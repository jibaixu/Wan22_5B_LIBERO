#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file


WANVIDEO_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = WANVIDEO_DIR / "model_training"
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WANVIDEO_DIR))
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(REPO_ROOT))

from libero_future_dataset import discover_libero_episodes  # noqa: E402
from libero_future_utils import (  # noqa: E402
    BASE_MODEL_ID,
    FEATURE_DIRECTORY,
    absolute_checkpoint_path,
    checkpoint_sha256,
    confidence_feature_directory,
    load_wan_pipe,
    teacher_seed,
)
from diffsynth.pipelines.wan_video import WanFeatureCaptureSpec  # noqa: E402


REQUIRED_TENSOR_KEYS = {"features", "anchor_frames", "valid_future_frames"}
CONFIDENCE_TENSOR_KEYS = {"confidence", "disagreement"}
REQUIRED_METADATA = {
    "complete": "true",
    "teacher": BASE_MODEL_ID,
    "fps": "20",
    "video_key": "observation.images.image",
    "prompt_source": "meta/episodes.jsonl:tasks[0]",
    "share_stride": "8",
    "future_horizon_frames": "16",
    "num_frames": "17",
    "height": "480",
    "width": "480",
    "num_inference_steps": "20",
    "saved_denoise_step": "19",
    "saved_denoise_step_indexing": "zero_based",
    "block_index": "20",
    "block_indexing": "zero_based",
    "cfg_branch": "positive",
    "cfg_scale": "5.0",
    "negative_prompt": "",
    "rand_device": "cpu",
    "tiled": "false",
    "sigma_shift": "5.0",
    "temporal_tokens": "5",
    "token_h": "15",
    "token_w": "15",
    "target_tokens": "1125",
    "feature_dim": "3072",
    "token_order": "temporal_height_width",
    "dtype": "bfloat16",
    "base_seed": "42",
}


def load_teacher_projection(path):
    if not path:
        return None
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        mean = reader.get_tensor("mean").float().contiguous()
        components = reader.get_tensor("components").float().contiguous()
        metadata = reader.metadata() or {}
    if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != mean.numel():
        raise ValueError(f"Invalid teacher projection shapes: {tuple(mean.shape)}, {tuple(components.shape)}")
    digest = hashlib.sha256(mean.numpy().tobytes() + components.numpy().tobytes()).hexdigest()
    if metadata.get("projection_sha256") and metadata["projection_sha256"] != digest:
        raise ValueError(f"Teacher projection hash mismatch: {metadata['projection_sha256']} != {digest}")
    return mean, components, digest


def project_teacher_feature(feature, projection):
    if projection is None:
        return feature
    mean, components, _ = projection
    if feature.ndim != 2 or feature.shape[-1] != mean.numel():
        raise ValueError(f"Teacher feature shape {tuple(feature.shape)} does not match projection input {mean.numel()}")
    return ((feature.float() - mean) @ components.T).to(torch.bfloat16).contiguous()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def expected_anchor_frames(episode_length, share_stride=8):
    return torch.arange(0, episode_length, share_stride, dtype=torch.int64)


def expected_valid_future_frames(episode_length, anchors, effective_future_steps=16):
    return torch.clamp(episode_length - 1 - anchors, min=0, max=effective_future_steps).to(torch.int16)


def future_frame_ids(anchor_frame, episode_length, num_frames=17, temporal_stride=1):
    return [min(anchor_frame + index * temporal_stride, episode_length - 1) for index in range(num_frames)]


def validate_episode_cache(
    path, episode, expected_metadata, allow_pending=False, check_finite=False
):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing or empty file"
    try:
        with safe_open(path, framework="pt", device="cpu") as file:
            keys = set(file.keys())
            if not REQUIRED_TENSOR_KEYS.issubset(keys):
                return False, f"missing tensor keys: {sorted(REQUIRED_TENSOR_KEYS - keys)}"
            metadata = file.metadata() or {}
            confidence_aware = metadata.get("confidence_aware") == "true"
            for key, value in expected_metadata.items():
                if confidence_aware and key == "complete":
                    continue
                if metadata.get(key) != str(value):
                    return False, f"metadata mismatch for {key}"
            if confidence_aware:
                if not CONFIDENCE_TENSOR_KEYS.issubset(keys):
                    return False, "missing confidence-aware tensor keys"
                complete = metadata.get("complete", "false") == "true"
                if not complete and not allow_pending:
                    return False, "cache is pending confidence calibration"
                if metadata.get("extraction_complete") != "true":
                    return False, "confidence extraction is incomplete"
                if complete and metadata.get("calibration") != "train_quantile_05_95":
                    return False, "cache has no finalized confidence calibration"
            elif metadata.get("complete") != "true":
                return False, "cache is incomplete"

            expected_anchors = expected_anchor_frames(
                int(episode["episode_length"]), int(episode.get("teacher_share_stride", 8))
            )
            expected_valid = expected_valid_future_frames(
                int(episode["episode_length"]), expected_anchors,
                16 * int(episode.get("teacher_temporal_stride", 1)),
            )
            feature_slice = file.get_slice("features")
            anchor_slice = file.get_slice("anchor_frames")
            valid_slice = file.get_slice("valid_future_frames")
            expected_feature_shape = [len(expected_anchors), 1125, int(episode.get("teacher_feature_dim", 3072))]
            if feature_slice.get_shape() != expected_feature_shape:
                return False, "feature shape mismatch"
            if feature_slice.get_dtype() != "BF16":
                return False, "feature dtype mismatch"
            if anchor_slice.get_shape() != [len(expected_anchors)]:
                return False, "anchor shape mismatch"
            if anchor_slice.get_dtype() != "I64":
                return False, "anchor dtype mismatch"
            if valid_slice.get_shape() != [len(expected_anchors)]:
                return False, "valid-future shape mismatch"
            if valid_slice.get_dtype() != "I16":
                return False, "valid-future dtype mismatch"

            if confidence_aware:
                if check_finite:
                    features = file.get_tensor("features")
                    if not torch.isfinite(features.float()).all():
                        return False, "non-finite teacher features"
                confidence_slice = file.get_slice("confidence")
                disagreement_slice = file.get_slice("disagreement")
                expected_scalar_shape = [len(expected_anchors)]
                if confidence_slice.get_shape() != expected_scalar_shape:
                    return False, "confidence shape mismatch"
                if disagreement_slice.get_shape() != expected_scalar_shape:
                    return False, "disagreement shape mismatch"
                if confidence_slice.get_dtype() != "F32" or disagreement_slice.get_dtype() != "F32":
                    return False, "confidence/disagreement dtype mismatch"
                confidence = file.get_tensor("confidence")
                disagreement = file.get_tensor("disagreement")
                if not torch.isfinite(confidence).all() or not torch.isfinite(disagreement).all():
                    return False, "non-finite confidence statistics"
                if torch.any(confidence < 0) or torch.any(confidence > 1):
                    return False, "confidence outside [0, 1]"

            anchors = file.get_tensor("anchor_frames")
            valid = file.get_tensor("valid_future_frames")
            if not torch.equal(anchors, expected_anchors):
                return False, "anchor values mismatch"
            if len(anchors) == 0 or anchors[0].item() != 0:
                return False, "first anchor is not zero"
            if not torch.equal(valid, expected_valid):
                return False, "valid-future values mismatch"
    except Exception as error:
        return False, f"cannot open cache: {error}"
    return True, "ok"


def cache_compatibility_metadata(
    episode, checkpoint_path, checkpoint_hash, base_seed,
    confidence_aware=False, num_teacher_samples=4,
):
    anchors = expected_anchor_frames(episode["episode_length"], int(episode.get("teacher_share_stride", 8)))
    seeds = [
        teacher_seed(
            base_seed,
            episode["suite_id"],
            episode["episode_index"],
            int(anchor),
        )
        for anchor in anchors.tolist()
    ]
    metadata = dict(REQUIRED_METADATA)
    if episode.get("dataset_format") == "robotwin":
        temporal_stride = int(episode.get("teacher_temporal_stride", 3))
        metadata.update({
            "dataset_format": "robotwin",
            "fps": str(episode.get("fps", 15)),
            "video_key": str(episode.get("video_key", "observation.images.image")),
            "prompt_source": "robotwin_teacher_manifest",
            "share_stride": str(episode.get("teacher_share_stride", 1)),
            "temporal_stride": str(temporal_stride),
            "effective_future_steps": str(16 * temporal_stride),
            "future_horizon_frames": str(16 * temporal_stride),
        })
    if episode.get("teacher_feature_dim") is not None:
        metadata["feature_dim"] = str(episode["teacher_feature_dim"])
    if episode.get("teacher_projection_sha256"):
        metadata.update({
            "feature_transform": "tokenwise_pca_v1",
            "projection_sha256": str(episode["teacher_projection_sha256"]),
            "projection_input_dim": "3072",
            "projection_output_dim": str(episode["teacher_feature_dim"]),
            "source_feature_dim": "3072",
        })
    metadata.update(
        {
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_hash,
            "suite": episode["suite"],
            "suite_id": str(episode["suite_id"]),
            "episode_index": str(episode["episode_index"]),
            "episode_length": str(episode["episode_length"]),
            "base_seed": str(base_seed),
            "seed_formula": "(base_seed + suite_id*100000000 + episode_index*1000003 + anchor_frame*9176) % 2**31",
            "anchor_seeds": ",".join(str(seed) for seed in seeds),
        }
    )
    if confidence_aware:
        metadata.update(
            {
                "confidence_aware": "true",
                "confidence_version": "v1",
                "num_teacher_samples": str(int(num_teacher_samples)),
                "target_aggregation": "mean",
            }
        )
    return {key: str(value) for key, value in metadata.items()}


def confidence_cache_metadata(
    episode, checkpoint_path, checkpoint_hash, base_seed, num_teacher_samples,
    complete=False, calibration=None, calibration_scope=None,
):
    metadata = cache_compatibility_metadata(
        episode, checkpoint_path, checkpoint_hash, base_seed,
        confidence_aware=True, num_teacher_samples=num_teacher_samples,
    )
    metadata.update(
        {
            "complete": str(bool(complete)).lower(),
            "extraction_complete": "true",
            "confidence_aware": "true",
            "confidence_version": "v1",
            "num_teacher_samples": str(int(num_teacher_samples)),
            "target_aggregation": "mean",
            "uncertainty_metric": "pairwise_cosine_disagreement",
            "feature_normalization": "layernorm_last_dim",
            "uncertainty_normalization": "l2_last_dim",
            "uncertainty_aggregation": "median_token",
            "calibration": "train_quantile_05_95" if calibration else "pending",
            "seed_policy": "explicit_per_anchor",
            "sample_seed_formula": "(base_seed + suite_id*100000000 + episode_index*1000003 + anchor_frame*9176 + sample_index*97531) % 2**31",
            "sample_seeds": json.dumps(
                [
                    [
                        teacher_seed(
                            base_seed,
                            episode["suite_id"],
                            episode["episode_index"],
                            int(anchor),
                            sample_index,
                        )
                        for sample_index in range(int(num_teacher_samples))
                    ]
                    for anchor in expected_anchor_frames(
                        episode["episode_length"], int(episode.get("teacher_share_stride", 8))
                    ).tolist()
                ],
                separators=(",", ":"),
            ),
        }
    )
    if calibration:
        if calibration_scope is None:
            calibration_scope = "suite_training_episodes"
        metadata.update(
            {
                "calibration_q_low": f"{calibration['q_low']:.9g}",
                "calibration_q_high": f"{calibration['q_high']:.9g}",
                "calibration_eps": f"{calibration['eps']:.9g}",
                "calibration_scope": calibration_scope,
                "calibration_suite": episode["suite"],
            }
        )
    return metadata


def cache_metadata(episode, checkpoint_path, checkpoint_hash, base_seed):
    metadata = cache_compatibility_metadata(
        episode, checkpoint_path, checkpoint_hash, base_seed
    )
    metadata.update(
        {
            "diffsynth_commit": git_commit(),
            "torch_version": torch.__version__,
            "cuda_version": str(torch.version.cuda),
        }
    )
    return metadata


def episode_output_path(episode, output_root=None, confidence_aware=False, num_teacher_samples=4):
    directory = confidence_feature_directory(num_teacher_samples) if confidence_aware else FEATURE_DIRECTORY
    if episode.get("dataset_format") == "robotwin":
        # A manifest row already points at its task-local LeRobot root.  Keep
        # each task's cache beside its source video; otherwise one global
        # output root would mix episode indices from different tasks.
        feature_root = Path(episode["dataset_root"]) / directory
    elif output_root is None:
        feature_root = Path(episode["dataset_root"]) / directory
    else:
        feature_root = Path(output_root).expanduser().resolve() / episode["suite"] / directory
    chunk_id = episode["episode_index"] // episode["chunks_size"]
    return feature_root / f"chunk-{chunk_id:03d}" / f"episode_{episode['episode_index']:06d}.safetensors"


def confidence_manifest_path(episode, output_root=None, num_teacher_samples=4):
    return episode_output_path(
        episode, output_root, confidence_aware=True,
        num_teacher_samples=num_teacher_samples,
    ).parents[1] / "confidence_manifest.json"


def read_anchor_image(reader, anchor_frame):
    image = Image.fromarray(reader.get_data(anchor_frame)).convert("RGB")
    return image.resize((480, 480), resample=Image.Resampling.BILINEAR)


def read_future_clip(reader, anchor_frame, episode_length, num_frames, temporal_stride):
    """Read the exact sparse clip used by the future-teacher contract."""
    frame_ids = future_frame_ids(anchor_frame, episode_length, num_frames, temporal_stride)
    return [read_anchor_image(reader, frame_id) for frame_id in frame_ids]


class _FeatureSampleAccumulator:
    """Streaming mean and disagreement statistics for one anchor."""

    def __init__(self):
        self.feature_sum = None
        self.normalized_sum = None
        self.normalized_sqnorm_sum = None
        self.sample_count = 0
        self.expected_shape = None

    def add(self, feature):
        feature = feature.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(feature).all():
            raise RuntimeError("Teacher feature contains non-finite values")
        if self.expected_shape is None:
            self.expected_shape = tuple(feature.shape)
            self.feature_sum = torch.zeros_like(feature)
            self.normalized_sum = torch.zeros_like(feature)
            self.normalized_sqnorm_sum = torch.zeros(feature.shape[0], dtype=torch.float32)
        elif tuple(feature.shape) != self.expected_shape:
            raise RuntimeError("Teacher sample shapes are inconsistent")
        normalized = torch.nn.functional.layer_norm(feature, (feature.shape[-1],))
        normalized = torch.nn.functional.normalize(normalized, dim=-1)
        self.feature_sum.add_(feature)
        self.normalized_sum.add_(normalized)
        self.normalized_sqnorm_sum.add_((normalized * normalized).sum(dim=-1))
        self.sample_count += 1

    def finalize(self):
        if self.sample_count == 0:
            raise RuntimeError("No teacher samples were produced")
        mean_feature = self.feature_sum / float(self.sample_count)
        if self.sample_count < 2:
            disagreement = torch.zeros((), dtype=torch.float32)
        else:
            pairwise_cosine = (
                self.normalized_sum.square().sum(dim=-1) - self.normalized_sqnorm_sum
            ) / float(self.sample_count * (self.sample_count - 1))
            disagreement = (1.0 - pairwise_cosine).clamp_(0.0, 2.0).median()
        return mean_feature, disagreement


def extract_episode(
    pipe, episode, checkpoint_path, checkpoint_hash, base_seed, output_path,
    confidence_aware=False, num_teacher_samples=4, share_stride=8, temporal_stride=1,
    projection=None,
):
    anchors = expected_anchor_frames(episode["episode_length"], share_stride)
    valid = expected_valid_future_frames(episode["episode_length"], anchors, 16 * temporal_stride)
    features = []
    disagreements = []
    reader = imageio.get_reader(episode["video_path"])
    try:
        for anchor in anchors.tolist():
            clip = read_future_clip(
                reader, anchor, episode["episode_length"], 17, temporal_stride
            )
            accumulator = _FeatureSampleAccumulator()
            sample_count = num_teacher_samples if confidence_aware else 1
            for sample_index in range(sample_count):
                seed = teacher_seed(
                    base_seed, episode["suite_id"], episode["episode_index"],
                    anchor, sample_index,
                )
                result = pipe.extract_dit_feature(
                    prompt=episode["prompt"], input_video=clip, height=480,
                    width=480, num_frames=17, seed=seed, num_inference_steps=20,
                    sigma_shift=5.0, cfg_scale=5.0, negative_prompt="",
                    rand_device="cpu", tiled=False, capture=WanFeatureCaptureSpec(),
                )
                accumulator.add(result.feature)
                del result
            mean_feature, disagreement = accumulator.finalize()
            features.append(project_teacher_feature(mean_feature, projection))
            disagreements.append(disagreement)
            del accumulator, mean_feature, disagreement
    finally:
        reader.close()

    tensors = {
        "features": torch.stack(features).to(dtype=torch.bfloat16, device="cpu"),
        "anchor_frames": anchors,
        "valid_future_frames": valid,
    }
    if confidence_aware:
        tensors["disagreement"] = torch.stack(disagreements).to(dtype=torch.float32)
        tensors["confidence"] = torch.ones(len(disagreements), dtype=torch.float32)
        metadata = confidence_cache_metadata(
            episode, checkpoint_path, checkpoint_hash, base_seed, num_teacher_samples,
        )
    else:
        metadata = cache_metadata(episode, checkpoint_path, checkpoint_hash, base_seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    save_file(tensors, temporary, metadata=metadata)
    valid_file, reason = validate_episode_cache(
        temporary, episode, metadata, allow_pending=confidence_aware
    )
    if not valid_file:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Temporary cache validation failed: {reason}")
    os.replace(temporary, output_path)
    return len(anchors), output_path.stat().st_size


def finalize_confidence_cache(episodes, output_root, checkpoint_path, checkpoint_hash, base_seed, num_teacher_samples, manifest_path):
    suites = {episode["suite"] for episode in episodes}
    if len(suites) != 1:
        raise ValueError("Confidence calibration requires episodes from exactly one suite")
    suite = next(iter(suites))
    paths = [episode_output_path(e, output_root, True, num_teacher_samples) for e in episodes]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize confidence cache; missing files: {missing[:3]}")
    raw_values = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as file:
            raw_values.append(file.get_tensor("disagreement").float())
    raw = torch.cat(raw_values)
    if raw.numel() == 0 or not torch.isfinite(raw).all():
        raise RuntimeError("Cannot calibrate empty or non-finite disagreement statistics")
    q_low = float(torch.quantile(raw, 0.05).item())
    q_high = float(torch.quantile(raw, 0.95).item())
    calibration = {"q_low": q_low, "q_high": q_high, "eps": 1e-8}
    for episode, path in zip(episodes, paths):
        with safe_open(path, framework="pt", device="cpu") as file:
            tensors = {
                "features": file.get_tensor("features").contiguous(),
                "anchor_frames": file.get_tensor("anchor_frames").contiguous(),
                "valid_future_frames": file.get_tensor("valid_future_frames").contiguous(),
                "disagreement": file.get_tensor("disagreement").float().contiguous(),
            }
            metadata = dict(file.metadata() or {})
        confidence = 1.0 - (tensors["disagreement"] - q_low) / (q_high - q_low + calibration["eps"])
        tensors["confidence"] = confidence.clamp_(0.0, 1.0).float().contiguous()
        metadata.update(confidence_cache_metadata(
            episode, checkpoint_path, checkpoint_hash, base_seed, num_teacher_samples,
            complete=True, calibration=calibration,
            calibration_scope="suite_training_episodes",
        ))
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        save_file(tensors, temporary, metadata=metadata)
        valid_file, reason = validate_episode_cache(
            temporary, episode, metadata, check_finite=True
        )
        if not valid_file:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Finalized cache validation failed: {reason}")
        os.replace(temporary, path)
    atomic_json(manifest_path, {
        "confidence_aware": True, "confidence_version": "v1",
        "num_teacher_samples": int(num_teacher_samples),
        "calibration": "train_quantile_05_95",
        "calibration_scope": "suite_training_episodes", "suite": suite,
        "q_low": q_low,
        "q_high": q_high, "eps": calibration["eps"],
        "episode_count": len(paths), "anchor_count": int(raw.numel()),
        "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint_hash,
    })
    return calibration


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True, default=str)
        file.write("\n")
    os.replace(temporary, path)


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def broadcast_checkpoint_hash(checkpoint_path, rank, world_size):
    value = [checkpoint_sha256(checkpoint_path) if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(value, src=0)
    return value[0]


def verify_validation_gate(
    report_path, checkpoint_path, checkpoint_hash, checkpoint_type
):
    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)
    if not (
        report.get("loss_passed", False)
        and report.get("video_passed", False)
        and report.get("passed", False)
    ):
        raise RuntimeError("Validation report has not passed both loss and video gates")
    selected = report.get("selected_checkpoint", {})
    if selected.get("path") != checkpoint_path:
        raise RuntimeError("Validation report selected a different checkpoint path")
    if selected.get("sha256") != checkpoint_hash:
        raise RuntimeError("Validation report checkpoint hash does not match")
    if selected.get("type") != checkpoint_type:
        raise RuntimeError("Validation report checkpoint type does not match")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract LIBERO Wan future-teacher features.")
    parser.add_argument(
        "--data-root",
        default="/data/jibaixu/Datasets/LIBERO/libero_no_noop",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-type", choices=("lora", "full"), default="lora")
    parser.add_argument("--projection", default=None, help="Optional token-wise PCA safetensors projection.")
    parser.add_argument("--manifest", default=None, help="Canonical RoboTwin teacher JSONL manifest.")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--feature-directory", default=None)
    parser.add_argument("--share-stride", type=int, default=8)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--progress-dir",
        default="/data/jibaixu/Ckpts/wan2_2_ti2v_5b_libero/extraction_progress",
    )
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--suites", nargs="*", default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--episode-indices", type=int, nargs="*", default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--confidence-aware",
        action="store_true",
        help="Run multiple teacher samples and write mean target/confidence tensors.",
    )
    parser.add_argument(
        "--num-teacher-samples",
        type=int,
        default=4,
        help="Number of teacher realizations per anchor in confidence-aware mode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def filter_episodes(indexed_episodes, args):
    episode_indices = set(args.episode_indices or [])
    selected = []
    for ordinal, episode in indexed_episodes:
        if args.suites and episode["suite"] not in args.suites:
            continue
        episode_index = episode["episode_index"]
        if args.episode_start is not None and episode_index < args.episode_start:
            continue
        if args.episode_end is not None and episode_index >= args.episode_end:
            continue
        if episode_indices and episode_index not in episode_indices:
            continue
        selected.append((ordinal, episode))
    return selected


def has_episode_filter(args):
    return (
        args.episode_start is not None
        or args.episode_end is not None
        or args.episode_indices is not None
    )


def main():
    global FEATURE_DIRECTORY
    args = parse_args()
    projection = load_teacher_projection(args.projection)
    if args.feature_directory:
        FEATURE_DIRECTORY = str(args.feature_directory)
    if args.audit_only and args.overwrite:
        raise ValueError("--audit-only and --overwrite cannot be used together")
    if args.num_teacher_samples < 1:
        raise ValueError("--num-teacher-samples must be positive")
    if args.confidence_aware and args.num_teacher_samples < 2:
        raise ValueError("--confidence-aware requires --num-teacher-samples >= 2")
    if args.share_stride <= 0 or args.temporal_stride <= 0:
        raise ValueError("--share-stride and --temporal-stride must be positive")
    rank, local_rank, world_size = init_distributed()
    checkpoint_path = absolute_checkpoint_path(args.checkpoint_path)
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)

    if args.manifest:
        episodes = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.task_id is not None:
            episodes = [episode for episode in episodes if episode.get("task_id") == args.task_id]
        for suite_id, episode in enumerate(episodes):
            episode["suite_id"] = suite_id
            episode["dataset_format"] = "robotwin"
            episode["teacher_share_stride"] = args.share_stride
            episode["teacher_temporal_stride"] = args.temporal_stride
            if projection is not None:
                episode["teacher_feature_dim"] = int(projection[1].shape[0])
                episode["teacher_projection_sha256"] = projection[2]
    else:
        episodes = discover_libero_episodes(args.data_root)
    selected = filter_episodes(list(enumerate(episodes)), args)
    if not selected:
        raise ValueError("Episode filters selected no work")
    if args.smoke_test:
        has_filter = bool(
            args.suites
            or args.episode_indices
            or args.episode_start is not None
            or args.episode_end is not None
        )
        if not has_filter or len(selected) > max(8, world_size):
            raise ValueError("--smoke-test requires explicit filters selecting at most 8 episodes")
    elif not args.dry_run and not args.audit_only and args.validation_report is None:
        raise ValueError("Formal extraction requires --validation-report")

    assignments = [
        (ordinal, episode)
        for ordinal, episode in selected
        if ordinal % world_size == rank
    ]
    progress_dir = Path(args.progress_dir).expanduser().resolve()
    progress_path = progress_dir / f"rank-{rank:03d}.json"
    progress = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "assigned_episodes": len(assignments),
        "completed_episodes": 0,
        "skipped_episodes": 0,
        "anchors": 0,
        "bytes": 0,
        "status": "dry_run" if args.dry_run else "running",
        "invalid_episodes": [],
        "episodes": [
            f"{episode['suite']}:{episode['episode_index']}" for _, episode in assignments
        ],
    }
    atomic_json(progress_path, progress)
    if args.dry_run:
        print(json.dumps(progress, indent=2, sort_keys=True))
        if world_size > 1:
            dist.barrier()
        return

    checkpoint_hash = broadcast_checkpoint_hash(checkpoint_path, rank, world_size)
    if not args.smoke_test and not args.audit_only:
        verify_validation_gate(
            args.validation_report,
            checkpoint_path,
            checkpoint_hash,
            args.checkpoint_type,
        )

    pending = []
    for _, episode in assignments:
        output_path = episode_output_path(
            episode, args.output_root, args.confidence_aware, args.num_teacher_samples
        )
        compatibility = cache_compatibility_metadata(
            episode, checkpoint_path, checkpoint_hash, args.base_seed,
            confidence_aware=args.confidence_aware,
            num_teacher_samples=args.num_teacher_samples,
        )
        if output_path.is_file() and not args.overwrite:
            is_valid, reason = validate_episode_cache(
                output_path, episode, compatibility,
                allow_pending=args.confidence_aware and not args.audit_only,
                check_finite=args.audit_only,
            )
            if is_valid:
                progress["skipped_episodes"] += 1
                progress["completed_episodes"] += 1
                progress["anchors"] += math.ceil(episode["episode_length"] / args.share_stride)
                progress["bytes"] += output_path.stat().st_size
                atomic_json(progress_path, progress)
                continue
        else:
            reason = "missing or overwrite requested"
        if args.audit_only:
            progress["invalid_episodes"].append(
                f"{episode['suite']}:{episode['episode_index']}:{reason}"
            )
            continue
        pending.append((episode, output_path))

    pipe = None
    if pending:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        pipe = load_wan_pipe(checkpoint_path, args.checkpoint_type, device=device)
    for episode, output_path in pending:
        anchors, file_size = extract_episode(
            pipe,
            episode,
            checkpoint_path,
            checkpoint_hash,
            args.base_seed,
            output_path,
            confidence_aware=args.confidence_aware,
            num_teacher_samples=args.num_teacher_samples,
            share_stride=args.share_stride,
            temporal_stride=args.temporal_stride,
            projection=projection,
        )
        progress["completed_episodes"] += 1
        progress["anchors"] += anchors
        progress["bytes"] += file_size
        progress["last_episode"] = f"{episode['suite']}:{episode['episode_index']}"
        atomic_json(progress_path, progress)

    progress["status"] = (
        "complete" if not progress["invalid_episodes"] else "invalid"
    )
    progress["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(progress_path, progress)
    if world_size > 1:
        dist.barrier()

    suite_confidence_run = (
        args.confidence_aware
        and not has_episode_filter(args)
        and not args.dry_run
        and not args.audit_only
    )
    calibrations = {}
    if rank == 0 and suite_confidence_run:
        selected_suites = sorted({episode["suite"] for _, episode in selected})
        for suite in selected_suites:
            suite_episodes = [
                episode for episode in episodes if episode["suite"] == suite
            ]
            manifest_path = confidence_manifest_path(
                suite_episodes[0], args.output_root, args.num_teacher_samples
            )
            finalized = manifest_path.is_file()
            if finalized:
                for episode in suite_episodes:
                    path = episode_output_path(
                        episode, args.output_root, True, args.num_teacher_samples
                    )
                    compatibility = cache_compatibility_metadata(
                        episode, checkpoint_path, checkpoint_hash, args.base_seed,
                        confidence_aware=True,
                        num_teacher_samples=args.num_teacher_samples,
                    )
                    valid_cache, _ = validate_episode_cache(path, episode, compatibility)
                    if not valid_cache:
                        finalized = False
                        break
                    with safe_open(path, framework="pt", device="cpu") as file:
                        metadata = file.metadata() or {}
                    if (
                        metadata.get("calibration_scope")
                        != "suite_training_episodes"
                        or metadata.get("calibration_suite") != suite
                    ):
                        finalized = False
                        break
                if finalized:
                    with open(manifest_path, "r", encoding="utf-8") as file:
                        manifest = json.load(file)
                    if (
                        manifest.get("suite") != suite
                        or manifest.get("calibration_scope")
                        != "suite_training_episodes"
                    ):
                        finalized = False
            if finalized:
                calibrations[suite] = {
                    **manifest, "manifest_path": str(manifest_path),
                }
            else:
                calibration = finalize_confidence_cache(
                    suite_episodes,
                    args.output_root,
                    checkpoint_path,
                    checkpoint_hash,
                    args.base_seed,
                    args.num_teacher_samples,
                    manifest_path,
                )
                calibrations[suite] = {
                    **calibration,
                    "suite": suite,
                    "calibration_scope": "suite_training_episodes",
                    "manifest_path": str(manifest_path),
                }
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        summary_allows_pending = (
            args.confidence_aware
            and not args.audit_only
            and not suite_confidence_run
        )
        rank_progress = []
        for rank_id in range(world_size):
            path = progress_dir / f"rank-{rank_id:03d}.json"
            with open(path, "r", encoding="utf-8") as file:
                rank_progress.append(json.load(file))
        summary = {
            "complete": all(item["status"] == "complete" for item in rank_progress),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_type": args.checkpoint_type,
            "lora_alpha": 1.0 if args.checkpoint_type == "lora" else None,
            "selected_episodes": len(selected),
            "completed_episodes": sum(item["completed_episodes"] for item in rank_progress),
            "anchors": sum(item["anchors"] for item in rank_progress),
            "bytes": sum(item["bytes"] for item in rank_progress),
            "feature_directory": confidence_feature_directory(args.num_teacher_samples) if args.confidence_aware else FEATURE_DIRECTORY,
            "confidence_aware": args.confidence_aware,
            "num_teacher_samples": args.num_teacher_samples if args.confidence_aware else 1,
            "calibrations": calibrations,
            "world_size": world_size,
            "smoke_test": args.smoke_test,
            "audit_only": args.audit_only,
            "invalid_episodes": sum(
                (item["invalid_episodes"] for item in rank_progress), []
            ),
            "ranks": rank_progress,
        }
        atomic_json(progress_dir / "extraction_summary.json", summary)
        for suite in sorted({episode["suite"] for _, episode in selected}):
            suite_episodes = [episode for _, episode in selected if episode["suite"] == suite]
            feature_root = episode_output_path(
                suite_episodes[0], args.output_root, args.confidence_aware, args.num_teacher_samples
            ).parents[1]
            suite_completed = 0
            suite_anchors = 0
            suite_bytes = 0
            suite_invalid = []
            for episode in suite_episodes:
                output_path = episode_output_path(
                    episode, args.output_root, args.confidence_aware, args.num_teacher_samples
                )
                compatibility = cache_compatibility_metadata(
                    episode, checkpoint_path, checkpoint_hash, args.base_seed,
                    confidence_aware=args.confidence_aware,
                    num_teacher_samples=args.num_teacher_samples,
                )
                is_valid, reason = validate_episode_cache(
                    output_path, episode, compatibility,
                    allow_pending=summary_allows_pending,
                    check_finite=args.audit_only,
                )
                if is_valid:
                    suite_completed += 1
                    suite_anchors += math.ceil(
                        episode["episode_length"] / args.share_stride
                    )
                    suite_bytes += output_path.stat().st_size
                else:
                    suite_invalid.append(
                        f"{suite}:{episode['episode_index']}:{reason}"
                    )
            expected_full_suite_episodes = sum(
                episode["suite"] == suite for episode in episodes
            )
            suite_summary = {
                "complete": suite_completed == len(suite_episodes),
                "complete_full_suite": (
                    len(suite_episodes) == expected_full_suite_episodes
                    and suite_completed == expected_full_suite_episodes
                ),
                "suite": suite,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_type": args.checkpoint_type,
                "lora_alpha": 1.0 if args.checkpoint_type == "lora" else None,
                "selected_episodes": len(suite_episodes),
                "expected_full_suite_episodes": expected_full_suite_episodes,
                "completed_episodes": suite_completed,
                "anchors": suite_anchors,
                "bytes": suite_bytes,
                "feature_directory": confidence_feature_directory(args.num_teacher_samples) if args.confidence_aware else FEATURE_DIRECTORY,
                "confidence_aware": args.confidence_aware,
                "num_teacher_samples": args.num_teacher_samples if args.confidence_aware else 1,
                "calibration": calibrations.get(suite),
                "world_size": world_size,
                "smoke_test": args.smoke_test,
                "audit_only": args.audit_only,
                "invalid_episodes": suite_invalid,
            }
            atomic_json(feature_root / "extraction_summary.json", suite_summary)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
