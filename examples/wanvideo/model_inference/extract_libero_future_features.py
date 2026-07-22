#!/usr/bin/env python3
import argparse
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
    load_wan_pipe,
    teacher_seed,
)
from diffsynth.pipelines.wan_video import WanFeatureCaptureSpec  # noqa: E402


REQUIRED_TENSOR_KEYS = {"features", "anchor_frames", "valid_future_frames"}
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


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def expected_anchor_frames(episode_length):
    return torch.arange(0, episode_length, 8, dtype=torch.int64)


def expected_valid_future_frames(episode_length, anchors):
    return torch.clamp(episode_length - 1 - anchors, min=0, max=16).to(torch.int16)


def validate_episode_cache(path, episode, expected_metadata):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing or empty file"
    try:
        with safe_open(path, framework="pt", device="cpu") as file:
            keys = set(file.keys())
            if not REQUIRED_TENSOR_KEYS.issubset(keys):
                return False, f"missing tensor keys: {sorted(REQUIRED_TENSOR_KEYS - keys)}"
            metadata = file.metadata() or {}
            for key, value in expected_metadata.items():
                if metadata.get(key) != str(value):
                    return False, f"metadata mismatch for {key}"

            expected_anchors = expected_anchor_frames(int(episode["episode_length"]))
            expected_valid = expected_valid_future_frames(
                int(episode["episode_length"]), expected_anchors
            )
            feature_slice = file.get_slice("features")
            anchor_slice = file.get_slice("anchor_frames")
            valid_slice = file.get_slice("valid_future_frames")
            expected_feature_shape = [len(expected_anchors), 1125, 3072]
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


def cache_compatibility_metadata(episode, checkpoint_path, checkpoint_hash, base_seed):
    anchors = expected_anchor_frames(episode["episode_length"])
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
    return {key: str(value) for key, value in metadata.items()}


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


def episode_output_path(episode, output_root=None):
    if output_root is None:
        feature_root = Path(episode["dataset_root"]) / FEATURE_DIRECTORY
    else:
        feature_root = Path(output_root).expanduser().resolve() / episode["suite"] / FEATURE_DIRECTORY
    chunk_id = episode["episode_index"] // episode["chunks_size"]
    return feature_root / f"chunk-{chunk_id:03d}" / f"episode_{episode['episode_index']:06d}.safetensors"


def read_anchor_image(reader, anchor_frame):
    image = Image.fromarray(reader.get_data(anchor_frame)).convert("RGB")
    return image.resize((480, 480), resample=Image.Resampling.BILINEAR)


def extract_episode(pipe, episode, checkpoint_path, checkpoint_hash, base_seed, output_path):
    anchors = expected_anchor_frames(episode["episode_length"])
    valid = expected_valid_future_frames(episode["episode_length"], anchors)
    features = []
    reader = imageio.get_reader(episode["video_path"])
    try:
        for anchor in anchors.tolist():
            image = read_anchor_image(reader, anchor)
            seed = teacher_seed(
                base_seed,
                episode["suite_id"],
                episode["episode_index"],
                anchor,
            )
            result = pipe.extract_dit_feature(
                prompt=episode["prompt"],
                input_image=image,
                height=480,
                width=480,
                num_frames=17,
                seed=seed,
                num_inference_steps=20,
                sigma_shift=5.0,
                cfg_scale=5.0,
                negative_prompt="",
                rand_device="cpu",
                tiled=False,
                capture=WanFeatureCaptureSpec(),
            )
            features.append(result.feature)
    finally:
        reader.close()

    tensors = {
        "features": torch.stack(features).to(dtype=torch.bfloat16, device="cpu"),
        "anchor_frames": anchors,
        "valid_future_frames": valid,
    }
    metadata = cache_metadata(episode, checkpoint_path, checkpoint_hash, base_seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    save_file(tensors, temporary, metadata=metadata)
    valid_file, reason = validate_episode_cache(temporary, episode, metadata)
    if not valid_file:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Temporary cache validation failed: {reason}")
    os.replace(temporary, output_path)
    return len(anchors), output_path.stat().st_size


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


def main():
    args = parse_args()
    if args.audit_only and args.overwrite:
        raise ValueError("--audit-only and --overwrite cannot be used together")
    rank, local_rank, world_size = init_distributed()
    checkpoint_path = absolute_checkpoint_path(args.checkpoint_path)
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)

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
        output_path = episode_output_path(episode, args.output_root)
        compatibility = cache_compatibility_metadata(
            episode, checkpoint_path, checkpoint_hash, args.base_seed
        )
        if output_path.is_file() and not args.overwrite:
            is_valid, reason = validate_episode_cache(output_path, episode, compatibility)
            if is_valid:
                progress["skipped_episodes"] += 1
                progress["completed_episodes"] += 1
                progress["anchors"] += math.ceil(episode["episode_length"] / 8)
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

    if rank == 0:
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
            "feature_directory": FEATURE_DIRECTORY,
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
            feature_root = episode_output_path(suite_episodes[0], args.output_root).parents[1]
            suite_completed = 0
            suite_anchors = 0
            suite_bytes = 0
            suite_invalid = []
            for episode in suite_episodes:
                output_path = episode_output_path(episode, args.output_root)
                compatibility = cache_compatibility_metadata(
                    episode, checkpoint_path, checkpoint_hash, args.base_seed
                )
                is_valid, reason = validate_episode_cache(
                    output_path, episode, compatibility
                )
                if is_valid:
                    suite_completed += 1
                    suite_anchors += math.ceil(episode["episode_length"] / 8)
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
                "feature_directory": FEATURE_DIRECTORY,
                "world_size": world_size,
                "smoke_test": args.smoke_test,
                "audit_only": args.audit_only,
                "invalid_episodes": suite_invalid,
            }
            atomic_json(feature_root / "extraction_summary.json", suite_summary)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
