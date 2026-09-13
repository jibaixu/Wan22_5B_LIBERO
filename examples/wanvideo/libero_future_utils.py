import hashlib
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import torch

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


BASE_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
FEATURE_DIRECTORY = "wan2_2_future_latent_h16_step20_block20z_stride8_480x480"


def prepare_model_cache(cache_dir):
    """Copy the Wan validation weights to a node-local, reusable cache.

    ModelConfig expects ``<base>/<model_id>/<origin_file_pattern>``.  Keeping
    that layout in the cache lets the normal downloader/loader code run
    unchanged while avoiding concurrent reads of the large DiT shards from a
    shared filesystem.  A lock serializes the first copy and a completion
    manifest makes subsequent ranks and runs cheap no-ops.
    """
    if not cache_dir:
        return None

    source_root = Path(
        os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models")
    ).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    if cache_root == source_root:
        return str(cache_root)

    model_specs = [
        (
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            "models_t5_umt5-xxl-enc-bf16.safetensors",
        ),
        (BASE_MODEL_ID, "diffusion_pytorch_model*.safetensors"),
        (
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            "Wan2.2_VAE.safetensors",
        ),
    ]
    files = []
    for model_id, pattern in model_specs:
        source_dir = source_root / model_id
        matches = sorted(source_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No model files matched {source_dir / pattern}. "
                "Set DIFFSYNTH_MODEL_BASE_PATH to a complete local model directory."
            )
        for source_path in matches:
            if not source_path.is_file():
                continue
            relative_path = Path(model_id) / source_path.relative_to(source_dir)
            files.append((source_path, cache_root / relative_path))

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".wan2.2-ti2v-5b.lock"
    complete_path = cache_root / ".wan2.2-ti2v-5b.complete.json"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            expected = {
                str(target.relative_to(cache_root)): source.stat().st_size
                for source, target in files
            }
            if complete_path.is_file():
                try:
                    metadata = json.loads(complete_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = None
                if metadata and metadata.get("files") == expected and all(
                    (cache_root / relative).is_file()
                    and (cache_root / relative).stat().st_size == size
                    for relative, size in expected.items()
                ):
                    return str(cache_root)

            started = time.monotonic()
            for source_path, target_path in files:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path = target_path.with_name(target_path.name + ".partial")
                shutil.copyfile(source_path, partial_path)
                if partial_path.stat().st_size != source_path.stat().st_size:
                    partial_path.unlink(missing_ok=True)
                    raise IOError(f"Incomplete model copy: {source_path}")
                os.replace(partial_path, target_path)
            temporary = complete_path.with_name(complete_path.name + ".partial")
            temporary.write_text(
                json.dumps(
                    {"source_root": str(source_root), "files": expected},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, complete_path)
            elapsed = time.monotonic() - started
            print(f"Prepared local Wan model cache at {cache_root} ({elapsed:.1f}s)")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return str(cache_root)


def confidence_feature_directory(num_teacher_samples=4):
    """Directory name for mean-target, confidence-aware future features."""
    suffix = f"_samples{int(num_teacher_samples)}_confv1"
    # Callers may pass an already-versioned directory name through
    # ``--feature-directory``.  Do not append the confidence suffix twice;
    # doing so silently puts extracted caches in a path different from the
    # one used by the training configuration.
    return FEATURE_DIRECTORY if FEATURE_DIRECTORY.endswith(suffix) else f"{FEATURE_DIRECTORY}{suffix}"


def wan_model_configs():
    return [
        ModelConfig(
            model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors",
        ),
        ModelConfig(
            model_id=BASE_MODEL_ID,
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            origin_file_pattern="Wan2.2_VAE.safetensors",
        ),
    ]


def load_wan_pipe(checkpoint_path=None, checkpoint_type="base", device="cuda"):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=wan_model_configs(),
    )
    if checkpoint_type == "base":
        if checkpoint_path is not None:
            raise ValueError("Base model does not accept a checkpoint path")
    elif checkpoint_type == "lora":
        if checkpoint_path is None:
            raise ValueError("LoRA checkpoint path is required")
        pipe.load_lora(pipe.dit, str(checkpoint_path), alpha=1.0)
    elif checkpoint_type == "full":
        if checkpoint_path is None:
            raise ValueError("Full checkpoint path is required")
        state_dict = load_state_dict(
            str(checkpoint_path), torch_dtype=torch.bfloat16, device=device
        )
        missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected full-checkpoint keys: {unexpected[:10]}")
        if missing:
            raise RuntimeError(f"Missing full-checkpoint keys: {missing[:10]}")
    else:
        raise ValueError(f"Unsupported checkpoint type {checkpoint_type!r}")
    return pipe


def checkpoint_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_seed(base_seed, suite_id, episode_index, anchor_frame, sample_index=0):
    return (
        base_seed
        + suite_id * 100_000_000
        + episode_index * 1_000_003
        + anchor_frame * 9_176
        + sample_index * 97_531
    ) % (2**31)


def absolute_checkpoint_path(path):
    return str(Path(path).expanduser().resolve())
