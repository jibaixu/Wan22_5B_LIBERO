import hashlib
from pathlib import Path

import torch

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


BASE_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
FEATURE_DIRECTORY = "wan2_2_future_latent_h16_step20_block20z_stride8_480x480"


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


def teacher_seed(base_seed, suite_id, episode_index, anchor_frame):
    return (
        base_seed
        + suite_id * 100_000_000
        + episode_index * 1_000_003
        + anchor_frame * 9_176
    ) % (2**31)


def absolute_checkpoint_path(path):
    return str(Path(path).expanduser().resolve())
