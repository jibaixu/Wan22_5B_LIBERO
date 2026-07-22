import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import accelerate
import torch
from accelerate.utils import set_seed

from diffsynth.diffusion import ModelLogger, launch_training_task

from libero_future_dataset import LiberoFutureWindowDataset
from train import WanTrainingModule, wan_parser


class LiberoWanTrainingModule(WanTrainingModule):
    def forward(self, data, inputs=None):
        loss = super().forward(data, inputs=inputs)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("LIBERO future training produced a non-finite loss")
        return loss


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run_metadata(args):
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": "Wan-AI/Wan2.2-TI2V-5B",
        "git_commit": git_commit(),
        "manifest_path": str(Path(args.dataset_metadata_path).expanduser().resolve()),
        "manifest_sha256": sha256_file(args.dataset_metadata_path),
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "accelerate_version": accelerate.__version__,
        "command": " ".join(os.sys.argv),
        "arguments": vars(args),
    }
    temporary = output_path / "run_metadata.json.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True, default=str)
        file.write("\n")
    os.replace(temporary, output_path / "run_metadata.json")


def parse_args():
    parser = wan_parser()
    parser.description = "Train Wan2.2-TI2V-5B on online LIBERO future windows."
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit unique manifest rows for smoke tests.",
    )
    args = parser.parse_args()
    if args.dataset_metadata_path is None:
        parser.error("--dataset_metadata_path must point to a LIBERO JSONL manifest")
    if (args.height, args.width, args.num_frames) != (480, 480, 17):
        parser.error("LIBERO future training requires --height 480 --width 480 --num_frames 17")
    extra_inputs = set((args.extra_inputs or "").split(","))
    if "input_image" not in extra_inputs:
        parser.error('--extra_inputs must include "input_image"')
    return args


def main():
    args = parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        write_run_metadata(args)

    dataset = LiberoFutureWindowDataset(
        manifest_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        max_samples=args.max_samples,
    )
    model = LiberoWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device=(
            "cpu"
            if args.initialize_model_on_cpu or args.enable_model_cpu_offload
            else accelerator.device
        ),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launch_training_task(
        accelerator, dataset, model, model_logger, args=args
    )


if __name__ == "__main__":
    main()
