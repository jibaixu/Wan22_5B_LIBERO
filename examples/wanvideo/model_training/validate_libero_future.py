#!/usr/bin/env python3
import argparse
import csv
import gc
import hashlib
import html
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diffsynth.diffusion import FlowMatchSFTLoss
from diffsynth.core import load_state_dict
from diffsynth.utils.data import save_video


WANVIDEO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WANVIDEO_DIR))

from libero_future_dataset import LiberoFutureWindowDataset, read_jsonl  # noqa: E402
from libero_future_utils import (  # noqa: E402
    BASE_MODEL_ID,
    checkpoint_sha256,
    load_wan_pipe,
    teacher_seed,
)
from train import WanTrainingModule  # noqa: E402


MODEL_PATHS = (
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors:"
    "models_t5_umt5-xxl-enc-bf16.safetensors,"
    "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,"
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors:Wan2.2_VAE.safetensors"
)


def stable_score(seed, namespace, row):
    key = (
        f"{seed}\0{namespace}\0{row['suite']}\0{row['episode_index']}\0"
        f"{row['anchor_frame']}\0{row['prompt']}"
    )
    return hashlib.sha256(key.encode("utf-8")).digest()


def stratified_selection(rows, count, seed, namespace):
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from a manifest containing {len(rows)}")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["suite"], row["prompt"])].append((index, row))
    selected = []
    selected_indices = set()
    if count >= len(groups):
        for group_key in sorted(groups):
            index, row = min(
                groups[group_key], key=lambda item: stable_score(seed, namespace, item[1])
            )
            selected.append(index)
            selected_indices.add(index)
    else:
        # The 32-video set cannot cover all 40 suite/task strata; guarantee suites first.
        for suite in sorted({row["suite"] for row in rows}):
            candidates = [
                (index, row) for index, row in enumerate(rows) if row["suite"] == suite
            ]
            index, _ = min(
                candidates, key=lambda item: stable_score(seed, namespace, item[1])
            )
            selected.append(index)
            selected_indices.add(index)
    remaining = sorted(
        ((index, row) for index, row in enumerate(rows) if index not in selected_indices),
        key=lambda item: stable_score(seed, namespace, item[1]),
    )
    selected.extend(index for index, _ in remaining[: count - len(selected)])
    return selected


def _zero_lora_parameters(module):
    """Disable an injected LoRA adapter without changing base weights."""
    with torch.no_grad():
        for name, parameter in module.pipe.dit.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                parameter.zero_()


def load_lora_checkpoint(module, checkpoint_path):
    """Replace the parameters of the existing DiT LoRA adapter."""
    if checkpoint_path is None:
        raise ValueError("LoRA checkpoint path is required")
    lora = load_state_dict(
        str(checkpoint_path),
        torch_dtype=module.pipe.torch_dtype,
        device=module.pipe.device,
    )
    lora_loader = module.pipe.lora_loader(
        torch_dtype=module.pipe.torch_dtype, device=module.pipe.device
    )
    lora = lora_loader.convert_state_dict(lora)
    lora = module.mapping_lora_state_dict(lora)
    adapter_names = {
        name
        for name, _ in module.pipe.dit.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    checkpoint_names = set(lora)
    missing = sorted(adapter_names - checkpoint_names)
    unexpected = sorted(checkpoint_names - adapter_names)
    if missing or unexpected:
        raise RuntimeError(
            "LoRA checkpoint does not match the injected adapter: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    load_result = module.pipe.dit.load_state_dict(lora, strict=False)
    missing_adapter = sorted(set(load_result.missing_keys) & adapter_names)
    if missing_adapter or load_result.unexpected_keys:
        raise RuntimeError(
            "Failed to replace LoRA checkpoint: "
            f"missing={missing_adapter[:10]}, unexpected={load_result.unexpected_keys[:10]}"
        )
    return module


def validation_module(checkpoint_path, checkpoint_type, device):
    kwargs = {
        "model_id_with_origin_paths": MODEL_PATHS,
        "extra_inputs": "input_image",
        "device": device,
        "task": "sft",
    }
    if checkpoint_type == "lora":
        kwargs.update(
            {
                "lora_base_model": "dit",
                "lora_target_modules": "q,k,v,o,ffn.0,ffn.2",
                "lora_rank": 32,
                # Inject the adapter once; individual checkpoints are loaded
                # later by load_lora_checkpoint so the base model is reusable.
                "lora_checkpoint": None,
            }
        )
    elif checkpoint_type == "full":
        kwargs.update(
            {
                "resume_from_checkpoint": checkpoint_path,
                "remove_prefix_in_ckpt": "pipe.dit.",
            }
        )
    elif checkpoint_type != "base":
        raise ValueError(checkpoint_type)
    module = WanTrainingModule(**kwargs)
    if checkpoint_type == "lora":
        _zero_lora_parameters(module)
        if checkpoint_path is not None:
            load_lora_checkpoint(module, checkpoint_path)
    module.eval()
    return module


def deterministic_loss(module, data, timestep_id, noise_seed):
    inputs_shared, inputs_posi, inputs_nega = module.get_pipeline_inputs(data)
    inputs_shared, inputs_posi, inputs_nega = module.transfer_data_to_device(
        (inputs_shared, inputs_posi, inputs_nega),
        module.pipe.device,
        module.pipe.torch_dtype,
    )
    with torch.inference_mode():
        for unit in module.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = module.pipe.unit_runner(
                unit, module.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        generator = torch.Generator("cpu").manual_seed(noise_seed)
        noise = torch.randn(
            inputs_shared["input_latents"].shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        loss = FlowMatchSFTLoss(
            module.pipe,
            **inputs_shared,
            **inputs_posi,
            validation_timestep_id=timestep_id,
            validation_noise=noise,
        )
    return float(loss.detach().cpu())


def aggregate_losses(samples):
    finite = [row for row in samples if math.isfinite(row["loss"])]
    result = {
        "mean": statistics.fmean(row["loss"] for row in finite) if finite else None,
        "std": statistics.pstdev(row["loss"] for row in finite) if len(finite) > 1 else 0.0,
        "non_finite_count": len(samples) - len(finite),
        "num_samples": len(samples),
    }
    by_suite = defaultdict(list)
    by_task = defaultdict(list)
    for row in finite:
        by_suite[row["suite"]].append(row["loss"])
        by_task[row["prompt"]].append(row["loss"])
    result["suites"] = {
        key: {"mean": statistics.fmean(values), "count": len(values)}
        for key, values in sorted(by_suite.items())
    }
    result["tasks"] = {
        key: {"mean": statistics.fmean(values), "count": len(values)}
        for key, values in sorted(by_task.items())
    }
    return result


def evaluate_checkpoint(checkpoint_path, checkpoint_type, dataset, indices, seed, device, module=None):
    owns_module = module is None
    if owns_module:
        module = validation_module(checkpoint_path, checkpoint_type, device)
    elif checkpoint_type == "lora":
        if checkpoint_path is None:
            _zero_lora_parameters(module)
        else:
            load_lora_checkpoint(module, checkpoint_path)
    samples = []
    for selection_index, dataset_index in enumerate(indices):
        data = dataset[dataset_index]
        row = dataset.data[dataset_index]
        timestep_digest = stable_score(seed, "timestep", row)
        noise_digest = stable_score(seed, "noise", row)
        timestep_id = int.from_bytes(timestep_digest[:8], "little") % 1000
        noise_seed = int.from_bytes(noise_digest[:8], "little") % (2**31)
        loss = deterministic_loss(module, data, timestep_id, noise_seed)
        samples.append(
            {
                "sample_id": data["sample_id"],
                "suite": data["suite"],
                "prompt": data["prompt"],
                "timestep_id": timestep_id,
                "noise_seed": noise_seed,
                "loss": loss,
            }
        )
        print(
            f"[{checkpoint_type}] {selection_index + 1}/{len(indices)} "
            f"loss={loss:.6f} {data['sample_id']}"
        )
    aggregate = aggregate_losses(samples)
    if owns_module:
        del module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"aggregate": aggregate, "samples": samples}


def image_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def ssim(image_a, image_b):
    x = image_tensor(image_a)
    y = image_tensor(image_b)
    coordinates = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).expand(3, 1, 11, 11)
    mu_x = F.conv2d(x, window, groups=3, padding=5)
    mu_y = F.conv2d(y, window, groups=3, padding=5)
    sigma_x = F.conv2d(x * x, window, groups=3, padding=5) - mu_x**2
    sigma_y = F.conv2d(y * y, window, groups=3, padding=5) - mu_y**2
    sigma_xy = F.conv2d(x * y, window, groups=3, padding=5) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean())


def psnr(image_a, image_b):
    difference = image_tensor(image_a) - image_tensor(image_b)
    mse = float(torch.mean(difference**2))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def video_metrics(generated, ground_truth):
    if len(generated) != 17 or len(ground_truth) != 17:
        raise ValueError("Expected 17 generated and ground-truth frames")
    future_psnr = [psnr(generated[i], ground_truth[i]) for i in range(1, 17)]
    future_ssim = [ssim(generated[i], ground_truth[i]) for i in range(1, 17)]
    first = image_tensor(generated[0])
    temporal_change = [
        float(torch.mean(torch.abs(image_tensor(frame) - first)))
        for frame in generated[1:]
    ]
    return {
        "condition_psnr": psnr(generated[0], ground_truth[0]),
        "condition_ssim": ssim(generated[0], ground_truth[0]),
        "future_psnr_mean": statistics.fmean(future_psnr),
        "future_ssim_mean": statistics.fmean(future_ssim),
        "final_frame_psnr": future_psnr[-1],
        "final_frame_ssim": future_ssim[-1],
        "temporal_change_mean": statistics.fmean(temporal_change),
        "exact_static_copy": all(value == 0 for value in temporal_change),
    }


def safe_sample_name(sample_id):
    return sample_id.replace(":", "_")


def generate_videos(checkpoint_path, checkpoint_type, dataset, indices, output_dir, seed, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("base", None, "base"),
        ("tuned", checkpoint_path, checkpoint_type),
    ]
    report = {"samples": {}}
    for variant, variant_checkpoint, variant_type in variants:
        pipe = load_wan_pipe(variant_checkpoint, variant_type, device=device)
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for dataset_index in indices:
            data = dataset[dataset_index]
            sample_name = safe_sample_name(data["sample_id"])
            generation_seed = teacher_seed(
                seed,
                dataset.data[dataset_index]["suite_id"],
                data["episode_index"],
                data["anchor_frame"],
            )
            generated = pipe(
                prompt=data["prompt"],
                negative_prompt="",
                input_image=data["video"][0],
                height=480,
                width=480,
                num_frames=17,
                seed=generation_seed,
                rand_device="cpu",
                cfg_scale=5.0,
                num_inference_steps=20,
                sigma_shift=5.0,
                tiled=False,
            )
            video_path = variant_dir / f"{sample_name}.mp4"
            save_video(generated, str(video_path), fps=20, quality=7)
            sample_report = report["samples"].setdefault(
                data["sample_id"],
                {
                    "suite": data["suite"],
                    "prompt": data["prompt"],
                    "seed": generation_seed,
                },
            )
            sample_report[variant] = {
                "video": str(video_path),
                "metrics": video_metrics(generated, data["video"]),
            }
            if variant == "base":
                gt_dir = output_dir / "gt"
                gt_dir.mkdir(parents=True, exist_ok=True)
                gt_path = gt_dir / f"{sample_name}.mp4"
                save_video(data["video"], str(gt_path), fps=20, quality=7)
                sample_report["gt_video"] = str(gt_path)
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for variant in ("base", "tuned"):
        metrics = [sample[variant]["metrics"] for sample in report["samples"].values()]
        report[f"{variant}_aggregate"] = {
            key: statistics.fmean(item[key] for item in metrics)
            for key in (
                "condition_psnr",
                "condition_ssim",
                "future_psnr_mean",
                "future_ssim_mean",
                "final_frame_psnr",
                "final_frame_ssim",
                "temporal_change_mean",
            )
        }
        report[f"{variant}_aggregate"]["exact_static_copy_count"] = sum(
            item["exact_static_copy"] for item in metrics
        )
    return report


def write_video_index(output_dir, report):
    output_dir = Path(output_dir)
    rows = []
    for sample_id, data in report["samples"].items():
        links = []
        for variant in ("base", "tuned"):
            relative = os.path.relpath(data[variant]["video"], output_dir)
            links.append(f'<a href="{html.escape(relative)}">{variant}</a>')
        gt_relative = os.path.relpath(data["gt_video"], output_dir)
        links.append(f'<a href="{html.escape(gt_relative)}">gt</a>')
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(sample_id), html.escape(data["prompt"]), " / ".join(links)
            )
        )
    document = """<!doctype html><meta charset="utf-8"><title>LIBERO validation</title>
<table><thead><tr><th>Sample</th><th>Prompt</th><th>Videos</th></tr></thead>
<tbody>{}</tbody></table>""".format("\n".join(rows))
    with open(output_dir / "index.html", "w", encoding="utf-8") as file:
        file.write(document)


def atomic_json(path, data):
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LIBERO future-video checkpoints.")
    parser.add_argument(
        "--val-manifest",
        default="/data/jibaixu/Ckpts/wan2_2_ti2v_5b_libero/manifests/val.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/jibaixu/Ckpts/wan2_2_ti2v_5b_libero/validation",
    )
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--checkpoint-type", choices=("lora", "full"), default="lora")
    parser.add_argument("--selected-checkpoint", default=None)
    parser.add_argument("--mode", choices=("loss", "video", "both"), default="loss")
    parser.add_argument("--loss-samples", type=int, default=256)
    parser.add_argument("--video-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--approve-video-quality",
        action="store_true",
        help="Confirm that the generated comparison index has no general collapse, flicker, or color failure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.val_manifest)
    if not 1 <= args.loss_samples <= len(rows):
        raise ValueError("--loss-samples must be within the validation manifest")
    if not 4 <= args.video_samples <= len(rows):
        raise ValueError("--video-samples must be at least 4 and within the validation manifest")
    dataset = LiberoFutureWindowDataset(args.val_manifest)
    selected_checkpoint = args.selected_checkpoint
    report = None

    if args.mode == "video":
        report_path = output_dir / "validation_report.json"
        if not report_path.is_file():
            raise ValueError("Video validation requires an existing loss validation report")
        with open(report_path, "r", encoding="utf-8") as file:
            report = json.load(file)
        if not report.get("loss_passed", False):
            raise ValueError("Video validation requires a passing loss validation report")
        if selected_checkpoint is None:
            selected_checkpoint = report["selected_checkpoint"]["path"]
        selected_checkpoint = str(Path(selected_checkpoint).expanduser().resolve())
        if selected_checkpoint != report["selected_checkpoint"]["path"]:
            raise ValueError("Video checkpoint does not match the loss-selected checkpoint")
        if args.checkpoint_type != report["selected_checkpoint"]["type"]:
            raise ValueError("Video checkpoint type does not match the loss report")
        if checkpoint_sha256(selected_checkpoint) != report["selected_checkpoint"]["sha256"]:
            raise ValueError("Video checkpoint hash does not match the loss report")

    if args.mode in {"loss", "both"}:
        if not args.checkpoints:
            raise ValueError("Loss validation requires --checkpoints")
        indices = stratified_selection(rows, args.loss_samples, args.seed, "loss_selection")
        selection = [
            {
                "suite": rows[index]["suite"],
                "episode_index": rows[index]["episode_index"],
                "anchor_frame": rows[index]["anchor_frame"],
                "prompt": rows[index]["prompt"],
            }
            for index in indices
        ]
        atomic_json(output_dir / "loss_selection.json", selection)
        evaluations = []
        reusable_module = validation_module(None, "lora", args.device) if args.checkpoint_type == "lora" else None
        base_result = evaluate_checkpoint(
            None,
            "lora" if reusable_module is not None else "base",
            dataset,
            indices,
            args.seed,
            args.device,
            module=reusable_module,
        )
        evaluations.append(
            {
                "label": "base",
                "path": None,
                "sha256": None,
                **base_result,
            }
        )
        for checkpoint in args.checkpoints:
            checkpoint = str(Path(checkpoint).expanduser().resolve())
            result = evaluate_checkpoint(
                checkpoint,
                args.checkpoint_type,
                dataset,
                indices,
                args.seed,
                args.device,
                module=reusable_module,
            )
            evaluations.append(
                {
                    "label": Path(checkpoint).name,
                    "path": checkpoint,
                    "sha256": checkpoint_sha256(checkpoint),
                    **result,
                }
            )
            atomic_json(
                output_dir / "validation_report.partial.json",
                {"base_model": BASE_MODEL_ID, "evaluations": evaluations},
            )
        if reusable_module is not None:
            del reusable_module
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        tuned = min(
            evaluations[1:],
            key=lambda item: (
                item["aggregate"]["non_finite_count"] > 0,
                item["aggregate"]["mean"] if item["aggregate"]["mean"] is not None else math.inf,
            ),
        )
        base = evaluations[0]
        loss_passed = (
            base["aggregate"]["non_finite_count"] == 0
            and tuned["aggregate"]["non_finite_count"] == 0
            and tuned["aggregate"]["mean"] < base["aggregate"]["mean"]
        )
        report = {
            "base_model": BASE_MODEL_ID,
            "seed": args.seed,
            "manifest": str(Path(args.val_manifest).resolve()),
            "loss_passed": loss_passed,
            "video_passed": False,
            "passed": False,
            "selected_checkpoint": {
                "path": tuned["path"],
                "sha256": tuned["sha256"],
                "type": args.checkpoint_type,
                "mean_loss": tuned["aggregate"]["mean"],
            },
            "base_mean_loss": base["aggregate"]["mean"],
            "evaluations": evaluations,
        }
        selected_checkpoint = tuned["path"]
        atomic_json(output_dir / "validation_report.json", report)
        with open(output_dir / "validation_summary.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=("label", "path", "mean", "std", "non_finite_count")
            )
            writer.writeheader()
            for evaluation in evaluations:
                writer.writerow(
                    {
                        "label": evaluation["label"],
                        "path": evaluation["path"],
                        "mean": evaluation["aggregate"]["mean"],
                        "std": evaluation["aggregate"]["std"],
                        "non_finite_count": evaluation["aggregate"]["non_finite_count"],
                    }
                )

    if args.mode in {"video", "both"}:
        if selected_checkpoint is None:
            raise ValueError("Video validation requires --selected-checkpoint or loss selection")
        video_indices = stratified_selection(
            rows, args.video_samples, args.seed, "video_selection"
        )
        video_dir = output_dir / "videos"
        video_report = generate_videos(
            str(Path(selected_checkpoint).expanduser().resolve()),
            args.checkpoint_type,
            dataset,
            video_indices,
            video_dir,
            args.seed,
            args.device,
        )
        atomic_json(video_dir / "metrics.json", video_report)
        write_video_index(video_dir, video_report)
        tuned_metrics = video_report["tuned_aggregate"]
        numeric_keys = (
            "condition_psnr",
            "condition_ssim",
            "future_psnr_mean",
            "future_ssim_mean",
            "final_frame_psnr",
            "final_frame_ssim",
            "temporal_change_mean",
        )
        auto_checks = {
            "all_metrics_finite": all(
                math.isfinite(float(tuned_metrics[key])) for key in numeric_keys
            ),
            "no_exact_static_copies": tuned_metrics["exact_static_copy_count"] == 0,
        }
        report["video_validation"] = str(video_dir / "metrics.json")
        report["video_index"] = str(video_dir / "index.html")
        report["video_auto_checks"] = auto_checks
        report["video_quality_approved"] = args.approve_video_quality
        report["video_passed"] = all(auto_checks.values()) and args.approve_video_quality
        report["passed"] = report.get("loss_passed", False) and report["video_passed"]
        atomic_json(output_dir / "validation_report.json", report)


if __name__ == "__main__":
    main()
