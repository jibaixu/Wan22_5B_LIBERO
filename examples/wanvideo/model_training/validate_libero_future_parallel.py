#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_libero_future import (  # noqa: E402
    BASE_MODEL_ID,
    LiberoFutureWindowDataset,
    atomic_json,
    checkpoint_sha256,
    evaluate_checkpoint,
    prepare_model_cache,
    read_jsonl,
    stratified_selection,
    validation_module,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel checkpoint-level LIBERO future-video loss validation."
    )
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
    parser.add_argument("--mode", choices=("loss",), default="loss")
    parser.add_argument("--loss-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help=(
            "Optional node-local directory for a one-time copy of the Wan "
            "weights before any rank loads a model."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device for single-process use. Under torchrun, leave unset so each rank "
            "uses cuda:${LOCAL_RANK}."
        ),
    )
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def rank_device(args, local_rank):
    if args.device is not None:
        return args.device
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def loss_selection(rows, count, seed):
    indices = stratified_selection(rows, count, seed, "loss_selection")
    selection = [
        {
            "suite": rows[index]["suite"],
            "episode_index": rows[index]["episode_index"],
            "anchor_frame": rows[index]["anchor_frame"],
            "prompt": rows[index]["prompt"],
        }
        for index in indices
    ]
    return indices, selection


def rank_checkpoint_assignments(checkpoints, rank, world_size):
    return [
        (index, checkpoint)
        for index, checkpoint in enumerate(checkpoints)
        if index % world_size == rank
    ]


def write_summary_csv(path, evaluations):
    with open(path, "w", newline="", encoding="utf-8") as file:
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


def merge_rank_results(output_dir, checkpoints, base_evaluation, checkpoint_type, seed, manifest):
    rank_dir = output_dir / "loss_rank_results"
    by_index = {}
    for rank_path in sorted(rank_dir.glob("rank-*.json")):
        with open(rank_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        for evaluation in payload["evaluations"]:
            by_index[int(evaluation["checkpoint_index"])] = evaluation

    missing = sorted(set(range(len(checkpoints))) - set(by_index))
    if missing:
        raise RuntimeError(f"Missing validation results for checkpoint indices: {missing}")

    checkpoint_evaluations = [by_index[index] for index in range(len(checkpoints))]
    evaluations = [base_evaluation] + checkpoint_evaluations
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
        "seed": seed,
        "manifest": str(Path(manifest).resolve()),
        "loss_passed": loss_passed,
        "video_passed": False,
        "passed": False,
        "selected_checkpoint": {
            "path": tuned["path"],
            "sha256": tuned["sha256"],
            "type": checkpoint_type,
            "mean_loss": tuned["aggregate"]["mean"],
        },
        "base_mean_loss": base["aggregate"]["mean"],
        "evaluations": evaluations,
    }
    atomic_json(output_dir / "validation_report.json", report)
    write_summary_csv(output_dir / "validation_summary.csv", evaluations)
    atomic_json(
        output_dir / "validation_report.partial.json",
        {"base_model": BASE_MODEL_ID, "evaluations": evaluations},
    )


def main():
    args = parse_args()
    if not args.checkpoints:
        raise ValueError("Parallel loss validation requires --checkpoints")
    rank, local_rank, world_size = init_distributed()
    device = rank_device(args, local_rank)

    if args.model_cache_dir:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = prepare_model_cache(
            args.model_cache_dir
        )
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        if world_size > 1:
            dist.barrier()

    output_dir = Path(args.output_dir).expanduser().resolve()
    rank_dir = output_dir / "loss_rank_results"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        rank_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in rank_dir.glob("rank-*.json"):
            stale_path.unlink()
    if world_size > 1:
        dist.barrier()

    rows = read_jsonl(args.val_manifest)
    if not 1 <= args.loss_samples <= len(rows):
        raise ValueError("--loss-samples must be within the validation manifest")
    indices, selection = loss_selection(rows, args.loss_samples, args.seed)
    if rank == 0:
        atomic_json(output_dir / "loss_selection.json", selection)

    dataset = LiberoFutureWindowDataset(args.val_manifest)
    checkpoints = [str(Path(checkpoint).expanduser().resolve()) for checkpoint in args.checkpoints]
    assignments = rank_checkpoint_assignments(checkpoints, rank, world_size)

    # Keep one base pipeline and one reusable LoRA adapter per active rank.
    # Ranks without checkpoints do not need to load the large base model.
    reusable_module = None
    if args.checkpoint_type == "lora" and (rank == 0 or assignments):
        reusable_module = validation_module(None, "lora", device)

    base_evaluation = None
    if rank == 0:
        base_result = evaluate_checkpoint(
            None,
            "lora" if reusable_module is not None else "base",
            dataset,
            indices,
            args.seed,
            device,
            module=reusable_module,
        )
        base_evaluation = {
            "label": "base",
            "path": None,
            "sha256": None,
            **base_result,
        }
        atomic_json(output_dir / "base_loss_result.json", base_evaluation)

    rank_evaluations = []
    for checkpoint_index, checkpoint in assignments:
        result = evaluate_checkpoint(
            checkpoint,
            args.checkpoint_type,
            dataset,
            indices,
            args.seed,
            device,
            module=reusable_module,
        )
        rank_evaluations.append(
            {
                "checkpoint_index": checkpoint_index,
                "label": Path(checkpoint).name,
                "path": checkpoint,
                "sha256": checkpoint_sha256(checkpoint),
                **result,
            }
        )
        atomic_json(
            rank_dir / f"rank-{rank:03d}.json",
            {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "device": device,
                "evaluations": rank_evaluations,
            },
        )

    if reusable_module is not None:
        del reusable_module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not assignments:
        atomic_json(
            rank_dir / f"rank-{rank:03d}.json",
            {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "device": device,
                "evaluations": [],
            },
        )

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        if base_evaluation is None:
            with open(output_dir / "base_loss_result.json", "r", encoding="utf-8") as file:
                base_evaluation = json.load(file)
        merge_rank_results(
            output_dir=output_dir,
            checkpoints=checkpoints,
            base_evaluation=base_evaluation,
            checkpoint_type=args.checkpoint_type,
            seed=args.seed,
            manifest=args.val_manifest,
        )

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
