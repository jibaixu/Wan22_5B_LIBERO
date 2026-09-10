#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.wanvideo.model_training.libero_future_dataset import (  # noqa: E402
    LIBERO_SUITES,
    discover_libero_episodes,
)


EXPECTED_EPISODES = 1693
EXPECTED_WINDOWS_STRIDE4 = 68991
EXPECTED_ANCHORS_STRIDE8 = 34918


def _split_score(seed, suite, task, episode_index):
    value = f"{seed}\0{suite}\0{task}\0{episode_index}".encode("utf-8")
    return hashlib.sha256(value).digest()


def split_episodes(episodes, val_ratio=0.1, split_seed=42):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["suite"], episode["prompt"])].append(episode)

    val_keys = set()
    for (suite, task), rows in sorted(groups.items()):
        rows = sorted(
            rows,
            key=lambda row: _split_score(
                split_seed, suite, task, row["episode_index"]
            ),
        )
        num_val = max(1, math.floor(len(rows) * val_ratio + 0.5))
        if num_val >= len(rows):
            raise ValueError(f"Validation split consumes stratum {(suite, task)!r}")
        val_keys.update((row["suite"], row["episode_index"]) for row in rows[:num_val])

    train = []
    val = []
    for episode in episodes:
        key = (episode["suite"], episode["episode_index"])
        (val if key in val_keys else train).append(episode)
    return train, val


def episode_windows(episode, stride, num_frames, split):
    horizon = (num_frames - 1) * int(episode.get("temporal_stride", 1))
    for anchor_frame in range(0, episode["episode_length"], stride):
        yield {
            "suite": episode["suite"],
            "suite_id": episode["suite_id"],
            "dataset_root": episode["dataset_root"],
            "episode_index": episode["episode_index"],
            "episode_length": episode["episode_length"],
            "anchor_frame": anchor_frame,
            "video_path": episode["video_path"],
            "prompt": episode["prompt"],
            "valid_future_frames": min(
                horizon, episode["episode_length"] - 1 - anchor_frame
            ),
            "temporal_stride": int(episode.get("temporal_stride", 1)),
            "split": split,
        }


def build_manifest_rows(episodes, stride, num_frames, split):
    rows = []
    for episode in sorted(
        episodes, key=lambda row: (row["suite_id"], row["episode_index"])
    ):
        rows.extend(episode_windows(episode, stride, num_frames, split))
    return rows


def _atomic_write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def build_manifests(
    data_root,
    output_dir,
    train_stride=4,
    val_stride=4,
    val_ratio=0.1,
    split_seed=42,
    video_key="observation.images.image",
    num_frames=17,
    temporal_stride=1,
    episode_manifest=None,
    verify_expected_counts=True,
):
    if train_stride <= 0 or val_stride <= 0 or num_frames <= 0:
        raise ValueError("strides and num_frames must be positive")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    if episode_manifest:
        episodes = [json.loads(line) for line in Path(episode_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        for suite_id, episode in enumerate(episodes):
            episode.setdefault("suite_id", suite_id)
            episode.setdefault("temporal_stride", temporal_stride)
    else:
        episodes = discover_libero_episodes(data_root, video_key=video_key)
    train_episodes, val_episodes = split_episodes(episodes, val_ratio, split_seed)
    train_rows = build_manifest_rows(train_episodes, train_stride, num_frames, "train")
    val_rows = build_manifest_rows(val_episodes, val_stride, num_frames, "val")
    all_rows = sorted(
        train_rows + val_rows,
        key=lambda row: (row["suite_id"], row["episode_index"], row["anchor_frame"]),
    )

    train_keys = {(row["suite"], row["episode_index"]) for row in train_episodes}
    val_keys = {(row["suite"], row["episode_index"]) for row in val_episodes}
    if train_keys & val_keys:
        raise AssertionError("Train and validation episode sets overlap")

    stride8_anchors = sum(
        math.ceil(episode["episode_length"] / 8) for episode in episodes
    )
    if verify_expected_counts:
        assert len(episodes) == EXPECTED_EPISODES, len(episodes)
        if train_stride == val_stride == 4:
            assert len(all_rows) == EXPECTED_WINDOWS_STRIDE4, len(all_rows)
        assert stride8_anchors == EXPECTED_ANCHORS_STRIDE8, stride8_anchors

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_dir / "train.jsonl", train_rows)
    _atomic_write_jsonl(output_dir / "val.jsonl", val_rows)
    _atomic_write_jsonl(output_dir / "all.jsonl", all_rows)

    suite_summary = {}
    for suite in LIBERO_SUITES:
        suite_episodes = [row for row in episodes if row["suite"] == suite]
        suite_train = [row for row in train_episodes if row["suite"] == suite]
        suite_val = [row for row in val_episodes if row["suite"] == suite]
        suite_summary[suite] = {
            "episodes": len(suite_episodes),
            "frames": sum(row["episode_length"] for row in suite_episodes),
            "train_episodes": len(suite_train),
            "val_episodes": len(suite_val),
            "train_windows": sum(row["suite"] == suite for row in train_rows),
            "val_windows": sum(row["suite"] == suite for row in val_rows),
            "stride8_anchors": sum(
                math.ceil(row["episode_length"] / 8) for row in suite_episodes
            ),
        }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(Path(data_root).expanduser().resolve()),
        "output_dir": str(output_dir),
        "parameters": {
            "train_stride": train_stride,
            "val_stride": val_stride,
            "val_ratio": val_ratio,
            "split_seed": split_seed,
            "video_key": video_key,
            "num_frames": num_frames,
        },
        "totals": {
            "episodes": len(episodes),
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
            "train_windows": len(train_rows),
            "val_windows": len(val_rows),
            "all_windows": len(all_rows),
            "stride8_anchors": stride8_anchors,
        },
        "suites": suite_summary,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build LIBERO future-video manifests.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--val-stride", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--episode-manifest", default=None,
                        help="Canonical RoboTwin episode manifest.")
    parser.add_argument("--skip-expected-count-checks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_manifests(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        video_key=args.video_key,
        num_frames=args.num_frames,
        temporal_stride=args.temporal_stride,
        episode_manifest=args.episode_manifest,
        verify_expected_counts=not args.skip_expected_count_checks,
    )
    print(json.dumps(result["totals"], indent=2, sort_keys=True))
