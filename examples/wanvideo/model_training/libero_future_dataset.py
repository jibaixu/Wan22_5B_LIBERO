import json
from collections import OrderedDict
from pathlib import Path

import imageio
import torch
from PIL import Image


LIBERO_SUITES = OrderedDict(
    (
        ("libero_10", "libero_10_no_noops_1.0.0_lerobot"),
        ("libero_goal", "libero_goal_no_noops_1.0.0_lerobot"),
        ("libero_object", "libero_object_no_noops_1.0.0_lerobot"),
        ("libero_spatial", "libero_spatial_no_noops_1.0.0_lerobot"),
    )
)
SUITE_IDS = {suite: suite_id for suite_id, suite in enumerate(LIBERO_SUITES)}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def episode_video_path(dataset_root, info, episode_index, video_key):
    chunk_size = int(info["chunks_size"])
    relative_path = info["video_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=video_key,
    )
    return str((Path(dataset_root) / relative_path).resolve())


def discover_libero_episodes(data_root, video_key="observation.images.image"):
    data_root = Path(data_root).expanduser().resolve()
    episodes = []
    for suite, directory_name in LIBERO_SUITES.items():
        dataset_root = data_root / directory_name
        info_path = dataset_root / "meta" / "info.json"
        episodes_path = dataset_root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata under {dataset_root}")

        with open(info_path, "r", encoding="utf-8") as file:
            info = json.load(file)
        if video_key not in info.get("features", {}):
            raise KeyError(f"Video key {video_key!r} is not available in {dataset_root}")
        if int(info["fps"]) != 20:
            raise ValueError(f"Expected 20 FPS in {dataset_root}, got {info['fps']}")

        suite_episodes = read_jsonl(episodes_path)
        if len(suite_episodes) != int(info["total_episodes"]):
            raise ValueError(f"Episode count mismatch in {dataset_root}")
        for episode in suite_episodes:
            prompts = episode.get("tasks", [])
            if len(prompts) != 1 or not prompts[0]:
                raise ValueError(
                    f"Expected one task for {suite} episode {episode.get('episode_index')}"
                )
            episode_index = int(episode["episode_index"])
            video_path = episode_video_path(dataset_root, info, episode_index, video_key)
            if not Path(video_path).is_file():
                raise FileNotFoundError(video_path)
            episodes.append(
                {
                    "suite": suite,
                    "suite_id": SUITE_IDS[suite],
                    "dataset_root": str(dataset_root),
                    "episode_index": episode_index,
                    "episode_length": int(episode["length"]),
                    "video_path": video_path,
                    "prompt": prompts[0],
                    "video_key": video_key,
                    "chunks_size": int(info["chunks_size"]),
                }
            )
    return episodes


def future_frame_ids(anchor_frame, episode_length, num_frames=17, temporal_stride=1):
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if not 0 <= anchor_frame < episode_length:
        raise ValueError(
            f"anchor_frame {anchor_frame} is outside episode length {episode_length}"
        )
    return [min(anchor_frame + offset * temporal_stride, episode_length - 1) for offset in range(num_frames)]


class LiberoFutureWindowDataset(torch.utils.data.Dataset):
    """Online fixed-window reader for LIBERO LeRobot episode videos."""

    def __init__(
        self,
        manifest_path,
        repeat=1,
        height=480,
        width=480,
        num_frames=17,
        max_open_readers=4,
        max_samples=None,
        temporal_stride=1,
    ):
        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self.data = read_jsonl(self.manifest_path)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.data = self.data[:max_samples]
        if not self.data:
            raise ValueError(f"No rows found in {self.manifest_path}")
        if repeat <= 0 or max_open_readers <= 0:
            raise ValueError("repeat and max_open_readers must be positive")

        self.repeat = int(repeat)
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.temporal_stride = int(temporal_stride)
        self.max_open_readers = int(max_open_readers)
        self.load_from_cache = False
        self._readers = OrderedDict()

    def __len__(self):
        return len(self.data) * self.repeat

    def _get_reader(self, video_path):
        reader = self._readers.pop(video_path, None)
        if reader is None:
            reader = imageio.get_reader(video_path)
        self._readers[video_path] = reader
        while len(self._readers) > self.max_open_readers:
            _, old_reader = self._readers.popitem(last=False)
            old_reader.close()
        return reader

    def _read_frame(self, reader, frame_id):
        frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
        return frame.resize((self.width, self.height), resample=Image.Resampling.BILINEAR)

    def __getitem__(self, index):
        row = self.data[index % len(self.data)]
        anchor_frame = int(row["anchor_frame"])
        episode_length = int(row["episode_length"])
        # The manifest is the source of truth for the sampling interval.  This
        # matters for future-window validation: a RoboTwin row can cover 96
        # source frames with 17 samples (temporal_stride=6), while older
        # manifests may omit the field and should retain the constructor
        # default for backwards compatibility.
        temporal_stride = int(row.get("temporal_stride", self.temporal_stride))
        if temporal_stride <= 0:
            raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")
        frame_ids = future_frame_ids(anchor_frame, episode_length, self.num_frames, temporal_stride)
        reader = self._get_reader(row["video_path"])
        frames = [self._read_frame(reader, frame_id) for frame_id in frame_ids]
        suite = str(row["suite"])
        episode_index = int(row["episode_index"])
        sample_id = f"{suite}:episode_{episode_index:06d}:frame_{anchor_frame:06d}"
        return {
            "video": frames,
            "prompt": str(row["prompt"]),
            "sample_id": sample_id,
            "suite": suite,
            "episode_index": episode_index,
            "anchor_frame": anchor_frame,
            "temporal_stride": temporal_stride,
            "valid_future_frames": int(row["valid_future_frames"]),
        }

    def close(self):
        while self._readers:
            _, reader = self._readers.popitem(last=False)
            reader.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"] = OrderedDict()
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
