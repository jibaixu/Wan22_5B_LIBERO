import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "examples" / "wanvideo" / "model_training"
INFERENCE_DIR = REPO_ROOT / "examples" / "wanvideo" / "model_inference"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(INFERENCE_DIR))

from libero_future_dataset import (  # noqa: E402
    LiberoFutureWindowDataset,
    discover_libero_episodes,
    future_frame_ids,
)
from tools.build_libero_future_manifest import (  # noqa: E402
    build_manifest_rows,
    split_episodes,
)
from extract_libero_future_features import (  # noqa: E402
    REQUIRED_METADATA,
    cache_metadata,
    expected_anchor_frames,
    expected_valid_future_frames,
    validate_episode_cache,
)
from diffsynth.diffusion.base_pipeline import PipelineUnit  # noqa: E402
from diffsynth.pipelines.wan_video import (  # noqa: E402
    WanFeatureCaptureSpec,
    WanVideoPipeline,
    _WanFeatureCollector,
)


DATA_ROOT = Path("/data/jibaixu/Datasets/LIBERO/libero_no_noop")


class TestManifestAndDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_ROOT.is_dir():
            raise unittest.SkipTest(f"LIBERO data is unavailable at {DATA_ROOT}")
        cls.episodes = discover_libero_episodes(DATA_ROOT)

    def test_manifest_counts_split_and_reproducibility(self):
        train_a, val_a = split_episodes(self.episodes, val_ratio=0.1, split_seed=42)
        train_b, val_b = split_episodes(self.episodes, val_ratio=0.1, split_seed=42)
        train_keys = [(row["suite"], row["episode_index"]) for row in train_a]
        val_keys = [(row["suite"], row["episode_index"]) for row in val_a]
        self.assertEqual(train_keys, [(row["suite"], row["episode_index"]) for row in train_b])
        self.assertEqual(val_keys, [(row["suite"], row["episode_index"]) for row in val_b])
        self.assertFalse(set(train_keys) & set(val_keys))
        self.assertEqual(len(self.episodes), 1693)

        train_rows = build_manifest_rows(train_a, 4, 17, "train")
        val_rows = build_manifest_rows(val_a, 4, 17, "val")
        self.assertEqual(len(train_rows) + len(val_rows), 68991)
        self.assertEqual(
            sum(math.ceil(row["episode_length"] / 8) for row in self.episodes),
            34918,
        )

    def test_frame_ids_pad_without_crossing_episode(self):
        self.assertEqual(future_frame_ids(3, 30), list(range(3, 20)))
        self.assertEqual(future_frame_ids(8, 10), [8, 9] + [9] * 15)
        self.assertEqual(future_frame_ids(9, 10), [9] * 17)

    def test_real_tail_window_is_rgb_square_and_repeats_last_frame(self):
        episode = next(row for row in self.episodes if row["suite"] == "libero_goal")
        row = {
            **episode,
            "anchor_frame": episode["episode_length"] - 1,
            "valid_future_frames": 0,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = Path(temporary_dir) / "tail.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = LiberoFutureWindowDataset(manifest, max_open_readers=1)
            sample = dataset[0]
            self.assertEqual(len(sample["video"]), 17)
            self.assertTrue(all(image.mode == "RGB" for image in sample["video"]))
            self.assertTrue(all(image.size == (480, 480) for image in sample["video"]))
            reference = np.asarray(sample["video"][0])
            self.assertTrue(all(np.array_equal(reference, np.asarray(image)) for image in sample["video"][1:]))
            dataset.close()


class TestCacheValidation(unittest.TestCase):
    def setUp(self):
        self.episode = {
            "suite": "libero_goal",
            "suite_id": 1,
            "episode_index": 7,
            "episode_length": 1,
        }
        self.metadata = cache_metadata(self.episode, "/tmp/checkpoint.safetensors", "abc", 42)
        self.anchors = expected_anchor_frames(1)
        self.valid = expected_valid_future_frames(1, self.anchors)
        self.features = torch.zeros((1, 1125, 3072), dtype=torch.bfloat16)
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_dir.name) / "episode.safetensors"

    def tearDown(self):
        self.temporary_dir.cleanup()

    def write(self, tensors=None, metadata=None):
        save_file(
            tensors
            or {
                "features": self.features,
                "anchor_frames": self.anchors,
                "valid_future_frames": self.valid,
            },
            self.path,
            metadata=metadata or self.metadata,
        )

    def test_accepts_complete_cache(self):
        self.write()
        self.assertEqual(
            validate_episode_cache(self.path, self.episode, self.metadata), (True, "ok")
        )

    def test_rejects_bad_dtype_shape_metadata_keys_and_anchor(self):
        cases = [
            (
                {
                    "features": self.features.float(),
                    "anchor_frames": self.anchors,
                    "valid_future_frames": self.valid,
                },
                self.metadata,
            ),
            (
                {
                    "features": torch.zeros((1, 1124, 3072), dtype=torch.bfloat16),
                    "anchor_frames": self.anchors,
                    "valid_future_frames": self.valid,
                },
                self.metadata,
            ),
            (
                {
                    "features": self.features,
                    "anchor_frames": self.anchors,
                    "valid_future_frames": self.valid,
                },
                {**self.metadata, "complete": "false"},
            ),
            (
                {"features": self.features, "anchor_frames": self.anchors},
                self.metadata,
            ),
            (
                {
                    "features": self.features,
                    "anchor_frames": torch.tensor([1], dtype=torch.int64),
                    "valid_future_frames": self.valid,
                },
                self.metadata,
            ),
        ]
        for tensors, metadata in cases:
            with self.subTest(tensors=list(tensors), complete=metadata.get("complete")):
                self.write(tensors, metadata)
                valid, _ = validate_episode_cache(self.path, self.episode, self.metadata)
                self.assertFalse(valid)


class _LatentUnit(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("latents",))

    def process(self, pipe):
        return {"latents": torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32)}


class _FakeScheduler:
    def set_timesteps(self, count, **kwargs):
        self.timesteps = torch.arange(count - 1, -1, -1, dtype=torch.float32)

    def step(self, noise_pred, timestep, latents):
        return latents


class _FakeVAE:
    def __init__(self):
        self.decode_calls = 0

    def decode(self, *args, **kwargs):
        self.decode_calls += 1
        return "decoded"


class TestFeatureCapture(unittest.TestCase):
    def fake_pipe(self):
        pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
        pipe.scheduler = _FakeScheduler()
        pipe.units = [_LatentUnit()]
        pipe.post_units = []
        pipe.in_iteration_models = ("dit",)
        pipe.dit = types.SimpleNamespace(
            blocks=[types.SimpleNamespace(num_heads=24) for _ in range(30)],
            dim=3072,
            patch_size=(1, 2, 2),
        )
        pipe.vae = _FakeVAE()
        pipe.load_models_to_device = lambda names: None
        return pipe

    def test_collector_uses_exact_block_and_shape(self):
        collector = _WanFeatureCollector(WanFeatureCaptureSpec())
        for block_index in range(30):
            hidden = torch.full(
                (1, 1125, 3072), block_index, dtype=torch.bfloat16
            )
            collector.collect(block_index, hidden, (5, 15, 15))
        result = collector.result
        self.assertEqual(result.feature.shape, (1125, 3072))
        self.assertEqual(result.feature.dtype, torch.bfloat16)
        self.assertEqual(result.feature.device.type, "cpu")
        self.assertTrue(torch.all(result.feature == 20))
        self.assertEqual(result.token_order, "temporal_height_width")

    def test_positive_final_step_only_and_no_vae_decode(self):
        pipe = self.fake_pipe()
        calls = []

        def model_fn(dit, latents, timestep, feature_collector=None, **kwargs):
            branch = "positive" if "prompt" in kwargs else "negative"
            calls.append((branch, feature_collector is not None))
            if feature_collector is not None:
                hidden = torch.zeros((1, 1125, 3072), dtype=torch.bfloat16)
                feature_collector.collect(20, hidden, (5, 15, 15))
            return torch.zeros_like(latents)

        pipe.model_fn = model_fn
        result = pipe.extract_dit_feature(
            prompt="put the bowl on the plate",
            negative_prompt="",
            input_image=Image.new("RGB", (480, 480)),
            num_inference_steps=20,
            cfg_scale=5.0,
            progress_bar_cmd=lambda values: values,
        )
        self.assertEqual(sum(captured for _, captured in calls), 1)
        self.assertEqual(calls[-2], ("positive", True))
        self.assertEqual(calls[-1], ("negative", False))
        self.assertEqual(result.feature.shape, (1125, 3072))
        self.assertEqual(pipe.vae.decode_calls, 0)

    def test_normal_inference_return_and_decoder_are_unchanged(self):
        pipe = self.fake_pipe()

        def model_fn(dit, latents, timestep, **kwargs):
            self.assertNotIn("feature_collector", kwargs)
            return torch.zeros_like(latents)

        pipe.model_fn = model_fn
        output = pipe(
            prompt="test",
            input_image=None,
            num_inference_steps=1,
            cfg_scale=1.0,
            output_type="floatpoint",
            progress_bar_cmd=lambda values: values,
        )
        self.assertEqual(output, "decoded")
        self.assertEqual(pipe.vae.decode_calls, 1)


if __name__ == "__main__":
    unittest.main()
