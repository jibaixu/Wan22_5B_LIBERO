import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "examples" / "wanvideo" / "model_training"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAINING_DIR))

try:
    import validate_libero_future
except ModuleNotFoundError as error:
    validate_libero_future = None
    VALIDATION_IMPORT_ERROR = error.name
else:
    VALIDATION_IMPORT_ERROR = None


class FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2, 2))
        self.lora_B = torch.nn.Parameter(torch.ones(2, 2))


class FakeLoRALoader:
    def convert_state_dict(self, state_dict):
        return state_dict


def fake_module():
    dit = FakeDiT()
    pipe = SimpleNamespace(
        dit=dit,
        device="cpu",
        torch_dtype=torch.float32,
        lora_loader=lambda **_: FakeLoRALoader(),
    )
    return SimpleNamespace(pipe=pipe, mapping_lora_state_dict=lambda state_dict: state_dict)


@unittest.skipIf(
    VALIDATION_IMPORT_ERROR is not None,
    f"Validation dependencies are unavailable: {VALIDATION_IMPORT_ERROR}",
)
class TestLoRAValidationReuse(unittest.TestCase):
    def test_replacing_checkpoint_overwrites_every_adapter_parameter(self):
        module = fake_module()
        first = {
            "lora_A": torch.full((2, 2), 2.0),
            "lora_B": torch.full((2, 2), 3.0),
        }
        second = {
            "lora_A": torch.full((2, 2), 4.0),
            "lora_B": torch.full((2, 2), 5.0),
        }

        with patch.object(validate_libero_future, "load_state_dict", side_effect=[first, second]):
            validate_libero_future.load_lora_checkpoint(module, "first.safetensors")
            validate_libero_future.load_lora_checkpoint(module, "second.safetensors")

        self.assertTrue(torch.equal(module.pipe.dit.lora_A, second["lora_A"]))
        self.assertTrue(torch.equal(module.pipe.dit.lora_B, second["lora_B"]))

    def test_base_evaluation_zeros_the_reusable_adapter(self):
        module = fake_module()
        validate_libero_future._zero_lora_parameters(module)

        self.assertEqual(torch.count_nonzero(module.pipe.dit.lora_A), 0)
        self.assertEqual(torch.count_nonzero(module.pipe.dit.lora_B), 0)

    def test_mismatched_checkpoint_is_rejected_before_loading(self):
        module = fake_module()
        incomplete = {"lora_A": torch.ones(2, 2)}

        with patch.object(validate_libero_future, "load_state_dict", return_value=incomplete):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_libero_future.load_lora_checkpoint(module, "bad.safetensors")


if __name__ == "__main__":
    unittest.main()
