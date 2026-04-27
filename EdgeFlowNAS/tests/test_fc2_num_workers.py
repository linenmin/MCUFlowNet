import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from efnas.data.fc2_dataset import FC2BatchProvider


class TestFC2NumWorkers(unittest.TestCase):
    def test_parallel_provider_uses_deterministic_per_sample_seeds(self):
        samples = [f"sample_{idx}-img_0.png" for idx in range(4)]
        provider = FC2BatchProvider(
            samples=samples,
            crop_h=2,
            crop_w=2,
            seed=123,
            sampling_mode="sequential",
            crop_mode="random",
            num_workers=2,
        )

        def fake_load(sample_path, rng):
            value = len(sample_path) + rng.randint(0, 1000)
            arr = np.full((2, 2, 3), value, dtype=np.float32)
            flow = np.full((2, 2, 2), value, dtype=np.float32)
            return arr, arr, flow

        with mock.patch.object(provider, "_load_one_from_sample", side_effect=fake_load):
            batch_a = provider.next_batch(batch_size=4)[3]

        provider.reset_cursor(0)
        provider.rng.seed(123)
        with mock.patch.object(provider, "_load_one_from_sample", side_effect=fake_load):
            batch_b = provider.next_batch(batch_size=4)[3]

        np.testing.assert_array_equal(batch_a, batch_b)

    def test_builder_passes_worker_counts(self):
        from efnas.data.dataloader_builder import build_fc2_provider

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train"
            train.mkdir()
            config = {
                "runtime": {"seed": 42},
                "train": {},
                "data": {
                    "base_path": str(root),
                    "train_dir": "train",
                    "val_dir": "train",
                    "input_height": 2,
                    "input_width": 2,
                    "fc2_num_workers": 8,
                    "fc2_eval_num_workers": 3,
                },
            }
            train_provider = build_fc2_provider(config, split="train", provider_mode="train")
            val_provider = build_fc2_provider(config, split="val", provider_mode="eval")
            self.assertEqual(train_provider.num_workers, 8)
            self.assertEqual(val_provider.num_workers, 3)


if __name__ == "__main__":
    unittest.main()
