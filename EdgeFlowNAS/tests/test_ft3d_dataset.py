"""Unit tests for FT3D dataset scanning/provider behavior."""

import tempfile
import unittest
from pathlib import Path

from efnas.data.dataloader_builder import build_ft3d_provider
from efnas.data.ft3d_dataset import resolve_ft3d_samples_from_folder


class TestFT3DDataset(unittest.TestCase):
    """Validate FT3D directory scanning without legacy list files."""

    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    def test_resolve_ft3d_samples_from_folder_detects_future_pairs(self) -> None:
        """Resolver should keep only frames with a valid next frame and future flow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._touch(root / "frames_cleanpass" / "TRAIN" / "A" / "0001" / "left" / "0006.png")
            self._touch(root / "frames_cleanpass" / "TRAIN" / "A" / "0001" / "left" / "0007.png")
            self._touch(
                root
                / "optical_flow"
                / "TRAIN"
                / "A"
                / "0001"
                / "into_future"
                / "left"
                / "OpticalFlowIntoFuture_0006_L.pfm"
            )

            samples = resolve_ft3d_samples_from_folder(base_path=str(root), split_dir="TRAIN")
            self.assertEqual(len(samples), 1)
            self.assertTrue(samples[0].endswith("0006.png"))

    def test_build_ft3d_provider_uses_train_test_roots(self) -> None:
        """Provider builder should support split roots like ../Datasets/optical_flow/TRAIN and TEST."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._touch(root / "frames_cleanpass" / "TRAIN" / "A" / "0001" / "left" / "0006.png")
            self._touch(root / "frames_cleanpass" / "TRAIN" / "A" / "0001" / "left" / "0007.png")
            self._touch(
                root
                / "optical_flow"
                / "TRAIN"
                / "A"
                / "0001"
                / "into_future"
                / "left"
                / "OpticalFlowIntoFuture_0006_L.pfm"
            )
            self._touch(root / "frames_cleanpass" / "TEST" / "A" / "0001" / "left" / "0006.png")
            self._touch(root / "frames_cleanpass" / "TEST" / "A" / "0001" / "left" / "0007.png")
            self._touch(
                root
                / "optical_flow"
                / "TEST"
                / "A"
                / "0001"
                / "into_future"
                / "left"
                / "OpticalFlowIntoFuture_0006_L.pfm"
            )
            config = {
                "runtime": {"seed": 42},
                "train": {"train_sampling_mode": "shuffle_no_replacement", "train_crop_mode": "random"},
                "data": {
                    "base_path": str(root),
                    "train_dir": "TRAIN",
                    "val_dir": "TEST",
                    "input_height": 480,
                    "input_width": 640,
                    "dataset": "FT3D",
                },
            }
            provider = build_ft3d_provider(config=config, split="train", seed_offset=0, provider_mode="train")
            self.assertEqual(len(provider), 1)
            self.assertIn("TRAIN", provider.source_dir)


if __name__ == "__main__":
    unittest.main()
