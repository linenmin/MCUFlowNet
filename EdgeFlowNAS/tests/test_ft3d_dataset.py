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
            frames_root = root / "frames_cleanpass"
            flow_root = root / "optical_flow"
            self._touch(frames_root / "TRAIN" / "A" / "0001" / "left" / "0006.png")
            self._touch(frames_root / "TRAIN" / "A" / "0001" / "left" / "0007.png")
            self._touch(
                flow_root
                / "TRAIN"
                / "A"
                / "0001"
                / "into_future"
                / "left"
                / "OpticalFlowIntoFuture_0006_L.pfm"
            )

            samples = resolve_ft3d_samples_from_folder(
                frames_base_path=str(frames_root),
                flow_base_path=str(flow_root),
                split_dir="TRAIN",
            )
            self.assertEqual(len(samples), 1)
            self.assertTrue(samples[0].endswith("0006.png"))

    def test_build_ft3d_provider_uses_separate_frame_and_flow_roots(self) -> None:
        """Provider builder should support explicit frame and flow roots."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frames_root = root / "frames_cleanpass"
            flow_root = root / "optical_flow"
            self._touch(frames_root / "TRAIN" / "A" / "0001" / "left" / "0006.png")
            self._touch(frames_root / "TRAIN" / "A" / "0001" / "left" / "0007.png")
            self._touch(
                flow_root
                / "TRAIN"
                / "A"
                / "0001"
                / "into_future"
                / "left"
                / "OpticalFlowIntoFuture_0006_L.pfm"
            )
            self._touch(frames_root / "TEST" / "A" / "0001" / "left" / "0006.png")
            self._touch(frames_root / "TEST" / "A" / "0001" / "left" / "0007.png")
            self._touch(
                flow_root
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
                    "ft3d_frames_base_path": str(frames_root),
                    "ft3d_flow_base_path": str(flow_root),
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
