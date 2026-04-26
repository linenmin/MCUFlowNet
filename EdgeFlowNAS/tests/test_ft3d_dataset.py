"""Unit tests for FT3D dataset scanning/provider behavior."""

import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from efnas.data.dataloader_builder import build_ft3d_provider
from efnas.data.ft3d_dataset import (
    _apply_spatial_augment,
    _build_ft3d_triplet,
    FT3DBatchProvider,
    resolve_ft3d_samples_from_folder,
)


class TestFT3DDataset(unittest.TestCase):
    """Validate FT3D directory scanning without legacy list files."""

    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    def test_resolve_ft3d_samples_from_folder_detects_future_and_past_pairs(self) -> None:
        """Resolver should include both future and past supervision when requested."""
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
            self._touch(
                flow_root
                / "TRAIN"
                / "A"
                / "0001"
                / "into_past"
                / "left"
                / "OpticalFlowIntoPast_0007_L.pfm"
            )

            samples = resolve_ft3d_samples_from_folder(
                frames_base_path=str(frames_root),
                flow_base_path=str(flow_root),
                split_dir="TRAIN",
                include_directions=("into_future", "into_past"),
            )
            self.assertEqual(len(samples), 2)
            self.assertEqual(
                sorted(Path(sample[2]).name for sample in samples),
                ["OpticalFlowIntoFuture_0006_L.pfm", "OpticalFlowIntoPast_0007_L.pfm"],
            )

    def test_build_ft3d_provider_merges_multiple_frame_roots(self) -> None:
        """Provider builder should merge cleanpass/finalpass roots into one train pool."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            clean_root = root / "frames_cleanpass"
            final_root = root / "frames_finalpass"
            flow_root = root / "optical_flow"
            self._touch(clean_root / "TRAIN" / "A" / "0001" / "left" / "0006.png")
            self._touch(clean_root / "TRAIN" / "A" / "0001" / "left" / "0007.png")
            self._touch(final_root / "TRAIN" / "A" / "0002" / "left" / "0006.png")
            self._touch(final_root / "TRAIN" / "A" / "0002" / "left" / "0007.png")
            self._touch(
                flow_root
                / "TRAIN"
                / "A"
                / "0001"
                / "into_future"
                / "left"
                / "OpticalFlowIntoFuture_0006_L.pfm"
            )
            self._touch(
                flow_root
                / "TRAIN"
                / "A"
                / "0002"
                / "into_future"
                / "left"
                / "OpticalFlowIntoFuture_0006_L.pfm"
            )
            self._touch(clean_root / "TEST" / "A" / "0001" / "left" / "0006.png")
            self._touch(clean_root / "TEST" / "A" / "0001" / "left" / "0007.png")
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
                    "ft3d_frames_base_paths": [str(clean_root), str(final_root)],
                    "ft3d_flow_base_path": str(flow_root),
                    "train_dir": "TRAIN",
                    "val_dir": "TEST",
                    "input_height": 480,
                    "input_width": 640,
                    "dataset": "FT3D",
                },
            }
            provider = build_ft3d_provider(config=config, split="train", seed_offset=0, provider_mode="train")
            self.assertEqual(len(provider), 2)
            self.assertIn("TRAIN", provider.source_dir)
            self.assertEqual(provider.num_workers, 1)

    def test_build_ft3d_provider_forwards_num_workers(self) -> None:
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
            config = {
                "runtime": {"seed": 42},
                "train": {"train_sampling_mode": "shuffle_no_replacement", "train_crop_mode": "random"},
                "data": {
                    "ft3d_frames_base_path": str(frames_root),
                    "ft3d_flow_base_path": str(flow_root),
                    "train_dir": "TRAIN",
                    "val_dir": "TRAIN",
                    "input_height": 480,
                    "input_width": 640,
                    "dataset": "FT3D",
                    "ft3d_num_workers": 6,
                },
            }
            provider = build_ft3d_provider(config=config, split="train", seed_offset=0, provider_mode="train")
            self.assertEqual(provider.num_workers, 6)

    def test_build_ft3d_eval_provider_defaults_to_train_worker_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frames_root = root / "frames_cleanpass"
            flow_root = root / "optical_flow"
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
                    "train_dir": "TEST",
                    "val_dir": "TEST",
                    "input_height": 480,
                    "input_width": 640,
                    "dataset": "FT3D",
                    "ft3d_num_workers": 6,
                },
            }
            provider = build_ft3d_provider(config=config, split="val", seed_offset=0, provider_mode="eval")
            self.assertEqual(provider.num_workers, 6)

    def test_ft3d_provider_uses_parallel_loader_when_num_workers_gt_one(self) -> None:
        provider = FT3DBatchProvider(
            samples=[("a.png", "b.png", "c.pfm")],
            crop_h=4,
            crop_w=4,
            seed=42,
            sampling_mode="random",
            crop_mode="random",
            flow_divisor=12.5,
            augment_cfg={"enabled": False},
            num_workers=3,
        )

        class _FakeExecutor:
            def __init__(self, *args, **kwargs):
                self.map_called = False

            def map(self, fn, jobs):
                self.map_called = True
                return [fn(job) for job in jobs]

            def shutdown(self, wait=True):
                return None

        fake_executor = _FakeExecutor()
        with mock.patch("efnas.data.ft3d_dataset.ThreadPoolExecutor", return_value=fake_executor):
            with mock.patch.object(
                provider,
                "_load_one_from_sample",
                side_effect=lambda sample_path, rng: (
                    np.zeros((4, 4, 3), dtype=np.float32),
                    np.zeros((4, 4, 3), dtype=np.float32),
                    np.zeros((4, 4, 2), dtype=np.float32),
                ),
            ):
                batch = provider.next_batch(batch_size=2)

        self.assertTrue(fake_executor.map_called)
        self.assertEqual(batch[0].shape, (2, 4, 4, 6))
        self.assertEqual(batch[3].shape, (2, 4, 4, 2))

    def test_ft3d_provider_empty_samples_raises_runtime_error(self) -> None:
        provider = FT3DBatchProvider(
            samples=[],
            crop_h=4,
            crop_w=4,
            seed=42,
            sampling_mode="random",
            crop_mode="random",
            flow_divisor=12.5,
            augment_cfg={"enabled": False},
            num_workers=2,
        )
        with self.assertRaisesRegex(RuntimeError, "sample list is empty"):
            provider.next_batch(batch_size=2)

    def test_build_ft3d_triplet_accepts_relative_sample_with_absolute_roots(self) -> None:
        """Runtime provider paths may be relative while cached roots are absolute."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            work_root = root / "EdgeFlowNAS"
            frames_root = root / "Datasets" / "FlyingThings3D" / "frames_cleanpass" / "TRAIN"
            flow_root = root / "Datasets" / "FlyingThings3D" / "optical_flow" / "TRAIN"
            self._touch(frames_root / "A" / "0001" / "left" / "0006.png")
            self._touch(frames_root / "A" / "0001" / "left" / "0007.png")
            self._touch(
                flow_root
                / "A"
                / "0001"
                / "into_future"
                / "left"
                / "OpticalFlowIntoFuture_0006_L.pfm"
            )
            work_root.mkdir(parents=True, exist_ok=True)
            old_cwd = os.getcwd()
            try:
                os.chdir(work_root)
                img0, img1, flow = _build_ft3d_triplet(
                    img0_path="../Datasets/FlyingThings3D/frames_cleanpass/TRAIN/A/0001/left/0006.png",
                    frames_root=str(frames_root),
                    flow_root=str(flow_root),
                )
            finally:
                os.chdir(old_cwd)

            self.assertTrue(img0.endswith("0006.png"))
            self.assertTrue(img1.endswith("0007.png"))
            self.assertTrue(flow.endswith("OpticalFlowIntoFuture_0006_L.pfm"))

    def test_build_ft3d_triplet_supports_past_direction(self) -> None:
        """Past-direction supervision should reverse the frame pair and flow filename."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frames_root = root / "frames_cleanpass" / "TRAIN"
            flow_root = root / "optical_flow" / "TRAIN"
            self._touch(frames_root / "A" / "0001" / "left" / "0006.png")
            self._touch(frames_root / "A" / "0001" / "left" / "0007.png")
            self._touch(
                flow_root / "A" / "0001" / "into_past" / "left" / "OpticalFlowIntoPast_0007_L.pfm"
            )

            img0, img1, flow = _build_ft3d_triplet(
                img0_path=str(frames_root / "A" / "0001" / "left" / "0007.png"),
                frames_root=str(frames_root),
                flow_root=str(flow_root),
                direction="into_past",
            )
            self.assertTrue(img0.endswith("0007.png"))
            self.assertTrue(img1.endswith("0006.png"))
            self.assertTrue(flow.endswith("OpticalFlowIntoPast_0007_L.pfm"))

    def test_apply_spatial_augment_forced_horizontal_flip_negates_x_flow(self) -> None:
        """Forced horizontal flip should negate the x component of flow."""
        img0 = np.zeros((10, 10, 3), dtype=np.float32)
        img1 = np.zeros((10, 10, 3), dtype=np.float32)
        flow = np.zeros((10, 10, 2), dtype=np.float32)
        flow[:, :, 0] = 2.0
        flow[:, :, 1] = 1.0

        aug_cfg = {
            "enabled": True,
            "spatial_aug_prob": 0.0,
            "stretch_prob": 0.0,
            "min_scale": 0.0,
            "max_scale": 0.0,
            "do_flip": True,
            "h_flip_prob": 1.0,
            "v_flip_prob": 0.0,
        }
        out0, out1, out_flow = _apply_spatial_augment(
            img0=img0,
            img1=img1,
            flow=flow,
            crop_h=8,
            crop_w=8,
            rng=np.random.RandomState(7),
            aug_cfg=aug_cfg,
        )
        self.assertEqual(out0.shape, (8, 8, 3))
        self.assertEqual(out1.shape, (8, 8, 3))
        self.assertTrue(np.allclose(out_flow[:, :, 0], -2.0))
        self.assertTrue(np.allclose(out_flow[:, :, 1], 1.0))


if __name__ == "__main__":
    unittest.main()
