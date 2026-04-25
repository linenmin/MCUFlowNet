"""Unit tests for retrain_v2 in-training Sintel evaluation helpers."""

import unittest

from efnas.engine.retrain_v2_sintel_runtime import (
    _append_sintel_metrics,
    _resolve_sintel_eval_config,
)


class TestRetrainV2SintelRuntime(unittest.TestCase):
    def test_resolve_sintel_eval_config_returns_none_when_dataset_root_missing(self) -> None:
        config = {"eval": {"eval_every_epoch": 2}}
        self.assertIsNone(_resolve_sintel_eval_config(config))

    def test_resolve_sintel_eval_config_uses_defaults_and_train_gpu(self) -> None:
        config = {
            "train": {"gpu_device": 0},
            "eval": {
                "eval_every_epoch": 2,
                "sintel": {
                    "dataset_root": "../Datasets/Sintel",
                },
            },
        }
        resolved = _resolve_sintel_eval_config(config)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["dataset_root"], "../Datasets/Sintel")
        self.assertEqual(resolved["sintel_list"], "EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt")
        self.assertEqual(resolved["patch_size"], (416, 1024))
        self.assertEqual(resolved["gpu_device"], 0)
        self.assertEqual(resolved["ckpt_name"], "sintel_best")

    def test_append_sintel_metrics_adds_per_model_and_mean_columns(self) -> None:
        row = {"epoch": 2, "mean_epe": 2.1, "epe_knee": 2.0, "epe_fast": 2.2}
        updated = _append_sintel_metrics(
            row=row,
            sintel_epes={"knee": 6.1, "fast": 6.4},
            model_names=["knee", "fast"],
        )
        self.assertEqual(updated["sintel_epe_knee"], 6.1)
        self.assertEqual(updated["sintel_epe_fast"], 6.4)
        self.assertAlmostEqual(updated["mean_sintel_epe"], 6.25, places=6)


if __name__ == "__main__":
    unittest.main()
