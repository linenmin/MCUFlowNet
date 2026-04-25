"""Unit tests for retrain_v2 in-training Sintel evaluation helpers."""

import csv
import tempfile
import unittest
from pathlib import Path

from efnas.engine.retrain_v2_sintel_runtime import (
    _append_sintel_metrics,
    _restore_retrain_histories,
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

    def test_restore_retrain_histories_reads_existing_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            for name, epe, sintel in (("knee", "2.3", "5.9"), ("fast", "2.5", "6.2")):
                model_dir = base_dir / f"model_{name}"
                model_dir.mkdir(parents=True, exist_ok=True)
                with (model_dir / "eval_history.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "epe", "best_epe", "sintel_epe"])
                    writer.writeheader()
                    writer.writerow({"epoch": "2", "loss": "1.1", "epe": epe, "best_epe": epe, "sintel_epe": sintel})

            with (base_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["epoch", "mean_epe", "epe_knee", "epe_fast", "sintel_epe_knee", "sintel_epe_fast", "mean_sintel_epe"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": "2",
                        "mean_epe": "2.4",
                        "epe_knee": "2.3",
                        "epe_fast": "2.5",
                        "sintel_epe_knee": "5.9",
                        "sintel_epe_fast": "6.2",
                        "mean_sintel_epe": "6.05",
                    }
                )

            eval_histories, comparison_rows = _restore_retrain_histories(base_dir=base_dir, model_names=["knee", "fast"])
            self.assertEqual(len(eval_histories["knee"]), 1)
            self.assertEqual(eval_histories["knee"][0]["epoch"], 2)
            self.assertAlmostEqual(eval_histories["knee"][0]["sintel_epe"], 5.9, places=6)
            self.assertEqual(len(comparison_rows), 1)
            self.assertAlmostEqual(comparison_rows[0]["mean_sintel_epe"], 6.05, places=6)


if __name__ == "__main__":
    unittest.main()
