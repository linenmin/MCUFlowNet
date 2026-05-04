"""Tests for Retrain V3 wrapper and launcher plumbing."""

import csv
import tempfile
import unittest
from pathlib import Path


class TestRetrainV3Wrappers(unittest.TestCase):
    def test_default_candidates_are_the_final_three_subnets(self):
        from wrappers.run_retrain_v3_batch import load_candidates

        candidates = load_candidates(Path("plan/retrain_v3/retrain_v3_candidates.csv"))

        self.assertEqual(
            [(item["model_name"], item["arch_code"]) for item in candidates],
            [
                ("v3_acc", "0,1,2,2,2,2,0,0,0,0,1"),
                ("v3_efn_fps", "2,0,0,2,2,1,0,0,0,0,0"),
                ("v3_light", "0,0,0,0,0,0,0,0,0,0,0"),
            ],
        )

    def test_batch_launcher_assigns_three_gpus_round_robin(self):
        from wrappers.run_retrain_v3_batch import assign_gpu

        self.assertEqual([assign_gpu(i, ["0", "1", "2"]) for i in range(5)], ["0", "1", "2", "0", "1"])

    def test_fc2_wrapper_accepts_confirmed_retrain_controls(self):
        from wrappers.run_retrain_v3_fc2 import _build_parser

        args = _build_parser().parse_args(
            [
                "--config",
                "configs/retrain_v3_fc2.yaml",
                "--experiment_name",
                "retrain_v3_fc2_run1",
                "--model_name",
                "v3_light",
                "--arch_code",
                "0,0,0,0,0,0,0,0,0,0,0",
                "--gpu_device",
                "0",
                "--num_epochs",
                "400",
                "--batch_size",
                "32",
                "--sintel_eval_every_epoch",
                "5",
                "--fc2_eval_every_epoch",
                "1",
                "--fc2_num_workers",
                "6",
                "--prefetch_batches",
                "2",
            ]
        )

        self.assertEqual(args.model_name, "v3_light")
        self.assertEqual(args.num_epochs, 400)
        self.assertEqual(args.sintel_eval_every_epoch, 5)
        self.assertEqual(args.fc2_num_workers, 6)

    def test_ft3d_wrapper_accepts_fc2_init_and_ft3d_controls(self):
        from wrappers.run_retrain_v3_ft3d import _build_parser

        args = _build_parser().parse_args(
            [
                "--config",
                "configs/retrain_v3_ft3d.yaml",
                "--experiment_name",
                "retrain_v3_ft3d_run1",
                "--model_name",
                "v3_acc",
                "--arch_code",
                "0,1,2,2,2,2,0,0,0,0,1",
                "--fc2_experiment_dir",
                "outputs/retrain_v3_fc2/retrain_v3_fc2_run1",
                "--ft3d_num_workers",
                "8",
                "--ft3d_eval_num_workers",
                "2",
                "--sintel_eval_every_epoch",
                "2",
            ]
        )

        self.assertEqual(args.model_name, "v3_acc")
        self.assertEqual(args.fc2_experiment_dir, "outputs/retrain_v3_fc2/retrain_v3_fc2_run1")
        self.assertEqual(args.ft3d_num_workers, 8)
        self.assertEqual(args.sintel_eval_every_epoch, 2)

    def test_candidate_loader_accepts_model_name_column(self):
        from wrappers.run_retrain_v3_batch import load_candidates

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model_name", "arch_code", "role"])
                writer.writeheader()
                writer.writerow({"model_name": "x", "arch_code": "0,0,0,0,0,0,0,0,0,0,0", "role": "light"})

            candidates = load_candidates(path)

        self.assertEqual(candidates[0]["model_name"], "x")
        self.assertEqual(candidates[0]["role"], "light")


if __name__ == "__main__":
    unittest.main()
