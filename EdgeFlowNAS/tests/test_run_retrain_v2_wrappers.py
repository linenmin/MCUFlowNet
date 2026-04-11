"""Unit tests for retrain_v2 CLI wrappers."""

import unittest

from wrappers.run_retrain_v2_fc2 import _build_parser as build_fc2_parser
from wrappers.run_retrain_v2_ft3d import _build_parser as build_ft3d_parser
from wrappers.run_retrain_v2_sintel_test import _build_parser as build_sintel_parser


class TestRunRetrainV2Wrappers(unittest.TestCase):
    """Validate wrapper parser defaults and key CLI contracts."""

    def test_fc2_parser_defaults_to_fc2_config(self) -> None:
        parser = build_fc2_parser()
        args = parser.parse_args([])
        self.assertEqual(args.config, "configs/retrain_v2_fc2.yaml")

    def test_ft3d_parser_defaults_to_ft3d_config(self) -> None:
        parser = build_ft3d_parser()
        args = parser.parse_args([])
        self.assertEqual(args.config, "configs/retrain_v2_ft3d.yaml")

    def test_ft3d_parser_accepts_fc2_init_dir(self) -> None:
        parser = build_ft3d_parser()
        args = parser.parse_args(["--fc2_experiment_dir", "outputs/retrain_v2_fc2/demo"])
        self.assertEqual(args.fc2_experiment_dir, "outputs/retrain_v2_fc2/demo")

    def test_sintel_parser_accepts_experiment_dir_and_best_mode(self) -> None:
        parser = build_sintel_parser()
        args = parser.parse_args(
            [
                "--experiment_dir",
                "outputs/retrain_v2_ft3d/demo",
                "--dataset_root",
                "/tmp/Sintel",
                "--ckpt_name",
                "best",
            ]
        )
        self.assertEqual(args.experiment_dir, "outputs/retrain_v2_ft3d/demo")
        self.assertEqual(args.ckpt_name, "best")


if __name__ == "__main__":
    unittest.main()
