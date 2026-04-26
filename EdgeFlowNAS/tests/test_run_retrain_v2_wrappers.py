"""Unit tests for retrain_v2 CLI wrappers."""

import unittest

from wrappers.run_retrain_v2_fc2 import _build_parser as build_fc2_parser
from wrappers.run_retrain_v2_ft3d import _build_parser as build_ft3d_parser
from wrappers.run_retrain_v2_sintel_test import _build_parser as build_sintel_parser
from wrappers.run_retrain_v2_vela_compare import _build_parser as build_vela_compare_parser


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
        args = parser.parse_args(
            [
                "--fc2_experiment_dir",
                "outputs/retrain_v2_fc2/demo",
                "--frames_base_path",
                "../Datasets/frames_cleanpass",
                "--flow_base_path",
                "../Datasets/optical_flow",
                "--ft3d_num_workers",
                "8",
            ]
        )
        self.assertEqual(args.fc2_experiment_dir, "outputs/retrain_v2_fc2/demo")
        self.assertEqual(args.frames_base_path, "../Datasets/frames_cleanpass")
        self.assertEqual(args.flow_base_path, "../Datasets/optical_flow")
        self.assertEqual(args.ft3d_num_workers, 8)

    def test_sintel_parser_accepts_experiment_dir_and_best_mode(self) -> None:
        parser = build_sintel_parser()
        args = parser.parse_args(
            [
                "--experiment_dir",
                "outputs/retrain_v2_ft3d/demo",
                "--dataset_root",
                "/tmp/Sintel",
                "--ckpt_name",
                "sintel_best",
            ]
        )
        self.assertEqual(args.experiment_dir, "outputs/retrain_v2_ft3d/demo")
        self.assertEqual(args.ckpt_name, "sintel_best")

    def test_vela_compare_parser_accepts_checkpoint_and_arch(self) -> None:
        parser = build_vela_compare_parser()
        args = parser.parse_args(
            [
                "--checkpoint_prefix",
                "outputs/supernet/demo/checkpoints/supernet_best.ckpt",
                "--arch_code",
                "2,1,0,1,2,1,0,0,0,0,1",
            ]
        )
        self.assertEqual(args.checkpoint_prefix, "outputs/supernet/demo/checkpoints/supernet_best.ckpt")
        self.assertEqual(args.arch_code, "2,1,0,1,2,1,0,0,0,0,1")


if __name__ == "__main__":
    unittest.main()
