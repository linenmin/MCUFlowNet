"""Unit tests for supernet V3 train CLI wrapper overrides."""

import unittest

from wrappers.run_supernet_train_v3 import _build_overrides_v3, _build_parser


class TestRunSupernetTrainWrapperV3(unittest.TestCase):
    """Validate CLI to config override mapping for V3 train wrapper."""

    def test_default_config_points_to_v3_yaml(self) -> None:
        """Parser default config should target the formal V3 172x224 yaml."""
        args = _build_parser().parse_args([])
        self.assertEqual(args.config, "configs/supernet_v3_fc2_172x224.yaml")

    def test_parallel_and_prefetch_overrides_are_forwarded(self) -> None:
        """V3 wrapper should expose arch-parallel and prefetch controls."""
        args = _build_parser().parse_args(
            [
                "--gpu_devices",
                "0,1,2",
                "--multi_gpu_mode",
                "arch_parallel",
                "--fc2_num_workers",
                "12",
                "--fc2_eval_num_workers",
                "2",
                "--prefetch_batches",
                "2",
            ]
        )
        overrides = _build_overrides_v3(args)
        self.assertEqual(overrides["train.gpu_devices"], "0,1,2")
        self.assertEqual(overrides["train.multi_gpu_mode"], "arch_parallel")
        self.assertEqual(overrides["data.fc2_num_workers"], 12)
        self.assertEqual(overrides["data.fc2_eval_num_workers"], 2)
        self.assertEqual(overrides["data.prefetch_batches"], 2)


if __name__ == "__main__":
    unittest.main()
