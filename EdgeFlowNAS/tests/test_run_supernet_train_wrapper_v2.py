"""Unit tests for supernet V2 train CLI wrapper overrides."""

import unittest

from wrappers.run_supernet_train_v2 import _build_parser
from wrappers.run_supernet_train import _build_overrides


class TestRunSupernetTrainWrapperV2(unittest.TestCase):
    """Validate CLI to config override mapping for V2 train wrapper."""

    def test_default_config_points_to_v2_yaml(self) -> None:
        """Parser default config should target V2 yaml."""
        parser = _build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.config, "configs/supernet_fc2_180x240_v2.yaml")

    def test_distill_teacher_arch_code_override_allows_11d_text(self) -> None:
        """V2 wrapper should forward 11-d teacher arch code text unchanged."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--distill_enabled",
                "--distill_teacher_type",
                "supernet",
                "--distill_teacher_arch_code",
                "0 0 0 0 0 0 1 1 1 1 1",
            ]
        )
        overrides = _build_overrides(args)
        self.assertTrue(overrides["train.distill.enabled"])
        self.assertEqual(overrides["train.distill.teacher_type"], "supernet")
        self.assertEqual(overrides["train.distill.teacher_arch_code"], "0 0 0 0 0 0 1 1 1 1 1")


if __name__ == "__main__":
    unittest.main()
