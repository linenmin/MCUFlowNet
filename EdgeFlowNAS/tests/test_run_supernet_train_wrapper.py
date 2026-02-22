"""Unit tests for supernet train CLI wrapper overrides."""

import unittest

from wrappers.run_supernet_train import _build_overrides, _build_parser


class TestRunSupernetTrainWrapper(unittest.TestCase):
    """Validate CLI to config override mapping for train wrapper."""

    def test_eval_every_epoch_override_present(self) -> None:
        """CLI value should be forwarded into eval overrides."""
        parser = _build_parser()
        args = parser.parse_args(["--eval_every_epoch", "5"])
        overrides = _build_overrides(args)
        self.assertIn("eval.eval_every_epoch", overrides)
        self.assertEqual(overrides["eval.eval_every_epoch"], 5)

    def test_eval_every_epoch_override_absent_by_default(self) -> None:
        """When CLI flag is not provided, override should not be emitted."""
        parser = _build_parser()
        args = parser.parse_args([])
        overrides = _build_overrides(args)
        self.assertNotIn("eval.eval_every_epoch", overrides)

    def test_distill_overrides_present(self) -> None:
        """Distill CLI args should map into nested train.distill config."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--distill_enabled",
                "--distill_lambda",
                "1.5",
                "--distill_teacher_type",
                "supernet",
                "--distill_teacher_ckpt",
                "outputs/supernet/exp/checkpoints/supernet_best.ckpt",
                "--distill_teacher_arch_code",
                "0 0 0 0 2 1 2 2 2",
                "--distill_layer_weights",
                "1.0 0.5 0.25",
            ]
        )
        overrides = _build_overrides(args)
        self.assertTrue(overrides["train.distill.enabled"])
        self.assertAlmostEqual(overrides["train.distill.lambda"], 1.5)
        self.assertEqual(overrides["train.distill.teacher_type"], "supernet")
        self.assertEqual(
            overrides["train.distill.teacher_ckpt"],
            "outputs/supernet/exp/checkpoints/supernet_best.ckpt",
        )
        self.assertEqual(overrides["train.distill.teacher_arch_code"], "0 0 0 0 2 1 2 2 2")
        self.assertEqual(overrides["train.distill.layer_weights"], "1.0 0.5 0.25")


if __name__ == "__main__":
    unittest.main()
