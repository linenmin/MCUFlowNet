import unittest


class TestAblationV1ConfigAndWrappers(unittest.TestCase):
    def test_wrapper_parsers_accept_required_overrides(self):
        from wrappers.run_ablation_v1_fc2 import _build_parser as build_fc2_parser
        from wrappers.run_ablation_v1_ft3d import _build_parser as build_ft3d_parser

        fc2_args = build_fc2_parser().parse_args(
            [
                "--config",
                "configs/ablation_v1_fc2.yaml",
                "--experiment_name",
                "exp",
                "--gpu_device",
                "0",
                "--fc2_num_workers",
                "16",
                "--grad_clip_global_norm",
                "200.0",
                "--micro_batch_size",
                "4",
                "--variants",
                "edgeflownet_deconv,bilinear_eca_gate4x",
                "--resume_ckpt_name",
                "sintel_best",
            ]
        )
        self.assertEqual(fc2_args.fc2_num_workers, 16)
        self.assertEqual(fc2_args.grad_clip_global_norm, 200.0)
        self.assertEqual(fc2_args.micro_batch_size, 4)
        self.assertEqual(fc2_args.variants, "edgeflownet_deconv,bilinear_eca_gate4x")
        self.assertEqual(fc2_args.resume_ckpt_name, "sintel_best")

        ft3d_args = build_ft3d_parser().parse_args(
            [
                "--config",
                "configs/ablation_v1_ft3d.yaml",
                "--experiment_name",
                "exp",
                "--init_experiment_dir",
                "outputs/ablation_v1_fc2/exp",
                "--init_ckpt_name",
                "best",
                "--ft3d_num_workers",
                "16",
                "--grad_clip_global_norm",
                "200.0",
                "--micro_batch_size",
                "2",
                "--variants",
                "edgeflownet_deconv",
            ]
        )
        self.assertEqual(ft3d_args.ft3d_num_workers, 16)
        self.assertEqual(ft3d_args.grad_clip_global_norm, 200.0)
        self.assertEqual(ft3d_args.micro_batch_size, 2)
        self.assertEqual(ft3d_args.variants, "edgeflownet_deconv")
        self.assertEqual(ft3d_args.init_ckpt_name, "best")

    def test_micro_batch_helpers_preserve_logical_batch_shape(self):
        try:
            import tensorflow  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("TensorFlow is required to import ablation_v1_trainer")
        from efnas.engine.ablation_v1_trainer import _iter_micro_slices, _resolve_micro_batch_size

        self.assertEqual(_resolve_micro_batch_size({"micro_batch_size": 4}, batch_size=32), 4)
        self.assertEqual(_resolve_micro_batch_size({}, batch_size=32), 32)
        self.assertEqual(_iter_micro_slices(total_size=10, micro_batch_size=4), [slice(0, 4), slice(4, 8), slice(8, 10)])

    def test_grad_stats_summary_counts_clip_rate(self):
        try:
            import tensorflow  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("TensorFlow is required to import ablation_v1_trainer")
        from efnas.engine.ablation_v1_trainer import _summarize_grad_norms

        stats = _summarize_grad_norms([10.0, 250.0, 50.0, 300.0], clip_threshold=200.0)
        self.assertAlmostEqual(stats["mean"], 152.5)
        self.assertAlmostEqual(stats["clip_rate"], 0.5)
        self.assertEqual(stats["count"], 4)


if __name__ == "__main__":
    unittest.main()
