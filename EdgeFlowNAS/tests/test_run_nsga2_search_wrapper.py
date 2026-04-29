import unittest

from wrappers.run_nsga2_search import _build_parser


class TestRunNSGA2SearchWrapper(unittest.TestCase):
    def test_parser_defaults_to_nsga2_v2_config(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.config, "configs/nsga2_v2.yaml")
        self.assertFalse(args.resume)

    def test_parser_accepts_resume_and_experiment_name(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--config",
                "configs/custom.yaml",
                "--resume",
                "--experiment_name",
                "trial_a",
            ]
        )
        self.assertEqual(args.config, "configs/custom.yaml")
        self.assertTrue(args.resume)
        self.assertEqual(args.experiment_name, "trial_a")

    def test_parser_accepts_v3_runtime_overrides(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--supernet_experiment_dir",
                "outputs/supernet/v3_no_distill",
                "--gpu_devices",
                "0,1,2,3,4,5",
                "--max_workers",
                "6",
                "--num_workers",
                "2",
                "--prefetch_batches",
                "2",
            ]
        )

        self.assertEqual(args.supernet_experiment_dir, "outputs/supernet/v3_no_distill")
        self.assertEqual(args.gpu_devices, "0,1,2,3,4,5")
        self.assertEqual(args.max_workers, 6)
        self.assertEqual(args.num_workers, 2)
        self.assertEqual(args.prefetch_batches, 2)


if __name__ == "__main__":
    unittest.main()
