import unittest

from wrappers.run_supernet_subnet_distribution_v3 import _build_parser


class TestRunSupernetSubnetDistributionV3Wrapper(unittest.TestCase):
    def test_parser_defaults_to_v3_training_config(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.config, "configs/supernet_v3_fc2_172x224.yaml")
        self.assertIsNone(args.experiment_dir)
        self.assertIsNone(args.max_fc2_val_samples)

    def test_parser_accepts_experiment_dir_workers_and_prefetch(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--fixed_arch",
                "0,1,2,0,1,2,0,1,0,1,0",
                "--experiment_dir",
                "outputs/supernet/v3_no_distill",
                "--num_workers",
                "4",
                "--prefetch_batches",
                "2",
                "--max_fc2_val_samples",
                "128",
            ]
        )

        self.assertEqual(args.fixed_arch, "0,1,2,0,1,2,0,1,0,1,0")
        self.assertEqual(args.experiment_dir, "outputs/supernet/v3_no_distill")
        self.assertEqual(args.num_workers, 4)
        self.assertEqual(args.prefetch_batches, 2)
        self.assertEqual(args.max_fc2_val_samples, 128)


if __name__ == "__main__":
    unittest.main()
