import unittest

from wrappers.run_supernet_subnet_distribution_v2 import _build_parser


class TestRunSupernetSubnetDistributionV2Wrapper(unittest.TestCase):
    def test_parser_defaults_to_v2_training_config(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.config, "configs/supernet_fc2_172x224_v2.yaml")
        self.assertIsNone(args.max_fc2_val_samples)

    def test_parser_accepts_v2_fixed_arch_and_fc2_cap(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--fixed_arch",
                "0,1,2,0,1,2,0,1,0,1,0",
                "--max_fc2_val_samples",
                "128",
            ]
        )

        self.assertEqual(args.fixed_arch, "0,1,2,0,1,2,0,1,0,1,0")
        self.assertEqual(args.max_fc2_val_samples, 128)


if __name__ == "__main__":
    unittest.main()
