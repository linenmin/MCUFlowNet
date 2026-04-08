"""Unit tests for the V2 rank-consistency wrapper CLI."""

import unittest

from wrappers.run_supernet_v2_rank_consistency import _build_parser


class TestRunSupernetV2RankConsistencyWrapper(unittest.TestCase):
    """Validate wrapper defaults for the V2 diagnostic."""

    def test_default_config_points_to_v2_yaml(self) -> None:
        """Wrapper should default to the 172x224 V2 config used by the current run."""
        parser = _build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.config, "configs/supernet_fc2_172x224_v2.yaml")
        self.assertEqual(args.num_arch_samples, 50)
        self.assertEqual(args.checkpoint_type, "best")


if __name__ == "__main__":
    unittest.main()
