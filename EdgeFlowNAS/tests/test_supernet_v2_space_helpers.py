"""Unit tests for Supernet V2 search-space helpers."""

import unittest
import random

from efnas.nas.arch_codec_v2 import decode_arch_code
from efnas.nas.eval_pool_builder_v2 import build_eval_pool, check_eval_pool_coverage
from efnas.nas.fair_sampler_v2 import generate_fair_cycle, run_cycles
from efnas.nas.search_space_v2 import get_num_blocks, get_num_choices


class TestSupernetV2SpaceHelpers(unittest.TestCase):
    """Validate V2 helper behavior without TensorFlow runtime."""

    def test_arch_codec_v2_decodes_groups(self) -> None:
        """Decoder should expose stem, backbone, and head groups."""
        decoded = decode_arch_code([0, 2, 0, 1, 2, 1, 0, 1, 0, 1, 0])
        self.assertIn("stem", decoded)
        self.assertIn("backbone", decoded)
        self.assertIn("head", decoded)
        self.assertEqual(decoded["stem"]["E0"], "7x7Conv")
        self.assertEqual(decoded["head"]["H1"], "5x5Conv")

    def test_eval_pool_builder_v2_covers_all_valid_options(self) -> None:
        """Evaluation pool should cover all valid options across mixed-cardinality blocks."""
        pool = build_eval_pool(seed=42, size=12)
        coverage = check_eval_pool_coverage(pool=pool)
        self.assertTrue(coverage["ok"])

    def test_fair_sampler_v2_outputs_valid_codes(self) -> None:
        """Sampler should emit valid 11-d arch codes with mixed ranges."""
        cycle_codes = generate_fair_cycle(rng=random.Random(42))
        self.assertEqual(len(cycle_codes), 3)
        for arch_code in cycle_codes:
            self.assertEqual(len(arch_code), get_num_blocks())
            for block_idx, value in enumerate(arch_code):
                self.assertGreaterEqual(int(value), 0)
                self.assertLess(int(value), get_num_choices(block_idx))

    def test_fair_sampler_v2_keeps_gap_small_under_balanced_duplication(self) -> None:
        """Balanced duplication should keep the global gap very small."""
        result = run_cycles(cycles=20, seed=42)
        self.assertLessEqual(int(result["fairness_gap"]), 1)


if __name__ == "__main__":
    unittest.main()
