"""Unit tests for Supernet V3 search-space helpers."""

import random
import unittest

from efnas.nas.arch_codec_v3 import decode_arch_code
from efnas.nas.eval_pool_builder_v3 import build_eval_pool, check_eval_pool_coverage
from efnas.nas.fair_sampler_v3 import generate_fair_cycle, run_cycles
from efnas.nas.search_space_v3 import (
    get_arch_semantics_version,
    get_block_specs,
    get_num_blocks,
    get_num_choices,
)


class TestSupernetV3SpaceHelpers(unittest.TestCase):
    """Validate V3 helper behavior without TensorFlow runtime."""

    def test_stem_choices_are_ordered_light_to_heavy(self) -> None:
        """The first two arch-code dimensions should use light-to-heavy labels."""
        specs = get_block_specs()
        self.assertEqual(get_arch_semantics_version(), "supernet_v3_mixed_11d_light_to_heavy_fixed_bilinear_bneckeca_gate4x")
        self.assertEqual(specs[0]["labels"], ["3x3Conv", "5x5Conv", "7x7Conv"])
        self.assertEqual(specs[1]["labels"], ["3x3Conv", "5x5Conv", "3x3Stride2DilatedConv"])

    def test_arch_codec_v3_decodes_corrected_stem_labels(self) -> None:
        """Decoder should expose corrected V3 stem semantics."""
        decoded = decode_arch_code([0, 2, 0, 1, 2, 1, 0, 1, 0, 1, 0])
        self.assertEqual(decoded["stem"]["E0"], "3x3Conv")
        self.assertEqual(decoded["stem"]["E1"], "3x3Stride2DilatedConv")
        self.assertEqual(decoded["head"]["H1"], "5x5Conv")

    def test_eval_pool_builder_v3_covers_all_valid_options(self) -> None:
        """Evaluation pool should cover all valid options across mixed-cardinality blocks."""
        pool = build_eval_pool(seed=42, size=12)
        coverage = check_eval_pool_coverage(pool=pool)
        self.assertTrue(coverage["ok"])

    def test_fair_sampler_v3_outputs_valid_codes(self) -> None:
        """Sampler should emit valid 11-d arch codes with mixed ranges."""
        cycle_codes = generate_fair_cycle(rng=random.Random(42))
        self.assertEqual(len(cycle_codes), 3)
        for arch_code in cycle_codes:
            self.assertEqual(len(arch_code), get_num_blocks())
            for block_idx, value in enumerate(arch_code):
                self.assertGreaterEqual(int(value), 0)
                self.assertLess(int(value), get_num_choices(block_idx))

    def test_fair_sampler_v3_keeps_gap_small_under_balanced_duplication(self) -> None:
        """Balanced duplication should keep the global gap very small."""
        result = run_cycles(cycles=20, seed=42)
        self.assertLessEqual(int(result["fairness_gap"]), 1)


if __name__ == "__main__":
    unittest.main()
