"""Unit tests for subnet distribution helpers."""

import unittest

from code.nas.supernet_subnet_distribution import (
    ARCH_SPACE_SIZE,
    compute_complexity_scores,
    sample_arch_pool,
)


class TestSubnetDistributionHelpers(unittest.TestCase):
    """Test pure helper functions used by subnet distribution script."""

    def test_sample_arch_pool_is_unique_and_deterministic(self) -> None:
        """Sampling should be reproducible and unique for same seed."""
        eval_pool = [[0] * 9, [1] * 9, [2] * 9]
        pool_a = sample_arch_pool(num_arch_samples=32, seed=42, include_eval_pool=True, eval_pool=eval_pool)
        pool_b = sample_arch_pool(num_arch_samples=32, seed=42, include_eval_pool=True, eval_pool=eval_pool)
        self.assertEqual(pool_a, pool_b)
        self.assertEqual(len(pool_a), len({tuple(code) for code in pool_a}))
        self.assertEqual(pool_a[0], [0] * 9)
        self.assertEqual(pool_a[1], [1] * 9)
        self.assertEqual(pool_a[2], [2] * 9)

    def test_sample_arch_pool_full_space(self) -> None:
        """Requesting >= 3^9 should return full unique architecture space."""
        pool = sample_arch_pool(num_arch_samples=ARCH_SPACE_SIZE, seed=7, include_eval_pool=False, eval_pool=[])
        self.assertEqual(len(pool), ARCH_SPACE_SIZE)
        self.assertEqual(len(pool), len({tuple(code) for code in pool}))

    def test_complexity_score_direction(self) -> None:
        """Complexity proxy should align heavy direction across blocks."""
        light = [0, 0, 0, 0, 2, 2, 2, 2, 2]  # shallow + smallest kernels
        heavy = [2, 2, 2, 2, 0, 0, 0, 0, 0]  # deep + largest kernels
        light_score = compute_complexity_scores(light)["complexity_score"]
        heavy_score = compute_complexity_scores(heavy)["complexity_score"]
        self.assertGreater(heavy_score, light_score)


if __name__ == "__main__":
    unittest.main()
