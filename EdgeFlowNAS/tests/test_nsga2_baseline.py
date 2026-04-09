import random
import unittest

from efnas.baselines import nsga2_search


class TestNSGA2BaselineHelpers(unittest.TestCase):
    def test_resolve_generation_count_requires_clean_budget_multiple(self) -> None:
        self.assertEqual(nsga2_search.resolve_generation_count(800, 50), 16)
        with self.assertRaises(ValueError):
            nsga2_search.resolve_generation_count(810, 50)

    def test_sample_random_arch_respects_v2_choice_ranges(self) -> None:
        rng = random.Random(2026)
        arch = nsga2_search.sample_random_arch(rng)
        self.assertEqual(len(arch), 11)
        self.assertTrue(all(value in (0, 1, 2) for value in arch[:6]))
        self.assertTrue(all(value in (0, 1) for value in arch[6:]))

    def test_mutate_arch_keeps_shape_and_valid_ranges(self) -> None:
        rng = random.Random(7)
        original = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        mutated = nsga2_search.mutate_arch(original, rng=rng, mutation_prob=1.0)
        self.assertEqual(len(mutated), 11)
        self.assertNotEqual(mutated, original)
        self.assertTrue(all(value in (0, 1, 2) for value in mutated[:6]))
        self.assertTrue(all(value in (0, 1) for value in mutated[6:]))

    def test_fast_non_dominated_sort_separates_fronts(self) -> None:
        objective_values = [
            (4.0, -5.0),
            (4.2, -6.0),
            (3.9, -4.8),
            (4.3, -4.0),
            (4.4, -3.8),
        ]
        fronts = nsga2_search.fast_non_dominated_sort(objective_values)
        self.assertEqual(fronts[0], [0, 1, 2])
        self.assertEqual(fronts[1], [3])
        self.assertEqual(fronts[2], [4])

    def test_select_next_population_prefers_better_fronts(self) -> None:
        rows = [
            {"arch_code": "a", "epe": 4.0, "fps": 5.0},
            {"arch_code": "b", "epe": 4.2, "fps": 6.0},
            {"arch_code": "c", "epe": 3.9, "fps": 4.8},
            {"arch_code": "d", "epe": 4.3, "fps": 4.0},
            {"arch_code": "e", "epe": 4.4, "fps": 3.8},
        ]
        selected = nsga2_search.select_next_population(rows, population_size=4)
        selected_codes = [row["arch_code"] for row in selected]
        self.assertEqual(len(selected_codes), 4)
        self.assertIn("a", selected_codes)
        self.assertIn("b", selected_codes)
        self.assertIn("c", selected_codes)
        self.assertIn("d", selected_codes)
        self.assertNotIn("e", selected_codes)


if __name__ == "__main__":
    unittest.main()
