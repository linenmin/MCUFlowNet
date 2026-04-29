import random
import tempfile
import unittest
from pathlib import Path

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

    def test_v3_search_space_adapter_uses_v3_module(self) -> None:
        space = nsga2_search.load_search_space("efnas.nas.search_space_v3")
        self.assertEqual(space.num_blocks(), 11)
        self.assertEqual([space.num_choices(i) for i in range(11)], [3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2])
        space.validate([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            space.validate([2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1])

    def test_gpu_assignment_round_robins_visible_devices(self) -> None:
        assignments = nsga2_search.assign_gpus_to_arches(["a", "b", "c", "d"], ["0", "2"])
        self.assertEqual(assignments, [("a", "0"), ("b", "2"), ("c", "0"), ("d", "2")])

    def test_prunes_vela_tflite_artifacts_when_enabled(self) -> None:
        cfg = {
            "search": {
                "total_evaluations": 50,
                "population_size": 50,
                "seed": 2026,
                "search_space_module": "efnas.nas.search_space_v3",
            },
            "concurrency": {"max_workers": 1, "gpu_devices": "0"},
            "evaluation": {"vela_prune_tflite_after_reduce": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            artifact_dir = exp_dir / "dashboard" / "eval_outputs" / "run_000" / "analysis" / "vela_tmp" / "arch_000"
            artifact_dir.mkdir(parents=True)
            tflite_path = artifact_dir / "model_int8.tflite"
            keep_path = artifact_dir / "vela_summary.csv"
            tflite_path.write_bytes(b"tflite")
            keep_path.write_text("fps,1\n", encoding="utf-8")

            runner = nsga2_search.NSGA2SearchRunner(cfg=cfg, exp_dir=str(exp_dir), project_root=str(Path.cwd()))
            removed = runner._maybe_prune_vela_tflite(stage="test")

            self.assertEqual(removed, 1)
            self.assertFalse(tflite_path.exists())
            self.assertTrue(keep_path.exists())


if __name__ == "__main__":
    unittest.main()
