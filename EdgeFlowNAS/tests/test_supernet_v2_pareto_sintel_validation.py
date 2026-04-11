"""Unit tests for V2 Pareto/near-Pareto Sintel validation helpers."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from efnas.nas.supernet_v2_pareto_sintel_validation import (
    compute_sintel_retention_summary,
    run_pareto_sintel_validation,
    select_pareto_and_near_candidates,
)


class TestParetoSintelValidationHelpers(unittest.TestCase):
    def test_select_pareto_and_near_candidates_marks_categories_and_limits_near(self) -> None:
        rows = [
            {"arch_code": "a", "epe": 4.0, "fps": 5.0},
            {"arch_code": "b", "epe": 4.1, "fps": 6.0},
            {"arch_code": "c", "epe": 4.2, "fps": 4.9},
            {"arch_code": "d", "epe": 4.25, "fps": 5.9},
            {"arch_code": "e", "epe": 4.8, "fps": 4.0},
        ]

        selected = select_pareto_and_near_candidates(rows, near_rel_gap=0.05, max_near=1)
        selected_by_code = {row["arch_code"]: row for row in selected}

        self.assertEqual(selected_by_code["a"]["selection_type"], "pareto")
        self.assertEqual(selected_by_code["b"]["selection_type"], "pareto")
        self.assertEqual(selected_by_code["d"]["selection_type"], "near_pareto")
        self.assertNotIn("c", selected_by_code)
        self.assertNotIn("e", selected_by_code)
        self.assertEqual(selected_by_code["d"]["closest_pareto_arch"], "b")

    def test_compute_sintel_retention_summary_reports_retained_and_promoted(self) -> None:
        records = [
            {"arch_code": "a", "selection_type": "pareto", "fc2_epe": 4.0, "fps": 5.0, "sintel_epe": 5.0},
            {"arch_code": "b", "selection_type": "pareto", "fc2_epe": 4.1, "fps": 5.8, "sintel_epe": 6.0},
            {"arch_code": "c", "selection_type": "near_pareto", "fc2_epe": 4.25, "fps": 5.9, "sintel_epe": 5.1},
        ]

        summary = compute_sintel_retention_summary(records)
        self.assertEqual(summary["num_selected"], 3)
        self.assertEqual(summary["num_original_pareto"], 2)
        self.assertEqual(summary["num_near_pareto"], 1)
        self.assertEqual(summary["num_sintel_front"], 2)
        self.assertEqual(summary["original_pareto_retained_count"], 1)
        self.assertEqual(summary["near_pareto_promoted_count"], 1)
        self.assertEqual(summary["retained_original_pareto_arches"], ["a"])
        self.assertEqual(summary["promoted_near_pareto_arches"], ["c"])

    def test_dry_run_selects_candidates_and_uses_generated_output_dir(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            history_csv = tmp_path / "history_archive.csv"
            history_csv.write_text(
                "arch_code,epe,fps\n"
                "a,4.0,5.0\n"
                "b,4.1,6.0\n"
                "c,4.25,5.9\n",
                encoding="utf-8",
            )
            result = run_pareto_sintel_validation(
                config_path="configs/supernet_fc2_172x224_v2.yaml",
                overrides={},
                options={
                    "history_csv": str(history_csv),
                    "dry_run": True,
                    "near_rel_gap": 0.05,
                    "max_near": 5,
                },
            )

        self.assertEqual(result["mode"], "dry_run")
        self.assertIn("outputs", str(result["resolved_output_dir"]))
        self.assertIn("selected_count", result)
        self.assertGreaterEqual(int(result["selected_count"]), 2)


if __name__ == "__main__":
    unittest.main()
