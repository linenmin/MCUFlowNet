"""Unit tests for V2 rank-consistency diagnostic helpers."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from efnas.nas.search_space_v2 import V2_REFERENCE_ARCH_CODE, get_num_blocks, get_num_choices
from efnas.nas.supernet_v2_rank_consistency import (
    _resolve_path,
    _option_or_default,
    build_fc2_eval_windows,
    compute_rank_consistency_summary,
    compute_v2_complexity_score,
    run_rank_consistency_diagnostic,
    sample_probe_arch_pool_v2,
)


class TestSupernetV2RankConsistencyHelpers(unittest.TestCase):
    """Validate pure helper behavior for the V2 FC2/Sintel diagnostic."""

    def test_sample_probe_arch_pool_v2_is_deterministic_and_valid(self) -> None:
        """Sampling should be reproducible, unique, and respect mixed choice ranges."""
        pool_a = sample_probe_arch_pool_v2(num_arch_samples=50, seed=42)
        pool_b = sample_probe_arch_pool_v2(num_arch_samples=50, seed=42)
        self.assertEqual(pool_a, pool_b)
        self.assertEqual(len(pool_a), 50)
        self.assertEqual(len(pool_a), len({tuple(code) for code in pool_a}))
        self.assertIn([int(v) for v in V2_REFERENCE_ARCH_CODE], pool_a)
        for arch_code in pool_a:
            self.assertEqual(len(arch_code), get_num_blocks())
            for block_idx, value in enumerate(arch_code):
                self.assertGreaterEqual(int(value), 0)
                self.assertLess(int(value), get_num_choices(block_idx))

    def test_sample_probe_arch_pool_v2_spans_complexity_range(self) -> None:
        """Probe pool should cover clearly distinct complexity levels."""
        pool = sample_probe_arch_pool_v2(num_arch_samples=50, seed=7)
        scores = [compute_v2_complexity_score(code) for code in pool]
        self.assertGreater(max(scores) - min(scores), 4.0)

    def test_compute_rank_consistency_summary_reports_overlap_and_correlation(self) -> None:
        """Summary should expose strong disagreement when FC2 and Sintel ranks diverge."""
        records = [
            {"arch_code": "a", "fc2_epe": 1.0, "sintel_epe": 5.0},
            {"arch_code": "b", "fc2_epe": 2.0, "sintel_epe": 4.0},
            {"arch_code": "c", "fc2_epe": 3.0, "sintel_epe": 3.0},
            {"arch_code": "d", "fc2_epe": 4.0, "sintel_epe": 2.0},
            {"arch_code": "e", "fc2_epe": 5.0, "sintel_epe": 1.0},
        ]
        summary = compute_rank_consistency_summary(records=records, top_ks=(1, 3))
        self.assertAlmostEqual(float(summary["spearman"]), -1.0, places=6)
        self.assertEqual(int(summary["topk_overlap"]["1"]["overlap"]), 0)
        self.assertEqual(int(summary["topk_overlap"]["3"]["overlap"]), 1)
        self.assertEqual(summary["largest_rank_shift"]["arch_code"], "a")

    def test_build_fc2_eval_windows_covers_full_set_without_wrap(self) -> None:
        """Full FC2 evaluation should consume each validation sample exactly once."""
        windows = build_fc2_eval_windows(num_samples=100, batch_size=32, max_samples=None)
        self.assertEqual(windows, [(0, 32), (32, 32), (64, 32), (96, 4)])

    def test_build_fc2_eval_windows_supports_optional_cap(self) -> None:
        """Optional sample cap should still avoid wraparound in the tail batch."""
        windows = build_fc2_eval_windows(num_samples=100, batch_size=32, max_samples=50)
        self.assertEqual(windows, [(0, 32), (32, 18)])

    def test_option_or_default_treats_none_as_missing(self) -> None:
        """Optional CLI args set to None should fall back to config defaults."""
        self.assertEqual(_option_or_default({"bn_recal_batch_size": None}, "bn_recal_batch_size", 32), 32)
        self.assertEqual(_option_or_default({"bn_recal_batch_size": 8}, "bn_recal_batch_size", 32), 8)

    def test_resolve_path_falls_back_to_sibling_edgeflownet_repo(self) -> None:
        """Relative EdgeFlowNet paths should resolve to the sibling repo when absent in EdgeFlowNAS."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            edgeflownas_root = root / "EdgeFlowNAS"
            edgeflownet_root = root / "EdgeFlowNet"
            edgeflownas_root.mkdir()
            target = edgeflownet_root / "code" / "dataset_paths" / "MPI_Sintel_Final_train_list.txt"
            target.parent.mkdir(parents=True)
            target.write_text("dummy\n", encoding="utf-8")
            with patch("efnas.nas.supernet_v2_rank_consistency.project_root", return_value=edgeflownas_root):
                resolved = _resolve_path("EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt")
            self.assertEqual(resolved, target.resolve())

    def test_dry_run_uses_generated_search_v2_output_dir_when_not_provided(self) -> None:
        """Dry-run should synthesize an outputs/search_v2 path instead of stringifying None."""
        result = run_rank_consistency_diagnostic(
            config_path="configs/supernet_fc2_172x224_v2.yaml",
            overrides={},
            options={"num_arch_samples": 5, "sample_seed": 1, "dry_run": True, "output_dir": None},
        )
        self.assertIn("outputs", str(result["resolved_output_dir"]))
        self.assertIn("search_v2", str(result["resolved_output_dir"]))
        self.assertNotIn("None", str(result["resolved_output_dir"]))


if __name__ == "__main__":
    unittest.main()
