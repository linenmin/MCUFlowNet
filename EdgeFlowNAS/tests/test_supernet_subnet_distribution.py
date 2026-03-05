"""Unit tests for subnet distribution helpers."""

import tempfile
import unittest
from pathlib import Path

from efnas.nas.supernet_subnet_distribution import (
    ARCH_SPACE_SIZE,
    _compute_metric_summary,
    _extract_vela_summary_metrics,
    _save_vela_csv,
    _safe_fps,
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
        light = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # shallow + smallest kernels
        heavy = [2, 2, 2, 2, 2, 2, 2, 2, 2]  # deep + largest kernels
        light_score = compute_complexity_scores(light)["complexity_score"]
        heavy_score = compute_complexity_scores(heavy)["complexity_score"]
        self.assertGreater(heavy_score, light_score)

    def test_safe_fps(self) -> None:
        """FPS conversion should be stable and reject invalid input."""
        self.assertAlmostEqual(_safe_fps(10.0), 100.0)
        self.assertIsNone(_safe_fps(0.0))
        self.assertIsNone(_safe_fps(-1.0))
        self.assertIsNone(_safe_fps(None))

    def test_metric_summary_counts(self) -> None:
        """Metric summary should skip invalid values and keep counts."""
        summary = _compute_metric_summary([1.0, None, 3.0, float("nan")])
        self.assertEqual(summary["count_total"], 4)
        self.assertEqual(summary["count_valid"], 2)
        self.assertEqual(summary["count_invalid"], 2)
        self.assertAlmostEqual(summary["mean"], 2.0)

    def test_extract_vela_summary_metrics_reads_cycles_and_macs(self) -> None:
        """Should parse cycles_npu and nn_macs from Vela summary CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "arch_0000_summary_Grove_Sys_Config.csv"
            summary_path.write_text(
                "experiment,cycles_npu,nn_macs,inference_time,sram_memory_used\n"
                "default,124441439.0,5827230720,0.3111035975,1620.0\n",
                encoding="utf-8",
            )
            metrics = _extract_vela_summary_metrics(Path(tmp_dir))
            self.assertEqual(metrics["cycles_npu"], 124441439)
            self.assertEqual(metrics["macs"], 5827230720)

    def test_save_vela_csv_contains_cycles_and_macs_columns(self) -> None:
        """Vela csv should include cycles_npu and macs columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "vela_metrics.csv"
            _save_vela_csv(
                csv_path,
                records=[
                    {
                        "sample_index": 0,
                        "arch_code": "0,0,0,0,0,0,0,0,0",
                        "sram_peak_mb": 1.58203125,
                        "inference_ms": 120.0,
                        "fps": 8.33,
                        "cycles_npu": 124441439,
                        "macs": 5827230720,
                        "vela_status": "ok",
                        "vela_error": "",
                    }
                ],
            )
            content = csv_path.read_text(encoding="utf-8")
            self.assertIn("cycles_npu", content.splitlines()[0])
            self.assertIn("macs", content.splitlines()[0])
            self.assertIn("124441439", content)
            self.assertIn("5827230720", content)


if __name__ == "__main__":
    unittest.main()
