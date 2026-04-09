"""Unit tests for V2 FC2-versus-Sintel timing probe helpers."""

import unittest

from efnas.nas.supernet_v2_timing_probe import compute_timing_comparison, run_timing_probe


class TestSupernetV2TimingProbe(unittest.TestCase):
    """Validate pure helper behavior for the V2 timing probe."""

    def test_compute_timing_comparison_reports_ratios(self) -> None:
        """Timing summary should expose both absolute and per-sample ratios."""
        result = compute_timing_comparison(fc2_seconds=20.0, fc2_samples=100, sintel_seconds=60.0, sintel_samples=50)
        self.assertAlmostEqual(float(result["fc2_seconds_per_sample"]), 0.2, places=6)
        self.assertAlmostEqual(float(result["sintel_seconds_per_sample"]), 1.2, places=6)
        self.assertAlmostEqual(float(result["sintel_over_fc2_eval_ratio"]), 3.0, places=6)
        self.assertAlmostEqual(float(result["sintel_over_fc2_per_sample_ratio"]), 6.0, places=6)

    def test_dry_run_uses_generated_search_v2_output_dir(self) -> None:
        """Dry-run should synthesize an outputs/search_v2 path instead of requiring datasets."""
        result = run_timing_probe(
            config_path="configs/supernet_fc2_172x224_v2.yaml",
            overrides={},
            options={"dry_run": True, "output_dir": None, "arch_code": "0,0,0,0,0,0,0,0,0,0,0"},
        )
        self.assertIn("outputs", str(result["resolved_output_dir"]))
        self.assertIn("search_v2", str(result["resolved_output_dir"]))
        self.assertEqual(str(result["arch_code"]), "0,0,0,0,0,0,0,0,0,0,0")


if __name__ == "__main__":
    unittest.main()
