import tempfile
import unittest
from pathlib import Path

from efnas.search.file_io import prune_vela_tflite_artifacts


class TestFileIOPrune(unittest.TestCase):
    def test_prune_vela_tflite_artifacts_removes_only_tflite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir)
            vela_dir = exp_dir / "dashboard" / "eval_outputs" / "run_000000000" / "analysis" / "vela_tmp" / "arch_0000"
            vela_dir.mkdir(parents=True, exist_ok=True)

            tflite_a = vela_dir / "arch_0000.tflite"
            tflite_b = vela_dir / "arch_0000_vela.tflite"
            csv_per_layer = vela_dir / "arch_0000_per-layer.csv"
            csv_summary = vela_dir / "arch_0000_summary_Grove_Sys_Config.csv"
            txt_detail = vela_dir / "detailed_performance.txt"

            tflite_a.write_text("a", encoding="utf-8")
            tflite_b.write_text("b", encoding="utf-8")
            csv_per_layer.write_text("layer,metric\n", encoding="utf-8")
            csv_summary.write_text("cycles_npu,nn_macs\n1,2\n", encoding="utf-8")
            txt_detail.write_text("details", encoding="utf-8")

            removed = prune_vela_tflite_artifacts(str(exp_dir))
            self.assertEqual(removed, 2)
            self.assertFalse(tflite_a.exists())
            self.assertFalse(tflite_b.exists())
            self.assertTrue(csv_per_layer.exists())
            self.assertTrue(csv_summary.exists())
            self.assertTrue(txt_detail.exists())


if __name__ == "__main__":
    unittest.main()
