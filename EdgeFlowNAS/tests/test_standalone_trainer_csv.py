"""Regression tests for standalone_trainer CSV history writing."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestStandaloneTrainerCsv(unittest.TestCase):
    def test_write_eval_history_accepts_rows_with_new_columns(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "eval_history.csv"
            code = f"""
from pathlib import Path
from efnas.engine.standalone_trainer import _write_eval_history

rows = [
    {{"epoch": 18, "loss": 1.0, "epe": 2.5, "best_epe": 2.5}},
    {{"epoch": 20, "loss": 0.9, "epe": 2.4, "best_epe": 2.4, "sintel_epe": 6.1, "best_sintel_epe": 6.1}},
]
csv_path = Path(r"{csv_path}")
_write_eval_history(csv_path, rows)
print(csv_path.read_text(encoding="utf-8"))
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("sintel_epe", csv_text)
            self.assertIn("best_sintel_epe", csv_text)
            self.assertIn("18,1.0,2.5,2.5", csv_text)


if __name__ == "__main__":
    unittest.main()
