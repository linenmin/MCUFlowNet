"""Regression tests for retrain_v2 resume state reconciliation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestRetrainV2ResumeState(unittest.TestCase):
    def test_resume_prefers_newer_last_checkpoint_meta(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = {
                "trainer_state": {"epoch": 19, "global_step": 100, "best_mean_epe": 2.0, "no_improve_evals": 1},
                "ckpt_meta": {"epoch": 20, "global_step": 110},
            }
            code = f"""
import json
from efnas.engine.retrain_v2_resume import _reconcile_resume_progress

payload = json.loads(r'''{json.dumps(payload)}''')
start_epoch, global_step = _reconcile_resume_progress(payload["trainer_state"], [payload["ckpt_meta"]])
print(start_epoch, global_step)
assert start_epoch == 21
assert global_step == 110
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

    def test_resume_from_best_uses_checkpoint_meta_over_later_trainer_state(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        payload = {
            "trainer_state": {"epoch": 36, "global_step": 45360, "best_mean_epe": 2.0, "no_improve_evals": 2},
            "ckpt_meta": {"epoch": 32, "global_step": 40320},
        }
        code = f"""
import json
from efnas.engine.retrain_v2_resume import _reconcile_resume_progress

payload = json.loads(r'''{json.dumps(payload)}''')
start_epoch, global_step = _reconcile_resume_progress(
    payload["trainer_state"],
    [payload["ckpt_meta"]],
    prefer_checkpoint_meta=True,
)
print(start_epoch, global_step)
assert start_epoch == 33
assert global_step == 40320
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

    def test_trim_retrain_histories_drops_rows_after_resume_epoch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        payload = {
            "histories": {
                "fast": [{"epoch": 32, "epe": 2.1}, {"epoch": 34, "epe": float("nan")}],
                "knee": [{"epoch": 32, "epe": 1.9}, {"epoch": 34, "epe": float("nan")}],
            },
            "comparison": [{"epoch": 32, "mean_epe": 2.0}, {"epoch": 34, "mean_epe": float("nan")}],
        }
        code = f"""
import json
from efnas.engine.retrain_v2_resume import _trim_retrain_histories

payload = json.loads(r'''{json.dumps(payload)}''')
histories, comparison = _trim_retrain_histories(payload["histories"], payload["comparison"], max_epoch=32)
print(histories, comparison)
assert len(histories["fast"]) == 1
assert len(histories["knee"]) == 1
assert len(comparison) == 1
assert histories["fast"][0]["epoch"] == 32
assert comparison[0]["epoch"] == 32
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
