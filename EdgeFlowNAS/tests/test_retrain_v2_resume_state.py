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
from efnas.engine.retrain_v2_trainer import _reconcile_resume_progress

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


if __name__ == "__main__":
    unittest.main()
