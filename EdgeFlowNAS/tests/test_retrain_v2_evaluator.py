"""Regression tests for retrain_v2 evaluator graph-mode setup."""

import subprocess
import sys
import unittest
from pathlib import Path


class TestRetrainV2Evaluator(unittest.TestCase):
    """Validate eager/graph mode behavior for retrain_v2 evaluation."""

    def test_ensure_graph_mode_disables_eager_execution(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        code = """
import tensorflow as tf
from efnas.engine.retrain_v2_evaluator import _ensure_graph_mode
print("before", tf.executing_eagerly())
_ensure_graph_mode()
print("after", tf.executing_eagerly())
assert tf.executing_eagerly() is False
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
        self.assertIn("before True", result.stdout)
        self.assertIn("after False", result.stdout)


if __name__ == "__main__":
    unittest.main()
