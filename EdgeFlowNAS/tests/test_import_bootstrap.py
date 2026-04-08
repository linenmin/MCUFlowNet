"""Tests for shared import path bootstrap helpers."""

import sys
import unittest
from pathlib import Path


class TestImportBootstrap(unittest.TestCase):
    """Validate shared path bootstrap behavior."""

    def test_bootstrap_project_paths_adds_expected_entries(self) -> None:
        """Helper should add project root plus EdgeFlowNet/EdgeFlowNAS code roots."""
        from efnas.utils.import_bootstrap import bootstrap_project_paths

        before = list(sys.path)
        try:
            added = bootstrap_project_paths()
            expected_suffixes = [
                str(Path("MCUFlowNet")),
                str(Path("MCUFlowNet") / "EdgeFlowNet"),
                str(Path("MCUFlowNet") / "EdgeFlowNet" / "code"),
                str(Path("MCUFlowNet") / "EdgeFlowNAS"),
                str(Path("MCUFlowNet") / "EdgeFlowNAS" / "code"),
            ]
            joined = "\n".join(added)
            for suffix in expected_suffixes:
                self.assertIn(suffix, joined)
        finally:
            sys.path[:] = before

    def test_bootstrap_project_paths_is_idempotent(self) -> None:
        """Calling the helper repeatedly should not keep prepending duplicates."""
        from efnas.utils.import_bootstrap import bootstrap_project_paths

        before = list(sys.path)
        try:
            bootstrap_project_paths()
            first = list(sys.path)
            bootstrap_project_paths()
            second = list(sys.path)
            self.assertEqual(first, second)
        finally:
            sys.path[:] = before


if __name__ == "__main__":
    unittest.main()
