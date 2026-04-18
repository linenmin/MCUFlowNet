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

    def test_resolve_project_paths_from_wrapper_anchor_stays_at_mcu_root(self) -> None:
        """Wrapper anchors under EdgeFlowNAS should resolve MCUFlowNet as the shared root."""
        from efnas.utils.import_bootstrap import resolve_project_paths

        anchor = Path("/tmp/test/MCUFlowNet/EdgeFlowNAS/wrappers/run_retrain_v2_sintel_test.py")
        paths = resolve_project_paths(anchor_file=anchor)
        self.assertEqual(paths["mcu_root"].parts[-2:], ("test", "MCUFlowNet"))
        self.assertEqual(paths["edgeflownet_root"].parts[-3:], ("test", "MCUFlowNet", "EdgeFlowNet"))
        self.assertEqual(paths["edgeflownas_root"].parts[-3:], ("test", "MCUFlowNet", "EdgeFlowNAS"))


if __name__ == "__main__":
    unittest.main()
