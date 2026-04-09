import os
import tempfile
import unittest

from efnas.search import file_io


class TestFileIOResumeSelection(unittest.TestCase):
    def test_find_latest_experiment_dir_ignores_non_search_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid_old = file_io.init_experiment_dir(tmp_dir, "search_v2_old")
            valid_new = file_io.init_experiment_dir(tmp_dir, "search_v2_new")

            timing_probe_dir = os.path.join(tmp_dir, "timing_probe_ref")
            os.makedirs(os.path.join(timing_probe_dir, "metadata"), exist_ok=True)
            with open(
                os.path.join(timing_probe_dir, "metadata", "timing_probe_summary.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("{}")

            selected = file_io.find_latest_experiment_dir(tmp_dir)
            self.assertEqual(os.path.normpath(selected), os.path.normpath(valid_new))


if __name__ == "__main__":
    unittest.main()
