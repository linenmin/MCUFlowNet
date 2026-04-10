import shutil
import tempfile
import unittest

import pandas as pd

from efnas.search import file_io
from efnas.search.coordinator import SearchCoordinator


class TestFileIORegistryAndRunState(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = tempfile.mkdtemp(prefix="efnas_fileio_")
        self.exp_dir = file_io.init_experiment_dir(self.tmp_root, "registry")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_init_creates_findings_json_and_run_state(self) -> None:
        registry_path = file_io.os.path.join(self.exp_dir, "metadata", "findings.json")
        run_state_path = file_io.os.path.join(self.exp_dir, "metadata", "run_state.json")
        self.assertTrue(file_io.os.path.exists(registry_path))
        self.assertTrue(file_io.os.path.exists(run_state_path))

    def test_count_findings_uses_json_registry(self) -> None:
        file_io.write_findings_registry(
            self.exp_dir,
            [
                {"id": "A01", "active": True},
                {"id": "A02", "active": False},
            ],
        )
        self.assertEqual(file_io.count_findings(self.exp_dir), 1)

    def test_infer_start_epoch_prefers_incomplete_run_state_epoch(self) -> None:
        cfg = {
            "llm": {
                "models": {
                    "agent_a_strategist": "gemini/test",
                    "agent_b_generator": "gemini/test",
                    "agent_c_distiller": "gemini/test",
                    "agent_d_scientist": "gemini/test",
                    "agent_d_coder": "gemini/test",
                    "agent_d_rule_manager": "gemini/test",
                },
                "temperature": 1.0,
            },
            "search": {
                "total_epochs": 10,
                "batch_size": 4,
                "scientist_trigger_interval": 5,
                "assumption_confidence_threshold": 0.95,
                "search_space_size": 23328,
            },
            "concurrency": {"max_workers": 1},
            "evaluation": {},
        }
        coordinator = SearchCoordinator(cfg=cfg, exp_dir=self.exp_dir, project_root="dummy_root")
        file_io.append_history_rows(
            self.exp_dir,
            [
                {
                    "arch_code": "0,0,0,0,0,0,0,0,0,0,0",
                    "epe": 4.0,
                    "fps": 6.0,
                    "sram_kb": 100.0,
                    "cycles_npu": 1,
                    "macs": 1,
                    "micro_insight": "",
                    "epoch": 3,
                    "timestamp": "2026-04-10T00:00:00",
                }
            ],
        )
        file_io.write_run_state(
            self.exp_dir,
            {
                "current_epoch": 3,
                "phase": "agent_b",
                "completed": False,
            },
        )

        self.assertEqual(coordinator._infer_start_epoch(), 3)


if __name__ == "__main__":
    unittest.main()
