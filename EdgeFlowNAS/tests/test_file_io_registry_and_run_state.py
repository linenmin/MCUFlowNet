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

    def test_render_active_finding_hints_only_includes_hard_filter_rules(self) -> None:
        file_io.write_findings_registry(
            self.exp_dir,
            [
                {
                    "id": "A01",
                    "active": True,
                    "generator_hint": "hard rule hint",
                    "enforcement": "hard_filter",
                },
                {
                    "id": "A02",
                    "active": True,
                    "generator_hint": "soft rule hint",
                    "enforcement": "generator_hint_only",
                },
            ],
        )

        rendered = file_io.render_active_finding_hints(self.exp_dir)
        self.assertIn("hard rule hint", rendered)
        self.assertNotIn("soft rule hint", rendered)

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

    def test_record_epoch_metrics_includes_best_fps(self) -> None:
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
                    "epe": 4.5,
                    "fps": 8.2,
                    "sram_kb": 100.0,
                    "cycles_npu": 1,
                    "macs": 1,
                    "micro_insight": "",
                    "epoch": 0,
                    "timestamp": "2026-04-10T00:00:00",
                },
                {
                    "arch_code": "1,0,0,0,0,0,0,0,0,0,0",
                    "epe": 4.2,
                    "fps": 7.8,
                    "sram_kb": 100.0,
                    "cycles_npu": 1,
                    "macs": 1,
                    "micro_insight": "",
                    "epoch": 0,
                    "timestamp": "2026-04-10T00:00:01",
                },
            ],
        )

        coordinator._record_epoch_metrics(epoch=0, new_evaluated=2, duplicates=0)
        metrics_df = file_io.read_epoch_metrics(self.exp_dir)
        self.assertIn("best_fps", metrics_df.columns)
        self.assertEqual(float(metrics_df.iloc[0]["best_fps"]), 8.2)

    def test_append_epoch_metrics_migrates_legacy_schema(self) -> None:
        path = file_io.os.path.join(self.exp_dir, "metadata", "epoch_metrics.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(
                "epoch,total_evaluated,new_evaluated,duplicates,rule_rejected,best_epe,pareto_count,findings_count,assumptions_count,coverage_pct\n"
            )
            f.write("0,16,16,0,0,4.0,5,0,0,0.07\n")

        file_io.append_epoch_metrics(
            self.exp_dir,
            {
                "epoch": 1,
                "total_evaluated": 32,
                "new_evaluated": 16,
                "duplicates": 0,
                "rule_rejected": 0,
                "best_epe": 3.9,
                "best_fps": 8.5,
                "pareto_count": 8,
                "findings_count": 0,
                "assumptions_count": 0,
                "coverage_pct": 0.14,
            },
        )

        metrics_df = file_io.read_epoch_metrics(self.exp_dir)
        self.assertIn("best_fps", metrics_df.columns)
        self.assertEqual(len(metrics_df), 2)
        self.assertTrue(pd.isna(metrics_df.iloc[0]["best_fps"]) or metrics_df.iloc[0]["best_fps"] == "")
        self.assertEqual(float(metrics_df.iloc[1]["best_fps"]), 8.5)


if __name__ == "__main__":
    unittest.main()
