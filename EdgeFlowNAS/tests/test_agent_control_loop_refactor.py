import os
import json
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(suppress_debug_info=True)

from efnas.search import agents, file_io


class _CaptureLLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def chat_json(self, role, system_prompt, user_message, **kwargs):
        self.calls.append(
            {
                "role": role,
                "system_prompt": system_prompt,
                "user_message": user_message,
                "kwargs": kwargs,
            }
        )
        return self.result


class TestAgentControlLoopRefactor(unittest.TestCase):
    def test_agent_a_uses_fact_only_inputs_and_coverage_view(self) -> None:
        llm = _CaptureLLM(
            {
                "strategic_reflection": "focus on uncovered pareto regions",
                "allocation": {
                    "free_exploration": {
                        "count": 16,
                        "direction_describe": "extend balanced regime",
                    }
                },
            }
        )
        history_df = pd.DataFrame(
            {
                "arch_code": [
                    "0,0,0,0,0,0,0,0,0,0,0",
                    "2,2,0,1,2,0,0,0,0,0,0",
                    "1,0,0,1,0,0,0,0,0,0,0",
                ],
                "epe": [4.3, 4.0, 4.6],
                "fps": [9.1, 6.0, 8.4],
                "epoch": [0, 1, 2],
                "micro_insight": ["insight-a", "insight-b", "insight-c"],
            }
        )
        metrics_df = pd.DataFrame(
            {
                "epoch": [0, 1, 2],
                "new_evaluated": [16, 15, 14],
                "duplicates": [0, 1, 2],
                "pareto_count": [8, 10, 11],
                "best_epe": [4.3, 4.0, 4.0],
                "best_fps": [9.1, 9.1, 9.1],
            }
        )

        with patch("efnas.search.agents.file_io.read_history", return_value=history_df), patch(
            "efnas.search.agents.file_io.read_epoch_metrics",
            create=True,
            return_value=metrics_df,
        ) as mock_metrics, patch(
            "efnas.search.agents.file_io.read_strategy_log"
        ) as mock_strategy, patch(
            "efnas.search.agents.file_io.read_assumptions"
        ) as mock_assumptions, patch(
            "efnas.search.agents.file_io.read_findings"
        ) as mock_findings, patch(
            "efnas.search.agents.file_io.append_strategy_log"
        ):
            agents.invoke_agent_a(llm, exp_dir="dummy_exp", epoch=3, batch_size=16)

        self.assertEqual(len(llm.calls), 1)
        user_msg = llm.calls[0]["user_message"]
        mock_metrics.assert_called_once()
        mock_strategy.assert_not_called()
        mock_assumptions.assert_not_called()
        mock_findings.assert_not_called()
        self.assertIn("Pareto", user_msg)
        self.assertNotIn("过往战术日志", user_msg)
        self.assertNotIn("当前猜想簿", user_msg)
        self.assertNotIn("绝对真理碑", user_msg)
        self.assertIn("搜索空间覆盖率结构", user_msg)
        self.assertIn("dim[0] 分布", user_msg)
        self.assertIn("当前 Pareto 前沿完整列表", user_msg)
        self.assertIn("最近 Pareto 动态变化", user_msg)
        self.assertNotIn("Top-5 最低 EPE", user_msg)
        self.assertNotIn("近3轮是否有改进", user_msg)
        self.assertNotIn("insight-a", user_msg)

    def test_agent_a_receives_full_pareto_and_recent_front_dynamics(self) -> None:
        llm = _CaptureLLM(
            {
                "strategic_reflection": "use full front and recent turnover",
                "allocation": {
                    "free_exploration": {
                        "count": 48,
                        "direction_describe": "probe middle-front gaps",
                    }
                },
            }
        )
        history_rows = []
        for i in range(24):
            history_rows.append(
                {
                    "arch_code": f"{i % 3},{(i + 1) % 3},0,0,{i % 3},{(i + 2) % 3},0,0,{i % 2},0,{(i + 1) % 2}",
                    "epe": 4.8 - i * 0.03,
                    "fps": 5.0 + i * 0.12,
                    "epoch": i // 2,
                }
            )
        history_df = pd.DataFrame(history_rows)
        metrics_df = pd.DataFrame(
            {
                "epoch": list(range(6)),
                "new_evaluated": [48, 48, 47, 48, 46, 48],
                "duplicates": [0, 0, 1, 0, 2, 0],
                "rule_rejected": [0, 0, 0, 0, 0, 0],
                "pareto_count": [10, 12, 13, 13, 14, 15],
                "best_epe": [4.2, 4.15, 4.10, 4.05, 4.02, 3.99],
                "best_fps": [8.2, 8.4, 8.6, 8.8, 8.9, 9.0],
            }
        )

        with patch("efnas.search.agents.file_io.read_history", return_value=history_df), patch(
            "efnas.search.agents.file_io.read_epoch_metrics",
            create=True,
            return_value=metrics_df,
        ), patch("efnas.search.agents.file_io.append_strategy_log"):
            agents.invoke_agent_a(llm, exp_dir="dummy_exp", epoch=6, batch_size=48)

        user_msg = llm.calls[0]["user_message"]
        self.assertIn("## 当前 Pareto 前沿完整列表", user_msg)
        self.assertIn("## 最近 Pareto 动态变化", user_msg)
        self.assertIn("entered_count", user_msg)
        self.assertIn("removed_count", user_msg)
        self.assertIn("24 条", user_msg)
        self.assertNotIn("Lowest-EPE end", user_msg)
        self.assertNotIn("Highest-FPS end", user_msg)

    def test_agent_b_uses_generator_hints_instead_of_raw_findings_markdown(self) -> None:
        llm = _CaptureLLM({"generated_candidates": ["0,0,0,0,0,0,0,0,0,0,0"]})
        allocation = {"free_exploration": {"count": 1, "direction_describe": "test"}}

        with patch(
            "efnas.search.agents.file_io.render_active_finding_hints",
            create=True,
            return_value="When targeting FPS >= 8.0, avoid H2Out=1.",
        ) as mock_hints, patch(
            "efnas.search.agents.file_io.read_findings"
        ) as mock_read_findings, patch(
            "efnas.search.agents.file_io.get_evaluated_arch_codes",
            return_value=set(),
        ):
            agents.invoke_agent_b(llm, exp_dir="dummy_exp", allocation=allocation, batch_size=1)

        self.assertEqual(len(llm.calls), 1)
        user_msg = llm.calls[0]["user_message"]
        mock_hints.assert_called_once()
        mock_read_findings.assert_not_called()
        self.assertIn("avoid H2Out=1", user_msg)
        self.assertNotIn("findings.md", user_msg)

    def test_agent_d1_uses_fact_table_without_micro_insight_column(self) -> None:
        llm = _CaptureLLM({"assumptions": [{"id": "A01", "description": "conditional trend"}]})
        history_df = pd.DataFrame(
            {
                "arch_code": ["0,0,0,0,0,0,0,0,0,0,0"],
                "epe": [4.3],
                "fps": [9.1],
                "micro_insight": ["do not leak this"],
                "epoch": [0],
            }
        )

        with patch("efnas.search.agents.file_io.read_history", return_value=history_df), patch(
            "efnas.search.agents.file_io.get_next_assumption_id",
            return_value=1,
        ), patch(
            "efnas.search.agents._extract_topic_summary",
            return_value="### 现有猜想: (无)",
        ), patch(
            "efnas.search.agents.file_io.append_assumptions"
        ):
            agents.invoke_agent_d1(llm, exp_dir="dummy_exp")

        self.assertEqual(len(llm.calls), 1)
        call = llm.calls[0]
        self.assertNotIn("micro_insight", call["user_message"])
        self.assertIn("条件趋势", call["system_prompt"])
        self.assertIn("存在反例", call["system_prompt"])


class TestAgentD3FindingRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = tempfile.mkdtemp(prefix="efnas_finding_registry_")
        self.exp_dir = file_io.init_experiment_dir(self.tmp_root, "registry")
        file_io.write_assumptions(
            self.exp_dir,
            [{"id": "A01", "description": "test description"}],
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_d3_writes_json_registry_entry_and_removes_assumption(self) -> None:
        llm = _CaptureLLM(
            {
                "finding": {
                    "id": "A01",
                    "title": "Test rule",
                    "summary": "Test summary",
                    "generator_hint": "Avoid this pattern.",
                    "enforcement": "hard_filter",
                    "scope": {"target_fps_min": 8.0},
                }
            }
        )

        agents.invoke_agent_d3(
            llm=llm,
            exp_dir=self.exp_dir,
            assumption={"id": "A01", "description": "test description"},
            confidence=0.97,
        )

        findings = file_io.read_findings_registry(self.exp_dir)
        assumptions = file_io.read_assumptions(self.exp_dir)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["id"], "A01")
        self.assertEqual(findings[0]["title"], "Test rule")
        self.assertEqual(assumptions, [])


class TestSearchV2Config(unittest.TestCase):
    def test_search_v2_uses_48_batch_17_epochs_and_interval_2(self) -> None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "search_v2.yaml",
        )
        with open(config_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.assertIn("total_epochs: 17", text)
        self.assertIn("batch_size: 48", text)
        self.assertIn("scientist_trigger_interval: 2", text)

    def test_agent_a_prompt_forbids_100_percent_desert_only_from_endpoint_stability(self) -> None:
        from efnas.search import prompts

        self.assertIn("禁止仅因 best EPE/best FPS 端点稳定", prompts.AGENT_A_SYSTEM)
        self.assertIn("100% 投入荒漠探索", prompts.AGENT_A_SYSTEM)


if __name__ == "__main__":
    unittest.main()
