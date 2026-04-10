import os
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
    def test_agent_a_uses_fact_only_inputs(self) -> None:
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
                ],
                "epe": [4.3, 4.0],
                "fps": [9.1, 6.0],
                "epoch": [0, 1],
                "micro_insight": ["a", "b"],
            }
        )
        metrics_df = pd.DataFrame(
            {
                "epoch": [0, 1],
                "new_evaluated": [16, 15],
                "duplicates": [0, 1],
                "pareto_count": [8, 10],
                "best_epe": [4.3, 4.0],
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


if __name__ == "__main__":
    unittest.main()
