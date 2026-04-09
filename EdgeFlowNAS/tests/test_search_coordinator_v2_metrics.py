import unittest
import sys
import types
from unittest.mock import patch

import pandas as pd

if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(suppress_debug_info=True)

from efnas.search.coordinator import SearchCoordinator


class TestSearchCoordinatorV2Metrics(unittest.TestCase):
    def test_record_epoch_metrics_uses_v2_space_size_for_coverage(self) -> None:
        cfg = {
            "llm": {
                "models": {
                    "agent_a_strategist": "gemini/gemini-3.1-pro-preview",
                    "agent_b_generator": "gemini/gemini-3.1-flash-lite-preview",
                    "agent_c_distiller": "gemini/gemini-3-flash-preview",
                    "agent_d_scientist": "gemini/gemini-3.1-pro-preview",
                    "agent_d_coder": "gemini/gemini-3-flash-preview",
                    "agent_d_rule_manager": "gemini/gemini-3.1-pro-preview",
                },
                "temperature": 0.4,
            },
            "search": {
                "total_epochs": 1,
                "batch_size": 16,
                "scientist_trigger_interval": 5,
                "assumption_confidence_threshold": 0.95,
                "search_space_size": 23328,
            },
            "concurrency": {"max_workers": 1},
            "evaluation": {},
        }
        coordinator = SearchCoordinator(cfg=cfg, exp_dir="dummy_exp", project_root="dummy_root")
        df = pd.DataFrame(
            {
                "arch_code": ["0,0,0,0,0,0,0,0,0,0,0"] * 233,
                "epe": [1.0] * 233,
                "fps": [10.0] * 233,
            }
        )

        with patch("efnas.search.coordinator.file_io.read_history", return_value=df), patch(
            "efnas.search.coordinator.file_io.count_findings", return_value=0
        ), patch("efnas.search.coordinator.file_io.read_assumptions", return_value=[]), patch(
            "efnas.search.coordinator.file_io.append_epoch_metrics"
        ) as mock_append:
            coordinator._record_epoch_metrics(epoch=0, new_evaluated=10, duplicates=2)

        metrics = mock_append.call_args.args[1]
        self.assertEqual(metrics["coverage_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
