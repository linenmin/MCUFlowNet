"""Phase 2 (search_hybrid_v1): Warmstart Agent + NSGA-II hook 测试.

覆盖:
- invoke_warmstart_agent: 正常解析、LLM 抛错回 fallback、非 dict 响应、
  非 list arch_codes 字段
- compute_warmstart_diagnostics: 合法/非法/重复混合统计、entropy 计算、
  rationale 透传
- save_warmstart_diagnostics: 落盘 + JSON 可读
- warmstart_pipeline: 端到端 (mocked LLM), 返回去重合法列表 + 落盘 diagnostics
- NSGA2SearchRunner._consume_external_initial_population: 合法/非法/重复混合
  时的 partial random fill 行为
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(suppress_debug_info=True)

from efnas.search import warmstart_agent
from efnas.baselines.nsga2_search import (
    NSGA2SearchRunner,
    load_search_space,
)


# ---------------------------------------------------------------------------
# Helper: mock LLM client
# ---------------------------------------------------------------------------

def _make_mock_llm(response):
    """Build a MagicMock LLMClient whose chat_json returns ``response``."""
    mock = MagicMock()
    if isinstance(response, Exception):
        mock.chat_json.side_effect = response
    else:
        mock.chat_json.return_value = response
    return mock


# ---------------------------------------------------------------------------
# invoke_warmstart_agent
# ---------------------------------------------------------------------------

class TestInvokeWarmstartAgent(unittest.TestCase):
    def test_normal_response_parsed(self) -> None:
        llm = _make_mock_llm({
            "rationale": "spread across extremes",
            "arch_codes": ["0,0,0,0,0,0,0,0,0,0,0", "2,2,2,2,2,2,1,1,1,1,1"],
        })
        result = warmstart_agent.invoke_warmstart_agent(llm, population_size=2)
        self.assertEqual(result["rationale"], "spread across extremes")
        self.assertEqual(len(result["arch_codes"]), 2)
        self.assertIsNotNone(result["raw_response"])

    def test_llm_exception_returns_fallback(self) -> None:
        llm = _make_mock_llm(RuntimeError("LLM timeout"))
        result = warmstart_agent.invoke_warmstart_agent(llm, population_size=50)
        self.assertEqual(result["rationale"], "")
        self.assertEqual(result["arch_codes"], [])
        self.assertIsNone(result["raw_response"])

    def test_non_dict_response_returns_fallback(self) -> None:
        llm = _make_mock_llm("not a dict")
        result = warmstart_agent.invoke_warmstart_agent(llm)
        self.assertEqual(result["arch_codes"], [])
        self.assertEqual(result["rationale"], "")

    def test_arch_codes_field_not_list_treated_as_empty(self) -> None:
        llm = _make_mock_llm({"rationale": "x", "arch_codes": "should be a list"})
        result = warmstart_agent.invoke_warmstart_agent(llm)
        self.assertEqual(result["arch_codes"], [])
        self.assertEqual(result["rationale"], "x")

    def test_strips_blank_arch_codes(self) -> None:
        llm = _make_mock_llm({
            "rationale": "",
            "arch_codes": ["0,0,0,0,0,0,0,0,0,0,0", "", "  ", "1,1,1,1,1,1,1,1,1,1,1"],
        })
        result = warmstart_agent.invoke_warmstart_agent(llm)
        self.assertEqual(len(result["arch_codes"]), 2)


# ---------------------------------------------------------------------------
# compute_warmstart_diagnostics
# ---------------------------------------------------------------------------

class TestComputeWarmstartDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = load_search_space("efnas.nas.search_space_v2")

    def test_all_valid_unique(self) -> None:
        codes = [
            "0,0,0,0,0,0,0,0,0,0,0",
            "2,2,2,2,2,2,1,1,1,1,1",
            "1,1,1,1,1,1,0,0,0,0,0",
        ]
        diag = warmstart_agent.compute_warmstart_diagnostics(
            codes, search_space=self.search_space, requested_count=3,
            rationale="spread", llm_model="test-model",
        )
        self.assertEqual(diag["returned_count"], 3)
        self.assertEqual(diag["unique_valid_count"], 3)
        self.assertEqual(diag["invalid_count"], 0)
        self.assertEqual(diag["duplicate_within_batch"], 0)
        self.assertEqual(diag["valid_count"], 3)
        self.assertEqual(diag["llm_model"], "test-model")
        self.assertEqual(diag["rationale"], "spread")
        self.assertEqual(len(diag["per_dim_entropy"]), 11)
        # 三个互斥个体, dim 0 取值 (0, 2, 1) → entropy ≈ log(3)
        import math
        self.assertAlmostEqual(diag["per_dim_entropy"][0], math.log(3), places=2)

    def test_mix_invalid_and_duplicate(self) -> None:
        codes = [
            "0,0,0,0,0,0,0,0,0,0,0",  # valid
            "9,9,9,9,9,9,9,9,9,9,9",  # invalid (out of range)
            "0,0,0,0,0,0,0,0,0,0,0",  # duplicate of #1
            "TOO,SHORT",              # invalid (parse error)
            "1,1,1,1,1,1,0,0,0,0,0",  # valid
        ]
        diag = warmstart_agent.compute_warmstart_diagnostics(
            codes, search_space=self.search_space, requested_count=5,
        )
        self.assertEqual(diag["returned_count"], 5)
        self.assertEqual(diag["invalid_count"], 2)
        self.assertEqual(diag["duplicate_within_batch"], 1)
        self.assertEqual(diag["unique_valid_count"], 2)
        # valid_count = unique + duplicate (一开始通过 validate 但被去重)
        self.assertEqual(diag["valid_count"], 3)

    def test_empty_input(self) -> None:
        diag = warmstart_agent.compute_warmstart_diagnostics(
            [], search_space=self.search_space, requested_count=50,
        )
        self.assertEqual(diag["returned_count"], 0)
        self.assertEqual(diag["unique_valid_count"], 0)
        # entropy 全 0
        self.assertEqual(diag["per_dim_entropy"], [0.0] * 11)


# ---------------------------------------------------------------------------
# save_warmstart_diagnostics
# ---------------------------------------------------------------------------

class TestSaveWarmstartDiagnostics(unittest.TestCase):
    def test_writes_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            diag = {"requested_count": 50, "valid_count": 45, "rationale": "test"}
            path = warmstart_agent.save_warmstart_diagnostics(tmp, diag)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, diag)


# ---------------------------------------------------------------------------
# warmstart_pipeline (端到端)
# ---------------------------------------------------------------------------

class TestWarmstartPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = load_search_space("efnas.nas.search_space_v2")

    def test_end_to_end_returns_valid_unique_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = _make_mock_llm({
                "rationale": "test",
                "arch_codes": [
                    "0,0,0,0,0,0,0,0,0,0,0",
                    "2,2,2,2,2,2,1,1,1,1,1",
                    "0,0,0,0,0,0,0,0,0,0,0",  # duplicate
                    "BAD",                     # invalid
                ],
            })
            valid = warmstart_agent.warmstart_pipeline(
                llm, tmp, search_space=self.search_space, population_size=4,
                llm_model="test-model",
            )
            self.assertEqual(len(valid), 2)
            self.assertIn("0,0,0,0,0,0,0,0,0,0,0", valid)
            # diagnostics 应已落盘
            self.assertTrue(os.path.exists(
                os.path.join(tmp, "metadata", "warmstart_diagnostics.json"),
            ))

    def test_llm_failure_returns_empty_list_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = _make_mock_llm(RuntimeError("LLM down"))
            valid = warmstart_agent.warmstart_pipeline(
                llm, tmp, search_space=self.search_space, population_size=50,
            )
            self.assertEqual(valid, [])
            # diagnostics 仍然应该落盘 (空 valid)
            diag_path = os.path.join(tmp, "metadata", "warmstart_diagnostics.json")
            self.assertTrue(os.path.exists(diag_path))
            with open(diag_path, "r", encoding="utf-8") as f:
                diag = json.load(f)
            self.assertEqual(diag["unique_valid_count"], 0)


# ---------------------------------------------------------------------------
# NSGA2SearchRunner._consume_external_initial_population (Phase 2.1 hook)
# ---------------------------------------------------------------------------

class TestNSGA2ConsumeExternalPopulation(unittest.TestCase):
    """单元测试 NSGA-II runner 的 warmstart hook 行为, 不实际跑评估."""

    def _make_runner(self, external):
        cfg = {
            "search": {
                "total_evaluations": 800,
                "population_size": 50,
                "search_space_module": "efnas.nas.search_space_v2",
                "search_space_size": 23328,
                "seed": 2026,
            },
            "concurrency": {"max_workers": 1},
            "evaluation": {},
            "files": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "metadata"), exist_ok=True)
            os.makedirs(os.path.join(tmp, "dashboard", "tmp_workers"), exist_ok=True)
            os.makedirs(os.path.join(tmp, "dashboard", "eval_outputs"), exist_ok=True)
            runner = NSGA2SearchRunner(
                cfg=cfg, exp_dir=tmp, project_root=tmp,
                external_initial_population=external,
            )
            return runner

    def test_consume_all_valid(self) -> None:
        external = [
            "0,0,0,0,0,0,0,0,0,0,0",
            "1,1,1,1,1,1,0,0,0,0,0",
            "2,2,2,2,2,2,1,1,1,1,1",
        ]
        runner = self._make_runner(external)
        with patch.object(runner, '_sample_unique_random_arches') as fallback:
            arches, dups = runner._consume_external_initial_population(target_count=3)
            fallback.assert_not_called()
        self.assertEqual(len(arches), 3)
        self.assertEqual(dups, 0)

    def test_partial_fill_when_external_short(self) -> None:
        external = ["0,0,0,0,0,0,0,0,0,0,0"]
        runner = self._make_runner(external)
        with patch.object(
            runner, '_sample_unique_random_arches',
            return_value=(["1,1,1,1,1,1,0,0,0,0,0", "2,2,2,2,2,2,1,1,1,1,1"], 0),
        ) as fallback:
            arches, dups = runner._consume_external_initial_population(target_count=3)
            fallback.assert_called_once()
            # 调用时 shortfall = 3 - 1 = 2
            args, kwargs = fallback.call_args
            self.assertEqual(args[0], 2)
        self.assertEqual(len(arches), 3)

    def test_skips_invalid_codes(self) -> None:
        external = [
            "0,0,0,0,0,0,0,0,0,0,0",  # valid
            "9,9,9,9,9,9,9,9,9,9,9",  # invalid (range)
            "BAD",                     # invalid (parse)
            "1,1,1,1,1,1,0,0,0,0,0",  # valid
        ]
        runner = self._make_runner(external)
        with patch.object(
            runner, '_sample_unique_random_arches',
            return_value=([], 0),
        ) as fallback:
            arches, dups = runner._consume_external_initial_population(target_count=2)
            # 2 valid 直接吸收, 不需要 fallback
            fallback.assert_not_called()
        self.assertEqual(len(arches), 2)

    def test_skips_duplicates_within_external(self) -> None:
        external = [
            "0,0,0,0,0,0,0,0,0,0,0",
            "0,0,0,0,0,0,0,0,0,0,0",  # dup
            "1,1,1,1,1,1,0,0,0,0,0",
        ]
        runner = self._make_runner(external)
        with patch.object(
            runner, '_sample_unique_random_arches',
            return_value=([], 0),
        ) as fallback:
            arches, dups = runner._consume_external_initial_population(target_count=2)
            fallback.assert_not_called()
        self.assertEqual(len(arches), 2)
        self.assertEqual(dups, 1)


if __name__ == "__main__":
    unittest.main()
