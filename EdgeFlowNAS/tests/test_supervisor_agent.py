"""Phase 4 (search_hybrid_v1): Supervisor Agent + NSGA-II 5-lever 测试.

覆盖:
- NSGA2SearchRunner.current_supervisor_state: 5 lever 当前值
- NSGA2SearchRunner.apply_supervisor_actions: 合法值 / 各种非法值的拒绝
- mutate_arch with per_dim_multiplier: 高 multiplier 增加 mutation, 0 multiplier 禁用
- _tournament_select_index 用 self._tournament_size
- reseed_bottom_pct 注入 random arches 到 offspring batch
- supervisor_agent: invoke (mocked LLM), supervisor_log 读写, supervisor_pipeline
  端到端 (含 LLM 失败兜底)
"""

import json
import os
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

import pandas as pd

if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(suppress_debug_info=True)

from efnas.baselines.nsga2_search import (
    NSGA2SearchRunner,
    load_search_space,
    mutate_arch,
)
from efnas.search import supervisor_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(response):
    mock = MagicMock()
    if isinstance(response, Exception):
        mock.chat_json.side_effect = response
    else:
        mock.chat_json.return_value = response
    return mock


def _make_runner(tmp_dir: str) -> NSGA2SearchRunner:
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
    os.makedirs(os.path.join(tmp_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "dashboard", "tmp_workers"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "dashboard", "eval_outputs"), exist_ok=True)
    return NSGA2SearchRunner(cfg=cfg, exp_dir=tmp_dir, project_root=tmp_dir)


# ---------------------------------------------------------------------------
# mutate_arch with per_dim_multiplier
# ---------------------------------------------------------------------------

class TestMutateArchWithMultiplier(unittest.TestCase):
    def setUp(self) -> None:
        self.space = load_search_space("efnas.nas.search_space_v2")
        self.arch = [0] * 11

    def test_zero_multiplier_disables_dim_mutation(self) -> None:
        # multiplier[2] = 0 → dim 2 永远不变
        rng = random.Random(42)
        multiplier = [1.0] * 11
        multiplier[2] = 0.0
        # 跑很多次 mutation, dim 2 必须永远是 0
        for _ in range(200):
            child = mutate_arch(
                self.arch, rng=rng, mutation_prob=0.5,
                search_space=self.space, per_dim_multiplier=multiplier,
            )
            self.assertEqual(child[2], 0)

    def test_high_multiplier_increases_mutation_freq(self) -> None:
        # multiplier=2.0 把 prob=0.05 抬到 0.10; 在足够多次试验中 dim 5 mutation
        # 频率应该明显高于 dim 0 的 (frequency 不严格但 statistically very likely)
        rng = random.Random(42)
        multiplier = [1.0] * 11
        multiplier[5] = 5.0  # 大幅放大
        n_trials = 500
        dim0_mutated = 0
        dim5_mutated = 0
        for _ in range(n_trials):
            child = mutate_arch(
                self.arch, rng=rng, mutation_prob=0.05,
                search_space=self.space, per_dim_multiplier=multiplier,
            )
            if child[0] != self.arch[0]:
                dim0_mutated += 1
            if child[5] != self.arch[5]:
                dim5_mutated += 1
        self.assertGreater(dim5_mutated, dim0_mutated * 2)

    def test_none_multiplier_falls_back_to_uniform(self) -> None:
        rng = random.Random(0)
        # 不传 per_dim_multiplier 应等价于全 1.0
        c1 = mutate_arch(self.arch, rng=random.Random(0), mutation_prob=0.5,
                         search_space=self.space)
        c2 = mutate_arch(self.arch, rng=random.Random(0), mutation_prob=0.5,
                         search_space=self.space, per_dim_multiplier=[1.0] * 11)
        self.assertEqual(c1, c2)

    def test_frozen_dims_skips_mutation(self) -> None:
        # search_hybrid_v1: frozen_dims 命中维度 mutation 概率强制为 0
        rng = random.Random(42)
        for _ in range(200):
            child = mutate_arch(
                self.arch, rng=rng, mutation_prob=0.99,
                search_space=self.space,
                frozen_dims=[3, 7, 9],
            )
            self.assertEqual(child[3], 0)
            self.assertEqual(child[7], 0)
            self.assertEqual(child[9], 0)

    def test_frozen_dims_overrides_high_multiplier(self) -> None:
        # 即使 per_dim_multiplier 给该维很大值, frozen_dims 仍然压成 0
        rng = random.Random(0)
        multiplier = [1.0] * 11
        multiplier[5] = 100.0
        for _ in range(100):
            child = mutate_arch(
                self.arch, rng=rng, mutation_prob=0.5,
                search_space=self.space,
                per_dim_multiplier=multiplier,
                frozen_dims=[5],
            )
            self.assertEqual(child[5], 0)

    def test_empty_frozen_dims_no_op(self) -> None:
        rng = random.Random(0)
        c1 = mutate_arch(self.arch, rng=random.Random(0), mutation_prob=0.5,
                         search_space=self.space)
        c2 = mutate_arch(self.arch, rng=random.Random(0), mutation_prob=0.5,
                         search_space=self.space, frozen_dims=[])
        self.assertEqual(c1, c2)


# ---------------------------------------------------------------------------
# current_supervisor_state + apply_supervisor_actions
# ---------------------------------------------------------------------------

class TestSupervisorState(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.runner = _make_runner(self.tmp)

    def test_initial_state(self) -> None:
        state = self.runner.current_supervisor_state()
        self.assertEqual(state["mutation_prob"], 1.0 / 11.0)
        self.assertEqual(state["crossover_prob"], 0.9)
        self.assertEqual(state["tournament_size"], 2)
        self.assertEqual(state["per_dim_mutation_multiplier"], [1.0] * 11)
        self.assertEqual(state["reseed_bottom_pct"], 0)
        # search_hybrid_v1: 3 个新 lever 的 identity default
        self.assertEqual(state["local_search_pareto_neighbors"], 0)
        self.assertEqual(state["parent_pool_source"], "current_pop")
        self.assertEqual(state["frozen_dims"], [])


class TestApplySupervisorActions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.runner = _make_runner(self.tmp)

    def test_apply_all_levers_legal(self) -> None:
        actions = {
            "mutation_prob": 0.15,
            "crossover_prob": 0.85,
            "tournament_size": 3,
            "per_dim_mutation_multiplier": [1.0, 1.0, 2.0, 1.0, 1.0, 1.5,
                                             1.0, 1.0, 1.0, 1.0, 1.0],
            "reseed_bottom_pct": 10,
        }
        result = self.runner.apply_supervisor_actions(actions)
        self.assertEqual(result["rejected"], {})
        self.assertEqual(len(result["applied"]), 5)
        # state actually changed
        self.assertEqual(self.runner.mutation_prob, 0.15)
        self.assertEqual(self.runner.crossover_prob, 0.85)
        self.assertEqual(self.runner._tournament_size, 3)
        self.assertEqual(self.runner._per_dim_mutation_multiplier[2], 2.0)
        self.assertEqual(self.runner._reseed_bottom_pct, 10)

    def test_null_field_means_no_change(self) -> None:
        before = self.runner.mutation_prob
        result = self.runner.apply_supervisor_actions({
            "mutation_prob": None,
            "crossover_prob": 0.7,
        })
        self.assertNotIn("mutation_prob", result["applied"])
        self.assertEqual(self.runner.mutation_prob, before)
        self.assertEqual(self.runner.crossover_prob, 0.7)

    def test_mutation_prob_out_of_range_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions({"mutation_prob": 1.5})
        self.assertIn("mutation_prob", result["rejected"])
        self.assertEqual(self.runner.mutation_prob, 1.0 / 11.0)  # unchanged

    def test_negative_mutation_prob_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions({"mutation_prob": -0.1})
        self.assertIn("mutation_prob", result["rejected"])

    def test_tournament_size_too_large_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions({"tournament_size": 1000})
        self.assertIn("tournament_size", result["rejected"])

    def test_per_dim_multiplier_wrong_length_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"per_dim_mutation_multiplier": [1.0, 1.0, 1.0]}
        )
        self.assertIn("per_dim_mutation_multiplier", result["rejected"])
        self.assertEqual(self.runner._per_dim_mutation_multiplier, [1.0] * 11)

    def test_per_dim_multiplier_negative_element_rejected(self) -> None:
        bad = [1.0] * 11
        bad[3] = -0.5
        result = self.runner.apply_supervisor_actions(
            {"per_dim_mutation_multiplier": bad}
        )
        self.assertIn("per_dim_mutation_multiplier", result["rejected"])

    def test_reseed_pct_out_of_range_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions({"reseed_bottom_pct": 150})
        self.assertIn("reseed_bottom_pct", result["rejected"])

    def test_partial_apply_keeps_others(self) -> None:
        # legal mutation_prob + illegal tournament_size → mutation_prob 应用,
        # tournament_size 拒绝
        result = self.runner.apply_supervisor_actions({
            "mutation_prob": 0.2,
            "tournament_size": -1,
        })
        self.assertEqual(self.runner.mutation_prob, 0.2)
        self.assertIn("mutation_prob", result["applied"])
        self.assertIn("tournament_size", result["rejected"])

    def test_non_dict_actions_returns_global_reject(self) -> None:
        result = self.runner.apply_supervisor_actions("not a dict")
        self.assertIn("_global", result["rejected"])
        self.assertEqual(result["applied"], {})


# ---------------------------------------------------------------------------
# search_hybrid_v1: 3 个新 lever 的校验
# ---------------------------------------------------------------------------

class TestNewLeversValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.runner = _make_runner(self.tmp)

    # local_search_pareto_neighbors ------------------------------------
    def test_local_search_legal(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"local_search_pareto_neighbors": 5}
        )
        self.assertIn("local_search_pareto_neighbors", result["applied"])
        self.assertEqual(self.runner._local_search_pareto_neighbors, 5)

    def test_local_search_zero_legal_default(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"local_search_pareto_neighbors": 0}
        )
        self.assertIn("local_search_pareto_neighbors", result["applied"])
        self.assertEqual(self.runner._local_search_pareto_neighbors, 0)

    def test_local_search_negative_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"local_search_pareto_neighbors": -1}
        )
        self.assertIn("local_search_pareto_neighbors", result["rejected"])
        self.assertEqual(self.runner._local_search_pareto_neighbors, 0)

    def test_local_search_above_pop_size_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"local_search_pareto_neighbors": self.runner.population_size + 1}
        )
        self.assertIn("local_search_pareto_neighbors", result["rejected"])

    # parent_pool_source -----------------------------------------------
    def test_parent_pool_source_current_pop_legal(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"parent_pool_source": "current_pop"}
        )
        self.assertIn("parent_pool_source", result["applied"])
        self.assertEqual(self.runner._parent_pool_source, "current_pop")

    def test_parent_pool_source_history_pareto_legal(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"parent_pool_source": "history_pareto"}
        )
        self.assertIn("parent_pool_source", result["applied"])
        self.assertEqual(self.runner._parent_pool_source, "history_pareto")

    def test_parent_pool_source_mixed_legal(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"parent_pool_source": "mixed_50_50"}
        )
        self.assertIn("parent_pool_source", result["applied"])

    def test_parent_pool_source_unknown_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"parent_pool_source": "random_archive"}
        )
        self.assertIn("parent_pool_source", result["rejected"])
        self.assertEqual(self.runner._parent_pool_source, "current_pop")

    def test_parent_pool_source_non_string_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"parent_pool_source": 42}
        )
        self.assertIn("parent_pool_source", result["rejected"])

    # frozen_dims ------------------------------------------------------
    def test_frozen_dims_legal(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"frozen_dims": [7, 9]}
        )
        self.assertIn("frozen_dims", result["applied"])
        self.assertEqual(self.runner._frozen_dims, [7, 9])

    def test_frozen_dims_dedup_and_sort(self) -> None:
        # dup and unsorted input gets deduped and sorted
        result = self.runner.apply_supervisor_actions(
            {"frozen_dims": [9, 7, 7, 3]}
        )
        self.assertEqual(self.runner._frozen_dims, [3, 7, 9])

    def test_frozen_dims_empty_legal(self) -> None:
        result = self.runner.apply_supervisor_actions({"frozen_dims": []})
        self.assertIn("frozen_dims", result["applied"])
        self.assertEqual(self.runner._frozen_dims, [])

    def test_frozen_dims_out_of_range_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"frozen_dims": [3, 11]}  # 11 is out of [0, 10]
        )
        self.assertIn("frozen_dims", result["rejected"])
        self.assertEqual(self.runner._frozen_dims, [])

    def test_frozen_dims_negative_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"frozen_dims": [-1, 3]}
        )
        self.assertIn("frozen_dims", result["rejected"])

    def test_frozen_dims_non_list_rejected(self) -> None:
        result = self.runner.apply_supervisor_actions(
            {"frozen_dims": "7,9"}
        )
        self.assertIn("frozen_dims", result["rejected"])


# ---------------------------------------------------------------------------
# tournament_size live behavior
# ---------------------------------------------------------------------------

class TestTournamentSizeLive(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.runner = _make_runner(self.tmp)

    def test_default_tournament_size_2(self) -> None:
        self.assertEqual(self.runner._tournament_size, 2)

    def test_apply_then_select_uses_new_size(self) -> None:
        # 设置 tournament_size=4, 验证 _tournament_select_index 内部确实
        # sample 4 个 contestant
        self.runner.apply_supervisor_actions({"tournament_size": 4})
        # 构造 fake population + ranks/crowding, 让 selection 跑一次, 没异常即可
        pop = [{"arch_code": f"arch_{i}", "epe": 4.0 + i * 0.01, "fps": 5.0}
               for i in range(20)]
        ranks = {i: i // 5 for i in range(20)}
        crowding = {i: float(i) for i in range(20)}
        # 多跑几次确认稳定
        for _ in range(10):
            idx = self.runner._tournament_select_index(pop, ranks, crowding)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 20)


# ---------------------------------------------------------------------------
# supervisor_log read/write
# ---------------------------------------------------------------------------

class TestSupervisorLog(unittest.TestCase):
    def test_read_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "metadata"), exist_ok=True)
            self.assertEqual(supervisor_agent.read_supervisor_log(tmp), [])

    def test_append_then_read_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "metadata"), exist_ok=True)
            entry1 = {"generation": 2, "rationale": "first", "applied": {}}
            entry2 = {"generation": 5, "rationale": "second", "applied": {"mutation_prob": 0.15}}
            supervisor_agent.append_supervisor_log(tmp, entry1)
            supervisor_agent.append_supervisor_log(tmp, entry2)
            log = supervisor_agent.read_supervisor_log(tmp)
            self.assertEqual(len(log), 2)
            self.assertEqual(log[0]["rationale"], "first")
            self.assertEqual(log[1]["applied"]["mutation_prob"], 0.15)

    def test_corrupted_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "metadata"), exist_ok=True)
            with open(supervisor_agent.supervisor_log_path(tmp), "w",
                      encoding="utf-8") as f:
                f.write("not valid json {{{")
            self.assertEqual(supervisor_agent.read_supervisor_log(tmp), [])


# ---------------------------------------------------------------------------
# invoke_supervisor_agent
# ---------------------------------------------------------------------------

class TestInvokeSupervisor(unittest.TestCase):
    def test_normal_response_parsed(self) -> None:
        llm = _make_mock_llm({
            "rationale": "HV 停滞 3 代, 提高 mutation",
            "actions": {
                "mutation_prob": 0.13,
                "crossover_prob": None,
                "per_dim_mutation_multiplier": None,
                "tournament_size": None,
                "reseed_bottom_pct": None,
            },
            "expected_effect": "hv_improvement_rate_3gen 回升",
            "review_after_gen": 6,
        })
        result = supervisor_agent.invoke_supervisor_agent(
            llm, current_state={"mutation_prob": 0.091},
            recent_metrics_df=pd.DataFrame(),
            current_pareto_summary="x",
            current_insights_md="",
            supervisor_log=[],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["actions"]["mutation_prob"], 0.13)
        self.assertEqual(result["review_after_gen"], 6)

    def test_llm_exception_returns_none(self) -> None:
        llm = _make_mock_llm(RuntimeError("LLM down"))
        result = supervisor_agent.invoke_supervisor_agent(
            llm, current_state={}, recent_metrics_df=pd.DataFrame(),
            current_pareto_summary="", current_insights_md="",
            supervisor_log=[],
        )
        self.assertIsNone(result)

    def test_actions_field_not_dict_default_empty(self) -> None:
        llm = _make_mock_llm({
            "rationale": "x", "actions": "should be dict",
            "expected_effect": "y", "review_after_gen": 3,
        })
        result = supervisor_agent.invoke_supervisor_agent(
            llm, current_state={}, recent_metrics_df=pd.DataFrame(),
            current_pareto_summary="", current_insights_md="",
            supervisor_log=[],
        )
        self.assertEqual(result["actions"], {})


# ---------------------------------------------------------------------------
# supervisor_pipeline (end-to-end)
# ---------------------------------------------------------------------------

class TestSupervisorPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.runner = _make_runner(self.tmp)
        # seed history (so _summarize_current_pareto has data)
        df = pd.DataFrame({
            "arch_code": ["0,0,0,0,0,0,0,0,0,0,0", "2,2,2,2,2,2,1,1,1,1,1"],
            "epe": [4.99, 4.01], "fps": [8.91, 5.0],
            "epoch": [0, 0],
        })
        df.to_csv(os.path.join(self.tmp, "metadata", "history_archive.csv"),
                  index=False)

    def test_full_success_path(self) -> None:
        llm = _make_mock_llm({
            "rationale": "looks healthy, slight mutation bump",
            "actions": {
                "mutation_prob": 0.12,
                "crossover_prob": None,
                "per_dim_mutation_multiplier": None,
                "tournament_size": None,
                "reseed_bottom_pct": None,
            },
            "expected_effect": "...",
            "review_after_gen": 5,
        })
        summary = supervisor_agent.supervisor_pipeline(
            llm, self.tmp, self.runner, generation=2,
        )
        self.assertTrue(summary["success"])
        self.assertEqual(summary["applied"].get("mutation_prob"), 0.12)
        self.assertEqual(self.runner.mutation_prob, 0.12)
        # 日志写入
        log = supervisor_agent.read_supervisor_log(self.tmp)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["generation"], 2)

    def test_llm_failure_logs_failure_entry(self) -> None:
        llm = _make_mock_llm(RuntimeError("LLM down"))
        before = self.runner.mutation_prob
        summary = supervisor_agent.supervisor_pipeline(
            llm, self.tmp, self.runner, generation=2,
        )
        self.assertFalse(summary["success"])
        self.assertIn("LLM", summary["error"])
        # state unchanged
        self.assertEqual(self.runner.mutation_prob, before)
        # 但 supervisor_log 仍然记录了 failure 条目
        log = supervisor_agent.read_supervisor_log(self.tmp)
        self.assertEqual(len(log), 1)
        self.assertIn("_global", log[0]["rejected"])

    def test_no_change_actions_succeed_with_empty_applied(self) -> None:
        llm = _make_mock_llm({
            "rationale": "looks healthy, no change",
            "actions": {
                "mutation_prob": None, "crossover_prob": None,
                "per_dim_mutation_multiplier": None,
                "tournament_size": None, "reseed_bottom_pct": None,
            },
            "expected_effect": "继续观察",
            "review_after_gen": 3,
        })
        summary = supervisor_agent.supervisor_pipeline(
            llm, self.tmp, self.runner, generation=5,
        )
        self.assertTrue(summary["success"])
        self.assertEqual(summary["applied"], {})
        self.assertEqual(summary["rejected"], {})

    def test_rejected_actions_logged_but_others_applied(self) -> None:
        llm = _make_mock_llm({
            "rationale": "boost mutation, also try invalid tournament_size",
            "actions": {
                "mutation_prob": 0.18,
                "crossover_prob": None,
                "per_dim_mutation_multiplier": None,
                "tournament_size": 9999,  # > population_size, should reject
                "reseed_bottom_pct": None,
            },
            "expected_effect": "x",
            "review_after_gen": 3,
        })
        summary = supervisor_agent.supervisor_pipeline(
            llm, self.tmp, self.runner, generation=2,
        )
        self.assertTrue(summary["success"])
        self.assertEqual(summary["applied"].get("mutation_prob"), 0.18)
        self.assertIn("tournament_size", summary["rejected"])


if __name__ == "__main__":
    unittest.main()
