"""Phase 3 (search_hybrid_v1): scientist_agent 测试.

Mock LLM client + 真实 sandbox + 真实 file_io.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

import pandas as pd

if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(suppress_debug_info=True)

from efnas.search import file_io, insights, scientist_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(response):
    """Single-response mock; chat_json returns response (or raises if Exception)."""
    mock = MagicMock()
    if isinstance(response, Exception):
        mock.chat_json.side_effect = response
    else:
        mock.chat_json.return_value = response
    return mock


def _make_sequenced_llm(*responses):
    """LLM that returns different responses per call (or raises Exception items)."""
    mock = MagicMock()
    side_effects = []
    for r in responses:
        if isinstance(r, Exception):
            side_effects.append(r)
        else:
            side_effects.append(r)
    mock.chat_json.side_effect = side_effects
    return mock


def _make_history_df(n: int = 6) -> pd.DataFrame:
    """Synthetic small history with diverse arch_codes."""
    archs = [
        "0,0,0,0,0,0,0,0,0,0,0",
        "2,2,2,2,2,2,1,1,1,1,1",
        "1,1,1,1,1,1,0,0,0,0,0",
        "2,1,2,1,2,2,0,1,1,0,0",
        "0,2,1,2,0,1,1,0,1,0,1",
        "1,1,0,0,1,1,0,1,0,1,0",
    ][:n]
    return pd.DataFrame({
        "arch_code": archs,
        "epe": [4.99, 4.01, 4.5, 4.10, 4.78, 4.32][:n],
        "fps": [8.91, 5.0, 6.5, 5.5, 5.8, 6.7][:n],
        "epoch": [0, 0, 0, 1, 1, 1][:n],
    })


def _make_temp_exp_dir() -> str:
    tmp = tempfile.mkdtemp(prefix="scientist_test_")
    os.makedirs(os.path.join(tmp, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "dashboard", "tmp_workers"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "dashboard", "eval_outputs"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "scripts"), exist_ok=True)
    return tmp


# ---------------------------------------------------------------------------
# _matches_pattern
# ---------------------------------------------------------------------------

class TestMatchesPattern(unittest.TestCase):
    def test_full_wildcard_matches_anything(self) -> None:
        pattern = [None] * 11
        self.assertTrue(scientist_agent._matches_pattern("0,1,2,0,0,1,0,0,0,1,0", pattern))
        self.assertTrue(scientist_agent._matches_pattern("2,2,2,2,2,2,1,1,1,1,1", pattern))

    def test_specific_position_match(self) -> None:
        # require dim 2 = 2 AND dim 5 = 2
        pattern = [None, None, 2, None, None, 2, None, None, None, None, None]
        self.assertTrue(scientist_agent._matches_pattern("0,1,2,0,0,2,0,0,0,1,0", pattern))
        self.assertFalse(scientist_agent._matches_pattern("0,1,1,0,0,2,0,0,0,1,0", pattern))

    def test_wrong_length_returns_false(self) -> None:
        self.assertFalse(scientist_agent._matches_pattern("0,1,2", [None] * 11))

    def test_string_int_comparison(self) -> None:
        # pattern uses int 2, arch_code parts are str "2"
        pattern = [None, None, 2, None, None, None, None, None, None, None, None]
        self.assertTrue(scientist_agent._matches_pattern("0,1,2,0,0,0,0,0,0,0,0", pattern))


# ---------------------------------------------------------------------------
# resolve_vela_queries
# ---------------------------------------------------------------------------

class TestResolveVelaQueries(unittest.TestCase):
    def setUp(self) -> None:
        self.exp_dir = _make_temp_exp_dir()
        self.history_df = _make_history_df()

    def test_by_arch_codes_filters_to_evaluated(self) -> None:
        queries = [{
            "insight_id": "I-001",
            "by_arch_codes": ["0,0,0,0,0,0,0,0,0,0,0", "9,9,9,9,9,9,9,9,9,9,9"],
            "limit": 5,
        }]
        results = scientist_agent.resolve_vela_queries(
            queries, self.history_df, self.exp_dir,
        )
        self.assertIn("I-001", results)
        # 9,9,... 不在 history; 应该只剩第一条
        self.assertEqual(results["I-001"]["match_count"], 1)
        self.assertEqual(
            results["I-001"]["matched_archs"][0]["arch_code"],
            "0,0,0,0,0,0,0,0,0,0,0",
        )

    def test_by_pattern_with_wildcards(self) -> None:
        # 找 EB0 (dim 2) = 2 的 arch
        queries = [{
            "insight_id": "I-002",
            "by_arch_code_pattern": [None, None, 2, None, None, None,
                                      None, None, None, None, None],
            "limit": 5,
        }]
        results = scientist_agent.resolve_vela_queries(
            queries, self.history_df, self.exp_dir,
        )
        # history 里 dim 2 = 2 的: "2,2,2,2,2,2,...", "2,1,2,1,2,2,..."
        # 还有 "0,2,1,..." 的 dim 2 = 1, 不算
        match_codes = [m["arch_code"] for m in results["I-002"]["matched_archs"]]
        self.assertIn("2,2,2,2,2,2,1,1,1,1,1", match_codes)
        self.assertIn("2,1,2,1,2,2,0,1,1,0,0", match_codes)

    def test_pareto_filter(self) -> None:
        # 拿 history 中 Pareto 前沿的 arch
        queries = [{
            "insight_id": "I-003",
            "by_arch_code_pattern": [None] * 11,
            "from_pareto_front_only": True,
            "limit": 20,
        }]
        results = scientist_agent.resolve_vela_queries(
            queries, self.history_df, self.exp_dir,
        )
        # 至少 best_epe 的 arch 在 (epe=4.01, fps=5.0) 是 Pareto
        match_codes = [m["arch_code"] for m in results["I-003"]["matched_archs"]]
        self.assertIn("2,2,2,2,2,2,1,1,1,1,1", match_codes)
        # best_fps 的 arch (epe=4.99, fps=8.91) 也在
        self.assertIn("0,0,0,0,0,0,0,0,0,0,0", match_codes)

    def test_sort_by_epe_asc_picks_lowest(self) -> None:
        queries = [{
            "insight_id": "I-004",
            "by_arch_code_pattern": [None] * 11,
            "sort_by_epe": "asc",
            "limit": 2,
        }]
        results = scientist_agent.resolve_vela_queries(
            queries, self.history_df, self.exp_dir,
        )
        epes = [m["epe"] for m in results["I-004"]["matched_archs"]]
        self.assertEqual(epes[0], 4.01)  # 最低 EPE 排第一
        self.assertEqual(len(epes), 2)

    def test_no_match_returns_note(self) -> None:
        queries = [{
            "insight_id": "I-005",
            "by_arch_codes": ["7,7,7,7,7,7,1,1,1,1,1"],
            "limit": 5,
        }]
        results = scientist_agent.resolve_vela_queries(
            queries, self.history_df, self.exp_dir,
        )
        self.assertEqual(results["I-005"]["match_count"], 0)
        self.assertIn("note", results["I-005"])

    def test_invalid_pattern_length_returns_note(self) -> None:
        queries = [{
            "insight_id": "I-006",
            "by_arch_code_pattern": [None, None, 2],  # too short
            "limit": 5,
        }]
        results = scientist_agent.resolve_vela_queries(
            queries, self.history_df, self.exp_dir,
        )
        self.assertEqual(results["I-006"]["match_count"], 0)
        self.assertIn("invalid pattern", results["I-006"]["note"])

    def test_limit_capped_at_20(self) -> None:
        queries = [{
            "insight_id": "I-007",
            "by_arch_code_pattern": [None] * 11,
            "limit": 100,
        }]
        results = scientist_agent.resolve_vela_queries(
            queries, self.history_df, self.exp_dir,
        )
        # history 只有 6 行, 自然不会超 6; 但确认 limit 上限工作
        self.assertLessEqual(results["I-007"]["match_count"], 20)


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

class TestInvokeStageA(unittest.TestCase):
    def test_normal_drafts(self) -> None:
        llm = _make_mock_llm({
            "drafts": [
                {"id": "I-001", "status": "active", "title": "test", "body": "正文"},
                {"id": "I-002", "status": "retired", "title": "old", "body": ""},
            ],
        })
        drafts = scientist_agent.invoke_scientist_stage_a(
            llm, history_df=_make_history_df(),
            metrics_df=pd.DataFrame(),
            prev_insights_md="",
            generation=2, total_generations=16,
            next_id_hint="I-001",
        )
        self.assertEqual(len(drafts), 2)
        self.assertEqual(drafts[0]["id"], "I-001")

    def test_filters_invalid_id(self) -> None:
        llm = _make_mock_llm({
            "drafts": [
                {"id": "BAD-no-prefix", "status": "active", "title": "x", "body": ""},
                {"id": "I-001", "status": "active", "title": "ok", "body": ""},
            ],
        })
        drafts = scientist_agent.invoke_scientist_stage_a(
            llm, history_df=_make_history_df(),
            metrics_df=None, prev_insights_md="",
            generation=0, total_generations=16,
            next_id_hint="I-001",
        )
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["id"], "I-001")

    def test_filters_invalid_status(self) -> None:
        llm = _make_mock_llm({
            "drafts": [
                {"id": "I-001", "status": "Active", "title": "x", "body": ""},  # capital A
                {"id": "I-002", "status": "active", "title": "ok", "body": ""},
            ],
        })
        drafts = scientist_agent.invoke_scientist_stage_a(
            llm, history_df=_make_history_df(),
            metrics_df=None, prev_insights_md="",
            generation=0, total_generations=16,
            next_id_hint="I-001",
        )
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["id"], "I-002")

    def test_llm_exception_returns_empty(self) -> None:
        llm = _make_mock_llm(RuntimeError("LLM down"))
        drafts = scientist_agent.invoke_scientist_stage_a(
            llm, history_df=_make_history_df(),
            metrics_df=None, prev_insights_md="",
            generation=0, total_generations=16,
            next_id_hint="I-001",
        )
        self.assertEqual(drafts, [])

    def test_empty_history_returns_empty_no_call(self) -> None:
        llm = MagicMock()
        drafts = scientist_agent.invoke_scientist_stage_a(
            llm, history_df=pd.DataFrame(),
            metrics_df=None, prev_insights_md="",
            generation=0, total_generations=16,
            next_id_hint="I-001",
        )
        self.assertEqual(drafts, [])
        llm.chat_json.assert_not_called()


# ---------------------------------------------------------------------------
# Stage B-1
# ---------------------------------------------------------------------------

class TestInvokeStageB1(unittest.TestCase):
    def test_normal_plan(self) -> None:
        llm = _make_mock_llm({
            "vela_queries": [{
                "insight_id": "I-001",
                "purpose": "test",
                "by_arch_codes": ["0,0,0,0,0,0,0,0,0,0,0"],
                "limit": 1,
            }],
            "verifications": [{
                "insight_id": "I-001",
                "purpose": "stat",
                "code": "import json\nprint(json.dumps({'count': 1}))",
            }],
            "annotations_no_code": [{
                "insight_id": "I-002", "annotation": "no need",
            }],
        })
        plan = scientist_agent.invoke_scientist_stage_b1(
            llm, drafts=[], history_df=_make_history_df(),
        )
        self.assertEqual(len(plan["vela_queries"]), 1)
        self.assertEqual(len(plan["verifications"]), 1)
        self.assertEqual(len(plan["annotations_no_code"]), 1)

    def test_llm_exception_returns_none(self) -> None:
        llm = _make_mock_llm(RuntimeError("down"))
        plan = scientist_agent.invoke_scientist_stage_b1(
            llm, drafts=[], history_df=_make_history_df(),
        )
        self.assertIsNone(plan)

    def test_missing_fields_default_empty(self) -> None:
        llm = _make_mock_llm({})  # no fields at all
        plan = scientist_agent.invoke_scientist_stage_b1(
            llm, drafts=[], history_df=_make_history_df(),
        )
        self.assertEqual(plan["vela_queries"], [])
        self.assertEqual(plan["verifications"], [])
        self.assertEqual(plan["annotations_no_code"], [])

    def test_non_list_field_replaced_with_empty(self) -> None:
        llm = _make_mock_llm({
            "vela_queries": "should be list",
            "verifications": [{"insight_id": "I-001", "code": "print(1)"}],
            "annotations_no_code": None,
        })
        plan = scientist_agent.invoke_scientist_stage_b1(
            llm, drafts=[], history_df=_make_history_df(),
        )
        self.assertEqual(plan["vela_queries"], [])
        self.assertEqual(len(plan["verifications"]), 1)
        self.assertEqual(plan["annotations_no_code"], [])


# ---------------------------------------------------------------------------
# execute_verifications (real sandbox)
# ---------------------------------------------------------------------------

class TestExecuteVerifications(unittest.TestCase):
    def test_simple_verification_runs(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8",
        ) as f:
            f.write("arch_code,epe,fps\n0,0,0,0,0,0,0,0,0,0,0,4.99,8.91\n")
            csv_path = f.name
        try:
            verifications = [{
                "insight_id": "I-001",
                "purpose": "count rows",
                "code": (
                    "import sys, json\n"
                    "import pandas as pd\n"
                    "df = pd.read_csv(sys.argv[1])\n"
                    "print(json.dumps({'rows': len(df)}))\n"
                ),
            }]
            results = scientist_agent.execute_verifications(
                verifications,
                history_csv_path=csv_path,
                query_results={},
                timeout=10,
            )
            self.assertEqual(results["I-001"]["status"], "ok")
            self.assertEqual(results["I-001"]["parsed_json"], {"rows": 1})
        finally:
            os.remove(csv_path)

    def test_disallowed_import_caught_as_validation_error(self) -> None:
        verifications = [{
            "insight_id": "I-001",
            "purpose": "bad",
            "code": "import os\nprint(os.getcwd())",
        }]
        results = scientist_agent.execute_verifications(
            verifications,
            history_csv_path="dummy",
            query_results={},
            timeout=5,
        )
        self.assertEqual(results["I-001"]["status"], "validation_error")

    def test_empty_code_validation_error(self) -> None:
        verifications = [{
            "insight_id": "I-001", "code": "",
        }]
        results = scientist_agent.execute_verifications(
            verifications, history_csv_path="dummy", query_results={},
        )
        self.assertEqual(results["I-001"]["status"], "validation_error")
        self.assertIn("empty", results["I-001"]["error"])


# ---------------------------------------------------------------------------
# Stage B-2
# ---------------------------------------------------------------------------

class TestInvokeStageB2(unittest.TestCase):
    def test_returns_markdown(self) -> None:
        llm = _make_mock_llm({
            "insights_md": "# Search Insights\n\n### I-001 (active): t\nbody\n",
        })
        md = scientist_agent.invoke_scientist_stage_b2(
            llm, drafts=[], query_results={}, verification_results={},
            annotations_no_code=[],
        )
        self.assertIn("### I-001", md)

    def test_llm_failure_returns_none(self) -> None:
        llm = _make_mock_llm(RuntimeError())
        md = scientist_agent.invoke_scientist_stage_b2(
            llm, drafts=[], query_results={}, verification_results={},
            annotations_no_code=[],
        )
        self.assertIsNone(md)

    def test_empty_md_field_returns_none(self) -> None:
        llm = _make_mock_llm({"insights_md": ""})
        md = scientist_agent.invoke_scientist_stage_b2(
            llm, drafts=[], query_results={}, verification_results={},
            annotations_no_code=[],
        )
        self.assertIsNone(md)


# ---------------------------------------------------------------------------
# _drafts_to_markdown (fallback)
# ---------------------------------------------------------------------------

class TestDraftsToMarkdown(unittest.TestCase):
    def test_renders_valid_drafts(self) -> None:
        drafts = [
            {"id": "I-001", "status": "active", "title": "x", "body": "y"},
            {"id": "I-002", "status": "retired", "title": "z", "body": ""},
        ]
        md = scientist_agent._drafts_to_markdown(drafts)
        self.assertIn("### I-001 (active): x", md)
        self.assertIn("### I-002 (retired): z", md)
        # 解析回去 → 2 条
        parsed = insights.parse_insights(md)
        self.assertEqual(len(parsed), 2)

    def test_skips_invalid_drafts(self) -> None:
        drafts = [
            {"id": "BAD", "status": "active", "title": "x"},  # invalid id
            {"id": "I-002", "status": "Active", "title": "y"},  # invalid status
            {"id": "I-003", "status": "active", "title": "ok", "body": "b"},
        ]
        md = scientist_agent._drafts_to_markdown(drafts)
        parsed = insights.parse_insights(md)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["id"], "I-003")


# ---------------------------------------------------------------------------
# scientist_pipeline (full orchestration)
# ---------------------------------------------------------------------------

class TestScientistPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.exp_dir = _make_temp_exp_dir()
        # Seed history_archive.csv
        df = _make_history_df()
        df.to_csv(
            os.path.join(self.exp_dir, "metadata", "history_archive.csv"),
            index=False,
        )
        # Seed insights.md from template
        file_io._touch_md(
            os.path.join(self.exp_dir, "metadata", "insights.md"),
            default="# Search Insights\n\n---\n\n",
        )

    def test_full_success_path(self) -> None:
        llm = _make_sequenced_llm(
            # Stage A
            {"drafts": [
                {"id": "I-001", "status": "active",
                 "title": "test insight", "body": "EB0=2 archs have lower EPE"},
            ]},
            # Stage B-1
            {
                "vela_queries": [],
                "verifications": [],
                "annotations_no_code": [
                    {"insight_id": "I-001", "annotation": "no verification needed"},
                ],
            },
            # Stage B-2
            {
                "insights_md": (
                    "# Search Insights\n\n"
                    "### I-001 (active): test insight\n\n"
                    "EB0=2 archs have lower EPE (no verification needed)\n"
                ),
            },
        )
        summary = scientist_agent.scientist_pipeline(
            llm, self.exp_dir, generation=2, total_generations=16,
        )
        self.assertTrue(summary["success"])
        self.assertEqual(summary["drafts_count"], 1)
        self.assertEqual(summary["error"], "")
        self.assertIn("stage_a", summary["stages_completed"])
        self.assertIn("stage_b1", summary["stages_completed"])
        self.assertIn("stage_b2", summary["stages_completed"])

        # 文件应该被改写
        with open(os.path.join(self.exp_dir, "metadata", "insights.md"),
                  "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("### I-001 (active): test insight", content)

        # 备份文件应存在
        backup_path = os.path.join(
            self.exp_dir, "metadata", "insights.md.gen2.bak",
        )
        self.assertTrue(os.path.exists(backup_path))

    def test_stage_a_failure_does_not_modify_insights(self) -> None:
        llm = _make_mock_llm(RuntimeError("stage A down"))
        # 写入 baseline 内容
        with open(os.path.join(self.exp_dir, "metadata", "insights.md"),
                  "w", encoding="utf-8") as f:
            f.write("# Search Insights\n\n### I-PRE (active): existing\nbody\n")
        summary = scientist_agent.scientist_pipeline(
            llm, self.exp_dir, generation=2, total_generations=16,
        )
        self.assertFalse(summary["success"])
        self.assertIn("Stage A", summary["error"])

        # insights.md 不应被改
        with open(os.path.join(self.exp_dir, "metadata", "insights.md"),
                  "r", encoding="utf-8") as f:
            self.assertIn("I-PRE", f.read())

    def test_stage_b1_failure_writes_drafts_fallback(self) -> None:
        llm = _make_sequenced_llm(
            # Stage A succeeds
            {"drafts": [
                {"id": "I-001", "status": "active", "title": "fallback test",
                 "body": "stage A ok"},
            ]},
            # Stage B-1 fails (raise exception on second call)
            RuntimeError("B1 down"),
        )
        summary = scientist_agent.scientist_pipeline(
            llm, self.exp_dir, generation=2, total_generations=16,
        )
        self.assertTrue(summary["success"])  # success because fallback wrote
        self.assertIn("Stage B-1 failed", summary["error"])

        with open(os.path.join(self.exp_dir, "metadata", "insights.md"),
                  "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("### I-001 (active): fallback test", content)

    def test_stage_b2_failure_writes_drafts_fallback(self) -> None:
        llm = _make_sequenced_llm(
            # Stage A
            {"drafts": [
                {"id": "I-001", "status": "active", "title": "B2 fallback",
                 "body": "test"},
            ]},
            # Stage B-1 succeeds with empty plan
            {"vela_queries": [], "verifications": [], "annotations_no_code": []},
            # Stage B-2 fails
            RuntimeError("B2 down"),
        )
        summary = scientist_agent.scientist_pipeline(
            llm, self.exp_dir, generation=2, total_generations=16,
        )
        self.assertTrue(summary["success"])
        self.assertIn("Stage B-2 failed", summary["error"])

        with open(os.path.join(self.exp_dir, "metadata", "insights.md"),
                  "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("### I-001 (active): B2 fallback", content)


# ---------------------------------------------------------------------------
# read/write/backup helpers
# ---------------------------------------------------------------------------

class TestInsightsIOHelpers(unittest.TestCase):
    def test_round_trip_atomic_write(self) -> None:
        tmp = _make_temp_exp_dir()
        scientist_agent.write_insights_md_atomic(tmp, "# Hello\n\n### I-001 (active): x\nbody\n")
        text = scientist_agent.read_insights_md(tmp)
        self.assertIn("### I-001", text)

    def test_backup_creates_gen_file(self) -> None:
        tmp = _make_temp_exp_dir()
        scientist_agent.write_insights_md_atomic(tmp, "before\n")
        backup = scientist_agent.backup_insights_md(tmp, generation=5)
        self.assertIsNotNone(backup)
        self.assertTrue(os.path.exists(backup))
        self.assertTrue(backup.endswith("insights.md.gen5.bak"))

    def test_backup_when_no_file_returns_none(self) -> None:
        tmp = _make_temp_exp_dir()
        # 无 insights.md
        backup = scientist_agent.backup_insights_md(tmp, generation=0)
        self.assertIsNone(backup)


if __name__ == "__main__":
    unittest.main()
