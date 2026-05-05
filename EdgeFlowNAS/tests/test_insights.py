"""Phase 1.5 (search_hybrid_v1): insights.md 解析与 ID 管理测试.

覆盖最小契约的全部行为:
- parse_insights: 各种合法/不合法 heading + body 边界
- list_active / list_active_ids: status 过滤
- next_insight_id: 空文件、纯 sequential、混合、自定义宽度
- validate_id: 合法/不合法 ID 形式
- find_insight_by_id, count_by_status
- 模板文件存在且可解析为空
- init_experiment_dir 写出可读的 insights.md
"""

import os
import tempfile
import unittest

from efnas.search import file_io, insights


class TestParseInsights(unittest.TestCase):
    def test_empty_text_returns_empty(self) -> None:
        self.assertEqual(insights.parse_insights(""), [])

    def test_only_header_no_insights_returns_empty(self) -> None:
        text = "# Search Insights\n\n<!-- 注释 -->\n\n---\n\n"
        self.assertEqual(insights.parse_insights(text), [])

    def test_single_active_insight(self) -> None:
        text = (
            "# Search Insights\n\n---\n\n"
            "### I-001 (active): EB0=2 与 DB1=2 强相关\n"
            "正文一段。\n"
            "另起一段。\n"
        )
        result = insights.parse_insights(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "I-001")
        self.assertEqual(result[0]["status"], "active")
        self.assertEqual(result[0]["title"], "EB0=2 与 DB1=2 强相关")
        self.assertIn("正文一段", result[0]["body"])

    def test_multiple_mixed_status(self) -> None:
        text = (
            "### I-001 (active): A\nbody A\n\n"
            "### I-002 (retired): B\nbody B\n\n"
            "### I-003 (under_review): C\nbody C\n"
        )
        result = insights.parse_insights(text)
        self.assertEqual([r["id"] for r in result], ["I-001", "I-002", "I-003"])
        self.assertEqual([r["status"] for r in result], ["active", "retired", "under_review"])

    def test_body_includes_subheadings_that_dont_match(self) -> None:
        # ### 但不是 INSIGHT_HEADING_RE 的标题应该归入正文
        text = (
            "### I-001 (active): main\n"
            "正文起.\n"
            "### Subsection without insight format\n"
            "继续正文.\n"
            "### I-002 (active): next\n"
            "next body.\n"
        )
        result = insights.parse_insights(text)
        self.assertEqual(len(result), 2)
        self.assertIn("Subsection without insight format", result[0]["body"])
        self.assertIn("继续正文", result[0]["body"])

    def test_malformed_heading_with_unknown_status_ignored(self) -> None:
        # status 必须严格三选一; "Active" 大写应被忽略
        text = (
            "### I-001 (Active): wrong case\nbody A\n"
            "### I-002 (active): correct\nbody B\n"
        )
        result = insights.parse_insights(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "I-002")

    def test_body_with_python_code_block_preserved(self) -> None:
        text = (
            "### I-001 (active): with code\n"
            "Some prose.\n\n"
            "```python\n"
            "df.query('eb0 == 2').epe.mean()\n"
            "```\n\n"
            "More prose.\n"
        )
        result = insights.parse_insights(text)
        self.assertEqual(len(result), 1)
        self.assertIn("```python", result[0]["body"])
        self.assertIn("df.query", result[0]["body"])

    def test_id_with_dashes_and_alphanumerics(self) -> None:
        text = "### I-EB0-DB1 (active): semantic ID\nbody\n"
        result = insights.parse_insights(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "I-EB0-DB1")

    def test_crlf_line_endings_handled(self) -> None:
        text = (
            "### I-001 (active): crlf test\r\n"
            "body line\r\n"
        )
        result = insights.parse_insights(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "crlf test")


class TestListActive(unittest.TestCase):
    def test_filters_only_active(self) -> None:
        text = (
            "### I-001 (active): a\nbody\n"
            "### I-002 (retired): b\nbody\n"
            "### I-003 (active): c\nbody\n"
            "### I-004 (under_review): d\nbody\n"
        )
        active = insights.list_active_insights(text)
        self.assertEqual([it["id"] for it in active], ["I-001", "I-003"])
        self.assertEqual(insights.list_active_ids(text), ["I-001", "I-003"])


class TestNextInsightId(unittest.TestCase):
    def test_empty_returns_001(self) -> None:
        self.assertEqual(insights.next_insight_id(""), "I-001")

    def test_only_header_returns_001(self) -> None:
        self.assertEqual(
            insights.next_insight_id("# Insights\n\n---\n\n"),
            "I-001",
        )

    def test_sequential_ids(self) -> None:
        text = (
            "### I-001 (active): a\nbody\n"
            "### I-002 (retired): b\nbody\n"
        )
        self.assertEqual(insights.next_insight_id(text), "I-003")

    def test_skips_non_sequential_ids(self) -> None:
        # I-EB0 类语义化 ID 不参与下一个数字编号
        text = (
            "### I-001 (active): a\nbody\n"
            "### I-EB0-DB1 (active): semantic\nbody\n"
        )
        self.assertEqual(insights.next_insight_id(text), "I-002")

    def test_max_id_used_not_count(self) -> None:
        # 即使有 retire 后留下的 I-005, 下一个仍然是 I-006
        text = (
            "### I-001 (active): a\nbody\n"
            "### I-005 (retired): b\nbody\n"
        )
        self.assertEqual(insights.next_insight_id(text), "I-006")

    def test_custom_width(self) -> None:
        self.assertEqual(insights.next_insight_id("", width=4), "I-0001")


class TestValidateId(unittest.TestCase):
    def test_valid_forms(self) -> None:
        for valid in ["I-001", "I-EB0-DB1", "I-A1B2", "I-x"]:
            self.assertTrue(insights.validate_id(valid),
                            f"should be valid: {valid}")

    def test_invalid_forms(self) -> None:
        for invalid in ["", "I", "I-", "001", "I_001", "I-001 ", " I-001",
                        "I-中文", "I-001!"]:
            self.assertFalse(insights.validate_id(invalid),
                             f"should be invalid: {invalid}")


class TestFindInsightById(unittest.TestCase):
    def test_found(self) -> None:
        text = "### I-001 (active): a\nbody A\n### I-002 (retired): b\nbody B\n"
        item = insights.find_insight_by_id(text, "I-002")
        self.assertIsNotNone(item)
        self.assertEqual(item["status"], "retired")

    def test_not_found(self) -> None:
        text = "### I-001 (active): a\nbody\n"
        self.assertIsNone(insights.find_insight_by_id(text, "I-999"))


class TestCountByStatus(unittest.TestCase):
    def test_returns_all_three_keys_even_when_zero(self) -> None:
        counts = insights.count_by_status("")
        self.assertEqual(counts, {"active": 0, "retired": 0, "under_review": 0})

    def test_counts_correctly(self) -> None:
        text = (
            "### I-001 (active): a\nbody\n"
            "### I-002 (retired): b\nbody\n"
            "### I-003 (active): c\nbody\n"
            "### I-004 (under_review): d\nbody\n"
            "### I-005 (active): e\nbody\n"
        )
        counts = insights.count_by_status(text)
        self.assertEqual(counts["active"], 3)
        self.assertEqual(counts["retired"], 1)
        self.assertEqual(counts["under_review"], 1)


class TestTemplateFile(unittest.TestCase):
    """模板文件本身必须存在且 parse_insights 返回空 list (没有真实 insight)."""

    def test_template_exists_and_parses_to_empty(self) -> None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "efnas", "search", "templates", "insights_template.md",
        )
        self.assertTrue(os.path.exists(path), f"模板缺失: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # 模板里只有 header 和注释, 不含真实 insight
        self.assertEqual(insights.parse_insights(text), [])
        # next_insight_id 应该返回 I-001
        self.assertEqual(insights.next_insight_id(text), "I-001")
        # 必须含 "# Search Insights" 标题
        self.assertIn("# Search Insights", text)


class TestInitExperimentDirCreatesInsightsMd(unittest.TestCase):
    """init_experiment_dir 必须 eagerly 创建 insights.md 并使其可解析."""

    def test_insights_md_created_and_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = file_io.init_experiment_dir(tmp, "test_phase15")
            insights_path = os.path.join(exp_dir, "metadata", "insights.md")
            self.assertTrue(os.path.exists(insights_path))
            with open(insights_path, "r", encoding="utf-8") as f:
                text = f.read()
            # 内容应该等同模板, 解析为空
            self.assertEqual(insights.parse_insights(text), [])
            self.assertEqual(insights.next_insight_id(text), "I-001")


if __name__ == "__main__":
    unittest.main()
