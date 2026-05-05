"""Phase 1.2 (search_hybrid_v1): Vela 层级数据 Python parser 测试。

覆盖：
- _extract_block_tag 对各种 raw_name 形式的标签推断
- parse_vela_layer_profile 对合成 CSV 的解析
- parse_vela_layer_profile 对真实样本文件的兼容性 (smoke test)
- find_per_layer_csv 在两种目录布局下的定位
- write_layer_profile + read 回环
- parse_and_persist_layer_profile 端到端
- query_vela_for_arch 优先读 JSON, 兜底回 CSV
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from efnas.search import vela_parser


# 合成 CSV 内容（最小可解析版本，列数 = 15，3 行数据覆盖典型场景）
_SYNTHETIC_PER_LAYER_CSV = (
    "TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,"
    "NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name\n"
    "CONV_2D,Conv2DBias,615968,37.13,2292566.0,4.74,2292566.0,529200.0,0.0,0.0,0.0,"
    "101606400,4.88,69.25,"
    "supernet_backbone/conv_bn_relu0/relu3/E0_relu;supernet_backbone/conv_bn_relu0/conv1/E0_conv/Conv2D\n"
    "CONV_2D,Conv2DBias,158688,9.56,4315240.0,8.93,4315240.0,202368.0,0.0,0.0,0.0,"
    "106168320,5.10,38.44,"
    "supernet_backbone/DB0/branch1_block1/conv_bn_relu98/conv99/conv1_conv/Conv2D\n"
    "RESIZE_BILINEAR,ResizeBilinear,1658880,100.0,185088.0,0.38,185088.0,122400.0,0.0,0.0,0.0,"
    "737280,0.035,6.22,AccumResize2\n"
)


class TestExtractBlockTag(unittest.TestCase):
    def test_eb0_path_returns_eb0(self) -> None:
        name = "supernet_backbone/EB0/branch1_block1/conv_bn_relu8/relu11/conv1_relu"
        self.assertEqual(vela_parser._extract_block_tag(name), "EB0")

    def test_h1out_takes_priority_over_h1(self) -> None:
        # H1Out 的子串里也含 H1，但模式表里 /H1Out 在 /H1/ 之前
        name = "supernet_head/H1Out/conv203/k3/Conv2D1"
        self.assertEqual(vela_parser._extract_block_tag(name), "H1Out")

    def test_h1_alone_returns_h1(self) -> None:
        name = "supernet_head/H1/resize_conv197/k3_resize"
        self.assertEqual(vela_parser._extract_block_tag(name), "H1")

    def test_e0_via_fused_marker(self) -> None:
        name = "supernet_backbone/conv_bn_relu0/relu3/E0_relu;other/E0_conv"
        self.assertEqual(vela_parser._extract_block_tag(name), "E0")

    def test_accum_resize_keeps_precise_tag(self) -> None:
        self.assertEqual(vela_parser._extract_block_tag("AccumResize2"), "AccumResize2")
        self.assertEqual(vela_parser._extract_block_tag("AccumAdd1"), "AccumAdd1")

    def test_strided_slice_tail(self) -> None:
        self.assertEqual(
            vela_parser._extract_block_tag("strided_slice_avgpool"),
            "StridedSliceTail",
        )

    def test_unknown_returns_other(self) -> None:
        self.assertEqual(vela_parser._extract_block_tag("foo/bar/baz"), "Other")

    def test_empty_returns_unknown(self) -> None:
        self.assertEqual(vela_parser._extract_block_tag(""), "Unknown")
        self.assertEqual(vela_parser._extract_block_tag("   "), "Unknown")


class TestParseVelaLayerProfile(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmpdir.name, "arch_0000_per-layer.csv")
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(_SYNTHETIC_PER_LAYER_CSV)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_parses_three_rows_with_correct_block_tags(self) -> None:
        rows = vela_parser.parse_vela_layer_profile(self.csv_path)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["block_tag"], "E0")
        self.assertEqual(rows[1]["block_tag"], "DB0")
        self.assertEqual(rows[2]["block_tag"], "AccumResize2")

    def test_numeric_fields_have_correct_types(self) -> None:
        rows = vela_parser.parse_vela_layer_profile(self.csv_path)
        first = rows[0]
        self.assertIsInstance(first["sram_bytes"], int)
        self.assertIsInstance(first["cycles"], int)
        self.assertIsInstance(first["macs"], int)
        self.assertIsInstance(first["peak_pct"], float)
        self.assertIsInstance(first["util_pct"], float)
        self.assertEqual(first["sram_bytes"], 615968)
        self.assertEqual(first["cycles"], 2292566)
        self.assertEqual(first["macs"], 101606400)
        self.assertAlmostEqual(first["util_pct"], 69.25, places=2)

    def test_returns_empty_for_missing_file(self) -> None:
        rows = vela_parser.parse_vela_layer_profile("/nonexistent/path.csv")
        self.assertEqual(rows, [])

    def test_returns_empty_for_empty_path(self) -> None:
        self.assertEqual(vela_parser.parse_vela_layer_profile(""), [])

    def test_handles_csv_with_too_few_columns_gracefully(self) -> None:
        bad_path = os.path.join(self.tmpdir.name, "bad.csv")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("col1,col2\n1,2\n3,4\n")
        self.assertEqual(vela_parser.parse_vela_layer_profile(bad_path), [])

    def test_skips_malformed_data_row_but_keeps_valid_rows(self) -> None:
        path = os.path.join(self.tmpdir.name, "mixed.csv")
        # 第一行数据列数不够，第二行正常；parser 应跳过坏行保留好行
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(
                "TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,"
                "NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name\n"
                "BAD,ROW\n"
                "CONV_2D,Conv2DBias,100,5.0,1000,1.0,1000,0,0,0,0,500,1.0,80.0,"
                "supernet_backbone/EB1/conv\n"
            )
        rows = vela_parser.parse_vela_layer_profile(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["block_tag"], "EB1")


class TestFindPerLayerCsv(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.run_dir = self.tmpdir.name

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_finds_csv_in_vela_tmp_layout(self) -> None:
        # 模拟标准 Vela 输出: analysis/vela_tmp/arch_0000/arch_0000_per-layer.csv
        target_dir = os.path.join(self.run_dir, "analysis", "vela_tmp", "arch_0000")
        os.makedirs(target_dir, exist_ok=True)
        target_csv = os.path.join(target_dir, "arch_0000_per-layer.csv")
        Path(target_csv).touch()

        found = vela_parser.find_per_layer_csv(self.run_dir)
        self.assertEqual(os.path.normcase(found), os.path.normcase(target_csv))

    def test_finds_csv_in_flat_analysis_layout(self) -> None:
        # 兜底布局: analysis/per-layer.csv
        analysis_dir = os.path.join(self.run_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        target_csv = os.path.join(analysis_dir, "per-layer.csv")
        Path(target_csv).touch()

        found = vela_parser.find_per_layer_csv(self.run_dir)
        self.assertEqual(os.path.normcase(found), os.path.normcase(target_csv))

    def test_returns_none_when_no_match(self) -> None:
        os.makedirs(os.path.join(self.run_dir, "analysis"), exist_ok=True)
        self.assertIsNone(vela_parser.find_per_layer_csv(self.run_dir))

    def test_returns_none_for_missing_dir(self) -> None:
        self.assertIsNone(vela_parser.find_per_layer_csv("/nonexistent"))


class TestWriteLayerProfile(unittest.TestCase):
    def test_round_trip_via_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = [
                {"block_tag": "EB0", "cycles": 1000, "util_pct": 80.0, "raw_name": "x"},
                {"block_tag": "DB0", "cycles": 2000, "util_pct": 33.0, "raw_name": "y"},
            ]
            path = vela_parser.write_layer_profile(tmp, profile)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.path.basename(path), "layer_profile.json")
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, profile)


class TestParseAndPersistEndToEnd(unittest.TestCase):
    def test_end_to_end_writes_layer_profile_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run_00000000000")
            csv_dir = os.path.join(run_dir, "analysis", "vela_tmp", "arch_0000")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, "arch_0000_per-layer.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(_SYNTHETIC_PER_LAYER_CSV)

            json_path = vela_parser.parse_and_persist_layer_profile(run_dir)
            self.assertIsNotNone(json_path)
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            self.assertEqual(len(profile), 3)
            self.assertEqual({row["block_tag"] for row in profile},
                             {"E0", "DB0", "AccumResize2"})

    def test_returns_none_when_csv_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run_empty")
            os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
            self.assertIsNone(vela_parser.parse_and_persist_layer_profile(run_dir))


class TestQueryVelaForArch(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.exp_dir = self.tmpdir.name
        self.arch_code = "0,1,2,0,0,1,2,1,0,1,0"
        self.safe_name = self.arch_code.replace(",", "")
        self.run_dir = os.path.join(
            self.exp_dir, "dashboard", "eval_outputs", f"run_{self.safe_name}",
        )
        os.makedirs(os.path.join(self.run_dir, "analysis"), exist_ok=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_reads_layer_profile_json_when_present(self) -> None:
        profile = [{"block_tag": "EB0", "cycles": 100, "util_pct": 90.0,
                    "raw_name": "x"}]
        with open(os.path.join(self.run_dir, "analysis", "layer_profile.json"),
                  "w", encoding="utf-8") as f:
            json.dump(profile, f)
        result = vela_parser.query_vela_for_arch(self.exp_dir, self.arch_code)
        self.assertEqual(result, profile)

    def test_falls_back_to_per_layer_csv_when_json_absent(self) -> None:
        csv_dir = os.path.join(self.run_dir, "analysis", "vela_tmp", "arch_0000")
        os.makedirs(csv_dir, exist_ok=True)
        with open(os.path.join(csv_dir, "arch_0000_per-layer.csv"), "w",
                  encoding="utf-8", newline="") as f:
            f.write(_SYNTHETIC_PER_LAYER_CSV)
        result = vela_parser.query_vela_for_arch(self.exp_dir, self.arch_code)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    def test_returns_none_when_neither_json_nor_csv(self) -> None:
        self.assertIsNone(
            vela_parser.query_vela_for_arch(self.exp_dir, self.arch_code)
        )

    def test_returns_none_for_empty_arch_code(self) -> None:
        self.assertIsNone(vela_parser.query_vela_for_arch(self.exp_dir, ""))


class TestRealVelaSmokeIfAvailable(unittest.TestCase):
    """Smoke test against a real Vela per-layer CSV in outputs/ if present.

    使用 search_v1_buggy_backup 里的现成样本（如果 repo 里仍保留）。
    本机找不到时跳过，不影响 CI。
    """

    SAMPLE_CSV = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "search_v1_20260305_141936_buggy_backup",
        "dashboard", "eval_outputs", "run_000000000",
        "analysis", "vela_tmp", "arch_0000", "arch_0000_per-layer.csv",
    )

    def test_real_csv_parses_and_includes_known_blocks(self) -> None:
        if not os.path.exists(self.SAMPLE_CSV):
            self.skipTest(f"Real Vela sample missing: {self.SAMPLE_CSV}")
        rows = vela_parser.parse_vela_layer_profile(self.SAMPLE_CSV)
        self.assertGreater(len(rows), 10)
        tags = {row["block_tag"] for row in rows}
        # 已知的 V1 9D 子网应该至少触及这几个块
        for expected_tag in ("EB0", "DB0", "H1Out", "AccumResize2"):
            self.assertIn(expected_tag, tags,
                          f"block_tag={expected_tag} 不在解析结果里: {sorted(tags)}")

    def test_real_csv_util_pct_is_in_zero_to_hundred_range(self) -> None:
        if not os.path.exists(self.SAMPLE_CSV):
            self.skipTest(f"Real Vela sample missing: {self.SAMPLE_CSV}")
        rows = vela_parser.parse_vela_layer_profile(self.SAMPLE_CSV)
        for row in rows:
            self.assertGreaterEqual(row["util_pct"], 0.0)
            self.assertLessEqual(row["util_pct"], 100.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
