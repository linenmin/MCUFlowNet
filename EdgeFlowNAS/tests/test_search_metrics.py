"""Phase 1.3 (search_hybrid_v1): search_metrics module 测试。

覆盖 8 大类指标：
- hypervolume_2d (含闭式验证)
- shannon_entropy (空 / 单值 / 均匀)
- per_dim_gene_entropy (11 维)
- mean_crowding_distance_excluding_inf (≤2 点 / 多点)
- largest_pareto_gap (空 / 单点 / 多点)
- stagnation_count (decrease / increase / 边界)
- compute_full_generation_metrics 端到端
"""

import math
import unittest

import pandas as pd

from efnas.search import search_metrics as sm


class TestHypervolume2D(unittest.TestCase):
    def test_empty_front_returns_zero(self) -> None:
        self.assertEqual(sm.hypervolume_2d([]), 0.0)

    def test_single_point_closed_form(self) -> None:
        # 单点 (3.99, 5.0)，ref (5.5, 3.0)
        # HV = (5.5 - 3.99) * (5.0 - 3.0) = 1.51 * 2.0 = 3.02
        hv = sm.hypervolume_2d([(3.99, 5.0)], ref_epe=5.5, ref_fps=3.0)
        self.assertAlmostEqual(hv, 3.02, places=4)

    def test_three_point_staircase_closed_form(self) -> None:
        # 来自 search_metrics.py 注释里的手算示例
        front = [(3.99, 5.0), (4.5, 7.0), (5.0, 9.0)]
        # strip 1: (5.5-3.99)*(5.0-3.0) = 3.02
        # strip 2: (5.5-4.5)*(7.0-5.0) = 2.00
        # strip 3: (5.5-5.0)*(9.0-7.0) = 1.00
        # total = 6.02
        hv = sm.hypervolume_2d(front, ref_epe=5.5, ref_fps=3.0)
        self.assertAlmostEqual(hv, 6.02, places=4)

    def test_points_outside_reference_filtered(self) -> None:
        # epe ≥ ref_epe 或 fps ≤ ref_fps 的点不计入
        front = [(5.5, 8.0), (3.0, 3.0), (4.0, 5.0)]
        # 只 (4.0, 5.0) 有效；HV = (5.5-4.0)*(5.0-3.0) = 3.0
        hv = sm.hypervolume_2d(front, ref_epe=5.5, ref_fps=3.0)
        self.assertAlmostEqual(hv, 3.0, places=4)

    def test_unsorted_input_handled(self) -> None:
        front_a = [(3.99, 5.0), (5.0, 9.0), (4.5, 7.0)]
        front_b = [(3.99, 5.0), (4.5, 7.0), (5.0, 9.0)]
        self.assertAlmostEqual(
            sm.hypervolume_2d(front_a, 5.5, 3.0),
            sm.hypervolume_2d(front_b, 5.5, 3.0),
            places=6,
        )


class TestShannonEntropy(unittest.TestCase):
    def test_empty_returns_zero(self) -> None:
        self.assertEqual(sm.shannon_entropy([]), 0.0)

    def test_single_value_returns_zero(self) -> None:
        self.assertEqual(sm.shannon_entropy(["a", "a", "a"]), 0.0)

    def test_uniform_two_choices_equals_log2(self) -> None:
        h = sm.shannon_entropy(["a", "b"] * 50)
        self.assertAlmostEqual(h, math.log(2), places=6)

    def test_uniform_three_choices_equals_log3(self) -> None:
        h = sm.shannon_entropy(["a", "b", "c"] * 10)
        self.assertAlmostEqual(h, math.log(3), places=6)


class TestPerDimGeneEntropy(unittest.TestCase):
    def test_empty_returns_all_zeros(self) -> None:
        self.assertEqual(sm.per_dim_gene_entropy([]), [0.0] * 11)

    def test_uniform_population_high_entropy(self) -> None:
        # 50 个个体，dim 0 在 {0, 1, 2} 上均匀
        codes = [
            f"{i % 3},0,0,0,0,0,0,0,0,0,0" for i in range(48)
        ]
        ents = sm.per_dim_gene_entropy(codes)
        self.assertEqual(len(ents), 11)
        # dim 0 接近 log(3)
        self.assertAlmostEqual(ents[0], math.log(3), places=2)
        # 其他维都是 0（全 0 单值）
        for d in range(1, 11):
            self.assertEqual(ents[d], 0.0)

    def test_skips_malformed_codes(self) -> None:
        codes = ["0,1,2,0,0,0,0,0,0,0,0", "TOO,SHORT", "1,1,1,1,1,1,1,1,1,1,1"]
        ents = sm.per_dim_gene_entropy(codes)
        # 只有 2 个有效，dim 0 在 {0, 1} 上均匀
        self.assertAlmostEqual(ents[0], math.log(2), places=6)


class TestMeanCrowdingDistance(unittest.TestCase):
    def test_zero_when_two_or_fewer_points(self) -> None:
        self.assertEqual(sm.mean_crowding_distance_excluding_inf([]), 0.0)
        self.assertEqual(sm.mean_crowding_distance_excluding_inf([(1.0, 2.0)]), 0.0)
        self.assertEqual(
            sm.mean_crowding_distance_excluding_inf([(1.0, 2.0), (2.0, 3.0)]),
            0.0,
        )

    def test_three_collinear_points_returns_finite_distance(self) -> None:
        # 中间点 (2, 4) 有 finite crowding distance
        front = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]
        cd = sm.mean_crowding_distance_excluding_inf(front)
        # epe range = [1, 3], 中间点 epe=2，间距 = (3-1)/(3-1) = 1
        # fps range = [2, 6]，注意 fps 取负：obj = -fps，中间 -4
        # -fps range = [-6, -2]，中间点 -4，间距 = (-2 - (-6))/(-2-(-6)) = 1
        # 总 crowding = 1 + 1 = 2
        self.assertAlmostEqual(cd, 2.0, places=6)


class TestLargestParetoGap(unittest.TestCase):
    def test_empty_returns_empty_strings(self) -> None:
        gap = sm.largest_pareto_gap([])
        self.assertEqual(gap["fps_low"], "")

    def test_single_point_returns_empty_strings(self) -> None:
        gap = sm.largest_pareto_gap([(4.0, 5.0)])
        self.assertEqual(gap["fps_high"], "")

    def test_finds_largest_gap_among_three_points(self) -> None:
        # FPS 间距: 5.0->5.5 (0.5), 5.5->8.0 (2.5)，最大 2.5
        front = [(4.0, 5.0), (4.2, 5.5), (4.8, 8.0)]
        gap = sm.largest_pareto_gap(front)
        self.assertAlmostEqual(gap["fps_low"], 5.5, places=6)
        self.assertAlmostEqual(gap["fps_high"], 8.0, places=6)
        self.assertAlmostEqual(gap["epe_low"], 4.2, places=6)
        self.assertAlmostEqual(gap["epe_high"], 4.8, places=6)
        self.assertAlmostEqual(gap["fps_span"], 2.5, places=6)


class TestStagnationCount(unittest.TestCase):
    def test_empty_returns_zero(self) -> None:
        self.assertEqual(sm.stagnation_count([]), 0)

    def test_single_value_returns_zero(self) -> None:
        self.assertEqual(sm.stagnation_count([4.0]), 0)

    def test_decrease_strict_improvement_resets(self) -> None:
        # values = [4.0, 4.0, 3.99]; 最后一个改进 → stagnation = 0
        self.assertEqual(sm.stagnation_count([4.0, 4.0, 3.99]), 0)

    def test_decrease_no_improvement_counts_gap(self) -> None:
        # values = [4.0, 3.99, 3.99, 3.99]; 上次改进在 idx=1，当前 idx=3 → 2
        self.assertEqual(sm.stagnation_count([4.0, 3.99, 3.99, 3.99]), 2)

    def test_decrease_first_value_only(self) -> None:
        # values = [4.0]; 长度 1 → 0
        self.assertEqual(sm.stagnation_count([4.0], direction="decrease"), 0)

    def test_increase_direction(self) -> None:
        # FPS: 增大算改进；values = [5.0, 5.5, 5.5, 5.5]; idx=1 改进，当前 idx=3 → 2
        self.assertEqual(
            sm.stagnation_count([5.0, 5.5, 5.5, 5.5], direction="increase"),
            2,
        )

    def test_invalid_direction_raises(self) -> None:
        with self.assertRaises(ValueError):
            sm.stagnation_count([1.0, 2.0], direction="bogus")


class TestComputeFullGenerationMetrics(unittest.TestCase):
    def test_first_generation_no_history(self) -> None:
        history = pd.DataFrame({
            "arch_code": [
                "0,0,0,0,0,0,0,0,0,0,0",
                "2,2,2,2,2,2,1,1,1,1,1",
                "1,1,1,1,1,1,0,0,0,0,0",
            ],
            "epe": [4.99, 4.01, 4.5],
            "fps": [8.91, 5.0, 6.5],
        })
        pop_codes = [
            "0,0,0,0,0,0,0,0,0,0,0",
            "2,2,2,2,2,2,1,1,1,1,1",
            "1,1,1,1,1,1,0,0,0,0,0",
        ]
        metrics = sm.compute_full_generation_metrics(
            history_df=history,
            current_population_arch_codes=pop_codes,
            metrics_history_df=None,
            epoch=0,
            new_evaluated=3,
            duplicates=0,
            population_size=3,
            search_space_size=23328,
        )
        # Pareto 前沿: 三个点全部非支配（确认）
        self.assertEqual(metrics["pareto_count"], 3)
        # best EPE 是 4.01，best FPS 是 8.91
        self.assertAlmostEqual(metrics["best_epe"], 4.01, places=4)
        self.assertAlmostEqual(metrics["best_fps"], 8.91, places=4)
        # HV > 0
        self.assertGreater(metrics["hv"], 0.0)
        # 首代 stagnation 全是 0
        self.assertEqual(metrics["stagnation_best_epe"], 0)
        self.assertEqual(metrics["stagnation_best_fps"], 0)
        self.assertEqual(metrics["stagnation_hv"], 0)
        # HV 改进率：首代没有上一代 → 空字符串
        self.assertEqual(metrics["hv_improvement_rate_3gen"], "")
        # 列数对齐 schema
        self.assertEqual(set(metrics.keys()), set(sm.GENERATION_METRICS_COLUMNS))

    def test_stagnation_picks_up_when_hv_does_not_improve(self) -> None:
        history = pd.DataFrame({
            "arch_code": ["1,1,1,1,1,1,0,0,0,0,0"] * 2,
            "epe": [4.5, 4.5],
            "fps": [6.5, 6.5],
        })
        # 当前 HV = (5.5-4.5)*(6.5-3.0) = 3.5；为了构造"未改进"，过去几代 HV 需要
        # 等于或大于 3.5
        metrics_history = pd.DataFrame({
            "epoch": [0, 1, 2],
            "best_epe": [4.5, 4.5, 4.5],
            "best_fps": [6.5, 6.5, 6.5],
            "hv": [3.5, 3.5, 3.5],
            "duplicate_rate": [0.0, 0.0, 0.0],
        })
        metrics = sm.compute_full_generation_metrics(
            history_df=history,
            current_population_arch_codes=["1,1,1,1,1,1,0,0,0,0,0"] * 2,
            metrics_history_df=metrics_history,
            epoch=3,
            new_evaluated=0,
            duplicates=0,
            population_size=2,
            search_space_size=23328,
        )
        # 当前 HV 应该等于历史 HV (单点不变)，stagnation_hv >= 1
        self.assertGreaterEqual(metrics["stagnation_hv"], 1)
        self.assertGreaterEqual(metrics["stagnation_best_epe"], 1)
        self.assertGreaterEqual(metrics["stagnation_best_fps"], 1)

    def test_hv_improvement_rate_with_three_prior_generations(self) -> None:
        history = pd.DataFrame({
            "arch_code": ["1,1,1,1,1,1,0,0,0,0,0"],
            "epe": [4.0],
            "fps": [7.0],
        })
        metrics_history = pd.DataFrame({
            "epoch": [0, 1, 2],
            "best_epe": [4.5, 4.3, 4.1],
            "best_fps": [6.0, 6.5, 6.8],
            "hv": [1.0, 2.0, 3.0],
            "duplicate_rate": [0.0, 0.0, 0.0],
        })
        metrics = sm.compute_full_generation_metrics(
            history_df=history,
            current_population_arch_codes=["1,1,1,1,1,1,0,0,0,0,0"],
            metrics_history_df=metrics_history,
            epoch=3,
            new_evaluated=1,
            duplicates=0,
            population_size=1,
            search_space_size=23328,
        )
        # 序列含当前: [1.0, 2.0, 3.0, current_hv]
        # 预期 (current_hv - 1.0) / 3
        expected = (metrics["hv"] - 1.0) / 3.0
        self.assertAlmostEqual(
            metrics["hv_improvement_rate_3gen"], round(expected, 6), places=4,
        )

    def test_duplicate_rate_3gen_avg_uses_recent_window(self) -> None:
        history = pd.DataFrame({
            "arch_code": ["1,1,1,1,1,1,0,0,0,0,0"],
            "epe": [4.0],
            "fps": [7.0],
        })
        metrics_history = pd.DataFrame({
            "epoch": [0, 1],
            "duplicate_rate": [0.1, 0.2],
        })
        metrics = sm.compute_full_generation_metrics(
            history_df=history,
            current_population_arch_codes=[],
            metrics_history_df=metrics_history,
            epoch=2,
            new_evaluated=0,
            duplicates=3,
            population_size=10,
            search_space_size=23328,
        )
        # duplicate_rate = 0.3，window 应取 [0.1, 0.2, 0.3]，均值 = 0.2
        self.assertAlmostEqual(metrics["duplicate_rate_3gen_avg"], 0.2, places=6)

    def test_rank1_saturation_uses_population(self) -> None:
        # 当前种群 5 个个体，全部 Pareto-equivalent (各自一个目标极端) → saturation = 1.0
        history = pd.DataFrame({
            "arch_code": [
                "0,0,0,0,0,0,0,0,0,0,0",
                "2,2,2,2,2,2,1,1,1,1,1",
                "1,1,1,1,1,1,0,0,0,0,0",
            ],
            "epe": [5.0, 4.0, 4.5],
            "fps": [9.0, 5.0, 6.5],
        })
        metrics = sm.compute_full_generation_metrics(
            history_df=history,
            current_population_arch_codes=[
                "0,0,0,0,0,0,0,0,0,0,0",
                "2,2,2,2,2,2,1,1,1,1,1",
                "1,1,1,1,1,1,0,0,0,0,0",
            ],
            metrics_history_df=None,
            epoch=0,
            new_evaluated=3,
            duplicates=0,
            population_size=3,
            search_space_size=23328,
        )
        self.assertAlmostEqual(metrics["rank1_saturation"], 1.0, places=4)

    def test_columns_match_schema_exactly(self) -> None:
        # 所有 metric dict key 必须和 GENERATION_METRICS_COLUMNS 完全一致
        history = pd.DataFrame({
            "arch_code": ["0,0,0,0,0,0,0,0,0,0,0"],
            "epe": [4.5],
            "fps": [6.0],
        })
        metrics = sm.compute_full_generation_metrics(
            history_df=history,
            current_population_arch_codes=["0,0,0,0,0,0,0,0,0,0,0"],
            metrics_history_df=None,
            epoch=0,
            new_evaluated=1,
            duplicates=0,
            population_size=1,
            search_space_size=23328,
        )
        self.assertEqual(set(metrics.keys()), set(sm.GENERATION_METRICS_COLUMNS))
        # 列总数: 7 基础 + 4 Pareto + 4 HV + 1 crowding + 11 entropy + 3 stagnation
        # + 4 gap + 2 legacy + 1 coverage = 37
        self.assertEqual(len(sm.GENERATION_METRICS_COLUMNS), 37)


if __name__ == "__main__":
    unittest.main()
