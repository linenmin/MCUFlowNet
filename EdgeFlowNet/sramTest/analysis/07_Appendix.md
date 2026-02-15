# 附录: 关键数据来源

> 返回 [目录](00_README.md)

---

| 数据集 | 目录 | 说明 |
|--------|------|------|
| 原始多尺度 | `output_benchmark_all_multiscale/` | 原始 7×7/5×5 Head，5 种骨干 |
| 3×3 decoder | `output_benchmark_all_multiscale_33decoder/` | 全 3×3 Head，5 种骨干 |
| Head 上采样搜索 | `output_benchmark_upsample/` | 7 种 Head Choice，ShuffleNet 骨干 |
| 网络定义 | `network/MultiScaleResNet_cell.py` | 原始网络 |
| 网络定义 | `network/MultiScaleResNet_cell_33decoder.py` | 3×3 decoder 版本 |
| 网络定义 | `network/MultiScaleResNet_cell_upsample_search.py` | Head NAS 搜索版本 |
| Benchmark 脚本 | `auto_benchmark_all.py` | 骨干 + Head 综合测试 |
| Benchmark 脚本 | `auto_benchmark_upsample.py` | Head 上采样 NAS 测试 |

