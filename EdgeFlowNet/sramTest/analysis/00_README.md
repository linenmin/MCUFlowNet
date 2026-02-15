# EdgeFlowNet NPU Benchmark 分析报告

> **平台**: Arm Ethos-U55 NPU (128 MAC, ~1.7 MB SRAM)
> **编译器**: Vela TFLite Compiler
> **量化**: INT8 全量化
> **日期**: 2026-02-15

---

## 目录

| 文件 | 内容 | 关键结论 |
|------|------|---------|
| [`01_Architecture_Overview.md`](01_Architecture_Overview.md) | 模型架构概览 | Encoder-Decoder + 多尺度 Head 结构 |
| [`02_Bottleneck_and_33Decoder.md`](02_Bottleneck_and_33Decoder.md) | 原始瓶颈分析 + 3×3 Decoder 优化 | Head 大核卷积占 44% cycles，改 3×3 后 FPS +60-68% |
| [`03_Backbone_Block_Comparison.md`](03_Backbone_Block_Comparison.md) | 骨干块替换对比 + MBConv NPU 根因分析 | ShuffleNet -87%，MBConv E6≈0%；DW Conv NPU 利用率仅 5-9% |
| [`04_Head_Upsample_NAS.md`](04_Head_Upsample_NAS.md) | Head 上采样 NAS 搜索结果 + 推荐 | Conv3×3/5×5/DSConv3×3 三选项；排除 MBConv 和 Shuffle |
| [`05_Block_Type_NAS_Brainstorm.md`](05_Block_Type_NAS_Brainstorm.md) | NAS 骨干块搜索策略（头脑风暴） | FusedMBConv + 非对称搜索（**已被第 6 篇修订**） |
| [`06_Final_NAS_Strategy.md`](06_Final_NAS_Strategy.md) | **纯标准卷积 NAS 最终方案** ⭐ | 仅 ResBlock，搜索每位置深度（36 骨干 × 162 Head = 5,832 种组合） |
| [`07_Appendix.md`](07_Appendix.md) | 附录：关键数据来源 | 文件路径索引 |

---

## 分析演进路径

```
原始瓶颈定位 (Ch2)
    │
    ├─ 3×3 Decoder 优化 (Ch3) → FPS +60-68%
    │
    ├─ 骨干块对比 (Ch4) → ShuffleNet 最快
    │   ├─ MBConv 慢的根因 (Ch5) → SRAM 溢出 + DW 低效
    │   └─ E6 vs E4 (Ch6) → 扩展比超临界点
    │
    ├─ Head NAS 搜索 (Ch7-8) → Conv3×3/5×5 + DSConv3×3
    │
    ├─ Block Type NAS 头脑风暴 (Ch9) → FusedMBConv 方案
    │
    └─ 最终修订 (Ch10) ⭐ → 放弃 DW 块，纯标准 Conv + 宽度/深度搜索
```

