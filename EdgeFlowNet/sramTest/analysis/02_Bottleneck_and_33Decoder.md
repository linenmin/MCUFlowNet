# 2. 原始模型瓶颈分析 & 3×3 Decoder 优化

> 返回 [目录](00_README.md)

---

## 2.1 原始配置（7×7/5×5 Head 卷积 + ResNet 骨干）

**测试数据来源**: `output_benchmark_all_multiscale/`

在原始网络 (`MultiScaleResNet_cell.py`) 中，多尺度输出头使用了大核卷积：

| Head | Upsample 卷积 | Predict 卷积 | 工作分辨率 |
|------|---------------|-------------|-----------|
| Head0 | — | 7×7 | H/4 × W/4 |
| Head1 | 5×5 | 7×7 | H/2 × W/2 |
| Head2 | 7×7 | 7×7 | H × W (全分辨率) |

## 2.2 瓶颈定位

以 ResNet_Bilinear @ 192×256 per-layer 数据为例：

| 层名 | Op Cycles | 占比 | MAC Count | 说明 |
|------|----------|------|-----------|------|
| Conv46 (Head1 predict 7×7) | 5,420,423 | 13.1% | 154M | 全分辨率 7×7 |
| Conv50 (Head2 predict 7×7) | 5,420,423 | 13.1% | 154M | 全分辨率 7×7 |
| Head2 upsample 7×7 | 3,596,766 | 8.7% | 226M | 全分辨率 7×7 |
| Head1 upsample 5×5 | 3,596,766 | 8.7% | 226M | 半分辨率 5×5 |
| **Head 总计** | **~18M** | **~44%** | | **近一半的开销** |

## 2.3 关键发现

- **Head 占整个网络约 44% 的 cycles**，是最大瓶颈
- **SRAM 峰值不由 Head 决定**，而是由 AccumPreds 中的全分辨率 Resize+Add 决定
- 骨干块仅占 ~17M cycles，替换骨干的效果被 Head 瓶颈掩盖

---

## 3.1 改动内容

将所有 Head 卷积从大核改为 3×3 (`MultiScaleResNet_cell_33decoder.py`)：

| 位置 | 原始 | 3×3 Decoder |
|------|------|-------------|
| Head0 predict | 7×7 | 3×3 |
| Head1 upsample | 5×5 | 3×3 |
| Head1 predict | 7×7 | 3×3 |
| Head2 upsample | 7×7 | 3×3 |
| Head2 predict | 7×7 | 3×3 |

## 3.2 FPS 提升

**测试数据来源**: `output_benchmark_all_multiscale_33decoder/`

| 配置 (192×256) | 原始 FPS | 3×3 Decoder FPS | 提升 |
|---------------|---------|-----------------|------|
| ResNet_Bilinear | 5.18 | 8.41 | **+62%** |
| ShuffleNet_Bilinear | 7.76 | 12.32 | **+59%** |
| MBConv_E4_Bilinear | 5.78 | 9.73 | **+68%** |

## 3.3 SRAM 不变

SRAM 峰值完全不受影响（由 AccumPreds 决定），3×3 decoder 的 SRAM 与原始版本完全相同。

## 3.4 结论

> **Head 大核卷积是原始模型的最大性能瓶颈。改为 3×3 后 FPS 提升 60-68%，且 SRAM 不变。**

