# 3. 骨干块替换对比 & MBConv NPU 根因分析

> 返回 [目录](00_README.md)

---

## 4.1 33decoder 版本下的骨干对比

在 Head 统一为 3×3 后，骨干差异变得显著：

**测试数据来源**: `output_benchmark_all_multiscale_33decoder/benchmark_all_results.csv`

| 骨干 | 192×256 FPS | SRAM (MB) | vs ResNet |
|------|-------------|-----------|-----------|
| ShuffleNet_Bilinear | **12.32** | 1.69 | **+46.5%** |
| MBConv_E4_Bilinear | 9.73 | 1.69 | +15.7% |
| MBConv_E6_Bilinear | 8.46 | 1.69 | +0.6% |
| ResNet_Bilinear | 8.41 | 1.69 | 基线 |
| ResNet_Transpose | 8.89 | 2.06 | +5.7% (SRAM↑) |

## 4.2 骨干块 Cycles 拆解（192×256）

从 per-layer 数据中提取 4 个骨干块的总 cycles：

| 骨干 | 骨干 Cycles | 其他 Cycles | 总 Cycles | 节省 |
|------|-----------|-----------|----------|------|
| ResNet | 17.3M | ~30M | ~47.3M | 基线 |
| MBConv_E4 | 10.9M | ~30M | ~40.9M | -37% |
| **MBConv_E6** | **17.0M** | ~30M | **~47.0M** | **-1.7%** |
| ShuffleNet | 2.2M | ~30M | ~32.2M | -87% |

## 4.3 骨干对比结论

> **ShuffleNet 骨干最优（-87%），MBConv_E4 次之（-37%），MBConv_E6 几乎无提升（-1.7%）。**

---

## 5. MBConv 在 NPU 上慢的根因

### 5.1 MBConv 的结构

```
MBConv (Inverted Residual Block):
  输入 C_in → [1×1 扩展] → C_in × E → [DW 3×3] → C_in × E → [1×1 投影] → C_out
                             ↑ 通道膨胀 E 倍
```

### 5.2 核心问题：通道扩展导致 SRAM 溢出

中间特征图大小 = H × W × C_in × E (INT8)

| 使用位置 | 分辨率 | 通道 | E4 扩展后 | 特征图大小 | vs SRAM |
|---------|-------|------|----------|-----------|---------|
| 骨干 enc_0 | 96×128 | 32ch | 128ch | 1.5 MB | ✅ 放得下 |
| 骨干 enc_1 | 48×64 | 64ch | 256ch | 0.75 MB | ✅ 轻松 |
| **Head1 upsample** | 96×128 | 64ch | 256ch | 3 MB | ⚠️ 分块处理 |
| **Head2 upsample** | **192×256** | 32ch | 128ch | **6 MB** | ❌ **SRAM 3.5倍** |

### 5.3 铁证：OffFlash AC（外部 Flash 访问开销）

OffFlash AC 表示 NPU 因 SRAM 不足而访问外部 Flash 的 cycles 数。

**骨干中的 MBConv（所有层 OffFlash = 0）**：
```
mbconv_enc_0: 1x1 expand  → OffFlash AC = 0     ✅
mbconv_enc_0: DW 3x3      → OffFlash AC = 0     ✅
mbconv_enc_0: 1x1 project → OffFlash AC = 0     ✅
```

**Head2 中的 MBConv_E6（OffFlash 爆炸）**：
```
head_mbconv_e6_head2: 1x1 expand  → OffFlash AC = 75,709,440  ❌ (38.4% of total!)
head_mbconv_e6_head2: DW 3x3      → OffFlash AC = 15,621,120  ❌
head_mbconv_e6_head2: 1x1 project → OffFlash AC = 18,973,952  ❌
Conv15 (7×7 predict)               → OffFlash AC = 42,578,240  ❌ (被挤出 SRAM)
```

### 5.4 DW Conv 的 NPU 利用率天生极低

| 操作类型 | 典型 NPU 利用率 | 原因 |
|---------|----------------|------|
| 标准 Conv 1×1 | 90-105% | MAC 密集，NPU 满载 |
| 标准 Conv 3×3 | 40-99% | 取决于通道数和分辨率 |
| **DW Conv 3×3** | **5-9%** | 每通道仅 9 次 MAC，NPU 大部分时间在搬数据 |

### 5.5 MBConv 的设计初衷 vs NPU 现实

| 维度 | 手机 CPU/GPU（MBConv 设计目标） | Ethos-U55 NPU（实际部署平台） |
|------|-------------------------------|------------------------------|
| 优化目标 | 减少 FLOPs (MAC 数) | 减少内存带宽和搬运 |
| 缓存/SRAM | 数十 MB L2/L3 缓存 | **~1.7 MB SRAM** |
| DW Conv 效率 | 高（cache 友好） | **极低 (~5-9% 利用率)** |
| 通道扩展代价 | 低（内存充裕） | **灾难性（SRAM 溢出 → Flash 访问）** |

### 5.6 MBConv 结论

> **MBConv 通过「扩展-压缩」减少了 MAC 总数，但在 SRAM 极其有限的 NPU 上，通道扩展带来的内存膨胀完全抵消甚至反超了计算量的减少。**
>
> - **骨干中可用**：低分辨率（H/2、H/4）下扩展后的特征图仍可放入 SRAM，净收益为正。
> - **Head 中不可用**：全分辨率下扩展后的特征图远超 SRAM，触发 Flash 访问，速度灾难性下降。

---

## 6. MBConv E6 vs E4：扩展比的影响

### 6.1 为什么 E6 骨干几乎没有比 ResNet 快？

骨干 enc_0（96×128, 32ch）中各子层的 Cycles 对比：

| 子层 | ResNet (2×Conv3×3) | MBConv_E4 (128ch) | MBConv_E6 (192ch) |
|------|-------------------|-------------------|-------------------|
| 3×3 or 1×1 expand | 1.90M + 1.90M | 0.88M | 1.18M |
| DW 3×3 | — | 1.18M (Util 9.4%) | **3.43M (Util 4.8%)** |
| 1×1 project | — | 0.79M | 1.18M |
| Add | 0.59M | 0.59M | 0.59M |
| **小计** | **4.39M** | **3.44M** | **6.38M** |

### 6.2 DW Conv 超线性增长

| 位置 | E4 DW Cycles | E6 DW Cycles | 通道比 | Cycles 比 |
|------|-------------|-------------|--------|----------|
| enc_0 (96×128) | 1,180K | **3,428K** | 1.5× | **2.9×** |
| enc_1 (48×64) | 591K | 886K | 1.5× | 1.5× |
| dec_0 (48×64) | 394K | 591K | 1.5× | 1.5× |
| dec_1 (96×128) | 591K | 886K | 1.5× | 1.5× |

**关键发现**: enc_0 处通道从 128 增到 192（×1.5），但 DW cycles 增长 ×2.9。

原因：192ch × 96 × 128 = 2.25 MB > SRAM 容量，Vela 必须将特征图切成更多更小的 tile，每个 tile 都有额外的搬运开销。SRAM AC 从 252K 增到 1,032K（×4.1），印证了大量数据搬运。

### 6.3 E6 骨干的损益分析

```
ResNet: 2 × Conv3×3 (C, C)
  → MAC 多，但 NPU 利用率高 (80-93%)
  → 骨干总计: 17.3M cycles

MBConv_E4: 1×1(C→4C) + DW(4C) + 1×1(4C→C)
  → 1×1 高效 (90-100%), DW 低效但通道不多
  → 骨干总计: 10.9M cycles ← 净省 6.4M

MBConv_E6: 1×1(C→6C) + DW(6C) + 1×1(6C→C)
  → 1×1 省的量被 DW 超线性增长吃掉
  → 骨干总计: 17.0M cycles ← 净省仅 0.3M
```

### 6.4 扩展比结论

> **E6 的扩展比已经超过了 NPU 的效率临界点。DW Conv 在大通道数时的超线性增长（tiling 开销 + 低利用率），完全抵消了 1×1 卷积替代标准 3×3 卷积所节省的计算量。E4 是该 NPU 上 MBConv 的最佳扩展比。**

