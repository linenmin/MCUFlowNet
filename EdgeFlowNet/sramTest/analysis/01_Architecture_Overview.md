# 1. 模型架构概览

> 返回 [目录](00_README.md)

EdgeFlowNet 是一个用于光流估计的轻量级 Encoder-Decoder 网络，支持多尺度输出。

```
输入 (H×W×6)
    │
    ├─ Conv Stem (标准 Conv, 6ch → 32ch, stride=2)    ← H/2 × W/2
    │
    ├─ Encoder Block 0 (可替换骨干块, 32ch)            ← H/2 × W/2
    ├─ Stride Conv (32ch → 64ch, stride=2)             ← H/4 × W/4
    ├─ Encoder Block 1 (可替换骨干块, 64ch)            ← H/4 × W/4
    │
    ├─ Stride Conv (降采样)                             ← H/4 × W/4
    │
    ├─ Decoder Block 0 (可替换骨干块, 32ch)            ← H/4 × W/4
    ├─ BilinearResize 2x + Conv                        ← H/2 × W/2
    ├─ Decoder Block 1 (可替换骨干块, 64ch)            ← H/2 × W/2
    ├─ BilinearResize 2x + Conv                        ← H × W
    │
    ├─ Head0: Conv predict (7×7) → 输出0               ← H/4 × W/4
    ├─ Head1: Upsample + Conv predict (7×7) → 输出1    ← H/2 × W/2
    ├─ Head2: Upsample + Conv predict (7×7) → 输出2    ← H × W
    │
    └─ AccumPreds: Resize + Add 多尺度累加 → 最终输出   ← H × W
```

**可替换骨干块选项**: ResBlock / MBConv / ShuffleNetV2

**可替换 Head 上采样选项**: Conv3×3 / Conv5×5 / Conv7×7 / DSConv3×3 / MBConv / ShuffleNet

