# 搜索环境与空间语义定义 (Search Environment and Space Semantic)

**更新时间**: 2026-03-04
**状态**: Active
**项目**: MCUFlowNet/EdgeFlowNAS

## 1. 核心目标
为了让 Agent Team 能够准确理解搜索空间并制定合理策略，必须将 9 维 `arch_code` (0,1,2 数组) 从纯代码逻辑映射为结构化的自然语言描述。本文件是 Agent A (Strategist) 和 Agent B (Generator) 的系统提示词基座。

---

## 2. 搜索空间与宏观架构语义

EdgeFlowNAS 是一个直接处理图像特征的非 Cost-Volume（无成本计算层）网络。
它本质是一个基于残差块 (Residual Blocks) 与空间多尺度预测的全卷积**Encoder-Decoder 变体**，不包含传统光流网络中的 Warp 操作或 Cost Volume 算子。

整个网络分为两大核心部分，对应搜索空间的 9 个可变维度：

### 2.1 Encoder-Decoder Backbone (特征提取与解码骨干网)
该部分负责从输入中提取视觉特征，并通过上采样重构高分辨率特征。
**深度 (Depth)** 直接决定了特征的表达能力，同时也决定了网络的参数量总量和特征图传递所需的额外内存。
搜索空间中的**前 4 位**控制 Backbone 某些阶段的深度结构：
*   **EB0 (位置 0)**: Encoder Block 0 的深度组合。
*   **EB1 (位置 1)**: Encoder Block 1 的深度组合。
*   **DB0 (位置 2)**: Decoder Backbone 0 的深度组合。
*   **DB1 (位置 3)**: Decoder Backbone 1 的深度组合。

**深度维度的取值 (index 0 ~ 3)**:
*   `0` = **Deep1** (单层 block 结构)
*   `1` = **Deep2** (双层 block 结构)
*   `2` = **Deep3** (三层 block 结构)

### 2.2 Multi-Scale Heads (多尺度预测头)
不同于传统的光流残差累加模式，该部分接收高分辨率的 Decoder 输出特征后，**串行进行多次上采样与卷积**，在不同尺度（1/4，1/2，1/1 全尺寸）分别独立预测输出。
**卷积核感受野大小 (Kernel Size)** 决定了该层捕获位移的能力，同时也深刻影响 NPU 的计算乘加数量 (FLOPs/MACs)。
搜索空间中的**后 5 位**控制多尺度 Heads 各个节点使用的卷积核大小：
*   **H0Out (位置 4)**: 产生 1/4 尺度输出的特征级卷积核大小。
*   **H1 (位置 5)**: 生成给 1/2 尺度的前置特征图时的卷积核大小。
*   **H1Out (位置 6)**: 产生 1/2 尺度输出的特征级卷积核大小。
*   **H2 (位置 7)**: 生成给 1/1 全尺寸的前置特征图时的卷积核大小。
*   **H2Out (位置 8)**: 产生 1/1 全尺度输出的特征级卷积核大小。

**感受野/核大小维度的取值 (index 4 ~ 8)**:
*   `0` = **7x7** 
*   `1` = **5x5** 
*   `2` = **3x3** 

---

## 3. 搜索空间严格声明 (Guardrails)
*   **唯一的调节手段**：仅可调整这 9 位代码的整数值 (`0, 1, 2`)。
*   **禁区**：绝对禁止引入对“通道数(Channels)”、“引入Cost Volume”、“改变上采样插值方法”的分析或指令。

---

## 4. 核心评估指标 (Metrics Dictionary)

Coordinator 驱动底层 Vela 仿真器以及 Supernet 提供以下基准数据供 Agent 分析：

1.  **EPE (Endpoint Error)**
    *   反映网络预测精度的唯一复合指标（越低越好）。
2.  **inferences_per_second (FPS)**
    *   板端硬件的实时帧率（越高越好）。
3.  **sram_memory_used (KB) / sram_total_bytes**
    *   反映网络内存驻留和换页压力的特征（Memory-bound 指标）。
4.  **cycles_npu**
    *   反映纯乘加计算开销的周期（Compute-bound 指标）。

---

## 5. 注入 Agent Prompt 的标准世界观 (System Identity Rule)

以下文本必须直接嵌入 Agent A, B, C, D 的 System Prompt 前序中，统一“世界观”认知：

> "当前任务是对只包含传统卷积的 Encoder-Decoder 网络进行架构搜索。
> 此网络并非基于 Cost Volume，且在 Head 阶段输出相互独立的多尺度特征。
> 你的探索范围是一个严格的 9 维整数数组 (`arch_code`)。其中每一位只能是 `[0,1,2]`。
> 严禁试图修改通道数，严禁更改除此 9 位数组以外的任何机制。
> - 前 4 位 `[0,1,2,3]` 代表网络骨干层的深度（0最浅->2最深）。
> - 后 5 位 `[4,5,6,7,8]` 代表最终几个多尺度预测头所使用的卷积核大小（0=7x7 -> 1=5x5 -> 2=3x3）。
> - 如果遇到因参数或计算量过大导致的 FPS 低下问题，你只能通过削减这 9 个位置的层数或卷积核大小来尝试解决。"
