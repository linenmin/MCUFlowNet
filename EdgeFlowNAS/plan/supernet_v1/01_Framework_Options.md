# EdgeFlowNAS 大框架方案（归档精简版）
更新时间: 2026-03-03
状态: Superseded（执行以 10/11/14 为准）

## 1. 已落地主决策
1. 训练路线采用 `Strict Fairness One-Shot Supernet`。
2. 代码组织采用 `EdgeFlowNAS` 独立可运行副本，不依赖跨项目运行。
3. 搜索与最终子网重训从超网训练文档中拆分，不与超网主流程混写。

---

## 2. 保留背景
1. 搜索空间按 9 维 `arch_code` 设计（4 个深度位 + 5 个 head kernel 位）。
2. 该文档原先的 A/B/C 方案讨论仅用于第一次框架定向，当前不再作为执行入口。

---

## 3. 后续读取入口
1. 训练规格: `02_Supernet_Training_Spec.md`
2. 任务拆分: `03_Supernet_Task_Breakdown.md`
3. 执行门禁: `04_Supernet_Execution_Gates.md`
4. 工程规范: `05_Engineering_File_Management_and_Code_Style.md`
5. 当前执行主线: `10_Eval_Consistency_and_Clip_Diagnostics_Plan.md`、`11_200Epoch_结果诊断与续训决策计划.md`

---

## 4. 归档细节（保留首轮选型依据）

### 4.1 已对齐的技术前提（首轮）
1. 骨架基于 Bilinear 版本，目标是保持 SRAM 峰值稳定且低于转置卷积骨架。
2. 首轮搜索空间定义为：
- Backbone Choice Block: `EB0, EB1, DB0, DB1`
- Backbone 选项: `Deep1, Deep2, Deep3`
- Head Choice Block: `H0Out, H1, H1Out, H2, H2Out`
- Head 选项: `7x7Conv, 5x5Conv, 3x3Conv`
3. 首轮估算搜索空间总数：
- Backbone: `3^4 = 81`
- Head: `3^5 = 243`
- Total: `81 * 243 = 19,683`
4. 当时希望训练入口风格尽量沿用 `MCUFlowNet/EdgeFlowNet/wrappers/run_train.py`。
5. 代码需要落在 `MCUFlowNet/EdgeFlowNAS` 内独立运行，不直接依赖其他项目目录。
6. 本机验证环境为 `conda activate tf_work_hpc`，正式训练在 HPC。

### 4.2 FairNAS 代码库现实情况（首轮风险）
1. 当时本地 `FairNAS` 目录主要只有模型定义与验证脚本，缺少完整超网训练与搜索流水线。
2. 这意味着项目实际采取的是“采用 FairNAS 方法论”，但训练、采样、搜索链路需要在 `EdgeFlowNAS` 内自行实现。

### 4.3 首轮方案对比

#### 方案 A（最终采用）
1. 构建包含 9 个 Choice Block 的 one-shot supernet。
2. 用 strict fairness 采样，使每个 block 的 3 个选项在公平周期内恰好训练一次。
3. 超网收敛后按约束筛子网并排序。
4. 对 Top-K 子网做独立训练，输出最终 Pareto。

优点：
1. 最接近 FairNAS 核心方法。
2. 计算成本显著低于 19,683 个子网全量独立训练。
3. 便于后续扩展新选项。

缺点：
1. 在 TF1 风格训练框架里实现复杂度偏高。
2. 超网排序与独立重训精度可能有偏差，需要 Top-K 重训校正。

#### 方案 B（分阶段 FairNAS）
1. 阶段 1 固定 head，只搜 backbone。
2. 阶段 2 固定 backbone 最优若干，只搜 head。
3. 阶段 3 联合微调与重训。

优点：
1. 每阶段搜索空间更小，调试更快。
2. 更适合先把链路跑通再扩大。

缺点：
1. 不是单阶段联合搜索，可能错过 backbone/head 交互最优。
2. 实验设计与日志管理更复杂。

#### 方案 C（不做超网，直接抽样独立训练）
1. 不做权重共享。
2. 按公平采样策略抽一批子网逐个独立训练。

优点：
1. 训练评估最“真实”，无共享权重偏置。
2. 实现逻辑直观。

缺点：
1. 训练成本最高，周期可能不可接受。
2. 对 HPC 资源与排队系统要求高。

### 4.4 代码落地方案对比

#### 独立副本方案（最终采用）
1. 在 `EdgeFlowNAS` 内复制并裁剪最小运行集（`code/`, `wrappers/`, `network/`, `misc/` 所需子集），再加 NAS 模块。
2. 优点是满足“代码归一到 EdgeFlowNAS”的约束，便于 HPC 迁移和长期维护。

#### 软链接/路径复用方案（未采用）
1. 起步快，但不满足“独立可运行副本”的硬约束。
2. 因此仅保留为首轮讨论记录，不作为当前工程路径。

### 4.5 首轮建议结论
1. 搜索策略选择 `Strict Fairness One-Shot Supernet`。
2. 代码组织选择 `EdgeFlowNAS` 独立可运行副本。
3. 验证节奏为本机 `tf_work_hpc` 跑最小超网 1-2 epoch，再上 HPC 全量。
