# EdgeFlowNAS 大框架方案（第一轮确认）

更新时间: 2026-02-16

## 1. 已对齐的技术前提

1. 骨架基于 Bilinear 版本，目的是保持 SRAM 峰值稳定且低于转置卷积骨架。
2. 你指定的搜索空间：
- Backbone Choice Block: `EB0, EB1, DB0, DB1`（4 个位置）
- Backbone 选项: `Deep1, Deep2, Deep3`（表示该位置 ResBlock 重复次数）
- Head Choice Block: `H0Out, H1, H1Out, H2, H2Out`（5 个位置）
- Head 选项: `7x7Conv, 5x5Conv, 3x3Conv`
3. 搜索空间总数（按你当前定义）：
- Backbone: `3^4 = 81`
- Head: `3^5 = 243`
- Total: `81 * 243 = 19,683`
4. 训练入口希望沿用 `MCUFlowNet/EdgeFlowNet/wrappers/run_train.py` 的单模型模板思路。
5. 代码需要落在 `MCUFlowNet/EdgeFlowNAS` 内（复制或重建），不直接依赖其他项目目录运行。
6. 本机验证环境：`conda activate tf_work_hpc`，正式训练在 HPC。

## 2. FairNAS 代码库现实情况（关键风险）

本地 `./FairNAS` 目录当前主要是模型定义与验证脚本（ImageNet/CIFAR 评估），没有看到完整超网训练与搜索流水线脚本。  
这意味着我们需要“采用 FairNAS 方法论”，但训练/采样/搜索执行链路需要在 `EdgeFlowNAS` 内自行实现。

## 3. 大框架选项

## 选项 A（推荐）：Strict Fairness One-Shot Supernet + 候选重训

思路：

1. 构建一个包含 9 个 Choice Block 的超网（4 backbone + 5 head）。
2. 用 FairNAS strict fairness 采样：每个公平周期内，每个 Choice Block 的 3 个选项都恰好训练一次。
3. 超网收敛后，按约束（SRAM/FPS）筛子网并排序。
4. 对 Top-K 子网做独立训练，输出最终 Pareto。

优点：

- 最接近 FairNAS 核心方法。
- 计算成本显著低于 19,683 全量独立训练。
- 便于后续扩展新选项。

缺点：

- 在现有 TF1 风格训练框架里实现复杂度中等偏高。
- 超网排序与真实重训精度可能存在偏差，需要 Top-K 重训校正。

## 选项 B：分阶段 FairNAS（先 Backbone 后 Head）

思路：

1. 阶段 1：固定 Head（如全 3x3 或指定模板）只搜 Backbone。
2. 阶段 2：固定 Backbone 最优若干，只搜 Head。
3. 阶段 3：联合微调与重训。

优点：

- 每阶段搜索空间更小，调试更快。
- 更适合先把链路跑通再放大。

缺点：

- 不是单阶段联合搜索，可能错过 backbone/head 的交互最优组合。
- 实验设计与日志管理更复杂。

## 选项 C：无超网、直接采样子网独立训练（Fairness 仅用于采样频次）

思路：

1. 不做权重共享。
2. 按公平采样策略抽取一批子网（比如 100~500 个），逐个独立训练。

优点：

- 训练评估最“真实”，无共享权重偏置。
- 实现逻辑直观。

缺点：

- 训练成本最高，周期可能不可接受。
- 对 HPC 资源和排队系统要求高。

## 4. 目录级落地方案（与选项 A/B/C 正交）

## 方案 1（推荐）：EdgeFlowNAS 独立可运行副本

在 `EdgeFlowNAS` 内复制并裁剪最小运行集（`code/`, `wrappers/`, `network/`, `misc/` 所需子集），再加 NAS 模块。  
优点：符合你“不要直接引用其他项目目录”的要求，后续迁移 HPC 更干净。  
缺点：初期搬运量稍大。

## 方案 2：先软链接/路径复用，后期再收敛到独立副本

优点：起步快。  
缺点：不满足你当前“代码归一到 EdgeFlowNAS”的硬约束，不建议。

## 5. 第一轮建议

1. 搜索策略：选 **A（Strict Fairness One-Shot）**。
2. 代码组织：选 **方案 1（EdgeFlowNAS 独立副本）**。
3. 验证节奏：先本机 `tf_work_hpc` 跑通最小超网训练 1-2 epoch，再上 HPC 全量。

## 6. 请先确认一个主决策

请先在下面三项中选一个：

1. `A`：Strict Fairness One-Shot Supernet（推荐）
2. `B`：分阶段 FairNAS
3. `C`：独立训练采样子网

你选定后，我会创建 `02_Architecture_Spec.md`，把网络编码、配置文件格式、采样器接口细化到可直接实现的层级。
