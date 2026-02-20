# 14 单模型与 Supernet 训练细节对比及数据利用率优化计划
更新时间: 2026-02-20

## 0. 输入与结论范围
1. 对比对象:
- 单模型训练入口: `MCUFlowNet/EdgeFlowNet/wrappers/run_train.py`
- Supernet 训练入口: `MCUFlowNet/EdgeFlowNAS/wrappers/run_supernet_train.py`
2. 对比链路:
- 单模型主循环: `MCUFlowNet/EdgeFlowNet/code/train.py`
- 单模型数据采样: `MCUFlowNet/EdgeFlowNet/code/misc/BatchCreationTF.py`
- Supernet 主循环: `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`
- Supernet 数据采样: `MCUFlowNet/EdgeFlowNAS/code/data/fc2_dataset.py`
3. 数据规模假设:
- train: 17785
- val/test: 4447（你提供）
4. 本文聚焦: 训练细节、数据利用率、可改进点，不涉及最终子网搜索决策。

## 0.1 状态更正（2026-02-20）
1. 你选择的选项 A 已落地: supernet 现已使用 `epoch_mode=full_pass` + `train_sampling_mode=shuffle_no_replacement`。
2. `gpu_device` 已在 supernet 训练中生效（日志可见 `gpu_device=0 applied`）。
3. `fast_mode` 已从 supernet 训练链路移除，不再作为“看似可配但不生效”的死参数。
4. supernet 已补齐 `run_manifest`，恢复训练会做关键语义一致性校验。

---

## 1. 核心差异总览（先看结论）
注: 本节保留 run1 历史基线视角，部分工程项已在 `0.1` 中完成修复。
1. 单模型把“epoch”定义为接近全数据一次遍历（`NumTrainSamples / batch`），而 supernet 当前是固定 `steps_per_epoch=50`，每个 epoch 只抽样少量训练数据。
2. 两者当前都属于“随机有放回抽样”，不是无放回遍历。
3. supernet 多了严格公平采样（每步 3 条架构路径）、BN 重校准、固定 12 子网 eval、梯度累积与全局裁剪，训练逻辑更复杂。
4. supernet 当前配置下，单 epoch 对训练集覆盖明显偏低，早期统计波动会更大，且“epoch”不再代表完整数据遍历。
5. supernet wrapper 的部分参数在主训练实现里未生效（`gpu_device`、`fast_mode`），存在工程一致性问题。

---

## 2. 逐项细节对比与优劣

### 2.1 Epoch 定义
1. 单模型:
- `NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)`
- 对 FC2+batch16，约 `1111` step/epoch。
2. Supernet:
- 固定 `steps_per_epoch`（当前 yaml: 50）。

优劣:
1. 固定 step 优点: 训练时长可控，适合超网高成本训练。
2. 固定 step 缺点: 与数据规模脱钩，epoch 统计含义弱，容易低覆盖。
3. 近全遍历 epoch 优点: 口径直观，数据利用更充分。
4. 近全遍历 epoch 缺点: 超网成本高，单 epoch 很慢。

### 2.2 训练采样方式（是否有放回）
1. 单模型: `random.randint` 每次独立抽样（有放回）。
2. Supernet: `sampling_mode="random"` + 每样本随机抽（有放回）。

优劣:
1. 有放回优点: 实现简单，随机性强。
2. 有放回缺点: 同 epoch 重复样本多，覆盖率低，统计噪声大。

### 2.3 数据划分使用方式
1. 单模型训练主循环只吃 train 列表，训练时无固定 val 监控（主循环内）。
2. Supernet 每 epoch 会跑固定 12 子网 eval（train 做 BN recal，val 做 EPE）。

优劣:
1. Supernet 可监控性更强。
2. 但 supernet 当前 eval 覆盖很小（见 2.10），对泛化变化的敏感性有限。

### 2.4 输入分辨率
1. 单模型 FC2 patch: `352x480`。
2. Supernet FC2 输入: `180x240`。

优劣:
1. 小分辨率速度更快，适合超网迭代。
2. 任务难度/误差尺度与单模型基准不完全同口径。

### 2.5 输入归一化
1. 单模型: batch 直接 `float32`，未统一映射到 `[-1,1]`。
2. Supernet: `standardize_image_tensor` 映射到 `[-1,1]`。

优劣:
1. Supernet 输入尺度更规范。
2. 与旧单模型训练口径存在差异，直接数值对比需谨慎。

### 2.6 数据增强
1. 单模型: 代码里 `args.Augmentations='None'`，基本等价无增强，仅随机裁剪。
2. Supernet: 随机裁剪（train），中心裁剪（eval）。

优劣:
1. 当前两者都偏“弱增强”，泛化上限可能受限。

### 2.7 优化器与学习率策略
1. 单模型: Adam + 常数 LR（`OptimizerParams[0]=LR`）。
2. Supernet: Adam + cosine LR（按 `global_step / (num_epochs*steps_per_epoch)`）。

优劣:
1. cosine 更平滑，后期更稳。
2. 但 supernet 的 schedule 强绑定 `total_steps`，续训时改 `num_epochs` 会改变轨迹。

### 2.8 梯度更新语义
1. 单模型: 每个 mini-batch 一次更新。
2. Supernet: 每 step 内对 3 个公平架构（再乘 micro slices）累积，再 batch 级一次 clip+apply。

优劣:
1. Supernet 更符合严格公平共享权重训练目标。
2. 但每 step 使用同一批样本服务 3 个架构，数据多样性利用偏弱。

### 2.9 梯度裁剪
1. 单模型: 默认无有效全局裁剪（注释掉）。
2. Supernet: 已改为“累积后一次 global clip”，当前阈值 200。

优劣:
1. Supernet 稳定性更可控。
2. 阈值需要结合日志持续校准。

### 2.10 训练内评估口径
1. Supernet 当前 eval:
- `eval_pool_size=12`
- `bn_recal_batches=8`
- `eval_batches_per_arch=4`
- `batch=32`
2. 即每架构 val 仅 `128` 样本；且每轮固定前序 batch。

优劣:
1. 优点: 高一致性、可比性强。
2. 缺点: 覆盖有限，可能对真实泛化变化不敏感。

### 2.11 Checkpoint/恢复
1. 单模型: 有 `run_manifest` 严格校验恢复一致性。
2. Supernet: 当前缺少等价严格 manifest 校验。

优劣:
1. 单模型恢复安全性更高。
2. Supernet 仍有“参数改了但误恢复”的工程风险。

### 2.12 可复现性
1. 单模型: 随机种子控制较弱。
2. Supernet: 有统一 seed 设置，复现性更好。

### 2.13 参数对齐完整性
1. Supernet wrapper 传了 `gpu_device`，但训练实现中未生效。
2. Supernet wrapper `fast_mode` 仅写入配置，训练实现未消费。

优劣:
1. 这是工程一致性问题，不是算法问题。
2. 需要尽快修复，避免“以为生效其实未生效”。

### 2.14 输入管线与 CPU 预取
1. 单模型: 具备 `BatchPrefetcher` 线程队列，`prefetch_batches` 可开启 CPU 预取（`fast_mode` 默认会拉高到 8）。
2. Supernet: 当前 `FC2BatchProvider` 为同步 `next_batch` 取数，未内置预取线程队列。

优劣:
1. 单模型在 I/O 偏重场景更容易隐藏数据加载开销。
2. Supernet实现更直接、调试更简单，但在高分辨率或慢盘环境下更容易被 I/O 卡住。

### 2.15 日志粒度与可观测性
1. 单模型: 以 `global_step` 为核心写 TensorBoard（`summary_every`/`summary_flush_every`），步级信号细。
2. Supernet: 以 epoch 为核心写 CSV/JSON 与控制台日志，步级细节默认较少。

优劣:
1. 单模型更适合排查短时抖动。
2. Supernet日志更轻量，但定位“某一段 step 异常”时信息密度偏低。

### 2.16 Checkpoint 策略与最佳模型判据
1. 单模型: 训练中按固定 step 间隔 + 每 epoch 保存，无内置“按验证指标最佳”分支。
2. Supernet: 每 epoch 保存 `last`，并按 `mean_epe_12`（固定 12 子网评估）更新 `best`，同时受 early-stop 配置驱动。

优劣:
1. Supernet更贴合“按评估指标保留最优权重”的实验目标。
2. 但 `best` 依赖于小规模固定 eval 池，若 eval 口径偏窄，`best` 可能过拟合该池。

### 2.17 数据源组织与 split 定义
1. 单模型: 走 `dataset_paths` 索引文件（`*_train.txt/*_val.txt/*_test.txt`）。
2. Supernet: 走 `train_dir/val_dir` 文件夹扫描。
3. 当前仓库索引文件检查结果: `FC2_train.txt=17785`，`FC2_test.txt=4446`，`FC2_val.txt=0`。

优劣:
1. 文件夹扫描更直观，不依赖额外索引文件维护。
2. 若两套流程 split 口径不一致（例如一个用 test 当验证，另一个用 val 文件夹），则结果不可直接横向比较。

---

## 3. run1（当前 yaml）数据利用率量化
当前超网配置:
- `steps_per_epoch=50`
- `batch_size=32`
- `num_epochs=200`
- train=17785

量化结果:
1. 每 epoch 抽样数:
- `50 * 32 = 1600`
2. 每 epoch 抽样占 train 比例:
- `1600 / 17785 = 8.996%`
3. 若按有放回抽样，单 epoch 期望唯一样本数约:
- `~1530`（约 `8.60%` train）
4. batch32 下“近全遍历”所需 step:
- `floor(17785/32)=555`（或 ceil 556）
5. 当前 50 step 相对全遍历 step 比例:
- `50 / 555 = 9.01%`
6. 200 epoch 总抽样数:
- `1600 * 200 = 320000`
7. 等价“完整数据遍历次数”:
- `320000 / 17785 = 18.02`
8. 口径提醒:
- 上述量化仅针对 train split；验证集规模请固定到同一来源后再比较（索引文件口径与文件夹口径当前可能不一致）。

结论:
1. 你的判断成立: run1 的“每个 epoch”远未充分利用完整 train 集。
2. 但 200 epoch 累计抽样并不低，约等价 18 次全数据量级。
3. 真正问题是“单 epoch 覆盖低 + 统计噪声高 + epoch 含义弱”，不是“总样本量绝对不足”。

---

## 4. 当前优先问题清单（2026-02-20 版本）

### P0（高优先级，直接影响你现在的训练判断）
1. 训练损失与评估指标口径不完全一致: 训练优化的是 `multiscale_uncertainty_loss + weight_decay`，而模型选择看的是 `mean_epe_12`（最终尺度 flow EPE），会出现 loss 继续下降但 EPE 几乎不动。
2. eval 覆盖偏窄: 每子网仅 `eval_batches_per_arch=4`（batch32 即 128 样本），且每次 `reset_cursor(0)` 固定评估同一前缀样本，对真实泛化变化不够敏感。
3. BN 重校准覆盖偏窄: 每子网仅 `bn_recal_batches=8`（256 样本），同样固定前缀，可能对“困难子网”校准不足。
4. 固定池均值受困难子网强影响: 当前 `arch_index=2` 长期垫底，`mean_epe_12` 会被单一子网显著拖动，影响 best checkpoint 判据。
5. 数据集口径仍需统一: 本次日志显示 `val_samples=640`，和你前面给出的 `4447` 不一致，跨实验横向比较有风险。

### P1（中优先级，影响收敛效率和可诊断性）
1. cosine 调度对 200 epoch 前段衰减偏慢，前 49 epoch 学习率仅从 `1e-4` 到 `8.59e-5`，平台期后缺少明显“再收敛触发”。
2. `grad_clip_global_norm=200` 基本很少触发，当前等效于“几乎不裁剪”，需要重新标定是否保留该超参。
3. 训练 `random crop`、评估 `center crop` 的口径差异，会放大“train loss 改善但 eval 指标不动”的观感。
4. 日志没有把 per-arch 历史曲线结构化落盘（目前主要在 `arch_rank_12` 字符串），后续定位慢子网瓶颈成本高。

### P2（可选增强）
1. 每 step 的 3 子网共用同一 batch，可考虑轻量改造为“子网间错位 micro-slice”，进一步提高样本多样性利用率。
2. 在保持高频小 eval 的同时，增加低频大 eval（例如每 5~10 epoch 用更大 `eval_batches_per_arch`）做纠偏。

---

## 5. 改进选项（含优缺点）
注: 选项 A 已作为 run2 主配置执行，以下保留用于后续扩展选择。

### 选项 A: 全遍历 epoch（无放回 + 每 epoch shuffle）
做法:
1. 增加 `train_sampling_mode=shuffle_no_replacement`。
2. 每 epoch 遍历全 train（`steps_per_epoch=ceil(N/batch)`）。

优点:
1. epoch 语义清晰。
2. 数据利用充分，统计更稳定。

缺点:
1. 超网耗时显著增加。
2. 同等总训练时长下可跑 epoch 数变少。

### 选项 B: 保持 fixed steps，但提高步数（例如 50 -> 150/200）
优点:
1. 实现改动最小。
2. 显著提升每 epoch 覆盖率。

缺点:
1. 仍是有放回随机，重复样本仍多。
2. epoch 仍不等于全遍历。

### 选项 C: 混合模式（推荐）
做法:
1. 增加 `epoch_mode`:
- `fixed_steps`
- `full_pass`
- `by_samples`（按目标样本数）
2. 增加 `train_samples_per_epoch`（例如 0.3N / 0.5N / 1.0N）。
3. 采样策略支持 `random_with_replacement` 与 `shuffle_no_replacement`。

优点:
1. 兼顾算力预算与数据利用率。
2. 可平滑迁移，不需要一次性切到最重配置。

缺点:
1. 实现稍复杂。
2. 需要配套日志和默认值设计。

### 选项 D: 维持现状
优点:
1. 风险最小。
2. 与现有历史结果完全同口径。

缺点:
1. 每 epoch 覆盖率低的问题持续存在。
2. 调参效率与训练信号质量受限。

---

## 6. 推荐执行路线（建议按阶段）

### 阶段 1（已完成：工程一致性）
1. 已让 `gpu_device` 在 supernet train 真正生效。
2. 已移除 supernet 训练链路中的 `fast_mode` 死参数。
3. 已增加 supernet 的 `run_manifest`/配置哈希恢复校验。

### 阶段 2（提升数据利用率，控制成本）
1. 引入 `epoch_mode` + `train_sampling_mode`。
2. 先跑中档配置验证:
- `epoch_mode=by_samples`
- `train_samples_per_epoch ≈ 0.5 * train_size`
- `train_sampling_mode=shuffle_no_replacement`

### 阶段 3（增强评估稳健性）
1. 保留当前固定小 eval（用于高频监控）。
2. 增加低频全量 eval（例如每 5~10 epoch），用于纠偏。

### 阶段 4（学习率口径收敛）
1. 将 LR 调度从“按总 step 比例”改为“按累计 seen samples 比例”或明确固定 total_steps 政策。
2. 保证续训改 `num_epochs` 时不会发生隐式调度漂移。

---

## 7. 验收指标
1. 训练参数一致性:
- `gpu_device/fast_mode` 日志可证明生效。
2. 数据利用率:
- 每 epoch 样本覆盖率有明确统计输出。
3. 稳定性:
- `mean_epe_12` 下降趋势更平滑，`std_epe_12` 不恶化。
4. 工程安全:
- 恢复训练出现配置不一致时能被阻断并报错。

---

## 8. 对“是否充分利用数据”的直接回答
1. 按“每个 epoch”的定义: 当前 run1（50 step）确实远未充分利用 17785 条 train。
2. 按“200 epoch 总体”看: 总抽样约等价 18 次全数据量级，不算绝对少。
3. 当前最值得改进的是“每 epoch 覆盖口径与采样方式”，而不是单纯把 epoch 数继续堆高。

---

## 9. run2 日志复盘（full_pass, epoch 1~49）

### 9.1 运行口径
1. 日志确认: `epoch_mode=full_pass`，`train_sampling_mode=shuffle_no_replacement`，`steps_per_epoch=556`。
2. 数据规模: `train_samples=17785`，`val_samples=640`。
3. 评估设置: `eval_pool_size=12`，`bn_recal_batches=8`，`eval_batches_per_arch=4`，`batch_size=32`。

### 9.2 量化现象
1. 训练 loss 持续下降: `11.218 -> 4.525`（约 `-59.7%`）。
2. `mean_epe_12` 仅小幅下降: `9.7749 -> 9.7207`（约 `-0.0542`, `-0.55%`）。
3. 约从 epoch 20 开始进入平台区，后续多数波动在 `9.72x`。
4. `std_epe_12` 大体从 `~0.05` 降到 `~0.01`，但 epoch 47 出现 `0.039` 的异常抬升。
5. `arch_index=2` 长期位于末位，且与最佳架构常有 `0.03~0.14` 的 EPE 差距，是当前最明显“困难子网”。
6. 梯度范数从 `36.1` 降到约 `13~15`；`clip_rate` 几乎常年 `0~0.0036`，说明 clip(200)几乎不参与训练动态。
7. 学习率从 `1.00e-4` 到 `8.59e-5`，前 49/200 epoch 仅衰减约 `14.1%`，属于慢退火阶段。

### 9.3 问题诊断（针对“loss 降、EPE 不动”）
1. 目标错配是首因: loss 优化含不确定性项与多尺度项，EPE 只看最终流分支，二者不必同步单调。
2. eval 统计分辨率不足: 每子网仅 128 个 val 样本，且固定前缀，容易出现“指标平台/不敏感”。
3. BN 重校准样本不足且固定: 每子网固定 256 train 样本，对分布复杂或较重内核子网可能不够。
4. 固定 12 子网的均值判据被困难子网拖动: 即便多数子网在改进，`mean_epe_12` 也可能近似不变。
5. `fairness_gap=0` 仅代表“采样次数公平”，不代表“性能公平”或“各子网同速收敛”。
6. val 口径未统一（640 vs 4447）会掩盖真实进展，尤其在跨 run、跨脚本对比时。

### 9.4 当前训练效率风险
1. 单 epoch 用时约 12~13 分钟，若平台期持续到 200 epoch，中后段算力利用率偏低。
2. 在当前日志信号下，继续“原样长跑”风险是成本增加但关键指标收益有限。

---

## 10. 针对 run2 的执行清单（建议按顺序）
1. 先做评估口径增强（不改训练）：保持 `eval_every_epoch=1`，把 `eval_batches_per_arch` 从 `4` 提高到 `16` 做对照。
2. 同步把 `bn_recal_batches` 从 `8` 提高到 `32`，观察 `arch_index=2` 是否明显回升。
3. 增加“低频大评估”：每 `5` 或 `10` 个 epoch 追加一次更大覆盖评估，用来校正高频小评估的偏差。
4. 增加结构化日志：每 epoch 落盘 `per_arch_epe`（12 维）与 `bilinear_arch_epe`（固定 baseline 子网）时间序列。
5. 评估指标从单一均值升级为组合判据：同时跟踪 `mean_epe_12`、`median_epe_12`、`p90_epe_12`，避免被单一困难子网完全主导。
6. 学习率策略做一次快收敛对照：保持其他超参不变，单独对照“更快退火曲线”或“阶段式退火”，验证平台期是否缩短。
7. split 口径对齐：明确 run2 的 `val=640` 来源，并在后续对比表中单列“口径标签”，避免与 `4447` 口径混用。
