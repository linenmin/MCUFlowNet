# 02 超网训练专项计划（仅 Supernet）

更新时间: 2026-02-16

## 1. 文档目标

本计划只聚焦 **EdgeFlowNAS 的超网训练阶段**。  
不包含搜索（NSGA-II）与最终子网重训，这两部分暂时全部排除在本计划外。

## 2. 范围定义

`In Scope`

1. 在 `MCUFlowNet/EdgeFlowNAS` 内建立可独立运行的 TF1 超网训练链路。
2. 实现 FairNAS 风格的 Strict Fairness 单路径训练循环。
3. 固化输入分辨率、数据流、训练预算与验收规则。
4. 形成可复现的日志、checkpoint、配置与基础评估输出。

`Out of Scope`

1. 候选架构搜索与 Pareto 排序。
2. 上板批量复测流程。
3. Top-K 子网独立重训。

## 3. 已确认选择（已锁定）

1. 训练栈：沿用 TF1 训练栈（从 EdgeFlowNet 复制最小必要代码到 EdgeFlowNAS）。
2. 超网方法：Strict Fairness One-Shot（3-path same-batch 累积梯度后单次更新）。
3. BN 策略：共享 BN；评估子网前执行 BN 统计重估。
4. 超网训练数据：FC2 only。
5. 输入策略：随机裁剪到 `H=180, W=240`。
6. 搜索空间可变维度：仅变 Backbone 深度（Deep1/2/3）与 Head kernel（7/5/3），其余结构固定。
7. 超网预算：`200 epochs`。
8. 超网完成验收主标准：公平计数 + 固定验证子网池 EPE。
9. EPE 评估频率：每 `1 epoch`。
10. 固定验证子网池大小：`12`。
11. 早停：`patience=15`, `min_delta=0.002`。
12. 优化器与学习率：`Adam + cosine decay`。
13. BN 重估批次数：`8 batches`。
14. 随机种子策略：`单种子固定`。
15. Weight Decay：`4e-5`。
16. 梯度裁剪：`Global norm clip=5.0`。
17. 固定验证子网池构建：`分层覆盖采样（12 个）`。
18. FC2 验证集来源：`直接使用 FC2_test`（用于 supernet 排名阶段）。

## 4. 超网编码规范

统一使用 9 维架构码 `arch_code`，每位取值为 `0/1/2`。

1. `arch_code[0]` -> `EB0` 深度选择（`0=Deep1,1=Deep2,2=Deep3`）
2. `arch_code[1]` -> `EB1` 深度选择
3. `arch_code[2]` -> `DB0` 深度选择
4. `arch_code[3]` -> `DB1` 深度选择
5. `arch_code[4]` -> `H0Out` kernel 选择（`0=7x7,1=5x5,2=3x3`）
6. `arch_code[5]` -> `H1` kernel 选择
7. `arch_code[6]` -> `H1Out` kernel 选择
8. `arch_code[7]` -> `H2` kernel 选择
9. `arch_code[8]` -> `H2Out` kernel 选择

## 5. 目录与文件落地（超网阶段）

计划新增或迁移的核心文件（均在 `EdgeFlowNAS` 下）：

1. `EdgeFlowNAS/code/network/MultiScaleResNet_supernet.py`
2. `EdgeFlowNAS/code/nas/arch_codec.py`
3. `EdgeFlowNAS/code/nas/fair_sampler.py`
4. `EdgeFlowNAS/code/nas/supernet_eval.py`
5. `EdgeFlowNAS/code/train_supernet.py`
6. `EdgeFlowNAS/wrappers/run_supernet_train.py`
7. `EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`
8. `EdgeFlowNAS/outputs/supernet/`（日志、ckpt、fairness 计数、EPE 曲线）

## 6. 训练循环规范（Strict Fairness）

每个训练 step（公平周期）执行：

1. 对 9 个 Choice Block 分别生成一个 `[0,1,2]` 随机排列。
2. 组合得到 3 个子网编码 `arch_0, arch_1, arch_2`，保证每个 block 的 3 个选项在本周期各出现一次。
3. 从 FC2 取同一个 batch（180x240 随机裁剪）；
4. 依次执行 `arch_0 -> arch_1 -> arch_2` 前向反向；
5. 梯度累计完成后执行一次 optimizer step；
6. 更新公平计数字典（每个 block 每个选项的累积被训练次数）。

## 7. 评估与早停规范

每个 epoch 结束执行：

1. 在固定 12 个验证子网上评估 EPE（FC2 验证子集）。
2. 当前选择：`code/dataset_paths/FC2_test.txt` 直接作为 supernet 阶段的验证来源。
3. 说明：该选择仅服务于超网阶段的子网相对排序，不作为最终子网训练/报告的独立测试结论。
4. 每个验证子网评估前执行 BN 统计重估（仅刷新均值方差，不更新权重）。
5. 记录指标：
1. `mean_epe_12`
2. `std_epe_12`
3. `fairness_gap`（各选项计数最大最小差）
6. 验收条件：
1. `fairness_gap == 0`（按周期检查）
2. `mean_epe_12` 在验证记录上达到最优或进入稳定区间
7. 早停条件：
1. 连续 `15` 次 epoch 评估未提升超过 `0.002`，触发早停。

## 8. 分阶段任务与小测试

### 阶段 A：训练骨架迁移

1. 复制并裁剪 TF1 训练必需模块到 EdgeFlowNAS。
2. 小测试：
1. `python wrappers/run_supernet_train.py --help` 正常输出。
2. 1 step smoke run 可启动并写日志。

### 阶段 B：超网网络结构

1. 将 Bilinear 骨架改造为可按 `arch_code` 动态选择深度与 head kernel。
2. 小测试：
1. 随机 5 组 `arch_code` 前向均通过。
2. 输出张量尺寸与 multiscale 逻辑正确。

### 阶段 C：Strict Fairness 采样与训练

1. 实装公平采样器和三路径同 batch 累积更新。
2. 小测试：
1. 连续 300 step 统计每个 block 三选项计数严格相等。
2. 固定 seed 下采样序列可复现。

### 阶段 D：评估与早停

1. 固定 12 子网评估器与 BN 重估流程。
2. 小测试：
1. 评估可输出 `mean_epe_12/std_epe_12`。
2. 早停逻辑在模拟曲线上按预期触发。

## 9. 产出物清单（超网阶段）

1. `supernet_best.ckpt`
2. `supernet_last.ckpt`
3. `fairness_counts.json`
4. `eval_epe_history.csv`
5. `train_manifest.json`（记录 seed、配置、commit、输入分辨率）
6. `supernet_training_report.md`

## 10. 单模型训练指令参考（用于超网 wrapper 兼容设计）

你提供的单模型训练命令如下，后续 `run_supernet_train.py` 参数风格将尽量与之对齐：

```bash
python wrappers/run_train.py --gpu_device 0 --num_epochs 400 --batch_size 32 --lr 1e-4 --network_module sramTest.network.MultiScaleResNet_bilinear --load_checkpoint --resume_experiment_name multiscaleresnet_bilinear_fc2_20260212_150419_fast --experiment_name multiscaleresnet_bilinear_fc2_20260212_150419_fast --fast_mode
```

超网训练 wrapper 会在兼容上述通用参数的基础上，新增 supernet 专用参数（如 fairness cycle、arch seed、eval pool 配置、bn recalibration 配置）。

## 11. 后续可选优化（非阻塞）

以下优化不影响当前超网训练计划执行，可在超网跑通后按需要追加：

1. 将 FC2_test 替换为 train-holdout 验证集，对比排序一致性。
2. 增加小规模 Kendall Tau spot-check，验证超网排序保真度。
3. 追加第二随机种子重复实验，评估排名稳定性。
