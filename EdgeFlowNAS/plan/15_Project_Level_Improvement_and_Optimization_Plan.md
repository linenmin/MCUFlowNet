# 15_项目级改进与架构优化计划

更新时间: 2026-02-22  
状态: Draft v3（已按最新反馈收敛）

## 0. 本轮决策确认（按你的反馈）

| 项目 | 结论 | 说明 |
|---|---|---|
| P0-1 评估协议强校验 | 不改 | 这是中途手动操作导致，不作为当前代码改造优先项。 |
| P0-2 特征蒸馏 | 采纳 A | 需要适配当前“含不确定性输出”的训练图。 |
| P0-3 Deep Block 解耦 | 采纳 A | 直接改为解耦版本，不保留现有嵌套方案。 |
| P0-4 评估策略 | 部分采纳 | 保持固定评估样本，不做轮转偏移；新增 CLI 可控评估频率，用于断点恢复时手动切换。 |
| P0-5 困难子网课程化 | 不改 | 训练期 `mean_epe_12` 主要用于判断是否逐渐收敛。 |
| P0-6 split 统一 | 已确认 | 当前 `val=640`，按该口径执行。 |

结论：  
当前只做三件事：`CLI 评估频率透传`、`Deep Block 解耦`、`蒸馏适配不确定性输出`。

---

## 1. 保留的核心问题（不再扩展其他方向）

1. `loss` 下降但 `mean_epe_12` 后期平台化，说明训练目标与评估目标存在错配。  
2. `_deep_choice_block` 嵌套结构会引入共享路径冲突，影响超网稳定收敛。  
3. 训练后期需要用更高频评估来确认“是否逐渐稳定”，但目前 `eval_every_epoch` 不能从 CLI 覆写。  

---

## 2. 接下来最应该做的改动计划（执行版）

执行顺序（推荐）：  
1) 先做 CLI 评估频率透传（低风险，马上可用）  
2) 再做 Deep Block 解耦（结构主改动）  
3) 最后做蒸馏适配（基于解耦后的稳定骨架接入）

这样可以减少反复改图，避免蒸馏实现跟着骨架改两次。

### 2.1 任务 A：评估频率透传到 CLI（支持断点恢复调频）

目标：  
在 `run_supernet_train.py` 增加可选参数，让你在续训时直接改 `eval.eval_every_epoch`。

涉及文件（计划）：
1. `MCUFlowNet/EdgeFlowNAS/wrappers/run_supernet_train.py`

改动要点：
1. 新增参数 `--eval_every_epoch`（可选 `int`）。  
2. 在 `_build_overrides` 中透传到 `eval.eval_every_epoch`。  
3. 不改默认值行为，不传即保持 yaml 原值。  
4. 代码注释保持当前风格：每行简短中文注释，说明“为什么要这样做”。  

测试计划：
1. 新增/更新 wrapper 单测，验证 `args.eval_every_epoch=5` 时输出覆写正确。  
2. `--dry_run` 验证合并配置中 `eval.eval_every_epoch` 变化正确。  
3. 断点恢复 smoke：`--load_checkpoint --resume_experiment_name ... --eval_every_epoch 5`，日志应打印 `eval_every_epoch=5`。  

验收标准：
1. 仅在传参时生效，未传参行为完全不变。  
2. 续训日志中的频率与 CLI 一致。  
3. 不影响现有训练流程与 checkpoint 恢复逻辑。  

---

### 2.2 任务 B：Deep Block 直接改为解耦分支（不保留嵌套）

目标：  
将 `_deep_choice_block` 从 `out1->out2->out3` 嵌套改为“同输入的三条独立深度分支”。

涉及文件（计划）：
1. `MCUFlowNet/EdgeFlowNAS/code/network/MultiScaleResNet_supernet.py`

改动要点：
1. `branch1`: 输入 `x`，1 个残差块。  
2. `branch2`: 输入 `x`，2 个残差块（分支内串联）。  
3. `branch3`: 输入 `x`，3 个残差块（分支内串联）。  
4. 保持 `arch_code` 语义不变（0/1/2 对应浅/中/深）。  
5. 删除旧嵌套路径，不做兼容分支。  
6. 输出尺度、head 结构、接口签名全部保持不变。  

测试计划：
1. 图构建测试：三种 choice 都可前向，输出 shape 不变。  
2. 变量命名检查：三分支参数独立，避免误共享。  
3. 1 epoch smoke 训练：确认无 shape/BN/update_op 错误。  
4. 现有 `supernet_subnet_distribution` 相关测试保持通过。  

验收标准：
1. 训练可正常启动并跑完 smoke。  
2. `preds` 仍是 3 尺度，维度与原版一致。  
3. 无新增临时脚本和冗余兼容代码，结构保持干净。  

---

### 2.3 任务 C：蒸馏适配当前“不确定性输出”训练版本（P0-2 A）

目标：  
在保留现有 uncertainty 输出训练的前提下，新增无参数特征蒸馏损失。

涉及文件（计划）：
1. `MCUFlowNet/EdgeFlowNAS/code/engine/train_step.py`  
2. `MCUFlowNet/EdgeFlowNAS/code/engine/supernet_trainer.py`  
3. `MCUFlowNet/EdgeFlowNAS/code/network/MultiScaleResNet_supernet.py`（若需要暴露 encoder 特征）  
4. `MCUFlowNet/EdgeFlowNAS/configs/supernet_fc2_180x240.yaml`（新增 distill 配置项）

设计约束：
1. 不改当前输出头格式（仍是 flow + uncertainty）。  
2. 蒸馏作用于中间特征，不直接改预测头张量结构。  
3. 采用无参数 `Channel Maximize` 对齐后计算 L2。  
4. 通过开关启用：`distill.enabled=false` 时行为与当前完全一致。  

建议配置项（示例）：
1. `train.distill.enabled`  
2. `train.distill.lambda`  
3. `train.distill.teacher_ckpt`  
4. `train.distill.layers`（可选，控制蒸馏层位）

测试计划：
1. 单测：`distill.enabled=false/true` 两种图都能构建。  
2. 单测：特征压缩后的 shape 与 loss 数值可计算。  
3. smoke：小步训练无 NaN，无梯度断裂。  
4. 日志：新增 `loss_flow/loss_unc/loss_distill/loss_total` 可观测字段。  

验收标准：
1. 关闭蒸馏时，与当前训练链路数值一致（允许浮点微差）。  
2. 打开蒸馏时，训练稳定不报错，且日志包含 distill 指标。  
3. 在同口径短跑对照中，`mean_epe_12` 至少不劣化（先过“不退化门”）。  

---

## 3. 统一测试与验收清单（必须执行）

### 3.1 静态与单测
1. 运行新增/更新的 wrapper 与网络单测。  
2. 运行已有 `tests`，确认无回归。  

### 3.2 训练 smoke（最小开销）
1. 先跑 1~3 epoch smoke，验证图、日志、checkpoint、resume 正常。  
2. 分别验证：  
  - `仅改评估频率`  
  - `解耦后无蒸馏`  
  - `解耦后有蒸馏`  

### 3.3 对照验收（同口径）
1. 固定 `val=640`、固定 eval pool、固定 eval 样本顺序。  
2. 同一配置仅改一个变量做 AB。  
3. 通过条件：  
  - 功能通过：无报错、可续训、日志字段完整  
  - 指标通过：先过“不退化门”，再追求明显提升  

---

## 4. 本文档后的执行原则

1. 本轮不再扩展其他优化项，避免任务膨胀。  
2. 代码保持当前目录分层，不引入跨层耦合。  
3. 所有新增代码保持简洁、可维护，注释延续“每行简短中文说明”风格。  
4. 每个阶段先有测试门，再进入下一阶段。  

