# EdgeFlowNAS 计划索引

更新时间: 2026-02-16

## 目标

基于 `EdgeFlowNet` 的 Bilinear 骨架，在 `MCUFlowNet/EdgeFlowNAS` 内构建 FairNAS 风格 NAS 方案。当前阶段只推进 **Supernet 训练**，不推进搜索与重训。

## 已完成阅读与对齐

已阅读文档：

1. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/00_README.md`
2. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/01_Architecture_Overview.md`
3. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/02_Bottleneck_and_33Decoder.md`
4. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/03_Backbone_Block_Comparison.md`
5. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/04_Head_Upsample_NAS.md`
6. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/05_Block_Type_NAS_Brainstorm.md`
7. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/06_Final_NAS_Strategy.md`
8. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/07_Appendix.md`

已阅读源码：

1. `MCUFlowNet/EdgeFlowNet/sramTest/network/MultiScaleResNet_bilinear.py`
2. `MCUFlowNet/EdgeFlowNet/sramTest/network/MultiScaleResNet_cell.py`
3. `MCUFlowNet/EdgeFlowNet/sramTest/network/MultiScaleResNet_cell_33decoder.py`
4. `MCUFlowNet/EdgeFlowNet/sramTest/network/MultiScaleResNet_cell_upsample_search.py`
5. `MCUFlowNet/EdgeFlowNet/sramTest/auto_benchmark_all.py`
6. `MCUFlowNet/EdgeFlowNet/wrappers/run_train.py`
7. `MCUFlowNet/EdgeFlowNet/code/train.py`
8. `MCUFlowNet/EdgeFlowNet/code/misc/DataHandling.py`
9. `MCUFlowNet/EdgeFlowNet/code/network/MultiScaleResNet.py`
10. `MCUFlowNet/EdgeFlowNet/code/network/BaseLayers.py`
11. `FairNAS/README.md`
12. `FairNAS/models/FairNAS_A.py`
13. `FairNAS/models/FairNAS_B.py`

## 计划文档序列

1. `00_Plan_Index.md`：索引与进度总览（本文件）
2. `01_Framework_Options.md`：大框架方案与首轮决策项（已完成）
3. `02_Supernet_Training_Spec.md`：超网训练专项规格（已完成，主规格文档）
4. `03_Supernet_Task_Breakdown.md`：任务拆分与验收标准（已完成）
5. `04_Supernet_Execution_Gates.md`：按执行顺序的门禁清单与测试模板（已完成）
6. `05_Engineering_File_Management_and_Code_Style.md`：工程级文件管理与代码注释规范（已完成）

## 当前锁定决策

1. 超网方法：Strict Fairness One-Shot（3-path same-batch 累积后单次更新）。
2. 训练栈：TF1 兼容链路，代码落地在 `EdgeFlowNAS`。
3. 数据与分辨率：FC2，随机裁剪到 `180x240`。
4. 验证来源：`FC2_test`（仅用于 supernet 排名阶段）。
5. 优化器：`Adam + cosine decay`。
6. 正则与稳定化：`weight_decay=4e-5`，`grad_clip_global_norm=5.0`。
7. BN：共享 BN，评估前重估 `8 batches`。
8. 验证池：固定 `12` 子网，分层覆盖采样。
9. 评估与停止：每 epoch 评估，`patience=15`，`min_delta=0.002`。
10. 预算与复现：`200 epochs`，单种子固定。

## 当前状态

1. 已完成：计划框架、超网规格、任务拆分、验收门禁模板、工程级文件管理规范。
2. 下一步：先按 `05` 完成目录与函数拆分，再按 `04` 的 Gate-1 到 Gate-4 代码落地与本机 smoke。
