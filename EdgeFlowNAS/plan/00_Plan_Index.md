# EdgeFlowNAS 计划索引
更新时间: 2026-02-19

## 项目目标
本目录用于管理 `EdgeFlowNAS` 在 FC2 数据集上的 supernet 训练、诊断与工程化落地文档。

## 文档清单
1. `00_Plan_Index.md`
说明: 总索引与阅读入口。

2. `01_Framework_Options.md`
说明: 超网训练框架与技术路线选项。

3. `02_Supernet_Training_Spec.md`
说明: supernet 训练规格与约束定义。

4. `03_Supernet_Task_Breakdown.md`
说明: 任务拆分与执行分解。

5. `04_Supernet_Execution_Gates.md`
说明: 分阶段执行门禁与验收标准。

6. `05_Engineering_File_Management_and_Code_Style.md`
说明: 工程文件组织、分层边界、代码风格规范。

7. `06_Implementation_Log.md`
说明: 关键实现记录与阶段日志。

8. `07_User_Manual.md`
说明: 训练与评估使用手册。

9. `08_Supernet_Retrospective.md`
说明: 历史问题复盘与根因分析。

10. `09_Fold_Run_Diagnostics_and_CPU_Eval_Plan.md`
说明: fold 训练阶段诊断与后续修订计划（当前聚焦评估稳定性与训练日志口径）。

11. `10_Eval_Consistency_and_Clip_Diagnostics_Plan.md`
说明: Eval 口径一致性与梯度裁剪机制变更（batch 级 clip）诊断计划。

12. `11_200Epoch_结果诊断与续训决策计划.md`
说明: 200 epoch 收敛状态复盘、是否 +100 续训的分档方案与门槛判据。

13. `12_BestCkpt_子网扩展抽样分布分析计划.md`
说明: 解释 best checkpoint 判据，并在最佳权重下做大样本子网分布统计与可视化。

## 当前主线
1. 使用 folder-based FC2 数据加载（`train_dir` / `val_dir`）。
2. 保持 strict-fairness supernet 训练主流程。
3. 优先改进评估稳定性、日志可解释性、排名稳定性指标（Kendall tau）。
4. CPU 并发评估方案已降级为非主线，不作为当前优先项。
