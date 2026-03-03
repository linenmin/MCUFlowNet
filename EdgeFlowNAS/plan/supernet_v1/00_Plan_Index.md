# EdgeFlowNAS 计划索引
更新时间: 2026-03-03

## 项目目标
本目录用于管理 `EdgeFlowNAS` 在 FC2 数据集上的 supernet 训练、诊断与工程化落地文档。

## 推荐阅读顺序（当前主线）
1. `10_Eval_Consistency_and_Clip_Diagnostics_Plan.md`
2. `11_200Epoch_结果诊断与续训决策计划.md`
3. `06_Implementation_Log.md`

## 文档清单（按文件号）
| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `00_Plan_Index.md` | Active | 总索引与阅读入口 |
| `01_Framework_Options.md` | Superseded | 大框架首轮选型记录（归档） |
| `02_Supernet_Training_Spec.md` | Reference | supernet 训练规格与约束定义 |
| `03_Supernet_Task_Breakdown.md` | Reference | 任务拆分与执行分解 |
| `04_Supernet_Execution_Gates.md` | Reference | 分阶段执行门禁与验收标准 |
| `05_Engineering_File_Management_and_Code_Style.md` | Reference | 工程组织、分层边界、代码风格 |
| `06_Implementation_Log.md` | Active | 关键实现记录与阶段日志 |
| `07_User_Manual.md` | Active | 训练与评估使用手册 |
| `08_Supernet_Retrospective.md` | Superseded | 历史复盘（归档精简版） |
| `09_Fold_Run_Diagnostics_and_CPU_Eval_Plan.md` | Superseded | 早期 fold 诊断计划（已精简归档） |
| `10_Eval_Consistency_and_Clip_Diagnostics_Plan.md` | Active | Eval 一致性与 clip 机制诊断执行 |
| `11_200Epoch_结果诊断与续训决策计划.md` | Active | 200 epoch 诊断与续训门槛决策 |
| `12_BestCkpt_子网扩展抽样分布分析计划.md` | Active | 最优 checkpoint 下的大样本子网分布分析 |
| `13_Vela_Bilinear基准_Supernet骨架一次性对齐自查计划.md` | Done | Vela 基准对齐与一次性自查 |
| `14_单模型与Supernet训练细节对比及数据利用率优化计划.md` | Active | 单模型与 supernet 训练链路对比优化 |
| `15_Project_Level_Improvement_and_Optimization_Plan.md` | Superseded | 项目级问题排查与修补方案（归档） |
| `16_单模型重训计划_Standalone_Retrain.md` | Active | 单模型重训执行计划 |

## 当前主线约束
1. 使用 folder-based FC2 数据加载（`train_dir` / `val_dir`）。
2. 保持 strict-fairness supernet 训练主流程。
3. 优先保证 eval 稳定性与日志可解释性。
4. CPU 并发评估已降级为非主线。
