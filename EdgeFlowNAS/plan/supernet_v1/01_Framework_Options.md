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
