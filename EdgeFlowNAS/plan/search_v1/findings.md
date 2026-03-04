# Findings & Decisions - EdgeFlowNAS search_v1

## Requirements
- **搜索空间**：9 维 `arch_code` (3^9 = 19,683 种组合)，无通道搜索。
- **语义分布**：0-3 位控制深度 (Backbone)，4-8 位控制卷积核大小 (Head)。
- **目标硬件**：ARM Ethos-U55 NPU (Vela 编译器)。
- **核心指标**：EPE (光流误差) + SRAM/Cycles (Vela summary 导出)。
- **核心逻辑**：基于 Agent D 科学猜想驱动的假设验证搜索。

## Research & Environment Findings
- **Vela 输出分析**：Vela 编译器生成的 `per-layer.csv` 包含逐算子的 Cycle 数和溢出状况，Agent C 可据此判断子网是“计算密集”还是“访存密集”。
- **tf_work_hpc 环境**：
  - Python 3.11.14 / TF 2.15.1
  - 核心缺失库：`litellm` (LiteLLM 是框架调用多模型 Gemini 的中间件)。
- **Supernet 兼容性**：底层 `supernet_subnet_distribution.py` 已经完成补丁，支持 `--fixed_arch` 架构锁。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| LiteLLM 路由 | A/D 用 Pro 模型增强逻辑，B/C 用 Flash 模型保证并行速度和成本平衡 |
| 无状态 Agent 设计 | Agent 每次交互后上下文清空，全局事实表 (`history_archive.csv`) 是唯一的真相源 |
| Findings 增量追加 | Agent D 仅在发现逻辑矛盾时重写 Findings.md，平时通过“追记”模式保持历史可追溯性 |

## Resources
- 搜索总体规划索引：`plan/search_v1/00_Search_Plan_Index.md`
- 搜索空间语义定义：`plan/search_v1/01_Search_Environment_and_Space_Semantic.md`
- 系统提示词定义包：`plan/search_v1/02_Agent_Prompts_Definition.md`
- 本次搜索主配置文件：`configs/search_v1.yaml`

## Visual/Browser Findings
- 参考 **CoLLM-NAS** 论文，发现 LLM 对“访存与频率关系”的直觉往往能加速硬件感知 NAS。
- **Vela Summary** 格式已解析：核心关注 `Summary` 表中的 `Total cycles` 和 `SRAM used`。
