# Task Plan: Multi-Agent NAS (search_v1) Implementation & HPC Deployment

## Goal
构建并运行基于 Multi-Agent (LLM) 的神经架构搜索系统，在已训练好的 Supernet 上寻找针对 MCU (Ethos-U NPU) 优化的最优子网架构。

## Current Phase
Phase 2: HPC Environment Setup & Verification

## Phases

### Phase 1: Local Framework Build-out
- [x] 定义 4 个 Agent 的系统提示词 (Strategist, Generator, HW Distiller, Scientist)
- [x] 实现协调器 (Coordinator) 与文件 I/O (file_io, llm_client)
- [x] 实现本地子进程评估、Vela 解析与 Agent C 蒸馏逻辑
- [x] 验证本地生命周期（干跑测试、断点恢复）
- [x] 提交代码仓库并推送日志
- **Status:** complete

### Phase 2: HPC Environment Setup & Verification
- [ ] 在 HPC 环境执行 `pip install litellm`
- [ ] 配置 `GEMINI_API_KEY` 环境变量
- [ ] 运行 `python tests/test_search_dryrun.py` 验证环境与密钥
- **Status:** pending

### Phase 3: Real Search Execution (Epoch 0-N)
- [ ] 执行 `python wrappers/run_agentic_search.py --config configs/search_v1.yaml`
- [ ] 监控 `history_archive.csv` 与 `findings.md` 的增量更新
- [ ] 处理 TF1 内存泄漏或进程挂起等潜在 HPC 异常
- **Status:** pending

### Phase 4: Results Analysis & Scientific Refinement
- [ ] 分析最终 Pareto 前沿 (EPE vs Cycles/SRAM)
- [ ] 验证 Agent D 提出的硬约束 (Findings) 是否在真实测试集中成立
- **Status:** pending

## Key Questions
1. HPC 上的 TensorFlow 1.x 版本的子进程调用是否稳定？ (需通过 Dry-run 验证)
2. LiteLLM 是否能成功穿透 HPC 的网络防火墙调用 Gemini API？ (可能需要代理设置)

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Map-Reduce 并发 (C2) | TF1 无法高效清理 Session，物理进程隔离 (subprocess) 是避免内存崩溃的唯一可靠方法 |
| B1 混合持久化 | 机器可读 (JSON) 保证流程，人机可读 (Findings.md) 保证科学可解释性 |
| 独立 Scientist (Agent D) 会话 | 将“提出猜想”、“编写验证脚本”和“规则合并”解耦，提高长链路逻辑的稳定性 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Import "litellm" could not be resolved | 1 | 本地开发环境未安装，但在 tf_work_hpc 分析中确认可安装，HPC 侧需手动 pip install |
| TF1 Graph Cleanup | 1 | 放弃在 Python 内部清理，全面转向 subprocess 物理进程重置 |
