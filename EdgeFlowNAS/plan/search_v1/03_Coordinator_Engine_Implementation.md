# EdgeFlowNAS 协调引擎实现方案 (Coordinator Engine)

**更新时间**: 2026-03-04
**状态**: Active
**项目**: MCUFlowNet/EdgeFlowNAS

## 1. 核心目标
本规划定义了主控程序 `run_agentic_search.py` 的工程架构。该引擎是整个 Multi-Agent 系统的唯一物理驱动中枢。它不仅负责多模态 LLM (A,B,C,D) 的状态机时序调度，还通过多线程并行化本地 `Supernet + Vela` 的仿真评估，实现“无极并发，断点续传”。

---

## 2. API 网关选型：A2 (LiteLLM)
基于高可用性和并发弹性考虑，我们在 LLM 调用底层放弃原生 SDK 裸写，转而采用 **LiteLLM** 统一网关库。

**收益与约定：**
1. **自动容灾与并发流控**：使用 `litellm.completion` 代替原生 requests。它原生处理因并发过高导致的 `ResourceExhausted (429)` 错误（自带挂起与退避重试）。
2. **多模型无缝切换**：只需在函数内变更 `model="gemini/gemini-1.5-pro"` 或 `model="gemini/gemini-1.5-flash"`。
3. **JSON 强约束支持**：通过给 liteLLM 传参 `response_format={ "type": "json_object" }`，确保解析极度稳健，防范 Agent 的 Markdown 碎嘴。

---

## 3. 文件系统材质选型：B1 (混合持久化)
系统的灵魂资产依循其应用场景分布：
1. **`history_archive.csv`**: 全局唯一事实表（包含 `arch_code, epe, fps, sram, macs, micro_insight`）。
2. **`assumptions.json`**: Agent D 生成的机器可读队列。存有未被完全证实的科研假说及置信度计算规则。纯 JSON 格式有助于 Python 取用和删除。
3. **`findings.md`**: 全局绝对纪律。由人与 Agent 共同维护的自然语言红线。纯 Markdown 格式，因为是要灌入 Agent B/A system prompt 的真理经文。

---

## 4. 并发架构设计：C2 (Map-Reduce 无文件锁归并模式)
鉴于本地算力限制（如 4 线程物理瓶颈）与稳定持久化诉求，Vela 和硬件蒸馏的并发被切割为非常干练的分治收集架构，彻底抹除了文件写入死锁（Race Condition）。

### 4.1 执行流与并发粒度 (Max Workers = 4)
当 Agent B 抛出本次需要判别的 40 个合法 `arch_code` 后：
1. **去重拦截**：Coordinator 瞬间在主线程扫过 `history_archive.csv`，其中 5 个架构因为早前搜过直接截断，剩下 35 个全新配置。
2. **Map 阶段（多线程下发）**：
   采用 `concurrent.futures.ThreadPoolExecutor(max_workers=4)` 发配任务。
   此时本地最多有 4 个 Python 线程同时在做以下事情（对某一个 `arch_code`）：
   - `subprocess.run(vela_eval)`。
   - 读取该操作引发的 `per-layer.csv`。
   - 将日志打到 LiteLLM (Agent C - Flash) 取回一句 `micro_insight`。
   - 【关键】**不碰全局 CSV！** 而是将结果字典存入一个由 `arch_code` 命名的独立小快照文件，例如 `dashboard/tmp/eval_012001210.json`。
3. **Reduce 阶段（单线程主轴拾取）**：
   线程池汇合 (`executor.map` 彻底执行完毕或因故中断退出)。主线程单骑出马，利用 `glob.glob` 捕获 `tmp/*.json` 里的全部 35 个安全小件。合并后一条 `append` 命令优雅落表于 `history_archive.csv`，接着清空 `tmp` 堆。

> **容灾能力(Fault Tolerance)**: 如果程序跑到第 18 个时你电脑断电重启，`history_archive.csv` 没变化。但 `tmp/` 里躺着极具价值的 18 个 JSON。引擎重启时第一个动作就叫 `Rescue_Map_Reduce`：扫描是否有残存的临时JSON立刻入表拯救资产。不浪费任何一分算力。

---

## 5. Main Loop 状态机伪代码 (生命周期)

整个核心脚本 `run_agentic_search.py` 的轮询主轴被设计为一个永不宕机的 `while` 循环体。

```python
# 主循环伪代码
def run_autonomous_nas_search(total_epochs=50, batch_size=40):
    for epoch in range(total_epochs):
        logger.info(f"=== Starting Epoch {epoch} ===")

        # 1. 触发大反思 (每 10 个 Epoch) -> Agent D
        if epoch % 10 == 0 and epoch != 0:
            execute_scientist_macro_reflection_loop() 
            # 内部包括: 
            # 1. 查阅 CSV -> 吐出 assumptions.json (Session D-1)
            # 2. 从 assumptions 生成 eval_xxx.py 验证脚本 (Session D-2)

        # 2. 验证已有猜想 (Engine)
        evaluate_pending_assumptions_and_promote() 
        # Python 子进程执行 eval_xxx.py
        # 如果 return confidence > 0.95 -> 触发 Session D-3 写 Findings.md

        # 3. 规划战术 -> Agent A
        strategy, allocation = invoke_strategist_agent_a()

        # 4. 生成子网 -> Agent B
        pending_arch_codes = invoke_generator_agent_b(strategy, allocation, batch_size)

        # 5. 过滤与验证 (Engine Multi-Thread + Agent C)
        new_archs = filter_duplicate_archs(pending_arch_codes, "history_archive.csv")
        
        # 4 个物理线程执行 Map
        with ThreadPoolExecutor(max_workers=4) as executor:
             executor.map(worker_evaluate_and_distill, new_archs)
        
        # 6. 后处理 Reduce
        commit_tmp_results_to_global_csv()
        
        logger.info(f"=== Epoch {epoch} Finished ===")
```

## 6. 后续任务清算
至此，设计层面的逻辑推演已尽数封闭化。此文档作为实际编码规范锁库。
下一步（脱离计划与架构设计）：正式开始准备真正的框架目录和预植入文件！
