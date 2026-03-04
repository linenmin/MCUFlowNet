# 实验日志与状态落盘规范 (05 Experiment & Logging Rules)

**更新时间**: 2026-03-04
**状态**: Active
**项目**: MCUFlowNet/EdgeFlowNAS

## 1. 核心目标
为了实现绝对意义上的**无锁、防断电、Stateless 断点续传**，本规划详细拟定了 `search_v1` 运行期间所产生的临时文件、档案文件和知识库文件的目录层级结构及维护规范。

---

## 2. 目录结构设计 (Folders & Files)

当主控制脚本 `run_agentic_search.py` （或同等启动脚本）运行时，它会在工程下创建并强依赖以下目录拓扑：

```text
MCUFlowNet/EdgeFlowNAS/
  outputs/
    search_v1/                                # Search 结果的统一根目录
      [experiment_name]_YYYYMMDD_HHMMSS/      # 单次大搜索实验根目录
      
        # -- 【第一象限：全局永续记忆库 (Global Memory)】 --
        metadata/
          history_archive.csv                 # 【核心事实记录】每行一个子网 (Arch, EPE, FPS, micro_insight...)
          search_strategy_log.md              # 【战术日记】Agent A 的长期宏观战略思维演进记录
          assumptions.json                    # 【科学猜想簿】待验证的 Agent D 提出的各种规律
          findings.md                         # 【宇宙真理碑】固化的，绝不能被违逆的硬条件

        # -- 【第二象限：多线程分布式缓冲 (Map-Reduce Temp)】 --
        dashboard/
          tmp_workers/                        # worker_n 完成评估后吐出的小 JSON 文件
            arch_012001210.json
            arch_022011210.json
            ...
          eval_outputs/                       # subprocess 调用原 supernet cli 时隔离用的输出产物
            run_012001210/                    # Vela CSVs、TFLite logs... 一旦提炼完可择机删除

        # -- 【第三象限：代码生成验证区 (Dynamic Scripting)】 --
        scripts/                              # Agent D 写出来的 Python 校验代码统一存放处
          eval_assumption_A01.py
          ...
```

---

## 3. 文件读写守则 (Guardrails)

### 3.1 Map-Reduce 无锁聚合机制
1. **多线程写入 (Map)**：所有的 4 个线程，只被允许向 `tmp_workers/` 文件夹下方写入以当前 `arch_code` 命名的私有小快照，不允许碰任何公共的 CSV 或 MDC。
2. **主线程聚合 (Reduce)**：只有主线程的统管大循环，才被允许使用 `pd.concat` 或者 `.write` 统一地将 `tmp_workers/` 里的 JSON 整合进 `history_archive.csv`。落表后，立刻清空 `tmp_workers/`。
3. **断电重启 (Rescue)**：任何时候重启 `run_agentic_search.py`，程序必须首先检查 `tmp_workers/` 是否非空。如果有，先执行一次 Reduce 抢救性合并，再唤醒模型接着干活。

### 3.2 档案去重过滤
在唤醒 Agent B (Generator) 要求它产出比方说 40 个子网配置时。由于它是随机生成且并不知道过往，它大概率会生成之前已经测过的。
*   主引擎在拿到这 40 个列表后，必须读取 `history_archive.csv` 对 `arch_code` 列进行强校验。如果发现历史中已有某架构，**直接使用已有成绩并在心中打勾**，绝不要去拉起 Vela 或 Supernet 做重复无用功，白白浪费计算资源。

---

## 4. 后续落地节点
至此，针对 Search 阶段规划的 01~05 号核心档案蓝图已经全部封存结束。
下一个实质性阶段动作将是：**根据《04_Evaluation规划》与《05_Logging规划》，用 Python 撰写并运行 `run_agentic_search.py`。**
