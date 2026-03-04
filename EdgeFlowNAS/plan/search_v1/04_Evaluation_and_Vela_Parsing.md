# EdgeFlowNAS 评估指标拉取与解析机制 (04 Evaluation & Vela Parsing)

**更新时间**: 2026-03-04
**状态**: Active
**项目**: MCUFlowNet/EdgeFlowNAS

## 1. 核心目标
本规划文档规定了主调度引擎 `run_agentic_search.py` 如何与底层超网环境通信。引擎在多线程中对某一个子网 (如 `[0,1,2,0,0,1,2,1,0]`) 跑完测算后，需要自动化解析出包含 `EPE`, `FPS`, `SRAM` 等维度的干净字典。

---

## 2. 工具下沉与复用判定

回顾我们在上一步 `supernet_v1` 阶段写的底层脚本，目前你的系统中已经有了一个强大且成熟的评测引擎命令行入口：
**CLI 文件:** `wrappers/run_supernet_subnet_distribution.py`

我们 **无需** 重写这套底层的算力图、TensorFlow 会话或 Vela 编译链，而是直接通过 Python 自带的 `subprocess` 在子线程中唤起该 CLI，再解析它的执行产物。

### 为什么选择 CLI 而不是直接 `import` 调用函数？
1. **进程隔离 (Process Isolation)**：TensorFlow 1.x 的图 (Graph) 和会话 (Session) 清理极其困难，如果在主引进程中用多线程大量 `import` 生成并评估网络，100% 会引发 OOM (内存溢出) 和变量域污染。
2. **多核多线程天然兼容**：`subprocess` 把每个子网评估当成一个独立的系统进程执行，干净利索，跑完就释放显存和物理内存。

---

## 3. 标准化执行命令 (The Evaluation Command)

Coordinator 内部派发出的每一个 Worker 线程，应该组合并执行以下标准的命令行指令：

```bash
# 以评估架构 0,1,2,0,0,1,2,1,0 为例的伪命令
python wrappers/run_supernet_subnet_distribution.py \
    --config [YOUR_CONFIG] \
    --checkpoint_type best \
    --enable_vela \
    --vela_mode verbose \
    --vela_keep_artifacts \
    --output_tag agent_eval_012001210 # 给输出独立打标
# 注意：原代码 `run_supernet_subnet_distribution.py` 是测随机采样的池子。
# 所以我们需要对它极其微小的改造，让它支持 `--force_evaluate_single_arch 0,1,2,0,0,1,2,1,0` 并跳过随机采样。
```

---

## 4. 输出解析逻辑设计 (Artifacts Parsing)

一旦上面的子进程返回退出状态码 `0`，Worker 线程紧接着就要去指定目录下“收割”文件。

### 4.1 获取 EPE 得分
运行上述脚本后，系统通常会生成一份类似 `summary_agent_eval_012001210.csv` 的汇总表。
Python 脚本只需要用 `pandas` 读取它并拿到该行架构对应的那列 `epe`。

### 4.2 获取 Vela 核心硬件数据
读取在上述文件夹中同步生成的 `sram_test_modified_summary_Grove_Sys_Config.csv` 或者主表的合并列。
提取：
1. `inferences_per_second` (FPS)
2. `sram_memory_used`
3. `cycles_npu`

### 4.3 喂给 Agent C 的长文报表
读取原生的 `sram_test_modified_per-layer.csv`，将这个文件当中的长文本字符串提取出来，并限制长度如果太长则截断，喂给 Agent C 的 Prompt。

---

## 5. 对已有代码的一处“外科手术式”改造需求 (Action Required)

你当前的 `run_supernet_subnet_distribution.py` 以及核心下层模块 `code.nas.supernet_subnet_distribution` 是设计给**大批量随机抽样 (num_arch_samples)** 和**读取预设池子**使用的。

为了配合 Agent 的精确指定索求，我们需要对那个文件（或其平级逻辑）加上一个参数通道：
*   **任务点**: 需要让 CLI 支持类似 `--fixed_arch 0,1,2,0,0,1,2,1,0` 的传参。这会在之后的代码落地阶段快速打好补丁。

至此，评价管线设计闭环。

