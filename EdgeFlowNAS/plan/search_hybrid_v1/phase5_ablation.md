# Phase 5: Ablation Execution Manual

**Status**: ready to submit (2026-05-05)
**Goal**: 量化每个 LLM 角色 (Phase 2 Warmstart / Phase 3 Scientist / Phase 4
Supervisor) 在已有 NSGA-II 上的增量贡献.

## 1. Ablation 设计 (4 组单跑, 不做多 seed)

| 组 | 配置 | 数据来源 |
|---|---|---|
| (a) | NSGA-II only | **复用** `outputs/nsga2_v3/nsga2_v3_distill_run1_20260429_201744/` |
| (b) | + Warmstart | 新跑 → `outputs/ablation_phase5/group_b_*/` |
| (c) | + Scientist | 新跑 → `outputs/ablation_phase5/group_c_*/` |
| (d) | + Supervisor (full system) | 新跑 → `outputs/ablation_phase5/group_d_*/` |

**HPC 跑量**: 3 个新 slurm × ~3h = ~9 小时墙钟 (4 P100 / 24 CPU / 160G 一组).

**单 seed (默认 2026)**: 用户决策 —— 多 seed 太贵; 红线判定如果差距小, 再考虑追加 seed.

## 2. 提交前 checklist

### 2.1 GEMINI_API_KEY 必须已配

```bash
# 在 ~/.bashrc 添加 (或 source 一个独立的 secret 文件):
export GEMINI_API_KEY="your-api-key-here"
```

每个 slurm 脚本会检查这个变量, 不存在直接退出.

### 2.2 Supernet checkpoint 路径

3 个 slurm 都用 V3 distill checkpoint:
```
outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill
```

确认它在 HPC `$VSC_DATA/test/MCUFlowNet/EdgeFlowNAS/outputs/supernet/...` 下存在.

### 2.3 LLM 配置

`configs/nsga2_v3.yaml` 已加 `llm:` 段:
- 5 个 role (warmstart_agent / scientist_stage_a/b1/b2 / supervisor_agent)
  全部路由到 `gemini/gemini-3.1-pro-preview`
- temperature 设为 1.0 (Gemini 3 推荐) / 0.6 (代码 stage B-1) / 0.7 (supervisor)
- max_tokens 8192 / timeout 180s / max_retries 3

如要换模型: 改 `configs/nsga2_v3.yaml` 的 `llm.models` map.

## 3. 提交命令

```bash
cd $VSC_DATA/test/MCUFlowNet/EdgeFlowNAS

# 顺序提交 3 组 (b → c → d), 每组 ~3h, 加起来 ~9h
sbatch plan/ablation_phase5/run_group_b.slurm
sbatch plan/ablation_phase5/run_group_c.slurm
sbatch plan/ablation_phase5/run_group_d.slurm
```

**4 P100 限制下只能一组一组跑** (每组 slurm 都 request `--gpus-per-node=4`).
slurm scheduler 会排队. 你可以一口气提 3 个, 它会自动顺序执行.

## 4. 监控

每组 slurm 跑起来后:
- 标准输出: `slurm-hybrid_v1_group_X-<jobid>.out` (在提交目录)
- NSGA-II 进度: `outputs/ablation_phase5/group_X_*/metadata/epoch_metrics.csv`
- Scientist 输出 (c/d): `outputs/ablation_phase5/group_X_*/metadata/insights.md`
  和 `insights.md.gen{N}.bak` 历史快照
- Supervisor 输出 (d): `outputs/ablation_phase5/group_X_*/metadata/supervisor_log.json`

如果某组中途失败 (LLM API 故障 / Vela 解析异常等), 用 `--resume` 续跑:
```bash
python wrappers/run_nsga2_search.py \
  --config configs/nsga2_v3.yaml \
  --output_root outputs/ablation_phase5 \
  --ablation_group d \
  --resume \
  --supernet_experiment_dir <path> \
  --gpu_devices 0,1,2,3 --max_workers 4 ...
```

`--resume` 时 warmstart 不会重新跑 (避免污染 gen 0).

## 5. 跑完后分析

```bash
python wrappers/run_phase5_ablation_analysis.py
```

(默认从 4 组的标准路径读 history_archive.csv, 输出到
`outputs/ablation_phase5/analysis_<timestamp>/` 下:)

- `summary.json`: 完整结构化摘要
- `hv_trajectory.png`: 4 组 HV vs 评估数对比折线
- `pareto_fronts.png`: 4 组最终 Pareto 前沿在 (FPS, EPE) 空间散点
- `report.md`: 人类可读的分析报告 + 红线判定

如果某组路径要单独覆盖 (比如 (a) 用别的目录):
```bash
python wrappers/run_phase5_ablation_analysis.py \
  --path_a outputs/nsga2_v3/<some_other_run>/metadata/history_archive.csv
```

## 6. 红线判定 (摘自 task_plan.md Success Criteria)

跑完后看 `report.md` 末尾的 verdict:

| 比较 | 期望 | 触发 FAIL 时建议 |
|---|---|---|
| HV(b) ≥ HV(a) | Phase 2 Warmstart 有正贡献 | 考虑移除 Phase 2 |
| HV(c) ≥ HV(b) | Phase 3 Scientist 有正贡献 | 考虑移除 Phase 3 |
| HV(d) ≥ HV(c) | Phase 4 Supervisor 有正贡献 | **强烈建议**移除 Phase 4 |

**论文叙事原则**: FAIL 是诚实数据, 不应硬撑. 写"我们尝试了 Phase X, 数据显示
无显著贡献, 已移除"比"勉强保留并附弱对比"可信得多.

**单 seed 警告**: 如果 FAIL 但差距小 (HV 相差 < 5%), 不能一棒打死; 建议追加
1-2 个 seed 看方差再判定.

## 7. 预期结果 (可信区间)

基于 search_v2_refactor_run2 实验 + CoLLM-NAS 论文经验:

- **(a) baseline HV** (V3 distill): 7.5217 (smoke-test 已确认)
- **(b) Warmstart**: 期望 +0~+5% (主要贡献在前 100-200 evals 区段, 800 evals
  点可能拉平)
- **(c) Scientist**: 期望 0~+5% (Scientist 的产出主要是 insights.md design
  rule corpus, 对 HV 直接增量较弱; 但 corpus 本身是论文产出)
- **(d) Supervisor**: 期望 -5%~+10% (高方差, 风险与收益并存; 这是 Phase 4
  设计文档明确写下"允许被消融出局"的角色)

如果 (d) 比 (c) 低 5%+: Phase 4 失败, 移除. 论文叙事变成"我们尝试动态超参
监督但发现固定参数 NSGA-II 已经足够".

## 8. Troubleshooting

### Slurm 超时 (5h 不够)

修改 slurm 顶部 `#SBATCH --time=5:00:00` 改成 `8:00:00` 或更长. 目前 5h
buffer 是基于"Vela 编译 + LLM 慢调用"经验估算; 真跑可能略超.

### LLM rate limit / API 错误

Gemini 3.1 Pro preview 有 rate limit. 如果出现 429, 调整
`configs/nsga2_v3.yaml` 的 `llm.max_retries=5` + `llm.request_timeout=240`.
Phase 4 只 5 次调用 / run, Scientist 也只 15 次, 总量不大, 限流概率不高.

### Warmstart 输出 < 50 个合法 arch

Phase 2 设计已自动 partial random fill; 看 `metadata/warmstart_diagnostics.json`
的 `unique_valid_count` 字段. 如果 < 30, 可能 prompt 有问题, 需要调整.

### Scientist sandbox 代码全 timeout

调高 `--scientist_sandbox_timeout 60` (默认 30s). Vela layer profile JSON 文件
比较大时 pandas 操作可能慢.

### Supervisor 一直 no_change

正常. supervisor prompt 明确说 "no_change 是合理输出". 看 supervisor_log.json
的 rationale 是不是合理 (search_v3 数据健康, 没有强信号需要调整).

## 9. 后续工作 (out of Phase 5 scope)

- 多 seed 重跑 (如果某组 verdict 临界)
- LLM 模型替换 ablation (Gemini 3.1 Pro vs DeepSeek V4 Pro)
- 单 lever 消融 (比如把 Supervisor 限制到只调 mutation_prob, 看动作空间贡献)
- 跑 V3 no-distill checkpoint 做 supernet ablation
