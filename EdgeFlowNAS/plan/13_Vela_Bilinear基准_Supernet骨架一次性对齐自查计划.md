# 13 Vela Bilinear 基准 Supernet骨架一次性对齐自查计划
更新时间: 2026-02-19

## 0. 输入与边界
1. 目标: 以 bilinear 基线为参照，用 Vela 编译结果驱动 supernet 骨架结构一次性对齐。
2. 边界: 本计划聚焦“结构定义与导出一致性”，不涉及权重恢复与训练效果。
3. 环境: `conda activate tf_work_hpc`。
4. 评估口径: 固定输入 `1x180x240x6`，INT8 量化，`vela --optimise Size --verbose-performance --verbose-allocation`。

---

## 1. 历史问题（修复前）
修复前证据:
- `outputs/supernet/vela_structcheck/20260219_223226/summary.json`

修复前差异（supernet `0,0,0,0,0,1,0,0,0` vs bilinear）:
1. bilinear:
- `sram_peak_mb = 1.58203125`
- `inference_ms = 192.2270575`
- `fps = 5.202181279812807`
2. supernet:
- `sram_peak_mb = 5.361328125`
- `inference_ms = 3588.2749975`
- `fps = 0.2786854409700242`
3. 差值:
- `SRAM +3.779296875 MB`
- `inference +3396.04794 ms`
- `FPS -4.92349584`

结论: 差异主因是骨架定义不一致，不是权重。

---

## 2. Option A 执行结果（已完成）

### 2.1 代码改造
1. 对齐训练图骨架:
- `code/network/MultiScaleResNet_supernet.py`
2. 对齐导出图骨架:
- `code/nas/supernet_subnet_distribution.py` 中 `_FixedSubnetForExport`
3. 新增无权重对比入口:
- `code/nas/supernet_bilinear_vela_compare.py` 增加 `--skip_checkpoint`
- `wrappers/run_supernet_bilinear_vela_compare.py` 透传 `--skip_checkpoint`

### 2.2 对齐点
1. `E0/E1` 下采样语义对齐 bilinear:
- 从 stride `(1,1)` 改为 `(2,2)`。
2. head 路径改为 bilinear 同节奏通道衰减:
- `net_high(/4) -> out_1_4`
- `H1`: 上采样到 `/2` 并衰减通道
- `H2`: 再上采样到 `/1` 并再次衰减通道
3. 保持 9 维搜索空间不变:
- 4 个深度选择 + 5 个 head kernel 选择，未改编码长度。

### 2.3 无权重 Vela 回归（结构验证）
证据文件:
1. `outputs/supernet/edgeflownas_supernet_fc2_180x240/vela_compare/best_20260219_214935_optionA_structcheck/compare_summary.json`
2. `outputs/supernet/edgeflownas_supernet_fc2_180x240/vela_compare/best_20260219_215000_optionA_structcheck_111111111/compare_summary.json`
3. `outputs/supernet/edgeflownas_supernet_fc2_180x240/vela_compare/best_20260219_215015_optionA_structcheck_222222222/compare_summary.json`

结果摘要:
1. arch `0,0,0,0,0,1,0,0,0`
- `sram diff = 0.0 MB`
- `inference diff = 0.0 ms`
- `fps diff = 0.0`
2. arch `1,1,1,1,1,1,1,1,1`
- `sram diff = 0.0 MB`
- `inference ratio = 211.79 / 192.23 = 1.10`
3. arch `2,2,2,2,2,2,2,2,2`
- `sram diff = 0.0 MB`
- `inference ratio = 227.20 / 192.23 = 1.18`

验收阈值检查:
1. `SRAM diff <= 0.30 MB`: 通过。
2. `inference ratio <= 1.30`: 通过。
3. Top 峰值算子阶段与 bilinear 同级: 通过（核大小选择引入可控浮动）。

---

## 3. 风险与约束
1. 由于骨架已调整，旧 checkpoint 不保证可直接恢复。
2. 本次完成的是“结构等价”闭环，不是“历史权重兼容”闭环。
3. 若要继续旧权重，需要单独做变量映射或兼容构图。

---

## 4. 下一步建议
1. 新开一次短程训练（例如 10~20 epoch）验证训练曲线与 eval 稳定性。
2. 如必须续用旧 checkpoint，单开一条“兼容分支”，不要在当前结构对齐分支上混做。
3. 后续子网分布评估、Vela 分析都优先走 `--skip_checkpoint` 做结构基准回归。

---

## 5. 当前状态
1. P0: 完成。
2. P1: 完成。
3. P2: 完成。
4. P3: 完成。
5. P4: 完成（含证据与风险归档）。
