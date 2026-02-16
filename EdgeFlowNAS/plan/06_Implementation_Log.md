# 06 实施日志

更新时间: 2026-02-16

## Part-01（E01 + E02 + Gate-1 基础入口）

### 本部分目标

1. 建立工程级目录分层骨架。
2. 完成 supernet 训练入口 `wrapper -> app -> engine` 最小闭环。
3. 完成 `arch_codec` 自测能力。
4. 固化配置文件与规则文件。

### 代码改动清单

1. 新增目录：`wrappers/`, `configs/`, `code/*`, `tests/*`, `outputs/supernet`, `scripts/hpc`。
2. 新增入口：`wrappers/run_supernet_train.py`。
3. 新增编排：`code/app/train_supernet_app.py`。
4. 新增执行层骨架：`code/engine/supernet_trainer.py` 与相关子模块。
5. 新增 NAS 核心：`code/nas/arch_codec.py`、`code/nas/fair_sampler.py`。
6. 新增工具：`code/utils/*.py`（日志、路径、json、seed、manifest、检查脚本）。
7. 新增配置：`configs/supernet_fc2_180x240.yaml`、`configs/layering_rules.yaml`。

### 验收命令

1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.nas.arch_codec --self_test`

### 结果记录

1. 状态：已完成并通过自检。
2. 命令结果：
- `python wrappers/run_supernet_train.py --help` 通过。
- `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run` 通过。
- `python -m code.nas.arch_codec --self_test` 通过。
- `python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 5` 通过。
- `python -m code.nas.fair_sampler --cycles 300 --seed 42 --check` 通过。
- `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8` 通过。
- `python -m code.utils.check_comments --root code --strict` 通过。
- `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml` 通过。
3. 备注：当前训练环节为可运行占位骨架，下一部分会逐步替换为真实训练逻辑。

## Part-02（E04 增强 + 训练公平计数实装）

### 本部分目标

1. 将验证池从随机占位升级为分层覆盖构建。
2. 将公平计数从静态占位升级为按公平周期真实累计。
3. 让训练报告与评估产物包含覆盖状态与公平统计。

### 代码改动清单

1. 更新 `code/nas/eval_pool_builder.py`：
- 增加基础覆盖编码策略。
- 增加 `check_eval_pool_coverage` 覆盖检查函数。
- 增加可直接命令行检查的 `--check` 入口。
2. 更新 `code/nas/supernet_eval.py`：
- 评估输出写入覆盖检查结果。
3. 更新 `code/engine/supernet_trainer.py`：
- 接入 `generate_fair_cycle` 周期采样并累积公平计数。
- 接入分层验证池构建与覆盖检查写盘。
- 在训练报告里记录 `final_fairness_gap` 与 `eval_pool_coverage_ok`。
4. 更新 `code/engine/train_step.py`：
- 接收 `cycle_codes` 输入，显式表达 3-path 周期语义。

### 验收命令

1. `python -m code.utils.check_comments --root code --strict`
2. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
3. `python -m code.nas.eval_pool_builder --seed 42 --size 12 --check`
4. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 3 --fast_mode`
5. `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8`

### 结果记录

1. 状态：已完成并通过自检。
2. 覆盖检查：`coverage.ok = true`。
3. 公平计数：训练短跑中 `fairness_gap=0`。
