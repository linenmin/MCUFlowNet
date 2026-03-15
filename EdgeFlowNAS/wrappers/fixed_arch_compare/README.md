# Fixed-Arch Compare Wrappers

这个目录只放固定骨架对比训练入口，不和通用 NAS / standalone wrapper 混在一起。

当前目标用例：

- 固定骨架：`0,2,1,1,0,0,0,0,0`
- 默认训练输入：`172x224`
- 默认配置：`configs/fixed_arch_compare_fc2_172x224.yaml`
- 多模型联合训练：
  - `baseline`
  - `globalgate4x_bneckeca`
  - `globalgate4x_bneckeca_skip8x4x2x`
- 六模型冲榜配置：
  - `configs/fixed_arch_compare_fc2_172x224_leaderboard6.yaml`
  - `globalgate4x_bneckeca_skip8x4x`
  - `globalgate8x4x_bneckeca`
  - `globalgate8x4x_bneckeca_skip8x`
  - `globalgate4x_dual_eca8_bneckeca`
  - `globalgate8x4x_bneckeca_skip8x4x`
  - `skip8x4x`

## Training

六模型联合训练：

```bash
python wrappers/fixed_arch_compare/run_train.py \
  --config configs/fixed_arch_compare_fc2_172x224_leaderboard6.yaml \
  --gpu_device 0 \
  --experiment_name fixed_arch_compare_fc2_172x224_leaderboard6_v1
```

单模型训练：

```bash
python wrappers/fixed_arch_compare/run_train.py \
  --config configs/fixed_arch_compare_fc2_172x224_leaderboard6.yaml \
  --model_variants globalgate8x4x_bneckeca \
  --model_names candidate \
  --gpu_device 0 \
  --experiment_name fixed_arch_compare_fc2_172x224_gate8x4x_single_v1
```

## Sintel Evaluation

训练后可以直接评估整个 fixed-arch compare experiment 的 `best` 或 `last` checkpoint：

```bash
python wrappers/fixed_arch_compare/run_sintel_test.py \
  --experiment_dir outputs/fixed_arch_compare/fixed_arch_compare_fc2_172x224_v1 \
  --ckpt_name best \
  --dataset_root ../Datasets/sintel
```

也可以单独评估某一个模型目录：

```bash
python wrappers/fixed_arch_compare/run_sintel_test.py \
  --model_dir outputs/fixed_arch_compare/fixed_arch_compare_fc2_172x224_v1/model_full \
  --ckpt_name best \
  --dataset_root ../Datasets/sintel
```

输出会写到 experiment 目录下的：

- `sintel_eval_best.json`
- `sintel_eval_best.csv`
