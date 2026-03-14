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

## Sintel Evaluation

训练后可以直接评估整个 fixed-arch compare experiment 的 `best` 或 `last` checkpoint：

```bash
python wrappers/fixed_arch_compare/run_sintel_test.py \
  --experiment_dir outputs/fixed_arch_compare/fixed_arch_compare_fc2_172x224_v1 \
  --ckpt_name best \
  --dataset_root ../Datasets/Sintel
```

也可以单独评估某一个模型目录：

```bash
python wrappers/fixed_arch_compare/run_sintel_test.py \
  --model_dir outputs/fixed_arch_compare/fixed_arch_compare_fc2_172x224_v1/model_full \
  --ckpt_name best \
  --dataset_root ../Datasets/Sintel
```

输出会写到 experiment 目录下的：

- `sintel_eval_best.json`
- `sintel_eval_best.csv`
