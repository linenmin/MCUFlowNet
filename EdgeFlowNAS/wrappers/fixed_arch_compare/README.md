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
