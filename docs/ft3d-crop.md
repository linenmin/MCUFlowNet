# FT3D裁剪大小对照

`FT3D-CROP-01`从FT3D-RECIPE-01的S/L 150轮起点分支末尾（FT3D累计20000步）完整恢复模型、BN、Adam和数据随机状态。每个模型各两支：训练裁剪352×480或512×896；各追加10000步，固定LR 1e-6，batch/microbatch32、seed42、prefetch1，无额外增强。保持父实验数据、光流单位与训练±50规则。

内部FT3D验证始终352×480；较大训练图使用共享原有变量的独立推理图验证，不新建参数或Adam，不更新BN。Sintel仍416×1024、共同845对，原始GT/历史两列并存，196对留出不用。每500步记录，每2500步完整监控。比较25000/27500/30000步均值，改善≥0.05且至少2/3节点占优才进入独立重复；这不是显著性检验。大裁剪每步像素约为原来的2.72倍，并非等计算量对照。

配方：`EdgeFlowNAS/configs/experiments/ft3d_crop.json`。运行器只在GPU allocation内执行，不自行提交或取消作业：

```bash
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/ft3d_crop.json --variant s_large --probe
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/ft3d_crop.json --variant s_large --probe --action continue --stop-step 21000
```

四支各做500+500步验收，保留父实验的500步计数单位；probe仍执行原验证及末尾845对监控。独立输出`FT3D-CROP-01-PROBE`，不作为正式起点。先在allocation内用`audit_crop_geometry.py`从原FT3D配方抽取128个不同TRAIN光流，统计端点本在原图内、因裁剪落到裁剪区外的比例；使用未截断标签。这是几何统计，不等于真实遮挡，不能代替训练结果。

`check_crop_probes.py --recipe ... --output /runs/FT3D-CROP-01/submission/acceptance.json`核对四支尺寸、父权重、完整状态恢复、累计步数、监控和启动记录，通过后正式运行去掉`--probe`，到30000步自动停。正式目录必须为空，每支从原父权重重开，不接probe。失败可用`--action resume`从最近完整边界恢复，禁止悄悄改配方。

输出、Slurm控制脚本、作业ID与验收JSON存Runs/control；代码与固定配方进Git，Sofia运行固定提交的只读release。其他实验和队列不参与本轮操作。
