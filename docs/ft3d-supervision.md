# FT3D-SUPERVISION-01：真实位移与缩放对照

从当前 FC2 clip50 第150轮末尾分别初始化 S/L，保留模型和 BN，重置 Adam。
每个模型五种设置：raw、raw_scale、raw_mask400、raw_mask400_scale、raw_l1。
配置见 `EdgeFlowNAS/configs/experiments/ft3d_supervision.json`。

共同设置沿用 FT3D-SCHEDULE-01 higher：60000步、batch32、seed42、352×480、
clean+final、双方向、prefetch1、余弦3e-5→1e-6。训练和评估均为原始像素单位；
新实验训练标签不截断。既有 clip50 六万步是参照，不重复训练。

- scale：一半样本原尺寸，一半统一缩放0.70–1.00，再随机裁剪。
  图像与标签一起缩放，光流按舍入后的实际横纵比例换算；独立几何随机数不改变样本顺序。
- mask400：训练中排除模长≥400的像素，阈值在缩放后判断。
  所有三尺度 L1、不确定性重建及正则项使用同一有效范围，按有效像素×通道数归一化。
  粗尺度标签插值涉及排除像素时，该位置也不计分，避免异常值混入正常标签。
  这不是 RAFT 原代码的全网格平均。评估仍使用全部原始 GT。
- l1：保留相同网络和四通道输出，只将额外不确定性项权重设为0；
  光流仍是三个尺度累积预测的 L1，权重0.125、0.25、0.5。
- 有效监督比例、梯度、双口径 Sintel 和完整恢复信息随每个报告点保存。

执行顺序：本机数值测试 → Sofia 标签抽样/样本配对检查、既有60k权重独立复验 →
十个配置各50步停止并恢复到100步 → 短跑通过后启动十条正式训练。
短跑使用100步余弦，仅验证工程运行；正式阶段始终采用完整60000步余弦。
正式20k/40k/60k保存恢复快照，监控845对，196对留出集不参与选择。

沿用 `tools/hpc/run_retrain_experiment.py` 的 `--recipe`、`--variant`、`--probe`、
`--action start/resume` 和 `--stop-step`。短跑与正式输出分别在
`/runs/FT3D-SUPERVISION-01-PROBE` 和 `/runs/FT3D-SUPERVISION-01`，不写入Git。
运行状态和结论在 Obsidian 原实验总表维护，不在这里复制实时状态。
