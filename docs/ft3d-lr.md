# FT3D-LR-01：延长训练，还是适度提高学习率？

从DATA-ROUTE-01各模型FT3D第20000步末尾分出两支，恢复完整权重、BN、Adam和数据RNG；父目录只读，分支输出在`/runs/FT3D-LR-01/{s,l}_{low,rewarm}`。同模型两支的唯一区别是学习率安排。

| 分支 | 学习率 | 首次停止位置 |
| --- | --- | --- |
| low | 固定1e-6 | 累计30000步，即追加10000步 |
| rewarm | 追加前500步从1e-6升到3e-6，随后余弦降至1e-6 | 同上 |

学习率完整周期为追加20000步，不能因中期停止而缩短。每500步一个记录区间，累计epoch从41开始；配置总epoch=80，首次stop_after_epoch=60。FT3D数据、batch32、352×480裁剪、无额外增强、标签单位、seed42、prefetch1及评测清单保持不变。每1000步快检，每5000步845对完整监控，196对留出不用。

配置位于`EdgeFlowNAS/configs/experiments/ft3d_lr.json`，包括父运行路径和四个分支的参数；不依赖旧Slurm作业号。入口从父运行的run_manifest读取实际配置，并核对模型、状态边界和协议。恢复完整Adam/BN张量，保存源权重指纹；每次完成检查连续步数、完整监控次数和有限loss/EPE。配对分支首批输入指纹应一致。分支最佳值重新记录，父最佳权重仍留在父目录。

中期比较相同追加步数的完整监控、最佳可恢复权重和异常情况。不因一次波动淘汰；后半段必须得到审批再执行continue。下列命令运行于GPU作业内，不会自行申请资源。

```bash
# 首次分叉：到累计30000步停止
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/ft3d_lr.json --variant s_low --action start
# 中断恢复：回到本分支最后完整保存点，目标保持此前已记录的停止位置
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/ft3d_lr.json --variant s_low --action resume
# 中期批准后：同一分支继续到40000步，不重置Adam或学习率周期
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/ft3d_lr.json --variant s_low --action continue --stop-step 40000
```

其他variant为s_rewarm/l_low/l_rewarm。run_lr_stage.py保留为很薄的旧命令兼容入口；同样支持action。每次操作生成新的job manifest和配置，已有目录不能用start覆盖。Linux/WSL使用运行锁，避免两个新入口同时写同一分支；旧版本作业不持有此锁，提交恢复前仍必须检查Slurm确认旧作业退出。

现有71289ba运行可直接被新入口读取，不需要移动目录或修改其配置。若旧权重和状态不一致则明确报错，不能通过手改轮次绕过。新代码的完整保存与恢复机制见experiment-workflow.md。
