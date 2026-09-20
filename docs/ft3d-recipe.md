# FT3D起点与弱颜色增强对照

配置：`EdgeFlowNAS/configs/experiments/ft3d_recipe.json`。
S/L各三支：`100_plain`、`150_plain`、`100_color`。前两支比较FC2切换时机；第一与第三支比较颜色增强。150轮多用50轮FC2，不能称为总计算量相同。没有同时测试150轮加增强，不能据此推断二者组合收益。

所有分支接FC2末尾权重，保留模型/BN，新建Adam和FT3D随机序列。FT3D clean+final、left、前后向，batch32、352×480、seed42、prefetch1、20000更新、LR从1e-5余弦降至1e-6。每500步保存完整恢复点，每1000步76对快检，每5000步845对共同监控；196对留出不使用。光流保持像素单位、训练分量±50、验证原值，Sintel同时保留原值和历史GT±50两列。

颜色增强先执行与无增强相同的随机裁剪，再用独立的样本内随机数副本，以50%概率对两帧应用相同亮度/对比度系数，各在[0.9,1.1]。对比度围绕每帧各通道的裁剪后均值变化，输出限制0–255；不改变标签，不调用几何变化或遮挡。读取线程、预取、暂停恢复不改变样本/裁剪随机数。完整配置保存在每次作业记录中。

在GPU分配内，通过已有Sofia容器包装器运行：

```bash
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/ft3d_recipe.json --variant s_100_color --probe --stop-step 50
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/ft3d_recipe.json --variant s_100_color --probe --action resume
```

工程验收输出单独位于`FT3D-RECIPE-01-PROBE`，不作为正式起点。正式运行去掉`--probe --stop-step 50`，一次授权范围为20000步。失败只从已发布完整边界恢复，配方变更必须开新实验。输出、权重、日志在Runs；Git只存代码和配方。六支启动前核验各自初始化张量和50+50步恢复，正式训练使用全新目录。已有DATA-ROUTE-01结果作为历史参照，新同期对照减少代码版本差异影响。

评价先比较10000/15000/20000步的845对原值EPE。平均降低至少0.05、且其中至少两次占优，作为值得换种子复验的筛选线；这不是统计显著性结论。单个最低点、工程短跑成绩都不作为配方成功证据。不得自动追加训练预算。
