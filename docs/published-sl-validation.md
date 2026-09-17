# 已发布S/L权重的本机复测（2026-09-17）

实际运行代码：`765256ab7aa473530b0469830225692e9b415ee8`。
配置：`EdgeFlowNAS/configs/experiments/published_sl_sintel.json`。
实验：`20260917-sl-restore01`（各2张加载检查）、`20260917-sl-sintel01`（全量）。

| 模型 | 样本数 | 历史截断GT EPE | 未截断GT EPE | 权重元信息旧EPE |
| --- | --- | --- | --- | --- |
| S | 1041 | 5.5818726600 | 6.7856218940 | 5.5818572044 |
| L | 1041 | 4.8878811094 | 5.7700334218 | 4.8878574371 |

Sintel training Final，436×1024原图中心裁为416×1024；同一预测分别计分，无跳过或重复。输出按历史FT3D配置乘12.5还原像素单位。逐张断言新构造的裁剪与历史读取器一致；没有训练或调整权重。原始GT列仍是裁剪结果，不能称为完整图像标准评测，不能直接与未经协议核实的文献分数排序。

模型变量S为86个、L为136个，名称与形状匹配并恢复；不包含优化器恢复验证。TensorFlow2.17.0、CUDA12.8、tf-keras2.17.0、RTX5060Ti，关闭TF32。历史TF2.15.1环境并非完全复制；旧EPE接近复现而非逐位一致，未做重复测量。

全量运行命令（再次执行需使用新的输出目录）：

```powershell
./tools/setup/run-local.ps1 python tools/validation/evaluate_published_sl.py --config EdgeFlowNAS/configs/experiments/published_sl_sintel.json --output /runs/新的实验编号
```

本机完整日志位于`C:/00Work/Runs/MCUFlowNet/validation-logs/`，结果在`C:/00Work/Runs/MCUFlowNet/20260917-sl-sintel01/`。运行清单、镜像ID、逐图CSV、权重SHA256及汇总副本位于`C:/00Work/Lem_brain/wiki/项目/MCUFlowNet/附件/20260917-sl-sintel01/`，解释见同项目`主题-旧S与L权重的Sintel复测.md`。

原运行清单保留Linux Git的CRLF差异报告。补查Windows Git工作区干净；Linux `git -c core.autocrlf=true -c core.filemode=false status --porcelain`为空，`git diff --ignore-cr-at-eol --stat`为空，未发现内容差异。补充核查记录单独保存，不修改原清单。

本次结果用于ISCAS新训练准备，不修改已定稿硕士论文，也不设旧论文审查门槛。固定旧权重、改变计分标签不能判断取消训练截断的收益。下一步从随机初始化做真实数据短跑和持续监控，再以相同评测、预算及其他设置比较保留/取消训练标签截断；FT3D的12.5单位缩放与截断分开处理。EdgeFlowNet同条件补测仍待做。
