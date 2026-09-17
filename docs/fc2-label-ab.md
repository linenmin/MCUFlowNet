# FC2 训练标签对照：准备与运行

这轮只检查 S/L 从头训练时，保留或取消 ±50 标签截断的影响。FT3D 的单位缩放不变，暂不开展 FT3D 实验。正式训练尚未启动。

## 四组设置

配置位于 `EdgeFlowNAS/configs/experiments/label_ab/`：`s_clip50.json`、`s_raw.json`、`l_clip50.json`、`l_raw.json`。同一模型两组仅实验编号和 `fc2_train_label_clip` 不同；50 表示分量截断，null 表示保留原值。`fc2_eval_label_clip` 全部为 null，所以 FC2 验证成绩可以直接比较。未写这些字段的旧配置仍默认 ±50。

第一版正式配置：种子42、352×480随机裁剪、逻辑batch32、microbatch8、Adam、学习率1e-4到1e-6余弦下降、50轮、梯度范数上限200、无额外增广或权重衰减。它是后续HPC试跑的起点，不代表最优方案。BN实际看到的是microbatch8；不能只保证逻辑batch相同。正式配置的batch尚未在HPC验收。

四个独立单卡任务使用同一软件环境。单个程序没有多卡并行能力。先测读取和显存再提交长作业。变更batch或训练周期需成组调整配置；续跑不能偷偷改变对照条件。

## 怎样监控

`prepare_label_ab.py`只读取官方列表中的路径，不读取标签或模型成绩；按场景家族名称的固定SHA256排序留出约四分之一家族，同一家族的不同序列不拆开。

| 清单 | 数量 | 使用方式 |
| --- | --- | --- |
| monitor_quick.txt | 76对，19个场景各4对 | 每轮检查趋势，单独保存sintel_best |
| monitor_all.txt | 845对，19个场景 | 每5轮及最后一轮检查，保存sintel_monitor_best；正式候选优先依据这一列选择 |
| holdout.txt | 196对，sleeping与temple家族 | 不接入训练程序；第一轮方案确定后再检查 |

两个监控集合分别计算原始GT和历史截断GT的EPE，同一份预测，416×1024中心裁剪，FC2输出乘数1。它们不是完整图像官方测试成绩。76对是固定、跨场景的小样本，不能保证代表整个Sintel。

此前旧权重已在全部1041对上复测；留出集合只能用于此后方案的内部检查，不是从未接触过的独立测试集。完整Sintel也不能被宣称为完全未参与调参的数据。此次划分不是官方划分。清单及SHA256已提交，修改后应作为新的实验版本。

FC2每轮验证全部640对。不同训练目标的训练loss不能直接排名；先比较统一验证EPE。不要因为一次监控波动立即淘汰分支。第一轮应至少观察多个完整监控点，再决定继续、重复种子或提出新修改。

## 本机检查与恢复

`smoke_*.json`专供工程检查：batch2，两个各50步的短段，中间退出进程并恢复；它们不是完整FC2轮次。四组总计400步。每段监控76对，第二段额外检查845对，不读取留出集合。运行入口：

```powershell
.\tools\setup\run-local.ps1 python tools/validation/test_label_ab.py
.\tools\setup\run-local.ps1 python tools/validation/test_retrain_validation.py
.\tools\setup\run-local.ps1 python tools/validation/run_label_ab_smoke.py
```

运行摘要写入 `C:/00Work/Runs/MCUFlowNet/20260917-label-prep-summary/`，四组权重和历史分别存入 `20260917-label-prep-{s,l}-{clip50,raw}`。存在的实验目录不会覆盖。

同模型两组记录初始化参数摘要、每段第一批输入摘要；专项测试另外检查连续批次抽样和裁剪一致。摘要相同不是跨硬件逐位可复现承诺。恢复会核对参数/BN/优化器，保护标签设置、训练周期、batch和监控清单；当前只支持这组配置的轮次边界恢复。中间突然被杀的未完成轮次需要重跑，不支持逐步精确恢复。

`last`用于继续训练；`best`用于FC2验证最佳；`sintel_best`对应76对快速监控；`sintel_monitor_best`对应845对完整监控部分。不要混用它们的指标。正式候选确认后再做留出检查和所需论文评测。

HPC路径和运行环境由本机配置派生为忽略的 `*.local.json`，启动前再记录完整配置。不要直接在HPC使用本机 `/datasets` 与 `/runs` 路径。当前不包含集群提交脚本，不自动提交作业。
