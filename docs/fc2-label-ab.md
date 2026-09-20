# FC2 训练标签对照：准备与运行

这轮只检查 S/L 从头训练时，保留或取消 ±50 标签截断的影响。FT3D 的单位缩放不变，暂不开展 FT3D 实验。正式训练尚未启动。

## 四组设置

配置位于 `EdgeFlowNAS/configs/experiments/label_ab/`：`s_clip50.json`、`s_raw.json`、`l_clip50.json`、`l_raw.json`。同一模型两组仅实验编号和 `fc2_train_label_clip` 不同；50 表示分量截断，null 表示保留原值。`fc2_eval_label_clip` 全部为 null，所以 FC2 验证成绩可以直接比较。未写这些字段的旧配置仍默认 ±50。

第一版正式配置：种子42、352×480随机裁剪、逻辑batch32、microbatch32、Adam、学习率1e-4到1e-6余弦下降、400轮学习率周期，先在第15轮停下来检查、梯度范数上限200、无额外增广或权重衰减。它是后续HPC试跑的起点，不代表最优方案。BN实际看到的是microbatch32；不能只保证逻辑batch相同。正式配置的batch尚未在HPC验收。batch32和400轮周期取自发布库configs/retrain_fc2.yaml；第15轮暂停是本轮新增的阶段检查点，届时已有第5/10/15轮的完整监控。继续时提高stop_after_epoch并恢复last，不改变num_epochs；400轮是学习率计划，不是必须跑满的承诺。

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

运行摘要写入 `C:/00Work/Runs/MCUFlowNet/20260917-label-prep-v2-summary/`，四组权重和历史分别存入 `20260917-label-prep-v2-{s,l}-{clip50,raw}`。存在的实验目录不会覆盖。

同模型两组记录初始化参数摘要、每段第一批输入摘要；专项测试另外检查连续批次抽样和裁剪一致。摘要相同不是跨硬件逐位可复现承诺。恢复会核对参数/BN/优化器，保护标签设置、训练周期、batch和监控清单；当前只支持这组配置的轮次边界恢复。中间突然被杀的未完成轮次需要重跑，不支持逐步精确恢复。

`last`用于继续训练；`best`用于FC2验证最佳；`sintel_best`对应76对快速监控；`sintel_monitor_best`对应845对完整监控部分。不要混用它们的指标。正式候选确认后再做留出检查和所需论文评测。

HPC路径和运行环境由本机配置派生为忽略的 `*.local.json`，启动前再记录完整配置。不要直接在HPC使用本机 `/datasets` 与 `/runs` 路径。集群提交与接续使用 `tools/hpc/run_sofia_job.sh` 和 `run_campaign.py`；完整操作见 `sofia-workflow.md`。


## 本机验收结果（2026-09-18，北京时间）

四组全部通过，各100步，共400步；每组第50步退出，由新进程恢复到第100步。总墙钟时间1466.66秒，包含扫描、构图、训练、保存、恢复和评测；不用于推算HPC训练速度。

| 核对项目 | 结果 |
| --- | --- |
| 成对初始化 | S两组相同，L两组相同；已修复原函数漏设TensorFlow种子的问题 |
| 输入配对 | 每个训练段的第一批输入摘要一致；专项测试另检查连续批次裁剪一致 |
| 跨进程恢复 | 每个S核对250个张量，每个L核对390个，包含BN和优化器 |
| Sintel数量 | 每组两次76对、一次845对；留出196对未计分 |
| 数值检查 | 记录的训练loss、梯度、FC2与Sintel误差均有限 |
| 最佳权重 | FC2、快速Sintel、845对监控的保存点元信息均等于对应曲线最低值 |
| 专项测试 | test_label_ab.py与test_retrain_validation.py共4项通过 |

原始结果在 `C:/00Work/Runs/MCUFlowNet/20260917-label-prep-v2-summary/summary.json`，核对和环境在同目录audit.json；wiki小型副本位于 `C:/00Work/Lem_brain/wiki/项目/MCUFlowNet/附件/20260917-label-prep/`。旧编号首次启动在参数更新前停止，仅保留日志；不作为一次训练结果。

实际进程清单记录44ffbd8、1aed3d0或90c9314：后两个提交分别修改正式配置/说明和错误续跑保护，训练数值计算与短跑配置相同。输入起点与监控检查通过不等于跨硬件逐位可复现。工程短跑不用于判断截断收益；这段记录仅描述当时的本机验收；后续HPC记录在wiki实验总表。

当时的后续步骤是HPC路径、环境和batch32试跑；完成证据见wiki实验总表。任意时刻被抢占后的恢复尚未验收；当前只确认正常训练段结束后的恢复。


## 标签初筛结束后的参照续训

标签专项只比较四组前100轮；不继续安排四组第二种子重复。后续从S/L截断组第100轮恢复，先到150轮作为数据路线、学习率和增强实验的参照。原值分支及100轮独立备份保留；原始GT与历史GT两列继续记录。

沿用原campaign与目录，避免复制整套运行历史。作业参数为 `20260918-fc2-label01 resume s_clip50 150` 和 `20260918-fc2-label01 resume l_clip50 150`，传给 `run_sofia_job.sh`，前面仍需项目和固定Git工作树路径。只改变停止位置，不改变配置中的400轮余弦周期；每组从69500步续到104250步，保留Adam/BN状态。运行提交40fadbd已包含保存器修复，实际作业与状态查manifest和实验总表。

启动前核对100轮state、last元信息、index/data及独立备份；启动后须检查本次日志和恢复结果，不能把上次留下的restore_check当成本次验收。若改变LR、数据或增强，必须另建运行目录并记录父保存点，不覆盖这两条参照。

标签专项的结束是研究优先级决定，不表示已证明两种处理等价。旧正式V3 FC2每轮556步，当前695步；比较历史训练量须按更新次数，不能直接按epoch。后续完整方案重复的随机种子仍应独立。

## 将150轮参照继续训练到400轮

`EdgeFlowNAS/configs/experiments/fc2_long.json`登记FC2-LONG-01：S/L各从150轮last完整恢复到400轮。使用新目录，保留原参照；每轮695次更新、总278000次，原来的400轮余弦下降不重启，Adam、BN和抽样随机状态全部继承。唯一执行调整为训练prefetch从0改为1，batch32、352×480裁剪、标签和评测不变。启动时检查实际每轮步数，避免数据数量变化悄悄改变学习率曲线。

在H200作业内执行：

```bash
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/fc2_long.json --variant s
# 进程中断后的恢复（不是重新从150轮开始）：
python tools/hpc/run_retrain_experiment.py --recipe EdgeFlowNAS/configs/experiments/fc2_long.json --variant s --action resume
```

L将`s`替换为`l`。该配方已批准完整400轮；200/300轮只是保存节点，不自动暂停或转入FT3D。工程验收加`--probe`：先只跑到151轮，再用`--action continue --stop-step 105640 --probe`恢复到152轮；配置中的余弦周期仍为400轮。两支通过`tools/validation/check_schedule_probes.py`后，正式作业才可更新权重。验收核对S/L全部状态张量、旧格式随机状态、原曲线学习率、跨进程恢复、独立快照和完整640对FC2/76对Sintel快速监控；845对完整监控仍沿用每5轮一次。

正式输出`/runs/FC2-LONG-01/{s,l}/model_*`，固定节点存于同一分支的`milestones/epoch-0200/model_*`、`epoch-0300/model_*`、`epoch-0400/model_*`。每个节点包含可独立恢复的权重、Adam/BN、随机状态、曲线和配置，不受滚动恢复点只保留两份的规则影响。新分支的最佳权重从续训阶段重新记录，150轮之前的最佳仍在父目录；比较全程最佳时须同时查父运行。

这次回答“补完FC2训练周期是否仍有收益”。后续400轮末尾权重是否接同样的FT3D对照，按实验计划另行决定；不会由本作业自动开启。代码/配置保存在Git，作业、曲线和权重在Runs，当前状态只在wiki实验总表维护。
