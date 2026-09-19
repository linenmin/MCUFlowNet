# 本机修改、Sofia训练、实验记录怎样配合

本机是代码修改和结果解释的主要位置；Sofia负责按已确定的代码与配置运行作业。服务器不必再开一个会自行修改代码的AI进程。Slurm会独立执行训练，本机休眠、SSH断开或聊天关闭都不影响已提交的作业；恢复协作时先读取作业状态和结果，不能直接重复提交。

## 文件分工

| 位置 | 保存什么 |
| --- | --- |
| 本机 `C:/00Work/Code/MCUFlowNet` | 实现、测试、实验配置；每个完成阶段提交并上传 |
| Sofia home 下 `Code/MCUFlowNet` | 从GitHub同步的代码仓库，不在这里直接改训练实现 |
| Sofia home 下 `Code/MCUFlowNet-releases/<完整提交号>` | 每批作业使用的固定代码副本；运行期间不修改、不更新 |
| Sofia项目的 `datasets` | 数据集，容器内只读；不放Git或home |
| 项目的 `mcuflownet/containers`、`environments` | 固定基础镜像及独立Python补充依赖，不修改系统软件 |
| 项目的 `mcuflownet/runs` | 权重、配置、每轮曲线和恢复状态 |
| 项目的 `mcuflownet/control`、`slurm` | 作业ID和Slurm日志；用于防止重复提交和核对实际运行 |
| 本机 `C:/00Work/Runs/MCUFlowNet/sofia-20260918` | 连接与作业记录、下载的小型结果；不包含认证秘密 |
| 本机 `C:/00Work/Lem_brain/wiki/项目/MCUFlowNet` | 可读进度、结论和下一步；附件只保存小型证据 |

具体账号与项目路径保存在本机运行记录，不固定进通用脚本。认证使用Pageant；不向服务器复制本机私钥。GitHub公开代码读取不需要把GitHub凭据复制到Sofia。

## 当前环境选择

2026-09-18实际查询Sofia：`module avail TensorFlow`没有结果，`module spider TensorFlow`报告找不到；zen4-ib对应模块和软件目录也没有TensorFlow。Tier2的 `TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1` 和 `~/tf_work` 不能直接在Sofia加载。

因此使用已有Apptainer运行与本机相同digest的NVIDIA25.02 TensorFlow基础镜像；补充依赖沿用本机版本，写入项目中的独立venv。容器版本不等于原HPC TF2.15.1；新实验统一使用这一套环境，并先实际验收。不能只看H200被识别就宣布训练正常。

H200作业每卡申请24 CPU核心，不设置内存覆盖，不使用`--export=ALL`。GPU分配显式传入容器，不能让某条实验使用未分配的GPU。规则来源：[Sofia官方说明](https://docs.vscentrum.be/brussel/tier1_sofia.html)，2026-09-18核对。

## 执行顺序

1. `bootstrap_sofia.sh`在计算节点准备环境，检查真实GPU算子、S/L参数更新、FC2数量和Sintel清单。登录节点只负责同步小代码、提交与读状态。
2. `run_sofia_job.sh`配合`run_campaign.py --mode probe`：四组各用一张H200，以实际batch32训练50步、退出后恢复到100步；每轮完整验证FC2的640对，监控Sintel的76/845对。探针权重不进入正式训练。
3. 四组通过后，用新的campaign执行`--mode train`，全部从零开始。保留400轮学习率周期，先到第15轮检查；代码仍是同一个固定提交。每个任务只用一张GPU。
4. 本机用`squeue`、`sacct`和`snapshot_campaign.py`检查进度，下载CSV/JSON更新wiki。判断收益使用正式曲线，不能用100步探针排名。
5. 保留原条件继续训练时，使用同一campaign与`--mode resume --stop-after 30`等新的暂停轮次；标签、batch、学习率周期及监控清单改变时另建实验，不覆盖原来的记录。

本机与HPC的数学实现和对照条件一致。HPC使用4个数据读取线程、关闭异步预取；专项测试核对单线程与4线程产生相同图片裁剪和标签。线程数相同的成对实验共享种子设置。CSV新增数据读取、参数更新和整轮耗时，帮助判断瓶颈来自GPU还是数据。

## 状态不明时怎么接续

标签对照的分组诊断使用 `tools/hpc/run_label_diagnostics.sh`，传入项目目录、固定代码副本、训练campaign、分支及新的评测编号。每个任务先复制第100轮last及完整监控最佳检查点（含优化器/BN保存内容），核对SHA256，再对固定845对监控集推理。输出逐样本CSV、按运动范围/分量阈值/场景汇总的JSON；总分须在1e-5以内复现对应历史记录。它不训练、不读取196对留出集，也不覆盖训练目录。权重和输出保存在项目runs中，不进入Git或wiki；wiki只收小型结果与判断。

若旧最佳检查点仅剩元信息、缺少index/data，诊断清单明确记录缺失，只对仍完整的保存点评测，不用另一轮次冒充。2026-09-19发现V3训练保存器的`max_to_keep=3`小于四类命名保存点的数量；旧最佳可能被自动清理，JSON元信息仍残留。诊断之后的修复提交`4a26c53`取消这项自动清理，固定名称只覆盖自身；`tools/validation/test_checkpoint_retention.py`用真实S/L图测试四类保存点跨进程恢复及再次保存。未来作业必须使用包含修复的固定提交，旧worktree不会自行升级。已有缺失权重不能由成绩JSON恢复。

先读本机session.json中的job ID和代码版本，再查Slurm。`run_campaign.py`保存每次作业的配置、命令、提交号和成功/失败状态，`snapshot_campaign.py`汇总小型记录。突然终止的进程可能还显示running，因此以Slurm最终状态和CSV、checkpoint共同判断。

不能只因SSH无回执就重新提交，不能让两个作业同时写同一个实验目录，也不自动删除失败产物。当前已测试的是正常训练段结束后的恢复；抢占或写检查点时中断仍需检查状态一致性，不能声称任意中断都能无损恢复。

服务器的活跃训练由Slurm维持，不需要本机agent持续在线。agent重新开始工作时依靠这些文件恢复上下文，不依靠聊天记忆。如果未来确实启用服务器上的另一个AI agent，让它只读日志、输出建议，代码修改仍先提交并经本机验证后发布新版本。

## 2026-09-18实际验收

固定运行提交 `2906163d2342b8a7675ac477ace176dbe801a446`。环境检查完成，实测TensorFlow 2.17.0、CUDA 12.8、Python 3.12.3，GPU算子和S/L更新通过。第一次环境创建遇到缺少ensurepip，改用基础镜像现有pip后成功；失败目录与日志保留，不作为可用环境。

`20260918-h200-probe01` 四组均完成：每组batch32、100步，中间退出并恢复，S各250、L各390个张量相同；成对初始化和两个训练段首批输入摘要一致。640对FC2验证、两次76对和一次845对Sintel监控通过，Slurm退出码均0。每组约4分53–54秒，包括启动、恢复和评测，不是纯训练吞吐。100步的误差不用于判断标签方案优劣。

正式 `20260918-fc2-label01` 已提交并开始运行：四个独立单H200任务，使用上述相同提交；随机初始化，batch32，400轮学习率周期，第15轮暂停，每任务时限6小时。运行编号及私人项目路径保存在本机Runs和wiki，公开仓库不复制私人运行登记。当前尚无完整正式训练结论。FT3D图像缺失暂不阻塞此轮FC2实验。


## 运行位置索引与备份

结果收集时用`tools/hpc/build_run_index.py`从登记目录的manifest重建`runs/index.csv`，下载到本机对应结果目录。配置例子和命令统一见[实验管理说明](experiment-workflow.md)。续跑按相同运行目录关联，作业ID逐次保留；索引不复制指标或状态。

重要阶段完成且作业停止写入后，在Slurm中运行`tools/hpc/backup_tree.py`，把指定运行目录打包到runs外的backups目录。脚本逐文件计算SHA256，打包后逐文件读回核对。将ZIP及同名`.zip.json`清单下载到本机`C:/00Work/Backups/MCUFlowNet/`，核对整个ZIP的SHA256；校验通过才记为备份成功。不是定时备份，不自动删除旧快照。解压恢复时用新目录并显式指定检查点前缀。wiki由用户已有的Obsidian官方同步管理，不重复上传到Sofia或代码仓库。
