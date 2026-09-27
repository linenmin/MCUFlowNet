# LOWRES-BENCH-01：三模型共同小分辨率重训

## 研究问题与固定配方

比较EdgeFlowNet、MCUFlowNet-S/L在同一输入尺寸、数据和预算下的表现。不是逐模型寻找最低EPE，也不是原作者训练的逐项复现。

| 项目 | 共同设置 |
|---|---|
| 输入 | 宽208×高160；整图面积插值；BGR转[-1,1] |
| 光流 | 双线性缩放，u/v分别乘宽/高比例；单位为缩图后的像素；不截断、不除12.5、不设400掩码 |
| 初始化 | seed42，从零开始；后续正式比较补43/44，不在本次首轮自动提交 |
| FC2 | 400完整epoch，固定1e-4；每轮打乱一次，无放回，尾批不补样 |
| FT3D | TRAIN左相机Clean+Final、future+past；50完整epoch，固定1e-5；继承FC2末尾模型及BN，重置Adam |
| 优化器 | batch32、FP32、TF32关闭；Adam β=0.9/0.999、ε=1e-8；无权重衰减/梯度裁剪 |
| 损失 | 复用共同三层累积L1+LinearSoftplus不确定性损失，权重0.125/0.25/0.5，不确定性系数1 |
| BN | 三模型训练开启、验证关闭；momentum0.9、epsilon1e-5；Edge原代码补训练开关 |
| 加载 | 8线程、预取1批；无额外颜色/几何增强、EMA或QAT |
| 监控 | FC2每10轮、FT3D每轮：完整FC2 val和既有845对Sintel Final监控集；阶段最后一轮必测 |
| Sintel口径 | 中心416×1024区域缩到208×160预测，还原到原坐标；全部像素、不截断 |
| 主结果 | 固定训练终点；best_monitor仅作共同规则选优补充，不按某模型成绩单独改日程 |

日程参照[EdgeFlowNet训练设置](https://arxiv.org/html/2411.14576v1#S3.SS1)。统一归一化、真实像素监督及显式BN训练是本研究修订；旧权重推理输入和输出单位不能直接套到新权重。845对用于调试和选优，不宣称独立测试集；196对也曾被旧权重评测，不宣称从未使用的留出。

## 代码和输出放在哪里

- 本机工作树：`C:/00Work/Code/MCUFlowNet-lowres`。
- 新分支：`training/lowres-208x160-tier2`；从已完成板端检查的分支创建。
- 共同实验代码：`tools/lowres/`；data负责清单和单位，model复用现有三模型与损失，train负责完整阶段和恢复，verify负责跨进程验收，tier2.sh负责环境与阶段衔接。
- Tier2独立checkout：`/user/leuven/379/vsc37996/Code/MCUFlowNet-lowres-208x160`。
- 数据只读：`/data/leuven/379/vsc37996/test/MCUFlowNet/Datasets`。
- 运行根目录：`/data/leuven/379/vsc37996/MCUFlowNet-runs/LOWRES-BENCH-01`。manifests放共同清单及SHA，probe放短跑证据，seed42/<edge|S|L>/<fc2|ft3d>放训练，slurm放日志。
- 状态只维护在Obsidian光流Benchmark实验记录；完整日志、权重不入Git，完成后按已有Globus约定归档到本机Runs。

## 开跑验收与恢复

本机test_data验证已知位移缩图、65样本尾批32/32/1、无重复遗漏；test_graph验证三模型GPU参数/BN更新、验证BN不变及共同损失的独立常数解。

服务器先在CPU作业扫描清单和核验路径，再对三模型并行GPU短跑：65对真实FC2、两轮；连续跑与跨进程中断恢复比较所有模型/BN/Adam变量和样本顺序；FT3D两轮短跑核对FC2权重/BN完整继承、Adam归零。通过后才能提交正式训练。

共同清单为FC2训练22,232对、验证640对、FT3D训练80,578对、Sintel监控845对。FT3D沿用历史明确名单：13个TRAIN光流文件实测各含4个非有限分量，统一排除其Clean/Final共26对图像；不是按位移大小筛样本。`audit_exclusions.py`保存原清单、文件SHA、无效数值计数和排除后的清单SHA到运行目录。

每个完整epoch保存所有TensorFlow变量和确定性样本序列对应的epoch/step；检查点写完后原子替换current.json。保留当前与前一轮检查点、共同监控最优及阶段末尾，不累积400份权重。中途被终止则从上一完整epoch重跑。恢复时核对配方、数据清单SHA和全部变量，禁止悄悄改变日程。

正式启动：tier2.sh train按FC2→FT3D自动衔接；已有current.json则恢复。每任务1张GPU、8CPU、32GB内存。提交时明确集群/账户/分区/时限；运行中不更新服务器checkout。TensorFlow使用Tier2已有2.15.1-foss-2023a-CUDA-12.1.1模块和~/tf_work，不安装新环境。模块配套NumPy/SciPy优先，tf_work仅补充依赖，避免旧虚拟环境覆盖模块的兼容组合。Python、NumPy、Keras和TensorFlow共同设种子，短跑还核对跨进程初始权重SHA。

## 板端预检结论

夜间续跑使用`continue.sh`＋`continue.py`，通过Slurm的afterany依赖预排两段、每段最多12小时。仅前段TIMEOUT/NODE_FAIL/PREEMPTED才恢复；FAILED/OOM/CANCELLED停止。全部阶段完成则直接退出，不重复训练。若首轮中断而尚无完整检查点，将该阶段半成品改名保留，再从该阶段起点恢复。控制器可从已提交版本复制到独立运行控制目录，实际训练始终调用原来锁定的checkout，不在线更新训练代码。具体作业编号与状态只记实验记录及运行回执。

2026-09-27旧权重实测：Edge208×160为6.097 FPS，S/L224×160为10.390/5.918 FPS，均通过模型CRC及连续预览；下一档Edge224×160缺131064 B，S/L240×176分别缺123128 B。限当前带相机/预览固件，不代表板卡绝对上限。

Edge共同208×160已经可运行，不再沿用早期误把旧arena当硬上限的缩小建议。新权重仍需重新量化、编译、验收；本次板端检查不是EPE测量。证据：本机Runs/MCUFlowNet/LOWRES-BENCH-01/board-results-20260927.json及wiki同一实验条目。
