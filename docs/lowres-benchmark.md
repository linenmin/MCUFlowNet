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

服务器先在CPU作业扫描清单和核验路径，再对三模型并行GPU短跑：65对真实FC2、三轮；连续跑与跨进程中断恢复比较所有模型/BN/Adam变量和样本顺序；FT3D三轮短跑核对FC2权重/BN完整继承、Adam归零。通过后才能提交正式训练。

共同清单为FC2训练22,232对、验证640对、FT3D训练80,578对、Sintel监控845对。FT3D沿用历史明确名单：13个TRAIN光流文件实测各含4个非有限分量，统一排除其Clean/Final共26对图像；不是按位移大小筛样本。`audit_exclusions.py`保存原清单、文件SHA、无效数值计数和排除后的清单SHA到运行目录。

每个完整epoch保存所有TensorFlow变量和确定性样本序列对应的epoch/step；检查点写完后原子替换current.json。保留当前与前一轮检查点、共同监控最优及阶段末尾，不累积400份权重。中途被终止则从上一完整epoch重跑。恢复时核对配方、数据清单SHA和全部变量，禁止悄悄改变日程。

正式启动：tier2.sh train按FC2→FT3D自动衔接；已有current.json则恢复。每任务1张GPU、8CPU、32GB内存。提交时明确集群/账户/分区/时限；运行中不更新服务器checkout。TensorFlow使用Tier2已有2.15.1-foss-2023a-CUDA-12.1.1模块和~/tf_work，不安装新环境。模块配套NumPy/SciPy优先，tf_work仅补充依赖，避免旧虚拟环境覆盖模块的兼容组合。Python、NumPy、Keras和TensorFlow共同设种子，短跑还核对跨进程初始权重SHA。

夜间续跑使用`continue.sh`＋`continue.py`，通过Slurm的afterany依赖预排两段、每段最多12小时。仅前段TIMEOUT/NODE_FAIL/PREEMPTED才恢复；FAILED/OOM/CANCELLED停止。全部阶段完成则直接退出，不重复训练。若首轮中断而尚无完整检查点，将该阶段半成品改名保留，再从该阶段起点恢复。控制器可从已提交版本复制到独立运行控制目录，实际训练始终调用原来锁定的checkout，不在线更新训练代码。具体作业编号与状态只记实验记录及运行回执。

## 板端预检结论

2026-09-27旧权重实测：Edge208×160为6.097 FPS，S/L224×160为10.390/5.918 FPS，均通过模型CRC及连续预览；下一档Edge224×160缺131064 B，S/L240×176分别缺123128 B。限当前带相机/预览固件，不代表板卡绝对上限。

Edge共同208×160已经可运行，不再沿用早期误把旧arena当硬上限的缩小建议。新权重仍需重新量化、编译、验收；本次板端检查不是EPE测量。证据：本机Runs/MCUFlowNet/LOWRES-BENCH-01/board-results-20260927.json及wiki同一实验条目。

2026-09-28恢复修复：恢复核验后立即释放检查点读取器；旧文件清理失败只记警告并在下轮重试，不能使已保存的新一轮训练退出。三轮短跑覆盖首次删除恢复来源检查点的边界；最初两轮短跑未覆盖这一情况。模型、损失、学习率和样本顺序不变。

## 对已有权重排查评分与BN波动

`tools/lowres/audit_bn.py`在本机读取独立下载的权重副本，不运行优化器、不写检查点、不改变服务器训练。输入目录需要`manifests/{sintel_monitor,ft3d_train}.json`以及各模型`{edge,S,L}/{best,epoch18}/model.{index,data-00000-of-00001}`；不存在的epoch18副本跳过。

程序独立解析Sintel的.flo、缩图和计算EPE，复测全部845对；仍复用原模型图，不声称独立重写了网络。还与生产数据读取及评分入口交叉核对，并逐张保存EPE。每次恢复逐变量检查与checkpoint完全一致；推理前后核对所有BN均值/方差不变。

诊断固定从FT3D TRAIN按seed20260929取1024对，batch32，等权平均各批BN统计；这是BN校准诊断，非完整总体方差的精确估计。可训练参数及优化器状态始终不动。对第18轮副本，再分别用该轮最后2对、含这2对的末32对，仅更新一次BN（momentum0.9），以检验小尾批的影响；这不是完整重放最后一次梯度更新，也不是32+2与34的训练对照。

Python3.12下仅对tf-keras初始化时传入randint的整数浮点上界作兼容；实际权重全部从文件恢复并逐值核验。结果写Runs中的JSON，包含checkpoint SHA256、校准索引、末批文件名、逐图成绩和TensorFlow版本。正式训练配方与评分表不因该诊断自动改变。

## 尾批训练对照：32＋2与34

用户于2026-09-29批准此诊断。入口`tools/lowres/tail_diagnostic.py`，HPC包装为`tail_diagnostic.sh`。每模型先固定一份完整FT3D检查点（包含Adam及BN），保存在独立运行目录`source/<model>/`，校验复制前后SHA256；不能直接依赖正式训练随时清理的旧轮次。

固定使用seed42、第18轮的数据排列，专门检查已发现异常的最后两对图像。先共同训练前80544对（2517批），再从完全相同的完整状态分别执行末32＋2、合并末34；不漏样、不补样、学习率均1e-5，两路仅相差最后分批方式及一次优化器更新。用各模型已固定的近期权重开始，起点轮次写source-receipt；不是从历史第17轮恢复，不宣称重现原第18轮，也不根据这个诊断跨模型排准确率名次。

记录分叉前、32对之后、2对之后、34对之后的完整FC2 val与845对Sintel；另外交换实际最后2对更新前后的BN/非BN状态，分别检查BN变化与参数更新的影响，两者可能交互。末批额外记录梯度范数、参数变化、BN均值/方差变化；之前的共同前缀不额外计算梯度范数。输出保留分叉前和各真实分支的完整检查点，不改生产输出。

开跑验收用66对真实样本（共同32对＋末34），并只用2对FC2与2对Sintel评分。检查恢复全部变量一致、推理不改状态、每对样本仅出现一次、两分支起点完全相同、两路步数分别+2/+1；分支结束后从共同状态重跑32＋2，要求全部模型/BN/Adam变量逐值相同。每个HPC任务先通过自己的短跑再执行完整诊断；失败则停止。活动训练checkout和既有日程保持不动，诊断使用独立按提交固定的checkout。

验收结论：若合并尾批稳定改善且BN状态交换重现差距，支持采用合并尾批；若梯度更新后仍明显退化，另查梯度或学习率。单个固定尾批只是因果诊断，正式推广仍需多轮观察；不以是否让MCU赢过Edge作为通过条件。程序不自动改正式训练。

## 修正尾批后的20轮学习率对照

2026-09-29用户批准六条FT3D训练：Edge/S/L各一条固定学习率、一条余弦衰减，均20轮。各模型两组从同一份FC2第400轮模型及BN初始化，Adam与FT3D计步归零；来源复制到独立实验目录并校验SHA256。保留原50轮训练作参照，不在线更新旧工作区。

两组均为seed42、208×160整图缩放、原80,578对FT3D清单、原损失与无额外增强；batch32，最后32＋2合为34，每轮2,518次更新。通用规则只把小于半批的尾批并入前一批；FC2尾批24不受此规则影响，本次不重跑FC2。每对图像恰好使用一次，样本顺序保持一致。

固定组每轮1e-5。衰减组每轮内恒定，轮间按`1e-6 + (1e-5 - 1e-6) * (1 + cos(pi*(epoch-1)/19))/2`变化，第1轮1e-5、第20轮1e-6；无warmup、无重启。`train.py --epochs 20 --merge-tail --keep-every 5 --lr-schedule {constant,cosine} --min-lr 1e-6`。恢复必须匹配原配置，不能把20改50来悄悄重画余弦日程；未来续训需明确的新阶段方案，完整状态已保留。

每轮验证原FC2 640对与Sintel 845对；比较同轮曲线、20轮最佳、末5轮中位数及第20轮值。Sintel是开发监控集，不能称为未参与选型的独立测试集。第10轮检查运行/曲线，但不以某模型是否领先来单独早停。训练每5轮、起点、第20轮、最佳及最近两轮保留完整模型/BN/Adam；best_monitor也附可恢复状态。记录每轮实际学习率、样本顺序指纹及实际末批大小。

`verify_comparison.py`以65对真实训练图像运行完整20轮日程，比较连续与第7轮中断后恢复的所有变量逐值一致，检查两配方起点/样本顺序相同、最佳权重含Adam及保留策略。短跑只用2对FC2与2对Sintel验证，不报告为模型成绩；实际66对→32/34及全量清单的分批覆盖另由数值测试核对。`ft3d_compare.sh probe`运行此验收；正式数组0/1/2为固定Edge/S/L，3/4/5为衰减Edge/S/L，验收失败不启动正式训练。

每任务1张H100、8CPU、32GB RAM；正式阶段一次请求12小时，并预排一段同配方续跑，仅TIMEOUT/NODE_FAIL/PREEMPTED允许恢复。FAILED/OOM/CANCELLED停止，已完成20轮则退出而不重复训练。任务号、实测状态和输出目录统一留在实验记录与Runs回执。此20轮对照可判断该预算下的两种安排，不声称已经充分优化或证明架构最终排名。

## 已有权重的误差定位

`tools/lowres/audit_gap.py`只读取20轮余弦组的三份最佳权重，逐对评测固定845对Sintel；记录原图/部署尺寸EPE、水平/垂直误差、逐场景结果和按真实运动大小划分的像素误差。检查所有模型张量与checkpoint逐值相同、推理前后未变，并要求总分复现训练记录至1e-5以内。不执行优化器或BN更新，不生成新候选权重。

在已有本机GPU容器执行：`python tools/lowres/audit_gap.py --data /datasets --experiment /runs/LOWRES-BENCH-01/ft3d-lr20-20260929 --out /runs/LOWRES-BENCH-01/gap-audit-20260929/inference`。输出目录须不存在，避免覆盖历史诊断。随后`python tools/lowres/summarize_gap.py --runs /runs`将逐图结果、FC2/FT3D曲线与旧公开权重的同845对评分汇总，保留来源SHA256。

原图与小图的像素单位不同，不跨列比较绝对误差。运动分组采用原图GT模长，边界为10和40像素；分组对总差距的贡献按像素数加权。旧S/L缩图参照为224×160、Edge为208×160，不能冒充同尺寸对照。场景统计使用已参与开发的监控集，不解释为独立测试显著性。

## 输入尺寸与Edge BN行为的两个对照

2026-09-30用户确认先做短程验证：416×320的Edge/S/L，以及208×160的Edge冻结BN统计组（宽×高）。四条只训练FC2 50轮，不进入FT3D、不自动延长到100或400轮。此前400＋20轮计划撤回，旧作业暂停后由短程作业替换。

共同设置：随机初始化seed42、batch32、固定学习率1e-4、Adam、每轮695步，共34,750步；数据清单、顺序、整图AREA缩放、BGR归一化、真实缩图像素标签及多尺度损失不变，FP32、TF32关闭，无额外增强。FC2尾批为24对，保持原设置。每10轮验证FC2和同845对Sintel，保留0/10/20/30/40/50轮、最佳及最近两轮完整状态。

比较第10/20/50轮与已有208×160正常BN训练的相同轮次，不与旧400轮或FT3D最佳分数直接比较。Sintel都换回原图坐标评分；FC2跨尺寸比较须换成同坐标。50轮只筛查尺寸和BN是否明显改变趋势，无收益不证明充分训练后仍无收益；是否延至100轮另行决定。

大图像素数为四倍、位移标签数值也加倍，因此不能单独区分图像信息和标签尺度的作用。参照FC2在A100，新组在H100，记录硬件差别。冻结组始终用均值0/方差1，gamma/beta继续训练，epsilon保持1e-5；采用fused=False支持确定性反向传播，正常组保持原算子。这是BN行为对照，不是作者历史环境逐项复现。

`factor_compare.sh`数组0/1/2为大图Edge/S/L，3为小图冻结Edge。服务器先用`verify_factors.py --fc2-only`验证真实65对的连续训练与恢复全部状态一致、冻结统计不变且gamma/beta更新，再允许正式50轮。已完成50轮的续跑任务直接退出；只有TIMEOUT/NODE_FAIL/PREEMPTED允许从完整轮恢复，训练错误/OOM/人工取消停止。采用独立checkout与运行目录，任务号、有限续跑段数及进度只写Runs回执与wiki实验记录。

### Mindwell B200运行方式

B200使用与本机/Sofia相同摘要的NVIDIA TensorFlow 25.02容器和固定补充依赖，环境在Mindwell的GPFS scratch中独立重建；旧TensorFlow 2.15模块不能在该节点加载。数据仅按FC2与Sintel清单迁移，经Globus checksum校验后写DATA_READY.json，再做四条FC2恢复验收。factor_b200.sh调用共同factor_compare.sh，训练预算仍为50轮，不改模型和损失。续跑在宿主机按SLURM_CLUSTER_NAME查询正确集群。

B200环境与旧A100参照的软件/硬件均不同，结果用于筛查，不能宣称严格单因素因果证据。先比较短跑初始化指纹与本机既有验收，记录版本与恢复一致性；若出现差别应先定位。输出留持久data目录，容器及数据在GPFS，完成后归档本机。
## 已归档权重的训练集BN诊断

`audit_factor_bn.py`检查三模型的大图FC2第30/50轮、小图FC2第400轮，或小图FT3D余弦组各自最佳权重。分别选择`--kind large-fc2`、`small-fc2`、`small-ft3d`；实验目录对应原运行根目录，可用`--manifests`显式传共同清单。大图采用416×320，小图208×160，Sintel始终是同845对、原图416×1024坐标、原始GT。

从对应训练集按seed20260930固定取1024对，batch32，只重新估计内存中的BN均值和方差；按批等权平均，这是诊断近似，不是完整总体方差。三模型同样处理，不使用Sintel估计统计，不执行优化器、不保存或覆盖模型检查点。推理前后检查BN不变，校准前后检查非BN状态和源文件SHA不变；原始完整评分须与服务器相差小于1e-5。

`--smoke`仅用64对训练图像与2对Sintel做恢复/状态检查，不报告为模型成绩。完整JSON、逐图分数和运行日志放Runs；诊断结果用于解释排名，不能静默替换benchmark成绩。具体结果及代码提交记在wiki实验记录。

## V3超网提取验收（不训练）

`tools/lowres/extract_supernet.py`读取历史V3非蒸馏及蒸馏超网的best checkpoint，分别提取S（00000000000）和L（20022100000）。源文件只读挂载；输入固定208×160整图缩放，BGR映射到[-1,1]。原超网的FC2预训练使用224×172随机裁剪、原始flow分量clip±50，没有除12.5，故提取后的输出直接按输入像素解释。

层的累计编号在超网与独立模型中不同。程序只移除外层名称和层计数，保留模块、分支、显式层名及参数名；每个对应必须唯一、形状相同，全部所需张量必须精确加载。用2对FC2和2对Sintel逐一比较三个尺度的四通道输出及累积光流，误差须满足atol=1e-4、rtol=1e-5；保存独立checkpoint后重载，所有张量指纹必须完全相同。源码复核确认当前完整V3网络与历史commit 2ccee133的网络文件正文相同。

完整验收使用FC2验证640对、Sintel监控845对。FC2按208×160像素评分，Sintel还原到416×1024原图坐标，真值与预测均不截断。另从FC2 TRAIN按seed20260930固定取1024对，以batch32重估BN；仅改均值和方差，不更新卷积、ECA、Gate、gamma/beta，不使用验证集校准。方法为批统计等权平均，未补偿批间均值差，与此前BN诊断一致。原统计与重估统计都保留并报告，不能只展示较好的一项。

在已有本机GPU容器中额外将历史outputs/supernet只读挂载到`/supernets`，执行：

```text
python tools/lowres/extract_supernet.py --data /datasets --supernets /supernets --manifests /runs/LOWRES-BENCH-01/ft3d-lr20-20260929/manifests --out /runs/LOWRES-BENCH-01/supernet-extract-20261001/full
```

先用独立目录加`--smoke`检查64对训练图像和少量评分；输出目录已存在时拒绝覆盖。完整目录保存来源SHA256、逐张量映射、逐图EPE及八份独立模型（两来源×S/L×两种BN统计），不保存超网Adam槽，也不执行优化器。此阶段只判断提取是否正确及第0轮表现；不表示继承已胜过从头训练，更不启动FC2/FT3D。

后续建议的第一轮共同适配仍为208×160整图缩放、真实未截断标签：这样对应整幅画面的部署输入。随机裁剪若要比较，应给Edge/S/L共同设置，另列实验。去掉预训练的标签截断允许学习真实大位移，不要求预测也clip；数值和图像分布都在变化，应保留第0轮及早期监控。FC2继承使用下节显式的独立checkpoint及低学习率参数，不得用伪造current.json绕过阶段约束。

### 蒸馏继承的候选适配配方（2026-10-01）

主来源固定蒸馏超网best第190轮，不新增非蒸馏训练。硕士论文§5.5、表5.4已经据排名消融选择蒸馏来源；本轮检验继承后的部署尺寸适配，不重做该消融。S/L使用`supernet-extract-20261001/full/distill/{S,L}/original/model`，路径相对Runs/LOWRES-BENCH-01，完整来源指纹见已有提取证据。Edge的新适配起点建议用作者公开`C:/00Work/Code/optical-flow-upstream/EdgeFlowNet/checkpoints/best.ckpt`。已有小尺寸Edge FC2第400轮已经适应该输入，不需要额外补20轮FC2；其FC2及FT3D结果保留作另一条参照，不能用较弱的新结果覆盖。

原Edge接入必须先验收：已复现公开best的推理入口接收BGR 0–255，累积输出直接为输入像素位移，不乘12.5；共同训练器接收[-1,1]，并显式改变BN训练开关与epsilon。权重名称/形状相符不保证预测一致。须保留原输入与BN推理语义或进行经过验证的等价转换，在同样图像上重现原入口预测，再固定并验收训练时BN更新规则。此项尚未实施，不能只替换checkpoint路径就开训。来源为Runs/MCUFlowNet/SINTEL-BASE-01/edge-full/manifest.json及tools/lowres/model.py。

用户于2026-10-01同意此路线；先通过验收，再执行三模型各FC2适配20轮，FT3D另审。20轮是共同新增预算，不是必要轮数，也不能使历史预训练总预算相等。seed42、batch32、FP32、TF32关闭；输入宽208×高160整图缩放、原真实标签及共同损失。S/L沿用原BN正常更新。Edge保留原均值/方差及epsilon1e-3，训练卷积和gamma/beta；用fused=False避免冻结BN反向的非确定算子，须与作者默认推理验收一致。三模型验证均冻结统计。这不是BN规则完全相同的架构对照，而是保留各自原模型输入与BN行为的继承适配。Adam β=(0.9,0.999)、ε=1e-8，阶段开始重置；学习率逐轮按现有20轮余弦公式1e-5→1e-6，无warmup/重启，不加教师损失、额外增强或BN重估。原22,232对训练样本每轮695步，共13,900步，尾批24。起点及每轮测FC2 640对与Sintel 845对；保留0/5/10/15/20轮、FC2最佳、Sintel最佳及最近两轮完整状态。

第5/10轮检查异常与趋势，不以MCU未反超Edge单独停训；20轮后审阅FT3D或共同延长。若进入FT3D，三模型均选择FC2验证EPE最低的权重，第0轮也参与，避免额外适配损坏较强起点。FT3D候选为20轮，重新初始化Adam，1e-5余弦降至1e-6，尾批合并34，20×2,518步；其提交另行决定。保留原FT3D Edge6.904、S7.032、L7.062作参照，最终比较最佳、末5轮中位数与末轮；不把从第0轮改善等同于胜过已有模型。

实现入口为`train.py --init-checkpoint <prefix> --initial-lr 1e-5 --eval-every 1 --lr-schedule cosine --epochs 20 --keep-every 5`；原Edge另传`--edge-public --bn-mode frozen`。只支持FC2独立权重初始化，不伪造既有阶段状态，也不改变旧scratch与FC2→FT3D默认协议。全部模型/BN张量按明确作用域精确恢复，重置Adam槽、beta powers和计步；第0轮参与FC2与Sintel最佳保存。两种最佳目录均含完整优化器状态与对应指标，最终FC2最佳要包括第0轮，不能强制使用适配后的末轮。进入FT3D时的最佳选模加载仍需该阶段的实现与验收，本次不提交FT3D。

`verify_adaptation.py`用真实65对训练图像做3轮短跑，检查连续与第1轮中断恢复后的所有张量逐值相同、卷积实际更新、BN行为正确、两个最佳检查点完整及源文件未变。短跑每次只测2对FC2与2对Sintel，不报告为模型成绩。原Edge还与保留作者BN默认行为的独立推理图比较三尺度四通道和最终flow；本机追加同845对的完整评分一致性检查。测试保留源码指纹，开发阶段结果不能仅凭基准提交号声称代码未经修改。

HPC包装`adapt.sh`复用Mindwell已验收的TensorFlow25.02容器，先在服务器做每模型短跑，三条均成功才允许三条正式FC2。每任务1张B200、8CPU、32GB；首段4小时，另排一段有限续跑，只允许TIMEOUT/NODE_FAIL/PREEMPTED恢复，训练错误或人工取消不自动重试；完成20轮则跳过。新权重、日志和模型输出放GPFS scratch上的独立`inherit-adapt-20261001`目录，避免占用已近满的NFS data；代码放Home固定提交的独立Git工作区，完成后完整归档本机。任务号与实际状态只记Runs回执及wiki实验记录。

### 延长适配日程的50轮对照（2026-10-01）

用户批准三模型从上述相同的各自初始权重重新适配FC2 50轮，检验更长预算及更慢余弦衰减的共同效果。不是从20轮末尾接续，也不改写原20轮成果。Adam重置、seed42、batch32、每轮695步，共34,750步；学习率从第1轮1e-5到第50轮1e-6。数据、208×160 AREA整图输入、真实未截断标签、损失、各自输入与BN规则保持不变；不新增增强、蒸馏损失或FT3D阶段。

包装参数为`adapt.sh <probe|train> <repo> <data> <runs> <predecessor-or-empty> <epochs>`；未传第六参数仍为20轮。50轮首段传空的前序任务和`50`，有限续跑同样传`50`，防止训练与完成门禁的总轮数不同。第0轮和每轮测FC2 val640对与Sintel845对，保留每5轮、两个最佳及最近两轮的完整状态；后续仍统一按FC2验证最佳选下一阶段起点。

运行目录为独立的`inherit-adapt50-20261001`；源权重和三份数据清单按SHA核对原20轮实验。计算节点先执行原继承／恢复验收，三条通过后再开始正式训练。每任务1张B200、8CPU、32GB，首段4小时，最多一段4小时超时续跑；50轮完成则跳过，不因错误或取消自动重试。比较第20/30/40/50轮最佳和后期五轮通常水平，不把学习率变化造成的平台直接解释为架构极限。完整回执、配置和当前状态只写Runs及wiki实验记录。

### FC2验证最佳接FT3D 20轮（2026-10-03）

用户批准Edge/S/L各训练FT3D 20轮，从完成的50轮FC2运行中按FC2验证最佳选权重：Edge第46轮、S第50轮、L第48轮。`train.py --phase ft3d --init-fc2-best <completed-fc2-directory>`核验完整父阶段、选择记录、模型、尺寸及输入/BN语义；只恢复模型和BN，Adam、beta powers和步数重置。作者原始Edge权重没有`edge/`前缀，经过FC2适配的权重已有该前缀；本阶段保持原Edge输入和BN语义，但按已适配作用域加载，不能再剥去前缀。原FC2继承和默认FC2末尾接FT3D入口保持原行为。

宽208×高160整图AREA缩放、真实未截断位移、seed42、batch32、末尾32＋2合34；沿用80,578对FT3D TRAIN左相机Clean+Final、future+past的清单，每轮2,518步，共50,360步。Adam参数与损失不变，学习率1e-5余弦降到1e-6；不新增增强或蒸馏。Edge统计冻结，S/L正常更新，FP32且TF32关闭。第0轮和每轮测FC2 val640对及Sintel Final845对；正式第0轮必须复现所选FC2检查点的评分（容差1e-5），第0轮也参与最佳保存。保留每5轮、两个最佳及最近两轮完整状态。

`verify_adaptation.py --fc2-best <directory> --full-reference`在三模型上核验全部FC2/Sintel起点评分、精确模型/BN加载及Adam重置，再用65对真实FT3D做三轮训练（每轮32+33两批）。连续与中断恢复的所有变量逐值一致，源权重只读，BN行为正确且参数实际更新。`adapt.sh`第七参数设为`ft3d`，第六参数为`20`；服务器三条验收全部通过后才运行正式训练。每任务一张B200、8CPU、32GB，首段8小时、最多一次8小时的超时/节点故障/抢占续跑；训练错误或取消不自动重试，完成20轮则跳过。输出独立于FC2，使用`LOWRES-BENCH-01/inherit-ft3d20-20261003`，完整产物按原规则归档本机。

### 继承FT3D权重的BN诊断

`audit_inherited_bn.py`检查已完成的继承实验，比较起点／末轮参数与起点／末轮BN均值、方差的四种组合。gamma/beta随参数组保留。可再用固定1024对FC2 TRAIN及FT3D TRAIN分别重估末轮统计；样本由预先固定的seed选择，不使用Sintel图像校准。只在内存中改变统计，不运行优化器、不保存检查点，源文件SHA及非BN参数保持不变。原始两个端点必须复现日志EPE；冻结Edge是统计交换应无效果的对照。混合参数和旧统计可能不匹配，不能把其分数直接称作“冻结BN训练”的结果，不能相减计算BN贡献百分比。

```powershell
./tools/setup/run-local.ps1 python tools/lowres/audit_inherited_bn.py --data /datasets --experiment /runs/LOWRES-BENCH-01/inherit-ft3d20-20261003 --out /runs/LOWRES-BENCH-01/causes-audit-20261003/bn-verified --calibrate 1024 --fc2-manifest /runs/LOWRES-BENCH-01/inherit-adapt50-20261001/manifests/fc2_train.json
```

输出目录必须是新目录。先加`--smoke --models S --calibrate 64`并改输出目录做小样本验收；smoke分数只验代码。诊断不改变正式评测协议，完整结果仍留Runs，wiki只收关键判断。

### 共同取图方式对照：每条10,000步

2026-10-03用户批准三模型各两条FC2短程对照。所有分支从原共同scratch训练的FC2第400轮末尾初始化，不用各自Sintel最佳或蒸馏继承权重；只恢复模型与BN，Adam和计步重置。三模型均BGR [-1,1]、训练态BN（momentum0.9、epsilon1e-5），避免引入公开Edge的固定BN差别。输出宽208×高160，FP32、TF32关闭，真实位移不截断；损失、Adam参数和batch32均沿用共同配方，原FC2尾批为24。

每条恰好10,000次更新，逐步余弦1e-5降至1e-6；总步数相同，最后只走完末次遍历的一部分。整图组保持原AREA缩图。随机组20%保留整图，80%按原图面积的15%–100%、源区域宽高比1.3–2.5抽取同一个矩形，同时裁两帧与光流，再缩到208×160。面积均匀抽样，宽高比按对数均匀抽样；最多尝试10次，失败则采用最后比例下能放入原图的最大区域。图片用AREA，标签用双线性，并分别乘横／纵缩放系数；不新增颜色、翻转、遮挡增强或教师损失。此对照检查整套取图方式，不能单独分离视野、尺度和形状影响。

样本顺序按seed42和遍历次数确定；几何按seed42、遍历次数、样本序号与固定命名空间20261004确定，不受模型或读取线程影响。记录每1,000步的样本与矩形指纹，以检查三个模型是否实际使用同一对照。验证始终整图缩放，FC2 val640与Sintel Final845在第0步及每1,000步评分；第0步必须复现FC2来源记录（容差2e-5），且参与最佳选择。保留每个监控点的完整模型、BN、Adam及计步；中断从已发布检查点按样本游标重放，不继承中断后的部分更新。

`test_geometry.py`检查合成图像的共同裁剪、位移单位、无验证增强，以及多线程／中断游标一致性；`verify_geometry.py`用真实65对FC2，在两组各5步中于第3步（遍历中间）中断，要求恢复后所有模型、BN、优化器变量逐值相同。它还核验Adam重置、实际参数更新、起点预测相同、样本顺序相同和源文件未变。服务器三模型短跑全部通过，才启动六条正式训练。包装入口为`geometry_compare.sh`，Mindwell每任务1张B200、8CPU、32GB，首段4小时，最多一段同预算的超时／节点故障／抢占续跑；取消或训练错误不自动重试。

报告每条相对第0步的改善、固定步数的两组差值、后五次监控中位数和末步，不只挑一次最佳。若随机组使S/L相对Edge的差距稳定缩小，支持取图方式与架构的交互；若三者均改善但Edge仍领先，保留这一结果。正向结论需要另补独立种子，不把六条seed42分支当作六次独立重复。完整配置、代码提交、验收与权重留`Runs/LOWRES-BENCH-01/geometry10k-20261004`，wiki只维护同一实验记录。

资源收尾修订：首段4小时的六任务预留过大（SAM在10-04核对约648,250 credits），不等于实际已花费。当前实验首段改为每任务1小时，训练目标仍为10,000步。用低成本CPU任务在首段全部结束后检查完成状态；全部完成就不申请GPU，仅对TIMEOUT/NODE_FAIL/PREEMPTED且未完成的分支申请一次续跑。`geometry_recovery.py`在额度释放后尝试60／40／30／20分钟的预算，保存每个拒绝和成功回执；只针对明确的额度拒绝尝试更小预算，其他提交异常停住，不重复创建不确定的任务。额度释放最多等8分钟，仍不足则记录待处理；取消和训练错误不重试，第二段结束后不再自动提交。改变的是Slurm预留和提交时机，Python训练参数、步数与中断恢复规则保持原样。

### 取图方式对照的FT3D阶段

已完成整图FT3D短程的三个模型均在第8,000步取得最低监控EPE。
`prepare_ft3d_deployment.py`固定这些检查点，沿用已验收FC2审计的1041/845/196对和64对FC2训练校准。
共同208×160及S/L224×160共五配置；原生FP32、TFLite FP32、INT8分别全量评分，并保留原转换阈值。
`summarize_deployment.py`识别FT3D协议，复算逐图指标并与同尺寸FC2候选配对比较；旧八配置逻辑保留。
Vela沿用既有配置并明确列出五个case，编译前评分和实机验证分开。本步骤不更新权重或BN、不延长训练。

2026-10-04用户批准：Edge/S/L各从上述FC2随机取图第10,000步继承模型和BN，重置Adam；各自分成FT3D整图与随机取图两条，共六条。三个FC2起点也恰好是各自FC2验证最低点，但不按Sintel重新挑选。共同输入208×160、batch32合并尾批，保持相同损失、BN、样本顺序和随机种子42。只改变FT3D取图方式：整图AREA缩放，随机组沿用20%整图＋80%共同裁剪后缩放；没有新增颜色或其他增强。

每条10,000次更新，逐步余弦3e-6降至1e-6，每1,000步验证640对FC2及845对Sintel。FT3D用既有80,578对TRAIN数据（Clean＋Final、左相机、前后方向）；入口核对每对源尺寸为960×540，裁剪和光流缩放系数随源尺寸计算，不沿用FC2的512×384。先在本机、再在服务器做两组各5步及中断恢复验收；正式第0步必须复现FC2起点。对应入口增加`--phase ft3d`，既有FC2默认行为保留。完整产物在`Runs/LOWRES-BENCH-01/ft3d-geometry10k-20261004`。

比较固定末步、后五次验证中位数、各自相对起点的改善及三模型差距，不只比较最佳。2026-10-05调整顺序：其余候选上板数值和速度复核放到最终选模后，不作为FT3D开训门槛；已有S部署修复和失败证据保留。训练前的来源、单位、第0步评分、GPU执行与恢复短跑检查仍须通过。较强旧Edge单列为已训练方法的部署参考，不能替换六条中的共同起点。Slurm时限与GPU选择须在提交前按实际资源和SAM余额核对；不把旧的4小时预留直接照搬。

同一包装入口也支持Sofia已经验收的TensorFlow25.02环境，路径为`$HOME/Software/MCUFlowNet/environments/tf2502-v2`；拒绝未知集群，恢复查询使用实际集群。H200任务按官方规则申请每卡24 CPU、不覆盖内存、不传`--export=ALL`。若从Mindwell换到Sofia，六条正式对照统一使用H200，先重新通过三模型服务器短跑；硬件和实际任务号写运行回执，不改变取图、损失或更新预算。`geometry_recovery.py`默认用于Mindwell，并支持下段所述wICE；Sofia只能使用明确最多一次的同集群续跑，不能调用Tier2的计费与提交分支。

2026-10-05增加wICE A100入口：wICE与Mindwell均显式使用GPFS上的同份TensorFlow25.02容器及环境；实际节点访问与三模型恢复短跑通过后才启动正式任务，不改科学配方。`geometry_recovery.py --cluster wice`仅在wICE的`gpu_a100`续跑，省略仍为Mindwell/B200；核对提交记录中的集群，Sofia不使用此Tier2恢复器。按用户偏好在A100能较快分配时优先A100，一组六条保持同型号GPU，等待时间以实际队列为准。

### 既有较强Edge与新权重的实机验收

既有继承适配Edge的FT3D第1轮`best_monitor/model`单列为部署参考。`audit_deployment.py`的case设`edge_public:true`，导出增加`--edge-public`，保留作者0–255 BGR输入与固定BN；输出仍为真实输入像素位移。沿用共同1041／845／196对及64对FC2 TRAIN校准。`compile_deployment.py --case reference-edge-208`选择这一额外导出，不替换旧五配置；`summarize_reference.py`独立归约逐图记录，核对清单、源权重、浮点转换及Vela文件身份。训练历史不同，不能当作同预算架构对照。

`board_fixtures.py`从640对FC2验证清单固定取第0／320／639对，生成量化输入与CPU TFLite INT8参考输出。标签不参与参考生成或量化校准。三个输入／输出顺序写入独立Flash槽，避免占用固件SRAM；SDK的`prepare_optical_bench.py --export-report ... --fixtures ...`读取实际量化参数及输入规则，不再仅按模型名猜测。输出乘数为1，不能继承旧MCU的12.5。

`check_board_reference.py`独立重算已保存的三个INT8参考，分别用不启用委托的优化CPU算子和参考CPU算子，保留逐对整数差异；同时核对源／Vela模型的I/O与输入、参考输出SHA。它不执行NPU，不能替代实机验收，也不能据此声称两种整数算子必然逐字节相同。板端出现数值失败时保存原日志，排查缓存、转换与算子差异，不直接放宽通过标准。

板端先核对模型与固定输入CRC、INT8 I/O、三组输出差值，再在保留相机／JPEG内存配置的条件下，使用同一输入预热5次、计时20次纯Invoke。固定输入的统一初始数值界限为最大分量差不超过2个量化档位、平均不超过0.05档位；不通过就保留差值并排查，不能把失败改成通过。三组小样本接近不等于全部1041对的板端EPE已测；离线INT8 EPE、实机推理速度、相机总耗时和Vela估计分开报告。完整产物放在既有`geometry10k-20261004/deployment-followup`，不新增wiki实验页。

### ECA部署偏差的等价调整与核验（2026-10-05）

共同训练S208的三组板端输出与两套CPU均有大幅差异。中间输出诊断发现：编码器、通道平均值、通道卷积及恢复形状前的sigmoid均精确，恢复形状后的系数异常。`board_activation_diagnostic.py prepare/check`可截取两个内部输出并校验完整UART；内部整数档不能当成光流EPE。额外输出会改变编译布局，定位后须回到完整模型验证。

`reshape-before-sigmoid`将ECA中的`sigmoid→reshape`改为`reshape→sigmoid`。全部常量、输入输出量化及原测试数据保留；临时张量现在保存变形后的卷积值，继承原卷积量化。两套CPU各用三对真实输入和四个额外输入验证，14份完整光流输出逐字节相同。该工具要求显式算子编号并检查连接关系，不按模型名称猜编号；输出目录须不存在。

```text
python tools/lowres/board_activation_diagnostic.py reshape-before-sigmoid --export <original-export-directory> --fixtures <original-fixtures-directory> --sigmoid-op 16 --reshape-op 17 --out <new-export-directory>
```

上述16／17仅适用于已核验的这份S208图。全图重新Vela编译并烧录后，`check_board_reference.py --board-log <uart.bin>`核对模型CRC、全部输入CRC及完整输出，再分别比较两套CPU。该S208三对与`BUILTIN_REF`逐字节相同，大幅偏差消除；原优化CPU参考仍有小差异，原2／0.05档门槛仍失败。保留两项事实及原失败，不换参考后覆盖旧结果。证据在`deployment-followup/board/random-S-208/sigmoid-reshape-reorder-v1/`；该局部验收不能证明所有模型、尺寸或全量Sintel都已通过，也不能解释FP32训练排名。

### 新权重的全量评分与PTQ验收（2026-10-04）

`audit_deployment.py prepare`从本机完整Sintel构建1041对清单，核对原845对包含其中；六份取图对照末步权重各测208×160，随机组S/L另测224×160。评分继续使用中心416×1024全部像素，图片AREA缩放、BGR归一化，预测LINEAR还原并分别换算u/v；不乘旧12.5、不截断GT或预测、不更新BN或优化器。每图保存原图EPE，按场景及原始GT运动大小（<10、10–40、≥40）归约；845与额外196分别报告，均是开发评测，不称盲测。

`export_deployment.py`使用同一份当前模型图、推理BN和batch1，保留原四通道输出头及最终u/v切片。三模型共享64对固定等距FC2 TRAIN整图校准；224×160重新校准，不仅修改旧TFLite形状。原生GPU卷积、精确加载、源文件及BN不变、冻结图一致性、全部64对FP32转换差异和全整数INT8张量均验收，具体差异及SHA留export.json。原生FP32使用5060 Ti，TFLite浮点与INT8使用CPU；INT8 EPE来自编译前的TFLite，不能冒充板端评分。旧公开权重的输入范围、BN和12.5规则不能套用。

运行示例（容器路径；通过`tools/setup/run-local.ps1`启动）：

```text
python tools/lowres/audit_deployment.py prepare --data /datasets --experiment /runs/LOWRES-BENCH-01/geometry10k-20261004 --out /runs/LOWRES-BENCH-01/geometry10k-20261004/deployment-audit
python tools/lowres/run_deployment_audit.py --data /datasets --audit /runs/LOWRES-BENCH-01/geometry10k-20261004/deployment-audit --phase native
python tools/lowres/run_deployment_audit.py --data /datasets --audit /runs/LOWRES-BENCH-01/geometry10k-20261004/deployment-audit --phase quantized
```

同一个控制器内顺序执行，任何子进程失败停止并保留日志，不自动覆盖失败产物；已有完成证据可跳过。`--phase exports`仅导出，`--only-case random-L-224`限定一个独立配置；可将独立GPU导出与CPU评分重叠，但不得同时操作同一配置或并发争用GPU。运行配置、完整逐图结果、转换模型和日志放同一Runs目录，不进入Git；状态及结果仅更新已有wiki实验记录和Benchmark总表。不自动启动FT3D训练。

转换验收统一要求：64对校准图每对最大u/v差不超过1e-3输入像素，平均向量差不超过1e-4输入像素；全1041对原生／TFLite FP32逐图EPE最大绝对差小于1e-3原图像素、绝对差均值小于1e-4。最初1e-4的最大分量门限被CPU后端差异触发，冻结TF CPU与TFLite CPU复查确认同样微小差异后，采用上述双重界限；原失败报告保留，INT8误差另计。GPU核验使用实际执行分区记录，不启用本机不兼容的CUPTI FULL_TRACE；最初崩溃日志也保留。

`compile_deployment.py`在已有独立Vela环境运行，五份量化文件均保持原四通道累加图，最后才切出u/v，不替换旧板端的双通道累加图。配置固定为已核验的`Runs/MCUFlowNet/GROVE-INT8-01/scan/grove-1p4mib.ini`（SHA256：`a07260cb487d49de1f93c93295ec9959ede034d6e847bb5aa92fe6650131e2da`），Ethos-U55-64、400MHz、Size、arena_cache_size=1,468,006 B。该预算不等于板卡2MiB总SRAM或固件可用arena。Windows复跑示例：

```powershell
& C:/00Work/Envs/stdc1-seg-vela/Scripts/python.exe tools/lowres/compile_deployment.py --audit C:/00Work/Runs/MCUFlowNet/LOWRES-BENCH-01/geometry10k-20261004/deployment-audit --config C:/00Work/Runs/MCUFlowNet/GROVE-INT8-01/scan/grove-1p4mib.ini --config-sha256 a07260cb487d49de1f93c93295ec9959ede034d6e847bb5aa92fe6650131e2da
```

编译输出目录已存在时拒绝覆盖；各项失败独立保留。五配置的输入SHA必须与INT8评分一致，报告CPU算子、SRAM峰值及**估计**FPS，新模型的上板数值、内存与实测FPS另验。

`summarize_deployment.py --audit <audit>`只做CPU归约，不导入TensorFlow或重新推理：重算全部18份逐图报告的1041／845／196对均值、场景与运动分组，核对样本、源权重、导出、转换验收；若已有Vela报告，也核对五份编译的身份及产物SHA。报告保存在同目录`summary.json`与`summary.md`，不以场景bootstrap代替独立训练种子。

### FC2方向损失对照（2026-10-06）

Edge/S/L从各自FC2随机取图10,000步端点开始，原损失和方向加权各一条、共六条5,000步。原组u/v=(1,1)，加权组=(1.30879345603272,0.6912065439672802)，按Sintel评分几何归一化；逐方向加权普通L1、不确定性重建与正则，三尺度权重不变。batch32、原FC2随机取图、逐步余弦3e-6→1e-6、Adam重置、FP32/TF32关闭、seed42，每1,000步正式计分和保存全部恢复状态。没有FT3D、结构变化或新的数据增强，两个损失的数值不能直接横比；输入EPE与原图EPE仍按共同协议。

复用`geometry_compare.py --fc2-source-step 10000 --direction-weights ...`；旧入口默认行为和旧恢复配置保留。`test_direction_loss.py`用已知向量核完整损失/梯度，权重(1,1)必须与原图完全一致；`verify_geometry.py --direction-compare`核两组相同数据及裁剪、不同优化轨迹、模型/BN初始化、Adam重置、实际GPU反向及中断恢复所有变量。`direction_compare.sh`提供Tier2准备完成后的probe/train数组入口；三组probe全部通过才放行正式六条。

`geometry_recovery.py --direction-compare`在首段释放后检查六条：已完成不提交；仅TIMEOUT/NODE_FAIL/PREEMPTED且未完成才提交一次剩余步数的续段；训练错误、取消或记账不明交人工检查。GPU分区沿用首次提交，方向对照的总目标保持5,000步，续段不重置学习率或Adam。

Tier2干净作业环境使用`--export=NIL`、登录shell初始化。软件路径从已核实的实验绝对路径解析；不要假定`VSC_SCRATCH_GPFS1`一定含个人目录，干净批处理环境可能只给出`/gpfs1/scratch`。续跑沿用提交记录的环境导出方式，旧记录默认保持ALL。

### 评分几何分解（2026-10-06）

`tools/lowres/audit_score_geometry.py`对Edge/S/L的FC2随机10k及整图FT3D10k两个端点做六配置推理，沿用原845对Sintel Final监控。输出依次为输入网格EPE、同网格换原图单位的EPE、预测与缩小GT共同恢复后的EPE、正式原GT EPE，以及横纵MAE和场景均值。后三项为原图像素，第一项为208×160像素；缩小GT回放只是参照，不能当理论误差下界或直接相加分摊贡献。

先以`--smoke`验收固定0/320/639三对，再全量执行。Windows工作树的Git指针不能直接在Docker里解析，因此在本机确认干净提交后冻结运行源码SHA，以`--code-manifest`提供，容器在开头和结束逐文件核对。每份权重必须精确恢复、实际GPU卷积执行、独立原评测入口一致、BN及全部推理状态不变，完整845对的输入/原图端点评分均须复现至2e-5。输出留外部Runs，源权重不写入；结果只决定下一训练建议，不自动启动方向加权训练。

### FT3D参数与BN统计的固定离线诊断（2026-10-06）

`audit_domain_bn.py`只对已有FC2随机第10,000步和整图FT3D第10,000步权重做推理。每个模型四个固定组合：A为FC2参数／FC2统计，B为FT3D参数／FT3D统计，C为FT3D参数／FC2统计，D为FC2参数／FT3D统计。只交换同一模型的moving_mean与moving_variance；gamma/beta、ECA、Gate和卷积属于参数组。所有赋值逐元素核验，推理前后检查全部模型和优化器变量指纹；不执行优化器，不保存新checkpoint，源文件SHA须不变。C/D仅作敏感性诊断，统计与特征可能不适配，不能自动替换正式benchmark。

准备阶段冻结FC2 val640、Sintel monitor845原清单，从FT3D TEST左相机按Clean/Final×future/past各160对、固定seed20261006选640对。序列随机排序后轮流选取，避免前640对集中在少数序列；只按文件可读性、有限标签和540×960尺寸剔除，记录拒绝原因，不按EPE选图。与TRAIN清单不交叉，记录新TEST文件SHA、数量、来源及清单SHA。该子集是开发诊断，不声明盲测。FC2与FT3D主EPE为208×160输入像素；Sintel为原416×1024像素，另列输入像素EPE；运动分组统一在160×208输入网格上按GT <2、2–8、≥8像素计分，不能冒充原图分组贡献。

`run --local-smoke`用本机两对FC2和两对Sintel逐项对照原生产评测入口，不含FT3D、不是实验成绩。服务器`run --smoke`再检查三个真实split；正式A/B须复现原完整FC2/Sintel记录至2e-5。使用不含CUPTI采样的执行分区信息核验GPU卷积。准备、probe和正式三个阶段必须依次成功；程序保留失败目录并拒绝覆盖输出。

```text
python tools/lowres/audit_domain_bn.py prepare --data /datasets --experiment /audit/experiment-source --out /audit/prepared --code-commit <pinned-commit>
python tools/lowres/audit_domain_bn.py run --data /datasets --prepared /audit/prepared --model S --out /audit/results/S --code-commit <pinned-commit>
```

`domain_bn.sh`复用Sofia已验收的home软件环境，数据、代码及源权重均只读绑定。按每卡24CPU、不覆盖内存、不传export=ALL申请H200；具体账户、任务号和排队情况写运行回执，不写成实时环境说明。三模型各一条、每条四组合，可并行评测，完整结果存LOWRES-BENCH-01/bn-domain-20261006，wiki维护原4n，不另开页面。

### FC2保留与FT3D混合微调（2026-10-07）

每个模型继承自己的FC2随机取图第10,000步模型与BN，重置Adam。六条新训练为Edge/S/L各一条纯FC2和一条混合；混合每次更新固定24对FC2＋8对FT3D，FC2沿用随机取图、FT3D沿用整图缩放。两套数据独立洗牌，读完一遍后接下一遍，不补复制样本；混合批次始终32对。纯FC2保留原尾批处理，每遍最后24对；旧纯FT3D参照保留原末尾合并34对，不能声称所有单数据集批次都恰好32。

208×160、FP32、TF32关闭、真实缩放像素标签不截断，共同原损失与正常训练BN。学习率按固定10,000步余弦从3e−6降至1e−6，**先停止于5,000步**，此时约2e−6。`--stop-after-steps`不改变完整日程；`status.json`的`pilot_completed`表示获批阶段结束，`completed`仍为false。后续至10,000步须审阅，不由资源恢复脚本决定。

```text
python tools/lowres/geometry_compare.py --model S --geometry random --phase fc2 --fc2-source-step 10000 --replay-arm mixture75_25 --data /datasets --manifests <frozen-manifests> --source <S-FC2-random10k> --out <new-run> --steps 10000 --stop-after-steps 5000 --eval-every 1000 --initial-lr 3e-6 --min-lr 1e-6 --seed 42 --code-commit <host-verified-commit>
```

`replay_data.py`预读一批，交付的是当前已消费批次后的两套游标；保存状态不能提前包含下一批。`verify_geometry.py --replay-compare --reference-repo <10ef4f0-checkout>`用真实数据核三配方的连续5步与3＋2恢复，跨过两套小清单末尾；要求所有变量、历史数值、顺序及裁剪指纹精确一致，并核模型／BN继承、Adam重置、实际GPU反向。纯FT3D短跑还须与旧提交的全部变量相同，才复用旧训练参照。

每1,000步验证FC2 val640、Sintel monitor845及锁定FT3D TEST640；第0／3,000／4,000／5,000步补评分清单中其余196对，按845＋196合并为同1041对原图EPE。`score_replay_reference.py`仅推理旧纯FT3D 3k／4k／5k权重，复现原845分数并补同两套验证和全量评分，不更新参数或BN。提交前必须核验**全部**验证文件，不能只检查前若干项就假定1041对齐全。

`replay.sh`提供三模型验收、六条训练及三条旧参照评分入口。`geometry_recovery.py --replay-compare`至多为TIMEOUT/NODE_FAIL/PREEMPTED安排一次续段，目标仍为5,000步；训练错误或取消不自动重跑。配置、权重及完整日志放外部Runs，Git仅保存代码与方法说明；wiki沿用实验记录4o。独立seed43／44按用户决定暂缓，本次实现核验的重放不作为科学实验重复。

用户批准后，`replay.sh continue`在原六个输出目录用`--resume --stop-after-steps 10000`继续；模型、BN、Adam、global_step及混合数据游标都继承，逐步余弦仍是原10k日程。训练器与数据／模型／损失代码必须逐文件匹配5k提交；只新增调度与离线评分。准备阶段将原current/status/metrics及来源指纹冻结到`control/continue10k/pilot-snapshot`，保留5k结果，旧检查点不覆盖。新调度代码、回执及READY独立留`control/continue10k`，不改旧提交记录。

原训练器仍只在0／3／4／5k补全1041评分，保证恢复配置完全相同；10k阶段按`replay.sh score10k`另对六条新权重及三模型旧纯FT的8／9／10k点统一评分。`score_replay_reference.py --folder <completed-run> --steps 8000 9000 10000`读取冻结模型并复现对应845分数，不更新参数或BN。`summarize_replay.py --end-step 10000`核完整曲线与全量评分，用后三点和末步比较，不拿5k三点代替10k结果。旧默认5k汇总可从冻结快照读取。

资源恢复使用`geometry_recovery.py --replay-compare --replay-end-step 10000`，只恢复至已经批准的10k终点；10k完成标记不能由5k的pilot_completed代替。`audit_replay_checkpoints.py --end-step 10000`在CPU检查66份保存状态和六条末步完整恢复。代码、结果验收和全量归档完成后，再按共同的64对FC2 TRAIN校准及现有Vela预算核候选，不把原生FP32成绩当作INT8部署成绩。

`prepare_replay_deployment.py`以通过验收的5k/10k固定末尾候选，在208×160完整1041原生FP32中选每模型较低误差的权重，平手用较早端点；不选8k/9k、场景最佳或INT8最佳。S/L224沿同一选中权重。它复用原FC2部署审计的完整／监控／64校准清单与源父权重指纹，冻结选择、代码提交和源检查点，再沿既有原生、导出、浮点／INT8评分、Vela与配对归约入口执行。`summarize_deployment.py`的replay分支仍核15份逐图报告、5导出及5同量化文件编译，原转换门槛、预算与零CPU要求不变；旧FC2与FT3D结果的归约和显示保留。

部署比较同时保留公开适配Edge和纯FT8k增强Edge，同1041逐图身份必须匹配；不同尺寸和训练历史单列，FP32与INT8不能分别取不同权重拼成一行。Vela速度是估计、配置内存不是固件可用arena，新模型编译通过不代替板端精度或计时。原S/L图在板端的ECA问题须按等价修复流程另验，不能仅凭电脑端INT8成绩跳过。

### 混合10k之后的配对学习率对照（2026-10-07）

`replay_lr_compare.py`从每模型已完成混合10k的完整checkpoint分为fixed/restart两条。完整模型、BN、Adam槽、beta powers、global_step及源游标精确继承；新阶段phase_step=global_step−10000，不重算旧日程。fixed恒1e−6，restart按预定新增10k余弦3e−6→1e−6；先各新增5k到global15000停止，不能默认接global20000。208×160、24FC2＋8FT3D、原几何、标签与损失不变；每1k按同1041原GT完整计分，另列原845，FC2val640和锁定FT3D TEST640，验证不更新BN。

`verify_replay_lr.py`用真实25/9小清单和两对验证检查两政策的连续5步与3＋2恢复；游标仅在probe按源位置取模以跨小清单末尾，正式起点保留[FC2:11,17680;FT3D:1,80000]。要求所有初始变量与10k源完全相同、Adam保留非零状态、global_step从10000继续、两政策数据／几何相同、恢复全部变量和数值一致、实际GPU反向执行。小清单重放是实现检查，不是独立seed实验。正式第0步必须复现四项完整源分数至2e−5。

`replay_lr.sh`验收三个模型后放行六条；输出在独立Runs实验根，原10k目录只读。`geometry_recovery.py --lr-compare`只为未完成TIMEOUT/NODE_FAIL/PREEMPTED安排一次剩余步数续段，止于global15000。完整配置、计分、恢复状态和回执留Runs，wiki继续原研究问题4o，不因政策或提交作业开页。
