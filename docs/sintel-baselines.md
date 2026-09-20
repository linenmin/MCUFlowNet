# Sintel公开基线复测（SINTEL-BASE-01）

目标：用本机Sintel真实图像和未经截断的光流标签，重新测量表中模型的EPE。这里只记录方法和来源；进度统一在Obsidian实验总表。

## 文件放在哪里

- 独立工作树：`C:/00Work/Code/MCUFlowNet-baselines`；分支`validation/sintel-baselines`。
- 上游代码：`C:/00Work/Code/optical-flow-upstream`，各自保留Git来源；不把上游代码和权重重复提交到本仓库。
- 输出、下载、完整环境清单：`C:/00Work/Runs/MCUFlowNet/SINTEL-BASE-01`。
- 数据只读：`C:/00Work/Datasets/Sintel`。
- Conda：`C:/00Work/Envs/sintel-torch`和`sintel-tf`。使用conda-forge，不修改base，不代用户接受Anaconda默认源条款。

## 来源（2026-09-20访问）

| 方法 | 作者代码及权重入口 | 需要明确的区别 |
| --- | --- | --- |
| RAFT | https://github.com/princeton-vl/RAFT ，download_models.sh的Dropbox压缩包 | 区分Things与Sintel微调权重；不能用训练过Sintel的模型冒充跨数据集泛化 |
| SPyNet | https://github.com/anuragranj/spynet | 原版Torch/Lua；README链接的https://github.com/sniklaus/pytorch-spynet明确是非官方PyTorch移植，使用时必须注明 |
| PWC-Net | https://github.com/NVlabs/PWC-Net ，PyTorch内附两份权重 | 作者说明PyTorch和Caffe结果不同；旧CUDA correlation算子需要兼容处理和数值检查 |
| Ajna | https://github.com/prgumd/ajna ，https://github.com/prgumd/ajna/wiki/RunningCode | 官方权重链接https://drive.google.com/file/d/1Eur8zt_662fN7CZTfqQnuaSHmX3yD3Kj/view ；说明为FC2 400轮再FT3D 50轮 |
| NanoFlowNet | https://github.com/tudelft/nanoflownet 的nanoflownet-cnns子模块 | 包含H5与TFLite；须核清输入112×160和输出单位，不直接拼入全分辨率成绩 |
| EdgeFlowNet Full/Chunking | https://github.com/pearwpi/EdgeFlowNet ，checkpoints/best.ckpt | 已读公开test_sintel.py和utils.py：默认416×1024，GT分量截断±50；这不证明其他模型的原表评分也截断 |

下载后运行`tools/baselines/inventory.py`保存各库提交和权重SHA256。该清单证明本次用了什么文件，不证明它们就是EdgeFlowNet表III使用的版本。

## 评测需要满足什么

首先核实上游预处理、颜色顺序、输出单位和权重结构。统一评分器读取原始.flo；不截断GT或预测，不过滤大运动、不静默跳过坏样本。保存逐图EPE、样本清单、样本数、模型参数、权重指纹、环境和实际命令。

为衔接已有S/L结果，首选相同1041对Final、中心416×1024区域。原图436×1024全图成绩必须另列；低分辨率模型须注明网络输入、预测还原方法和评分网格，不能把不同网格的像素误差当作同一个单位。Full和Chunking也不是相同推理方式。最终协议需在实际适配前固定，不能只依据原表脚注猜测所有模型均在1024宽度评分。

本轮是公开权重的重新测量，不预先承诺复现原表的4.23等数值。原表其他模型的具体权重与评测脚本尚未建立一一对应关系。Sintel训练集上的微调权重结果，应与未用Sintel训练的权重分开标注。

GPU验收必须实际运行算子。PyTorch选择CUDA12.8以适配5060Ti；Windows TensorFlow2.15.1为CPU环境，只用于兼容验证。若需要TensorFlow GPU，应使用已有WSL运行路径或另配Linux隔离环境，不能将Windows CPU推理写为GPU测量。

## 已固定的第一版评分方法

协议ID：`sintel-final-train1041-center416-raw-v1`。读取training/flow的1041个唯一文件，按场景和帧排序，取对应Final相邻两帧。原图436×1024，图像和标签共同裁去上下各10行；不使用遮挡掩码，所有像素计分。每图计算欧氏距离的像素平均，再对1041图等权平均。所有评分图同尺寸，因此等价于全像素平均。像素误差float32，汇总float64。另存历史±50 GT辅助列，主列始终未经截断。它是裁剪评测，不应写成436×1024标准全图结果。

`tools/baselines/evaluate.py`负责共用读取、计分、逐图CSV和运行清单；PyTorch模型接入函数与`tf_adapters.py`负责模型预处理和预测；`pwc_compat.py`集中保存旧算子兼容处理。`models.json`登记权重的训练数据、输入方式与能否用于正式比较，`summarize.py`核对1041个唯一样本、不同模型样本顺序相同、CSV重算与结果一致后再汇总。后续新增模型沿用评分器与登记表，不复制另一套评分程序，也不因新增模型重选MCUFlowNet权重。

| 接入 | 本次实际处理 |
| --- | --- |
| RAFT | 作者代码，RGB 0–255，内部归一化，32次迭代，FP32，关闭TF32；416×1024已整除8，不需改变尺寸 |
| SPyNet | 作者链接的非官方移植，BGR/255，经原estimate函数；416×1024已整除32。下载后的本地权重严格加载，禁止构造模型时再次下载其他权重 |
| PWC-Net | 作者PyTorch图和原权重；BGR/255，OpenCV线性插值到448×1024，预测乘20，插值回416×1024并校正向量坐标尺度。旧correlation替换为81个偏移的通道均值，逐元素标量参考检查通过；grid_sample显式恢复旧版align_corners=True |
| EdgeFlowNet Full | 作者图和best.ckpt，BGR 0–255，四通道输出取前两维，多尺度累计；全部图变量与checkpoint逐个比较完全相同；TF2.15.1 CPU、BN推理态 |
| EdgeFlowNet四块 | 同一权重，将416×1024切成4个208×512非重叠块，逐块预测后拼接，不做接缝平滑。这是本次固定的高分辨率分块方式，不冒充表III的4×176×240部署方式 |
| NanoFlowNet（暂定） | 发布nanoflownet.h5，输入缩到112×160、转灰度、(x−128)/128，第一路光流输出双线性插值回评分网格。按公开loader保留源图像素单位、不再乘缩放倍数。但作者训练NPY标签的转换来源未提供，单位尚待确认，结果不进入正式名次 |

NanoFlowNet也做过2对416×1024直接输入的工程探测；这与低分辨率主设置不同，不作排名。不能通过尝试多个输出系数并选择EPE最低者来决定单位。Ajna缺少论文对应权重，保持缺失，不用其他旧架构权重替代。

评测权重包括未用Sintel微调的主要组，以及单列的Sintel微调参考组。后者在Sintel training上的分数属于含训练数据的测量，不作为跨数据集泛化证据。其他模型与EdgeFlowNet表III权重是否相同仍未知；只对具有证据的对应关系作判断，不把所有差异归因于截断。

## 运行示例

在工作树根目录，以独立Conda环境的python直接运行，无需激活base。输出目录必须不存在，防止覆盖：

```powershell
& C:/00Work/Envs/sintel-torch/python.exe tools/baselines/evaluate.py --model raft --weights C:/00Work/Runs/MCUFlowNet/SINTEL-BASE-01/weights/raft/models/raft-things.pth --upstream C:/00Work/Code/optical-flow-upstream --dataset C:/00Work/Datasets/Sintel --output C:/00Work/Runs/MCUFlowNet/SINTEL-BASE-01/raft-things-new
& C:/00Work/Envs/sintel-torch/python.exe tools/baselines/summarize.py --runs C:/00Work/Runs/MCUFlowNet/SINTEL-BASE-01
```

TensorFlow模型改用`sintel-tf/python.exe`；NanoFlowNet低分辨率设置额外指定`--nano-native`。完整环境冻结文件在Runs的torch-freeze.txt/tf-freeze.txt。测量耗时包含并发任务影响，不作为模型FPS或能效比较。

重建环境可用`tools/baselines/environment.yml`创建两个不同名称的Conda环境，再分别安装`requirements-torch.txt`和`requirements-tf.txt`；完全复刻本次依赖应使用Runs的freeze文件。`dataset_manifest.py`一次性核对原图尺寸和所有文件SHA256；`summarize.py`还将每组逐图CSV与该数据清单核对。

参数量来自实际加载图的计数，不能直接沿用原表：当前公开实现RAFT 5,257,536、SPyNet 1,440,300、PWC-Net 9,374,340、EdgeFlowNet 2,743,804，NanoFlowNet H5总参数170,881（含辅助输出及非训练状态）。各自计数对象不同，最终论文如要比较参数量应另统一去除训练专用分支等计数规则。本轮只提供EPE复测，不复用端侧FPS形成新的性能名次。

SPyNet权重额外通过`check_spynet_weights.py`核查：Chairs/Final和Sintel/Final两套转换权重各60个张量，与作者仓库的Lua .t7数组逐个完全相等。原spynet.lua对Chairs明确让第六层复用第五层权重，所以PyTorch六套模块的参数计数包含重复存储，不能简单当作六套独立训练参数。该检查证明权重身份，不代表已在原Lua环境中逐像素验证推理一致。证据为Runs的spynet-weight-check.json；torchfile仅用于读取旧格式，不运行Lua程序。
