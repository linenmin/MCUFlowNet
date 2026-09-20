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
