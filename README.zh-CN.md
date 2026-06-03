<div align="center">

# MCUFlowNet

**面向 Cortex-M55 + Ethos-U55 微型 NPU 的硬件感知光流神经架构搜索**

[English](README.md) · 简体中文

</div>

---

MCUFlowNet 是硕士论文《面向 Grove Vision AI V2 的硬件感知光流神经架构搜索》的公开代码库。
它在一块微控制器级 NPU（ARM Cortex-M55 + Ethos-U55，仅支持 INT8、约 1.4 MB SRAM arena）上
搜索真正能放得下、跑得动的稠密光流网络，采用**三层硬件–软件协同设计**：

1. **L1 — 宏观骨干重设计**：用双线性（resize + 卷积）解码器替换转置卷积解码器以降低 SRAM 峰值，
   再把两个轻量注意力模块（瓶颈处 ECA + 全局广播 gate）塞进释放出的内存里。
2. **L2 — 搜索空间与超网**：一个 11 维、编译器可行的搜索空间（23,328 个候选），以权重共享超网 +
   公平采样器训练；每个候选只有在 Vela 编译器能在 SRAM 上限内调度时才被纳入。
3. **L3 — 硬件在环搜索**：NSGA-II 基线与 LLM agent 系统（warm-start / scientist / supervisor）的混合，
   优化 (Sintel EPE, Vela 预测 FPS) 帕累托前沿，帧率由实际编译每个候选得到。

## 结果

在 Grove Vision AI V2 上 172×224 的同硬件对比（Vela 预测 FPS），与
[EdgeFlowNet](https://github.com/pearwpi/EdgeFlowNet) 基线相比：

| 模型 | Sintel Final EPE ↓ | FPS ↑ | FPS / W ↑ | 硬件 |
|------|:---:|:---:|:---:|:---:|
| EdgeFlowNet 基线 | 6.31 | 5.3 | — | Ethos-U55 |
| **MCUFlowNet-L** | **4.89** | 5.3 | 15.2 | Ethos-U55 |
| **MCUFlowNet-S** | 5.58 | **9.1** | 26.1 | Ethos-U55 |

- **MCUFlowNet-L** 在基线相同帧率下把 Sintel Final EPE 降低 **22%**。
- **MCUFlowNet-S** 在比基线低 **12%** 的 EPE 下达到 **高 72%** 的 FPS。

## 发布的两个模型

| 名称 | 角色 | 架构编码（11 维） |
|------|------|---------------------|
| **MCUFlowNet-L** | 较大模型，帧率对齐 EdgeFlowNet 基线 | `2,0,0,2,2,1,0,0,0,0,0` |
| **MCUFlowNet-S** | 最轻的帕累托端点 | `0,0,0,0,0,0,0,0,0,0,0` |

## 仓库结构

```
MCUFlowNet/
├── EdgeFlowNAS/              # 我们的 NAS 框架（论文核心贡献）
│   ├── efnas/               #   核心包：network / nas / search / engine / data / ...
│   ├── configs/             #   超网/搜索/重训/消融/部署的 YAML 配置
│   ├── wrappers/            #   命令行入口（run_*.py）
│   ├── scripts/             #   绘图 / 数据检查 / HPC 辅助脚本
│   ├── tools/               #   辅助工具（如 Sintel demo 微调）
│   └── outputs/             #   curate 后的结果文件（CSV/JSON/图）；权重在外部
├── EdgeFlowNet/             # 修改版 MIT 基线 + 我们的硬件剖析工具
│   ├── sramTest/            #   Vela/SRAM 基准 + 论文出图脚本（make_thesis_ch*.py）
│   ├── MODIFICATIONS.md     #   我们对基线的确切改动
│   └── LICENSE              #   上游 EdgeFlowNet MIT 许可证（原样保留）
├── ATTRIBUTION.md           # 哪些是我们的、哪些是 EdgeFlowNet 基线
├── LICENSE                  # MIT
└── .env.example             # API key 模板（仅 L3 LLM 搜索需要）
```

## 环境配置

训练与评估使用 **TensorFlow 2.15**（超网/重训代码是 TF1 图风格，在 TF2 下运行）。
器件可行性与 FPS 由 Ethos-U55 的 **ARM Vela** 编译器给出。

```bash
# （示例）conda 环境
conda create -n mcuflownet python=3.11
conda activate mcuflownet
pip install "tensorflow==2.15.*" pymoo matplotlib tqdm pyyaml
pip install ethos-u-vela          # 用于 L2 准入与 L3 FPS 信号的 Ethos-U55 编译器
# 仅 L3 LLM agent 搜索需要：
pip install anthropic openai
cp .env.example .env              # 然后填入你自己的 API key
```

### 数据集

请自行下载并在配置里指向：

- **FlyingChairs2 (FC2)** — 超网/消融训练：https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs2
- **FlyingThings3D (FT3D)** — 重训：https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
- **MPI-Sintel** — 跨域评估：http://sintel.is.tue.mpg.de/downloads

## 预训练权重

训练好的 checkpoint 托管在 Google Drive（太大无法进 git）：
**[MCUFlowNet_checkpoint](https://drive.google.com/drive/folders/1M598SgCXy6i3bcrOnD88zv5tp30RHeoF?usp=sharing)**。
下载每个模型的子文件夹，把其中的 checkpoint 文件解压到下表路径
（每个子文件夹内也有 `README.txt` 注明架构编码）：

| Drive 子文件夹 | 解压到 |
|----------------|--------|
| `MCUFlowNet-L/` | `EdgeFlowNAS/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1/model_v3_efn_fps/checkpoints/` |
| `MCUFlowNet-S/` | `EdgeFlowNAS/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1/model_v3_light/checkpoints/` |
| `MCUFlowNet-Supernet/` | `EdgeFlowNAS/outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill/checkpoints/` |

或者一键自动下载并放置三个权重：

```bash
pip install gdown
python download_weights.py
```

## 复现流程

每个阶段都有命令行 wrapper（加 `--help` 查看全部参数）。代表性命令：

```bash
cd EdgeFlowNAS

# L1 — 宏观骨干消融（第 4 章）
python wrappers/run_ablation_fc2.py     --config configs/ablation_fc2.yaml     --experiment_name ablation_fc2

# L2 — 训练权重共享超网（第 5 章）
python wrappers/run_supernet_train.py   --config configs/supernet_fc2_172x224.yaml

# L3 — 混合 NSGA-II + LLM agent 搜索（第 6 章）；需要 .env 里的 API key
python wrappers/run_nsga2_search.py     --config configs/search_nsga2.yaml      --experiment_name search_run

# 从头重训选中的子网（第 7 章）—— 例：MCUFlowNet-S
python wrappers/run_retrain_ft3d.py     --config configs/retrain_ft3d.yaml --arch_code "0,0,0,0,0,0,0,0,0,0,0"

# 部署分辨率微调（第 7 章）
python wrappers/run_deploy_ft_one.py    --config configs/deploy_ft.yaml --model_name MCUFlowNet-S --arch_family fixed_v3 --arch_code "0,0,0,0,0,0,0,0,0,0,0"
```

硬件剖析与所有论文插图由 `EdgeFlowNet/sramTest/`（`make_thesis_ch*.py`）生成，
它们读取 `EdgeFlowNAS/outputs/` 下 curate 后的 CSV。

## 引用

如果你使用本代码，请引用本论文及其所基于的 EdgeFlowNet 基线：

```bibtex
@mastersthesis{lin2026mcuflownet,
  author = {Lin, Enmin},
  title  = {Hardware-Aware Neural Architecture Search for Optical Flow on the Grove Vision AI V2},
  school = {KU Leuven},
  year   = {2026}
}

@article{Raju2024EdgeFlowNet,
  author  = {Raju, Sai Ramana Kiran Pinnama and Singh, Rishabh and Velmurugan, Manoj and Sanket, Nitin J.},
  title   = {EdgeFlowNet: 100FPS@1W Dense Optical Flow For Tiny Mobile Robots},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2024},
  doi     = {10.1109/LRA.2024.3496336}
}
```

## 许可证与署名

MIT — 见 [LICENSE](LICENSE)。本仓库在 `EdgeFlowNet/` 下捆绑了
[EdgeFlowNet](https://github.com/pearwpi/EdgeFlowNet)（© 2024 PeAR Group, WPI）的修改版（同为 MIT）；
详见 [ATTRIBUTION.md](ATTRIBUTION.md) 与 [EdgeFlowNet/MODIFICATIONS.md](EdgeFlowNet/MODIFICATIONS.md)。
