<div align="center">

# MCUFlowNet

**Hardware-Aware Neural Architecture Search for Dense Optical Flow on a Cortex-M55 + Ethos-U55 microNPU**

English · [简体中文](README.zh-CN.md)

</div>

---

MCUFlowNet is the public code release for the master's thesis *"Hardware-Aware Neural
Architecture Search for Optical Flow on the Grove Vision AI V2."* It searches for dense
optical-flow networks that actually fit and run on a microcontroller-class NPU
(ARM Cortex-M55 + Ethos-U55, INT8-only, ~1.4 MB SRAM arena), using a **three-level
hardware–software co-design**:

1. **L1 — Macro-backbone redesign.** Replace the transposed-convolution decoder with a
   bilinear (resize-plus-convolution) decoder to cut the SRAM peak, then add two lightweight
   attention modules (bottleneck ECA + a global broadcast gate) into the freed memory.
2. **L2 — Search space & supernet.** An 11-dimensional, compiler-feasible search space
   (23,328 candidates) trained as a weight-sharing supernet with a fairness sampler, with
   every candidate admitted only if the Vela compiler can schedule it within the SRAM cap.
3. **L3 — Hardware-in-the-loop search.** A hybrid of an NSGA-II baseline and an LLM-agent
   system (warm-start / scientist / supervisor) that optimises the (Sintel EPE, Vela-predicted
   FPS) Pareto front, with the on-device frame rate measured by compiling each candidate.

## Results

Same-hardware comparison on the Grove Vision AI V2 at 172×224 (Vela-predicted FPS), against
the [EdgeFlowNet](https://github.com/pearwpi/EdgeFlowNet) baseline:

| Model | Sintel Final EPE ↓ | FPS ↑ | FPS / W ↑ | Hardware |
|-------|:---:|:---:|:---:|:---:|
| EdgeFlowNet baseline | 6.31 | 5.3 | — | Ethos-U55 |
| **MCUFlowNet-L** | **4.89** | 5.3 | 15.2 | Ethos-U55 |
| **MCUFlowNet-S** | 5.58 | **9.1** | 26.1 | Ethos-U55 |

- **MCUFlowNet-L** lowers Sintel Final EPE by **22 %** at the baseline's own frame rate.
- **MCUFlowNet-S** reaches **72 % higher FPS** at a **12 % lower** EPE than the baseline.

## The two released models

| Name | Role | Architecture code (11D) |
|------|------|--------------------------|
| **MCUFlowNet-L** | larger model, matched to the EdgeFlowNet baseline frame rate | `2,0,0,2,2,1,0,0,0,0,0` |
| **MCUFlowNet-S** | lightest Pareto endpoint | `0,0,0,0,0,0,0,0,0,0,0` |

## Repository layout

```
MCUFlowNet/
├── EdgeFlowNAS/              # our NAS framework (the thesis contribution)
│   ├── efnas/               #   core package: network / nas / search / engine / data / ...
│   ├── configs/             #   YAML configs for supernet, search, retrain, ablation, deploy
│   ├── wrappers/            #   CLI entry points (run_*.py)
│   ├── scripts/             #   plotting / data-check / HPC helper scripts
│   ├── tools/               #   auxiliary tools (e.g. Sintel demo fine-tune)
│   └── outputs/             #   curated result files (CSV/JSON/figures); weights are external
├── EdgeFlowNet/             # modified MIT baseline + our hardware-profiling tools
│   ├── sramTest/            #   Vela/SRAM benchmarks + thesis figure scripts (make_thesis_ch*.py)
│   ├── MODIFICATIONS.md     #   exactly what we changed in the baseline
│   └── LICENSE              #   upstream EdgeFlowNet MIT license (preserved)
├── ATTRIBUTION.md           # what is ours vs. the EdgeFlowNet baseline
├── LICENSE                  # MIT
└── .env.example             # API-key template (only needed for the L3 LLM search)
```

## Setup

Training and evaluation use **TensorFlow 2.15** (the supernet/retrain code is TF1-graph style
run under TF2). On-device feasibility and FPS use the **ARM Vela** compiler for the Ethos-U55.

```bash
# (example) conda environment
conda create -n mcuflownet python=3.11
conda activate mcuflownet
pip install "tensorflow==2.15.*" pymoo matplotlib tqdm pyyaml
pip install ethos-u-vela          # Ethos-U55 compiler used for the L2 admission + L3 FPS signal
# for the L3 LLM-agent search only:
pip install anthropic openai
cp .env.example .env              # then add your own API keys
```

### Datasets

Download separately and point the configs at them:

- **FlyingChairs2 (FC2)** — supernet/ablation training: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs2
- **FlyingThings3D (FT3D)** — retraining: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
- **MPI-Sintel** — cross-domain evaluation: http://sintel.is.tue.mpg.de/downloads

## Pretrained weights

The trained checkpoints are hosted on Google Drive (too large for git):
**[MCUFlowNet_checkpoint](https://drive.google.com/drive/folders/1M598SgCXy6i3bcrOnD88zv5tp30RHeoF?usp=sharing)**.
Download each model's subfolder and extract its checkpoint files into the path below
(each subfolder also includes a `README.txt` with the architecture code):

| Drive subfolder | Extract to |
|-----------------|------------|
| `MCUFlowNet-L/` | `EdgeFlowNAS/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1/model_v3_efn_fps/checkpoints/` |
| `MCUFlowNet-S/` | `EdgeFlowNAS/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1/model_v3_light/checkpoints/` |
| `MCUFlowNet-Supernet/` | `EdgeFlowNAS/outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill/checkpoints/` |

Or fetch and place all three automatically:

```bash
pip install gdown
python download_weights.py
```

## Reproducing the pipeline

Each stage has a CLI wrapper (pass `--help` to see all options). Representative commands:

```bash
cd EdgeFlowNAS

# L1 — macro-backbone ablation (Ch.4)
python wrappers/run_ablation_fc2.py     --config configs/ablation_fc2.yaml     --experiment_name ablation_fc2

# L2 — train the weight-sharing supernet (Ch.5)
python wrappers/run_supernet_train.py   --config configs/supernet_fc2_172x224.yaml

# L3 — hybrid NSGA-II + LLM-agent search (Ch.6); needs .env API keys
python wrappers/run_nsga2_search.py     --config configs/search_nsga2.yaml      --experiment_name search_run

# Retrain a selected subnet from scratch (Ch.7) — example: MCUFlowNet-S
python wrappers/run_retrain_ft3d.py     --config configs/retrain_ft3d.yaml --arch_code "0,0,0,0,0,0,0,0,0,0,0"

# Deployment-resolution fine-tune (Ch.7)
python wrappers/run_deploy_ft_one.py    --config configs/deploy_ft.yaml --model_name MCUFlowNet-S --arch_family fixed_v3 --arch_code "0,0,0,0,0,0,0,0,0,0,0"
```

Hardware profiling and all thesis figures are generated from `EdgeFlowNet/sramTest/`
(`make_thesis_ch*.py`), which read the curated CSVs under `EdgeFlowNAS/outputs/`.

## Citation

If you use this code, please cite the thesis and the EdgeFlowNet baseline it builds on:

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

## License & attribution

MIT — see [LICENSE](LICENSE). This repository bundles a modified, MIT-licensed copy of
[EdgeFlowNet](https://github.com/pearwpi/EdgeFlowNet) (© 2024 PeAR Group, WPI) under
`EdgeFlowNet/`; see [ATTRIBUTION.md](ATTRIBUTION.md) and
[EdgeFlowNet/MODIFICATIONS.md](EdgeFlowNet/MODIFICATIONS.md).
