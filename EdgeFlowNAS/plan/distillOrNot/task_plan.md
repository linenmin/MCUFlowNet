# Task Plan: Distill-Or-Not Same-FPS Short Retrain Probe

## Goal

Run a controlled short-retrain experiment for 8 V3 subnets selected from the two V3 NSGA-II runs in:

`D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/nsga2_v3`

The experiment asks:

> Within the same FPS/hardware niche, which supernet's EPE ranking is closer to a scratch short-retrain ranking: distill or no-distill?

This is a rank-calibration probe. It does not test whether inherited distill weights can be deployed directly.

## Candidate Source

The current candidate list is:

`D:/Dataset/MCUFlowNet/EdgeFlowNAS/plan/distillOrNot/same_fps_rank_gap_top8.csv`

It is copied from:

`D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/nsga2_v3/same_fps_rank_gap_probe_20260430/same_fps_rank_gap_top8.csv`

Selection rule:

- Align common architecture codes from the distill and no-distill V3 NSGA-II `history_archive.csv` files.
- Restrict comparison to a contiguous FPS window with <= 5% relative span.
- Compute EPE rank inside that FPS window separately for distill and no-distill.
- Select the 8 subnets with the largest within-window EPE rank gap.

Selected FPS niche:

- Window FPS range: `6.589863 - 6.895185`.
- Window size: 21 common architectures.
- Selected FPS range: `6.597428 - 6.895185`.
- Selected FPS relative span: `4.45%`.
- Selected rank-gap range: `8 - 10`.

## Fixed Training Requirements

| Requirement | Current Decision |
| --- | --- |
| Dataset | FlyingChairs2 |
| Input resolution | `172x224`, matching V3 supernet/search resolution |
| Candidate count | 8 same-FPS rank-gap subnets |
| Initialization | Scratch/common random initialization |
| GPU usage | One Slurm job using 4 GPUs, one architecture process per GPU |
| CPU loading | FC2 threaded loading plus bounded prefetch |
| Epochs | 50 |
| FC2 validation | Every epoch |
| Sintel validation | Every 5 epochs |
| Learning rate policy | Adam, cosine LR `1e-4 -> 1e-6` |
| Output root | `outputs/distill_or_not_fc2_short` |
| Experiment name | `distill_or_not_same_fps_rank_gap_run1` |

## Implemented Code Surface

- `efnas.network.fixed_arch_models_v3.FixedArchModelV3`
  - selected-only hard-routed V3 model
  - 11D architecture code
  - bilinear decoder
  - bottleneck ECA
  - 1/4 global gate
  - multi-scale outputs

- `efnas.engine.distill_or_not_trainer`
  - single-architecture scratch FC2 training
  - multi-scale uncertainty loss
  - gradient accumulation through `micro_batch_size`
  - FC2 validation with progress bars
  - optional Sintel validation
  - saves `last`, `best`, and `sintel_best`

- `wrappers/run_distill_or_not_fc2_one.py`
  - trains one candidate on one visible GPU

- `wrappers/run_distill_or_not_fc2_batch.py`
  - launches candidates across multiple GPUs
  - defaults to 4 GPUs and the same-FPS top-8 CSV

- `plan/distillOrNot/distill_or_not_fc2_short_4gpu.slurm`
  - HPC entrypoint for the 8-subnet experiment

## Why One Process Per Architecture

The launcher intentionally does not put all candidates into one TensorFlow graph.

Instead, each child process:

- owns one architecture
- sees one GPU through `CUDA_VISIBLE_DEVICES`
- builds one TF1 graph/session
- has its own FC2 providers and prefetch queue

This is different from the older retrain code that trained multiple models in one graph and shared batches. That old pattern was useful for one-GPU pairwise comparison, but it does not scale cleanly to 4 GPUs because TensorFlow 1 graph/session placement, checkpointing, failure isolation, and GPU memory ownership become tightly coupled.

The current design favors:

- clean multi-GPU utilization
- independent checkpoint/resume per candidate
- failure isolation
- lower per-process graph memory
- simpler Slurm scheduling

The trade-off is duplicated data loading. With `--fc2_num_workers 4` and four concurrent processes, the effective training workers are about 16 plus four Python main processes, which matches the 24-CPU Slurm template.

## Comparison After Training

After the 8 short runs finish:

1. Compute final FC2 EPE rank across the 8 subnets.
2. Compute final Sintel EPE rank across the 8 subnets.
3. Compare those ranks against:
   - distill EPE rank inside the FPS window
   - no-distill EPE rank inside the FPS window
4. Report:
   - Spearman correlation
   - Kendall correlation if available
   - per-candidate absolute rank error
   - winner count: distill closer vs no-distill closer

The expected claim, if supported, is:

> Distillation improves or does not improve supernet EPE rank calibration within a fixed hardware/FPS niche.
