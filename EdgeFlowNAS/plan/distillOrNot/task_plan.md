# Task Plan: Distill-Or-Not Short Retrain Probe

## Goal

Run a controlled short-retrain experiment for the 10 V3 NSGA-II rank-disagreement subnets selected in:

`D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/nsga2_v3/frontier_top5_rank_gap_probe_20260430/top10.csv`

The goal is to test whether the V3 no-distill or distill supernet inherited ranking better predicts short FC2 retrain ranking.

## Current Phase

Planning and code-surface audit complete. Implementation is pending user confirmation on the initialization policy.

## Fixed Requirements

| Requirement | Current Decision |
| --- | --- |
| Dataset | FlyingChairs2 |
| Input resolution | `172x224`, matching V3 supernet/search resolution |
| Candidate count | 10 selected rank-gap subnets |
| GPU usage | One Slurm job/script using 5 GPUs, one architecture process per GPU |
| CPU loading | Use FC2 threaded loading plus bounded prefetch |
| Epochs | 50 |
| FC2 validation | Every epoch |
| Sintel validation | Every 5 epochs |
| Learning rate policy | Same as V3 FC2 supernet unless user chooses otherwise: Adam, cosine LR `1e-4 -> 1e-6` |
| Output root | Proposed: `outputs/distill_or_not_fc2_short` |

## Proposed Architecture

### Phase 1: V3 Fixed-Architecture Training Surface

- [ ] Add a V3 fixed-architecture model implementation for training.
- [ ] Reuse V3 semantics:
  - 11D code
  - first six dimensions with choices `0/1/2`
  - last five dimensions with choices `0/1`
  - fixed bilinear decoder
  - bottleneck ECA
  - 1/4 global gate
- [ ] Preserve multi-scale output and uncertainty loss used by current retraining.
- [ ] Save `last`, `best`, and `sintel_best` checkpoints.

### Phase 2: FC2 Short-Retrain Trainer

- [ ] Add config, proposed: `configs/distill_or_not_fc2_short.yaml`.
- [ ] Add a single-architecture trainer wrapper, proposed: `wrappers/run_distill_or_not_fc2_one.py`.
- [ ] Support:
  - `--arch_code`
  - `--model_name`
  - `--experiment_name`
  - `--gpu_device`
  - `--num_epochs`
  - `--fc2_num_workers`
  - `--fc2_eval_num_workers`
  - `--prefetch_batches`
  - `--sintel_eval_every_epoch`
  - `--fc2_eval_every_epoch`
  - `--resume`

### Phase 3: Five-GPU Launcher

- [ ] Add a launcher wrapper, proposed: `wrappers/run_distill_or_not_fc2_batch.py`.
- [ ] Read `top10.csv`.
- [ ] Launch at most 5 child processes concurrently.
- [ ] Assign each child process exactly one GPU through `CUDA_VISIBLE_DEVICES`.
- [ ] Keep one architecture per child process to avoid graph/memory coupling.
- [ ] Collect process return codes and write a launcher manifest.
- [ ] Support resume by skipping completed or launching with `--resume`.

### Phase 4: Validation

- [ ] Unit-test candidate CSV parsing.
- [ ] Unit-test command construction and GPU assignment.
- [ ] Dry-run launcher with `--dry_run`.
- [ ] Smoke-test one architecture for 1 epoch on a small FC2 cap if available.
- [ ] Verify generated outputs contain:
  - per-model `eval_history.csv`
  - FC2 EPE every epoch
  - Sintel EPE every 5 epochs
  - `comparison.csv` or equivalent summary
  - `run_manifest.json`

### Phase 5: HPC Script

- [ ] Write Slurm example for 5 GPUs.
- [ ] Recommended starting resources:
  - `--gpus-per-node=5`
  - `--cpus-per-task=20` to `30`
  - `--mem=140G` to `210G`
  - `--time=72:00:00`
- [ ] Default worker plan:
  - `--fc2_num_workers 4`
  - `--fc2_eval_num_workers 4`
  - `--prefetch_batches 2`
  - total loader threads approximately `5 * 4 = 20`, plus TensorFlow/Vela/runtime overhead.

## Important Design Decision Pending

The main unresolved decision is initialization:

1. **Train from scratch / common random initialization**
   - Best for measuring architecture quality independent of either supernet.
   - Least biased for deciding which inherited rank predicts true retrain quality.
   - This is my recommendation.

2. **Warm-start all candidates from one common supernet checkpoint**
   - Closer to the original retrain workflow if the project defines retrain as inherited-weight fine-tuning.
   - But choosing no-distill or distill checkpoint would bias the experiment toward that supernet.

3. **Warm-start each candidate from both supernets**
   - Most complete but doubles compute to 20 short runs.
   - Tests both ranking and inherited-weight quality, but makes interpretation less clean.

## Recommended Default

Use common random initialization with the same seed family, then compare the final 50-epoch FC2/Sintel ranks against:

- no-distill inherited front rank
- distill inherited front rank

This directly tests ranking predictiveness without giving either supernet a warm-start advantage.

## Open Questions For User

1. Should the 50-epoch short retrain start from scratch, or should it warm-start from a supernet checkpoint?
2. Should all 10 subnets share the same seed, or should each use a stable different seed derived from candidate ID?
3. Should early stopping be disabled for this probe?

