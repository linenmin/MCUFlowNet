# Progress Log: Distill-Or-Not Short Retrain Probe

## Session: 2026-04-30

### Status

- **Status:** implementation verified locally; git commit/push pending.
- **Initialization policy:** confirmed by user as scratch/common random initialization.

### Actions taken

- Loaded planning workflow.
- Created `D:/Dataset/MCUFlowNet/EdgeFlowNAS/plan/distillOrNot`.
- Reviewed selected candidate CSV:
  - `outputs/nsga2_v3/frontier_top5_rank_gap_probe_20260430/top10.csv`
- Reviewed available wrappers:
  - `wrappers/run_retrain_v2_fc2.py`
  - `wrappers/run_ablation_v1_fc2.py`
  - `wrappers/run_supernet_subnet_distribution_v3.py`
- Reviewed relevant configs:
  - `configs/retrain_v2_fc2.yaml`
  - `configs/ablation_v1_fc2.yaml`
  - `configs/supernet_v3_fc2_172x224.yaml`
- Reviewed relevant trainer/evaluator surfaces:
  - `efnas.engine.retrain_v2_trainer`
  - `efnas.engine.ablation_v1_trainer`
  - `efnas.engine.ablation_v1_sintel_runtime`
  - `efnas.nas.supernet_subnet_distribution_v3`
  - `efnas.data.dataloader_builder`

### Current conclusion

Existing training code has most of the needed pieces, but there is no clean V3 fixed-architecture short-retrain entrypoint yet.

Recommended implementation:

1. Add a V3 fixed-architecture trainable model.
2. Add a single-architecture FC2 short trainer.
3. Add a five-GPU batch launcher.
4. Add a config and Slurm example.

### Implementation update

- Added TDD coverage for candidate CSV parsing, GPU round-robin assignment, wrapper CLI controls, and fixed V3 selected-branch graph construction.
- Added `efnas.network.fixed_arch_models_v3.FixedArchModelV3`.
- Added `efnas.engine.distill_or_not_trainer` for single-candidate scratch FC2 retraining.
- Added `efnas.engine.distill_or_not_sintel_runtime` for V3 fixed checkpoint Sintel validation.
- Added wrappers:
  - `wrappers/run_distill_or_not_fc2_one.py`
  - `wrappers/run_distill_or_not_fc2_batch.py`
- Added config:
  - `configs/distill_or_not_fc2_short.yaml`
- Added versioned candidate list and Slurm template:
  - `rank_gap_top10.csv`
  - `distill_or_not_fc2_short_5gpu.slurm`

### Pending

- Commit and push.

### Verification

- `python -m unittest tests.test_distill_or_not_short_retrain` passed.
- `python wrappers/run_distill_or_not_fc2_batch.py --dry_run ...` passed and showed 10 candidates assigned round-robin over GPUs 0-4.
- `python -m py_compile ...` passed for the new trainer/runtime/model/wrapper files.
