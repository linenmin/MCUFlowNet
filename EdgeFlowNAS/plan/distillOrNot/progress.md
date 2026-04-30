# Progress Log: Distill-Or-Not Same-FPS Probe

## Session: 2026-04-30

### Current Status

- Corrected candidate selection from old 10-subnet Pareto/front-rank probe to new 8-subnet same-FPS rank-gap probe.
- Training code remains scratch/common random initialization.
- Slurm template is configured for 4 P100 GPUs.

### Completed

- Created and verified V3 fixed-architecture training path:
  - `efnas.network.fixed_arch_models_v3.FixedArchModelV3`
  - `efnas.engine.distill_or_not_trainer`
  - `efnas.engine.distill_or_not_sintel_runtime`
  - `wrappers/run_distill_or_not_fc2_one.py`
  - `wrappers/run_distill_or_not_fc2_batch.py`
- Fixed prefetch wrapper signature bug after HPC log showed `PrefetchBatchProvider.__init__()` did not accept `name=`.
- Verified focused tests:
  - `python -m unittest tests.test_distill_or_not_short_retrain`
- Verified fixed V3 Vela structure using local `vela` environment:
  - SRAM `1.353515625 MB`
  - inference `147.233625 ms`
  - FPS `6.791926776`
- Generated same-FPS candidate analysis:
  - `outputs/nsga2_v3/same_fps_rank_gap_probe_20260430/same_fps_rank_gap_top8.csv`
  - `outputs/nsga2_v3/same_fps_rank_gap_probe_20260430/fps_window_all_candidates.csv`
  - `outputs/nsga2_v3/same_fps_rank_gap_probe_20260430/summary.md`
- Copied the selected 8 candidates into a versioned plan CSV:
  - `plan/distillOrNot/same_fps_rank_gap_top8.csv`
- Updated 4-GPU Slurm template:
  - `plan/distillOrNot/distill_or_not_fc2_short_4gpu.slurm`

### Current Candidate Set

| ID | Arch code | FPS | Distill rank | No-distill rank | Gap |
| --- | --- | --- | --- | --- | --- |
| SF01 | `0,1,0,2,0,2,0,0,1,0,0` | 6.6722 | 5 | 15 | 10 |
| SF02 | `0,1,0,2,0,2,0,0,0,0,1` | 6.6015 | 4 | 14 | 10 |
| SF03 | `0,0,0,2,0,2,0,0,0,0,0` | 6.8952 | 8 | 17 | 9 |
| SF04 | `2,0,1,1,0,2,1,0,1,0,0` | 6.6233 | 17 | 9 | 8 |
| SF05 | `1,0,1,1,0,2,0,0,1,0,1` | 6.6329 | 15 | 7 | 8 |
| SF06 | `1,0,1,1,0,2,0,0,0,0,1` | 6.7032 | 14 | 6 | 8 |
| SF07 | `1,0,1,1,0,2,1,0,1,0,1` | 6.5974 | 13 | 5 | 8 |
| SF08 | `1,0,1,1,0,2,0,0,0,0,0` | 6.8494 | 16 | 8 | 8 |

### Next

- Run the 8-candidate short retrain on HPC with:
  - `sbatch plan/distillOrNot/distill_or_not_fc2_short_4gpu.slurm`
- After completion, compare FC2/Sintel final ranks against the two within-FPS-window supernet ranks.
