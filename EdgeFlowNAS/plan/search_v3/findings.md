# Findings: Search V3 NSGA-II Planning

## Search V2 Baseline State

- `search_v2` selected NSGA-II as the first non-agent baseline because the project is a two-objective Pareto search: `EPE↓` and `FPS↑`.
- The existing NSGA-II implementation lives in:
  - `D:/Dataset/MCUFlowNet/EdgeFlowNAS/efnas/baselines/nsga2_search.py`
  - `D:/Dataset/MCUFlowNet/EdgeFlowNAS/wrappers/run_nsga2_search.py`
  - `D:/Dataset/MCUFlowNet/EdgeFlowNAS/configs/nsga2_v2.yaml`
- The V2 NSGA-II runner is not fully version-agnostic yet. It imports `efnas.nas.search_space_v2` directly.
- The V2 config points evaluation to:
  - `wrappers/run_supernet_subnet_distribution_v2.py`
  - `configs/supernet_fc2_172x224_v2.yaml`

## V3 Supernet Output Summary

### No-Distill V3

- Path: `D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel`
- Reported `epochs_finished`: `196`
- Reported `eval_epochs`: `40`
- Reported `best_metric`: `3.695306803410252`
- Final eval history at `epoch=200`: `mean_epe_12 = 3.695188325519363`
- Final `fairness_gap`: `0`
- Checkpoints present under `checkpoints/`.

### Distill V3

- Path: `D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill`
- Reported `epochs_finished`: `200`
- Reported `eval_epochs`: `40`
- Reported `best_metric`: `4.313483780870835`
- Final eval history at `epoch=200`: `mean_epe_12 = 4.31512763351202`
- Final `fairness_gap`: `0`
- Checkpoints present under `checkpoints/`.

## Current Supernet Choice Interpretation

- The no-distill V3 supernet is the stronger default for inherited-weight NSGA-II search based on FC2 eval-pool metric.
- The distill V3 supernet should not be discarded, but it should be treated as a controlled comparison rather than the default.
- If distill changes ranking but not inherited EPE, it may still be useful as a robustness check; however, there is not enough evidence to make it the primary search checkpoint.

## V3 Search-Space Facts

- V3 uses an 11D architecture code.
- Choice counts are `[3,3,3,3,3,3,2,2,2,2,2]`.
- Total size remains `3^6 * 2^5 = 23328`.
- V3 semantics version from manifest:
  - `supernet_v3_mixed_11d_light_to_heavy_fixed_bilinear_bneckeca_gate4x`
- V3 block names:
  - `E0`
  - `E1`
  - `EB0`
  - `EB1`
  - `DB0`
  - `DB1`
  - `H0Out`
  - `H1`
  - `H1Out`
  - `H2`
  - `H2Out`

## Key Difference From V2

- V2 and V3 can both accept an 11-number code, so a mistaken V2 evaluator may not fail early.
- That makes semantic validation important:
  - check manifest `arch_semantics_version`
  - use `search_space_v3.validate_arch_code`
  - use `MultiScaleResNet_supernet_v3`
- The first two positions in V3 are explicitly light-to-heavy aligned, fixing the earlier V2 semantic issue.

## Existing Evaluation Surface

- Existing `run_supernet_subnet_distribution_v2.py` is V2-specific by name and import.
- Existing `supernet_v2_evaluator.py` restores `MultiScaleResNetSupernetV2`.
- No `supernet_v3_evaluator.py` currently exists.
- V3 NAS helper files currently present:
  - `efnas/nas/arch_codec_v3.py`
  - `efnas/nas/eval_pool_builder_v3.py`
  - `efnas/nas/fair_sampler_v3.py`
  - `efnas/nas/search_space_v3.py`

## Design Finding

The cleanest implementation is to keep one NSGA-II algorithm implementation and make the search-space binding configurable, while adding a V3-specific evaluator surface. The evaluator has real model/checkpoint differences; the NSGA-II algorithm mostly does not.

## Recommended Minimal Implementation

1. Add V3 fixed-subnet evaluation support.
2. Parameterize or lightly wrap NSGA-II so it samples V3 search-space choices.
3. Add `configs/nsga2_v3.yaml`.
4. Run a smoke test on one fixed architecture.
5. Run the full no-distill NSGA-II baseline.
6. Optionally run the same budget or a smaller comparison on the distill checkpoint.

## Feasibility Finding: No-Distill And Distill Support

- Supporting both completed V3 supernets is straightforward if the evaluator config can select the experiment folder.
- The code should not fork separate evaluators for distill and no-distill. They share architecture semantics and graph structure.
- The clean setup is either:
  - one `configs/nsga2_v3.yaml` with CLI overrides for `runtime.experiment_name`, or
  - two thin config files that differ only in selected V3 experiment name and output root.
- The default should remain no-distill because its inherited FC2 eval-pool metric is better.
- The distill run should be a second controlled run to test whether the Pareto front and top candidates are stable.

## Feasibility Finding: Multi-GPU Search

- Multi-GPU search is feasible and should speed up NSGA-II because candidate evaluations are independent.
- The speedup is across architectures, not within a single architecture.
- Current V2 worker code only sets `TF_FORCE_GPU_ALLOW_GROWTH=true`; it does not bind a subprocess to a specific GPU.
- Therefore simply requesting six P100s is not enough. Without explicit GPU assignment, multiple TensorFlow subprocesses may all try to use GPU 0 or see all GPUs.
- Required implementation:
  - add `concurrency.gpu_devices`
  - assign one GPU id per evaluation subprocess
  - set subprocess environment `CUDA_VISIBLE_DEVICES=<assigned_gpu>`
  - log assigned GPU with the candidate arch code
- Expected speedup:
  - close to linear only if each single-arch eval is GPU-bound and CPU/I/O can keep up
  - lower if FC2 loading, Vela, or filesystem I/O dominates
- With six P100s, a practical first target is `max_workers=6`, one eval process per GPU.

## Feasibility Finding: CPU Data Loading And Prefetch

- FC2 provider already supports threaded loading through `fc2_num_workers` and `fc2_eval_num_workers`.
- Current fixed-subnet V2 wrapper accepts `--num_workers`, but the help text says it is currently unused.
- Current V2 fixed-subnet evaluator builds providers directly and does not wrap them with `PrefetchBatchProvider`.
- Therefore CPU prefetch is feasible, but not automatic in the current fixed-subnet search path.
- Required implementation:
  - make `--num_workers` override `data.fc2_num_workers` and `data.fc2_eval_num_workers`
  - add `--prefetch_batches`
  - wrap both BN-recal train provider and FC2-val provider in `PrefetchBatchProvider` when enabled
  - close providers cleanly after each eval
- Recommended starting point for six P100s:
  - if `cpus-per-task >= 24`: use `num_workers=4` per eval process
  - if `cpus-per-task = 12-16`: use `num_workers=2` per eval process
  - use `prefetch_batches=1` or `2`

## Practical HPC Conclusion

- Six P100s should help materially if the implementation binds workers to GPUs.
- The first pilot should not immediately run all 800 evaluations.
- Recommended pilot:
  - `max_workers=2`, `gpu_devices=0,1`, `max_fc2_val_samples=64`
  - then `max_workers=6`, `gpu_devices=0,1,2,3,4,5`, `max_fc2_val_samples=64`
  - compare per-candidate wall time and GPU utilization
- After pilot, run full FC2 val with `max_fc2_val_samples=null`.

## Validation Finding

The most dangerous failure mode is not a crash. It is silently running V2 semantics against V3-shaped codes. Therefore validation should assert manifest compatibility and search-space module identity before a full NSGA-II launch.

## Suggested Reporting Artifacts

For each V3 NSGA-II run, keep:

- `metadata/history_archive.csv`
- `metadata/epoch_metrics.csv`
- `metadata/nsga2_state.json`
- `metadata/pareto_front.csv`
- `metadata/run_manifest.json` or equivalent config snapshot
- per-arch `dashboard/eval_outputs/run_<arch>/analysis/records.csv`
- Vela summaries for each candidate

## Future Diagnostic

After the V3 fixed-subnet evaluator exists, repeat a smaller version of the V2 rank-consistency diagnostic:

- 30-50 mixed-complexity V3 subnets
- evaluate FC2 inherited EPE
- evaluate Sintel inherited EPE
- compute Spearman, Kendall, top-k overlap
- decide whether V3 search still uses FC2-only first-stage search plus Sintel shortlist re-ranking

This diagnostic should be done before relying on FC2-only final selection, but it does not block implementing the NSGA-II V3 runtime.
