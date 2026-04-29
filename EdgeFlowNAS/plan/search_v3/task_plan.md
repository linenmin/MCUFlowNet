# Task Plan: Search V3 NSGA-II Adaptation For Supernet V3

## Goal

Adapt the existing `search_v2` NSGA-II baseline to the completed third-version supernet outputs while keeping the implementation maintainable. This plan is scoped to the NSGA-II search path first; the agentic search path will be planned separately later.

The central change is not the dimensionality alone. `supernet_v3` keeps the same 11D mixed search-space size as V2, but its architecture semantics, network class, checkpoint manifests, and best available trained supernet runs are different.

## Current Phase

Phase 1 complete: planning and code-surface audit.

## Inputs

| Input | Path | Role |
| --- | --- | --- |
| V3 no-distill supernet | `D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel` | Default search checkpoint candidate |
| V3 distill supernet | `D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill` | Secondary comparison checkpoint |
| Previous NSGA-II plan | `D:/Dataset/MCUFlowNet/EdgeFlowNAS/plan/search_v2` | Design baseline to adapt |
| Existing NSGA-II config | `D:/Dataset/MCUFlowNet/EdgeFlowNAS/configs/nsga2_v2.yaml` | Config template |
| Existing NSGA-II runner | `D:/Dataset/MCUFlowNet/EdgeFlowNAS/efnas/baselines/nsga2_search.py` | Algorithm implementation to refactor or parameterize |

## Phases

### Phase 1: Audit V2 Search And V3 Supernet Outputs

- [x] Review `search_v2` planning decisions.
- [x] Confirm existing NSGA-II baseline entrypoint and config.
- [x] Inspect both completed V3 supernet output folders.
- [x] Confirm V3 search-space semantics and manifest fields.
- **Status:** complete

### Phase 2: Define V3 Evaluation Surface

- [ ] Add or generalize a V3 fixed-subnet evaluator.
- [ ] Add a V3 wrapper equivalent to `wrappers/run_supernet_subnet_distribution_v2.py`.
- [ ] Ensure evaluator restores `MultiScaleResNet_supernet_v3` checkpoints.
- [ ] Ensure evaluator validates with `efnas.nas.search_space_v3`, not `search_space_v2`.
- [ ] Preserve full FC2 validation coverage by default.
- [ ] Preserve BN recalibration before inherited-weight evaluation.
- [ ] Preserve Vela export and hardware metric parsing.
- **Status:** pending

### Phase 3: Refactor NSGA-II Search-Space Binding

- [ ] Remove hard dependency on `efnas.nas.search_space_v2` from `efnas/baselines/nsga2_search.py`, or create a small V3-specific runner only if parameterization becomes too invasive.
- [ ] Support configurable search-space module, with `search_space_v3` as the V3 config value.
- [ ] Keep the categorical operators unchanged:
  - uniform crossover
  - per-gene mutation
  - duplicate rejection
  - NSGA-II non-dominated sorting
  - crowding distance
- [ ] Keep `search_space_size = 23328`.
- **Status:** pending

### Phase 4: Create V3 NSGA-II Config And CLI Defaults

- [ ] Add `configs/nsga2_v3.yaml`.
- [ ] Point `evaluation.eval_script` to the V3 fixed-subnet wrapper.
- [ ] Point `evaluation.supernet_config` to a V3-compatible config that resolves the desired V3 experiment folder.
- [ ] Make the selected V3 supernet experiment configurable so the same code supports no-distill and distill checkpoints.
- [ ] Add either two explicit config files or two documented experiment-name overrides:
  - no-distill: `edgeflownas_supernet_v3_fc2_172x224_run1_archparallel`
  - distill: `edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill`
- [ ] Set default experiment name/output root under `outputs/nsga2_v3`.
- [ ] Decide whether `wrappers/run_nsga2_search.py` remains generic or receives a `run_nsga2_v3_search.py` alias for safer HPC use.
- **Status:** pending

### Phase 5: Supernet Choice Gate

- [ ] Support both no-distill and distill V3 checkpoints in the search runtime.
- [ ] Use no-distill V3 checkpoint as the default first full NSGA-II run.
- [ ] Treat distill V3 checkpoint as a second full run or a reduced-budget comparison run.
- [ ] If compute allows, run a small fixed probe on both checkpoints before the full NSGA-II run.
- [ ] Record whether top candidates are stable across no-distill and distill inherited weights.
- **Status:** pending

### Phase 6: Multi-GPU And CPU-Input Pipeline

- [ ] Add `concurrency.gpu_devices`, e.g. `"0,1,2,3,4,5"`.
- [ ] Set `concurrency.max_workers` independently from GPU count, with the default rule `max_workers <= len(gpu_devices)`.
- [ ] Modify the search worker so each evaluation subprocess receives a single-GPU `CUDA_VISIBLE_DEVICES` assignment.
- [ ] Avoid launching multiple TensorFlow evaluation subprocesses on the same GPU unless explicitly requested.
- [ ] Keep each architecture evaluation single-GPU; parallelism is across candidate architectures, not within one candidate.
- [ ] Make `evaluation.num_workers` actually override `data.fc2_num_workers` and `data.fc2_eval_num_workers` in the fixed-subnet evaluator.
- [ ] Add `evaluation.prefetch_batches` and wrap FC2 train/val providers with `PrefetchBatchProvider` in V3 evaluation.
- [ ] Add logging for:
  - assigned GPU per candidate
  - FC2 loader workers
  - prefetch batches
  - wall time per candidate
- **Status:** pending

### Phase 7: Validation And HPC Readiness

- [ ] Add unit tests for V3 arch parsing, mutation, crossover, and validation.
- [ ] Add wrapper dry-run tests for the V3 fixed-subnet evaluator.
- [ ] Add unit tests for GPU assignment and non-overlapping worker-device mapping.
- [ ] Add unit tests that `--num_workers` and `--prefetch_batches` reach the V3 evaluator config.
- [ ] Add NSGA-II config dry-run or initialization test.
- [ ] Run one local CPU-only/small-sample evaluation if dataset paths permit.
- [ ] Prepare Slurm commands for:
  - no-distill full NSGA-II
  - distill comparison NSGA-II
  - resume mode
- **Status:** pending

## Core Design Decisions

| Decision | Current Choice | Rationale |
| --- | --- | --- |
| First search algorithm | NSGA-II only | User explicitly wants NSGA-II adaptation first; agentic search comes later |
| Default V3 supernet | no-distill `edgeflownas_supernet_v3_fc2_172x224_run1_archparallel` | Completed run has better reported best metric than distill |
| Distill supernet role | controlled comparison | Distill run is complete but weaker by inherited FC2 eval; still worth checking ranking stability |
| First-stage objective | `EPE↓` and `FPS↑` | Same as V2 baseline; keep comparison clean |
| SRAM handling | record, not hard constraint | V2 plan already treated the space as deployment-oriented; retain unless Vela reveals violations |
| Resolution | `172x224` | Matches V3 supernet training and deployment-aligned resolution |
| Evaluation proxy | FC2 val first | V2 diagnostic supported FC2 as a strong coarse proxy; V3 should re-check with a smaller probe later |
| Multi-GPU policy | one architecture evaluation per GPU | NSGA-II candidates are independent, so process-level parallelism is the lowest-risk speedup |
| CPU input policy | threaded loader plus bounded prefetch per eval process | Current FC2 provider already supports threads; V3 evaluator must wire the CLI/config knobs through |

## Required Code Changes

1. **V3 evaluator**
   - Add `efnas/engine/supernet_v3_evaluator.py`.
   - Mirror `supernet_v2_evaluator.py`, but import:
     - `efnas.nas.search_space_v3`
     - `efnas.network.MultiScaleResNet_supernet_v3`
   - Use the V3 manifest semantics:
     - `supernet_v3_mixed_11d_light_to_heavy_fixed_bilinear_bneckeca_gate4x`
     - block choice counts `[3,3,3,3,3,3,2,2,2,2,2]`

2. **V3 fixed-subnet evaluation wrapper**
   - Add `wrappers/run_supernet_subnet_distribution_v3.py`.
   - Add or generalize `efnas/nas/supernet_subnet_distribution_v3.py`.
   - Preserve output contract:
     - `analysis/records.csv`
     - Vela summary artifacts
     - stdout/stderr logs via the existing search worker

3. **NSGA-II search-space generalization**
   - Preferred: parameterize `efnas/baselines/nsga2_search.py` with `search.search_space_module`.
   - V3 config should set:
     - `search_space_module: "efnas.nas.search_space_v3"`
     - `search_space_size: 23328`
   - Avoid copy-pasting the whole NSGA-II runner unless tests show import parameterization is fragile.

4. **V3 NSGA-II config**
   - Add `configs/nsga2_v3.yaml`.
   - Default evaluation should target the no-distill V3 experiment.
   - Add comments showing how to switch to the distill experiment.

5. **Multi-GPU evaluator scheduling**
   - Extend `efnas/search/eval_worker.py` or the NSGA-II runner to pass a worker-local GPU id into the subprocess environment.
   - Recommended config shape:
     - `concurrency.gpu_devices: "0,1,2,3,4,5"`
     - `concurrency.max_workers: 6`
   - Each subprocess should see exactly one logical GPU via `CUDA_VISIBLE_DEVICES=<one id>`.

6. **CPU data pipeline**
   - Add `--prefetch_batches` to the V3 fixed-subnet wrapper.
   - Make `--num_workers` update both FC2 train and eval provider worker counts for the evaluation process.
   - Wrap BN-recal train provider and FC2-val provider with `PrefetchBatchProvider` when `prefetch_batches > 0`.

## Suggested V3 NSGA-II Defaults

| Parameter | Suggested Value | Notes |
| --- | --- | --- |
| `total_evaluations` | `800` | Matches V2 plan for fair baseline comparison |
| `population_size` | `50` | Keeps 16 generations |
| `crossover_prob` | `0.9` | Standard NSGA-II setting retained |
| `mutation_prob` | `null` | Resolve to `1 / 11` per gene |
| `max_workers` | `6` on six P100s after smoke test | Use one evaluation subprocess per GPU |
| `gpu_devices` | `"0,1,2,3,4,5"` | Required for deterministic GPU binding |
| `num_workers` | `2-4` per eval process | Tune against allocated CPU cores and storage bandwidth |
| `prefetch_batches` | `1-2` | Enough to overlap FC2 loading without excessive RAM |
| `checkpoint_type` | `best` for initial run | Best inherited FC2 metric is the natural search checkpoint |
| `bn_recal_batches` | `32` initially | Matches V3 eval during training; can reduce to `16` only for speed pilots |
| `batch_size` | `32` | Same as V3 training/eval |
| `max_fc2_val_samples` | `null` for real run | Full FC2 val coverage |

## V3-Specific Risks

1. **V2 and V3 share 11D shape but not semantics**
   - Accidentally using `search_space_v2` would produce syntactically valid but semantically wrong candidates.

2. **Evaluator checkpoint mismatch**
   - `MultiScaleResNet_supernet_v2` cannot restore V3 checkpoints.
   - Manifest compatibility should be checked before evaluation.

3. **Distill checkpoint may bias ranking**
   - Distill V3 trained to a worse inherited FC2 best metric.
   - It should not be the default until a ranking-stability probe supports it.

4. **Vela export path may still be V2-specific**
   - Any helper named `_build_tflite_for_arch_v2` must either be generalized or wrapped carefully for V3.

5. **Concurrent evaluation can over-allocate GPU memory**
   - NSGA-II `max_workers=4` from V2 is unsafe if all workers see all GPUs.
   - This must be fixed by explicit per-subprocess `CUDA_VISIBLE_DEVICES` assignment.
   - With six P100s, `max_workers=6` is reasonable only after the smoke test confirms one eval fits one P100.

6. **CPU oversubscription**
   - Six concurrent eval subprocesses multiplied by `num_workers=4` creates at least 24 FC2 loader threads, plus TensorFlow and Vela overhead.
   - If Slurm CPU allocation is small, GPU utilization will be poor even with six GPUs.
   - Start with `num_workers=2` per process if CPU cores are limited; increase only if GPUs show input stalls.

## Implementation Gate Before Full Search

Before launching the 800-evaluation run:

1. `python wrappers/run_supernet_subnet_distribution_v3.py --dry_run ...` succeeds.
2. One fixed V3 architecture evaluates on a small FC2 cap.
3. One fixed V3 architecture exports through Vela.
4. A two-GPU pilot evaluates at least four candidates with non-overlapping GPU assignment.
5. The fixed-subnet evaluator logs `num_workers` and `prefetch_batches`.
6. `wrappers/run_nsga2_search.py --config configs/nsga2_v3.yaml --experiment_name ...` initializes metadata correctly.
7. `history_archive.csv` rows contain valid:
   - `arch_code`
   - `epe`
   - `fps`
   - `sram_kb`
   - `cycles_npu`
   - `macs`

## Open Questions

1. Should the first full NSGA-II run use `checkpoint_type=best` or `last`?
   - Current plan: start with `best`.
   - Later comparison: check `last` if inherited rank looks unstable.

2. Should V3 redo the FC2-vs-Sintel rank-consistency diagnostic?
   - Current plan: yes, but after the fixed-subnet V3 evaluator exists.
   - It can be smaller than V2 initially, e.g. 30-50 probe architectures.

3. Should the next supernet stage include FT3D before search?
   - Current plan: not required for the immediate FC2-trained V3 NSGA-II baseline.
   - If the research goal shifts to inherited weights that skip retrain, add an FT3D-supernet stage and redo search on that checkpoint.
