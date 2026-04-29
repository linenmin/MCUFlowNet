# Progress Log: Search V3

## Session: 2026-04-29

### Planning Initialization

- **Status:** complete
- **Actions taken:**
  - Loaded the `pi-planning-with-files` workflow.
  - Created `D:/Dataset/MCUFlowNet/EdgeFlowNAS/plan/search_v3`.
  - Reviewed `plan/search_v2/task_plan.md`.
  - Reviewed `plan/search_v2/findings.md`.
  - Reviewed V3 no-distill and distill supernet output folders.
  - Reviewed V3 `run_manifest.json` files.
  - Reviewed final rows from both `eval_epe_history.csv` files.
  - Reviewed existing NSGA-II config and runner code.
  - Reviewed existing V2 fixed-subnet evaluator surface.
  - Confirmed V3 has search-space helpers but does not yet have a V3 fixed-subnet evaluator.

### Key Observations

- V3 no-distill supernet finished with a better inherited eval-pool metric than V3 distill.
- V3 and V2 share 11D code shape and search-space size, but not safe-to-mix semantics.
- The existing NSGA-II runner is currently V2-bound through `search_space_v2` imports.
- The existing evaluator is V2-bound through `MultiScaleResNetSupernetV2`.
- NSGA-II itself can remain largely unchanged if search-space sampling/validation is parameterized.

### Files Created

- `D:/Dataset/MCUFlowNet/EdgeFlowNAS/plan/search_v3/task_plan.md`
- `D:/Dataset/MCUFlowNet/EdgeFlowNAS/plan/search_v3/findings.md`
- `D:/Dataset/MCUFlowNet/EdgeFlowNAS/plan/search_v3/progress.md`

## Next Recommended Action

Implement Phase 2:

1. Add `efnas/engine/supernet_v3_evaluator.py`.
2. Add `efnas/nas/supernet_subnet_distribution_v3.py`.
3. Add `wrappers/run_supernet_subnet_distribution_v3.py`.
4. Verify one fixed architecture can run through FC2 inherited evaluation and Vela.

## Session: 2026-04-29 (Feasibility Review For Distill, Multi-GPU, Prefetch)

### Status

- **Status:** complete

### Actions taken

- Re-read the current `search_v3` plan.
- Checked V2 fixed-subnet evaluator for GPU assignment and data-loading behavior.
- Checked `efnas/search/eval_worker.py` subprocess environment handling.
- Checked FC2 provider threading support.
- Updated `task_plan.md` with a dedicated multi-GPU and CPU-input phase.
- Updated `findings.md` with feasibility conclusions.

### Conclusions

- Supporting no-distill and distill V3 supernets is feasible through config/CLI selection of the supernet experiment folder.
- Six P100s can speed up search, but only if each evaluation subprocess is explicitly bound to one GPU.
- Current worker code does not yet implement per-worker GPU assignment.
- CPU threaded loading exists in FC2 provider, but the fixed-subnet search wrapper currently does not really wire `--num_workers` into provider config.
- Bounded prefetch exists and can be reused, but fixed-subnet eval must explicitly wrap its train/val providers.

### Updated implementation requirement

Before full NSGA-II V3 search, implement:

1. V3 fixed-subnet evaluator.
2. Configurable no-distill/distill experiment selection.
3. Per-subprocess GPU assignment.
4. Real `num_workers` passthrough.
5. `prefetch_batches` passthrough and provider wrapping.

## Session: 2026-04-29 (NSGA-II V3 Runtime Implementation)

### Status

- **Status:** implementation complete, local dataset-backed eval not run because local FC2 dataset path is absent.

### Actions taken

- Added `configs/nsga2_v3.yaml`.
- Added `efnas/engine/supernet_v3_evaluator.py`.
- Added `efnas/nas/supernet_subnet_distribution_v3.py`.
- Added `wrappers/run_supernet_subnet_distribution_v3.py`.
- Updated `efnas/baselines/nsga2_search.py` so the search-space module is configurable.
- Updated `efnas/search/eval_worker.py` to pass:
  - `--experiment_dir`
  - `--prefetch_batches`
  - per-worker `CUDA_VISIBLE_DEVICES`
- Updated `wrappers/run_nsga2_search.py` with CLI overrides for:
  - `--supernet_experiment_dir`
  - `--gpu_devices`
  - `--max_workers`
  - `--num_workers`
  - `--prefetch_batches`
  - `--max_fc2_val_samples`
- Added unit tests for V3 wrapper parsing, V3 search-space adapter, GPU round-robin assignment, eval command passthrough, and NSGA-II CLI overrides.

### Verification

- `D:/Anaconda3/envs/tf_work_hpc/python.exe -m unittest tests.test_eval_worker_command tests.test_run_supernet_subnet_distribution_v3_wrapper tests.test_nsga2_baseline tests.test_run_nsga2_search_wrapper`
  - Result: `18 tests OK`
- `D:/Anaconda3/envs/tf_work_hpc/python.exe wrappers/run_supernet_subnet_distribution_v3.py --dry_run --fixed_arch 0,0,0,0,0,0,0,0,0,0,0 --experiment_dir outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel --num_workers 2 --prefetch_batches 2 --max_fc2_val_samples 64 --enable_vela --vela_mode verbose`
  - Result: printed the expected `python -m efnas.nas.supernet_subnet_distribution_v3 ...` command.
- `D:/Anaconda3/envs/tf_work_hpc/python.exe -m unittest tests.test_prefetch_provider tests.test_run_supernet_train_wrapper_v3 tests.test_supernet_v3_space_helpers`
  - Result: `10 tests OK`

### Local limitation

- `D:/Dataset/MCUFlowNet/Datasets/FlyingChairs2/val` does not exist locally, so the actual one-architecture FC2 evaluation must be run on HPC or on a machine with the dataset mounted.

## Errors Or Tooling Notes

| Issue | Impact | Resolution |
| --- | --- | --- |
| `rg.exe` from the Codex Windows app failed with access denied | Could not use `rg` for code search in this session | Used PowerShell `Get-ChildItem | Select-String` fallback |
