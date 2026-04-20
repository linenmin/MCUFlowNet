# Progress: Retrain V2

## 2026-04-11

### Session Start

- Created planning workspace:
  - `plan/retrain_v2/task_plan.md`
  - `plan/retrain_v2/findings.md`
  - `plan/retrain_v2/progress.md`

### Context Gathered

- Read current retrain entrypoints:
  - `wrappers/run_standalone_train.py`
  - `efnas/engine/standalone_trainer.py`
  - `wrappers/fixed_arch_compare/run_train.py`
  - `efnas/engine/fixed_arch_compare_trainer.py`
- Read original EdgeFlowNet training wrapper:
  - `EdgeFlowNet/wrappers/run_train.py`
  - `EdgeFlowNet/code/train.py`
- Read existing configs:
  - `configs/standalone_fc2_180x240.yaml`
  - `configs/fixed_arch_compare_fc2_172x224.yaml`

### Key Conclusions Logged

- Old standalone retrain path is still 9D-oriented and not suitable for direct V2 reuse.
- Fixed-arch compare trainer is structurally closer to the desired dual-model retrain pipeline.
- Original EdgeFlowNet already supports `FT3D` by dataset switch; there is no separate FT3D wrapper.
- New retrain work should likely be based on:
  - V2 fixed architectures
  - warm-start from supernet weights
  - larger-resolution training
  - staged `FC2 -> FT3D` continuation

### Pending Next Actions

1. Run first FC2-stage warm-start on HPC.
2. Verify checkpoint resume and early stopping with a controlled interruption.
3. Run FT3D continuation from FC2 best checkpoints.
4. Run the independent Sintel test wrapper against the latest best weights.

### Implementation Completed

- Added V2 retrain configs:
  - `configs/retrain_v2_fc2.yaml`
  - `configs/retrain_v2_ft3d.yaml`
- Added V2 retrain wrappers:
  - `wrappers/run_retrain_v2_fc2.py`
  - `wrappers/run_retrain_v2_ft3d.py`
  - `wrappers/run_retrain_v2_sintel_test.py`
- Added FT3D folder-scanning provider:
  - `efnas/data/ft3d_dataset.py`
  - `efnas/data/dataloader_builder.py` extension
- Added retrain/eval engines:
  - `efnas/engine/retrain_v2_trainer.py`
  - `efnas/engine/retrain_v2_evaluator.py`
- Added V2 fixed-model helper:
  - `efnas/network/fixed_arch_models_v2.py`

### Verification Completed

- New tests added:
  - `tests/test_fixed_arch_models_v2.py`
  - `tests/test_ft3d_dataset.py`
  - `tests/test_run_retrain_v2_wrappers.py`
- Verified:
  - V2 retrain graph construction
  - supernet-compatible warm-start naming path
  - FT3D TRAIN/TEST folder scanning
  - wrapper default configs and parser contracts
- `py_compile` passed for all newly added retrain_v2 files.

## 2026-04-18

### HPC Execution Feedback Integrated

- First real `FC2` HPC launch failed during warm-start restore.
- Error was not a path issue in the graph itself; it was a restore-contract issue:
  - retrain warm-start attempted to restore Adam slot tensors from the source supernet checkpoint
  - the supernet checkpoint does not contain those tensors

### Fix Applied

- Updated `efnas/engine/retrain_v2_trainer.py` so `warmstart_saver` excludes optimizer-slot variables.
- Added regression coverage in:
  - `tests/test_fixed_arch_models_v2.py`
- Verified locally:
  - `python -m unittest tests.test_fixed_arch_models_v2`
  - `py_compile` for the modified retrain trainer and test
- Fix committed and pushed:
  - `856493e`

### HPC Data Contract Clarified

- Confirmed that `../Datasets` on HPC is a symlink, which explained why earlier `find` commands returned no results without `-L`.
- Confirmed FT3D flow data exists under:
  - `FlyingThings3D/optical_flow`
- Confirmed RGB frames were initially absent on HPC.
- User then downloaded and merged the required frame data under:
  - `FlyingThings3D/frames_cleanpass`

### Current Runnable Commands

- FC2 stage:
  - `python wrappers/run_retrain_v2_fc2.py --config configs/retrain_v2_fc2.yaml --experiment_name retrain_v2_fc2_run1 --supernet_checkpoint outputs/supernet/edgeflownas_supernet_v2_fc2_172x224_run1_balance/checkpoints/supernet_best.ckpt --gpu_device 0`
- FT3D stage:
  - `python wrappers/run_retrain_v2_ft3d.py --config configs/retrain_v2_ft3d.yaml --experiment_name retrain_v2_ft3d_run1 --fc2_experiment_dir outputs/retrain_v2_fc2/retrain_v2_fc2_run1 --frames_base_path FlyingThings3D/frames_cleanpass --flow_base_path FlyingThings3D/optical_flow --gpu_device 0`
- Sintel test:
  - `python wrappers/run_retrain_v2_sintel_test.py --experiment_dir outputs/retrain_v2_ft3d/retrain_v2_ft3d_run1 --dataset_root /lustre1/scratch/379/vsc37996/dataset/Sintel --sintel_list EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt --patch_size 416,1024 --ckpt_name best --gpu_device 0`

### Remaining Verification Work

1. Run FC2 end-to-end long enough to confirm:
   - warm-start really succeeds on HPC
   - checkpoint save path is valid
2. Run FT3D continuation once FC2 produces a `best.ckpt`.
3. Interrupt and resume at least one stage to validate resume semantics.
4. Check whether early stopping writes the expected trainer state JSON and exits cleanly.

### Fixed-Subnet Retrain Refactor

- Switched `retrain_v2` training graph from constant-arch `MultiScaleResNetSupernetV2` to true `FixedArchModelV2`.
- Switched `retrain_v2` evaluator graph to the same fixed-subnet model so training/eval/deployment contracts match.
- Added normalized supernet-name mapping in:
  - `efnas/engine/retrain_v2_trainer.py`
  so fixed-subnet variables can warm-start directly from the V2 supernet checkpoint.
- Added parser coverage for a new optional Vela comparison wrapper:
  - `wrappers/run_retrain_v2_vela_compare.py`
- Verified locally:
  - `python -m unittest tests.test_fixed_arch_models_v2 tests.test_run_retrain_v2_wrappers tests.test_ft3d_dataset`
  - `py_compile` for trainer / evaluator / Vela wrapper
  - a real restore from local `supernet_best.ckpt` into the fixed-subnet graph

### Validation Boundary

- Local checkpoint-restore validation now succeeds for the fixed-subnet graph.
- Local Vela comparison is still blocked by missing `vela` executable in the Windows environment.
- HPC commands for `FC2`, `FT3D`, and `Sintel test` remain CLI-compatible; only the internal graph and warm-start path changed.

## 2026-04-20

### FT3D To Sintel Evaluation Contract Fixed

- Investigated why early `FT3D` Sintel evaluations jumped to roughly `10.6` EPE while stage metrics were improving.
- Confirmed this was not immediately interpretable as real generalization collapse.
- Root cause:
  - `FT3D` training labels are divided by `12.5` in the retrain data loader
  - the external Sintel test wrapper was comparing predictions against raw Sintel flow without multiplying predictions back to the original flow scale
- Added a dedicated evaluation-scaling helper:
  - `efnas/engine/retrain_v2_eval_scaling.py`
- Updated:
  - `wrappers/run_retrain_v2_sintel_test.py`
  so `FT3D` checkpoints automatically restore prediction scale before Sintel EPE is computed
- `FC2` evaluation behavior remains unchanged

### Verification Completed

- Added regression coverage in:
  - `tests/test_retrain_v2_sintel_test.py`
- Verified locally:
  - `python -m unittest tests.test_retrain_v2_sintel_test`
  - `python -m py_compile efnas/engine/retrain_v2_eval_scaling.py wrappers/run_retrain_v2_sintel_test.py`

### Practical Implication

- Previously observed `FT3D`-stage Sintel CSV values before this fix should not be treated as trustworthy cross-stage comparison points.
- Re-running the same Sintel test command on HPC is sufficient; no CLI change is required because the correction is internal to the wrapper.
