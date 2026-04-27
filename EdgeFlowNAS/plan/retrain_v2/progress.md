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

## 2026-04-27

### FT3D Run2 NaN Incident Logged

- User reported that `comparison.csv` contains finite metrics through epoch 32 and `nan` for all metrics at epochs 34 and 36.
- Inspected copied run artifacts under:
  - `outputs/retrain_v2_ft3d/retrain_v2_ft3d_run2`
  - `outputs/retrain_v2_ft3d/retrain_v2_ft3d_run2/retrain_v2_ft3d_run2`
- Confirmed the NaN is present in both model histories:
  - `model_fast/eval_history.csv`: `loss`, FT3D `epe`, and `sintel_epe` become `nan` at epoch 34
  - `model_knee/eval_history.csv`: same pattern at epoch 34
- Confirmed epoch-32 `best.ckpt` is still the latest clean checkpoint:
  - `fast` best FT3D EPE: `2.132976692745356`
  - `knee` best FT3D EPE: `1.9421447476720421`
- Confirmed epoch-36 `last.ckpt` metadata contains `metric: NaN` for both models and should not be used as a continuation source.

### Initial Debugging Interpretation

- Because both models become NaN at the same evaluation boundary, the first suspect is shared state:
  - shared FT3D batch/data path
  - shared optimizer/training step instability
  - shared resume/checkpoint state
- A static bad validation PFM is less likely to be the only cause because validation runs had been finite before epoch 34.
- A static bad training PFM is also not the cleanest explanation if `shuffle_no_replacement` truly traverses the whole training split every epoch; however, this still needs a direct PFM/batch scan because NaN values survive `np.clip`.

### Diagnostic Tool Added

- Added read-only diagnostic script:
  - `scripts/diagnose_retrain_v2_ft3d_nan.py`
- Purpose:
  - scan FT3D PFM files for NaN/Inf/extreme values
  - replay FT3D provider batches and check pre-TensorFlow input/label tensors
- Full PFM scan command for HPC:
  - `python scripts/diagnose_retrain_v2_ft3d_nan.py --config configs/retrain_v2_ft3d.yaml --frames_base_path ../Datasets/FlyingThings3D/frames_cleanpass --flow_base_path ../Datasets/FlyingThings3D/optical_flow --split both --output_csv outputs/retrain_v2_ft3d/ft3d_pfm_scan.csv`
- Provider replay command for the resumed run window:
  - `python scripts/diagnose_retrain_v2_ft3d_nan.py --config configs/retrain_v2_ft3d.yaml --frames_base_path ../Datasets/FlyingThings3D/frames_cleanpass --flow_base_path ../Datasets/FlyingThings3D/optical_flow --skip_scan --probe_batches --probe_split train --epochs_to_probe 14 --batch_size 32`

### Next Actions

1. Run the PFM scan on HPC.
2. If clean, run provider replay for the resumed epoch window.
3. If provider replay is clean, instrument training with TensorFlow numerics checks before changing LR or grad clipping.
4. Continue only from epoch-32 `best.ckpt` into a fresh experiment name after the root cause is identified.

### PFM Scan Result Integrated

- Full optical-flow scan completed on HPC:
  - total PFM files scanned: `107040`
  - bad PFM files: `14`
  - failure mode: `non-finite`
- Bad files reported:
  - `../Datasets/FlyingThings3D/optical_flow/TEST/A/0085/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0012/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0096/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0186/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0441/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0456/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0483/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0728/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/B/0459/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0031/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0080/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0140/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0260/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
  - `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0323/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- Interpretation:
  - This is sufficient evidence that the FT3D dataset contains corrupt/non-finite labels.
  - Because the current loader performs `np.clip(flow / divisor, ...)`, NaN values would survive preprocessing and can turn loss, gradients, and weights into NaN.
  - The next step is to exclude/quarantine these samples, then restart from epoch-32 `best.ckpt` in a fresh experiment directory.

### Data Exclusion Fix Implemented

- Added the 14 known non-finite PFM paths to:
  - `configs/retrain_v2_ft3d.yaml`
- Updated FT3D sample resolution and provider build path:
  - `efnas/data/ft3d_dataset.py`
  - `efnas/data/dataloader_builder.py`
- Behavior:
  - configured bad flow paths are excluded before sampling
  - any remaining non-finite flow loaded at runtime is skipped and replaced
  - multi-threaded augmentation now uses the per-job RNG instead of sharing the provider RNG across loader threads
- Added clearer FT3D restart CLI args:
  - `--init_experiment_dir`
  - `--init_ckpt_name`
- Tests added and passing:
  - configured flow exclusions are honored by resolver and provider
  - non-finite flow is skipped by provider
  - generic FT3D init args parse correctly
