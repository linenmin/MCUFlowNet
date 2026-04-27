# Task Plan: Retrain V2 Fixed-Arch Candidates

## Goal

Build a clean V2 retraining pipeline for two shortlisted 11D architectures, starting from inherited supernet weights and training them at larger resolution for fairer final comparison against EdgeFlowNet on Sintel-style evaluation.

Target architectures:

1. `2,1,0,1,2,1,0,0,0,0,1`
2. `2,1,0,0,0,0,0,0,0,0,0`

## Why This Exists

The current standalone retrain path in EdgeFlowNAS was written for the old 9D supernet setting and FC2-only retraining. It is not yet aligned with:

- the V2 11D search space
- dual-model retraining of selected fixed architectures
- warm-starting from V2 supernet weights
- larger-resolution retraining
- two-stage `FC2 -> FlyingThings3D` continuation

## Desired End State

One reproducible retraining entrypoint that can:

1. load the two V2 fixed architectures
2. initialize them from the trained V2 supernet shared weights
3. retrain both models jointly under the same data stream
4. continue from `FC2` with small LR and cosine schedule
5. then continue on `FlyingThings3D` with `1e-5` scale LR
6. support early stopping and checkpoint resume
7. produce checkpoints suitable for final Sintel evaluation

## Current Phase

Phase 4 - FT3D runtime NaN investigation

## Phases

### Phase 1: Scope Lock And Gap Audit

- [x] Confirm which current trainer is the best base:
  - `wrappers/run_standalone_train.py`
  - `wrappers/fixed_arch_compare/run_train.py`
- [x] Confirm V2 retraining must use 11D arch codes end-to-end
- [x] Confirm warm-start path from V2 supernet checkpoint into fixed-arch models
- [x] Confirm larger-resolution training target and how it maps to dataset loader/crop policy
- [x] Confirm two-stage schedule:
  - stage 1: FC2 continuation with small LR and early stopping
  - stage 2: FT3D continuation with `1e-5`

### Phase 2: Training Design

- [x] Decide whether to extend `fixed_arch_compare_trainer.py` or build a new V2 retrain trainer
- [x] Define config contract for:
  - two candidate arch codes
  - stage-wise datasets
  - stage-wise LR / epoch / early-stop / cosine settings
  - warm-start checkpoint source
  - high-resolution patch size
- [x] Define manifest/checkpoint layout for stage 1 and stage 2
- [x] Define evaluation checkpoints and final Sintel export path

### Phase 3: Implementation

- [x] Add a V2 fixed-arch retrain wrapper/config
- [x] Add or refactor trainer to support:
  - 11D fixed architectures
  - supernet weight import
  - multi-stage dataset schedule
  - early stopping
  - cosine LR
  - dual-model shared-data training
- [x] Add any required data loader/config support for larger resolution

### Phase 4: Verification

- [x] Smoke test config parsing and graph construction
- [x] Verify checkpoint import from V2 supernet
- [x] Produce exact HPC commands for the full retrain run
- [x] Replace constant-arch supernet retrain graph with pure fixed-subnet graph
- [x] Correct external `FT3D -> Sintel` evaluation scale
- [ ] Verify stage transition `FC2 -> FT3D`
- [ ] Verify resume / early stopping state
- [ ] Diagnose FT3D epoch-34 NaN source before continuing from `last.ckpt`
  - [x] Check whether FT3D PFM files contain NaN/Inf or extreme values
  - [x] Exclude known non-finite FT3D flow files from sample resolution
  - [x] Add runtime loader guard so non-finite flow is skipped instead of entering TensorFlow
  - [x] Resume only from epoch-32 `best.ckpt` or a clean pre-NaN checkpoint, not epoch-36 `last.ckpt`
  - [ ] Verify clean restart on HPC for at least one evaluation interval

## Open Technical Questions

1. Which current code path is the least risky foundation:
   - old standalone retrain path
   - fixed-arch compare trainer
   - a new V2-specific retrain trainer
2. How should early stopping be defined:
   - FC2 val EPE patience
   - stage 2 FT3D val EPE patience
   - or checkpoint on external Sintel proxy only
3. Whether FT3D validation should remain `TEST`-root based or move to a custom held-out subset once the first run is stable.

## Current Execution Status

- `FC2` retrain command is now runnable on HPC with the confirmed supernet checkpoint path:
  - `outputs/supernet/edgeflownas_supernet_v2_fc2_172x224_run1_balance/checkpoints/supernet_best.ckpt`
- A real warm-start restore bug was found during first HPC launch:
  - TensorFlow tried to restore Adam slot tensors from the supernet checkpoint
  - fix committed by excluding optimizer-slot variables from `warmstart_saver`
- `FT3D` flow root was confirmed on HPC under:
  - `FlyingThings3D/optical_flow`
- `FT3D` RGB frames were initially missing on HPC; after download they were confirmed under:
  - `FlyingThings3D/frames_cleanpass`
- The retrain pipeline therefore now has a concrete stage-2 data contract:
  - frames root: `FlyingThings3D/frames_cleanpass`
  - flow root: `FlyingThings3D/optical_flow`
- The retrain graph has now been cleaned up from constant-arch supernet execution to true fixed-subnet execution:
  - train graph: `FixedArchModelV2`
  - eval graph: `FixedArchModelV2`
  - warm-start: explicit normalized variable-name mapping from fixed subnet to supernet checkpoint
- The external Sintel evaluation path has now been corrected for `FT3D` checkpoints:
  - `FT3D` predictions are multiplied back by the configured flow scale before Sintel EPE is computed
  - `FC2` predictions remain unchanged
- What remains unverified is not configuration anymore, but runtime behavior:
  - actual `FC2 -> FT3D` handoff
  - actual `resume`
  - actual early stopping behavior on HPC
- FT3D `retrain_v2_ft3d_run2` reached a new best at epoch 32, then both candidate models reported `nan` for train loss, FT3D validation EPE, and external Sintel EPE at epochs 34 and 36.
- The epoch-34 NaN is present in each model's `eval_history.csv`, not just in top-level `comparison.csv`; this points upstream of CSV aggregation.
- The last checkpoints at epoch 36 have `metric: NaN`; epoch-32 `best.ckpt` remains the latest clean model checkpoint for both `knee` and `fast`.
- A direct full scan of `../Datasets/FlyingThings3D/optical_flow` found 14 non-finite `.pfm` files out of 107040 files, including 13 in `TRAIN` and 1 in `TEST`. This is now confirmed as a real data-quality fault capable of producing the observed NaNs.
- There are two inconsistent history locations in the copied run output:
  - outer `outputs/retrain_v2_ft3d/retrain_v2_ft3d_run2/comparison.csv` stops at epoch 18
  - nested `outputs/retrain_v2_ft3d/retrain_v2_ft3d_run2/retrain_v2_ft3d_run2/comparison.csv` contains epochs 22-36 and the NaNs
  This should be treated as a resume/output-layout warning until the exact HPC command and copy path are confirmed.

## Active NaN Triage Plan

1. Preserve current artifacts before running more training:
   - keep epoch-32 `best.ckpt`
   - keep epoch-36 `last.ckpt` only as a failed-state artifact
   - do not use `last.ckpt` as the source for final evaluation or continuation
2. Run the read-only FT3D input scan on HPC:
   - `python scripts/diagnose_retrain_v2_ft3d_nan.py --config configs/retrain_v2_ft3d.yaml --frames_base_path ../Datasets/FlyingThings3D/frames_cleanpass --flow_base_path ../Datasets/FlyingThings3D/optical_flow --split both --output_csv outputs/retrain_v2_ft3d/ft3d_pfm_scan.csv`
3. If the full PFM scan is clean, replay provider batches for the number of epochs since the most recent resume, not necessarily absolute epoch 34:
   - after resume from epoch 21, epoch 34 is roughly the 14th in-process provider epoch
   - `python scripts/diagnose_retrain_v2_ft3d_nan.py --config configs/retrain_v2_ft3d.yaml --frames_base_path ../Datasets/FlyingThings3D/frames_cleanpass --flow_base_path ../Datasets/FlyingThings3D/optical_flow --skip_scan --probe_batches --probe_split train --epochs_to_probe 14 --batch_size 32`
4. Interpret results:
   - scan has found non-finite PFM files, so quarantine/exclude those samples first
   - still run provider replay or add loader logging once to verify the bad paths are actually reachable by the configured `frames_cleanpass + optical_flow` sample resolver
   - if NaNs persist after exclusion, then instrument TensorFlow-side loss, gradients, and variables before changing hyperparameters
5. Only after root cause is known, continue from the epoch-32 best checkpoint into a fresh experiment name.

## Implemented Data Guard

- `configs/retrain_v2_ft3d.yaml` now lists the 14 known non-finite PFM files under `data.ft3d_excluded_flow_paths`.
- `build_ft3d_provider` forwards that exclusion list to the FT3D resolver.
- `FT3DBatchProvider` now checks `np.isfinite(flow)` after PFM read and after flow scaling/clipping; bad samples are skipped and replaced.
- `wrappers/run_retrain_v2_ft3d.py` now accepts the clearer generic initialization arguments:
  - `--init_experiment_dir`
  - `--init_ckpt_name`
- `wrappers/run_retrain_v2_ft3d.py` also accepts `--resume_ckpt_name`; using `--resume --resume_ckpt_name best` preserves the original checkpoint epoch/global_step for LR scheduling while avoiding a contaminated `last.ckpt`.

## Non-Goals

- Reworking the search system again
- Changing the final two shortlisted architectures
- Reproducing the full original EdgeFlowNet training codebase inside EdgeFlowNAS

## Success Criteria

- Two selected V2 architectures can be retrained from inherited supernet weights
- Training uses larger resolution than deployment-time supernet search
- Pipeline supports `FC2 -> FT3D` staged continuation
- Outputs are directly usable for final Sintel comparison
