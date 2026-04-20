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

Phase 4

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

## Non-Goals

- Reworking the search system again
- Changing the final two shortlisted architectures
- Reproducing the full original EdgeFlowNet training codebase inside EdgeFlowNAS

## Success Criteria

- Two selected V2 architectures can be retrained from inherited supernet weights
- Training uses larger resolution than deployment-time supernet search
- Pipeline supports `FC2 -> FT3D` staged continuation
- Outputs are directly usable for final Sintel comparison
