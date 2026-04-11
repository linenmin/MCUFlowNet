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
