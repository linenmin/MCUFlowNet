# Retrain V3 Progress

## 2026-05-04

Created `plan/retrain_v3` with three planning documents:

- `task_plan.md`
- `findings.md`
- `progress.md`

Checked the distill NSGA2 metadata folder:

- `pareto_front.csv`
- `history_archive.csv`
- `epoch_metrics.csv`

Confirmed:

- The all-zero architecture is present in the Pareto front.
- The current Pareto front has 51 rows.
- The all-zero subnet is the fastest Pareto endpoint with FPS `9.1415`.
- A provisional EdgeFlowNet-FPS neighborhood around `6.79 FPS` has several Pareto candidates.

No code changes were made in this step.

## Next Step

Measure original EdgeFlowNet deconv FPS at `172x224` with the same Vela configuration, then lock the final two or three Retrain V3 candidates.

## 2026-05-04 Vela Baseline Update

Ran `plan/retrain_v3/run_edgeflownet_deconv_vela_172x224.py` in the local `vela` conda environment.

Result for original EdgeFlowNet deconv at `172x224`:

- SRAM: `1694.0 KB`
- inference time: `188.0440 ms`
- FPS: `5.3179`

Updated `task_plan.md` and `findings.md` with the measured baseline and new same-FPS Pareto candidates.

Current shortlist:

- `V3-LIGHT`: `0,0,0,0,0,0,0,0,0,0,0`
- `V3-EFN-FPS`: `2,0,0,2,2,1,0,0,0,0,0`
- `V3-EFN-ACC`: `0,0,0,2,2,2,0,0,0,0,1`

## 2026-05-04 Retrain Protocol Confirmation

User confirmed:

- Train from scratch.
- Use original retrain schedule.
- Disable early stopping.
- Effective batch size is `32`.
- FC2 stage: FC2 val every epoch, Sintel val every 5 epochs.
- FT3D stage: FT3D val every epoch, Sintel val every 2 epochs.
- Final three subnets:
  - `v3_acc`: `0,1,2,2,2,2,0,0,0,0,1`
  - `v3_efn_fps`: `2,0,0,2,2,1,0,0,0,0,0`
  - `v3_light`: `0,0,0,0,0,0,0,0,0,0,0`

Checked current Sintel validation against EdgeFlowNet `run_test.py --uncertainity`.

Finding:

- The current V3 runtime is behaviorally close for EPE because it uses the same data list, batch preparation, and processor, and it slices flow channels before processor update.
- It is not strict command-level equivalence because the current V3 args object sets `uncertainity=False`; implementation should make this explicit.

Next:

- Wait for user decision on whether to implement async Sintel validation with a fourth GPU now, or start with a simpler three-training-process launcher.

## 2026-05-04 Implementation

Implemented the three-GPU Retrain V3 pipeline:

- `efnas/engine/retrain_v3_trainer.py`
- `wrappers/run_retrain_v3_fc2.py`
- `wrappers/run_retrain_v3_ft3d.py`
- `wrappers/run_retrain_v3_batch.py`
- `configs/retrain_v3_fc2.yaml`
- `configs/retrain_v3_ft3d.yaml`
- `plan/retrain_v3/retrain_v3_candidates.csv`
- `plan/retrain_v3/retrain_v3_fc2_3gpu.slurm`
- `plan/retrain_v3/retrain_v3_ft3d_3gpu.slurm`
- `tests/test_retrain_v3_wrappers.py`

Implementation details:

- One child process per subnet.
- One GPU per child process.
- Per-model outputs under `model_<name>/`, avoiding shared concurrent writes to `trainer_state.json`.
- FC2 starts from scratch.
- FT3D initializes model weights from the FC2 `best` checkpoint and starts its optimizer/scheduler fresh.
- Effective batch size is `32`, with `micro_batch_size` configurable.
- FC2 validation runs every epoch, Sintel every 5 epochs.
- FT3D validation runs every epoch, Sintel every 2 epochs.
- FT3D config carries the known 14 non-finite PFM exclusions.

Verification:

- `python -m unittest tests.test_retrain_v3_wrappers`
- `python -m py_compile efnas/engine/retrain_v3_trainer.py wrappers/run_retrain_v3_fc2.py wrappers/run_retrain_v3_ft3d.py wrappers/run_retrain_v3_batch.py`
- `python -m unittest tests.test_retrain_v3_wrappers tests.test_distill_or_not_short_retrain`
- dry-run checked FC2 and FT3D launcher commands for all three candidates on GPUs `0,1,2`.
