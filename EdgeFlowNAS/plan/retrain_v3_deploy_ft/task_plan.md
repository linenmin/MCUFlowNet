# Task Plan: Retrain V3 Deploy-Resolution Fine-Tune

## Goal

Fine-tune the 3 retrain_v3 subnets (v3_acc / v3_efn_fps / v3_light) **and** the EdgeFlowNet mainline baseline from their current checkpoints at the board's deploy resolution **157×203** on FT3D, then produce a fair INT8 PTQ + Vela report and a final EPE comparison vs the un-fine-tuned baselines.

End state = updated `sintel_best.ckpt`-family checkpoints under a new `retrain_v3_deploy_ft_*` experiment dir + a Δ-table that shows whether (and how much) deploy-res fine-tune closes the train/deploy input-size mismatch identified in Seeed plan §7d.

## Current Phase

Phase 2 (Code Implementation) — Phase 1 closed

## Phases

### Phase 1: Plan & Design — **complete**

- [x] Confirm hard facts about mainline checkpoint (training data, input size, flow_divisor) — Key Q §1-3 all answered: FT3D / 480×640 / no divisor (findings §6)
- [x] Decide Sintel-leakage policy for in-training val (Key Q §4 → Option A: FT3D TEST split for val, Sintel Final eval offline-only)
- [x] Lock final hyperparameter set (findings §9)
- [x] Lock code structure (findings §11)
- [x] Output: this file + findings.md + progress.md fully filled
- **Status:** complete

### Phase 2: Code Implementation

- [ ] `efnas/network/edgeflownet_mainline.py`: wrap EdgeFlowNet `MultiScaleResNet` (transpose-conv decoder) behind the same `build()` interface used by `FixedArchModelV3`
- [ ] `efnas/engine/deploy_ft_trainer.py`: deploy-res fine-tune trainer with `arch_family ∈ {fixed_v3, edgeflownet_mainline}` switch
  - reuses `efnas.data.dataloader_builder.build_ft3d_provider` for data
  - reuses `efnas.engine.checkpoint_manager` for ckpt I/O
  - reuses `efnas.engine.distill_or_not_sintel_runtime.evaluate_v3_checkpoint_dir_on_sintel` for in-training Sintel-Final eval (with `arch_family` dispatch)
  - explicit support for `init_ckpt_path` override (load from arbitrary external ckpt prefix)
  - explicit support for per-family `flow_divisor` (12.5 for v3, 1.0 for mainline)
- [ ] `configs/retrain_v3_deploy_ft.yaml`: config template (input 157×203, FT3D data, low LR, fine-tune epoch budget, aug scaled down)
- [ ] `wrappers/run_deploy_ft_batch.py`: dispatch one trainer process per candidate per GPU (mirror of `run_retrain_v3_batch.py`)
- [ ] `plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_candidates.csv`: 4 rows (v3_acc, v3_efn_fps, v3_light, mainline) with `arch_family`, `init_ckpt_path`, `flow_divisor`
- [ ] `plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_4gpu.slurm`: slurm script for HPC submission
- [ ] Smoke test on local CPU / single GPU with `--limit-batches 2 --num-epochs 1` to validate end-to-end before HPC submit
- **Status:** pending

### Phase 3: V3 Subnets Fine-Tune (HPC)

- [ ] Submit fine-tune for v3_acc, v3_efn_fps, v3_light in parallel (3 GPUs, one per candidate)
- [ ] Monitor: train loss + per-epoch Sintel Final EPE
- [ ] Save: `outputs/retrain_v3_deploy_ft/retrain_v3_deploy_ft_run1/model_<name>/checkpoints/{best,last,sintel_best}.ckpt`
- [ ] Sync ckpts back to local for downstream PTQ INT8 export
- **Status:** pending

### Phase 4: Mainline Fine-Tune (HPC)

- [ ] Submit mainline fine-tune (1 GPU; can run in parallel with Phase 3 if 4-GPU slot)
- [ ] Same eval cadence as v3
- [ ] Sync ckpt back
- **Status:** pending

### Phase 5: Eval + Final Comparison

- [ ] For each fine-tuned ckpt:
  - [ ] Re-export INT8 via Seeed `tools/model_export/edgeflownas_v3/run_export.py` (or mainline-equivalent) at 157×203
  - [ ] Vela compile, check SRAM peak stays within 1432 KiB arena
  - [ ] Sintel Final FP32 + INT8 EPE @ 157×203 (Seeed evaluators)
  - [ ] Also rerun on the original Sintel-Final HPC eval (FP32 @ 416×1024) to verify the fine-tune didn't catastrophically destroy the high-res capability
- [ ] Fill in the deliverable table (see findings.md §10)
- [ ] Decide which model gets flashed to the board (likely whoever has lowest INT8 EPE @ 157×203 + within SRAM budget)
- **Status:** pending

## Key Questions

1. ~~Did EdgeFlowNet mainline training apply a flow_divisor?~~ **NO** — confirmed by reading `EdgeFlowNet/code/misc/utils.py` (only `np.clip(gt, -50, 50)`, no division). Mainline FT uses `flow_divisor=1.0`.
2. ~~What dataset was mainline trained on?~~ **FT3D** at hardcoded patch **480×640** (per `DataHandling.SetupAll`). Same as v3. No domain shift.
3. ~~What input size was mainline trained at?~~ **480×640** for FT3D path (hardcoded in `SetupAll`).
4. **Sintel leakage policy**: We evaluate on Sintel **train Final** 1041 frames. For in-training val we want a labeled set the model has never seen flow GT for.
   - Option A: Use a held-out subset of FT3D val (`TEST` split per `ft3d_dataset.py`). No leakage but val and eval distributions differ.
   - Option B: Use Sintel **train Clean** for val. Different visual pass but same scene geometry → mild leakage; commonly accepted in literature.
   - Option C: Hold out a random 5 scenes from Sintel Final train as in-training val, eval on the remaining 18 scenes. Cleanest but breaks the existing 1041-frame eval comparability with prior numbers.
   - **Tentative decision**: Option A (FT3D TEST) for in-training val + Sintel Final 1041 for **final** offline eval only (don't peek during training). Confirms compatibility with prior numbers in plan/MCUFlowNet_Deployment.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Fine-tune (not full retrain) | ckpts already work well at HPC eval-res; only need spatial-scale adaptation. User explicit ask. |
| Fine-tune data = FT3D (TRAIN split) | Sintel train any-pass would leak into Sintel Final eval. FT3D was original v3 training data. |
| Fine-tune input = 157×203 | Match deploy res exactly. Decided not to try 172×224 (can't compare to mainline fairly). |
| Include mainline in same fine-tune sweep | Mainline original ckpt was trained at 416×1024; comparing FT-v3 vs unFT-mainline would be unfair. |
| One unified trainer file (`deploy_ft_trainer.py`) with `arch_family` switch | Avoid polluting `retrain_v3_trainer.py`; keep mainline + v3 paths converged. |
| LR start 1e-6, cosine to 1e-7 | Original retrain_v3 used 1e-5; fine-tune typically uses 1/10 that. |
| 5-10 epochs | Original 50 epochs from scratch; fine-tune mostly needs BN-stat re-adjustment + late-layer tweaks. |
| Spatial aug scaled down | At 157×203 a `min_scale=-0.4 max_scale=0.8` crop produces ~94×122 to ~283×366 effective — too aggressive; drop `spatial_aug_prob` to 0.3, `max_stretch` to 0.05, disable eraser. |
| Photometric aug kept full strength | Doesn't depend on spatial size. |
| In-training val = FT3D TEST split (Option A in §4) | Avoid any Sintel leakage; final Sintel Final eval is offline-only. |
| early-stop monitor = FT3D val EPE | Aligned with what training optimizes. Final Sintel EPE only used post-hoc. |
| Each candidate gets its own model_dir like retrain_v3 layout | Reuses existing `run_export.py --ckpt-name` flow with minimal changes. |
| mainline `flow_divisor=1.0` during FT | Mainline original training did not divide GT; confirmed pending (Key Q §1). |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
|       |         |            |

## Notes

- This plan lives in `EdgeFlowNAS/plan/`; the matching DEPLOYMENT plan (where INT8 export + board burn happen) lives in `Seeed_Grove_Vision_AI_Module_V2/plan/MCUFlowNet_Deployment/`. Cross-reference both.
- Don't burn or update Seeed `findings.md` from within this folder; the Seeed side records deployment outcomes only.
- Slurm template lives at `plan/retrain_v3/retrain_v3_ft3d_3gpu.slurm`; copy + tweak for fine-tune (lower mem, shorter time, 4 GPUs if possible).
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions
- Log ALL errors here, including HPC failures (OOM, NCCL issues, etc.)
- Never repeat a failed approach without mutation
