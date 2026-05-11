# Progress Log: Retrain V3 Deploy-Resolution Fine-Tune

Append entries with the most recent at the top. Don't rewrite history.

---

## Next Action

**Phase 2 done; submit on HPC.**

Steps for HPC (user does this):
1. `git pull` in `$VSC_DATA/test/MCUFlowNet/EdgeFlowNAS/` to sync new files.
2. Verify FT3D paths exist: `ls ../Datasets/FlyingThings3D/frames_cleanpass/TRAIN/`.
3. Verify Sintel paths exist (for in-training Sintel eval): `ls ../Datasets/Sintel/training/final/`.
4. Verify EdgeFlowNet ckpt is at `../EdgeFlowNet/checkpoints/best.ckpt` relative to EdgeFlowNAS root.
5. Submit:
   ```
   sbatch plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_4gpu.slurm
   ```
6. Monitor:
   ```
   squeue -u $USER
   tail -f outputs/retrain_v3_deploy_ft/retrain_v3_deploy_ft_run1/launcher_logs/v3_efn_fps.log
   tail -f outputs/retrain_v3_deploy_ft/retrain_v3_deploy_ft_run1/launcher_logs/mainline.log
   ```
7. Expected timeline (rough): ~30 min/epoch × ~10 epochs (early stop) ≈ 5 hours per candidate; all 4 run in parallel so wall clock ~5h total.

After training completes (Phase 3 / Phase 4 done), Phase 5 kicks in:
- Sync fine-tuned ckpts to local
- Re-export INT8 via Seeed `tools/model_export/edgeflownas_v3/run_export.py` (need a small flag for ckpt-path override per candidate)
- For mainline FT, add a parallel export entry in Seeed tools
- Sintel Final eval at 157×203 + 416×1024 to fill findings.md §10 deliverable table
- Pick the winner and flash to board

User instruction was "开始写 fine-tune 计划". The plan is now complete (Phase 1 done). Awaiting confirmation before starting code work.

Phase 2 starting tasks (drop into EdgeFlowNAS source after user approval):
1. `efnas/network/edgeflownet_mainline.py` — wrap EdgeFlowNet `MultiScaleResNet` (transpose-conv decoder, UncType="LinearSoftplus"). Verify the wrapper restores `EdgeFlowNet/checkpoints/best.ckpt` cleanly (variable scopes must match).
2. `efnas/engine/deploy_ft_trainer.py` — arch_family-aware fine-tune trainer; reuse FT3D dataloader, sintel-eval runtime, ckpt manager.
3. `configs/retrain_v3_deploy_ft.yaml` — input 157×203, epochs 5-10, LR 1e-6 → 1e-7 cosine, aug knobs per findings §9.
4. `wrappers/run_deploy_ft_batch.py` — multi-candidate dispatcher (mirror `run_retrain_v3_batch.py`).
5. `plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_candidates.csv` — 4 rows.
6. `plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_4gpu.slurm` — copy + modify from `plan/retrain_v3/retrain_v3_ft3d_3gpu.slurm`.
7. Smoke test locally with `--limit-batches 2 --num-epochs 1` before HPC submission.

---

## 2026-05-11 — Day 0: Plan Created + Phase 1 closed + Phase 2 complete

**Phase 2 (code implementation) done in one session.** Smoke tests pass for both arch families.

New files (all committed):
- `efnas/network/edgeflownet_mainline.py` — self-contained port of MultiScaleResNet + BaseLayers + Decorators (no cross-repo sys.path tricks). Verified `EdgeFlowNet/checkpoints/best.ckpt` restores cleanly (86 fwd vars).
- `efnas/engine/deploy_ft_trainer.py` — arch_family-aware FT trainer with constant LR + early stopping. Reuses `_build_graph` (v3) + adds `_build_mainline_graph` (mainline, no outer scope, BN inference mode).
- `efnas/engine/deploy_ft_sintel_runtime.py` — Sintel eval dispatcher; mainline uses no flow_scale (matches training).
- `configs/retrain_v3_deploy_ft.yaml` — input 157×203, FT3D, LR 1e-6 constant, 20 ep / 5 patience.
- `wrappers/run_deploy_ft_one.py`, `wrappers/run_deploy_ft_batch.py` — runner + 4-GPU dispatcher.
- `plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_candidates.csv` — 4 rows (v3_acc / v3_efn_fps / v3_light / mainline).
- `plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_4gpu.slurm` — 4× p100 / 36 cores / 180G / 24h.

Smoke results:
- Dry-run: 4-candidate commands generated correctly.
- Build+restore: v3_efn_fps (136 fwd vars from sintel_best.ckpt) ✓, mainline (86 fwd vars from EdgeFlowNet/best.ckpt) ✓.
- 1-step training on synthetic data: both families forward + loss + grad accum + train_op + EPE ✓.

---

## 2026-05-11 — Day 0 earlier: Plan Created + Phase 1 closed

**Phase 1 close-out (mainline metadata investigation):**
- Read `EdgeFlowNet/code/train.py`, `wrappers/run_train.py`, `misc/DataHandling.py`, `misc/utils.py`.
- Answered Key Q §1-§3:
  - mainline trained on **FT3D** (same as v3)
  - mainline patch size **480×640** (hardcoded in `SetupAll`, same as v3)
  - mainline `flow_divisor = 1.0` (no GT division in `get_sintel_batch`; only clip ±50)
- **Big finding**: v3 and mainline have IDENTICAL training data + input size; the Δ_downsample gap (v3 +2.86~+5.02 vs mainline +1.40) is **architectural**, not data. v3's bilinear-ResizeConv + ECA-bottleneck + global-broadcast-gate stack is more sensitive to spatial-scale than mainline's transpose-conv decoder.
- Implication: mainline + v3 can share the same FT3D data pipeline at 157×203. No domain shift.
- §12 risk "Domain shift for mainline FT" marked RESOLVED in findings.md.
- task_plan Phase 1 marked complete; Current Phase advanced to Phase 2.



**Done:**
- Created `D:\Dataset\MCUFlowNet\EdgeFlowNAS\plan\retrain_v3_deploy_ft\` and the three pi-planning files (task_plan / findings / progress).
- Anchored all hard facts from the Seeed deployment plan (§7b/§7d/§7e1) into `findings.md §1-§3` so this experiment doesn't lose context.
- Listed the 4 candidates (v3_acc / v3_efn_fps / v3_light / mainline) with their original checkpoints and HPC eval numbers.
- Drafted hyperparameter deltas vs the retrain_v3 from-scratch run (`findings.md §9`).
- Drafted code layout (`findings.md §11`).
- Drafted deliverable table skeleton (`findings.md §10`).

**Carried over (still open):**
- Key Questions §1-§4 in `task_plan.md` — need investigation before code work begins.
- Choice of in-training val data — leaning Option A (FT3D TEST split), to be confirmed.
- Mainline `MultiScaleResNet` port correctness — flagged risk in `findings.md §12`.

**Files created / modified:**
| File | Type |
|------|------|
| `plan/retrain_v3_deploy_ft/task_plan.md` | created |
| `plan/retrain_v3_deploy_ft/findings.md` | created |
| `plan/retrain_v3_deploy_ft/progress.md` | created (this file) |

**Phase status:**
- Phase 1: in_progress (3 of ~5 sub-bullets done)
- Phase 2-5: pending

---

## Session Reboot Checklist

If you pick this up cold later:
1. Read `task_plan.md` — see Current Phase + Key Questions.
2. Read `findings.md §10` to recall the deliverable table.
3. Read this file top entry (Next Action).
4. Spot-check pre-fine-tune baseline still matches Seeed `plan/MCUFlowNet_Deployment/findings.md §7b` (no upstream change).
5. Verify FT3D paths exist on HPC: `ls $VSC_DATA/test/MCUFlowNet/Datasets/FlyingThings3D/`.
6. Continue from Next Action.

---

## Test Results Log (populated in Phase 5)

| Model | FP32 @ 416×1024 (FT) | FP32 @ 157×203 (FT) | INT8 @ 157×203 (FT) | Δ vs orig | Vela peak | Vela ms | Notes |
|---|---|---|---|---|---|---|---|
| Mainline    |  |  |  |  |  |  |  |
| v3_acc      |  |  |  |  |  |  |  |
| v3_efn_fps  |  |  |  |  |  |  |  |
| v3_light    |  |  |  |  |  |  |  |
