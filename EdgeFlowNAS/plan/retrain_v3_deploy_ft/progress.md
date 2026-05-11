# Progress Log: Retrain V3 Deploy-Resolution Fine-Tune

Append entries with the most recent at the top. Don't rewrite history.

---

## Next Action

**Close out Phase 1 (Plan & Design) by answering Key Questions §1–3 (mainline metadata).**

Concretely:
1. Open `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\train_sintel.py` (or whatever the training entry is) and read:
   - default `--patch_dim_0`, `--patch_dim_1`, `--patch_channels`
   - whether `flow_divisor` (or any `/ N` operation on GT) appears in the data loader
   - what dataset (Sintel-only, FT3D-only, mixed) is iterated
   - augmentation settings
2. Open `EdgeFlowNet/wrappers/run_train.py` to see how the production training was launched.
3. Record findings in `findings.md §6` (replace the "Verification still needed" subsection).
4. Re-decide §12 risk on mainline FT data: if mainline was trained on Sintel, propose alternative FT data (e.g., Sintel train Clean + 80/20 hold-out, or FT3D + small Sintel mix).
5. Once Key Q §1–3 are answered AND the in-training val policy (Key Q §4 → tentative Option A) is locked, mark Phase 1 complete in `task_plan.md` and move to Phase 2 (code implementation).

Estimated time: 30–60 min for the metadata investigation, then proceed to Phase 2.

---

## 2026-05-11 — Day 0: Plan Created

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
