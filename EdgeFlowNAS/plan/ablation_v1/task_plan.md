# Task Plan: Ablation V1 Backbone Validation

## Goal

Build a maintainable ablation pipeline to test whether the proposed EdgeFlowNAS backbone changes are justified relative to original EdgeFlowNet under the same FC2 -> FT3D training protocol.

## Target Ablation Matrix

All models should keep the same EdgeFlowNet-style block-depth architecture once that mapping is verified.

| ID | Model | Intended Difference |
|---|---|---|
| A0 | `edgeflownet_deconv` | original EdgeFlowNet decoder with transposed convolution upsampling |
| A1 | `edgeflownet_bilinear` | same backbone/block depths, replace transposed convolution upsampling with bilinear/resize-conv |
| A2 | `edgeflownet_bilinear_eca` | A1 plus bottleneck/middle ECA channel attention |
| A3 | `edgeflownet_bilinear_eca_gate4x` | A2 plus 1/4-resolution global broadcast gate |

## Current Phase

Phase 1 - design and implementation plan. A detailed execution proposal now exists in `execution_plan.md` and is waiting for user approval.

## Updated Baseline Finding

After rechecking `plan/supernet_v2`, the V2 run manifest, and the V2 trainer/model code, the trained second supernet appears to use `MultiScaleResNet_supernet_v2`, not `globalgate4x_bneckeca`. The `globalgate4x_bneckeca` implementation exists in the older fixed-arch variant module. Therefore this ablation should be framed as both:

- a backbone-design ablation for `deconv -> bilinear -> bilinear + ECA -> bilinear + ECA + gate4x`
- a correction/verification path for the possible mismatch between the intended V2 backbone and the actual V2 supernet implementation

## Phases

### Phase 1: Scope And Code Audit

- [x] Check current V2 supernet/fixed-subnet skeleton for bilinear/ECA/global-gate components.
- [x] Check original EdgeFlowNet decoder implementation.
- [x] Check existing fixed-arch compare variants and trainer.
- [x] Draft detailed ablation execution settings for user approval.
- [ ] Verify exact arch-code mapping that reproduces original EdgeFlowNet block depths.
- [ ] Decide whether A0 should wrap original `EdgeFlowNet/code/network/MultiScaleResNet.py` directly or be reimplemented in EdgeFlowNAS with matching naming and tests.

### Phase 2: Maintainable Model Design

- [x] Create a single ablation model implementation with explicit feature flags:
  - `upsample_mode`: `deconv` or `bilinear`
  - `bottleneck_eca`: true/false
  - `gate_4x`: true/false
- [x] Keep common encoder/decoder block-depth wiring shared across all variants.
- [x] Add graph-construction tests for all four variants.
- [ ] Add parameter/FLOP or Vela-summary export hooks so architecture changes can be reported alongside accuracy.

### Phase 3: Training Pipeline

- [x] Build `ablation_v1` configs for:
  - FC2: 400 epochs
  - FT3D: 50 epochs
- [x] Add FC2 multi-worker prefetch/data loading parity with the current FT3D loader.
- [x] Add FT3D bad-PFM filtering and runtime finite-flow guard to ablation training.
- [x] Support FC2 -> FT3D continuation from each model's FC2 best checkpoint.
- [x] Preserve resume semantics including checkpoint epoch/global-step based LR schedule.
- [x] Add useful `train.log` logging for startup, epoch, validation, checkpoint, resume, and FT3D skipped-sample summaries.
- [x] Add tqdm progress bars for in-domain validation and Sintel evaluation, not only training.

### Phase 4: Evaluation And Checkpoint Policy

- [x] Save `best.ckpt` on in-domain validation EPE for each model.
- [x] Save `sintel_best.ckpt` during training using the same external Sintel proxy path as retrain_v2.
- [x] Produce per-stage `comparison.csv` with FC2/FT3D EPE, Sintel EPE, delta versus A0, and delta versus A1.
- [ ] Add a final reporting script that combines FC2, FT3D, Sintel, model size, and latency/Vela metrics.

### Phase 5: HPC Execution Plan

- [ ] Define one reproducible sbatch template for FC2 and one for FT3D.
- [ ] Decide whether to train all four variants jointly on one GPU or split into separate jobs.
- [ ] Benchmark FC2 loader worker counts before launching 400-epoch jobs.
- [ ] Record exact commands and expected output directories.

## Key Decisions

- Prefer a single parameterized ablation model over four copied model classes.
- Prefer training all variants under the same data stream when memory permits; if GPU memory is too high, split into per-model jobs but keep the exact same seed/sampling contract.
- Treat V2 `MultiScaleResNetSupernetV2` as evidence for bilinear/resize-conv only. It is not evidence that ECA/global-gate was present in the trained second supernet.
- FC2 LR is `1.0e-4 -> 1.0e-6` by user request.
- Keep early stopping enabled by user request.
- Set `grad_clip_global_norm=200.0` by default and expose it as a CLI override.
- Use EdgeFlowNAS-native A0 implementation and validate equivalence against original EdgeFlowNet with Vela.
- Enable online Sintel validation during training for `sintel_best.ckpt` selection.

## Open Questions

1. What exact 9D fixed-arch code should represent "same as EdgeFlowNet"?
   - Existing compare configs use `0,2,1,1,0,0,0,0,0`.
   - This may be a prior comparison code, not necessarily a faithful original EdgeFlowNet block-depth mapping.
2. What exact Vela acceptance tolerance should be used for A0 equivalence metrics?
3. Should ECA be placed only at the bottleneck (`eca_bottleneck`) or also at another "middle" decoder point?
4. Should the 1/4 global gate use bottleneck context only, matching existing `_global_broadcast_gate(context_inputs=bottleneck_context, target_inputs=net_high)`?

## Success Criteria

- Four variants train under a single documented config system.
- FC2 and FT3D stages are reproducible and resumable.
- FC2 uses multi-worker data loading, and FT3D keeps the bad-PFM exclusion guard.
- Each model produces `best.ckpt`, `sintel_best.ckpt`, histories, manifests, and final comparison tables.
- The final results can support or falsify the claim that `bilinear + ECA + 1/4 global gate` is a reasonable backbone design.

## Pending Approval

- Detailed execution proposal: `plan/ablation_v1/execution_plan.md`
- Core training-policy decisions are now approved; implementation can start from `execution_plan.md`.

## Errors Encountered

| Date | Error | Resolution |
|---|---|---|
| 2026-04-27 | User noted that V2 was intended to use `globalgate4x_bneckeca`; re-audit found `supernet_v2` plan, manifest, trainer, and model code do not show that implementation. | Record as a likely intended-spec versus actual-implementation mismatch; keep ablation focused on verifying this backbone design explicitly. |
