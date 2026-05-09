# Findings: Ablation V1

## Current Code Evidence

### Supernet V2 Recheck After User Challenge

- Rechecked `plan/supernet_v2`; it does not contain `globalgate4x_bneckeca`, `bneckeca`, or ECA/global-gate design text. The only "gate" hit is a generic decision-gate heading.
- Rechecked the trained V2 run manifest at `outputs/edgeflownas_supernet_v2_fc2_172x224_run1_balance/run_manifest.json`.
  - `network`: `MultiScaleResNet_supernet_v2`
  - `arch_semantics_version`: `supernet_v2_mixed_11d_front6_3choice_head5_2choice`
  - block names: `E0`, `E1`, `EB0`, `EB1`, `DB0`, `DB1`, `H0Out`, `H1`, `H1Out`, `H2`, `H2Out`
- Rechecked `efnas/engine/supernet_trainer_v2.py`; the graph is instantiated as `MultiScaleResNetSupernetV2(...)`.
- Rechecked `efnas/network/MultiScaleResNet_supernet_v2.py`; decoder/head upsampling uses `resize_conv`, but there is no ECA block and no 1/4 global broadcast gate module.
- `globalgate4x_bneckeca` exists in `efnas/network/fixed_arch_models.py`, not in the V2 supernet/retrain-v2 11D implementation.
- Interpretation: if the intended V2 supernet specification was "use `globalgate4x_bneckeca` as the backbone", the current evidence points to a spec-implementation mismatch. The actually trained V2 supernet appears to be the bilinear/resize-conv 11D search space without ECA/global gate.

### Original EdgeFlowNet

- File: `../EdgeFlowNet/code/network/MultiScaleResNet.py`
- Encoder:
  - initial `7x7` stride-2 convolution
  - `5x5` stride-2 convolution
  - repeated residual blocks with additional downsampling convolutions
- Decoder:
  - uses `ResBlockTranspose`
  - uses `ConvTranspose` for decoder upsampling and prediction heads
- Base layer source:
  - `../EdgeFlowNet/code/network/BaseLayers.py`
  - `ConvTranspose` is implemented with `tf.compat.v1.layers.conv2d_transpose`

### V2 Supernet / Retrain Subnet

- Files:
  - `efnas/network/MultiScaleResNet_supernet_v2.py`
  - `efnas/network/fixed_arch_models_v2.py`
- Confirmed properties:
  - V2 is "Bilinear-aligned" by file docstring and implementation.
  - Decoder upsampling uses `resize_conv`.
  - V2 search space is 11D:
    - E0/E1 stem choices
    - deep block choices
    - 3x3/5x5 head choices
- Important negative finding:
  - No ECA implementation was found in V2 supernet/fixed-subnet files.
  - No global broadcast gate was found in V2 supernet/fixed-subnet files.

### Existing ECA / Global Gate Implementation

- File: `efnas/network/fixed_arch_models.py`
- Existing variants include:
  - `baseline`
  - `globalgate4x_bneckeca`
  - `globalgate4x_bneckeca_skip8x4x`
  - `globalgate4x_bneckeca_skip8x4x2x`
  - `globalgate8x4x_bneckeca`
  - `globalgate8x4x_bneckeca_skip8x`
  - `globalgate8x4x_bneckeca_skip8x4x`
  - `globalgate4x_dual_eca8_bneckeca`
  - `skip8x4x`
- ECA:
  - `_eca_block` global-average-pools spatial dimensions, runs a channel-wise 1D-style conv, applies sigmoid, then scales the input.
  - Existing bottleneck ECA is applied after `DB0` when `_use_bneck_eca()` is true.
- 1/4 global gate:
  - `_global_broadcast_gate` pools a context tensor, projects to target channels with `1x1`, sigmoid-gates the target tensor.
  - Existing 1/4 gate is applied to `net_high` using bottleneck context when `_use_global_gate_4x()` is true.
- Current limitation:
  - This model family is 9D and not the same as V2 11D.
  - Its `baseline` is already a bilinear/resize-conv baseline, not original transposed-conv EdgeFlowNet.

## Training Pipeline Evidence

### Retrain V2

- FC2 command used previously:
  - `python wrappers/run_retrain_v2_fc2.py --config configs/retrain_v2_fc2.yaml --experiment_name retrain_v2_fc2_run1 --supernet_checkpoint outputs/supernet/edgeflownas_supernet_v2_fc2_172x224_run1_balance/checkpoints/supernet_best.ckpt --gpu_device 0`
- FT3D command used previously:
  - `python wrappers/run_retrain_v2_ft3d.py --config configs/retrain_v2_ft3d.yaml --experiment_name retrain_v2_ft3d_run2 --fc2_experiment_dir outputs/retrain_v2_fc2/retrain_v2_fc2 --gpu_device 0`
- Recent fixes relevant to ablation:
  - FT3D excludes 14 known non-finite PFM files.
  - FT3D loader has runtime finite-flow guard.
  - FT3D resume can restore from `best.ckpt` with checkpoint epoch/global-step preserved.

### Fixed-Arch Compare Trainer

- File: `efnas/engine/fixed_arch_compare_trainer.py`
- Strengths:
  - trains multiple variants jointly under the same FC2 data stream
  - writes per-model histories and comparison CSV
  - supports fixed variants from `efnas/network/fixed_arch_models.py`
- Current gaps for requested ablation:
  - FC2 provider is single-threaded.
  - trainer is FC2-only.
  - no FT3D continuation path.
  - no built-in external Sintel evaluation / `sintel_best.ckpt`.
  - no original transposed-conv EdgeFlowNet variant.

### FC2 Provider

- File: `efnas/data/fc2_dataset.py`
- Current behavior:
  - sequentially loads each batch sample in Python.
  - no `ThreadPoolExecutor`.
  - no per-job RNG.
  - no `num_workers` config.
- Performance implication:
  - FC2 likely has the same data-loading bottleneck that FT3D had before multi-worker loading.

## Preliminary Design Judgment

The cleanest ablation implementation is not to keep extending the old 9D `FixedArchModel` in place indefinitely. A better plan is:

1. Create a dedicated ablation model module with the original EdgeFlowNet skeleton represented once.
2. Use feature flags for:
   - deconv vs bilinear upsampling
   - bottleneck ECA
   - 1/4 global gate
3. Build trainer support around named ablation variants and staged `FC2 -> FT3D` training.
4. Reuse proven retrain_v2 infrastructure where possible:
   - FT3D provider
   - finite-flow guards
   - resume-state handling
   - external Sintel runtime
   - checkpoint metadata layout

## Risk Register

| Risk | Why It Matters | Mitigation |
|---|---|---|
| Original EdgeFlowNet arch-code mapping may be ambiguous | A misleading arch code could make the ablation scientifically weak. | Add graph/shape tests and document the exact mapping before training. |
| Jointly training all four variants may exceed A100 memory | Four full models plus optimizer slots can be heavy at FC2 352x480 or FT3D 480x640. | Start with graph/memory smoke test; split jobs if needed. |
| Sintel proxy during training is expensive | Evaluating four models every few epochs may dominate wall time. | Use configurable cadence or max samples for early smoke, full evaluation for final runs. |
| A0 direct original wrapper may not share checkpoint/eval contracts | Hard to compare with A1-A3. | Prefer behavior-matched EdgeFlowNAS implementation unless direct original parity is required. |

## 2026-05-09 Skeleton Mismatch vs Supernet V3 (ECA / gate4x placement)

Ablation V1 is intended to ablate the fixed backbone of Supernet V3, so A2/A3 must wire ECA and the 1/4 global gate at the same graph positions as `efnas/network/MultiScaleResNet_supernet_v3.py`. They currently do not.

### Helper functions are identical
- `_eca_block` and `_global_broadcast_gate` in `efnas/network/ablation_edgeflownet_v1.py` (lines 106-136) and in `efnas/network/MultiScaleResNet_supernet_v3.py` (lines 78-110) are byte-for-byte equivalent (mean -> reshape-1d -> 1xK conv -> sigmoid -> scale; and mean -> 1x1 conv -> sigmoid -> multiply).

### `eca_bottleneck` is at different stages
- Supernet V3 (`MultiScaleResNet_supernet_v3.py:176-178`): ECA is applied to `net_low` immediately after `DB0`, i.e. at the encoder's deepest 1/16 feature, before any decoder upsampling. `bottleneck_context` is set from this same deep tensor.
- Ablation V1 (`ablation_edgeflownet_v1.py:148-164`): the encoder ResBlock+Down stack and the decoder ResBlock+Upsample stack both run first. Only after the entire decoder stack does the variant set `net = _eca_block(net)` and `bottleneck_context = net`. The tensor named "bottleneck" in the ablation code is actually `feat_low`, the input of `OutLow` (the 1/4 head), not the encoder bottleneck.

### `global_gate_4x` covers a different output head
- Supernet V3 (`MultiScaleResNet_supernet_v3.py:185-196`): the gate modulates `net_high` after `Up2`, which is the 1/4-stride feature feeding `H0Out` (the `out_1_4` head). Context is the deep `bottleneck_context` from after `DB0`. Context resolution is far smaller than target resolution, so the broadcast is genuine.
- Ablation V1 (`ablation_edgeflownet_v1.py:167-180`): `out_1_4 = OutLow(net)` is emitted before any gating; the gate is applied to the `UpMid` output and feeds `out_1_2`, not `out_1_4`. The context (`feat_low`) is at the same resolution as the gated target's predecessor, so the "global" broadcast collapses to a same-or-near-same-resolution gating.

### Consequence
A2 and A3 in `outputs/ablation_v1_fc2/` were trained against this mis-aligned skeleton:
- `ablation_v1_fc2_run1_p100_edgeflownet_bilinear_eca`
- `ablation_v1_fc2_run1_p100_edgeflownet_bilinear_eca_gate4x`

Their results do not isolate the ECA-at-bottleneck and 1/4-broadcast-gate behaviors that Supernet V3 is built around. A1 (`edgeflownet_bilinear`) and A0 (`edgeflownet_deconv`) are unaffected because they set `bottleneck_eca=False` and `gate_4x=False`.

### Required fix
Re-wire `ABlationEdgeFlowNetV1.build` so the encoder/decoder layout mirrors Supernet V3:
1. Build a true encoder bottleneck (deepest feature before any decoder upsampling), apply `_eca_block` there, and snapshot `bottleneck_context` from this deep tensor.
2. Continue the decoder up to the 1/4 stage. Apply `_global_broadcast_gate(context_inputs=bottleneck_context, target_inputs=<1/4 feature>)` before the 1/4 head reads from it.
3. Emit `out_1_4` after the gate, then continue upsampling to 1/2 and 1/1 heads as before.
4. Keep `_eca_block` / `_global_broadcast_gate` helper bodies unchanged; only their call sites move.

### Retraining
- A0 (`edgeflownet_deconv`) and A1 (`edgeflownet_bilinear`): no retraining needed (skeleton change does not affect their graphs).
- A2 (`edgeflownet_bilinear_eca`) and A3 (`edgeflownet_bilinear_eca_gate4x`): must be retrained from scratch under the corrected skeleton. The existing checkpoints under `outputs/ablation_v1_fc2/...edgeflownet_bilinear_eca[_gate4x]/` are not weight-compatible with the new placement (different parent feature, different channel widths at gate sites) and should be archived rather than fine-tuned.
