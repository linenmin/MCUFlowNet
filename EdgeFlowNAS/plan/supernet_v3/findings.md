# Findings: Supernet V3 Planning

## V2 Plan Review

V2 planned and implemented an 11D mixed-cardinality search space:

- first 6 blocks use 3 choices
- last 5 head blocks use 2 choices
- total space size is `3^6 * 2^5 = 23328`

The V2 plan identified `172x224` as the formal deployment-aligned search-training resolution and kept `180x240` as a historical smoke-compatible config.

## V2 Choice-Order Issue

V2 `search_space_v2.py` currently defines:

- `E0`: `["7x7Conv", "5x5Conv", "3x3Conv"]`
- `E1`: `["5x5Conv", "3x3Conv", "3x3Stride2DilatedConv"]`

This means V2 does not follow the desired convention that `0` is lightest and larger indices are heavier.

V3 should correct this to:

- `E0`: `0 = 3x3`, `1 = 5x5`, `2 = 7x7`
- `E1`: `0 = 3x3`, `1 = 5x5`, `2 = 3x3 + dilated 3x3`

The remaining depth and head dimensions already naturally support light-to-heavy ordering:

- `Deep1`, `Deep2`, `Deep3`
- `3x3`, `5x5`

## V2 Backbone Status

`efnas/network/MultiScaleResNet_supernet_v2.py` is bilinear / resize-conv aligned:

- decoder `Up1` and `Up2` use `resize_conv`
- head upsample blocks use `_head_choice_resize_conv`

However, V2 does not currently include:

- bottleneck ECA
- 1/4 global gate

The only `gate` mention inside the V2 supernet is the one-hot candidate selection gate.

## Existing ECA and Global Gate Sources

Useful reference implementations exist in:

- `efnas/network/fixed_arch_models.py`
- `efnas/network/ablation_edgeflownet_v1.py`

Relevant behavior:

- ECA is applied at the bottleneck as `eca_bottleneck`.
- The global gate uses bottleneck context and scales decoder features.
- The ablation model variant `edgeflownet_bilinear_eca_gate4x` matches the intended fixed skeleton at a high level:
  - bilinear decoder
  - bottleneck ECA
  - 1/4 global gate

V3 should port these operations into the supernet backbone as fixed operations, not add new arch-code dimensions.

## Current V2 Trainer Status

`efnas/engine/supernet_trainer_v2.py` already has several useful pieces for V3:

- 11D arch placeholder
- mixed-cardinality FairNAS-style sampler
- micro-batch accumulation
- global gradient norm clipping
- run manifest resume checks
- eval pool
- checkpoint best/last
- train log with gradient statistics

V3 can reuse the V2 structure by copying and updating the parallel V3 path.

## Current Data Loading Status

`efnas/data/fc2_dataset.py` supports `num_workers` via `ThreadPoolExecutor`.

Current behavior:

- inside `next_batch`, it maps per-sample loading over a thread pool
- it returns only after the whole batch is loaded
- it does not prefetch the next batch while GPU computation is running

Implication:

- increasing workers can help until I/O/CPU saturates
- it can also increase memory pressure
- it does not remove the GPU waiting bubble between training steps

V3 should add bounded asynchronous batch prefetching on top of existing providers.

## Multi-GPU Status

V2 trainer currently only applies a single `gpu_device` setting and creates one TensorFlow session graph.

There is no current multi-GPU tower construction, no `gpu_devices` parser, and no gradient averaging across devices.

V3 should add explicit multi-GPU support rather than relying on TensorFlow implicit placement.

## Multi-GPU Design Finding

For Supernet V3, there are two viable multi-GPU directions.

Data-parallel towers are the most general mode:

- works with 2 or more GPUs
- keeps one logical supernet
- keeps the FairNAS cycle semantics
- preserves global LR/global-step/checkpoint behavior

However, FairNAS-cycle arch parallel is more aligned with the actual supernet training loop:

- one strict-fairness cycle samples 3 subnet architectures
- the current V2 trainer executes those 3 subnet forward/backward passes sequentially
- with 3 GPUs, those 3 subnet passes can be placed on 3 GPUs and accumulated into one optimizer step
- this targets the main repeated compute in the supernet trainer without increasing the logical batch size

The main caveat is BatchNorm in both multi-GPU modes:

- current `tf.layers.batch_normalization` is not synchronized across GPUs
- per-tower or concurrent BN updates mean multi-GPU is not exactly bitwise equivalent to single-GPU training
- this must be logged and documented

Current assessment:

- `arch_parallel` should be the primary V3 multi-GPU target for 3-GPU nodes.
- `data_parallel` should remain a later fallback/generalization for 2-GPU or 4+-GPU nodes.
- Before implementation, the trainer design must explicitly decide how BN update ops are handled in arch-parallel mode.

## Resolution Recommendation

Use `172x224` for formal V3 supernet training.

Reasons:

- this is already identified by V2 as the deployment-aligned resolution
- stem and head choices are resolution-sensitive
- architecture search should match hardware deployment constraints
- high-resolution FC2/FT3D retraining can still be used after search for final subnet training

User confirmed `172x224` on 2026-04-27.

## Vela Precheck Finding

A direct Supernet V3 export with a constant arch code is not valid for Vela subnet comparison because TFLite can retain the full branchy supernet graph.

Observed invalid full-supernet export:

- `sram_mb = 2.857421875`
- `inference_ms = 1571.526265`

This was caused by exporting the branch-selecting supernet graph, not a hard-routed child subnet.

After switching to hard-routed fixed-subnet export for the V3 reference code `[2,1,0,0,0,0,1,1,1,1,1]` at `172x224`, Vela reported:

- `sram_mb = 1.353515625`
- `inference_ms = 147.23362500000002`

The SRAM value is about `1386 KiB`, which matches the Grove model-design target and confirms the V3 fixed backbone can stay within the intended SRAM peak when exported as an actual subnet.

## Open Decisions

1. Confirm `172x224` as the formal V3 supernet training resolution.
2. Confirm whether the first multi-GPU mode should be `arch_parallel` with controlled BN semantics.
3. Confirm V3 should use a parallel code path and leave V2 intact.
4. Confirm bounded prefetch default depth should be `2`.
