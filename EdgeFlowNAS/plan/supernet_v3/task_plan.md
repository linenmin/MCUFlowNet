# Task Plan: Supernet V3 Backbone and Training

## Goal

Implement Supernet V3 as the next formal supernet training path for EdgeFlowNAS.

V3 should correct the V2 architecture-code semantics, fix the intended backbone design into the supernet, improve the input pipeline with bounded prefetching, and add a maintainable multi-GPU training path.

This plan is a design draft pending final multi-GPU strategy approval. It does not authorize code changes yet.

## Confirmed Requirements

- Use an 11D architecture code.
- Keep the same 11 searchable positions as V2:
  - `E0`, `E1`, `EB0`, `EB1`, `DB0`, `DB1`, `H0Out`, `H1`, `H1Out`, `H2`, `H2Out`.
- Correct the first two dimensions so that all choices follow the same convention:
  - `0 = lightest`
  - `1 = medium`
  - `2 = heaviest`
- Fix the intended backbone into the supernet:
  - bilinear / resize-conv decoder
  - bottleneck ECA
  - 1/4 global gate from bottleneck context
- Include faster data loading with data prefetching.
- Include a multi-GPU acceleration path.
- Keep the code maintainable and avoid broad edits to V1/V2 paths unless necessary.

## Recommended Training Resolution

Use `172x224` as the formal Supernet V3 training resolution.

Rationale:

- V2 planning already identified `172x224` as the deployment-aligned resolution.
- The supernet search includes stem and head choices whose latency and quality tradeoffs are resolution-sensitive.
- The supernet's purpose is architecture selection under the deployment target, so the search-training resolution should match the deployed inference resolution rather than later high-resolution retraining.
- `180x240` can remain a historical smoke-test resolution only if needed.

Recommended config names:

- formal config: `configs/supernet_v3_fc2_172x224.yaml`
- optional smoke config: `configs/supernet_v3_fc2_180x240_smoke.yaml`

## V3 11D Search Space

### Corrected Choice Semantics

All dimensions should be ordered from light to heavy.

| Index | Name | Choices |
| --- | --- | --- |
| 0 | `E0` | `0 = 3x3 stride-2`, `1 = 5x5 stride-2`, `2 = 7x7 stride-2` |
| 1 | `E1` | `0 = 3x3 stride-2`, `1 = 5x5 stride-2`, `2 = 3x3 stride-2 + 3x3 dilated` |
| 2 | `EB0` | `0 = Deep1`, `1 = Deep2`, `2 = Deep3` |
| 3 | `EB1` | `0 = Deep1`, `1 = Deep2`, `2 = Deep3` |
| 4 | `DB0` | `0 = Deep1`, `1 = Deep2`, `2 = Deep3` |
| 5 | `DB1` | `0 = Deep1`, `1 = Deep2`, `2 = Deep3` |
| 6 | `H0Out` | `0 = 3x3`, `1 = 5x5` |
| 7 | `H1` | `0 = 3x3`, `1 = 5x5` |
| 8 | `H1Out` | `0 = 3x3`, `1 = 5x5` |
| 9 | `H2` | `0 = 3x3`, `1 = 5x5` |
| 10 | `H2Out` | `0 = 3x3`, `1 = 5x5` |

Search-space size remains:

```text
3^6 * 2^5 = 23328
```

### Reference Codes

- Lightest code: `[0,0,0,0,0,0,0,0,0,0,0]`
- Heaviest front/backbone with heavy heads: `[2,2,2,2,2,2,1,1,1,1,1]`
- V2 reference equivalent after reordered stem semantics: `[2,1,0,0,0,0,1,1,1,1,1]`

The `arch_semantics_version` must change so V2 checkpoints cannot be resumed accidentally as V3:

```text
supernet_v3_mixed_11d_light_to_heavy_fixed_bilinear_bneckeca_gate4x
```

## Backbone Design

V3 should use a new network class rather than mutating V2 in place:

- `efnas/network/MultiScaleResNet_supernet_v3.py`

Required fixed backbone behavior:

- Use resize-conv / bilinear upsampling in decoder paths.
- Apply ECA at the bottleneck after `DB0`.
- Use bottleneck context to gate the 1/4 decoder feature after `Up2`.
- Keep only the 11D operator choices searchable; do not add ECA/gate as arch-code dimensions.

Expected feature points:

- `net_low`: bottleneck feature after `DB0` and bottleneck ECA.
- `net_mid`: decoder feature after `Up1` and `DB1`.
- `net_high`: 1/4 decoder feature after `Up2`, BN/ReLU, and 1/4 global gate.

## Data Pipeline Plan

Current FC2 loading uses a `ThreadPoolExecutor` inside each `next_batch` call. That parallelizes per-sample loading, but it does not prefetch the next batch while the GPU is computing.

V3 should add bounded asynchronous batch prefetch:

- add a reusable prefetch wrapper, for example `efnas/data/prefetch_provider.py`
- wrap FC2 providers when `data.prefetch_batches > 0`
- use a bounded queue to avoid the memory growth observed in P100 jobs
- default to conservative prefetching:
  - `fc2_num_workers: 12` on full A100 jobs
  - `fc2_eval_num_workers: 2` or `4`
  - `prefetch_batches: 2`
- log queue depth, worker counts, and effective provider type at startup

The prefetcher must support:

- `next_batch(batch_size)`
- `start_epoch(shuffle=True)`
- `reset_cursor(index=0)` if the wrapped provider supports it
- `len(provider)`
- clean shutdown on trainer exit

## Multi-GPU Plan

V3 should support multi-GPU training as an explicit mode.

### Candidate Mode A: Data Parallel

Generic data-parallel mode:

```yaml
train:
  gpu_devices: "0,1"
  multi_gpu_mode: "data_parallel"
```

Data-parallel semantics:

- Keep global `batch_size` as the logical batch size.
- Split each logical batch across GPUs.
- Build one shared-variable tower per GPU.
- Use the same arch code on every tower for a given accumulation item.
- Average tower gradients.
- Preserve the existing FairNAS cycle accumulation over the 3 sampled arch codes.
- Apply optimizer once per logical FairNAS training step.
- Keep LR schedule, global step, fairness counters, checkpointing, and eval semantics tied to the logical step.

Important caveat:

- Current `tf.layers.batch_normalization` is not synchronized across GPUs.
- V3 multi-GPU data parallel will have per-tower BN statistics unless sync-BN is implemented.
- The first version should log this clearly and keep single-GPU as the reproducibility baseline.

### Candidate Mode B: FairNAS Arch Parallel

FairNAS-specific arch-parallel mode:

```yaml
train:
  multi_gpu_mode: "arch_parallel"
```

This maps the 3 subnets sampled in one strict-fairness cycle onto 3 GPUs:

- GPU 0 trains subnet A.
- GPU 1 trains subnet B.
- GPU 2 trains subnet C.
- All towers use the same logical input batch.
- Gradients are averaged or summed exactly as the current sequential 3-subnet accumulation does before one optimizer step.
- The optimizer still applies once per FairNAS cycle.
- LR, global step, fairness counters, and checkpointing remain identical at the logical-step level.

This is algorithmically attractive because V2/V3 currently spend one training step running 3 subnet forward/backward passes sequentially. With 3 GPUs, arch parallel can target nearly 3x speedup for the supernet training step without increasing the logical batch size.

Risks and caveats:

- Shared fixed-layer BatchNorm moving averages are updated by all three towers. Naive parallel update ops can be nondeterministic or last-writer-wins.
- Choice-branch BN variables are mostly disjoint per sampled operation, but fixed stem/decoder/head layers are shared across towers.
- To be faithful, V3 must either:
  - keep BN updates controlled on one tower / sequentially for BN only, or
  - explicitly average BN moving-stat updates, or
  - accept and log that arch-parallel BN semantics differ from single-GPU sequential training.
- Requires exactly 3 GPUs for the cleanest mapping. With 2 or 4 GPUs, scheduling is less natural.

### Current Recommendation

Prioritize `arch_parallel` as the Supernet V3-specific multi-GPU design, with `single_gpu` as the reproducibility baseline.

Keep `data_parallel` as a later generalization or fallback because it is less aligned with FairNAS strict-fairness cycles and changes effective per-tower batch/BN behavior.

## File Layout

Recommended new files:

- `configs/supernet_v3_fc2_172x224.yaml`
- `wrappers/run_supernet_train_v3.py`
- `efnas/app/train_supernet_app_v3.py`
- `efnas/network/MultiScaleResNet_supernet_v3.py`
- `efnas/nas/search_space_v3.py`
- `efnas/nas/arch_codec_v3.py`
- `efnas/nas/fair_sampler_v3.py`
- `efnas/nas/eval_pool_builder_v3.py`
- `efnas/engine/supernet_trainer_v3.py`
- `efnas/data/prefetch_provider.py`
- `tests/test_supernet_v3_space_helpers.py`
- `tests/test_supernet_network_structure_v3.py`
- `tests/test_run_supernet_train_wrapper_v3.py`
- `tests/test_prefetch_provider.py`

V2 should remain available for checkpoint analysis and comparison.

## Implementation Phases

### Phase 1: V3 Scope Lock

- [x] Confirm `172x224` as the formal training resolution.
- [x] Confirm 11D block names remain identical to V2.
- [x] Confirm the corrected light-to-heavy choice order.
- [x] Confirm bottleneck ECA and 1/4 global gate are fixed backbone operations, not architecture-code dimensions.
- **Status:** complete

### Phase 2: Search-Space and Codec

- [x] Add `search_space_v3.py` with corrected choice labels.
- [x] Add `arch_codec_v3.py` with strict validation and readable labels.
- [x] Add V3 eval-pool builder.
- [x] Add tests for valid/invalid 11D codes and V2-to-V3 reference mapping.
- **Status:** complete

### Phase 3: Network Graph

- [x] Add `MultiScaleResNet_supernet_v3.py`.
- [x] Port V2 resize-conv supernet structure.
- [x] Add bottleneck ECA.
- [x] Add 1/4 global gate.
- [x] Verify multiscale outputs and feature pyramid names.
- [x] Add TensorFlow graph-construction tests.
- **Status:** complete

### Phase 4: Bounded Prefetch

- [x] Add reusable prefetch wrapper.
- [x] Add config fields for train/eval prefetch depth.
- [x] Wrap FC2 providers in V3 trainer.
- [x] Add shutdown handling and memory-safe bounded queues.
- [x] Add tests for ordering, reset, and shutdown behavior.
- **Status:** complete

### Phase 5: Trainer V3

- [x] Add V3 wrapper/app/trainer files.
- [x] Preserve V2 resume safety and manifest behavior.
- [x] Update manifest signatures to include V3 arch semantics, backbone flags, resolution, prefetch, and multi-GPU mode.
- [x] Preserve micro-batch accumulation.
- [x] Preserve gradient clipping statistics.
- [x] Preserve eval-pool validation and best/last checkpoints.
- **Status:** complete

### Phase 6: Multi-GPU Training

- [x] Add `gpu_devices` parser and config support.
- [x] Implement `single_gpu` baseline mode.
- [x] Implement `arch_parallel` mode for 3-GPU FairNAS-cycle training.
- [x] Build shared-variable arch towers with one sampled subnet per GPU.
- [x] Keep one logical batch shared across the 3 arch towers.
- [x] Average/sum tower gradients to match current sequential FairNAS-cycle accumulation.
- [x] Keep BN update caveat explicit for first production smoke; exact sync-BN is not implemented.
- [x] Log BN caveat, GPU count, and effective per-tower subnet assignment.
- [x] Add graph-level tests where possible.
- **Status:** complete

### Phase 7: Validation

- [x] Run non-TF unit tests.
- [x] Run TensorFlow graph tests in `tf_work_hpc`.
- [x] Run `--dry_run`.
- [x] Run Vela fixed-subnet precheck at `172x224`.
- [ ] Run 1-step smoke test at `172x224` on HPC data.
- [ ] Run single-GPU 1-epoch smoke test.
- [ ] Run 3-GPU arch-parallel smoke test if a multi-GPU node is available.
- [ ] Check GPU utilization, memory, and epoch time.
- **Status:** in_progress

## Decision Gate

Before implementation, confirm:

1. `172x224` is the formal Supernet V3 training resolution. `Confirmed`
2. multi-GPU first implementation should be `arch_parallel` for 3-GPU FairNAS-cycle training, unless BN semantics make it too risky.
3. prefetch should default to bounded `prefetch_batches=2`.
4. V3 should be a parallel code path rather than modifying V2 in place.
