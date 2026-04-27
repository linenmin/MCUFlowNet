# Findings: Retrain V2

## Confirmed Facts

### 2026-04-27 FT3D Run2 NaN Evidence

The current FT3D run has a clean metric trajectory through epoch 32 and then simultaneous NaNs for both selected architectures at epoch 34:

| Epoch | mean_epe | epe_knee | epe_fast | sintel_epe_knee | sintel_epe_fast |
|---:|---:|---:|---:|---:|---:|
| 32 | 2.037560720208699 | 1.9421447476720421 | 2.132976692745356 | 5.472796440124512 | 5.75335693359375 |
| 34 | nan | nan | nan | nan | nan |
| 36 | nan | nan | nan | nan | nan |

This is not just a `comparison.csv` aggregation problem:

- `model_fast/eval_history.csv` has `loss=nan`, `epe=nan`, and `sintel_epe=nan` at epoch 34.
- `model_knee/eval_history.csv` has the same transition at epoch 34.
- `best_epe` and `best_sintel_epe` remain finite, so the best-checkpoint bookkeeping did not overwrite the clean epoch-32 checkpoints.
- `last.ckpt.meta.json` at epoch 36 records `metric: NaN` for both models; `last.ckpt` should be considered contaminated until proven otherwise.

The strongest current hypotheses are:

1. A shared train batch contained non-finite or extreme labels/inputs and pushed both models into non-finite weights.
2. The FT3D data is finite, but the shared training graph became numerically unstable after epoch 32 because `grad_clip_global_norm` is currently `0.0`.
3. Resume/output bookkeeping is confusing analysis because copied artifacts contain inconsistent outer and nested histories, but this does not by itself explain NaN losses inside both model histories.

Important nuance:

- With `shuffle_no_replacement`, a static corrupt training PFM should usually be encountered in every full training epoch, so "bad file appears only at epoch 34" is not the most natural explanation unless the dataset changed during training, the run was resumed with a fresh provider RNG state, or the bad values are triggered by stochastic augmentation/batch composition.
- Because `np.clip` preserves NaN values, the loader currently would not filter a NaN-containing PFM; a direct scan remains necessary.

The direct PFM scan has now confirmed this data-quality fault:

- total scanned: `107040`
- bad files: `14`
- bad mode: `non-finite`
- distribution: `13` under `TRAIN`, `1` under `TEST`

This makes "FT3D data contains corrupt labels" a confirmed contributor, not just a hypothesis. A corrupt `TRAIN` PFM can poison model weights through NaN loss/gradients; a corrupt `TEST` PFM can also make validation EPE NaN. The simultaneous NaNs in both candidate models are consistent with both models consuming the same corrupted shared batch.

### Selected Architectures

The two candidate architectures to retrain are:

1. `2,1,0,1,2,1,0,0,0,0,1`
2. `2,1,0,0,0,0,0,0,0,0,0`

From the Sintel validation shortlist:

| Arch Code | FPS | FC2 EPE | Sintel EPE |
|---|---:|---:|---:|
| `2,1,0,1,2,1,0,0,0,0,1` | 5.896121 | 4.107600 | 5.607823 |
| `2,1,0,0,0,0,0,0,0,0,0` | 9.549454 | 4.976324 | 6.479133 |

### V2 Search Space Is 11D

The retrain target is not the old 9D search space.

V2 uses:

- first 6 blocks: 3 choices each
- last 5 head blocks: 2 choices each
- total size: `3^6 * 2^5 = 23328`

The first two positions mean:

- `[0] E0`: `0=7x7 stride-2`, `1=5x5 stride-2`, `2=3x3 stride-2`
- `[1] E1`: `0=5x5 stride-2`, `1=3x3 stride-2`, `2=3x3 stride-2 + 3x3 dilated`

Therefore:

- `2,1,...` means a light frontend:
  - `E0 = 3x3 stride-2`
  - `E1 = 3x3 stride-2`

### Existing NAS Retrain Path Is Still Old-Format

`wrappers/run_standalone_train.py` still documents and parses `9`-dim architecture codes, not V2 `11`-dim codes.

`efnas/engine/standalone_trainer.py` also defaults to `num_blocks=9` and builds `MultiScaleResNetSupernet` directly from fixed arch constants.

Implication:

- the old standalone retrain path is not V2-ready as-is

### Existing Fixed-Arch Joint Trainer Is A Better Structural Starting Point

`wrappers/fixed_arch_compare/run_train.py` and `efnas/engine/fixed_arch_compare_trainer.py` already support:

- multiple models trained together
- shared data stream
- cosine LR
- checkpointing
- comparison-oriented output structure

But it still currently assumes:

- old-style `backbone_arch_code` length `9`
- FC2-only data build path via `build_fc2_provider`
- no explicit supernet warm-start import mechanism

### Current NAS Retrain Configs Use Small Training Resolution

Existing configs:

- `configs/standalone_fc2_180x240.yaml`
- `configs/fixed_arch_compare_fc2_172x224.yaml`

Both are FC2-only and use search/deployment-oriented resolutions, not a larger final retrain resolution for stronger Sintel comparison.

### Original EdgeFlowNet Training Wrapper Already Supports FT3D As A Dataset Argument

`EdgeFlowNet/wrappers/run_train.py` exposes:

- `--dataset FC2/FT3D/MSCOCO`
- `--num_epochs`
- `--batch_size`
- `--lr`

`EdgeFlowNet/code/train.py` also documents:

- `Dataset: FC2 for Flying Chairs 2 or FT3D for Flying Things 3D`

Implication:

- there is no separate dedicated FT3D wrapper
- the original code switches dataset through arguments and data lists

### Literature/Implementation Target To Mirror

The desired high-level recipe is:

- train / continue on FC2 with `1e-4` scale LR
- fine-tune on FT3D with `1e-5`
- batch size target `32`

For this project, the intended adaptation is:

- do not train from scratch
- initialize from supernet-inherited subnet weights
- use a smaller continuation LR on FC2
- keep cosine schedule
- use early stopping instead of blindly finishing all epochs

### Training Resolution Decision Is Locked

The retrain pipeline will follow the original EdgeFlowNet crop recipe instead of inventing a new wide-screen training regime:

- FC2 stage: `352 x 480`
- FT3D stage: `480 x 640`

This keeps training aligned with the intended 4:3 deployment setting while still moving beyond the smaller supernet-search resolution.

### Fixed-Subnet Retraining Needs Explicit Name Normalization

The constant-arch supernet fallback was useful to prove the warm-start path, but it is not the right long-run training graph because it still materializes all candidate branches and only selects them late.

The real issue behind fixed-subnet warm-start was narrower:

- `BaseLayers` uses counting-based internal scopes
- pruning branches changes some raw TensorFlow variable names
- therefore fixed-subnet graph names do not raw-match supernet checkpoint names

The workable solution is now confirmed:

- build retrain/eval graphs with `FixedArchModelV2`
- normalize fixed-subnet variable names back to the supernet naming scheme
- restore from the supernet checkpoint through an explicit source-name map

This keeps the training/eval graph as a true subnet while preserving auditable supernet warm-start.

### FT3D Loader Should Scan Folders Directly

EdgeFlowNAS does not currently carry the old FT3D dirname/list files.

The new retrain path should therefore scan:

- `frames_cleanpass/{TRAIN,TEST}`
- `optical_flow/{TRAIN,TEST}`

and derive valid future-frame pairs by file existence.

### HPC FlyingThings3D Layout Is Now Confirmed

The actual HPC data layout for stage-2 retraining is now known.

Confirmed flow path example:

- `FlyingThings3D/optical_flow/TRAIN/A/0000/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`

Confirmed frame path example:

- `FlyingThings3D/frames_cleanpass/TRAIN/A/0000/left/0006.png`

Implications:

- stage-2 retraining only needs `frames_cleanpass` plus `optical_flow`
- `finalpass` is not required for the current retrain pipeline
- PNG frames are the correct asset type; WebP would not match the current code path

### First HPC FC2 Run Exposed A Real Warm-Start Saver Bug

The first live FC2 launch failed during checkpoint restore because the retrain graph tried to restore optimizer slot tensors from the supernet checkpoint.

Observed failure shape:

- `.../bn2/beta/Adam not found in checkpoint`

Root cause:

- `warmstart_saver` was built from all scope global variables
- this accidentally included Adam slot variables created by the retrain optimizer
- the source supernet checkpoint does not contain those slot tensors

Resolution:

- warm-start restore now filters out optimizer-slot variables such as `/Adam`
- save/resume for the retrain experiment still keeps full scope variables
- only the one-time import path was narrowed

This is the correct contract split:

- warm-start import: model weights + BN state only
- retrain checkpoint save/resume: full scope variables including optimizer state

### Current HPC Commands Are Structurally Ready

The pipeline now has concrete command-level contracts for:

- FC2 warm-start from supernet checkpoint
- FT3D continuation from FC2 best checkpoint
- standalone Sintel testing from latest retrain experiment directory

The remaining uncertainty is execution validation, not command definition.

### FT3D External Sintel Evaluation Needs Inverse Flow Scaling

The retrain pipeline currently uses different target scales between `FC2` and `FT3D`:

- `FC2`: raw flow is clipped directly
- `FT3D`: flow is divided by `12.5` before clipping and supervision

This `12.5` factor is not a new invention in the retrain code; it is inherited from the original `EdgeFlowNet` FT3D batching path.

Implications:

- internal `FT3D` stage metrics are still self-consistent because train/val share the same scaled label space
- external Sintel evaluation is *not* self-consistent unless predictions are multiplied back by `12.5` before EPE is measured against raw Sintel ground truth

This explains the earlier confusing pattern:

- stage metric improved
- external Sintel EPE appeared to jump to about `10.6`

That number should not be over-interpreted as real model collapse unless the prediction scale has first been restored.

Resolution now applied:

- `wrappers/run_retrain_v2_sintel_test.py` rescales predictions back to raw-flow units for `FT3D` checkpoints before computing Sintel EPE
- `FC2` checkpoints remain on identity scale

Therefore:

- old pre-fix `FT3D` Sintel CSVs are not reliable for scientific comparison
- post-fix Sintel CSVs are the ones that should be used to judge whether `FT3D` continuation is helping or hurting cross-dataset generalization

## Preliminary Design Judgement

The most plausible low-risk route is:

1. base the new work on `fixed_arch_compare_trainer.py`
2. upgrade it to V2 `11D`
3. add supernet-to-fixed-model warm-start loading
4. add multi-stage dataset schedule `FC2 -> FT3D`
5. add larger-resolution configs

This is likely lower risk than trying to rescue the older `standalone_trainer.py` path.

## Risks

1. FT3D `TEST` as validation is operationally simple but may not be the final protocol you want for reporting.
2. Early stopping must not silently fight cosine scheduling or stage transition logic.
3. Fixed-subnet warm-start now depends on normalized variable-name mapping; this is structurally cleaner than constant-arch masking, but final HPC validation should still confirm deployment export and training metrics stay aligned.
