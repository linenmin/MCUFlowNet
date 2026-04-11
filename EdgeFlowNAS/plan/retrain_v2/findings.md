# Findings: Retrain V2

## Confirmed Facts

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

### Warm-Start Must Use Constant-Arch Supernet Graphs

The first implementation attempt with a branch-pruned fixed-only V2 model exposed a real naming problem:

- `BaseLayers` uses counting-based internal scopes
- pruning branches changes scope-number order
- therefore fixed-only graph variable names do not stably match supernet checkpoint names

The stable solution is:

- build retrain/eval graphs with `MultiScaleResNetSupernetV2`
- feed a constant 11D arch code per model
- restrict the graph behavior through the fixed code, not through branch deletion

This preserves checkpoint-compatible variable names and makes supernet warm-start auditable.

### FT3D Loader Should Scan Folders Directly

EdgeFlowNAS does not currently carry the old FT3D dirname/list files.

The new retrain path should therefore scan:

- `frames_cleanpass/{TRAIN,TEST}`
- `optical_flow/{TRAIN,TEST}`

and derive valid future-frame pairs by file existence.

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
3. The new retrain graph now depends on constant-arch supernet masking; this is correct for warm-start, but first HPC run should still confirm only selected paths receive effective gradient updates.
