# Execution Plan: Ablation V1

## Status

Core training-policy decisions approved by the user. Implementation can proceed from this plan.

## Fixed HPC Assumption

Use the current default single-A100 allocation:

```bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --gpus-per-node=1
```

The training code should expose worker-count CLI overrides, with the intended default:

- FC2: `--fc2_num_workers 16`
- FT3D: `--ft3d_num_workers 16`
- thread-library environment:
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`

These exports keep NumPy/OpenCV/BLAS thread pools from competing with the explicit data-loader workers.

## Scientific Question

Test whether the proposed backbone changes are justified under a controlled FC2 -> FT3D training protocol:

1. original transposed-conv EdgeFlowNet decoder
2. same skeleton with bilinear/resize-conv upsampling
3. bilinear plus bottleneck ECA
4. bilinear plus bottleneck ECA plus 1/4-resolution global gate

This also verifies the suspected mismatch between the intended second-supernet backbone and the actual `MultiScaleResNet_supernet_v2` implementation.

## Ablation Matrix

| ID | Variant Name | Upsampling | ECA | 1/4 Global Gate | Purpose |
|---|---|---|---|---|---|
| A0 | `edgeflownet_deconv` | transposed conv | no | no | Original EdgeFlowNet baseline |
| A1 | `edgeflownet_bilinear` | bilinear/resize-conv | no | no | Isolate upsampling change |
| A2 | `edgeflownet_bilinear_eca` | bilinear/resize-conv | bottleneck ECA | no | Isolate ECA gain |
| A3 | `edgeflownet_bilinear_eca_gate4x` | bilinear/resize-conv | bottleneck ECA | yes | Full proposed backbone |

All variants should share the same EdgeFlowNet-style encoder/decoder depth. The implementation should not reuse the V2 11D search code as the scientific baseline because V2 contains additional search-space semantics that are not the original EdgeFlowNet skeleton.

## Model Implementation Plan

Create one parameterized ablation model instead of four copied classes.

Proposed module:

- `efnas/network/ablation_edgeflownet_v1.py`

Core constructor settings:

- `variant_name`
- `upsample_mode`: `deconv` or `bilinear`
- `bottleneck_eca`: bool
- `gate_4x`: bool
- `init_neurons`: `32`
- `expansion_factor`: `2.0`
- `num_sub_blocks`: `2`
- `num_out`: `flow_channels * 2` for uncertainty loss compatibility

Backbone policy:

- A0 mirrors original `EdgeFlowNet/code/network/MultiScaleResNet.py` behavior:
  - `7x7` stride-2 stem
  - `5x5` stride-2 second encoder conv
  - two encoder residual/downsample stages
  - two decoder residual/upsample stages
  - three multi-scale flow heads
- A1-A3 change only the explicitly named ablation components.
- A2 applies ECA after the bottleneck feature, matching the existing `eca_bottleneck` location.
- A3 gates the 1/4-resolution decoder feature using bottleneck context, matching the existing `global_gate_4x` concept.

Required tests before training:

- graph builds for all four variants
- output list has three scales
- output channels equal `flow_channels * 2`
- A0 uses transposed-conv variables/ops
- A1-A3 use resize-conv variables/ops
- A2/A3 include `eca_bottleneck`
- A3 includes `global_gate_4x`

## Training Strategy

Prefer one trainer that can run either:

- joint mode: all four variants in the same graph and same data stream
- single-variant mode: one variant per job with fixed seed and deterministic sample order

Initial implementation should support both, but the first smoke test decides which mode is used for the full runs.

Decision rule:

- If joint mode fits A100 memory and `sec/step` is clearly better than four separate jobs, use joint mode.
- If joint mode is close to 4x single-model time or risks memory pressure, run four separate jobs with identical seed/sampling contracts.

Early stopping remains enabled, by user request. In joint mode, early stopping should stop the whole stage based on the shared mean validation EPE so all variants receive the same epoch budget. In single-variant mode, early stopping is per variant and the stopped epoch must be reported explicitly.

## FC2 Stage

Config target:

- dataset: `FC2`
- epochs: `400`
- input size: `352x480`
- batch size: `32`
- seed: `42`
- sampling: `shuffle_no_replacement`
- crop: random
- optimizer: Adam, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`
- base LR: `1.0e-4`
- min LR: `1.0e-6`
- weight decay: `0.0`
- grad clip: `200.0` by default, configurable from CLI
- eval cadence: every `5` epochs
- eval batches: `0`, meaning full validation provider
- early stop: enabled, patience `20`, min delta `1.0e-4`

Learning-rate schedule:

```text
total_steps = 400 * ceil(len(FC2_train) / batch_size)
lr(step) = lr_min + (base_lr - lr_min) * 0.5 * (1 + cos(pi * step / total_steps))
```

Resolution and crop policy:

- train resolution: `352x480`
- validation resolution: `352x480`
- train crop mode: random
- validation crop mode: center

Checkpoint policy per variant:

- save `last.ckpt` every epoch
- save `best.ckpt` when FC2 validation EPE improves
- save trainer state with `epoch` and `global_step`
- write per-variant history and a shared `comparison.csv`

FC2 data-loader update:

- add `fc2_num_workers`
- use per-sample deterministic seeds so random crop remains reproducible under parallel loading
- keep loader output identical in shape/range to current FC2 provider
- do not introduce TFRecord/cache/mixed precision in this plan

## FT3D Stage

Config target:

- dataset: `FT3D`
- epochs: `50`
- input size: `480x640`
- batch size: `32`
- seed: `42`
- sampling: `shuffle_no_replacement`
- crop/augmentation: keep current `retrain_v2_ft3d.yaml` augmentation settings
- optimizer: Adam, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`
- base LR: `1.0e-5`
- min LR: `1.0e-6`
- weight decay: `0.0`
- grad clip: `200.0` by default, configurable from CLI
- eval cadence: every `2` epochs
- eval batches: `0`, meaning full validation provider
- early stop: enabled, patience `10`, min delta `1.0e-4`

Learning-rate schedule:

```text
total_steps = 50 * ceil(len(FT3D_train_after_filtering) / batch_size)
lr(step) = lr_min + (base_lr - lr_min) * 0.5 * (1 + cos(pi * step / total_steps))
```

Resolution and crop policy:

- train resolution: `480x640`
- validation resolution: `480x640`
- train crop/augmentation: random crop plus current FT3D augmentation settings
- validation crop mode: center

Initialization:

- each FT3D variant initializes from its corresponding FC2 `best.ckpt`
- do not initialize A1-A3 from A0 or from the V2 supernet checkpoint
- resume must restore checkpoint epoch/global-step when using non-`last` checkpoints, preserving LR schedule semantics

FT3D abnormal-data handling:

- carry over the known `ft3d_excluded_flow_paths` list from `configs/retrain_v2_ft3d.yaml`
- support optional `ft3d_excluded_flow_list` file for future bad-path additions
- build sample lists after excluding those flow paths
- at runtime, skip samples whose raw PFM flow is not finite
- after `flow / ft3d_flow_divisor` and clipping, skip samples that are still not finite
- keep `ft3d_flow_divisor = 12.5`
- keep clipping to `[-50, 50]` after scaling

Known bad PFM paths to carry into the ablation config:

- `../Datasets/FlyingThings3D/optical_flow/TEST/A/0085/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0012/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0096/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0186/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0441/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0456/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0483/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/A/0728/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/B/0459/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0031/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0080/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0140/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0260/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`
- `../Datasets/FlyingThings3D/optical_flow/TRAIN/C/0323/into_future/left/OpticalFlowIntoFuture_0006_L.pfm`

Checkpoint policy per variant:

- save `last.ckpt` every epoch
- save `best.ckpt` when FT3D validation EPE improves
- save `sintel_best.ckpt` when Sintel EPE improves
- write checkpoint metadata with `epoch`, `global_step`, in-domain metric, and best metric

## Sintel Evaluation During Training

Use the same external Sintel path as retrain_v2:

- dataset root: `../Datasets/Sintel`
- list: `EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt`
- patch size: `416,1024`
- checkpoint name for best Sintel: `sintel_best`

Resolution:

- Sintel evaluation patch size: `416x1024`

Evaluation cadence:

- FC2 stage: proposed every `5` epochs if Sintel evaluation is enabled for FC2; otherwise run only final FC2 Sintel evaluation.
- FT3D stage: every `2` epochs, matching current FT3D eval cadence.

Recommended final setting:

- FC2 online Sintel enabled at the FC2 validation cadence.
- FT3D online Sintel enabled at the FT3D validation cadence.
- Rationale: prior runs showed Sintel validation EPE can decrease and then increase, so `sintel_best.ckpt` must be selected during training rather than only after the final epoch.

## Output Layout

Proposed roots:

- `outputs/ablation_v1_fc2/<experiment_name>/`
- `outputs/ablation_v1_ft3d/<experiment_name>/`

Per variant:

```text
model_<variant_name>/
  checkpoints/
    last.ckpt.*
    best.ckpt.*
    sintel_best.ckpt.*      # when Sintel eval is enabled
  history.csv
  manifest.json
```

Shared files:

- `comparison.csv`
- `trainer_state.json`
- `run_manifest.json`
- `train.log`
- copied config snapshot

## Logging Requirements

`train.log` must be useful enough to diagnose long HPC jobs without opening every CSV.

Log once at startup:

- git commit and dirty-worktree flag if available
- full output directory
- config path and copied config snapshot path
- selected variants and their feature flags
- dataset roots and split sample counts
- train/validation resolutions and crop modes
- batch size, worker counts, expected `steps_per_epoch`
- optimizer and LR schedule summary
- checkpoint/init/resume mode
- FT3D excluded-flow count and optional exclusion-list path
- gradient clipping threshold

Log every epoch:

- epoch number and total epochs
- `global_step` at epoch start/end
- LR at epoch start/end
- epoch wall time
- per-variant mean train loss
- per-variant gradient norm statistics: mean, p50, p90, p99
- per-variant `clip_rate`, computed as the fraction of train steps whose pre-clip global norm exceeded `grad_clip_global_norm`
- whether validation/Sintel evaluation ran

Log every validation event:

- validation sample count or batch count
- per-variant validation EPE
- whether each model improved `best.ckpt`
- Sintel EPE when enabled
- whether each model improved `sintel_best.ckpt`
- checkpoint paths written

Log resume/init details:

- exact checkpoint prefix restored for every variant
- checkpoint metadata: `epoch`, `global_step`, metric, best metric
- reconciled `start_epoch` and `global_step`
- if histories are trimmed, log the cutoff epoch and resulting row counts

Log abnormal-data events compactly:

- FT3D excluded-path count during sample-list build
- runtime skipped non-finite samples as aggregate counters per epoch, not one line per sample

## Progress Bar Requirements

Training and validation must both expose tqdm progress.

- Training tqdm: current epoch, total epochs, step progress, current LR, and recent/average loss.
- In-domain validation tqdm: variant name, validation batch progress, running EPE.
- Sintel tqdm: variant name, checkpoint name, sample progress, running Sintel EPE if available.
- Progress bars should be visible in Slurm stdout/stderr and should not replace persistent CSV/log outputs.

## Resume Semantics

Resume must be robust enough for long HPC runs and mid-evaluation interruptions.

Rules:

- `--resume --resume_experiment_name X --resume_ckpt_name last` restores each variant from `model_<variant>/checkpoints/last.ckpt`.
- `--resume_ckpt_name best` and `--resume_ckpt_name sintel_best` are allowed and must prefer checkpoint metadata over stale trainer state.
- resume restores model variables and optimizer slots when available.
- resume restores or reconciles `epoch`, `global_step`, best in-domain metric, best Sintel metric, and no-improve counters.
- LR must continue from restored `global_step`; it must not restart from epoch 1 unless intentionally starting a new experiment.
- histories and `comparison.csv` must be trimmed to the restored checkpoint epoch when resuming from non-`last` checkpoints.
- if any variant checkpoint is missing, fail fast before training starts.
- if config identity differs from the run manifest, fail unless an explicit `--allow_config_mismatch` flag is provided.
- `last.ckpt` is saved after every epoch; `trainer_state.json` is written atomically after checkpoints and histories are written.
- a mid-validation crash should resume from the last completed epoch, not from a half-written validation row.

`comparison.csv` columns should include:

- `epoch`
- per-variant train loss
- per-variant in-domain EPE
- per-variant Sintel EPE if available
- delta versus A0
- delta versus A1
- current LR

## Proposed Wrappers

Add dedicated wrappers rather than overloading retrain_v2:

- `wrappers/run_ablation_v1_fc2.py`
- `wrappers/run_ablation_v1_ft3d.py`

Important CLI args:

- `--config`
- `--experiment_name`
- `--gpu_device`
- `--variants`, default all four variants
- `--joint` / `--single_variant`
- `--fc2_num_workers`
- `--ft3d_num_workers`
- `--grad_clip_global_norm`, default `200.0`
- `--early_stop_patience`
- `--early_stop_min_delta`
- `--resume`
- `--resume_experiment_name`
- `--resume_ckpt_name`
- `--init_experiment_dir`
- `--init_ckpt_name`

## Proposed Configs

Add:

- `configs/ablation_v1_fc2.yaml`
- `configs/ablation_v1_ft3d.yaml`

Both configs should explicitly list the four variants and feature flags, rather than relying on hidden defaults.

## HPC Command Drafts

FC2:

```bash
python wrappers/run_ablation_v1_fc2.py \
  --config configs/ablation_v1_fc2.yaml \
  --experiment_name ablation_v1_fc2_run1 \
  --gpu_device 0 \
  --fc2_num_workers 16 \
  --grad_clip_global_norm 200.0
```

FT3D:

```bash
python wrappers/run_ablation_v1_ft3d.py \
  --config configs/ablation_v1_ft3d.yaml \
  --experiment_name ablation_v1_ft3d_run1 \
  --init_experiment_dir outputs/ablation_v1_fc2/ablation_v1_fc2_run1 \
  --init_ckpt_name best \
  --gpu_device 0 \
  --ft3d_num_workers 16 \
  --grad_clip_global_norm 200.0
```

Resume examples:

```bash
python wrappers/run_ablation_v1_fc2.py \
  --config configs/ablation_v1_fc2.yaml \
  --experiment_name ablation_v1_fc2_run1 \
  --resume \
  --resume_experiment_name ablation_v1_fc2_run1 \
  --resume_ckpt_name last \
  --gpu_device 0 \
  --fc2_num_workers 16 \
  --grad_clip_global_norm 200.0
```

```bash
python wrappers/run_ablation_v1_ft3d.py \
  --config configs/ablation_v1_ft3d.yaml \
  --experiment_name ablation_v1_ft3d_run1 \
  --resume \
  --resume_experiment_name ablation_v1_ft3d_run1 \
  --resume_ckpt_name last \
  --gpu_device 0 \
  --ft3d_num_workers 16 \
  --grad_clip_global_norm 200.0
```

## Smoke Tests Before Full Run

Before launching the full 400/50 epoch jobs:

1. graph-build test for all variants
2. one train step on synthetic tensors
3. FC2 20-step loader/trainer smoke test with `fc2_num_workers=16`
4. FT3D 20-step loader/trainer smoke test with known bad-PFM exclusion enabled
5. checkpoint save/restore test for `last`, `best`, and `sintel_best`
6. resume test that confirms `global_step` and LR continue from checkpoint metadata
7. short A100 benchmark:
   - single-variant 200 steps
   - joint four-variant 200 steps

## Vela Equivalence Validation

A0 should be implemented inside EdgeFlowNAS rather than imported directly from `EdgeFlowNet/code`.

Validation requirements:

- use the HPC Python environment `tf_work_hpc`
- use the existing Vela config at `tests/vela/vela.ini`
- export the original EdgeFlowNet A0 graph and the EdgeFlowNAS-native A0 graph to TFLite under the same input shape and export settings
- run both through Vela
- compare:
  - export success/failure
  - supported/fallback ops from Vela logs
  - SRAM estimate
  - inference time estimate
  - cycles/MACs from Vela summary when available
- accept the A0 reimplementation only if op structure and Vela metrics are effectively equivalent; otherwise keep the discrepancy documented and fix the implementation before full training

## Resolved User Decisions

1. Keep early stopping enabled.
2. Set `grad_clip_global_norm = 200.0` by default and expose it as a CLI override.
3. Record gradient norm statistics and clip rate in `train.log` and structured histories.
4. Use an EdgeFlowNAS-native reimplementation of original EdgeFlowNet for A0.
5. Validate A0 equivalence with Vela in `tf_work_hpc` using the existing `tests/vela/vela.ini`.
6. Enable online Sintel validation during training so `sintel_best.ckpt` tracks the best Sintel epoch.
