# Retrain V3 Candidate Selection Plan

## Goal

Select two to three subnets from the distill Supernet V3 NSGA2 result for full retraining and final validation.

Primary source:

- `outputs/nsga2_v3/nsga2_v3_distill_run1_20260429_201744/metadata/pareto_front.csv`

The selection should include:

1. the lightest subnet `0,0,0,0,0,0,0,0,0,0,0`
2. one Pareto subnet whose FPS is as close as possible to original EdgeFlowNet at the same input resolution
3. optionally one additional Pareto subnet in the same EdgeFlowNet-FPS neighborhood but with better predicted EPE

## Current Input Resolution

Use the Supernet V3 deployment-aligned input resolution:

- height: `172`
- width: `224`

All FPS comparisons must use this same input shape and the same Vela configuration/toolchain.

## Selection Method

### Step 1: Fix the EdgeFlowNet Baseline FPS

Measure the original EdgeFlowNet deconv baseline with Vela using:

- same input shape: `172x224`
- same quantization/export path if possible
- same `vela.ini`
- same FPS formula used by NSGA2 V3 metadata

Record:

- `edgeflownet_deconv_fps`
- `edgeflownet_deconv_cycles_npu`
- `edgeflownet_deconv_sram_kb`
- exact exported model path
- exact Vela command/output

Do not use P100/A100 training throughput as this baseline. The matching criterion is deployment FPS from Vela, not GPU training speed.

### Step 2: Always Include the Lightest Pareto Subnet

Candidate:

```text
0,0,0,0,0,0,0,0,0,0,0
```

Reason:

- It is the highest-FPS/lightest Pareto endpoint in the current distill NSGA2 front.
- It gives the lower-bound accuracy reference for the V3 search space.

### Step 3: Select the EdgeFlowNet-FPS Pareto Match

Filter `pareto_front.csv` by:

```text
relative_fps_gap = abs(candidate_fps - edgeflownet_deconv_fps) / edgeflownet_deconv_fps
```

Recommended rule:

1. first try `relative_fps_gap <= 0.02`
2. if no candidate exists, relax to `<= 0.05`
3. if still no candidate exists, choose the Pareto point with minimum absolute FPS gap

Within the accepted FPS window, pick:

1. the candidate with lowest predicted EPE
2. tie-break by smaller FPS gap
3. tie-break by lower cycles

### Step 4: Optional Third Candidate

Add a third candidate only if it provides a useful tradeoff not covered by the first two:

- same FPS neighborhood but clearly lower predicted EPE, or
- a knee point where a modest FPS loss buys a large EPE gain

Avoid retraining redundant neighbors whose FPS and predicted EPE are nearly identical.

## Measured EdgeFlowNet Baseline

Measured on 2026-05-04 with `plan/retrain_v3/run_edgeflownet_deconv_vela_172x224.py`:

| Baseline | Input | Vela optimise | SRAM KB | Time ms | FPS |
|---|---:|---|---:|---:|---:|
| original EdgeFlowNet deconv | `172x224` | `Size` | 1694.0 | 188.0440 | 5.3179 |

The script imports the original `EdgeFlowNet/code/network/MultiScaleResNet.py`, restores `EdgeFlowNet/checkpoints/best.ckpt`, exports int8 TFLite, and runs Vela with `EdgeFlowNet/vela/vela.ini`.

## Updated Candidates From Current Metadata

These candidates use the measured original EdgeFlowNet deconv FPS target `5.3179`.

| Role | Arch code | Predicted EPE | FPS | Notes |
|---|---|---:|---:|---|
| Lightest endpoint | `0,0,0,0,0,0,0,0,0,0,0` | 4.6832 | 9.1415 | Must include |
| EdgeFlowNet-FPS closest match | `2,0,0,2,2,1,0,0,0,0,0` | 4.0971 | 5.3123 | FPS gap is about 0.10% vs original EdgeFlowNet |
| EdgeFlowNet-FPS accuracy-preferred | `0,0,0,2,2,2,0,0,0,0,1` | 4.0584 | 5.0987 | Within 5% FPS window, best predicted EPE |

If retraining budget allows only two subnets, use the lightest endpoint plus the closest EdgeFlowNet-FPS match. If budget allows three, include the accuracy-preferred same-neighborhood point.

## Final Candidate Set

Retrain exactly these three V3 subnets:

| Model name | Arch code | Role |
|---|---|---|
| `v3_acc` | `0,1,2,2,2,2,0,0,0,0,1` | strongest predicted-accuracy Pareto point |
| `v3_efn_fps` | `2,0,0,2,2,1,0,0,0,0,0` | closest FPS match to original EdgeFlowNet deconv |
| `v3_light` | `0,0,0,0,0,0,0,0,0,0,0` | lightest Pareto endpoint |

## Confirmed Retraining Protocol

These settings were confirmed by the user on 2026-05-04:

- initialization: train fixed V3 subnets from scratch
- FC2 schedule: `400 epochs`, Adam, cosine LR `1e-4 -> 1e-6`
- FT3D schedule: `50 epochs`, Adam, cosine LR `1e-5 -> 1e-6`
- early stopping: disabled
- effective batch size: `32`
- gradient clipping: keep configurable, default `200.0`
- FC2 stage validation:
  - FC2 val every epoch
  - Sintel online val every 5 epochs
- FT3D stage validation:
  - FT3D val every epoch
  - Sintel online val every 2 epochs
- FC2 training resolution: `352x480`
- FT3D training resolution: `480x640`
- Sintel online validation resolution: `416x1024`
- FT3D fine-tuning: initialize from FC2 `best` checkpoint
- FT3D fine-tuning: use cleaned FT3D flow list, excluding known non-finite PFM samples
- checkpointing: save `best`, `last`, and `sintel_best`
- resume: restore epoch, optimizer, LR scheduler, and best metrics from trainer state

## Implementation Plan

### Code Modules

Create a V3-specific retrain pipeline instead of reusing `retrain_v2_trainer.py` directly:

- `efnas/engine/retrain_v3_trainer.py`
- `wrappers/run_retrain_v3_fc2.py`
- `wrappers/run_retrain_v3_ft3d.py`
- `wrappers/run_retrain_v3_batch.py`
- `configs/retrain_v3_fc2.yaml`
- `configs/retrain_v3_ft3d.yaml`

Reuse existing components where appropriate:

- `FixedArchModelV3`
- FC2 and FT3D data providers
- `PrefetchBatchProvider`
- V3 Sintel runtime
- standalone checkpoint helpers
- atomic JSON state writing pattern from `distill_or_not_trainer.py`

### GPU/CPU Scheduling

Preferred launcher strategy:

- run three independent child processes
- bind one model to one GPU
- keep each model isolated for checkpointing and fault recovery

For a 3 GPU / 18 CPU allocation:

- `max_workers=3`
- `gpu_devices=0,1,2`
- FC2: `fc2_num_workers=4`, `fc2_eval_num_workers=1`
- FT3D: `ft3d_num_workers=4`, `ft3d_eval_num_workers=1`
- `prefetch_batches=2`
- `eval_prefetch_batches=0`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`

For the available 3 GPU / 27 CPU / 135 GB allocation:

- `max_workers=3`
- `gpu_devices=0,1,2`
- FC2: `fc2_num_workers=6`, `fc2_eval_num_workers=2`
- FT3D: `ft3d_num_workers=6`, `ft3d_eval_num_workers=2`
- `prefetch_batches=2`
- `eval_prefetch_batches=0`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`

For a 4 GPU / 36 CPU / 180 GB allocation:

- still train only three models concurrently unless async Sintel validation is implemented
- recommended first-pass worker settings:
  - FC2: `fc2_num_workers=6`, `fc2_eval_num_workers=2`
  - FT3D: `ft3d_num_workers=8`, `ft3d_eval_num_workers=2`
  - `prefetch_batches=2`
- reserve the fourth GPU only after measuring Sintel validation as a real bottleneck, because making validation asynchronous needs extra checkpoint coordination and delayed `sintel_best` bookkeeping

## Sintel Validation Equivalence Note

Current EdgeFlowNAS V3 Sintel validation is close to, but not exactly the same as:

```bash
python wrappers/run_test.py --checkpoint checkpoints/best.ckpt --dataset sintel --gpu_device 0 --uncertainity
```

Shared behavior:

- uses EdgeFlowNet Sintel list parsing
- uses EdgeFlowNet `get_sintel_batch`
- uses EdgeFlowNet `FlowPostProcessor("full", is_multiscale=True)`
- uses patch size `416x1024`
- evaluates mean EPE through the same processor logic

Known difference:

- EdgeFlowNet `--uncertainity` passes `Args.uncertainity=True` and then the processor slices prediction channels `[..., 0:2]`.
- Current V3 runtime already slices `preds[:, :, :, :2]` before `processor.update`, while `_build_processor_args()` sets `uncertainity=False`.
- For EPE this should be numerically equivalent if the prediction fed to the processor is already only flow channels, but it is not a strict command-level equivalence.

Implementation requirement:

- Add an explicit `sintel_uncertainty: true` option or set `_build_processor_args().uncertainity=True` for V3 retrain evaluation, then verify one checkpoint against the EdgeFlowNet-style processor path.

## Remaining Decisions

No scientific settings are currently blocked. Before coding, only the implementation choice below remains:

1. Whether to implement async Sintel validation on the fourth GPU now, or first build the simpler three-training-process launcher and only add async validation if logs show Sintel dominates wall time.
