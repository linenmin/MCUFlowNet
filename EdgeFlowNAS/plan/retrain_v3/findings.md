# Retrain V3 Findings

## Source Files Checked

- `outputs/nsga2_v3/nsga2_v3_distill_run1_20260429_201744/metadata/pareto_front.csv`
- `outputs/nsga2_v3/nsga2_v3_distill_run1_20260429_201744/metadata/history_archive.csv`
- `outputs/nsga2_v3/nsga2_v3_distill_run1_20260429_201744/metadata/epoch_metrics.csv`

## NSGA2 Distill Run Summary

The distill NSGA2 run has:

- `800` evaluated rows in `history_archive.csv`
- `51` Pareto rows in `pareto_front.csv`
- Pareto FPS range: `4.1822` to `9.1415`
- Pareto predicted EPE range: `4.0256` to `4.6832`

The all-zero architecture is present in both the archive and the Pareto front:

| Arch code | Predicted EPE | FPS | SRAM KB | NPU cycles | MACs |
|---|---:|---:|---:|---:|---:|
| `0,0,0,0,0,0,0,0,0,0,0` | 4.6832 | 9.1415 | 1386.0 | 43622513 | 1651223296 |

This confirms it is a valid Pareto endpoint and should be included in Retrain V3.

## Original EdgeFlowNet Deconv Vela Baseline

Measured on 2026-05-04 with:

- script: `plan/retrain_v3/run_edgeflownet_deconv_vela_172x224.py`
- network: `EdgeFlowNet/code/network/MultiScaleResNet.py`
- checkpoint: `EdgeFlowNet/checkpoints/best.ckpt`
- input: `[1, 172, 224, 6]`
- output: `AccumPreds(outputs)[0][...,0:2]`
- Vela config: `EdgeFlowNet/vela/vela.ini`
- accelerator: `ethos-u55-64`
- optimise: `Size`

Result:

| SRAM KB | SRAM MB | Time ms | FPS |
|---:|---:|---:|---:|
| 1694.0 | 1.6543 | 188.0440 | 5.3179 |

Artifacts:

- `outputs/vela_edgeflownet_deconv_172x224/edgeflownet_deconv_original_172x224_int8.tflite`
- `outputs/vela_edgeflownet_deconv_172x224/edgeflownet_deconv_172x224_vela_metrics.json`
- `outputs/vela_edgeflownet_deconv_172x224/vela_size/edgeflownet_deconv_original_172x224_int8_summary_Grove_Sys_Config.csv`

## Pareto Points Near the Measured 5.3179 FPS Target

Nearest Pareto candidates:

| Arch code | Predicted EPE | FPS | FPS gap vs 5.3179 | NPU cycles |
|---|---:|---:|---:|---:|
| `2,0,0,2,2,1,0,0,0,0,0` | 4.0971 | 5.3123 | 0.0056 | 75162074 |
| `0,0,0,2,2,1,0,0,0,0,1` | 4.1074 | 5.3414 | 0.0235 | 74753133 |
| `0,0,0,2,2,1,0,0,0,0,0` | 4.1077 | 5.4338 | 0.1159 | 73478765 |
| `0,0,0,2,2,2,0,0,0,0,0` | 4.0586 | 5.1829 | 0.1350 | 77042457 |
| `0,0,0,2,2,2,0,0,0,0,1` | 4.0584 | 5.0987 | 0.2192 | 78316825 |

Interpretation:

- `2,0,0,2,2,1,0,0,0,0,0` is the closest FPS match to original EdgeFlowNet deconv.
- `0,0,0,2,2,2,0,0,0,0,1` has the best predicted EPE inside a 5% FPS window around the original EdgeFlowNet deconv baseline.

## Important Caveat

The EdgeFlowNet original baseline requested by the user is the deconv/original model, not necessarily the V3 bilinear/gate skeleton. Therefore the baseline FPS must be measured from the original EdgeFlowNet-equivalent export path before final candidate locking.

## Recommended Initial Shortlist

User finalized the shortlist on 2026-05-04:

| Candidate ID | Arch code | Rationale |
|---|---|---|
| `v3_acc` | `0,1,2,2,2,2,0,0,0,0,1` | strongest predicted-accuracy Pareto point |
| `v3_efn_fps` | `2,0,0,2,2,1,0,0,0,0,0` | closest FPS match to original EdgeFlowNet deconv |
| `v3_light` | `0,0,0,0,0,0,0,0,0,0,0` | lightest Pareto endpoint |

## Sintel Validation Equivalence Finding

Checked:

- `EdgeFlowNet/wrappers/run_test.py`
- `EdgeFlowNet/code/test_sintel.py`
- `EdgeFlowNet/code/misc/processor.py`
- `EdgeFlowNAS/efnas/engine/distill_or_not_sintel_runtime.py`
- `EdgeFlowNAS/efnas/engine/retrain_v2_sintel_runtime.py`

Current EdgeFlowNAS V3 Sintel validation reuses the important EdgeFlowNet components:

- `read_sintel_list`
- `get_sintel_batch`
- `FlowPostProcessor("full", is_multiscale=True)`
- patch size `416x1024`

The main difference is uncertainty handling:

- EdgeFlowNet command with `--uncertainity` sets `Args.NumOut=4`, then `FlowPostProcessor` slices prediction `[..., 0:2]`.
- Current V3 runtime builds `FixedArchModelV3(num_out=4)`, accumulates multiscale predictions, slices `preds[:, :, :, :2]`, and passes an args object with `uncertainity=False`.

For mean EPE, this should be equivalent in effect because only the first two flow channels reach the processor. It is not strictly equivalent at the command/API level, so the V3 retrain implementation should make uncertainty handling explicit and preferably set `uncertainity=True` in the processor args for auditability.
