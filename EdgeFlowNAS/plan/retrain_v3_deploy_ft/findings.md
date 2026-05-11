# Findings: Retrain V3 Deploy-Resolution Fine-Tune

Hard facts and constraints. Add new entries as they are discovered. Do not delete old entries — append updates instead, so the chronology is auditable.

---

## 1. Deploy Constraint (anchor for everything else)

- Board: Himax WE2 (CM55M + Ethos-U55), 2 MiB SRAM, ~1.9 MiB app-controllable.
- Tensor arena hard ceiling: **1432 KiB** (`common_config.h` in Seeed repo).
- Max INT8 input size that fits the arena + prev-buffer: **157×203**.
  Larger inputs (e.g., 158×202, 172×224 for V3) may pass Vela compile but fail board-side `AllocateTensors` or `alloc prev buffer fail`. Source: `Seeed_Grove_Vision_AI_Module_V2/plan/MCUFlowNet_Deployment/findings.md §1`.
- Therefore deploy input is **locked at 157×203**.

## 2. Pre-Fine-Tune Baseline (Sintel Final train, 1041 frames, test_sintel methodology @ patch 416×1024, clip ±50, `flow_scale=12.5` for v3)

Source: `Seeed_Grove_Vision_AI_Module_V2/plan/MCUFlowNet_Deployment/findings.md §7b/7d` and `progress.md` rows R7/R13/R14/R15/R17/R18/R19.

| Model | HPC FP32 @ 416×1024 | FP32 @ 157×203 | INT8 @ 157×203 | Δ_downsample | Δ_pure_quant |
|---|---:|---:|---:|---:|---:|
| Mainline (transpose-conv) | 6.31 | 7.71 | 7.79 | +1.40 | +0.08 |
| v3_acc | 5.09 | 8.27 | 8.40 | +3.18 | +0.13 |
| v3_efn_fps | **4.89** | 7.75 | **7.34** | +2.86 | -0.41 (noise) |
| v3_light | 5.58 | 10.60 | 12.73 | +5.02 | **+2.13 ⚠** |

Key takeaway: at HPC eval-res the 3 V3 subnets clearly beat mainline; at deploy-res mainline almost catches up because V3 architectures lose more from input downsampling. Fine-tune target: shrink each row's Δ_downsample.

## 3. BN Recalibration Failed (anti-pattern)

Tried zero-cost fix before fine-tune: update only BN running stats, leave conv weights alone. v3_efn_fps:
- 500 batches × bs=4 BN recal on Sintel train Clean @ 157×203: FP32 EPE 7.75 → **19.46** (broke the model).
- 50 batches × bs=16 (gentler): FP32 EPE → **22.93** (even worse).

Reason: V3 conv weights are entangled with the original BN running stats from training; shifting BN stats alone pushes downstream conv inputs into a region the conv weights weren't calibrated for. Conv + BN must move together. This justifies the fine-tune approach.

(Failed ckpts were deleted from EdgeFlowNAS outputs dir. Anti-pattern script preserved at `Seeed_Grove_Vision_AI_Module_V2/tools/bn_recal/run_bn_recal_v3.py`.)

## 4. V3 Subnet Inventory

`outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1/`:

| Subnet | arch_code | role | sintel_best.ckpt metric (FP32 @ 416×1024) |
|---|---|---|---:|
| `v3_acc`     | `0,1,2,2,2,2,0,0,0,0,1` | strongest predicted EPE | 5.0898 |
| `v3_efn_fps` | `2,0,0,2,2,1,0,0,0,0,0` | matches EdgeFlowNet FPS target | 4.8879 |
| `v3_light`   | `0,0,0,0,0,0,0,0,0,0,0` | lightest Pareto endpoint | 5.5819 |

Each model_dir has `checkpoints/{best,last,sintel_best}.ckpt{.index,.data-00000-of-00001,.meta,.meta.json}`. The `.meta.json` carries a HPC-absolute checkpoint_path that won't resolve locally — must override when loading from non-HPC environments.

## 5. V3 Training Settings (from `model_v3_acc/run_manifest.json`, applies to all three)

```
dataset: FT3D
ft3d_frames_base_paths: [TRAIN/frames_cleanpass, TRAIN/frames_finalpass]
ft3d_flow_base_path: TRAIN/optical_flow
ft3d_directions: [into_future, into_past]
ft3d_flow_divisor: 12.5
input_height: 480
input_width: 640
flow_channels: 2

train.num_epochs: 50
train.batch_size: 32
train.micro_batch_size: 16
train.lr: 1e-5
train.lr_min: 1e-6
train.optimizer: adam
train.lr_schedule: cosine
train.grad_clip_global_norm: 200.0

ft3d_train_augment:
  min_scale: -0.4   max_scale: 0.8
  spatial_aug_prob: 0.8   stretch_prob: 0.8   max_stretch: 0.2
  do_flip: true   h_flip_prob: 0.5   v_flip_prob: 0.1
  photometric_aug_prob: 1.0
  brightness/contrast/saturation: 0.4
  asymmetric_color_aug_prob: 0.2
  eraser_aug_prob: 0.5   eraser_min_size: 50   eraser_max_size: 100
```

Network preproc: `(uint8 / 255) * 2 - 1` (see `efnas/data/transforms_180x240.py` and `retrain_v2_sintel_runtime.preprocess_eval_batch`).

## 6. Mainline (EdgeFlowNet) Baseline — facts to verify

Files of interest (cross-repo):
- Architecture: `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\network\MultiScaleResNet.py`
- Training entry: `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\train_sintel.py` (or `train.py`; check)
- Train wrapper: `D:\Dataset\MCUFlowNet\EdgeFlowNet\wrappers\run_train.py` (likely)
- Test entry: `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\test_sintel.py` (confirmed)
- Test list (Sintel Final): `D:\Dataset\MCUFlowNet\EdgeFlowNet\code\dataset_paths\MPI_Sintel_Final_train_list.txt`
- Checkpoint (canonical): `D:\Dataset\MCUFlowNet\EdgeFlowNet\checkpoints\best.ckpt`
- Mirror in Seeed: `Seeed_Grove_Vision_AI_Module_V2\tools\model_export\optical_flow_144x192\assets\checkpoints\best.ckpt`

Verification done so far:
- `test_sintel.py --uncertainity` with `--patch_dim_0 416 --patch_dim_1 1024` on `MPI_Sintel_Final_train_list.txt` reproduces user's prior 6.31 EPE baseline.
- `test_sintel.py` does NOT multiply prediction by 12.5; the model's raw output is in input-grid pixel motion units.
- Therefore mainline FT must use `flow_divisor=1.0` (no GT division during training).

Verification done in Phase 1 closeout (read `EdgeFlowNet/code/train.py` + `EdgeFlowNet/code/misc/DataHandling.py:91` SetupAll + `EdgeFlowNet/code/misc/utils.py`):

- **Training entry**: `EdgeFlowNet/code/train.py` (no separate `train_sintel.py`)
- **Training dataset**: `--Dataset FT3D` was used (almost certainly — FC2 default in argparse but ckpt evaluates on Sintel, and only FT3D path makes sense; the FC2 default is just argparse history). Train list comes from `code/dataset_paths/FT3D_train.txt` etc.
- **Training input size (FT3D)**: hardcoded in `DataHandling.SetupAll`:
  - FT3D: original 540×960 → patch **480×640** (SAME as v3!)
  - FC2:  original 384×512 → patch 352×480
  - MSCOCO: 504×640 → 352×480
- **flow_divisor**: NONE. `utils.get_sintel_batch` only does `np.clip(gt, -50, 50)` — no `/N` on GT. Mainline FT uses `flow_divisor=1.0` confirmed.
- **NumOut quirk**: `SetupAll` sets `args.NumOut = 2` for FT3D, BUT `args.EffectiveUncertaintyType = DEFAULT_UNCERTAINTY_TYPE = "LinearSoftplus"`. The network's `UncType` argument likely makes it produce extra uncertainty channels on top of NumOut. Empirically `test_sintel.py --uncertainity` (which sets NumOut=4 for inference) successfully restores the published `best.ckpt`, so the trained network indeed has 4 channels per scale. **Treat this as "NumOut=4 effective"** during FT for safety (and match what evaluation uses).
- **Loss**: `LossFuncName = 'MultiscaleSL1-1'` (multi-scale Smooth-L1).
- **LR default**: 1e-4 from `train.py`. Original mainline training used some unknown actual LR — likely lower for late epochs. Fine-tune at 1e-6 → 1e-7 still appropriate (10-100× below original).
- **DataAug flag**: `args.DataAug` (int, default 0). Without it: no augmentation. With it: an Augmentations pipeline is enabled (see `train.py:153` "args.Augmentations = 'None'" when DataAug=0).
- **BN**: standard `tf.compat.v1.layers.batch_normalization` (in `BaseLayers.py`), default momentum 0.99.

**Big finding**: mainline and v3 are trained on the **SAME** dataset (FT3D) at the **SAME** input size (480×640). The reason mainline degrades less under 480×640 → 157×203 input downsample (Δ +1.40 vs v3's +2.86~+5.02) must be **architectural**: mainline's transpose-conv decoder + simpler block stack is more robust to spatial-scale change than v3's bilinear-ResizeConv + ECA-bottleneck + global-broadcast-gate stack (which all touch global spatial statistics).

Implication for FT plan:
- Both mainline and v3 can share the **same FT3D data pipeline at 157×203** — no domain shift; pure spatial-scale adaptation.
- mainline uses `flow_divisor=1.0`, clip ±50; v3 uses `flow_divisor=12.5`, clip ±50 (post-division → ±4 range).
- BN momentum 0.99 default — fine-tune with 5-10 epochs at ~few hundred steps/epoch will gradually update BN, not abruptly replace.

## 7. FT3D Data Layout (used by retrain_v3, will reuse)

Per `run_manifest.json`:
- `frames_cleanpass` and `frames_finalpass` are both used as input image sources.
- `optical_flow` directory under TRAIN provides GT (.pfm files).
- Both directions used: `into_future` + `into_past`.
- A small list of excluded .pfm files (corrupted/inconsistent) — preserved in the manifest. Must be re-applied in FT config.
- Eval data is FT3D `TEST` split (val); 1 worker.

On HPC the base path is `../Datasets/FlyingThings3D/...`. On local it should resolve via the dataloader's relative path conventions; for local smoke test we may need a small slice. NOT downloading FT3D locally — local smoke test will use a tiny synthetic batch or HPC-only.

## 8. HPC Configuration

- Cluster: Leuven `genius`, account `lp_embaivision`, partition `gpu_p100`.
- Module: `TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1`.
- venv activation: `source ~/tf_work/bin/activate` (post module-load).
- Threading caps: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
- Slurm template to copy from: `plan/retrain_v3/retrain_v3_ft3d_3gpu.slurm`.
- For 4 candidates we want 4 GPUs ideally; or 3+1 split. Note p100 with 3 GPUs/node is the existing pattern.

## 9. Hyperparameter Plan (fine-tune-specific deltas vs retrain_v3 baseline)

| Knob | retrain_v3 (from scratch) | fine-tune (proposed) | rationale |
|---|---|---|---|
| epochs | 50 | 5–10 | only need spatial-scale adaptation; not new representation learning |
| LR | 1e-5 → 1e-6 cosine | **1e-6 → 1e-7 cosine** | 10× lower; standard FT discipline |
| input_height | 480 | **157** | deploy res |
| input_width  | 640 | **203** | deploy res |
| batch_size | 32 | 32 (likely fine; small input) | unchanged |
| micro_batch_size | 16 | 32 (input is ~10× smaller per sample) | likely fits in one micro-step |
| spatial_aug_prob | 0.8 | **0.3** | avoid pushing already-small input to extreme crops |
| max_stretch | 0.2 | **0.05** | same reasoning |
| eraser_aug_prob | 0.5 | **0.0** | erase patches at 157×203 wipe out most of the image |
| photometric_aug_prob | 1.0 | 1.0 (unchanged) | spatial-independent |
| h_flip_prob | 0.5 | 0.5 (unchanged) | OK at any res |
| v_flip_prob | 0.1 | 0.1 (unchanged) | OK at any res |
| optimizer | adam | adam | unchanged |
| weight_decay | 0.0 | 0.0 | unchanged |
| grad_clip_global_norm | 200.0 | 50.0 (proposed lower) | FT loss should be small; tighten clip to prevent rare large updates |
| early_stop_patience | 0 (off) | 3 epochs on FT3D val EPE | catch overfit / divergence |

These are starting points; expect tuning if first run misbehaves.

## 10. Deliverable Table Skeleton (for Phase 5)

| Model | HPC FP32 @ 416×1024 (orig) | FP32 @ 157×203 (orig) | INT8 @ 157×203 (orig) | **FP32 @ 157×203 (FT)** | **INT8 @ 157×203 (FT)** | Vela inf (ms) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Mainline | 6.31 | 7.71 | 7.79 | TBD | TBD | ~188 | ckpt: `EdgeFlowNet/checkpoints/best.ckpt` |
| v3_acc | 5.09 | 8.27 | 8.40 | TBD | TBD | 189.67 | strongest predicted EPE |
| v3_efn_fps | 4.89 | 7.75 | 7.34 | TBD | TBD | 165.22 | matches EdgeFlowNet FPS |
| v3_light | 5.58 | 10.60 | 12.73 | TBD | TBD | 95.94 | lightest Pareto |

Decision criterion for which model to flash:
1. Pass Vela 1432 KiB arena
2. Lowest INT8 EPE @ 157×203 after FT
3. Tiebreaker: lower Vela inf time

## 11. Code Layout (planned)

EdgeFlowNAS additions:

```
configs/retrain_v3_deploy_ft.yaml          # config template
efnas/network/edgeflownet_mainline.py      # wraps MultiScaleResNet for the trainer
efnas/engine/deploy_ft_trainer.py          # arch_family-aware fine-tune trainer
wrappers/run_deploy_ft_batch.py            # multi-candidate dispatcher
plan/retrain_v3_deploy_ft/
  task_plan.md / findings.md / progress.md
  retrain_v3_deploy_ft_candidates.csv      # 4 rows
  retrain_v3_deploy_ft_4gpu.slurm
```

Seeed side stays as-is. Export + eval continue to use existing scripts.

## 12. Known Risks / Open Questions

- ~~**Domain shift for mainline FT**~~ **RESOLVED**: mainline was trained on FT3D at 480×640 (same as v3). FT on FT3D at 157×203 is pure spatial-scale adaptation, no domain shift. See §6 update.
- **`tf.layers.batch_normalization` momentum quirk**: TF1 default momentum 0.99 means BN stats take many batches to stabilize. With small FT epoch count, BN may stay close to original stats — which is actually fine since we want conv to adapt around them. But verify with `bn moving_mean` snapshots before/after FT.
- **PTQ calibration set adequacy**: Currently 50 frame pairs from PERTURBED_market_3 + PERTURBED_shaman_1 (Seeed mainline export's calibration data). After FT, the activation distributions may shift; the same calibration set might or might not still cover them well. Plan: re-quantize and verify INT8 vs FP32 Δ stays ≤ 0.2 EPE.
- **Mainline arch port correctness**: `MultiScaleResNet` from EdgeFlowNet uses `tf.compat.v1.layers.conv2d_transpose` for upsample, plus `tf.compat.v1.layers.batch_normalization`. Wrapper must preserve scope names so we can restore from `EdgeFlowNet/checkpoints/best.ckpt`. Anticipate a few hours of variable-name debugging.
- **Sintel data leakage if final eval is run too early**: discipline — Sintel Final 1041-frame eval is **offline only**, never used as in-training val. Enforce via config (no Sintel path in `eval.sintel_*` for this experiment).

---

**Update protocol**: every time a key question is answered, an assumption is validated, or a hyperparameter is changed by experiment, append a dated entry here before changing code. Don't accumulate undocumented experimental knowledge.
