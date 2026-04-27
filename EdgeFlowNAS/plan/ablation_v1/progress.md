# Progress: Ablation V1

## 2026-04-27

### Supernet V2 Backbone Recheck

- Rechecked `plan/supernet_v2` after the user's note that V2 was intended to use `globalgate4x_bneckeca`.
- Confirmed the plan files do not mention `globalgate4x_bneckeca`, `bneckeca`, or ECA/global gate as the V2 backbone.
- Confirmed the trained run manifest reports `network = MultiScaleResNet_supernet_v2` and `arch_semantics_version = supernet_v2_mixed_11d_front6_3choice_head5_2choice`.
- Confirmed `supernet_trainer_v2.py` instantiates `MultiScaleResNetSupernetV2`, whose decoder uses `resize_conv` but does not contain ECA/global-gate modules.
- Updated the ablation plan to treat this as a possible intended-spec versus actual-implementation mismatch.

### Detailed Execution Draft

- Created `plan/ablation_v1/execution_plan.md`.
- Fixed the proposed resource assumption to the current single-A100 setting: 16 CPU cores and 112G memory.
- Recorded the intended FC2 settings: 400 epochs, batch size 32, Adam, cosine-with-min LR from `1e-4` to `1e-6`, full validation every 5 epochs.
- Recorded the intended FT3D settings: 50 epochs, batch size 32, Adam, cosine-with-min LR from `1e-5` to `1e-6`, full validation every 2 epochs.
- Carried over FT3D abnormal-data handling: known bad PFM exclusion list, optional exclusion file, and runtime finite-flow guards before and after scaling/clipping.
- Initially marked open approval decisions around early stopping, grad clipping, A0 implementation, and Sintel evaluation.

### User Requested Plan Refinements

- Updated FC2 LR to `1e-4 -> 1e-6`.
- Recorded explicit resolutions:
  - FC2 train/validation: `352x480`
  - FT3D train/validation: `480x640`
  - Sintel evaluation patch: `416x1024`
- Added requirements for useful `train.log` content covering startup, epoch summaries, validation results, checkpoint writes, resume details, and FT3D skipped-sample counters.
- Added requirement for tqdm progress bars during in-domain validation and Sintel evaluation.
- Strengthened resume requirements so `last`, `best`, and `sintel_best` checkpoint resumes preserve `epoch/global_step`, LR schedule, metric state, and history consistency.

### User Decisions Resolved

- Keep early stopping enabled.
- Set `grad_clip_global_norm=200.0` by default and expose `--grad_clip_global_norm` as a CLI override.
- Record per-variant gradient norm statistics and `clip_rate` in `train.log` and structured histories.
- Use an EdgeFlowNAS-native implementation of original EdgeFlowNet for A0.
- Validate A0 equivalence using Vela in `tf_work_hpc`; the existing config path is `tests/vela/vela.ini`.
- Enable online Sintel validation during training for `sintel_best.ckpt` selection.

### Implementation Completed

- Added EdgeFlowNAS-native ablation model implementation:
  - `efnas/network/ablation_edgeflownet_v1.py`
  - four variants: deconv, bilinear, bilinear+ECA, bilinear+ECA+gate4x
- Added ablation trainer and Sintel runtime:
  - `efnas/engine/ablation_v1_trainer.py`
  - `efnas/engine/ablation_v1_sintel_runtime.py`
- Added wrappers:
  - `wrappers/run_ablation_v1_fc2.py`
  - `wrappers/run_ablation_v1_ft3d.py`
  - `wrappers/run_ablation_v1_vela_equivalence.py`
- Added configs:
  - `configs/ablation_v1_fc2.yaml`
  - `configs/ablation_v1_ft3d.yaml`
- Added FC2 multi-worker loading support with `fc2_num_workers` and `fc2_eval_num_workers`.
- Added FT3D non-finite skip counters for compact logging.
- Added tests:
  - `tests/test_ablation_v1_model.py`
  - `tests/test_ablation_v1_config_and_wrappers.py`
  - `tests/test_fc2_num_workers.py`

### Verification

- Local non-TensorFlow tests:
  - `python -m pytest tests/test_ablation_v1_model.py tests/test_ablation_v1_config_and_wrappers.py tests/test_fc2_num_workers.py -q`
  - initial result: `3 passed, 3 skipped`
- TensorFlow graph/unit tests in `tf_work_hpc`:
  - `D:\Anaconda3\envs\tf_work_hpc\python.exe -m unittest tests.test_ablation_v1_model tests.test_ablation_v1_config_and_wrappers`
  - initial result: `OK`
- Low-memory ablation update:
  - added `train.micro_batch_size` to FC2/FT3D configs; default remains `32`, so full training behavior is unchanged unless overridden.
  - added `--micro_batch_size` CLI override for FC2/FT3D ablation wrappers.
  - added `--variants` CLI override so 8GB smoke jobs can run one model at a time instead of keeping all four model graphs resident.
  - trainer now accumulates weighted micro-batch gradients and applies Adam, LR schedule, `global_step`, checkpointing, and gradient clipping once per logical batch.
  - `train.log` records logical batch, micro-batch, and the BatchNorm caveat when accumulation is enabled.
- Low-memory verification:
  - `python -m pytest tests/test_ablation_v1_model.py tests/test_ablation_v1_config_and_wrappers.py tests/test_fc2_num_workers.py -q`
  - result: `3 passed, 5 skipped`
  - `D:\Anaconda3\envs\tf_work_hpc\python.exe -m unittest tests.test_ablation_v1_model tests.test_ablation_v1_config_and_wrappers`
  - result: `OK`
- Vela equivalence smoke at `64x64`:
  - original and EdgeFlowNAS A0 both produced SRAM `0.1171875 MB`
  - original inference `89.98667 ms`; EdgeFlowNAS A0 `89.98699 ms`
- Vela equivalence at FC2 train resolution `352x480`:
  - original and EdgeFlowNAS A0 both produced SRAM `4.833984375 MB`
  - both produced inference `1719.8559899999998 ms`

### Planning Workspace Created

- Created:
  - `plan/ablation_v1/task_plan.md`
  - `plan/ablation_v1/findings.md`
  - `plan/ablation_v1/progress.md`

### Context Audited

- Checked V2 supernet/fixed-subnet files:
  - `efnas/network/MultiScaleResNet_supernet_v2.py`
  - `efnas/network/fixed_arch_models_v2.py`
- Checked original EdgeFlowNet files:
  - `../EdgeFlowNet/code/network/MultiScaleResNet.py`
  - `../EdgeFlowNet/code/network/BaseLayers.py`
- Checked existing ablation-like variant code:
  - `efnas/network/fixed_arch_models.py`
  - `efnas/engine/fixed_arch_compare_trainer.py`
- Checked current FC2 provider:
  - `efnas/data/fc2_dataset.py`

### Key Findings Logged

- Original EdgeFlowNet uses transposed convolution upsampling.
- V2 supernet/retrain subnet is bilinear/resize-conv aligned.
- V2 supernet/retrain subnet does not currently contain ECA or global gate logic.
- Existing ECA and 1/4 global gate code exists in the older 9D `FixedArchModel` variant system.
- FC2 data loading is still single-threaded and should be upgraded before 400-epoch ablation training.

### Next Actions

1. Verify the exact EdgeFlowNet-equivalent fixed arch code.
2. Decide direct original-model wrapper versus behavior-matched EdgeFlowNAS ablation implementation.
3. Draft implementation tasks for:
   - ablation model module
   - FC2 multi-worker provider
   - staged FC2/FT3D trainer
   - Sintel-best checkpointing
