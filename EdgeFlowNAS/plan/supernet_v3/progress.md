# Progress Log: Supernet V3

## Session: 2026-04-27

### Planning Initialization

- **Status:** complete
- Created planning directory:
  - `plan/supernet_v3/`
- Created planning files:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Context Reviewed

- Read Supernet V2 planning files:
  - `plan/supernet_v2/task_plan.md`
  - `plan/supernet_v2/findings.md`
  - `plan/supernet_v2/progress.md`
- Read V2 search-space code:
  - `efnas/nas/search_space_v2.py`
- Read V2 supernet network:
  - `efnas/network/MultiScaleResNet_supernet_v2.py`
- Read V2 trainer:
  - `efnas/engine/supernet_trainer_v2.py`
- Read V2 formal config:
  - `configs/supernet_fc2_172x224_v2.yaml`
- Checked existing ECA/global-gate implementations:
  - `efnas/network/fixed_arch_models.py`
  - `efnas/network/ablation_edgeflownet_v1.py`
- Checked FC2 provider builder and worker support:
  - `efnas/data/dataloader_builder.py`
  - `efnas/data/fc2_dataset.py`

### Key Findings

- V2 already uses an 11D mixed-cardinality architecture code.
- V2 stem choice order is inconsistent with the desired light-to-heavy semantics.
- V2 supernet is bilinear / resize-conv aligned.
- V2 supernet does not currently include fixed bottleneck ECA or fixed 1/4 global gate.
- Existing ECA/global-gate implementation is available in fixed-arch and ablation model code.
- FC2 loading has per-batch threaded sample loading but does not yet have asynchronous bounded batch prefetch.
- V2 trainer has no explicit multi-GPU tower implementation.
- `172x224` remains the recommended formal V3 supernet training resolution.

### Current Recommendation

- Implement V3 as a parallel path instead of modifying V2 in place.
- Use `172x224` for formal V3 supernet training. User confirmed this resolution on 2026-04-27.
- Keep the search space size unchanged at `23328`.
- Correct all architecture-code dimensions so `0` means light and larger values mean heavier.
- Fix bilinear + bottleneck ECA + 1/4 global gate into the backbone.
- Add bounded prefetching with conservative defaults.
- Re-evaluate multi-GPU design around FairNAS arch-parallel training:
  - one strict-fairness cycle samples 3 subnets
  - 3 GPUs can naturally run those 3 subnets in parallel
  - this may be more effective than generic data parallel for Supernet V3
  - BN update semantics must be explicitly controlled before production use

### Next Step

Implement V3 after user approved `172x224` and arch-parallel as the fastest multi-GPU direction.

### Implementation Progress

- Added V3 search-space, codec, sampler, and eval-pool modules.
- Added V3 supernet network with:
  - corrected light-to-heavy stem semantics
  - bilinear / resize-conv decoder
  - fixed bottleneck ECA
  - fixed 1/4 global gate
- Added bounded prefetch wrapper.
- Added V3 wrapper/app/trainer/config path.
- Added `arch_parallel` graph mode for 3-GPU FairNAS-cycle training:
  - three sampled subnets are placed on three GPU towers
  - one logical optimizer step is applied after accumulated tower gradients
  - single-GPU mode remains available as baseline
- Added V3 Vela reference export wrapper using hard-routed fixed subnet export.

### Verification

- Local tests:
  - `python -m pytest tests/test_supernet_v3_space_helpers.py tests/test_run_supernet_train_wrapper_v3.py tests/test_prefetch_provider.py tests/test_supernet_network_structure_v3.py tests/test_supernet_v2_space_helpers.py tests/test_run_supernet_train_wrapper_v2.py -q`
  - result: `15 passed, 2 skipped`
- Syntax compile:
  - `python -m py_compile efnas/nas/search_space_v3.py efnas/nas/arch_codec_v3.py efnas/nas/fair_sampler_v3.py efnas/nas/eval_pool_builder_v3.py efnas/data/prefetch_provider.py efnas/network/MultiScaleResNet_supernet_v3.py efnas/engine/supernet_trainer_v3.py efnas/app/train_supernet_app_v3.py wrappers/run_supernet_train_v3.py`
  - result: success
- TensorFlow graph tests:
  - `D:\Anaconda3\envs\tf_work_hpc\python.exe -m unittest tests.test_supernet_network_structure_v3`
  - result: `OK`
- Dry run:
  - `python wrappers/run_supernet_train_v3.py --config configs/supernet_v3_fc2_172x224.yaml --dry_run --multi_gpu_mode arch_parallel --gpu_devices 0,1,2 --fc2_num_workers 12 --fc2_eval_num_workers 2 --prefetch_batches 2`
  - result: merged config is valid
- Vela reference precheck:
  - first attempt exported the full supernet graph and was invalid for subnet comparison because unselected branches were retained
  - fixed by changing Vela export to a hard-routed fixed subnet graph
  - final command:
    `D:\Anaconda3\envs\tf_work_hpc\python.exe wrappers/run_supernet_v3_vela_reference.py --input_height 172 --input_width 224 --rep_dataset_samples 1 --mode basic --output_dir outputs/supernet_v3_vela_reference_172x224_fixed`
  - result:
    - `sram_mb = 1.353515625`
    - `inference_ms = 147.23362500000002`
  - SRAM equals about `1386 KiB`, matching the Grove model-design SRAM target.
