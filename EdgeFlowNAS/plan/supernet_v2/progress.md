# Progress Log

## Session: 2026-04-06

### Phase 1: V2 Scope Lock and Assessment

- **Status:** in_progress
- Actions taken:
  - Read `wrappers/run_supernet_train.py`
  - Read `efnas/network/MultiScaleResNet_supernet.py`
  - Read `efnas/engine/supernet_trainer.py` around graph build
  - Read `efnas/nas/arch_codec.py`
  - Read `efnas/nas/fair_sampler.py`
  - Read `efnas/nas/eval_pool_builder.py`
  - Read `efnas/app/train_supernet_app.py`
  - Read `configs/supernet_fc2_180x240.yaml`
  - Read `tests/test_supernet_network_structure.py`
  - Read selected v1 planning docs:
    - `plan/supernet_v1/02_Supernet_Training_Spec.md`
    - `plan/supernet_v1/05_Engineering_File_Management_and_Code_Style.md`
  - Confirmed the V2 space from the user and supervisor:
    - front 6 blocks use 3 choices each
    - last 5 head blocks use 2 choices each
    - total space size is `23328`
  - Confirmed the main implementation issue is the fairness cycle, not only the network graph
  - Confirmed `tf_work_hpc` is not present in current WSL conda environments
  - Confirmed the wrapper/app split is already clean, so adding `run_supernet_train_v2.py` and `train_supernet_app_v2.py` fits the current style
  - Confirmed the current V1 trainer, codec, sampler, eval-pool builder, and tests all assume a 9-block all-3-choice space
  - Confirmed V2 training can be implemented without touching search-side tools in the first pass if separate V2 modules are used
  - Ran `git fetch origin` and confirmed `HEAD...origin/master = 0 0`
  - Checked the FairNAS paper and official repository
  - Confirmed FairNAS strict fairness is described for uniform-cardinality choice blocks
  - Confirmed Appendix A.1 gives an irregular-space extension based on temporary resampling up to the max choice count
  - Locked the V2 fairness mode to FairNAS A.1 approximated-SF with 3-path cycles
  - Implemented a parallel V2 file set for wrapper/app/network/codec/sampler/eval-pool/trainer/tests
  - Completed syntax compilation for all new V2 files
  - Completed `python wrappers/run_supernet_train_v2.py --dry_run`
  - Completed unit tests:
    - `tests.test_run_supernet_train_wrapper_v2`
    - `tests.test_supernet_v2_space_helpers`
  - Completed helper smoke checks:
    - `python -m efnas.nas.arch_codec_v2 --self_test`
    - `python -m efnas.nas.eval_pool_builder_v2 --size 12 --check`
    - `python -m efnas.nas.fair_sampler_v2 --cycles 20 --seed 42`
  - Confirmed current WSL base Python does not have TensorFlow, so TF graph smoke and 1-step training smoke remain pending
  - Added `configs/supernet_fc2_172x224_v2.yaml` as the formal V2 search-training config
  - Kept `configs/supernet_fc2_180x240_v2.yaml` unchanged as the historical smoke-compatible config
- Files created/modified:
  - `plan/supernet_v2/task_plan.md`
  - `plan/supernet_v2/findings.md`
  - `plan/supernet_v2/progress.md`
  - `configs/supernet_fc2_172x224_v2.yaml`
  - `configs/supernet_fc2_180x240_v2.yaml`
  - `wrappers/run_supernet_train_v2.py`
  - `efnas/app/train_supernet_app_v2.py`
  - `efnas/network/MultiScaleResNet_supernet_v2.py`
  - `efnas/nas/search_space_v2.py`
  - `efnas/nas/arch_codec_v2.py`
  - `efnas/nas/fair_sampler_v2.py`
  - `efnas/nas/eval_pool_builder_v2.py`
  - `efnas/engine/supernet_trainer_v2.py`
  - `tests/test_run_supernet_train_wrapper_v2.py`
  - `tests/test_supernet_v2_space_helpers.py`

## Notes

- V2 training can stay isolated from search in the first pass if we add parallel v2 files instead of modifying the entire v1 stack immediately.
- The cleanest engineering decision still depends on the fairness-cycle choice for mixed 3-choice and 2-choice blocks.
