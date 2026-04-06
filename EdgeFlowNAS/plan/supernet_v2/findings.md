# Findings

## V2 Search Space

- V2 uses 11 searchable blocks.
- Front 6 blocks each have 3 choices.
- Last 5 head blocks each have 2 choices.
- Search space size is `3^6 * 2^5 = 23328`.

## Confirmed Operator Choices

- `E0`: `7x7 stride-2`, `5x5 stride-2`, `3x3 stride-2`
- `E1`: `5x5 stride-2`, `3x3 stride-2`, `3x3 stride-2 + 3x3 dilated`
- `EB0/EB1/DB0/DB1`: `Deep1/Deep2/Deep3`
- `H0Out/H1/H1Out/H2/H2Out`: `3x3/5x5`

## V1 Assumptions That Conflict With V2

- `MultiScaleResNet_supernet.py` hard-codes a 9-choice arch code.
- `supernet_trainer.py` hard-codes:
  - `arch_code_ph` shape `[9]`
  - `teacher_arch_code_ph` shape `[9]`
  - `num_blocks=9` in multiple helper paths
- `arch_codec.py` assumes:
  - length `9`
  - every block uses `0/1/2`
  - head labels are `3x3/5x5/7x7`
- `fair_sampler.py` assumes:
  - `num_blocks=9`
  - every block has exactly 3 options
  - each fairness cycle contains 3 paths
- `eval_pool_builder.py` assumes:
  - every block uses `0/1/2`
  - seed codes are built from 3-option logic
- unit tests currently build graph with `shape=[9]`

## Implementation Impact

The V2 change is moderate.

It is larger than a simple new wrapper because the mixed-cardinality search space changes the training logic, not only the network graph.

The core files that matter for V2 training are:

- `efnas/network/MultiScaleResNet_supernet.py`
- `efnas/engine/supernet_trainer.py`
- `efnas/nas/arch_codec.py`
- `efnas/nas/fair_sampler.py`
- `efnas/nas/eval_pool_builder.py`
- `tests/test_supernet_network_structure.py`
- `wrappers/run_supernet_train.py`

## V2 Training-Only Scope

This round can stop at the training path.

The following areas do not need to change yet if V2 keeps separate files:

- search prompts
- subnet distribution tools
- search-side visualization and ranking helpers
- standalone fixed-arch training tools

This reduces the initial implementation risk.

## Key Technical Risk

The main technical issue is not the 11-d arch code itself.

The main issue is fairness.

V1 uses a 3-path fairness cycle because every block has 3 options. V2 mixes:

- 6 blocks with 3 options
- 5 blocks with 2 options

So V2 must redefine what one fairness cycle means.

## Fairness Design Options

### Current fairness mode

- Keep `3` single-path models per cycle.
- Keep the FairNAS Appendix A.1 irregular-space idea of expanding 2-choice head blocks to temporary 3-slot lists.
- Replace random duplication with under-sampled-first duplication.
- If both options have the same count, break the tie randomly.
- Then run the same without-replacement 3-path cycle used by FairNAS strict fairness.

This is a project-specific balanced irregular-fairness extension.

It is no longer the exact A.1 random-resampling rule.

Its purpose is simple:

- preserve the low 3-path training cost
- reduce long-run count drift on 2-choice head blocks

## Recommended File Strategy

Current assessment: add a parallel V2 path first.

Reasons:

- V1 is internally consistent but tightly coupled to 9 blocks and 3 choices.
- A shared refactor is possible, but it would spread across more files before the first smoke test.
- A parallel V2 path keeps the change set readable and easier to debug.

## WSL Environment Status

- In the current WSL conda list, `tf_work_hpc` is not available.
- Available environments are:
  - `base`
  - `raw2event`
  - `torch`
  - `vela`

This means local smoke validation in WSL needs one of these paths:

1. create a new TF training env in WSL
2. use an existing env if it already has the required TF1 stack
3. write code in WSL and only run syntax-level checks locally

Current assessment: option 3 is the safest immediate path unless the user wants to create a new TF env in WSL now.

## Git Synchronization Status

- `master` tracks `origin/master`
- `git fetch origin` completed successfully
- `HEAD...origin/master = 0 0`
- Current worktree has only untracked local directories:
  - `outputs/fixed_arch_vela_compare/`
  - `plan/supernet_v2/`

Current assessment: the repository is up to date enough to start V2 implementation without pulling new code first.

## FairNAS Reference Check

I checked the FairNAS paper and the official repository.

The confirmed FairNAS core rule is:

- one strict-fairness training step samples `m` single-path models without replacement
- each choice block has `m` candidate operations
- each operation is sampled equally within the same step

The most direct evidence appears in the FairNAS sampling figure, which states:

- `sample m models without replacement and train them sequentially`
- `All operations are thus ensured to be equally sampled and trained within every step t`

Current assessment:

- FairNAS strict fairness is defined for a uniform-cardinality search space
- the paper supplement Appendix A.1 gives an irregular-space extension
- V2 now follows that Appendix A.1 rule instead of using a 6-path custom cycle

## V2 Implementation Status

The first implementation pass now exists as a parallel V2 path.

New files:

- `configs/supernet_fc2_180x240_v2.yaml`
- `configs/supernet_fc2_172x224_v2.yaml`
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

## WSL Static Validation Status

Completed:

- Python syntax compilation for all new V2 files
- `run_supernet_train_v2.py --dry_run`
- wrapper and helper unit tests
- `arch_codec_v2` self test
- `eval_pool_builder_v2 --check`
- `fair_sampler_v2` smoke run
- balanced irregular sampler smoke run shows `fairness_gap = 0` after 20 cycles with seed 42

Blocked in current WSL environment:

- TensorFlow graph build smoke test
- 1-step training smoke test

Reason:

- current WSL base Python does not have TensorFlow installed

## Resume Compatibility Note

The fairness mode string now changed from the earlier A.1 random version to:

- `balanced_irregular_fairness`

This means a new run should use a new experiment name.

Resuming an older run that was started with the earlier random-irregular config is not recommended.

## Training Resolution Decision

The formal V2 supernet training resolution should align with the deployment and fixed-arch comparison resolution.

That aligned resolution is `172x224`.

Reason:

- stem operator evaluation in `model_design` uses `172x224`
- fixed-arch compare training already uses `172x224`
- the V2 search space now includes `E0`, `E1`, and head kernel choices, which are resolution-sensitive

Implementation choice:

- keep `configs/supernet_fc2_180x240_v2.yaml` as the historical smoke-compatible config
- add `configs/supernet_fc2_172x224_v2.yaml` as the formal V2 search-training config
