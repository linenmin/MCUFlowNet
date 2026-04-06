# Task Plan: Supernet V2 Definition and Training

## Goal

Implement a clean and maintainable Supernet V2 training path in `MCUFlowNet/EdgeFlowNAS`.

This round only covers supernet definition and training. It does not cover search.

The confirmed V2 search space is:

- first 6 blocks: 3 choices each
- last 5 head blocks: 2 choices each
- total space size: `3^6 * 2^5 = 23328`

## Confirmed Search Space

### Front 6 blocks

1. `E0`
   - `0 = 7x7 stride-2`
   - `1 = 5x5 stride-2`
   - `2 = 3x3 stride-2`
2. `E1`
   - `0 = 5x5 stride-2`
   - `1 = 3x3 stride-2`
   - `2 = 3x3 stride-2 + 3x3 dilated`
3. `EB0`
   - `0 = Deep1`
   - `1 = Deep2`
   - `2 = Deep3`
4. `EB1`
   - `0 = Deep1`
   - `1 = Deep2`
   - `2 = Deep3`
5. `DB0`
   - `0 = Deep1`
   - `1 = Deep2`
   - `2 = Deep3`
6. `DB1`
   - `0 = Deep1`
   - `1 = Deep2`
   - `2 = Deep3`

### Last 5 head blocks

7. `H0Out`
   - `0 = 3x3`
   - `1 = 5x5`
8. `H1`
   - `0 = 3x3`
   - `1 = 5x5`
9. `H1Out`
   - `0 = 3x3`
   - `1 = 5x5`
10. `H2`
    - `0 = 3x3`
    - `1 = 5x5`
11. `H2Out`
    - `0 = 3x3`
    - `1 = 5x5`

## Current Phase

Phase 2

## Phases

### Phase 1: V2 Scope Lock

- [x] Confirm V2 block list and choice definitions
- [x] Confirm search space size
- [x] Confirm this round only covers supernet training, not search
- **Status:** complete

### Phase 2: Codebase Impact Assessment

- [x] Inspect v1 supernet network definition
- [x] Inspect v1 trainer assumptions
- [x] Inspect v1 arch codec and fairness sampler assumptions
- [x] Identify files that hard-code `9` blocks and `0/1/2` choices
- [x] Estimate implementation size for training-only V2
- [x] Decide V2 training fairness strategy
- [x] Decide V2 file organization strategy
- **Status:** complete

### Phase 3: V2 File Layout Design

- [x] Decide whether V2 reuses shared trainer modules or adds parallel v2 files
- [x] Define `run_supernet_train_v2.py`
- [x] Define `configs/supernet_v2_*.yaml`
- [x] Define network / codec / sampler / eval module names
- **Status:** complete

### Phase 4: V2 Supernet Graph Implementation

- [x] Implement V2 network definition with 11-choice arch code
- [x] Implement E0 3-choice operator block
- [x] Implement E1 3-choice operator block
- [x] Implement 2-choice head blocks
- [x] Keep output shapes and multiscale heads aligned with v1
- **Status:** complete

### Phase 5: V2 Training Path Implementation

- [x] Update or extend trainer to support 11-d arch code
- [x] Update fairness counting and eval pool generation
- [x] Update dry-run and manifest outputs
- [x] Add minimal tests for graph build and fair sampling
- **Status:** complete

### Phase 6: Local Dry-Run Validation

- [x] Run `--dry_run`
- [ ] Run graph build smoke test
- [ ] Run 1-step CPU smoke test if environment allows
- [x] Check non-TF sampler / codec / eval-pool validation
- **Status:** in_progress

## Key Questions

1. How should Supernet V2 keep FairNAS-style strict fairness when the search space mixes 3-choice blocks and 2-choice blocks?
2. Should V2 extend the existing trainer stack with config-driven block specs, or create a clean parallel V2 path?
3. Which local environment should be used for smoke validation in WSL if `tf_work_hpc` is not available here?

## Recommended Implementation Path

### Recommended default

- keep v1 untouched
- add a parallel V2 training path
- only share small helpers when they are already generic
- keep FairNAS A.1 irregular-space preprocessing with 3-path cycles

### Proposed V2 file set

- `wrappers/run_supernet_train_v2.py`
- `configs/supernet_fc2_180x240_v2.yaml`
- `efnas/app/train_supernet_app_v2.py`
- `efnas/network/MultiScaleResNet_supernet_v2.py`
- `efnas/nas/arch_codec_v2.py`
- `efnas/nas/fair_sampler_v2.py`
- `efnas/nas/eval_pool_builder_v2.py`
- `efnas/engine/supernet_trainer_v2.py`
- `tests/test_supernet_network_structure_v2.py`

### Why this path is recommended

- V1 has many hard-coded 9-block and 3-choice assumptions.
- Training-only V2 does not need to touch the whole search stack yet.
- A parallel V2 path keeps the current training entry stable.
- This path is easier to review and easier to roll back.

## Working Decisions

| Decision | Rationale |
| --- | --- |
| V2 keeps the v1 engineering style | The user wants `run_supernet_train_v2.py` and related modules to stay clean and maintainable |
| This round excludes search | The user explicitly wants to start from supernet definition and training only |
| Head 7x7 options are removed | The supervisor and user already decided they are not needed in V2 |
| V2 must preserve `EdgeFlowNAS` layering style | The existing wrapper/app/network/engine split should remain intact |
| V2 should avoid broad refactors before smoke validation | The current priority is a clean training path, not a full framework generalization |

## Expected Impact

V2 is not a small wrapper-only change. The main impact is moderate and concentrated in the supernet stack:

- network definition
- arch code parsing
- fairness sampler
- eval pool builder
- trainer graph placeholders and fairness bookkeeping
- smoke tests

Search-specific code can stay untouched in the first implementation pass if V2 training uses separate files.

## Decision Gate Before Implementation

1. fairness mode
   - `Chosen`: `3-path approximated SF` with FairNAS A.1 irregular-space preprocessing
2. code organization
   - `Chosen`: parallel v2 modules
3. local validation
   - `Chosen`: only do static checks in WSL, then smoke-test in Windows or HPC
