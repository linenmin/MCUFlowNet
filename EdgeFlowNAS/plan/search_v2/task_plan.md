# Task Plan: Search V2 Diagnostic and Search Redesign

## Goal

Before launching the second multi-agent NAS round, verify whether inherited-weight `supernet_v2` subnet ranking on `FC2 val` is a reliable proxy for ranking on `Sintel`, then decide whether `search_v2` should first try a parameter-tuned `v1.5` search loop or immediately move to a larger mechanism rewrite.

This phase is still diagnostic-first, but it now includes a concrete search-parameter decision gate for the next run.

## Current Phase

Phase 3

## Phases

### Phase 1: Rank-Consistency Diagnostic Design

- [x] Confirm the first concrete question for `search_v2`
- [x] Confirm the diagnostic should use inherited-weight `supernet_v2` subnets
- [x] Confirm comparison target is `FC2 val` rank vs `Sintel` rank
- [x] Confirm this phase should produce runnable HPC commands
- **Status:** complete

### Phase 2: Diagnostic Tooling

- [x] Add a V2 supernet evaluator that restores inherited weights from `supernet_best.ckpt` or `supernet_last.ckpt`
- [x] Add a V2 probe-pool sampler for 50 mixed-complexity subnets
- [x] Add a wrapper script for one-command execution on HPC
- [x] Add unit tests for pure helper logic and wrapper defaults
- **Status:** complete

### Phase 3: Local Static Validation

- [x] Verify unit tests in `tf_work_hpc`
- [x] Verify wrapper `--dry_run` in `tf_work_hpc`
- [x] Confirm no dataset access is required for dry-run validation
- **Status:** complete

### Phase 4: HPC Execution

- [x] Run the full 50-subnet diagnostic on HPC
- [x] Export `rank_consistency_records.csv`
- [x] Export `rank_consistency_summary.json`
- [x] Review Spearman / Kendall / top-k overlap
- **Status:** complete

## Validation Notes

- `BN recal` intentionally stays fixed-batch for runtime control.
- `FC2 val` must use full sequential coverage for rank-comparison validity.
- `Sintel` uses full list coverage unless a manual cap is requested for pilot runs.

### Phase 5: Search V2 Decision Gate

- [x] If correlation is high, keep FC2-driven search and use Sintel only for shortlist re-ranking
- [ ] If correlation is medium, design a mixed FC2+Sintel search pipeline
- [ ] If correlation is low, redesign search_v2 around Sintel-aware evaluation
- **Status:** complete

### Phase 6: Search V1.5 Parameter Re-tuning

- [x] Re-assess whether late-stage collapse in `search_v1` is pure failure or near-convergence plus bad scheduling
- [x] Decide whether to test parameter-only changes before redesigning the agent mechanism
- [x] Lock the next candidate search settings for a `v1.5` verification run
- [ ] If needed, create a dedicated config for the `v1.5` rerun
- **Status:** in progress

### Phase 7: Prompt Review And V2 Execution Readiness

- [x] Re-review shared worldview and agent prompts against the V2 11D search space
- [x] Reduce `assumptions` influence on `Agent A` without removing the scientist loop
- [x] Fix V1-only prompt examples that conflict with V2 head choices
- [x] Check whether the current agentic-search runtime can be reused for V2 as-is
- [ ] Create a dedicated `search_v2` config and V2 evaluation entrypoint before any real rerun
- **Status:** in progress

## Decision Rules

1. `Spearman >= 0.8`
   - FC2 rank is a strong proxy
   - Search V2 can stay FC2-driven with Sintel used for shortlist selection
2. `0.5 <= Spearman < 0.8`
   - Proxy is only partially reliable
   - Search V2 should add periodic Sintel re-scoring
3. `Spearman < 0.5`
   - Proxy is unreliable
   - Search V2 must treat Sintel as a core ranking signal

## Working Decisions

| Decision | Rationale |
| --- | --- |
| Use inherited-weight V2 subnets first | This directly informs the next search design before retraining any standalone models |
| Use 50 subnets with stratified complexity coverage | Random-only sampling is too likely to miss ranking inversions in key regions |
| Recalibrate BN with FC2 train batches before per-arch evaluation | This is more defensible than evaluating stale moving statistics directly |
| Keep this phase outside the agentic loop | The question is methodological, not LLM-agent-specific |

## Expected Outputs

- `wrappers/run_supernet_v2_rank_consistency.py`
- `metadata/sampled_probe_arches.csv`
- `metadata/rank_consistency_records.csv`
- `metadata/rank_consistency_summary.json`
- `metadata/rank_consistency_summary.md`

## Final Decision From This Diagnostic

1. `FC2 val` is strong enough to remain the first-stage ranking signal for `search_v2`
   - measured `Spearman = 0.8996`
   - measured `Kendall tau = 0.7339`
2. `FC2 val` is not strong enough to be the only final selection criterion
   - `top-5 overlap = 0.6`
   - `top-10 overlap = 0.8`
   - the largest observed rank inversion was `17` places
3. `search_v2` should therefore use a two-stage evaluation policy
   - stage 1: search and coarse filtering by `FC2 + hardware`
   - stage 2: full-list `Sintel` re-ranking on the `FC2` shortlist
4. Practical shortlist rule supported by this diagnostic
   - all `Sintel top-10` probe architectures were contained inside `FC2 top-20`
   - `FC2 top-15` already covered `9/10` of the `Sintel top-10`

## Current Search-V1.5 Decision

1. Do not jump directly to a heavy search-loop rewrite
   - `search_v1` likely entered a strong convergence region early
   - the later collapse is now interpreted as near-convergence plus over-frequent reflection and prompt-level self-reinforcement
2. First verify a parameter-tuned `v1.5` run
   - keep `batch_size = 20`
   - change `scientist_trigger_interval = 5`
   - keep `assumption_confidence_threshold = 0.95`
3. Current LLM temperature baseline from `configs/search_v1.yaml`
   - `agent_a = 0.35`
   - `agent_b = 0.40`
4. Current temperature experiment recommendation
   - test `agent_a = 0.60`
   - test `agent_b = 0.60`
   - keep `agent_c`, `agent_d1`, `agent_d2`, `agent_d3` unchanged for the first rerun
5. Why this parameter-first pass is preferred
   - it preserves the existing agent-team design
   - it directly tests the user's hypothesis that the system itself may be basically sound
   - it avoids mixing mechanism redesign with search-space and evaluation changes in the same round

## Prompt Review Decisions For Search V2

1. Keep the agent-team structure
   - no role was removed
   - the main issue was prompt alignment with the new V2 search space and team protocol
2. Shared worldview was updated to V2 semantics
   - the search space is now explicitly defined as `11D`
   - the front `6` blocks are `3-choice`
   - the last `5` head blocks are `2-choice`
   - the backbone blocks are described as concrete `ResNet-style residual block stack` choices rather than vague "depth blocks"
3. `Agent B` received one hard reliability fix
   - candidates must not repeat history
   - candidates must also not repeat within the same batch
4. `Agent A` was intentionally rebalanced
   - `assumptions.json` is now framed as a weak signal
   - `findings.md` and historical Pareto evidence remain the strong signals
   - assumption-testing budget is now optional and capped at `30%`
   - `verify_assumptions` can be `0` when there is no good reason to test a guess
5. `Agent D3` was cleaned of a V1-only example
   - the old example used an invalid head value `2`
   - it now uses a V2-valid `0/1` head-choice example

## Search-V2 Runtime Readiness Check

1. The current prompt layer is now broadly aligned with V2
2. The current runtime layer is not yet fully V2-ready
   - `configs/search_v1.yaml` still points to V1 evaluation assets:
     - `wrappers/run_supernet_subnet_distribution.py`
     - `configs/supernet_fc2_180x240.yaml`
   - `SearchCoordinator._record_epoch_metrics()` still hard-codes V1 space size as `3 ** 9`
   - the current worker path still evaluates architectures through the V1 subnet-distribution stack
3. Therefore, before a real V2 agentic rerun, the minimum required work is:
   - add a dedicated `search_v2` config
   - point evaluation to a V2-capable worker entrypoint
   - update epoch coverage accounting to `23328`
   - verify the end-to-end evaluation path accepts 11D codes all the way through
