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
5. Runtime-cost confirmation from the dedicated timing probe
   - `FC2 eval = 17.53s` over `640` samples
   - `Sintel eval = 124.85s` over `1041` samples
   - `Sintel / FC2 eval ratio = 7.12x`
   - therefore stage-1 search should stay on `FC2 val`

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

## Current Operating Decision

1. Search evaluation
   - proceed with `FC2 val` as the first-stage search evaluator
   - do not switch the full search loop to `Sintel`
2. Gemini temperature policy
   - if warnings persist on Gemini `3 / 3.1` roles, prefer moving those roles to `temperature = 1.0`
   - treat `1.0` as the model-family baseline rather than as a deliberate creativity boost
   - if output discipline becomes the concern, fix that through prompting/schema/model choice first
3. Baseline policy
   - if only one non-agent baseline is implemented, choose `NSGA-II`
   - do not spend extra time implementing `Random Search` just to re-prove a weaker baseline
   - treat `Local Search` as a respected NAS baseline, but not the primary one for this paper's multi-objective Pareto setting

## Current Baseline Decision

1. The first comparison baseline for the agent team will be `NSGA-II`
   - rationale: this project is explicitly a two-objective Pareto search (`EPE↓`, `FPS↑`)
   - rationale: `NSGA-II` is the canonical and most widely accepted multi-objective evolutionary baseline
   - rationale: this choice is better aligned with the paper narrative than `Local Search`, which is strong but less canonical as a Pareto optimizer
2. `Local Search` remains a backup option
   - if `NSGA-II` integration proves unexpectedly awkward, `Local Search` can still be revived as a secondary NAS-style baseline
   - but it is not the first implementation target
3. `Random Search` is intentionally de-prioritized
   - the paper does not need to spend effort re-establishing that `NSGA-II` beats random exploration
   - if needed later, random can be added as a lightweight appendix-only sanity baseline rather than a main comparison

## NSGA-II Baseline Preparation Gate

Before implementation starts, confirm:

1. Search budget parity
   - use the same nominal total evaluation budget as the agent run
   - note that the agent run may realize fewer than `800` valid evaluations because of duplicates, but the baseline target remains the same planned budget
2. Objective definition
   - optimize `EPE↓` and `FPS↑` only
   - do not add SRAM as a hard constraint because the current V2 subnet space is already SRAM-safe
3. Population schedule
   - stopping rule should use fixed generations
   - the only remaining open hyperparameter is the preferred `population_size`
4. Candidate encoding operators
   - use standard `crossover + mutation`
5. External reference check
   - before implementation, anchor the first parameter choice to a recognized NAS / one-shot NAS codebase that already uses `NSGA-II`, to avoid purely ad-hoc settings

## NSGA-II Baseline Implementation Status

1. Scope
   - create a standalone `NSGA-II` baseline entrypoint rather than mixing it into the agentic runtime
   - reuse the existing V2 fixed-subnet evaluator and metadata conventions where practical
2. Directory structure
   - baseline algorithm code should live under `efnas/baselines/`
   - CLI entrypoint should live under `wrappers/`
   - dedicated config should live under `configs/`
3. Current implementation target
   - `population_size = 50`
   - `total_evaluations = 800`
   - derived `total_generations = 16`
   - objectives = `EPE↓`, `FPS↑`
   - operators = standard crossover + mutation

## Current Control-Loop Refactor Direction

### Priority 1: Findings Contract

1. Replace prose-style `findings.md` with machine-owned `findings.json`
2. Do not maintain a parallel human-facing findings document in the control loop
3. Treat each finding as a script-backed rule rather than a fixed template sentence
   - each rule keeps one reusable script such as `scripts/rule_Axx.py`
   - the same rule script is reused across assumption verification, finding promotion, revalidation, and runtime candidate checking
4. `findings.json` should act only as a registry/governance layer
   - it stores identity, active/inactive state, scope, confidence/support, enforcement type, and script path
   - it does not try to encode the full scientific logic in a rigid mini-language

### Priority 2: Agent-A Input Simplification

1. Do not add extra summary files just to feed Agent A
2. Agent A should consume only search facts and runtime-computed Pareto context
3. Current agreed direction:
   - keep `history_archive.csv` as the global fact table
   - keep `epoch_metrics.csv` as the global search-health table
   - compute the current Pareto-point list on demand at invocation time rather than persisting a large family of summary artifacts
4. Current agreed exclusions:
   - `assumptions.json` should not be fed back to Agent A
   - `findings` should not be used as raw strategist input in the first refactor pass
   - `search_strategy_log.md` should not be fed back to Agent A
5. Implementation note:
   - Agent A may still append a human-readable `strategic_reflection` to `search_strategy_log.md`
   - but that file is now for audit / human reading only, not strategist memory

### Priority 3: Runtime Consistency

1. Add a minimal `run_state.json`
   - store current epoch, current phase, and which side effects have already been committed
   - use it to make resume behavior phase-aware instead of replaying blindly
2. Add `chat_json()` malformed-JSON retry / repair logic
   - avoid hard failure of a full epoch due to one truncated LLM response
3. Keep the runtime state surface small
   - do not introduce a broad family of new metadata files unless a later need is proven

### Role Split Under The New Direction

1. Agent A
   - should plan from search facts only
   - should not schedule scientist validation work directly
2. Agent B
   - remains the candidate generator
   - should be aware of active finding constraints in a generator-friendly form
3. Coordinator / engine
   - owns scientist scheduling
   - owns assumption promotion / finding revalidation
   - remains the final rule executor for candidate legality
