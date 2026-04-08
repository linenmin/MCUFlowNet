# Task Plan: Search V2 Diagnostic and Search Redesign

## Goal

Before launching the second multi-agent NAS round, verify whether inherited-weight `supernet_v2` subnet ranking on `FC2 val` is a reliable proxy for ranking on `Sintel`.

This phase is diagnostic-first. It does not yet modify the formal agentic search loop.

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

- [ ] Run the full 50-subnet diagnostic on HPC
- [ ] Export `rank_consistency_records.csv`
- [ ] Export `rank_consistency_summary.json`
- [ ] Review Spearman / Kendall / top-k overlap
- **Status:** pending

## Validation Notes

- `BN recal` intentionally stays fixed-batch for runtime control.
- `FC2 val` must use full sequential coverage for rank-comparison validity.
- `Sintel` uses full list coverage unless a manual cap is requested for pilot runs.

### Phase 5: Search V2 Decision Gate

- [ ] If correlation is high, keep FC2-driven search and use Sintel only for shortlist re-ranking
- [ ] If correlation is medium, design a mixed FC2+Sintel search pipeline
- [ ] If correlation is low, redesign search_v2 around Sintel-aware evaluation
- **Status:** pending

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
