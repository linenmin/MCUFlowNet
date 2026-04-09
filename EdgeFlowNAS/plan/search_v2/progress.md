# Progress Log

## Session: 2026-04-08

### Phase 1-2: Diagnostic Setup

- **Status:** complete
- **Actions taken:**
  - Confirmed the first `search_v2` task should be a methodological diagnostic, not immediate search execution.
  - Added a V2 evaluator for inherited-weight checkpoint restoration.
  - Added a V2 probe-pool sampler and FC2/Sintel rank-consistency summary logic.
  - Added a wrapper script for HPC execution.
  - Added unit tests for helper logic and wrapper defaults.

## Pending Verification

- Completed:
  - `python -m unittest tests.test_supernet_v2_rank_consistency tests.test_run_supernet_v2_rank_consistency_wrapper`
  - `conda run -n tf_work_hpc python -m unittest tests.test_supernet_v2_rank_consistency tests.test_run_supernet_v2_rank_consistency_wrapper`
  - `python wrappers/run_supernet_v2_rank_consistency.py --dry_run`
  - `conda run -n tf_work_hpc python wrappers/run_supernet_v2_rank_consistency.py --dry_run`
- Result:
  - tests passed in both default Python and `tf_work_hpc`
  - dry-run produced a valid `outputs/search_v2/rank_consistency_*` path
  - no dataset access was required for dry-run validation

## Session: 2026-04-08 (Coverage Fix)

### FC2 Evaluation Control

- **Status:** complete
- **Actions taken:**
  - reviewed the initial diagnostic and confirmed FC2 validation still used a fixed batch count rather than full-set coverage
  - added `build_fc2_eval_windows()` so FC2 val now runs sequentially across the whole split without wraparound duplication
  - kept BN recalibration as fixed-batch calibration rather than full-train evaluation
  - added regression tests for full-set and capped FC2 window generation
  - re-ran local and `tf_work_hpc` tests plus wrapper dry-run

## Intended HPC Deliverable

- One output directory under `outputs/search_v2/`
- CSV of all sampled architectures and both metrics
- JSON/Markdown summary of correlation and overlap statistics

## Session: 2026-04-08 (HPC Diagnostic Review)

### Rank Consistency Result

- **Status:** complete
- **Output reviewed:**
  - `outputs/rank_consistency_supernet_v2_fc2_vs_sintel/metadata/rank_consistency_records.csv`
  - `outputs/rank_consistency_supernet_v2_fc2_vs_sintel/metadata/rank_consistency_summary.json`
  - `outputs/rank_consistency_supernet_v2_fc2_vs_sintel/metadata/rank_consistency_summary.md`
- **Key numbers:**
  - `Spearman = 0.8996`
  - `Kendall tau = 0.7339`
  - `top-5 overlap = 0.6`
  - `top-10 overlap = 0.8`
  - largest rank inversion = `17`
- **Additional analysis:**
  - mean absolute rank shift = `4.88`
  - median absolute rank shift = `4`
  - all `Sintel top-10` probe architectures were contained inside `FC2 top-20`
  - `FC2 top-15` already contained `9/10` of the `Sintel top-10`

### Search-V2 Decision

- **Status:** complete
- **Decision taken:**
  - keep `FC2 val` as the first-stage proxy signal because it remains informative and much cheaper than full-list `Sintel`
  - reject `FC2-only` final selection because top-end rank inversions are still meaningful
  - adopt a two-stage evaluation policy for the next search round:
    - coarse search on `FC2 + hardware`
    - shortlist re-ranking on full-list `Sintel`

### Next Planned Step

- Convert this decision into the concrete `search_v2` execution design:
  - shortlist width
  - when Sintel re-ranking is triggered
  - how shortlisted Pareto subnets are retrained on `FC2 + FlyingThings3D`

## Session: 2026-04-09 (Search-V1 Reassessment)

### Search-V1 Late-Phase Interpretation

- **Status:** complete
- **Actions taken:**
  - revisited `outputs/search_v5_20260311_003305/metadata/epoch_metrics.csv`
  - separated `best_epe` stagnation from Pareto-front growth
  - re-evaluated whether the first agent-team run should be treated as mechanism failure
- **Conclusion:**
  - the run likely entered a strong convergence region early
  - later collapse is better interpreted as near-convergence plus over-frequent reflection and self-reinforcing generation
  - a parameter-only rerun is justified before any large redesign

### Agreed V1.5 Parameter Direction

- **Status:** complete
- **Decision taken:**
  - keep `batch_size = 20`
  - change `scientist_trigger_interval = 5`
  - keep `assumption_confidence_threshold = 0.95`

### LLM Temperature Review

- **Status:** complete
- **Observed current settings from `configs/search_v1.yaml`:**
  - `agent_a = 0.35`
  - `agent_b = 0.40`
  - `agent_c = 0.15`
  - `agent_d1 = 0.45`
  - `agent_d2 = 0.15`
  - `agent_d3 = 0.25`
- **Decision taken for the next rerun candidate:**
  - test `agent_a = 0.60`
  - test `agent_b = 0.60`
  - keep other agent temperatures unchanged in the first `v1.5` rerun

### Immediate Next Step

- create or stage a dedicated `search_v1.5` config that changes only:
  - `scientist_trigger_interval: 5`
  - `llm.temperature.agent_a: 0.60`
  - `llm.temperature.agent_b: 0.60`

## Session: 2026-04-09 (Prompt Review And V2 Search Readiness)

### Prompt-Layer Cleanup

- **Status:** complete
- **Actions taken:**
  - re-reviewed the shared worldview and all active agent prompts against the V2 `11D` search space
  - updated the public team/worldview text so it no longer carries V1-only wording
  - aligned the shared search-space description to `3^6 * 2^5 = 23328`
  - clarified concrete block semantics for the searchable V2 code
  - added the missing `Agent B` hard rule forbidding within-batch duplicates
  - corrected the stale V1-only `Agent D3` example that used an invalid head value

### Agent-A Reweighting Decision

- **Status:** complete
- **Decision taken:**
  - keep `assumptions.json` visible to `Agent A`
  - explicitly downgrade assumptions to weak signals
  - keep `findings.md` and historical Pareto evidence as strong signals
  - retain assumption-testing as an optional budget type, but cap it at `30%`
  - allow `verify_assumptions = 0` in any round where no active guess deserves testing

### V2 Runtime Reuse Check

- **Status:** complete
- **What was checked:**
  - `wrappers/run_agentic_search.py`
  - `configs/search_v1.yaml`
  - `efnas/search/coordinator.py`
  - `efnas/search/eval_worker.py`
  - `efnas/nas/supernet_subnet_distribution.py`
- **Conclusion:**
  - the prompt layer is now V2-aware
  - the runtime layer is still partially V1-bound
  - the current agentic search loop cannot be treated as V2-ready without at least:
    - a dedicated `search_v2` config
    - a V2-capable evaluation entrypoint
    - corrected coverage accounting in epoch metrics

## Session: 2026-04-09 (V2 First-Stage Runtime Implementation)

### Implemented V2 Search Runtime Surface

- **Status:** complete
- **Actions taken:**
  - added [search_v2.yaml](d:/Dataset/MCUFlowNet/EdgeFlowNAS/configs/search_v2.yaml)
  - kept Gemini routing
  - set `batch_size = 16`
  - set `scientist_trigger_interval = 5`
  - set `agent_a = 0.60`
  - set `agent_b = 0.60`
  - routed evaluation to a dedicated V2 wrapper:
    - `wrappers/run_supernet_subnet_distribution_v2.py`
    - `efnas/nas/supernet_subnet_distribution_v2.py`

### FC2 Validation Policy In Code

- **Status:** complete
- **Decision implemented:**
  - first-stage `search_v2` now treats `FC2 val` as full sequential coverage by default
  - `eval_batches_per_arch` is no longer the governing control for V2 first-stage evaluation
  - an optional `max_fc2_val_samples` override exists only for pilot/debug runs

### Runtime Compatibility Fixes

- **Status:** complete
- **Actions taken:**
  - added `max_fc2_val_samples` passthrough in `efnas/search/eval_worker.py`
  - added configurable `search_space_size` handling in `efnas/search/coordinator.py`
  - set V2 coverage accounting to `23328` through config rather than leaving it hard-coded to V1

### Verification

- **Status:** complete
- **Checks passed:**
  - `conda run -n tf_work_hpc python -m unittest tests.test_run_supernet_subnet_distribution_v2_wrapper tests.test_eval_worker_command tests.test_search_coordinator_v2_metrics`
  - module import check for `efnas.nas.supernet_subnet_distribution_v2`
  - wrapper `--dry_run` for `run_supernet_subnet_distribution_v2.py`
