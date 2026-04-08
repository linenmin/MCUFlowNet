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
