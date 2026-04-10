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

## Session: 2026-04-09 (FC2 vs Sintel Timing Probe)

### Single-Subnet Timing Result

- **Status:** complete
- **Output reviewed:**
  - `outputs/search_v2/timing_probe_ref/metadata/timing_probe_summary.json`
- **Probe target:**
  - `arch_code = 0,0,0,0,0,0,1,1,1,1,1`
  - checkpoint = `supernet_best.ckpt`
- **Measured times:**
  - `BN recal = 25.57s`
  - `FC2 eval = 17.53s` over `640` samples
  - `Sintel eval = 124.85s` over `1041` samples
  - `Sintel / FC2 eval ratio = 7.12x`
  - `Sintel / FC2 per-sample ratio = 4.38x`

### Search Implication

- **Status:** complete
- **Decision reinforced:**
  - do not replace first-stage `FC2 val` with full-list `Sintel`
  - keep `FC2 val` as the stage-1 search evaluator for throughput
  - reserve `Sintel` for later re-ranking / validation because the inherited-weight eval path is materially more expensive on Sintel

### Gemini Temperature Note

- **Status:** review complete
- **Current interpretation:**
  - if the run stays on Gemini `3 / 3.1` preview models, `temperature = 1.0` should be treated as the compatibility baseline, not as an intentionally high-creativity setting
  - future control over output discipline should rely primarily on prompt constraints, schema constraints, and model choice rather than low-temperature tuning on Gemini 3-family models

## Session: 2026-04-09 (Baseline Selection)

### Single Baseline Decision

- **Status:** complete
- **Decision taken:**
  - if only one non-agent baseline is implemented for the paper, use `NSGA-II`
  - do not prioritize `Random Search` as a main baseline
  - keep `Local Search` only as a backup option, not the first implementation target

### Rationale

- **Status:** complete
- **Why `NSGA-II`:**
  - the current search problem is explicitly a two-objective Pareto search on `EPE↓` and `FPS↑`
  - `NSGA-II` is the canonical and most widely recognized multi-objective evolutionary baseline
  - this makes it a cleaner authority-backed comparison point for the agent-team search than `Local Search`
- **Why not `Random Search` first:**
  - the paper does not need to spend implementation budget re-proving that a mature Pareto optimizer is stronger than random exploration
  - if random is ever needed, it can be added later as a lightweight sanity check

### Pending Confirmation Before Implementation

- **Status:** pending
- **Need to confirm:**
  - preferred `population_size`
  - exact generation count once `population_size` is fixed

## Session: 2026-04-09 (NSGA-II Baseline Scope Lock)

### Confirmed Design Choices

- **Status:** complete
- **User-confirmed settings:**
  - match the agent run's planned total evaluation budget
  - optimize only `EPE↓` and `FPS↑`
  - do not add SRAM as a hard constraint because the current V2 search space is already SRAM-safe
  - use a fixed-generation stop rule
  - use standard `crossover + mutation`

### Remaining Open Point

- **Status:** pending
- **Still to settle:**
  - `population_size`, which will also determine the final generation count under the fixed budget
  - this choice should be informed by an existing recognized NAS / multi-objective NAS implementation rather than chosen ad hoc

## Session: 2026-04-09 (NSGA-II Baseline Implementation)

### Implemented Runtime Surface

- **Status:** complete
- **Actions taken:**
  - added `efnas/baselines/` as a dedicated home for non-agent baselines
  - implemented `efnas/baselines/nsga2_search.py`
  - added CLI entrypoint `wrappers/run_nsga2_search.py`
  - added dedicated config `configs/nsga2_v2.yaml`

### Implementation Decisions

- **Status:** complete
- **Decisions applied in code:**
  - keep the baseline separate from the agentic search coordinator
  - reuse the existing V2 subprocess evaluator via `evaluate_single_arch(...)`
  - reuse the existing `history_archive.csv` and `epoch_metrics.csv` conventions for easier comparison
  - derive `total_generations = total_evaluations / population_size = 800 / 50 = 16`
  - use project-native categorical `uniform crossover + mutation` rather than adding a new external optimization dependency in the first version

### Verification

- **Status:** partial local verification complete
- **Checks passed:**
  - `python -m unittest tests.test_nsga2_baseline tests.test_run_nsga2_search_wrapper`
  - `python -m py_compile efnas\\baselines\\__init__.py efnas\\baselines\\nsga2_search.py wrappers\\run_nsga2_search.py`
  - `python wrappers\\run_nsga2_search.py --help`
- **Known local limitation:**
  - full end-to-end execution was not run in the default Windows Python because the local `pandas/pyarrow` stack is currently binary-incompatible with local NumPy
  - the real runtime validation therefore still belongs on the user's HPC / `tf_work` environment

## Session: 2026-04-10 (Agent-Team Control-Loop Refactor Decision)

### Over-Design Rejected

- **Status:** complete
- **Decision taken:**
  - reject the earlier idea of creating many new persistent summary files just to steer the agents
  - keep the control-loop state surface small and maintainable
  - prefer runtime-computed prompt sections over document proliferation

### Findings Direction Locked

- **Status:** complete
- **Decision taken:**
  - retire prose-style `findings.md` from the control loop
  - move to a script-first finding system backed by `findings.json`
  - each promoted finding should keep one reusable rule script (for verification, revalidation, and candidate checking)
  - `findings.json` should remain a compact registry rather than a rigid rule-DSL

### Agent-A Input Direction Locked

- **Status:** partial decision complete
- **Decision taken:**
  - `assumptions.json` should no longer feed back into Agent A
  - Agent A should be driven by:
    - `history_archive.csv`
    - `epoch_metrics.csv`
    - runtime-computed current Pareto-point list
  - do not introduce dedicated `strategy_memory.json` / `search_progress.json` / similar summary artifacts just to feed A
- **Still open:**
  - whether Agent A should continue seeing its own previous strategy record in any form

### Runtime-Consistency Direction Locked

- **Status:** complete
- **Decision taken:**
  - add a minimal `run_state.json` for phase-aware resume
  - add malformed-JSON retry/repair in `chat_json()`
  - avoid expanding the metadata surface beyond what is needed for correctness

### Role Split Direction

- **Status:** complete
- **Decision taken:**
  - Agent A should focus on search planning only
  - Agent A should not directly allocate scientist verification budget
  - Agent B should remain the generator and receive active-rule guidance
  - the coordinator/engine should own scientist scheduling and final rule enforcement

## Session: 2026-04-10 (Control-Loop Refactor Implementation)

### Implemented Refactor Surface

- **Status:** complete
- **Actions taken:**
  - switched the control loop from prose-style `findings.md` to registry-style `findings.json`
  - kept legacy `findings.md` helpers only as compatibility shims for old tools and old experiment folders
  - added `run_state.json` and made coordinator resume phase-aware
  - removed `assumptions / findings / strategy_log` from Agent A runtime input
  - changed Agent A to use:
    - `history_archive.csv`
    - `epoch_metrics.csv`
    - runtime-computed current Pareto-point list
  - kept `search_strategy_log.md` only as a human-facing audit log
  - updated Agent B to consume active finding hints rather than raw findings markdown
  - updated D2/D3 prompts toward a script-first rule workflow
  - added malformed-JSON retry in `LLMClient.chat_json()`

### Verification Status

- **Status:** mostly complete
- **Checks passed:**
  - `D:\\Anaconda3\\envs\\tf_work_hpc\\python.exe -m py_compile ...`
  - `D:\\Anaconda3\\envs\\tf_work_hpc\\python.exe -m unittest tests.test_search_coordinator_v2_metrics tests.test_eval_worker_command tests.test_file_io_resume_selection tests.test_agent_control_loop_refactor tests.test_llm_json_retry tests.test_file_io_registry_and_run_state`
  - `D:\\Anaconda3\\envs\\tf_work_hpc\\python.exe tests/test_search_dryrun.py`
- **Known local limitation:**
  - local Windows environments here still do not have `litellm` installed, so `run_agentic_search.py --help` could not be verified locally through a real import path
  - this is an environment gap, not a failing unit test in the refactor itself
