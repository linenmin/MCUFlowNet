# Findings

## Diagnostic Scope

- The first `search_v2` question is not "which subnet wins", but "does FC2 ranking transfer to Sintel ranking under inherited weights?"
- This question should be answered before modifying the full agentic search loop.

## Why This Diagnostic Matters

- Previous search outputs were optimized around FC2-side inherited-weight evaluation plus hardware signals.
- The user observed architecture changes such as `eca` and `global gate` can look similar on FC2 while separating on Sintel.
- If the same phenomenon appears inside the V2 search space, then FC2-only search ranking is an unreliable proxy.

## Current V2 Search Space

- `supernet_v2` uses 11 searchable blocks.
- Front 6 blocks have 3 choices each.
- Last 5 head blocks have 2 choices each.
- Total search space size is `23328`.

## Tooling Decision

- Old `supernet_subnet_distribution.py` is hard-coded to the V1 9-block search space.
- A separate V2 diagnostic path is required instead of patching the old V1 script in place.

## Evaluation Decision

- The diagnostic uses inherited-weight subnets directly from the trained `supernet_v2` checkpoint.
- Each probe subnet is evaluated on:
  - FC2 validation EPE
  - Sintel EPE
- The implementation restores the original supernet checkpoint before each arch, then performs FC2-train BN recalibration before measuring FC2 and Sintel.
- BN recalibration remains fixed-batch by design.
- FC2 validation was updated to full sequential coverage without wraparound repetition.
- Sintel evaluation already used full list coverage unless an explicit sample cap is provided.

## Decision Metrics

- Spearman rank correlation
- Kendall tau
- Top-k overlap (`k=5,10`)
- Largest rank inversion case

## Interpretation Thresholds

- `Spearman >= 0.8`: FC2 rank is a strong proxy
- `0.5 <= Spearman < 0.8`: proxy is partial, mixed evaluation is needed
- `Spearman < 0.5`: FC2 rank is not reliable enough for search_v2

## HPC Diagnostic Result: `rank_consistency_supernet_v2_fc2_vs_sintel`

- Output directory: `outputs/rank_consistency_supernet_v2_fc2_vs_sintel`
- Probe size: `50` inherited-weight `supernet_v2` subnets
- FC2 validation coverage: full split, `640` samples
- Sintel coverage: full `MPI_Sintel_Final_train_list.txt`, `1041` samples
- BN recalibration: fixed `16` FC2-train batches per arch

## Measured Agreement

- `Spearman = 0.8996`
- `Kendall tau = 0.7339`
- `top-5 overlap = 3/5 = 0.6`
- `top-10 overlap = 8/10 = 0.8`
- mean absolute rank shift: `4.88`
- median absolute rank shift: `4`
- largest rank shift:
  - `arch_code = 2,1,2,0,2,0,0,1,1,0,1`
  - `FC2 rank = 29`
  - `Sintel rank = 46`
  - `rank delta = +17`

## What This Means

- `FC2 val` is a strong coarse proxy for `Sintel` under inherited-weight evaluation.
- `FC2 val` is not a reliable final-tiebreak metric for the very best subnets.
- The diagnostic therefore supports keeping `FC2` inside the search loop for throughput, but not as the only final ranking signal.

## Why FC2 Should Still Be Kept

- `FC2` is dramatically cheaper than full-list `Sintel` evaluation in the current implementation.
- `FC2` evaluation is batchable over the full validation split.
- `Sintel` evaluation is currently sample-wise, uses larger `416x1024` crops, and includes heavier per-sample preprocessing/postprocessing.
- Because of this cost gap, replacing FC2 entirely with Sintel would reduce search throughput sharply.

## Search V2 Design Decision

- Recommended policy: `FC2 + hardware` for stage-1 coarse search, then full-list `Sintel` for shortlist re-ranking.
- This is justified by the probe result:
  - all `Sintel top-10` probe architectures were inside `FC2 top-20`
  - `FC2 top-15` already covered `9/10` of the `Sintel top-10`
- Therefore the second search round does not need full Sintel evaluation for every candidate, but it should include Sintel before final subnet selection.

## Concrete Implication For Search V2

- Do not run a pure `FC2-only` final selection pipeline.
- Do not run full-list `Sintel` for every explored candidate.
- Use a two-stage shortlist strategy:
  - broad exploration on `FC2 + hardware`
  - periodic or end-of-round `Sintel` re-scoring on the top `15-20` FC2 candidates

## Reassessment Of Search-V1 Collapse

- The first search round should not be summarized as simple mechanism failure.
- `best_epe` stabilized extremely early, which supports the idea that the system entered a strong convergence region fast.
- But the late phase is not explained by convergence alone:
  - duplicate proposal ratio rose sharply in the middle stage
  - the search kept adding Pareto points after the best-EPE stopped improving
  - this means the loop still had useful frontier-completion work left, but was exploring inefficiently

## Updated Interpretation

- `search_v1` was likely effective in the early phase.
- The late-stage collapse is currently best interpreted as:
  - near-convergence on the strongest low-EPE region
  - plus over-frequent scientist reflection
  - plus overly self-reinforcing strategist/generator behavior inside a small search space

## Parameter-Level Decision Before Mechanism Rewrite

- A full redesign is not yet justified as the first next step.
- The next prudent move is a `search_v1.5` rerun with lighter scheduling changes.

## Agreed Search-V1.5 Settings

- Keep `batch_size = 20`
- Change `scientist_trigger_interval = 5`
- Keep `assumption_confidence_threshold = 0.95`

## Current Temperature Settings

- From `configs/search_v1.yaml`:
  - `agent_a = 0.35`
  - `agent_b = 0.40`
  - `agent_c = 0.15`
  - `agent_d1 = 0.45`
  - `agent_d2 = 0.15`
  - `agent_d3 = 0.25`

## Temperature Decision For The Next Rerun

- The current strategist and generator temperatures are materially below the `0.6` setting the user observed in the CoLLM temperature ablation.
- For the first parameter-only rerun, the cleanest test is:
  - raise `agent_a` to `0.60`
  - raise `agent_b` to `0.60`
  - leave the remaining agent temperatures unchanged

## Prompt Review Findings For V2

- The first real prompt problem was not the existence of the agent team, but stale V1 search-space language.
- The shared worldview needed to explicitly describe:
  - the V2 `11D` code
  - the `3^6 * 2^5 = 23328` search space
  - the concrete block semantics for `E0/E1/EB0/EB1/DB0/DB1/H0Out/H1/H1Out/H2/H2Out`
- The public wording "you only need to output architecture codes" was wrong because it only applied to `Agent B`, not the whole team.
- The public wording about "超物理纲常" was also wrong; the real constraint is simply "do not propose changes outside the current discrete search round."

## Agent-A Specific Finding

- The original `Agent A` design gave `assumptions.json` too much structural weight.
- The problem was not that A could see assumptions at all.
- The real problem was that assumption-testing was encoded as a default budget category alongside exploration, which risks anchoring the strategist toward scientist guesses even when the historical Pareto evidence points elsewhere.
- The prompt has now been corrected so that:
  - `assumptions` are weak signals
  - `findings` and historical search evidence are strong signals
  - verification budget is optional
  - verification budget is capped at `30%`
  - zero-budget validation is allowed when no assumption deserves active testing

## Agent-B Specific Finding

- `Agent B` needed one explicit hard rule even if the model could often infer it:
  - generated candidates must not duplicate history
  - generated candidates must also not duplicate each other within the same batch
- This matters because the engine currently deduplicates against history, not against within-batch repeats.

## Agent-D3 Specific Finding

- `Agent D3` still contained a V1-style example that used an invalid V2 head value (`2`).
- This was a real semantic error, not just wording.
- The example has been replaced with a V2-valid `0/1` head-choice example.

## Runtime Readiness Finding

- Prompt alignment alone is not enough to run `search_v2`.
- The current search runtime still contains V1 evaluation assumptions:
  - `configs/search_v1.yaml` still routes to `run_supernet_subnet_distribution.py`
  - that evaluation stack is still the old 9-block subnet-distribution path
  - `SearchCoordinator._record_epoch_metrics()` still uses `3 ** 9` rather than `23328`
- Therefore the next agentic V2 run still requires a dedicated V2 runtime config and a V2-capable evaluation path.

## Why Only A/B Temperature Should Move First

- `agent_a` controls strategic diversity and willingness to leave local habits.
- `agent_b` controls candidate diversity under the given strategy.
- `agent_c` and `agent_d2` are precision-oriented roles, so raising their temperature would mainly add noise.
- `agent_d1` and `agent_d3` can be revisited later if the `v1.5` rerun still shows premature narrative lock-in.
