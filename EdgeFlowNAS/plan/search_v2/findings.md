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

## Decision Metrics

- Spearman rank correlation
- Kendall tau
- Top-k overlap (`k=5,10`)
- Largest rank inversion case

## Interpretation Thresholds

- `Spearman >= 0.8`: FC2 rank is a strong proxy
- `0.5 <= Spearman < 0.8`: proxy is partial, mixed evaluation is needed
- `Spearman < 0.5`: FC2 rank is not reliable enough for search_v2
