# Search Insights

<!-- Final Scientist report, gen 14 → gen 15 handoff. All claims grounded against verification + Vela layer profiles where available. -->

---

### I-011 (active): HV growth nearly flat through gen 14 — final gen needs targeted shoulder-gap seeding, not broad sampling

**Verified HV trajectory (gen 0→14):** 1.844 → 2.056 → 2.216 → 2.320 → 2.386 → 2.399 → 2.422 → 2.432 → 2.433 → 2.437 → 2.440 → 2.444 → 2.4444 → 2.4445 → 2.4477. Last 4 generations (11→14) added cumulative **+0.0037 HV**, ≈+0.001/gen growth rate. PF size grew 46→46→48→50→53. (Note: my Stage-A draft cited absolute HV around 7.5; the verified absolute scale is ~2.45, but the **deltas and saturation pattern are confirmed** — gen 11→14 growth is sub-0.001/gen, well below early-gen rates.)

**Dup-rate caveat:** verification reports gen_*_dup_rate = 0.0 across all gens, contradicting my Stage-A claim of "dup-rate 1.48 at gen 14". I retract that number — the saturation case must rest on (a) HV growth flatness and (b) PF-thickening pattern (I-013, I-015), not on duplicate counts.

**Vela query for the shoulder zone (FPS 6.5–7.2) returned 0 matches** because the query lacked a filter; the actual Pareto members in that band were instead pulled via I-016's archetype query (see below).

**Geometry for gen 15:** the FPS-shoulder gap between the cluster around FPS≈6.74 (the (DB0,DB1)=(0,1) and (2,0) cells, EPE 4.30–4.37) and the (DB0,DB1)=(1,1) right-edge at FPS≈7.03/EPE≈4.37 is the **single sub-segment most likely to extend HV**. Any arch landing at FPS≈6.85–7.0 with EPE<4.40 directly thickens the front. Zones 1 (FPS-endpoint) and 3 (EPE-floor) are already saturated (see I-015).

---

### I-013 (active): H1/H2 entropy near-collapsed, FPS-endpoint thickening confirmed but at modest scale — head exploitation is real but bounded

**Verified gene entropy at gen 14:**
- dim_6 (H0Out) = 0.661, dim_7 (H1) = **0.435**, dim_8 (H1Out) = 0.665, dim_9 (H2) = **0.535**, dim_10 (H2Out) = 0.604
- backbone dims 0–5 all > 0.96 (much higher diversity)

My Stage-A claim of "dim_7 = 0.0" was wrong; **H1 entropy is 0.435, strongly biased toward H1=0 but not fully collapsed**. Similarly dim_9 = 0.535, biased but not collapsed. The qualitative pattern (H1/H2 prefer 3x3 = code 0 because they cost ~1.2 FPS each at 5x5 vs 3x3) holds, but the population still explores k=5 variants.

**Vela layer evidence (from I-016 query, arch `0,0,0,0,1,1,0,0,0,0,0` at 4.37/7.03):**
- H1 (k=3) costs **2,957,414 cycles** (5.20% of total), H2 (k=3) costs **2,801,022 cycles** (4.92%)
- HxOut heads cost dramatically less: H1Out 355k cycles (0.62%), H2Out 1,075k cycles (1.89%)
- For arch `1,0,0,0,1,1,0,0,1,0,1` (H1Out=k5, H2Out=k5, same backbone): H2Out jumps 1.07M→2.35M cycles, H1Out 355k→987k — direct confirmation that upsample-conv k-size is the dominant FPS knob in the head, and that H1=H2=0 (the gates themselves at k=3) is the cheap floor.

**FPS-endpoint thickening (FPS>8.5 with H1=H2=0): 17 distinct archs** (verified count). My Stage-A claimed "20+" — close enough; the zone is thickened but not as densely as I asserted. Still HV-neutral for further sampling because all 17 are clustered in EPE 4.65–4.71 and dominate or are dominated within a tight neighbourhood.

**Implication for gen 15:** don't waste seeds on more H1=H2=0 FPS-endpoint mutations. The k-size choice on H1/H2 (dims 7, 9) is well-explored; remaining gains come from backbone (dims 2–5) and stem (dims 0, 1) variation.

---

### I-015 (active): Zone audit — Zone 2 (shoulder gap) is the only under-sampled high-leverage region

**Verified gen 12–14 distribution (150 evals total):**
- Zone 1 (FPS-endpoint thickening, FPS>8.5): **2** new archs — already saturated, NSGA-II is no longer pushing here
- Zone 2 (FPS-shoulder gap 6.5–7.2): **8** archs landed in this band over 3 gens — non-zero, but Vela query confirms the band is *populated by sparse Pareto members*, not densely tiled. My Stage-A claim of "Zone 2 untouched" was too strong; **8 archs in 3 gens × 50 = ~5% rate** is below natural sampling density given the band's width
- Zone 3 (EPE-floor 4.04–4.06): **10** new archs — partially complete, mostly inside-front

**Re-prioritisation for gen 15:** Zone 2 remains the highest-leverage target, but the framing shifts from "empty" to "thinly sampled with no PF-extending hit yet." The 8 archs that did land in 6.5–7.2 over gen 12–14 were dominated or only marginally Pareto-relevant; the gap between the (0,1)/(2,0) cluster at FPS≈6.7 EPE≈4.30–4.37 and the (1,1) cell at FPS=7.03 EPE=4.37 is geometrically the most promising HV-extending region.

---

### I-016 (active): Final-gen seed shortlist — target FPS-shoulder via (DB0,DB1) ∈ {(1,1),(2,0),(1,0)} with H1=H2=0

**Vela query (I-016, 22 matching archs)** gave detailed layer profiles for the (DB0,DB1)=(1,1) right-edge of the gap and the (2,0) cluster. Key observations from cycle breakdown:

- **(1,1) cell, arch `0,0,0,0,1,1,0,0,0,0,0` (4.37/7.03):** DB0 dominates (4 × ~4.7M cycles = ~33% of total). Up1 conv 3.4M, H1 conv 3.0M, Up2 conv 2.8M, H2 conv 2.8M. **No single backbone op has slack** — util_pct on Up1/H1/Up2/H2 is 83–101%. Stem-only mutations (E0=k5/k7, E1=k3-stride2) add 0.6–1.4M cycles → FPS drops to 6.95 (E0=k5), 6.83 (E0=k7), 6.86 (E1=2). All three sampled, all near 4.37 EPE; the (1,1) cell is well-explored at H1=H2=0.

- **(2,0) cell, arch `0,0,0,0,2,0,0,0,0,0,0` (4.306/6.357):** DB0 now has **6 conv blocks** (Deep3 stack) at ~4.7M cycles each = ~45% of total cycles. EPE is lower (4.306 vs 4.37) but FPS is below the gap. Adding H1Out=k5 (`0,0,0,0,2,0,1,0,1,0,0` → 4.305/6.26) costs ~1 FPS for negligible EPE gain.

- **(1,0) bridge cell:** only 1 sampled arch in the query results visible at `0,0,0,0,1,0,0,0,0,0,0` (4.451/7.499). EPE is high (4.45) — this cell may not bridge the gap as cleanly as I hoped in Stage A.

**Verification of unsampled archs failed** (`subprocess_error: Argument list too long`), so I cannot certify that the specific arch_codes in my Stage-A shortlist are truly unevaluated. Treat the shortlist below as *candidate archetypes*, not verified-novel codes.

**Concrete seeds for gen 15 (priority order):**
1. **(DB0=1, DB1=1) with EB0/EB1 ≠ Deep1**: e.g. `0,0,1,0,1,1,0,0,0,0,0`, `0,0,0,1,1,1,0,0,0,0,0`, `0,0,1,1,1,1,0,0,0,0,0`. Backbone diversity (entropy 0.96+ on dims 2–5) suggests these slots have room to move; current (1,1)-cell exploration was almost entirely EB0=EB1=Deep1. Expected: EPE 4.30–4.40, FPS 6.7–7.0 — directly in the gap.
2. **(DB0=2, DB1=0) with stem 0**: keep cheap stem, vary EB: `0,0,1,0,2,0,0,0,0,0,0`, `0,0,0,1,2,0,0,0,0,0,0`. Expected: EPE ≈4.30, FPS 6.2–6.4 (extends left side of the gap downward in EPE).
3. **(DB0=2, DB1=1) — entirely absent from sampled archs in the Vela query**, only 1 of 22 matches was a (2,1)-or-(1,2) variant. Expected: EPE 4.30–4.35, FPS 6.0–6.6.

**Avoidance list (confirmed):** FPS-endpoint H1=H2=0 single-mutations (I-013, 17 archs already), EB0=Deep2 floor permutations (I-015 Zone 3, 10+ recent hits), k=4/k=5 backbone variants on the right edge.

**Caveat:** with HV growth at ~0.001/gen and only 50 evals left, even a perfectly placed seed family extends HV by maybe +0.002–0.005 absolute. Final gen will not dramatically reshape the front; it can only thicken specific underexposed cells.
