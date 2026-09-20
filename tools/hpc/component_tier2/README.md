# COMP-ABL-01 Tier2 deployment

Training stays pinned to 680eba5c21a234c6fffbf8fe77082433b3d87435. These deployment
scripts add wICE/A100 scheduling and storage only; they do not change the model,
loss, data recipe, prefetch=1, seed matrix, or 400-epoch LR horizon.

Copy these scripts into the isolated DATA root's control directory. The existing
legacy checkout at DATA/test/MCUFlowNet is not modified. Code releases, container,
environment and control logs use DATA/MCUFlowNet-component; dataset reads use the
existing SCRATCH/dataset. Results use SCRATCH/MCUFlowNet-component/runs.

Submit bootstrap to a CPU batch node, then the five probe indices 0-4 on A100
with afterok dependency, then the release script on one CPU after all probes.
Explicitly select --clusters=wice --account=lp_embaivision; GPU tasks use one
A100 and 18 CPU cores. Site limits: https://docs.vscentrum.be/leuven/slurm_specifics.html
(checked 2026-09-20). All GPU tasks stay within the 72-hour limit.

Before release, supply sofia-probe-fingerprints.json from the existing probe
and sofia-migration-release.json proving that only pending indices 2-14 were
cancelled on Sofia. Keep the running A0/A1 seed42 there. Release checks complete
FC2 and 845-pair Sintel decoding, matching dataset lists and cross-site input/
label fingerprints, all five exact within-device recovery comparisons, and
current credit budget. Failed checks stop submission. Partial submission writes
its job IDs immediately and must be inspected before any retry.

The release job submits all 13 migrated runs (array2-14, max4 concurrent) after
acceptance. The measured conservative wall estimate determines a single400 or
200+200 epoch split; the latter resumes all state and retains the400 horizon.
The second array waits for the entire first array; failed jobs do not silently
advance. Budget quoting includes all requested slots and keeps 5% of current
available credits unused. Maximum wall time beyond two segments is rejected.

A100 and H200 need not produce bit-identical trajectories. Seed42 has mixed
hardware, while seeds43/44 cover all five variants on A100; report this explicitly
and inspect the same-hardware repeats separately. The audit does not decode the
196 held-out pairs and makes no claim about FT3D integrity.

Run the five release-safety checks with python test-tier2-release.py. They use
temporary fixtures and mocked submission calls; no jobs are submitted by tests.
When production A100s are occupied, tier2-debug-probes.sh runs all five probes
sequentially in one <=1h gpu_a100_debug allocation. That card is A100 80GB PCIe;
production is A100 80GB SXM. Treat its timings as a planning estimate with the
configured margin, not a production speed measurement. Point release dependency
to that debug job, and cancel unused pending probe/release jobs before switching.
