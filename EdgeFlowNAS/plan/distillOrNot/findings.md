# Findings: Distill-Or-Not Same-FPS Probe

## Experimental Scope Correction

The original 10-subnet selection used Pareto/front-rank disagreement. That selection is useful for a broad Pareto-disagreement probe, but it is not the cleanest test of whether distill or no-distill gives a better EPE rank signal.

The corrected experiment fixes the hardware niche first:

- use common architecture codes across both V3 NSGA-II runs
- restrict to a narrow FPS range
- compare only EPE ranks inside that same FPS window

This better matches the research question:

> Under similar deployment speed, which supernet ranks architecture quality more faithfully?

## Current Candidate Set

Current source:

`plan/distillOrNot/same_fps_rank_gap_top8.csv`

External analysis output:

`outputs/nsga2_v3/same_fps_rank_gap_probe_20260430/`

Selection summary:

- common architectures: 154
- FPS window: `6.589863 - 6.895185`
- selected FPS range: `6.597428 - 6.895185`
- selected relative FPS span: `4.45%`
- selected within-window EPE rank-gap range: `8 - 10`

## Multi-GPU Design Finding

The implemented launcher uses one child process per architecture. Four child processes run concurrently on four GPUs.

This means batches are not physically shared across candidates. Each process constructs its own FC2 provider and prefetch queue.

Why this design:

- TensorFlow 1 multi-model single-graph training is easier when everything stays on one GPU, but less clean across multiple GPUs.
- One process per architecture gives robust GPU isolation through `CUDA_VISIBLE_DEVICES`.
- A failed candidate does not kill checkpoints for every candidate in a shared graph.
- Per-candidate logs and checkpoints are simpler.
- GPU memory is bounded by one fixed subnet graph, not a shared graph containing many subnet graphs.

Trade-off:

- The FC2 batches are loaded independently by each process, so disk/CPU load is duplicated.
- With the current 4-GPU Slurm template, `fc2_num_workers=4` gives about 16 train loader workers total. This is intentional for `--cpus-per-task=24`.

Common-randomness note:

- All processes currently receive the same default seed unless overridden.
- That gives comparable initialization and data-order intent, but separate multiprocessing data loaders may still make exact batch identity an implementation detail.
- If the final comparison requires strict common batches, a future one-GPU shared-batch runner can be added, but it would be slower and is not necessary for this rank-calibration probe.

## Vela/Fix-Subnet Verification

The fixed V3 subnet graph was checked with the local `vela` environment.

For reference arch:

`2,1,0,0,0,0,1,1,1,1,1`

`FixedArchModelV3` export produced:

- SRAM: `1.353515625 MB`
- inference time: `147.233625 ms`
- FPS: `6.791926776`

This matches the previously validated fixed V3 reference and confirms the new fixed model is not exporting the full mixed-branch supernet graph.
