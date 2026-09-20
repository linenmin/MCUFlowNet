# COMP-ABL-01: component ablation

Five macro-backbone variants (A0 deconv, A1 bilinear, A2 ECA, A4 gate, A3 full),
each from scratch with seeds42/43/44. They are not NAS-selected S/L subnets.
Use `tools/hpc/run_component_ablation.py --index 0..14 --mode probe|train|resume`.
Indices0..4 use seed42, 5..9 seed43, 10..14 seed44. This experiment is isolated
under `/runs/COMP-ABL-01`; no historical checkpoint initialization.

Shared retrain loop uses the explicit `component_variant` model factory, original
multiscale uncertainty loss, batch32, Adam, FC2 clip50 training, 400-epoch cosine
1e-4 to1e-6, prefetch1 and eight loader threads. At22232 samples this is695
updates per epoch (last full batch wraps eight samples), 278000 updates total.
The schedule horizon is independent of the run's stop_after_epoch. Disabling
early stopping ensures an equal budget; seeds42/43/44 cover every combination.

FC2 full640 validation is recorded every epoch with raw and clip50 GT EPE from
the same prediction (legacy FC2 sqrt epsilon1e-6 is preserved). Sintel Final845
monitor runs every5 epochs,416x1024 center crop, both raw/clip50, raw selects
best. No196-pair holdout evaluation. Per-pair error sums/counts are retained,
including large-motion grouping. FC2 clip50 best, FC2 raw best, Sintel raw best,
last and epochs130/200/300/400 are retained separately.

Run manifests include model config, data list hashes and software version.
Parent control manifests record the pinned code commit/job and exact config.
At epoch boundaries prefetch pauses and rolls back unconsumed state before RNG
is saved. Resume restores full model/BN/Adam and consumed-data RNG; it is not
mid-epoch continuation. On interrupted checkpoint writing, inspect consistency
before resuming. Shared layers have the same initialization distribution, but
not guaranteed identical tensor values across different model graphs.

Probe executes deterministic 50+50 and continuous100 updates, all five graphs,
full640 FC2 and845 Sintel checks, exact checkpoint tensor/RNG comparisons and
reloads each selection type. Probe checkpoints are never used to initialize
formal runs. Regression suite: `python -m unittest discover -s tools/validation`.
Normal production does not request deterministic GPU kernels; probe timing is
therefore a conservative engineering estimate, not a speed guarantee.

Preflight code review (2026-09-20) covers model factory, loss/update parity,
providers/prefetch, RNG and epoch recovery, dual metrics, checkpoint selection,
portable inference, configuration and Slurm entry points. The23-test suite
includes all five original-vs-shared graphs with identical initial model/BN
values, matching loss and one-step updated tensors; corrupt history/step
recovery is rejected. Standalone evaluation also disables TF32. Verification
mode copies probe outputs, resumes at the completed boundary without updates,
and recomputes76 fixed pairs against the saved per-pair results; it must load
the copy rather than the stale absolute path inside checkpoint metadata.

Historical distinction: the old ablation trainer evaluated FC2 with
`is_training=True`. This campaign uses inference BN statistics for FC2 and
Sintel. Consequently the new FC2 curve is not an exact reproduction of the
historical protocol even when the GT clipping is the same. Model family is
identified by`component_variant`; the legacy11-zero`arch_code` field is unused
for component construction and does not mean the S subnet is trained.

Sofia array wrapper: `tools/hpc/component_array.sh PROJECT PINNED_CODE MODE`.
Use explicit account/partition,1GPU,24CPU,no memory override. Cap arrays at2
concurrent jobs. Submit first seed then the remaining seeds with afterok on the
first array. Job manifests live under`COMP-ABL-01/control/<mode>-<index>/` and
can be registered with the existing run-index builder. Monitor actual early
epoch times before trusting the full-run duration estimate.
