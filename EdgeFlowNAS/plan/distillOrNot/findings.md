# Findings: Distill-Or-Not Short Retrain Probe

## Candidate Source

The selected 10 candidates are from:

`D:/Dataset/MCUFlowNet/EdgeFlowNAS/outputs/nsga2_v3/frontier_top5_rank_gap_probe_20260430/top10.csv`

Selection rule:

- common architecture evaluated by both V3 NSGA-II runs
- at least one run has `front_rank <= 5`
- largest cross-run `front_rank_gap`
- tie-break by `dominance_count_gap`, then `rank_score_gap`
- no cross-supernet absolute EPE comparison

## Existing Code Surface

### Not directly usable as-is

- `wrappers/run_retrain_v2_fc2.py`
  - supports multiple `arch_codes`
  - but imports `FixedArchModelV2`
  - warm-start mapping uses `MultiScaleResNetSupernetV2`
  - parses V2 arch semantics through `arch_codec_v2`
  - therefore unsafe for V3 rank probe without a V3-specific adaptation

- `configs/retrain_v2_fc2.yaml`
  - uses `input_height=352`, `input_width=480`
  - LR default is `5e-5 -> 1e-6`
  - not aligned with current V3 supernet resolution or requested LR consistency

- `wrappers/run_ablation_v1_fc2.py`
  - already has FC2/Sintel validation and useful logging
  - but trains named ablation variants, not arbitrary 11D V3 architecture codes

### Useful reusable parts

- `efnas.engine.retrain_v2_trainer`
  - multi-scale uncertainty loss
  - FC2 train/eval loop
  - checkpoint saving
  - FC2 comparison CSV
  - Sintel validation integration

- `efnas.engine.ablation_v1_trainer`
  - has validation progress bars
  - has Sintel best checkpoint logic
  - has gradient statistics and richer logs
  - can serve as the better pattern for a new short-probe trainer

- `efnas.nas.supernet_subnet_distribution_v3`
  - contains a selected-only `_FixedSubnetForExportV3`
  - useful as a reference for V3 hard-routed architecture semantics
  - should not be imported as a private training model directly; better to move or duplicate into a proper network module

- `efnas.data.dataloader_builder`
  - already supports `fc2_num_workers` and `fc2_eval_num_workers`

- `efnas.data.prefetch_provider.PrefetchBatchProvider`
  - can be reused to provide bounded CPU prefetch for train/eval providers

## Key Technical Finding

The requested experiment should not train all 10 subnets inside one TensorFlow graph on one GPU. That would be memory-heavy, difficult to parallelize cleanly across 5 GPUs, and would couple failures.

The safer design is:

- one child process per architecture
- each process sees exactly one GPU via `CUDA_VISIBLE_DEVICES`
- a launcher keeps 5 child processes active until all 10 complete

This matches the successful NSGA-style multi-GPU pattern already used in this project.

## Initialization Finding

The initialization policy materially affects the interpretation:

- If warm-started from no-distill, no-distill may look better because the weights are from that supernet.
- If warm-started from distill, distill may get the same advantage.
- If training from scratch, the comparison becomes a cleaner test of architecture rank predictiveness.

Therefore common random initialization is the recommended default unless the research question is explicitly about inherited-weight fine-tuning quality.

