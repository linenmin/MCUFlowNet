"""Analyze subnet distribution and export data files only."""

import argparse
import csv
import itertools
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from code.utils.json_io import write_json

NUM_BLOCKS = 9
ARCH_SPACE_SIZE = 3 ** NUM_BLOCKS
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _iter_all_arch_codes(num_blocks: int = NUM_BLOCKS) -> Iterable[List[int]]:
    """Iterate full architecture search space."""
    for code in itertools.product((0, 1, 2), repeat=num_blocks):
        yield [int(item) for item in code]


def _dedup_codes(codes: Sequence[Sequence[int]]) -> List[List[int]]:
    """Keep first occurrence order while dropping duplicates."""
    seen = set()
    out: List[List[int]] = []
    for code in codes:
        key = tuple(int(item) for item in code)
        if key in seen:
            continue
        seen.add(key)
        out.append([int(item) for item in code])
    return out


def sample_arch_pool(
    num_arch_samples: int,
    seed: int,
    include_eval_pool: bool,
    eval_pool: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Sample unique architecture codes from 3^9 space."""
    sample_size = int(num_arch_samples)
    if sample_size <= 0:
        raise ValueError("num_arch_samples must be > 0")

    if sample_size >= ARCH_SPACE_SIZE:
        return list(_iter_all_arch_codes())

    rng = random.Random(int(seed))
    pool: List[List[int]] = []
    if include_eval_pool:
        pool.extend(_dedup_codes(eval_pool))
    if len(pool) >= sample_size:
        return pool[:sample_size]

    seen = {tuple(code) for code in pool}
    while len(pool) < sample_size:
        candidate = [rng.randint(0, 2) for _ in range(NUM_BLOCKS)]
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        pool.append(candidate)
    return pool


def compute_complexity_scores(arch_code: Sequence[int]) -> Dict[str, float]:
    """Compute unified-direction complexity proxy scores."""
    if len(arch_code) != NUM_BLOCKS:
        raise ValueError("arch_code length must be 9")
    code = [int(item) for item in arch_code]
    depth_score = float(sum(code[:4]))  # 前四位越大表示骨干越重。
    kernel_light_score = float(sum(code[4:]))  # 后五位越大表示头部越轻。
    kernel_heavy_score = float(sum(2 - item for item in code[4:]))  # 统一成越大越重方向。
    total_score = float(depth_score + kernel_heavy_score)  # 总复杂度代理分数。
    return {
        "depth_score": depth_score,
        "kernel_light_score": kernel_light_score,
        "kernel_heavy_score": kernel_heavy_score,
        "complexity_score": total_score,
    }


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float with best effort."""
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if np.isfinite(number):
        return float(number)
    return None


def _safe_fps(inference_ms: Optional[float]) -> Optional[float]:
    """Convert inference time(ms) to FPS."""
    time_ms = _safe_float(inference_ms)
    if time_ms is None or time_ms <= 0.0:
        return None
    return float(1000.0 / time_ms)


def _compute_metric_summary(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Compute robust summary for one metric."""
    cleaned = [float(item) for item in values if _safe_float(item) is not None]
    total_count = int(len(values))
    valid_count = int(len(cleaned))
    if valid_count <= 0:
        return {
            "count_total": total_count,
            "count_valid": 0,
            "count_invalid": total_count,
        }
    arr = np.asarray(cleaned, dtype=np.float64)
    summary: Dict[str, Any] = {
        "count_total": total_count,
        "count_valid": valid_count,
        "count_invalid": int(total_count - valid_count),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }
    for pct in PERCENTILES:
        summary[f"p{pct:02d}"] = float(np.percentile(arr, pct))
    return summary


def _to_arch_text(arch_code: Sequence[int]) -> str:
    """Format arch code for CSV/export."""
    return ",".join(str(int(item)) for item in arch_code)


def _save_records_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Save sampled subnet records to CSV."""
    fields = [
        "sample_index",
        "arch_code",
        "epe",
        "depth_score",
        "kernel_light_score",
        "kernel_heavy_score",
        "complexity_score",
        "sram_peak_mb",
        "inference_ms",
        "fps",
        "vela_status",
        "vela_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in fields})


def _save_ranking_csv(path: Path, ranking: Sequence[Dict[str, Any]]) -> None:
    """Save sorted ranking by EPE to CSV."""
    fields = ["rank", "sample_index", "arch_code", "epe", "complexity_score"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ranking:
            writer.writerow({key: row.get(key, "") for key in fields})


def _save_vela_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Save Vela-only metrics to CSV."""
    fields = [
        "sample_index",
        "arch_code",
        "sram_peak_mb",
        "inference_ms",
        "fps",
        "vela_status",
        "vela_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in fields})


def _evaluate_arch_pool(
    config: Dict[str, Any],
    checkpoint_prefix: Path,
    arch_pool: List[List[int]],
    bn_recal_batches: int,
    batch_size: int,
    eval_batches_per_arch: int,
    num_workers: int,
    cpu_only: bool,
) -> Dict[str, Any]:
    """Evaluate sampled subnets and return per-arch EPE."""
    from code.data.dataloader_builder import build_fc2_provider
    from code.nas.supernet_eval import (
        _build_eval_graph,
        _run_eval_pool,
        _run_eval_pool_parallel,
    )
    import tensorflow as tf

    workers_used = min(max(1, int(num_workers)), max(1, len(arch_pool)))
    if workers_used <= 1:
        if cpu_only:
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        graph_obj = _build_eval_graph(config=config, batch_size=batch_size)
        train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="eval")
        val_provider = build_fc2_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
        if len(train_provider) == 0:
            raise RuntimeError(f"train sample count is 0; source={train_provider.source_dir}")
        if len(val_provider) == 0:
            raise RuntimeError(f"val sample count is 0; source={val_provider.source_dir}")
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_obj["saver"].restore(sess, str(checkpoint_prefix))
            metrics = _run_eval_pool(
                sess=sess,
                graph_obj=graph_obj,
                train_provider=train_provider,
                val_provider=val_provider,
                eval_pool=arch_pool,
                bn_recal_batches=int(bn_recal_batches),
                batch_size=int(batch_size),
                eval_batches_per_arch=int(eval_batches_per_arch),
            )
        metrics["num_workers_used"] = 1
        return metrics

    metrics = _run_eval_pool_parallel(
        config=config,
        checkpoint_prefix=checkpoint_prefix,
        eval_pool=arch_pool,
        bn_recal_batches=int(bn_recal_batches),
        batch_size=int(batch_size),
        eval_batches_per_arch=int(eval_batches_per_arch),
        num_workers=int(workers_used),
        cpu_only=bool(cpu_only),
    )
    metrics["num_workers_used"] = int(workers_used)
    return metrics


def _build_tflite_for_arch(
    config: Dict[str, Any],
    checkpoint_prefix: Path,
    arch_code: Sequence[int],
    tflite_path: Path,
    rep_dataset_samples: int,
    quantize_int8: bool,
) -> None:
    """Export one fixed-arch subnet to TFLite."""
    import tensorflow as tf

    from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet

    data_cfg = config.get("data", {})
    input_h = int(data_cfg.get("input_height", 180))
    input_w = int(data_cfg.get("input_width", 240))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = int(flow_channels * 2)

    tf.compat.v1.disable_eager_execution()  # 保持 TF1 图模式一致。
    graph = tf.Graph()
    with graph.as_default():
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_h, input_w, 6], name="Input")
        arch_const = tf.constant([int(v) for v in arch_code], dtype=tf.int32, name="ArchCodeConst")
        is_training_const = tf.constant(False, dtype=tf.bool, name="IsTrainingConst")
        model = MultiScaleResNetSupernet(
            input_ph=input_ph,
            arch_code_ph=arch_const,
            is_training_ph=is_training_const,
            num_out=pred_channels,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()
        output_tensor = preds[-1]
        saver = tf.compat.v1.train.Saver(max_to_keep=1)

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, str(checkpoint_prefix))
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [output_tensor])
            if quantize_int8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                rng = np.random.default_rng(seed=2026)

                def representative_dataset_gen():
                    for _ in range(max(1, int(rep_dataset_samples))):
                        sample = rng.random((1, input_h, input_w, 6)).astype(np.float32)
                        yield [sample]

                converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def _run_vela_for_arch(
    tflite_path: Path,
    output_dir: Path,
    vela_mode: str,
    vela_optimise: str,
    vela_silent: bool,
) -> Dict[str, Any]:
    """Run Vela and return parsed metrics."""
    try:
        from tests.vela.vela_compiler import run_vela
    except Exception as exc:
        raise ImportError(
            "failed to import tests.vela.vela_compiler.run_vela; "
            "please ensure EdgeFlowNAS/tests/vela exists"
        ) from exc

    sram_mb, inference_ms = run_vela(
        str(tflite_path),
        mode=str(vela_mode),
        output_dir=str(output_dir),
        optimise=str(vela_optimise),
        silent=bool(vela_silent),
    )
    sram_peak_mb = _safe_float(sram_mb)
    inference_ms = _safe_float(inference_ms)
    fps = _safe_fps(inference_ms)
    status = "ok" if sram_peak_mb is not None and inference_ms is not None and fps is not None else "fail"
    error_text = "" if status == "ok" else "missing_vela_metrics"
    return {
        "sram_peak_mb": sram_peak_mb,
        "inference_ms": inference_ms,
        "fps": fps,
        "vela_status": status,
        "vela_error": error_text,
    }


def _collect_vela_metrics(
    config: Dict[str, Any],
    checkpoint_prefix: Path,
    arch_pool: List[List[int]],
    analysis_dir: Path,
    vela_mode: str,
    vela_optimise: str,
    vela_silent: bool,
    vela_limit: Optional[int],
    rep_dataset_samples: int,
    quantize_int8: bool,
    keep_artifacts: bool,
) -> Dict[int, Dict[str, Any]]:
    """Collect Vela metrics for sampled subnets."""
    results: Dict[int, Dict[str, Any]] = {}
    limit = len(arch_pool) if vela_limit is None else max(0, min(len(arch_pool), int(vela_limit)))
    vela_root = analysis_dir / "vela_tmp"
    vela_root.mkdir(parents=True, exist_ok=True)
    ok_count = 0
    fail_count = 0

    with tqdm(total=limit, desc="subnet vela", unit="arch") as progress:
        for sample_index, arch_code in enumerate(arch_pool[:limit]):
            arch_tag = f"arch_{sample_index:04d}"
            arch_dir = vela_root / arch_tag
            tflite_path = arch_dir / f"{arch_tag}.tflite"
            arch_dir.mkdir(parents=True, exist_ok=True)
            item = {
                "sram_peak_mb": None,
                "inference_ms": None,
                "fps": None,
                "vela_status": "fail",
                "vela_error": "",
            }
            try:
                _build_tflite_for_arch(
                    config=config,
                    checkpoint_prefix=checkpoint_prefix,
                    arch_code=arch_code,
                    tflite_path=tflite_path,
                    rep_dataset_samples=rep_dataset_samples,
                    quantize_int8=quantize_int8,
                )
                vela_item = _run_vela_for_arch(
                    tflite_path=tflite_path,
                    output_dir=arch_dir,
                    vela_mode=vela_mode,
                    vela_optimise=vela_optimise,
                    vela_silent=vela_silent,
                )
                item.update(vela_item)
            except Exception as exc:
                item["vela_status"] = "fail"
                item["vela_error"] = str(exc)
            if item["vela_status"] == "ok":
                ok_count += 1
            else:
                fail_count += 1
            results[int(sample_index)] = item
            if not keep_artifacts:
                shutil.rmtree(arch_dir, ignore_errors=True)
            progress.update(1)
            progress.set_postfix(ok=ok_count, fail=fail_count)

    if not keep_artifacts and vela_root.exists():
        shutil.rmtree(vela_root, ignore_errors=True)
    return results


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="analyze subnet distribution under best/last checkpoint")
    parser.add_argument("--config", required=True, help="path to supernet config yaml")
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type")
    parser.add_argument("--num_arch_samples", type=int, default=512, help="number of sampled subnets")
    parser.add_argument("--sample_seed", type=int, default=None, help="seed for subnet sampling")
    parser.add_argument("--exclude_eval_pool", action="store_true", help="exclude fixed eval_pool_12 from samples")

    parser.add_argument("--bn_recal_batches", type=int, default=None, help="override BN recalibration batches")
    parser.add_argument("--eval_batches_per_arch", type=int, default=None, help="override val eval batches per arch")
    parser.add_argument("--batch_size", type=int, default=None, help="override eval batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="parallel workers for arch eval")
    parser.add_argument("--cpu_only", action="store_true", help="force CPU-only eval")

    parser.add_argument("--enable_vela", action="store_true", help="enable Vela benchmark for sampled subnets")
    parser.add_argument("--vela_mode", choices=["basic", "verbose"], default="basic", help="Vela mode")
    parser.add_argument("--vela_optimise", choices=["Performance", "Size"], default="Size", help="Vela optimise policy")
    parser.add_argument("--vela_limit", type=int, default=None, help="limit Vela runs by sample count")
    parser.add_argument("--vela_rep_dataset_samples", type=int, default=3, help="rep-dataset samples for INT8 export")
    parser.add_argument("--vela_float32", action="store_true", help="export float32 TFLite instead of INT8")
    parser.add_argument("--vela_keep_artifacts", action="store_true", help="keep Vela temp folders and tflite files")
    parser.add_argument("--vela_verbose_log", action="store_true", help="disable Vela silent mode")

    parser.add_argument("--plots", default="", help="deprecated: plotting moved to separate script")
    parser.add_argument("--hist_bins", type=int, default=30, help="deprecated: plotting moved to separate script")
    parser.add_argument("--top_k", type=int, default=10, help="top/bottom K summary")
    parser.add_argument("--output_tag", default="", help="optional suffix for output folder")

    parser.add_argument("--experiment_name", default=None, help="override runtime.experiment_name")
    parser.add_argument("--base_path", default=None, help="override data.base_path")
    parser.add_argument("--train_dir", default=None, help="override data.train_dir")
    parser.add_argument("--val_dir", default=None, help="override data.val_dir")
    parser.add_argument("--train_batch_size", type=int, default=None, help="override train.batch_size in config")
    parser.add_argument("--seed", type=int, default=None, help="override runtime.seed")
    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    from code.nas.supernet_eval import (
        _apply_cli_overrides,
        _build_checkpoint_paths,
        _find_existing_checkpoint,
        _load_checkpoint_meta,
        _load_config,
        _load_or_build_eval_pool,
        _resolve_output_dir,
    )

    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    config = _apply_cli_overrides(config=_load_config(args.config), args=args)
    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    seed = int(runtime_cfg.get("seed", 42))

    pool_size = int(eval_cfg.get("eval_pool_size", 12))
    eval_pool_info = _load_or_build_eval_pool(
        output_dir=_resolve_output_dir(config=config),
        seed=seed,
        pool_size=pool_size,
    )
    include_eval_pool = not bool(args.exclude_eval_pool)
    sample_seed = int(args.sample_seed) if args.sample_seed is not None else int(seed + 9973)
    arch_pool = sample_arch_pool(
        num_arch_samples=int(args.num_arch_samples),
        seed=sample_seed,
        include_eval_pool=include_eval_pool,
        eval_pool=eval_pool_info["pool"],
    )

    output_root = _resolve_output_dir(config=config)
    ckpt_paths = _build_checkpoint_paths(experiment_dir=output_root)
    chosen = ckpt_paths[args.checkpoint_type]
    checkpoint_prefix = _find_existing_checkpoint(path_prefix=chosen)
    if checkpoint_prefix is None:
        raise RuntimeError(f"checkpoint not found: {chosen}")
    checkpoint_prefix = Path(checkpoint_prefix)

    bn_recal_batches = int(args.bn_recal_batches) if args.bn_recal_batches is not None else int(eval_cfg.get("bn_recal_batches", 8))
    eval_batches_per_arch = (
        int(args.eval_batches_per_arch) if args.eval_batches_per_arch is not None else int(eval_cfg.get("eval_batches_per_arch", 4))
    )
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 32))

    tag = str(args.output_tag).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.checkpoint_type}_{len(arch_pool)}_{stamp}" + (f"_{tag}" if tag else "")
    run_dir = output_root / "subnet_distribution" / folder_name
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics = _evaluate_arch_pool(
        config=config,
        checkpoint_prefix=checkpoint_prefix,
        arch_pool=arch_pool,
        bn_recal_batches=bn_recal_batches,
        batch_size=batch_size,
        eval_batches_per_arch=eval_batches_per_arch,
        num_workers=int(args.num_workers),
        cpu_only=bool(args.cpu_only),
    )

    records: List[Dict[str, Any]] = []
    per_arch_epe = [float(item) for item in metrics.get("per_arch_epe", [])]
    for idx, (arch_code, epe_val) in enumerate(zip(arch_pool, per_arch_epe)):
        score = compute_complexity_scores(arch_code=arch_code)
        records.append(
            {
                "sample_index": int(idx),
                "arch_code": _to_arch_text(arch_code),
                "epe": float(epe_val),
                "depth_score": float(score["depth_score"]),
                "kernel_light_score": float(score["kernel_light_score"]),
                "kernel_heavy_score": float(score["kernel_heavy_score"]),
                "complexity_score": float(score["complexity_score"]),
                "sram_peak_mb": None,
                "inference_ms": None,
                "fps": None,
                "vela_status": "skipped",
                "vela_error": "",
            }
        )

    if bool(args.enable_vela):
        vela_map = _collect_vela_metrics(
            config=config,
            checkpoint_prefix=checkpoint_prefix,
            arch_pool=arch_pool,
            analysis_dir=analysis_dir,
            vela_mode=str(args.vela_mode),
            vela_optimise=str(args.vela_optimise),
            vela_silent=not bool(args.vela_verbose_log),
            vela_limit=args.vela_limit,
            rep_dataset_samples=int(args.vela_rep_dataset_samples),
            quantize_int8=not bool(args.vela_float32),
            keep_artifacts=bool(args.vela_keep_artifacts),
        )
        for row in records:
            sample_index = int(row["sample_index"])
            if sample_index not in vela_map:
                row["vela_status"] = "skipped_limit"
                continue
            row.update(vela_map[sample_index])

    ranked = sorted(records, key=lambda item: (float(item["epe"]), int(item["sample_index"])))
    ranking_rows: List[Dict[str, Any]] = []
    for rank_idx, row in enumerate(ranked, start=1):
        ranking_rows.append(
            {
                "rank": int(rank_idx),
                "sample_index": int(row["sample_index"]),
                "arch_code": row["arch_code"],
                "epe": float(row["epe"]),
                "complexity_score": float(row["complexity_score"]),
            }
        )

    top_k = max(1, int(args.top_k))
    vela_ok_rows = [item for item in records if str(item.get("vela_status", "")) == "ok"]
    rank_fps = sorted(vela_ok_rows, key=lambda item: (-float(item["fps"]), int(item["sample_index"])))
    rank_sram = sorted(vela_ok_rows, key=lambda item: (float(item["sram_peak_mb"]), int(item["sample_index"])))
    rank_epe = ranked

    summary_payload: Dict[str, Any] = {
        "status": "ok",
        "checkpoint_type": args.checkpoint_type,
        "checkpoint_path": str(checkpoint_prefix),
        "checkpoint_meta": _load_checkpoint_meta(path_prefix=checkpoint_prefix),
        "num_arch_samples_requested": int(args.num_arch_samples),
        "num_arch_samples_used": int(len(arch_pool)),
        "sample_seed": int(sample_seed),
        "include_eval_pool": bool(include_eval_pool),
        "eval_pool_size": int(pool_size),
        "bn_recal_batches": int(bn_recal_batches),
        "eval_batches_per_arch": int(eval_batches_per_arch),
        "batch_size": int(batch_size),
        "num_workers_requested": int(args.num_workers),
        "num_workers_used": int(metrics.get("num_workers_used", 1)),
        "vela_enabled": bool(args.enable_vela),
        "vela_mode": str(args.vela_mode),
        "vela_optimise": str(args.vela_optimise),
        "vela_limit": args.vela_limit,
        "distribution": {
            "epe": _compute_metric_summary([item.get("epe") for item in records]),
            "fps": _compute_metric_summary([item.get("fps") for item in records]),
            "sram_peak_mb": _compute_metric_summary([item.get("sram_peak_mb") for item in records]),
            "inference_ms": _compute_metric_summary([item.get("inference_ms") for item in records]),
        },
        "vela_status_count": {
            "ok": int(sum(1 for item in records if item.get("vela_status") == "ok")),
            "fail": int(sum(1 for item in records if item.get("vela_status") == "fail")),
            "skipped": int(sum(1 for item in records if item.get("vela_status") == "skipped")),
            "skipped_limit": int(sum(1 for item in records if item.get("vela_status") == "skipped_limit")),
        },
        "top_k_by_epe": rank_epe[:top_k],
        "bottom_k_by_epe": rank_epe[-top_k:],
        "top_k_by_fps": rank_fps[:top_k],
        "top_k_by_sram": rank_sram[:top_k],
    }

    records_csv_path = analysis_dir / "records.csv"
    ranking_csv_path = analysis_dir / "ranking_by_epe.csv"
    vela_csv_path = analysis_dir / "vela_metrics.csv"
    summary_json_path = analysis_dir / "summary.json"
    pool_json_path = analysis_dir / "sampled_arch_pool.json"

    _save_records_csv(path=records_csv_path, records=records)
    _save_ranking_csv(path=ranking_csv_path, ranking=ranking_rows)
    _save_vela_csv(path=vela_csv_path, records=records)
    write_json(str(summary_json_path), summary_payload)
    write_json(
        str(pool_json_path),
        {
            "sample_seed": int(sample_seed),
            "num_arch_samples": int(len(arch_pool)),
            "arch_pool": arch_pool,
            "source_eval_pool_path": str(eval_pool_info.get("pool_path", "")),
        },
    )

    elapsed_seconds = float(time.perf_counter() - start_perf)
    finished_at = datetime.now(timezone.utc)
    result = {
        "status": "ok",
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "summary_json": str(summary_json_path),
        "records_csv": str(records_csv_path),
        "ranking_csv": str(ranking_csv_path),
        "vela_csv": str(vela_csv_path),
        "num_arch_samples_used": int(len(arch_pool)),
        "mean_epe": float(summary_payload["distribution"]["epe"].get("mean", 0.0)),
        "mean_fps": float(summary_payload["distribution"]["fps"].get("mean", 0.0))
        if summary_payload["distribution"]["fps"].get("count_valid", 0) > 0
        else None,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
