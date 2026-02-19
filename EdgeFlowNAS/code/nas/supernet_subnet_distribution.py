"""Analyze subnet EPE distribution under a trained supernet checkpoint."""

import argparse
import csv
import itertools
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from code.utils.json_io import write_json

NUM_BLOCKS = 9
ARCH_SPACE_SIZE = 3 ** NUM_BLOCKS
DEFAULT_PLOTS = ("hist", "ecdf", "rank", "complexity_scatter")


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
    depth_score = float(sum(code[:4]))  # Larger index means deeper/heavier.
    kernel_light_score = float(sum(code[4:]))  # Larger index means lighter kernel.
    kernel_heavy_score = float(sum(2 - item for item in code[4:]))  # Unified heavy direction.
    total_score = float(depth_score + kernel_heavy_score)
    return {
        "depth_score": depth_score,
        "kernel_light_score": kernel_light_score,
        "kernel_heavy_score": kernel_heavy_score,
        "complexity_score": total_score,
    }


def _parse_plot_list(text: str) -> List[str]:
    """Parse comma separated plot names."""
    if not text.strip():
        return []
    names = [item.strip().lower() for item in text.split(",") if item.strip()]
    valid = {"hist", "ecdf", "rank", "complexity_scatter", "box"}
    unknown = [item for item in names if item not in valid]
    if unknown:
        raise ValueError(f"unknown plot names: {unknown}; valid={sorted(valid)}")
    out: List[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _compute_summary(epe_values: Sequence[float]) -> Dict[str, float]:
    """Compute distribution summary statistics."""
    values = np.asarray(list(epe_values), dtype=np.float64)
    if values.size == 0:
        return {
            "num_arch_samples": 0,
            "mean_epe": 0.0,
            "std_epe": 0.0,
            "min_epe": 0.0,
            "max_epe": 0.0,
            "median_epe": 0.0,
        }
    summary: Dict[str, float] = {
        "num_arch_samples": int(values.size),
        "mean_epe": float(np.mean(values)),
        "std_epe": float(np.std(values)),
        "min_epe": float(np.min(values)),
        "max_epe": float(np.max(values)),
        "median_epe": float(np.median(values)),
    }
    for pct in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        summary[f"p{pct:02d}_epe"] = float(np.percentile(values, pct))
    return summary


def _to_arch_text(arch_code: Sequence[int]) -> str:
    """Format arch code for CSV/export."""
    return ",".join(str(int(item)) for item in arch_code)


def _save_samples_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Save sampled subnet records to CSV."""
    fields = [
        "sample_index",
        "arch_code",
        "epe",
        "depth_score",
        "kernel_light_score",
        "kernel_heavy_score",
        "complexity_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fields})


def _save_ranking_csv(path: Path, ranking: Sequence[Dict[str, Any]]) -> None:
    """Save sorted ranking to CSV."""
    fields = ["rank", "sample_index", "arch_code", "epe", "complexity_score"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ranking:
            writer.writerow({key: row[key] for key in fields})


def _render_plots(records: Sequence[Dict[str, Any]], plot_names: Sequence[str], output_dir: Path, bins: int) -> List[str]:
    """Render requested PNG plots and return generated filenames."""
    if not plot_names:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epe = np.asarray([float(row["epe"]) for row in records], dtype=np.float64)
    complexity = np.asarray([float(row["complexity_score"]) for row in records], dtype=np.float64)
    generated: List[str] = []

    if "hist" in plot_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(epe, bins=max(5, int(bins)), alpha=0.85, edgecolor="black")
        ax.axvline(float(np.mean(epe)), linestyle="--", linewidth=1.5, label=f"mean={np.mean(epe):.4f}")
        ax.axvline(float(np.median(epe)), linestyle=":", linewidth=1.5, label=f"median={np.median(epe):.4f}")
        ax.set_title("Subnet EPE Histogram")
        ax.set_xlabel("EPE")
        ax.set_ylabel("Count")
        ax.legend()
        path = output_dir / "plot_hist.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path.name)

    if "ecdf" in plot_names:
        sorted_epe = np.sort(epe)
        y = np.arange(1, len(sorted_epe) + 1) / float(len(sorted_epe))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sorted_epe, y, linewidth=2)
        ax.set_title("Subnet EPE ECDF")
        ax.set_xlabel("EPE")
        ax.set_ylabel("F(x)")
        path = output_dir / "plot_ecdf.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path.name)

    if "rank" in plot_names:
        sorted_epe = np.sort(epe)
        x = np.arange(1, len(sorted_epe) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, sorted_epe, linewidth=2)
        ax.set_title("Subnet Rank Curve (Lower is Better)")
        ax.set_xlabel("Rank")
        ax.set_ylabel("EPE")
        path = output_dir / "plot_rank_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path.name)

    if "complexity_scatter" in plot_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(complexity, epe, alpha=0.75, s=18)
        ax.set_title("Complexity Proxy vs EPE")
        ax.set_xlabel("Complexity Score (Higher means heavier)")
        ax.set_ylabel("EPE")
        path = output_dir / "plot_complexity_scatter.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path.name)

    if "box" in plot_names:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.boxplot(epe, vert=False)
        ax.set_title("Subnet EPE Boxplot")
        ax.set_xlabel("EPE")
        path = output_dir / "plot_box.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path.name)

    return generated


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

    parser.add_argument("--plots", default=",".join(DEFAULT_PLOTS), help="plot list: hist,ecdf,rank,complexity_scatter,box")
    parser.add_argument("--hist_bins", type=int, default=30, help="histogram bins")
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

    bn_recal_batches = int(args.bn_recal_batches) if args.bn_recal_batches is not None else int(eval_cfg.get("bn_recal_batches", 8))
    eval_batches_per_arch = (
        int(args.eval_batches_per_arch) if args.eval_batches_per_arch is not None else int(eval_cfg.get("eval_batches_per_arch", 4))
    )
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 32))
    plot_names = _parse_plot_list(args.plots)

    tag = str(args.output_tag).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.checkpoint_type}_{len(arch_pool)}_{stamp}" + (f"_{tag}" if tag else "")
    run_dir = output_root / "subnet_distribution" / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = _evaluate_arch_pool(
        config=config,
        checkpoint_prefix=Path(checkpoint_prefix),
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
            }
        )

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

    summary = _compute_summary(epe_values=per_arch_epe)
    top_k = max(1, int(args.top_k))
    summary_payload: Dict[str, Any] = {
        "status": "ok",
        "checkpoint_type": args.checkpoint_type,
        "checkpoint_path": str(checkpoint_prefix),
        "checkpoint_meta": _load_checkpoint_meta(path_prefix=Path(checkpoint_prefix)),
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
        "plots": plot_names,
        "distribution": summary,
        "top_k": ranking_rows[:top_k],
        "bottom_k": ranking_rows[-top_k:],
    }

    samples_csv_path = run_dir / "subnet_samples.csv"
    ranking_csv_path = run_dir / "subnet_ranking.csv"
    summary_json_path = run_dir / "summary.json"
    pool_json_path = run_dir / "sampled_arch_pool.json"

    _save_samples_csv(path=samples_csv_path, records=records)
    _save_ranking_csv(path=ranking_csv_path, ranking=ranking_rows)
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

    generated_plots = _render_plots(
        records=records,
        plot_names=plot_names,
        output_dir=run_dir,
        bins=int(args.hist_bins),
    )
    elapsed_seconds = float(time.perf_counter() - start_perf)
    finished_at = datetime.now(timezone.utc)

    result = {
        "status": "ok",
        "output_dir": str(run_dir),
        "summary_json": str(summary_json_path),
        "samples_csv": str(samples_csv_path),
        "ranking_csv": str(ranking_csv_path),
        "plots_generated": generated_plots,
        "num_arch_samples_used": int(len(arch_pool)),
        "mean_epe": float(summary.get("mean_epe", 0.0)),
        "std_epe": float(summary.get("std_epe", 0.0)),
        "min_epe": float(summary.get("min_epe", 0.0)),
        "max_epe": float(summary.get("max_epe", 0.0)),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
