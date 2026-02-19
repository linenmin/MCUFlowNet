"""Evaluate trained supernet checkpoints."""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from code.data.dataloader_builder import build_fc2_provider
from code.data.transforms_180x240 import standardize_image_tensor
from code.engine.eval_step import build_epe_metric
from code.nas.eval_pool_builder import BILINEAR_BASELINE_ARCH_CODE, build_eval_pool, check_eval_pool_coverage
from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet
from code.utils.json_io import read_json, write_json
from code.utils.path_utils import ensure_directory, project_root

try:
    import yaml
except Exception:
    yaml = None


def _parse_scalar(value_text: str) -> Any:
    """Parse simple YAML scalar values."""
    lowered = value_text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value_text.startswith('"') and value_text.endswith('"'):
        return value_text[1:-1]
    if value_text.startswith("'") and value_text.endswith("'"):
        return value_text[1:-1]
    try:
        return int(value_text)
    except Exception:
        pass
    try:
        return float(value_text)
    except Exception:
        pass
    return value_text


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    """Fallback YAML parser for minimal config files."""
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            content = raw.split("#", 1)[0].rstrip("\n")
            if not content.strip():
                continue
            indent = len(content) - len(content.lstrip(" "))
            stripped = content.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if stripped.endswith(":"):
                key = stripped[:-1].strip()
                node: Dict[str, Any] = {}
                parent[key] = node
                stack.append((indent, node))
                continue
            key, value_text = stripped.split(":", 1)
            parent[key.strip()] = _parse_scalar(value_text.strip())
    return root


def _load_config(path_like: str) -> Dict[str, Any]:
    """Load YAML config."""
    config_path = Path(path_like)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    if yaml is None:
        payload = _load_simple_yaml(config_path)
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("config top level must be a mapping")
    return payload


def _set_nested(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set nested config value by dot-separated key path."""
    keys = key_path.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _put_override(overrides: Dict[str, Any], key_path: str, value: Optional[Any]) -> None:
    """Append one override when value is provided."""
    if value is None:
        return
    overrides[key_path] = value


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides without editing YAML manually."""
    overrides: Dict[str, Any] = {}
    _put_override(overrides, "runtime.experiment_name", args.experiment_name)
    _put_override(overrides, "data.base_path", args.base_path)
    _put_override(overrides, "data.train_dir", args.train_dir)
    _put_override(overrides, "data.val_dir", args.val_dir)
    _put_override(overrides, "train.batch_size", args.train_batch_size)
    _put_override(overrides, "runtime.seed", args.seed)
    merged = deepcopy(config)
    for key_path, value in overrides.items():
        _set_nested(merged, key_path, value)
    return merged


def _resolve_output_dir(config: Dict[str, Any]) -> Path:
    """Resolve experiment output directory."""
    runtime_cfg = config.get("runtime", {})
    output_root = runtime_cfg.get("output_root", "outputs/supernet")
    experiment_name = runtime_cfg.get("experiment_name", "default_experiment")
    return ensure_directory(str(project_root() / output_root / experiment_name))


def _build_checkpoint_paths(experiment_dir: Path) -> Dict[str, Path]:
    """Build checkpoint path prefixes."""
    ckpt_root = ensure_directory(str(experiment_dir / "checkpoints"))
    return {"best": ckpt_root / "supernet_best.ckpt", "last": ckpt_root / "supernet_last.ckpt"}


def _checkpoint_exists(path_prefix: Path) -> bool:
    """Check if checkpoint index exists."""
    return Path(str(path_prefix) + ".index").exists()


def _find_existing_checkpoint(path_prefix: Path) -> Optional[Path]:
    """Find explicit or latest checkpoint in folder."""
    if _checkpoint_exists(path_prefix=path_prefix):
        return path_prefix
    latest = tf.train.latest_checkpoint(str(path_prefix.parent))
    if latest:
        return Path(latest)
    return None


def _load_checkpoint_meta(path_prefix: Path) -> Dict[str, Any]:
    """Load sidecar checkpoint metadata."""
    meta_path = Path(str(path_prefix) + ".meta.json")
    if not meta_path.exists():
        return {}
    payload = read_json(str(meta_path))
    if isinstance(payload, dict):
        return payload
    return {}


def _build_eval_graph(config: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    """Build TF1 graph for supernet eval."""
    data_cfg = config.get("data", {})
    input_h = int(data_cfg.get("input_height", 180))
    input_w = int(data_cfg.get("input_width", 240))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = int(flow_channels * 2)

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, input_h, input_w, flow_channels], name="Label")
    arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[9], name="ArchCode")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")

    model = MultiScaleResNetSupernet(
        input_ph=input_ph,
        arch_code_ph=arch_code_ph,
        is_training_ph=is_training_ph,
        num_out=pred_channels,
        init_neurons=32,
        expansion_factor=2.0,
    )
    preds = model.build()

    epe_tensor = build_epe_metric(pred_tensor=preds[-1], label_ph=label_ph, num_out=flow_channels)
    saver = tf.compat.v1.train.Saver(max_to_keep=5)
    return {
        "input_ph": input_ph,
        "label_ph": label_ph,
        "arch_code_ph": arch_code_ph,
        "is_training_ph": is_training_ph,
        "pred_tensor": preds[-1],
        "epe": epe_tensor,
        "saver": saver,
    }


def _build_arch_ranking(eval_pool: List[List[int]], per_arch_epe: List[float]) -> List[Dict[str, Any]]:
    """Build per-arch rank summary sorted by EPE ascending."""
    indexed = []
    for arch_idx, (arch_code, epe_val) in enumerate(zip(eval_pool, per_arch_epe)):
        indexed.append((int(arch_idx), [int(v) for v in arch_code], float(epe_val)))
    indexed.sort(key=lambda item: (item[2], item[0]))
    ranking = []
    for rank_idx, (arch_idx, arch_code, epe_val) in enumerate(indexed, start=1):
        ranking.append(
            {
                "rank": int(rank_idx),
                "arch_index": int(arch_idx),
                "arch_code": arch_code,
                "epe": float(epe_val),
            }
        )
    return ranking


def _run_eval_pool(
    sess,
    graph_obj: Dict[str, Any],
    train_provider,
    val_provider,
    eval_pool: List[List[int]],
    bn_recal_batches: int,
    batch_size: int,
    eval_batches_per_arch: int,
) -> Dict[str, Any]:
    """Run BN recalibration and EPE on fixed eval pool."""
    per_arch_epe = []
    bn_batches = int(bn_recal_batches)
    eval_batches = max(1, int(eval_batches_per_arch))
    total_steps = max(1, len(eval_pool) * (bn_batches + eval_batches))
    with tqdm(total=total_steps, desc="supernet eval", unit="step") as progress:
        for arch_idx, arch_code in enumerate(eval_pool, start=1):
            if hasattr(train_provider, "reset_cursor"):
                train_provider.reset_cursor(0)
            for _ in range(bn_batches):
                train_input, _, _, train_label = train_provider.next_batch(batch_size=batch_size)
                train_input = standardize_image_tensor(train_input)
                sess.run(
                    graph_obj["pred_tensor"],
                    feed_dict={
                        graph_obj["input_ph"]: train_input,
                        graph_obj["label_ph"]: train_label,
                        graph_obj["arch_code_ph"]: arch_code,
                        graph_obj["is_training_ph"]: True,
                    },
                )
                progress.update(1)

            if hasattr(val_provider, "reset_cursor"):
                val_provider.reset_cursor(0)
            arch_batch_epes = []
            for _ in range(eval_batches):
                val_input, _, _, val_label = val_provider.next_batch(batch_size=batch_size)
                val_input = standardize_image_tensor(val_input)
                epe_val = sess.run(
                    graph_obj["epe"],
                    feed_dict={
                        graph_obj["input_ph"]: val_input,
                        graph_obj["label_ph"]: val_label,
                        graph_obj["arch_code_ph"]: arch_code,
                        graph_obj["is_training_ph"]: False,
                    },
                )
                arch_batch_epes.append(float(epe_val))
                progress.update(1)
            arch_epe_mean = float(np.mean(arch_batch_epes)) if arch_batch_epes else 0.0
            per_arch_epe.append(arch_epe_mean)
            progress.set_postfix(arch=f"{arch_idx}/{len(eval_pool)}", epe=f"{arch_epe_mean:.4f}")

    mean_epe = float(np.mean(per_arch_epe)) if per_arch_epe else 0.0
    std_epe = float(np.std(per_arch_epe)) if per_arch_epe else 0.0
    arch_ranking = _build_arch_ranking(eval_pool=eval_pool, per_arch_epe=per_arch_epe)
    return {"mean_epe_12": mean_epe, "std_epe_12": std_epe, "per_arch_epe": per_arch_epe, "arch_ranking": arch_ranking}


def _run_one_arch_eval(
    sess,
    graph_obj: Dict[str, Any],
    train_provider,
    val_provider,
    arch_code: List[int],
    bn_recal_batches: int,
    batch_size: int,
    eval_batches_per_arch: int,
) -> float:
    """Run BN recalibration + val EPE for one arch."""
    if hasattr(train_provider, "reset_cursor"):
        train_provider.reset_cursor(0)
    for _ in range(int(bn_recal_batches)):
        train_input, _, _, train_label = train_provider.next_batch(batch_size=batch_size)
        train_input = standardize_image_tensor(train_input)
        sess.run(
            graph_obj["pred_tensor"],
            feed_dict={
                graph_obj["input_ph"]: train_input,
                graph_obj["label_ph"]: train_label,
                graph_obj["arch_code_ph"]: arch_code,
                graph_obj["is_training_ph"]: True,
            },
        )
    if hasattr(val_provider, "reset_cursor"):
        val_provider.reset_cursor(0)
    eval_batches = max(1, int(eval_batches_per_arch))
    arch_batch_epes = []
    for _ in range(eval_batches):
        val_input, _, _, val_label = val_provider.next_batch(batch_size=batch_size)
        val_input = standardize_image_tensor(val_input)
        epe_val = sess.run(
            graph_obj["epe"],
            feed_dict={
                graph_obj["input_ph"]: val_input,
                graph_obj["label_ph"]: val_label,
                graph_obj["arch_code_ph"]: arch_code,
                graph_obj["is_training_ph"]: False,
            },
        )
        arch_batch_epes.append(float(epe_val))
    return float(np.mean(arch_batch_epes)) if arch_batch_epes else 0.0


def _split_arch_chunks(eval_pool: List[List[int]], num_chunks: int) -> List[List[List[int]]]:
    """Split arch list into balanced chunks."""
    buckets: List[List[List[int]]] = [[] for _ in range(max(1, int(num_chunks)))]
    for idx, arch_code in enumerate(eval_pool):
        buckets[idx % len(buckets)].append(arch_code)
    return [chunk for chunk in buckets if chunk]


def _eval_arch_chunk_worker(
    config: Dict[str, Any],
    checkpoint_prefix: str,
    arch_chunk: List[List[int]],
    bn_recal_batches: int,
    batch_size: int,
    eval_batches_per_arch: int,
    worker_id: int,
    cpu_only: bool,
) -> List[Dict[str, Any]]:
    """Evaluate a chunk of arch codes in one worker process."""
    if cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    graph_obj = _build_eval_graph(config=config, batch_size=batch_size)
    train_provider = build_fc2_provider(config=config, split="train", seed_offset=1000 + int(worker_id), provider_mode="eval")
    val_provider = build_fc2_provider(config=config, split="val", seed_offset=2000 + int(worker_id), provider_mode="eval")

    if len(train_provider) == 0:
        raise RuntimeError(f"worker={worker_id} train sample count is 0; source={train_provider.source_dir}")
    if len(val_provider) == 0:
        raise RuntimeError(f"worker={worker_id} val sample count is 0; source={val_provider.source_dir}")

    outputs: List[Dict[str, Any]] = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        graph_obj["saver"].restore(sess, checkpoint_prefix)
        for arch_code in arch_chunk:
            epe_val = _run_one_arch_eval(
                sess=sess,
                graph_obj=graph_obj,
                train_provider=train_provider,
                val_provider=val_provider,
                arch_code=arch_code,
                bn_recal_batches=bn_recal_batches,
                batch_size=batch_size,
                eval_batches_per_arch=eval_batches_per_arch,
            )
            outputs.append({"arch_code": [int(item) for item in arch_code], "epe": float(epe_val)})
    return outputs


def _run_eval_pool_parallel(
    config: Dict[str, Any],
    checkpoint_prefix: Path,
    eval_pool: List[List[int]],
    bn_recal_batches: int,
    batch_size: int,
    eval_batches_per_arch: int,
    num_workers: int,
    cpu_only: bool,
) -> Dict[str, Any]:
    """Run eval pool using multi-process arch parallelism."""
    chunks = _split_arch_chunks(eval_pool=eval_pool, num_chunks=max(1, int(num_workers)))
    arch_to_epe: Dict[tuple, float] = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as executor:
        future_to_worker = {
            executor.submit(
                _eval_arch_chunk_worker,
                config,
                str(checkpoint_prefix),
                chunk,
                int(bn_recal_batches),
                int(batch_size),
                int(eval_batches_per_arch),
                worker_id,
                bool(cpu_only),
            ): worker_id
            for worker_id, chunk in enumerate(chunks, start=1)
        }
        with tqdm(total=len(eval_pool), desc=f"supernet eval x{len(chunks)}", unit="arch") as progress:
            for future in as_completed(future_to_worker):
                worker_outputs = future.result()
                for item in worker_outputs:
                    arch_key = tuple(int(x) for x in item["arch_code"])
                    arch_to_epe[arch_key] = float(item["epe"])
                progress.update(len(worker_outputs))
                progress.set_postfix(done=f"{len(arch_to_epe)}/{len(eval_pool)}")

    epe_values = [float(arch_to_epe[tuple(int(x) for x in arch_code)]) for arch_code in eval_pool]
    mean_epe = float(np.mean(epe_values)) if epe_values else 0.0
    std_epe = float(np.std(epe_values)) if epe_values else 0.0
    arch_ranking = _build_arch_ranking(eval_pool=eval_pool, per_arch_epe=epe_values)
    return {"mean_epe_12": mean_epe, "std_epe_12": std_epe, "per_arch_epe": epe_values, "arch_ranking": arch_ranking}


def _load_or_build_eval_pool(output_dir: Path, seed: int, pool_size: int) -> Dict[str, Any]:
    """Load existing eval pool or build a new one."""
    pool_path = output_dir / f"eval_pool_{pool_size}.json"
    baseline_key = tuple(int(item) for item in BILINEAR_BASELINE_ARCH_CODE)
    if pool_path.exists():
        payload = read_json(str(pool_path))
        if isinstance(payload, dict) and isinstance(payload.get("pool", None), list):
            pool = payload["pool"]
            pool_keys = {tuple(int(item) for item in code) for code in pool}
            if baseline_key not in pool_keys:
                # 评估池语义升级后强制重建，保证包含 bilinear 对齐子网。
                pool = build_eval_pool(seed=seed, size=pool_size, num_blocks=9)
                coverage = check_eval_pool_coverage(pool=pool, num_blocks=9)
                write_json(str(pool_path), {"pool": pool, "coverage": coverage})
                return {"pool": pool, "coverage": coverage, "pool_path": pool_path}
            coverage = payload.get("coverage", check_eval_pool_coverage(pool=pool, num_blocks=9))
            return {"pool": pool, "coverage": coverage, "pool_path": pool_path}

    pool = build_eval_pool(seed=seed, size=pool_size, num_blocks=9)
    coverage = check_eval_pool_coverage(pool=pool, num_blocks=9)
    write_json(str(pool_path), {"pool": pool, "coverage": coverage})
    return {"pool": pool, "coverage": coverage, "pool_path": pool_path}


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="evaluate trained supernet checkpoint")
    parser.add_argument("--config", required=True, help="path to supernet config yaml")
    parser.add_argument("--eval_only", action="store_true", help="run eval only flow")
    parser.add_argument("--bn_recal_batches", type=int, default=None, help="override bn recalibration batches")
    parser.add_argument("--eval_batches_per_arch", type=int, default=None, help="override eval batches per arch")
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type to evaluate")
    parser.add_argument("--batch_size", type=int, default=None, help="override eval batch size")
    parser.add_argument("--experiment_name", default=None, help="override runtime.experiment_name")
    parser.add_argument("--base_path", default=None, help="override data.base_path")
    parser.add_argument("--train_dir", default=None, help="override data.train_dir")
    parser.add_argument("--val_dir", default=None, help="override data.val_dir")
    parser.add_argument("--train_batch_size", type=int, default=None, help="override train.batch_size in config")
    parser.add_argument("--seed", type=int, default=None, help="override runtime.seed")
    parser.add_argument("--cpu_only", action="store_true", help="force CPU eval by hiding GPUs")
    parser.add_argument("--num_workers", type=int, default=1, help="arch-eval worker processes")
    return parser


def main() -> int:
    """Entry point for supernet eval."""
    parser = _build_parser()
    args = parser.parse_args()
    if not args.eval_only:
        parser.error("supernet_eval requires --eval_only")
    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    config = _apply_cli_overrides(config=_load_config(args.config), args=args)
    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})

    seed = int(runtime_cfg.get("seed", 42))
    pool_size = int(eval_cfg.get("eval_pool_size", 12))
    bn_recal_batches = int(args.bn_recal_batches) if args.bn_recal_batches is not None else int(eval_cfg.get("bn_recal_batches", 8))
    eval_batches_per_arch = (
        int(args.eval_batches_per_arch) if args.eval_batches_per_arch is not None else int(eval_cfg.get("eval_batches_per_arch", 4))
    )
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 32))
    num_workers = max(1, int(args.num_workers))

    output_dir = _resolve_output_dir(config=config)
    pool_info = _load_or_build_eval_pool(output_dir=output_dir, seed=seed, pool_size=pool_size)
    eval_pool = pool_info["pool"]
    coverage = pool_info["coverage"]

    train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="eval")
    val_provider = build_fc2_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
    if len(train_provider) == 0:
        raise RuntimeError(f"train sample count is 0; source={train_provider.source_dir}")
    if len(val_provider) == 0:
        raise RuntimeError(f"val sample count is 0; source={val_provider.source_dir}")

    checkpoint_paths = _build_checkpoint_paths(experiment_dir=output_dir)
    chosen_prefix = checkpoint_paths[args.checkpoint_type]
    checkpoint_prefix = _find_existing_checkpoint(path_prefix=chosen_prefix)
    if checkpoint_prefix is None:
        raise RuntimeError(f"checkpoint not found: {chosen_prefix}")
    workers_used = min(num_workers, max(1, len(eval_pool)))
    if workers_used <= 1:
        if args.cpu_only:
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        graph_obj = _build_eval_graph(config=config, batch_size=batch_size)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            graph_obj["saver"].restore(sess, str(checkpoint_prefix))
            metrics = _run_eval_pool(
                sess=sess,
                graph_obj=graph_obj,
                train_provider=train_provider,
                val_provider=val_provider,
                eval_pool=eval_pool,
                bn_recal_batches=bn_recal_batches,
                batch_size=batch_size,
                eval_batches_per_arch=eval_batches_per_arch,
            )
    else:
        metrics = _run_eval_pool_parallel(
            config=config,
            checkpoint_prefix=checkpoint_prefix,
            eval_pool=eval_pool,
            bn_recal_batches=bn_recal_batches,
            batch_size=batch_size,
            eval_batches_per_arch=eval_batches_per_arch,
            num_workers=workers_used,
            cpu_only=bool(args.cpu_only),
        )

    elapsed_seconds = float(time.perf_counter() - start_perf)
    elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(int(round(elapsed_seconds))))
    finished_at = datetime.now(timezone.utc)
    checkpoint_meta = _load_checkpoint_meta(path_prefix=checkpoint_prefix)
    result = {
        "status": "ok",
        "checkpoint_type": args.checkpoint_type,
        "checkpoint_path": str(checkpoint_prefix),
        "checkpoint_meta": checkpoint_meta,
        "eval_pool_path": str(pool_info["pool_path"]),
        "eval_pool_coverage_ok": bool(coverage.get("ok", False)),
        "mean_epe_12": float(metrics["mean_epe_12"]),
        "std_epe_12": float(metrics["std_epe_12"]),
        "bn_recal_batches": int(bn_recal_batches),
        "eval_batches_per_arch": int(eval_batches_per_arch),
        "batch_size": int(batch_size),
        "num_workers_requested": int(num_workers),
        "num_workers_used": int(workers_used),
        "train_samples": len(train_provider),
        "val_samples": len(val_provider),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hms": elapsed_hms,
    }

    result_path = output_dir / f"supernet_eval_result_{args.checkpoint_type}.json"
    write_json(
        str(result_path),
        {
            "summary": result,
            "per_arch_epe": metrics.get("per_arch_epe", []),
            "arch_ranking": metrics.get("arch_ranking", []),
            "coverage": coverage,
        },
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "result_path": str(result_path),
                "mean_epe_12": result["mean_epe_12"],
                "std_epe_12": result["std_epe_12"],
                "coverage_ok": result["eval_pool_coverage_ok"],
                "num_workers_used": result["num_workers_used"],
                "eval_batches_per_arch": result["eval_batches_per_arch"],
                "elapsed_seconds": result["elapsed_seconds"],
                "elapsed_hms": result["elapsed_hms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
