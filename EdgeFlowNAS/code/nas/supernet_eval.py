"""Evaluate trained supernet checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from code.data.dataloader_builder import build_fc2_provider
from code.data.transforms_180x240 import standardize_image_tensor
from code.engine.eval_step import build_epe_metric
from code.nas.eval_pool_builder import build_eval_pool, check_eval_pool_coverage
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


def _run_eval_pool(
    sess,
    graph_obj: Dict[str, Any],
    train_provider,
    val_provider,
    eval_pool: List[List[int]],
    bn_recal_batches: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Run BN recalibration and EPE on fixed eval pool."""
    epe_values = []
    for arch_code in eval_pool:
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
        epe_values.append(float(epe_val))

    mean_epe = float(np.mean(epe_values)) if epe_values else 0.0
    std_epe = float(np.std(epe_values)) if epe_values else 0.0
    return {"mean_epe_12": mean_epe, "std_epe_12": std_epe, "per_arch_epe": epe_values}


def _load_or_build_eval_pool(output_dir: Path, seed: int, pool_size: int) -> Dict[str, Any]:
    """Load existing eval pool or build a new one."""
    pool_path = output_dir / f"eval_pool_{pool_size}.json"
    if pool_path.exists():
        payload = read_json(str(pool_path))
        if isinstance(payload, dict) and isinstance(payload.get("pool", None), list):
            pool = payload["pool"]
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
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type to evaluate")
    parser.add_argument("--batch_size", type=int, default=None, help="override eval batch size")
    return parser


def main() -> int:
    """Entry point for supernet eval."""
    parser = _build_parser()
    args = parser.parse_args()
    if not args.eval_only:
        parser.error("supernet_eval requires --eval_only")

    config = _load_config(args.config)
    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})

    seed = int(runtime_cfg.get("seed", 42))
    pool_size = int(eval_cfg.get("eval_pool_size", 12))
    bn_recal_batches = int(args.bn_recal_batches) if args.bn_recal_batches is not None else int(eval_cfg.get("bn_recal_batches", 8))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 32))

    output_dir = _resolve_output_dir(config=config)
    pool_info = _load_or_build_eval_pool(output_dir=output_dir, seed=seed, pool_size=pool_size)
    eval_pool = pool_info["pool"]
    coverage = pool_info["coverage"]

    train_provider = build_fc2_provider(config=config, split="train", seed_offset=0)
    val_provider = build_fc2_provider(config=config, split="val", seed_offset=999)
    if len(train_provider) == 0:
        raise RuntimeError(f"train sample count is 0; source={train_provider.source_dir}")
    if len(val_provider) == 0:
        raise RuntimeError(f"val sample count is 0; source={val_provider.source_dir}")

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    graph_obj = _build_eval_graph(config=config, batch_size=batch_size)

    checkpoint_paths = _build_checkpoint_paths(experiment_dir=output_dir)
    chosen_prefix = checkpoint_paths[args.checkpoint_type]
    checkpoint_prefix = _find_existing_checkpoint(path_prefix=chosen_prefix)
    if checkpoint_prefix is None:
        raise RuntimeError(f"checkpoint not found: {chosen_prefix}")

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
        )

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
        "batch_size": int(batch_size),
        "train_samples": len(train_provider),
        "val_samples": len(val_provider),
    }

    result_path = output_dir / f"supernet_eval_result_{args.checkpoint_type}.json"
    write_json(str(result_path), {"summary": result, "per_arch_epe": metrics.get("per_arch_epe", []), "coverage": coverage})

    print(
        json.dumps(
            {
                "status": "ok",
                "result_path": str(result_path),
                "mean_epe_12": result["mean_epe_12"],
                "std_epe_12": result["std_epe_12"],
                "coverage_ok": result["eval_pool_coverage_ok"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
