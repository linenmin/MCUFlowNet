import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from efnas.utils.import_bootstrap import bootstrap_project_paths, resolve_project_paths

bootstrap_project_paths(anchor_file=__file__)
project_root = resolve_project_paths(anchor_file=__file__)["mcu_root"]

from efnas.engine.eval_step import accumulate_predictions
from efnas.network.fixed_arch_models import FixedArchModel, SUPPORTED_VARIANTS


def _load_checkpoint_meta(model_dir: Path, ckpt_name: str = "best") -> Dict[str, Any]:
    meta_path = model_dir / "checkpoints" / f"{ckpt_name}.ckpt.meta.json"
    if not meta_path.exists() and ckpt_name == "best":
        meta_path = model_dir / "checkpoints" / "last.ckpt.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No checkpoint meta found for {model_dir} ({ckpt_name})")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_run_manifest(model_dir: Path) -> Dict[str, Any]:
    manifest_path = model_dir.parent / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _resolve_scope_name(model_dir: Path) -> str:
    dir_name = model_dir.name
    return dir_name[len("model_") :] if dir_name.startswith("model_") else dir_name


def _resolve_variant(model_dir: Path, explicit_variant: Optional[str] = None) -> str:
    if explicit_variant:
        variant = str(explicit_variant).strip()
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        return variant

    scope_name = _resolve_scope_name(model_dir)
    run_manifest = _load_run_manifest(model_dir)
    manifest_variants = run_manifest.get("model_variants", {})
    if isinstance(manifest_variants, dict) and scope_name in manifest_variants:
        variant = str(manifest_variants[scope_name]).strip()
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"unsupported variant from run_manifest: {variant}")
        return variant

    if scope_name in SUPPORTED_VARIANTS:
        return scope_name

    raise ValueError(
        f"Cannot resolve variant for {model_dir}. "
        "Provide --variant explicitly or ensure run_manifest.json exists."
    )


def setup_fixed_arch_eval_model(
    checkpoint_dir: Path,
    patch_size: Tuple[int, int],
    ckpt_name: str = "best",
    variant: Optional[str] = None,
) -> Tuple[tf.compat.v1.Session, tf.Tensor, tf.Tensor, Dict[str, Any]]:
    meta_data = _load_checkpoint_meta(checkpoint_dir, ckpt_name)
    arch_code = meta_data["arch_code"]
    if isinstance(arch_code, str):
        arch_list = [int(x) for x in arch_code.split(",") if str(x).strip()]
    else:
        arch_list = [int(x) for x in arch_code]

    resolved_variant = _resolve_variant(checkpoint_dir, explicit_variant=variant)
    scope_name = _resolve_scope_name(checkpoint_dir)

    tf.compat.v1.reset_default_graph()

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, patch_size[0], patch_size[1], 6], name="input_ph")
    is_training_ph = tf.compat.v1.placeholder_with_default(
        tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph"
    )

    with tf.compat.v1.variable_scope(scope_name):
        model = FixedArchModel(
            input_ph=input_ph,
            is_training_ph=is_training_ph,
            arch_code=arch_list,
            variant=resolved_variant,
            num_out=4,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()

    preds_accumulated = accumulate_predictions(preds)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{scope_name}/")]
    if not scope_global_vars:
        raise RuntimeError(f"No variables found with scope '{scope_name}'")

    saver = tf.compat.v1.train.Saver(var_list=scope_global_vars)

    ckpt_path = meta_data.get("checkpoint_path")
    if not ckpt_path:
        fallback = checkpoint_dir / "checkpoints" / f"{ckpt_name}.ckpt"
        ckpt_path = str(fallback)

    saver.restore(sess, ckpt_path)

    meta = dict(meta_data)
    meta["variant"] = resolved_variant
    meta["scope_name"] = scope_name
    meta["checkpoint_dir"] = str(checkpoint_dir)
    meta["checkpoint_path"] = str(ckpt_path)
    return sess, input_ph, preds_accumulated, meta


def preprocess_eval_batch(input_batch):
    return (input_batch / 255.0) * 2.0 - 1.0
