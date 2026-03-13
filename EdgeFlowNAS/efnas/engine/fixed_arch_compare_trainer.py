"""Joint trainer for fixed-arch baseline/ablation/full comparison."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from efnas.data.dataloader_builder import build_fc2_provider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.eval_step import accumulate_predictions, build_epe_metric
from efnas.engine.standalone_trainer import (
    _apply_gpu_device_setting,
    _build_standalone_checkpoint_paths,
    _cosine_lr_with_min,
    _evaluate_model,
    _git_commit_hash,
    _resolve_output_dir,
    _resolve_resume_dir,
    _save_standalone_checkpoint,
    _write_eval_history,
)
from efnas.engine.train_step import add_weight_decay, build_multiscale_uncertainty_loss
from efnas.network.fixed_arch_models import FixedArchModel, SUPPORTED_VARIANTS
from efnas.utils.json_io import read_json, write_json
from efnas.utils.logger import build_logger
from efnas.utils.seed import set_global_seed


def _parse_arch_code(raw: str, num_blocks: int = 9) -> List[int]:
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if len(tokens) != num_blocks:
        raise ValueError(f"backbone_arch_code must have {num_blocks} values, got {len(tokens)}")
    code = [int(v) for v in tokens]
    if any(v not in (0, 1, 2) for v in code):
        raise ValueError(f"backbone_arch_code only allows 0/1/2: {raw}")
    return code


def _parse_model_variants(raw: str) -> List[str]:
    variants = [v.strip() for v in str(raw).split("+") if v.strip()]
    if not variants:
        raise ValueError("model_variants must contain at least one variant")
    unsupported = [v for v in variants if v not in SUPPORTED_VARIANTS]
    if unsupported:
        raise ValueError(f"unsupported variants: {unsupported}")
    return variants


def _parse_model_names(raw: Optional[str], variants: List[str]) -> List[str]:
    if not raw or not raw.strip():
        return [v for v in variants]
    names = [n.strip() for n in str(raw).split("+") if n.strip()]
    if len(names) != len(variants):
        raise ValueError("model_names count must match model_variants count")
    return names


def _build_single_model_graph(
    scope_name: str,
    arch_code: List[int],
    variant: str,
    input_ph: tf.Tensor,
    label_ph: tf.Tensor,
    lr_ph: tf.Tensor,
    is_training_ph: tf.Tensor,
    flow_channels: int,
    pred_channels: int,
    weight_decay: float,
    grad_clip_global_norm: float,
) -> Dict[str, Any]:
    with tf.compat.v1.variable_scope(scope_name):
        model = FixedArchModel(
            input_ph=input_ph,
            is_training_ph=is_training_ph,
            arch_code=arch_code,
            variant=variant,
            num_out=pred_channels,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()
        loss_terms = build_multiscale_uncertainty_loss(
            preds=preds,
            label_ph=label_ph,
            num_out=flow_channels,
            return_terms=True,
        )
        loss_core = loss_terms["total"]

        trainable_vars = tf.compat.v1.trainable_variables(scope=scope_name)
        if not trainable_vars:
            raise RuntimeError(f"scope '{scope_name}' has no trainable variables")

        loss_tensor = add_weight_decay(
            loss_tensor=loss_core,
            weight_decay=weight_decay,
            trainable_vars=trainable_vars,
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr_ph,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        grads_and_vars = optimizer.compute_gradients(loss_tensor, var_list=trainable_vars)
        all_bn_updates = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        bn_updates = [op for op in all_bn_updates if op.name.startswith(f"{scope_name}/")]

        if grad_clip_global_norm > 0:
            grads = [g for g, _ in grads_and_vars if g is not None]
            vars_ = [v for g, v in grads_and_vars if g is not None]
            clipped_grads, grad_norm = tf.clip_by_global_norm(grads, clip_norm=grad_clip_global_norm)
            with tf.control_dependencies(bn_updates):
                train_op = optimizer.apply_gradients(list(zip(clipped_grads, vars_)))
        else:
            grad_norm = tf.constant(0.0, dtype=tf.float32)
            with tf.control_dependencies(bn_updates):
                train_op = optimizer.apply_gradients(grads_and_vars)

        pred_accum = accumulate_predictions(preds)
        epe_tensor = build_epe_metric(pred_tensor=pred_accum, label_ph=label_ph, num_out=flow_channels)
        scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{scope_name}/")]
        saver = tf.compat.v1.train.Saver(var_list=scope_global_vars, max_to_keep=3)

    return {
        "scope_name": scope_name,
        "arch_code": [int(v) for v in arch_code],
        "variant": variant,
        "loss": loss_tensor,
        "loss_core": loss_core,
        "loss_optical": loss_terms["optical_total"],
        "loss_uncertainty": loss_terms["uncertainty_total"],
        "train_op": train_op,
        "grad_norm": grad_norm,
        "epe": epe_tensor,
        "saver": saver,
        "trainable_vars": trainable_vars,
        "scope_global_vars": scope_global_vars,
    }


def train_fixed_arch_compare(config: Dict[str, Any]) -> int:
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    data_cfg = config.get("data", {})

    seed = int(runtime_cfg.get("seed", 42))
    set_global_seed(seed)

    experiment_dir = _resolve_output_dir(config)
    logger = build_logger("edgeflownas_fixed_arch_compare", str(experiment_dir / "train.log"))
    logger.info("start fixed-arch joint compare training")
    _apply_gpu_device_setting(train_cfg=train_cfg, logger=logger)

    backbone_arch_code = _parse_arch_code(config.get("backbone_arch_code", ""))
    model_variants = _parse_model_variants(config.get("model_variants", ""))
    model_names = _parse_model_names(config.get("model_names", None), model_variants)
    num_models = len(model_variants)
    for name, variant in zip(model_names, model_variants):
        logger.info("model=%s variant=%s arch_code=%s", name, variant, ",".join(str(v) for v in backbone_arch_code))

    batch_size = int(train_cfg.get("batch_size", 32))
    num_epochs = int(train_cfg.get("num_epochs", 400))
    base_lr = float(train_cfg.get("lr", 1e-4))
    lr_min = float(train_cfg.get("lr_min", 1e-6))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip_global_norm", 0.0))
    eval_every_epoch = max(1, int(eval_cfg.get("eval_every_epoch", 5)))
    eval_batches = int(eval_cfg.get("eval_batches", 0))
    input_h = int(data_cfg.get("input_height", 180))
    input_w = int(data_cfg.get("input_width", 240))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = flow_channels * 2

    train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    val_provider = build_fc2_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
    if len(train_provider) == 0:
        raise RuntimeError("train split is empty")
    if len(val_provider) == 0:
        raise RuntimeError("val split is empty")

    steps_per_epoch = int(math.ceil(float(len(train_provider)) / float(max(1, batch_size))))
    total_steps = max(1, num_epochs * steps_per_epoch)
    logger.info("num_models=%d batch_size=%d num_epochs=%d", num_models, batch_size, num_epochs)
    logger.info("steps_per_epoch=%d total_steps=%d", steps_per_epoch, total_steps)

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, flow_channels], name="Label")
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")

    model_graphs: Dict[str, Dict[str, Any]] = {}
    for name, variant in zip(model_names, model_variants):
        logger.info("build graph: scope=%s variant=%s", name, variant)
        mg = _build_single_model_graph(
            scope_name=name,
            arch_code=backbone_arch_code,
            variant=variant,
            input_ph=input_ph,
            label_ph=label_ph,
            lr_ph=lr_ph,
            is_training_ph=is_training_ph,
            flow_channels=flow_channels,
            pred_channels=pred_channels,
            weight_decay=weight_decay,
            grad_clip_global_norm=grad_clip,
        )
        model_graphs[name] = mg
        num_params = sum(int(np.prod(v.shape.as_list())) for v in mg["trainable_vars"])
        logger.info("model=%s variant=%s trainable_params=%d", name, variant, num_params)

    model_dirs: Dict[str, Path] = {}
    model_ckpt_paths: Dict[str, Dict[str, Path]] = {}
    for name in model_names:
        mdir = experiment_dir / f"model_{name}"
        mdir.mkdir(parents=True, exist_ok=True)
        model_dirs[name] = mdir
        model_ckpt_paths[name] = _build_standalone_checkpoint_paths(mdir)

    run_manifest = {
        "backbone_arch_code": backbone_arch_code,
        "model_variants": {name: variant for name, variant in zip(model_names, model_variants)},
        "config": config,
        "git_commit": _git_commit_hash(),
    }
    write_json(str(experiment_dir / "run_manifest.json"), run_manifest)

    best_epe: Dict[str, float] = {name: float("inf") for name in model_names}
    eval_histories: Dict[str, List[Dict[str, Any]]] = {name: [] for name in model_names}
    comparison_rows: List[Dict[str, Any]] = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        checkpoint_cfg = config.get("checkpoint", {})
        global_step = 0
        start_epoch = 1
        if bool(checkpoint_cfg.get("load_checkpoint", False)):
            resume_dir = _resolve_resume_dir(config=config, experiment_dir=experiment_dir)
            for name, mg in model_graphs.items():
                resume_ckpt = model_ckpt_paths[name]["last"]
                resume_model_ckpt = resume_dir / f"model_{name}" / "checkpoints" / "last.ckpt"
                if Path(str(resume_model_ckpt) + ".index").exists():
                    ckpt_to_load = resume_model_ckpt
                elif Path(str(resume_ckpt) + ".index").exists():
                    ckpt_to_load = resume_ckpt
                else:
                    logger.warning("model=%s checkpoint not found, train from scratch", name)
                    continue
                mg["saver"].restore(sess, str(ckpt_to_load))
                meta_path = Path(str(ckpt_to_load) + ".meta.json")
                if meta_path.exists():
                    meta = read_json(str(meta_path))
                    if isinstance(meta, dict):
                        start_epoch = max(start_epoch, int(meta.get("epoch", 0)) + 1)
                        global_step = max(global_step, int(meta.get("global_step", 0)))
                        best_epe[name] = float(meta.get("best_metric", float("inf")))
                logger.info("model=%s restored checkpoint=%s start_epoch=%d", name, str(ckpt_to_load), start_epoch)

        for epoch_idx in range(start_epoch, num_epochs + 1):
            if hasattr(train_provider, "start_epoch"):
                train_provider.start_epoch(shuffle=True)

            train_fetch = {}
            for name, mg in model_graphs.items():
                train_fetch[f"loss_{name}"] = mg["loss"]
                train_fetch[f"train_{name}"] = mg["train_op"]

            epoch_losses: Dict[str, float] = {name: 0.0 for name in model_names}
            step_count = 0
            step_iter = tqdm(
                range(steps_per_epoch),
                total=steps_per_epoch,
                desc=f"epoch {epoch_idx}/{num_epochs}",
                leave=False,
            )
            for _ in step_iter:
                input_batch, _, _, label_batch = train_provider.next_batch(batch_size=batch_size)
                input_batch = standardize_image_tensor(input_batch)
                current_lr = _cosine_lr_with_min(
                    base_lr=base_lr,
                    lr_min=lr_min,
                    step_idx=global_step,
                    total_steps=total_steps,
                )
                feed = {
                    input_ph: input_batch,
                    label_ph: label_batch,
                    lr_ph: current_lr,
                    is_training_ph: True,
                }
                results = sess.run(train_fetch, feed_dict=feed)
                for name in model_names:
                    epoch_losses[name] += float(results[f"loss_{name}"])
                step_count += 1
                global_step += 1

            avg_losses = {name: epoch_losses[name] / max(1, step_count) for name in model_names}
            do_eval = (epoch_idx % eval_every_epoch == 0) or (epoch_idx == num_epochs)
            epoch_epes: Dict[str, float] = {}
            if do_eval:
                for name, mg in model_graphs.items():
                    epoch_epes[name] = _evaluate_model(
                        sess=sess,
                        model_graph=mg,
                        input_ph=input_ph,
                        label_ph=label_ph,
                        is_training_ph=is_training_ph,
                        val_provider=val_provider,
                        batch_size=batch_size,
                        eval_batches=eval_batches,
                    )

            lr_now = _cosine_lr_with_min(base_lr, lr_min, global_step, total_steps)
            log_parts = [f"epoch={epoch_idx}", f"lr={lr_now:.2e}"]
            for name in model_names:
                log_parts.append(f"loss_{name}={avg_losses[name]:.6f}")
            if do_eval:
                baseline_name = model_names[0]
                for name in model_names:
                    log_parts.append(f"epe_{name}={epoch_epes[name]:.4f}")
                for name in model_names[1:]:
                    delta = epoch_epes[name] - epoch_epes[baseline_name]
                    log_parts.append(f"Δepe_{name}_vs_{baseline_name}={delta:.4f}")
            else:
                log_parts.append("eval=skipped")
            logger.info(" ".join(log_parts))

            for name, mg in model_graphs.items():
                metric_val = epoch_epes.get(name, float("inf"))
                _save_standalone_checkpoint(
                    sess=sess,
                    saver=mg["saver"],
                    path_prefix=model_ckpt_paths[name]["last"],
                    epoch=epoch_idx,
                    global_step=global_step,
                    metric=metric_val,
                    best_metric=best_epe[name],
                    arch_code=mg["arch_code"],
                )
                if do_eval and metric_val < best_epe[name]:
                    best_epe[name] = metric_val
                    _save_standalone_checkpoint(
                        sess=sess,
                        saver=mg["saver"],
                        path_prefix=model_ckpt_paths[name]["best"],
                        epoch=epoch_idx,
                        global_step=global_step,
                        metric=metric_val,
                        best_metric=best_epe[name],
                        arch_code=mg["arch_code"],
                    )
                    logger.info("model=%s update best checkpoint: epe=%.4f", name, metric_val)

            if do_eval:
                for name in model_names:
                    eval_histories[name].append(
                        {
                            "epoch": epoch_idx,
                            "lr": lr_now,
                            "loss": avg_losses[name],
                            "epe": epoch_epes[name],
                            "best_epe": best_epe[name],
                            "variant": model_graphs[name]["variant"],
                        }
                    )
                    _write_eval_history(model_dirs[name] / "eval_history.csv", eval_histories[name])

                comp_row: Dict[str, Any] = {"epoch": epoch_idx, "lr": lr_now}
                baseline_name = model_names[0]
                for name in model_names:
                    comp_row[f"loss_{name}"] = avg_losses[name]
                    comp_row[f"epe_{name}"] = epoch_epes[name]
                for name in model_names[1:]:
                    comp_row[f"delta_epe_{name}_vs_{baseline_name}"] = epoch_epes[name] - epoch_epes[baseline_name]
                comparison_rows.append(comp_row)
                _write_eval_history(experiment_dir / "comparison.csv", comparison_rows)

    logger.info("fixed-arch joint compare training finished")
    for name in model_names:
        logger.info("model=%s variant=%s best_epe=%.4f", name, model_graphs[name]["variant"], best_epe[name])
    return 0
