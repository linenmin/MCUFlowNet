"""Short FC2 scratch retraining for V3 distill-or-not rank probes."""

from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from efnas.data.dataloader_builder import build_fc2_provider
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.eval_step import accumulate_predictions, build_epe_metric
from efnas.engine.standalone_trainer import (
    _apply_gpu_device_setting,
    _build_standalone_checkpoint_paths,
    _cosine_lr_with_min,
    _git_commit_hash,
    _resolve_output_dir,
    _save_standalone_checkpoint,
)
from efnas.engine.train_step import add_weight_decay, build_multiscale_uncertainty_loss
from efnas.nas.search_space_v3 import validate_arch_code
from efnas.network.fixed_arch_models_v3 import FixedArchModelV3
from efnas.utils.json_io import read_json, write_json
from efnas.utils.logger import build_logger
from efnas.utils.seed import set_global_seed


def parse_arch_code(raw: Any) -> List[int]:
    """Parse one 11D V3 architecture code."""
    if isinstance(raw, (list, tuple)):
        code = [int(item) for item in raw]
    else:
        text = str(raw).replace(":", ",").replace(" ", ",")
        code = [int(token) for token in text.split(",") if token.strip()]
    validate_arch_code(code)
    return code


def _wrap_prefetch(provider: Any, prefetch_batches: int) -> Any:
    if int(prefetch_batches) <= 0:
        return provider
    return PrefetchBatchProvider(provider, prefetch_batches=int(prefetch_batches))


def _close_provider(provider: Any) -> None:
    if hasattr(provider, "close"):
        provider.close()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    tmp_path = path.with_name(tmp_name)
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        for attempt in range(20):
            try:
                tmp_path.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _summarize_grad_norms(values: Sequence[float], clip_threshold: float) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "clip_rate": 0.0}
    threshold = float(clip_threshold)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "clip_rate": float(np.mean(arr > threshold)) if threshold > 0 else 0.0,
    }


def _iter_micro_slices(total_size: int, micro_batch_size: int) -> List[slice]:
    if total_size <= 0:
        return []
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    return [slice(start, min(start + micro_batch_size, total_size)) for start in range(0, total_size, micro_batch_size)]


def _build_graph(
    scope_name: str,
    arch_code: Sequence[int],
    input_ph,
    label_ph,
    lr_ph,
    grad_scale_ph,
    is_training_ph,
    flow_channels: int,
    pred_channels: int,
    weight_decay: float,
    grad_clip_global_norm: float,
) -> Dict[str, Any]:
    with tf.compat.v1.variable_scope(scope_name):
        model = FixedArchModelV3(
            input_ph=input_ph,
            is_training_ph=is_training_ph,
            arch_code=arch_code,
            num_out=pred_channels,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()
        loss_terms = build_multiscale_uncertainty_loss(preds=preds, label_ph=label_ph, num_out=flow_channels, return_terms=True)
        loss_core = loss_terms["total"]
        trainable_vars = tf.compat.v1.trainable_variables(scope=scope_name)
        if not trainable_vars:
            raise RuntimeError(f"scope '{scope_name}' has no trainable variables")
        loss_tensor = add_weight_decay(loss_tensor=loss_core, weight_decay=weight_decay, trainable_vars=trainable_vars)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_ph, beta1=0.9, beta2=0.999, epsilon=1e-8)
        scaled_loss = loss_tensor * grad_scale_ph
        grads_and_vars = optimizer.compute_gradients(scaled_loss, var_list=trainable_vars)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        vars_ = [v for _, v in grads_and_vars]
        accum_vars = [
            tf.Variable(tf.zeros(shape=v.shape, dtype=v.dtype.base_dtype), trainable=False, name=f"{v.op.name.replace('/', '_')}_grad_accum")
            for v in vars_
        ]
        zero_grad_op = tf.group(*[var.assign(tf.zeros_like(var)) for var in accum_vars], name="zero_gradients") if accum_vars else tf.no_op()
        bn_updates = [op for op in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS) if op.name.startswith(f"{scope_name}/")]
        with tf.control_dependencies(bn_updates):
            accum_op = (
                tf.group(*[accum.assign_add(grad) for accum, (grad, _) in zip(accum_vars, grads_and_vars)], name="accumulate_gradients")
                if accum_vars
                else tf.no_op()
            )
        averaged_grads = list(accum_vars)
        if grad_clip_global_norm > 0:
            clipped_grads, grad_norm = tf.clip_by_global_norm(averaged_grads, clip_norm=grad_clip_global_norm)
            apply_pairs = list(zip(clipped_grads, vars_))
        else:
            grad_norm = tf.linalg.global_norm(averaged_grads) if averaged_grads else tf.constant(0.0, dtype=tf.float32)
            apply_pairs = list(zip(averaged_grads, vars_))
        train_op = optimizer.apply_gradients(apply_pairs)
        epe_tensor = build_epe_metric(pred_tensor=accumulate_predictions(preds), label_ph=label_ph, num_out=flow_channels)
        scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{scope_name}/")]
        saver = tf.compat.v1.train.Saver(var_list=scope_global_vars, max_to_keep=3)
    return {
        "scope_name": scope_name,
        "arch_code": [int(v) for v in arch_code],
        "loss": loss_tensor,
        "loss_optical": loss_terms["optical_total"],
        "loss_uncertainty": loss_terms["uncertainty_total"],
        "zero_grad_op": zero_grad_op,
        "accum_op": accum_op,
        "train_op": train_op,
        "grad_norm": grad_norm,
        "epe": epe_tensor,
        "saver": saver,
        "trainable_vars": trainable_vars,
        "scope_global_vars": scope_global_vars,
    }


def _evaluate_fc2_with_progress(
    sess,
    graph_obj: Dict[str, Any],
    input_ph,
    label_ph,
    is_training_ph,
    val_provider,
    batch_size: int,
    eval_batches: int,
    desc: str,
) -> float:
    if hasattr(val_provider, "reset_cursor"):
        val_provider.reset_cursor(0)
    num_batches = max(1, int(math.ceil(len(val_provider) / float(batch_size)))) if eval_batches <= 0 else int(eval_batches)
    values: List[float] = []
    for _ in tqdm(range(num_batches), total=num_batches, desc=desc, leave=False):
        input_batch, _, _, label_batch = val_provider.next_batch(batch_size=batch_size)
        input_batch = standardize_image_tensor(input_batch)
        epe = sess.run(graph_obj["epe"], feed_dict={input_ph: input_batch, label_ph: label_batch, is_training_ph: True})
        values.append(float(epe))
    return float(np.mean(values)) if values else float("inf")


def _save_checkpoint_with_arch(sess, saver, path_prefix: Path, epoch: int, global_step: int, metric: float, best_metric: float, arch_code: Sequence[int]) -> Path:
    return _save_standalone_checkpoint(
        sess=sess,
        saver=saver,
        path_prefix=path_prefix,
        epoch=epoch,
        global_step=global_step,
        metric=metric,
        best_metric=best_metric,
        arch_code=[int(v) for v in arch_code],
    )


def _run_sintel_if_configured(model_dir: Path, config: Dict[str, Any], epoch_idx: int, ckpt_name: str) -> Dict[str, Any] | None:
    from efnas.engine.distill_or_not_sintel_runtime import evaluate_v3_checkpoint_dir_on_sintel
    from efnas.engine.retrain_v2_sintel_runtime import _resolve_sintel_eval_config

    sintel_cfg = _resolve_sintel_eval_config(config)
    if sintel_cfg is None:
        return None
    return evaluate_v3_checkpoint_dir_on_sintel(
        model_dir=model_dir,
        dataset_root=str(sintel_cfg["dataset_root"]),
        sintel_list=str(sintel_cfg["sintel_list"]),
        patch_size=tuple(sintel_cfg["patch_size"]),
        ckpt_name=ckpt_name,
        max_samples=sintel_cfg.get("max_samples", None),
        progress_desc=f"Sintel {model_dir.name} e{epoch_idx}",
    )


def train_distill_or_not_fc2(config: Dict[str, Any]) -> int:
    """Run one V3 fixed subnet FC2 scratch retraining job."""
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("eval", {})
    checkpoint_cfg = config.get("checkpoint", {})

    seed = int(runtime_cfg.get("seed", 42))
    set_global_seed(seed)
    experiment_dir = _resolve_output_dir(config)
    logger = build_logger("distill_or_not_fc2", str(experiment_dir / "train.log"))
    logger.info("start distill-or-not V3 scratch FC2 retrain")
    _apply_gpu_device_setting(train_cfg=train_cfg, logger=logger)

    arch_code = parse_arch_code(config.get("arch_code"))
    model_name = str(config.get("model_name", "candidate")).strip() or "candidate"
    batch_size = int(train_cfg.get("batch_size", 32))
    micro_batch_size = min(int(train_cfg.get("micro_batch_size", batch_size)), batch_size)
    num_epochs = int(train_cfg.get("num_epochs", 50))
    base_lr = float(train_cfg.get("lr", 1e-4))
    lr_min = float(train_cfg.get("lr_min", 1e-6))
    weight_decay = float(train_cfg.get("weight_decay", 4e-5))
    grad_clip = float(train_cfg.get("grad_clip_global_norm", 200.0))
    eval_every_epoch = max(1, int(eval_cfg.get("eval_every_epoch", 1)))
    eval_batches = int(eval_cfg.get("eval_batches", 0))
    sintel_every = int(eval_cfg.get("sintel", {}).get("eval_every_epoch", 5)) if isinstance(eval_cfg.get("sintel", {}), dict) else 0
    input_h = int(data_cfg.get("input_height", 172))
    input_w = int(data_cfg.get("input_width", 224))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = flow_channels * 2

    train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    val_provider = build_fc2_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
    train_provider = _wrap_prefetch(train_provider, int(data_cfg.get("prefetch_batches", 0)))
    val_provider = _wrap_prefetch(val_provider, int(data_cfg.get("eval_prefetch_batches", 0)))
    steps_per_epoch = int(math.ceil(len(train_provider) / float(max(1, batch_size))))
    total_steps = max(1, steps_per_epoch * num_epochs)

    logger.info("model=%s arch=%s scratch_init=true", model_name, ",".join(str(v) for v in arch_code))
    logger.info("input=%dx%d batch=%d micro_batch=%d epochs=%d steps_per_epoch=%d", input_h, input_w, batch_size, micro_batch_size, num_epochs, steps_per_epoch)
    logger.info("lr=%.2e lr_min=%.2e optimizer=Adam weight_decay=%.2e grad_clip=%.1f", base_lr, lr_min, weight_decay, grad_clip)
    logger.info("fc2_eval_every=%d sintel_eval_every=%d workers train/eval=%s/%s prefetch train/eval=%s/%s", eval_every_epoch, sintel_every, data_cfg.get("fc2_num_workers"), data_cfg.get("fc2_eval_num_workers"), data_cfg.get("prefetch_batches"), data_cfg.get("eval_prefetch_batches"))

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, flow_channels], name="Label")
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")
    grad_scale_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="GradScale")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")
    graph_obj = _build_graph(
        scope_name=model_name,
        arch_code=arch_code,
        input_ph=input_ph,
        label_ph=label_ph,
        lr_ph=lr_ph,
        grad_scale_ph=grad_scale_ph,
        is_training_ph=is_training_ph,
        flow_channels=flow_channels,
        pred_channels=pred_channels,
        weight_decay=weight_decay,
        grad_clip_global_norm=grad_clip,
    )

    model_dir = experiment_dir / f"model_{model_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_paths = _build_standalone_checkpoint_paths(model_dir)
    state_path = experiment_dir / "trainer_state.json"
    eval_history_path = model_dir / "eval_history.csv"
    comparison_path = experiment_dir / "comparison.csv"
    manifest = {
        "model_name": model_name,
        "arch_code": [int(v) for v in arch_code],
        "scratch_init": True,
        "config": config,
        "git_commit": _git_commit_hash(),
    }
    write_json(str(experiment_dir / "run_manifest.json"), manifest)

    eval_history: List[Dict[str, Any]] = _load_csv_rows(eval_history_path)
    comparison_rows: List[Dict[str, Any]] = _load_csv_rows(comparison_path)
    best_epe = float("inf")
    best_sintel_epe = float("inf")
    start_epoch = 1
    global_step = 0

    try:
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            if bool(checkpoint_cfg.get("load_checkpoint", False)):
                resume_root = experiment_dir
                resume_name = str(checkpoint_cfg.get("resume_experiment_name", "")).strip()
                if resume_name:
                    resume_root = experiment_dir.parent / resume_name
                resume_ckpt_name = str(checkpoint_cfg.get("resume_ckpt_name", "last")).strip() or "last"
                resume_ckpt = resume_root / f"model_{model_name}" / "checkpoints" / f"{resume_ckpt_name}.ckpt"
                if not Path(str(resume_ckpt) + ".index").exists():
                    raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")
                graph_obj["saver"].restore(sess, str(resume_ckpt))
                meta_path = Path(str(resume_ckpt) + ".meta.json")
                if meta_path.exists():
                    meta = read_json(str(meta_path))
                    start_epoch = int(meta.get("epoch", 0)) + 1
                    global_step = int(meta.get("global_step", 0))
                    best_epe = float(meta.get("best_metric", float("inf")))
                if state_path.exists():
                    state = read_json(str(state_path))
                    best_sintel_epe = float(state.get("best_sintel_epe", float("inf")))
                logger.info("resume checkpoint=%s start_epoch=%d global_step=%d", resume_ckpt, start_epoch, global_step)

            for epoch_idx in range(start_epoch, num_epochs + 1):
                epoch_start = time.time()
                if hasattr(train_provider, "start_epoch"):
                    train_provider.start_epoch(shuffle=True)
                epoch_loss = 0.0
                optical_loss = 0.0
                uncertainty_loss = 0.0
                grad_norms: List[float] = []
                lr_last = base_lr
                iterator = tqdm(range(steps_per_epoch), total=steps_per_epoch, desc=f"{model_name} epoch {epoch_idx}/{num_epochs}", leave=False)
                for _ in iterator:
                    input_batch, _, _, label_batch = train_provider.next_batch(batch_size=batch_size)
                    input_batch = standardize_image_tensor(input_batch)
                    logical_batch = int(input_batch.shape[0])
                    micro_slices = _iter_micro_slices(logical_batch, micro_batch_size)
                    if not micro_slices:
                        continue
                    lr_now = _cosine_lr_with_min(base_lr, lr_min, global_step, total_steps)
                    lr_last = lr_now
                    sess.run(graph_obj["zero_grad_op"])
                    step_loss = 0.0
                    step_optical = 0.0
                    step_uncertainty = 0.0
                    for micro_slice in micro_slices:
                        micro_input = input_batch[micro_slice]
                        micro_label = label_batch[micro_slice]
                        grad_scale = float(micro_input.shape[0]) / float(logical_batch)
                        result = sess.run(
                            {
                                "loss": graph_obj["loss"],
                                "optical": graph_obj["loss_optical"],
                                "uncertainty": graph_obj["loss_uncertainty"],
                                "accum": graph_obj["accum_op"],
                            },
                            feed_dict={
                                input_ph: micro_input,
                                label_ph: micro_label,
                                lr_ph: lr_now,
                                grad_scale_ph: grad_scale,
                                is_training_ph: True,
                            },
                        )
                        step_loss += float(result["loss"]) * grad_scale
                        step_optical += float(result["optical"]) * grad_scale
                        step_uncertainty += float(result["uncertainty"]) * grad_scale
                    apply_result = sess.run({"grad_norm": graph_obj["grad_norm"], "train": graph_obj["train_op"]}, feed_dict={lr_ph: lr_now})
                    grad_norms.append(float(apply_result["grad_norm"]))
                    epoch_loss += step_loss
                    optical_loss += step_optical
                    uncertainty_loss += step_uncertainty
                    global_step += 1
                    iterator.set_postfix(lr=f"{lr_now:.2e}", loss=f"{epoch_loss / max(1, len(grad_norms)):.4f}")

                avg_loss = epoch_loss / max(1, steps_per_epoch)
                avg_optical = optical_loss / max(1, steps_per_epoch)
                avg_uncertainty = uncertainty_loss / max(1, steps_per_epoch)
                grad_stats = _summarize_grad_norms(grad_norms, grad_clip)
                fc2_epe = float("inf")
                sintel_epe = None
                do_fc2_eval = (epoch_idx % eval_every_epoch == 0) or (epoch_idx == num_epochs)
                if do_fc2_eval:
                    fc2_epe = _evaluate_fc2_with_progress(sess, graph_obj, input_ph, label_ph, is_training_ph, val_provider, batch_size, eval_batches, desc=f"FC2 val {model_name} e{epoch_idx}")

                _save_checkpoint_with_arch(sess, graph_obj["saver"], ckpt_paths["last"], epoch_idx, global_step, fc2_epe, best_epe, arch_code)
                if do_fc2_eval and fc2_epe < best_epe:
                    best_epe = fc2_epe
                    _save_checkpoint_with_arch(sess, graph_obj["saver"], ckpt_paths["best"], epoch_idx, global_step, fc2_epe, best_epe, arch_code)
                    logger.info("best updated model=%s fc2_epe=%.4f", model_name, fc2_epe)

                if sintel_every > 0 and do_fc2_eval and (epoch_idx % sintel_every == 0 or epoch_idx == num_epochs):
                    sintel_result = _run_sintel_if_configured(model_dir=model_dir, config=config, epoch_idx=epoch_idx, ckpt_name="last")
                    if sintel_result is not None:
                        sintel_epe = float(sintel_result["sintel_epe"])
                        if sintel_epe < best_sintel_epe:
                            best_sintel_epe = sintel_epe
                            _save_checkpoint_with_arch(
                                sess,
                                graph_obj["saver"],
                                ckpt_paths["root"] / "sintel_best.ckpt",
                                epoch_idx,
                                global_step,
                                sintel_epe,
                                best_sintel_epe,
                                arch_code,
                            )
                            logger.info("sintel_best updated model=%s sintel_epe=%.4f", model_name, sintel_epe)

                row: Dict[str, Any] = {
                    "epoch": epoch_idx,
                    "global_step": global_step,
                    "lr": lr_last,
                    "loss": avg_loss,
                    "loss_optical": avg_optical,
                    "loss_uncertainty": avg_uncertainty,
                    "fc2_epe": fc2_epe,
                    "best_fc2_epe": best_epe,
                    **{f"grad_norm_{k}": v for k, v in grad_stats.items()},
                }
                if sintel_epe is not None:
                    row["sintel_epe"] = sintel_epe
                    row["best_sintel_epe"] = best_sintel_epe
                eval_history.append(row)
                comparison_rows.append({"model_name": model_name, "arch_code": ",".join(str(v) for v in arch_code), **row})
                _write_csv(eval_history_path, eval_history)
                _write_csv(comparison_path, comparison_rows)
                _save_json_atomic(
                    state_path,
                    {
                        "epoch": epoch_idx,
                        "global_step": global_step,
                        "best_fc2_epe": best_epe,
                        "best_sintel_epe": best_sintel_epe,
                        "model_name": model_name,
                        "arch_code": [int(v) for v in arch_code],
                    },
                )
                logger.info(
                    "epoch=%d/%d global_step=%d lr=%.2e time_sec=%.1f loss=%.6f optical=%.6f uncertainty=%.6f fc2_epe=%s sintel_epe=%s grad_mean=%.4f grad_p90=%.4f clip_rate=%.4f",
                    epoch_idx,
                    num_epochs,
                    global_step,
                    lr_last,
                    time.time() - epoch_start,
                    avg_loss,
                    avg_optical,
                    avg_uncertainty,
                    "" if not np.isfinite(fc2_epe) else f"{fc2_epe:.4f}",
                    "" if sintel_epe is None else f"{sintel_epe:.4f}",
                    grad_stats["mean"],
                    grad_stats["p90"],
                    grad_stats["clip_rate"],
                )
    finally:
        _close_provider(train_provider)
        _close_provider(val_provider)

    logger.info("training complete model=%s best_fc2_epe=%.4f best_sintel_epe=%.4f", model_name, best_epe, best_sintel_epe)
    return 0
