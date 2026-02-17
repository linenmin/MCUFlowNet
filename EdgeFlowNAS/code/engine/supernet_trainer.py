"""Supernet training engine."""

import csv
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from code.data.dataloader_builder import build_fc2_provider
from code.data.transforms_180x240 import standardize_image_tensor
from code.engine.bn_recalibration import run_bn_recalibration_session
from code.engine.checkpoint_manager import (
    build_checkpoint_paths,
    find_existing_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from code.engine.early_stop import EarlyStopState, update_early_stop
from code.engine.eval_step import build_epe_metric
from code.engine.train_step import add_weight_decay, build_multiscale_uncertainty_loss
from code.nas.eval_pool_builder import build_eval_pool, check_eval_pool_coverage
from code.nas.fair_sampler import generate_fair_cycle
from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet
from code.optim.lr_scheduler import cosine_lr
from code.utils.json_io import write_json
from code.utils.logger import build_logger
from code.utils.manifest import build_manifest
from code.utils.path_utils import ensure_directory, project_root
from code.utils.seed import set_global_seed


def _resolve_output_dir(config: Dict[str, Any]) -> Path:
    """Resolve experiment output directory."""
    runtime_cfg = config.get("runtime", {})
    output_root = runtime_cfg.get("output_root", "outputs/supernet")
    experiment_name = runtime_cfg.get("experiment_name", "default_experiment")
    root_path = project_root() / output_root / experiment_name
    return ensure_directory(str(root_path))


def _resolve_resume_dir(config: Dict[str, Any], experiment_dir: Path) -> Path:
    """Resolve checkpoint source directory for resume."""
    runtime_cfg = config.get("runtime", {})
    checkpoint_cfg = config.get("checkpoint", {})
    resume_name = str(checkpoint_cfg.get("resume_experiment_name", "")).strip()
    if not resume_name:
        return experiment_dir
    output_root = runtime_cfg.get("output_root", "outputs/supernet")
    return project_root() / output_root / resume_name


def _git_commit_hash() -> str:
    """Return current git commit hash."""
    try:
        raw = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root()))
    except Exception:
        return "unknown"
    return raw.decode("utf-8").strip()


def _init_fairness_counts(num_blocks: int = 9) -> Dict[str, Dict[str, int]]:
    """Initialize strict fairness counters."""
    counts: Dict[str, Dict[str, int]] = {}
    for block_idx in range(num_blocks):
        counts[str(block_idx)] = {"0": 0, "1": 0, "2": 0}
    return counts


def _sanitize_fairness_counts(raw_counts: Any, num_blocks: int = 9) -> Dict[str, Dict[str, int]]:
    """Normalize fairness counters loaded from checkpoint metadata."""
    clean = _init_fairness_counts(num_blocks=num_blocks)
    if not isinstance(raw_counts, dict):
        return clean
    for block_idx in range(num_blocks):
        block_key = str(block_idx)
        block_raw = raw_counts.get(block_key, raw_counts.get(block_idx, {}))
        if not isinstance(block_raw, dict):
            continue
        for option in (0, 1, 2):
            option_key = str(option)
            raw_value = block_raw.get(option_key, block_raw.get(option, 0))
            try:
                clean[block_key][option_key] = int(raw_value)
            except Exception:
                clean[block_key][option_key] = 0
    return clean


def _update_fairness_counts(counts: Dict[str, Dict[str, int]], cycle_codes: List[List[int]]) -> None:
    """Accumulate fairness counters for one cycle."""
    for arch_code in cycle_codes:
        for block_idx, option in enumerate(arch_code):
            counts[str(block_idx)][str(int(option))] += 1


def _fairness_gap(counts: Dict[str, Dict[str, int]]) -> int:
    """Return max option-count gap across all blocks."""
    gap = 0
    for block_counts in counts.values():
        values = list(block_counts.values())
        gap = max(gap, max(values) - min(values))
    return int(gap)


def _write_eval_history(csv_path: Path, rows: List[Dict[str, float]]) -> None:
    """Write epoch-level eval history."""
    headers = [
        "epoch",
        "mean_epe_12",
        "std_epe_12",
        "fairness_gap",
        "lr",
        "bn_recal_batches",
        "train_loss_epoch_avg",
        "train_grad_norm_epoch_avg",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_graph(config: Dict[str, Any]) -> Dict[str, object]:
    """Build TF1 supernet training graph."""
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    input_h = int(data_cfg.get("input_height", 180))
    input_w = int(data_cfg.get("input_width", 240))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = int(flow_channels * 2)

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, flow_channels], name="Label")
    arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[9], name="ArchCode")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")
    accum_divisor_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="AccumDivisor")

    model = MultiScaleResNetSupernet(
        input_ph=input_ph,
        arch_code_ph=arch_code_ph,
        is_training_ph=is_training_ph,
        num_out=pred_channels,
        init_neurons=32,
        expansion_factor=2.0,
    )
    preds = model.build()

    loss_tensor = build_multiscale_uncertainty_loss(preds=preds, label_ph=label_ph, num_out=flow_channels)
    loss_tensor = add_weight_decay(loss_tensor=loss_tensor, weight_decay=float(train_cfg.get("weight_decay", 0.0)))

    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=lr_ph,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    grads_and_vars = optimizer.compute_gradients(loss_tensor)
    grads = [grad for grad, _ in grads_and_vars if grad is not None]
    vars_ = [var for grad, var in grads_and_vars if grad is not None]
    clipped, global_norm = tf.clip_by_global_norm(grads, clip_norm=float(train_cfg.get("grad_clip_global_norm", 5.0)))

    accum_vars = []
    zero_ops = []
    add_ops = []
    for idx, (grad, var) in enumerate(zip(clipped, vars_)):
        accum_var = tf.compat.v1.get_variable(
            name=f"strict_accum_{idx}",
            shape=var.shape,
            dtype=var.dtype.base_dtype,
            initializer=tf.zeros_initializer(),
            trainable=False,
        )
        accum_vars.append(accum_var)
        zero_ops.append(tf.compat.v1.assign(accum_var, tf.zeros_like(accum_var), name=f"strict_zero_{idx}"))
        add_ops.append(tf.compat.v1.assign_add(accum_var, grad, name=f"strict_add_{idx}"))

    bn_updates = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(add_ops + bn_updates):
        accum_op = tf.no_op(name="strict_accum_done")

    avg_divisor = tf.maximum(accum_divisor_ph, 1.0, name="strict_avg_divisor")
    avg_grads = [accum_var / avg_divisor for accum_var in accum_vars]
    apply_op = optimizer.apply_gradients(list(zip(avg_grads, vars_)), name="strict_apply")

    epe_tensor = build_epe_metric(pred_tensor=preds[-1], label_ph=label_ph, num_out=flow_channels)
    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    return {
        "input_ph": input_ph,
        "label_ph": label_ph,
        "arch_code_ph": arch_code_ph,
        "is_training_ph": is_training_ph,
        "lr_ph": lr_ph,
        "accum_divisor_ph": accum_divisor_ph,
        "preds": preds,
        "loss": loss_tensor,
        "epe": epe_tensor,
        "global_grad_norm": global_norm,
        "zero_ops": zero_ops,
        "accum_op": accum_op,
        "apply_op": apply_op,
        "saver": saver,
    }


def _run_eval_epoch(
    sess,
    graph_obj: Dict[str, object],
    train_provider,
    val_provider,
    eval_pool: List[List[int]],
    bn_recal_batches: int,
    batch_size: int,
) -> Dict[str, float]:
    """Run one eval epoch on a fixed architecture pool."""
    epe_values = []
    for arch_code in eval_pool:
        run_bn_recalibration_session(
            sess=sess,
            forward_fetch=graph_obj["preds"][-1],
            input_ph=graph_obj["input_ph"],
            label_ph=graph_obj["label_ph"],
            arch_code_ph=graph_obj["arch_code_ph"],
            is_training_ph=graph_obj["is_training_ph"],
            batch_provider=train_provider,
            arch_code=arch_code,
            batch_size=batch_size,
            recal_batches=bn_recal_batches,
        )
        input_batch, _, _, label_batch = val_provider.next_batch(batch_size=batch_size)
        input_batch = standardize_image_tensor(input_batch)
        epe_val = sess.run(
            graph_obj["epe"],
            feed_dict={
                graph_obj["input_ph"]: input_batch,
                graph_obj["label_ph"]: label_batch,
                graph_obj["arch_code_ph"]: arch_code,
                graph_obj["is_training_ph"]: False,
            },
        )
        epe_values.append(float(epe_val))
    mean_epe = float(np.mean(epe_values)) if epe_values else 0.0
    std_epe = float(np.std(epe_values)) if epe_values else 0.0
    return {"mean_epe_12": mean_epe, "std_epe_12": std_epe}


def _default_restore_state() -> Dict[str, Any]:
    """Default state when not resuming."""
    return {
        "start_epoch": 1,
        "global_step": 0,
        "fairness_counts": _init_fairness_counts(num_blocks=9),
        "best_metric": float("inf"),
        "bad_epochs": 0,
        "last_metric": float("inf"),
    }


def _try_restore_training_state(
    sess,
    saver: tf.compat.v1.train.Saver,
    config: Dict[str, Any],
    experiment_dir: Path,
    logger,
) -> Dict[str, Any]:
    """Restore checkpoint and resume metadata if enabled."""
    checkpoint_cfg = config.get("checkpoint", {})
    if not bool(checkpoint_cfg.get("load_checkpoint", False)):
        return _default_restore_state()

    resume_dir = _resolve_resume_dir(config=config, experiment_dir=experiment_dir)
    resume_paths = build_checkpoint_paths(str(resume_dir))
    resume_prefix = find_existing_checkpoint(path_prefix=resume_paths["last"])
    if resume_prefix is None:
        logger.warning("load_checkpoint=true but no checkpoint found in %s", str(resume_dir))
        return _default_restore_state()

    restore_info = restore_checkpoint(sess=sess, saver=saver, path_prefix=resume_prefix)
    meta = restore_info.get("meta", {}) if isinstance(restore_info, dict) else {}

    start_epoch = int(meta.get("epoch", 0)) + 1
    global_step = int(meta.get("global_step", 0))
    fairness_counts = _sanitize_fairness_counts(raw_counts=meta.get("fairness_counts", {}), num_blocks=9)
    best_metric = float(meta.get("best_metric", float("inf")))
    bad_epochs = int(meta.get("bad_epochs", 0))
    last_metric = float(meta.get("metric", best_metric))

    if bool(checkpoint_cfg.get("reset_early_stop_on_resume", False)):
        best_metric = float("inf")
        bad_epochs = 0
        last_metric = float("inf")
        logger.info("reset early-stop state on resume")

    logger.info("resume checkpoint=%s start_epoch=%d global_step=%d", str(resume_prefix), start_epoch, global_step)
    return {
        "start_epoch": start_epoch,
        "global_step": global_step,
        "fairness_counts": fairness_counts,
        "best_metric": best_metric,
        "bad_epochs": bad_epochs,
        "last_metric": last_metric,
    }


def train_supernet(config: Dict[str, Any]) -> int:
    """Run strict-fairness supernet training."""
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})

    seed = int(runtime_cfg.get("seed", 42))
    set_global_seed(seed)

    experiment_dir = _resolve_output_dir(config)
    logger = build_logger("edgeflownas_supernet", str(experiment_dir / "train.log"))
    logger.info("start strict-fairness supernet training")

    batch_size = int(train_cfg.get("batch_size", 32))
    micro_batch_size = int(train_cfg.get("micro_batch_size", batch_size))
    if micro_batch_size <= 0:
        micro_batch_size = batch_size
    micro_batch_size = min(micro_batch_size, batch_size)
    logger.info("batch_size=%d micro_batch_size=%d", batch_size, micro_batch_size)

    num_epochs = int(train_cfg.get("num_epochs", 200))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 50))
    base_lr = float(train_cfg.get("lr", 1e-4))
    bn_recal_batches = int(eval_cfg.get("bn_recal_batches", 8))
    eval_pool_size = int(eval_cfg.get("eval_pool_size", 12))
    eval_every_epoch = max(1, int(eval_cfg.get("eval_every_epoch", 1)))
    patience = int(eval_cfg.get("early_stop_patience", 15))
    min_delta = float(eval_cfg.get("early_stop_min_delta", 0.002))
    logger.info("eval_every_epoch=%d", eval_every_epoch)

    train_provider = build_fc2_provider(config=config, split="train", seed_offset=0)
    val_provider = build_fc2_provider(config=config, split="val", seed_offset=999)
    logger.info("train_source=%s train_samples=%d", train_provider.source_dir, len(train_provider))
    logger.info("val_source=%s val_samples=%d", val_provider.source_dir, len(val_provider))

    if len(train_provider) == 0:
        raise RuntimeError("train sample count is 0; aborting")
    if len(val_provider) == 0:
        raise RuntimeError("val sample count is 0; aborting")

    eval_pool = build_eval_pool(seed=seed, size=eval_pool_size, num_blocks=9)
    eval_pool_cov = check_eval_pool_coverage(pool=eval_pool, num_blocks=9)
    write_json(str(experiment_dir / f"eval_pool_{eval_pool_size}.json"), {"pool": eval_pool, "coverage": eval_pool_cov})

    graph_obj = _build_graph(config=config)
    early_stop = EarlyStopState()
    checkpoint_paths = build_checkpoint_paths(str(experiment_dir))
    eval_rows = []
    sampler_rng = random.Random(seed)
    total_steps = max(1, num_epochs * max(1, steps_per_epoch))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        restore_state = _try_restore_training_state(
            sess=sess,
            saver=graph_obj["saver"],
            config=config,
            experiment_dir=experiment_dir,
            logger=logger,
        )
        start_epoch = int(restore_state["start_epoch"])
        global_step = int(restore_state["global_step"])
        fairness_counts = _sanitize_fairness_counts(raw_counts=restore_state["fairness_counts"], num_blocks=9)
        early_stop.best_metric = float(restore_state["best_metric"])
        early_stop.bad_epochs = int(restore_state["bad_epochs"])
        last_eval_metric = float(restore_state.get("last_metric", early_stop.best_metric))
        if not np.isfinite(last_eval_metric):
            last_eval_metric = float(early_stop.best_metric) if np.isfinite(early_stop.best_metric) else float("inf")
        epochs_ran = 0

        for epoch_idx in range(start_epoch, num_epochs + 1):
            epochs_ran += 1
            step_iterator = tqdm(
                range(steps_per_epoch),
                total=steps_per_epoch,
                desc=f"train epoch {epoch_idx}/{num_epochs}",
                leave=False,
            )
            epoch_loss_sum = 0.0
            epoch_grad_norm_sum = 0.0
            epoch_step_count = 0
            for _ in step_iterator:
                cycle_codes = generate_fair_cycle(rng=sampler_rng, num_blocks=9)
                _update_fairness_counts(counts=fairness_counts, cycle_codes=cycle_codes)

                input_batch, _, _, label_batch = train_provider.next_batch(batch_size=batch_size)
                input_batch = standardize_image_tensor(input_batch)

                current_lr = cosine_lr(base_lr=base_lr, step_idx=global_step, total_steps=total_steps)
                micro_slices = [
                    (start, min(start + micro_batch_size, batch_size))
                    for start in range(0, batch_size, micro_batch_size)
                ]
                accum_runs = int(len(cycle_codes) * len(micro_slices))

                sess.run(graph_obj["zero_ops"])
                loss_val = 0.0
                grad_norm_val = 0.0

                for arch_idx, arch_code in enumerate(cycle_codes):
                    for micro_idx, (start_idx, end_idx) in enumerate(micro_slices):
                        feed = {
                            graph_obj["input_ph"]: input_batch[start_idx:end_idx],
                            graph_obj["label_ph"]: label_batch[start_idx:end_idx],
                            graph_obj["arch_code_ph"]: arch_code,
                            graph_obj["is_training_ph"]: True,
                            graph_obj["lr_ph"]: current_lr,
                        }
                        is_last_accum = bool(
                            arch_idx == len(cycle_codes) - 1 and micro_idx == len(micro_slices) - 1
                        )
                        if is_last_accum:
                            loss_val, grad_norm_val, _ = sess.run(
                                [graph_obj["loss"], graph_obj["global_grad_norm"], graph_obj["accum_op"]],
                                feed_dict=feed,
                            )
                        else:
                            sess.run(graph_obj["accum_op"], feed_dict=feed)

                sess.run(
                    graph_obj["apply_op"],
                    feed_dict={
                        graph_obj["lr_ph"]: current_lr,
                        graph_obj["accum_divisor_ph"]: float(accum_runs),
                    },
                )
                global_step += 1
                epoch_loss_sum += float(loss_val)
                epoch_grad_norm_sum += float(grad_norm_val)
                epoch_step_count += 1
                step_iterator.set_postfix(loss=f"{float(loss_val):.4f}", lr=f"{float(current_lr):.2e}")

            step_iterator.close()
            train_loss_epoch_avg = epoch_loss_sum / max(1, epoch_step_count)
            train_grad_norm_epoch_avg = epoch_grad_norm_sum / max(1, epoch_step_count)
            should_eval = bool(epoch_idx % eval_every_epoch == 0)
            if should_eval:
                eval_info = _run_eval_epoch(
                    sess=sess,
                    graph_obj=graph_obj,
                    train_provider=train_provider,
                    val_provider=val_provider,
                    eval_pool=eval_pool,
                    bn_recal_batches=bn_recal_batches,
                    batch_size=batch_size,
                )
                row = {
                    "epoch": int(epoch_idx),
                    "mean_epe_12": float(eval_info["mean_epe_12"]),
                    "std_epe_12": float(eval_info["std_epe_12"]),
                    "fairness_gap": float(_fairness_gap(fairness_counts)),
                    "lr": float(cosine_lr(base_lr=base_lr, step_idx=global_step, total_steps=total_steps)),
                    "bn_recal_batches": float(bn_recal_batches),
                    "train_loss_epoch_avg": float(train_loss_epoch_avg),
                    "train_grad_norm_epoch_avg": float(train_grad_norm_epoch_avg),
                }
                eval_rows.append(row)
                improved = update_early_stop(state=early_stop, metric=row["mean_epe_12"], min_delta=min_delta)
                last_eval_metric = float(row["mean_epe_12"])
                save_checkpoint(
                    sess=sess,
                    saver=graph_obj["saver"],
                    path_prefix=checkpoint_paths["last"],
                    epoch=epoch_idx,
                    metric=last_eval_metric,
                    global_step=global_step,
                    best_metric=early_stop.best_metric,
                    bad_epochs=early_stop.bad_epochs,
                    fairness_counts=fairness_counts,
                    extra_payload={"row": row},
                )
                if improved:
                    save_checkpoint(
                        sess=sess,
                        saver=graph_obj["saver"],
                        path_prefix=checkpoint_paths["best"],
                        epoch=epoch_idx,
                        metric=last_eval_metric,
                        global_step=global_step,
                        best_metric=early_stop.best_metric,
                        bad_epochs=early_stop.bad_epochs,
                        fairness_counts=fairness_counts,
                        extra_payload={"row": row},
                    )
                logger.info(
                    "epoch=%d loss=%.6f mean_epe_12=%.6f std_epe_12=%.6f fairness_gap=%.2f grad_norm=%.4f",
                    epoch_idx,
                    float(train_loss_epoch_avg),
                    row["mean_epe_12"],
                    row["std_epe_12"],
                    row["fairness_gap"],
                    float(train_grad_norm_epoch_avg),
                )
                if early_stop.bad_epochs >= patience:
                    logger.info("early stop triggered at epoch=%d", epoch_idx)
                    break
            else:
                save_checkpoint(
                    sess=sess,
                    saver=graph_obj["saver"],
                    path_prefix=checkpoint_paths["last"],
                    epoch=epoch_idx,
                    metric=float(last_eval_metric),
                    global_step=global_step,
                    best_metric=early_stop.best_metric,
                    bad_epochs=early_stop.bad_epochs,
                    fairness_counts=fairness_counts,
                    extra_payload={
                        "eval_skipped": True,
                        "eval_every_epoch": int(eval_every_epoch),
                        "train_loss_epoch_avg": float(train_loss_epoch_avg),
                        "train_grad_norm_epoch_avg": float(train_grad_norm_epoch_avg),
                    },
                )
                logger.info(
                    "epoch=%d loss=%.6f eval=skipped fairness_gap=%.2f grad_norm=%.4f",
                    epoch_idx,
                    float(train_loss_epoch_avg),
                    float(_fairness_gap(fairness_counts)),
                    float(train_grad_norm_epoch_avg),
                )

    _write_eval_history(csv_path=experiment_dir / "eval_epe_history.csv", rows=eval_rows)
    write_json(str(experiment_dir / "fairness_counts.json"), fairness_counts)

    manifest = build_manifest(config=config, git_commit=_git_commit_hash())
    write_json(str(experiment_dir / "train_manifest.json"), manifest)

    report_path = experiment_dir / "supernet_training_report.md"
    report_path.write_text(
        "# Supernet Training Report\n\n"
        f"- epochs_finished: {epochs_ran}\n"
        f"- eval_epochs: {len(eval_rows)}\n"
        f"- eval_every_epoch: {eval_every_epoch}\n"
        f"- best_metric: {early_stop.best_metric}\n"
        f"- final_fairness_gap: {_fairness_gap(fairness_counts)}\n"
        f"- eval_pool_coverage_ok: {bool(eval_pool_cov['ok'])}\n"
        f"- checkpoint_best: {checkpoint_paths['best']}\n"
        f"- checkpoint_last: {checkpoint_paths['last']}\n",
        encoding="utf-8",
    )

    logger.info("strict-fairness supernet training finished")
    return 0
