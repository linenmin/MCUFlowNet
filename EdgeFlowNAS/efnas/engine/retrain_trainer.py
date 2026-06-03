"""Stage trainer for fixed-architecture Retrain V3 candidates."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from efnas.data.dataloader_builder import build_fc2_provider, build_ft3d_provider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.distill_or_not_trainer import (
    _build_graph,
    _close_provider,
    _iter_micro_slices,
    _load_csv_rows,
    _run_sintel_if_configured,
    _save_json_atomic,
    _summarize_grad_norms,
    _wrap_prefetch,
    _write_csv,
    parse_arch_code,
)
from efnas.engine.standalone_trainer import (
    _apply_gpu_device_setting,
    _build_standalone_checkpoint_paths,
    _cosine_lr_with_min,
    _git_commit_hash,
    _resolve_output_dir,
    _save_standalone_checkpoint,
)
from efnas.utils.json_io import read_json, write_json
from efnas.utils.logger import build_logger
from efnas.utils.seed import set_global_seed


def _build_provider(config: Dict[str, Any], split: str, seed_offset: int, provider_mode: str):
    dataset = str(config.get("data", {}).get("dataset", "FC2")).strip().upper()
    if dataset == "FC2":
        return build_fc2_provider(config=config, split=split, seed_offset=seed_offset, provider_mode=provider_mode)
    if dataset == "FT3D":
        return build_ft3d_provider(config=config, split=split, seed_offset=seed_offset, provider_mode=provider_mode)
    raise ValueError(f"unsupported retrain_v3 dataset: {dataset}")


def _evaluate_with_progress(
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


def _resolve_init_checkpoint_path(config: Dict[str, Any], model_name: str) -> Optional[Path]:
    checkpoint_cfg = config.get("checkpoint", {})
    init_mode = str(checkpoint_cfg.get("init_mode", "")).strip().lower()
    if not init_mode or init_mode == "none":
        return None
    if init_mode != "experiment_dir":
        raise ValueError(f"unsupported retrain_v3 checkpoint.init_mode: {init_mode}")
    exp_dir = str(checkpoint_cfg.get("init_experiment_dir", "")).strip()
    ckpt_name = str(checkpoint_cfg.get("init_ckpt_name", "best")).strip() or "best"
    if not exp_dir:
        raise ValueError("checkpoint.init_experiment_dir is required when init_mode=experiment_dir")
    return Path(exp_dir) / f"model_{model_name}" / "checkpoints" / f"{ckpt_name}.ckpt"


def _resolve_resume_root(config: Dict[str, Any], experiment_dir: Path) -> Path:
    checkpoint_cfg = config.get("checkpoint", {})
    resume_name = str(checkpoint_cfg.get("resume_experiment_name", "")).strip()
    if not resume_name:
        return experiment_dir
    return experiment_dir.parent / resume_name


def _model_weight_vars(scope_name: str) -> List[tf.Variable]:
    vars_: List[tf.Variable] = []
    for var in tf.compat.v1.global_variables():
        if not var.name.startswith(f"{scope_name}/"):
            continue
        name = var.op.name
        if "/Adam" in name or name.endswith("/Adam") or name.endswith("/Adam_1"):
            continue
        if name in (f"{scope_name}/beta1_power", f"{scope_name}/beta2_power"):
            continue
        vars_.append(var)
    return vars_


def _checkpoint_exists(path_prefix: Path) -> bool:
    return Path(str(path_prefix) + ".index").exists()


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(str(path))
    return payload if isinstance(payload, dict) else {}


def train_retrain_v3(config: Dict[str, Any]) -> int:
    """Run one fixed V3 subnet retrain stage."""
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("eval", {})
    checkpoint_cfg = config.get("checkpoint", {})
    dataset = str(data_cfg.get("dataset", "FC2")).strip().upper()

    seed = int(runtime_cfg.get("seed", 42))
    set_global_seed(seed)
    experiment_dir = _resolve_output_dir(config)
    model_name = str(config.get("model_name", "candidate")).strip() or "candidate"
    model_dir = experiment_dir / f"model_{model_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(f"retrain_v3_{model_name}", str(model_dir / "train.log"))
    logger.info("start retrain_v3 dataset=%s model=%s", dataset, model_name)
    _apply_gpu_device_setting(train_cfg=train_cfg, logger=logger)

    arch_code = parse_arch_code(config.get("arch_code"))
    batch_size = int(train_cfg.get("batch_size", 32))
    micro_batch_size = min(int(train_cfg.get("micro_batch_size", batch_size)), batch_size)
    num_epochs = int(train_cfg.get("num_epochs", 400))
    base_lr = float(train_cfg.get("lr", 1e-4))
    lr_min = float(train_cfg.get("lr_min", 1e-6))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip_global_norm", 200.0))
    eval_every_epoch = max(1, int(eval_cfg.get("eval_every_epoch", 1)))
    eval_batches = int(eval_cfg.get("eval_batches", 0))
    sintel_cfg = eval_cfg.get("sintel", {}) if isinstance(eval_cfg.get("sintel", {}), dict) else {}
    sintel_every = int(sintel_cfg.get("eval_every_epoch", 0) or 0)
    input_h = int(data_cfg.get("input_height", 352))
    input_w = int(data_cfg.get("input_width", 480))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = flow_channels * 2

    train_provider = _build_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    val_provider = _build_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
    if len(train_provider) == 0:
        raise RuntimeError("train split is empty")
    if len(val_provider) == 0:
        raise RuntimeError("val split is empty")
    train_provider = _wrap_prefetch(train_provider, int(data_cfg.get("prefetch_batches", 0)))
    val_provider = _wrap_prefetch(val_provider, int(data_cfg.get("eval_prefetch_batches", 0)))
    steps_per_epoch = int(math.ceil(len(train_provider) / float(max(1, batch_size))))
    total_steps = max(1, steps_per_epoch * num_epochs)

    logger.info("arch=%s", ",".join(str(v) for v in arch_code))
    logger.info("input=%dx%d batch=%d micro_batch=%d epochs=%d steps_per_epoch=%d", input_h, input_w, batch_size, micro_batch_size, num_epochs, steps_per_epoch)
    logger.info("lr=%.2e lr_min=%.2e weight_decay=%.2e grad_clip=%.1f", base_lr, lr_min, weight_decay, grad_clip)
    logger.info("eval_every=%d sintel_every=%d prefetch train/eval=%s/%s", eval_every_epoch, sintel_every, data_cfg.get("prefetch_batches"), data_cfg.get("eval_prefetch_batches"))

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
    init_saver = tf.compat.v1.train.Saver(var_list=_model_weight_vars(model_name))

    ckpt_paths = _build_standalone_checkpoint_paths(model_dir)
    state_path = model_dir / "trainer_state.json"
    eval_history_path = model_dir / "eval_history.csv"
    write_json(
        str(model_dir / "run_manifest.json"),
        {
            "model_name": model_name,
            "arch_code": [int(v) for v in arch_code],
            "dataset": dataset,
            "scratch_init": str(checkpoint_cfg.get("init_mode", "none")).strip().lower() in ("", "none"),
            "config": config,
            "git_commit": _git_commit_hash(),
        },
    )

    eval_history = _load_csv_rows(eval_history_path)
    best_epe = float("inf")
    best_sintel_epe = float("inf")
    start_epoch = 1
    global_step = 0

    try:
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            if bool(checkpoint_cfg.get("load_checkpoint", False)):
                resume_root = _resolve_resume_root(config=config, experiment_dir=experiment_dir)
                resume_ckpt_name = str(checkpoint_cfg.get("resume_ckpt_name", "last")).strip() or "last"
                resume_ckpt = resume_root / f"model_{model_name}" / "checkpoints" / f"{resume_ckpt_name}.ckpt"
                if not _checkpoint_exists(resume_ckpt):
                    raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")
                graph_obj["saver"].restore(sess, str(resume_ckpt))
                meta_path = Path(str(resume_ckpt) + ".meta.json")
                if meta_path.exists():
                    meta = read_json(str(meta_path))
                    start_epoch = int(meta.get("epoch", 0)) + 1
                    global_step = int(meta.get("global_step", 0))
                    best_epe = float(meta.get("best_metric", float("inf")))
                state = _load_state(resume_root / f"model_{model_name}" / "trainer_state.json")
                if state:
                    best_sintel_epe = float(state.get("best_sintel_epe", best_sintel_epe))
                logger.info("resume checkpoint=%s start_epoch=%d global_step=%d", resume_ckpt, start_epoch, global_step)
            else:
                init_ckpt = _resolve_init_checkpoint_path(config=config, model_name=model_name)
                if init_ckpt is not None:
                    if not _checkpoint_exists(init_ckpt):
                        raise FileNotFoundError(f"init checkpoint not found: {init_ckpt}")
                    init_saver.restore(sess, str(init_ckpt))
                    logger.info("initialized model weights from %s", init_ckpt)

            for epoch_idx in range(start_epoch, num_epochs + 1):
                epoch_start = time.time()
                if hasattr(train_provider, "start_epoch"):
                    train_provider.start_epoch(shuffle=True)
                epoch_loss = 0.0
                optical_loss = 0.0
                uncertainty_loss = 0.0
                grad_norms: List[float] = []
                lr_last = base_lr
                desc = f"{model_name} {dataset} epoch {epoch_idx}/{num_epochs}"
                iterator = tqdm(range(steps_per_epoch), total=steps_per_epoch, desc=desc, leave=False)
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
                do_eval = (epoch_idx % eval_every_epoch == 0) or (epoch_idx == num_epochs)
                val_epe = float("inf")
                sintel_epe = None
                if do_eval:
                    val_epe = _evaluate_with_progress(
                        sess,
                        graph_obj,
                        input_ph,
                        label_ph,
                        is_training_ph,
                        val_provider,
                        batch_size,
                        eval_batches,
                        desc=f"{dataset} val {model_name} e{epoch_idx}",
                    )

                _save_standalone_checkpoint(sess, graph_obj["saver"], ckpt_paths["last"], epoch_idx, global_step, val_epe, best_epe, arch_code)
                if do_eval and val_epe < best_epe:
                    best_epe = val_epe
                    _save_standalone_checkpoint(sess, graph_obj["saver"], ckpt_paths["best"], epoch_idx, global_step, val_epe, best_epe, arch_code)
                    logger.info("best updated val_epe=%.4f", val_epe)

                if sintel_every > 0 and do_eval and (epoch_idx % sintel_every == 0 or epoch_idx == num_epochs):
                    sintel_result = _run_sintel_if_configured(model_dir=model_dir, config=config, epoch_idx=epoch_idx, ckpt_name="last")
                    if sintel_result is not None:
                        sintel_epe = float(sintel_result["sintel_epe"])
                        if sintel_epe < best_sintel_epe:
                            best_sintel_epe = sintel_epe
                            _save_standalone_checkpoint(
                                sess,
                                graph_obj["saver"],
                                ckpt_paths["root"] / "sintel_best.ckpt",
                                epoch_idx,
                                global_step,
                                sintel_epe,
                                best_sintel_epe,
                                arch_code,
                            )
                            logger.info("sintel_best updated sintel_epe=%.4f", sintel_epe)

                row: Dict[str, Any] = {
                    "epoch": epoch_idx,
                    "global_step": global_step,
                    "lr": lr_last,
                    "loss": avg_loss,
                    "loss_optical": avg_optical,
                    "loss_uncertainty": avg_uncertainty,
                    "val_epe": val_epe,
                    "best_val_epe": best_epe,
                    **{f"grad_norm_{k}": v for k, v in grad_stats.items()},
                }
                if sintel_epe is not None:
                    row["sintel_epe"] = sintel_epe
                    row["best_sintel_epe"] = best_sintel_epe
                eval_history.append(row)
                _write_csv(eval_history_path, eval_history)
                _save_json_atomic(
                    state_path,
                    {
                        "epoch": epoch_idx,
                        "global_step": global_step,
                        "best_val_epe": best_epe,
                        "best_sintel_epe": best_sintel_epe,
                        "model_name": model_name,
                        "arch_code": [int(v) for v in arch_code],
                        "dataset": dataset,
                    },
                )
                logger.info(
                    "epoch=%d/%d global_step=%d lr=%.2e time_sec=%.1f loss=%.6f optical=%.6f uncertainty=%.6f val_epe=%s sintel_epe=%s grad_mean=%.4f grad_p90=%.4f clip_rate=%.4f",
                    epoch_idx,
                    num_epochs,
                    global_step,
                    lr_last,
                    time.time() - epoch_start,
                    avg_loss,
                    avg_optical,
                    avg_uncertainty,
                    "" if not np.isfinite(val_epe) else f"{val_epe:.4f}",
                    "" if sintel_epe is None else f"{sintel_epe:.4f}",
                    grad_stats["mean"],
                    grad_stats["p90"],
                    grad_stats["clip_rate"],
                )
    finally:
        _close_provider(train_provider)
        _close_provider(val_provider)

    logger.info("training complete model=%s dataset=%s best_val_epe=%.4f best_sintel_epe=%.4f", model_name, dataset, best_epe, best_sintel_epe)
    return 0
