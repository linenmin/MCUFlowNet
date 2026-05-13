"""Deploy-resolution fine-tune trainer for retrain_v3 subnets + EdgeFlowNet mainline.

Fine-tunes a pretrained checkpoint at the board deploy resolution (e.g.
157×203) on FT3D to fix the train (480×640) → deploy (157×203) input-size
mismatch documented in
`plan/retrain_v3_deploy_ft/findings.md` §7d (cross-reference: Seeed
`plan/MCUFlowNet_Deployment/findings.md` §7d).

Two model families are supported via `config["arch_family"]`:

* `fixed_v3`: the 3 NSGA-II V3 subnets, scope-wrapped under `<model_name>/`
  to match the retrain_v3 ckpt layout. Reuses
  `distill_or_not_trainer._build_graph`. BN runs with `is_training_ph=True`
  (UPDATE_OPS captured via `control_dependencies`).

* `edgeflownet_mainline`: the original EdgeFlowNet transpose-conv network
  (`efnas.network.edgeflownet_mainline.EdgeFlowNetMainline`). Forward vars
  live at root scope (`EncoderDecoderBlock0/...`) to match the published
  `EdgeFlowNet/checkpoints/best.ckpt`. BN is left in inference mode (no
  `training=` arg), matching how the published checkpoint was trained.
  Only conv kernels + BN γ/β are updated during fine-tune.

Differences from `retrain_v3_trainer`:
- Constant LR by default (`train.lr_schedule: constant`), no cosine.
- Early stopping on FT3D val EPE (`train.early_stop_patience`).
- Always saves `sintel_best.ckpt` (in addition to `best.ckpt`) for
  diagnostic reference, but ckpt selection metric remains FT3D val EPE.
- Supports `checkpoint.init_mode: explicit_path` with arbitrary
  `init_ckpt_path` (needed for mainline whose ckpt sits outside the
  EdgeFlowNAS outputs tree).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from efnas.data.dataloader_builder import build_fc2_provider, build_ft3d_provider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.deploy_ft_sintel_runtime import run_sintel_for_deploy_ft
from efnas.engine.distill_or_not_trainer import (
    _build_graph as _build_v3_graph_impl,
    _close_provider,
    _iter_micro_slices,
    _load_csv_rows,
    _save_json_atomic,
    _summarize_grad_norms,
    _wrap_prefetch,
    _write_csv,
    parse_arch_code,
)
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
from efnas.network.edgeflownet_mainline import EdgeFlowNetMainline, MAINLINE_DEFAULT_CONFIG
from efnas.utils.json_io import read_json, write_json
from efnas.utils.logger import build_logger
from efnas.utils.seed import set_global_seed


ARCH_FAMILY_V3 = "fixed_v3"
ARCH_FAMILY_MAINLINE = "edgeflownet_mainline"


# ----------------------------------------------------------------------------
# Mainline graph (no outer scope; BN in inference mode)
# ----------------------------------------------------------------------------
def _build_mainline_graph(
    input_ph,
    label_ph,
    lr_ph,
    grad_scale_ph,
    flow_channels: int,
    pred_channels: int,
    weight_decay: float,
    grad_clip_global_norm: float,
) -> Dict[str, Any]:
    """Build mainline forward + optimizer at root scope.

    Returned dict mirrors `_build_v3_graph_impl` for unified trainer code."""
    model = EdgeFlowNetMainline(InputPH=input_ph, **MAINLINE_DEFAULT_CONFIG)
    preds = model.build()  # [low, mid, full], each [B, h, w, 4]

    loss_terms = build_multiscale_uncertainty_loss(
        preds=preds, label_ph=label_ph, num_out=flow_channels, return_terms=True
    )
    loss_core = loss_terms["total"]

    # Forward weights = everything not in the optimizer scope.
    fwd_vars = [v for v in tf.compat.v1.trainable_variables()
                if not v.name.startswith("Optimizer/")]
    if not fwd_vars:
        raise RuntimeError("mainline graph has no trainable variables")

    loss_tensor = add_weight_decay(
        loss_tensor=loss_core, weight_decay=weight_decay, trainable_vars=fwd_vars
    )

    with tf.compat.v1.variable_scope("Optimizer"):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr_ph, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
        scaled_loss = loss_tensor * grad_scale_ph
        grads_and_vars = optimizer.compute_gradients(scaled_loss, var_list=fwd_vars)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        vars_ = [v for _, v in grads_and_vars]
        accum_vars = [
            tf.Variable(
                tf.zeros(shape=v.shape, dtype=v.dtype.base_dtype),
                trainable=False,
                name=f"{v.op.name.replace('/', '_')}_grad_accum",
            )
            for v in vars_
        ]
        zero_grad_op = (
            tf.group(*[var.assign(tf.zeros_like(var)) for var in accum_vars],
                     name="zero_gradients")
            if accum_vars
            else tf.no_op()
        )
        # NOTE: mainline BN is in inference mode (no training= arg in BaseLayers),
        # so UPDATE_OPS for mainline ops is intentionally empty. We do NOT
        # `control_dependencies` against any UPDATE_OPS here.
        accum_op = (
            tf.group(*[accum.assign_add(grad)
                       for accum, (grad, _) in zip(accum_vars, grads_and_vars)],
                     name="accumulate_gradients")
            if accum_vars
            else tf.no_op()
        )
        averaged_grads = list(accum_vars)
        if grad_clip_global_norm > 0:
            clipped_grads, grad_norm = tf.clip_by_global_norm(
                averaged_grads, clip_norm=grad_clip_global_norm
            )
            apply_pairs = list(zip(clipped_grads, vars_))
        else:
            grad_norm = (
                tf.linalg.global_norm(averaged_grads)
                if averaged_grads
                else tf.constant(0.0, dtype=tf.float32)
            )
            apply_pairs = list(zip(averaged_grads, vars_))
        train_op = optimizer.apply_gradients(apply_pairs)

    epe_tensor = build_epe_metric(
        pred_tensor=accumulate_predictions(preds),
        label_ph=label_ph,
        num_out=flow_channels,
    )

    # Saver covers ONLY forward weights (mainline ckpt layout doesn't have
    # the optimizer state, and we don't want to save Adam slots into the
    # published ckpt either).
    saver = tf.compat.v1.train.Saver(var_list=fwd_vars, max_to_keep=3)

    return {
        "scope_name": "",  # mainline has no outer scope
        "arch_family": ARCH_FAMILY_MAINLINE,
        "arch_code": None,
        "loss": loss_tensor,
        "loss_optical": loss_terms["optical_total"],
        "loss_uncertainty": loss_terms["uncertainty_total"],
        "zero_grad_op": zero_grad_op,
        "accum_op": accum_op,
        "train_op": train_op,
        "grad_norm": grad_norm,
        "epe": epe_tensor,
        "saver": saver,
        "trainable_vars": fwd_vars,
        "scope_global_vars": fwd_vars,
    }


# ----------------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------------
def _build_provider(config: Dict[str, Any], split: str, seed_offset: int, provider_mode: str):
    dataset = str(config.get("data", {}).get("dataset", "FT3D")).strip().upper()
    if dataset == "FC2":
        return build_fc2_provider(
            config=config, split=split, seed_offset=seed_offset, provider_mode=provider_mode
        )
    if dataset == "FT3D":
        return build_ft3d_provider(
            config=config, split=split, seed_offset=seed_offset, provider_mode=provider_mode
        )
    raise ValueError(f"unsupported deploy_ft dataset: {dataset}")


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
    num_batches = (
        max(1, int(math.ceil(len(val_provider) / float(batch_size))))
        if eval_batches <= 0
        else int(eval_batches)
    )
    values: List[float] = []
    feed_dict_extra = {is_training_ph: True} if is_training_ph is not None else {}
    for _ in tqdm(range(num_batches), total=num_batches, desc=desc, leave=False):
        input_batch, _, _, label_batch = val_provider.next_batch(batch_size=batch_size)
        input_batch = standardize_image_tensor(input_batch)
        feed = {input_ph: input_batch, label_ph: label_batch, **feed_dict_extra}
        epe = sess.run(graph_obj["epe"], feed_dict=feed)
        values.append(float(epe))
    return float(np.mean(values)) if values else float("inf")


def _resolve_init_checkpoint(config: Dict[str, Any], model_name: str) -> Optional[Path]:
    """Two init modes:
        - experiment_dir: look under init_experiment_dir/model_<name>/checkpoints/<init_ckpt_name>.ckpt
        - explicit_path:  use init_ckpt_path verbatim
    """
    checkpoint_cfg = config.get("checkpoint", {})
    init_mode = str(checkpoint_cfg.get("init_mode", "")).strip().lower()
    if not init_mode or init_mode == "none":
        return None
    if init_mode == "experiment_dir":
        exp_dir = str(checkpoint_cfg.get("init_experiment_dir", "")).strip()
        ckpt_name = str(checkpoint_cfg.get("init_ckpt_name", "sintel_best")).strip() or "sintel_best"
        if not exp_dir:
            raise ValueError("checkpoint.init_experiment_dir required for init_mode=experiment_dir")
        return Path(exp_dir) / f"model_{model_name}" / "checkpoints" / f"{ckpt_name}.ckpt"
    if init_mode == "explicit_path":
        explicit = str(checkpoint_cfg.get("init_ckpt_path", "")).strip()
        if not explicit:
            raise ValueError("checkpoint.init_ckpt_path required for init_mode=explicit_path")
        return Path(explicit)
    raise ValueError(f"unsupported deploy_ft checkpoint.init_mode: {init_mode}")


def _resolve_lr(base_lr: float, lr_min: float, schedule: str, step: int, total: int) -> float:
    schedule = (schedule or "constant").strip().lower()
    if schedule == "constant":
        return float(base_lr)
    if schedule == "cosine":
        return _cosine_lr_with_min(base_lr, lr_min, step, total)
    raise ValueError(f"unsupported lr_schedule: {schedule}")


def _model_weight_vars_v3(scope_name: str) -> List[tf.Variable]:
    vars_: List[tf.Variable] = []
    for var in tf.compat.v1.global_variables():
        if not var.name.startswith(f"{scope_name}/"):
            continue
        name = var.op.name
        if "/Adam" in name or name.endswith("/Adam") or name.endswith("/Adam_1"):
            continue
        if name in (f"{scope_name}/beta1_power", f"{scope_name}/beta2_power"):
            continue
        if "_grad_accum" in name:
            continue
        vars_.append(var)
    return vars_


def _model_weight_vars_mainline() -> List[tf.Variable]:
    vars_: List[tf.Variable] = []
    for var in tf.compat.v1.global_variables():
        if var.name.startswith("Optimizer/"):
            continue
        if "_grad_accum" in var.op.name:
            continue
        vars_.append(var)
    return vars_


def _checkpoint_exists(path_prefix: Path) -> bool:
    return Path(str(path_prefix) + ".index").exists()


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def train_deploy_ft(config: Dict[str, Any]) -> int:
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("eval", {})
    checkpoint_cfg = config.get("checkpoint", {})

    dataset = str(data_cfg.get("dataset", "FT3D")).strip().upper()
    arch_family = str(config.get("arch_family", ARCH_FAMILY_V3)).strip().lower()
    if arch_family not in (ARCH_FAMILY_V3, ARCH_FAMILY_MAINLINE):
        raise ValueError(f"unsupported arch_family: {arch_family}")

    seed = int(runtime_cfg.get("seed", 42))
    set_global_seed(seed)
    experiment_dir = _resolve_output_dir(config)
    model_name = str(config.get("model_name", "candidate")).strip() or "candidate"
    model_dir = experiment_dir / f"model_{model_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(f"deploy_ft_{model_name}", str(model_dir / "train.log"))
    logger.info("start deploy_ft family=%s dataset=%s model=%s", arch_family, dataset, model_name)
    _apply_gpu_device_setting(train_cfg=train_cfg, logger=logger)

    arch_code = (
        parse_arch_code(config.get("arch_code"))
        if arch_family == ARCH_FAMILY_V3
        else None
    )

    batch_size = int(train_cfg.get("batch_size", 32))
    micro_batch_size = min(int(train_cfg.get("micro_batch_size", batch_size)), batch_size)
    num_epochs = int(train_cfg.get("num_epochs", 20))
    base_lr = float(train_cfg.get("lr", 1e-6))
    lr_min = float(train_cfg.get("lr_min", base_lr))
    lr_schedule = str(train_cfg.get("lr_schedule", "constant"))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip_global_norm", 50.0))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 5))
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 1e-4))

    eval_every_epoch = max(1, int(eval_cfg.get("eval_every_epoch", 1)))
    eval_batches = int(eval_cfg.get("eval_batches", 0))
    sintel_cfg = eval_cfg.get("sintel", {}) if isinstance(eval_cfg.get("sintel", {}), dict) else {}
    sintel_every = int(sintel_cfg.get("eval_every_epoch", 0) or 0)

    input_h = int(data_cfg.get("input_height", 157))
    input_w = int(data_cfg.get("input_width", 203))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = flow_channels * 2  # uncertainty mode

    # Build dataloaders (FT3D defaults; flow_divisor comes from config)
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

    logger.info("arch_family=%s arch_code=%s", arch_family,
                ",".join(str(v) for v in arch_code) if arch_code else "(none)")
    logger.info("input=%dx%d batch=%d micro_batch=%d epochs=%d steps_per_epoch=%d",
                input_h, input_w, batch_size, micro_batch_size, num_epochs, steps_per_epoch)
    logger.info("lr=%.2e (schedule=%s, min=%.2e) weight_decay=%.2e grad_clip=%.1f",
                base_lr, lr_schedule, lr_min, weight_decay, grad_clip)
    logger.info("eval_every=%d sintel_every=%d early_stop_patience=%d",
                eval_every_epoch, sintel_every, early_stop_patience)

    # Graph
    input_ph = tf.compat.v1.placeholder(tf.float32, [None, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, [None, input_h, input_w, flow_channels], name="Label")
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")
    grad_scale_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="GradScale")
    is_training_ph: Optional[tf.Tensor] = None

    if arch_family == ARCH_FAMILY_V3:
        is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")
        graph_obj = _build_v3_graph_impl(
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
        init_saver = tf.compat.v1.train.Saver(var_list=_model_weight_vars_v3(model_name))
    else:
        graph_obj = _build_mainline_graph(
            input_ph=input_ph,
            label_ph=label_ph,
            lr_ph=lr_ph,
            grad_scale_ph=grad_scale_ph,
            flow_channels=flow_channels,
            pred_channels=pred_channels,
            weight_decay=weight_decay,
            grad_clip_global_norm=grad_clip,
        )
        init_saver = tf.compat.v1.train.Saver(var_list=_model_weight_vars_mainline())

    ckpt_paths = _build_standalone_checkpoint_paths(model_dir)
    state_path = model_dir / "trainer_state.json"
    eval_history_path = model_dir / "eval_history.csv"
    write_json(
        str(model_dir / "run_manifest.json"),
        {
            "model_name": model_name,
            "arch_family": arch_family,
            "arch_code": [int(v) for v in arch_code] if arch_code else None,
            "dataset": dataset,
            "init_from": str(_resolve_init_checkpoint(config, model_name) or ""),
            "config": config,
            "git_commit": _git_commit_hash(),
        },
    )

    eval_history = _load_csv_rows(eval_history_path)
    best_epe = float("inf")
    best_sintel_epe = float("inf")
    no_improve_epochs = 0
    global_step = 0

    try:
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            init_ckpt = _resolve_init_checkpoint(config=config, model_name=model_name)
            if init_ckpt is None:
                raise RuntimeError("deploy_ft requires an init checkpoint; set checkpoint.init_mode")
            if not _checkpoint_exists(init_ckpt):
                raise FileNotFoundError(f"init checkpoint missing: {init_ckpt}")
            init_saver.restore(sess, str(init_ckpt))
            logger.info("restored init weights from %s", init_ckpt)

            for epoch_idx in range(1, num_epochs + 1):
                epoch_start = time.time()
                if hasattr(train_provider, "start_epoch"):
                    train_provider.start_epoch(shuffle=True)
                epoch_loss = 0.0
                optical_loss = 0.0
                uncertainty_loss = 0.0
                grad_norms: List[float] = []
                lr_last = base_lr
                desc = f"{model_name} {arch_family} epoch {epoch_idx}/{num_epochs}"
                iterator = tqdm(range(steps_per_epoch), total=steps_per_epoch, desc=desc, leave=False)
                for _ in iterator:
                    input_batch, _, _, label_batch = train_provider.next_batch(batch_size=batch_size)
                    input_batch = standardize_image_tensor(input_batch)
                    logical_batch = int(input_batch.shape[0])
                    micro_slices = _iter_micro_slices(logical_batch, micro_batch_size)
                    if not micro_slices:
                        continue
                    lr_now = _resolve_lr(base_lr, lr_min, lr_schedule, global_step, total_steps)
                    lr_last = lr_now
                    sess.run(graph_obj["zero_grad_op"])
                    step_loss = 0.0
                    step_optical = 0.0
                    step_uncertainty = 0.0
                    for micro_slice in micro_slices:
                        micro_input = input_batch[micro_slice]
                        micro_label = label_batch[micro_slice]
                        grad_scale = float(micro_input.shape[0]) / float(logical_batch)
                        feed = {
                            input_ph: micro_input,
                            label_ph: micro_label,
                            lr_ph: lr_now,
                            grad_scale_ph: grad_scale,
                        }
                        if is_training_ph is not None:
                            feed[is_training_ph] = True
                        result = sess.run(
                            {
                                "loss": graph_obj["loss"],
                                "optical": graph_obj["loss_optical"],
                                "uncertainty": graph_obj["loss_uncertainty"],
                                "accum": graph_obj["accum_op"],
                            },
                            feed_dict=feed,
                        )
                        step_loss += float(result["loss"]) * grad_scale
                        step_optical += float(result["optical"]) * grad_scale
                        step_uncertainty += float(result["uncertainty"]) * grad_scale
                    apply_result = sess.run(
                        {"grad_norm": graph_obj["grad_norm"], "train": graph_obj["train_op"]},
                        feed_dict={lr_ph: lr_now},
                    )
                    grad_norms.append(float(apply_result["grad_norm"]))
                    epoch_loss += step_loss
                    optical_loss += step_optical
                    uncertainty_loss += step_uncertainty
                    global_step += 1
                    iterator.set_postfix(
                        lr=f"{lr_now:.2e}",
                        loss=f"{epoch_loss / max(1, len(grad_norms)):.4f}",
                    )

                avg_loss = epoch_loss / max(1, steps_per_epoch)
                avg_optical = optical_loss / max(1, steps_per_epoch)
                avg_uncertainty = uncertainty_loss / max(1, steps_per_epoch)
                grad_stats = _summarize_grad_norms(grad_norms, grad_clip)

                do_eval = (epoch_idx % eval_every_epoch == 0) or (epoch_idx == num_epochs)
                val_epe = float("inf")
                if do_eval:
                    val_epe = _evaluate_with_progress(
                        sess, graph_obj, input_ph, label_ph, is_training_ph,
                        val_provider, batch_size, eval_batches,
                        desc=f"{dataset} val {model_name} e{epoch_idx}",
                    )

                _save_standalone_checkpoint(
                    sess, graph_obj["saver"], ckpt_paths["last"],
                    epoch_idx, global_step, val_epe, best_epe,
                    arch_code if arch_code is not None else [],
                )
                if do_eval and val_epe < best_epe - early_stop_min_delta:
                    best_epe = val_epe
                    no_improve_epochs = 0
                    _save_standalone_checkpoint(
                        sess, graph_obj["saver"], ckpt_paths["best"],
                        epoch_idx, global_step, val_epe, best_epe,
                        arch_code if arch_code is not None else [],
                    )
                    logger.info("best updated val_epe=%.4f", val_epe)
                elif do_eval:
                    no_improve_epochs += 1

                sintel_epe = None
                if sintel_every > 0 and do_eval and (
                    epoch_idx % sintel_every == 0 or epoch_idx == num_epochs
                ):
                    # Pass flow_scale EXPLICITLY (12.5 for FT3D-trained V3,
                    # 1.0 for mainline). Bypasses the broken
                    # _resolve_prediction_flow_scale path-hint fallback.
                    sintel_flow_scale = float(data_cfg.get("ft3d_flow_divisor", 12.5))
                    sintel_result = run_sintel_for_deploy_ft(
                        model_dir=model_dir, config=config,
                        epoch_idx=epoch_idx, ckpt_name="last",
                        arch_family=arch_family,
                        flow_scale=sintel_flow_scale,
                    )
                    if sintel_result is not None:
                        sintel_epe = float(sintel_result["sintel_epe"])
                        if sintel_epe < best_sintel_epe:
                            best_sintel_epe = sintel_epe
                            _save_standalone_checkpoint(
                                sess, graph_obj["saver"],
                                ckpt_paths["root"] / "sintel_best.ckpt",
                                epoch_idx, global_step, sintel_epe, best_sintel_epe,
                                arch_code if arch_code is not None else [],
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
                    "no_improve_epochs": no_improve_epochs,
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
                        "no_improve_epochs": no_improve_epochs,
                        "model_name": model_name,
                        "arch_family": arch_family,
                        "arch_code": [int(v) for v in arch_code] if arch_code else None,
                        "dataset": dataset,
                    },
                )
                logger.info(
                    "epoch=%d/%d global_step=%d lr=%.2e time=%.1fs loss=%.6f "
                    "optical=%.6f uncertainty=%.6f val_epe=%s sintel_epe=%s "
                    "no_improve=%d grad_mean=%.4f grad_p90=%.4f clip_rate=%.4f",
                    epoch_idx, num_epochs, global_step, lr_last,
                    time.time() - epoch_start, avg_loss, avg_optical, avg_uncertainty,
                    "" if not np.isfinite(val_epe) else f"{val_epe:.4f}",
                    "" if sintel_epe is None else f"{sintel_epe:.4f}",
                    no_improve_epochs,
                    grad_stats["mean"], grad_stats["p90"], grad_stats["clip_rate"],
                )

                if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                    logger.info("early stop triggered: no improvement for %d epochs", no_improve_epochs)
                    break
    finally:
        _close_provider(train_provider)
        _close_provider(val_provider)

    logger.info(
        "training complete model=%s family=%s best_val_epe=%.4f best_sintel_epe=%.4f",
        model_name, arch_family, best_epe, best_sintel_epe,
    )
    return 0
