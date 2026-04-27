"""Supernet V3 training engine."""

import hashlib
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from efnas.data.dataloader_builder import build_fc2_provider
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.checkpoint_manager import (
    build_checkpoint_paths,
    find_existing_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from efnas.engine.early_stop import EarlyStopState, update_early_stop
from efnas.engine.eval_step import accumulate_predictions, build_epe_metric
from efnas.engine.supernet_trainer import (
    _apply_gpu_device_setting,
    _build_arch_ranking,
    _build_teacher_restore_map,
    _format_arch_ranking,
    _git_commit_hash,
    _load_edgeflownet_network_class,
    _parse_float_list,
    _resolve_checkpoint_prefix,
    _resolve_distill_teacher_type,
    _resolve_output_dir,
    _resolve_resume_dir,
    _run_eval_epoch,
    _try_restore_teacher_state,
    _write_eval_history,
)
from efnas.engine.train_step import add_weight_decay, build_channel_max_distill_loss, build_multiscale_uncertainty_loss
from efnas.nas.eval_pool_builder_v3 import build_eval_pool, check_eval_pool_coverage
from efnas.nas.fair_sampler_v3 import generate_fair_cycle
from efnas.nas.search_space_v3 import (
    V3_REFERENCE_ARCH_CODE,
    get_arch_semantics_version,
    get_block_name,
    get_block_specs,
    get_num_blocks,
    get_num_choices,
)
from efnas.network.MultiScaleResNet_supernet_v3 import MultiScaleResNetSupernetV3
from efnas.optim.lr_scheduler import cosine_lr
from efnas.utils.json_io import read_json, write_json
from efnas.utils.logger import build_logger
from efnas.utils.manifest import compare_run_manifest
from efnas.utils.path_utils import project_root
from efnas.utils.seed import set_global_seed


def _hash_config(config: Dict[str, Any]) -> str:
    """Compute stable config hash."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_resume_signature_v3(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build V3 resume-critical signature."""
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    distill_cfg = train_cfg.get("distill", {})
    return {
        "dataset": data_cfg.get("dataset", ""),
        "input_shape": [int(data_cfg.get("input_height", 0)), int(data_cfg.get("input_width", 0))],
        "flow_channels": int(data_cfg.get("flow_channels", 2)),
        "supernet_mode": train_cfg.get("supernet_mode", "balanced_irregular_fairness"),
        "uncertainty_type": train_cfg.get("uncertainty_type", "LinearSoftplus"),
        "distill_enabled": bool(distill_cfg.get("enabled", False)),
        "distill_teacher_type": _resolve_distill_teacher_type(distill_cfg.get("teacher_type", "edgeflownet")),
        "distill_lambda": float(distill_cfg.get("lambda", 1.0)),
        "multi_gpu_mode": str(train_cfg.get("multi_gpu_mode", "single_gpu")).strip().lower(),
        "arch_semantics_version": get_arch_semantics_version(),
        "network": "MultiScaleResNet_supernet_v3",
        "backbone": "bilinear_bottleneck_eca_gate4x",
        "block_names": [str(spec["name"]) for spec in get_block_specs()],
        "block_choice_counts": [int(spec["num_choices"]) for spec in get_block_specs()],
        "init_neurons": 32,
        "expansion_factor": 2.0,
        "multiscale_outputs": 3,
    }


def _build_run_manifest_v3(config: Dict[str, Any], git_commit: str) -> Dict[str, Any]:
    """Build V3 run manifest."""
    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("eval", {})
    checkpoint_cfg = config.get("checkpoint", {})
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "manifest_type": "run_manifest",
        "experiment_name": runtime_cfg.get("experiment_name", "unknown"),
        "git_commit": git_commit,
        "config_hash": _hash_config(config),
        "resume_signature": _build_resume_signature_v3(config=config),
        "train_snapshot": {
            "num_epochs": int(train_cfg.get("num_epochs", 0)),
            "epoch_mode": str(train_cfg.get("epoch_mode", "fixed_steps")),
            "steps_per_epoch": int(train_cfg.get("steps_per_epoch", 0)),
            "train_sampling_mode": str(train_cfg.get("train_sampling_mode", "random")),
            "batch_size": int(train_cfg.get("batch_size", 0)),
            "micro_batch_size": int(train_cfg.get("micro_batch_size", 0)),
            "lr": float(train_cfg.get("lr", 0.0)),
            "lr_schedule": str(train_cfg.get("lr_schedule", "")),
            "grad_clip_global_norm": float(train_cfg.get("grad_clip_global_norm", 0.0)),
            "multi_gpu_mode": str(train_cfg.get("multi_gpu_mode", "single_gpu")),
            "gpu_devices": str(train_cfg.get("gpu_devices", train_cfg.get("gpu_device", "0"))),
        },
        "eval_snapshot": {
            "eval_every_epoch": int(eval_cfg.get("eval_every_epoch", 1)),
            "eval_pool_size": int(eval_cfg.get("eval_pool_size", 12)),
            "bn_recal_batches": int(eval_cfg.get("bn_recal_batches", 8)),
            "eval_batches_per_arch": int(eval_cfg.get("eval_batches_per_arch", 4)),
        },
        "checkpoint_snapshot": {
            "load_checkpoint": bool(checkpoint_cfg.get("load_checkpoint", False)),
            "resume_experiment_name": str(checkpoint_cfg.get("resume_experiment_name", "")),
        },
        "data_snapshot": {
            "dataset": str(data_cfg.get("dataset", "")),
            "base_path": str(data_cfg.get("base_path", "")),
            "train_dir": str(data_cfg.get("train_dir", "")),
            "val_dir": str(data_cfg.get("val_dir", "")),
            "fc2_num_workers": int(data_cfg.get("fc2_num_workers", 1)),
            "fc2_eval_num_workers": int(data_cfg.get("fc2_eval_num_workers", data_cfg.get("fc2_num_workers", 1))),
            "prefetch_batches": int(data_cfg.get("prefetch_batches", 0)),
        },
    }


def _build_manifest_v3(config: Dict[str, Any], git_commit: str) -> Dict[str, Any]:
    """Build V3 legacy train manifest payload."""
    train_cfg = config.get("train", {})
    runtime_cfg = config.get("runtime", {})
    data_cfg = config.get("data", {})
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "experiment_name": runtime_cfg.get("experiment_name", "unknown"),
        "seed": runtime_cfg.get("seed", 0),
        "git_commit": git_commit,
        "config_hash": _hash_config(config),
        "input_shape": [data_cfg.get("input_height", 0), data_cfg.get("input_width", 0)],
        "flow_channels": data_cfg.get("flow_channels", 2),
        "optimizer": train_cfg.get("optimizer", "adam"),
        "lr_schedule": train_cfg.get("lr_schedule", "cosine"),
        "uncertainty_type": train_cfg.get("uncertainty_type", "LinearSoftplus"),
        "wd": train_cfg.get("weight_decay", 0.0),
        "grad_clip": train_cfg.get("grad_clip_global_norm", 0.0),
        "arch_semantics_version": get_arch_semantics_version(),
    }


def _init_fairness_counts() -> Dict[str, Dict[str, int]]:
    """Initialize V3 fairness counters."""
    counts: Dict[str, Dict[str, int]] = {}
    for block_idx in range(get_num_blocks()):
        counts[str(block_idx)] = {}
        for option in range(get_num_choices(block_idx)):
            counts[str(block_idx)][str(option)] = 0
    return counts


def _sanitize_fairness_counts(raw_counts: Any) -> Dict[str, Dict[str, int]]:
    """Normalize fairness counters loaded from checkpoint metadata."""
    clean = _init_fairness_counts()
    if not isinstance(raw_counts, dict):
        return clean
    for block_idx in range(get_num_blocks()):
        block_key = str(block_idx)
        block_raw = raw_counts.get(block_key, raw_counts.get(block_idx, {}))
        if not isinstance(block_raw, dict):
            continue
        for option in range(get_num_choices(block_idx)):
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
    for block_idx in range(get_num_blocks()):
        values = list(counts[str(block_idx)].values())
        gap = max(gap, max(values) - min(values))
    return int(gap)


def _parse_teacher_arch_code(raw_value: Any) -> List[int]:
    """Parse V3 teacher arch code with fallback to V3 reference code."""
    default_code = [int(item) for item in V3_REFERENCE_ARCH_CODE]
    if raw_value is None:
        return default_code
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return default_code
        tokens = [token for token in re.split(r"[,\s]+", text) if token]
    elif isinstance(raw_value, (list, tuple)):
        tokens = [str(token) for token in raw_value]
    else:
        raise ValueError(f"unsupported teacher_arch_code type: {type(raw_value)}")
    if len(tokens) != int(get_num_blocks()):
        raise ValueError(f"teacher_arch_code length must be {get_num_blocks()}, got {len(tokens)}")
    parsed: List[int] = []
    for block_idx, token in enumerate(tokens):
        value = int(token)
        num_choices = get_num_choices(block_idx)
        if value < 0 or value >= num_choices:
            raise ValueError(
                f"teacher_arch_code out of range at idx={block_idx} "
                f"name={get_block_name(block_idx)} value={value} valid=[0,{num_choices - 1}]"
            )
        parsed.append(value)
    return parsed


def _parse_gpu_devices(train_cfg: Dict[str, Any]) -> List[int]:
    """Parse configured GPU devices."""
    raw = train_cfg.get("gpu_devices", None)
    if raw is None or str(raw).strip() == "":
        return [int(train_cfg.get("gpu_device", 0))]
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _build_graph(config: Dict[str, Any]) -> Dict[str, object]:
    """Build TF1 supernet V3 training graph."""
    train_cfg = config.get("train", {})
    distill_cfg = train_cfg.get("distill", {})
    data_cfg = config.get("data", {})
    input_h = int(data_cfg.get("input_height", 180))
    input_w = int(data_cfg.get("input_width", 240))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = int(flow_channels * 2)
    distill_enabled = bool(distill_cfg.get("enabled", False))
    distill_lambda = float(distill_cfg.get("lambda", 1.0))
    distill_teacher_type = _resolve_distill_teacher_type(distill_cfg.get("teacher_type", "edgeflownet"))
    distill_layer_weights = _parse_float_list(distill_cfg.get("layer_weights", ""))
    teacher_arch_code = _parse_teacher_arch_code(distill_cfg.get("teacher_arch_code", None))
    teacher_ckpt = str(distill_cfg.get("teacher_ckpt", "")).strip()
    if distill_enabled and not teacher_ckpt:
        raise RuntimeError("distill.enabled=true requires train.distill.teacher_ckpt")

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, flow_channels], name="Label")
    arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[get_num_blocks()], name="ArchCode")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")
    accum_divisor_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="AccumDivisor")

    multi_gpu_mode = str(train_cfg.get("multi_gpu_mode", "single_gpu")).strip().lower()
    gpu_devices = _parse_gpu_devices(train_cfg)
    if multi_gpu_mode == "arch_parallel":
        if len(gpu_devices) < 3:
            raise RuntimeError("multi_gpu_mode=arch_parallel requires at least 3 gpu_devices")
        arch_codes_ph = tf.compat.v1.placeholder(tf.int32, shape=[3, get_num_blocks()], name="ArchCodes")
        tower_losses = []
        tower_optical = []
        tower_uncertainty = []
        tower_distill = []
        teacher_arch_code_ph = None
        teacher_vars: List[tf.Variable] = []
        teacher_features = []

        if distill_enabled:
            with tf.compat.v1.variable_scope("teacher"):
                if distill_teacher_type == "supernet":
                    teacher_arch_code_ph = tf.compat.v1.placeholder(
                        tf.int32,
                        shape=[get_num_blocks()],
                        name="TeacherArchCode",
                    )
                    teacher_model = MultiScaleResNetSupernetV3(
                        input_ph=input_ph,
                        arch_code_ph=teacher_arch_code_ph,
                        is_training_ph=tf.constant(False, dtype=tf.bool),
                        num_out=pred_channels,
                        init_neurons=32,
                        expansion_factor=2.0,
                    )
                    teacher_model.build()
                    teacher_features = teacher_model.feature_pyramid()
                else:
                    edge_network_cls = _load_edgeflownet_network_class()
                    teacher_model = edge_network_cls(
                        InputPH=input_ph,
                        NumOut=pred_channels,
                        InitNeurons=32,
                        ExpansionFactor=2.0,
                        NumSubBlocks=2,
                        NumBlocks=1,
                        Suffix="",
                        UncType=None,
                    )
                    teacher_model.Network()
                    if hasattr(teacher_model, "FeaturePyramidOutputs"):
                        teacher_features = teacher_model.FeaturePyramidOutputs()
                    else:
                        raise RuntimeError("EdgeFlowNet teacher model missing FeaturePyramidOutputs")
            if distill_layer_weights and len(distill_layer_weights) != len(teacher_features):
                raise RuntimeError("distill.layer_weights length must match feature pyramid length")
            teacher_vars = [var for var in tf.compat.v1.global_variables() if var.name.startswith("teacher/")]
            if not teacher_vars:
                raise RuntimeError("distill enabled but teacher graph variables are empty")

        with tf.compat.v1.variable_scope("shared_supernet") as shared_scope:
            for tower_idx, gpu_id in enumerate(gpu_devices[:3]):
                with tf.device(f"/GPU:{gpu_id}"):
                    if tower_idx > 0:
                        shared_scope.reuse_variables()
                    tower_model = MultiScaleResNetSupernetV3(
                        input_ph=input_ph,
                        arch_code_ph=arch_codes_ph[tower_idx],
                        is_training_ph=is_training_ph,
                        num_out=pred_channels,
                        init_neurons=32,
                        expansion_factor=2.0,
                    )
                    tower_preds = tower_model.build()
                    tower_terms = build_multiscale_uncertainty_loss(
                        preds=tower_preds,
                        label_ph=label_ph,
                        num_out=flow_channels,
                        return_terms=True,
                    )
                    tower_loss_core = tower_terms["total"]
                    if distill_enabled:
                        student_features = tower_model.feature_pyramid()
                        if len(student_features) != len(teacher_features):
                            raise RuntimeError("student and teacher feature pyramid lengths do not match")
                        tower_loss_distill = build_channel_max_distill_loss(
                            student_features=student_features,
                            teacher_features=teacher_features,
                            layer_weights=(distill_layer_weights or None),
                        )
                        tower_loss_core = tf.add(
                            tower_loss_core,
                            float(distill_lambda) * tower_loss_distill,
                            name=f"tower{tower_idx}_loss_core_with_distill",
                        )
                        tower_distill.append(tower_loss_distill)
                    tower_losses.append(tower_loss_core)
                    tower_optical.append(tower_terms["optical_total"])
                    tower_uncertainty.append(tower_terms["uncertainty_total"])

            tower_bn_updates = [
                op
                for op in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                if op.name.startswith("shared_supernet/")
            ]
            shared_scope.reuse_variables()
            eval_model = MultiScaleResNetSupernetV3(
                input_ph=input_ph,
                arch_code_ph=arch_code_ph,
                is_training_ph=is_training_ph,
                num_out=pred_channels,
                init_neurons=32,
                expansion_factor=2.0,
            )
            preds = eval_model.build()

        loss_core_for_grads = tf.add_n(tower_losses, name="arch_parallel_loss_sum")
        loss_tensor_for_log = tf.reduce_mean(tf.stack(tower_losses), name="arch_parallel_loss_mean")
        loss_optical = tf.reduce_mean(tf.stack(tower_optical), name="arch_parallel_optical_mean")
        loss_uncertainty = tf.reduce_mean(tf.stack(tower_uncertainty), name="arch_parallel_uncertainty_mean")
        if distill_enabled:
            loss_distill = tf.reduce_mean(tf.stack(tower_distill), name="arch_parallel_distill_mean")
        else:
            loss_distill = tf.constant(0.0, dtype=tf.float32, name="distill_loss_disabled")
        student_trainable_vars = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith("shared_supernet/")]
        if not student_trainable_vars:
            raise RuntimeError("student trainable variables are empty")
        loss_for_grads = add_weight_decay(
            loss_tensor=loss_core_for_grads,
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            trainable_vars=student_trainable_vars,
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_ph, beta1=0.9, beta2=0.999, epsilon=1e-8)
        grads_and_vars = optimizer.compute_gradients(loss_for_grads, var_list=student_trainable_vars)
        grads = [grad for grad, _ in grads_and_vars if grad is not None]
        vars_ = [var for grad, var in grads_and_vars if grad is not None]
        clip_norm = float(train_cfg.get("grad_clip_global_norm", 5.0))
        accum_vars = []
        zero_ops = []
        add_ops = []
        for idx, (grad, var) in enumerate(zip(grads, vars_)):
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
        with tf.control_dependencies(add_ops + tower_bn_updates):
            accum_op = tf.no_op(name="strict_accum_done")
        avg_divisor = tf.maximum(accum_divisor_ph, 1.0, name="strict_avg_divisor")
        avg_grads = [accum_var / avg_divisor for accum_var in accum_vars]
        clipped_avg_grads, global_norm = tf.clip_by_global_norm(avg_grads, clip_norm=clip_norm)
        clip_trigger = tf.cast(tf.greater(global_norm, clip_norm), tf.int32, name="strict_clip_trigger")
        apply_op = optimizer.apply_gradients(list(zip(clipped_avg_grads, vars_)), name="strict_apply")
        pred_accum = accumulate_predictions(preds)
        epe_tensor = build_epe_metric(pred_tensor=pred_accum, label_ph=label_ph, num_out=flow_channels)
        student_global_vars = [var for var in tf.compat.v1.global_variables() if not var.name.startswith("teacher/")]
        saver = tf.compat.v1.train.Saver(var_list=student_global_vars, max_to_keep=5)
        return {
            "input_ph": input_ph,
            "label_ph": label_ph,
            "arch_code_ph": arch_code_ph,
            "arch_codes_ph": arch_codes_ph,
            "teacher_arch_code_ph": teacher_arch_code_ph,
            "is_training_ph": is_training_ph,
            "lr_ph": lr_ph,
            "accum_divisor_ph": accum_divisor_ph,
            "preds": preds,
            "loss": loss_tensor_for_log,
            "loss_core": loss_tensor_for_log,
            "loss_optical": loss_optical,
            "loss_uncertainty": loss_uncertainty,
            "loss_distill": loss_distill,
            "epe": epe_tensor,
            "global_grad_norm": global_norm,
            "clip_trigger": clip_trigger,
            "clip_norm": clip_norm,
            "zero_ops": zero_ops,
            "accum_op": accum_op,
            "apply_op": apply_op,
            "distill_enabled": bool(distill_enabled),
            "distill_lambda": float(distill_lambda),
            "distill_teacher_type": distill_teacher_type,
            "teacher_arch_code": [int(item) for item in teacher_arch_code],
            "teacher_ckpt": teacher_ckpt,
            "teacher_vars": teacher_vars,
            "multi_gpu_mode": "arch_parallel",
            "gpu_devices": gpu_devices[:3],
            "saver": saver,
        }

    model = MultiScaleResNetSupernetV3(
        input_ph=input_ph,
        arch_code_ph=arch_code_ph,
        is_training_ph=is_training_ph,
        num_out=pred_channels,
        init_neurons=32,
        expansion_factor=2.0,
    )
    preds = model.build()
    student_features = model.feature_pyramid()
    base_loss_terms = build_multiscale_uncertainty_loss(
        preds=preds,
        label_ph=label_ph,
        num_out=flow_channels,
        return_terms=True,
    )
    loss_optical = base_loss_terms["optical_total"]
    loss_uncertainty = base_loss_terms["uncertainty_total"]
    loss_distill = tf.constant(0.0, dtype=tf.float32, name="distill_loss_disabled")
    loss_core = base_loss_terms["total"]
    teacher_arch_code_ph = None
    teacher_vars: List[tf.Variable] = []

    if distill_enabled:
        with tf.compat.v1.variable_scope("teacher"):
            if distill_teacher_type == "supernet":
                teacher_arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[get_num_blocks()], name="TeacherArchCode")
                teacher_model = MultiScaleResNetSupernetV3(
                    input_ph=input_ph,
                    arch_code_ph=teacher_arch_code_ph,
                    is_training_ph=tf.constant(False, dtype=tf.bool),
                    num_out=pred_channels,
                    init_neurons=32,
                    expansion_factor=2.0,
                )
                teacher_model.build()
                teacher_features = teacher_model.feature_pyramid()
            else:
                edge_network_cls = _load_edgeflownet_network_class()
                teacher_model = edge_network_cls(
                    InputPH=input_ph,
                    NumOut=pred_channels,
                    InitNeurons=32,
                    ExpansionFactor=2.0,
                    NumSubBlocks=2,
                    NumBlocks=1,
                    Suffix="",
                    UncType=None,
                )
                teacher_model.Network()
                if hasattr(teacher_model, "FeaturePyramidOutputs"):
                    teacher_features = teacher_model.FeaturePyramidOutputs()
                else:
                    raise RuntimeError("EdgeFlowNet teacher model missing FeaturePyramidOutputs")
        if distill_layer_weights and len(distill_layer_weights) != len(student_features):
            raise RuntimeError("distill.layer_weights length must match feature pyramid length")
        loss_distill = build_channel_max_distill_loss(
            student_features=student_features,
            teacher_features=teacher_features,
            layer_weights=(distill_layer_weights or None),
        )
        loss_core = tf.add(loss_core, float(distill_lambda) * loss_distill, name="loss_core_with_distill")
        teacher_vars = [var for var in tf.compat.v1.global_variables() if var.name.startswith("teacher/")]
        if not teacher_vars:
            raise RuntimeError("distill enabled but teacher graph variables are empty")

    student_trainable_vars = [var for var in tf.compat.v1.trainable_variables() if not var.name.startswith("teacher/")]
    if not student_trainable_vars:
        raise RuntimeError("student trainable variables are empty")
    loss_tensor = add_weight_decay(
        loss_tensor=loss_core,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        trainable_vars=student_trainable_vars,
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=lr_ph,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    grads_and_vars = optimizer.compute_gradients(loss_tensor, var_list=student_trainable_vars)
    grads = [grad for grad, _ in grads_and_vars if grad is not None]
    vars_ = [var for grad, var in grads_and_vars if grad is not None]
    clip_norm = float(train_cfg.get("grad_clip_global_norm", 5.0))

    accum_vars = []
    zero_ops = []
    add_ops = []
    for idx, (grad, var) in enumerate(zip(grads, vars_)):
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

    bn_updates_all = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    bn_updates = [op for op in bn_updates_all if not op.name.startswith("teacher/")]
    with tf.control_dependencies(add_ops + bn_updates):
        accum_op = tf.no_op(name="strict_accum_done")

    avg_divisor = tf.maximum(accum_divisor_ph, 1.0, name="strict_avg_divisor")
    avg_grads = [accum_var / avg_divisor for accum_var in accum_vars]
    clipped_avg_grads, global_norm = tf.clip_by_global_norm(avg_grads, clip_norm=clip_norm)
    clip_trigger = tf.cast(tf.greater(global_norm, clip_norm), tf.int32, name="strict_clip_trigger")
    apply_op = optimizer.apply_gradients(list(zip(clipped_avg_grads, vars_)), name="strict_apply")

    pred_accum = accumulate_predictions(preds)
    epe_tensor = build_epe_metric(pred_tensor=pred_accum, label_ph=label_ph, num_out=flow_channels)
    student_global_vars = [var for var in tf.compat.v1.global_variables() if not var.name.startswith("teacher/")]
    saver = tf.compat.v1.train.Saver(var_list=student_global_vars, max_to_keep=5)

    return {
        "input_ph": input_ph,
        "label_ph": label_ph,
        "arch_code_ph": arch_code_ph,
        "teacher_arch_code_ph": teacher_arch_code_ph,
        "is_training_ph": is_training_ph,
        "lr_ph": lr_ph,
        "accum_divisor_ph": accum_divisor_ph,
        "preds": preds,
        "loss": loss_tensor,
        "loss_core": loss_core,
        "loss_optical": loss_optical,
        "loss_uncertainty": loss_uncertainty,
        "loss_distill": loss_distill,
        "epe": epe_tensor,
        "global_grad_norm": global_norm,
        "clip_trigger": clip_trigger,
        "clip_norm": clip_norm,
        "zero_ops": zero_ops,
        "accum_op": accum_op,
        "apply_op": apply_op,
        "distill_enabled": bool(distill_enabled),
        "distill_lambda": float(distill_lambda),
        "distill_teacher_type": distill_teacher_type,
        "teacher_arch_code": [int(item) for item in teacher_arch_code],
        "teacher_ckpt": teacher_ckpt,
        "teacher_vars": teacher_vars,
        "multi_gpu_mode": "single_gpu",
        "gpu_devices": _parse_gpu_devices(train_cfg),
        "saver": saver,
    }


def _default_restore_state() -> Dict[str, Any]:
    """Default state when not resuming."""
    return {
        "start_epoch": 1,
        "global_step": 0,
        "fairness_counts": _init_fairness_counts(),
        "best_metric": float("inf"),
        "bad_epochs": 0,
        "last_metric": float("inf"),
    }


def _validate_resume_run_manifest(config: Dict[str, Any], resume_dir: Path, logger) -> None:
    """Validate resume checkpoint compatibility with V3 run manifest."""
    resume_manifest_path = resume_dir / "run_manifest.json"
    if not resume_manifest_path.exists():
        logger.warning("resume manifest not found: %s (skip compatibility check)", str(resume_manifest_path))
        return
    resume_manifest = read_json(str(resume_manifest_path))
    if not isinstance(resume_manifest, dict):
        raise RuntimeError(f"resume manifest is not a valid dict: {resume_manifest_path}")
    current_manifest = _build_run_manifest_v3(config=config, git_commit=_git_commit_hash())
    mismatches = compare_run_manifest(current_manifest=current_manifest, resume_manifest=resume_manifest)
    if mismatches:
        mismatch_text = "\n".join(f"- {item}" for item in mismatches)
        raise RuntimeError(
            "resume run_manifest mismatch detected:\n"
            f"{mismatch_text}\n"
            "please use a checkpoint with matching data/model semantics"
        )


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
    _validate_resume_run_manifest(config=config, resume_dir=resume_dir, logger=logger)
    resume_paths = build_checkpoint_paths(str(resume_dir))
    resume_prefix = find_existing_checkpoint(path_prefix=resume_paths["last"])
    if resume_prefix is None:
        logger.warning("load_checkpoint=true but no checkpoint found in %s", str(resume_dir))
        return _default_restore_state()

    restore_info = restore_checkpoint(sess=sess, saver=saver, path_prefix=resume_prefix)
    meta = restore_info.get("meta", {}) if isinstance(restore_info, dict) else {}

    start_epoch = int(meta.get("epoch", 0)) + 1
    global_step = int(meta.get("global_step", 0))
    fairness_counts = _sanitize_fairness_counts(raw_counts=meta.get("fairness_counts", {}))
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
    """Run balanced irregular-fairness supernet V3 training."""
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    distill_cfg = train_cfg.get("distill", {})

    seed = int(runtime_cfg.get("seed", 42))
    set_global_seed(seed)

    experiment_dir = _resolve_output_dir(config)
    logger = build_logger("edgeflownas_supernet_v3", str(experiment_dir / "train.log"))
    logger.info("start supernet V3 training with balanced irregular-space sampling")
    multi_gpu_mode_cfg = str(train_cfg.get("multi_gpu_mode", "single_gpu")).strip().lower()
    if multi_gpu_mode_cfg == "arch_parallel":
        logger.info("gpu visibility left unchanged for arch_parallel gpu_devices=%s", train_cfg.get("gpu_devices", ""))
        logger.info("arch_parallel BN caveat: tf.layers batch-norm moving statistics are not synchronized across towers")
    else:
        _apply_gpu_device_setting(train_cfg=train_cfg, logger=logger)

    batch_size = int(train_cfg.get("batch_size", 32))
    micro_batch_size = int(train_cfg.get("micro_batch_size", batch_size))
    if micro_batch_size <= 0:
        micro_batch_size = batch_size
    micro_batch_size = min(micro_batch_size, batch_size)
    logger.info("batch_size=%d micro_batch_size=%d", batch_size, micro_batch_size)

    num_epochs = int(train_cfg.get("num_epochs", 200))
    epoch_mode = str(train_cfg.get("epoch_mode", "fixed_steps")).strip().lower()
    if epoch_mode not in ("fixed_steps", "full_pass"):
        raise ValueError(f"unsupported epoch_mode: {epoch_mode}")
    base_lr = float(train_cfg.get("lr", 1e-4))
    bn_recal_batches = int(eval_cfg.get("bn_recal_batches", 8))
    eval_pool_size = int(eval_cfg.get("eval_pool_size", 12))
    eval_batches_per_arch = max(1, int(eval_cfg.get("eval_batches_per_arch", 4)))
    eval_every_epoch = max(1, int(eval_cfg.get("eval_every_epoch", 1)))
    patience = int(eval_cfg.get("early_stop_patience", 15))
    min_delta = float(eval_cfg.get("early_stop_min_delta", 0.002))
    logger.info("eval_every_epoch=%d", eval_every_epoch)
    logger.info("eval_batches_per_arch=%d", eval_batches_per_arch)
    logger.info("grad_clip_global_norm=%.4f", float(train_cfg.get("grad_clip_global_norm", 5.0)))
    logger.info("distill_enabled=%s", str(bool(distill_cfg.get("enabled", False))).lower())
    if bool(distill_cfg.get("enabled", False)):
        logger.info("distill_teacher_type=%s", _resolve_distill_teacher_type(distill_cfg.get("teacher_type", "edgeflownet")))
        logger.info("distill_lambda=%.4f", float(distill_cfg.get("lambda", 1.0)))
        logger.info("distill_teacher_ckpt=%s", str(distill_cfg.get("teacher_ckpt", "")).strip())

    train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    eval_train_provider = build_fc2_provider(config=config, split="train", seed_offset=1000, provider_mode="eval")
    eval_val_provider = build_fc2_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
    prefetch_batches = int(config.get("data", {}).get("prefetch_batches", 0))
    eval_prefetch_batches = int(config.get("data", {}).get("eval_prefetch_batches", 0))
    if prefetch_batches > 0:
        train_provider = PrefetchBatchProvider(train_provider, prefetch_batches=prefetch_batches)
    if eval_prefetch_batches > 0:
        eval_train_provider = PrefetchBatchProvider(eval_train_provider, prefetch_batches=eval_prefetch_batches)
        eval_val_provider = PrefetchBatchProvider(eval_val_provider, prefetch_batches=eval_prefetch_batches)
    logger.info("train_source=%s train_samples=%d", train_provider.source_dir, len(train_provider))
    logger.info("val_source=%s val_samples=%d", eval_val_provider.source_dir, len(eval_val_provider))
    logger.info("prefetch_batches train=%d eval=%d", prefetch_batches, eval_prefetch_batches)

    if len(train_provider) == 0:
        raise RuntimeError("train sample count is 0; aborting")
    if len(eval_train_provider) == 0:
        raise RuntimeError("eval train sample count is 0; aborting")
    if len(eval_val_provider) == 0:
        raise RuntimeError("val sample count is 0; aborting")

    if epoch_mode == "full_pass":
        steps_per_epoch = int(math.ceil(float(len(train_provider)) / float(max(1, batch_size))))
        if str(getattr(train_provider, "sampling_mode", "")).lower() != "shuffle_no_replacement":
            raise RuntimeError("epoch_mode=full_pass requires train_sampling_mode=shuffle_no_replacement")
    else:
        steps_per_epoch = int(train_cfg.get("steps_per_epoch", 50))
    steps_per_epoch = max(1, int(steps_per_epoch))
    logger.info(
        "epoch_mode=%s train_sampling_mode=%s train_samples=%d steps_per_epoch=%d",
        epoch_mode,
        str(getattr(train_provider, "sampling_mode", "unknown")),
        len(train_provider),
        steps_per_epoch,
    )

    eval_pool = build_eval_pool(seed=seed, size=eval_pool_size)
    eval_pool_cov = check_eval_pool_coverage(pool=eval_pool)
    write_json(str(experiment_dir / f"eval_pool_{eval_pool_size}.json"), {"pool": eval_pool, "coverage": eval_pool_cov})

    graph_obj = _build_graph(config=config)
    early_stop = EarlyStopState()
    checkpoint_paths = build_checkpoint_paths(str(experiment_dir))
    eval_rows = []
    sampler_rng = random.Random(seed)
    total_steps = max(1, num_epochs * max(1, steps_per_epoch))

    run_manifest = _build_run_manifest_v3(config=config, git_commit=_git_commit_hash())
    write_json(str(experiment_dir / "run_manifest.json"), run_manifest)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        restore_state = _try_restore_training_state(
            sess=sess,
            saver=graph_obj["saver"],
            config=config,
            experiment_dir=experiment_dir,
            logger=logger,
        )
        _try_restore_teacher_state(sess=sess, graph_obj=graph_obj, logger=logger)
        start_epoch = int(restore_state["start_epoch"])
        global_step = int(restore_state["global_step"])
        fairness_counts = _sanitize_fairness_counts(raw_counts=restore_state["fairness_counts"])
        early_stop.best_metric = float(restore_state["best_metric"])
        early_stop.bad_epochs = int(restore_state["bad_epochs"])
        last_eval_metric = float(restore_state.get("last_metric", early_stop.best_metric))
        if not np.isfinite(last_eval_metric):
            last_eval_metric = float(early_stop.best_metric) if np.isfinite(early_stop.best_metric) else float("inf")
        epochs_ran = 0
        distill_enabled = bool(graph_obj.get("distill_enabled", False))
        distill_teacher_type = str(graph_obj.get("distill_teacher_type", "edgeflownet"))
        multi_gpu_mode = str(graph_obj.get("multi_gpu_mode", "single_gpu"))
        logger.info("multi_gpu_mode=%s gpu_devices=%s", multi_gpu_mode, graph_obj.get("gpu_devices", []))
        teacher_arch_code_ph = graph_obj.get("teacher_arch_code_ph", None)
        teacher_arch_code = np.asarray(graph_obj.get("teacher_arch_code", []), dtype=np.int32)

        for epoch_idx in range(start_epoch, num_epochs + 1):
            epochs_ran += 1
            if hasattr(train_provider, "start_epoch"):
                train_provider.start_epoch(shuffle=True)
            step_iterator = tqdm(
                range(steps_per_epoch),
                total=steps_per_epoch,
                desc=f"train epoch {epoch_idx}/{num_epochs}",
                leave=False,
            )
            epoch_loss_sum = 0.0
            epoch_grad_norm_sum = 0.0
            epoch_step_count = 0
            epoch_clip_trigger_count = 0
            epoch_clip_check_count = 0
            epoch_grad_norm_values: List[float] = []
            epoch_distill_sum = 0.0
            epoch_lr_last = float(base_lr)
            for _ in step_iterator:
                cycle_codes = generate_fair_cycle(rng=sampler_rng, fairness_counts=fairness_counts)
                _update_fairness_counts(counts=fairness_counts, cycle_codes=cycle_codes)

                input_batch, _, _, label_batch = train_provider.next_batch(batch_size=batch_size)
                input_batch = standardize_image_tensor(input_batch)

                current_lr = cosine_lr(base_lr=base_lr, step_idx=global_step, total_steps=total_steps)
                epoch_lr_last = float(current_lr)
                micro_slices = [
                    (start, min(start + micro_batch_size, batch_size))
                    for start in range(0, batch_size, micro_batch_size)
                ]
                accum_runs = int(len(cycle_codes) * len(micro_slices))

                sess.run(graph_obj["zero_ops"])
                loss_val = 0.0
                grad_norm_val = 0.0

                if multi_gpu_mode == "arch_parallel":
                    for micro_idx, (start_idx, end_idx) in enumerate(micro_slices):
                        feed = {
                            graph_obj["input_ph"]: input_batch[start_idx:end_idx],
                            graph_obj["label_ph"]: label_batch[start_idx:end_idx],
                            graph_obj["arch_codes_ph"]: np.asarray(cycle_codes, dtype=np.int32),
                            graph_obj["is_training_ph"]: True,
                            graph_obj["lr_ph"]: current_lr,
                        }
                        if distill_enabled and distill_teacher_type == "supernet" and teacher_arch_code_ph is not None:
                            feed[teacher_arch_code_ph] = teacher_arch_code
                        if micro_idx == len(micro_slices) - 1:
                            if distill_enabled:
                                loss_val, distill_val, _ = sess.run(
                                    [graph_obj["loss"], graph_obj["loss_distill"], graph_obj["accum_op"]],
                                    feed_dict=feed,
                                )
                            else:
                                loss_val, _ = sess.run([graph_obj["loss"], graph_obj["accum_op"]], feed_dict=feed)
                                distill_val = 0.0
                        else:
                            sess.run(graph_obj["accum_op"], feed_dict=feed)
                else:
                    for arch_idx, arch_code in enumerate(cycle_codes):
                        for micro_idx, (start_idx, end_idx) in enumerate(micro_slices):
                            feed = {
                                graph_obj["input_ph"]: input_batch[start_idx:end_idx],
                                graph_obj["label_ph"]: label_batch[start_idx:end_idx],
                                graph_obj["arch_code_ph"]: arch_code,
                                graph_obj["is_training_ph"]: True,
                                graph_obj["lr_ph"]: current_lr,
                            }
                            if distill_enabled and distill_teacher_type == "supernet" and teacher_arch_code_ph is not None:
                                feed[teacher_arch_code_ph] = teacher_arch_code
                            is_last_accum = bool(
                                arch_idx == len(cycle_codes) - 1 and micro_idx == len(micro_slices) - 1
                            )
                            if is_last_accum:
                                if distill_enabled:
                                    loss_val, distill_val, _ = sess.run(
                                        [graph_obj["loss"], graph_obj["loss_distill"], graph_obj["accum_op"]],
                                        feed_dict=feed,
                                    )
                                else:
                                    loss_val, _ = sess.run(
                                        [graph_obj["loss"], graph_obj["accum_op"]],
                                        feed_dict=feed,
                                    )
                                    distill_val = 0.0
                            else:
                                sess.run(graph_obj["accum_op"], feed_dict=feed)

                grad_norm_val, clip_trigger_val, _ = sess.run(
                    [graph_obj["global_grad_norm"], graph_obj["clip_trigger"], graph_obj["apply_op"]],
                    feed_dict={
                        graph_obj["lr_ph"]: current_lr,
                        graph_obj["accum_divisor_ph"]: float(accum_runs),
                    },
                )
                epoch_grad_norm_values.append(float(grad_norm_val))
                epoch_clip_trigger_count += int(clip_trigger_val)
                epoch_clip_check_count += 1
                global_step += 1
                epoch_loss_sum += float(loss_val)
                epoch_distill_sum += float(distill_val)
                epoch_grad_norm_sum += float(grad_norm_val)
                epoch_step_count += 1
                step_iterator.set_postfix(loss=f"{float(loss_val):.4f}", lr=f"{float(current_lr):.2e}")

            step_iterator.close()
            train_loss_epoch_avg = epoch_loss_sum / max(1, epoch_step_count)
            train_grad_norm_epoch_avg = epoch_grad_norm_sum / max(1, epoch_step_count)
            if epoch_grad_norm_values:
                train_grad_norm_p50 = float(np.percentile(epoch_grad_norm_values, 50))
                train_grad_norm_p90 = float(np.percentile(epoch_grad_norm_values, 90))
                train_grad_norm_p99 = float(np.percentile(epoch_grad_norm_values, 99))
            else:
                train_grad_norm_p50 = 0.0
                train_grad_norm_p90 = 0.0
                train_grad_norm_p99 = 0.0
            clip_trigger_rate = float(epoch_clip_trigger_count) / max(1, epoch_clip_check_count)
            train_distill_loss_epoch_avg = epoch_distill_sum / max(1, epoch_step_count)
            should_eval = bool(epoch_idx % eval_every_epoch == 0)
            if should_eval:
                eval_info = _run_eval_epoch(
                    sess=sess,
                    graph_obj=graph_obj,
                    train_provider=eval_train_provider,
                    val_provider=eval_val_provider,
                    eval_pool=eval_pool,
                    bn_recal_batches=bn_recal_batches,
                    batch_size=batch_size,
                    eval_batches_per_arch=eval_batches_per_arch,
                )
                arch_rank_12 = _format_arch_ranking(eval_info.get("arch_ranking", []))
                row = {
                    "epoch": int(epoch_idx),
                    "mean_epe_12": float(eval_info["mean_epe_12"]),
                    "std_epe_12": float(eval_info["std_epe_12"]),
                    "fairness_gap": float(_fairness_gap(fairness_counts)),
                    "lr": float(epoch_lr_last),
                    "bn_recal_batches": float(bn_recal_batches),
                    "eval_batches_per_arch": float(eval_batches_per_arch),
                    "train_loss_epoch_avg": float(train_loss_epoch_avg),
                    "train_grad_norm_epoch_avg": float(train_grad_norm_epoch_avg),
                    "train_grad_norm_p50": float(train_grad_norm_p50),
                    "train_grad_norm_p90": float(train_grad_norm_p90),
                    "train_grad_norm_p99": float(train_grad_norm_p99),
                    "clip_trigger_count": float(epoch_clip_trigger_count),
                    "clip_trigger_rate": float(clip_trigger_rate),
                    "train_distill_loss_epoch_avg": float(train_distill_loss_epoch_avg),
                    "arch_rank_12": arch_rank_12,
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
                    "epoch=%d lr=%.2e loss=%.6f mean_epe_12=%.6f std_epe_12=%.6f fairness_gap=%.2f grad_norm=%.4f grad_p90=%.4f clip_count=%d clip_rate=%.4f arch_rank_12=%s",
                    epoch_idx,
                    float(row["lr"]),
                    float(train_loss_epoch_avg),
                    row["mean_epe_12"],
                    row["std_epe_12"],
                    row["fairness_gap"],
                    float(train_grad_norm_epoch_avg),
                    float(train_grad_norm_p90),
                    int(epoch_clip_trigger_count),
                    float(clip_trigger_rate),
                    arch_rank_12,
                )
                if distill_enabled:
                    logger.info("epoch=%d distill_loss=%.6f", epoch_idx, float(train_distill_loss_epoch_avg))
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
                        "train_grad_norm_p50": float(train_grad_norm_p50),
                        "train_grad_norm_p90": float(train_grad_norm_p90),
                        "train_grad_norm_p99": float(train_grad_norm_p99),
                        "clip_trigger_count": int(epoch_clip_trigger_count),
                        "clip_trigger_rate": float(clip_trigger_rate),
                        "train_distill_loss_epoch_avg": float(train_distill_loss_epoch_avg),
                    },
                )
                logger.info(
                    "epoch=%d lr=%.2e loss=%.6f distill=%.6f eval=skipped fairness_gap=%.2f grad_norm=%.4f grad_p90=%.4f clip_count=%d clip_rate=%.4f",
                    epoch_idx,
                    float(epoch_lr_last),
                    float(train_loss_epoch_avg),
                    float(train_distill_loss_epoch_avg),
                    float(_fairness_gap(fairness_counts)),
                    float(train_grad_norm_epoch_avg),
                    float(train_grad_norm_p90),
                    int(epoch_clip_trigger_count),
                    float(clip_trigger_rate),
                )

    _write_eval_history(csv_path=experiment_dir / "eval_epe_history.csv", rows=eval_rows)
    write_json(str(experiment_dir / "fairness_counts.json"), fairness_counts)

    manifest = _build_manifest_v3(config=config, git_commit=_git_commit_hash())
    write_json(str(experiment_dir / "train_manifest.json"), manifest)

    checkpoint_paths = build_checkpoint_paths(str(experiment_dir))
    report_path = experiment_dir / "supernet_training_report.md"
    report_path.write_text(
        "# Supernet V3 Training Report\n\n"
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

    for provider in (train_provider, eval_train_provider, eval_val_provider):
        if hasattr(provider, "close"):
            provider.close()

    logger.info("supernet V3 training finished")
    return 0

