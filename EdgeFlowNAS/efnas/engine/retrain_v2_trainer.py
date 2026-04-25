"""Stage trainer for V2 fixed-architecture retraining."""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from efnas.data.dataloader_builder import build_fc2_provider, build_ft3d_provider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.eval_step import accumulate_predictions, build_epe_metric
from efnas.engine.retrain_v2_sintel_runtime import (
    _append_sintel_metrics,
    _resolve_sintel_eval_config,
    evaluate_model_graph_on_sintel,
)
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
from efnas.nas.arch_codec_v2 import parse_arch_code_text
from efnas.network.fixed_arch_models_v2 import FixedArchModelV2
from efnas.network.MultiScaleResNet_supernet_v2 import MultiScaleResNetSupernetV2
from efnas.utils.json_io import read_json, write_json
from efnas.utils.logger import build_logger
from efnas.utils.seed import set_global_seed


def _parse_arch_codes(raw: str) -> List[List[int]]:
    groups = [part.strip() for part in str(raw).split("+") if part.strip()]
    if not groups:
        raise ValueError("arch_codes must contain at least one 11D arch code")
    return [parse_arch_code_text(group) for group in groups]


def _parse_model_names(raw: Optional[str], arch_codes: List[List[int]]) -> List[str]:
    if not raw or not str(raw).strip():
        if len(arch_codes) == 1:
            return ["model"]
        return [f"candidate_{i}" for i in range(len(arch_codes))]
    names = [part.strip() for part in str(raw).split("+") if part.strip()]
    if len(names) != len(arch_codes):
        raise ValueError("model_names count must match arch_codes count")
    return names


def _build_provider(config: Dict[str, Any], split: str, seed_offset: int, provider_mode: str):
    dataset = str(config.get("data", {}).get("dataset", "FC2")).strip().upper()
    if dataset == "FC2":
        return build_fc2_provider(config=config, split=split, seed_offset=seed_offset, provider_mode=provider_mode)
    if dataset == "FT3D":
        return build_ft3d_provider(config=config, split=split, seed_offset=seed_offset, provider_mode=provider_mode)
    raise ValueError(f"unsupported dataset for retrain_v2: {dataset}")


def _normalize_supernet_var_name(var_name: str) -> str:
    parts = []
    for part in str(var_name).split("/"):
        part = re.sub(r"^(conv_bn_relu)\d+$", r"\1", part)
        part = re.sub(r"^(resize_conv)\d+$", r"\1", part)
        part = re.sub(r"^(conv)\d+$", r"\1", part)
        part = re.sub(r"^(bn)\d+$", r"\1", part)
        parts.append(part)
    return "/".join(parts)


def _build_supernet_source_name_map(pred_channels: int) -> Dict[str, str]:
    with tf.Graph().as_default():
        tf.compat.v1.disable_eager_execution()
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 64, 64, 6], name="warm_input_ph")
        arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[11], name="warm_arch_code_ph")
        is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="warm_is_training_ph")
        model = MultiScaleResNetSupernetV2(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=pred_channels,
            init_neurons=32,
            expansion_factor=2.0,
        )
        model.build()
        source_vars = tf.compat.v1.global_variables()
        source_map: Dict[str, str] = {}
        for var in source_vars:
            normalized = _normalize_supernet_var_name(var.op.name)
            source_map[normalized] = var.op.name
        return source_map


def _build_warmstart_var_map(
    scope_name: str,
    scope_global_vars: List[tf.Variable],
    source_name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, tf.Variable]:
    warmstart_map: Dict[str, tf.Variable] = {}
    for var in scope_global_vars:
        var_name = var.op.name
        if "/Adam" in var_name:
            continue
        if var_name in (f"{scope_name}/beta1_power", f"{scope_name}/beta2_power"):
            continue
        warm_name = var_name[len(scope_name) + 1 :]
        if source_name_map is not None:
            normalized = _normalize_supernet_var_name(warm_name)
            source_name = source_name_map.get(normalized)
            if source_name is None:
                raise KeyError(f"missing source mapping for retrain var '{warm_name}'")
            warm_name = source_name
        warmstart_map[warm_name] = var
    return warmstart_map


def _build_single_model_graph(
    scope_name: str,
    arch_code: List[int],
    input_ph: tf.Tensor,
    label_ph: tf.Tensor,
    lr_ph: tf.Tensor,
    is_training_ph: tf.Tensor,
    flow_channels: int,
    pred_channels: int,
    weight_decay: float,
    grad_clip_global_norm: float,
    supernet_source_name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    with tf.compat.v1.variable_scope(scope_name):
        model = FixedArchModelV2(
            input_ph=input_ph,
            is_training_ph=is_training_ph,
            arch_code=arch_code,
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

        loss_tensor = add_weight_decay(loss_tensor=loss_core, weight_decay=weight_decay, trainable_vars=trainable_vars)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_ph, beta1=0.9, beta2=0.999, epsilon=1e-8)
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
        warmstart_map = _build_warmstart_var_map(
            scope_name=scope_name,
            scope_global_vars=scope_global_vars,
            source_name_map=supernet_source_name_map,
        )
        warmstart_saver = tf.compat.v1.train.Saver(var_list=warmstart_map)

    return {
        "scope_name": scope_name,
        "arch_code": [int(v) for v in arch_code],
        "loss": loss_tensor,
        "loss_core": loss_core,
        "loss_optical": loss_terms["optical_total"],
        "loss_uncertainty": loss_terms["uncertainty_total"],
        "train_op": train_op,
        "grad_norm": grad_norm,
        "epe": epe_tensor,
        "pred_accum": pred_accum,
        "saver": saver,
        "warmstart_saver": warmstart_saver,
        "trainable_vars": trainable_vars,
        "scope_global_vars": scope_global_vars,
        "warmstart_keys": sorted(warmstart_map.keys()),
    }


def _resolve_init_checkpoint_path(config: Dict[str, Any], model_name: str) -> Optional[Path]:
    checkpoint_cfg = config.get("checkpoint", {})
    init_mode = str(checkpoint_cfg.get("init_mode", "")).strip().lower()
    if not init_mode or init_mode == "none":
        return None
    if init_mode == "supernet":
        raw = str(checkpoint_cfg.get("init_checkpoint_path", "")).strip()
        if not raw:
            raise ValueError("checkpoint.init_checkpoint_path is required when init_mode=supernet")
        return Path(raw)
    if init_mode == "experiment_dir":
        exp_dir = str(checkpoint_cfg.get("init_experiment_dir", "")).strip()
        ckpt_name = str(checkpoint_cfg.get("init_ckpt_name", "best")).strip()
        if not exp_dir:
            raise ValueError("checkpoint.init_experiment_dir is required when init_mode=experiment_dir")
        return Path(exp_dir) / f"model_{model_name}" / "checkpoints" / f"{ckpt_name}.ckpt"
    raise ValueError(f"unsupported checkpoint.init_mode: {init_mode}")


def _normalize_checkpoint_prefix(path_prefix: Path) -> Path:
    if path_prefix.suffix == ".ckpt":
        return path_prefix
    return Path(str(path_prefix).replace(".index", "").replace(".meta", ""))


def _load_trainer_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(str(path))
    return payload if isinstance(payload, dict) else {}


def train_retrain_v2(config: Dict[str, Any]) -> int:
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    data_cfg = config.get("data", {})
    checkpoint_cfg = config.get("checkpoint", {})

    seed = int(runtime_cfg.get("seed", 42))
    set_global_seed(seed)

    experiment_dir = _resolve_output_dir(config)
    logger = build_logger("edgeflownas_retrain_v2", str(experiment_dir / "train.log"))
    logger.info("start retrain_v2")
    _apply_gpu_device_setting(train_cfg=train_cfg, logger=logger)

    arch_codes = _parse_arch_codes(config.get("arch_codes", ""))
    model_names = _parse_model_names(config.get("model_names", None), arch_codes)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_epochs = int(train_cfg.get("num_epochs", 400))
    base_lr = float(train_cfg.get("lr", 1e-4))
    lr_min = float(train_cfg.get("lr_min", 1e-6))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip_global_norm", 0.0))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
    eval_every_epoch = max(1, int(eval_cfg.get("eval_every_epoch", 5)))
    eval_batches = int(eval_cfg.get("eval_batches", 0))
    input_h = int(data_cfg.get("input_height", 352))
    input_w = int(data_cfg.get("input_width", 480))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = flow_channels * 2
    supernet_source_name_map = _build_supernet_source_name_map(pred_channels=pred_channels)
    sintel_eval_cfg = _resolve_sintel_eval_config(config)
    sintel_ckpt_name = str(sintel_eval_cfg.get("ckpt_name", "sintel_best")).strip() if sintel_eval_cfg is not None else "sintel_best"
    prediction_flow_scale = float(data_cfg.get("ft3d_flow_divisor", 1.0)) if str(data_cfg.get("dataset", "")).strip().upper() == "FT3D" else 1.0

    train_provider = _build_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    val_provider = _build_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
    if len(train_provider) == 0:
        raise RuntimeError("train split is empty")
    if len(val_provider) == 0:
        raise RuntimeError("val split is empty")

    steps_per_epoch = int(math.ceil(float(len(train_provider)) / float(max(1, batch_size))))
    total_steps = max(1, num_epochs * steps_per_epoch)

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, flow_channels], name="Label")
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")

    model_graphs: Dict[str, Dict[str, Any]] = {}
    for name, code in zip(model_names, arch_codes):
        logger.info("build graph: model=%s arch=%s", name, ",".join(str(v) for v in code))
        model_graphs[name] = _build_single_model_graph(
            scope_name=name,
            arch_code=code,
            input_ph=input_ph,
            label_ph=label_ph,
            lr_ph=lr_ph,
            is_training_ph=is_training_ph,
            flow_channels=flow_channels,
            pred_channels=pred_channels,
            weight_decay=weight_decay,
            grad_clip_global_norm=grad_clip,
            supernet_source_name_map=supernet_source_name_map,
        )

    model_dirs: Dict[str, Path] = {}
    model_ckpt_paths: Dict[str, Dict[str, Path]] = {}
    for name in model_names:
        mdir = experiment_dir / f"model_{name}"
        mdir.mkdir(parents=True, exist_ok=True)
        model_dirs[name] = mdir
        model_ckpt_paths[name] = _build_standalone_checkpoint_paths(mdir)

    trainer_state_path = experiment_dir / "trainer_state.json"
    write_json(
        str(experiment_dir / "run_manifest.json"),
        {
            "arch_codes": {name: code for name, code in zip(model_names, arch_codes)},
            "model_names": model_names,
            "config": config,
            "git_commit": _git_commit_hash(),
        },
    )

    best_epe: Dict[str, float] = {name: float("inf") for name in model_names}
    best_sintel_epe: Dict[str, float] = {name: float("inf") for name in model_names}
    eval_histories: Dict[str, List[Dict[str, Any]]] = {name: [] for name in model_names}
    comparison_rows: List[Dict[str, Any]] = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        global_step = 0
        start_epoch = 1
        best_mean_epe = float("inf")
        no_improve_evals = 0

        if bool(checkpoint_cfg.get("load_checkpoint", False)):
            resume_dir = _resolve_resume_dir(config=config, experiment_dir=experiment_dir)
            trainer_state = _load_trainer_state(resume_dir / "trainer_state.json")
            start_epoch = int(trainer_state.get("epoch", 0)) + 1 if trainer_state else 1
            global_step = int(trainer_state.get("global_step", 0)) if trainer_state else 0
            best_mean_epe = float(trainer_state.get("best_mean_epe", float("inf"))) if trainer_state else float("inf")
            no_improve_evals = int(trainer_state.get("no_improve_evals", 0)) if trainer_state else 0
            best_sintel_payload = trainer_state.get("best_sintel_epe", {}) if trainer_state else {}
            for name, mg in model_graphs.items():
                resume_ckpt = resume_dir / f"model_{name}" / "checkpoints" / "last.ckpt"
                if not Path(str(resume_ckpt) + ".index").exists():
                    raise FileNotFoundError(f"resume checkpoint not found for model={name}: {resume_ckpt}")
                mg["saver"].restore(sess, str(resume_ckpt))
                meta_path = Path(str(resume_ckpt) + ".meta.json")
                if meta_path.exists():
                    meta = read_json(str(meta_path))
                    best_epe[name] = float(meta.get("best_metric", float("inf")))
                best_sintel_epe[name] = float(best_sintel_payload.get(name, float("inf")))
        else:
            init_mode = str(checkpoint_cfg.get("init_mode", "")).strip().lower()
            if init_mode and init_mode != "none":
                for name, mg in model_graphs.items():
                    ckpt_path = _normalize_checkpoint_prefix(_resolve_init_checkpoint_path(config=config, model_name=name))
                    if not Path(str(ckpt_path) + ".index").exists():
                        raise FileNotFoundError(f"init checkpoint not found for model={name}: {ckpt_path}")
                    if init_mode == "supernet":
                        mg["warmstart_saver"].restore(sess, str(ckpt_path))
                    else:
                        mg["saver"].restore(sess, str(ckpt_path))

        for epoch_idx in range(start_epoch, num_epochs + 1):
            if hasattr(train_provider, "start_epoch"):
                train_provider.start_epoch(shuffle=True)

            train_fetch = {}
            for name, mg in model_graphs.items():
                train_fetch[f"loss_{name}"] = mg["loss"]
                train_fetch[f"train_{name}"] = mg["train_op"]

            epoch_losses = {name: 0.0 for name in model_names}
            step_count = 0
            step_iter = tqdm(range(steps_per_epoch), total=steps_per_epoch, desc=f"epoch {epoch_idx}/{num_epochs}", leave=False)
            for _ in step_iter:
                input_batch, _, _, label_batch = train_provider.next_batch(batch_size=batch_size)
                input_batch = standardize_image_tensor(input_batch)
                current_lr = _cosine_lr_with_min(base_lr=base_lr, lr_min=lr_min, step_idx=global_step, total_steps=total_steps)
                feed = {input_ph: input_batch, label_ph: label_batch, lr_ph: current_lr, is_training_ph: True}
                results = sess.run(train_fetch, feed_dict=feed)
                for name in model_names:
                    epoch_losses[name] += float(results[f"loss_{name}"])
                step_count += 1
                global_step += 1

            avg_losses = {name: epoch_losses[name] / max(1, step_count) for name in model_names}
            do_eval = (epoch_idx % eval_every_epoch == 0) or (epoch_idx == num_epochs)
            epoch_epes: Dict[str, float] = {}
            sintel_epes: Dict[str, float] = {}
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
                if sintel_eval_cfg is not None:
                    for name, mg in model_graphs.items():
                        sintel_epes[name] = evaluate_model_graph_on_sintel(
                            sess=sess,
                            input_ph=input_ph,
                            is_training_ph=is_training_ph,
                            pred_tensor=mg["pred_accum"],
                            flow_scale=prediction_flow_scale,
                            dataset_root=str(sintel_eval_cfg["dataset_root"]),
                            sintel_list=str(sintel_eval_cfg["sintel_list"]),
                            patch_size=tuple(sintel_eval_cfg["patch_size"]),
                            max_samples=sintel_eval_cfg.get("max_samples", None),
                            progress_desc=f"Sintel {name} e{epoch_idx}",
                        )

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
                if do_eval and name in sintel_epes and sintel_epes[name] < best_sintel_epe[name]:
                    best_sintel_epe[name] = float(sintel_epes[name])
                    _save_standalone_checkpoint(
                        sess=sess,
                        saver=mg["saver"],
                        path_prefix=model_ckpt_paths[name]["root"] / f"{sintel_ckpt_name}.ckpt",
                        epoch=epoch_idx,
                        global_step=global_step,
                        metric=sintel_epes[name],
                        best_metric=best_sintel_epe[name],
                        arch_code=mg["arch_code"],
                    )

            if do_eval:
                mean_epe = float(np.mean(list(epoch_epes.values())))
                if mean_epe + early_stop_min_delta < best_mean_epe:
                    best_mean_epe = mean_epe
                    no_improve_evals = 0
                else:
                    no_improve_evals += 1

                for name in model_names:
                    history_row = {"epoch": epoch_idx, "loss": avg_losses[name], "epe": epoch_epes[name], "best_epe": best_epe[name]}
                    if name in sintel_epes:
                        history_row["sintel_epe"] = float(sintel_epes[name])
                        history_row["best_sintel_epe"] = float(best_sintel_epe[name])
                    eval_histories[name].append(history_row)
                    _write_eval_history(model_dirs[name] / "eval_history.csv", eval_histories[name])
                comparison_row = {"epoch": epoch_idx, "mean_epe": mean_epe, **{f"epe_{n}": epoch_epes[n] for n in model_names}}
                if sintel_epes:
                    comparison_row = _append_sintel_metrics(comparison_row, sintel_epes=sintel_epes, model_names=model_names)
                comparison_rows.append(comparison_row)
                _write_eval_history(experiment_dir / "comparison.csv", comparison_rows)

            write_json(
                str(trainer_state_path),
                {
                    "epoch": epoch_idx,
                    "global_step": global_step,
                    "best_epe": best_epe,
                    "best_sintel_epe": best_sintel_epe,
                    "best_mean_epe": best_mean_epe,
                    "no_improve_evals": no_improve_evals,
                    "dataset": data_cfg.get("dataset", "FC2"),
                    "model_names": model_names,
                },
            )

            if do_eval and early_stop_patience > 0 and no_improve_evals >= early_stop_patience:
                logger.info("early stop triggered at epoch=%d", epoch_idx)
                break

    logger.info("retrain_v2 finished")
    return 0
