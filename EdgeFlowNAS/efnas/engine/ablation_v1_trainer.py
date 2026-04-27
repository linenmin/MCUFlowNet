"""Trainer for Ablation V1 backbone experiments."""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from efnas.data.dataloader_builder import build_fc2_provider, build_ft3d_provider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.ablation_v1_sintel_runtime import evaluate_ablation_checkpoint_dir_on_sintel
from efnas.engine.eval_step import accumulate_predictions, build_epe_metric
from efnas.engine.retrain_v2_resume import _reconcile_resume_progress, _trim_retrain_histories
from efnas.engine.retrain_v2_sintel_runtime import (
    _append_sintel_metrics,
    _resolve_sintel_eval_config,
    _restore_retrain_histories,
)
from efnas.engine.standalone_trainer import (
    _apply_gpu_device_setting,
    _build_standalone_checkpoint_paths,
    _cosine_lr_with_min,
    _git_commit_hash,
    _resolve_output_dir,
    _resolve_resume_dir,
    _save_standalone_checkpoint,
    _write_eval_history,
)
from efnas.engine.train_step import add_weight_decay, build_multiscale_uncertainty_loss
from efnas.network.ablation_edgeflownet_v1 import ABlationEdgeFlowNetV1, build_ablation_variants
from efnas.utils.json_io import read_json, write_json
from efnas.utils.logger import build_logger
from efnas.utils.seed import set_global_seed


def _build_provider(config: Dict[str, Any], split: str, seed_offset: int, provider_mode: str):
    dataset = str(config.get("data", {}).get("dataset", "FC2")).strip().upper()
    if dataset == "FC2":
        return build_fc2_provider(config=config, split=split, seed_offset=seed_offset, provider_mode=provider_mode)
    if dataset == "FT3D":
        return build_ft3d_provider(config=config, split=split, seed_offset=seed_offset, provider_mode=provider_mode)
    raise ValueError(f"unsupported dataset for ablation_v1: {dataset}")


def _summarize_grad_norms(values: List[float], clip_threshold: float) -> Dict[str, float]:
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


def _load_trainer_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(str(path))
    return payload if isinstance(payload, dict) else {}


def _save_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(str(tmp), payload)
    tmp.replace(path)


def _resume_signature(config: Dict[str, Any], variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("eval", {})
    return {
        "variants": variants,
        "dataset": str(data_cfg.get("dataset", "")),
        "input_height": int(data_cfg.get("input_height", 0)),
        "input_width": int(data_cfg.get("input_width", 0)),
        "flow_channels": int(data_cfg.get("flow_channels", 2)),
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "num_epochs": int(train_cfg.get("num_epochs", 0)),
        "lr": float(train_cfg.get("lr", 0.0)),
        "lr_min": float(train_cfg.get("lr_min", 0.0)),
        "grad_clip_global_norm": float(train_cfg.get("grad_clip_global_norm", 0.0)),
        "eval_every_epoch": int(eval_cfg.get("eval_every_epoch", 0)),
    }


def _check_resume_signature(resume_dir: Path, signature: Dict[str, Any], allow_mismatch: bool) -> None:
    manifest_path = resume_dir / "run_manifest.json"
    if not manifest_path.exists():
        if allow_mismatch:
            return
        raise FileNotFoundError(f"resume run_manifest.json not found: {manifest_path}")
    manifest = read_json(str(manifest_path))
    old_signature = manifest.get("resume_signature", {}) if isinstance(manifest, dict) else {}
    if old_signature != signature and not allow_mismatch:
        raise ValueError("resume config signature mismatch; pass --allow_config_mismatch only if intentional")


def _save_checkpoint_with_variant(sess, saver, path_prefix: Path, epoch: int, global_step: int, metric: float, best_metric: float, variant_config: Dict[str, Any]):
    path = _save_standalone_checkpoint(
        sess=sess,
        saver=saver,
        path_prefix=path_prefix,
        epoch=epoch,
        global_step=global_step,
        metric=metric,
        best_metric=best_metric,
        arch_code=[],
    )
    meta_path = Path(str(path_prefix) + ".meta.json")
    meta = read_json(str(meta_path))
    meta["variant_config"] = variant_config
    meta["init_neurons"] = 32
    meta["expansion_factor"] = 2.0
    write_json(str(meta_path), meta)
    return path


def _build_single_model_graph(scope_name: str, variant_config: Dict[str, Any], input_ph, label_ph, lr_ph, is_training_ph, flow_channels: int, pred_channels: int, weight_decay: float, grad_clip_global_norm: float) -> Dict[str, Any]:
    with tf.compat.v1.variable_scope(scope_name):
        model = ABlationEdgeFlowNetV1(
            input_ph=input_ph,
            is_training_ph=is_training_ph,
            num_out=pred_channels,
            variant_config=variant_config,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()
        loss_terms = build_multiscale_uncertainty_loss(preds=preds, label_ph=label_ph, num_out=flow_channels, return_terms=True)
        loss_tensor = add_weight_decay(
            loss_tensor=loss_terms["total"],
            weight_decay=weight_decay,
            trainable_vars=tf.compat.v1.trainable_variables(scope=scope_name),
        )
        trainable_vars = tf.compat.v1.trainable_variables(scope=scope_name)
        if not trainable_vars:
            raise RuntimeError(f"scope '{scope_name}' has no trainable variables")
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_ph, beta1=0.9, beta2=0.999, epsilon=1e-8)
        grads_and_vars = optimizer.compute_gradients(loss_tensor, var_list=trainable_vars)
        bn_updates = [op for op in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS) if op.name.startswith(f"{scope_name}/")]
        grads = [g for g, _ in grads_and_vars if g is not None]
        vars_ = [v for g, v in grads_and_vars if g is not None]
        if grad_clip_global_norm > 0:
            clipped_grads, grad_norm = tf.clip_by_global_norm(grads, clip_norm=grad_clip_global_norm)
            apply_pairs = list(zip(clipped_grads, vars_))
        else:
            grad_norm = tf.linalg.global_norm(grads) if grads else tf.constant(0.0, dtype=tf.float32)
            apply_pairs = [(g, v) for g, v in grads_and_vars if g is not None]
        with tf.control_dependencies(bn_updates):
            train_op = optimizer.apply_gradients(apply_pairs)
        pred_accum = accumulate_predictions(preds)
        epe_tensor = build_epe_metric(pred_tensor=pred_accum, label_ph=label_ph, num_out=flow_channels)
        scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{scope_name}/")]
        saver = tf.compat.v1.train.Saver(var_list=scope_global_vars, max_to_keep=3)
    return {
        "scope_name": scope_name,
        "variant_config": variant_config,
        "loss": loss_tensor,
        "loss_core": loss_terms["total"],
        "loss_optical": loss_terms["optical_total"],
        "loss_uncertainty": loss_terms["uncertainty_total"],
        "train_op": train_op,
        "grad_norm": grad_norm,
        "epe": epe_tensor,
        "saver": saver,
        "trainable_vars": trainable_vars,
    }


def _evaluate_model_with_progress(sess, model_graph: Dict[str, Any], input_ph, label_ph, is_training_ph, val_provider, batch_size: int, eval_batches: int, desc: str) -> float:
    if hasattr(val_provider, "reset_cursor"):
        val_provider.reset_cursor(0)
    total_samples = len(val_provider)
    num_batches = max(1, int(math.ceil(total_samples / batch_size))) if eval_batches <= 0 else int(eval_batches)
    batch_epes: List[float] = []
    iterator = tqdm(range(num_batches), total=num_batches, desc=desc, leave=False, unit="batch")
    for _ in iterator:
        input_batch, _, _, label_batch = val_provider.next_batch(batch_size=batch_size)
        input_batch = standardize_image_tensor(input_batch)
        epe_val = sess.run(
            model_graph["epe"],
            feed_dict={input_ph: input_batch, label_ph: label_batch, is_training_ph: True},
        )
        batch_epes.append(float(epe_val))
        iterator.set_postfix(epe=f"{float(np.mean(batch_epes)):.4f}")
    return float(np.mean(batch_epes)) if batch_epes else 0.0


def _resolve_init_checkpoint_path(config: Dict[str, Any], model_name: str) -> Optional[Path]:
    checkpoint_cfg = config.get("checkpoint", {})
    init_mode = str(checkpoint_cfg.get("init_mode", "")).strip().lower()
    if not init_mode or init_mode == "none":
        return None
    if init_mode != "experiment_dir":
        raise ValueError(f"unsupported ablation checkpoint.init_mode: {init_mode}")
    exp_dir = str(checkpoint_cfg.get("init_experiment_dir", "")).strip()
    ckpt_name = str(checkpoint_cfg.get("init_ckpt_name", "best")).strip()
    if not exp_dir:
        raise ValueError("checkpoint.init_experiment_dir is required when init_mode=experiment_dir")
    return Path(exp_dir) / f"model_{model_name}" / "checkpoints" / f"{ckpt_name}.ckpt"


def train_ablation_v1(config: Dict[str, Any]) -> int:
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
    logger = build_logger("edgeflownas_ablation_v1", str(experiment_dir / "train.log"))
    logger.info("start ablation_v1")
    _apply_gpu_device_setting(train_cfg=train_cfg, logger=logger)

    variants = build_ablation_variants(config.get("variants", None))
    model_names = [str(v["name"]) for v in variants]
    batch_size = int(train_cfg.get("batch_size", 32))
    num_epochs = int(train_cfg.get("num_epochs", 400))
    base_lr = float(train_cfg.get("lr", 1e-4))
    lr_min = float(train_cfg.get("lr_min", 1e-6))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip_global_norm", 200.0))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
    eval_every_epoch = max(1, int(eval_cfg.get("eval_every_epoch", 5)))
    eval_batches = int(eval_cfg.get("eval_batches", 0))
    input_h = int(data_cfg.get("input_height", 352))
    input_w = int(data_cfg.get("input_width", 480))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = flow_channels * 2
    sintel_eval_cfg = _resolve_sintel_eval_config(config)
    sintel_ckpt_name = str(sintel_eval_cfg.get("ckpt_name", "sintel_best")).strip() if sintel_eval_cfg is not None else "sintel_best"

    train_provider = _build_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    val_provider = _build_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
    if len(train_provider) == 0:
        raise RuntimeError("train split is empty")
    if len(val_provider) == 0:
        raise RuntimeError("val split is empty")
    steps_per_epoch = int(math.ceil(float(len(train_provider)) / float(max(1, batch_size))))
    total_steps = max(1, num_epochs * steps_per_epoch)
    logger.info("output_dir=%s git_commit=%s", experiment_dir, _git_commit_hash())
    logger.info("dataset=%s train_samples=%d val_samples=%d input=%dx%d batch=%d workers(fc2/ft3d)=%s/%s", data_cfg.get("dataset"), len(train_provider), len(val_provider), input_h, input_w, batch_size, data_cfg.get("fc2_num_workers", ""), data_cfg.get("ft3d_num_workers", ""))
    logger.info("optimizer=Adam lr=%.2e lr_min=%.2e grad_clip_global_norm=%.1f early_stop_patience=%d eval_every=%d steps_per_epoch=%d", base_lr, lr_min, grad_clip, early_stop_patience, eval_every_epoch, steps_per_epoch)

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, flow_channels], name="Label")
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")

    model_graphs: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        name = str(variant["name"])
        logger.info("build graph model=%s variant=%s", name, variant)
        model_graphs[name] = _build_single_model_graph(name, variant, input_ph, label_ph, lr_ph, is_training_ph, flow_channels, pred_channels, weight_decay, grad_clip)

    model_dirs: Dict[str, Path] = {}
    model_ckpt_paths: Dict[str, Dict[str, Path]] = {}
    for name in model_names:
        mdir = experiment_dir / f"model_{name}"
        mdir.mkdir(parents=True, exist_ok=True)
        write_json(str(mdir / "variant_config.json"), model_graphs[name]["variant_config"])
        model_dirs[name] = mdir
        model_ckpt_paths[name] = _build_standalone_checkpoint_paths(mdir)

    trainer_state_path = experiment_dir / "trainer_state.json"
    signature = _resume_signature(config, variants)
    if bool(checkpoint_cfg.get("load_checkpoint", False)):
        _check_resume_signature(
            resume_dir=_resolve_resume_dir(config=config, experiment_dir=experiment_dir),
            signature=signature,
            allow_mismatch=bool(checkpoint_cfg.get("allow_config_mismatch", False)),
        )
    write_json(str(experiment_dir / "run_manifest.json"), {"variants": variants, "resume_signature": signature, "config": config, "git_commit": _git_commit_hash()})
    best_epe = {name: float("inf") for name in model_names}
    best_sintel_epe = {name: float("inf") for name in model_names}
    eval_histories = {name: [] for name in model_names}
    comparison_rows: List[Dict[str, Any]] = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        global_step = 0
        start_epoch = 1
        best_mean_epe = float("inf")
        no_improve_evals = 0

        if bool(checkpoint_cfg.get("load_checkpoint", False)):
            resume_dir = _resolve_resume_dir(config=config, experiment_dir=experiment_dir)
            resume_ckpt_name = str(checkpoint_cfg.get("resume_ckpt_name", "last")).strip() or "last"
            prefer_checkpoint_meta = resume_ckpt_name != "last"
            trainer_state = _load_trainer_state(resume_dir / "trainer_state.json")
            best_mean_epe = float(trainer_state.get("best_mean_epe", float("inf"))) if trainer_state else float("inf")
            no_improve_evals = 0 if prefer_checkpoint_meta else int(trainer_state.get("no_improve_evals", 0)) if trainer_state else 0
            checkpoint_metas: List[Dict[str, Any]] = []
            for name, mg in model_graphs.items():
                resume_ckpt = resume_dir / f"model_{name}" / "checkpoints" / f"{resume_ckpt_name}.ckpt"
                if not Path(str(resume_ckpt) + ".index").exists():
                    raise FileNotFoundError(f"resume checkpoint not found for model={name}: {resume_ckpt}")
                mg["saver"].restore(sess, str(resume_ckpt))
                logger.info("restored model=%s checkpoint=%s", name, resume_ckpt)
                meta_path = Path(str(resume_ckpt) + ".meta.json")
                if meta_path.exists():
                    meta = read_json(str(meta_path))
                    best_epe[name] = float(meta.get("best_metric", float("inf")))
                    checkpoint_metas.append(meta if isinstance(meta, dict) else {})
            best_sintel_epe.update({k: float(v) for k, v in dict(trainer_state.get("best_sintel_epe", {})).items()} if trainer_state else {})
            start_epoch, global_step = _reconcile_resume_progress(trainer_state, checkpoint_metas, prefer_checkpoint_meta=prefer_checkpoint_meta)
            eval_histories, comparison_rows = _restore_retrain_histories(base_dir=resume_dir, model_names=model_names)
            if prefer_checkpoint_meta:
                resume_epoch = max([int(meta.get("epoch", 0)) for meta in checkpoint_metas] or [0])
                eval_histories, comparison_rows = _trim_retrain_histories(eval_histories, comparison_rows, max_epoch=resume_epoch)
                logger.info("trimmed histories to epoch=%d", resume_epoch)
            logger.info("resume start_epoch=%d global_step=%d", start_epoch, global_step)
        else:
            init_mode = str(checkpoint_cfg.get("init_mode", "")).strip().lower()
            if init_mode and init_mode != "none":
                for name, mg in model_graphs.items():
                    ckpt_path = _resolve_init_checkpoint_path(config, name)
                    if ckpt_path is None or not Path(str(ckpt_path) + ".index").exists():
                        raise FileNotFoundError(f"init checkpoint not found for model={name}: {ckpt_path}")
                    mg["saver"].restore(sess, str(ckpt_path))
                    logger.info("initialized model=%s checkpoint=%s", name, ckpt_path)

        for epoch_idx in range(start_epoch, num_epochs + 1):
            epoch_start = time.time()
            if hasattr(train_provider, "start_epoch"):
                train_provider.start_epoch(shuffle=True)
            train_fetch = {}
            for name, mg in model_graphs.items():
                train_fetch[f"loss_{name}"] = mg["loss"]
                train_fetch[f"grad_norm_{name}"] = mg["grad_norm"]
                train_fetch[f"train_{name}"] = mg["train_op"]

            epoch_losses = {name: 0.0 for name in model_names}
            grad_norms = {name: [] for name in model_names}
            lr_first = None
            lr_last = None
            iterator = tqdm(range(steps_per_epoch), total=steps_per_epoch, desc=f"epoch {epoch_idx}/{num_epochs}", leave=False)
            for _ in iterator:
                input_batch, _, _, label_batch = train_provider.next_batch(batch_size=batch_size)
                input_batch = standardize_image_tensor(input_batch)
                current_lr = _cosine_lr_with_min(base_lr, lr_min, global_step, total_steps)
                lr_first = current_lr if lr_first is None else lr_first
                lr_last = current_lr
                results = sess.run(train_fetch, feed_dict={input_ph: input_batch, label_ph: label_batch, lr_ph: current_lr, is_training_ph: True})
                for name in model_names:
                    epoch_losses[name] += float(results[f"loss_{name}"])
                    grad_norms[name].append(float(results[f"grad_norm_{name}"]))
                global_step += 1
                iterator.set_postfix(lr=f"{current_lr:.2e}", loss=f"{np.mean([epoch_losses[n] for n in model_names]) / max(1, len(grad_norms[model_names[0]])):.4f}")

            avg_losses = {name: epoch_losses[name] / max(1, steps_per_epoch) for name in model_names}
            grad_stats = {name: _summarize_grad_norms(grad_norms[name], grad_clip) for name in model_names}
            do_eval = (epoch_idx % eval_every_epoch == 0) or (epoch_idx == num_epochs)
            epoch_epes: Dict[str, float] = {}
            sintel_epes: Dict[str, float] = {}
            if do_eval:
                for name, mg in model_graphs.items():
                    epoch_epes[name] = _evaluate_model_with_progress(sess, mg, input_ph, label_ph, is_training_ph, val_provider, batch_size, eval_batches, desc=f"val {name} e{epoch_idx}")

            for name, mg in model_graphs.items():
                metric_val = epoch_epes.get(name, float("inf"))
                _save_checkpoint_with_variant(sess, mg["saver"], model_ckpt_paths[name]["last"], epoch_idx, global_step, metric_val, best_epe[name], mg["variant_config"])
                if do_eval and metric_val < best_epe[name]:
                    best_epe[name] = metric_val
                    _save_checkpoint_with_variant(sess, mg["saver"], model_ckpt_paths[name]["best"], epoch_idx, global_step, metric_val, best_epe[name], mg["variant_config"])
                    logger.info("best updated model=%s epe=%.4f", name, metric_val)

            if do_eval and sintel_eval_cfg is not None:
                for name in model_names:
                    sintel_result = evaluate_ablation_checkpoint_dir_on_sintel(
                        model_dir=model_dirs[name],
                        dataset_root=str(sintel_eval_cfg["dataset_root"]),
                        sintel_list=str(sintel_eval_cfg["sintel_list"]),
                        patch_size=tuple(sintel_eval_cfg["patch_size"]),
                        ckpt_name="last",
                        max_samples=sintel_eval_cfg.get("max_samples", None),
                        progress_desc=f"Sintel {name} e{epoch_idx}",
                    )
                    sintel_epes[name] = float(sintel_result["sintel_epe"])
                for name, mg in model_graphs.items():
                    if sintel_epes[name] < best_sintel_epe[name]:
                        best_sintel_epe[name] = float(sintel_epes[name])
                        _save_checkpoint_with_variant(sess, mg["saver"], model_ckpt_paths[name]["root"] / f"{sintel_ckpt_name}.ckpt", epoch_idx, global_step, sintel_epes[name], best_sintel_epe[name], mg["variant_config"])
                        logger.info("sintel_best updated model=%s sintel_epe=%.4f", name, sintel_epes[name])

            if do_eval:
                mean_epe = float(np.mean(list(epoch_epes.values())))
                no_improve_evals = 0 if mean_epe + early_stop_min_delta < best_mean_epe else no_improve_evals + 1
                best_mean_epe = min(best_mean_epe, mean_epe)
                for name in model_names:
                    row = {"epoch": epoch_idx, "lr": lr_last, "loss": avg_losses[name], "epe": epoch_epes[name], "best_epe": best_epe[name], **{f"grad_norm_{k}": v for k, v in grad_stats[name].items()}}
                    if name in sintel_epes:
                        row["sintel_epe"] = sintel_epes[name]
                        row["best_sintel_epe"] = best_sintel_epe[name]
                    eval_histories[name].append(row)
                    _write_eval_history(model_dirs[name] / "eval_history.csv", eval_histories[name])
                comp_row = {"epoch": epoch_idx, "lr": lr_last, "mean_epe": mean_epe}
                for name in model_names:
                    comp_row[f"loss_{name}"] = avg_losses[name]
                    comp_row[f"epe_{name}"] = epoch_epes[name]
                    comp_row[f"grad_clip_rate_{name}"] = grad_stats[name]["clip_rate"]
                if sintel_epes:
                    comp_row = _append_sintel_metrics(comp_row, sintel_epes=sintel_epes, model_names=model_names)
                comparison_rows.append(comp_row)
                _write_eval_history(experiment_dir / "comparison.csv", comparison_rows)

            logger.info("epoch=%d/%d global_step=%d lr_start=%.2e lr_end=%.2e time_sec=%.1f eval=%s", epoch_idx, num_epochs, global_step, float(lr_first or 0.0), float(lr_last or 0.0), time.time() - epoch_start, bool(do_eval))
            if hasattr(train_provider, "skipped_nonfinite_count") or hasattr(val_provider, "skipped_nonfinite_count"):
                logger.info(
                    "ft3d_skipped_nonfinite train=%s val=%s",
                    getattr(train_provider, "skipped_nonfinite_count", ""),
                    getattr(val_provider, "skipped_nonfinite_count", ""),
                )
            for name in model_names:
                gs = grad_stats[name]
                logger.info("model=%s loss=%.6f grad_mean=%.4f grad_p50=%.4f grad_p90=%.4f grad_p99=%.4f clip_rate=%.4f epe=%s sintel=%s", name, avg_losses[name], gs["mean"], gs["p50"], gs["p90"], gs["p99"], gs["clip_rate"], epoch_epes.get(name, ""), sintel_epes.get(name, ""))

            _save_json_atomic(
                trainer_state_path,
                {"epoch": epoch_idx, "global_step": global_step, "best_epe": best_epe, "best_sintel_epe": best_sintel_epe, "best_mean_epe": best_mean_epe, "no_improve_evals": no_improve_evals, "dataset": data_cfg.get("dataset", ""), "model_names": model_names},
            )
            if do_eval and early_stop_patience > 0 and no_improve_evals >= early_stop_patience:
                logger.info("early stop triggered epoch=%d no_improve_evals=%d", epoch_idx, no_improve_evals)
                break

    logger.info("ablation_v1 finished")
    return 0
