"""单模型/多模型独立重训引擎。

支持同时训练多个固定架构模型（共享数据），用于 NAS 搜索后的 retrain 阶段。
"""

import csv
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from code.data.dataloader_builder import build_fc2_provider
from code.data.transforms_180x240 import standardize_image_tensor
from code.engine.eval_step import accumulate_predictions, build_epe_metric
from code.engine.train_step import add_weight_decay, build_multiscale_uncertainty_loss
from EdgeFlowNAS.code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet
from code.optim.lr_scheduler import cosine_lr
from code.utils.json_io import read_json, write_json
from code.utils.logger import build_logger
from code.utils.path_utils import ensure_directory, project_root
from code.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _git_commit_hash() -> str:
    """获取当前 git commit hash。"""
    try:
        raw = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root()))
    except Exception:
        return "unknown"
    return raw.decode("utf-8").strip()


def _parse_arch_codes(raw: str, num_blocks: int = 9) -> List[List[int]]:
    """解析架构码字符串，支持 '+' 分隔多个架构码。

    示例: "0,2,1,1,0,0,1,0,1" => [[0,2,1,1,0,0,1,0,1]]
          "0,2,1,1,0,0,1,0,1+0,0,0,0,2,1,2,2,2" => [[0,2,1,1,0,0,1,0,1], [0,0,0,0,2,1,2,2,2]]
    """
    arch_groups = raw.strip().split("+")
    result: List[List[int]] = []
    for group in arch_groups:
        tokens = [t for t in re.split(r"[,\s]+", group.strip()) if t]
        if len(tokens) != num_blocks:
            raise ValueError(f"每个架构码必须有 {num_blocks} 位，实际得到 {len(tokens)}: {group}")
        code = []
        for t in tokens:
            v = int(t)
            if v not in (0, 1, 2):
                raise ValueError(f"架构码每位只能取 0/1/2，实际得到 {v}")
            code.append(v)
        result.append(code)
    return result


def _parse_arch_names(raw: Optional[str], num_models: int) -> List[str]:
    """解析架构名，如果未提供则自动编号。"""
    if not raw or not raw.strip():
        if num_models == 1:
            return ["model"]
        return [f"model_{i}" for i in range(num_models)]
    names = [n.strip() for n in raw.split("+")]
    if len(names) != num_models:
        raise ValueError(f"arch_names 数量 ({len(names)}) 必须与 arch_codes 数量 ({num_models}) 一致")
    return names


def _resolve_output_dir(config: Dict[str, Any]) -> Path:
    """构建实验输出目录。"""
    runtime_cfg = config.get("runtime", {})
    output_root = runtime_cfg.get("output_root", "outputs/standalone")
    experiment_name = runtime_cfg.get("experiment_name", "default_experiment")
    return ensure_directory(str(project_root() / output_root / experiment_name))


def _resolve_resume_dir(config: Dict[str, Any], experiment_dir: Path) -> Path:
    """解析恢复训练源目录。"""
    checkpoint_cfg = config.get("checkpoint", {})
    resume_name = str(checkpoint_cfg.get("resume_experiment_name", "")).strip()
    if not resume_name:
        return experiment_dir
    runtime_cfg = config.get("runtime", {})
    output_root = runtime_cfg.get("output_root", "outputs/standalone")
    return project_root() / output_root / resume_name


def _apply_gpu_device_setting(train_cfg: Dict[str, Any], logger) -> None:
    """设置 GPU 可见性。"""
    raw_value = train_cfg.get("gpu_device", None)
    if raw_value is None:
        logger.info("gpu_device=auto")
        return
    gpu_device = int(raw_value)
    if gpu_device < 0:
        try:
            tf.config.set_visible_devices([], "GPU")
            logger.info("gpu_device=%d -> force CPU mode", gpu_device)
            return
        except Exception as exc:
            raise RuntimeError(f"无法强制 CPU 模式: {exc}") from exc
    all_gpus = tf.config.list_physical_devices("GPU")
    if not all_gpus:
        logger.warning("gpu_device=%d 但未找到 GPU; 回退到 CPU/默认设备", gpu_device)
        return
    if gpu_device >= len(all_gpus):
        raise RuntimeError(f"gpu_device={gpu_device} 超出范围; 可见 GPU 数量={len(all_gpus)}")
    try:
        tf.config.set_visible_devices(all_gpus[gpu_device], "GPU")
        try:
            tf.config.experimental.set_memory_growth(all_gpus[gpu_device], True)
        except Exception:
            logger.warning("set_memory_growth 失败; 继续运行")
        logger.info("gpu_device=%d applied (visible GPUs=%d)", gpu_device, len(all_gpus))
    except Exception as exc:
        raise RuntimeError(f"gpu_device={gpu_device} 设置失败: {exc}") from exc


def _cosine_lr_with_min(base_lr: float, lr_min: float, step_idx: int, total_steps: int) -> float:
    """带有最低学习率的余弦退火。"""
    if total_steps <= 0:
        return float(base_lr)
    ratio = min(max(step_idx / float(total_steps), 0.0), 1.0)
    return float(lr_min) + (float(base_lr) - float(lr_min)) * 0.5 * (1.0 + math.cos(math.pi * ratio))


# ---------------------------------------------------------------------------
# 单模型图构建
# ---------------------------------------------------------------------------


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
) -> Dict[str, Any]:
    """在指定 variable_scope 下为一个固定架构构建完整的训练图。"""
    with tf.compat.v1.variable_scope(scope_name):
        # 使用固定 arch_code 的 placeholder（常量化）
        arch_code_ph = tf.constant(arch_code, dtype=tf.int32, name="FixedArchCode")

        model = MultiScaleResNetSupernet(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=pred_channels,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()

        # Loss
        loss_terms = build_multiscale_uncertainty_loss(
            preds=preds,
            label_ph=label_ph,
            num_out=flow_channels,
            return_terms=True,
        )
        loss_core = loss_terms["total"]

        # 可训练变量（仅本 scope）
        trainable_vars = tf.compat.v1.trainable_variables(scope=scope_name)
        if not trainable_vars:
            raise RuntimeError(f"scope '{scope_name}' 内无可训练变量")

        # Weight Decay
        loss_tensor = add_weight_decay(
            loss_tensor=loss_core,
            weight_decay=weight_decay,
            trainable_vars=trainable_vars,
        )

        # Optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr_ph, beta1=0.9, beta2=0.999, epsilon=1e-8,
        )
        grads_and_vars = optimizer.compute_gradients(loss_tensor, var_list=trainable_vars)

        # BN 更新操作（仅本 scope）
        all_bn_updates = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        bn_updates = [op for op in all_bn_updates if op.name.startswith(f"{scope_name}/")]

        # 梯度裁剪（可选）
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

        # EPE 指标（用于评估）
        pred_accum = accumulate_predictions(preds)
        epe_tensor = build_epe_metric(pred_tensor=pred_accum, label_ph=label_ph, num_out=flow_channels)

        # Saver（仅本 scope 的全局变量）
        scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{scope_name}/")]
        saver = tf.compat.v1.train.Saver(var_list=scope_global_vars, max_to_keep=3)

    return {
        "scope_name": scope_name,
        "arch_code": arch_code,
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


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def _evaluate_model(
    sess,
    model_graph: Dict[str, Any],
    input_ph: tf.Tensor,
    label_ph: tf.Tensor,
    is_training_ph: tf.Tensor,
    val_provider,
    batch_size: int,
    eval_batches: int,
) -> float:
    """评估单个模型在 val set 上的 EPE。使用 Train-Mode BN（与超网评估一致）。"""
    if hasattr(val_provider, "reset_cursor"):
        val_provider.reset_cursor(0)

    total_samples = len(val_provider)
    if eval_batches <= 0:
        # 全量评估
        num_batches = max(1, int(math.ceil(total_samples / batch_size)))
    else:
        num_batches = eval_batches

    batch_epes: List[float] = []
    for _ in range(num_batches):
        input_batch, _, _, label_batch = val_provider.next_batch(batch_size=batch_size)
        input_batch = standardize_image_tensor(input_batch)
        epe_val = sess.run(
            model_graph["epe"],
            feed_dict={
                input_ph: input_batch,
                label_ph: label_batch,
                # Train-Mode BN: 使用当前 batch 统计量，避免全局 moving stats 可能的问题
                is_training_ph: True,
            },
        )
        batch_epes.append(float(epe_val))

    return float(np.mean(batch_epes)) if batch_epes else 0.0


# ---------------------------------------------------------------------------
# Checkpoint 管理（简化版，不依赖 fairness_counts）
# ---------------------------------------------------------------------------


def _build_standalone_checkpoint_paths(model_dir: Path) -> Dict[str, Path]:
    """为单个模型构建 checkpoint 路径。"""
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": ckpt_dir,
        "best": ckpt_dir / "best.ckpt",
        "last": ckpt_dir / "last.ckpt",
    }


def _save_standalone_checkpoint(
    sess,
    saver: tf.compat.v1.train.Saver,
    path_prefix: Path,
    epoch: int,
    global_step: int,
    metric: float,
    best_metric: float,
    arch_code: List[int],
) -> Path:
    """保存 checkpoint 及其 meta 信息。"""
    save_path = saver.save(sess, str(path_prefix))
    meta = {
        "checkpoint_path": str(save_path),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metric": float(metric),
        "best_metric": float(best_metric),
        "arch_code": [int(v) for v in arch_code],
    }
    write_json(path_like=str(path_prefix) + ".meta.json", payload=meta)
    return Path(save_path)


# ---------------------------------------------------------------------------
# 评估历史 CSV
# ---------------------------------------------------------------------------


def _write_eval_history(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """写入评估历史 CSV。"""
    if not rows:
        return
    headers = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------


def train_standalone(config: Dict[str, Any]) -> int:
    """运行单模型/多模型独立重训。"""
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    data_cfg = config.get("data", {})

    seed = int(runtime_cfg.get("seed", 42))
    set_global_seed(seed)

    experiment_dir = _resolve_output_dir(config)
    logger = build_logger("edgeflownas_standalone", str(experiment_dir / "train.log"))
    logger.info("开始单模型独立重训")
    _apply_gpu_device_setting(train_cfg=train_cfg, logger=logger)

    # --- 架构码解析 ---
    arch_codes_raw = str(config.get("arch_codes", "")).strip()
    if not arch_codes_raw:
        raise RuntimeError("必须通过 --arch_codes 指定至少一个架构码")
    arch_codes = _parse_arch_codes(arch_codes_raw)
    arch_names = _parse_arch_names(config.get("arch_names", None), len(arch_codes))
    num_models = len(arch_codes)
    for name, code in zip(arch_names, arch_codes):
        logger.info("model=%s arch_code=%s", name, ",".join(str(v) for v in code))

    # --- 训练参数 ---
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
    pred_channels = flow_channels * 2  # flow + uncertainty

    logger.info("batch_size=%d num_epochs=%d lr=%.2e lr_min=%.2e", batch_size, num_epochs, base_lr, lr_min)
    logger.info("weight_decay=%.2e grad_clip=%.1f eval_every=%d", weight_decay, grad_clip, eval_every_epoch)
    logger.info("num_models=%d", num_models)

    # --- 数据加载 ---
    train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    val_provider = build_fc2_provider(config=config, split="val", seed_offset=999, provider_mode="eval")
    logger.info("train_source=%s train_samples=%d", train_provider.source_dir, len(train_provider))
    logger.info("val_source=%s val_samples=%d", val_provider.source_dir, len(val_provider))

    if len(train_provider) == 0:
        raise RuntimeError("训练集样本数为 0; 中止训练")
    if len(val_provider) == 0:
        raise RuntimeError("验证集样本数为 0; 中止训练")

    steps_per_epoch = int(math.ceil(float(len(train_provider)) / float(max(1, batch_size))))
    total_steps = max(1, num_epochs * steps_per_epoch)
    logger.info("steps_per_epoch=%d total_steps=%d", steps_per_epoch, total_steps)

    # --- 构建 TF 图 ---
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, 6], name="Input")
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_h, input_w, flow_channels], name="Label")
    lr_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name="LearningRate")
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")

    model_graphs: Dict[str, Dict[str, Any]] = {}
    for name, code in zip(arch_names, arch_codes):
        logger.info("构建模型图: scope=%s arch=%s", name, ",".join(str(v) for v in code))
        mg = _build_single_model_graph(
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
        )
        model_graphs[name] = mg
        num_params = sum(int(np.prod(v.shape.as_list())) for v in mg["trainable_vars"])
        logger.info("model=%s trainable_params=%d", name, num_params)

    # --- 输出目录 ---
    model_dirs: Dict[str, Path] = {}
    model_ckpt_paths: Dict[str, Dict[str, Path]] = {}
    for name in arch_names:
        mdir = experiment_dir / f"model_{name}"
        mdir.mkdir(parents=True, exist_ok=True)
        model_dirs[name] = mdir
        model_ckpt_paths[name] = _build_standalone_checkpoint_paths(mdir)

    # --- run_manifest ---
    run_manifest = {
        "arch_codes": {name: [int(v) for v in code] for name, code in zip(arch_names, arch_codes)},
        "num_models": num_models,
        "config": config,
        "git_commit": _git_commit_hash(),
    }
    write_json(str(experiment_dir / "run_manifest.json"), run_manifest)

    # --- 训练循环 ---
    best_epe: Dict[str, float] = {name: float("inf") for name in arch_names}
    eval_histories: Dict[str, List[Dict[str, Any]]] = {name: [] for name in arch_names}
    comparison_rows: List[Dict[str, Any]] = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # --- 恢复 checkpoint（如果指定） ---
        checkpoint_cfg = config.get("checkpoint", {})
        global_step = 0
        start_epoch = 1
        if bool(checkpoint_cfg.get("load_checkpoint", False)):
            resume_dir = _resolve_resume_dir(config=config, experiment_dir=experiment_dir)
            for name, mg in model_graphs.items():
                resume_ckpt = model_ckpt_paths[name]["last"]
                # 尝试从 resume_dir 的子目录加载
                resume_model_dir = resume_dir / f"model_{name}" / "checkpoints" / "last.ckpt"
                if Path(str(resume_model_dir) + ".index").exists():
                    ckpt_to_load = resume_model_dir
                elif Path(str(resume_ckpt) + ".index").exists():
                    ckpt_to_load = resume_ckpt
                else:
                    logger.warning("model=%s 未找到 checkpoint，从头训练", name)
                    continue
                mg["saver"].restore(sess, str(ckpt_to_load))
                # 读取 meta 信息
                meta_path = Path(str(ckpt_to_load) + ".meta.json")
                if meta_path.exists():
                    meta = read_json(str(meta_path))
                    if isinstance(meta, dict):
                        start_epoch = max(start_epoch, int(meta.get("epoch", 0)) + 1)
                        global_step = max(global_step, int(meta.get("global_step", 0)))
                        best_epe[name] = float(meta.get("best_metric", float("inf")))
                logger.info("model=%s 恢复 checkpoint=%s start_epoch=%d", name, str(ckpt_to_load), start_epoch)

        logger.info("训练开始: start_epoch=%d global_step=%d", start_epoch, global_step)

        for epoch_idx in range(start_epoch, num_epochs + 1):
            if hasattr(train_provider, "start_epoch"):
                train_provider.start_epoch(shuffle=True)

            # --- 构建 sess.run 的 fetch 字典 ---
            train_fetch = {}
            for name, mg in model_graphs.items():
                train_fetch[f"loss_{name}"] = mg["loss"]
                train_fetch[f"train_{name}"] = mg["train_op"]

            epoch_losses: Dict[str, float] = {name: 0.0 for name in arch_names}
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
                    base_lr=base_lr, lr_min=lr_min,
                    step_idx=global_step, total_steps=total_steps,
                )

                feed = {
                    input_ph: input_batch,
                    label_ph: label_batch,
                    lr_ph: current_lr,
                    is_training_ph: True,
                }
                results = sess.run(train_fetch, feed_dict=feed)

                for name in arch_names:
                    epoch_losses[name] += float(results[f"loss_{name}"])
                step_count += 1
                global_step += 1

            # --- Epoch 平均 loss ---
            avg_losses: Dict[str, float] = {}
            for name in arch_names:
                avg_losses[name] = epoch_losses[name] / max(1, step_count)

            # --- 评估 ---
            do_eval = (epoch_idx % eval_every_epoch == 0) or (epoch_idx == num_epochs)
            epoch_epes: Dict[str, float] = {}

            if do_eval:
                for name, mg in model_graphs.items():
                    epe_val = _evaluate_model(
                        sess=sess,
                        model_graph=mg,
                        input_ph=input_ph,
                        label_ph=label_ph,
                        is_training_ph=is_training_ph,
                        val_provider=val_provider,
                        batch_size=batch_size,
                        eval_batches=eval_batches,
                    )
                    epoch_epes[name] = epe_val

            # --- 日志 ---
            lr_now = _cosine_lr_with_min(base_lr, lr_min, global_step, total_steps)
            log_parts = [f"epoch={epoch_idx}", f"lr={lr_now:.2e}"]
            for name in arch_names:
                log_parts.append(f"loss_{name}={avg_losses[name]:.6f}")
            if do_eval:
                for name in arch_names:
                    log_parts.append(f"epe_{name}={epoch_epes[name]:.4f}")
                if num_models == 2:
                    names_list = list(arch_names)
                    delta = epoch_epes[names_list[0]] - epoch_epes[names_list[1]]
                    marker = "✓" if delta < 0 else "✗"
                    log_parts.append(f"Δepe={delta:.4f} {marker}")
            else:
                log_parts.append("eval=skipped")
            logger.info(" ".join(log_parts))

            # --- 保存 checkpoint（每 epoch） ---
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
                # Best checkpoint 更新
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
                    logger.info("model=%s 更新 best checkpoint: epe=%.4f", name, metric_val)

            # --- 评估历史记录 ---
            if do_eval:
                for name in arch_names:
                    eval_histories[name].append({
                        "epoch": epoch_idx,
                        "lr": lr_now,
                        "loss": avg_losses[name],
                        "epe": epoch_epes[name],
                        "best_epe": best_epe[name],
                    })
                    _write_eval_history(
                        csv_path=model_dirs[name] / "eval_history.csv",
                        rows=eval_histories[name],
                    )

                # 对比表（多模型时有用）
                if num_models > 1:
                    comp_row: Dict[str, Any] = {"epoch": epoch_idx, "lr": lr_now}
                    for name in arch_names:
                        comp_row[f"loss_{name}"] = avg_losses[name]
                        comp_row[f"epe_{name}"] = epoch_epes[name]
                    if num_models == 2:
                        names_list = list(arch_names)
                        comp_row["delta_epe"] = epoch_epes[names_list[0]] - epoch_epes[names_list[1]]
                    comparison_rows.append(comp_row)
                    _write_eval_history(
                        csv_path=experiment_dir / "comparison.csv",
                        rows=comparison_rows,
                    )

    # --- 训练结束 ---
    logger.info("训练完成: total_epochs=%d", num_epochs)
    for name in arch_names:
        logger.info("model=%s best_epe=%.4f", name, best_epe[name])
    return 0
