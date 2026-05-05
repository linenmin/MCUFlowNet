"""Evaluation worker for one architecture.

The worker runs subnet evaluation via subprocess and parses artifacts into a
single row for `history_archive.csv`.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _analysis_dir(run_output_dir: str) -> str:
    """Return analysis folder used by subnet-distribution script."""
    return os.path.join(run_output_dir, "analysis")


def _append_opt(cmd: list, name: str, value: Any) -> None:
    """Append CLI option when value is explicitly provided."""
    if value is None:
        return
    cmd.extend([name, str(value)])


def _resolve_project_path(project_root: str, value: Any) -> str:
    """Resolve config path values relative to the project root."""
    path = str(value)
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def _build_eval_command(
    project_root: str,
    eval_cfg: Dict[str, Any],
    arch_code_str: str,
    output_tag: str,
    run_output_dir: str,
) -> list:
    """Build eval subprocess command from `evaluation` config."""
    eval_script = _resolve_project_path(project_root, eval_cfg["eval_script"])
    supernet_config = _resolve_project_path(project_root, eval_cfg["supernet_config"])

    cmd = [
        sys.executable,
        eval_script,
        "--config",
        supernet_config,
        "--checkpoint_type",
        str(eval_cfg.get("checkpoint_type", "best")),
        "--fixed_arch",
        arch_code_str,
        "--output_tag",
        output_tag,
        "--output_dir",
        run_output_dir,
    ]

    # Throughput and eval behavior overrides.
    _append_opt(cmd, "--bn_recal_batches", eval_cfg.get("bn_recal_batches"))
    _append_opt(cmd, "--eval_batches_per_arch", eval_cfg.get("eval_batches_per_arch"))
    _append_opt(cmd, "--max_fc2_val_samples", eval_cfg.get("max_fc2_val_samples"))
    _append_opt(cmd, "--batch_size", eval_cfg.get("batch_size"))
    _append_opt(cmd, "--num_workers", eval_cfg.get("num_workers"))
    _append_opt(cmd, "--prefetch_batches", eval_cfg.get("prefetch_batches"))
    experiment_dir = eval_cfg.get("supernet_experiment_dir") or eval_cfg.get("experiment_dir")
    if experiment_dir is not None and str(experiment_dir).strip() != "":
        _append_opt(cmd, "--experiment_dir", _resolve_project_path(project_root, experiment_dir))
    if bool(eval_cfg.get("cpu_only", False)):
        cmd.append("--cpu_only")

    # Vela related options.
    if bool(eval_cfg.get("enable_vela", False)):
        cmd.append("--enable_vela")
        _append_opt(cmd, "--vela_mode", eval_cfg.get("vela_mode", "verbose"))
    if bool(eval_cfg.get("vela_keep_artifacts", False)):
        cmd.append("--vela_keep_artifacts")
    _append_opt(cmd, "--vela_optimise", eval_cfg.get("vela_optimise"))
    _append_opt(cmd, "--vela_limit", eval_cfg.get("vela_limit"))
    _append_opt(cmd, "--vela_rep_dataset_samples", eval_cfg.get("vela_rep_dataset_samples"))
    if bool(eval_cfg.get("vela_float32", False)):
        cmd.append("--vela_float32")
    if bool(eval_cfg.get("vela_verbose_log", False)):
        cmd.append("--vela_verbose_log")

    return cmd


def _build_worker_env(assigned_gpu: Optional[str] = None) -> Dict[str, str]:
    """Build subprocess environment for one architecture evaluation."""
    env = os.environ.copy()
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if assigned_gpu is not None and str(assigned_gpu).strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu).strip()
    return env


def evaluate_single_arch(
    arch_code_str: str,
    epoch: int,
    exp_dir: str,
    project_root: str,
    cfg: Dict[str, Any],
    llm_client: Any = None,
    assigned_gpu: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate one architecture and return parsed metrics row."""
    eval_cfg = cfg["evaluation"]
    safe_name = arch_code_str.replace(",", "")
    output_tag = f"agent_eval_{safe_name}"

    run_output_dir = os.path.join(exp_dir, "dashboard", "eval_outputs", f"run_{safe_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    logger.info("[Worker] 开始评估架构: %s", arch_code_str)

    cmd = _build_eval_command(
        project_root=project_root,
        eval_cfg=eval_cfg,
        arch_code_str=arch_code_str,
        output_tag=output_tag,
        run_output_dir=run_output_dir,
    )

    env = _build_worker_env(assigned_gpu=assigned_gpu)
    if assigned_gpu is not None:
        logger.info("[Worker] arch=%s assigned_gpu=%s", arch_code_str, assigned_gpu)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=project_root,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("[Worker] 评估超时 arch=%s (>600s)", arch_code_str)
        return None
    except Exception:
        logger.exception("[Worker] subprocess 异常 arch=%s", arch_code_str)
        return None

    # Always persist raw child logs for troubleshooting.
    try:
        with open(os.path.join(run_output_dir, "stdout.log"), "w", encoding="utf-8") as f_out:
            f_out.write(result.stdout or "")
        with open(os.path.join(run_output_dir, "stderr.log"), "w", encoding="utf-8") as f_err:
            f_err.write(result.stderr or "")
    except Exception:
        logger.warning("[Worker] failed to write child process logs: %s", run_output_dir)

    if result.returncode != 0:
        logger.error(
            "[Worker] 评估失败 arch=%s, returncode=%d\nstderr:\n%s",
            arch_code_str,
            result.returncode,
            result.stderr[-2000:] if result.stderr else "(empty)",
        )
        return None

    epe = _parse_epe(run_output_dir=run_output_dir, output_tag=output_tag)
    vela_metrics = _parse_vela_summary(run_output_dir=run_output_dir)

    # Phase 1.2: 替代旧 Agent C per-arch LLM 调用 —— 用纯 Python parser 把 Vela
    # 报告里的逐层硬件信息持久化为 analysis/layer_profile.json。Phase 3
    # Scientist 通过 query_vela_for_arch() 按需读取，零 LLM 调用。
    try:
        from efnas.search.vela_parser import parse_and_persist_layer_profile
        parse_and_persist_layer_profile(run_output_dir)
    except Exception:
        logger.exception("[Worker] Vela 层级 profile 持久化失败 arch=%s", arch_code_str)

    # micro_insight 列保留为向后兼容字段，新评估始终为空字符串
    micro_insight = ""

    row = {
        "arch_code": arch_code_str,
        "epe": epe,
        "fps": vela_metrics.get("fps", ""),
        "sram_kb": vela_metrics.get("sram_kb", ""),
        "cycles_npu": vela_metrics.get("cycles_npu", ""),
        "macs": vela_metrics.get("macs", ""),
        "micro_insight": micro_insight,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
    }

    from efnas.search.file_io import write_worker_result

    json_path = write_worker_result(exp_dir, arch_code_str, row)
    logger.info("[Worker] 评估完成 arch=%s -> %s", arch_code_str, json_path)
    return row


def _parse_epe(run_output_dir: str, output_tag: str) -> Optional[float]:
    """Parse EPE from modern and legacy output formats."""
    import glob

    analysis_dir = _analysis_dir(run_output_dir)

    # Preferred modern artifact.
    records_csv = os.path.join(analysis_dir, "records.csv")
    if os.path.exists(records_csv):
        try:
            df = pd.read_csv(records_csv)
            if "epe" in df.columns and len(df) > 0:
                val = df["epe"].iloc[0]
                return float(val) if pd.notna(val) else None
        except Exception:
            pass

    # Legacy fallback scan.
    for root in [run_output_dir, analysis_dir]:
        patterns = [
            os.path.join(root, f"*{output_tag}*summary*.csv"),
            os.path.join(root, "*summary*.csv"),
            os.path.join(root, "*.csv"),
        ]
        for pattern_item in patterns:
            for file_path in glob.glob(pattern_item):
                if "per-layer" in file_path or "per_layer" in file_path:
                    continue
                try:
                    df = pd.read_csv(file_path)
                    epe_cols = [c for c in df.columns if "epe" in c.lower()]
                    if epe_cols:
                        val = df[epe_cols[0]].iloc[0]
                        return float(val) if pd.notna(val) else None
                except Exception:
                    continue

    # Last fallback: parse from stdout.
    stdout_path = os.path.join(run_output_dir, "stdout.log")
    if os.path.exists(stdout_path):
        try:
            with open(stdout_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "epe" in line.lower():
                        for token in line.strip().split():
                            try:
                                return float(token)
                            except ValueError:
                                continue
        except Exception:
            pass

    logger.warning("未能解析 EPE: %s (expected analysis/records.csv)", run_output_dir)
    return None


def _parse_vela_summary(run_output_dir: str) -> Dict[str, Any]:
    """Parse Vela metrics from modern and legacy output formats."""
    import glob

    metrics: Dict[str, Any] = {}
    required_keys = ("fps", "sram_kb", "cycles_npu", "macs")
    analysis_dir = _analysis_dir(run_output_dir)

    # Preferred modern artifact.
    vela_metrics_csv = os.path.join(analysis_dir, "vela_metrics.csv")
    if os.path.exists(vela_metrics_csv):
        try:
            df = pd.read_csv(vela_metrics_csv)
            if len(df) > 0:
                row = df.iloc[0]
                if "fps" in df.columns and pd.notna(row.get("fps")):
                    metrics["fps"] = float(row.get("fps"))
                if "sram_peak_mb" in df.columns and pd.notna(row.get("sram_peak_mb")):
                    metrics["sram_kb"] = float(row.get("sram_peak_mb")) * 1024.0
                if "cycles_npu" in df.columns and pd.notna(row.get("cycles_npu")):
                    metrics["cycles_npu"] = int(float(row.get("cycles_npu")))
                if "macs" in df.columns and pd.notna(row.get("macs")):
                    metrics["macs"] = int(float(row.get("macs")))
                if all(k in metrics for k in required_keys):
                    return metrics
        except Exception:
            pass

    # Legacy fallback scan.
    for root in [run_output_dir, analysis_dir]:
        patterns = [
            os.path.join(root, "*summary*Grove*Sys*Config*.csv"),
            os.path.join(root, "*summary*.csv"),
            os.path.join(root, "**", "*summary*Grove*Sys*Config*.csv"),
            os.path.join(root, "**", "*summary*.csv"),
        ]
        for pattern_item in patterns:
            for file_path in glob.glob(pattern_item, recursive=True):
                if "per-layer" in file_path or "per_layer" in file_path:
                    continue
                try:
                    df = pd.read_csv(file_path)
                    cols_lower = {c.lower(): c for c in df.columns}

                    if "fps" not in metrics:
                        for key in ["inferences_per_second", "fps", "inferences/s"]:
                            if key in cols_lower:
                                val = df[cols_lower[key]].iloc[0]
                                if pd.notna(val):
                                    metrics["fps"] = float(val)
                                break

                    if "sram_kb" not in metrics:
                        for key in ["sram_memory_used", "sram_total_bytes", "sram_used_bytes"]:
                            if key in cols_lower:
                                val = df[cols_lower[key]].iloc[0]
                                if pd.notna(val):
                                    raw = float(val)
                                    metrics["sram_kb"] = raw / 1024.0 if raw > 10000 else raw
                                break

                    if "cycles_npu" not in metrics:
                        for key in ["cycles_npu", "npu_cycles", "total_cycles"]:
                            if key in cols_lower:
                                val = df[cols_lower[key]].iloc[0]
                                if pd.notna(val):
                                    metrics["cycles_npu"] = int(float(val))
                                break

                    if "macs" not in metrics:
                        for key in ["macs", "nn_macs", "total_macs", "mac_count"]:
                            if key in cols_lower:
                                val = df[cols_lower[key]].iloc[0]
                                if pd.notna(val):
                                    metrics["macs"] = int(float(val))
                                break

                    if all(k in metrics for k in required_keys):
                        return metrics
                except Exception:
                    continue

    # If modern/legacy artifacts partially exist, return what we have without warning.
    if metrics:
        return metrics

    logger.warning("未能解析 Vela summary: %s (expected analysis/vela_metrics.csv)", run_output_dir)
    return metrics


# Phase 1.1 removed: `_read_per_layer_report` and `_invoke_agent_c`.
# Reason: 771 per-arch LLM calls produced near-zero-entropy "DB0 Util ~33%"
# duplications. Phase 1.2 replaces this path with a deterministic Python
# parser that emits structured `(layer_name, op_type, cycles, util_pct)`
# vectors into `analysis/layer_profile.json`, queried on-demand by Phase 3
# Scientist instead of being eagerly translated to Chinese narrative.
