"""评估 Worker：subprocess 调用 Supernet + Vela 编译器，解析输出产物。

每个 Worker 在独立线程中执行以下流程：
1. 通过 subprocess 调用现有的 run_supernet_subnet_distribution.py CLI
2. 解析产出的 EPE、Vela summary CSV (FPS / SRAM / Cycles)
3. 读取 per-layer.csv 原始报告（供 Agent C 蒸馏）
4. 将结果写入 tmp_workers/ 下的独立 JSON 文件

关键设计：Worker 绝不碰全局 CSV，只写私有 JSON 快照。
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_single_arch(
    arch_code_str: str,
    epoch: int,
    exp_dir: str,
    project_root: str,
    cfg: Dict[str, Any],
    llm_client: Any = None,
) -> Optional[Dict[str, Any]]:
    """评估单个子网架构并返回结果字典。

    此函数被 ThreadPoolExecutor 的工作线程调用。

    Args:
        arch_code_str: 逗号分隔的架构编码 (如 "0,1,2,0,0,1,2,1,0")。
        epoch: 当前搜索轮次。
        exp_dir: 实验根目录。
        project_root: 项目根目录 (EdgeFlowNAS/)。
        cfg: 全局配置字典。
        llm_client: LLMClient 实例（用于调用 Agent C 蒸馏，可选）。

    Returns:
        包含所有评估指标的字典，或 None（如果评估失败）。
    """
    eval_cfg = cfg["evaluation"]
    safe_name = arch_code_str.replace(",", "")
    output_tag = f"agent_eval_{safe_name}"

    # 为此次评估创建隔离的输出目录
    run_output_dir = os.path.join(exp_dir, "dashboard", "eval_outputs", f"run_{safe_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    logger.info("[Worker] 开始评估架构: %s", arch_code_str)

    # ---------------------------------------------------------------
    # Step 1: 通过 subprocess 调用底层 CLI
    # ---------------------------------------------------------------
    eval_script = os.path.join(project_root, eval_cfg["eval_script"])
    supernet_config = os.path.join(project_root, eval_cfg["supernet_config"])

    cmd = [
        sys.executable, eval_script,
        "--config", supernet_config,
        "--checkpoint_type", eval_cfg["checkpoint_type"],
        "--fixed_arch", arch_code_str,
        "--output_tag", output_tag,
        "--output_dir", run_output_dir,
    ]
    if eval_cfg.get("enable_vela", False):
        cmd.append("--enable_vela")
        cmd.extend(["--vela_mode", eval_cfg.get("vela_mode", "verbose")])
    if eval_cfg.get("vela_keep_artifacts", False):
        cmd.append("--vela_keep_artifacts")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 单个子网评估超时 10 分钟
            cwd=project_root,
        )
        if result.returncode != 0:
            logger.error(
                "[Worker] 评估失败 arch=%s, returncode=%d\nstderr:\n%s",
                arch_code_str, result.returncode,
                result.stderr[-2000:] if result.stderr else "(empty)",
            )
            return None
    except subprocess.TimeoutExpired:
        logger.error("[Worker] 评估超时 arch=%s (>600s)", arch_code_str)
        return None
    except Exception:
        logger.exception("[Worker] subprocess 异常 arch=%s", arch_code_str)
        return None

    # ---------------------------------------------------------------
    # Step 2: 解析 EPE 得分
    # ---------------------------------------------------------------
    epe = _parse_epe(run_output_dir, output_tag)

    # ---------------------------------------------------------------
    # Step 3: 解析 Vela summary CSV (FPS / SRAM / Cycles / MACs)
    # ---------------------------------------------------------------
    vela_metrics = _parse_vela_summary(run_output_dir)

    # ---------------------------------------------------------------
    # Step 4: 读取 per-layer.csv 供 Agent C 蒸馏
    # ---------------------------------------------------------------
    per_layer_text = _read_per_layer_report(
        run_output_dir,
        max_rows=cfg["evaluation"].get("per_layer_csv_max_rows", 200),
    )

    # ---------------------------------------------------------------
    # Step 5: 调用 Agent C 蒸馏（如果 llm_client 可用）
    # ---------------------------------------------------------------
    micro_insight = ""
    if llm_client is not None and per_layer_text:
        micro_insight = _invoke_agent_c(
            llm_client, arch_code_str, per_layer_text, vela_metrics,
        )

    # ---------------------------------------------------------------
    # Step 6: 组装结果字典并写入 tmp_workers/
    # ---------------------------------------------------------------
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

    # 写入 tmp_workers/ 私有 JSON
    from efnas.search.file_io import write_worker_result
    json_path = write_worker_result(exp_dir, arch_code_str, row)
    logger.info("[Worker] 评估完成 arch=%s -> %s", arch_code_str, json_path)

    return row


# ===================================================================
# 内部解析函数
# ===================================================================

def _parse_epe(run_output_dir: str, output_tag: str) -> Optional[float]:
    """从评估产物中解析 EPE 得分。

    尝试多种可能的文件名模式来定位 EPE 数据。
    """
    # 模式 1: 直接查找 summary CSV (常见形式)
    import glob
    patterns = [
        os.path.join(run_output_dir, f"*{output_tag}*summary*.csv"),
        os.path.join(run_output_dir, "*summary*.csv"),
        os.path.join(run_output_dir, "*.csv"),
    ]

    for pattern in patterns:
        files = glob.glob(pattern)
        for f in files:
            if "per-layer" in f or "per_layer" in f:
                continue  # 跳过 per-layer 报告
            try:
                df = pd.read_csv(f)
                # 查找包含 epe 的列
                epe_cols = [c for c in df.columns if "epe" in c.lower()]
                if epe_cols:
                    val = df[epe_cols[0]].iloc[0]
                    return float(val) if pd.notna(val) else None
            except Exception:
                continue

    # 模式 2: 从 stdout 解析 (如果 CLI 打印了 EPE)
    stdout_path = os.path.join(run_output_dir, "stdout.log")
    if os.path.exists(stdout_path):
        try:
            with open(stdout_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "epe" in line.lower():
                        parts = line.strip().split()
                        for p in parts:
                            try:
                                return float(p)
                            except ValueError:
                                continue
        except Exception:
            pass

    logger.warning("未能解析 EPE: %s", run_output_dir)
    return None


def _parse_vela_summary(run_output_dir: str) -> Dict[str, Any]:
    """从 Vela summary CSV 中提取关键硬件指标。"""
    import glob
    metrics: Dict[str, Any] = {}

    # 查找 Vela summary 文件
    patterns = [
        os.path.join(run_output_dir, "*summary*Grove*Sys*Config*.csv"),
        os.path.join(run_output_dir, "*summary*.csv"),
    ]

    for pattern in patterns:
        files = glob.glob(pattern)
        for f in files:
            if "per-layer" in f or "per_layer" in f:
                continue
            try:
                df = pd.read_csv(f)
                cols_lower = {c.lower(): c for c in df.columns}

                # FPS
                for key in ["inferences_per_second", "fps", "inferences/s"]:
                    if key in cols_lower:
                        val = df[cols_lower[key]].iloc[0]
                        if pd.notna(val):
                            metrics["fps"] = float(val)
                        break

                # SRAM
                for key in ["sram_memory_used", "sram_total_bytes", "sram_used_bytes"]:
                    if key in cols_lower:
                        val = df[cols_lower[key]].iloc[0]
                        if pd.notna(val):
                            # 转换为 KB
                            raw = float(val)
                            metrics["sram_kb"] = raw / 1024.0 if raw > 10000 else raw
                        break

                # NPU Cycles
                for key in ["cycles_npu", "npu_cycles", "total_cycles"]:
                    if key in cols_lower:
                        val = df[cols_lower[key]].iloc[0]
                        if pd.notna(val):
                            metrics["cycles_npu"] = int(float(val))
                        break

                # MACs
                for key in ["macs", "total_macs", "mac_count"]:
                    if key in cols_lower:
                        val = df[cols_lower[key]].iloc[0]
                        if pd.notna(val):
                            metrics["macs"] = int(float(val))
                        break

                if metrics:
                    return metrics
            except Exception:
                continue

    if not metrics:
        logger.warning("未能解析 Vela summary: %s", run_output_dir)
    return metrics


def _read_per_layer_report(run_output_dir: str, max_rows: int = 200) -> str:
    """读取 per-layer.csv 的文本内容，截断到最大行数。"""
    import glob
    patterns = [
        os.path.join(run_output_dir, "*per-layer*.csv"),
        os.path.join(run_output_dir, "*per_layer*.csv"),
    ]

    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            try:
                with open(files[0], "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # 截断
                if len(lines) > max_rows + 1:  # +1 for header
                    lines = lines[:max_rows + 1]
                    lines.append(f"\n... (截断: 共 {len(lines)} 行, 仅展示前 {max_rows} 行)\n")
                return "".join(lines)
            except Exception:
                continue

    return ""


def _invoke_agent_c(
    llm_client: Any,
    arch_code_str: str,
    per_layer_text: str,
    vela_metrics: Dict[str, Any],
) -> str:
    """调用 Agent C (HW Distiller) 对单个子网的 Vela 报告进行蒸馏。"""
    from efnas.search.prompts import AGENT_C_SYSTEM

    # 组装 user message：包含 per-layer.csv 内容和 summary 数据
    summary_info = json.dumps(vela_metrics, ensure_ascii=False, indent=2) if vela_metrics else "N/A"
    user_msg = (
        f"## 当前子网架构编码: {arch_code_str}\n\n"
        f"## Vela Summary 关键指标:\n```json\n{summary_info}\n```\n\n"
        f"## Vela Per-Layer 详细报告:\n```csv\n{per_layer_text}\n```\n"
    )

    try:
        insight = llm_client.chat(
            role="agent_c",
            system_prompt=AGENT_C_SYSTEM,
            user_message=user_msg,
            force_json=False,  # Agent C 输出纯文本
        )
        return insight.strip()
    except Exception:
        logger.exception("[Agent C] 蒸馏调用失败 arch=%s", arch_code_str)
        return "Agent C 调用失败, 无法生成硬件洞察"
