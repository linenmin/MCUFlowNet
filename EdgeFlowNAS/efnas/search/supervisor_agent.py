"""Phase 4 (search_hybrid_v1): Supervisor Agent —— NSGA-II 5-lever 监督.

每 K 代触发一次 (默认 K=3, 与 Scientist 同频, Scientist 之后立即跑).

单次调用流程:
  1. 读 current_state (5 个 lever 的当前值) + 最近 metrics + 最新 insights +
     supervisor_log (自己的历史)
  2. 调 LLM 一次, 输出 actions (5 个 lever 任意组合)
  3. 调 NSGA2SearchRunner.apply_supervisor_actions 应用 actions; runner 只做
     数学合法性校验 (策略层完全交给 agent)
  4. 把这次调用的 (rationale, before, after, applied, rejected, expected_effect,
     review_after_gen) 追加到 supervisor_log.json

设计原则:
- 5-lever 动作空间, 无策略边界, 仅合法性校验
- agent 知道当前状态 + 历史调整 → 自我纠偏
- 失败兜底: LLM 调用失败 / 输出非法 → log warning, 不调任何 lever, 不阻断
  NSGA-II 主搜索
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from efnas.search import file_io, prompts
from efnas.search.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ===================================================================
# supervisor_log.json 读写
# ===================================================================

def supervisor_log_path(exp_dir: str) -> str:
    return os.path.join(exp_dir, "metadata", "supervisor_log.json")


def read_supervisor_log(exp_dir: str) -> List[Dict[str, Any]]:
    """读取 supervisor_log.json. 不存在或损坏返回空 list."""
    path = supervisor_log_path(exp_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("[Supervisor] supervisor_log.json 不是 list, 视为空")
        return []
    except (json.JSONDecodeError, OSError):
        logger.exception("[Supervisor] 读 supervisor_log.json 失败")
        return []


def append_supervisor_log(exp_dir: str, entry: Dict[str, Any]) -> str:
    """追加一条 supervisor 调用记录. 文件不存在时创建."""
    path = supervisor_log_path(exp_dir)
    log = read_supervisor_log(exp_dir)
    log.append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return path


# ===================================================================
# Supervisor LLM 调用
# ===================================================================

def invoke_supervisor_agent(
    llm: LLMClient,
    *,
    current_state: Dict[str, Any],
    recent_metrics_df: pd.DataFrame,
    current_pareto_summary: str,
    current_insights_md: str,
    supervisor_log: List[Dict[str, Any]],
    role: str = "supervisor_agent",
) -> Optional[Dict[str, Any]]:
    """单次 LLM 调用, 返回 actions / rationale / expected_effect / review_after_gen.

    Returns:
        - dict 含上述四字段; LLM 失败 / 输出非 dict / 缺关键字段 → None
    """
    metrics_csv = ""
    if recent_metrics_df is not None and not getattr(recent_metrics_df, "empty", True):
        metrics_csv = recent_metrics_df.tail(8).to_csv(index=False)

    log_summary = json.dumps(
        supervisor_log[-5:] if supervisor_log else [],
        ensure_ascii=False,
        indent=2,
    )

    user_msg = (
        f"# Current 5-Lever State\n"
        f"```json\n{json.dumps(current_state, ensure_ascii=False, indent=2)}\n```\n\n"
        f"# Recent Search Health Metrics (generation_metrics.csv tail-8)\n"
        f"```csv\n{metrics_csv or '(empty)'}\n```\n\n"
        f"# Current Pareto Front Summary\n"
        f"```\n{current_pareto_summary}\n```\n\n"
        f"# Latest insights.md (from Phase 3 Scientist)\n"
        f"```markdown\n{current_insights_md or '(empty - scientist has not run yet)'}\n```\n\n"
        f"# Your Supervisor Log (last 5 entries)\n"
        f"```json\n{log_summary}\n```\n\n"
        f"请根据上面所有信号, 判断要不要调整 NSGA-II 参数. 输出 JSON 含\n"
        f"`rationale`, `actions` (5 字段, 允许 null), `expected_effect`,\n"
        f"`review_after_gen` 四个键. 不调任何 lever 时, actions 全部设 null."
    )

    try:
        result = llm.chat_json(
            role=role,
            system_prompt=prompts.SUPERVISOR_AGENT_SYSTEM,
            user_message=user_msg,
        )
    except Exception:
        logger.exception("[Supervisor] LLM 调用失败")
        return None

    if not isinstance(result, dict):
        logger.warning("[Supervisor] 响应非 dict: %r", type(result))
        return None

    rationale = str(result.get("rationale", "")).strip()
    actions = result.get("actions", {})
    if not isinstance(actions, dict):
        logger.warning("[Supervisor] actions 字段非 dict")
        actions = {}
    expected_effect = str(result.get("expected_effect", "")).strip()
    try:
        review_after_gen = int(result.get("review_after_gen", 0))
    except (TypeError, ValueError):
        review_after_gen = 0

    return {
        "rationale": rationale,
        "actions": actions,
        "expected_effect": expected_effect,
        "review_after_gen": review_after_gen,
    }


# ===================================================================
# Pareto / metrics 摘要 (供 agent 输入)
# ===================================================================

def _summarize_current_pareto(history_df: pd.DataFrame) -> str:
    """简短文字摘要当前 Pareto 前沿端点 + 大小, 给 Supervisor 看."""
    if history_df is None or history_df.empty:
        return "(no evaluations yet)"
    try:
        df = history_df.copy()
        df["epe_num"] = pd.to_numeric(df["epe"], errors="coerce")
        df["fps_num"] = pd.to_numeric(df["fps"], errors="coerce")
        df = df.dropna(subset=["epe_num", "fps_num"])
        if df.empty:
            return "(no valid (epe, fps) data)"
        from efnas.search.search_metrics import _compute_pareto_front_2d
        front = _compute_pareto_front_2d(
            list(zip(df["epe_num"].tolist(), df["fps_num"].tolist())),
        )
        if not front:
            return "(empty Pareto front)"
        epes = [p[0] for p in front]
        fpss = [p[1] for p in front]
        return (
            f"Pareto front size: {len(front)}\n"
            f"  EPE range:  [{min(epes):.4f}, {max(epes):.4f}]\n"
            f"  FPS range:  [{min(fpss):.4f}, {max(fpss):.4f}]\n"
            f"  Best EPE point:  EPE={min(epes):.4f}, FPS={fpss[epes.index(min(epes))]:.4f}\n"
            f"  Best FPS point:  EPE={epes[fpss.index(max(fpss))]:.4f}, FPS={max(fpss):.4f}"
        )
    except Exception:
        logger.exception("[Supervisor] 计算 Pareto 摘要失败")
        return "(pareto summary unavailable)"


# ===================================================================
# Pipeline (orchestrator)
# ===================================================================

def supervisor_pipeline(
    llm: LLMClient,
    exp_dir: str,
    runner: Any,
    *,
    generation: int,
    role: str = "supervisor_agent",
) -> Dict[str, Any]:
    """完整 Supervisor 调用 (1 LLM call + apply_actions + log).

    Args:
        llm: LLMClient
        exp_dir: 实验输出根
        runner: NSGA2SearchRunner 实例 (供 current_state 读 / actions 应用)
        generation: 当前代号 (用于 log)
        role: LLM client role 名

    Returns:
        dict 含执行摘要:
            - success (bool): LLM 调用 + apply_actions 都成功
            - llm_responded (bool)
            - applied (Dict): 实际生效的 lever
            - rejected (Dict): 被拒 lever 及原因
            - rationale (str)
            - error (str)
    """
    summary: Dict[str, Any] = {
        "success": False,
        "llm_responded": False,
        "applied": {},
        "rejected": {},
        "rationale": "",
        "error": "",
    }

    try:
        current_state = runner.current_supervisor_state()
        history_df = file_io.read_history(exp_dir)
        metrics_df = file_io.read_epoch_metrics(exp_dir)
        pareto_summary = _summarize_current_pareto(history_df)
        try:
            from efnas.search.scientist_agent import read_insights_md
            insights_md = read_insights_md(exp_dir)
        except Exception:
            insights_md = ""
        log = read_supervisor_log(exp_dir)
    except Exception as exc:
        summary["error"] = f"failed gathering inputs: {exc}"
        logger.exception("[Supervisor] 准备输入时异常")
        return summary

    response = invoke_supervisor_agent(
        llm,
        current_state=current_state,
        recent_metrics_df=metrics_df,
        current_pareto_summary=pareto_summary,
        current_insights_md=insights_md,
        supervisor_log=log,
        role=role,
    )

    if response is None:
        summary["error"] = "LLM 调用失败或响应格式不合法"
        # 仍记录一条日志, 让 agent 下次能看到这次失败了
        append_supervisor_log(exp_dir, {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "before_state": current_state,
            "after_state": current_state,
            "applied": {},
            "rejected": {"_global": "LLM call failed or response malformed"},
            "rationale": "",
            "expected_effect": "",
            "review_after_gen": 0,
        })
        return summary

    summary["llm_responded"] = True
    summary["rationale"] = response["rationale"]

    # 应用 actions 到 runner
    try:
        apply_result = runner.apply_supervisor_actions(response["actions"])
    except Exception as exc:
        summary["error"] = f"apply_supervisor_actions raised: {exc}"
        logger.exception("[Supervisor] apply_actions 异常")
        return summary

    summary["applied"] = apply_result["applied"]
    summary["rejected"] = apply_result["rejected"]
    summary["success"] = True  # 即使 actions 全 null 或全被拒, LLM 调用成功就算 success

    # 写日志
    append_supervisor_log(exp_dir, {
        "generation": generation,
        "timestamp": datetime.now().isoformat(),
        "before_state": apply_result["before_state"],
        "after_state": apply_result["after_state"],
        "applied": apply_result["applied"],
        "rejected": apply_result["rejected"],
        "rationale": response["rationale"],
        "expected_effect": response["expected_effect"],
        "review_after_gen": response["review_after_gen"],
    })
    return summary
