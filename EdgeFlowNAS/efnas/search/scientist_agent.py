"""Phase 3 (search_hybrid_v1): Scientist Agent —— 三阶段大反思.

每 K 代触发一次 (默认 K=3). 单次 Scientist invocation 内部:

  Stage A (LLM call, 纯架构归纳)
    ↓
  Stage B-1 (LLM call, 规划 vela_queries + verifications)
    ↓
  resolve_vela_queries (Python, 按 pattern/code 过滤 + 加载 layer_profile.json)
    ↓
  execute_verifications (sandbox, 跑 agent 写的 Python 验证代码)
    ↓
  Stage B-2 (LLM call, 综合所有数据写最终 insights.md)
    ↓
  atomic-write to metadata/insights.md (失败时保持上一次成功状态)

设计哲学:
- Stage A / B 分离防止 hardware-verifiable 模式被偏向 (用户的关键 insight)
- Vela query 由 agent 在 Stage B-1 自己规划, 而不是 coordinator 预选
- 每阶段失败兜底独立, 不阻断 NSGA-II 主搜索
- 备份策略: 每次调用前 snapshot 成 insights.md.gen{N}.bak
"""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from efnas.search import file_io, insights, sandbox, vela_parser, prompts
from efnas.search.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ===================================================================
# 公共辅助: insights.md 读写 + 备份
# ===================================================================

def read_insights_md(exp_dir: str) -> str:
    """读取 ``metadata/insights.md`` 当前内容. 不存在或读取失败返回空串."""
    path = os.path.join(exp_dir, "metadata", "insights.md")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        logger.exception("读取 insights.md 失败: %s", path)
        return ""


def write_insights_md_atomic(exp_dir: str, content: str) -> str:
    """原子写 ``metadata/insights.md``. tempfile + os.replace 防写一半."""
    metadata_dir = os.path.join(exp_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    target = os.path.join(metadata_dir, "insights.md")
    fd, tmp = tempfile.mkstemp(
        dir=metadata_dir, prefix=".insights_", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, target)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return target


def backup_insights_md(exp_dir: str, generation: int) -> Optional[str]:
    """把当前 insights.md snapshot 成 ``insights.md.gen{N}.bak``.

    用于 Phase 5 ablation 时审计 insights 的演化轨迹.

    Returns:
        备份文件路径; 当前 insights.md 不存在时返回 None.
    """
    src = os.path.join(exp_dir, "metadata", "insights.md")
    if not os.path.exists(src):
        return None
    dst = os.path.join(
        exp_dir, "metadata", f"insights.md.gen{generation}.bak",
    )
    try:
        shutil.copy2(src, dst)
        return dst
    except OSError:
        logger.exception("备份 insights.md 失败")
        return None


# ===================================================================
# Stage A: 纯架构归纳
# ===================================================================

def invoke_scientist_stage_a(
    llm: LLMClient,
    *,
    history_df: pd.DataFrame,
    metrics_df: Optional[pd.DataFrame],
    prev_insights_md: str,
    generation: int,
    total_generations: int,
    next_id_hint: str,
    role: str = "scientist_stage_a",
) -> List[Dict[str, Any]]:
    """单次 LLM 调用产出 insight drafts (无 hardware / 无代码).

    失败兜底: LLM 调用异常 / 输出非 JSON / drafts 字段非 list / 全部 drafts
    格式不合法 → 返回空 list, 上层不修改 insights.md.
    """
    if history_df is None or history_df.empty:
        logger.info("[Stage A] history 为空, 跳过")
        return []

    history_csv_text = history_df.to_csv(index=False)
    metrics_text = ""
    if metrics_df is not None and not getattr(metrics_df, "empty", True):
        metrics_text = metrics_df.tail(8).to_csv(index=False)

    user_msg = (
        f"# Progress\n"
        f"当前生成: {generation + 1} / {total_generations} 代\n"
        f"剩余: {max(0, total_generations - generation - 1)} 代\n"
        f"建议新 insight ID: {next_id_hint}\n\n"
        f"# History (history_archive.csv full content)\n"
        f"```csv\n{history_csv_text}\n```\n\n"
        f"# Recent Search Health Metrics (epoch_metrics.csv tail-8)\n"
        f"```csv\n{metrics_text or '(empty)'}\n```\n\n"
        f"# Previous insights.md (your prior memory)\n"
        f"```markdown\n{prev_insights_md or '(empty - first invocation)'}\n```\n\n"
        f"请根据上面所有数据, 输出 JSON 含 `drafts` 字段 (list of insight 草稿).\n"
        f"绝对不要在 body 里引用硬件 cycles / util 数据, 不要写 Python 代码 ——\n"
        f"那些是 Stage B 的事."
    )

    try:
        result = llm.chat_json(
            role=role,
            system_prompt=prompts.SCIENTIST_STAGE_A_SYSTEM,
            user_message=user_msg,
        )
    except Exception:
        logger.exception("[Stage A] LLM 调用失败")
        return []

    if not isinstance(result, dict):
        logger.warning("[Stage A] 响应非 dict: %r", type(result))
        return []

    raw_drafts = result.get("drafts", [])
    if not isinstance(raw_drafts, list):
        logger.warning("[Stage A] drafts 字段非 list: %r", type(raw_drafts))
        return []

    drafts: List[Dict[str, Any]] = []
    for d in raw_drafts:
        if not isinstance(d, dict):
            continue
        iid = str(d.get("id", "")).strip()
        status = str(d.get("status", "")).strip()
        title = str(d.get("title", "")).strip()
        body = str(d.get("body", "")).strip()
        if not insights.validate_id(iid):
            logger.warning("[Stage A] 跳过非法 ID: %r", iid)
            continue
        if status not in insights.VALID_STATUSES:
            logger.warning("[Stage A] 跳过非法 status: %r (id=%s)", status, iid)
            continue
        if not title:
            logger.warning("[Stage A] 跳过空标题 (id=%s)", iid)
            continue
        drafts.append({
            "id": iid, "status": status, "title": title, "body": body,
        })

    logger.info("[Stage A] 产出 %d 个合法 drafts (LLM 返回 %d, 过滤 %d)",
                 len(drafts), len(raw_drafts), len(raw_drafts) - len(drafts))
    return drafts


# ===================================================================
# Stage B-1: 规划 vela_queries + verifications
# ===================================================================

def _summarize_history_meta(history_df: pd.DataFrame) -> str:
    """给 Stage B-1 的简短 history schema 描述 (不含 row 数据)."""
    if history_df is None or history_df.empty:
        return "history_archive.csv: empty"
    try:
        epe = pd.to_numeric(history_df["epe"], errors="coerce").dropna()
        fps = pd.to_numeric(history_df["fps"], errors="coerce").dropna()
        gen = pd.to_numeric(history_df.get("epoch", pd.Series(dtype=float)),
                            errors="coerce").dropna()
        gen_max = int(gen.max()) if len(gen) > 0 else 0
        return (
            f"history_archive.csv: {len(history_df)} rows; "
            f"columns: {list(history_df.columns)}\n"
            f"  epe range: [{epe.min():.4f}, {epe.max():.4f}], "
            f"median {epe.median():.4f}\n"
            f"  fps range: [{fps.min():.4f}, {fps.max():.4f}], "
            f"median {fps.median():.4f}\n"
            f"  generation max: {gen_max}"
        )
    except Exception:
        return f"history_archive.csv: {len(history_df)} rows; columns: {list(history_df.columns)}"


def invoke_scientist_stage_b1(
    llm: LLMClient,
    *,
    drafts: List[Dict[str, Any]],
    history_df: pd.DataFrame,
    role: str = "scientist_stage_b1",
) -> Optional[Dict[str, Any]]:
    """单次 LLM 调用规划 vela_queries + verifications + annotations_no_code.

    Returns:
        plan dict 含三个 key (其中任一允许为空 list); LLM 失败返回 None.
    """
    drafts_json = json.dumps({"drafts": drafts}, ensure_ascii=False, indent=2)
    history_meta = _summarize_history_meta(history_df)

    user_msg = (
        f"# Stage A drafts\n"
        f"```json\n{drafts_json}\n```\n\n"
        f"# History schema\n"
        f"```\n{history_meta}\n```\n\n"
        f"请输出 JSON 含三个字段: `vela_queries`, `verifications`, "
        f"`annotations_no_code`. 每个字段都是 list (允许空 list).\n"
        f"参考 system prompt 的 OUTPUT FORMAT 示例."
    )

    try:
        result = llm.chat_json(
            role=role,
            system_prompt=prompts.SCIENTIST_STAGE_B1_SYSTEM,
            user_message=user_msg,
        )
    except Exception:
        logger.exception("[Stage B-1] LLM 调用失败")
        return None

    if not isinstance(result, dict):
        logger.warning("[Stage B-1] 响应非 dict")
        return None

    plan = {
        "vela_queries": result.get("vela_queries", []) or [],
        "verifications": result.get("verifications", []) or [],
        "annotations_no_code": result.get("annotations_no_code", []) or [],
    }
    for key in plan:
        if not isinstance(plan[key], list):
            logger.warning("[Stage B-1] 字段 %s 非 list, 改为空", key)
            plan[key] = []
    logger.info(
        "[Stage B-1] 规划: %d vela_queries, %d verifications, %d no-code annotations",
        len(plan["vela_queries"]),
        len(plan["verifications"]),
        len(plan["annotations_no_code"]),
    )
    return plan


# ===================================================================
# Resolve vela_queries (deterministic Python, 不调 LLM)
# ===================================================================

def _matches_pattern(arch_code: str, pattern: Sequence[Optional[int]]) -> bool:
    """判断 arch_code 是否匹配 pattern (None 通配)."""
    parts = [p.strip() for p in str(arch_code).split(",")]
    if len(parts) != len(pattern):
        return False
    for i, p in enumerate(pattern):
        if p is None:
            continue
        if str(parts[i]) != str(p):
            return False
    return True


def _compute_pareto_arch_set(history_df: pd.DataFrame) -> set:
    """计算当前 Pareto 前沿成员的 arch_code 集合."""
    if history_df is None or history_df.empty:
        return set()
    try:
        from efnas.search.search_metrics import _compute_pareto_front_2d
        df = history_df.copy()
        df["epe_num"] = pd.to_numeric(df["epe"], errors="coerce")
        df["fps_num"] = pd.to_numeric(df["fps"], errors="coerce")
        df = df.dropna(subset=["epe_num", "fps_num"])
        points = list(zip(df["epe_num"].tolist(), df["fps_num"].tolist()))
        front = _compute_pareto_front_2d(points)
        front_set = set((round(e, 6), round(f, 6)) for e, f in front)
        out = set()
        for _, row in df.iterrows():
            key = (round(float(row["epe_num"]), 6), round(float(row["fps_num"]), 6))
            if key in front_set:
                out.add(str(row["arch_code"]))
        return out
    except Exception:
        logger.exception("[VelaQuery] 计算 Pareto 集合失败")
        return set()


def resolve_vela_queries(
    queries: List[Dict[str, Any]],
    history_df: pd.DataFrame,
    exp_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """按 vela_queries 过滤 history + 加载对应 arch 的 layer_profile.json.

    Returns:
        Dict mapping insight_id -> {"purpose", "matched_archs": [...],
        "match_count", "note"}.
        多条 query 同 insight_id 会被合并 (后写覆盖, 但 matched_archs 累积).
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not queries or history_df is None or history_df.empty:
        return results

    pareto_set: Optional[set] = None
    history_arch_set = set(history_df["arch_code"].astype(str)) \
        if "arch_code" in history_df.columns else set()

    for q in queries:
        if not isinstance(q, dict):
            continue
        insight_id = str(q.get("insight_id", "")).strip()
        if not insight_id:
            continue
        purpose = str(q.get("purpose", "")).strip()

        # 选 candidates
        candidates: List[str] = []
        note = ""
        if q.get("by_arch_codes"):
            raw = q["by_arch_codes"]
            if isinstance(raw, list):
                candidates = [
                    str(c).strip() for c in raw
                    if str(c).strip() in history_arch_set
                ]
                if not candidates:
                    note = "no archs match the explicit list (none evaluated yet)"
        elif q.get("by_arch_code_pattern"):
            pattern = q["by_arch_code_pattern"]
            if isinstance(pattern, list) and len(pattern) == 11:
                candidates = [
                    str(c) for c in history_df["arch_code"].astype(str).tolist()
                    if _matches_pattern(c, pattern)
                ]
                if not candidates:
                    note = "no archs match this pattern in history"
            else:
                note = "invalid pattern (must be list of length 11)"
        else:
            note = "no filter specified (need by_arch_codes or by_arch_code_pattern)"

        # Pareto 过滤
        if candidates and q.get("from_pareto_front_only"):
            if pareto_set is None:
                pareto_set = _compute_pareto_arch_set(history_df)
            candidates = [c for c in candidates if c in pareto_set]
            if not candidates:
                note = "no archs match after Pareto-front filter"

        # 排序 + limit
        if candidates:
            sub = history_df[history_df["arch_code"].astype(str).isin(candidates)].copy()
            sub["_epe_num"] = pd.to_numeric(sub["epe"], errors="coerce")
            sub["_fps_num"] = pd.to_numeric(sub["fps"], errors="coerce")
            if q.get("sort_by_epe"):
                sub = sub.sort_values(
                    "_epe_num", ascending=(q["sort_by_epe"] == "asc"),
                )
            elif q.get("sort_by_fps"):
                sub = sub.sort_values(
                    "_fps_num", ascending=(q["sort_by_fps"] == "asc"),
                )
            limit = int(q.get("limit", 5))
            limit = max(1, min(limit, 20))
            sub = sub.head(limit)

            matched: List[Dict[str, Any]] = []
            for _, row in sub.iterrows():
                arch_code = str(row["arch_code"])
                profile = vela_parser.query_vela_for_arch(exp_dir, arch_code)
                matched.append({
                    "arch_code": arch_code,
                    "epe": float(row["_epe_num"]) if pd.notna(row["_epe_num"]) else None,
                    "fps": float(row["_fps_num"]) if pd.notna(row["_fps_num"]) else None,
                    "layer_profile": profile,
                })
        else:
            matched = []

        # 同 insight_id 多 query: 累积 matched_archs, 第一条的 purpose
        if insight_id in results:
            results[insight_id]["matched_archs"].extend(matched)
            if note and not results[insight_id].get("note"):
                results[insight_id]["note"] = note
        else:
            entry: Dict[str, Any] = {
                "purpose": purpose,
                "matched_archs": matched,
                "match_count": len(matched),
            }
            if note:
                entry["note"] = note
            results[insight_id] = entry

    # 重新计算 match_count 反映合并后实际数量
    for entry in results.values():
        entry["match_count"] = len(entry["matched_archs"])

    return results


# ===================================================================
# Execute verifications (sandbox)
# ===================================================================

def execute_verifications(
    verifications: List[Dict[str, Any]],
    *,
    history_csv_path: str,
    query_results: Dict[str, Dict[str, Any]],
    timeout: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """对每条 verification, sandbox 跑 code, 收集结果.

    传给 sandbox 的 argv:
        argv[1] = history_csv_path
        argv[2] = JSON 字符串, 该 insight_id 下的 vela query 结果
                  (matched_archs + purpose + note); 若无则 ``"{}"``

    Returns:
        Dict mapping insight_id -> {"status", "purpose", "parsed_json",
        "stdout", "stderr", "error"} (字段同 sandbox.execute_verification)
    """
    results: Dict[str, Dict[str, Any]] = {}
    for v in verifications:
        if not isinstance(v, dict):
            continue
        insight_id = str(v.get("insight_id", "")).strip()
        if not insight_id:
            continue
        purpose = str(v.get("purpose", "")).strip()
        code = v.get("code", "")
        if not isinstance(code, str) or not code.strip():
            results[insight_id] = {
                "status": "validation_error",
                "purpose": purpose,
                "error": "empty code",
                "stdout": "",
                "stderr": "",
                "parsed_json": None,
            }
            continue

        query_arg = json.dumps(
            query_results.get(insight_id, {}), ensure_ascii=False,
        )
        try:
            out = sandbox.execute_verification(
                code,
                args=[history_csv_path, query_arg],
                timeout=timeout,
            )
        except Exception as exc:
            logger.exception("[Verification] sandbox 异常 (insight=%s)", insight_id)
            out = {
                "status": "subprocess_error",
                "error": f"{type(exc).__name__}: {exc}",
                "stdout": "",
                "stderr": "",
                "parsed_json": None,
            }
        out["purpose"] = purpose
        results[insight_id] = out
        logger.info(
            "[Verification] insight=%s status=%s",
            insight_id, out.get("status"),
        )
    return results


# ===================================================================
# Stage B-2: 收尾
# ===================================================================

def invoke_scientist_stage_b2(
    llm: LLMClient,
    *,
    drafts: List[Dict[str, Any]],
    query_results: Dict[str, Dict[str, Any]],
    verification_results: Dict[str, Dict[str, Any]],
    annotations_no_code: List[Dict[str, Any]],
    role: str = "scientist_stage_b2",
) -> Optional[str]:
    """单次 LLM 调用产出最终 insights.md 全文.

    Returns:
        markdown 字符串. LLM 调用失败 / 输出非 dict / 无 insights_md 字段
        返回 None.
    """
    payload = {
        "drafts": drafts,
        "query_results": _trim_query_results_for_prompt(query_results),
        "verification_results": _trim_verification_results_for_prompt(verification_results),
        "annotations_no_code": annotations_no_code,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    user_msg = (
        f"# Inputs\n"
        f"```json\n{payload_json}\n```\n\n"
        f"请输出 JSON 含 `insights_md` 字段 (string), 内容是完整的最终 insights.md\n"
        f"markdown. 三级标题格式严格匹配 `### I-{{id}} ({{status}}): {{title}}`."
    )

    try:
        result = llm.chat_json(
            role=role,
            system_prompt=prompts.SCIENTIST_STAGE_B2_SYSTEM,
            user_message=user_msg,
        )
    except Exception:
        logger.exception("[Stage B-2] LLM 调用失败")
        return None

    if not isinstance(result, dict):
        return None
    md = result.get("insights_md")
    if not isinstance(md, str) or not md.strip():
        logger.warning("[Stage B-2] insights_md 字段为空或非 str")
        return None
    return md


def _trim_query_results_for_prompt(
    query_results: Dict[str, Dict[str, Any]],
    max_layers_per_arch: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """精简 query_results 给 LLM. 每个 layer_profile 截前 max_layers_per_arch 行
    (按 cycles 降序), 避免 prompt 灌爆."""
    trimmed: Dict[str, Dict[str, Any]] = {}
    for insight_id, entry in query_results.items():
        new_entry = {
            "purpose": entry.get("purpose", ""),
            "match_count": entry.get("match_count", 0),
            "note": entry.get("note", ""),
            "matched_archs": [],
        }
        for arch in entry.get("matched_archs", []):
            profile = arch.get("layer_profile") or []
            if isinstance(profile, list) and len(profile) > max_layers_per_arch:
                profile = sorted(
                    profile,
                    key=lambda p: int(p.get("cycles", 0) or 0),
                    reverse=True,
                )[:max_layers_per_arch]
            new_entry["matched_archs"].append({
                "arch_code": arch.get("arch_code"),
                "epe": arch.get("epe"),
                "fps": arch.get("fps"),
                "layer_profile": profile,
            })
        trimmed[insight_id] = new_entry
    return trimmed


def _trim_verification_results_for_prompt(
    verification_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """精简 verification_results: 删除大块 stdout, 保留 parsed_json + status."""
    trimmed: Dict[str, Dict[str, Any]] = {}
    for insight_id, entry in verification_results.items():
        trimmed[insight_id] = {
            "purpose": entry.get("purpose", ""),
            "status": entry.get("status", ""),
            "parsed_json": entry.get("parsed_json"),
            "error": entry.get("error", ""),
            "stdout_tail": (entry.get("stdout") or "")[-500:],
        }
    return trimmed


# ===================================================================
# 完整 pipeline (orchestrator)
# ===================================================================

def _drafts_to_markdown(drafts: List[Dict[str, Any]]) -> str:
    """Stage B 失败时的兜底渲染: 把 stage A drafts 直接转成 insights.md 格式."""
    lines = [
        "# Search Insights",
        "",
        f"<!-- Stage B fallback render (last updated {datetime.now().isoformat()}) -->",
        "",
        "---",
        "",
    ]
    for d in drafts:
        if not insights.validate_id(d.get("id", "")):
            continue
        if d.get("status") not in insights.VALID_STATUSES:
            continue
        title = d.get("title", "").strip()
        if not title:
            continue
        lines.append(f"### {d['id']} ({d['status']}): {title}")
        lines.append("")
        body = d.get("body", "").strip()
        if body:
            lines.append(body)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def scientist_pipeline(
    llm: LLMClient,
    exp_dir: str,
    *,
    generation: int,
    total_generations: int,
    role_stage_a: str = "scientist_stage_a",
    role_stage_b1: str = "scientist_stage_b1",
    role_stage_b2: str = "scientist_stage_b2",
    sandbox_timeout: int = 30,
) -> Dict[str, Any]:
    """完整 Scientist 调用 (3 LLM calls + 0+ sandbox executions).

    流程:
        1. backup current insights.md → insights.md.gen{N}.bak
        2. Stage A
        3. Stage B-1
        4. resolve_vela_queries (Python)
        5. execute_verifications (sandbox)
        6. Stage B-2
        7. atomic write metadata/insights.md

    每阶段独立 fallback:
        - Stage A 失败 → 不修改 insights.md, return early
        - Stage B-1 失败 → 用 stage A drafts 直接渲染 fallback markdown
        - Stage B-2 失败 → 用 stage A drafts 直接渲染 fallback markdown
        - 任一阶段 sandbox 单条失败 → 该 verification 标 error, 流程继续

    Returns:
        dict 含执行摘要 (用于日志/审计):
            - success (bool): 至少 stage A 成功且 insights.md 被更新
            - stages_completed (List[str])
            - drafts_count (int)
            - vela_queries_count (int)
            - verifications_count (int)
            - error (str): 简短失败原因; 成功时空
    """
    summary: Dict[str, Any] = {
        "success": False,
        "stages_completed": [],
        "drafts_count": 0,
        "vela_queries_count": 0,
        "verifications_count": 0,
        "error": "",
    }

    # 0. 备份
    backup_path = backup_insights_md(exp_dir, generation)
    if backup_path:
        logger.info("[Scientist] 备份 insights.md → %s", backup_path)

    # 加载所需输入
    history_df = file_io.read_history(exp_dir)
    metrics_df = file_io.read_epoch_metrics(exp_dir)
    prev_md = read_insights_md(exp_dir)
    next_id_hint = insights.next_insight_id(prev_md)

    # 1. Stage A
    drafts = invoke_scientist_stage_a(
        llm,
        history_df=history_df,
        metrics_df=metrics_df,
        prev_insights_md=prev_md,
        generation=generation,
        total_generations=total_generations,
        next_id_hint=next_id_hint,
        role=role_stage_a,
    )
    if not drafts:
        summary["error"] = "Stage A returned no valid drafts"
        return summary
    summary["stages_completed"].append("stage_a")
    summary["drafts_count"] = len(drafts)

    # 2. Stage B-1
    plan = invoke_scientist_stage_b1(
        llm, drafts=drafts, history_df=history_df, role=role_stage_b1,
    )
    if plan is None:
        # Stage A 已成功; 用 drafts 直接渲染兜底
        fallback_md = _drafts_to_markdown(drafts)
        try:
            write_insights_md_atomic(exp_dir, fallback_md)
        except Exception as exc:
            summary["error"] = f"Stage B-1 failed and fallback write failed: {exc}"
            return summary
        summary["error"] = "Stage B-1 failed; wrote stage A drafts as fallback"
        summary["success"] = True
        return summary
    summary["stages_completed"].append("stage_b1")
    summary["vela_queries_count"] = len(plan["vela_queries"])
    summary["verifications_count"] = len(plan["verifications"])

    # 3. Resolve vela_queries (Python, 不调 LLM)
    query_results = resolve_vela_queries(
        plan["vela_queries"], history_df, exp_dir,
    )
    summary["stages_completed"].append("vela_resolve")

    # 4. Execute verifications (sandbox)
    history_csv = os.path.join(exp_dir, "metadata", "history_archive.csv")
    verification_results = execute_verifications(
        plan["verifications"],
        history_csv_path=history_csv,
        query_results=query_results,
        timeout=sandbox_timeout,
    )
    summary["stages_completed"].append("sandbox_exec")

    # 5. Stage B-2
    final_md = invoke_scientist_stage_b2(
        llm,
        drafts=drafts,
        query_results=query_results,
        verification_results=verification_results,
        annotations_no_code=plan["annotations_no_code"],
        role=role_stage_b2,
    )
    if final_md is None:
        # 兜底: Stage A drafts 直接渲染
        fallback_md = _drafts_to_markdown(drafts)
        try:
            write_insights_md_atomic(exp_dir, fallback_md)
        except Exception as exc:
            summary["error"] = f"Stage B-2 failed and fallback write failed: {exc}"
            return summary
        summary["error"] = "Stage B-2 failed; wrote stage A drafts as fallback"
        summary["success"] = True
        return summary
    summary["stages_completed"].append("stage_b2")

    # 6. Atomic write
    try:
        write_insights_md_atomic(exp_dir, final_md)
    except Exception as exc:
        summary["error"] = f"final atomic write failed: {exc}"
        return summary

    summary["success"] = True
    return summary
