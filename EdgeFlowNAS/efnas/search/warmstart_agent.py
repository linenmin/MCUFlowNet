"""Phase 2 (search_hybrid_v1): Warmstart Agent —— 单次 LLM 调用产生 NSGA-II
generation 0 初始种群。

设计哲学:
  - 给 agent 全局信息和角色定位，不规定具体策略 (多样性、覆盖、聚集程度等)
  - Engineering 层只管硬正确性 (合法 arch_code) 和总失败兜底 (LLM 全错时
    NSGA-II 改用 random init)
  - Diagnostics 落盘但不门控 (per-dim entropy, valid count, agent 自述策略)

整个生命周期: agent 在 NSGA-II generation 0 之前调用一次, 之后所有代由
NSGA-II 的 binary tournament + uniform per-gene crossover (90%) +
per-gene mutation (1/11) 演化你的种子。
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from efnas.search import prompts
from efnas.search.llm_client import LLMClient

logger = logging.getLogger(__name__)


def invoke_warmstart_agent(
    llm: LLMClient,
    *,
    population_size: int = 50,
    role: str = "warmstart_agent",
) -> Dict[str, Any]:
    """单次调用 warmstart agent, 返回原始 LLM 响应 (含 rationale + arch_codes).

    本函数不做任何 arch_code 合法性校验 —— 只把 LLM 输出直接 parse 出来。
    合法性校验和 partial random fill 由 NSGA2SearchRunner.
    `_consume_external_initial_population` 处理 (Phase 2.1)。

    Args:
        llm: LLMClient 实例 (须已加载好 warmstart_agent role 的模型路由)
        population_size: 期望生成的 arch_code 数量。注入 prompt, 但 agent 实际
            生成数量由 LLM 决定; 工程层会做 partial fill 处理。默认 50.
        role: LLM client 的 role 名 (用于路由到指定模型, 默认 "warmstart_agent")

    Returns:
        dict 含三个 key:
            - rationale (str): agent 自述的策略思考 (落盘到 diagnostics)
            - arch_codes (List[str]): agent 输出的 arch_code 列表 (未校验合法性)
            - raw_response (Any): 原始 LLM JSON 响应, 用于审计

        LLM 调用失败时返回 ``{"rationale": "", "arch_codes": [], "raw_response": None}``，
        让 NSGA-II 的 partial fill 兜底接手 (走 random init)。
    """
    system_prompt = prompts.WARMSTART_AGENT_SYSTEM.format(
        population_size=population_size,
    )
    user_msg = (
        f"请输出恰好 {population_size} 个 arch_code 作为 NSGA-II generation 0 的初始种群。\n"
        f"严格遵守上面的 OUTPUT FORMAT (JSON 含 rationale 和 arch_codes 两字段)。\n"
        f"具体的策略选择 (多样性 / 覆盖 / 聚集) 由你自己决定。"
    )

    try:
        result = llm.chat_json(
            role=role,
            system_prompt=system_prompt,
            user_message=user_msg,
        )
    except Exception:
        logger.exception("[Warmstart] LLM 调用失败")
        return {"rationale": "", "arch_codes": [], "raw_response": None}

    if not isinstance(result, dict):
        logger.warning("[Warmstart] LLM 响应不是 dict: %r", type(result))
        return {"rationale": "", "arch_codes": [], "raw_response": result}

    rationale = str(result.get("rationale", "")).strip()
    arch_codes_raw = result.get("arch_codes", [])
    if not isinstance(arch_codes_raw, list):
        logger.warning(
            "[Warmstart] arch_codes 字段不是 list: %r (type %s)",
            arch_codes_raw, type(arch_codes_raw).__name__,
        )
        arch_codes_raw = []

    arch_codes: List[str] = [str(c).strip() for c in arch_codes_raw if str(c).strip()]

    logger.info(
        "[Warmstart] 收到 %d 个 arch_code (期望 %d), rationale 长度 %d 字符",
        len(arch_codes), population_size, len(rationale),
    )
    return {
        "rationale": rationale,
        "arch_codes": arch_codes,
        "raw_response": result,
    }


def compute_warmstart_diagnostics(
    arch_codes: List[str],
    *,
    search_space: Any,
    requested_count: int,
    rationale: str = "",
    llm_model: str = "",
) -> Dict[str, Any]:
    """计算 warmstart 输出的多样性 / 合法性诊断指标。

    本函数**不门控**，只观察 + 落盘. 即使 LLM 输出多样性差或全部非法，也不
    抛错，让 NSGA-II 的 partial random fill 兜底接手。Phase 5 ablation 时会
    把这些诊断和最终 HV 关联做相关分析。

    Args:
        arch_codes: warmstart agent 输出的 arch_code 列表 (raw, 可能含非法/重复)
        search_space: SearchSpaceAdapter 实例 (用于 validate)
        requested_count: 我们要求 agent 输出的数量 (通常 50)
        rationale: agent 自述的策略思考 (将原样落盘)
        llm_model: 调用的 LLM 模型名 (用于审计)

    Returns:
        dict 含:
            - timestamp (ISO str)
            - llm_model (str)
            - requested_count (int)
            - returned_count (int): LLM 实际返回的条目数
            - valid_count (int): 通过 search_space.validate 的条目数
            - invalid_count (int)
            - duplicate_within_batch (int): 合法但 batch 内重复的条目数
            - unique_valid_count (int): 合法且去重后的条目数
            - per_dim_entropy (List[float]): 11 维 Shannon 熵 (基于 unique
              valid arch_codes)
            - rationale (str)
    """
    from efnas.search import search_metrics

    returned = len(arch_codes)
    valid_codes: List[str] = []
    invalid_count = 0
    seen: set = set()
    duplicate_within_batch = 0

    for raw in arch_codes:
        text = str(raw).strip()
        if not text:
            invalid_count += 1
            continue
        try:
            parts = [int(p.strip()) for p in text.split(",")]
            search_space.validate(parts)
        except Exception:
            invalid_count += 1
            continue
        normalized = ",".join(str(int(p)) for p in parts)
        if normalized in seen:
            duplicate_within_batch += 1
            continue
        seen.add(normalized)
        valid_codes.append(normalized)

    per_dim_entropy = search_metrics.per_dim_gene_entropy(valid_codes, num_dims=11)

    return {
        "timestamp": datetime.now().isoformat(),
        "llm_model": llm_model,
        "requested_count": int(requested_count),
        "returned_count": int(returned),
        "valid_count": int(len(valid_codes) + duplicate_within_batch),
        "invalid_count": int(invalid_count),
        "duplicate_within_batch": int(duplicate_within_batch),
        "unique_valid_count": int(len(valid_codes)),
        "per_dim_entropy": [round(e, 6) for e in per_dim_entropy],
        "rationale": rationale,
    }


def save_warmstart_diagnostics(
    exp_dir: str,
    diagnostics: Dict[str, Any],
    filename: str = "warmstart_diagnostics.json",
) -> str:
    """把 diagnostics 落盘到 ``<exp_dir>/metadata/<filename>``.

    Returns:
        写入的 JSON 文件绝对路径.
    """
    metadata_dir = os.path.join(exp_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    path = os.path.join(metadata_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    logger.info("[Warmstart] diagnostics 已落盘: %s", path)
    return path


def warmstart_pipeline(
    llm: LLMClient,
    exp_dir: str,
    *,
    search_space: Any,
    population_size: int = 50,
    llm_model: str = "",
    role: str = "warmstart_agent",
) -> List[str]:
    """完整 warmstart 流程: invoke → compute diagnostics → save → return arch_codes.

    供 wrapper 一行调用. 总失败兜底: 任何环节出错都返回空列表, 让 NSGA-II
    走 random init.

    Args:
        llm: LLMClient
        exp_dir: 实验输出根目录 (用于 diagnostics 落盘)
        search_space: SearchSpaceAdapter
        population_size: 目标种群大小
        llm_model: 仅用于 diagnostics 标注
        role: LLM 路由 role 名

    Returns:
        合法且去重后的 arch_code 列表 (可能少于 population_size, NSGA-II 会
        partial random fill).
    """
    response = invoke_warmstart_agent(
        llm, population_size=population_size, role=role,
    )
    arch_codes = response.get("arch_codes", [])
    rationale = response.get("rationale", "")

    diagnostics = compute_warmstart_diagnostics(
        arch_codes,
        search_space=search_space,
        requested_count=population_size,
        rationale=rationale,
        llm_model=llm_model,
    )
    try:
        save_warmstart_diagnostics(exp_dir, diagnostics)
    except Exception:
        logger.exception("[Warmstart] diagnostics 落盘失败 (不阻断)")

    # 提取合法 + 去重后的列表给 NSGA-II
    valid_codes: List[str] = []
    seen: set = set()
    for raw in arch_codes:
        text = str(raw).strip()
        if not text:
            continue
        try:
            parts = [int(p.strip()) for p in text.split(",")]
            search_space.validate(parts)
        except Exception:
            continue
        normalized = ",".join(str(int(p)) for p in parts)
        if normalized not in seen:
            seen.add(normalized)
            valid_codes.append(normalized)

    return valid_codes
