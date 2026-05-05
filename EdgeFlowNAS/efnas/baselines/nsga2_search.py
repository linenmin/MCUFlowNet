"""NSGA-II baseline search over the V2 supernet search space."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from efnas.nas.search_space_v2 import get_num_blocks, get_num_choices, validate_arch_code

logger = logging.getLogger(__name__)


class SearchSpaceAdapter:
    """Small adapter around a categorical search-space module."""

    def __init__(self, module_name: str):
        self.module_name = str(module_name)
        self.module = importlib.import_module(self.module_name)

    def num_blocks(self) -> int:
        return int(self.module.get_num_blocks())

    def num_choices(self, block_idx: int) -> int:
        return int(self.module.get_num_choices(block_idx))

    def validate(self, arch_code: Sequence[int]) -> None:
        self.module.validate_arch_code([int(item) for item in arch_code])


def load_search_space(module_name: Optional[str] = None) -> SearchSpaceAdapter:
    """Load a categorical search-space adapter."""
    return SearchSpaceAdapter(module_name or "efnas.nas.search_space_v2")


def _parse_gpu_devices(raw_devices: Any) -> List[str]:
    """Parse a comma/space separated GPU device list."""
    if raw_devices is None:
        return []
    if isinstance(raw_devices, (list, tuple)):
        return [str(item).strip() for item in raw_devices if str(item).strip()]
    return [item.strip() for item in str(raw_devices).replace(";", ",").split(",") if item.strip()]


def assign_gpus_to_arches(arch_codes: Sequence[str], gpu_devices: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
    """Assign candidate architectures to visible GPU ids in round-robin order."""
    devices = [str(item).strip() for item in gpu_devices if str(item).strip()]
    assignments: List[Tuple[str, Optional[str]]] = []
    for idx, arch_code in enumerate(arch_codes):
        assigned_gpu = devices[idx % len(devices)] if devices else None
        assignments.append((str(arch_code), assigned_gpu))
    return assignments


def _file_io():
    """Import search file I/O lazily so pure helper tests stay light."""
    from efnas.search import file_io as search_file_io

    return search_file_io


def _evaluate_single_arch():
    """Import eval worker lazily to avoid heavyweight imports in pure tests."""
    from efnas.search.eval_worker import evaluate_single_arch as worker_fn

    return worker_fn


def arch_to_text(arch_code: Sequence[int]) -> str:
    """Convert one architecture code to canonical comma-separated text."""
    return ",".join(str(int(value)) for value in arch_code)


def text_to_arch(arch_code_text: str, search_space: Optional[SearchSpaceAdapter] = None) -> List[int]:
    """Parse canonical architecture text."""
    arch_code = [int(token.strip()) for token in str(arch_code_text).split(",") if token.strip()]
    if search_space is None:
        validate_arch_code(arch_code)
    else:
        search_space.validate(arch_code)
    return arch_code


def resolve_generation_count(total_evaluations: int, population_size: int) -> int:
    """Resolve generation count from budget and population size."""
    total = int(total_evaluations)
    pop_size = int(population_size)
    if total <= 0 or pop_size <= 0:
        raise ValueError("total_evaluations and population_size must be positive")
    if total % pop_size != 0:
        raise ValueError("total_evaluations must be divisible by population_size")
    return total // pop_size


def sample_random_arch(rng: random.Random, search_space: Optional[SearchSpaceAdapter] = None) -> List[int]:
    """Sample one valid random V2 architecture."""
    space = search_space or load_search_space()
    arch_code = [rng.randrange(space.num_choices(block_idx)) for block_idx in range(space.num_blocks())]
    space.validate(arch_code)
    return arch_code


def mutate_arch(
    arch_code: Sequence[int],
    rng: random.Random,
    mutation_prob: float,
    search_space: Optional[SearchSpaceAdapter] = None,
    *,
    per_dim_multiplier: Optional[Sequence[float]] = None,
) -> List[int]:
    """Mutate one architecture with per-gene categorical mutation.

    Phase 4: optional ``per_dim_multiplier`` 给每维独立 mutation_prob 缩放系数.
    实际每维 mutation 概率 = ``mutation_prob × multiplier[d]``, 截断到 [0, 1].
    长度不足或 None 时退化为均匀 mutation_prob (Phase 1 行为).
    """
    space = search_space or load_search_space()
    child = [int(value) for value in arch_code]
    base_prob = float(mutation_prob)
    for block_idx in range(space.num_blocks()):
        effective_prob = base_prob
        if per_dim_multiplier is not None and block_idx < len(per_dim_multiplier):
            try:
                effective_prob = max(0.0, min(1.0, base_prob * float(per_dim_multiplier[block_idx])))
            except (TypeError, ValueError):
                effective_prob = base_prob
        if rng.random() >= effective_prob:
            continue
        num_choices = space.num_choices(block_idx)
        candidates = [value for value in range(num_choices) if value != child[block_idx]]
        child[block_idx] = int(rng.choice(candidates))
    space.validate(child)
    return child


def uniform_crossover(
    parent_a: Sequence[int],
    parent_b: Sequence[int],
    rng: random.Random,
    crossover_prob: float,
    search_space: Optional[SearchSpaceAdapter] = None,
) -> Tuple[List[int], List[int]]:
    """Perform uniform crossover on two categorical architectures."""
    space = search_space or load_search_space()
    child_a = [int(value) for value in parent_a]
    child_b = [int(value) for value in parent_b]
    if rng.random() >= float(crossover_prob):
        return child_a, child_b
    for block_idx in range(space.num_blocks()):
        if rng.random() < 0.5:
            child_a[block_idx], child_b[block_idx] = child_b[block_idx], child_a[block_idx]
    space.validate(child_a)
    space.validate(child_b)
    return child_a, child_b


def fast_non_dominated_sort(objective_values: Sequence[Tuple[float, float]]) -> List[List[int]]:
    """Compute non-dominated fronts for minimization objectives."""
    population_size = len(objective_values)
    dominates_map: List[List[int]] = [[] for _ in range(population_size)]
    domination_count = [0 for _ in range(population_size)]
    fronts: List[List[int]] = [[]]

    for idx_a in range(population_size):
        for idx_b in range(population_size):
            if idx_a == idx_b:
                continue
            value_a = objective_values[idx_a]
            value_b = objective_values[idx_b]
            if _dominates(value_a, value_b):
                dominates_map[idx_a].append(idx_b)
            elif _dominates(value_b, value_a):
                domination_count[idx_a] += 1
        if domination_count[idx_a] == 0:
            fronts[0].append(idx_a)

    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front: List[int] = []
        for idx_a in fronts[front_idx]:
            for idx_b in dominates_map[idx_a]:
                domination_count[idx_b] -= 1
                if domination_count[idx_b] == 0:
                    next_front.append(idx_b)
        if next_front:
            fronts.append(next_front)
        front_idx += 1
    return fronts


def select_next_population(rows: Sequence[Dict[str, Any]], population_size: int) -> List[Dict[str, Any]]:
    """Select the next NSGA-II population from evaluated rows."""
    if population_size <= 0 or not rows:
        return []
    objective_values = [_row_to_objectives(row) for row in rows]
    fronts = fast_non_dominated_sort(objective_values)
    selected_indices: List[int] = []

    for front in fronts:
        if len(selected_indices) + len(front) <= population_size:
            selected_indices.extend(front)
            continue
        crowding = crowding_distance(front, objective_values)
        ordered = sorted(front, key=lambda idx: crowding[idx], reverse=True)
        remaining = population_size - len(selected_indices)
        selected_indices.extend(ordered[:remaining])
        break

    return [dict(rows[idx]) for idx in selected_indices]


def crowding_distance(front: Sequence[int], objective_values: Sequence[Tuple[float, float]]) -> Dict[int, float]:
    """Compute NSGA-II crowding distance for one front."""
    distances = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distances[idx] = float("inf")
        return distances

    num_objectives = len(objective_values[0])
    for objective_idx in range(num_objectives):
        ordered = sorted(front, key=lambda idx: objective_values[idx][objective_idx])
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        min_value = objective_values[ordered[0]][objective_idx]
        max_value = objective_values[ordered[-1]][objective_idx]
        if max_value == min_value:
            continue
        for rank in range(1, len(ordered) - 1):
            if distances[ordered[rank]] == float("inf"):
                continue
            left_value = objective_values[ordered[rank - 1]][objective_idx]
            right_value = objective_values[ordered[rank + 1]][objective_idx]
            distances[ordered[rank]] += (right_value - left_value) / (max_value - min_value)
    return distances


def _dominates(left: Tuple[float, float], right: Tuple[float, float]) -> bool:
    """Return True when left Pareto-dominates right for minimization."""
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def _row_to_objectives(row: Dict[str, Any]) -> Tuple[float, float]:
    """Map row metrics to minimization objectives."""
    return float(row["epe"]), -float(row["fps"])


class NSGA2SearchRunner:
    """Project-native NSGA-II baseline runner."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        exp_dir: str,
        project_root: str,
        *,
        external_initial_population: Optional[Sequence[str]] = None,
        scientist_llm: Optional[Any] = None,
        scientist_interval: int = 3,
        scientist_sandbox_timeout: int = 30,
        supervisor_llm: Optional[Any] = None,
    ) -> None:
        """初始化 NSGA-II runner。

        Args:
            cfg: 完整配置字典（含 search / concurrency / evaluation / files 段）
            exp_dir: 本次实验输出根目录
            project_root: EdgeFlowNAS 项目根
            external_initial_population: Phase 2 warm-start hook —— 可选的外部
                generation-0 初始种群（arch_code 字符串列表）。如果提供，gen 0 不
                走纯 random sampling，而是消费这个列表并对非法/重复条目做 random
                partial fill。Phase 2.4 wrapper 会从 warmstart_agent 拿到这个列表。
        """
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.project_root = project_root

        search_cfg = cfg["search"]
        self.total_evaluations = int(search_cfg["total_evaluations"])
        self.population_size = int(search_cfg["population_size"])
        self.total_generations = resolve_generation_count(self.total_evaluations, self.population_size)
        self.seed = int(search_cfg.get("seed", 2026))
        self.crossover_prob = float(search_cfg.get("crossover_prob", 0.9))
        self.search_space = load_search_space(search_cfg.get("search_space_module", "efnas.nas.search_space_v2"))
        mutation_prob = search_cfg.get("mutation_prob")
        self.mutation_prob = float(mutation_prob) if mutation_prob is not None else 1.0 / float(self.search_space.num_blocks())
        self.duplicate_retry_factor = int(search_cfg.get("duplicate_retry_factor", 20))
        self.search_space_size = int(search_cfg.get("search_space_size", 23328))
        self.max_workers = int(cfg["concurrency"]["max_workers"])
        self.gpu_devices = _parse_gpu_devices(cfg.get("concurrency", {}).get("gpu_devices"))
        self.prune_tflite_after_reduce = bool(cfg.get("evaluation", {}).get("vela_prune_tflite_after_reduce", False))
        self.state_path = Path(self.exp_dir) / "metadata" / str(cfg.get("files", {}).get("state_json", "nsga2_state.json"))
        self.pareto_csv_path = Path(self.exp_dir) / "metadata" / str(cfg.get("files", {}).get("pareto_csv", "pareto_front.csv"))
        self._rng = random.Random(self.seed)
        # Phase 2.1: warm-start hook
        self._external_initial_population: List[str] = list(external_initial_population or [])
        # Phase 3: scientist hook (None 表示禁用)
        self._scientist_llm = scientist_llm
        self._scientist_interval = max(1, int(scientist_interval))
        self._scientist_sandbox_timeout = int(scientist_sandbox_timeout)
        # Phase 4: 5-lever 可变状态 (Supervisor 通过 apply_supervisor_actions 调整)
        # mutation_prob / crossover_prob 已在上面用 self.* 存; tournament_size /
        # per_dim_multiplier / reseed_bottom_pct 是 Phase 4 新增.
        self._tournament_size: int = 2
        self._per_dim_mutation_multiplier: List[float] = [1.0] * self.search_space.num_blocks()
        self._reseed_bottom_pct: int = 0
        self._supervisor_llm: Optional[Any] = supervisor_llm

    def run(self) -> None:
        """Run the NSGA-II baseline."""
        logger.info("=" * 60)
        logger.info("EdgeFlowNAS NSGA-II baseline search start")
        logger.info("experiment=%s pop_size=%d total_generations=%d budget=%d", self.exp_dir, self.population_size, self.total_generations, self.total_evaluations)
        logger.info("=" * 60)

        _file_io().rescue_orphaned_results(self.exp_dir)
        self._maybe_prune_vela_tflite(stage="startup_rescue")
        state = self._load_or_init_state()
        current_population = self._rows_from_arch_codes(state.get("population_arches", []))
        start_generation = int(state.get("next_generation", 0))

        try:
            for generation in range(start_generation, self.total_generations):
                logger.info("=" * 50)
                logger.info("=== Generation %d / %d ===", generation, self.total_generations - 1)
                logger.info("=" * 50)
                current_population = self._run_single_generation(generation, current_population)
                self._save_state(next_generation=generation + 1, population=current_population)
        except KeyboardInterrupt:
            _file_io().collect_and_commit_worker_results(self.exp_dir)
            self._maybe_prune_vela_tflite(stage="interrupt_rescue")
            raise
        except Exception:
            _file_io().collect_and_commit_worker_results(self.exp_dir)
            self._maybe_prune_vela_tflite(stage="exception_rescue")
            raise

        logger.info("=" * 60)
        logger.info("NSGA-II baseline finished")
        logger.info("=" * 60)

    def _run_single_generation(
        self,
        generation: int,
        current_population: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        existing_rows = self._rows_for_generation(generation)
        duplicate_count = 0

        needed = max(0, self.population_size - len(existing_rows))
        if needed > 0:
            if generation == 0 and not current_population:
                # Phase 2.1: 优先消费外部 warm-start 种群；不足部分 random 补齐
                if self._external_initial_population:
                    candidate_arches, duplicate_count = self._consume_external_initial_population(needed)
                else:
                    candidate_arches, duplicate_count = self._sample_unique_random_arches(needed)
            else:
                # Phase 4 lever 5: reseed_bottom_pct 把 needed 拆成 offspring +
                # random injection 两部分.
                reseed_pct = max(0, min(100, int(self._reseed_bottom_pct or 0)))
                n_reseed = int(round(needed * reseed_pct / 100.0)) if reseed_pct > 0 else 0
                n_offspring = max(0, needed - n_reseed)
                candidate_arches = []
                duplicate_count = 0
                if n_offspring > 0:
                    offspring, dup1 = self._generate_offspring_arches(
                        current_population, n_offspring,
                    )
                    candidate_arches.extend(offspring)
                    duplicate_count += dup1
                if n_reseed > 0:
                    reseed_arches, dup2 = self._sample_unique_random_arches(
                        n_reseed, extra_seen=set(candidate_arches),
                    )
                    candidate_arches.extend(reseed_arches)
                    duplicate_count += dup2
                    logger.info(
                        "[Reseed] gen=%d injected %d random arches (pct=%d%%, "
                        "offspring=%d, total=%d)",
                        generation, len(reseed_arches), reseed_pct,
                        n_offspring, len(candidate_arches),
                    )
            if candidate_arches:
                self._evaluate_arch_batch(candidate_arches, generation)
                _file_io().collect_and_commit_worker_results(self.exp_dir)
                self._maybe_prune_vela_tflite(stage=f"generation_{generation}_reduce")

        generation_rows = self._rows_for_generation(generation)

        if generation == 0 and not current_population:
            next_population = select_next_population(generation_rows, min(self.population_size, len(generation_rows)))
        else:
            combined_rows = list(current_population) + [row for row in generation_rows if row["arch_code"] not in {item["arch_code"] for item in current_population}]
            next_population = select_next_population(combined_rows, min(self.population_size, len(combined_rows)))

        self._record_generation_metrics(
            generation=generation,
            new_evaluated=len(generation_rows),
            duplicates=duplicate_count,
            current_population=next_population,
        )
        self._write_pareto_snapshot()
        logger.info(
            "generation=%d evaluated=%d duplicates=%d next_population=%d",
            generation,
            len(generation_rows),
            duplicate_count,
            len(next_population),
        )

        # Phase 3: 每 K 代触发 Scientist 大反思 (best-effort, 不阻断主搜索)
        self._maybe_invoke_scientist(generation)

        # Phase 4: Scientist 之后立刻 Supervisor (best-effort, 不阻断主搜索)
        self._maybe_invoke_supervisor(generation)

        return next_population

    def _maybe_invoke_scientist(self, generation: int) -> None:
        """Phase 3: 每 K=scientist_interval 代后触发 Scientist 三阶段反思.

        触发时机: ``(generation + 1) % K == 0``. 比如 K=3 时, 在 generation
        2 / 5 / 8 / 11 / 14 之后各触发一次 (16 代总长 → 5 次反思).

        失败兜底: scientist_pipeline 任何阶段失败都不抛异常出来, 也不阻断
        NSGA-II 主搜索. log 一行执行摘要即可.
        """
        if self._scientist_llm is None:
            return
        if (generation + 1) % self._scientist_interval != 0:
            return
        try:
            from efnas.search.scientist_agent import scientist_pipeline
            summary = scientist_pipeline(
                self._scientist_llm,
                self.exp_dir,
                generation=generation,
                total_generations=self.total_generations,
                sandbox_timeout=self._scientist_sandbox_timeout,
            )
            logger.info(
                "[Scientist] gen=%d success=%s stages=%s drafts=%d "
                "queries=%d verifications=%d %s",
                generation,
                summary.get("success"),
                ",".join(summary.get("stages_completed", [])),
                summary.get("drafts_count", 0),
                summary.get("vela_queries_count", 0),
                summary.get("verifications_count", 0),
                f"error={summary.get('error')}" if summary.get("error") else "",
            )
        except Exception:
            logger.exception("[Scientist] pipeline 异常 (不阻断主搜索)")

    def _maybe_invoke_supervisor(self, generation: int) -> None:
        """Phase 4: 每 K=scientist_interval 代后触发 Supervisor (5-lever 调参).

        触发频率与 Scientist 同步 (K=3), 在 Scientist 跑完之后立即跑, 让
        Supervisor 能消费 fresh insights.md.

        失败兜底: supervisor_pipeline 任何环节失败都不抛异常, 不阻断 NSGA-II
        主搜索. 失败时 supervisor_log 仍会记录一条 "LLM failed" 条目.
        """
        if self._supervisor_llm is None:
            return
        if (generation + 1) % self._scientist_interval != 0:
            return
        try:
            from efnas.search.supervisor_agent import supervisor_pipeline
            summary = supervisor_pipeline(
                self._supervisor_llm, self.exp_dir, self,
                generation=generation,
            )
            logger.info(
                "[Supervisor] gen=%d success=%s applied=%s rejected=%s %s",
                generation,
                summary.get("success"),
                list(summary.get("applied", {}).keys()),
                list(summary.get("rejected", {}).keys()),
                f"error={summary.get('error')}" if summary.get("error") else "",
            )
        except Exception:
            logger.exception("[Supervisor] pipeline 异常 (不阻断主搜索)")

    # ===============================================================
    # Phase 4: Supervisor 5-lever 接口
    # ===============================================================

    def current_supervisor_state(self) -> Dict[str, Any]:
        """返回 5 个 lever 的当前数值, 给 Supervisor agent 看自己上次调了啥."""
        return {
            "mutation_prob": float(self.mutation_prob),
            "crossover_prob": float(self.crossover_prob),
            "tournament_size": int(self._tournament_size),
            "per_dim_mutation_multiplier": list(self._per_dim_mutation_multiplier),
            "reseed_bottom_pct": int(self._reseed_bottom_pct),
        }

    def apply_supervisor_actions(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """应用 Supervisor agent 的 5-lever 调整, 仅做数学合法性校验.

        策略选择 (是否合理) 全交给 agent. 本函数只拒绝**数学上无效**的值
        (NaN / 越界到无法跑 NSGA-II 的范围). 对合法值不做"建议范围"约束.

        Args:
            actions: dict 含可选字段 mutation_prob / crossover_prob /
                tournament_size / per_dim_mutation_multiplier / reseed_bottom_pct.
                每字段允许 None 表示不调.

        Returns:
            dict 含:
                - applied: 实际生效的字段及新值
                - rejected: 被拒字段及原因
                - before_state: 调整前的 5 lever 值
                - after_state: 调整后的 5 lever 值
        """
        before = self.current_supervisor_state()
        applied: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}

        if actions is None or not isinstance(actions, dict):
            return {
                "applied": {},
                "rejected": {"_global": "actions is not a dict"},
                "before_state": before,
                "after_state": before,
            }

        # mutation_prob
        if actions.get("mutation_prob") is not None:
            try:
                v = float(actions["mutation_prob"])
                if not (0.0 <= v <= 1.0) or v != v:  # NaN check
                    rejected["mutation_prob"] = f"out of [0.0, 1.0]: {v}"
                else:
                    self.mutation_prob = v
                    applied["mutation_prob"] = v
            except (TypeError, ValueError) as e:
                rejected["mutation_prob"] = f"invalid: {e}"

        # crossover_prob
        if actions.get("crossover_prob") is not None:
            try:
                v = float(actions["crossover_prob"])
                if not (0.0 <= v <= 1.0) or v != v:
                    rejected["crossover_prob"] = f"out of [0.0, 1.0]: {v}"
                else:
                    self.crossover_prob = v
                    applied["crossover_prob"] = v
            except (TypeError, ValueError) as e:
                rejected["crossover_prob"] = f"invalid: {e}"

        # tournament_size
        if actions.get("tournament_size") is not None:
            try:
                v = int(actions["tournament_size"])
                if v < 2 or v > self.population_size:
                    rejected["tournament_size"] = (
                        f"out of [2, {self.population_size}]: {v}"
                    )
                else:
                    self._tournament_size = v
                    applied["tournament_size"] = v
            except (TypeError, ValueError) as e:
                rejected["tournament_size"] = f"invalid: {e}"

        # per_dim_mutation_multiplier
        if actions.get("per_dim_mutation_multiplier") is not None:
            raw = actions["per_dim_mutation_multiplier"]
            num_dims = self.search_space.num_blocks()
            if not isinstance(raw, list) or len(raw) != num_dims:
                rejected["per_dim_mutation_multiplier"] = (
                    f"must be list of length {num_dims}, got {type(raw).__name__} "
                    f"len={len(raw) if isinstance(raw, list) else 'n/a'}"
                )
            else:
                try:
                    parsed = [float(x) for x in raw]
                    if any(x < 0.0 or x != x for x in parsed):
                        rejected["per_dim_mutation_multiplier"] = (
                            "all elements must be non-negative finite"
                        )
                    else:
                        self._per_dim_mutation_multiplier = parsed
                        applied["per_dim_mutation_multiplier"] = parsed
                except (TypeError, ValueError) as e:
                    rejected["per_dim_mutation_multiplier"] = f"invalid: {e}"

        # reseed_bottom_pct
        if actions.get("reseed_bottom_pct") is not None:
            try:
                v = int(actions["reseed_bottom_pct"])
                if v < 0 or v > 100:
                    rejected["reseed_bottom_pct"] = f"out of [0, 100]: {v}"
                else:
                    self._reseed_bottom_pct = v
                    applied["reseed_bottom_pct"] = v
            except (TypeError, ValueError) as e:
                rejected["reseed_bottom_pct"] = f"invalid: {e}"

        return {
            "applied": applied,
            "rejected": rejected,
            "before_state": before,
            "after_state": self.current_supervisor_state(),
        }

    def _maybe_prune_vela_tflite(self, stage: str) -> int:
        """Optionally delete heavyweight Vela TFLite artifacts while keeping metrics."""
        if not self.prune_tflite_after_reduce:
            return 0
        removed = _file_io().prune_vela_tflite_artifacts(self.exp_dir)
        if removed > 0:
            logger.info("[Prune] %s: deleted %d Vela tflite files", stage, removed)
        return removed

    def _generate_offspring_arches(
        self,
        current_population: Sequence[Dict[str, Any]],
        target_count: int,
    ) -> Tuple[List[str], int]:
        objective_values = [_row_to_objectives(row) for row in current_population]
        fronts = fast_non_dominated_sort(objective_values)
        ranks = self._build_rank_lookup(fronts)
        crowding_lookup = self._build_crowding_lookup(fronts, objective_values)

        evaluated_arches = _file_io().get_evaluated_arch_codes(self.exp_dir)
        batch_arches: List[str] = []
        batch_seen = set()
        duplicate_count = 0
        max_attempts = max(target_count * self.duplicate_retry_factor, target_count * 4)
        attempts = 0

        while len(batch_arches) < target_count and attempts < max_attempts:
            parent_a = current_population[self._tournament_select_index(current_population, ranks, crowding_lookup)]
            parent_b = current_population[self._tournament_select_index(current_population, ranks, crowding_lookup)]
            child_a, child_b = uniform_crossover(
                text_to_arch(parent_a["arch_code"], search_space=self.search_space),
                text_to_arch(parent_b["arch_code"], search_space=self.search_space),
                rng=self._rng,
                crossover_prob=self.crossover_prob,
                search_space=self.search_space,
            )
            for child in (
                mutate_arch(
                    child_a, rng=self._rng, mutation_prob=self.mutation_prob,
                    search_space=self.search_space,
                    per_dim_multiplier=self._per_dim_mutation_multiplier,
                ),
                mutate_arch(
                    child_b, rng=self._rng, mutation_prob=self.mutation_prob,
                    search_space=self.search_space,
                    per_dim_multiplier=self._per_dim_mutation_multiplier,
                ),
            ):
                arch_text = arch_to_text(child)
                if arch_text in evaluated_arches or arch_text in batch_seen:
                    duplicate_count += 1
                    continue
                batch_arches.append(arch_text)
                batch_seen.add(arch_text)
                if len(batch_arches) >= target_count:
                    break
            attempts += 1

        if len(batch_arches) < target_count:
            fallback_arches, fallback_duplicates = self._sample_unique_random_arches(
                target_count - len(batch_arches),
                extra_seen=batch_seen,
            )
            batch_arches.extend(fallback_arches)
            duplicate_count += fallback_duplicates

        return batch_arches, duplicate_count

    def _consume_external_initial_population(
        self,
        target_count: int,
    ) -> Tuple[List[str], int]:
        """Phase 2.1: 消费 warm-start agent 提供的外部初始种群。

        策略:
        1. 按顺序遍历 self._external_initial_population
        2. 用 search_space.validate 验证每条 arch_code
        3. 跳过非法 / 已评估 / 同 batch 内重复
        4. 收集 target_count 个合法的；如果外部种群不够 target_count，剩余位置
           用 _sample_unique_random_arches 补齐 (partial fill)
        5. **不 fail**——这是 best-effort warm-start，LLM 出错也不让 NSGA-II 挂掉

        Returns:
            (collected_arch_codes, duplicate_count)
        """
        evaluated_arches = _file_io().get_evaluated_arch_codes(self.exp_dir)
        batch_arches: List[str] = []
        batch_seen: set = set()
        duplicate_count = 0
        invalid_count = 0
        external_total = len(self._external_initial_population)

        for raw_code in self._external_initial_population:
            if len(batch_arches) >= target_count:
                break
            arch_text = str(raw_code).strip()
            if not arch_text:
                continue
            try:
                parsed = text_to_arch(arch_text, search_space=self.search_space)
            except Exception:
                invalid_count += 1
                logger.warning("[Warmstart] 跳过非法 arch_code: %r", arch_text)
                continue
            normalized = arch_to_text(parsed)
            if normalized in evaluated_arches or normalized in batch_seen:
                duplicate_count += 1
                continue
            batch_arches.append(normalized)
            batch_seen.add(normalized)

        consumed = len(batch_arches)
        logger.info(
            "[Warmstart] consumed=%d / external_total=%d, invalid=%d, duplicates=%d",
            consumed, external_total, invalid_count, duplicate_count,
        )

        # Partial random fill 如果 LLM 给的不够
        if consumed < target_count:
            shortfall = target_count - consumed
            logger.warning(
                "[Warmstart] LLM 提供 %d 合法种子，缺 %d，用 random 补齐",
                consumed, shortfall,
            )
            random_arches, random_dups = self._sample_unique_random_arches(
                shortfall, extra_seen=batch_seen,
            )
            batch_arches.extend(random_arches)
            duplicate_count += random_dups

        return batch_arches, duplicate_count

    def _sample_unique_random_arches(
        self,
        target_count: int,
        extra_seen: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], int]:
        evaluated_arches = _file_io().get_evaluated_arch_codes(self.exp_dir)
        seen = set(extra_seen or [])
        arches: List[str] = []
        duplicate_count = 0
        max_attempts = max(target_count * self.duplicate_retry_factor, target_count * 4)
        attempts = 0

        while len(arches) < target_count and attempts < max_attempts:
            arch_text = arch_to_text(sample_random_arch(self._rng, search_space=self.search_space))
            if arch_text in evaluated_arches or arch_text in seen:
                duplicate_count += 1
                attempts += 1
                continue
            arches.append(arch_text)
            seen.add(arch_text)
            attempts += 1
        return arches, duplicate_count

    def _evaluate_arch_batch(self, arch_codes: Sequence[str], generation: int) -> None:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            worker_fn = _evaluate_single_arch()
            assignments = assign_gpus_to_arches(arch_codes, self.gpu_devices)
            futures = {
                executor.submit(
                    worker_fn,
                    arch_code_str=arch_code,
                    epoch=generation,
                    exp_dir=self.exp_dir,
                    project_root=self.project_root,
                    cfg=self.cfg,
                    llm_client=None,
                    assigned_gpu=assigned_gpu,
                ): arch_code
                for arch_code, assigned_gpu in assignments
            }
            for future in as_completed(futures):
                arch_code = futures[future]
                try:
                    result = future.result()
                    if result is None:
                        logger.warning("baseline evaluation failed: %s", arch_code)
                except Exception:
                    logger.exception("baseline evaluation crashed: %s", arch_code)

    def _record_generation_metrics(
        self,
        generation: int,
        new_evaluated: int,
        duplicates: int,
        *,
        current_population: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Phase 1.3: 写入完整 generation_metrics 行（36 列含 HV / crowding /
        entropy / gap / stagnation 等）。"""
        file_io = _file_io()
        from efnas.search import search_metrics

        history = file_io.read_history(self.exp_dir)
        metrics_history = file_io.read_epoch_metrics(self.exp_dir)
        pop_arch_codes: List[str] = [
            str(row["arch_code"])
            for row in (current_population or [])
            if "arch_code" in row
        ]

        metrics = search_metrics.compute_full_generation_metrics(
            history_df=history,
            current_population_arch_codes=pop_arch_codes,
            metrics_history_df=metrics_history,
            epoch=generation,
            new_evaluated=new_evaluated,
            duplicates=duplicates,
            population_size=self.population_size,
            search_space_size=self.search_space_size,
        )
        file_io.append_epoch_metrics(
            self.exp_dir,
            metrics,
            columns=search_metrics.GENERATION_METRICS_COLUMNS,
        )

    def _write_pareto_snapshot(self) -> None:
        file_io = _file_io()
        history = file_io.read_history(self.exp_dir)
        if history.empty:
            return
        rows = history.to_dict("records")
        front_rows = select_next_population(rows, population_size=len(rows))
        front_objectives = [_row_to_objectives(row) for row in front_rows]
        first_front = fast_non_dominated_sort(front_objectives)[0]
        pareto_rows = [front_rows[idx] for idx in first_front]
        with self.pareto_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=file_io.HISTORY_COLUMNS)
            writer.writeheader()
            for row in sorted(pareto_rows, key=lambda item: float(item["fps"])):
                writer.writerow({column: row.get(column, "") for column in file_io.HISTORY_COLUMNS})

    def _load_or_init_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        state = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "next_generation": 0,
            "population_arches": [],
            "population_size": self.population_size,
            "total_generations": self.total_generations,
            "seed": self.seed,
        }
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state

    def _save_state(self, next_generation: int, population: Sequence[Dict[str, Any]]) -> None:
        state = {
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "next_generation": int(next_generation),
            "population_arches": [row["arch_code"] for row in population],
            "population_size": self.population_size,
            "total_generations": self.total_generations,
            "seed": self.seed,
        }
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rows_from_arch_codes(self, arch_codes: Sequence[str]) -> List[Dict[str, Any]]:
        if not arch_codes:
            return []
        history = _file_io().read_history(self.exp_dir)
        if history.empty:
            return []
        mapping = {str(row["arch_code"]): row for row in history.to_dict("records")}
        return [mapping[arch_code] for arch_code in arch_codes if arch_code in mapping]

    def _rows_for_generation(self, generation: int) -> List[Dict[str, Any]]:
        history = _file_io().read_history(self.exp_dir)
        if history.empty or "epoch" not in history.columns:
            return []
        try:
            generation_df = history[history["epoch"].astype(int) == int(generation)]
        except (TypeError, ValueError):
            return []
        return generation_df.to_dict("records")

    def _build_rank_lookup(self, fronts: Sequence[Sequence[int]]) -> Dict[int, int]:
        ranks: Dict[int, int] = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
        return ranks

    def _build_crowding_lookup(
        self,
        fronts: Sequence[Sequence[int]],
        objective_values: Sequence[Tuple[float, float]],
    ) -> Dict[int, float]:
        crowding_lookup: Dict[int, float] = {}
        for front in fronts:
            crowding_lookup.update(crowding_distance(front, objective_values))
        return crowding_lookup

    def _tournament_select_index(
        self,
        current_population: Sequence[Dict[str, Any]],
        ranks: Dict[int, int],
        crowding_lookup: Dict[int, float],
    ) -> int:
        """K-way tournament selection.

        Phase 4: K = ``self._tournament_size`` (默认 2). 较大 K 增加选择压力.
        """
        n = len(current_population)
        if n == 1:
            return 0
        size = max(2, min(int(self._tournament_size), n))
        contestants = self._rng.sample(list(range(n)), size)
        best = contestants[0]
        for idx in contestants[1:]:
            rank_idx = ranks[idx]
            rank_best = ranks[best]
            if rank_idx < rank_best:
                best = idx
            elif rank_idx == rank_best:
                if crowding_lookup.get(idx, 0.0) > crowding_lookup.get(best, 0.0):
                    best = idx
        return best
