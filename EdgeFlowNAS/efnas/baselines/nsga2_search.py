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
) -> List[int]:
    """Mutate one architecture with per-gene categorical mutation."""
    space = search_space or load_search_space()
    child = [int(value) for value in arch_code]
    for block_idx in range(space.num_blocks()):
        if rng.random() >= float(mutation_prob):
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

    def __init__(self, cfg: Dict[str, Any], exp_dir: str, project_root: str) -> None:
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
                candidate_arches, duplicate_count = self._sample_unique_random_arches(needed)
            else:
                candidate_arches, duplicate_count = self._generate_offspring_arches(current_population, needed)
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

        self._record_generation_metrics(generation=generation, new_evaluated=len(generation_rows), duplicates=duplicate_count)
        self._write_pareto_snapshot()
        logger.info(
            "generation=%d evaluated=%d duplicates=%d next_population=%d",
            generation,
            len(generation_rows),
            duplicate_count,
            len(next_population),
        )
        return next_population

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
                mutate_arch(child_a, rng=self._rng, mutation_prob=self.mutation_prob, search_space=self.search_space),
                mutate_arch(child_b, rng=self._rng, mutation_prob=self.mutation_prob, search_space=self.search_space),
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

    def _record_generation_metrics(self, generation: int, new_evaluated: int, duplicates: int) -> None:
        file_io = _file_io()
        history = file_io.read_history(self.exp_dir)
        total = len(history)
        best_epe = ""
        pareto_count = 0
        if not history.empty:
            try:
                epe_values = history["epe"].astype(float)
                fps_values = history["fps"].astype(float)
                best_epe = round(float(epe_values.min()), 6)
                pareto_count = len(fast_non_dominated_sort(list(zip(epe_values.tolist(), (-fps_values).tolist())))[0])
            except (TypeError, ValueError):
                best_epe = ""
                pareto_count = 0
        metrics = {
            "epoch": generation,
            "total_evaluated": total,
            "new_evaluated": new_evaluated,
            "duplicates": duplicates,
            "best_epe": best_epe,
            "pareto_count": pareto_count,
            "findings_count": 0,
            "assumptions_count": 0,
            "coverage_pct": round(total / max(1, self.search_space_size) * 100, 2),
        }
        file_io.append_epoch_metrics(self.exp_dir, metrics)

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
        if len(current_population) == 1:
            return 0
        idx_a, idx_b = self._rng.sample(list(range(len(current_population))), 2)
        rank_a = ranks[idx_a]
        rank_b = ranks[idx_b]
        if rank_a < rank_b:
            return idx_a
        if rank_b < rank_a:
            return idx_b
        crowding_a = crowding_lookup.get(idx_a, 0.0)
        crowding_b = crowding_lookup.get(idx_b, 0.0)
        if crowding_a > crowding_b:
            return idx_a
        if crowding_b > crowding_a:
            return idx_b
        return idx_a if self._rng.random() < 0.5 else idx_b
