"""FairNAS A.1 irregular-space sampler for Supernet V2."""

import argparse
import json
import random
from typing import Dict, List

from efnas.nas.search_space_v2 import get_block_specs, get_max_choice_count, get_num_blocks, get_num_choices


def _expand_irregular_block_choices(rng: random.Random, num_choices: int, target_count: int) -> List[int]:
    """Expand one block into FairNAS A.1 temporary slots."""
    slots = list(range(int(num_choices)))
    while len(slots) < int(target_count):
        slots.append(int(rng.randrange(int(num_choices))))
    rng.shuffle(slots)
    return slots


def generate_fair_cycle(rng: random.Random) -> List[List[int]]:
    """Generate one approximated-SF cycle with 3 single-path models."""
    block_specs = get_block_specs()
    target_count = get_max_choice_count()
    block_slot_lists: List[List[int]] = []
    for block_idx, spec in enumerate(block_specs):
        slots = _expand_irregular_block_choices(
            rng=rng,
            num_choices=int(spec["num_choices"]),
            target_count=target_count,
        )
        block_slot_lists.append(slots)
    cycle_codes: List[List[int]] = []
    for path_idx in range(target_count):
        arch_code: List[int] = []
        for block_idx in range(len(block_slot_lists)):
            arch_code.append(int(block_slot_lists[block_idx][path_idx]))
        cycle_codes.append(arch_code)
    return cycle_codes


def _init_counts() -> Dict[str, Dict[str, int]]:
    """Initialize cumulative per-block counts."""
    counts: Dict[str, Dict[str, int]] = {}
    for block_idx in range(get_num_blocks()):
        counts[str(block_idx)] = {}
        for option in range(get_num_choices(block_idx)):
            counts[str(block_idx)][str(option)] = 0
    return counts


def run_cycles(cycles: int, seed: int) -> Dict[str, object]:
    """Run multiple approximated-SF cycles and collect counts."""
    rng = random.Random(seed)
    counts = _init_counts()
    records: List[List[List[int]]] = []
    for _ in range(int(cycles)):
        cycle_codes = generate_fair_cycle(rng=rng)
        records.append(cycle_codes)
        for arch_code in cycle_codes:
            for block_idx, option in enumerate(arch_code):
                counts[str(block_idx)][str(int(option))] += 1
    fairness_gap = 0
    for block_idx in range(get_num_blocks()):
        values = list(counts[str(block_idx)].values())
        fairness_gap = max(fairness_gap, max(values) - min(values))
    return {"counts": counts, "fairness_gap": int(fairness_gap), "records": records}


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="FairNAS A.1 sampler for EdgeFlowNAS V2")
    parser.add_argument("--cycles", type=int, default=1, help="number of approximated-SF cycles")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser


def main() -> int:
    """Run command-line entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    result = run_cycles(cycles=args.cycles, seed=args.seed)
    summary = {
        "cycles": int(args.cycles),
        "seed": int(args.seed),
        "paths_per_cycle": int(get_max_choice_count()),
        "fairness_gap": int(result["fairness_gap"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
