"""Evaluation pool builder for Supernet V2."""

import argparse
import json
import random
from typing import Dict, List

from efnas.nas.search_space_v2 import V2_REFERENCE_ARCH_CODE, get_block_specs, get_num_blocks, get_num_choices


def _seed_codes() -> List[List[int]]:
    """Build readable seed codes with mixed-cardinality validity."""
    block_specs = get_block_specs()
    num_blocks = get_num_blocks()
    codes: List[List[int]] = []
    codes.append([0 for _ in range(num_blocks)])
    codes.append([1 if int(spec["num_choices"]) >= 2 else 0 for spec in block_specs])
    codes.append([min(2, int(spec["num_choices"]) - 1) for spec in block_specs])
    codes.append([idx % int(spec["num_choices"]) for idx, spec in enumerate(block_specs)])
    codes.append([((idx + 1) % int(spec["num_choices"])) for idx, spec in enumerate(block_specs)])
    codes.append([((int(spec["num_choices"]) - 1) - (idx % int(spec["num_choices"]))) for idx, spec in enumerate(block_specs)])
    codes.append([int(item) for item in V2_REFERENCE_ARCH_CODE])
    deduped: List[List[int]] = []
    for code in codes:
        if code not in deduped:
            deduped.append(code)
    return deduped


def check_eval_pool_coverage(pool: List[List[int]]) -> Dict[str, object]:
    """Check whether pool covers all valid options of all blocks."""
    counts: Dict[str, Dict[str, int]] = {}
    missing = []
    for block_idx in range(get_num_blocks()):
        counts[str(block_idx)] = {}
        for option in range(get_num_choices(block_idx)):
            counts[str(block_idx)][str(option)] = 0
    for arch_code in pool:
        for block_idx, option in enumerate(arch_code):
            counts[str(block_idx)][str(int(option))] += 1
    for block_idx in range(get_num_blocks()):
        for option in range(get_num_choices(block_idx)):
            if counts[str(block_idx)][str(option)] <= 0:
                missing.append({"block": int(block_idx), "option": int(option)})
    return {"ok": len(missing) == 0, "counts": counts, "missing": missing}


def build_eval_pool(seed: int, size: int) -> List[List[int]]:
    """Build fixed-size mixed-cardinality evaluation pool."""
    if int(size) < 3:
        raise ValueError("eval pool size must be >= 3")
    rng = random.Random(seed)
    pool: List[List[int]] = []
    for code in _seed_codes():
        if len(pool) >= int(size):
            break
        pool.append(code)
    while len(pool) < int(size):
        candidate = [int(rng.randrange(get_num_choices(block_idx))) for block_idx in range(get_num_blocks())]
        if candidate in pool:
            continue
        pool.append(candidate)
    return pool


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="eval pool builder for EdgeFlowNAS V2")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--size", type=int, default=12, help="eval pool size")
    parser.add_argument("--check", action="store_true", help="run coverage check")
    return parser


def main() -> int:
    """Run command-line entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    pool = build_eval_pool(seed=args.seed, size=args.size)
    payload: Dict[str, object] = {"pool_size": len(pool), "pool": pool}
    if args.check:
        payload["coverage"] = check_eval_pool_coverage(pool=pool)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.check and not payload["coverage"]["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
