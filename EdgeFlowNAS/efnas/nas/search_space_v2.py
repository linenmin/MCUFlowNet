"""Supernet V2 search-space specification."""

from typing import Dict, List

V2_BLOCK_SPECS: List[Dict[str, object]] = [
    {"name": "E0", "group": "stem", "num_choices": 3, "labels": ["7x7Conv", "5x5Conv", "3x3Conv"]},
    {"name": "E1", "group": "stem", "num_choices": 3, "labels": ["5x5Conv", "3x3Conv", "3x3Stride2DilatedConv"]},
    {"name": "EB0", "group": "backbone", "num_choices": 3, "labels": ["Deep1", "Deep2", "Deep3"]},
    {"name": "EB1", "group": "backbone", "num_choices": 3, "labels": ["Deep1", "Deep2", "Deep3"]},
    {"name": "DB0", "group": "backbone", "num_choices": 3, "labels": ["Deep1", "Deep2", "Deep3"]},
    {"name": "DB1", "group": "backbone", "num_choices": 3, "labels": ["Deep1", "Deep2", "Deep3"]},
    {"name": "H0Out", "group": "head", "num_choices": 2, "labels": ["3x3Conv", "5x5Conv"]},
    {"name": "H1", "group": "head", "num_choices": 2, "labels": ["3x3Conv", "5x5Conv"]},
    {"name": "H1Out", "group": "head", "num_choices": 2, "labels": ["3x3Conv", "5x5Conv"]},
    {"name": "H2", "group": "head", "num_choices": 2, "labels": ["3x3Conv", "5x5Conv"]},
    {"name": "H2Out", "group": "head", "num_choices": 2, "labels": ["3x3Conv", "5x5Conv"]},
]

V2_REFERENCE_ARCH_CODE: List[int] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def get_block_specs() -> List[Dict[str, object]]:
    """Return block specs for V2 search space."""
    return list(V2_BLOCK_SPECS)


def get_num_blocks() -> int:
    """Return total block count."""
    return int(len(V2_BLOCK_SPECS))


def get_max_choice_count() -> int:
    """Return maximum choice count across blocks."""
    return max(int(spec["num_choices"]) for spec in V2_BLOCK_SPECS)


def get_num_choices(block_idx: int) -> int:
    """Return number of valid choices for one block."""
    return int(V2_BLOCK_SPECS[int(block_idx)]["num_choices"])


def get_block_name(block_idx: int) -> str:
    """Return readable block name."""
    return str(V2_BLOCK_SPECS[int(block_idx)]["name"])


def validate_arch_code(arch_code: List[int]) -> None:
    """Validate one V2 architecture code."""
    if len(arch_code) != get_num_blocks():
        raise ValueError(f"length={get_num_blocks()} required")
    for block_idx, value in enumerate(arch_code):
        num_choices = get_num_choices(block_idx=block_idx)
        if int(value) < 0 or int(value) >= num_choices:
            raise ValueError(
                f"value out of range at idx={block_idx} name={get_block_name(block_idx)} "
                f"value={value} valid=[0,{num_choices - 1}]"
            )

