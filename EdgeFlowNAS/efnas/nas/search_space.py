"""Supernet V3 search-space specification."""

from typing import Dict, List

V3_ARCH_SEMANTICS_VERSION = "supernet_v3_mixed_11d_light_to_heavy_fixed_bilinear_bneckeca_gate4x"

V3_BLOCK_SPECS: List[Dict[str, object]] = [
    {"name": "E0", "group": "stem", "num_choices": 3, "labels": ["3x3Conv", "5x5Conv", "7x7Conv"]},
    {"name": "E1", "group": "stem", "num_choices": 3, "labels": ["3x3Conv", "5x5Conv", "3x3Stride2DilatedConv"]},
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

V3_LIGHTEST_ARCH_CODE: List[int] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
V3_REFERENCE_ARCH_CODE: List[int] = [2, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def get_arch_semantics_version() -> str:
    """Return the V3 arch-code semantics version."""
    return V3_ARCH_SEMANTICS_VERSION


def get_block_specs() -> List[Dict[str, object]]:
    """Return block specs for V3 search space."""
    return list(V3_BLOCK_SPECS)


def get_num_blocks() -> int:
    """Return total block count."""
    return int(len(V3_BLOCK_SPECS))


def get_max_choice_count() -> int:
    """Return maximum choice count across blocks."""
    return max(int(spec["num_choices"]) for spec in V3_BLOCK_SPECS)


def get_num_choices(block_idx: int) -> int:
    """Return number of valid choices for one block."""
    return int(V3_BLOCK_SPECS[int(block_idx)]["num_choices"])


def get_block_name(block_idx: int) -> str:
    """Return readable block name."""
    return str(V3_BLOCK_SPECS[int(block_idx)]["name"])


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
