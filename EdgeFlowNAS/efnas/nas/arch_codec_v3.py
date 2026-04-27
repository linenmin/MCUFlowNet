"""11-dimensional architecture codec for Supernet V3."""

import argparse
import json
import re
from typing import Dict, List

from efnas.nas.search_space_v3 import V3_BLOCK_SPECS, validate_arch_code


def decode_arch_code(arch_code: List[int]) -> Dict[str, Dict[str, str]]:
    """Decode one V3 arch code into grouped readable labels."""
    validate_arch_code(arch_code)
    grouped: Dict[str, Dict[str, str]] = {"stem": {}, "backbone": {}, "head": {}}
    for block_idx, spec in enumerate(V3_BLOCK_SPECS):
        group = str(spec["group"])
        name = str(spec["name"])
        labels = list(spec["labels"])
        grouped[group][name] = str(labels[int(arch_code[block_idx])])
    return grouped


def parse_arch_code_text(text: str) -> List[int]:
    """Parse comma or whitespace separated arch code text."""
    tokens = [token for token in re.split(r"[,\s]+", str(text).strip()) if token]
    return [int(token) for token in tokens]


def run_self_test() -> Dict[str, str]:
    """Run minimal self test."""
    example = [0, 2, 0, 1, 2, 1, 0, 1, 0, 1, 0]
    decoded = decode_arch_code(example)
    assert decoded["stem"]["E0"] == "3x3Conv"
    assert decoded["stem"]["E1"] == "3x3Stride2DilatedConv"
    assert decoded["backbone"]["DB0"] == "Deep3"
    assert decoded["head"]["H1"] == "5x5Conv"
    try:
        decode_arch_code([0] * 10)
    except ValueError as exc:
        assert "length=11 required" in str(exc)
    try:
        decode_arch_code([0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0])
    except ValueError as exc:
        assert "value out of range" in str(exc)
    return {"status": "ok"}


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="arch code parser for EdgeFlowNAS V3")
    parser.add_argument("--arch_code", default="", help="comma or whitespace separated 11-d arch code")
    parser.add_argument("--self_test", action="store_true", help="run internal self test")
    return parser


def main() -> int:
    """Run command-line entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False))
        return 0
    if not args.arch_code:
        parser.error("please provide --arch_code or --self_test")
    arch_code = parse_arch_code_text(args.arch_code)
    decoded = decode_arch_code(arch_code)
    print(json.dumps(decoded, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
