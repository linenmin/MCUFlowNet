"""Shared sys.path bootstrap for MCUFlowNet multi-project imports."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union


def resolve_project_paths(anchor_file: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """Resolve common project paths from an anchor file inside EdgeFlowNAS."""
    anchor = Path(anchor_file) if anchor_file is not None else Path(__file__)
    anchor = anchor.resolve()

    mcu_root = anchor.parents[3]
    edgeflownas_root = mcu_root / "EdgeFlowNAS"
    edgeflownet_root = mcu_root / "EdgeFlowNet"
    return {
        "mcu_root": mcu_root,
        "edgeflownas_root": edgeflownas_root,
        "edgeflownas_code_root": edgeflownas_root / "code",
        "edgeflownet_root": edgeflownet_root,
        "edgeflownet_code_root": edgeflownet_root / "code",
    }


def bootstrap_project_paths(anchor_file: Optional[Union[str, Path]] = None) -> List[str]:
    """Ensure shared MCUFlowNet import roots exist in sys.path and return them."""
    paths = resolve_project_paths(anchor_file=anchor_file)
    ordered = [
        paths["mcu_root"],
        paths["edgeflownet_root"],
        paths["edgeflownet_code_root"],
        paths["edgeflownas_root"],
        paths["edgeflownas_code_root"],
    ]
    for path in reversed(ordered):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return [str(path) for path in ordered]
