"""Pure helpers for retrain_v2 checkpoint resume state."""

from typing import Any, Dict, List, Optional, Tuple


def _reconcile_resume_progress(
    trainer_state: Optional[Dict[str, Any]],
    checkpoint_metas: List[Dict[str, Any]],
    prefer_checkpoint_meta: bool = False,
) -> Tuple[int, int]:
    if prefer_checkpoint_meta and checkpoint_metas:
        start_epoch = 1
        global_step = 0
    else:
        state = trainer_state or {}
        start_epoch = int(state.get("epoch", 0)) + 1 if state else 1
        global_step = int(state.get("global_step", 0)) if state else 0

    for meta in checkpoint_metas:
        meta_epoch = int(meta.get("epoch", 0))
        meta_global_step = int(meta.get("global_step", 0))
        if meta_epoch + 1 > start_epoch:
            start_epoch = meta_epoch + 1
        if meta_global_step > global_step:
            global_step = meta_global_step
    return start_epoch, global_step


def _trim_retrain_histories(
    eval_histories: Dict[str, List[Dict[str, Any]]],
    comparison_rows: List[Dict[str, Any]],
    max_epoch: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    cutoff = int(max_epoch)
    trimmed_histories = {
        name: [row for row in rows if int(row.get("epoch", 0)) <= cutoff]
        for name, rows in eval_histories.items()
    }
    trimmed_comparison = [row for row in comparison_rows if int(row.get("epoch", 0)) <= cutoff]
    return trimmed_histories, trimmed_comparison
