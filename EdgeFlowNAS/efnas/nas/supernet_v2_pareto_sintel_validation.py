"""Validate whether FC2 Pareto and near-Pareto subnets stay near the edge on Sintel."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from efnas.app.train_supernet_app import _load_yaml, _merge_overrides
from efnas.data.dataloader_builder import build_fc2_provider
from efnas.engine.supernet_v2_evaluator import setup_supernet_v2_eval_model
from efnas.nas.supernet_v2_rank_consistency import (
    _evaluate_sintel_one_arch,
    _option_or_default,
    _parse_arch_text,
    _prepare_sintel_lists,
    _resolve_path,
    _restore_eval_checkpoint,
    _run_bn_recalibration,
    _to_arch_text,
)
from efnas.utils.path_utils import ensure_directory, project_root


def _load_history_archive(history_csv: str | Path) -> List[Dict[str, Any]]:
    """Load one history archive as typed dict rows."""
    path = Path(history_csv)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    typed: List[Dict[str, Any]] = []
    for row in rows:
        typed.append(
            {
                "arch_code": str(row["arch_code"]).strip().strip('"'),
                "epe": float(row["epe"]),
                "fps": float(row["fps"]),
                "epoch": int(float(row.get("epoch", 0) or 0)),
            }
        )
    return typed


def _compute_pareto_front(rows: Sequence[Dict[str, Any]], epe_key: str = "epe") -> List[Dict[str, Any]]:
    """Return non-dominated rows for minimize-EPE / maximize-FPS."""
    keep: List[Dict[str, Any]] = []
    for i, candidate in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if (
                float(other[epe_key]) <= float(candidate[epe_key])
                and float(other["fps"]) >= float(candidate["fps"])
                and (
                    float(other[epe_key]) < float(candidate[epe_key])
                    or float(other["fps"]) > float(candidate["fps"])
                )
            ):
                dominated = True
                break
        if not dominated:
            keep.append(dict(candidate))
    return sorted(keep, key=lambda item: (float(item["fps"]), float(item[epe_key]), str(item["arch_code"])))


def select_pareto_and_near_candidates(
    history_rows: Sequence[Dict[str, Any]],
    near_rel_gap: float = 0.05,
    max_near: int = 70,
) -> List[Dict[str, Any]]:
    """Select all current Pareto rows plus near-Pareto dominated rows."""
    pareto_rows = _compute_pareto_front(history_rows, epe_key="epe")
    pareto_map = {str(row["arch_code"]): dict(row) for row in pareto_rows}
    selected: List[Dict[str, Any]] = []
    for row in pareto_rows:
        enriched = dict(row)
        enriched["selection_type"] = "pareto"
        enriched["closest_pareto_arch"] = str(row["arch_code"])
        enriched["closest_epe_gap_rel"] = 0.0
        enriched["closest_fps_gap_rel"] = 0.0
        enriched["near_score"] = 0.0
        selected.append(enriched)

    near_candidates: List[Dict[str, Any]] = []
    for row in history_rows:
        arch_code = str(row["arch_code"])
        if arch_code in pareto_map:
            continue
        best_match: Optional[Dict[str, Any]] = None
        for pareto in pareto_rows:
            if not (
                float(pareto["epe"]) <= float(row["epe"])
                and float(pareto["fps"]) >= float(row["fps"])
                and (float(pareto["epe"]) < float(row["epe"]) or float(pareto["fps"]) > float(row["fps"]))
            ):
                continue
            epe_gap_rel = max(0.0, (float(row["epe"]) - float(pareto["epe"])) / max(float(pareto["epe"]), 1e-12))
            fps_gap_rel = max(0.0, (float(pareto["fps"]) - float(row["fps"])) / max(float(pareto["fps"]), 1e-12))
            if epe_gap_rel <= float(near_rel_gap) or fps_gap_rel <= float(near_rel_gap):
                score = min(epe_gap_rel, fps_gap_rel)
                candidate = {
                    **dict(row),
                    "selection_type": "near_pareto",
                    "closest_pareto_arch": str(pareto["arch_code"]),
                    "closest_epe_gap_rel": float(epe_gap_rel),
                    "closest_fps_gap_rel": float(fps_gap_rel),
                    "near_score": float(score),
                }
                if best_match is None or float(candidate["near_score"]) < float(best_match["near_score"]):
                    best_match = candidate
        if best_match is not None:
            near_candidates.append(best_match)

    near_candidates.sort(
        key=lambda item: (
            float(item["near_score"]),
            float(item["closest_epe_gap_rel"]),
            float(item["closest_fps_gap_rel"]),
            str(item["arch_code"]),
        )
    )
    selected.extend(near_candidates[: max(0, int(max_near))])
    return selected


def compute_sintel_retention_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize whether original FC2 Pareto points stay on the Sintel front."""
    sintel_front = _compute_pareto_front(records, epe_key="sintel_epe")
    front_arches = {str(row["arch_code"]) for row in sintel_front}
    original_pareto = [row for row in records if str(row["selection_type"]) == "pareto"]
    near_rows = [row for row in records if str(row["selection_type"]) == "near_pareto"]
    retained = sorted(str(row["arch_code"]) for row in original_pareto if str(row["arch_code"]) in front_arches)
    promoted = sorted(str(row["arch_code"]) for row in near_rows if str(row["arch_code"]) in front_arches)
    return {
        "num_selected": int(len(records)),
        "num_original_pareto": int(len(original_pareto)),
        "num_near_pareto": int(len(near_rows)),
        "num_sintel_front": int(len(sintel_front)),
        "original_pareto_retained_count": int(len(retained)),
        "original_pareto_retention_ratio": float(len(retained) / len(original_pareto)) if original_pareto else None,
        "near_pareto_promoted_count": int(len(promoted)),
        "retained_original_pareto_arches": retained,
        "promoted_near_pareto_arches": promoted,
        "sintel_front_arches": [str(row["arch_code"]) for row in sintel_front],
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write one list of dict rows to CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_summary_markdown(summary: Dict[str, Any], selected_rows: Sequence[Dict[str, Any]], records: Sequence[Dict[str, Any]]) -> str:
    """Render a concise markdown report."""
    retained = summary["retained_original_pareto_arches"]
    promoted = summary["promoted_near_pareto_arches"]
    lines = [
        "# V2 Pareto Sintel Validation",
        "",
        f"- selected candidates: {summary['num_selected']}",
        f"- original FC2 Pareto candidates: {summary['num_original_pareto']}",
        f"- near-Pareto candidates: {summary['num_near_pareto']}",
        f"- Sintel front size (within selected set): {summary['num_sintel_front']}",
        f"- retained original Pareto count: {summary['original_pareto_retained_count']}",
        f"- retained original Pareto ratio: {summary['original_pareto_retention_ratio']}",
        f"- promoted near-Pareto count: {summary['near_pareto_promoted_count']}",
        "",
        "## Retained Original Pareto Arches",
        "",
    ]
    if retained:
        for arch in retained:
            lines.append(f"- `{arch}`")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Promoted Near-Pareto Arches", ""])
    if promoted:
        for arch in promoted:
            lines.append(f"- `{arch}`")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Sintel Front Details", ""])
    records_map = {str(row["arch_code"]): row for row in records}
    for arch in summary["sintel_front_arches"]:
        row = records_map[arch]
        fc2_epe = row.get("fc2_epe", row.get("epe"))
        lines.append(
            f"- `{arch}` | type={row['selection_type']} fc2_epe={float(fc2_epe):.6f} "
            f"sintel_epe={float(row['sintel_epe']):.6f} fps={float(row['fps']):.6f}"
        )
    return "\n".join(lines) + "\n"


def run_pareto_sintel_validation(
    config_path: str,
    overrides: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Sintel validation for current Pareto and near-Pareto candidates."""
    config = _merge_overrides(_load_yaml(config_path), overrides)

    history_csv_text = str(options.get("history_csv", "")).strip()
    if not history_csv_text:
        raise RuntimeError("--history_csv is required")
    history_csv = _resolve_path(history_csv_text)
    if not history_csv.exists():
        raise FileNotFoundError(f"history_csv does not exist: {history_csv}")

    history_rows = _load_history_archive(history_csv)
    selected_rows = select_pareto_and_near_candidates(
        history_rows,
        near_rel_gap=float(_option_or_default(options, "near_rel_gap", 0.05)),
        max_near=int(_option_or_default(options, "max_near", 70)),
    )

    output_dir_raw = options.get("output_dir", "")
    output_dir_text = "" if output_dir_raw is None else str(output_dir_raw).strip()
    if output_dir_text:
        output_dir = _resolve_path(output_dir_text)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (project_root() / "outputs" / "search_v2" / f"pareto_sintel_validation_{timestamp}").resolve()

    if bool(options.get("dry_run", False)):
        return {
            "mode": "dry_run",
            "resolved_output_dir": str(output_dir),
            "history_csv": str(history_csv),
            "selected_count": int(len(selected_rows)),
            "selected_preview": selected_rows[:5],
        }

    experiment_dir_text = str(options.get("experiment_dir", "")).strip()
    dataset_root_text = str(options.get("dataset_root", "")).strip()
    if not experiment_dir_text:
        raise RuntimeError("--experiment_dir is required unless --dry_run is used")
    if not dataset_root_text:
        raise RuntimeError("--dataset_root is required unless --dry_run is used")

    experiment_dir = _resolve_path(experiment_dir_text)
    dataset_root = _resolve_path(dataset_root_text, base_path=config.get("data", {}).get("base_path", None))
    if not experiment_dir.exists():
        raise FileNotFoundError(f"experiment_dir does not exist: {experiment_dir}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    if "CUDA_VISIBLE_DEVICES" not in os.environ and options.get("gpu_device", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options["gpu_device"])

    metadata_dir = Path(ensure_directory(str(output_dir / "metadata")))
    selected_csv = metadata_dir / "selected_candidates.csv"
    records_csv = metadata_dir / "sintel_validation_records.csv"
    summary_json = metadata_dir / "sintel_validation_summary.json"
    summary_md = metadata_dir / "sintel_validation_summary.md"
    _write_csv(selected_csv, selected_rows)

    fc2_train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    img1_list, img2_list, flo_list, sintel_list_path = _prepare_sintel_lists(
        dataset_root=dataset_root,
        sintel_list_text=str(options.get("sintel_list", "EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt")),
    )
    sintel_patch_size = [int(item) for item in str(options.get("sintel_patch_size", "416,1024")).split(",")]
    if len(sintel_patch_size) != 2:
        raise ValueError("sintel_patch_size must be H,W")

    eval_model = setup_supernet_v2_eval_model(
        experiment_dir=experiment_dir,
        checkpoint_type=str(options.get("checkpoint_type", "best")),
        flow_channels=int(config.get("data", {}).get("flow_channels", 2)),
        allow_growth=True,
    )

    records: List[Dict[str, Any]] = []
    progress = tqdm(selected_rows, desc="ParetoSintel", unit="arch")
    try:
        for row in progress:
            arch_code = _parse_arch_text(str(row["arch_code"]))
            _restore_eval_checkpoint(eval_model)
            _run_bn_recalibration(
                eval_model=eval_model,
                train_provider=fc2_train_provider,
                arch_code=arch_code,
                num_batches=int(_option_or_default(options, "bn_recal_batches", 16)),
                batch_size=int(_option_or_default(options, "bn_recal_batch_size", config.get("train", {}).get("batch_size", 8))),
            )
            sintel_epe = _evaluate_sintel_one_arch(
                eval_model=eval_model,
                arch_code=arch_code,
                img1_list=img1_list,
                img2_list=img2_list,
                flo_list=flo_list,
                patch_size=sintel_patch_size,
                max_samples=_option_or_default(options, "max_sintel_samples", None),
            )
            record = dict(row)
            record["fc2_epe"] = float(row.get("fc2_epe", row.get("epe")))
            record["sintel_epe"] = float(sintel_epe)
            records.append(record)
            progress.set_postfix(type=row["selection_type"], sintel=f"{sintel_epe:.4f}")
    finally:
        eval_model["sess"].close()

    summary = compute_sintel_retention_summary(records)
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "history_csv": str(history_csv),
        "experiment_dir": str(experiment_dir),
        "checkpoint_type": str(options.get("checkpoint_type", "best")),
        "dataset_root": str(dataset_root),
        "sintel_list": sintel_list_path,
        "sintel_patch_size": sintel_patch_size,
        "near_rel_gap": float(_option_or_default(options, "near_rel_gap", 0.05)),
        "max_near": int(_option_or_default(options, "max_near", 70)),
        "bn_recal_batches": int(_option_or_default(options, "bn_recal_batches", 16)),
        "summary": summary,
    }
    _write_csv(records_csv, records)
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(_build_summary_markdown(summary, selected_rows, records), encoding="utf-8")

    return {
        "exit_code": 0,
        "output_dir": str(output_dir),
        "selected_csv": str(selected_csv),
        "records_csv": str(records_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "summary": summary,
    }


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser for direct module execution."""
    parser = argparse.ArgumentParser(description="validate FC2 Pareto and near-Pareto subnets on Sintel")
    parser.add_argument("--config", default="configs/supernet_fc2_172x224_v2.yaml")
    parser.add_argument("--history_csv", required=True)
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--sintel_list", default="EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt")
    parser.add_argument("--sintel_patch_size", default="416,1024")
    parser.add_argument("--max_sintel_samples", type=int, default=None)
    parser.add_argument("--bn_recal_batches", type=int, default=16)
    parser.add_argument("--bn_recal_batch_size", type=int, default=None)
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument("--near_rel_gap", type=float, default=0.05)
    parser.add_argument("--max_near", type=int, default=70)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    result = run_pareto_sintel_validation(
        config_path=args.config,
        overrides={},
        options={
            "history_csv": args.history_csv,
            "experiment_dir": args.experiment_dir,
            "checkpoint_type": args.checkpoint_type,
            "output_dir": args.output_dir,
            "dataset_root": args.dataset_root,
            "sintel_list": args.sintel_list,
            "sintel_patch_size": args.sintel_patch_size,
            "max_sintel_samples": args.max_sintel_samples,
            "bn_recal_batches": args.bn_recal_batches,
            "bn_recal_batch_size": args.bn_recal_batch_size,
            "gpu_device": args.gpu_device,
            "near_rel_gap": args.near_rel_gap,
            "max_near": args.max_near,
            "dry_run": args.dry_run,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
