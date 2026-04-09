"""Diagnostic for comparing FC2 and Sintel rankings on inherited-weight V2 subnets."""

import csv
import itertools
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from efnas.app.train_supernet_app import _load_yaml, _merge_overrides
from efnas.data.dataloader_builder import build_fc2_provider
from efnas.data.transforms_180x240 import standardize_image_tensor
from efnas.engine.supernet_v2_evaluator import setup_supernet_v2_eval_model
from efnas.nas.arch_codec_v2 import decode_arch_code
from efnas.nas.search_space_v2 import V2_REFERENCE_ARCH_CODE, get_num_blocks, get_num_choices, validate_arch_code
from efnas.utils.import_bootstrap import bootstrap_project_paths
from efnas.utils.path_utils import ensure_directory, project_root

PROJECT_ROOT = project_root()
bootstrap_project_paths(anchor_file=__file__)


def _iter_all_arch_codes_v2() -> Iterable[List[int]]:
    """Iterate over the full mixed-cardinality V2 space."""
    per_block_choices = [range(get_num_choices(block_idx)) for block_idx in range(get_num_blocks())]
    for code in itertools.product(*per_block_choices):
        yield [int(item) for item in code]


def _to_arch_text(arch_code: Sequence[int]) -> str:
    """Convert arch code to canonical comma-separated text."""
    return ",".join(str(int(item)) for item in arch_code)


def _parse_arch_text(raw: str) -> List[int]:
    """Parse one comma-separated V2 arch code string."""
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    arch_code = [int(token) for token in tokens]
    validate_arch_code(arch_code)
    return arch_code


def compute_v2_complexity_score(arch_code: Sequence[int]) -> float:
    """Compute a monotonic proxy score used only for stratified diagnostic sampling."""
    validate_arch_code([int(item) for item in arch_code])
    score_maps = (
        {0: 2.0, 1: 1.0, 2: 0.0},
        {0: 2.0, 1: 0.0, 2: 1.0},
        {0: 0.0, 1: 1.0, 2: 2.0},
        {0: 0.0, 1: 1.0, 2: 2.0},
        {0: 0.0, 1: 1.0, 2: 2.0},
        {0: 0.0, 1: 1.0, 2: 2.0},
        {0: 0.0, 1: 1.0},
        {0: 0.0, 1: 1.0},
        {0: 0.0, 1: 1.0},
        {0: 0.0, 1: 1.0},
        {0: 0.0, 1: 1.0},
    )
    return float(sum(score_maps[idx][int(value)] for idx, value in enumerate(arch_code)))


def sample_probe_arch_pool_v2(num_arch_samples: int, seed: int = 42) -> List[List[int]]:
    """Sample a deterministic, complexity-stratified probe pool from the V2 space."""
    sample_size = int(num_arch_samples)
    if sample_size <= 0:
        raise ValueError("num_arch_samples must be > 0")

    all_scored = [
        {"arch_code": code, "complexity_score": compute_v2_complexity_score(code)}
        for code in _iter_all_arch_codes_v2()
    ]
    all_scored.sort(key=lambda item: (float(item["complexity_score"]), _to_arch_text(item["arch_code"])))
    if sample_size >= len(all_scored):
        return [list(item["arch_code"]) for item in all_scored]

    anchors = [
        all_scored[0]["arch_code"],
        all_scored[len(all_scored) // 4]["arch_code"],
        list(V2_REFERENCE_ARCH_CODE),
        all_scored[len(all_scored) // 2]["arch_code"],
        all_scored[(3 * len(all_scored)) // 4]["arch_code"],
        all_scored[-1]["arch_code"],
    ]

    picked: List[List[int]] = []
    seen = set()
    for arch_code in anchors:
        key = tuple(int(item) for item in arch_code)
        if key in seen:
            continue
        seen.add(key)
        picked.append([int(item) for item in arch_code])
        if len(picked) >= sample_size:
            return picked[:sample_size]

    remaining = [item for item in all_scored if tuple(item["arch_code"]) not in seen]
    rng = random.Random(int(seed))
    num_bins = min(5, max(3, sample_size - len(picked)))
    bins: List[List[List[int]]] = []
    for bin_idx in range(num_bins):
        start = int(round(len(remaining) * bin_idx / num_bins))
        end = int(round(len(remaining) * (bin_idx + 1) / num_bins))
        members = [list(item["arch_code"]) for item in remaining[start:end]]
        rng.shuffle(members)
        bins.append(members)

    while len(picked) < sample_size:
        progressed = False
        for members in bins:
            if not members:
                continue
            arch_code = members.pop(0)
            key = tuple(int(item) for item in arch_code)
            if key in seen:
                continue
            seen.add(key)
            picked.append([int(item) for item in arch_code])
            progressed = True
            if len(picked) >= sample_size:
                break
        if not progressed:
            break
    return picked[:sample_size]


def _pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Compute Pearson correlation with small-sample guards."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _kendall_tau(xs: Sequence[int], ys: Sequence[int]) -> Optional[float]:
    """Compute Kendall tau-a for small probe sets without extra dependencies."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    concordant = 0
    discordant = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = int(xs[i]) - int(xs[j])
            dy = int(ys[i]) - int(ys[j])
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return float((concordant - discordant) / float(total))


def compute_rank_consistency_summary(
    records: Sequence[Dict[str, Any]],
    top_ks: Sequence[int] = (5, 10),
) -> Dict[str, Any]:
    """Summarize FC2/Sintel rank agreement from per-arch metric rows."""
    valid_rows: List[Dict[str, Any]] = []
    for row in records:
        try:
            fc2_epe = float(row["fc2_epe"])
            sintel_epe = float(row["sintel_epe"])
        except Exception:
            continue
        if not np.isfinite(fc2_epe) or not np.isfinite(sintel_epe):
            continue
        merged = dict(row)
        merged["fc2_epe"] = fc2_epe
        merged["sintel_epe"] = sintel_epe
        valid_rows.append(merged)

    if not valid_rows:
        return {
            "num_records": 0,
            "num_valid_records": 0,
            "spearman": None,
            "kendall_tau": None,
            "topk_overlap": {},
            "largest_rank_shift": None,
            "records": [],
        }

    fc2_sorted = sorted(valid_rows, key=lambda item: (float(item["fc2_epe"]), str(item["arch_code"])))
    sintel_sorted = sorted(valid_rows, key=lambda item: (float(item["sintel_epe"]), str(item["arch_code"])))
    fc2_rank = {str(item["arch_code"]): rank for rank, item in enumerate(fc2_sorted, start=1)}
    sintel_rank = {str(item["arch_code"]): rank for rank, item in enumerate(sintel_sorted, start=1)}

    enriched: List[Dict[str, Any]] = []
    for item in valid_rows:
        arch_code = str(item["arch_code"])
        row = dict(item)
        row["fc2_rank"] = int(fc2_rank[arch_code])
        row["sintel_rank"] = int(sintel_rank[arch_code])
        row["rank_delta"] = int(row["sintel_rank"] - row["fc2_rank"])
        row["rank_delta_abs"] = int(abs(row["rank_delta"]))
        enriched.append(row)

    aligned = sorted(enriched, key=lambda item: int(item["fc2_rank"]))
    fc2_ranks = [int(item["fc2_rank"]) for item in aligned]
    sintel_ranks = [int(item["sintel_rank"]) for item in aligned]
    spearman = _pearson_corr(fc2_ranks, sintel_ranks)
    kendall_tau = _kendall_tau(fc2_ranks, sintel_ranks)

    topk_overlap: Dict[str, Dict[str, Any]] = {}
    for top_k in top_ks:
        k = max(1, min(int(top_k), len(aligned)))
        fc2_top = {str(item["arch_code"]) for item in fc2_sorted[:k]}
        sintel_top = {str(item["arch_code"]) for item in sintel_sorted[:k]}
        topk_overlap[str(k)] = {
            "k": int(k),
            "overlap": int(len(fc2_top & sintel_top)),
            "overlap_ratio": float(len(fc2_top & sintel_top) / float(k)),
        }

    largest_rank_shift = sorted(
        aligned,
        key=lambda item: (-int(item["rank_delta_abs"]), int(item["fc2_rank"]), str(item["arch_code"])),
    )[0]

    return {
        "num_records": int(len(records)),
        "num_valid_records": int(len(valid_rows)),
        "spearman": None if spearman is None else float(spearman),
        "kendall_tau": None if kendall_tau is None else float(kendall_tau),
        "topk_overlap": topk_overlap,
        "largest_rank_shift": {
            "arch_code": str(largest_rank_shift["arch_code"]),
            "fc2_rank": int(largest_rank_shift["fc2_rank"]),
            "sintel_rank": int(largest_rank_shift["sintel_rank"]),
            "rank_delta": int(largest_rank_shift["rank_delta"]),
            "rank_delta_abs": int(largest_rank_shift["rank_delta_abs"]),
            "fc2_epe": float(largest_rank_shift["fc2_epe"]),
            "sintel_epe": float(largest_rank_shift["sintel_epe"]),
        },
        "records": aligned,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write dict rows to CSV with stable headers."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_summary_markdown(summary: Dict[str, Any]) -> str:
    """Render a concise markdown summary for quick review."""
    lines = [
        "# Supernet V2 Rank Consistency Summary",
        "",
        f"- num_records: {summary.get('num_records')}",
        f"- num_valid_records: {summary.get('num_valid_records')}",
        f"- spearman: {summary.get('spearman')}",
        f"- kendall_tau: {summary.get('kendall_tau')}",
        "",
        "## Top-K Overlap",
        "",
    ]
    for key, payload in summary.get("topk_overlap", {}).items():
        lines.append(
            f"- top-{key}: overlap={payload.get('overlap')} overlap_ratio={payload.get('overlap_ratio')}"
        )
    largest = summary.get("largest_rank_shift")
    if largest:
        lines.extend(
            [
                "",
                "## Largest Rank Shift",
                "",
                f"- arch_code: {largest.get('arch_code')}",
                f"- fc2_rank: {largest.get('fc2_rank')}",
                f"- sintel_rank: {largest.get('sintel_rank')}",
                f"- rank_delta: {largest.get('rank_delta')}",
                f"- fc2_epe: {largest.get('fc2_epe')}",
                f"- sintel_epe: {largest.get('sintel_epe')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _resolve_path(path_text: str, base_path: Optional[str] = None) -> Path:
    """Resolve one path from absolute text, base_path, or project root."""
    raw = Path(str(path_text))
    if raw.is_absolute():
        return raw
    if base_path:
        return (Path(str(base_path)) / raw).resolve()
    resolved = (project_root() / raw).resolve()
    if resolved.exists():
        return resolved
    if raw.parts and raw.parts[0] == "EdgeFlowNet":
        sibling = (project_root().parent / raw).resolve()
        if sibling.exists():
            return sibling
    return resolved


def _option_or_default(options: Dict[str, Any], key: str, default: Any) -> Any:
    """Return option value unless it is explicitly None, then use the provided default."""
    value = options.get(key, None)
    return default if value is None else value


def _prepare_sintel_lists(dataset_root: Path, sintel_list_text: str) -> Tuple[List[str], List[str], List[str], str]:
    """Resolve Sintel file lists in the same way as existing wrappers."""
    from argparse import Namespace

    from EdgeFlowNet.code.misc.utils import read_sintel_list

    list_path = _resolve_path(sintel_list_text)
    if not list_path.exists():
        raise FileNotFoundError(f"Sintel list not found: {list_path}")

    def strip_prefix(path_str: str) -> str:
        prefix = "Datasets/Sintel/"
        if path_str.startswith(prefix):
            return path_str[len(prefix) :]
        return path_str

    list_args = Namespace(data_list=str(list_path))
    rel_img1_list, rel_img2_list, rel_flo_list = read_sintel_list(list_args)
    img1_list = [str(dataset_root / strip_prefix(item)) for item in rel_img1_list]
    img2_list = [str(dataset_root / strip_prefix(item)) for item in rel_img2_list]
    flo_list = [str(dataset_root / strip_prefix(item)) for item in rel_flo_list]
    return img1_list, img2_list, flo_list, str(list_path)


def _restore_eval_checkpoint(eval_model: Dict[str, Any]) -> None:
    """Restore original checkpoint weights before a new arch evaluation."""
    eval_model["saver"].restore(eval_model["sess"], str(eval_model["checkpoint_path"]))


def _run_bn_recalibration(
    eval_model: Dict[str, Any],
    train_provider: Any,
    arch_code: Sequence[int],
    num_batches: int,
    batch_size: int,
) -> None:
    """Refresh BN moving stats with FC2 train batches for one fixed architecture."""
    batches = max(0, int(num_batches))
    if batches == 0 or not eval_model.get("update_ops"):
        return
    for _ in range(batches):
        input_batch, _, _, _ = train_provider.next_batch(batch_size=int(batch_size))
        input_batch = standardize_image_tensor(input_batch)
        eval_model["sess"].run(
            eval_model["update_ops"],
            feed_dict={
                eval_model["input_ph"]: input_batch,
                eval_model["arch_code_ph"]: [int(item) for item in arch_code],
                eval_model["is_training_ph"]: True,
            },
        )


def build_fc2_eval_windows(num_samples: int, batch_size: int, max_samples: Optional[int] = None) -> List[Tuple[int, int]]:
    """Build sequential FC2 eval windows that cover each chosen sample exactly once."""
    total = max(0, int(num_samples))
    bs = max(1, int(batch_size))
    if max_samples is not None:
        total = min(total, max(0, int(max_samples)))
    windows: List[Tuple[int, int]] = []
    offset = 0
    while offset < total:
        current_bs = min(bs, total - offset)
        windows.append((int(offset), int(current_bs)))
        offset += current_bs
    return windows


def _evaluate_fc2_one_arch(
    eval_model: Dict[str, Any],
    val_provider: Any,
    arch_code: Sequence[int],
    batch_size: int,
    max_samples: Optional[int] = None,
) -> float:
    """Evaluate one architecture on FC2 val with full sequential coverage."""
    windows = build_fc2_eval_windows(
        num_samples=len(val_provider),
        batch_size=int(batch_size),
        max_samples=max_samples,
    )
    epes: List[float] = []
    for offset, current_bs in windows:
        if hasattr(val_provider, "reset_cursor"):
            val_provider.reset_cursor(int(offset))
        input_batch, _, _, label_batch = val_provider.next_batch(batch_size=int(current_bs))
        input_batch = standardize_image_tensor(input_batch)
        epe_val = eval_model["sess"].run(
            eval_model["epe_tensor"],
            feed_dict={
                eval_model["input_ph"]: input_batch,
                eval_model["label_ph"]: label_batch,
                eval_model["arch_code_ph"]: [int(item) for item in arch_code],
                eval_model["is_training_ph"]: False,
            },
        )
        epes.append(float(epe_val))
    return float(np.mean(epes)) if epes else float("nan")


def _evaluate_sintel_one_arch(
    eval_model: Dict[str, Any],
    arch_code: Sequence[int],
    img1_list: Sequence[str],
    img2_list: Sequence[str],
    flo_list: Sequence[str],
    patch_size: Sequence[int],
    max_samples: Optional[int] = None,
) -> float:
    """Evaluate one architecture on Sintel using the inherited supernet weights."""
    from argparse import Namespace

    from EdgeFlowNet.code.misc.processor import FlowPostProcessor
    from EdgeFlowNet.code.misc.utils import get_sintel_batch

    processor = FlowPostProcessor("full", is_multiscale=True)
    limit = len(img1_list) if max_samples is None else min(len(img1_list), max(1, int(max_samples)))
    dummy_args = Namespace(
        Display=False,
        ShiftedFlow=False,
        ResizeToHalf=False,
        ResizeCropStack=False,
        ResizeNearestCropStack=False,
        NumberOfHalves=0,
        ResizeCropStackBlur=False,
        OverlapCropStack=False,
        PatchDelta=0,
        uncertainity=False,
    )

    for idx in range(limit):
        input_comb, gt_flow = get_sintel_batch(img1_list[idx], img2_list[idx], flo_list[idx], list(patch_size))
        if input_comb is None or gt_flow is None:
            raise RuntimeError(f"Failed to load Sintel sample idx={idx}")
        input_batch = np.expand_dims(input_comb, axis=0)
        input_batch = standardize_image_tensor(input_batch)
        preds = eval_model["sess"].run(
            eval_model["preds_tensor"],
            feed_dict={
                eval_model["input_ph"]: input_batch,
                eval_model["arch_code_ph"]: [int(item) for item in arch_code],
                eval_model["is_training_ph"]: False,
            },
        )
        flow_only = preds[:, :, :, :2]
        processor.update(label=gt_flow, prediction=flow_only, Args=dummy_args)

    if processor.counter == 0:
        return float("nan")
    return float(np.concatenate(processor.errorEPEs).mean())


def run_rank_consistency_diagnostic(
    config_path: str,
    overrides: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the FC2/Sintel ranking diagnostic or a dry-run preview."""
    config = _merge_overrides(_load_yaml(config_path), overrides)
    sample_pool = sample_probe_arch_pool_v2(
        num_arch_samples=int(_option_or_default(options, "num_arch_samples", 50)),
        seed=int(_option_or_default(options, "sample_seed", 42)),
    )

    output_dir_raw = options.get("output_dir", "")
    output_dir_text = "" if output_dir_raw is None else str(output_dir_raw).strip()
    if output_dir_text:
        output_dir = _resolve_path(output_dir_text)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (project_root() / "outputs" / "search_v2" / f"rank_consistency_{timestamp}").resolve()

    if bool(options.get("dry_run", False)):
        return {
            "mode": "dry_run",
            "config": config,
            "resolved_output_dir": str(output_dir),
            "num_arch_samples": len(sample_pool),
            "sample_preview": [_to_arch_text(code) for code in sample_pool[:5]],
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

    metadata_dir = ensure_directory(str(output_dir / "metadata"))
    samples_csv = metadata_dir / "sampled_probe_arches.csv"
    samples_json = metadata_dir / "sampled_probe_arches.json"
    records_csv = metadata_dir / "rank_consistency_records.csv"
    summary_json = metadata_dir / "rank_consistency_summary.json"
    summary_md = metadata_dir / "rank_consistency_summary.md"

    sample_rows = []
    for sample_index, arch_code in enumerate(sample_pool):
        decoded = decode_arch_code(arch_code)
        sample_rows.append(
            {
                "sample_index": int(sample_index),
                "arch_code": _to_arch_text(arch_code),
                "complexity_score": float(compute_v2_complexity_score(arch_code)),
                "decoded_json": json.dumps(decoded, ensure_ascii=False),
            }
        )
    _write_csv(samples_csv, sample_rows)
    samples_json.write_text(json.dumps(sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fc2_train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    fc2_val_provider = build_fc2_provider(config=config, split="val", seed_offset=1, provider_mode="eval")

    if "CUDA_VISIBLE_DEVICES" not in os.environ and options.get("gpu_device", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options["gpu_device"])

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
    progress = tqdm(sample_pool, desc="RankConsistency", unit="arch")
    try:
        for sample_index, arch_code in enumerate(progress):
            _restore_eval_checkpoint(eval_model)
            _run_bn_recalibration(
                eval_model=eval_model,
                train_provider=fc2_train_provider,
                arch_code=arch_code,
                num_batches=int(_option_or_default(options, "bn_recal_batches", 16)),
                batch_size=int(
                    _option_or_default(options, "bn_recal_batch_size", config.get("train", {}).get("batch_size", 8))
                ),
            )
            fc2_epe = _evaluate_fc2_one_arch(
                eval_model=eval_model,
                val_provider=fc2_val_provider,
                arch_code=arch_code,
                batch_size=int(
                    _option_or_default(options, "eval_batch_size", config.get("train", {}).get("batch_size", 8))
                ),
                max_samples=_option_or_default(options, "max_fc2_val_samples", None),
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
            records.append(
                {
                    "sample_index": int(sample_index),
                    "arch_code": _to_arch_text(arch_code),
                    "complexity_score": float(compute_v2_complexity_score(arch_code)),
                    "fc2_epe": float(fc2_epe),
                    "sintel_epe": float(sintel_epe),
                }
            )
            progress.set_postfix(fc2=f"{fc2_epe:.4f}", sintel=f"{sintel_epe:.4f}")
    finally:
        eval_model["sess"].close()

    summary = compute_rank_consistency_summary(records=records, top_ks=(5, 10))
    records_with_rank = summary.get("records", records)
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "experiment_dir": str(experiment_dir),
        "checkpoint_type": str(options.get("checkpoint_type", "best")),
        "dataset_root": str(dataset_root),
        "sintel_list": sintel_list_path,
        "sintel_patch_size": sintel_patch_size,
        "bn_recal_batches": int(_option_or_default(options, "bn_recal_batches", 16)),
        "fc2_val_total_samples": int(len(fc2_val_provider)),
        "max_fc2_val_samples": _option_or_default(options, "max_fc2_val_samples", None),
        "sample_seed": int(_option_or_default(options, "sample_seed", 42)),
        "num_arch_samples": int(len(sample_pool)),
        "summary": summary,
    }
    _write_csv(records_csv, records_with_rank)
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(_build_summary_markdown(summary), encoding="utf-8")
    return {
        "exit_code": 0,
        "output_dir": str(output_dir),
        "records_csv": str(records_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "summary": summary,
    }
