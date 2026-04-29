"""Launch the ten distill-or-not V3 scratch retrain candidates across GPUs."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path(_PROJECT_ROOT) / path


def load_candidates(csv_path: Path) -> List[Dict[str, object]]:
    """Load candidate rows from top10.csv."""
    rows: List[Dict[str, object]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            candidate_id = str(row.get("candidate_id") or row.get("id") or f"T{idx:02d}").strip()
            arch_code = str(row.get("arch_code", "")).strip()
            if not arch_code:
                raise ValueError(f"missing arch_code in row {idx}: {row}")
            distill_rank = row.get("distill_rank", row.get("distill_front_rank", ""))
            no_distill_rank = row.get("no_distill_rank", row.get("no_rank", row.get("no_distill_front_rank", "")))
            rank_gap = row.get("rank_gap", row.get("front_rank_gap", row.get("rank_score_gap", "")))
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "arch_code": arch_code,
                    "distill_rank": int(float(distill_rank)) if distill_rank not in (None, "") else "",
                    "no_distill_rank": int(float(no_distill_rank)) if no_distill_rank not in (None, "") else "",
                    "rank_gap": int(float(rank_gap)) if rank_gap not in (None, "") else "",
                }
            )
    return rows


def assign_gpu(index: int, gpu_devices: Sequence[str]) -> str:
    if not gpu_devices:
        raise ValueError("gpu_devices must not be empty")
    return str(gpu_devices[int(index) % len(gpu_devices)])


def _write_status(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = ["candidate_id", "arch_code", "gpu", "status", "returncode", "started_at", "finished_at", "log_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch V3 distill-or-not FC2 scratch probes")
    parser.add_argument("--config", default="configs/distill_or_not_fc2_short.yaml")
    parser.add_argument("--candidates_csv", default="outputs/nsga2_v3/frontier_top5_rank_gap_probe_20260430/top10.csv")
    parser.add_argument("--experiment_name", default="distill_or_not_fc2_short_run1")
    parser.add_argument("--gpu_devices", default="0,1,2,3,4")
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--base_path", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--fc2_num_workers", type=int, default=None)
    parser.add_argument("--fc2_eval_num_workers", type=int, default=None)
    parser.add_argument("--prefetch_batches", type=int, default=None)
    parser.add_argument("--eval_prefetch_batches", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--grad_clip_global_norm", type=float, default=None)
    parser.add_argument("--sintel_eval_every_epoch", type=int, default=None)
    parser.add_argument("--fc2_eval_every_epoch", type=int, default=None)
    parser.add_argument("--sintel_max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_ckpt_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _append_optional(cmd: List[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _child_command(args, candidate: Dict[str, object]) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(_PROJECT_ROOT) / "wrappers" / "run_distill_or_not_fc2_one.py"),
        "--config",
        str(args.config),
        "--experiment_name",
        str(args.experiment_name),
        "--model_name",
        str(candidate["candidate_id"]),
        "--arch_code",
        str(candidate["arch_code"]),
        "--gpu_device",
        "0",
    ]
    for attr, flag in [
        ("base_path", "--base_path"),
        ("train_dir", "--train_dir"),
        ("val_dir", "--val_dir"),
        ("fc2_num_workers", "--fc2_num_workers"),
        ("fc2_eval_num_workers", "--fc2_eval_num_workers"),
        ("prefetch_batches", "--prefetch_batches"),
        ("eval_prefetch_batches", "--eval_prefetch_batches"),
        ("num_epochs", "--num_epochs"),
        ("batch_size", "--batch_size"),
        ("micro_batch_size", "--micro_batch_size"),
        ("lr", "--lr"),
        ("lr_min", "--lr_min"),
        ("grad_clip_global_norm", "--grad_clip_global_norm"),
        ("sintel_eval_every_epoch", "--sintel_eval_every_epoch"),
        ("fc2_eval_every_epoch", "--fc2_eval_every_epoch"),
        ("sintel_max_samples", "--sintel_max_samples"),
        ("seed", "--seed"),
    ]:
        _append_optional(cmd, flag, getattr(args, attr))
    if args.resume:
        cmd.append("--resume")
        cmd.extend(["--resume_experiment_name", str(args.experiment_name)])
    _append_optional(cmd, "--resume_ckpt_name", args.resume_ckpt_name)
    return cmd


def main() -> int:
    args = _build_parser().parse_args()
    candidates = load_candidates(_project_path(args.candidates_csv))
    gpu_devices = [item.strip() for item in str(args.gpu_devices).split(",") if item.strip()]
    max_workers = min(max(1, int(args.max_workers)), len(gpu_devices), len(candidates))
    run_dir = Path(_PROJECT_ROOT) / "outputs" / "distill_or_not_fc2_short" / str(args.experiment_name)
    logs_dir = run_dir / "launcher_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": str(args.experiment_name),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "candidates_csv": str(_project_path(args.candidates_csv)),
        "gpu_devices": gpu_devices,
        "max_workers": max_workers,
        "candidates": candidates,
    }
    (run_dir / "launcher_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        for idx, candidate in enumerate(candidates):
            print(f"GPU {assign_gpu(idx, gpu_devices)}:", " ".join(_child_command(args, candidate)))
        return 0

    queue = list(enumerate(candidates))
    active: List[Dict[str, object]] = []
    statuses: List[Dict[str, object]] = []
    status_path = run_dir / "launcher_status.csv"
    while queue or active:
        while queue and len(active) < max_workers:
            idx, candidate = queue.pop(0)
            gpu = assign_gpu(idx, gpu_devices)
            cmd = _child_command(args, candidate)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = logs_dir / f"{candidate['candidate_id']}.log"
            handle = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=_PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            row = {
                "candidate_id": candidate["candidate_id"],
                "arch_code": candidate["arch_code"],
                "gpu": gpu,
                "status": "running",
                "returncode": "",
                "started_at": datetime.utcnow().isoformat() + "Z",
                "finished_at": "",
                "log_path": str(log_path),
            }
            active.append({"proc": proc, "handle": handle, "row": row})
            statuses.append(row)
            print(f"[launcher] started {candidate['candidate_id']} on GPU {gpu}")
            _write_status(status_path, statuses)

        time.sleep(5)
        still_active: List[Dict[str, object]] = []
        for item in active:
            proc = item["proc"]
            returncode = proc.poll()
            if returncode is None:
                still_active.append(item)
                continue
            item["handle"].close()
            row = item["row"]
            row["returncode"] = int(returncode)
            row["status"] = "done" if int(returncode) == 0 else "failed"
            row["finished_at"] = datetime.utcnow().isoformat() + "Z"
            print(f"[launcher] {row['status']} {row['candidate_id']} returncode={returncode}")
            _write_status(status_path, statuses)
        active = still_active

    failed = [row for row in statuses if row.get("status") != "done"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
