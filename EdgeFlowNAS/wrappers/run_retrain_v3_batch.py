"""Launch Retrain V3 candidates across multiple GPUs."""

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


FINAL_RETRAIN_V3_CANDIDATES: List[Dict[str, object]] = [
    {
        "model_name": "v3_acc",
        "arch_code": "0,1,2,2,2,2,0,0,0,0,1",
        "role": "strongest_predicted_accuracy",
    },
    {
        "model_name": "v3_efn_fps",
        "arch_code": "2,0,0,2,2,1,0,0,0,0,0",
        "role": "edgeflownet_fps_match",
    },
    {
        "model_name": "v3_light",
        "arch_code": "0,0,0,0,0,0,0,0,0,0,0",
        "role": "lightest_pareto_endpoint",
    },
]


def _project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path(_PROJECT_ROOT) / path


def load_candidates(csv_path: Path) -> List[Dict[str, object]]:
    resolved_path = _project_path(str(csv_path))
    default_path = _project_path("plan/retrain_v3/retrain_v3_candidates.csv")
    if not resolved_path.exists() and resolved_path == default_path:
        return [dict(item) for item in FINAL_RETRAIN_V3_CANDIDATES]
    rows: List[Dict[str, object]] = []
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            model_name = str(row.get("model_name") or row.get("candidate_id") or f"v3_{idx:02d}").strip()
            arch_code = str(row.get("arch_code", "")).strip()
            if not arch_code:
                raise ValueError(f"missing arch_code in row {idx}: {row}")
            rows.append({"model_name": model_name, "arch_code": arch_code, "role": str(row.get("role", "")).strip()})
    return rows


def assign_gpu(index: int, gpu_devices: Sequence[str]) -> str:
    if not gpu_devices:
        raise ValueError("gpu_devices must not be empty")
    return str(gpu_devices[int(index) % len(gpu_devices)])


def _write_status(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = ["model_name", "arch_code", "gpu", "status", "returncode", "started_at", "finished_at", "log_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Retrain V3 candidates across GPUs")
    parser.add_argument("--stage", choices=["fc2", "ft3d"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--candidates_csv", default="plan/retrain_v3/retrain_v3_candidates.csv")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--gpu_devices", default="0,1,2")
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--base_path", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--fc2_experiment_dir", default=None)
    parser.add_argument("--init_experiment_dir", default=None)
    parser.add_argument("--init_ckpt_name", default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--grad_clip_global_norm", type=float, default=None)
    parser.add_argument("--fc2_num_workers", type=int, default=None)
    parser.add_argument("--fc2_eval_num_workers", type=int, default=None)
    parser.add_argument("--ft3d_num_workers", type=int, default=None)
    parser.add_argument("--ft3d_eval_num_workers", type=int, default=None)
    parser.add_argument("--prefetch_batches", type=int, default=None)
    parser.add_argument("--eval_prefetch_batches", type=int, default=None)
    parser.add_argument("--fc2_eval_every_epoch", type=int, default=None)
    parser.add_argument("--ft3d_eval_every_epoch", type=int, default=None)
    parser.add_argument("--sintel_eval_every_epoch", type=int, default=None)
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
    wrapper = "run_retrain_v3_fc2.py" if args.stage == "fc2" else "run_retrain_v3_ft3d.py"
    config = args.config or ("configs/retrain_v3_fc2.yaml" if args.stage == "fc2" else "configs/retrain_v3_ft3d.yaml")
    cmd = [
        sys.executable,
        str(Path(_PROJECT_ROOT) / "wrappers" / wrapper),
        "--config",
        str(config),
        "--experiment_name",
        str(args.experiment_name),
        "--model_name",
        str(candidate["model_name"]),
        "--arch_code",
        str(candidate["arch_code"]),
        "--gpu_device",
        "0",
    ]
    common = [
        ("base_path", "--base_path"),
        ("train_dir", "--train_dir"),
        ("val_dir", "--val_dir"),
        ("num_epochs", "--num_epochs"),
        ("batch_size", "--batch_size"),
        ("micro_batch_size", "--micro_batch_size"),
        ("lr", "--lr"),
        ("lr_min", "--lr_min"),
        ("grad_clip_global_norm", "--grad_clip_global_norm"),
        ("prefetch_batches", "--prefetch_batches"),
        ("eval_prefetch_batches", "--eval_prefetch_batches"),
        ("sintel_eval_every_epoch", "--sintel_eval_every_epoch"),
        ("sintel_max_samples", "--sintel_max_samples"),
        ("seed", "--seed"),
    ]
    for attr, flag in common:
        _append_optional(cmd, flag, getattr(args, attr))
    if args.stage == "fc2":
        for attr, flag in [
            ("fc2_num_workers", "--fc2_num_workers"),
            ("fc2_eval_num_workers", "--fc2_eval_num_workers"),
            ("fc2_eval_every_epoch", "--fc2_eval_every_epoch"),
        ]:
            _append_optional(cmd, flag, getattr(args, attr))
    else:
        for attr, flag in [
            ("ft3d_num_workers", "--ft3d_num_workers"),
            ("ft3d_eval_num_workers", "--ft3d_eval_num_workers"),
            ("ft3d_eval_every_epoch", "--ft3d_eval_every_epoch"),
            ("fc2_experiment_dir", "--fc2_experiment_dir"),
            ("init_experiment_dir", "--init_experiment_dir"),
            ("init_ckpt_name", "--init_ckpt_name"),
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
    output_root = "outputs/retrain_v3_fc2" if args.stage == "fc2" else "outputs/retrain_v3_ft3d"
    run_dir = Path(_PROJECT_ROOT) / output_root / str(args.experiment_name)
    logs_dir = run_dir / "launcher_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": args.stage,
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
            log_path = logs_dir / f"{candidate['model_name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=_PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            row = {
                "model_name": candidate["model_name"],
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
            print(f"[launcher] started {candidate['model_name']} on GPU {gpu}")
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
            print(f"[launcher] {row['status']} {row['model_name']} returncode={returncode}")
            _write_status(status_path, statuses)
        active = still_active

    failed = [row for row in statuses if row.get("status") != "done"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
