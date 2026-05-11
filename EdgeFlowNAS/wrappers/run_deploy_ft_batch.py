"""Launch deploy-resolution fine-tune candidates across multiple GPUs.

Reads `plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_candidates.csv`
(or any CSV passed via --candidates_csv), spawns one
`run_deploy_ft_one.py` process per candidate, distributes them across
the GPU devices in --gpu_devices, and tracks completion in
`launcher_status.csv`.

Each CSV row must carry at minimum:
    model_name, arch_family, init_mode, flow_divisor
Plus, depending on init_mode:
    init_mode=experiment_dir → init_experiment_dir + init_ckpt_name
    init_mode=explicit_path  → init_ckpt_path
And, for fixed_v3 candidates:
    arch_code   (e.g. "0,1,2,2,2,2,0,0,0,0,1")
"""
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
    p = Path(path_text)
    return p if p.is_absolute() else Path(_PROJECT_ROOT) / p


def load_candidates(csv_path: Path) -> List[Dict[str, str]]:
    resolved = _project_path(str(csv_path))
    if not resolved.exists():
        raise FileNotFoundError(f"candidates csv missing: {resolved}")
    rows: List[Dict[str, str]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            model_name = (row.get("model_name") or "").strip()
            if not model_name:
                raise ValueError(f"row {idx}: missing model_name")
            arch_family = (row.get("arch_family") or "fixed_v3").strip()
            init_mode = (row.get("init_mode") or "experiment_dir").strip()
            if arch_family not in ("fixed_v3", "edgeflownet_mainline"):
                raise ValueError(f"row {idx}: unsupported arch_family={arch_family}")
            if init_mode not in ("experiment_dir", "explicit_path"):
                raise ValueError(f"row {idx}: unsupported init_mode={init_mode}")
            rows.append({
                "model_name": model_name,
                "arch_family": arch_family,
                "arch_code": (row.get("arch_code") or "").strip(),
                "role": (row.get("role") or "").strip(),
                "init_mode": init_mode,
                "init_experiment_dir": (row.get("init_experiment_dir") or "").strip(),
                "init_ckpt_name": (row.get("init_ckpt_name") or "").strip(),
                "init_ckpt_path": (row.get("init_ckpt_path") or "").strip(),
                "flow_divisor": (row.get("flow_divisor") or "12.5").strip(),
            })
    return rows


def assign_gpu(index: int, gpu_devices: Sequence[str]) -> str:
    if not gpu_devices:
        raise ValueError("gpu_devices must not be empty")
    return str(gpu_devices[int(index) % len(gpu_devices)])


def _write_status(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = ["model_name", "arch_family", "gpu", "status", "returncode",
               "started_at", "finished_at", "log_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch deploy_ft candidates across GPUs")
    p.add_argument("--config", default="configs/retrain_v3_deploy_ft.yaml")
    p.add_argument(
        "--candidates_csv",
        default="plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_candidates.csv",
    )
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--gpu_devices", default="0,1,2,3")
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--micro_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr_schedule", default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--grad_clip_global_norm", type=float, default=None)
    p.add_argument("--input_height", type=int, default=None)
    p.add_argument("--input_width", type=int, default=None)
    p.add_argument("--ft3d_num_workers", type=int, default=None)
    p.add_argument("--ft3d_eval_num_workers", type=int, default=None)
    p.add_argument("--prefetch_batches", type=int, default=None)
    p.add_argument("--eval_prefetch_batches", type=int, default=None)
    p.add_argument("--ft3d_eval_every_epoch", type=int, default=None)
    p.add_argument("--sintel_eval_every_epoch", type=int, default=None)
    p.add_argument("--sintel_max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p


def _append(cmd: List[str], flag: str, value) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def _child_command(args, candidate: Dict[str, str]) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(_PROJECT_ROOT) / "wrappers" / "run_deploy_ft_one.py"),
        "--config", str(args.config),
        "--experiment_name", str(args.experiment_name),
        "--model_name", candidate["model_name"],
        "--arch_family", candidate["arch_family"],
        "--gpu_device", "0",  # CUDA_VISIBLE_DEVICES already pins per-process
    ]
    if candidate["arch_code"]:
        cmd.extend(["--arch_code", candidate["arch_code"]])
    cmd.extend(["--init_mode", candidate["init_mode"]])
    if candidate["init_mode"] == "experiment_dir":
        _append(cmd, "--init_experiment_dir", candidate["init_experiment_dir"])
        _append(cmd, "--init_ckpt_name", candidate["init_ckpt_name"])
    else:
        _append(cmd, "--init_ckpt_path", candidate["init_ckpt_path"])
    _append(cmd, "--flow_divisor", candidate["flow_divisor"])

    for attr, flag in [
        ("num_epochs", "--num_epochs"),
        ("batch_size", "--batch_size"),
        ("micro_batch_size", "--micro_batch_size"),
        ("lr", "--lr"),
        ("lr_schedule", "--lr_schedule"),
        ("early_stop_patience", "--early_stop_patience"),
        ("grad_clip_global_norm", "--grad_clip_global_norm"),
        ("input_height", "--input_height"),
        ("input_width", "--input_width"),
        ("ft3d_num_workers", "--ft3d_num_workers"),
        ("ft3d_eval_num_workers", "--ft3d_eval_num_workers"),
        ("prefetch_batches", "--prefetch_batches"),
        ("eval_prefetch_batches", "--eval_prefetch_batches"),
        ("ft3d_eval_every_epoch", "--ft3d_eval_every_epoch"),
        ("sintel_eval_every_epoch", "--sintel_eval_every_epoch"),
        ("sintel_max_samples", "--sintel_max_samples"),
        ("seed", "--seed"),
    ]:
        _append(cmd, flag, getattr(args, attr))
    return cmd


def main() -> int:
    args = _build_parser().parse_args()
    candidates = load_candidates(_project_path(args.candidates_csv))
    gpu_devices = [g.strip() for g in str(args.gpu_devices).split(",") if g.strip()]
    max_workers = min(max(1, int(args.max_workers)), len(gpu_devices), len(candidates))
    run_dir = Path(_PROJECT_ROOT) / "outputs" / "retrain_v3_deploy_ft" / str(args.experiment_name)
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
    (run_dir / "launcher_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        for idx, candidate in enumerate(candidates):
            gpu = assign_gpu(idx, gpu_devices)
            print(f"GPU {gpu}:", " ".join(_child_command(args, candidate)))
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
            proc = subprocess.Popen(
                cmd, cwd=_PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
            row = {
                "model_name": candidate["model_name"],
                "arch_family": candidate["arch_family"],
                "gpu": gpu,
                "status": "running",
                "returncode": "",
                "started_at": datetime.utcnow().isoformat() + "Z",
                "finished_at": "",
                "log_path": str(log_path),
            }
            active.append({"proc": proc, "handle": handle, "row": row})
            statuses.append(row)
            print(f"[launcher] started {candidate['model_name']} ({candidate['arch_family']}) on GPU {gpu}")
            _write_status(status_path, statuses)

        time.sleep(5)
        still_active: List[Dict[str, object]] = []
        for item in active:
            proc = item["proc"]
            rc = proc.poll()
            if rc is None:
                still_active.append(item)
                continue
            item["handle"].close()
            row = item["row"]
            row["returncode"] = int(rc)
            row["status"] = "done" if int(rc) == 0 else "failed"
            row["finished_at"] = datetime.utcnow().isoformat() + "Z"
            print(f"[launcher] {row['status']} {row['model_name']} returncode={rc}")
            _write_status(status_path, statuses)
        active = still_active

    failed = [r for r in statuses if r.get("status") != "done"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
