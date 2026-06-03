#!/usr/bin/env python3
"""Download the MCUFlowNet pretrained checkpoints from Google Drive into the repo.

Requires `gdown` (pip install gdown). Fetches the shared "MCUFlowNet_checkpoint" folder
and places each model's checkpoint files into the path the code expects.

Usage:
    pip install gdown
    python download_weights.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FOLDER_URL = "https://drive.google.com/drive/folders/1M598SgCXy6i3bcrOnD88zv5tp30RHeoF"
REPO = Path(__file__).resolve().parent

# Drive subfolder name -> checkpoint directory in the repo
TARGETS = {
    "MCUFlowNet-L": REPO / "EdgeFlowNAS/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1/model_v3_efn_fps/checkpoints",
    "MCUFlowNet-S": REPO / "EdgeFlowNAS/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1/model_v3_light/checkpoints",
    "MCUFlowNet-Supernet": REPO / "EdgeFlowNAS/outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill/checkpoints",
}


def main() -> int:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown is not installed. Install it with:\n    pip install gdown", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print(f"Downloading checkpoints from:\n  {FOLDER_URL}\n(this may take a while; the supernet file is ~193 MB)")
        try:
            subprocess.run(
                ["gdown", "--folder", "--remaining-ok", FOLDER_URL, "-O", str(tmp_dir)],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"\ngdown download failed: {exc}\n"
                  "If a large file was blocked by Google Drive's virus-scan page, download the\n"
                  "three model folders from the link above manually and extract them per the README.",
                  file=sys.stderr)
            return 1

        # gdown --folder writes the folder's contents into tmp_dir, usually under one subdir.
        candidates = [tmp_dir, *[p for p in tmp_dir.iterdir() if p.is_dir()]]
        placed_any = False
        for name, target in TARGETS.items():
            src = next((c / name for c in candidates if (c / name).is_dir()), None)
            if src is None:
                print(f"  ! could not find '{name}' in the download — skipping")
                continue
            target.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in src.glob("*.ckpt.*"):  # the TensorFlow checkpoint shards
                shutil.copy2(f, target / f.name)
                n += 1
            print(f"  {name}: placed {n} file(s) -> {target.relative_to(REPO).as_posix()}/")
            placed_any = placed_any or n > 0

    if not placed_any:
        print("\nNo checkpoint files were placed. Check the Google Drive link / your access.", file=sys.stderr)
        return 1
    print("\nDone. Pretrained checkpoints are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
