"""Evaluate retrain_v2 checkpoints on Sintel."""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from efnas.engine.retrain_v2_evaluator import preprocess_eval_batch, setup_retrain_v2_eval_model
from efnas.utils.import_bootstrap import bootstrap_project_paths, resolve_project_paths

bootstrap_project_paths(anchor_file=__file__)
project_root = resolve_project_paths(anchor_file=__file__)["mcu_root"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrain_v2 experiment on Sintel")
    parser.add_argument("--experiment_dir", required=True, help="Retrain experiment directory containing model_* folders")
    parser.add_argument("--model_name", default=None, help="Optional single model name")
    parser.add_argument("--ckpt_name", default="best", choices=["best", "last"], help="Checkpoint selection")
    parser.add_argument("--dataset_root", required=True, help="Sintel dataset root")
    parser.add_argument("--sintel_list", default="EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt", help="Sintel split list")
    parser.add_argument("--patch_size", default="416,1024", help="Eval resolution H,W")
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU device")
    parser.add_argument("--output_csv", default=None, help="Optional explicit output CSV path")
    parser.add_argument("--Display", action="store_true")
    parser.add_argument("--ShiftedFlow", action="store_true")
    parser.add_argument("--ResizeToHalf", action="store_true")
    parser.add_argument("--ResizeCropStack", action="store_true")
    parser.add_argument("--ResizeNearestCropStack", action="store_true")
    parser.add_argument("--NumberOfHalves", type=int, default=0)
    parser.add_argument("--ResizeCropStackBlur", action="store_true")
    parser.add_argument("--OverlapCropStack", action="store_true")
    parser.add_argument("--PatchDelta", type=int, default=0)
    parser.add_argument("--uncertainity", action="store_true")
    return parser


def _strip_sintel_prefix(path_str: str) -> str:
    prefix = "Datasets/Sintel/"
    return path_str[len(prefix) :] if path_str.startswith(prefix) else path_str


def main() -> int:
    from EdgeFlowNet.code.misc.processor import FlowPostProcessor
    from EdgeFlowNet.code.misc.utils import get_sintel_batch, read_sintel_list

    args = _build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    experiment_dir = Path(args.experiment_dir)
    patch_size = [int(x) for x in args.patch_size.split(",")]
    dataset_root = Path(args.dataset_root)
    sintel_list_path = project_root / args.sintel_list

    from argparse import Namespace

    list_args = Namespace(data_list=str(sintel_list_path))
    rel_img1_list, rel_img2_list, rel_flo_list = read_sintel_list(list_args)
    img1_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_img1_list]
    img2_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_img2_list]
    flo_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_flo_list]

    if args.model_name:
        model_dirs = [experiment_dir / f"model_{args.model_name}"]
    else:
        model_dirs = sorted(p for p in experiment_dir.iterdir() if p.is_dir() and p.name.startswith("model_"))
    if not model_dirs:
        raise FileNotFoundError(f"no model_* directories found under {experiment_dir}")

    output_csv = Path(args.output_csv) if args.output_csv else experiment_dir / f"sintel_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    for model_dir in model_dirs:
        sess, input_ph, preds_tensor, meta_data = setup_retrain_v2_eval_model(model_dir, tuple(patch_size), ckpt_name=args.ckpt_name)
        processor = FlowPostProcessor("full", is_multiscale=True)
        for idx in tqdm(range(len(img1_list)), desc=f"Evaluating {model_dir.name}", unit="sample"):
            input_comb, gt_flow = get_sintel_batch(img1_list[idx], img2_list[idx], flo_list[idx], patch_size)
            if input_comb is None or gt_flow is None:
                continue
            input_batch = preprocess_eval_batch(np.expand_dims(input_comb, axis=0))
            preds_results = sess.run(preds_tensor, feed_dict={input_ph: input_batch})
            processor.update(label=gt_flow, prediction=preds_results[:, :, :, :2], Args=args)
        sess.close()
        rows.append(
            {
                "model_name": meta_data["scope_name"],
                "arch_code": ",".join(str(v) for v in meta_data["arch_code"]),
                "checkpoint_path": meta_data["checkpoint_path"],
                "fc2_or_stage_metric": meta_data.get("metric", ""),
                "sintel_epe": getattr(processor, "MeanEPE", None),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_name", "arch_code", "checkpoint_path", "fc2_or_stage_metric", "sintel_epe"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(str(output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
