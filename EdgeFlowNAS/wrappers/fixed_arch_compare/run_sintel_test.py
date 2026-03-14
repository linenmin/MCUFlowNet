import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

project_root = Path(__file__).resolve().parent.parent.parent.parent
edgeflownet_dir = project_root / "EdgeFlowNet"
edgeflownas_dir = project_root / "EdgeFlowNAS"

edgeflownet_code_dir = edgeflownet_dir / "code"
edgeflownas_code_dir = edgeflownas_dir / "code"

if str(edgeflownet_dir) not in sys.path:
    sys.path.insert(0, str(edgeflownet_dir))
if str(edgeflownet_code_dir) not in sys.path:
    sys.path.insert(0, str(edgeflownet_code_dir))
if str(edgeflownas_dir) not in sys.path:
    sys.path.insert(0, str(edgeflownas_dir))
if str(edgeflownas_code_dir) not in sys.path:
    sys.path.insert(0, str(edgeflownas_code_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from efnas.engine.fixed_arch_compare_evaluator import setup_fixed_arch_eval_model, preprocess_eval_batch
from EdgeFlowNet.code.misc.processor import FlowPostProcessor
from EdgeFlowNet.code.misc.utils import get_sintel_batch, read_sintel_list


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fixed-arch compare checkpoints on Sintel")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--experiment_dir", type=str, help="Joint-training experiment dir containing model_* folders")
    src_group.add_argument("--model_dir", type=str, help="Single model dir, e.g. outputs/.../model_baseline")

    parser.add_argument("--model_name", type=str, default="", help="Optional model name inside experiment_dir")
    parser.add_argument("--variant", type=str, default="", help="Required only when model_dir has no run_manifest.json")
    parser.add_argument("--ckpt_name", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to Sintel root (Datasets/Sintel)")
    parser.add_argument(
        "--sintel_list",
        type=str,
        default="EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt",
        help="Relative path to Sintel split list",
    )
    parser.add_argument("--patch_size", type=str, default="416,1024", help="H,W")
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--output_json", type=str, default="", help="Optional explicit output json path")

    parser.add_argument("--Display", action="store_true", help="Display Output")
    parser.add_argument("--ShiftedFlow", action="store_true", help="Shift Flow for Overlap Crop Method")
    parser.add_argument("--ResizeToHalf", action="store_true", help="ResizeToHalf")
    parser.add_argument("--ResizeCropStack", action="store_true", help="ResizeCropStack")
    parser.add_argument("--ResizeNearestCropStack", action="store_true", help="ResizeNearestCropStack")
    parser.add_argument("--NumberOfHalves", type=int, default=0, help="ResizeCropStack halves")
    parser.add_argument("--ResizeCropStackBlur", action="store_true", help="OverlapCropStack")
    parser.add_argument("--OverlapCropStack", action="store_true", help="OverlapCropStack")
    parser.add_argument("--PatchDelta", type=int, default=0, help="OverlapCropStack padding")
    parser.add_argument("--uncertainity", action="store_true", help="Use only flow channels")
    return parser.parse_args()


def _strip_sintel_prefix(path_str):
    prefix = "Datasets/Sintel/"
    if path_str.startswith(prefix):
        return path_str[len(prefix) :]
    return path_str


def _load_manifest(experiment_dir: Path):
    manifest_path = experiment_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _discover_model_dirs(experiment_dir: Path, model_name: str):
    manifest = _load_manifest(experiment_dir)
    variant_map = manifest.get("model_variants", {})

    if model_name:
        model_dir = experiment_dir / f"model_{model_name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {model_dir}")
        return [model_dir]

    if isinstance(variant_map, dict) and variant_map:
        ordered = []
        for name in variant_map.keys():
            model_dir = experiment_dir / f"model_{name}"
            if model_dir.exists():
                ordered.append(model_dir)
        if ordered:
            return ordered

    return sorted([p for p in experiment_dir.glob("model_*") if p.is_dir()])


def _prepare_sintel_lists(dataset_root: Path, sintel_list_rel: str):
    raw_path = Path(sintel_list_rel)
    if raw_path.is_absolute():
        sintel_list_path = raw_path
    elif raw_path.exists():
        sintel_list_path = raw_path.resolve()
    else:
        sintel_list_path = (project_root / raw_path).resolve()
    if not sintel_list_path.exists():
        raise FileNotFoundError(f"Sintel list not found: {sintel_list_path}")

    from argparse import Namespace

    list_args = Namespace(data_list=str(sintel_list_path))
    rel_img1_list, rel_img2_list, rel_flo_list = read_sintel_list(list_args)

    img1_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_img1_list]
    img2_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_img2_list]
    flo_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_flo_list]
    return img1_list, img2_list, flo_list, str(sintel_list_path)


def _evaluate_single_model(model_dir: Path, args, patch_size, img1_list, img2_list, flo_list):
    sess, input_ph, preds_tensor, meta_data = setup_fixed_arch_eval_model(
        checkpoint_dir=model_dir,
        patch_size=patch_size,
        ckpt_name=args.ckpt_name,
        variant=(args.variant or None),
    )

    processor = FlowPostProcessor("full", is_multiscale=True)
    for i in tqdm(range(len(img1_list)), desc=f"Evaluating {model_dir.name}", unit="sample"):
        img1_path = img1_list[i]
        img2_path = img2_list[i]
        flo_path = flo_list[i]
        missing = [p for p in [img1_path, img2_path, flo_path] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "Sintel sample path missing.\n"
                f"sample_idx={i}\n"
                f"img1={img1_path}\n"
                f"img2={img2_path}\n"
                f"flo={flo_path}\n"
                f"missing={missing}"
            )

        input_comb, gt_flow = get_sintel_batch(img1_list[i], img2_list[i], flo_list[i], patch_size)
        if input_comb is None or gt_flow is None:
            raise RuntimeError(
                "Failed to decode Sintel sample.\n"
                f"sample_idx={i}\n"
                f"img1={img1_path}\n"
                f"img2={img2_path}\n"
                f"flo={flo_path}\n"
                "The evaluator is fail-fast by design; fix the dataset path or corrupt file first."
            )

        input_batch = np.expand_dims(input_comb, axis=0)
        input_batch = preprocess_eval_batch(input_batch)
        preds_results = sess.run(preds_tensor, feed_dict={input_ph: input_batch})
        flow_only = preds_results[:, :, :, :2]
        processor.update(label=gt_flow, prediction=flow_only, Args=args)

    sess.close()

    if processor.counter == 0:
        epe = float("nan")
    else:
        epe = float(np.concatenate(processor.errorEPEs).mean())

    scope_name = meta_data.get("scope_name", model_dir.name)
    return {
        "model_name": scope_name,
        "variant": meta_data.get("variant"),
        "checkpoint_dir": str(model_dir),
        "checkpoint_path": meta_data.get("checkpoint_path"),
        "ckpt_name": args.ckpt_name,
        "epoch": meta_data.get("epoch"),
        "global_step": meta_data.get("global_step"),
        "fc2_val_epe": meta_data.get("metric"),
        "sintel_epe": epe,
        "arch_code": meta_data.get("arch_code"),
        "samples": processor.counter,
    }


def _default_output_json(args, root_dir: Path) -> Path:
    if args.output_json:
        return Path(args.output_json)
    stem = f"sintel_eval_{args.ckpt_name}"
    return root_dir / f"{stem}.json"


def _write_csv(csv_path: Path, rows):
    if not rows:
        return
    headers = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    patch_size = [int(x) for x in args.patch_size.split(",")]
    if len(patch_size) != 2:
        raise ValueError("patch_size must be H,W")

    dataset_root = Path(args.dataset_root)
    img1_list, img2_list, flo_list, sintel_list_path = _prepare_sintel_lists(
        dataset_root=dataset_root,
        sintel_list_rel=args.sintel_list,
    )
    print(f"[*] Reading Sintel List: {sintel_list_path}")
    print(f"[*] Found {len(img1_list)} samples.")

    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
        model_dirs = _discover_model_dirs(experiment_dir=experiment_dir, model_name=args.model_name)
        output_root = experiment_dir
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
        model_dirs = [model_dir]
        output_root = model_dir

    results = []
    for model_dir in model_dirs:
        print(f"\n[*] Evaluating fixed-arch model from {model_dir} ({args.ckpt_name}) ...")
        summary = _evaluate_single_model(
            model_dir=model_dir,
            args=args,
            patch_size=patch_size,
            img1_list=img1_list,
            img2_list=img2_list,
            flo_list=flo_list,
        )
        results.append(summary)
        print(
            f"[+] model={summary['model_name']} variant={summary['variant']} "
            f"epoch={summary['epoch']} step={summary['global_step']} "
            f"fc2_val_epe={summary['fc2_val_epe']} sintel_epe={summary['sintel_epe']:.6f}"
        )

    results = sorted(results, key=lambda item: item["sintel_epe"])
    output_json = _default_output_json(args=args, root_dir=output_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_json.with_suffix(".csv")
    payload = {
        "dataset_root": str(dataset_root),
        "sintel_list": sintel_list_path,
        "patch_size": patch_size,
        "ckpt_name": args.ckpt_name,
        "results": results,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_csv, results)

    print("\n" + "=" * 60)
    print("Fixed-Arch Sintel Evaluation Summary")
    for item in results:
        print(
            f"{item['model_name']}: variant={item['variant']} "
            f"epoch={item['epoch']} step={item['global_step']} "
            f"fc2_val_epe={item['fc2_val_epe']} sintel_epe={item['sintel_epe']:.6f}"
        )
    print(f"json: {output_json}")
    print(f"csv:  {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
