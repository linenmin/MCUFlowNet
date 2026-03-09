import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Adjust sys.path
project_root = Path(__file__).resolve().parent.parent.parent
edgeflownet_dir = project_root / "EdgeFlowNet"
edgeflownas_dir = project_root / "EdgeFlowNAS"

edgeflownet_code_dir = edgeflownet_dir / "code"
edgeflownas_code_dir = edgeflownas_dir / "code"

# Insert in reverse order of priority
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

from efnas.engine.edgeflownet_original_evaluator import setup_edgeflownet_original_model
from efnas.engine.standalone_evaluator import setup_eval_model, preprocess_eval_batch
from EdgeFlowNet.code.misc.utils import read_sintel_list, get_sintel_batch
from EdgeFlowNet.code.misc.processor import FlowPostProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Standalone/Supernet/Original EdgeFlowNet Model on Sintel")
    parser.add_argument(
        "--model_type",
        type=str,
        default="standalone_supernet",
        choices=["standalone_supernet", "edgeflownet_original"],
        help="Model loading mode. standalone_supernet expects EdgeFlowNAS retrain outputs; edgeflownet_original expects an original EdgeFlowNet checkpoint prefix.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="For standalone_supernet: path to the trained model directory (e.g., outputs/.../model_target). For edgeflownet_original: checkpoint prefix/path (e.g., EdgeFlowNet/checkpoints/best.ckpt).",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which checkpoint to load for standalone_supernet: 'best' (default) or 'last'",
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root of the Sintel dataset (usually Datasets/Sintel/)")
    parser.add_argument("--sintel_list", type=str, default="EdgeFlowNet/code/dataset_paths/MPI_Sintel_Final_train_list.txt",
                        help="Path to the split list file")
    parser.add_argument("--patch_size", type=str, default="416,1024",
                        help="Test resolution, comma separated (H,W). Default is 416,1024 (Baseline convention)")
    parser.add_argument("--gpu_device", type=int, default=0,
                        help="GPU device ID to use")

    # Flags required by FlowPostProcessor
    parser.add_argument('--Display', action='store_true', help='Display Output')
    parser.add_argument('--ShiftedFlow', action='store_true', help='Shift Flow for Overlap Crop Method')
    parser.add_argument('--ResizeToHalf', action='store_true', help='ResizeToHalf')
    parser.add_argument('--ResizeCropStack', action='store_true', help='ResizeCropStack')
    parser.add_argument('--ResizeNearestCropStack', action='store_true', help='ResizeNearestCropStack')
    parser.add_argument('--NumberOfHalves', type=int, default=0, help='ResizeCropStackBlur')
    parser.add_argument('--ResizeCropStackBlur', action='store_true', help='OverlapCropStack')
    parser.add_argument('--OverlapCropStack', action='store_true', help='OverlapCropStack')
    parser.add_argument('--PatchDelta', type=int, default=0, help='OverlapCropStack Padding Value')
    parser.add_argument('--uncertainity', action='store_true', help='Uncertainity Extraction enabled (only 2 channels evaluated)')

    return parser.parse_args()


def _strip_sintel_prefix(path_str):
    prefix = "Datasets/Sintel/"
    if path_str.startswith(prefix):
        return path_str[len(prefix):]
    return path_str


def evaluate_sintel(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    patch_size = [int(x) for x in args.patch_size.split(',')]
    assert len(patch_size) == 2, "Patch size must be H,W"

    ckpt_target = Path(args.checkpoint_dir)
    if args.model_type == "standalone_supernet" and not ckpt_target.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_target} does not exist.")

    dataset_root = Path(args.dataset_root)
    sintel_list_path = project_root / args.sintel_list

    print(f"[*] Reading Sintel List: {sintel_list_path}")

    from argparse import Namespace
    list_args = Namespace(data_list=str(sintel_list_path))
    rel_img1_list, rel_img2_list, rel_flo_list = read_sintel_list(list_args)

    img1_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_img1_list]
    img2_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_img2_list]
    flo_list = [str(dataset_root / _strip_sintel_prefix(p)) for p in rel_flo_list]

    num_samples = len(img1_list)
    print(f"[*] Found {num_samples} samples.")

    if args.model_type == "standalone_supernet":
        print(f"\n[*] Initializing standalone supernet model from {ckpt_target} (weight: {args.ckpt_name}) ...")
        sess, input_ph, preds_tensor, meta_data = setup_eval_model(ckpt_target, patch_size, ckpt_name=args.ckpt_name)
        requires_standardize = True
    else:
        print(f"\n[*] Initializing original EdgeFlowNet model from {ckpt_target} ...")
        sess, input_ph, preds_tensor, meta_data = setup_edgeflownet_original_model(
            checkpoint_path=ckpt_target,
            patch_size=patch_size,
            use_uncertainty=args.uncertainity,
        )
        requires_standardize = False

    processor = FlowPostProcessor("full", is_multiscale=True)

    print(f"\n[*] Starting Sintel Evaluation at Resolution {patch_size} ...")

    for i in tqdm(range(num_samples), desc=f"Evaluating {ckpt_target.name}", unit="sample"):
        input_comb, gt_flow = get_sintel_batch(
            img1_list[i],
            img2_list[i],
            flo_list[i],
            patch_size,
        )

        if input_comb is None or gt_flow is None:
            tqdm.write(f"Warning: Failed to load sample {i} ({img1_list[i]}). Skipping.")
            continue

        input_batch = np.expand_dims(input_comb, axis=0)
        if requires_standardize:
            input_batch = preprocess_eval_batch(input_batch)

        preds_results = sess.run(
            preds_tensor,
            feed_dict={input_ph: input_batch},
        )

        flow_only = preds_results[:, :, :, :2]
        processor.update(label=gt_flow, prediction=flow_only, Args=args)

    print("\n" + "=" * 50)
    print(f"Evaluation Results for Model: {ckpt_target.name}")
    print(f"Model Type: {args.model_type}")
    print(f"Arch Code: {meta_data.get('arch_code')}")
    print(f"Training Epoch: {meta_data.get('epoch', 'N/A')} (Global Step: {meta_data.get('global_step', 'N/A')})")
    print(f"Training Val EPE (on FC2): {meta_data.get('metric', 'N/A')}")
    if meta_data.get("auto_num_out_fallback"):
        print(f"Original EdgeFlowNet num_out auto-fallback used: {meta_data.get('num_out')}")
    processor.print()
    print("=" * 50 + "\n")


if __name__ == "__main__":
    args = parse_args()
    evaluate_sintel(args)
