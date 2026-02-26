import os
import sys
import argparse
from pathlib import Path

# Adjust sys.path
project_root = Path(__file__).resolve().parent.parent.parent
edgeflownet_dir = project_root / "EdgeFlowNet"
edgeflownas_dir = project_root / "EdgeFlowNAS"

edgeflownet_code_dir = edgeflownet_dir / "code"

if str(edgeflownet_dir) not in sys.path:
    sys.path.insert(0, str(edgeflownet_dir))
if str(edgeflownet_code_dir) not in sys.path:
    sys.path.insert(0, str(edgeflownet_code_dir))
if str(edgeflownas_dir) not in sys.path:
    sys.path.insert(0, str(edgeflownas_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from EdgeFlowNAS.code.engine.standalone_evaluator import setup_eval_model, preprocess_eval_batch
from EdgeFlowNet.code.misc.utils import read_sintel_list, get_sintel_batch
from EdgeFlowNet.code.misc.processor import FlowPostProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Retrained Standalone Model on Sintel")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the trained model directory (e.g., outputs/.../model_target) containing checkpoints and .meta.json")
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


def evaluate_sintel(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    
    patch_size = [int(x) for x in args.patch_size.split(',')]
    assert len(patch_size) == 2, "Patch size must be H,W"

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
        
    # 1. Setup Data Paths
    dataset_root = Path(args.dataset_root)
    sintel_list_path = project_root / args.sintel_list
    
    print(f"[*] Reading Sintel List: {sintel_list_path}")
    
    # read_sintel_list expects an object with .data_list attribute
    from argparse import Namespace
    list_args = Namespace(data_list=str(sintel_list_path))
    rel_img1_list, rel_img2_list, rel_flo_list = read_sintel_list(list_args)
    
    # Baseline returns relative paths from the .txt, we need to join them with dataset_root
    img1_list = [str(dataset_root / p) for p in rel_img1_list]
    img2_list = [str(dataset_root / p) for p in rel_img2_list]
    flo_list = [str(dataset_root / p) for p in rel_flo_list]
    
    num_samples = len(img1_list)
    print(f"[*] Found {num_samples} samples.")

    # 2. Setup Evaluation Engine (Loads variable scope, graph, and weights automatically)
    print(f"\n[*] Initializing Model from {ckpt_dir} ...")
    sess, input_ph, preds_tensor, meta_data = setup_eval_model(ckpt_dir, patch_size)
    
    # 3. Setup FlowPostProcessor
    # We use "full" suffix convention and is_multiscale=True since Supernet outputs multiscale lists just like MultiScaleResNet
    processor = FlowPostProcessor("full", is_multiscale=True)

    print(f"\n[*] Starting Sintel Evaluation at Resolution {patch_size} ...")
    
    # 4. Evaluation Loop
    for i in range(num_samples):
        # get_sintel_batch applies ResizeNearestCrop under the hood 
        # (resizes keeping aspect ratio, then center crops to patch_size)
        # Returns [1, H, W, 6] (stacked img1, img2) and [1, H, W, 2] (ground truth flow)
        input_comb, gt_flow = get_sintel_batch(
            img1_list[i], 
            img2_list[i], 
            flo_list[i], 
            patch_size
        )
        
        if input_comb is None or gt_flow is None:
            print(f"Warning: Failed to load sample {i} ({img1_list[i]}). Skipping.")
            continue
            
        # Add batch dimension: [1, H, W, 6]
        input_batch = np.expand_dims(input_comb, axis=0) 
        
        # VERY IMPORTANT: Standardize input identically to standalone_trainer [-1, 1]
        input_batch = preprocess_eval_batch(input_batch)

        # Run Inference
        preds_results = sess.run(
            preds_tensor, 
            feed_dict={input_ph: input_batch}
        )
        
        # Process and accumulate error
        # FlowPostProcessor expects list of multiscale outputs, which preds_tensor provides
        processor.update(label=gt_flow, prediction=preds_results, Args=args)
        
        # Optional: Print progress
        if (i + 1) % 50 == 0 or (i + 1) == num_samples:
            print(f"    Processing {i + 1}/{num_samples}...")

    # 5. Print Summary
    print("\n" + "="*50)
    print(f"Evaluation Results for Model: {ckpt_dir.name}")
    print(f"Arch Code: {meta_data.get('arch_code')}")
    print(f"Training Epoch: {meta_data.get('epoch', 'N/A')} (Global Step: {meta_data.get('global_step', 'N/A')})")
    print(f"Training Val EPE (on FC2): {meta_data.get('metric', 'N/A')}")
    processor.print()
    print("="*50 + "\n")


if __name__ == "__main__":
    args = parse_args()
    evaluate_sintel(args)
