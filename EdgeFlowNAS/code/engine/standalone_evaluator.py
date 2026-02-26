import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import tensorflow as tf
import numpy as np

# Adjust sys.path so we can import code from EdgeFlowNet/EdgeFlowNAS
project_root = Path(__file__).resolve().parent.parent.parent.parent
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

from EdgeFlowNAS.code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet
from EdgeFlowNet.code.misc.processor import FlowPostProcessor


def load_model_metadata(checkpoint_dir: Path) -> Dict[str, Any]:
    """
    Reads the .meta.json file associated with a checkpoint directory
    to extract arch_code, epoch, and other training metadata.
    """
    meta_path = checkpoint_dir / "best.ckpt.meta.json"
    if not meta_path.exists():
        # Fallback to last if best doesn't exist
        meta_path = checkpoint_dir / "last.ckpt.meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"No .meta.json found in {checkpoint_dir}. Cannot determine 'arch_code'.")
    
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
        
    return meta_data


def setup_eval_model(
    checkpoint_dir: Path, 
    patch_size: Tuple[int, int]
) -> Tuple[tf.compat.v1.Session, tf.Tensor, tf.Tensor, Dict[str, Any]]:
    """
    Reads metadata, builds the MultiScaleResNetSupernet graph under the correct variable scope,
    and restores the weights.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint and .meta.json
        patch_size: Target resolution [height, width] for the input placeholder.
        
    Returns:
        sess: TensorFlow session
        input_ph: The input placeholder tensor (shape: [1, H, W, 6])
        preds_tensor: The prediction tensor (for FlowPostProcessor)
        meta_data: The loaded metadata dictionary
    """
    meta_data = load_model_metadata(checkpoint_dir)
    arch_code = meta_data["arch_code"]
    
    # Try to extract the scope name from the directory name (e.g. 'model_target')
    scope_name = checkpoint_dir.name
    
    tf.compat.v1.reset_default_graph()
    
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, patch_size[0], patch_size[1], 6], name="input_ph")
    is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph")
    
    # We must construct the graph under the same variable scope used during Standalone Retraining
    with tf.compat.v1.variable_scope(scope_name):
        model = MultiScaleResNetSupernet(
            input_tensor=input_ph,
            arch_code=arch_code,
            is_training=is_training_ph,
            bn_decay=0.9, # Doesn't matter for pure eval, but required by API 
            # (Note: we use moving statistics, so is_training_ph=False is correct here)
            flow_channels=2,
            pred_channels=4,
            width_multiplier=1.0, # Fixed to 1.0 based on run_standalone_train setup
            build_activation='relu',
            build_stride=2
        )
        preds = model.build()

    # Create session and restorer
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    # Find the variables we need to restore
    scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(f"{scope_name}/")]
    
    if not scope_global_vars:
        raise ValueError(f"No variables found with scope '{scope_name}'. "
                         f"Please ensure the directory name matches the trained model's scope.")
        
    saver = tf.compat.v1.train.Saver(var_list=scope_global_vars)
    
    # Load the checkpoint path from meta
    ckpt_path = meta_data.get("checkpoint_path")
    if not ckpt_path:
        raise ValueError(f"Metadata file at {checkpoint_dir} does not contain 'checkpoint_path'.")
    
    saver.restore(sess, ckpt_path)
    print(f"[*] Successfully restored model weights from {ckpt_path}")
    print(f"[*] Target Arch Code: {arch_code}")
    print(f"[*] Training Progress: Epoch {meta_data.get('epoch', 'N/A')}")
    
    return sess, input_ph, preds, meta_data

def preprocess_eval_batch(input_batch: np.ndarray) -> np.ndarray:
    """
    Applies the exact same standardization used during standalone training.
    Mapping [0, 255] -> [-1, 1]
    """
    # The training code does: batch.input / 255.0 * 2.0 - 1.0
    return (input_batch / 255.0) * 2.0 - 1.0
