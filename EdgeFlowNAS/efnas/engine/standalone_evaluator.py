import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import tensorflow as tf
import numpy as np

from efnas.utils.import_bootstrap import bootstrap_project_paths, resolve_project_paths

bootstrap_project_paths(anchor_file=__file__)
project_root = resolve_project_paths(anchor_file=__file__)["mcu_root"]

from efnas.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet
from efnas.engine.eval_step import accumulate_predictions
from EdgeFlowNet.code.misc.processor import FlowPostProcessor


def load_model_metadata(checkpoint_dir: Path, ckpt_name: str = "best") -> Dict[str, Any]:
    """
    Reads the .meta.json file associated with a checkpoint directory
    to extract arch_code, epoch, and other training metadata.
    Search in the 'checkpoints' subdirectory.
    """
    meta_path = checkpoint_dir / "checkpoints" / f"{ckpt_name}.ckpt.meta.json"
    
    if not meta_path.exists():
        # Try fallback if picking best but only last exists
        if ckpt_name == "best":
            meta_path = checkpoint_dir / "checkpoints" / "last.ckpt.meta.json"
        
    if not meta_path.exists():
        raise FileNotFoundError(f"No .meta.json found at {meta_path}. Cannot determine 'arch_code'. Check your directory structure.")
    
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
        
    return meta_data


def setup_eval_model(
    checkpoint_dir: Path, 
    patch_size: Tuple[int, int],
    ckpt_name: str = "best"
) -> Tuple[tf.compat.v1.Session, tf.Tensor, tf.Tensor, Dict[str, Any]]:
    """
    Reads metadata, builds the MultiScaleResNetSupernet graph under the correct variable scope,
    and restores the weights.
    
    Args:
        checkpoint_dir: Directory containing the 'checkpoints' folder.
        patch_size: Target resolution [height, width] for the input placeholder.
        ckpt_name: 'best' or 'last'
        
    Returns:
        sess: TensorFlow session
        input_ph: The input placeholder tensor (shape: [1, H, W, 6])
        preds_tensor: The prediction tensor (for FlowPostProcessor)
        meta_data: The loaded metadata dictionary
    """
    meta_data = load_model_metadata(checkpoint_dir, ckpt_name)
    arch_code = meta_data["arch_code"]
    
    # Extract scope name from directory name:
    # Training uses arch_name (e.g. 'target') as variable_scope,
    # but saves to directory 'model_{name}' (e.g. 'model_target').
    # We need to strip the 'model_' prefix to get the correct scope.
    dir_name = checkpoint_dir.name
    if dir_name.startswith("model_"):
        scope_name = dir_name[len("model_"):]
    else:
        scope_name = dir_name
    
    tf.compat.v1.reset_default_graph()
    
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, patch_size[0], patch_size[1], 6], name="input_ph")
    is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph")
    
    # arch_code must be a TF int32 tensor (the supernet uses tf.one_hot on it)
    # meta.json stores arch_code as list[int], but handle string format too for safety
    if isinstance(arch_code, str):
        arch_list = [int(x) for x in arch_code.split(",")]
    else:
        arch_list = [int(x) for x in arch_code]
    arch_code_ph = tf.constant(arch_list, dtype=tf.int32, name="arch_code_ph")
    
    # We must construct the graph under the same variable scope used during Standalone Retraining
    with tf.compat.v1.variable_scope(scope_name):
        model = MultiScaleResNetSupernet(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=4,
        )
        preds = model.build()

    # Accumulate multi-scale predictions into a single tensor (same as training)
    preds_accumulated = accumulate_predictions(preds)

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
    
    return sess, input_ph, preds_accumulated, meta_data

def preprocess_eval_batch(input_batch: np.ndarray) -> np.ndarray:
    """
    Applies the exact same standardization used during standalone training.
    Mapping [0, 255] -> [-1, 1]
    """
    # The training code does: batch.input / 255.0 * 2.0 - 1.0
    return (input_batch / 255.0) * 2.0 - 1.0
