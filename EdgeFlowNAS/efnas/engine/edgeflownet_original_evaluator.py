import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import tensorflow as tf

from efnas.engine.eval_step import accumulate_predictions
from efnas.utils.import_bootstrap import bootstrap_project_paths, resolve_project_paths

bootstrap_project_paths(anchor_file=__file__)
project_root = resolve_project_paths(anchor_file=__file__)["mcu_root"]


def _resolve_checkpoint_prefix(path_like) -> Path:
    """Normalize checkpoint inputs like best.ckpt(.index/.meta/.meta.json)."""
    candidate = Path(str(path_like).strip())
    if not candidate.is_absolute():
        candidate = project_root / candidate
    text = str(candidate)
    for suffix in (".index", ".meta", ".meta.json"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = text.replace("\\", "/")
    if ".data-" in text and "-of-" in text:
        text = text.split(".data-")[0]
    return Path(text)


def _load_edgeflownet_network_class():
    """Load the original EdgeFlowNet MultiScaleResNet implementation."""
    module = importlib.import_module("network.MultiScaleResNet")
    network_cls = getattr(module, "MultiScaleResNet", None)
    if network_cls is None:
        raise RuntimeError("failed to load EdgeFlowNet MultiScaleResNet class")
    return network_cls


def _build_edgeflownet_original_graph(
    checkpoint_prefix: Path,
    patch_size: Tuple[int, int],
    num_out: int,
):
    """Build and restore the original EdgeFlowNet graph for Sintel evaluation."""
    tf.compat.v1.reset_default_graph()

    input_ph = tf.compat.v1.placeholder(
        tf.float32,
        shape=[1, patch_size[0], patch_size[1], 6],
        name="input_ph",
    )

    network_cls = _load_edgeflownet_network_class()
    model = network_cls(
        InputPH=input_ph,
        InitNeurons=32,
        NumSubBlocks=2,
        Suffix="",
        NumOut=int(num_out),
        ExpansionFactor=2.0,
        UncType=None,
    )
    preds = model.Network()
    preds_accumulated = accumulate_predictions(preds)

    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    saver.restore(sess, str(checkpoint_prefix))
    return sess, input_ph, preds_accumulated


def setup_edgeflownet_original_model(
    checkpoint_path,
    patch_size: Tuple[int, int],
    use_uncertainty: bool = False,
) -> Tuple[tf.compat.v1.Session, tf.Tensor, tf.Tensor, Dict[str, Any]]:
    """Restore the original EdgeFlowNet MultiScaleResNet checkpoint."""
    checkpoint_prefix = _resolve_checkpoint_prefix(checkpoint_path)
    if not Path(str(checkpoint_prefix) + ".index").exists():
        raise FileNotFoundError(f"Original EdgeFlowNet checkpoint not found: {checkpoint_prefix}")

    num_out_candidates = [4, 2] if bool(use_uncertainty) else [2, 4]
    last_error = None

    for num_out in num_out_candidates:
        sess = None
        try:
            sess, input_ph, preds_accumulated = _build_edgeflownet_original_graph(
                checkpoint_prefix=checkpoint_prefix,
                patch_size=patch_size,
                num_out=num_out,
            )
            meta_data = {
                "model_type": "edgeflownet_original",
                "checkpoint_path": str(checkpoint_prefix),
                "num_out": int(num_out),
                "arch_code": "edgeflownet_original",
                "epoch": "N/A",
                "global_step": "N/A",
                "metric": "N/A",
            }
            if bool(use_uncertainty) != bool(num_out == 4):
                meta_data["auto_num_out_fallback"] = True
            print(f"[*] Successfully restored original EdgeFlowNet weights from {checkpoint_prefix}")
            print(f"[*] Original EdgeFlowNet num_out={num_out}")
            return sess, input_ph, preds_accumulated, meta_data
        except Exception as exc:
            last_error = exc
            if sess is not None:
                sess.close()

    raise RuntimeError(
        f"failed to restore original EdgeFlowNet checkpoint {checkpoint_prefix} "
        f"with num_out candidates {num_out_candidates}: {last_error}"
    )
