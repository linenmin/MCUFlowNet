"""Export one supernet subnet and one bilinear baseline for Vela side-by-side comparison."""

import argparse
import importlib.util as importlib_util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from code.nas.supernet_eval import (
    _apply_cli_overrides,
    _build_checkpoint_paths,
    _find_existing_checkpoint,
    _load_checkpoint_meta,
    _load_config,
    _resolve_output_dir,
)
from code.nas.supernet_subnet_distribution import _build_tflite_for_arch, _run_vela_for_arch
from code.utils.json_io import write_json


def _parse_arch_code(text: str) -> List[int]:
    """Parse comma-separated 9-dim arch code."""
    parts = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(parts) != 9:
        raise ValueError(f"arch_code must contain 9 integers, got {len(parts)}")
    code = [int(item) for item in parts]
    if any(item not in (0, 1, 2) for item in code):
        raise ValueError(f"arch_code values must be in {{0,1,2}}, got {code}")
    return code


def _load_module(module_name: str, module_path: Path):
    """Load python module from file path."""
    spec = importlib_util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to build module spec: {module_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _accum_preds(pred_list):
    """Accumulate multi-scale predictions exactly like EdgeFlowNet misc.utils.AccumPreds."""
    import tensorflow as tf

    pred_accum = None
    pred_accum_list = []
    for pred_i in pred_list:
        if pred_accum is None:
            pred_accum = pred_i
            pred_accum_list.append(pred_accum)
            continue
        pred_accum = tf.compat.v1.image.resize_bilinear(
            pred_accum,
            [pred_i.shape[1], pred_i.shape[2]],
            align_corners=False,
            half_pixel_centers=False,
        )
        pred_accum += pred_i
        pred_accum_list.append(pred_accum)
    return pred_accum, pred_accum_list


def _build_bilinear_tflite(
    config: Dict[str, Any],
    tflite_path: Path,
    rep_dataset_samples: int,
    quantize_int8: bool,
) -> None:
    """Build bilinear baseline tflite from EdgeFlowNet/sramTest network definition."""
    import numpy as np
    import sys
    import tensorflow as tf

    repo_root = Path(__file__).resolve().parents[3]
    sram_test_dir = repo_root / "EdgeFlowNet" / "sramTest"
    edgeflownet_root = repo_root / "EdgeFlowNet"
    if str(sram_test_dir) not in sys.path:
        sys.path.insert(0, str(sram_test_dir))
    if str(edgeflownet_root) not in sys.path:
        sys.path.append(str(edgeflownet_root))

    bilinear_net_path = repo_root / "EdgeFlowNet" / "sramTest" / "network" / "MultiScaleResNet_bilinear.py"
    if not bilinear_net_path.exists():
        raise RuntimeError(f"bilinear network file not found: {bilinear_net_path}")
    bilinear_module = _load_module("edgeflownet_sram_bilinear", bilinear_net_path)

    data_cfg = config.get("data", {})
    input_h = int(data_cfg.get("input_height", 180))
    input_w = int(data_cfg.get("input_width", 240))
    flow_channels = int(data_cfg.get("flow_channels", 2))
    pred_channels = int(flow_channels * 2)

    tf.compat.v1.disable_eager_execution()
    graph = tf.Graph()
    with graph.as_default():
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_h, input_w, 6], name="Input")
        model_obj = bilinear_module.MultiScaleResNet(
            InputPH=input_ph,
            InitNeurons=32,
            ExpansionFactor=2.0,
            NumSubBlocks=2,
            NumOut=pred_channels,
            NumBlocks=1,
            Padding="same",
        )
        outputs = model_obj.Network()
        if isinstance(outputs, list) and len(outputs) > 1:
            accum_out, _ = _accum_preds(outputs)
            final_output = accum_out[..., 0:flow_channels]
        else:
            final_output = outputs[-1][..., 0:flow_channels] if isinstance(outputs, list) else outputs[..., 0:flow_channels]

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            if quantize_int8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                rng = np.random.default_rng(seed=2026)

                def representative_dataset_gen():
                    for _ in range(max(1, int(rep_dataset_samples))):
                        sample = rng.random((1, input_h, input_w, 6)).astype(np.float32)
                        yield [sample]

                converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def _collect_vela_artifacts(output_dir: Path) -> Dict[str, Any]:
    """Collect important vela output file paths."""
    detailed_txt = output_dir / "detailed_performance.txt"
    summary_csv = sorted(output_dir.glob("*_summary_*.csv"))
    return {
        "detailed_performance_txt": str(detailed_txt) if detailed_txt.exists() else "",
        "summary_csv_list": [str(item) for item in summary_csv],
    }


def _safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safe subtraction for optional float."""
    if a is None or b is None:
        return None
    return float(a - b)


def _build_parser() -> argparse.ArgumentParser:
    """Build cli parser."""
    parser = argparse.ArgumentParser(description="compare supernet subnet vs bilinear baseline with vela")
    parser.add_argument("--config", default="configs/supernet_fc2_180x240.yaml", help="path to supernet config yaml")
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type for supernet export")
    parser.add_argument("--skip_checkpoint", action="store_true", help="export supernet with random init (no checkpoint restore)")
    parser.add_argument("--arch_code", default="0,0,0,0,0,1,0,0,0", help="supernet arch code")
    parser.add_argument("--output_tag", default="", help="optional suffix for output folder")
    parser.add_argument("--vela_mode", choices=["basic", "verbose"], default="verbose", help="vela mode")
    parser.add_argument("--vela_optimise", choices=["Performance", "Size"], default="Size", help="vela optimise policy")
    parser.add_argument("--vela_rep_dataset_samples", type=int, default=5, help="rep-dataset samples for int8 export")
    parser.add_argument("--vela_float32", action="store_true", help="export float32 tflite instead of int8")
    parser.add_argument("--vela_verbose_log", action="store_true", help="disable vela silent mode")
    parser.add_argument("--keep_tflite", action="store_true", help="keep generated tflite files")
    parser.add_argument("--experiment_name", default=None, help="override runtime.experiment_name")
    parser.add_argument("--base_path", default=None, help="override data.base_path")
    parser.add_argument("--train_dir", default=None, help="override data.train_dir")
    parser.add_argument("--val_dir", default=None, help="override data.val_dir")
    parser.add_argument("--train_batch_size", type=int, default=None, help="override train.batch_size")
    parser.add_argument("--seed", type=int, default=None, help="override runtime.seed")
    return parser


def main() -> int:
    """Entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    config = _apply_cli_overrides(config=_load_config(args.config), args=args)
    output_root = _resolve_output_dir(config=config)

    checkpoint_prefix: Optional[Path] = None
    checkpoint_meta: Dict[str, Any] = {}
    if not bool(args.skip_checkpoint):
        ckpt_paths = _build_checkpoint_paths(experiment_dir=output_root)
        chosen = ckpt_paths[str(args.checkpoint_type)]
        found = _find_existing_checkpoint(path_prefix=chosen)
        if found is None:
            raise RuntimeError(f"checkpoint not found: {chosen}")
        checkpoint_prefix = Path(found)
        checkpoint_meta = _load_checkpoint_meta(path_prefix=checkpoint_prefix)

    arch_code = _parse_arch_code(args.arch_code)
    tag = str(args.output_tag).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.checkpoint_type}_{stamp}" + (f"_{tag}" if tag else "")
    run_dir = output_root / "vela_compare" / folder_name
    supernet_dir = run_dir / "supernet"
    bilinear_dir = run_dir / "bilinear"
    supernet_dir.mkdir(parents=True, exist_ok=True)
    bilinear_dir.mkdir(parents=True, exist_ok=True)

    quantize_int8 = not bool(args.vela_float32)
    rep_dataset_samples = int(args.vela_rep_dataset_samples)
    vela_silent = not bool(args.vela_verbose_log)

    supernet_tflite = supernet_dir / "supernet_subnet.tflite"
    bilinear_tflite = bilinear_dir / "bilinear_baseline.tflite"

    _build_tflite_for_arch(
        config=config,
        checkpoint_prefix=checkpoint_prefix,
        arch_code=arch_code,
        tflite_path=supernet_tflite,
        rep_dataset_samples=rep_dataset_samples,
        quantize_int8=quantize_int8,
        restore_checkpoint=not bool(args.skip_checkpoint),
    )
    supernet_vela = _run_vela_for_arch(
        tflite_path=supernet_tflite,
        output_dir=supernet_dir,
        vela_mode=str(args.vela_mode),
        vela_optimise=str(args.vela_optimise),
        vela_silent=vela_silent,
    )

    _build_bilinear_tflite(
        config=config,
        tflite_path=bilinear_tflite,
        rep_dataset_samples=rep_dataset_samples,
        quantize_int8=quantize_int8,
    )
    bilinear_vela = _run_vela_for_arch(
        tflite_path=bilinear_tflite,
        output_dir=bilinear_dir,
        vela_mode=str(args.vela_mode),
        vela_optimise=str(args.vela_optimise),
        vela_silent=vela_silent,
    )

    elapsed_seconds = float(time.perf_counter() - start_perf)
    elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(int(round(elapsed_seconds))))
    finished_at = datetime.now(timezone.utc)

    supernet_payload = {
        "arch_code": arch_code,
        "tflite_path": str(supernet_tflite),
        **supernet_vela,
        "vela_artifacts": _collect_vela_artifacts(output_dir=supernet_dir),
    }
    bilinear_payload = {
        "tflite_path": str(bilinear_tflite),
        **bilinear_vela,
        "vela_artifacts": _collect_vela_artifacts(output_dir=bilinear_dir),
    }
    diff_payload = {
        "sram_peak_mb_supernet_minus_bilinear": _safe_sub(supernet_payload.get("sram_peak_mb"), bilinear_payload.get("sram_peak_mb")),
        "inference_ms_supernet_minus_bilinear": _safe_sub(supernet_payload.get("inference_ms"), bilinear_payload.get("inference_ms")),
        "fps_supernet_minus_bilinear": _safe_sub(supernet_payload.get("fps"), bilinear_payload.get("fps")),
    }
    result = {
        "status": "ok",
        "checkpoint_type": str(args.checkpoint_type),
        "checkpoint_path": (str(checkpoint_prefix) if checkpoint_prefix is not None else ""),
        "checkpoint_meta": checkpoint_meta,
        "skip_checkpoint": bool(args.skip_checkpoint),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hms": elapsed_hms,
        "supernet": supernet_payload,
        "bilinear": bilinear_payload,
        "diff": diff_payload,
    }

    if not args.keep_tflite:
        try:
            supernet_tflite.unlink(missing_ok=True)
            bilinear_tflite.unlink(missing_ok=True)
        except Exception:
            pass

    summary_path = run_dir / "compare_summary.json"
    write_json(str(summary_path), result)
    print(
        json.dumps(
            {
                "status": "ok",
                "summary_path": str(summary_path),
                "supernet": {
                    "arch_code": arch_code,
                    "sram_peak_mb": supernet_payload.get("sram_peak_mb"),
                    "inference_ms": supernet_payload.get("inference_ms"),
                    "fps": supernet_payload.get("fps"),
                },
                "bilinear": {
                    "sram_peak_mb": bilinear_payload.get("sram_peak_mb"),
                    "inference_ms": bilinear_payload.get("inference_ms"),
                    "fps": bilinear_payload.get("fps"),
                },
                "diff": diff_payload,
                "elapsed_hms": elapsed_hms,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
