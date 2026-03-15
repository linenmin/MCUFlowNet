"""Export fixed-arch variants and run Vela side-by-side precheck."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except ModuleNotFoundError:
        payload: Dict[str, object] = {}
        current_section = None
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip()
                if not line or line.lstrip().startswith("#"):
                    continue
                if line.endswith(":") and not line.startswith(" "):
                    current_section = line[:-1].strip()
                    payload[current_section] = {}
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                target = payload if current_section is None or raw_line.startswith(" ") is False else payload[current_section]
                if value.lower() in ("true", "false"):
                    parsed = value.lower() == "true"
                else:
                    try:
                        if "." in value or "e" in value.lower():
                            parsed = float(value)
                        else:
                            parsed = int(value)
                    except ValueError:
                        parsed = value
                target[key] = parsed
        return payload


def _parse_arch_code(raw: str) -> List[int]:
    tokens = [item.strip() for item in str(raw).split(",") if item.strip()]
    if len(tokens) != 9:
        raise ValueError(f"arch_code must have 9 ints, got {len(tokens)}")
    return [int(item) for item in tokens]


def _parse_plus_list(raw: str) -> List[str]:
    items = [item.strip() for item in str(raw).split("+") if item.strip()]
    if not items:
        raise ValueError("expected at least one item")
    return items


def _accum_preds(pred_list):
    import tensorflow as tf

    pred_accum = None
    for pred_i in pred_list:
        if pred_accum is None:
            pred_accum = pred_i
            continue
        pred_accum = tf.compat.v1.image.resize_bilinear(
            pred_accum,
            [pred_i.shape[1], pred_i.shape[2]],
            align_corners=False,
            half_pixel_centers=False,
        )
        pred_accum = pred_accum + pred_i
    return pred_accum


def _build_variant_tflite(
    arch_code: List[int],
    variant: str,
    input_height: int,
    input_width: int,
    flow_channels: int,
    rep_dataset_samples: int,
    tflite_path: Path,
) -> Dict[str, List[int]]:
    import numpy as np
    import tensorflow as tf

    from efnas.network.fixed_arch_models import FixedArchModel

    pred_channels = int(flow_channels) * 2
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    input_ph = tf.compat.v1.placeholder(
        tf.float32, shape=[1, int(input_height), int(input_width), 6], name="input_ph"
    )
    is_training_ph = tf.compat.v1.placeholder_with_default(
        tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph"
    )

    with tf.compat.v1.variable_scope(variant):
        model = FixedArchModel(
            input_ph=input_ph,
            is_training_ph=is_training_ph,
            arch_code=arch_code,
            variant=variant,
            num_out=pred_channels,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()
        final_output = _accum_preds(preds)[..., 0:int(flow_channels)]

    feature_shapes = {
        key: tensor.get_shape().as_list() for key, tensor in model.feature_pyramid().items()
    }
    output_shape = final_output.get_shape().as_list()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
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
                sample = rng.random((1, int(input_height), int(input_width), 6)).astype(np.float32)
                yield [sample]

        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    return {"output_shape": output_shape, "feature_shapes": feature_shapes}


def _read_summary_csv(csv_path: Path) -> Dict[str, float]:
    import csv

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[0]
    return {
        "sram_kib": float(row["sram_memory_used"]),
        "sram_bytes": float(row["sram_memory_used"]) * 1024.0,
        "inference_ms": float(row["inference_time"]) * 1000.0,
        "fps": 1000.0 / (float(row["inference_time"]) * 1000.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-arch variants through INT8 TFLite + Vela")
    parser.add_argument("--config", default="configs/fixed_arch_compare_fc2_172x224_leaderboard6.yaml")
    parser.add_argument("--backbone_arch_code", default=None)
    parser.add_argument("--model_variants", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--optimise", choices=["Performance", "Size"], default="Size")
    parser.add_argument("--rep_dataset_samples", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = _load_yaml(config_path)

    arch_code = _parse_arch_code(args.backbone_arch_code or config.get("backbone_arch_code", "0,2,1,1,0,0,0,0,0"))
    variants = _parse_plus_list(args.model_variants or config.get("model_variants", "baseline"))
    data_cfg = config.get("data", {})
    input_height = int(data_cfg.get("input_height", 172))
    input_width = int(data_cfg.get("input_width", 224))
    flow_channels = int(data_cfg.get("flow_channels", 2))

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = PROJECT_ROOT / "outputs" / "fixed_arch_vela_compare" / f"{input_height}x{input_width}"
    output_root.mkdir(parents=True, exist_ok=True)

    from tests.vela.vela_compiler import run_vela

    results = []
    for variant in variants:
        variant_dir = output_root / variant
        vela_dir = variant_dir / "vela"
        tflite_path = variant_dir / f"{variant}.tflite"
        meta = _build_variant_tflite(
            arch_code=arch_code,
            variant=variant,
            input_height=input_height,
            input_width=input_width,
            flow_channels=flow_channels,
            rep_dataset_samples=args.rep_dataset_samples,
            tflite_path=tflite_path,
        )
        sram_mb, time_ms = run_vela(
            str(tflite_path),
            mode="verbose",
            output_dir=str(vela_dir),
            optimise=str(args.optimise),
            silent=True,
        )
        summary_csv = next(vela_dir.glob("*_summary_*.csv"))
        summary = _read_summary_csv(summary_csv)
        payload = {
            "variant": variant,
            "arch_code": arch_code,
            "input_shape": [1, input_height, input_width, 6],
            "output_shape": meta["output_shape"],
            "feature_shapes": meta["feature_shapes"],
            "summary_csv": str(summary_csv),
            "detailed_performance_txt": str(vela_dir / "detailed_performance.txt"),
            "sram_mb_reported": sram_mb,
            "inference_ms_reported": time_ms,
            **summary,
        }
        results.append(payload)
        print(
            f"{variant}: output={meta['output_shape']} "
            f"sram={summary.get('sram_kib', -1):.2f} KiB "
            f"infer={summary.get('inference_ms', -1):.3f} ms "
            f"fps={summary.get('fps', -1):.3f}"
        )

    summary_json = output_root / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "input_height": input_height,
                "input_width": input_width,
                "optimise": args.optimise,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"summary_json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
