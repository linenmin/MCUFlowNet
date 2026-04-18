"""Export V2 constant-arch supernet vs fixed-subnet graphs and compare with Vela."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _accum_preds(pred_list):
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


def _convert_to_tflite(sess, input_ph, final_output, tflite_path: Path, rep_dataset_samples: int) -> None:
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
        input_shape = input_ph.shape.as_list()
        for _ in range(max(1, int(rep_dataset_samples))):
            sample = rng.random(input_shape).astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare retrain_v2 export graphs with Vela")
    parser.add_argument("--checkpoint_prefix", required=True, help="supernet checkpoint prefix")
    parser.add_argument("--arch_code", required=True, help="11D arch code")
    parser.add_argument("--input_height", type=int, default=352, help="input height")
    parser.add_argument("--input_width", type=int, default=480, help="input width")
    parser.add_argument("--flow_channels", type=int, default=2, help="flow channels")
    parser.add_argument("--rep_dataset_samples", type=int, default=5, help="int8 representative dataset size")
    parser.add_argument("--optimise", choices=["Performance", "Size"], default="Size", help="Vela optimise mode")
    parser.add_argument("--output_dir", default=None, help="explicit output directory")
    return parser


def _export_constant_arch_supernet(
    checkpoint_prefix: Path,
    arch_code,
    input_height: int,
    input_width: int,
    flow_channels: int,
    rep_dataset_samples: int,
    tflite_path: Path,
):
    from efnas.network.MultiScaleResNet_supernet_v2 import MultiScaleResNetSupernetV2

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_height, input_width, 6], name="input_ph")
    is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph")
    arch_code_ph = tf.constant(arch_code, dtype=tf.int32, name="FixedArchCode")
    model = MultiScaleResNetSupernetV2(
        input_ph=input_ph,
        arch_code_ph=arch_code_ph,
        is_training_ph=is_training_ph,
        num_out=int(flow_channels) * 2,
        init_neurons=32,
        expansion_factor=2.0,
    )
    preds = model.build()
    final_output = _accum_preds(preds)[..., 0:int(flow_channels)]
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, str(checkpoint_prefix))
        _convert_to_tflite(sess, input_ph, final_output, tflite_path, rep_dataset_samples)


def _export_fixed_subnet(
    checkpoint_prefix: Path,
    arch_code,
    input_height: int,
    input_width: int,
    flow_channels: int,
    rep_dataset_samples: int,
    tflite_path: Path,
):
    from efnas.engine.retrain_v2_trainer import _build_supernet_source_name_map, _build_warmstart_var_map
    from efnas.network.fixed_arch_models_v2 import FixedArchModelV2

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_height, input_width, 6], name="input_ph")
    is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name="is_training_ph")
    with tf.compat.v1.variable_scope("export"):
        model = FixedArchModelV2(
            input_ph=input_ph,
            is_training_ph=is_training_ph,
            arch_code=arch_code,
            num_out=int(flow_channels) * 2,
            init_neurons=32,
            expansion_factor=2.0,
        )
        preds = model.build()
    final_output = _accum_preds(preds)[..., 0:int(flow_channels)]
    scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("export/")]
    warmstart_map = _build_warmstart_var_map(
        scope_name="export",
        scope_global_vars=scope_global_vars,
        source_name_map=_build_supernet_source_name_map(pred_channels=int(flow_channels) * 2),
    )
    saver = tf.compat.v1.train.Saver(var_list=warmstart_map)
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, str(checkpoint_prefix))
        _convert_to_tflite(sess, input_ph, final_output, tflite_path, rep_dataset_samples)


def main() -> int:
    from efnas.nas.arch_codec_v2 import parse_arch_code_text
    from tests.vela.vela_compiler import run_vela

    args = _build_parser().parse_args()
    checkpoint_prefix = Path(str(args.checkpoint_prefix).replace(".index", "").replace(".meta", ""))
    arch_code = parse_arch_code_text(args.arch_code)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "retrain_v2_vela_compare" / args.arch_code.replace(",", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("constant_arch_supernet", _export_constant_arch_supernet),
        ("fixed_subnet", _export_fixed_subnet),
    ]
    results = []
    for label, fn in variants:
        variant_dir = output_dir / label
        tflite_path = variant_dir / f"{label}.tflite"
        vela_dir = variant_dir / "vela"
        fn(
            checkpoint_prefix=checkpoint_prefix,
            arch_code=arch_code,
            input_height=int(args.input_height),
            input_width=int(args.input_width),
            flow_channels=int(args.flow_channels),
            rep_dataset_samples=int(args.rep_dataset_samples),
            tflite_path=tflite_path,
        )
        sram_mb, time_ms = run_vela(
            str(tflite_path),
            mode="verbose",
            output_dir=str(vela_dir),
            optimise=str(args.optimise),
            silent=True,
        )
        results.append(
            {
                "label": label,
                "tflite_path": str(tflite_path),
                "vela_dir": str(vela_dir),
                "sram_mb": sram_mb,
                "time_ms": time_ms,
            }
        )
        print(f"{label}: sram_mb={sram_mb} time_ms={time_ms}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"arch_code": arch_code, "results": results}, indent=2), encoding="utf-8")
    print(f"summary_json: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
