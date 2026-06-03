"""Export an Ablation V1 backbone variant and run Vela for SRAM profiling."""

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MCU_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MCU_ROOT) not in sys.path:
    sys.path.insert(0, str(MCU_ROOT))


VARIANTS = {
    "edgeflownet_bilinear_eca_gate4x": {
        "name": "edgeflownet_bilinear_eca_gate4x",
        "upsample_mode": "bilinear",
        "bottleneck_eca": True,
        "gate_4x": True,
    },
}


def _accum_preds(preds):
    import tensorflow as tf

    pred_accum = preds[0]
    accum_outputs = [pred_accum]
    for pred in preds[1:]:
        pred_accum = tf.compat.v1.image.resize_bilinear(
            pred_accum,
            [pred.shape.as_list()[1], pred.shape.as_list()[2]],
            align_corners=False,
            half_pixel_centers=False,
        )
        pred_accum = pred_accum + pred
        accum_outputs.append(pred_accum)
    return pred_accum, accum_outputs


def _convert_to_tflite(sess, input_ph, output_tensor, tflite_path: Path, rep_dataset_samples: int) -> None:
    import numpy as np
    import tensorflow as tf

    def representative_dataset():
        for _ in range(int(rep_dataset_samples)):
            yield [np.random.uniform(0.0, 1.0, size=input_ph.shape.as_list()).astype(np.float32)]

    converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [output_tensor])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(converter.convert())


def _export_variant(tflite_path: Path, variant_name: str, input_height: int, input_width: int, rep_dataset_samples: int) -> None:
    import tensorflow as tf
    from efnas.network.ablation_edgeflownet import ABlationEdgeFlowNetV1

    tf.compat.v1.reset_default_graph()
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_height, input_width, 6], name="Input")
    is_training_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="IsTraining")
    model = ABlationEdgeFlowNetV1(
        input_ph=input_ph,
        is_training_ph=is_training_ph,
        num_out=4,
        variant_config=VARIANTS[variant_name],
    )
    outputs = model.build()
    final_output, _ = _accum_preds(outputs)
    final_output = final_output[..., 0:2]
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        _convert_to_tflite(sess, input_ph, final_output, tflite_path, rep_dataset_samples)


def main() -> int:
    import tensorflow as tf
    from efnas.vela.vela_compiler import run_vela

    parser = argparse.ArgumentParser(description="Ablation V1 SRAM profile exporter")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="edgeflownet_bilinear_eca_gate4x")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "ablation_v1_sram_profile"))
    parser.add_argument("--input_height", type=int, default=172)
    parser.add_argument("--input_width", type=int, default=224)
    parser.add_argument("--rep_dataset_samples", type=int, default=5)
    parser.add_argument("--optimise", choices=["Performance", "Size"], default="Size")
    parser.add_argument("--mode", choices=["basic", "verbose"], default="verbose")
    args = parser.parse_args()

    tf.compat.v1.disable_eager_execution()
    output_dir = Path(args.output_dir)
    size = f"{args.input_height}x{args.input_width}"
    tflite_path = output_dir / f"{args.variant}_{size}.tflite"
    vela_dir = output_dir / f"{args.variant}_{size}_vela"

    _export_variant(tflite_path, args.variant, args.input_height, args.input_width, args.rep_dataset_samples)
    sram_mb, inference_ms = run_vela(
        str(tflite_path),
        mode=args.mode,
        output_dir=str(vela_dir),
        optimise=args.optimise,
        silent=False,
    )
    summary = {
        "variant": args.variant,
        "resolution": size,
        "tflite": str(tflite_path),
        "vela_dir": str(vela_dir),
        "sram_mb": sram_mb,
        "inference_ms": inference_ms,
        "output": "AccumPreds(outputs)[...,0:2]",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.variant}_{size}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
