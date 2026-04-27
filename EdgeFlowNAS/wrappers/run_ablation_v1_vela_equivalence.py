"""Export original EdgeFlowNet A0 and EdgeFlowNAS A0, then compare with Vela."""

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MCU_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MCU_ROOT) not in sys.path:
    sys.path.insert(0, str(MCU_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablation V1 A0 Vela equivalence precheck")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "ablation_v1_vela_equivalence"))
    parser.add_argument("--input_height", type=int, default=352)
    parser.add_argument("--input_width", type=int, default=480)
    parser.add_argument("--rep_dataset_samples", type=int, default=3)
    parser.add_argument("--optimise", choices=["Performance", "Size"], default="Size")
    parser.add_argument("--mode", choices=["basic", "verbose"], default="verbose")
    return parser


def _convert_to_tflite(sess, input_ph, output_tensor, tflite_path: Path, rep_dataset_samples: int) -> None:
    import numpy as np
    import tensorflow as tf

    def representative_dataset():
        for _ in range(int(rep_dataset_samples)):
            yield [np.zeros(input_ph.shape.as_list(), dtype=np.float32)]

    converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [output_tensor])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(converter.convert())


def _export_original(tflite_path: Path, input_height: int, input_width: int, rep_dataset_samples: int) -> None:
    import tensorflow as tf

    edgeflownet_root = MCU_ROOT / "EdgeFlowNet" / "code"
    if str(edgeflownet_root) not in sys.path:
        sys.path.insert(0, str(edgeflownet_root))
    from network.MultiScaleResNet import MultiScaleResNet

    tf.compat.v1.reset_default_graph()
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_height, input_width, 6], name="Input")
    model = MultiScaleResNet(InputPH=input_ph, NumOut=2, InitNeurons=32, ExpansionFactor=2.0, NumSubBlocks=2, NumBlocks=1, UncType="LinearSoftplus")
    outputs = model.Network()
    final_output = outputs[-1]
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        _convert_to_tflite(sess, input_ph, final_output, tflite_path, rep_dataset_samples)


def _export_ablation(tflite_path: Path, input_height: int, input_width: int, rep_dataset_samples: int) -> None:
    import tensorflow as tf
    from efnas.network.ablation_edgeflownet_v1 import ABlationEdgeFlowNetV1

    tf.compat.v1.reset_default_graph()
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_height, input_width, 6], name="Input")
    is_training_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="IsTraining")
    model = ABlationEdgeFlowNetV1(
        input_ph=input_ph,
        is_training_ph=is_training_ph,
        num_out=4,
        variant_config={"name": "edgeflownet_deconv", "upsample_mode": "deconv", "bottleneck_eca": False, "gate_4x": False},
    )
    final_output = model.build()[-1]
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        _convert_to_tflite(sess, input_ph, final_output, tflite_path, rep_dataset_samples)


def main() -> int:
    import tensorflow as tf
    from tests.vela.vela_compiler import run_vela

    tf.compat.v1.disable_eager_execution()
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    original_tflite = output_dir / "original_edgeflownet_a0.tflite"
    ablation_tflite = output_dir / "edgeflownas_ablation_a0.tflite"
    _export_original(original_tflite, args.input_height, args.input_width, args.rep_dataset_samples)
    _export_ablation(ablation_tflite, args.input_height, args.input_width, args.rep_dataset_samples)

    original_vela_dir = output_dir / "original_vela"
    ablation_vela_dir = output_dir / "ablation_vela"
    original_sram, original_ms = run_vela(str(original_tflite), mode=args.mode, output_dir=str(original_vela_dir), optimise=args.optimise, silent=False)
    ablation_sram, ablation_ms = run_vela(str(ablation_tflite), mode=args.mode, output_dir=str(ablation_vela_dir), optimise=args.optimise, silent=False)
    summary = {
        "original": {"tflite": str(original_tflite), "vela_dir": str(original_vela_dir), "sram_mb": original_sram, "inference_ms": original_ms},
        "ablation": {"tflite": str(ablation_tflite), "vela_dir": str(ablation_vela_dir), "sram_mb": ablation_sram, "inference_ms": ablation_ms},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
