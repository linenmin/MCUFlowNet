"""Export a Supernet V3 reference subnet and compile it with Vela."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_arch_code(text: str):
    return [int(item.strip()) for item in str(text).replace(" ", ",").split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supernet V3 reference-subnet Vela precheck")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "supernet_v3_vela_reference"))
    parser.add_argument("--input_height", type=int, default=172)
    parser.add_argument("--input_width", type=int, default=224)
    parser.add_argument("--arch_code", default="2,1,0,0,0,0,1,1,1,1,1")
    parser.add_argument("--rep_dataset_samples", type=int, default=3)
    parser.add_argument("--optimise", choices=["Performance", "Size"], default="Size")
    parser.add_argument("--mode", choices=["basic", "verbose"], default="verbose")
    return parser


def _accumulate_predictions(preds):
    import tensorflow as tf

    pred_accum = None
    for pred in preds:
        if pred_accum is None:
            pred_accum = pred
        else:
            pred_accum = tf.compat.v1.image.resize_bilinear(
                pred_accum,
                [pred.shape[1], pred.shape[2]],
                align_corners=False,
                half_pixel_centers=False,
            )
            pred_accum = pred_accum + pred
    return pred_accum


def _export_v3(tflite_path: Path, input_height: int, input_width: int, arch_code, rep_dataset_samples: int) -> None:
    import numpy as np
    import tensorflow as tf

    from efnas.nas.search_space_v3 import validate_arch_code
    from efnas.network.base_layers import BaseLayers

    class FixedSubnetV3(BaseLayers):
        def __init__(self, input_ph, is_training_ph, arch_code, num_out=4):
            super().__init__(is_training_ph=is_training_ph)
            self.input_ph = input_ph
            self.arch_code = [int(item) for item in arch_code]
            self.num_out = int(num_out)
            self.init_neurons = 32
            self.expansion_factor = 2.0

        def _res_block(self, inputs, filters, name):
            with tf.compat.v1.variable_scope(name):
                net = self.conv_bn_relu(inputs=inputs, filters=filters, strides=(1, 1), name="conv1")
                net = self.conv(inputs=net, filters=filters, strides=(1, 1), activation=None, name="conv2")
                net = self.bn(inputs=net, name="bn2")
                return self.relu(inputs=tf.add(net, inputs, name="res_add"), name="res_relu")

        def _deep_choice_block(self, inputs, filters, choice_idx, name):
            net = inputs
            with tf.compat.v1.variable_scope(name):
                for block_idx in range(int(choice_idx) + 1):
                    net = self._res_block(inputs=net, filters=filters, name=f"branch{int(choice_idx) + 1}_block{block_idx + 1}")
            return net

        def _conv_bn_relu_dilated(self, inputs, filters, kernel_size, dilation_rate, name):
            with tf.compat.v1.variable_scope(name):
                net = tf.compat.v1.layers.conv2d(
                    inputs=inputs,
                    filters=int(filters),
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    dilation_rate=dilation_rate,
                    padding="same",
                    activation=None,
                    use_bias=False,
                    name="conv",
                )
                net = self.bn(inputs=net, name="bn")
                return self.relu(inputs=net, name="relu")

        def _e0_choice_block(self, inputs, filters, choice_idx, name):
            kernels = [(3, 3), (5, 5), (7, 7)]
            labels = ["k3", "k5", "k7"]
            idx = int(choice_idx)
            return self.conv_bn_relu(
                inputs=inputs,
                filters=filters,
                kernel_size=kernels[idx],
                strides=(2, 2),
                name=f"{name}/{labels[idx]}",
            )

        def _e1_choice_block(self, inputs, filters, choice_idx, name):
            idx = int(choice_idx)
            if idx == 0:
                return self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(2, 2), name=f"{name}/k3")
            if idx == 1:
                return self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(5, 5), strides=(2, 2), name=f"{name}/k5")
            dilated = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(2, 2), name=f"{name}/k3_down")
            return self._conv_bn_relu_dilated(
                inputs=dilated,
                filters=filters,
                kernel_size=(3, 3),
                dilation_rate=(2, 2),
                name=f"{name}/k3_dilated",
            )

        def _eca_block(self, inputs, kernel_size=3, name="eca"):
            channels = inputs.get_shape().as_list()[-1]
            pooled = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True, name=f"{name}_mean")
            pooled_1d = tf.reshape(pooled, [-1, 1, channels, 1], name=f"{name}_reshape_to_1d")
            attn = tf.compat.v1.layers.conv2d(
                inputs=pooled_1d,
                filters=1,
                kernel_size=(1, kernel_size),
                strides=(1, 1),
                padding="same",
                activation=None,
                name=f"{name}_conv1d",
            )
            attn = tf.reshape(attn, [-1, 1, 1, channels], name=f"{name}_reshape_back")
            return tf.multiply(inputs, tf.math.sigmoid(attn, name=f"{name}_sigmoid"), name=f"{name}_scale")

        def _global_broadcast_gate(self, context_inputs, target_inputs, target_filters, name):
            context = tf.reduce_mean(context_inputs, axis=[1, 2], keepdims=True, name=f"{name}_mean")
            gate = self.conv(
                inputs=context,
                filters=int(target_filters),
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=None,
                name=f"{name}_proj",
            )
            return tf.multiply(target_inputs, tf.math.sigmoid(gate, name=f"{name}_sigmoid"), name=f"{name}_scale")

        def _head_choice_conv(self, inputs, filters, choice_idx, name):
            kernel = (3, 3) if int(choice_idx) == 0 else (5, 5)
            label = "k3" if int(choice_idx) == 0 else "k5"
            return self.conv(inputs=inputs, filters=filters, kernel_size=kernel, strides=(1, 1), activation=None, name=f"{name}/{label}")

        def _head_choice_resize_conv(self, inputs, filters, choice_idx, name):
            kernel = (3, 3) if int(choice_idx) == 0 else (5, 5)
            label = "k3" if int(choice_idx) == 0 else "k5"
            return self.resize_conv(inputs=inputs, filters=filters, kernel_size=kernel, name=f"{name}/{label}")

        def build(self):
            arch = self.arch_code
            c1 = int(self.init_neurons * self.expansion_factor)
            c2 = int(c1 * self.expansion_factor)
            c3 = int(c2 * self.expansion_factor)
            c4 = int(c3 / self.expansion_factor)
            c5 = int(c4 / self.expansion_factor)
            h1_filters = max(1, int(c5 / self.expansion_factor))
            h2_filters = max(1, int(h1_filters / self.expansion_factor))
            with tf.compat.v1.variable_scope("supernet_backbone"):
                net = self._e0_choice_block(self.input_ph, self.init_neurons, arch[0], "E0")
                net = self._e1_choice_block(net, c1, arch[1], "E1")
                net = self._deep_choice_block(net, c1, arch[2], "EB0")
                net = self.conv(inputs=net, filters=c2, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down1Conv")
                net = self.relu(inputs=self.bn(inputs=net, name="Down1BN"), name="Down1ReLU")
                net = self._deep_choice_block(net, c2, arch[3], "EB1")
                net = self.conv(inputs=net, filters=c3, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down2Conv")
                net = self.relu(inputs=self.bn(inputs=net, name="Down2BN"), name="Down2ReLU")
                net_low = self._deep_choice_block(net, c3, arch[4], "DB0")
                net_low = self._eca_block(net_low, name="eca_bottleneck")
                net_mid = self.resize_conv(inputs=net_low, filters=c4, kernel_size=(3, 3), name="Up1")
                net_mid = self.relu(inputs=self.bn(inputs=net_mid, name="Up1BN"), name="Up1ReLU")
                net_mid = self._deep_choice_block(net_mid, c4, arch[5], "DB1")
                net_high = self.resize_conv(inputs=net_mid, filters=c5, kernel_size=(3, 3), name="Up2")
                net_high = self.relu(inputs=self.bn(inputs=net_high, name="Up2BN"), name="Up2ReLU")
                net_high = self._global_broadcast_gate(net_low, net_high, c5, name="global_gate_4x")
            with tf.compat.v1.variable_scope("supernet_head"):
                out_1_4 = self._head_choice_conv(net_high, self.num_out, arch[6], "H0Out")
                h1 = self._head_choice_resize_conv(net_high, h1_filters, arch[7], "H1")
                h1 = self.relu(inputs=self.bn(inputs=h1, name="H1BN"), name="H1ReLU")
                out_1_2 = self._head_choice_conv(h1, self.num_out, arch[8], "H1Out")
                h2 = self._head_choice_resize_conv(h1, h2_filters, arch[9], "H2")
                h2 = self.relu(inputs=self.bn(inputs=h2, name="H2BN"), name="H2ReLU")
                out_1_1 = self._head_choice_conv(h2, self.num_out, arch[10], "H2Out")
            return [out_1_4, out_1_2, out_1_1]

    validate_arch_code(arch_code)
    tf.compat.v1.reset_default_graph()
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_height, input_width, 6], name="Input")
    is_training_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="IsTraining")
    model = FixedSubnetV3(
        input_ph=input_ph,
        is_training_ph=is_training_ph,
        arch_code=arch_code,
        num_out=4,
    )
    final_output = _accumulate_predictions(model.build())[:, :, :, :2]

    def representative_dataset():
        for _ in range(int(rep_dataset_samples)):
            yield [np.zeros(input_ph.shape.as_list(), dtype=np.float32)]

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        tflite_path.write_bytes(converter.convert())


def main() -> int:
    import tensorflow as tf
    from tests.vela.vela_compiler import run_vela

    tf.compat.v1.disable_eager_execution()
    args = _build_parser().parse_args()
    arch_code = _parse_arch_code(args.arch_code)
    output_dir = Path(args.output_dir)
    tflite_path = output_dir / "supernet_v3_reference.tflite"
    _export_v3(tflite_path, args.input_height, args.input_width, arch_code, args.rep_dataset_samples)
    vela_dir = output_dir / "vela"
    sram, inference_ms = run_vela(str(tflite_path), mode=args.mode, output_dir=str(vela_dir), optimise=args.optimise, silent=False)
    summary = {
        "arch_code": arch_code,
        "input_shape": [args.input_height, args.input_width, 6],
        "tflite": str(tflite_path),
        "vela_dir": str(vela_dir),
        "sram_mb": sram,
        "inference_ms": inference_ms,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
