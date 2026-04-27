"""Ablation V1 EdgeFlowNet-style backbones."""

from typing import Dict, Iterable, List, Optional

import tensorflow as tf

from efnas.network.base_layers import BaseLayers


DEFAULT_ABLATION_VARIANTS: List[Dict[str, object]] = [
    {"name": "edgeflownet_deconv", "upsample_mode": "deconv", "bottleneck_eca": False, "gate_4x": False},
    {"name": "edgeflownet_bilinear", "upsample_mode": "bilinear", "bottleneck_eca": False, "gate_4x": False},
    {"name": "edgeflownet_bilinear_eca", "upsample_mode": "bilinear", "bottleneck_eca": True, "gate_4x": False},
    {"name": "edgeflownet_bilinear_eca_gate4x", "upsample_mode": "bilinear", "bottleneck_eca": True, "gate_4x": True},
]


def build_ablation_variants(raw_variants: Optional[Iterable[Dict[str, object]]]) -> List[Dict[str, object]]:
    """Normalize ablation variant configs."""
    variants = list(DEFAULT_ABLATION_VARIANTS if raw_variants is None else raw_variants)
    normalized: List[Dict[str, object]] = []
    for item in variants:
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("ablation variant requires name")
        mode = str(item.get("upsample_mode", "bilinear")).strip().lower()
        if mode not in ("deconv", "bilinear"):
            raise ValueError(f"unsupported upsample_mode for {name}: {mode}")
        normalized.append(
            {
                "name": name,
                "upsample_mode": mode,
                "bottleneck_eca": bool(item.get("bottleneck_eca", False)),
                "gate_4x": bool(item.get("gate_4x", False)),
            }
        )
    return normalized


class ABlationEdgeFlowNetV1(BaseLayers):
    """EdgeFlowNet-style model with explicit ablation switches."""

    def __init__(
        self,
        input_ph,
        is_training_ph,
        num_out: int,
        variant_config: Dict[str, object],
        init_neurons: int = 32,
        expansion_factor: float = 2.0,
        num_sub_blocks: int = 2,
    ):
        super().__init__(is_training_ph=is_training_ph)
        self.input_ph = input_ph
        self.num_out = int(num_out)
        self.variant_config = dict(variant_config)
        self.variant_name = str(self.variant_config.get("name", "ablation_model"))
        self.upsample_mode = str(self.variant_config.get("upsample_mode", "bilinear")).strip().lower()
        self.bottleneck_eca = bool(self.variant_config.get("bottleneck_eca", False))
        self.gate_4x = bool(self.variant_config.get("gate_4x", False))
        self.init_neurons = int(init_neurons)
        self.expansion_factor = float(expansion_factor)
        self.num_sub_blocks = int(num_sub_blocks)
        self._preds = None
        self._feature_pyramid = None

    def _res_block(self, inputs, filters: int, name: str):
        with tf.compat.v1.variable_scope(name):
            net = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(1, 1), name="conv1")
            net = self.conv(inputs=net, filters=filters, kernel_size=(3, 3), strides=(1, 1), activation=None, name="conv2")
            net = self.bn(inputs=net, name="bn2")
            net = tf.add(net, inputs, name="add")
            return self.relu(inputs=net, name="relu")

    def _res_block_transpose(self, inputs, filters: int, name: str):
        with tf.compat.v1.variable_scope(name):
            net = self.conv_transpose_bn_relu(
                inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(1, 1), name="deconv1"
            )
            net = self.conv_transpose(
                inputs=net, filters=filters, kernel_size=(3, 3), strides=(1, 1), activation=None, name="deconv2"
            )
            net = self.bn(inputs=net, name="bn2")
            net = tf.add(net, inputs, name="add")
            return self.relu(inputs=net, name="relu")

    def _decoder_block(self, inputs, filters: int, name: str):
        if self.upsample_mode == "deconv":
            return self.conv_transpose(inputs=inputs, filters=filters, kernel_size=(3, 3), name=name)
        net = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(3, 3), name=name)
        net = self.bn(inputs=net, name=f"{name}_bn")
        return self.relu(inputs=net, name=f"{name}_relu")

    def _head_upsample(self, inputs, filters: int, kernel_size, name: str):
        if self.upsample_mode == "deconv":
            return self.conv_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size, name=name)
        return self.resize_conv(inputs=inputs, filters=filters, kernel_size=kernel_size, name=name)

    def _head_same_scale(self, inputs, filters: int, kernel_size, name: str):
        if self.upsample_mode == "deconv":
            return self.conv_transpose(
                inputs=inputs, filters=filters, kernel_size=kernel_size, strides=(1, 1), activation=None, name=name
            )
        return self.conv(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=(1, 1), activation=None, name=name)

    def _eca_block(self, inputs, kernel_size: int = 3, name: str = "eca"):
        channels = inputs.get_shape().as_list()[-1]
        if channels is None:
            raise ValueError("ECA block requires static channel dimension")
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
        attn = tf.math.sigmoid(attn, name=f"{name}_sigmoid")
        return tf.multiply(inputs, attn, name=f"{name}_scale")

    def _global_broadcast_gate(self, context_inputs, target_inputs, target_filters: int, name: str):
        context = tf.reduce_mean(context_inputs, axis=[1, 2], keepdims=True, name=f"{name}_mean")
        gate = self.conv(
            inputs=context,
            filters=int(target_filters),
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=None,
            name=f"{name}_proj",
        )
        gate = tf.math.sigmoid(gate, name=f"{name}_sigmoid")
        return tf.multiply(target_inputs, gate, name=f"{name}_scale")

    def build(self):
        if self._preds is not None:
            return self._preds

        num_filters = self.init_neurons
        with tf.compat.v1.variable_scope("ablation_backbone"):
            net = self.conv_bn_relu(inputs=self.input_ph, filters=num_filters, kernel_size=(7, 7), name="E0")
            num_filters = int(num_filters * self.expansion_factor)
            net = self.conv_bn_relu(inputs=net, filters=num_filters, kernel_size=(5, 5), name="E1")

            for idx in range(self.num_sub_blocks):
                net = self._res_block(inputs=net, filters=num_filters, name=f"EncoderRes{idx}")
                num_filters = int(num_filters * self.expansion_factor)
                net = self.conv(inputs=net, filters=num_filters, kernel_size=(3, 3), activation=None, name=f"Down{idx}")

            for idx in range(self.num_sub_blocks):
                if self.upsample_mode == "deconv":
                    net = self._res_block_transpose(inputs=net, filters=num_filters, name=f"DecoderRes{idx}")
                else:
                    net = self._res_block(inputs=net, filters=num_filters, name=f"DecoderRes{idx}")
                num_filters = int(num_filters / self.expansion_factor)
                net = self._decoder_block(inputs=net, filters=num_filters, name=f"Up{idx}")

            if self.bottleneck_eca:
                net = self._eca_block(inputs=net, kernel_size=3, name="eca_bottleneck")
            bottleneck_context = net
            feat_low = net
            out_1_4 = self._head_same_scale(inputs=net, filters=self.num_out, kernel_size=(7, 7), name="OutLow")

            num_filters = int(num_filters / self.expansion_factor)
            net = self._head_upsample(inputs=net, filters=num_filters, kernel_size=(5, 5), name="UpMid")
            if self.upsample_mode == "bilinear":
                net = self.bn(inputs=net, name="UpMid_bn")
                net = self.relu(inputs=net, name="UpMid_relu")
            if self.gate_4x:
                net = self._global_broadcast_gate(
                    context_inputs=bottleneck_context,
                    target_inputs=net,
                    target_filters=num_filters,
                    name="global_gate_4x",
                )
            feat_mid = net
            out_1_2 = self._head_same_scale(inputs=net, filters=self.num_out, kernel_size=(7, 7), name="OutMid")

            num_filters = int(num_filters / self.expansion_factor)
            net = self._head_upsample(inputs=net, filters=num_filters, kernel_size=(7, 7), name="UpHigh")
            if self.upsample_mode == "bilinear":
                net = self.bn(inputs=net, name="UpHigh_bn")
                net = self.relu(inputs=net, name="UpHigh_relu")
            feat_high = net
            out_1_1 = self._head_same_scale(inputs=net, filters=self.num_out, kernel_size=(7, 7), name="OutHigh")

        self._feature_pyramid = {"bottleneck": feat_low, "decoder_4x": feat_mid, "decoder_2x": feat_high}
        self._preds = [out_1_4, out_1_2, out_1_1]
        return self._preds

    def feature_pyramid(self):
        if self._feature_pyramid is None:
            self.build()
        return self._feature_pyramid


AblationEdgeFlowNetV1 = ABlationEdgeFlowNetV1
