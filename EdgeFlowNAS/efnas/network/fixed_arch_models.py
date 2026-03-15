"""Fixed-arch retrain models for joint comparison runs."""

from typing import Dict, List, Sequence

import tensorflow as tf

from efnas.network.base_layers import BaseLayers


SUPPORTED_VARIANTS = (
    "baseline",
    "globalgate4x_bneckeca",
    "globalgate4x_bneckeca_skip8x4x",
    "globalgate4x_bneckeca_skip8x4x2x",
    "globalgate8x4x_bneckeca",
    "globalgate8x4x_bneckeca_skip8x",
    "globalgate8x4x_bneckeca_skip8x4x",
    "globalgate4x_dual_eca8_bneckeca",
    "skip8x4x",
)


class FixedArchModel(BaseLayers):
    """Build one fixed-arch model, optionally with extra gate/skip modules."""

    def __init__(
        self,
        input_ph,
        is_training_ph,
        arch_code: Sequence[int],
        variant: str,
        num_out: int = 4,
        init_neurons: int = 32,
        expansion_factor: float = 2.0,
        suffix: str = "",
    ):
        super().__init__(is_training_ph=is_training_ph, suffix=suffix)
        if len(arch_code) != 9:
            raise ValueError("arch_code length must be 9")
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        self.input_ph = input_ph
        self.arch_code = [int(v) for v in arch_code]
        self.variant = str(variant)
        self.num_out = int(num_out)
        self.init_neurons = int(init_neurons)
        self.expansion_factor = float(expansion_factor)
        self._preds = None
        self._feature_pyramid = None

    def _res_block(self, inputs, filters, name):
        with tf.compat.v1.variable_scope(name):
            net = self.conv_bn_relu(inputs=inputs, filters=filters, strides=(1, 1), name="conv1")
            net = self.conv(inputs=net, filters=filters, strides=(1, 1), activation=None, name="conv2")
            net = self.bn(inputs=net, name="bn2")
            net = tf.add(net, inputs, name="res_add")
            return self.relu(inputs=net, name="res_relu")

    def _deep_choice_block(self, inputs, filters, choice_idx: int, name: str):
        choice = int(choice_idx)
        if choice not in (0, 1, 2):
            raise ValueError(f"invalid deep choice: {choice_idx}")
        with tf.compat.v1.variable_scope(name):
            out1 = self._res_block(inputs=inputs, filters=filters, name="branch1_block1")
            out2 = self._res_block(inputs=inputs, filters=filters, name="branch2_block1")
            out2 = self._res_block(inputs=out2, filters=filters, name="branch2_block2")
            out3 = self._res_block(inputs=inputs, filters=filters, name="branch3_block1")
            out3 = self._res_block(inputs=out3, filters=filters, name="branch3_block2")
            out3 = self._res_block(inputs=out3, filters=filters, name="branch3_block3")
            return [out1, out2, out3][choice]

    def _head_choice_conv(self, inputs, filters, choice_idx: int, name: str):
        choice = int(choice_idx)
        if choice not in (0, 1, 2):
            raise ValueError(f"invalid head choice: {choice_idx}")
        with tf.compat.v1.variable_scope(name):
            conv7 = self.conv(
                inputs=inputs, filters=filters, kernel_size=(7, 7), strides=(1, 1), activation=None, name="k7"
            )
            conv5 = self.conv(
                inputs=inputs, filters=filters, kernel_size=(5, 5), strides=(1, 1), activation=None, name="k5"
            )
            conv3 = self.conv(
                inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(1, 1), activation=None, name="k3"
            )
            return [conv3, conv5, conv7][choice]

    def _head_choice_resize_conv(self, inputs, filters, choice_idx: int, name: str):
        choice = int(choice_idx)
        if choice not in (0, 1, 2):
            raise ValueError(f"invalid head choice: {choice_idx}")
        with tf.compat.v1.variable_scope(name):
            conv7 = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(7, 7), name="k7")
            conv5 = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(5, 5), name="k5")
            conv3 = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(3, 3), name="k3")
            return [conv3, conv5, conv7][choice]

    def _compressed_additive_skip(self, decoder_inputs, encoder_inputs, filters, name):
        hidden_filters = max(int(filters) // 2, 8)
        projected = self.conv(
            inputs=encoder_inputs,
            filters=hidden_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=None,
            name=f"{name}_squeeze",
        )
        projected = self.bn(inputs=projected, name=f"{name}_squeeze_bn")
        projected = self.relu(inputs=projected, name=f"{name}_squeeze_relu")
        projected = self.conv(
            inputs=projected,
            filters=int(filters),
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=None,
            name=f"{name}_expand",
        )
        projected = self.bn(inputs=projected, name=f"{name}_expand_bn")

        decoder_shape = decoder_inputs.get_shape().as_list()
        projected_shape = projected.get_shape().as_list()
        pad_h = int(decoder_shape[1]) - int(projected_shape[1])
        pad_w = int(decoder_shape[2]) - int(projected_shape[2])
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"skip shape mismatch is not pad-only: decoder={decoder_shape}, projected={projected_shape}"
            )
        if pad_h > 0 or pad_w > 0:
            projected = tf.pad(
                projected,
                paddings=[[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
                mode="CONSTANT",
                name=f"{name}_pad",
            )
        return tf.add(decoder_inputs, projected, name=f"{name}_add")

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

    def _use_bneck_eca(self) -> bool:
        return self.variant in (
            "globalgate4x_bneckeca",
            "globalgate4x_bneckeca_skip8x4x",
            "globalgate4x_bneckeca_skip8x4x2x",
            "globalgate8x4x_bneckeca",
            "globalgate8x4x_bneckeca_skip8x",
            "globalgate8x4x_bneckeca_skip8x4x",
            "globalgate4x_dual_eca8_bneckeca",
        )

    def _use_encoder_eca_8x(self) -> bool:
        return self.variant == "globalgate4x_dual_eca8_bneckeca"

    def _use_global_gate_8x(self) -> bool:
        return self.variant in (
            "globalgate8x4x_bneckeca",
            "globalgate8x4x_bneckeca_skip8x",
            "globalgate8x4x_bneckeca_skip8x4x",
        )

    def _use_global_gate_4x(self) -> bool:
        return self.variant in (
            "globalgate4x_bneckeca",
            "globalgate4x_bneckeca_skip8x4x",
            "globalgate4x_bneckeca_skip8x4x2x",
            "globalgate8x4x_bneckeca",
            "globalgate8x4x_bneckeca_skip8x",
            "globalgate8x4x_bneckeca_skip8x4x",
            "globalgate4x_dual_eca8_bneckeca",
        )

    def _use_skip_8x(self) -> bool:
        return self.variant in (
            "globalgate4x_bneckeca_skip8x4x",
            "globalgate4x_bneckeca_skip8x4x2x",
            "globalgate8x4x_bneckeca_skip8x",
            "globalgate8x4x_bneckeca_skip8x4x",
            "skip8x4x",
        )

    def _use_skip_4x(self) -> bool:
        return self.variant in (
            "globalgate4x_bneckeca_skip8x4x",
            "globalgate4x_bneckeca_skip8x4x2x",
            "globalgate8x4x_bneckeca_skip8x4x",
            "skip8x4x",
        )

    def _use_skip_2x(self) -> bool:
        return self.variant == "globalgate4x_bneckeca_skip8x4x2x"

    def build(self):
        if self._preds is not None:
            return self._preds

        arch = self.arch_code
        c1 = int(self.init_neurons * self.expansion_factor)
        c2 = int(c1 * self.expansion_factor)
        c3 = int(c2 * self.expansion_factor)
        c4 = int(c3 / self.expansion_factor)
        c5 = int(c4 / self.expansion_factor)
        h1_filters = max(1, int(c5 / self.expansion_factor))
        h2_filters = max(1, int(h1_filters / self.expansion_factor))

        with tf.compat.v1.variable_scope("fixed_backbone"):
            net = self.conv_bn_relu(
                inputs=self.input_ph,
                filters=self.init_neurons,
                kernel_size=(7, 7),
                strides=(2, 2),
                name="E0",
            )
            skip_2x = net

            net = self.conv_bn_relu(inputs=net, filters=c1, kernel_size=(5, 5), strides=(2, 2), name="E1")
            net = self._deep_choice_block(inputs=net, filters=c1, choice_idx=arch[0], name="EB0")
            skip_4x = net

            net = self.conv(inputs=net, filters=c2, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down1Conv")
            net = self.bn(inputs=net, name="Down1BN")
            net = self.relu(inputs=net, name="Down1ReLU")
            net = self._deep_choice_block(inputs=net, filters=c2, choice_idx=arch[1], name="EB1")
            if self._use_encoder_eca_8x():
                net = self._eca_block(inputs=net, kernel_size=3, name="eca_encoder_8x")
            skip_8x = net

            net = self.conv(inputs=net, filters=c3, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down2Conv")
            net = self.bn(inputs=net, name="Down2BN")
            net = self.relu(inputs=net, name="Down2ReLU")
            net_low = self._deep_choice_block(inputs=net, filters=c3, choice_idx=arch[2], name="DB0")
            if self._use_bneck_eca():
                net_low = self._eca_block(inputs=net_low, kernel_size=3, name="eca_bottleneck")
            bottleneck_context = net_low

            net_mid = self.resize_conv(inputs=net_low, filters=c4, kernel_size=(3, 3), name="Up1")
            net_mid = self.bn(inputs=net_mid, name="Up1BN")
            net_mid = self.relu(inputs=net_mid, name="Up1ReLU")
            if self._use_global_gate_8x():
                net_mid = self._global_broadcast_gate(
                    context_inputs=bottleneck_context,
                    target_inputs=net_mid,
                    target_filters=c4,
                    name="global_gate_8x",
                )
            if self._use_skip_8x():
                net_mid = self._compressed_additive_skip(
                    decoder_inputs=net_mid,
                    encoder_inputs=skip_8x,
                    filters=c4,
                    name="skip_8x",
                )
            net_mid = self._deep_choice_block(inputs=net_mid, filters=c4, choice_idx=arch[3], name="DB1")

            net_high = self.resize_conv(inputs=net_mid, filters=c5, kernel_size=(3, 3), name="Up2")
            net_high = self.bn(inputs=net_high, name="Up2BN")
            net_high = self.relu(inputs=net_high, name="Up2ReLU")
            if self._use_global_gate_4x():
                net_high = self._global_broadcast_gate(
                    context_inputs=bottleneck_context,
                    target_inputs=net_high,
                    target_filters=c5,
                    name="global_gate_4x",
                )
            if self._use_skip_4x():
                net_high = self._compressed_additive_skip(
                    decoder_inputs=net_high,
                    encoder_inputs=skip_4x,
                    filters=c5,
                    name="skip_4x",
                )

        with tf.compat.v1.variable_scope("fixed_head"):
            out_1_4 = self._head_choice_conv(inputs=net_high, filters=self.num_out, choice_idx=arch[4], name="H0Out")

            h1 = self._head_choice_resize_conv(inputs=net_high, filters=h1_filters, choice_idx=arch[5], name="H1")
            h1 = self.bn(inputs=h1, name="H1BN")
            h1 = self.relu(inputs=h1, name="H1ReLU")
            if self._use_skip_2x():
                h1 = self._compressed_additive_skip(
                    decoder_inputs=h1,
                    encoder_inputs=skip_2x,
                    filters=h1_filters,
                    name="skip_2x",
                )
            out_1_2 = self._head_choice_conv(inputs=h1, filters=self.num_out, choice_idx=arch[6], name="H1Out")

            h2 = self._head_choice_resize_conv(inputs=h1, filters=h2_filters, choice_idx=arch[7], name="H2")
            h2 = self.bn(inputs=h2, name="H2BN")
            h2 = self.relu(inputs=h2, name="H2ReLU")
            out_1_1 = self._head_choice_conv(inputs=h2, filters=self.num_out, choice_idx=arch[8], name="H2Out")

        self._feature_pyramid = {
            "skip_2x": skip_2x,
            "skip_4x": skip_4x,
            "skip_8x": skip_8x,
            "bottleneck": net_low,
            "decoder_8x": net_mid,
            "decoder_4x": net_high,
        }
        self._preds = [out_1_4, out_1_2, out_1_1]
        return self._preds

    def feature_pyramid(self) -> Dict[str, tf.Tensor]:
        if self._preds is None:
            self.build()
        return self._feature_pyramid
