"""Fixed-architecture V3 model for scratch retraining.

This mirrors :mod:`efnas.network.MultiScaleResNet_supernet_v3` but materializes
only the selected branches for a concrete 11D architecture code.
"""

from typing import Dict, Sequence

import tensorflow as tf

from efnas.nas.search_space_v3 import validate_arch_code
from efnas.network.base_layers import BaseLayers


class FixedArchModelV3(BaseLayers):
    """Build one hard-routed V3 subnet with V3 stem/depth/head semantics."""

    def __init__(
        self,
        input_ph,
        is_training_ph,
        arch_code: Sequence[int],
        num_out: int = 4,
        init_neurons: int = 32,
        expansion_factor: float = 2.0,
        suffix: str = "",
    ):
        super().__init__(is_training_ph=is_training_ph, suffix=suffix)
        self.input_ph = input_ph
        self.arch_code = [int(v) for v in arch_code]
        validate_arch_code(self.arch_code)
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

    def _deep_choice_block(self, inputs, filters, choice_idx, name):
        choice = int(choice_idx)
        if choice not in (0, 1, 2):
            raise ValueError(f"invalid deep choice: {choice_idx}")
        with tf.compat.v1.variable_scope(name):
            out = inputs
            for block_idx in range(choice + 1):
                out = self._res_block(inputs=out, filters=filters, name=f"branch{choice + 1}_block{block_idx + 1}")
            return out

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

    def _e0_choice_block(self, inputs, filters, choice_idx, name):
        choice = int(choice_idx)
        if choice not in (0, 1, 2):
            raise ValueError(f"invalid E0 choice: {choice_idx}")
        kernels = [(3, 3), (5, 5), (7, 7)]
        names = ["k3", "k5", "k7"]
        with tf.compat.v1.variable_scope(name):
            return self.conv_bn_relu(
                inputs=inputs,
                filters=filters,
                kernel_size=kernels[choice],
                strides=(2, 2),
                name=names[choice],
            )

    def _e1_choice_block(self, inputs, filters, choice_idx, name):
        choice = int(choice_idx)
        if choice not in (0, 1, 2):
            raise ValueError(f"invalid E1 choice: {choice_idx}")
        with tf.compat.v1.variable_scope(name):
            if choice == 0:
                return self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(2, 2), name="k3")
            if choice == 1:
                return self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(5, 5), strides=(2, 2), name="k5")
            net = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(2, 2), name="k3_down")
            return self._conv_bn_relu_dilated(
                inputs=net,
                filters=filters,
                kernel_size=(3, 3),
                dilation_rate=(2, 2),
                name="k3_dilated",
            )

    def _head_choice_conv(self, inputs, filters, choice_idx, name):
        choice = int(choice_idx)
        if choice not in (0, 1):
            raise ValueError(f"invalid head choice: {choice_idx}")
        with tf.compat.v1.variable_scope(name):
            if choice == 0:
                return self.conv(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(1, 1), activation=None, name="k3")
            return self.conv(inputs=inputs, filters=filters, kernel_size=(5, 5), strides=(1, 1), activation=None, name="k5")

    def _head_choice_resize_conv(self, inputs, filters, choice_idx, name):
        choice = int(choice_idx)
        if choice not in (0, 1):
            raise ValueError(f"invalid resize head choice: {choice_idx}")
        with tf.compat.v1.variable_scope(name):
            if choice == 0:
                return self.resize_conv(inputs=inputs, filters=filters, kernel_size=(3, 3), name="k3")
            return self.resize_conv(inputs=inputs, filters=filters, kernel_size=(5, 5), name="k5")

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

        with tf.compat.v1.variable_scope("supernet_backbone"):
            net = self._e0_choice_block(inputs=self.input_ph, filters=self.init_neurons, choice_idx=arch[0], name="E0")
            net = self._e1_choice_block(inputs=net, filters=c1, choice_idx=arch[1], name="E1")
            net = self._deep_choice_block(inputs=net, filters=c1, choice_idx=arch[2], name="EB0")

            net = self.conv(inputs=net, filters=c2, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down1Conv")
            net = self.bn(inputs=net, name="Down1BN")
            net = self.relu(inputs=net, name="Down1ReLU")
            net = self._deep_choice_block(inputs=net, filters=c2, choice_idx=arch[3], name="EB1")

            net = self.conv(inputs=net, filters=c3, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down2Conv")
            net = self.bn(inputs=net, name="Down2BN")
            net = self.relu(inputs=net, name="Down2ReLU")
            net_low = self._deep_choice_block(inputs=net, filters=c3, choice_idx=arch[4], name="DB0")
            net_low = self._eca_block(inputs=net_low, kernel_size=3, name="eca_bottleneck")
            bottleneck_context = net_low

            net_mid = self.resize_conv(inputs=net_low, filters=c4, kernel_size=(3, 3), name="Up1")
            net_mid = self.bn(inputs=net_mid, name="Up1BN")
            net_mid = self.relu(inputs=net_mid, name="Up1ReLU")
            net_mid = self._deep_choice_block(inputs=net_mid, filters=c4, choice_idx=arch[5], name="DB1")

            net_high = self.resize_conv(inputs=net_mid, filters=c5, kernel_size=(3, 3), name="Up2")
            net_high = self.bn(inputs=net_high, name="Up2BN")
            net_high = self.relu(inputs=net_high, name="Up2ReLU")
            net_high = self._global_broadcast_gate(
                context_inputs=bottleneck_context,
                target_inputs=net_high,
                target_filters=c5,
                name="global_gate_4x",
            )

        with tf.compat.v1.variable_scope("supernet_head"):
            out_1_4 = self._head_choice_conv(inputs=net_high, filters=self.num_out, choice_idx=arch[6], name="H0Out")
            h1 = self._head_choice_resize_conv(inputs=net_high, filters=h1_filters, choice_idx=arch[7], name="H1")
            h1 = self.bn(inputs=h1, name="H1BN")
            h1 = self.relu(inputs=h1, name="H1ReLU")
            out_1_2 = self._head_choice_conv(inputs=h1, filters=self.num_out, choice_idx=arch[8], name="H1Out")
            h2 = self._head_choice_resize_conv(inputs=h1, filters=h2_filters, choice_idx=arch[9], name="H2")
            h2 = self.bn(inputs=h2, name="H2BN")
            h2 = self.relu(inputs=h2, name="H2ReLU")
            out_1_1 = self._head_choice_conv(inputs=h2, filters=self.num_out, choice_idx=arch[10], name="H2Out")

        self._feature_pyramid = {"bottleneck": net_low, "decoder_8x": net_mid, "decoder_4x": net_high}
        self._preds = [out_1_4, out_1_2, out_1_1]
        return self._preds

    def feature_pyramid(self) -> Dict[str, tf.Tensor]:
        if self._preds is None:
            self.build()
        return self._feature_pyramid
