"""Bilinear-aligned supernet network implementation."""

import tensorflow as tf

from code.network.base_layers import BaseLayers


class MultiScaleResNetSupernet(BaseLayers):
    """Bilinear-style supernet with 9 architecture choices."""

    def __init__(
        self,
        input_ph,
        arch_code_ph,
        is_training_ph,
        num_out=4,
        init_neurons=32,
        expansion_factor=2.0,
        suffix="",
    ):
        super().__init__(is_training_ph=is_training_ph, suffix=suffix)
        self.input_ph = input_ph
        self.arch_code_ph = arch_code_ph
        self.num_out = int(num_out)
        self.init_neurons = int(init_neurons)
        self.expansion_factor = float(expansion_factor)
        self._preds = None
        self._feature_pyramid = None

    def _select_by_index(self, candidates, index_tensor, name):
        """Select one candidate tensor by one-hot gate."""
        stacked = tf.stack(candidates, axis=0, name=f"{name}_stack")
        weights = tf.one_hot(index_tensor, depth=len(candidates), dtype=stacked.dtype, name=f"{name}_onehot")
        weights = tf.reshape(weights, [len(candidates), 1, 1, 1, 1], name=f"{name}_reshape")
        return tf.reduce_sum(stacked * weights, axis=0, name=f"{name}_mix")

    def _res_block(self, inputs, filters, name):
        """Basic residual block."""
        with tf.compat.v1.variable_scope(name):
            net = self.conv_bn_relu(inputs=inputs, filters=filters, strides=(1, 1), name="conv1")
            net = self.conv(inputs=net, filters=filters, strides=(1, 1), activation=None, name="conv2")
            net = self.bn(inputs=net, name="bn2")
            net = tf.add(net, inputs, name="res_add")
            return self.relu(inputs=net, name="res_relu")

    def _deep_choice_block(self, inputs, filters, choice_idx, name):
        """Depth choice with decoupled branches: deep1/deep2/deep3."""
        with tf.compat.v1.variable_scope(name):
            # 三个深度分支共享输入，避免原嵌套路径的梯度串扰。
            out1 = self._res_block(inputs=inputs, filters=filters, name="branch1_block1")

            out2 = self._res_block(inputs=inputs, filters=filters, name="branch2_block1")
            out2 = self._res_block(inputs=out2, filters=filters, name="branch2_block2")

            out3 = self._res_block(inputs=inputs, filters=filters, name="branch3_block1")
            out3 = self._res_block(inputs=out3, filters=filters, name="branch3_block2")
            out3 = self._res_block(inputs=out3, filters=filters, name="branch3_block3")

            return self._select_by_index([out1, out2, out3], choice_idx, name="deep_select")

    def _head_choice_conv(self, inputs, filters, choice_idx, name):
        """Kernel choice conv: 3x3 / 5x5 / 7x7."""
        with tf.compat.v1.variable_scope(name):
            conv7 = self.conv(inputs=inputs, filters=filters, kernel_size=(7, 7), strides=(1, 1), activation=None, name="k7")
            conv5 = self.conv(inputs=inputs, filters=filters, kernel_size=(5, 5), strides=(1, 1), activation=None, name="k5")
            conv3 = self.conv(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(1, 1), activation=None, name="k3")
            return self._select_by_index([conv3, conv5, conv7], choice_idx, name="kernel_select")

    def _head_choice_resize_conv(self, inputs, filters, choice_idx, name):
        """Kernel choice resize-conv: upsample x2 then conv (3x3 / 5x5 / 7x7)."""
        with tf.compat.v1.variable_scope(name):
            conv7 = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(7, 7), name="k7")
            conv5 = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(5, 5), name="k5")
            conv3 = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(3, 3), name="k3")
            return self._select_by_index([conv3, conv5, conv7], choice_idx, name="kernel_select")

    def build(self):
        """Build supernet forward graph."""
        if self._preds is not None:
            return self._preds

        arch = self.arch_code_ph
        c1 = int(self.init_neurons * self.expansion_factor)
        c2 = int(c1 * self.expansion_factor)
        c3 = int(c2 * self.expansion_factor)
        c4 = int(c3 / self.expansion_factor)
        c5 = int(c4 / self.expansion_factor)
        h1_filters = max(1, int(c5 / self.expansion_factor))
        h2_filters = max(1, int(h1_filters / self.expansion_factor))

        with tf.compat.v1.variable_scope("supernet_backbone"):
            # Align E0/E1 downsample semantics with bilinear baseline.
            net = self.conv_bn_relu(inputs=self.input_ph, filters=self.init_neurons, kernel_size=(7, 7), strides=(2, 2), name="E0")
            net = self.conv_bn_relu(inputs=net, filters=c1, kernel_size=(5, 5), strides=(2, 2), name="E1")
            net = self._deep_choice_block(inputs=net, filters=c1, choice_idx=arch[0], name="EB0")

            net = self.conv(inputs=net, filters=c2, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down1Conv")
            net = self.bn(inputs=net, name="Down1BN")
            net = self.relu(inputs=net, name="Down1ReLU")
            net = self._deep_choice_block(inputs=net, filters=c2, choice_idx=arch[1], name="EB1")

            net = self.conv(inputs=net, filters=c3, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down2Conv")
            net = self.bn(inputs=net, name="Down2BN")
            net = self.relu(inputs=net, name="Down2ReLU")
            net_low = self._deep_choice_block(inputs=net, filters=c3, choice_idx=arch[2], name="DB0")

            net_mid = self.resize_conv(inputs=net_low, filters=c4, kernel_size=(3, 3), name="Up1")
            net_mid = self.bn(inputs=net_mid, name="Up1BN")
            net_mid = self.relu(inputs=net_mid, name="Up1ReLU")
            net_mid = self._deep_choice_block(inputs=net_mid, filters=c4, choice_idx=arch[3], name="DB1")

            net_high = self.resize_conv(inputs=net_mid, filters=c5, kernel_size=(3, 3), name="Up2")
            net_high = self.bn(inputs=net_high, name="Up2BN")
            net_high = self.relu(inputs=net_high, name="Up2ReLU")

        with tf.compat.v1.variable_scope("supernet_head"):
            # Keep 5 head choices while aligning the channel decay path.
            out_1_4 = self._head_choice_conv(inputs=net_high, filters=self.num_out, choice_idx=arch[4], name="H0Out")

            h1 = self._head_choice_resize_conv(inputs=net_high, filters=h1_filters, choice_idx=arch[5], name="H1")
            h1 = self.bn(inputs=h1, name="H1BN")
            h1 = self.relu(inputs=h1, name="H1ReLU")
            out_1_2 = self._head_choice_conv(inputs=h1, filters=self.num_out, choice_idx=arch[6], name="H1Out")

            h2 = self._head_choice_resize_conv(inputs=h1, filters=h2_filters, choice_idx=arch[7], name="H2")
            h2 = self.bn(inputs=h2, name="H2BN")
            h2 = self.relu(inputs=h2, name="H2ReLU")
            out_1_1 = self._head_choice_conv(inputs=h2, filters=self.num_out, choice_idx=arch[8], name="H2Out")

        # 记录骨干多尺度特征，供蒸馏等训练增强模块复用。
        self._feature_pyramid = [net_low, net_mid, net_high]
        self._preds = [out_1_4, out_1_2, out_1_1]
        return self._preds

    def feature_pyramid(self):
        """Return cached multi-scale backbone features."""
        if self._preds is None:
            self.build()
        return self._feature_pyramid
