#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiScaleResNet with switchable Block types and Upsample types.
支持 BlockType: resblock/mbconv/shufflenet
支持 UpsampleType: transpose/bilinear
支持 MBConvExpandRatio: MBConv 的扩展比参数
"""

import tensorflow as tf  # TF 核心库
import numpy as np  # 数值计算
import sys  # 系统退出
from functools import wraps  # 装饰器工具
from misc.Decorators import *  # 计数装饰器
from network.BaseLayers import *  # 基础层定义


class MultiScaleResNet(BaseLayers):
    """支持多种 Block 和 Upsample 类型的 MultiScaleResNet"""

    def __init__(self, InputPH=None, Padding=None, NumOut=None, InitNeurons=None,
                 ExpansionFactor=None, NumSubBlocks=None, NumBlocks=None,
                 Suffix=None, UncType=None, BlockType='resblock',
                 UpsampleType='transpose', MBConvExpandRatio=6):
        """
        Args:
            BlockType: 'resblock' | 'mbconv' | 'shufflenet'
            UpsampleType: 'transpose' | 'bilinear'
            MBConvExpandRatio: MBConv 的通道扩展比 (默认 6)
        """
        super(MultiScaleResNet, self).__init__()
        
        # 输入验证
        if InputPH is None:
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        
        # 默认参数
        self.InitNeurons = InitNeurons if InitNeurons else 37
        self.ExpansionFactor = ExpansionFactor if ExpansionFactor else 2.0
        self.NumSubBlocks = NumSubBlocks if NumSubBlocks else 2
        self.NumBlocks = NumBlocks if NumBlocks else 1
        self.Suffix = Suffix if Suffix else ''
        self.NumOut = NumOut if NumOut else 1
        self.DropOutRate = 0.7
        self.currBlock = 0
        self.UncType = UncType
        
        # 不确定性类型处理
        if self.UncType in ('Aleatoric', 'Inlier', 'LinearSoftplus'):
            self.NumOut *= 2

        # 卷积默认参数
        self.kernel_size = (3, 3)
        self.strides = (2, 2)
        self.padding = Padding if Padding else 'same'
        
        # Block 类型验证
        self.BlockType = BlockType.lower()
        if self.BlockType not in ('resblock', 'mbconv', 'shufflenet'):
            raise ValueError(f"BlockType 必须是 'resblock', 'mbconv' 或 'shufflenet'")
        
        # Upsample 类型验证
        self.UpsampleType = UpsampleType.lower()
        if self.UpsampleType not in ('transpose', 'bilinear'):
            raise ValueError(f"UpsampleType 必须是 'transpose' 或 'bilinear'")
        
        # MBConv 扩展比
        self.MBConvExpandRatio = MBConvExpandRatio

    # ==================== Bilinear 上采样实现 ====================

    def ResizeConv(self, inputs, filters, kernel_size=None, name=None):
        """双线性插值上采样 + 卷积 (替代 ConvTranspose stride=2)"""
        if kernel_size is None: kernel_size = self.kernel_size
        
        # 获取输入尺寸
        shape = inputs.get_shape().as_list()
        new_h, new_w = shape[1] * 2, shape[2] * 2  # 2 倍上采样
        
        # 双线性插值 (align_corners=False 以满足 NPU 加速要求)
        upsampled = tf.compat.v1.image.resize_bilinear(
            inputs, [new_h, new_w], align_corners=False,
            name=(name + '_resize') if name else None
        )
        
        # 后接卷积
        output = self.Conv(inputs=upsampled, filters=filters, kernel_size=kernel_size,
                          strides=(1, 1), activation=None,
                          name=(name + '_conv') if name else None)
        return output

    def ResizeConvBNReLUBlock(self, inputs, filters, kernel_size=None, name=None):
        """双线性上采样 + 卷积 + BN + ReLU"""
        net = self.ResizeConv(inputs, filters, kernel_size, name)
        net = self.BN(net)
        net = self.ReLU(net)
        return net

    def _get_upsample(self, inputs, filters, kernel_size=None, with_bn_relu=True):
        """上采样工厂方法：根据 UpsampleType 选择实现"""
        if self.UpsampleType == 'transpose':
            if with_bn_relu:
                return self.ConvTransposeBNReLUBlock(inputs=inputs, filters=filters, kernel_size=kernel_size)
            else:
                return self.ConvTranspose(inputs=inputs, filters=filters, kernel_size=kernel_size)
        else:  # bilinear
            if with_bn_relu:
                return self.ResizeConvBNReLUBlock(inputs, filters, kernel_size)
            else:
                return self.ResizeConv(inputs, filters, kernel_size)

    # ==================== Block 实现 ====================

    @CountAndScope
    def ResBlock(self, inputs, filters, kernel_size=None, padding=None):
        """标准残差块 (stride=1)"""
        if kernel_size is None: kernel_size = self.kernel_size
        if padding is None: padding = self.padding
        
        Net = self.ConvBNReLUBlock(inputs=inputs, filters=filters, padding=padding, strides=(1, 1))
        Net = self.Conv(inputs=Net, filters=filters, padding=padding, strides=(1, 1), activation=None)
        Net = self.BN(inputs=Net)
        Net = tf.add(Net, inputs)  # 残差连接
        Net = self.ReLU(inputs=Net)
        return Net

    def MBConvBlock(self, inputs, filters_out, stride=1, name='mbconv'):
        """MBConv 倒残差块"""
        in_c = inputs.get_shape().as_list()[-1]
        hidden_c = int(in_c * self.MBConvExpandRatio)  # 使用实例变量
        
        with tf.compat.v1.variable_scope(name):
            # 1x1 扩展
            out = tf.compat.v1.layers.conv2d(inputs, hidden_c, 1, 1, padding='same', use_bias=False)
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            out = tf.nn.relu6(out)
            
            # 3x3 DepthwiseConv
            with tf.compat.v1.variable_scope('dw_conv'):
                dw_filter = tf.compat.v1.get_variable(
                    'dw_filter', [3, 3, hidden_c, 1],
                    initializer=tf.compat.v1.initializers.glorot_uniform()
                )
            out = tf.nn.depthwise_conv2d(out, dw_filter, [1, stride, stride, 1], padding='SAME')
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            out = tf.nn.relu6(out)
            
            # 1x1 投影
            out = tf.compat.v1.layers.conv2d(out, filters_out, 1, 1, padding='same', use_bias=False)
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            
            # 残差连接
            if stride == 1 and in_c == filters_out:
                out = tf.add(out, inputs)
            return out

    def _channel_shuffle(self, x, groups=2):
        """
        使用 1x1 卷积实现的 Channel Shuffle，完全支持 NPU 加速。
        原始的 Reshape+Transpose 方案因 Transpose 排列不被 NPU 支持而回退 CPU。
        1x1 卷积等价于通道重排，NPU 处理效率极高。
        """
        n, h, w, c = x.get_shape().as_list()
        assert c % groups == 0, f'Channel {c} must be divisible by groups {groups}'
        channels_per_group = c // groups

        # 1. 构建重排索引 (Permutation Index)
        # 原始逻辑: [0, 1, 2, 3, 4, 5] (groups=2) -> [0, 3, 1, 4, 2, 5]
        # 公式: new_idx = (old_idx % groups) * channels_per_group + (old_idx // groups)
        perm = np.zeros(c, dtype=np.int32)
        for i in range(c):
            perm[i] = (i % groups) * channels_per_group + (i // groups)

        # 2. 构建 1x1 卷积的权重矩阵 (Identity Matrix 的行重排)
        # Shape: [kernel_h, kernel_w, input_channels, output_channels] -> [1, 1, c, c]
        kernel_weights = np.zeros((1, 1, c, c), dtype=np.float32)
        for i in range(c):
            # 在输出通道 i，只接收输入通道 perm[i] 的数据
            input_idx = perm[i]
            kernel_weights[0, 0, input_idx, i] = 1.0

        # 3. 创建常量 Tensor (不可训练)
        kernel = tf.constant(kernel_weights, name='shuffle_1x1_weights')

        # 4. 执行 1x1 卷积 (NPU 对 Conv2D 支持极好)
        x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID', name='shuffle_conv')
        return x

    def ShuffleNetBlock(self, inputs, out_c, stride=1, name='shuffle'):
        """ShuffleNetV2 Block"""
        with tf.compat.v1.variable_scope(name):
            in_c = inputs.get_shape().as_list()[-1]
            mid_c = out_c // 2
            
            if stride == 1:
                x1, x2 = tf.split(inputs, num_or_size_splits=2, axis=3)
                
                x2 = tf.compat.v1.layers.conv2d(x2, mid_c, 1, 1, padding='same', use_bias=False)
                x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
                x2 = tf.nn.relu(x2)
                
                with tf.compat.v1.variable_scope('branch2_dw'):
                    dw_filter = tf.compat.v1.get_variable(
                        'dw_filter', [3, 3, mid_c, 1],
                        initializer=tf.compat.v1.initializers.glorot_uniform()
                    )
                x2 = tf.nn.depthwise_conv2d(x2, dw_filter, [1, 1, 1, 1], padding='SAME')
                x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
                x2 = tf.compat.v1.layers.conv2d(x2, mid_c, 1, 1, padding='same', use_bias=False)
                x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
                x2 = tf.nn.relu(x2)
                
                out = tf.concat([x1, x2], axis=3)
            else:
                x_shape = inputs.get_shape().as_list()
                with tf.compat.v1.variable_scope('branch1_dw'):
                    dw_filter1 = tf.compat.v1.get_variable(
                        'dw_filter', [3, 3, x_shape[-1], 1],
                        initializer=tf.compat.v1.initializers.glorot_uniform()
                    )
                x1 = tf.nn.depthwise_conv2d(inputs, dw_filter1, [1, stride, stride, 1], padding='SAME')
                x1 = tf.compat.v1.layers.batch_normalization(x1, momentum=0.9, epsilon=1e-5)
                x1 = tf.compat.v1.layers.conv2d(x1, mid_c, 1, 1, padding='same', use_bias=False)
                x1 = tf.compat.v1.layers.batch_normalization(x1, momentum=0.9, epsilon=1e-5)
                x1 = tf.nn.relu(x1)

                x2 = tf.compat.v1.layers.conv2d(inputs, mid_c, 1, 1, padding='same', use_bias=False)
                x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
                x2 = tf.nn.relu(x2)
                with tf.compat.v1.variable_scope('branch2_dw'):
                    dw_filter2 = tf.compat.v1.get_variable(
                        'dw_filter', [3, 3, mid_c, 1],
                        initializer=tf.compat.v1.initializers.glorot_uniform()
                    )
                x2 = tf.nn.depthwise_conv2d(x2, dw_filter2, [1, stride, stride, 1], padding='SAME')
                x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
                x2 = tf.compat.v1.layers.conv2d(x2, mid_c, 1, 1, padding='same', use_bias=False)
                x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
                x2 = tf.nn.relu(x2)

                out = tf.concat([x1, x2], axis=3)
            
            out = self._channel_shuffle(out, 2)
            return out

    def _get_block(self, inputs, filters, stride=1, name_suffix=''):
        """Block 工厂方法"""
        if self.BlockType == 'resblock':
            return self.ResBlock(inputs=inputs, filters=filters)
        elif self.BlockType == 'mbconv':
            return self.MBConvBlock(inputs, filters, stride=stride, name=f'mbconv_{name_suffix}')
        elif self.BlockType == 'shufflenet':
            return self.ShuffleNetBlock(inputs, filters, stride=stride, name=f'shuffle_{name_suffix}')

    # ==================== 网络结构 ====================

    @CountAndScope
    def ResNetBlock(self, inputs):
        """编码-解码主干"""
        # === Encoder ===
        NumFilters = self.InitNeurons
        
        Net = self.ConvBNReLUBlock(inputs=inputs, filters=NumFilters, kernel_size=(7, 7))
        NumFilters = int(NumFilters * self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs=Net, filters=NumFilters, kernel_size=(5, 5))

        for i in range(self.NumSubBlocks):
            Net = self._get_block(Net, NumFilters, stride=1, name_suffix=f'enc_{i}')
            NumFilters = int(NumFilters * self.ExpansionFactor)
            Net = self.Conv(inputs=Net, filters=NumFilters)

        # === Decoder ===
        Nets = []
        for i in range(self.NumSubBlocks):
            Net = self._get_block(Net, NumFilters, stride=1, name_suffix=f'dec_{i}')
            NumFilters = int(NumFilters / self.ExpansionFactor)
            Net = self._get_upsample(Net, NumFilters, with_bn_relu=False)  # 上采样

        # 多尺度输出头
        # NetOut = self.Conv(inputs=Net, filters=self.NumOut, kernel_size=(3, 3), strides=(1, 1), activation=None)
        NetOut = self.Conv(inputs=Net, filters=self.NumOut, kernel_size=(7, 7), strides=(1, 1), activation=None)

        Nets.append(NetOut)

        NumFilters = int(NumFilters / self.ExpansionFactor)
        # Net = self._get_upsample(Net, NumFilters, kernel_size=(3, 3), with_bn_relu=True)
        Net = self._get_upsample(Net, NumFilters, kernel_size=(5, 5), with_bn_relu=True)

        # NetOut = self.Conv(inputs=Net, filters=self.NumOut, kernel_size=(3, 3), strides=(1, 1), activation=None)
        NetOut = self.Conv(inputs=Net, filters=self.NumOut, kernel_size=(7, 7), strides=(1, 1), activation=None)

        Nets.append(NetOut)

        NumFilters = int(NumFilters / self.ExpansionFactor)
        # Net = self._get_upsample(Net, NumFilters, kernel_size=(3, 3), with_bn_relu=True)
        Net = self._get_upsample(Net, NumFilters, kernel_size=(7, 7), with_bn_relu=True)

        # Net = self.Conv(inputs=Net, filters=self.NumOut, kernel_size=(3, 3), strides=(1, 1), activation=None)
        Net = self.Conv(inputs=Net, filters=self.NumOut, kernel_size=(7, 7), strides=(1, 1), activation=None)

        Nets.append(Net)
        return Nets

    def Network(self):
        """构建完整网络"""
        OutNow = self.InputPH
        for count in range(self.NumBlocks):
            with tf.compat.v1.variable_scope('EncoderDecoderBlock' + str(count) + self.Suffix):
                OutNow = self.ResNetBlock(OutNow)
                self.currBlock += 1
        return OutNow
