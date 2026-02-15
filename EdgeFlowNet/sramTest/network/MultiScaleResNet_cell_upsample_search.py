#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiScaleResNet — Head Upsample NAS 搜索版本
基于 33decoder 版本，新增 HeadUpsampleChoice 参数用于搜索最优 Head 上采样策略。
Predict 卷积固定为 7x7，Decoder 上采样固定为 Bilinear+Conv3x3。

HeadUpsampleChoice 选项:
  resize_conv3x3   : Bilinear 2x + Conv 3x3 + BN + ReLU  (基线，最轻量)
  resize_conv5x5   : Bilinear 2x + Conv 5x5 + BN + ReLU  (中等)
  resize_conv7x7   : Bilinear 2x + Conv 7x7 + BN + ReLU  (原始，最重)
  resize_dsconv3x3 : Bilinear 2x + DWConv 3x3 + PW 1x1 + BN + ReLU  (轻量分离卷积)
  resize_mbconv_e4 : Bilinear 2x + MBConv(expand=4)  (倒残差块，扩展比4)
  resize_mbconv_e6 : Bilinear 2x + MBConv(expand=6)  (倒残差块，扩展比6)
  resize_shuffle   : Bilinear 2x + 1x1投影 + ShuffleNetV2  (通道混洗)
"""

import tensorflow as tf  # TF 核心库
import numpy as np  # 数值计算
import sys  # 系统退出
from functools import wraps  # 装饰器工具
from misc.Decorators import *  # 计数装饰器
from network.BaseLayers import *  # 基础层定义

# 所有合法的 Head 上采样选项
VALID_HEAD_CHOICES = (
    'resize_conv3x3', 'resize_conv5x5', 'resize_conv7x7',
    'resize_dsconv3x3',
    'resize_mbconv_e4', 'resize_mbconv_e6',
    'resize_shuffle',
)


class MultiScaleResNet(BaseLayers):
    """支持 Head Upsample NAS 搜索的 MultiScaleResNet"""

    def __init__(self, InputPH=None, Padding=None, NumOut=None, InitNeurons=None,
                 ExpansionFactor=None, NumSubBlocks=None, NumBlocks=None,
                 Suffix=None, UncType=None, BlockType='shufflenet',
                 MBConvExpandRatio=6, HeadUpsampleChoice='resize_conv3x3'):
        """
        Args:
            BlockType: 骨干 Block 类型 'resblock' | 'mbconv' | 'shufflenet'
            MBConvExpandRatio: 骨干 MBConv 的扩展比 (默认 6)
            HeadUpsampleChoice: Head 上采样策略 (见模块顶部说明)
        """
        super(MultiScaleResNet, self).__init__()

        # 输入占位符验证
        if InputPH is None:
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH

        # 网络超参数
        self.InitNeurons = InitNeurons if InitNeurons else 37  # 初始通道数
        self.ExpansionFactor = ExpansionFactor if ExpansionFactor else 2.0  # 通道扩展因子
        self.NumSubBlocks = NumSubBlocks if NumSubBlocks else 2  # 编解码子块数
        self.NumBlocks = NumBlocks if NumBlocks else 1  # 编解码大块数
        self.Suffix = Suffix if Suffix else ''  # 变量域后缀
        self.NumOut = NumOut if NumOut else 1  # 输出通道数
        self.DropOutRate = 0.7  # Dropout 率
        self.currBlock = 0  # 当前块计数
        self.UncType = UncType  # 不确定性类型

        # 不确定性类型处理: Aleatoric/Inlier/LinearSoftplus 需要双倍输出
        if self.UncType in ('Aleatoric', 'Inlier', 'LinearSoftplus'):
            self.NumOut *= 2

        # 卷积默认参数
        self.kernel_size = (3, 3)  # 默认卷积核
        self.strides = (2, 2)  # 默认步幅 (下采样)
        self.padding = Padding if Padding else 'same'  # 填充方式

        # 骨干 Block 类型验证
        self.BlockType = BlockType.lower()
        if self.BlockType not in ('resblock', 'mbconv', 'shufflenet'):
            raise ValueError(f"BlockType 必须是 'resblock', 'mbconv' 或 'shufflenet'")

        # 骨干 MBConv 扩展比
        self.MBConvExpandRatio = MBConvExpandRatio

        # Head 上采样策略验证
        self.HeadUpsampleChoice = HeadUpsampleChoice.lower()
        if self.HeadUpsampleChoice not in VALID_HEAD_CHOICES:
            raise ValueError(f"HeadUpsampleChoice 必须是 {VALID_HEAD_CHOICES} 之一")

    # ==================== 公共辅助方法 ====================

    def _bilinear_resize_2x(self, inputs, name=None):
        """双线性插值 2 倍上采样 (align_corners=False 以兼容 NPU)"""
        shape = inputs.get_shape().as_list()  # 获取输入尺寸
        new_h, new_w = shape[1] * 2, shape[2] * 2  # 2 倍空间尺寸
        return tf.compat.v1.image.resize_bilinear(
            inputs, [new_h, new_w], align_corners=False, name=name
        )

    # ==================== Decoder 上采样 (固定 Bilinear+Conv3x3) ====================

    def _decoder_upsample(self, inputs, filters, with_bn_relu=False):
        """Decoder 上采样: Bilinear 2x + Conv 3x3 (固定，不参与搜索)"""
        net = self._bilinear_resize_2x(inputs)  # 双线性插值 2x
        net = self.Conv(inputs=net, filters=filters, kernel_size=(3, 3),
                        strides=(1, 1), activation=None)  # 3x3 卷积调整通道
        if with_bn_relu:  # 可选 BN+ReLU
            net = self.BN(net)
            net = self.ReLU(net)
        return net

    # ==================== Head 上采样选项 ====================

    def _head_upsample_conv(self, inputs, filters, kernel_size, name_suffix):
        """Head 上采样: Bilinear 2x + Conv + BN + ReLU"""
        net = self._bilinear_resize_2x(inputs)  # 双线性插值 2x
        net = self.Conv(inputs=net, filters=filters, kernel_size=kernel_size,
                        strides=(1, 1), activation=None)  # 标准卷积
        net = self.BN(net)  # 批归一化
        net = self.ReLU(net)  # 激活
        return net

    def _head_upsample_dsconv(self, inputs, filters, name_suffix):
        """Head 上采样: Bilinear 2x + DWConv 3x3 + PWConv 1x1 + BN + ReLU"""
        net = self._bilinear_resize_2x(inputs)  # 双线性插值 2x
        in_c = net.get_shape().as_list()[-1]  # 输入通道数

        with tf.compat.v1.variable_scope(f'head_dsconv_{name_suffix}'):
            # DW 3x3: 逐通道卷积 (保持通道数)
            with tf.compat.v1.variable_scope('dw_conv'):
                dw_filter = tf.compat.v1.get_variable(
                    'dw_filter', [3, 3, in_c, 1],
                    initializer=tf.compat.v1.initializers.glorot_uniform()
                )
            net = tf.nn.depthwise_conv2d(net, dw_filter, [1, 1, 1, 1], padding='SAME')
            net = tf.compat.v1.layers.batch_normalization(net, momentum=0.9, epsilon=1e-5)
            net = tf.nn.relu(net)

            # PW 1x1: 逐点卷积 (调整通道数: in_c → filters)
            net = tf.compat.v1.layers.conv2d(net, filters, 1, 1, padding='same', use_bias=False)
            net = tf.compat.v1.layers.batch_normalization(net, momentum=0.9, epsilon=1e-5)
            net = tf.nn.relu(net)
        return net

    def _head_upsample_mbconv(self, inputs, filters, expand_ratio, name_suffix):
        """Head 上采样: Bilinear 2x + MBConv (倒残差块)"""
        net = self._bilinear_resize_2x(inputs)  # 双线性插值 2x
        in_c = net.get_shape().as_list()[-1]  # 输入通道数
        hidden_c = int(in_c * expand_ratio)  # 扩展后的隐藏通道数

        with tf.compat.v1.variable_scope(f'head_mbconv_e{expand_ratio}_{name_suffix}'):
            # 1x1 扩展: in_c → hidden_c
            out = tf.compat.v1.layers.conv2d(net, hidden_c, 1, 1, padding='same', use_bias=False)
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            out = tf.nn.relu6(out)

            # 3x3 DW: 逐通道卷积 (保持 hidden_c)
            with tf.compat.v1.variable_scope('dw_conv'):
                dw_filter = tf.compat.v1.get_variable(
                    'dw_filter', [3, 3, hidden_c, 1],
                    initializer=tf.compat.v1.initializers.glorot_uniform()
                )
            out = tf.nn.depthwise_conv2d(out, dw_filter, [1, 1, 1, 1], padding='SAME')
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            out = tf.nn.relu6(out)

            # 1x1 投影: hidden_c → filters
            out = tf.compat.v1.layers.conv2d(out, filters, 1, 1, padding='same', use_bias=False)
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)

            # 残差连接 (仅当通道数匹配时)
            if in_c == filters:
                out = tf.add(out, net)
        return out

    def _head_upsample_shuffle(self, inputs, filters, name_suffix):
        """Head 上采样: Bilinear 2x + 1x1 通道投影 + ShuffleNetV2 Block"""
        net = self._bilinear_resize_2x(inputs)  # 双线性插值 2x

        # 1x1 投影: 调整通道数以匹配 ShuffleNet 输入要求 (in_c → filters)
        with tf.compat.v1.variable_scope(f'head_shuffle_proj_{name_suffix}'):
            net = tf.compat.v1.layers.conv2d(net, filters, 1, 1, padding='same', use_bias=False)
            net = tf.compat.v1.layers.batch_normalization(net, momentum=0.9, epsilon=1e-5)
            net = tf.nn.relu(net)

        # ShuffleNetV2 Block: 通道混洗 + 特征精炼 (stride=1, 保持尺寸)
        net = self.ShuffleNetBlock(net, filters, stride=1, name=f'head_shuffle_{name_suffix}')
        return net

    def _get_head_upsample(self, inputs, filters, name_suffix=''):
        """Head 上采样工厂方法: 根据 HeadUpsampleChoice 选择实现"""
        choice = self.HeadUpsampleChoice  # 获取当前选项

        if choice == 'resize_conv3x3':
            return self._head_upsample_conv(inputs, filters, (3, 3), name_suffix)
        elif choice == 'resize_conv5x5':
            return self._head_upsample_conv(inputs, filters, (5, 5), name_suffix)
        elif choice == 'resize_conv7x7':
            return self._head_upsample_conv(inputs, filters, (7, 7), name_suffix)
        elif choice == 'resize_dsconv3x3':
            return self._head_upsample_dsconv(inputs, filters, name_suffix)
        elif choice == 'resize_mbconv_e4':
            return self._head_upsample_mbconv(inputs, filters, 4, name_suffix)
        elif choice == 'resize_mbconv_e6':
            return self._head_upsample_mbconv(inputs, filters, 6, name_suffix)
        elif choice == 'resize_shuffle':
            return self._head_upsample_shuffle(inputs, filters, name_suffix)
        else:
            raise ValueError(f"未知的 HeadUpsampleChoice: {choice}")

    # ==================== 骨干 Block 实现 ====================

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
        """MBConv 倒残差块 (骨干用)"""
        in_c = inputs.get_shape().as_list()[-1]  # 输入通道数
        hidden_c = int(in_c * self.MBConvExpandRatio)  # 扩展通道数

        with tf.compat.v1.variable_scope(name):
            # 1x1 扩展
            out = tf.compat.v1.layers.conv2d(inputs, hidden_c, 1, 1, padding='same', use_bias=False)
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            out = tf.nn.relu6(out)

            # 3x3 DW
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
        """1x1 卷积实现的 Channel Shuffle (完全 NPU 兼容)"""
        n, h, w, c = x.get_shape().as_list()
        assert c % groups == 0, f'Channel {c} must be divisible by groups {groups}'
        channels_per_group = c // groups

        # 构建重排索引
        perm = np.zeros(c, dtype=np.int32)
        for i in range(c):
            perm[i] = (i % groups) * channels_per_group + (i // groups)

        # 构建 1x1 卷积权重矩阵 (置换矩阵)
        kernel_weights = np.zeros((1, 1, c, c), dtype=np.float32)
        for i in range(c):
            kernel_weights[0, 0, perm[i], i] = 1.0

        kernel = tf.constant(kernel_weights, name='shuffle_1x1_weights')  # 不可训练常量
        x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID', name='shuffle_conv')
        return x

    def ShuffleNetBlock(self, inputs, out_c, stride=1, name='shuffle'):
        """ShuffleNetV2 Block"""
        with tf.compat.v1.variable_scope(name):
            in_c = inputs.get_shape().as_list()[-1]
            mid_c = out_c // 2  # 分支通道数

            if stride == 1:
                # stride=1: 通道分裂 → 一半直通 + 一半处理
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

                out = tf.concat([x1, x2], axis=3)  # 合并两个分支
            else:
                # stride=2: 两个分支都处理 (下采样)
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

            out = self._channel_shuffle(out, 2)  # 通道混洗
            return out

    def _get_block(self, inputs, filters, stride=1, name_suffix=''):
        """骨干 Block 工厂方法"""
        if self.BlockType == 'resblock':
            return self.ResBlock(inputs=inputs, filters=filters)
        elif self.BlockType == 'mbconv':
            return self.MBConvBlock(inputs, filters, stride=stride, name=f'mbconv_{name_suffix}')
        elif self.BlockType == 'shufflenet':
            return self.ShuffleNetBlock(inputs, filters, stride=stride, name=f'shuffle_{name_suffix}')

    # ==================== 网络主结构 ====================

    @CountAndScope
    def ResNetBlock(self, inputs):
        """编码-解码主干 + 多尺度 Head 输出"""
        # === Encoder: 逐步降分辨率、增通道 ===
        NumFilters = self.InitNeurons
        Net = self.ConvBNReLUBlock(inputs=inputs, filters=NumFilters, kernel_size=(7, 7))  # 初始 7x7

        NumFilters = int(NumFilters * self.ExpansionFactor)  # 通道翻倍
        Net = self.ConvBNReLUBlock(inputs=Net, filters=NumFilters, kernel_size=(5, 5))  # 5x5

        for i in range(self.NumSubBlocks):  # 编码子块
            Net = self._get_block(Net, NumFilters, stride=1, name_suffix=f'enc_{i}')
            NumFilters = int(NumFilters * self.ExpansionFactor)
            Net = self.Conv(inputs=Net, filters=NumFilters)  # stride=2 下采样

        # === Decoder: 逐步升分辨率、减通道 ===
        Nets = []
        for i in range(self.NumSubBlocks):  # 解码子块
            Net = self._get_block(Net, NumFilters, stride=1, name_suffix=f'dec_{i}')
            NumFilters = int(NumFilters / self.ExpansionFactor)
            Net = self._decoder_upsample(Net, NumFilters, with_bn_relu=False)  # Decoder 固定 Bilinear+3x3

        # === Head0: 最低分辨率输出 (无上采样) ===
        NetOut = self.Conv(inputs=Net, filters=self.NumOut,
                           kernel_size=(7, 7), strides=(1, 1), activation=None)  # 7x7 predict
        Nets.append(NetOut)

        # === Head1: 中间分辨率输出 (上采样 2x) ===
        NumFilters = int(NumFilters / self.ExpansionFactor)  # 通道减半
        Net = self._get_head_upsample(Net, NumFilters, name_suffix='head1')  # ← NAS 搜索目标
        NetOut = self.Conv(inputs=Net, filters=self.NumOut,
                           kernel_size=(7, 7), strides=(1, 1), activation=None)  # 7x7 predict
        Nets.append(NetOut)

        # === Head2: 最高分辨率输出 (再上采样 2x) ===
        NumFilters = int(NumFilters / self.ExpansionFactor)  # 通道再减半
        Net = self._get_head_upsample(Net, NumFilters, name_suffix='head2')  # ← NAS 搜索目标
        Net = self.Conv(inputs=Net, filters=self.NumOut,
                        kernel_size=(7, 7), strides=(1, 1), activation=None)  # 7x7 predict
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

