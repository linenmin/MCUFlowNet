#!/usr/bin/env python

import tensorflow as tf
import sys
from functools import wraps
# 保持原有引用结构
from misc.Decorators import *
from network.BaseLayers import *

class MultiScaleResNet(BaseLayers):
    # 初始化部分保持不变
    def __init__(self, InputPH = None, Padding = None,\
                 NumOut = None, InitNeurons = None, ExpansionFactor = None, NumSubBlocks = None, NumBlocks = None, Suffix = None, UncType = None):
        super(MultiScaleResNet, self).__init__()
        if(InputPH is None):
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        if(InitNeurons is None):
            InitNeurons = 37
        if(ExpansionFactor is None):
            ExpansionFactor =  2.0
        if(NumSubBlocks is None):
            NumSubBlocks = 2
        if(NumBlocks is None):
            NumBlocks = 1
        self.InitNeurons = InitNeurons
        self.ExpansionFactor = ExpansionFactor
        self.DropOutRate = 0.7
        self.NumSubBlocks = NumSubBlocks
        self.NumBlocks = NumBlocks
        if(Suffix is None):
            Suffix = ''
        self.Suffix = Suffix
        if(NumOut is None):
            NumOut = 1
        self.NumOut = NumOut
        self.currBlock = 0
        self.UncType = UncType
        if(self.UncType == 'Aleatoric' or self.UncType == 'Inlier' or self.UncType == 'LinearSoftplus'):
            self.NumOut *= 2

        self.kernel_size = (3,3)
        self.strides = (2,2)
        if(Padding is None):
            Padding = 'same'
        self.padding = Padding

    # --- 新增辅助函数：双线性插值 + 卷积 ---
    def ResizeConv(self, inputs, filters, kernel_size=None, activation=None, name=None):
        """替代转置卷积：先双线性插值放大2倍，再做卷积"""
        if kernel_size is None:
            kernel_size = self.kernel_size
            
        # 1. 获取输入形状
        input_shape = inputs.get_shape().as_list()
        # 2. 计算目标尺寸 (H*2, W*2)
        # 注意：这里假设输入是 N H W C 格式
        new_height = input_shape[1] * 2
        new_width = input_shape[2] * 2
        
        # 3. 双线性插值上采样
        # 关键修改：align_corners=False 才能满足 NPU 对整数倍 (2x, 4x...) 缩放的加速要求
        upsampled = tf.compat.v1.image.resize_bilinear(
            inputs, 
            size=[new_height, new_width],
            align_corners=False,
            # Keep this False for Ethos-U/Vela. True causes ResizeBilinear to be
            # decomposed into extra ops (AvgPool/DepthwiseConv), inflating SRAM.
            half_pixel_centers=False,
            name=(name + '_resize') if name else None
        )
        
        # 4. 普通卷积 (Stride=1)
        output = self.Conv(
            inputs=upsampled, 
            filters=filters, 
            kernel_size=kernel_size, 
            strides=(1,1), 
            activation=activation, 
            name=(name + '_conv') if name else None
        )
        return output

    # 对应 ConvTransposeBNReLUBlock 的 Resize 版本
    def ResizeConvBNReLUBlock(self, inputs, filters, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
            
        # 上采样 + 卷积
        net = self.ResizeConv(inputs, filters, kernel_size)
        # BN + ReLU
        net = self.BN(net)
        net = self.ReLU(net)
        return net

    # --- 复用基类的 ResBlock ---
    # 原有的 ResBlockTranspose 只是 stride=1 的转置卷积，其实等价于 stride=1 的普通卷积
    # 所以我们可以直接用 ResBlock 来替代 ResBlockTranspose

    @CountAndScope
    def ResBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        # 保持原有 ResBlock 逻辑
        if(kernel_size is None):
            kernel_size = self.kernel_size
        if(strides is None):
            strides = self.strides
        if(padding is None):
            padding = self.padding                  
        Net = self.ConvBNReLUBlock(inputs = inputs, filters = filters, padding = padding, strides=(1,1))
        Net = self.Conv(inputs = Net, filters = filters, padding = padding, strides=(1,1), activation=None)
        Net = self.BN(inputs = Net)
        Net = tf.add(Net, inputs)
        Net = self.ReLU(inputs = Net)
        return Net

    @CountAndScope
    def ResNetBlock(self, inputs):
        # --- Encoder (保持不变) ---
        # Encoder has (NumSubBlocks + 2) x (Conv + BN + ReLU) Layers
        
        NumFilters = self.InitNeurons
        # Conv
        Net = self.ConvBNReLUBlock(inputs = inputs, filters = NumFilters, kernel_size = (7,7)) # Stride 2 in Init? No, base strides defined in init
        
        # Conv
        NumFilters = int(NumFilters*self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (5,5))
        
        # Conv Loop
        for count in range(self.NumSubBlocks):
            Net = self.ResBlock(inputs = Net, filters = NumFilters)
            NumFilters = int(NumFilters*self.ExpansionFactor)
            # Extra Conv for downscaling (Stride由self.strides控制，默认为2)
            Net = self.Conv(inputs = Net, filters = NumFilters)
            
        # --- Decoder (修改为 Resize + Conv) ---
        Nets = []
        for count in range(self.NumSubBlocks):
            # 原本的 ResBlockTranspose (Stride=1) -> 替换为 ResBlock (Stride=1)
            # 因为不改变分辨率，只是特征提取
            Net = self.ResBlock(inputs = Net, filters = NumFilters)    
            
            NumFilters = int(NumFilters/self.ExpansionFactor)
            
            # 原本的 Extra ConvTranspose for upscaling -> 替换为 ResizeConv
            Net = self.ResizeConv(inputs = Net, filters = NumFilters)

        # ------------------------------------------------------------------
        # 修改点 1: 原本的 Upsample (ConvTranspose stride=1? No, logic depends on context)
        # 原代码: NetOut = self.ConvTranspose(..., strides=(1,1))
        # 这似乎仅仅是一个 Projection 层？
        # 如果是 stride=(1,1)，直接用 Conv 即可
        # ------------------------------------------------------------------
        # 原文 L121: NetOut = self.ConvTranspose(inputs = Net, filters = self.NumOut, strides=(1,1), ...)
        # 替换为 Conv (降维输出)
        NetOut = self.Conv(inputs = Net, filters = self.NumOut, kernel_size=(7,7), strides=(1,1), activation=None)
        Nets.append(NetOut)
        print(f"[*] Decoder Out 1: {NetOut.shape}")
            
        # ------------------------------------------------------------------
        # 修改点 2: 进一步上采样
        # 原文 L127: Net = self.ConvTransposeBNReLUBlock(..., kernel_size=(5,5)) (没写stride，默认2)
        # 替换为 ResizeConvBNReLU
        NumFilters = int(NumFilters/self.ExpansionFactor)
        Net = self.ResizeConvBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size=(5,5))
        
        # 原文 L129: NetOut = self.ConvTranspose(..., strides=(1,1)) 
        # 替换为 Conv
        NetOut = self.Conv(inputs = Net, filters = self.NumOut, kernel_size=(7,7), strides=(1,1), activation=None)
        Nets.append(NetOut)
        print(f"[*] Decoder Out 2: {NetOut.shape}")

        # ------------------------------------------------------------------
        # 修改点 3: 最后的上采样 (Upsample Final 2)
        # 原文 L135: Net = self.ConvTransposeBNReLUBlock(..., kernel_size=(7,7)) (默认stride 2)
        NumFilters = int(NumFilters/self.ExpansionFactor)
        Net = self.ResizeConvBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size=(7,7))
        print(f"[*] Upsample Final 2 output shape: {Net.shape}")
        
        # ------------------------------------------------------------------
        # 修改点 4: 主输出
        # 原文 L139: Net = self.ConvTranspose(..., strides=(1,1))
        # 替换为 Conv
        Net = self.Conv(inputs = Net, filters = self.NumOut, kernel_size=(7,7), strides=(1,1), activation=None)
        Nets.append(Net)
        print(f"[*] Main Output shape: {Net.shape}")
        
        return Nets
        
    def Network(self):
        OutNow = self.InputPH
        for count in range(self.NumBlocks):
            with tf.compat.v1.variable_scope('EncoderDecoderBlock' + str(count) + self.Suffix):
                OutNow = self.ResNetBlock(OutNow)
                self.currBlock += 1
        return OutNow
