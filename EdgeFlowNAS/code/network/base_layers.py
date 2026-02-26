"""Supernet 基础层实现。"""  # 定义模块用途

import tensorflow as tf  # 导入TensorFlow模块

from code.network.decorators import count_and_scope  # 导入计数作用域装饰器


class BaseLayers:  # 定义基础层类
    """封装常用卷积与激活模块。"""  # 说明类用途

    def __init__(self, is_training_ph, suffix=""):  # 定义初始化函数
        self.is_training_ph = is_training_ph  # 保存训练阶段占位符
        self.suffix = str(suffix)  # 保存作用域后缀
        self.curr_block = 0  # 初始化层计数器
        self.kernel_size = (3, 3)  # 设置默认卷积核尺寸
        self.strides = (2, 2)  # 设置默认步长
        self.padding = "same"  # 设置默认填充方式

    @count_and_scope  # 添加计数作用域装饰器
    def conv(self, inputs, filters, kernel_size=None, strides=None, activation=None, name=None):  # 定义卷积函数
        """执行二维卷积。"""  # 说明函数用途
        kernel_size = kernel_size or self.kernel_size  # 设置卷积核尺寸
        strides = strides or self.strides  # 设置步长
        return tf.compat.v1.layers.conv2d(  # 返回卷积输出
            inputs=inputs,  # 指定输入张量
            filters=int(filters),  # 指定输出通道数
            kernel_size=kernel_size,  # 指定卷积核尺寸
            strides=strides,  # 指定卷积步长
            padding=self.padding,  # 指定填充方式
            activation=activation,  # 指定激活函数
            use_bias=False,  # 关闭偏置以配合BN
            name=name,  # 指定层名称
        )

    @count_and_scope  # 添加计数作用域装饰器
    def bn(self, inputs, name=None):  # 定义批归一化函数
        """执行批归一化。"""  # 说明函数用途
        return tf.compat.v1.layers.batch_normalization(  # 返回BN输出
            inputs=inputs,  # 指定输入张量
            training=self.is_training_ph,  # 指定训练阶段开关
            momentum=0.9,  # 指定动量参数
            epsilon=1e-5,  # 指定数值稳定项
            name=name,  # 指定层名称
        )

    @count_and_scope  # 添加计数作用域装饰器
    def relu(self, inputs, name=None):  # 定义ReLU函数
        """执行ReLU激活。"""  # 说明函数用途
        return tf.nn.relu(inputs, name=name)  # 返回ReLU输出

    @count_and_scope  # 添加计数作用域装饰器
    def conv_bn_relu(self, inputs, filters, kernel_size=None, strides=None, name=None):  # 定义组合层函数
        """执行卷积+BN+ReLU组合。"""  # 说明函数用途
        net = self.conv(  # 执行卷积运算
            inputs=inputs,  # 传入输入张量
            filters=filters,  # 传入输出通道数
            kernel_size=kernel_size,  # 传入卷积核尺寸
            strides=strides,  # 传入步长参数
            name=(None if name is None else f"{name}_conv"),  # 传入卷积层名称
        )
        net = self.bn(  # 执行BN运算
            inputs=net,  # 传入卷积输出
            name=(None if name is None else f"{name}_bn"),  # 传入BN层名称
        )
        net = self.relu(  # 执行ReLU运算
            inputs=net,  # 传入BN输出
            name=(None if name is None else f"{name}_relu"),  # 传入ReLU层名称
        )
        return net  # 返回组合层输出

    @count_and_scope  # 添加计数作用域装饰器
    def resize_conv(self, inputs, filters, kernel_size=None, name=None):  # 定义上采样卷积函数
        """执行双线性上采样后卷积。"""  # 说明函数用途
        kernel_size = kernel_size or self.kernel_size  # 设置卷积核尺寸
        input_shape = inputs.get_shape().as_list()  # 读取输入静态形状
        height = input_shape[1] * 2  # 计算上采样后高度
        width = input_shape[2] * 2  # 计算上采样后宽度
        resized = tf.compat.v1.image.resize_bilinear(  # 执行双线性上采样
            images=inputs,  # 传入输入张量
            size=[height, width],  # 指定目标尺寸
            align_corners=False,  # 关闭角点对齐
            half_pixel_centers=False,  # 关闭半像素中心
            name=(None if name is None else f"{name}_resize"),  # 指定上采样层名称
        )
        return self.conv(  # 返回上采样后的卷积输出
            inputs=resized,  # 传入上采样张量
            filters=filters,  # 传入输出通道数
            kernel_size=kernel_size,  # 传入卷积核尺寸
            strides=(1, 1),  # 固定步长为1
            activation=None,  # 关闭卷积激活
            name=(None if name is None else f"{name}_conv"),  # 指定卷积层名称
        )
