"""Bilinear Supernet 网络实现。"""  # 定义模块用途

import tensorflow as tf  # 导入TensorFlow模块

from code.network.base_layers import BaseLayers  # 导入基础层实现


class MultiScaleResNetSupernet(BaseLayers):  # 定义超网模型类
    """支持9维架构编码的Bilinear超网模型。"""  # 说明类用途

    def __init__(  # 定义初始化函数
        self,  # 定义实例引用
        input_ph,  # 定义输入占位符
        arch_code_ph,  # 定义架构编码占位符
        is_training_ph,  # 定义训练标志占位符
        num_out=2,  # 定义输出通道数
        init_neurons=32,  # 定义初始通道数
        expansion_factor=2.0,  # 定义通道扩展倍率
        suffix="",  # 定义作用域后缀
    ):  # 定义模型初始化结束
        super().__init__(is_training_ph=is_training_ph, suffix=suffix)  # 调用父类初始化
        self.input_ph = input_ph  # 保存输入张量
        self.arch_code_ph = arch_code_ph  # 保存架构编码张量
        self.num_out = int(num_out)  # 保存输出通道数
        self.init_neurons = int(init_neurons)  # 保存初始通道数
        self.expansion_factor = float(expansion_factor)  # 保存通道扩展倍率
        self._preds = None  # 初始化预测缓存

    def _select_by_index(self, candidates, index_tensor, name):  # 定义候选选择函数
        """通过one-hot门控选择候选张量。"""  # 说明函数用途
        stacked = tf.stack(candidates, axis=0, name=f"{name}_stack")  # 按候选维度堆叠张量
        weights = tf.one_hot(index_tensor, depth=len(candidates), dtype=stacked.dtype, name=f"{name}_onehot")  # 生成one-hot权重
        weights = tf.reshape(weights, [len(candidates), 1, 1, 1, 1], name=f"{name}_reshape")  # 重塑权重形状
        mixed = tf.reduce_sum(stacked * weights, axis=0, name=f"{name}_mix")  # 按权重融合候选张量
        return mixed  # 返回融合后的选中张量

    def _res_block(self, inputs, filters, name):  # 定义残差块函数
        """执行基础残差块计算。"""  # 说明函数用途
        with tf.compat.v1.variable_scope(name):  # 进入残差块作用域
            net = self.conv_bn_relu(inputs=inputs, filters=filters, strides=(1, 1), name="conv1")  # 执行首层卷积块
            net = self.conv(inputs=net, filters=filters, strides=(1, 1), activation=None, name="conv2")  # 执行第二层卷积
            net = self.bn(inputs=net, name="bn2")  # 执行第二层BN
            net = tf.add(net, inputs, name="res_add")  # 执行残差连接
            net = self.relu(inputs=net, name="res_relu")  # 执行残差激活
            return net  # 返回残差块输出

    def _deep_choice_block(self, inputs, filters, choice_idx, name):  # 定义深度选择块函数
        """根据选择索引执行Deep1/2/3。"""  # 说明函数用途
        with tf.compat.v1.variable_scope(name):  # 进入深度选择作用域
            out1 = self._res_block(inputs=inputs, filters=filters, name="deep1")  # 构建Deep1候选
            out2 = self._res_block(inputs=out1, filters=filters, name="deep2")  # 构建Deep2候选
            out3 = self._res_block(inputs=out2, filters=filters, name="deep3")  # 构建Deep3候选
            return self._select_by_index([out1, out2, out3], choice_idx, name="deep_select")  # 按索引选择输出

    def _head_choice_conv(self, inputs, filters, choice_idx, name):  # 定义头部选择卷积函数
        """根据选择索引执行7/5/3卷积候选。"""  # 说明函数用途
        with tf.compat.v1.variable_scope(name):  # 进入头部选择作用域
            conv7 = self.conv(inputs=inputs, filters=filters, kernel_size=(7, 7), strides=(1, 1), activation=None, name="k7")  # 构建7x7候选
            conv5 = self.conv(inputs=inputs, filters=filters, kernel_size=(5, 5), strides=(1, 1), activation=None, name="k5")  # 构建5x5候选
            conv3 = self.conv(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(1, 1), activation=None, name="k3")  # 构建3x3候选
            return self._select_by_index([conv7, conv5, conv3], choice_idx, name="kernel_select")  # 按索引选择输出

    def build(self):  # 定义网络构建函数
        """构建超网前向并返回多尺度输出。"""  # 说明函数用途
        if self._preds is not None:  # 判断是否已构建前向图
            return self._preds  # 已构建时直接返回缓存输出
        arch = self.arch_code_ph  # 读取架构编码张量
        c1 = int(self.init_neurons * self.expansion_factor)  # 计算第一阶段通道数
        c2 = int(c1 * self.expansion_factor)  # 计算第二阶段通道数
        c3 = int(c2 * self.expansion_factor)  # 计算第三阶段通道数
        c4 = int(c3 / self.expansion_factor)  # 计算解码第一阶段通道数
        c5 = int(c4 / self.expansion_factor)  # 计算解码第二阶段通道数
        with tf.compat.v1.variable_scope("supernet_backbone"):  # 进入骨干网络作用域
            net = self.conv_bn_relu(inputs=self.input_ph, filters=self.init_neurons, kernel_size=(7, 7), strides=(1, 1), name="E0")  # 构建E0层
            net = self.conv_bn_relu(inputs=net, filters=c1, kernel_size=(5, 5), strides=(1, 1), name="E1")  # 构建E1层
            net = self._deep_choice_block(inputs=net, filters=c1, choice_idx=arch[0], name="EB0")  # 构建EB0选择块
            net = self.conv(inputs=net, filters=c2, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down1Conv")  # 构建下采样层1
            net = self.bn(inputs=net, name="Down1BN")  # 构建下采样层1BN
            net = self.relu(inputs=net, name="Down1ReLU")  # 构建下采样层1激活
            net = self._deep_choice_block(inputs=net, filters=c2, choice_idx=arch[1], name="EB1")  # 构建EB1选择块
            net = self.conv(inputs=net, filters=c3, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down2Conv")  # 构建下采样层2
            net = self.bn(inputs=net, name="Down2BN")  # 构建下采样层2BN
            net = self.relu(inputs=net, name="Down2ReLU")  # 构建下采样层2激活
            net_low = self._deep_choice_block(inputs=net, filters=c3, choice_idx=arch[2], name="DB0")  # 构建低尺度DB0特征
            net_mid = self.resize_conv(inputs=net_low, filters=c4, kernel_size=(3, 3), name="Up1")  # 构建到1/2尺度的上采样特征
            net_mid = self.bn(inputs=net_mid, name="Up1BN")  # 构建上采样层1BN
            net_mid = self.relu(inputs=net_mid, name="Up1ReLU")  # 构建上采样层1激活
            net_mid = self._deep_choice_block(inputs=net_mid, filters=c4, choice_idx=arch[3], name="DB1")  # 构建中尺度DB1特征
            net_high = self.resize_conv(inputs=net_mid, filters=c5, kernel_size=(3, 3), name="Up2")  # 构建到全尺度的上采样特征
            net_high = self.bn(inputs=net_high, name="Up2BN")  # 构建上采样层2BN
            net_high = self.relu(inputs=net_high, name="Up2ReLU")  # 构建上采样层2激活
        with tf.compat.v1.variable_scope("supernet_head"):  # 进入头部网络作用域
            out_1_4 = self._head_choice_conv(inputs=net_low, filters=self.num_out, choice_idx=arch[4], name="H0Out")  # 构建1/4尺度H0Out输出
            h1 = self._head_choice_conv(inputs=net_mid, filters=c4, choice_idx=arch[5], name="H1")  # 构建1/2尺度H1特征
            h1 = self.bn(inputs=h1, name="H1BN")  # 构建H1批归一化
            h1 = self.relu(inputs=h1, name="H1ReLU")  # 构建H1激活
            out_1_2 = self._head_choice_conv(inputs=h1, filters=self.num_out, choice_idx=arch[6], name="H1Out")  # 构建1/2尺度H1Out输出
            h2 = self._head_choice_conv(inputs=net_high, filters=c5, choice_idx=arch[7], name="H2")  # 构建全尺度H2特征
            h2 = self.bn(inputs=h2, name="H2BN")  # 构建H2批归一化
            h2 = self.relu(inputs=h2, name="H2ReLU")  # 构建H2激活
            out_1_1 = self._head_choice_conv(inputs=h2, filters=self.num_out, choice_idx=arch[8], name="H2Out")  # 构建全尺度H2Out输出
        self._preds = [out_1_4, out_1_2, out_1_1]  # 缓存多尺度输出
        return self._preds  # 返回多尺度输出列表
