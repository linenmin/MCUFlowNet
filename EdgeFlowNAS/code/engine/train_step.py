"""训练步图构建工具。"""  # 定义模块用途

from typing import Dict, List  # 导入类型注解

import tensorflow as tf  # 导入TensorFlow模块


def build_multiscale_l1_loss(preds: List[tf.Tensor], label_ph: tf.Tensor) -> tf.Tensor:  # 定义多尺度L1损失函数
    """构建累积式多尺度L1损失。"""  # 说明函数用途
    loss_scales = [0.125, 0.25, 0.5]  # 定义三尺度损失权重
    pred_accum = None  # 初始化累计预测张量
    loss_terms = []  # 初始化损失项列表
    for pred_tensor, weight in zip(preds, loss_scales):  # 遍历多尺度预测与权重
        if pred_accum is None:  # 判断是否为第一个尺度
            pred_accum = pred_tensor  # 初始化累计预测
        else:  # 处理后续尺度
            new_size = [int(pred_tensor.shape[1]), int(pred_tensor.shape[2])]  # 获取当前尺度尺寸
            pred_accum = tf.compat.v1.image.resize(pred_accum, size=new_size, method=tf.image.ResizeMethod.BILINEAR)  # 上采样累计预测
            pred_accum = tf.add(pred_accum, pred_tensor, name="pred_accum_add")  # 累加当前尺度预测
        label_i = tf.compat.v1.image.resize(label_ph, size=[int(pred_accum.shape[1]), int(pred_accum.shape[2])], method=tf.image.ResizeMethod.BILINEAR)  # 缩放标签到当前尺度
        loss_i = tf.reduce_mean(tf.abs(pred_accum - label_i), name="loss_abs_mean")  # 计算当前尺度L1损失
        loss_terms.append(float(weight) * loss_i)  # 记录加权损失项
    return tf.add_n(loss_terms, name="multiscale_l1_loss")  # 汇总并返回总损失


def add_weight_decay(loss_tensor: tf.Tensor, weight_decay: float) -> tf.Tensor:  # 定义权重衰减叠加函数
    """将L2权重衰减项叠加到损失。"""  # 说明函数用途
    reg_terms = []  # 初始化正则项列表
    for var in tf.compat.v1.trainable_variables():  # 遍历可训练变量
        if "kernel" in var.name or "weights" in var.name:  # 筛选卷积/全连接权重变量
            reg_terms.append(tf.nn.l2_loss(var))  # 添加L2正则项
    if reg_terms and float(weight_decay) > 0.0:  # 判断是否启用权重衰减
        loss_tensor = tf.add(loss_tensor, float(weight_decay) * tf.add_n(reg_terms), name="loss_with_weight_decay")  # 叠加权重衰减损失
    return loss_tensor  # 返回叠加后的损失张量
