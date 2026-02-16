"""训练步图构建工具。"""  # 定义模块用途
from typing import List  # 导入类型注解工具

import tensorflow as tf  # 导入TensorFlow模块


def _resize_like(src: tf.Tensor, ref: tf.Tensor, name: str) -> tf.Tensor:  # 定义按参考张量缩放函数
    """将源张量缩放到参考张量空间尺寸。"""  # 说明函数用途
    target_size = [int(ref.shape[1]), int(ref.shape[2])]  # 读取参考张量目标尺寸
    return tf.compat.v1.image.resize(src, size=target_size, method=tf.image.ResizeMethod.BILINEAR, name=name)  # 返回双线性缩放结果


def _split_flow_and_uncertainty(pred_tensor: tf.Tensor, num_out: int) -> List[tf.Tensor]:  # 定义预测分拆函数
    """将预测分拆为光流分支和不确定性分支。"""  # 说明函数用途
    flow_pred = pred_tensor[:, :, :, : int(num_out)]  # 切片提取光流分支
    unc_pred = pred_tensor[:, :, :, int(num_out) : int(num_out) * 2]  # 切片提取不确定性分支
    return [flow_pred, unc_pred]  # 返回分拆后的两个分支


def build_multiscale_l1_loss(preds: List[tf.Tensor], label_ph: tf.Tensor) -> tf.Tensor:  # 定义多尺度L1损失函数
    """构建累积式多尺度L1损失。"""  # 说明函数用途
    loss_scales = [0.125, 0.25, 0.5]  # 定义三尺度损失权重
    pred_accum = None  # 初始化累计预测张量
    loss_terms = []  # 初始化损失项列表
    for idx, (pred_tensor, weight) in enumerate(zip(preds, loss_scales)):  # 遍历多尺度预测与权重
        if pred_accum is None:  # 判断是否为首尺度
            pred_accum = pred_tensor  # 首尺度直接作为累计预测
        else:  # 处理后续尺度累计逻辑
            pred_accum = _resize_like(src=pred_accum, ref=pred_tensor, name=f"pred_accum_resize_{idx}")  # 将累计预测上采样到当前尺度
            pred_accum = tf.add(pred_accum, pred_tensor, name=f"pred_accum_add_{idx}")  # 累加当前尺度预测
        label_i = _resize_like(src=label_ph, ref=pred_accum, name=f"label_resize_{idx}")  # 将标签缩放到当前尺度
        loss_i = tf.reduce_mean(tf.abs(pred_accum - label_i), name=f"loss_abs_mean_{idx}")  # 计算当前尺度L1损失
        loss_terms.append(float(weight) * loss_i)  # 记录加权损失项
    return tf.add_n(loss_terms, name="multiscale_l1_loss")  # 汇总并返回总损失


def build_multiscale_uncertainty_loss(preds: List[tf.Tensor], label_ph: tf.Tensor, num_out: int) -> tf.Tensor:  # 定义多尺度不确定性损失函数
    """构建与单模型训练对齐的LinearSoftplus多尺度损失。"""  # 说明函数用途
    loss_scales = [0.125, 0.25, 0.5, 1.0]  # 定义与原训练一致的尺度权重
    eps = 1e-3  # 定义稳定项避免除零
    flow_accum = None  # 初始化累计光流预测
    logvar_accum = None  # 初始化累计不确定性预测
    optical_terms = []  # 初始化光流L1损失项列表
    uncertainty_terms = []  # 初始化不确定性损失项列表
    for idx, (pred_tensor, weight) in enumerate(zip(preds, loss_scales)):  # 遍历多尺度预测与权重
        flow_pred, unc_pred = _split_flow_and_uncertainty(pred_tensor=pred_tensor, num_out=num_out)  # 分拆当前尺度预测
        if flow_accum is None:  # 判断是否为首尺度
            flow_accum = flow_pred  # 首尺度直接作为累计光流预测
            logvar_accum = unc_pred  # 首尺度直接作为累计不确定性预测
        else:  # 处理后续尺度累计逻辑
            flow_accum = _resize_like(src=flow_accum, ref=flow_pred, name=f"flow_accum_resize_{idx}")  # 上采样累计光流到当前尺度
            flow_accum = tf.add(flow_accum, flow_pred, name=f"flow_accum_add_{idx}")  # 累加当前尺度光流预测
            logvar_accum = _resize_like(src=logvar_accum, ref=unc_pred, name=f"logvar_accum_resize_{idx}")  # 上采样累计不确定性到当前尺度
            logvar_accum = tf.add(logvar_accum, unc_pred, name=f"logvar_accum_add_{idx}")  # 累加当前尺度不确定性预测
        label_i = _resize_like(src=label_ph, ref=flow_accum, name=f"unc_label_resize_{idx}")  # 缩放标签到当前尺度
        abs_diff = tf.abs(flow_accum - label_i, name=f"unc_abs_diff_{idx}")  # 计算累计光流绝对误差
        loss_optical = tf.reduce_mean(abs_diff, name=f"unc_optical_l1_{idx}")  # 计算普通L1损失项
        sigma = tf.maximum(tf.math.softplus(logvar_accum + eps), eps, name=f"unc_sigma_{idx}")  # 计算稳定化方差项
        loss_unc_data = tf.reduce_mean((1.0 / sigma) * abs_diff, name=f"unc_data_{idx}")  # 计算不确定性加权重建项
        loss_unc_reg = tf.reduce_mean(tf.math.softplus(logvar_accum), name=f"unc_reg_{idx}")  # 计算不确定性正则项
        loss_unc = tf.add(loss_unc_data, loss_unc_reg, name=f"unc_total_{idx}")  # 汇总当前尺度不确定性损失
        optical_terms.append(float(weight) * loss_optical)  # 记录加权光流L1损失项
        uncertainty_terms.append(float(weight) * loss_unc)  # 记录加权不确定性损失项
    optical_total = tf.add_n(optical_terms, name="unc_optical_total")  # 汇总全部光流L1损失项
    uncertainty_total = tf.add_n(uncertainty_terms, name="unc_uncertainty_total")  # 汇总全部不确定性损失项
    return tf.add(optical_total, uncertainty_total, name="multiscale_uncertainty_loss")  # 返回最终联合损失


def add_weight_decay(loss_tensor: tf.Tensor, weight_decay: float) -> tf.Tensor:  # 定义权重衰减叠加函数
    """将L2权重衰减项叠加到损失。"""  # 说明函数用途
    reg_terms = []  # 初始化正则项列表
    for var in tf.compat.v1.trainable_variables():  # 遍历可训练变量
        if "kernel" in var.name or "weights" in var.name:  # 筛选卷积/全连接权重变量
            reg_terms.append(tf.nn.l2_loss(var))  # 添加L2正则项
    if reg_terms and float(weight_decay) > 0.0:  # 判断是否启用权重衰减
        loss_tensor = tf.add(loss_tensor, float(weight_decay) * tf.add_n(reg_terms), name="loss_with_weight_decay")  # 叠加权重衰减损失
    return loss_tensor  # 返回叠加后的损失张量
