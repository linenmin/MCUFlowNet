"""评估指标构建工具。"""  # 定义模块用途

import tensorflow as tf  # 导入TensorFlow模块


def build_epe_metric(pred_tensor: tf.Tensor, label_ph: tf.Tensor) -> tf.Tensor:  # 定义EPE指标构建函数
    """构建平均端点误差(EPE)指标。"""  # 说明函数用途
    pred_size = [int(pred_tensor.shape[1]), int(pred_tensor.shape[2])]  # 读取预测输出空间尺寸
    label_resized = tf.compat.v1.image.resize(label_ph, size=pred_size, method=tf.image.ResizeMethod.BILINEAR)  # 对标签执行同尺寸缩放
    diff = pred_tensor - label_resized  # 计算预测与标签差值
    sq = tf.square(diff)  # 计算逐元素平方误差
    sum_sq = tf.reduce_sum(sq, axis=3)  # 按通道聚合平方误差
    epe_map = tf.sqrt(sum_sq + 1e-6)  # 计算逐像素端点误差
    return tf.reduce_mean(epe_map, name="mean_epe")  # 返回平均端点误差

