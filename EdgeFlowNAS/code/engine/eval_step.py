"""评估指标构建工具。"""  # 定义模块用途
import tensorflow as tf  # 导入TensorFlow模块


def extract_flow_prediction(pred_tensor: tf.Tensor, num_out: int = 2) -> tf.Tensor:  # 定义流场切片函数
    """从预测张量中提取光流主分支。"""  # 说明函数用途
    return pred_tensor[:, :, :, : int(num_out)]  # 返回前num_out个光流通道


def build_epe_metric(pred_tensor: tf.Tensor, label_ph: tf.Tensor, num_out: int = 2) -> tf.Tensor:  # 定义EPE指标构建函数
    """构建平均端点误差(EPE)指标。"""  # 说明函数用途
    flow_pred = extract_flow_prediction(pred_tensor=pred_tensor, num_out=num_out)  # 提取光流预测分支
    pred_size = [int(flow_pred.shape[1]), int(flow_pred.shape[2])]  # 读取预测输出空间尺寸
    label_resized = tf.compat.v1.image.resize(label_ph, size=pred_size, method=tf.image.ResizeMethod.BILINEAR)  # 对标签执行同尺度缩放
    diff = flow_pred - label_resized  # 计算预测与标签差值
    sq = tf.square(diff)  # 计算逐元素平方误差
    sum_sq = tf.reduce_sum(sq, axis=3)  # 按通道聚合平方误差
    epe_map = tf.sqrt(sum_sq + 1e-6)  # 计算逐像素端点误差
    return tf.reduce_mean(epe_map, name="mean_epe")  # 返回平均端点误差
