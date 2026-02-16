"""Supernet 输出形状检查脚本。"""  # 定义模块用途

import argparse  # 导入参数解析模块
import json  # 导入JSON模块
import random  # 导入随机模块

import numpy as np  # 导入NumPy模块
import tensorflow as tf  # 导入TensorFlow模块

from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet  # 导入超网模型实现


def _build_parser():  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="model shape check for supernet")  # 创建解析器对象
    parser.add_argument("--h", type=int, default=180, help="input height")  # 添加输入高度参数
    parser.add_argument("--w", type=int, default=240, help="input width")  # 添加输入宽度参数
    parser.add_argument("--batch", type=int, default=2, help="batch size")  # 添加批大小参数
    parser.add_argument("--samples", type=int, default=20, help="random sample count")  # 添加采样数量参数
    parser.add_argument("--seed", type=int, default=42, help="random seed")  # 添加随机种子参数
    return parser  # 返回解析器对象


def main():  # 定义主函数
    """执行超网输出形状检查。"""  # 说明函数用途
    parser = _build_parser()  # 构建命令行参数解析器
    args = parser.parse_args()  # 解析命令行参数
    tf.compat.v1.disable_eager_execution()  # 关闭Eager以使用TF1图模式
    tf.compat.v1.reset_default_graph()  # 重置默认计算图
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[args.batch, args.h, args.w, 6], name="Input")  # 创建输入占位符
    arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[9], name="ArchCode")  # 创建架构编码占位符
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")  # 创建训练标志占位符
    model = MultiScaleResNetSupernet(  # 创建超网模型实例
        input_ph=input_ph,  # 传入输入占位符
        arch_code_ph=arch_code_ph,  # 传入架构编码占位符
        is_training_ph=is_training_ph,  # 传入训练标志占位符
        num_out=4,  # 传入输出通道数
        init_neurons=32,  # 传入初始通道数
        expansion_factor=2.0,  # 传入通道扩展倍率
    )
    preds = model.build()  # 构建模型前向输出
    rng = random.Random(args.seed)  # 创建随机数生成器
    dummy_input = np.zeros([args.batch, args.h, args.w, 6], dtype=np.float32)  # 创建零输入样本
    last_shapes = None  # 初始化最后一次输出形状
    with tf.compat.v1.Session() as sess:  # 创建TensorFlow会话
        sess.run(tf.compat.v1.global_variables_initializer())  # 初始化全局变量
        for _ in range(args.samples):  # 按采样次数循环测试
            arch_code = [rng.randint(0, 2) for _ in range(9)]  # 生成随机架构编码
            out_1_4, out_1_2, out_1_1 = sess.run(  # 执行前向推理
                preds,  # 指定输出张量列表
                feed_dict={  # 传入推理输入字典
                    input_ph: dummy_input,  # 传入输入样本
                    arch_code_ph: arch_code,  # 传入架构编码
                    is_training_ph: False,  # 指定推理模式
                },
            )
            last_shapes = {  # 记录最后一次输出形状
                "out_1_4": [int(out_1_4.shape[1]), int(out_1_4.shape[2])],  # 记录1/4尺度输出
                "out_1_2": [int(out_1_2.shape[1]), int(out_1_2.shape[2])],  # 记录1/2尺度输出
                "out_1_1": [int(out_1_1.shape[1]), int(out_1_1.shape[2])],  # 记录1/1尺度输出
            }
    print(json.dumps({"status": "ok", "last_shapes": last_shapes}, ensure_ascii=False, indent=2))  # 打印检查结果
    return 0  # 返回成功状态码


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出
