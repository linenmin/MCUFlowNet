import os
import sys

# 抑制 TF 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import pandas as pd
import shutil

# 添加项目根目录到路径以导入 vela 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from vela.vela_compiler import run_vela

def generate_sandwiched_op_model(op_type, input_shape, filters, output_path):
    """生成包含单个算子的 TFLite 模型，并在前后包裹 Conv 层以模拟真实环境"""
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    # 输入 Placeholder
    in_ph = tf.compat.v1.placeholder(tf.float32, shape=[1] + list(input_shape), name='Input')

    # 1. 前置卷积 (Pre-Conv): 固定输出 32 通道，确保输入环境稳定
    pre_net = tf.compat.v1.layers.conv2d(
        inputs=in_ph,
        filters=2,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        activation=tf.nn.relu,
        name='PreConv'
    )

    # 2. 目标测试算子 (Target Op): 通道数可变 (filters)
    if op_type == "Conv2D":
        # 使用 3x3 卷积，步长 1，Same Padding
        net = tf.compat.v1.layers.conv2d(
            inputs=pre_net,
            filters=filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='TestConv2D'
        )
    elif op_type == "Conv2DTranspose":
        # 使用 3x3 转置卷积，步长 2 (上采样)，Same Padding
        # 注意: 转置卷积会放大分辨率
        net = tf.compat.v1.layers.conv2d_transpose(
            inputs=pre_net,
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='TestConv2DTranspose'
        )
    elif op_type == "DepthwiseConv2D":
        # 深度可分离卷积 (Depthwise 部分)
        # 注意: Depthwise 的输出通道数通常等于输入通道数 * multiplier
        # 这里为了测试通道对齐，我们实际上是在测 Pointwise 的部分或者 SeparableConv 的整体输出
        # 使用 separable_conv2d 模拟，filters 参数控制最终输出通道
        net = tf.compat.v1.layers.separable_conv2d(
            inputs=pre_net,
            filters=filters, 
            kernel_size=(3, 3),
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            name='TestSeparableConv2D'
        )
    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    # 3. 后置卷积 (Post-Conv): 固定输出 16 通道，强制 NPU 缓存上一层的输出
    # 这能逼迫编译器分配中间 Buffer，而不是优化掉
    post_net = tf.compat.v1.layers.conv2d(
        inputs=net,
        filters=2,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        activation=tf.nn.relu,
        name='PostConv'
    )

    # 输出节点
    out = tf.identity(post_net, name='Output')

    # 初始化并导出
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # TFLite 转换
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [in_ph], [out])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 代表性数据集 (随机)
        def representative_dataset_gen():
            for _ in range(5):
                yield [np.random.rand(1, *input_shape).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset_gen
        # 强制 INT8 输入输出
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    
    return output_path

import datetime

def main():
    # 测试配置
    input_shape = (64, 64, 32) # 固定输入大小
    test_channels = [1, 8, 14, 15, 16, 17, 24, 32, 58, 64, 88, 128] # 测试的通道数列表
    op_types = ["Conv2D", "Conv2DTranspose", "DepthwiseConv2D"] # 添加了转置卷积和深度可分离卷积
    optimise_mode = "Size"
    
    output_dir = os.path.join(current_dir, "alignment_test_output")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"alignment_report_Conv2D_Opt{optimise_mode}_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)

    # 准备写入文件
    with open(report_path, "w", encoding="utf-8") as f:
        # 写入 Report Header
        header = (
            "============================================================\n"
            f"Experiment: Ethos-U Channel Alignment Verification\n"
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Input Shape: {input_shape}\n"
            f"Op Type(s): {op_types}\n"
            f"Vela Optimise Mode: {optimise_mode}\n"
            "============================================================\n\n"
            f"{'Op Type':<15} | {'Channels':<10} | {'SRAM (KiB)':<15} | {'Diff (KiB)':<10}\n"
            f"{'-' * 60}\n"
        )
        f.write(header)
        print(f"[*] Report file: {report_path}")
        print(header.strip()) # 打印 Header 到控制台

        results = []

        for op in op_types:
            prev_sram = 0
            for ch in test_channels:
                model_name = f"test_{op}_ch{ch}.tflite"
                model_path = os.path.join(output_dir, model_name)
                
                # 1. 生成模型
                try:
                    generate_sandwiched_op_model(op, input_shape, ch, model_path)
                except Exception as e:
                    error_msg = f"[!] 模型生成失败 ch={ch}: {e}"
                    print(error_msg)
                    f.write(f"Error: {error_msg}\n")
                    continue

                # 2. Vela 编译
                vela_out_dir = os.path.join(output_dir, f"vela_out_{op}_{ch}")
                sram_mb, _ = run_vela(model_path, mode="basic", output_dir=vela_out_dir, optimise=optimise_mode, silent=True)
                
                if sram_mb is not None:
                    sram_kib = sram_mb * 1024
                    diff = sram_kib - prev_sram if prev_sram > 0 else 0
                    
                    line = f"{op:<15} | {ch:<10} | {sram_kib:<15.2f} | {diff:<10.2f}"
                    print(line)
                    f.write(line + "\n")
                    f.flush()
                    
                    results.append({
                        "Op": op,
                        "Channels": ch,
                        "SRAM_KiB": sram_kib
                    })
                    prev_sram = sram_kib
                else:
                    line = f"{op:<15} | {ch:<10} | {'Failed':<15} | {'-':<10}"
                    print(line)
                    f.write(line + "\n")
                    f.flush()

        footer = "-" * 60 + "\n[*] 测试完成。"
        print(footer)
        f.write(footer + "\n")

if __name__ == "__main__":
    main()
