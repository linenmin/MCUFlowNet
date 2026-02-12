#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os  # 路径处理
import sys  # 系统退出
import argparse  # 命令行参数
import subprocess  # 运行 Vela
import numpy as np  # 数值库
import pandas as pd  # 读取 CSV
import tensorflow as tf  # TF1 兼容模式
from vela.vela_compiler import run_vela  # 引用 vela_compiler 中的编译函数

# 关闭 eager，保持 TF1 图模式
tf.compat.v1.disable_eager_execution()

# Vela 配置（与 auto_benchmark_extract_tflite.py 保持一致）
vela_config_file = "vela.ini"  # Vela 配置文件
sys_config = "Grove_Sys_Config"  # 系统配置名
mem_mode = "Grove_Mem_Mode"  # 内存模式
accelerator = "ethos-u55-64"  # 加速器类型
output_dir = "output_benchmark"  # Vela 输出目录
optimise = "Performance"  # 优化目标

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


# ---------- 基础算子 ----------
def conv_bn_relu(x, filters, k, s, name):  # 标准卷积+BN+ReLU
    with tf.compat.v1.variable_scope(name):
        x = tf.compat.v1.layers.conv2d(x, filters, k, s, padding="same", use_bias=False)
        x = tf.compat.v1.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5)
        x = tf.nn.relu(x)
        return x


def dwconv_bn_relu(x, k, s, name):  # 深度可分卷积+BN+ReLU
    with tf.compat.v1.variable_scope(name):
        in_c = x.get_shape().as_list()[-1]
        x = tf.compat.v1.layers.separable_conv2d(
            x, in_c, k, s, padding="same", depth_multiplier=1, use_bias=False
        )
        x = tf.compat.v1.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5)
        x = tf.nn.relu(x)
        return x


# ---------- MBConv (无 SE) ----------
def mbconv_block(x, filters_out, stride, expand_ratio, name):  # MBConv: 1x1扩展-3x3 DW-1x1投影
    in_c = x.get_shape().as_list()[-1]
    hidden_c = int(in_c * expand_ratio)
    with tf.compat.v1.variable_scope(name):
        # 1x1 扩展
        out = tf.compat.v1.layers.conv2d(x, hidden_c, 1, 1, padding="same", use_bias=False)
        out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
        out = tf.nn.relu6(out)
        # 3x3 DW
        # 修复：使用 tf.nn.depthwise_conv2d 替代不存在的 layers.depthwise_conv2d
        in_shape = out.get_shape().as_list()
        filter_shape = [3, 3, in_shape[-1], 1]  # [H, W, in_channels, channel_multiplier]
        with tf.compat.v1.variable_scope("dw_conv"):
            dw_filter = tf.compat.v1.get_variable(
                "dw_filter", filter_shape,
                initializer=tf.compat.v1.initializers.glorot_uniform()
            )
        if stride == 1:
            out = tf.nn.depthwise_conv2d(out, dw_filter, [1, 1, 1, 1], padding="SAME")
        else:
            out = tf.nn.depthwise_conv2d(out, dw_filter, [1, stride, stride, 1], padding="SAME")
        out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
        out = tf.nn.relu6(out)
        # 1x1 投影
        out = tf.compat.v1.layers.conv2d(out, filters_out, 1, 1, padding="same", use_bias=False)
        out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
        # 残差连接（仅 stride=1 且通道相同）
        if stride == 1 and in_c == filters_out:
            out = tf.add(out, x)
        return out


# ---------- ShuffleNetV2 block ----------
def channel_shuffle(x, groups):  # 优化版：维度折叠 (Dimension Folding)
    # 获取维度，注意 x 可能是动态 shape，但在 Vela 编译时通常 batch=1
    n, h, w, c = x.get_shape().as_list()
    
    # 确保通道数能被 groups 整除
    assert c % groups == 0
    channels_per_group = c // groups

    # --- 关键修改开始 ---
    # 原逻辑 (5D): [N, H, W, groups, channels_per_group] -> 违规
    # 新逻辑 (4D): [N, H*W, groups, channels_per_group]
    # 我们将 H 和 W 合并，以此来“欺骗”编译器保持在 4D 限制内
    
    # 1. Reshape: 合并 H 和 W
    # 注意：这里假设 N (Batch Size) 也是确定的，通常在 NPU 推理时 N=1
    # 如果 N 是动态的，可以用 -1 代替
    x = tf.reshape(x, [-1, h * w, groups, channels_per_group])
    
    # 2. Transpose: 交换 groups (dim 2) 和 channels_per_group (dim 3)
    # 这里的维度索引基于 4D Tensor: [0:N, 1:HW, 2:Groups, 3:C_per_group]
    x = tf.transpose(x, [0, 1, 3, 2])
    
    # 3. Reshape: 还原回原始的 [N, H, W, C]
    x = tf.reshape(x, [-1, h, w, c])
    # --- 关键修改结束 ---
    
    return x


def shufflenetv2_block(x, out_c, stride, name):  # ShuffleNetV2 block（1:1 通道拆分）
    with tf.compat.v1.variable_scope(name):
        in_c = x.get_shape().as_list()[-1]
        mid_c = out_c // 2
        if stride == 1:
            x1, x2 = tf.split(x, num_or_size_splits=2, axis=3)
            # branch2
            x2 = tf.compat.v1.layers.conv2d(x2, mid_c, 1, 1, padding="same", use_bias=False)
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.nn.relu(x2)
            # 修复：使用 tf.nn.depthwise_conv2d
            x2_shape = x2.get_shape().as_list()
            with tf.compat.v1.variable_scope("branch2_dw"):
                dw_filter = tf.compat.v1.get_variable(
                    "dw_filter", [3, 3, x2_shape[-1], 1],
                    initializer=tf.compat.v1.initializers.glorot_uniform()
                )
            x2 = tf.nn.depthwise_conv2d(x2, dw_filter, [1, 1, 1, 1], padding="SAME")
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.compat.v1.layers.conv2d(x2, mid_c, 1, 1, padding="same", use_bias=False)
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.nn.relu(x2)
            out = tf.concat([x1, x2], axis=3)
        else:
            # stride=2，下采样双分支
            # 修复：使用 tf.nn.depthwise_conv2d，使用不同的 variable_scope 避免冲突
            x_shape = x.get_shape().as_list()
            with tf.compat.v1.variable_scope("branch1_dw"):
                dw_filter1 = tf.compat.v1.get_variable(
                    "dw_filter", [3, 3, x_shape[-1], 1],
                    initializer=tf.compat.v1.initializers.glorot_uniform()
                )
            x1 = tf.nn.depthwise_conv2d(x, dw_filter1, [1, stride, stride, 1], padding="SAME")
            x1 = tf.compat.v1.layers.batch_normalization(x1, momentum=0.9, epsilon=1e-5)
            x1 = tf.compat.v1.layers.conv2d(x1, mid_c, 1, 1, padding="same", use_bias=False)
            x1 = tf.compat.v1.layers.batch_normalization(x1, momentum=0.9, epsilon=1e-5)
            x1 = tf.nn.relu(x1)

            x2 = tf.compat.v1.layers.conv2d(x, mid_c, 1, 1, padding="same", use_bias=False)
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.nn.relu(x2)
            # 修复：使用 tf.nn.depthwise_conv2d
            x2_shape = x2.get_shape().as_list()
            with tf.compat.v1.variable_scope("branch2_dw"):
                dw_filter2 = tf.compat.v1.get_variable(
                    "dw_filter", [3, 3, x2_shape[-1], 1],
                    initializer=tf.compat.v1.initializers.glorot_uniform()
                )
            x2 = tf.nn.depthwise_conv2d(x2, dw_filter2, [1, stride, stride, 1], padding="SAME")
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.compat.v1.layers.conv2d(x2, mid_c, 1, 1, padding="same", use_bias=False)
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.nn.relu(x2)

            out = tf.concat([x1, x2], axis=3)
        out = channel_shuffle(out, 2)
        return out


# ---------- 宏观骨架：编码-解码，多尺度输出 ----------
def build_backbone(inputs, cfg):  # 构建骨架，block_fn 决定使用何种单元
    block_fn = cfg["block_fn"]
    init_c = cfg["init_c"]
    expand = cfg["expand"]
    num_sub = cfg["num_sub"]
    num_blocks = cfg["num_blocks"]
    num_out = cfg["num_out"]

    nets = []
    x = inputs
    for b in range(num_blocks):
        with tf.compat.v1.variable_scope(f"EncoderDecoderBlock{b}"):
            # 编码：两层大卷积
            c = init_c
            x = conv_bn_relu(x, c, (7, 7), (2, 2), "enc_stem1")
            c = int(c * expand)
            x = conv_bn_relu(x, c, (5, 5), (2, 2), "enc_stem2")

            # 编码子块：block + 下采样
            for i in range(num_sub):
                x = block_fn(x, c, stride=1, name=f"enc_block_{i}")
                c = int(c * expand)
                x = conv_bn_relu(x, c, (3, 3), (2, 2), f"enc_down_{i}")

            # 解码子块：上采样 + block
            for i in range(num_sub):
                c = int(c / expand)
                x = tf.compat.v1.layers.conv2d_transpose(
                    x, c, (3, 3), (2, 2), padding="same", use_bias=False, name=f"dec_up_{i}"
                )
                x = tf.compat.v1.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5)
                x = tf.nn.relu(x)
                x = block_fn(x, c, stride=1, name=f"dec_block_{i}")

            # 生成多尺度输出（与原始类似 3 个头）
            out1 = tf.compat.v1.layers.conv2d_transpose(
                x, num_out, (7, 7), (1, 1), padding="same", name="out_head1"
            )
            nets.append(out1)

            c = int(c / expand)
            x_tmp = tf.compat.v1.layers.conv2d_transpose(
                x, c, (5, 5), (2, 2), padding="same", use_bias=False, name="out_up2"
            )
            x_tmp = tf.nn.relu(x_tmp)
            out2 = tf.compat.v1.layers.conv2d_transpose(
                x_tmp, num_out, (7, 7), (1, 1), padding="same", name="out_head2"
            )
            nets.append(out2)

            c = int(c / expand)
            x_tmp = tf.compat.v1.layers.conv2d_transpose(
                x_tmp, c, (7, 7), (2, 2), padding="same", use_bias=False, name="out_up3"
            )
            x_tmp = tf.nn.relu(x_tmp)
            out3 = tf.compat.v1.layers.conv2d_transpose(
                x_tmp, num_out, (7, 7), (1, 1), padding="same", name="out_head3"
            )
            nets.append(out3)
    return nets


# ---------- 运行 Vela 并解析算子支持 ----------
def parse_npu_support(perf_file):  # 解析详细报告，输出 NPU 支持情况
    if not os.path.exists(perf_file):
        print(f"Warning: Detail report not found at {perf_file}")
        return

    npu_ops = set()
    cpu_ops = set()
    
    with open(perf_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_npu_table = False
    for line in lines:
        line = line.strip()
        
        # 1. 捕获 CPU 回退警告
        # 格式: Warning: Unsupported ... for QUANTIZE 'tfl.quantize'. Placing on CPU instead
        if "Placing on CPU instead" in line:
            parts = line.split(" for ")
            if len(parts) > 1:
                op_part = parts[1].split("'")[0].strip() # 提取 QUANTIZE
                # 有时候格式可能是 ... for DEQUANTIZE 'NodeName'. ...
                # 尝试提取第一个单词作为算子名
                op_name = op_part.split(" ")[0]
                cpu_ops.add(op_name)
        
        # 2. 捕获 NPU 表格
        if line.startswith("TFLite_operator") and "NNG Operator" in line:
            in_npu_table = True
            continue
        if in_npu_table and line.startswith("----"):
            continue
        if in_npu_table:
            if not line: # 空行结束表格
                in_npu_table = False
                continue
            # 提取第一列
            parts = line.split()
            if parts:
                op_name = parts[0]
                # 过滤一些非算子行（虽然表格一般很规整）
                if op_name != "Network": 
                    npu_ops.add(op_name)

    print("\n" + "="*20 + " NPU 算子支持统计 " + "="*20)
    if cpu_ops:
        print(f"[!] CPU 回退算子 (不支持): {', '.join(sorted(cpu_ops))}")
    else:
        print("[-] CPU 回退算子: 无 (完美！)")
        
    if npu_ops:
        print(f"[+] NPU 加速算子 (支持): {', '.join(sorted(npu_ops))}")
    else:
        print("[-] NPU 加速算子: 无")
    print("="*56 + "\n")


# ---------- 构建并导出 TFLite ----------
def build_and_export(cell_type, h, w, output_path):  # 构建模型并导出 tflite，然后运行 Vela
    print(f"[1/2] Generating TFLite for {cell_type} with resolution: {h}x{w} ...")
    
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        with sess.as_default():
            input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, h, w, 6], name="input_image")  # 输入占位

            if cell_type == "mbconv":
                block_fn = lambda x, c, stride, name: mbconv_block(x, c, stride, expand_ratio=6, name=name)
            elif cell_type == "shufflenet":
                block_fn = lambda x, c, stride, name: shufflenetv2_block(x, c, stride, name=name)
            else:
                raise ValueError("cell_type must be mbconv or shufflenet")

            cfg = {
                "block_fn": block_fn,
                "init_c": 32,
                "expand": 2.0,
                "num_sub": 2,
                "num_blocks": 1,
                "num_out": 4,
            }
            nets = build_backbone(input_ph, cfg)
            # 选择最后一个输出作为导出
            final_out = nets[-1]

            sess.run(tf.compat.v1.global_variables_initializer())

            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_out])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]

            def representative_dataset_gen():  # 代表性数据（随机）
                for _ in range(10):
                    yield [np.random.uniform(0.0, 1.0, size=[1, h, w, 6]).astype(np.float32)]

            converter.representative_dataset = representative_dataset_gen
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            tflite_model = converter.convert()

            with open(output_path, "wb") as f:
                f.write(tflite_model)
            print(f"Saved TFLite to {output_path}")
    
    # 运行 Vela (调用 vela_compiler.py)
    # 引用 vela_compiler.py 中的 run_vela 函数，使用 verbose 模式以生成详细报告
    output_dir = "output_benchmark"
    sram, time_ms = run_vela(output_path, mode="verbose", output_dir=output_dir)
    
    if sram is not None:
        fps = 1000.0 / time_ms if time_ms > 0 else 0
        print(f"\n=== Vela Report ===")
        print(f"SRAM Used: {sram:.2f} MB")
        print(f"Inference Time: {time_ms:.2f} ms")
        print(f"FPS: {fps:.2f}")
        
        # 解析详细报告以显示算子支持情况
        perf_file = os.path.join(output_dir, "detailed_performance.txt")
        parse_npu_support(perf_file)
        
        return sram # 返回 SRAM 占用作为简单标志
    else:
        print(f"\nVela compilation failed.")
        return None


def parse_args():  # 参数解析
    parser = argparse.ArgumentParser(description="Export EdgeFlowNet backbone with custom cells")
    parser.add_argument("--cell", default="mbconv", choices=["mbconv", "shufflenet"], help="选择 block 类型")
    parser.add_argument("--height", type=int, default=192, help="输入高度")
    parser.add_argument("--width", type=int, default=256, help="输入宽度")
    parser.add_argument("--output", default="edgeflownet_custom.tflite", help="输出 tflite 文件名")
    return parser.parse_args()


def main():  # 脚本主入口
    args = parse_args()
    build_and_export(args.cell, args.height, args.width, args.output)


if __name__ == "__main__":  # 入口
    main()

