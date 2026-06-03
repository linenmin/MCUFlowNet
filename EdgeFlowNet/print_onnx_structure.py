#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印 ONNX 模型结构
用于分析模型层次、形状和潜在问题
"""

import sys
import onnx
from onnx import shape_inference
from pathlib import Path


def get_tensor_shape(vi):
    """获取张量的形状信息"""
    shape = []
    for dim in vi.type.tensor_type.shape.dim:
        if dim.dim_value:  # 固定维度
            shape.append(dim.dim_value)
        elif dim.dim_param:  # 动态维度
            shape.append(dim.dim_param)
        else:  # 未知维度
            shape.append("?")
    return shape


def print_model_structure(model_path):
    """打印模型结构"""
    print("=" * 70)
    print(f"分析模型: {model_path}")
    print("=" * 70)
    
    # 加载模型
    try:
        model = onnx.load(model_path)
        print("✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 运行形状推断
    try:
        model = shape_inference.infer_shapes(model)
        print("✅ 形状推断成功\n")
    except Exception as e:
        print(f"⚠️ 形状推断失败: {e}\n")
    
    # 收集所有张量的形状信息
    tensor_shapes = {}
    for vi in model.graph.value_info:  # 中间层
        tensor_shapes[vi.name] = get_tensor_shape(vi)
    for inp in model.graph.input:  # 输入
        tensor_shapes[inp.name] = get_tensor_shape(inp)
    for out in model.graph.output:  # 输出
        tensor_shapes[out.name] = get_tensor_shape(out)
    
    # 1. 输入信息
    print("[1] 输入信息")
    print("-" * 70)
    for inp in model.graph.input:
        shape = tensor_shapes.get(inp.name, [])
        print(f"  名称: {inp.name}")
        print(f"  形状: {shape}")
        print(f"  类型: {inp.type.tensor_type.elem_type}")
        print()
    
    # 2. 输出信息
    print("[2] 输出信息")
    print("-" * 70)
    for out in model.graph.output:
        shape = tensor_shapes.get(out.name, [])
        print(f"  名称: {out.name}")
        print(f"  形状: {shape}")
        print(f"  类型: {out.type.tensor_type.elem_type}")
        print()
    
    # 3. 节点统计
    print("[3] 节点统计")
    print("-" * 70)
    op_counts = {}
    for node in model.graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    print(f"  总节点数: {len(model.graph.node)}")
    print(f"  算子类型数: {len(op_counts)}")
    print("\n  算子统计:")
    for op_type, count in sorted(op_counts.items()):
        print(f"    {op_type}: {count}")
    print()
    
    # 4. 关键节点详情（Resize, Conv, ConvTranspose等）
    print("[4] 关键节点详情")
    print("-" * 70)
    key_ops = ["Resize", "Conv", "ConvTranspose", "Slice", "Concat", "Add"]
    for node in model.graph.node:
        if node.op_type in key_ops:
            print(f"  节点: {node.name}")
            print(f"    类型: {node.op_type}")
            print(f"    输入: {node.input}")
            print(f"    输出: {node.output}")
            
            # 打印输入输出形状
            for inp_name in node.input:
                if inp_name in tensor_shapes:
                    print(f"      输入 {inp_name}: {tensor_shapes[inp_name]}")
            for out_name in node.output:
                if out_name in tensor_shapes:
                    print(f"      输出 {out_name}: {tensor_shapes[out_name]}")
            
            # 打印属性
            if node.attribute:
                print(f"    属性:")
                for attr in node.attribute:
                    if attr.type == 1:  # FLOAT
                        print(f"      {attr.name}: {attr.f}")
                    elif attr.type == 2:  # INT
                        print(f"      {attr.name}: {attr.i}")
                    elif attr.type == 3:  # STRING
                        print(f"      {attr.name}: {attr.s.decode('utf-8')}")
                    elif attr.type == 6:  # INTS
                        print(f"      {attr.name}: {list(attr.ints)}")
                    elif attr.type == 7:  # FLOATS
                        print(f"      {attr.name}: {list(attr.floats)}")
            print()
    
    # 5. 检查可疑尺寸（包含516或其他非预期尺寸）
    print("[5] 可疑尺寸检查")
    print("-" * 70)
    suspicious = []
    for name, shape in tensor_shapes.items():
        for dim in shape:
            if isinstance(dim, int) and dim > 1:
                # 检查是否是非预期尺寸
                if dim == 516 or (dim not in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 576, 288, 144, 72, 36, 1024, 512, 256, 128, 64, 32, 18, 9]):
                    suspicious.append((name, shape, dim))
                    break
    
    if suspicious:
        print(f"  ⚠️ 发现 {len(suspicious)} 个可疑尺寸节点:")
        for name, shape, bad_dim in suspicious[:20]:  # 只显示前20个
            short_name = name if len(name) < 50 else name[:47] + "..."
            print(f"    {short_name}")
            print(f"      形状: {shape}, 问题尺寸: {bad_dim}")
    else:
        print("  ✅ 未发现可疑尺寸")
    print()
    
    # 6. 通道数统计
    print("[6] 通道数统计 (NCHW格式的第2个维度)")
    print("-" * 70)
    channel_sizes = set()
    for name, shape in tensor_shapes.items():
        if len(shape) >= 2 and isinstance(shape[1], int):
            channel_sizes.add(shape[1])
    
    print(f"  所有通道数: {sorted(channel_sizes)}")
    print("\n  不是64倍数的通道数:")
    has_padding_issue = False
    for ch in sorted(channel_sizes):
        if ch % 64 != 0:
            next_multiple = ((ch + 63) // 64) * 64
            print(f"    {ch} → 需要padding到 {next_multiple}")
            if next_multiple == 516:
                print(f"      ⚠️ 这个通道数padding后会变成516!")
                has_padding_issue = True
    
    if not has_padding_issue:
        print("    ✅ 未发现会导致516的通道padding问题")
    print()
    
    print("=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    # 默认模型路径
    default_model = "multiscale_bilinear_576_1024.onnx"
    
    # 从命令行参数获取模型路径，如果没有则使用默认值
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = default_model
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"❌ 文件不存在: {model_path}")
        print(f"用法: python {sys.argv[0]} <模型路径>")
        print(f"示例: python {sys.argv[0]} {default_model}")
        sys.exit(1)
    
    # 打印模型结构
    print_model_structure(model_path)

