#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 ONNX 模型中间层形状
验证 Crop Conv 是否正确裁剪
"""

import numpy as np
import onnx
from onnx import shape_inference

def check_shapes(model_path):
    """检查所有节点的输入输出形状"""
    print(f"加载模型: {model_path}")
    model = onnx.load(model_path)
    
    # Shape inference
    print("执行形状推断...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"形状推断失败: {e}")
        return
    
    # 创建形状字典
    shape_dict = {}
    for vi in model.graph.value_info:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    for vi in model.graph.input:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    for vi in model.graph.output:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    
    print("\n" + "=" * 80)
    print("ConvTranspose 和 Crop Conv 节点形状分析")
    print("=" * 80)
    
    for node in model.graph.node:
        if "ConvTranspose" in node.name or "_crop" in node.name.lower():
            print(f"\n节点: {node.op_type} - {node.name[:60]}...")
            
            # 输入形状
            for i, inp in enumerate(node.input):
                if inp in shape_dict:
                    print(f"  输入[{i}]: {shape_dict[inp]}")
            
            # 输出形状
            for i, out in enumerate(node.output):
                if out in shape_dict:
                    print(f"  输出[{i}]: {shape_dict[out]}")
                else:
                    print(f"  输出[{i}]: (形状未知)")
            
            # 对于 Conv (Crop)，显示 kernel 和 group
            if node.op_type == "Conv":
                for attr in node.attribute:
                    if attr.name in ["kernel_shape", "group", "pads"]:
                        if attr.name == "group":
                            print(f"  {attr.name}: {attr.i}")
                        else:
                            print(f"  {attr.name}: {list(attr.ints)}")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "edgeflownet_576_1024.onnx"
    check_shapes(model_path)
