#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ONNX 模型推理是否正常工作
确认模型结构正确，所有层的输出形状匹配
"""

import numpy as np

try:
    import onnxruntime as ort
    import onnx
except ImportError:
    print("请先安装: pip install onnxruntime onnx")
    exit(1)


def check_all_node_shapes(model_path):
    """检查所有节点的输出形状，找出尺寸问题"""
    print("\n" + "=" * 60)
    print("[详细形状检查] 检查所有节点的输出形状")
    print("=" * 60)
    
    model = onnx.load(model_path)
    
    # 运行形状推断
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  形状推断失败: {e}")
        return
    
    # 收集所有值信息
    value_info = {}
    for vi in model.graph.value_info:
        shape = []
        for dim in vi.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append(dim.dim_param or "?")
        value_info[vi.name] = shape
    
    # 添加输入输出
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else "?")
        value_info[inp.name] = shape
    
    for out in model.graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else "?")
        value_info[out.name] = shape
    
    # 查找问题节点 (包含 516 或其他非预期尺寸)
    print("\n[可疑尺寸节点] (包含 516, 258, 129 等非 2 的幂次尺寸)")
    suspicious = []
    for name, shape in value_info.items():
        for dim in shape:
            if isinstance(dim, int) and dim > 1:
                # 检查是否是非预期尺寸 (不是 512, 256, 128, 64, 32, 576, 1024 等)
                expected = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 576, 288, 144, 72, 36, 18]
                if dim not in expected:
                    suspicious.append((name, shape, dim))
                    break
    
    if suspicious:
        for name, shape, bad_dim in suspicious[:20]:  # 只显示前20个
            print(f"  ⚠️  {name}: {shape} (问题尺寸: {bad_dim})")
    else:
        print("  ✅ 未发现可疑尺寸")
    
    # 统计 Resize 和 ConvTranspose 节点
    print("\n[Resize/ConvTranspose 节点统计]")
    resize_count = 0
    convtrans_count = 0
    for node in model.graph.node:
        if node.op_type == "Resize":
            resize_count += 1
            out_shape = value_info.get(node.output[0], "未知")
            print(f"  Resize: {node.name} -> {out_shape}")
        elif node.op_type == "ConvTranspose":
            convtrans_count += 1
            out_shape = value_info.get(node.output[0], "未知")
            print(f"  ConvTranspose: {node.name} -> {out_shape}")
    
    print(f"\n  总计: {resize_count} Resize, {convtrans_count} ConvTranspose")


def test_inference(model_path, input_shape=(1, 576, 1024, 6)):
    """测试 ONNX 模型推理"""
    print("=" * 60)
    print(f"测试模型: {model_path}")
    print(f"输入形状: {input_shape}")
    print("=" * 60)
    
    # 创建推理会话
    try:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 获取输入输出信息
    input_info = sess.get_inputs()[0]
    output_info = sess.get_outputs()[0]
    
    print(f"\n[输入信息]")
    print(f"  名称: {input_info.name}")
    print(f"  形状: {input_info.shape}")
    print(f"  类型: {input_info.type}")
    
    print(f"\n[输出信息]")
    print(f"  名称: {output_info.name}")
    print(f"  形状: {output_info.shape}")
    print(f"  类型: {output_info.type}")
    
    # 创建随机输入
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # 运行推理
    print(f"\n[推理测试]")
    try:
        outputs = sess.run(None, {input_info.name: input_data})
        output = outputs[0]
        print(f"  ✅ 推理成功!")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
        return True
    except Exception as e:
        print(f"  ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "edgeflownet_576_1024.onnx"
    
    # 先检查详细形状
    check_all_node_shapes(model_path)
    
    # 再测试推理
    success = test_inference(model_path)
    print()
    if success:
        print("🎉 模型推理测试通过!")
    else:
        print("⛔ 模型推理测试失败!")

