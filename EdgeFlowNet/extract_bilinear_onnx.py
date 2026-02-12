#!/usr/bin/env python
"""
MultiScaleResNet_bilinear 模型导出脚本
使用随机权重，用于测试 Axelera 编译器兼容性
"""

import os
import sys

# 添加必要的路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sramTest'))

import tensorflow as tf
import numpy as np

# 设置 TensorFlow 1.x 兼容模式
tf.compat.v1.disable_eager_execution()

# 配置参数
INPUT_HEIGHT = 576
INPUT_WIDTH = 1024
INPUT_CHANNELS = 6  # 两帧 RGB
OUTPUT_CHANNELS = 2
OUTPUT_ONNX_PATH = "multiscale_bilinear_576_1024.onnx"
EXPORT_NCHW = True
FORCE_EXPLICIT_PADS = True


def AccumPreds(prVals):
    """累积多尺度预测输出 (复用自 misc/utils.py)"""
    prValAccum = None
    prValsAccum = []
    for prVali in prVals:
        if prValAccum == None:
            prValAccum = prVali
            prValsAccum.append(prValAccum)
            continue
                
        prValAccum = tf.compat.v1.image.resize_bilinear(
            prValAccum,
            [prVali.shape[1], prVali.shape[2]],
            align_corners=False,
            half_pixel_centers=True,
        )
        prValAccum += prVali
        prValsAccum.append(prValAccum)
    
    return prValAccum, prValsAccum


def create_model_with_random_weights():
    """创建带随机权重的模型"""
    print("=" * 50)
    print("MultiScaleResNet_bilinear TensorFlow → ONNX 转换")
    print("=" * 50)
    
    # 导入模型
    from sramTest.network.MultiScaleResNet_bilinear import MultiScaleResNet
    
    print(f"\n[1/4] 创建模型 (输入: {INPUT_HEIGHT}x{INPUT_WIDTH}x{INPUT_CHANNELS})...")
    
    # 创建 TensorFlow 会话和图
    tf.compat.v1.reset_default_graph()
    
    # 输入占位符 - NHWC 格式
    input_ph = tf.compat.v1.placeholder(
        tf.float32, 
        shape=[1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
        name='input'
    )
    
    # 创建模型实例
    model = MultiScaleResNet(
        InputPH=input_ph,
        Padding='same',
        NumOut=2,  # 光流输出 (u, v)
        InitNeurons=16,  # 初始通道数
        ExpansionFactor=2.0,
        NumSubBlocks=2,
        NumBlocks=1,
        Suffix='',
        UncType=None
    )
    
    # 构建网络 - 返回多尺度输出列表
    multi_scale_outputs = model.Network()
    print(f"      多尺度输出数量: {len(multi_scale_outputs)}")
    for i, out in enumerate(multi_scale_outputs):
        print(f"        尺度 {i}: {out.shape}")
    
    # 使用 AccumPreds 累积多尺度输出
    accum_output, accum_outputs = AccumPreds(multi_scale_outputs)
    
    # 取前2个通道 (u, v 光流)
    main_output = accum_output[..., 0:OUTPUT_CHANNELS]
    main_output = tf.identity(main_output, name='output')
    
    print(f"      ✓ 模型创建成功")
    print(f"      累积输出形状: {main_output.shape}")
    
    return input_ph, main_output


def convert_to_onnx(input_tensor, output_tensor):
    """转换为 ONNX 格式"""
    import tf2onnx
    
    print(f"\n[2/4] 初始化随机权重...")
    
    with tf.compat.v1.Session() as sess:
        # 使用随机权重初始化
        sess.run(tf.compat.v1.global_variables_initializer())
        print("      ✓ 随机权重初始化完成")
        
        # 测试推理
        print("\n[3/4] 测试 TensorFlow 推理...")
        test_input = np.random.randn(1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS).astype(np.float32)
        output_val = sess.run(output_tensor, feed_dict={input_tensor: test_input})
        print(f"      ✓ 推理成功，输出形状: {output_val.shape}")
        print(f"      输出范围: [{output_val.min():.4f}, {output_val.max():.4f}]")
        
        # 冻结图 - 在当前 Session 中进行
        print(f"\n[4/4] 冻结图并转换为 ONNX...")
        
        # 获取输出节点名称 (去掉 :0)
        output_node_name = output_tensor.name.split(':')[0]
        
        # 冻结图 - 把变量转换为常量
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            [output_node_name]
        )
        print(f"      ✓ 图冻结完成")
        
    # 在 Session 外用冻结的图进行转换
    tf.compat.v1.reset_default_graph()
    
    onnx_model, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,
        input_names=[input_tensor.name],
        output_names=[output_tensor.name],
        opset=17
    )
    if EXPORT_NCHW:
        onnx_model = strip_io_transposes(onnx_model)
    if FORCE_EXPLICIT_PADS:
        onnx_model = convert_auto_pad_to_explicit_pads(onnx_model)
    
    # 保存
    import onnx
    onnx.save(onnx_model, OUTPUT_ONNX_PATH)
    print(f"      ✓ 保存到: {OUTPUT_ONNX_PATH}")
    
    return True


def _get_attr_ints(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None


def _get_dims(vi):
    return [d.dim_value if d.dim_value else d.dim_param or None for d in vi.type.tensor_type.shape.dim]


def _set_dims(vi, dims):
    shape = vi.type.tensor_type.shape
    del shape.dim[:]
    for val in dims:
        dim = shape.dim.add()
        if isinstance(val, int):
            dim.dim_value = val
        elif val is not None:
            dim.dim_param = str(val)


def _reorder_nhwc_to_nchw(dims):
    if len(dims) != 4:
        return dims
    return [dims[0], dims[3], dims[1], dims[2]]


def strip_io_transposes(model):
    graph = model.graph
    input_name = graph.input[0].name
    output_name = graph.output[0].name

    output_to_node = {}
    for node in graph.node:
        for out_name in node.output:
            output_to_node[out_name] = node

    def _resolve_identity_chain(name):
        identities = []
        node = output_to_node.get(name)
        while node is not None and node.op_type == "Identity":
            identities.append(node)
            if not node.input:
                break
            name = node.input[0]
            node = output_to_node.get(name)
        return name, node, identities

    input_transpose = None
    for node in graph.node:
        if node.op_type != "Transpose":
            continue
        perm = _get_attr_ints(node, "perm")
        if perm == [0, 3, 1, 2] and node.input and node.input[0] == input_name:
            input_transpose = node
            break

    if input_transpose:
        trans_out = input_transpose.output[0]
        for node in graph.node:
            for idx, name in enumerate(node.input):
                if name == trans_out:
                    node.input[idx] = input_name
        graph.node.remove(input_transpose)

        in_dims = _get_dims(graph.input[0])
        _set_dims(graph.input[0], _reorder_nhwc_to_nchw(in_dims))

    orig_out_dims = _get_dims(graph.output[0])
    resolved_name, resolved_node, identity_nodes = _resolve_identity_chain(output_name)

    output_transpose = None
    if resolved_node is not None and resolved_node.op_type == "Transpose":
        perm = _get_attr_ints(resolved_node, "perm")
        if perm == [0, 2, 3, 1]:
            output_transpose = resolved_node

    if output_transpose:
        new_output = output_transpose.input[0]
        trans_out = output_transpose.output[0]

        for node in graph.node:
            for idx, name in enumerate(node.input):
                if name == trans_out:
                    node.input[idx] = new_output

        graph.output[0].name = new_output

        out_dims = None
        for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
            if vi.name == new_output:
                out_dims = _get_dims(vi)
                break
        if out_dims:
            _set_dims(graph.output[0], out_dims)
        else:
            _set_dims(graph.output[0], _reorder_nhwc_to_nchw(orig_out_dims))

        graph.node.remove(output_transpose)

        for node in identity_nodes:
            if node in graph.node:
                graph.node.remove(node)

    if EXPORT_NCHW and len(graph.output) > 0:
        out_dims = _get_dims(graph.output[0])
        if (
            not out_dims
            or any(dim is None for dim in out_dims)
            or (len(out_dims) == 4 and out_dims[1] == INPUT_HEIGHT and out_dims[2] == INPUT_WIDTH and out_dims[3] == OUTPUT_CHANNELS)
        ):
            _set_dims(
                graph.output[0],
                [1, OUTPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH],
            )

    return model


def _get_initializer_shape(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def _get_shape_from_value_info(model, name):
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == name:
            return _get_dims(vi)
    return None


def convert_auto_pad_to_explicit_pads(model):
    import math
    import onnx
    from onnx import shape_inference

    model = shape_inference.infer_shapes(model)

    for node in model.graph.node:
        if node.op_type != "Conv":
            continue

        auto_pad = None
        pads_attr = None
        for attr in node.attribute:
            if attr.name == "auto_pad" and attr.s:
                auto_pad = attr.s.decode("utf-8")
            elif attr.name == "pads":
                pads_attr = list(attr.ints)

        if auto_pad is None or auto_pad == "NOTSET":
            continue

        in_shape = _get_shape_from_value_info(model, node.input[0])
        if not in_shape or len(in_shape) < 4 or any(dim is None for dim in in_shape[0:4]):
            continue

        kernel_shape = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
                break
        if kernel_shape is None:
            weight_shape = _get_initializer_shape(model, node.input[1])
            if weight_shape and len(weight_shape) >= 4:
                kernel_shape = weight_shape[-2:]

        if not kernel_shape or len(kernel_shape) != 2:
            continue

        strides = None
        dilations = None
        for attr in node.attribute:
            if attr.name == "strides":
                strides = list(attr.ints)
            elif attr.name == "dilations":
                dilations = list(attr.ints)
        if not strides:
            strides = [1, 1]
        if not dilations:
            dilations = [1, 1]

        in_h, in_w = in_shape[2], in_shape[3]
        k_h, k_w = kernel_shape
        s_h, s_w = strides
        d_h, d_w = dilations

        if auto_pad == "VALID":
            pads = [0, 0, 0, 0]
        else:
            out_h = int(math.ceil(float(in_h) / float(s_h)))
            out_w = int(math.ceil(float(in_w) / float(s_w)))
            pad_h_total = max((out_h - 1) * s_h + (k_h - 1) * d_h + 1 - in_h, 0)
            pad_w_total = max((out_w - 1) * s_w + (k_w - 1) * d_w + 1 - in_w, 0)

            if auto_pad == "SAME_LOWER":
                pad_top = int(math.ceil(pad_h_total / 2.0))
                pad_left = int(math.ceil(pad_w_total / 2.0))
            else:
                pad_top = int(math.floor(pad_h_total / 2.0))
                pad_left = int(math.floor(pad_w_total / 2.0))

            pad_bottom = pad_h_total - pad_top
            pad_right = pad_w_total - pad_left

            pads = [pad_top, pad_left, pad_bottom, pad_right]

        new_attrs = []
        for attr in node.attribute:
            if attr.name in ("auto_pad", "pads"):
                continue
            new_attrs.append(attr)
        new_attrs.append(onnx.helper.make_attribute("pads", pads))
        new_attrs.append(onnx.helper.make_attribute("auto_pad", "NOTSET"))
        node.attribute[:] = new_attrs

    return model


def verify_onnx():
    """验证 ONNX 模型"""
    import onnx
    import onnxruntime as ort
    
    print("\n[验证] ONNX 模型检查...")
    
    # 加载并检查
    model = onnx.load(OUTPUT_ONNX_PATH)
    onnx.checker.check_model(model)
    print("      ✓ ONNX 模型结构有效")
    
    # 推理测试
    print("      测试 ONNX Runtime 推理...")
    sess = ort.InferenceSession(OUTPUT_ONNX_PATH)
    
    # 获取输入输出名称
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    if EXPORT_NCHW:
        test_input = np.random.randn(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH).astype(np.float32)
    else:
        test_input = np.random.randn(1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS).astype(np.float32)
    result = sess.run([output_name], {input_name: test_input})
    
    print(f"      ✓ 推理成功")
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {result[0].shape}")
    print(f"  输出范围: [{result[0].min():.4f}, {result[0].max():.4f}]")
    print("\n🎉 模型推理测试通过!")
    
    return True


def main():
    """主函数"""
    # 创建模型
    input_tensor, output_tensor = create_model_with_random_weights()
    
    # 转换为 ONNX
    convert_to_onnx(input_tensor, output_tensor)
    
    # 验证
    verify_onnx()
    
    print("\n" + "=" * 50)
    print("完成！请上传到 OrangePi 测试编译：")
    print(f"  scp {OUTPUT_ONNX_PATH} orangepi@orangepi5plus:~/.cache/axelera/weights/")
    print("=" * 50)


if __name__ == "__main__":
    main()
