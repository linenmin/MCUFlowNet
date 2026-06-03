#!/usr/bin/env python
"""
EdgeFlowNet 模型导出脚本
将 TensorFlow 模型转换为 ONNX 格式
支持多尺度输出累积
"""

import argparse  # 命令行参数解析
import os  # 文件系统操作
import sys  # 系统相关操作
from typing import Optional  # 类型提示

import numpy as np  # 数值计算
import tensorflow as tf  # TensorFlow

# 使用 TensorFlow 1.x 图模式（兼容旧代码）
tf.compat.v1.disable_eager_execution()

# 获取脚本所在目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 代码目录路径
CODE_DIR = os.path.join(ROOT_DIR, "code")
# 将代码目录添加到 Python 路径
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# 导入模型类
try:
    from network.MultiScaleResNet import MultiScaleResNet  # 多尺度 ResNet 模型
    from misc.utils import AccumPreds  # 多尺度输出累积函数
except ImportError as exc:
    raise SystemExit(f"Failed to import EdgeFlowNet code: {exc}")  

def resolve_checkpoint(path: Optional[str]) -> Optional[str]:
    """解析检查点路径，支持目录、文件或索引文件"""
    if not path:  # 路径为空
        return None
    if os.path.isdir(path):  # 如果是目录，查找最新的检查点
        return tf.train.latest_checkpoint(path)
    if os.path.exists(path):  # 如果文件存在
        return path
    if os.path.exists(path + ".index"):  # 如果索引文件存在
        return path
    return None


def detect_checkpoint_num_out(checkpoint_path):
    """从检查点中检测实际的 NumOut 值（通过检查 ConvTranspose45 的偏置维度）"""
    ckpt = resolve_checkpoint(checkpoint_path)  # 解析检查点路径
    if not ckpt:  # 如果检查点不存在
        return None
    
    # 创建一个临时会话来读取检查点
    tf.compat.v1.reset_default_graph()  # 重置图
    try:
        # 使用 NewCheckpointReader 读取检查点文件
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt)  # 创建检查点读取器
        var_names = reader.get_variable_to_shape_map()  # 获取所有变量名和形状的映射
        
        # 查找 ConvTranspose45 的偏置变量
        target_var = None  # 目标变量名
        for var_name in var_names:  # 遍历所有变量名
            if 'ConvTranspose45' in var_name and 'bias' in var_name.lower():  # 找到目标偏置变量
                target_var = var_name  # 保存变量名
                break
        
        if target_var:  # 如果找到目标变量
            bias_shape = reader.get_variable_to_shape_map()[target_var]  # 获取偏置形状
            if len(bias_shape) == 1:  # 如果是一维向量
                num_out = bias_shape[0]  # 获取偏置大小，即 NumOut
                print(f"[检测] 从检查点检测到 NumOut = {num_out} (来自 {target_var})")  # 打印检测结果
                return num_out  # 返回检测到的 NumOut
    except Exception as e:  # 捕获异常
        print(f"[检测] 检测 NumOut 时出错: {e}")  # 打印错误信息
    
    print("[检测] 无法从检查点检测 NumOut，使用默认值")  # 打印提示信息
    return None  # 返回 None


def build_graph(args, checkpoint_num_out=None):
    """构建 TensorFlow 计算图"""
    tf.compat.v1.reset_default_graph()  # 重置默认图

    # 如果提供了检查点的 NumOut，使用它；否则使用 args.num_out
    actual_num_out = checkpoint_num_out if checkpoint_num_out is not None else args.num_out  # 确定实际使用的 NumOut
    if checkpoint_num_out is not None and checkpoint_num_out != args.num_out:  # 如果检测到的值与参数不同
        print(f"[构建] 使用检查点的 NumOut={checkpoint_num_out} (而非命令行参数 {args.num_out})")  # 打印提示信息

    # 创建输入占位符（NHWC格式，两帧拼接所以通道数是 channels * 2）
    input_ph = tf.compat.v1.placeholder(
        tf.float32,  # 数据类型
        shape=[1, args.height, args.width, args.channels * 2],  # 输入形状：[批次, 高度, 宽度, 通道数*2]
        name="input",  # 节点名称
    )

    # 创建多尺度 ResNet 模型实例（使用检查点的 NumOut）
    model = MultiScaleResNet(
        InputPH=input_ph,  # 输入占位符
        Padding=args.padding,  # 填充方式
        NumOut=actual_num_out,  # 使用检查点的输出通道数
        InitNeurons=args.init_neurons,  # 初始神经元数
        ExpansionFactor=args.expansion_factor,  # 扩展因子
        NumSubBlocks=args.num_sub_blocks,  # 子块数量
        NumBlocks=args.num_blocks,  # 块数量
        Suffix="",  # 后缀
        UncType=None,  # 不确定性类型
    )

    # 构建网络，获取多尺度输出列表
    multi_scale_outputs = model.Network()  # 调用网络构建方法
    # 累积多尺度输出
    accum_output, _ = AccumPreds(multi_scale_outputs)  # 累积多尺度预测结果

    # 取前 output_channels 个通道作为最终输出（从检查点的通道数切片到目标通道数）
    output = accum_output[..., : args.output_channels]  # 切片操作，只保留前 output_channels 个通道
    # 添加输出节点名称
    output = tf.identity(output, name="output")  # 创建 Identity 节点并命名

    return input_ph, output  # 返回输入和输出张量


def freeze_graph(sess, output_tensor):
    """冻结图：将变量转换为常量"""
    output_node_name = output_tensor.name.split(":")[0]  # 获取输出节点名称（去掉:0）
    # 将图中的变量转换为常量
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,  # TensorFlow 会话
        sess.graph_def,  # 图定义
        [output_node_name]  # 输出节点名称列表
    )
    return frozen_graph_def  # 返回冻结后的图定义


def _get_attr_ints(node, name):
    """获取节点的整数列表属性"""
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)  # 返回整数列表
    return None


def _get_dims(vi):
    """获取张量的维度信息"""
    return [
        d.dim_value if d.dim_value else d.dim_param or None  # 固定值或动态参数
        for d in vi.type.tensor_type.shape.dim
    ]


def _set_dims(vi, dims):
    """设置张量的维度信息"""
    shape = vi.type.tensor_type.shape
    del shape.dim[:]  # 清空现有维度
    for val in dims:
        dim = shape.dim.add()  # 添加新维度
        if isinstance(val, int):  # 如果是整数，设为固定值
            dim.dim_value = val
        elif val is not None:  # 否则设为动态参数
            dim.dim_param = str(val)


def _reorder_nhwc_to_nchw(dims):
    """将 NHWC 格式转换为 NCHW 格式"""
    if len(dims) != 4:  # 不是4维张量，直接返回
        return dims
    return [dims[0], dims[3], dims[1], dims[2]]  # [N, H, W, C] -> [N, C, H, W]


def strip_io_transposes(model, input_height, input_width, output_channels):
    """去除输入输出的转置操作，直接使用NCHW格式"""
    graph = model.graph
    input_name = graph.input[0].name  # 输入节点名称
    output_name = graph.output[0].name  # 输出节点名称

    # 构建输出名称到节点的映射
    output_to_node = {}
    for node in graph.node:
        for out_name in node.output:
            output_to_node[out_name] = node

    def _resolve_identity_chain(name):
        """解析Identity链，找到真正的输出节点"""
        identities = []
        node = output_to_node.get(name)
        while node is not None and node.op_type == "Identity":  # 跳过Identity节点
            identities.append(node)
            if not node.input:
                break
            name = node.input[0]
            node = output_to_node.get(name)
        return name, node, identities

    # 查找并移除输入转置（NHWC -> NCHW）
    input_transpose = None
    for node in graph.node:
        if node.op_type != "Transpose":
            continue
        perm = _get_attr_ints(node, "perm")  # 获取转置顺序
        if perm == [0, 3, 1, 2] and node.input and node.input[0] == input_name:  # NHWC->NCHW
            input_transpose = node
            break

    if input_transpose:
        trans_out = input_transpose.output[0]
        # 将所有使用转置输出的节点改为直接使用输入
        for node in graph.node:
            for idx, name in enumerate(node.input):
                if name == trans_out:
                    node.input[idx] = input_name
        graph.node.remove(input_transpose)  # 移除转置节点

        # 更新输入形状为NCHW
        in_dims = _get_dims(graph.input[0])
        _set_dims(graph.input[0], _reorder_nhwc_to_nchw(in_dims))

    # 查找并移除输出转置（NCHW -> NHWC）
    orig_out_dims = _get_dims(graph.output[0])
    resolved_name, resolved_node, identity_nodes = _resolve_identity_chain(output_name)

    output_transpose = None
    if resolved_node is not None and resolved_node.op_type == "Transpose":
        perm = _get_attr_ints(resolved_node, "perm")
        if perm == [0, 2, 3, 1]:  # NCHW->NHWC
            output_transpose = resolved_node

    if output_transpose:
        new_output = output_transpose.input[0]  # 转置前的输出
        trans_out = output_transpose.output[0]

        # 将所有使用转置输出的节点改为直接使用新输出
        for node in graph.node:
            for idx, name in enumerate(node.input):
                if name == trans_out:
                    node.input[idx] = new_output

        # 更新输出名称和形状
        graph.output[0].name = new_output

        # 查找新输出的形状
        out_dims = None
        for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
            if vi.name == new_output:
                out_dims = _get_dims(vi)
                break
        if out_dims:
            _set_dims(graph.output[0], out_dims)
        else:
            _set_dims(graph.output[0], _reorder_nhwc_to_nchw(orig_out_dims))

        graph.node.remove(output_transpose)  # 移除转置节点

        # 移除Identity节点
        for node in identity_nodes:
            if node in graph.node:
                graph.node.remove(node)

    # 确保输出形状正确（NCHW格式）
    if len(graph.output) > 0:
        out_dims = _get_dims(graph.output[0])
        if (
            not out_dims  # 没有形状信息
            or any(dim is None for dim in out_dims)  # 有动态维度
            or (  # 形状看起来像NHWC格式
                len(out_dims) == 4
                and out_dims[1] == input_height
                and out_dims[2] == input_width
                and out_dims[3] == output_channels
            )
        ):
            # 强制设置为NCHW格式
            _set_dims(
                graph.output[0],
                [1, output_channels, input_height, input_width],
            )

    return model


def _get_initializer_shape(model, name):
    """获取初始化器（权重）的形状"""
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def _get_shape_from_value_info(model, name):
    """从value_info、input或output中获取形状信息"""
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == name:
            return _get_dims(vi)
    return None  

def convert_auto_pad_to_explicit_pads(model):
    """将自动padding转换为显式padding（Axelera编译器要求）"""
    import math
    import onnx
    from onnx import shape_inference

    # 运行形状推断
    model = shape_inference.infer_shapes(model)

    for node in model.graph.node:
        if node.op_type != "Conv":  # 只处理卷积层
            continue

        # 获取auto_pad属性
        auto_pad = None
        for attr in node.attribute:
            if attr.name == "auto_pad" and attr.s:
                auto_pad = attr.s.decode("utf-8")

        # 如果已经是显式padding，跳过
        if auto_pad is None or auto_pad == "NOTSET":
            continue

        # 获取输入形状
        in_shape = _get_shape_from_value_info(model, node.input[0])
        if not in_shape or len(in_shape) < 4 or any(dim is None for dim in in_shape[0:4]):
            continue

        # 获取卷积核形状
        kernel_shape = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
                break
        if kernel_shape is None:  # 从权重形状获取
            weight_shape = _get_initializer_shape(model, node.input[1])
            if weight_shape and len(weight_shape) >= 4:
                kernel_shape = weight_shape[-2:]  # 取最后两个维度（H, W）

        if not kernel_shape or len(kernel_shape) != 2:
            continue

        # 获取步长和膨胀
        strides = None
        dilations = None
        for attr in node.attribute:
            if attr.name == "strides":
                strides = list(attr.ints)
            elif attr.name == "dilations":
                dilations = list(attr.ints)
        if not strides:  # 默认步长为1
            strides = [1, 1]
        if not dilations:  # 默认膨胀为1
            dilations = [1, 1]

        # 提取尺寸参数
        in_h, in_w = in_shape[2], in_shape[3]  # 输入高度和宽度
        k_h, k_w = kernel_shape  # 卷积核高度和宽度
        s_h, s_w = strides  # 步长
        d_h, d_w = dilations  # 膨胀

        # 计算显式padding值
        if auto_pad == "VALID":  # 无padding
            pads = [0, 0, 0, 0]
        else:  # SAME padding
            # 计算输出尺寸
            out_h = int(math.ceil(float(in_h) / float(s_h)))
            out_w = int(math.ceil(float(in_w) / float(s_w)))
            # 计算总padding
            pad_h_total = max((out_h - 1) * s_h + (k_h - 1) * d_h + 1 - in_h, 0)
            pad_w_total = max((out_w - 1) * s_w + (k_w - 1) * d_w + 1 - in_w, 0)

            # 根据auto_pad类型分配padding
            if auto_pad == "SAME_LOWER":  # 向下取整
                pad_top = int(math.ceil(pad_h_total / 2.0))
                pad_left = int(math.ceil(pad_w_total / 2.0))
            else:  # SAME_UPPER，向上取整
                pad_top = int(math.floor(pad_h_total / 2.0))
                pad_left = int(math.floor(pad_w_total / 2.0))

            pad_bottom = pad_h_total - pad_top
            pad_right = pad_w_total - pad_left
            pads = [pad_top, pad_left, pad_bottom, pad_right]  # [top, left, bottom, right]

        # 更新节点属性：移除auto_pad，添加显式pads
        new_attrs = []
        for attr in node.attribute:
            if attr.name in ("auto_pad", "pads"):  # 跳过旧的padding属性
                continue
            new_attrs.append(attr)
        new_attrs.append(onnx.helper.make_attribute("pads", pads))  # 添加显式pads
        new_attrs.append(onnx.helper.make_attribute("auto_pad", "NOTSET"))  # 设置为NOTSET
        node.attribute[:] = new_attrs

    return model


def convert_to_onnx(args, input_tensor, output_tensor, frozen_graph_def):
    """将冻结的TensorFlow图转换为ONNX格式"""
    import tf2onnx

    # 使用tf2onnx转换
    onnx_model, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,
        input_names=[input_tensor.name],  # 输入节点名称
        output_names=[output_tensor.name],  # 输出节点名称
        opset=args.opset,  # ONNX opset版本
    )

    # 去除输入输出的转置（转换为NCHW格式）
    if args.export_nchw:
        onnx_model = strip_io_transposes(
            onnx_model, args.height, args.width, args.output_channels
        )
    # 将自动padding转换为显式padding
    if args.force_explicit_pads:
        onnx_model = convert_auto_pad_to_explicit_pads(onnx_model)

    import onnx

    # 保存ONNX模型
    onnx.save(onnx_model, args.output)


def verify_onnx(args):
    """验证ONNX模型：检查结构并测试推理"""
    try:
        import onnx
        import onnxruntime as ort
    except Exception as exc:
        print(f"ONNX verification skipped: {exc}")  # 如果库不存在，跳过验证
        return

    # 加载并检查模型结构
    model = onnx.load(args.output)
    onnx.checker.check_model(model)  # 验证模型结构

    # 创建推理会话
    sess = ort.InferenceSession(args.output)
    input_name = sess.get_inputs()[0].name  # 获取输入名称
    output_name = sess.get_outputs()[0].name  # 获取输出名称

    # 创建测试输入（根据格式）
    if args.export_nchw:  # NCHW格式
        test_input = np.random.randn(
            1, args.channels * 2, args.height, args.width
        ).astype(np.float32)
    else:  # NHWC格式
        test_input = np.random.randn(
            1, args.height, args.width, args.channels * 2
        ).astype(np.float32)

    # 运行推理测试
    result = sess.run([output_name], {input_name: test_input})[0]
    print(f"ONNX output shape: {result.shape}")


def main():
    """主函数：解析参数并执行导出流程"""
    parser = argparse.ArgumentParser(description="Export EdgeFlowNet (transpose conv) to ONNX")
    # 输入尺寸参数
    parser.add_argument("--height", type=int, default=384)  # 输入高度
    parser.add_argument("--width", type=int, default=512)  # 输入宽度
    parser.add_argument("--channels", type=int, default=3, help="Channels per frame")  # 每帧通道数
    # 模型结构参数
    parser.add_argument("--num-out", type=int, default=2)  # 模型输出通道数
    parser.add_argument("--output-channels", type=int, default=2)  # 最终输出通道数
    parser.add_argument("--init-neurons", type=int, default=32)  # 初始神经元数
    parser.add_argument("--expansion-factor", type=float, default=2.0)  # 扩展因子
    parser.add_argument("--num-sub-blocks", type=int, default=2)  # 子块数量
    parser.add_argument("--num-blocks", type=int, default=1)  # 块数量
    parser.add_argument("--padding", default="same")  # 填充方式
    # 文件路径参数
    parser.add_argument("--checkpoint", default=os.path.join(ROOT_DIR, "checkpoints", "best.ckpt"))  # 检查点路径
    parser.add_argument("--output", default="edgeflownet_384_512.onnx")  # 输出文件名
    # ONNX转换参数
    parser.add_argument("--opset", type=int, default=17)  # ONNX opset版本
    parser.add_argument("--export-nchw", action="store_true", default=True)  # 导出为NCHW格式
    parser.add_argument("--no-export-nchw", dest="export_nchw", action="store_false")  # 禁用NCHW导出
    parser.add_argument("--force-explicit-pads", action="store_true", default=True)  # 强制显式padding
    parser.add_argument(
        "--no-force-explicit-pads", dest="force_explicit_pads", action="store_false"  # 禁用显式padding
    )
    parser.add_argument("--verify", action="store_true", default=False)  # 是否验证模型
    args = parser.parse_args()  # 解析命令行参数

    # 从检查点检测实际的 NumOut 值
    checkpoint_num_out = detect_checkpoint_num_out(args.checkpoint)  # 检测检查点中的 NumOut
    
    # 参数验证（使用检测到的或指定的 num_out）
    actual_num_out = checkpoint_num_out if checkpoint_num_out is not None else args.num_out  # 确定实际使用的 NumOut
    if args.output_channels > actual_num_out:  # 如果输出通道数大于 NumOut
        raise SystemExit(f"output-channels ({args.output_channels}) must be <= num-out ({actual_num_out})")  # 抛出错误

    # 构建计算图（传入检测到的 NumOut）
    input_tensor, output_tensor = build_graph(args, checkpoint_num_out)  # 构建图，传入检测到的 NumOut

    # 创建会话并加载权重
    with tf.compat.v1.Session() as sess:  # 创建 TensorFlow 会话
        sess.run(tf.compat.v1.global_variables_initializer())  # 初始化变量

        # 尝试加载检查点
        ckpt = resolve_checkpoint(args.checkpoint)  # 解析检查点路径
        if ckpt:  # 如果找到检查点
            saver = tf.compat.v1.train.Saver()  # 创建保存器
            saver.restore(sess, ckpt)  # 恢复权重
            print(f"Loaded checkpoint: {ckpt}")  # 打印加载信息
        else:  # 如果没找到，使用随机权重
            print("Checkpoint not found; using random weights.")  # 打印提示信息

        # 冻结图（将变量转为常量）
        frozen_graph_def = freeze_graph(sess, output_tensor)  # 冻结图

    # 转换为ONNX并保存
    convert_to_onnx(args, input_tensor, output_tensor, frozen_graph_def)
    print(f"Saved ONNX model to: {args.output}")

    # 如果启用验证，测试模型
    if args.verify:
        verify_onnx(args)  

if __name__ == "__main__":
    main()
