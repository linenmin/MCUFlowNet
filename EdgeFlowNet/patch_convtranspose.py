#!/usr/bin/env python  # 指定Python解释器路径
import argparse  # 导入命令行参数解析模块
import onnx  # 导入ONNX模型处理库


def get_attr(node, name):  # 获取节点的指定属性
    for attr in node.attribute:  # 遍历节点的所有属性
        if attr.name == name:  # 如果属性名匹配
            return attr  # 返回该属性
    return None  # 未找到则返回None


def get_attr_ints(node, name):  # 获取节点的整数列表属性
    attr = get_attr(node, name)  # 获取属性对象
    if attr is None:  # 如果属性不存在
        return None  # 返回None
    return list(attr.ints)  # 返回整数列表


def set_attr(node, name, value):  # 设置节点的属性值
    attr = get_attr(node, name)  # 获取现有属性
    if attr is not None:  # 如果属性已存在
        node.attribute.remove(attr)  # 删除旧属性
    node.attribute.append(onnx.helper.make_attribute(name, value))  # 添加新属性


def del_attr(node, name):  # 删除节点的指定属性
    attr = get_attr(node, name)  # 获取属性对象
    if attr is not None:  # 如果属性存在
        node.attribute.remove(attr)  # 删除该属性


def get_initializer_shape(model, name):  # 获取初始化器的形状
    for init in model.graph.initializer:  # 遍历模型的所有初始化器
        if init.name == name:  # 如果名称匹配
            return list(init.dims)  # 返回维度列表
    return None  # 未找到则返回None


def infer_kernel_shape(node, model):  # 推断卷积核形状
    ks = get_attr_ints(node, "kernel_shape")  # 尝试从属性获取核形状
    if ks:  # 如果存在核形状属性
        return ks  # 直接返回
    if len(node.input) < 2:  # 如果输入数量不足
        return None  # 返回None
    w_shape = get_initializer_shape(model, node.input[1])  # 从权重初始化器获取形状
    if w_shape and len(w_shape) >= 4:  # 如果形状有效且维度>=4
        return w_shape[-2:]  # 返回最后两个维度（高度和宽度）
    return None  # 否则返回None


def infer_strides(node, rank):  # 推断步长
    strides = get_attr_ints(node, "strides")  # 尝试从属性获取步长
    if strides:  # 如果存在步长属性
        return strides  # 直接返回
    return [1] * rank  # 否则返回全1的列表


def infer_dilations(node, rank):  # 推断膨胀率
    dilations = get_attr_ints(node, "dilations")  # 尝试从属性获取膨胀率
    if dilations:  # 如果存在膨胀率属性
        return dilations  # 直接返回
    return [1] * rank  # 否则返回全1的列表


def patch_convtranspose(model):  # 修补ConvTranspose节点
    changed = 0  # 记录修改的节点数量
    for node in model.graph.node:  # 遍历模型的所有节点
        if node.op_type != "ConvTranspose":  # 如果不是ConvTranspose节点
            continue  # 跳过

        kernel = infer_kernel_shape(node, model)  # 推断核形状
        if not kernel:  # 如果无法推断核形状
            continue  # 跳过该节点

        rank = len(kernel)  # 获取卷积的维度数
        strides = infer_strides(node, rank)  # 推断步长
        dilations = infer_dilations(node, rank)  # 推断膨胀率

        pads_begin = []  # 初始化起始填充列表
        pads_end = []  # 初始化结束填充列表
        for k, d in zip(kernel, dilations):  # 遍历核大小和膨胀率
            k_eff = (k - 1) * d + 1  # 计算有效核大小
            p = k_eff // 2  # 计算填充值（向下取整）
            pads_begin.append(p)  # 添加到起始填充
            pads_end.append(p)  # 添加到结束填充

        pads = pads_begin + pads_end  # 合并起始和结束填充

        output_padding = []  # 初始化输出填充列表
        for s in strides:  # 遍历步长
            op = s - 1 if s > 1 else 0  # 计算输出填充（步长>1时为步长-1，否则为0）
            output_padding.append(op)  # 添加到输出填充列表

        set_attr(node, "auto_pad", "NOTSET")  # 设置auto_pad为NOTSET
        set_attr(node, "pads", pads)  # 设置填充属性
        set_attr(node, "output_padding", output_padding)  # 设置输出填充属性
        del_attr(node, "output_shape")  # 删除output_shape属性
        changed += 1  # 增加修改计数

    return changed  # 返回修改的节点数量


def main():  # 主函数
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("input", help="input onnx path", default="./edgeflownet_384_512.onnx")  # 添加输入文件路径参数
    parser.add_argument("output", help="output onnx path", default="./output_padding/edgeflownet_384_512.onnx")  # 添加输出文件路径参数
    args = parser.parse_args()  # 解析命令行参数

    model = onnx.load(args.input)  # 加载ONNX模型
    changed = patch_convtranspose(model)  # 修补ConvTranspose节点
    onnx.save(model, args.output)  # 保存修改后的模型
    print(f"patched ConvTranspose: {changed}")  # 打印修改的节点数量


if __name__ == "__main__":  # 如果作为主程序运行
    main()  # 调用主函数
