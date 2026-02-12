import os
import tensorflow as tf

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
checkpoint_path = os.path.join(root_dir, 'checkpoints', 'best.ckpt')

def check_checkpoint_shapes():
    print(f"[*] 正在检查权重文件: {checkpoint_path}")
    if not os.path.exists(checkpoint_path + ".index"):
        print("[!] 错误: 未找到权重文件！")
        return

    try:
        # 这种方式可以在不构建模型的情况下直接读取权重文件里的张量形状
        reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        
        # 寻找第一层卷积的 kernel
        # 路径通常包含 EncoderDecoderBlock0...ConvBNReLUBlock1...conv2d/kernel
        first_conv_key = None
        for key in sorted(var_to_shape_map.keys()):
            if "ConvBNReLUBlock1" in key and "conv2d/kernel" in key:
                first_conv_key = key
                break
        
        if first_conv_key:
            shape = var_to_shape_map[first_conv_key]
            # 卷积核形状通常是 [kh, kw, in_ch, out_ch]
            print(f"\n[+] 定位到第一层卷积权重: {first_conv_key}")
            print(f"[+] 权重形状 (Shape): {shape}")
            
            init_neurons = shape[-1]
            print(f"\n[结论] 该权重文件适配的 InitNeurons 数值为: {init_neurons}")
            
            if init_neurons == 32:
                print(">>> 权重适配 32 通道。这就是为什么你设为 32 时没报错。")
            elif init_neurons == 37:
                print(">>> 权重适配 37 通道。")
            else:
                print(f">>> 权重适配的是一个特殊的数值: {init_neurons}")

        else:
            print("[!] 未能在权重文件中定位到关键层，打印前 5 个变量供参考：")
            for key in list(var_to_shape_map.keys())[:5]:
                print(f"  - {key}: {var_to_shape_map[key]}")

    except Exception as e:
        print(f"[!] 检查失败: {e}")

if __name__ == "__main__":
    check_checkpoint_shapes()
