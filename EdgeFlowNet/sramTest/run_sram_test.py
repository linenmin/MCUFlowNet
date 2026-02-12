import os
import sys
import numpy as np
import tensorflow as tf
import shutil

# --- 0. 环境设置 ---
tf.compat.v1.disable_eager_execution()

# 获取当前脚本所在目录 (sramTest)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
root_dir = os.path.dirname(current_dir)

# 关键：将 sramTest 所在的目录加入 sys.path 的最前面
# 这样脚本在 import network 或 import misc 时会优先找 sramTest 内部的版本
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 同时也需要将项目根目录加入 path，以便导入 vela_compiler 等工具
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    # 强制重新刷新导入路径
    import network.MultiScaleResNet
    from network.MultiScaleResNet import MultiScaleResNet
    from misc.utils import AccumPreds  # 导入多尺度累加函数
    from vela.vela_compiler import run_vela
    print(f"[*] 成功加载模块。Network 路径: {network.__file__}")
except ImportError as e:
    print(f"[!] 导入失败: {e}")
    # 打印一下当前的 path 方便调试
    print(f"[DEBUG] sys.path: {sys.path}")
    sys.exit(1)
except NameError:
    # 兼容处理
    from network.MultiScaleResNet import MultiScaleResNet
    from vela.vela_compiler import run_vela
    print("[*] 成功加载模块。")

# --- 1. 配置 ---
height, width = 92, 128
checkpoint_path = os.path.join(root_dir, 'checkpoints', 'best.ckpt')
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

model_config = {
    'InitNeurons': 32,
    'ExpansionFactor': 2.0,
    'NumSubBlocks': 2,
    'NumOut': 4,
    'NumBlocks': 1,
    'Padding': 'same'
}

def export_tflite():
    print(f"\n[1/2] 正在导出 TFLite 模型 ({height}x{width}, 修改后的通道数)...")
    
    new_graph = tf.Graph()
    with new_graph.as_default():
        sess = tf.compat.v1.Session(graph=new_graph)
        try:
            input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, height, width, 6], name='input_image')
            
            model_obj = MultiScaleResNet(
                InputPH=input_ph,
                InitNeurons=model_config['InitNeurons'],
                ExpansionFactor=model_config['ExpansionFactor'],
                NumSubBlocks=model_config['NumSubBlocks'],
                NumOut=model_config['NumOut'],
                NumBlocks=model_config['NumBlocks'],
                Padding=model_config['Padding']
            )
            
            network_outputs = model_obj.Network()
            
            # 使用多尺度输出累加（参考 test_sintel.py 的处理方式）
            if isinstance(network_outputs, list) and len(network_outputs) > 1:
                accumOut, _ = AccumPreds(network_outputs)
                final_output = accumOut[..., 0:2]  # 取前 2 个通道 (u, v 光流)
                print(f"[*] 使用多尺度累加输出，共 {len(network_outputs)} 个尺度")
            else:
                final_output = network_outputs[-1] if isinstance(network_outputs, list) else network_outputs
                print("[*] 使用单一输出")
            
            # 关键：由于我们要微调通道数，权重加载会因为维度不匹配而报错。
            # 为了内存测试，我们只需导出结构，不需要真实权重。
            # 强制使用随机初始化。
            print("[*] 正在执行变量随机初始化 (绕过 checkpoint 以适配通道数修改)...")
            sess.run(tf.compat.v1.global_variables_initializer())
            
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            def representative_dataset_gen():
                for _ in range(5):
                    yield [np.random.uniform(0.0, 1.0, size=[1, height, width, 6]).astype(np.float32)]
            converter.representative_dataset = representative_dataset_gen
            
            tflite_model = converter.convert()
            
            tflite_path = os.path.join(current_dir, "sram_test_modified.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"[+] TFLite 已保存至: {tflite_path}")
            return tflite_path
            
        except Exception as e:
            print(f"[!] 导出失败: {e}")
            return None
        finally:
            sess.close()

def main():
    # 步骤 1: 导出
    tflite_path = export_tflite()
    if not tflite_path:
        return

    # 步骤 2: 编译 (使用 vela/vela_compiler.py 的逻辑)
    print(f"\n[2/2] 正在调用 Vela 进行编译...")
    # 尝试切换为 Size 模式，看看 SRAM 是否有显著改善
    sram_mb, time_ms = run_vela(tflite_path, mode="verbose", output_dir=output_dir, optimise="Size")

    if sram_mb is not None:
        print("\n" + "="*40)
        print(f"测试完成！")
        print(f"模型分辨率: {height}x{width}")
        print(f"修改后 SRAM 占用: {sram_mb:.3f} MB")
        print(f"预估推理时间: {time_ms:.2f} ms")
        print(f"生成的 CSV 文件位于: {output_dir}")
        print("="*40)
    else:
        print("\n[!] Vela 编译未产生结果，请检查控制台报错。")

if __name__ == "__main__":
    main()
