import os
import sys
import numpy as np
import tensorflow as tf
import shutil
import argparse

# --- 0. 环境设置 ---
tf.compat.v1.disable_eager_execution()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 导入新定义的 Bilinear 网络
try:
    from network.MultiScaleResNet_bilinear import MultiScaleResNet
    from misc.utils import AccumPreds  # 导入多尺度累加函数
    from vela.vela_compiler import run_vela
    print(f"[*] 成功加载 Bilinear Network 模块: {MultiScaleResNet}")
except ImportError as e:
    print(f"[!] 导入失败: {e}")
    sys.exit(1)

# --- 1. 配置 ---
height, width = 180, 240

model_config = {
    'InitNeurons': 32,
    'ExpansionFactor': 2.0,
    'NumSubBlocks': 2,
    'NumOut': 4,
    'NumBlocks': 1,
    'Padding': 'same'
}

def build_multiscale_final_output(network_outputs, accum_mode):
    if not isinstance(network_outputs, list) or len(network_outputs) <= 1:
        return network_outputs[-1] if isinstance(network_outputs, list) else network_outputs, network_outputs

    if accum_mode == "uv_only":
        accum_inputs = [output[..., 0:2] for output in network_outputs]
    elif accum_mode == "all_channels":
        accum_inputs = list(network_outputs)
    else:
        raise ValueError(f"未知 accum_mode: {accum_mode}")

    accum_out, _ = AccumPreds(accum_inputs)
    final_output = accum_out if accum_mode == "uv_only" else accum_out[..., 0:2]
    return final_output, accum_inputs


def export_tflite(accum_mode, tflite_path):
    print(f"\n[1/2] 正在导出 Bilinear TFLite 模型 ({height}x{width}, accum_mode={accum_mode})...")
    
    new_graph = tf.Graph()
    with new_graph.as_default():
        sess = tf.compat.v1.Session(graph=new_graph)
        try:
            input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, height, width, 6], name='input_image')
            
            # 实例化新网络
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
                final_output, accum_inputs = build_multiscale_final_output(network_outputs, accum_mode)
                print(
                    f"[*] 使用多尺度累加输出，共 {len(network_outputs)} 个尺度，"
                    f"累加输入通道数: {[tensor.shape.as_list()[-1] for tensor in accum_inputs]}"
                )
            else:
                final_output = network_outputs[-1] if isinstance(network_outputs, list) else network_outputs
                print("[*] 使用单一输出")
            
            print("[*] 正在执行变量随机初始化...")
            sess.run(tf.compat.v1.global_variables_initializer())
            
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # 确保使用 INT8
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
            
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"[+] TFLite 已保存至: {tflite_path}")
            return tflite_path
            
        except Exception as e:
            print(f"[!] 导出失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            sess.close()

def main():
    global height, width

    parser = argparse.ArgumentParser(description="Bilinear SRAM test exporter")
    parser.add_argument(
        "--accum-mode",
        choices=("all_channels", "uv_only"),
        default="uv_only",
        help="多尺度累加时是先保留 4 通道再切 2 通道，还是每尺度先切成 2 通道",
    )
    parser.add_argument("--height", type=int, default=height, help="Input height for export and Vela compile")
    parser.add_argument("--width", type=int, default=width, help="Input width for export and Vela compile")
    args = parser.parse_args()

    height, width = args.height, args.width

    suffix = "uv_only" if args.accum_mode == "uv_only" else "all_channels"
    size_suffix = f"{height}x{width}"
    output_dir = os.path.join(current_dir, f"output_bilinear_{suffix}_{size_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(current_dir, f"sram_test_bilinear_{suffix}_{size_suffix}.tflite")

    tflite_path = export_tflite(args.accum_mode, tflite_path)
    if not tflite_path:
        return

    print(f"\n[2/2] 正在调用 Vela 进行编译 (Resize+Conv 方案)...")
    # 依然使用 Size 策略
    sram_mb, time_ms = run_vela(tflite_path, mode="verbose", output_dir=output_dir, optimise="Size")

    if sram_mb is not None:
        print("\n" + "="*40)
        print(f"Bilinear 测试完成！")
        print(f"模型分辨率: {height}x{width}")
        print(f"SRAM 占用: {sram_mb:.3f} MB")
        print(f"预估推理时间: {time_ms:.2f} ms")
        print(f"累加模式: {args.accum_mode}")
        print(f"结果目录: {output_dir}")
        print("="*40)

if __name__ == "__main__":
    main()
