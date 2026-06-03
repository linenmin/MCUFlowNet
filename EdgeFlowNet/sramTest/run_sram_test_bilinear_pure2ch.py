import os
import sys
import numpy as np
import tensorflow as tf


tf.compat.v1.disable_eager_execution()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from network.MultiScaleResNet_bilinear import MultiScaleResNet
    from misc.utils import AccumPreds
    from vela.vela_compiler import run_vela
    print(f"[*] 成功加载 Bilinear Network 模块: {MultiScaleResNet}")
except ImportError as e:
    print(f"[!] 导入失败: {e}")
    sys.exit(1)


height, width = 180, 240

model_config = {
    "InitNeurons": 32,
    "ExpansionFactor": 2.0,
    "NumSubBlocks": 2,
    "NumOut": 2,
    "NumBlocks": 1,
    "Padding": "same",
}


def build_pure_uv_final_output(network_outputs):
    if not isinstance(network_outputs, list) or len(network_outputs) <= 1:
        return network_outputs[-1] if isinstance(network_outputs, list) else network_outputs

    accum_out, _ = AccumPreds(network_outputs)
    return accum_out


def export_tflite(tflite_path):
    print(f"\n[1/2] 正在导出 Bilinear 纯 2ch TFLite 模型 ({height}x{width})...")

    new_graph = tf.Graph()
    with new_graph.as_default():
        sess = tf.compat.v1.Session(graph=new_graph)
        try:
            input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, height, width, 6], name="input_image")

            model_obj = MultiScaleResNet(
                InputPH=input_ph,
                InitNeurons=model_config["InitNeurons"],
                ExpansionFactor=model_config["ExpansionFactor"],
                NumSubBlocks=model_config["NumSubBlocks"],
                NumOut=model_config["NumOut"],
                NumBlocks=model_config["NumBlocks"],
                Padding=model_config["Padding"],
            )

            network_outputs = model_obj.Network()
            final_output = build_pure_uv_final_output(network_outputs)

            if isinstance(network_outputs, list):
                print(
                    f"[*] 使用纯 2ch 多尺度输出，共 {len(network_outputs)} 个尺度，"
                    f"每尺度通道数: {[tensor.shape.as_list()[-1] for tensor in network_outputs]}"
                )
            else:
                print("[*] 使用单一 2ch 输出")

            print("[*] 正在执行变量随机初始化...")
            sess.run(tf.compat.v1.global_variables_initializer())

            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
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
    output_dir = os.path.join(current_dir, "output_bilinear_pure2ch")
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(current_dir, "sram_test_bilinear_pure2ch.tflite")

    tflite_path = export_tflite(tflite_path)
    if not tflite_path:
        return

    print("\n[2/2] 正在调用 Vela 进行编译 (纯 2ch head 方案)...")
    sram_mb, time_ms = run_vela(tflite_path, mode="verbose", output_dir=output_dir, optimise="Size")

    if sram_mb is not None:
        print("\n" + "=" * 40)
        print("Bilinear 纯 2ch 测试完成！")
        print(f"模型分辨率: {height}x{width}")
        print(f"SRAM 占用: {sram_mb:.3f} MB")
        print(f"预估推理时间: {time_ms:.2f} ms")
        print(f"结果目录: {output_dir}")
        print("=" * 40)


if __name__ == "__main__":
    main()
