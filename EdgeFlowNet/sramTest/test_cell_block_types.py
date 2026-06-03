"""
快速验证脚本：测试 MultiScaleResNet_cell.py 中的 BlockType 参数
"""
import os
import sys

# 环境设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TF 日志

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

from network.MultiScaleResNet_cell import MultiScaleResNet
from vela.vela_compiler import run_vela

# 测试配置
height, width = 384, 512  # 使用较小分辨率加速测试
output_dir = os.path.join(current_dir, "output_cell_test")
os.makedirs(output_dir, exist_ok=True)

def test_block_type(block_type):
    """测试指定 BlockType 的模型能否成功编译"""
    print(f"\n{'='*50}")
    print(f"测试 BlockType: {block_type}")
    print(f"{'='*50}")
    
    # 创建新图
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        try:
            input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, height, width, 6], name='input')
            
            # 使用偶数 InitNeurons (ShuffleNet 需要)
            model = MultiScaleResNet(
                InputPH=input_ph,
                InitNeurons=32,
                ExpansionFactor=2.0,
                NumSubBlocks=2,
                NumOut=4,
                NumBlocks=1,
                BlockType=block_type
            )
            
            outputs = model.Network()
            final_output = outputs[-1] if isinstance(outputs, list) else outputs
            
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # 转换 TFLite
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            def rep_data():
                for _ in range(3):
                    yield [np.random.rand(1, height, width, 6).astype(np.float32)]
            converter.representative_dataset = rep_data
            
            tflite_model = converter.convert()
            tflite_path = os.path.join(output_dir, f"test_{block_type}.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Vela 编译 (使用 Size 模式确保不会因 SRAM 不足失败)
            sram_mb, time_ms = run_vela(tflite_path, mode="verbose", output_dir=output_dir, optimise="Size", silent=True)
            
            if sram_mb is not None:
                fps = 1000.0 / time_ms if time_ms > 0 else 0
                print(f"✅ {block_type}: SRAM={sram_mb:.2f} MB, FPS={fps:.1f}")
                return True, sram_mb, fps
            else:
                print(f"❌ {block_type}: Vela 编译失败")
                return False, None, None
                
        except Exception as e:
            print(f"❌ {block_type}: 构建失败 - {e}")
            return False, None, None
        finally:
            sess.close()

def main():
    results = []
    # for bt in ['resblock', 'mbconv', 'shufflenet']:
    for bt in ['shufflenet']:
        ok, sram, fps = test_block_type(bt)
        results.append({'type': bt, 'ok': ok, 'sram': sram, 'fps': fps})
    
    print("\n" + "="*50)
    print("汇总结果:")
    print("="*50)
    for r in results:
        status = "✅ 通过" if r['ok'] else "❌ 失败"
        if r['ok']:
            print(f"  {r['type']:12s}: {status} | SRAM={r['sram']:.2f} MB | FPS={r['fps']:.1f}")
        else:
            print(f"  {r['type']:12s}: {status}")

if __name__ == "__main__":
    main()
