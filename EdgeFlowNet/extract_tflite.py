import tensorflow as tf
import os
import sys

# 1. 设置 TF1 兼容模式 (非常重要)
tf.compat.v1.disable_eager_execution()

# 添加代码路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'code'))

try:
    from network.MultiScaleResNet import MultiScaleResNet
    print("Successfully imported MultiScaleResNet.")
except ImportError as e:
    print(f"Error importing MultiScaleResNet: {e}")
    sys.exit(1)

# --- 配置参数 (注意首字母大写，匹配原代码) ---
model_config = {
    'InitNeurons': 32,
    'ExpansionFactor': 2.0,
    'NumSubBlocks': 2,
    'NumOut': 4,         # 光流输出通道
    'NumBlocks': 1,      # 默认值
    'Padding': 'same'
}

# input_height = 384
# input_width = 512
input_height = 192
input_width = 256
input_channels = 6 # RGB+RGB

# 2. 创建输入占位符 (TF1 风格必须步骤)
# 这里的 name='input_image' 很重要，后续转换可能会用到
input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, input_height, input_width, input_channels], name='input_image')

# 3. 实例化模型
print("Building model graph...")
# 注意：这里必须传入 InputPH，且参数名要大写
model_obj = MultiScaleResNet(
    InputPH=input_ph,
    InitNeurons=model_config['InitNeurons'],
    ExpansionFactor=model_config['ExpansionFactor'],
    NumSubBlocks=model_config['NumSubBlocks'],
    NumOut=model_config['NumOut'],
    NumBlocks=model_config['NumBlocks'],
    Padding=model_config['Padding']
)

# 4. 获取输出张量
# Network() 返回的是一个列表 [scale1, scale2, final_flow]
# 我们通常只需要最后一个（最高分辨率的）
network_outputs = model_obj.Network() 
if isinstance(network_outputs, list):
    final_output = network_outputs[-1] # 取最后一个输出
else:
    final_output = network_outputs

print(f"Output tensor shape: {final_output.shape}")

# 5. 启动 Session 并加载权重
sess = tf.compat.v1.Session()

# 这里的 Saver 会自动去匹配 checkpoint 中的变量名
saver = tf.compat.v1.train.Saver()

checkpoint_path = os.path.join(current_dir, 'checkpoints', 'best.ckpt')
print(f"Loading weights from: {checkpoint_path}")

try:
    saver.restore(sess, checkpoint_path)
    print("Weights restored successfully.")
except Exception as e:
    print(f"Error restoring weights: {e}")
    sys.exit(1)

# 6. 转换为 TFLite
print("Starting TFLite conversion from Session...")

# 使用 from_session 而不是 from_keras_model
converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])

# --- 量化配置 (针对 Vela NPU) ---
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 代表性数据集 (生成器)
def representative_dataset_gen():
    import numpy as np
    # 生成 10 个样本用于校准
    for _ in range(10):
        # 注意：这里的范围需要和你训练时的预处理一致
        # 假设是归一化到 0-1。如果是 -1 到 1，请改为 minval=-1.0, maxval=1.0
        yield [np.random.uniform(0.0, 1.0, size=[1, input_height, input_width, input_channels]).astype(np.float32)]

converter.representative_dataset = representative_dataset_gen

try:
    tflite_model = converter.convert()
    
    output_path = os.path.join(current_dir, "edgeflownet_192_256_int8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Success! Model saved to: {output_path}")
    
except Exception as e:
    print(f"Conversion failed: {e}")

finally:
    sess.close()