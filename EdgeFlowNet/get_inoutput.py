import tensorflow as tf

path = r"D:\Dataset\EdgeFlowNet\edgeflownet_192_256_int8.tflite"  # 补上结尾引号
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:", input_details)
print("Output:", output_details)