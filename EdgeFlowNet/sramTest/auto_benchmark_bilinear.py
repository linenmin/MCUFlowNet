import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# --- 0. 环境设置 ---
tf.compat.v1.disable_eager_execution()

# 获取当前脚本所在目录 (sramTest)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
root_dir = os.path.dirname(current_dir)

# 将 sramTest 所在的目录加入 sys.path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 尝试导入模型定义
try:
    # 导入原版 (Transposed Conv)
    import network.MultiScaleResNet as net_original
    # 导入 Bilinear 版 (Resize + Conv)
    import network.MultiScaleResNet_bilinear as net_bilinear
    from vela.vela_compiler import run_vela
    from misc.utils import AccumPreds  # 多尺度累加函数
    print(f"[*] 成功加载所有 Network 模块。")
except ImportError as e:
    print(f"[!] 导入失败: {e}")
    sys.exit(1)

# --- 1. 配置部分 ---

# 分辨率列表 (对应 auto_benchmark_extract_tflite.py)
resolutions_to_test = [
    (384, 512),  # 原始
    (336, 448),  # 较高
    (300, 400),  # 中高
    (264, 352),  # 中等
    (228, 304),  # 中低
    (192, 256),  # 推荐
    (156, 208),  # 快速
    (120, 160),  # 极速
    (96, 128),   # 极限
    (72, 96),    # 超极限
]

# 模型参数
model_config = {
    'InitNeurons': 32,
    'ExpansionFactor': 2.0,
    'NumSubBlocks': 2,
    'NumOut': 4,
    'NumBlocks': 1,
    'Padding': 'same'
}

output_dir = os.path.join(current_dir, "output_benchmark_bilinear")
os.makedirs(output_dir, exist_ok=True)
optimise_mode = "Size" # 绘图脚本通常关注性能模式下的上限

# --- 2. 函数定义 ---

def generate_tflite(height, width, model_class, suffix):
    """根据分辨率和模型类生成 TFLite 模型"""
    print(f"\n[1/2] 正在生成 TFLite ({suffix}) 分辨率: {height}x{width} ...")
    
    new_graph = tf.Graph() # 创建新图，避免变量冲突
    with new_graph.as_default():
        sess = tf.compat.v1.Session(graph=new_graph)
        try:
            input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, height, width, 6], name='input_image')
            
            # 使用传入的模型类实例化
            model_obj = model_class.MultiScaleResNet(
                InputPH=input_ph,
                InitNeurons=model_config['InitNeurons'],
                ExpansionFactor=model_config['ExpansionFactor'],
                NumSubBlocks=model_config['NumSubBlocks'],
                NumOut=model_config['NumOut'],
                NumBlocks=model_config['NumBlocks'],
                Padding=model_config['Padding']
            )
            
            network_outputs = model_obj.Network() # 构建网络
            
            # 多尺度累加输出 (AccumPreds)
            if isinstance(network_outputs, list) and len(network_outputs) > 1:
                accumOut, _ = AccumPreds(network_outputs)
                final_output = accumOut[..., 0:2]  # 取前 2 通道 (u, v)
            else:
                final_output = network_outputs[-1] if isinstance(network_outputs, list) else network_outputs
            
            # 随机初始化 (仅为了内存测试，不加载权重)
            sess.run(tf.compat.v1.global_variables_initializer())
            
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # 提供代表性数据集以完成量化
            def representative_dataset_gen():
                for _ in range(3):
                    yield [np.random.uniform(0.0, 1.0, size=[1, height, width, 6]).astype(np.float32)]
            converter.representative_dataset = representative_dataset_gen
            
            tflite_model = converter.convert()
            
            # 保存 TFLite 文件
            tflite_path = os.path.join(output_dir, f"{suffix}_{height}_{width}.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            
            return tflite_path
            
        except Exception as e:
            print(f"[!] 导出 TFLite 失败 ({suffix} {height}x{width}): {e}")
            return None
        finally:
            sess.close()

# --- 3. 主程序逻辑 ---

def main():
    # 数据存储：分辨率, 类型(Original/Bilinear), SRAM, FPS
    full_results = []

    test_configs = [
        {"module": net_original, "suffix": "original", "label": "Original (Transpose)"},
        {"module": net_bilinear, "suffix": "bilinear", "label": "Bilinear (Resize+Conv)"}
    ]

    print("=== 开始对比 Benchmark (Original vs Bilinear) ===\n")

    for config in test_configs:
        print(f"\n>>> 正在测试模式: {config['label']} ...")
        for h, w in resolutions_to_test:
            # 1. 生成 TFLite
            tflite_path = generate_tflite(h, w, config['module'], config['suffix'])
            if not tflite_path: continue
                
            # 2. 调用 Vela 编译
            sram_mb, time_ms = run_vela(tflite_path, mode="basic", output_dir=output_dir, optimise=optimise_mode, silent=True)
            
            res_str = f"{h}x{w}"
            if sram_mb is not None:
                fps = 1000.0 / time_ms if time_ms > 0 else 0
                print(f"[*] 结果 {res_str}: SRAM={sram_mb:.2f} MB, FPS={fps:.2f}")
                full_results.append({
                    "Resolution": res_str,
                    "Type": config['label'],
                    "SRAM (MB)": sram_mb,
                    "FPS": fps
                })

    if not full_results:
        print("[!] 没有成功的运行结果。")
        return

    # 保存 CSV
    df = pd.DataFrame(full_results)
    csv_path = os.path.join(current_dir, "benchmark_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] 对比数据已保存至: {csv_path}")

    # --- 绘图逻辑 ---
    resolutions = [f"{h}x{w}" for h, w in resolutions_to_test]
    # 过滤掉失败的项（确保顺序一致）
    df_orig = df[df['Type'] == "Original (Transpose)"]
    df_bili = df[df['Type'] == "Bilinear (Resize+Conv)"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    x = np.arange(len(resolutions))
    width = 0.35 # 柱状图宽度

    # 1. FPS 对比图 (柱状图)
    # 关键修复：使用 .values 避免 Pandas 索引对齐导致数据变空
    ax1.bar(x - width/2, df_orig['FPS'].values, width, label='Original FPS', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, df_bili['FPS'].values, width, label='Bilinear FPS', color='lightgreen', alpha=0.8)
    ax1.set_ylabel('Inference FPS')
    ax1.set_title('EdgeFlowNet Performance Comparison (FPS)')
    ax1.legend()
    # 标数值
    for i, val in enumerate(df_orig['FPS'].values): ax1.text(i - width/2, val, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    for i, val in enumerate(df_bili['FPS'].values): ax1.text(i + width/2, val, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # 2. SRAM 对比图 (折线图)
    # 同样使用 .values
    ax2.plot(x, df_orig['SRAM (MB)'].values, marker='o', label='Original SRAM', color='royalblue', linewidth=2)
    ax2.plot(x, df_bili['SRAM (MB)'].values, marker='s', label='Bilinear SRAM', color='darkgreen', linewidth=2)
    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='SRAM Limit (2MB)')
    ax2.set_ylabel('Peak SRAM Usage (MB)')
    ax2.set_xlabel('Input Resolution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(resolutions, rotation=45)
    ax2.set_title('EdgeFlowNet Memory Comparison (SRAM)')
    ax2.legend()
    # 标数值 (SRAM)
    for i, val in enumerate(df_orig['SRAM (MB)']): ax2.text(i, val + 0.1, f'{val:.2f}', color='blue', ha='right', fontsize=9)
    for i, val in enumerate(df_bili['SRAM (MB)']): ax2.text(i, val - 0.2, f'{val:.2f}', color='green', ha='left', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(current_dir, "benchmark_comparison_chart.png")
    plt.savefig(chart_path, dpi=120)
    print(f"[+] 对比图表已保存至: {chart_path}")
    
    # 无需在自动化脚本中显示 plt.show()
    # plt.show()

if __name__ == "__main__":
    main()
