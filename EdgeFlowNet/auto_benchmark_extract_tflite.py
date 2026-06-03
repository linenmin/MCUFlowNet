import os
import sys
import subprocess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# --- 0. 环境设置 ---
# TF 2.15 不需要设置 TF_USE_LEGACY_KERAS，直接跑即可

# [关键] 必须保留！设置 TF1 兼容模式，否则 placeholder 会报错
tf.compat.v1.disable_eager_execution()

# 添加代码路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'code'))

# 尝试导入模型定义
try:
    from network.MultiScaleResNet import MultiScaleResNet
except ImportError as e:
    print(f"Error importing MultiScaleResNet: {e}")
    sys.exit(1)

# --- 1. 配置部分 ---

# 分辨率列表
resolutions_to_test = [
    (384, 512),  # 原始（4:3）
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

checkpoint_path = os.path.join(current_dir, 'checkpoints', 'best.ckpt')

# Vela 配置
vela_config_file = "vela.ini" 
sys_config = "Grove_Sys_Config"
mem_mode = "Grove_Mem_Mode"
accelerator = "ethos-u55-64"
output_dir = "output_benchmark"
optimise = "Performance"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# --- 2. 函数定义 ---

def generate_tflite(height, width):
    """根据指定分辨率生成 INT8 量化的 TFLite 模型"""
    print(f"\n[1/2] Generating TFLite for resolution: {height}x{width} ...")
    
    # [关键修复] 创建一个全新的图对象
    # 这能确保每次循环都是在一个干净的“平行宇宙”中进行，防止递归错误和内存泄漏
    new_graph = tf.Graph()
    
    with new_graph.as_default():
        # 在这个 with 块内，所有的 tf 操作都会绑定到 new_graph 上
        
        # 创建 Session (必须绑定到这个新图)
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
            final_output = network_outputs[-1] if isinstance(network_outputs, list) else network_outputs
            
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, checkpoint_path)
            
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            def representative_dataset_gen():
                for _ in range(10):
                    yield [np.random.uniform(0.0, 1.0, size=[1, height, width, 6]).astype(np.float32)]
            converter.representative_dataset = representative_dataset_gen
            
            tflite_model = converter.convert()
            
            filename = f"edgeflownet_{height}_{width}_int8.tflite"
            with open(filename, "wb") as f:
                f.write(tflite_model)
            
            return filename
            
        except Exception as e:
            print(f"Error generating TFLite: {e}")
            return None
        finally:
            sess.close()
            # 显式清理图引用，帮助垃圾回收
            del new_graph

def run_vela_and_parse(tflite_file):
    """运行 Vela 编译器并解析生成的 CSV 报告"""
    print(f"[2/2] Running Vela compiler on {tflite_file} ...")
    
    # --- 关键修正：查找 Vela 可执行文件 ---
    # 尝试在当前 Python 环境的 Scripts 目录下找 vela.exe
    python_dir = os.path.dirname(sys.executable)
    vela_exe = os.path.join(python_dir, "Scripts", "vela.exe") # Windows 路径
    
    if not os.path.exists(vela_exe):
        # 如果找不到绝对路径，尝试直接用 "vela" (依赖系统 PATH)
        vela_cmd = "vela"
    else:
        vela_cmd = vela_exe
        
    # 构建 Vela 命令
    cmd = [
        vela_cmd, tflite_file,
        "--accelerator-config", accelerator,
        "--config", vela_config_file,
        "--system-config", sys_config,
        "--memory-mode", mem_mode,
        "--optimise", optimise, 
        "--output-dir", output_dir,        
        # "--supported-ops-report"    
    ]
    
    try:
        # --- 关键修正：shell=True ---
        # 在 Windows 上，有时需要 shell=True 才能正确解析命令
        # 但如果提供了绝对路径，shell=False 也可以。为了保险，我们先试绝对路径 + shell=False
        result = subprocess.run(cmd, capture_output=True, text=True, check=True) 
        
        # 寻找生成的 CSV 文件
        model_name = os.path.splitext(tflite_file)[0] 
        csv_path = os.path.join(output_dir, f"{model_name}_summary_{sys_config}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Error: Could not find Vela summary CSV at {csv_path}")
            print(f"Vela stderr:\n{result.stderr}") 
            return None, None
            
        # 读取 CSV 解析数据
        df = pd.read_csv(csv_path)
        
        # 提取关键指标
        sram_used = df['sram_memory_used'].values[0] / 1024.0  # MB
        inference_time = df['inference_time'].values[0] * 1000.0 # ms
        
        return sram_used, inference_time
        
    except subprocess.CalledProcessError as e:
        print(f"Vela compilation failed for {tflite_file} with exit code {e.returncode}:\n{e.stderr}")
        return None, None
    except Exception as e:
        print(f"Error running Vela or parsing CSV for {tflite_file}: {e}")
        # 打印一下当前尝试的 vela 路径，方便调试
        print(f"Tried executing: {vela_cmd}")
        return None, None
# --- 3. 主程序逻辑 ---

def main():
    results = {
        "Resolution": [],
        "SRAM (MB)": [],
        "FPS": []
    }

    print("=== Starting Automated Benchmark ===\n")

    for h, w in resolutions_to_test:
        # 1. 生成
        tflite_path = generate_tflite(h, w)
        if not tflite_path: continue
            
        # 2. 编译
        sram, time_ms = run_vela_and_parse(tflite_path)
        
        res_str = f"{h}x{w}"
        
        if sram is not None:
            fps = 1000.0 / time_ms if time_ms > 0 else 0
            print(f"Result {res_str}: SRAM={sram:.2f} MB, FPS={fps:.2f}")
            
            results["Resolution"].append(res_str)
            results["SRAM (MB)"].append(sram)
            results["FPS"].append(fps)
        else:
            print(f"Result {res_str}: Failed (Likely OOM)")

    if not results["Resolution"]:
        print("No successful runs.")
        return

    # 保存与绘图
    df_res = pd.DataFrame(results)
    df_res.to_csv("benchmark_results.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(df_res))
    
    # FPS 柱状图
    color = 'tab:blue'
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('FPS', color=color)
    bars = ax1.bar(x, df_res['FPS'], color=color, alpha=0.6, width=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_res['Resolution'], rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')

    # SRAM 折线图
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('SRAM (MB)', color=color)
    ax2.plot(x, df_res['SRAM (MB)'], color=color, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=2.0, color='gray', linestyle='--', label='Limit (2MB)') # 2MB 红线

    plt.title('EdgeFlowNet Benchmark on Grove Vision AI V2')
    plt.tight_layout()
    plt.savefig('benchmark_chart.png')
    print("\nDone! Chart saved to benchmark_chart.png")
    plt.show()

if __name__ == "__main__":
    main()