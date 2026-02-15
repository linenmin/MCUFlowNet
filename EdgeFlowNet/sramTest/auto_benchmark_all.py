#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合 Benchmark 脚本：比较 5 种模型配置
支持单尺度 (只取最后输出) 和多尺度 (AccumPreds 累加) 两种模式
"""

import os  # 路径处理
import sys  # 系统退出
import argparse  # 命令行参数
import numpy as np  # 数值计算
import pandas as pd  # 数据表格
import matplotlib.pyplot as plt  # 绘图
import tensorflow as tf  # TF 核心库

# --- 环境设置 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TF 日志
tf.compat.v1.disable_eager_execution()  # TF1 兼容模式

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))  # sramTest 目录
root_dir = os.path.dirname(current_dir)  # 项目根目录
if current_dir not in sys.path: sys.path.insert(0, current_dir)  # 添加 sramTest 到 path
if root_dir not in sys.path: sys.path.append(root_dir)  # 添加根目录

# --- 导入模块 ---
from network.MultiScaleResNet_cell_33decoder import MultiScaleResNet  # 统一的网络模块
from misc.utils import AccumPreds  # 多尺度累加函数
from vela.vela_compiler import run_vela  # Vela 编译器封装

# --- 测试配置 ---
RESOLUTIONS = [  # 测试分辨率列表
    (384, 512),  # 原始
    (300, 400),  # 中高
    (264, 352),  # 中等
    (192, 256),  # 推荐
    (156, 208),  # 快速
    (120, 160),  # 极速
]

MODEL_CONFIGS = [  # 5 种模型配置
    {'name': 'ResNet_Transpose',     'block': 'resblock',   'upsample': 'transpose', 'expand': 6},
    {'name': 'ResNet_Bilinear',      'block': 'resblock',   'upsample': 'bilinear',  'expand': 6},
    {'name': 'ShuffleNet_Bilinear',  'block': 'shufflenet', 'upsample': 'bilinear',  'expand': 6},
    {'name': 'MBConv_E6_Bilinear',   'block': 'mbconv',     'upsample': 'bilinear',  'expand': 6},
    {'name': 'MBConv_E4_Bilinear',   'block': 'mbconv',     'upsample': 'bilinear',  'expand': 4},
]

INIT_NEURONS = 32  # 初始通道数 (偶数，兼容 ShuffleNet)


def generate_tflite(h, w, config, output_dir, use_multiscale=False):
    """生成指定配置的 TFLite 模型"""
    scale_mode = "multiscale" if use_multiscale else "singlescale"
    print(f"  [1/2] 生成 TFLite: {h}x{w} {config['name']} ({scale_mode})...")
    
    graph = tf.Graph()  # 创建独立图
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        try:
            input_ph = tf.compat.v1.placeholder(tf.float32, [1, h, w, 6], name='input')  # 输入占位符
            
            # 实例化模型
            model = MultiScaleResNet(
                InputPH=input_ph,
                InitNeurons=INIT_NEURONS,
                ExpansionFactor=2.0,
                NumSubBlocks=2,
                NumOut=4,
                NumBlocks=1,
                BlockType=config['block'],
                UpsampleType=config['upsample'],
                MBConvExpandRatio=config['expand']
            )
            
            outputs = model.Network()  # 构建网络
            
            # 根据模式选择输出
            if use_multiscale and isinstance(outputs, list) and len(outputs) > 1:
                accumOut, _ = AccumPreds(outputs)
                final_output = accumOut[..., 0:2]  # 取前 2 通道 (u, v)
            else:
                final_output = outputs[-1] if isinstance(outputs, list) else outputs
            
            sess.run(tf.compat.v1.global_variables_initializer())  # 随机初始化
            
            # TFLite 转换
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 默认优化
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.inference_input_type = tf.int8  # INT8 输入
            converter.inference_output_type = tf.int8  # INT8 输出
            
            def rep_data():  # 代表性数据集
                for _ in range(3):
                    yield [np.random.rand(1, h, w, 6).astype(np.float32)]
            converter.representative_dataset = rep_data
            
            tflite_model = converter.convert()  # 执行转换
            
            tflite_path = os.path.join(output_dir, f"{config['name']}_{h}x{w}.tflite")  # 保存路径
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            return tflite_path
        except Exception as e:
            print(f"  [!] TFLite 生成失败: {e}")
            return None
        finally:
            sess.close()


def run_benchmark(output_base, use_multiscale=False):
    """运行全部 Benchmark"""
    all_results = []  # 存储所有结果
    scale_mode = "多尺度累加" if use_multiscale else "单尺度"
    
    print("=" * 60)
    print(f"综合 Benchmark: 5 种模型配置 × 多分辨率 ({scale_mode})")
    print("=" * 60)
    
    for config in MODEL_CONFIGS:  # 遍历模型配置
        config_name = config['name']
        config_dir = os.path.join(output_base, config_name)  # 每个配置一个目录
        os.makedirs(config_dir, exist_ok=True)
        
        print(f"\n>>> 配置: {config_name}")
        
        for h, w in RESOLUTIONS:  # 遍历分辨率
            # 生成 TFLite
            tflite_path = generate_tflite(h, w, config, config_dir, use_multiscale)
            if not tflite_path:
                continue
            
            # Vela 编译 (verbose 模式生成完整报告)
            sram_mb, time_ms = run_vela(
                tflite_path,
                mode='verbose',  # 详细模式
                output_dir=config_dir,  # 输出到配置目录
                optimise='Size',  # 优化模式
                silent=True  # 静默输出
            )
            
            if sram_mb is not None:
                fps = 1000.0 / time_ms if time_ms > 0 else 0
                print(f"  [*] {h}x{w}: SRAM={sram_mb:.2f}MB, FPS={fps:.1f}")
                all_results.append({
                    'Config': config_name,
                    'Resolution': f'{h}x{w}',
                    'H': h, 'W': w,
                    'SRAM_MB': sram_mb,
                    'Time_ms': time_ms,
                    'FPS': fps
                })
            else:
                print(f"  [!] {h}x{w}: Vela 编译失败")
    
    return all_results


def save_and_plot(results, output_base):
    """保存结果并绘图"""
    if not results:
        print("[!] 没有有效结果")
        return
    
    df = pd.DataFrame(results)  # 转换为 DataFrame
    
    # 保存 CSV
    csv_path = os.path.join(output_base, 'benchmark_all_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[+] 结果已保存: {csv_path}")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    resolutions = [f'{h}x{w}' for h, w in RESOLUTIONS]  # X 轴标签
    x = np.arange(len(resolutions))  # X 轴位置
    width = 0.15  # 柱宽
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']  # 颜色
    
    # FPS 对比图
    for i, config in enumerate(MODEL_CONFIGS):
        config_data = df[df['Config'] == config['name']]
        fps_vals = [config_data[config_data['Resolution'] == r]['FPS'].values[0] 
                   if len(config_data[config_data['Resolution'] == r]) > 0 else 0 
                   for r in resolutions]
        bars = ax1.bar(x + i * width, fps_vals, width, label=config['name'], color=colors[i], alpha=0.8)
        # 标注数值
        for bar, val in zip(bars, fps_vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', 
                        ha='center', va='bottom', fontsize=7, rotation=45)
    
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('FPS')
    ax1.set_title('EdgeFlowNet Inference Speed Comparison (FPS, higher is better)')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(resolutions, rotation=45)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # SRAM 对比图
    for i, config in enumerate(MODEL_CONFIGS):
        config_data = df[df['Config'] == config['name']]
        sram_vals = [config_data[config_data['Resolution'] == r]['SRAM_MB'].values[0] 
                    if len(config_data[config_data['Resolution'] == r]) > 0 else 0 
                    for r in resolutions]
        ax2.plot(x, sram_vals, marker='o', label=config['name'], color=colors[i], linewidth=2)
        # 标注数值
        for xi, val in zip(x, sram_vals):
            if val > 0:
                ax2.text(xi, val + 0.1, f'{val:.2f}', ha='center', fontsize=8, color=colors[i])
    
    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='SRAM Limit (2MB)')
    ax2.set_xlabel('Resolution')
    ax2.set_ylabel('Peak SRAM (MB)')
    ax2.set_title('EdgeFlowNet Memory Usage Comparison (SRAM, lower is better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(resolutions, rotation=45)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(output_base, 'benchmark_all_chart.png')
    plt.savefig(chart_path, dpi=150)
    print(f"[+] 图表已保存: {chart_path}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="综合 Benchmark 脚本")
    parser.add_argument('--multiscale', action='store_true', 
                        help='使用多尺度累加输出 (默认: 单尺度, 只取最后一个输出)')
    args = parser.parse_args()
    
    # 根据模式设置输出目录名
    if args.multiscale:
        output_base = os.path.join(current_dir, 'output_benchmark_all_multiscale_33decoder')
    else:
        output_base = os.path.join(current_dir, 'output_benchmark_all_singlescale')
    
    os.makedirs(output_base, exist_ok=True)  # 创建输出目录
    results = run_benchmark(output_base, use_multiscale=args.multiscale)  # 运行 Benchmark
    save_and_plot(results, output_base)  # 保存并绘图
    print("\n[+] 完成!")


if __name__ == '__main__':
    main()
