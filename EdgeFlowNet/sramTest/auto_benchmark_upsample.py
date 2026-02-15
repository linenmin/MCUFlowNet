#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Head Upsample NAS Benchmark 脚本
对比 7 种 Head 上采样策略在不同分辨率下的 FPS / SRAM / 效率表现。
骨干默认使用 ShuffleNet + Bilinear，可通过 --backbone 参数切换。
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
from network.MultiScaleResNet_cell_upsample_search import MultiScaleResNet  # NAS 搜索版网络
from misc.utils import AccumPreds  # 多尺度累加函数
from vela.vela_compiler import run_vela  # Vela 编译器封装

# --- 测试配置 ---
RESOLUTIONS = [  # 测试分辨率列表 (从大到小)
    (384, 512),  # 原始
    (300, 400),  # 中高
    (264, 352),  # 中等
    (192, 256),  # 推荐
    (156, 208),  # 快速
    (120, 160),  # 极速
]

HEAD_CHOICES = [  # 7 种 Head 上采样策略
    {'name': 'Conv3x3',    'choice': 'resize_conv3x3'},    # Bilinear + Conv 3x3 (基线)
    {'name': 'Conv5x5',    'choice': 'resize_conv5x5'},    # Bilinear + Conv 5x5
    {'name': 'Conv7x7',    'choice': 'resize_conv7x7'},    # Bilinear + Conv 7x7 (原始)
    {'name': 'DSConv3x3',  'choice': 'resize_dsconv3x3'},  # Bilinear + DW3x3 + PW1x1
    {'name': 'MBConv_E4',  'choice': 'resize_mbconv_e4'},  # Bilinear + MBConv(expand=4)
    {'name': 'MBConv_E6',  'choice': 'resize_mbconv_e6'},  # Bilinear + MBConv(expand=6)
    {'name': 'Shuffle',    'choice': 'resize_shuffle'},     # Bilinear + 1x1 + ShuffleNetV2
]

INIT_NEURONS = 32  # 初始通道数 (偶数，兼容 ShuffleNet)


def generate_tflite(h, w, head_choice, backbone_block, output_dir):
    """生成指定 Head 上采样策略的多尺度 TFLite 模型"""
    print(f"  [1/2] 生成 TFLite: {h}x{w} Head={head_choice['name']}...")

    graph = tf.Graph()  # 创建独立计算图
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        try:
            # 输入占位符: [batch=1, H, W, 6通道(双帧RGB)]
            input_ph = tf.compat.v1.placeholder(tf.float32, [1, h, w, 6], name='input')

            # 实例化模型
            model = MultiScaleResNet(
                InputPH=input_ph,
                InitNeurons=INIT_NEURONS,
                ExpansionFactor=2.0,
                NumSubBlocks=2,
                NumOut=4,
                NumBlocks=1,
                BlockType=backbone_block,         # 骨干 Block 类型
                MBConvExpandRatio=6,              # 骨干 MBConv 扩展比
                HeadUpsampleChoice=head_choice['choice'],  # Head 上采样策略
            )

            outputs = model.Network()  # 构建网络

            # 多尺度累加输出 (AccumPreds)
            if isinstance(outputs, list) and len(outputs) > 1:
                accumOut, _ = AccumPreds(outputs)
                final_output = accumOut[..., 0:2]  # 取前 2 通道 (u, v 光流)
            else:
                final_output = outputs[-1] if isinstance(outputs, list) else outputs

            sess.run(tf.compat.v1.global_variables_initializer())  # 随机初始化权重

            # TFLite INT8 量化转换
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 默认优化
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # INT8 算子
                tf.lite.OpsSet.TFLITE_BUILTINS          # 标准算子
            ]
            converter.inference_input_type = tf.int8   # INT8 输入
            converter.inference_output_type = tf.int8  # INT8 输出

            def rep_data():  # 代表性数据集 (量化校准)
                for _ in range(3):
                    yield [np.random.rand(1, h, w, 6).astype(np.float32)]
            converter.representative_dataset = rep_data

            tflite_model = converter.convert()  # 执行转换

            # 保存 TFLite 文件
            tflite_path = os.path.join(output_dir, f"{head_choice['name']}_{h}x{w}.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            return tflite_path
        except Exception as e:
            print(f"  [!] TFLite 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            sess.close()


def run_benchmark(output_base, backbone_block):
    """运行全部 Benchmark"""
    all_results = []  # 存储所有结果

    print("=" * 70)
    print(f"Head Upsample NAS Benchmark: 7 种策略 × {len(RESOLUTIONS)} 种分辨率")
    print(f"骨干 Block: {backbone_block} | Predict Conv: 7x7 | 输出模式: 多尺度累加")
    print("=" * 70)

    for hc in HEAD_CHOICES:  # 遍历 Head 上采样策略
        choice_name = hc['name']
        choice_dir = os.path.join(output_base, choice_name)  # 每个策略一个子目录
        os.makedirs(choice_dir, exist_ok=True)

        print(f"\n>>> Head 上采样策略: {choice_name} ({hc['choice']})")

        for h, w in RESOLUTIONS:  # 遍历分辨率
            # 生成 TFLite
            tflite_path = generate_tflite(h, w, hc, backbone_block, choice_dir)
            if not tflite_path:
                continue

            # Vela 编译 (verbose 模式, 生成 per-layer 报告)
            sram_mb, time_ms = run_vela(
                tflite_path,
                mode='verbose',       # 详细模式
                output_dir=choice_dir,  # 输出到策略目录
                optimise='Size',       # 优化 SRAM 使用
                silent=True            # 静默输出
            )

            if sram_mb is not None:
                fps = 1000.0 / time_ms if time_ms > 0 else 0  # 计算 FPS
                print(f"  [*] {h}x{w}: SRAM={sram_mb:.2f}MB, Time={time_ms:.1f}ms, FPS={fps:.1f}")
                all_results.append({
                    'HeadChoice': choice_name,
                    'Resolution': f'{h}x{w}',
                    'H': h, 'W': w,
                    'SRAM_MB': sram_mb,
                    'Time_ms': time_ms,
                    'FPS': fps,
                })
            else:
                print(f"  [!] {h}x{w}: Vela 编译失败")

    return all_results


def save_and_plot(results, output_base):
    """保存 CSV 并生成三子图对比图"""
    if not results:
        print("[!] 没有有效结果")
        return

    df = pd.DataFrame(results)  # 转换为 DataFrame

    # 保存 CSV
    csv_path = os.path.join(output_base, 'benchmark_upsample_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[+] 结果已保存: {csv_path}")

    # --- 绘图: 3 个子图 ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))

    resolutions = [f'{h}x{w}' for h, w in RESOLUTIONS]  # X 轴标签
    x = np.arange(len(resolutions))  # X 轴位置
    n_choices = len(HEAD_CHOICES)  # 策略数量
    width = 0.8 / n_choices  # 柱宽 (自适应)
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63',  # 7 种颜色
              '#9C27B0', '#F44336', '#00BCD4']

    # --- 子图 1: FPS 对比 (柱状图) ---
    for i, hc in enumerate(HEAD_CHOICES):
        hc_data = df[df['HeadChoice'] == hc['name']]  # 过滤当前策略数据
        fps_vals = []
        for r in resolutions:
            match = hc_data[hc_data['Resolution'] == r]
            fps_vals.append(match['FPS'].values[0] if len(match) > 0 else 0)

        offset = (i - n_choices / 2 + 0.5) * width  # 柱偏移
        bars = ax1.bar(x + offset, fps_vals, width, label=hc['name'],
                       color=colors[i % len(colors)], alpha=0.85)
        # 标注数值
        for bar, val in zip(bars, fps_vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{val:.1f}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('FPS (higher is better)')
    ax1.set_title('Head Upsample NAS — Inference Speed (FPS)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(resolutions, rotation=45)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # --- 子图 2: SRAM 对比 (折线图) ---
    for i, hc in enumerate(HEAD_CHOICES):
        hc_data = df[df['HeadChoice'] == hc['name']]
        sram_vals = []
        for r in resolutions:
            match = hc_data[hc_data['Resolution'] == r]
            sram_vals.append(match['SRAM_MB'].values[0] if len(match) > 0 else 0)

        ax2.plot(x, sram_vals, marker='o', label=hc['name'],
                 color=colors[i % len(colors)], linewidth=2)
        # 标注数值
        for xi, val in zip(x, sram_vals):
            if val > 0:
                ax2.text(xi, val + 0.05, f'{val:.2f}', ha='center', fontsize=7,
                         color=colors[i % len(colors)])

    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='SRAM Limit (2MB)')
    ax2.set_xlabel('Resolution')
    ax2.set_ylabel('Peak SRAM (MB)')
    ax2.set_title('Head Upsample NAS — Memory Usage (SRAM)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(resolutions, rotation=45)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # --- 子图 3: 效率指标 FPS/SRAM (柱状图) ---
    for i, hc in enumerate(HEAD_CHOICES):
        hc_data = df[df['HeadChoice'] == hc['name']]
        eff_vals = []
        for r in resolutions:
            match = hc_data[hc_data['Resolution'] == r]
            if len(match) > 0 and match['SRAM_MB'].values[0] > 0:
                eff = match['FPS'].values[0] / match['SRAM_MB'].values[0]  # FPS/MB
            else:
                eff = 0
            eff_vals.append(eff)

        offset = (i - n_choices / 2 + 0.5) * width
        bars = ax3.bar(x + offset, eff_vals, width, label=hc['name'],
                       color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, eff_vals):
            if val > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{val:.1f}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax3.set_xlabel('Resolution')
    ax3.set_ylabel('Efficiency (FPS / SRAM_MB)')
    ax3.set_title('Head Upsample NAS — Efficiency (FPS per MB SRAM)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(resolutions, rotation=45)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()  # 自动调整布局
    chart_path = os.path.join(output_base, 'benchmark_upsample_chart.png')
    plt.savefig(chart_path, dpi=150)
    print(f"[+] 图表已保存: {chart_path}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Head Upsample NAS Benchmark")
    parser.add_argument('--backbone', type=str, default='shufflenet',
                        choices=['resblock', 'mbconv', 'shufflenet'],
                        help='骨干 Block 类型 (默认: shufflenet)')
    args = parser.parse_args()

    # 输出目录
    output_base = os.path.join(current_dir, 'output_benchmark_upsample')
    os.makedirs(output_base, exist_ok=True)  # 创建输出目录

    results = run_benchmark(output_base, args.backbone)  # 运行 Benchmark
    save_and_plot(results, output_base)  # 保存结果并绘图

    print("\n[+] Head Upsample NAS Benchmark 完成!")


if __name__ == '__main__':
    main()

