#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vela Per-Layer CSV 可视化脚本
读取 Vela 生成的逐层性能 CSV，绘制 SRAM、Op Cycles、MAC Count、Util% 四个子图
"""

import os  # 路径处理
import pandas as pd  # 数据读取
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算

# ==================== 配置区 ====================
# 手动指定 CSV 文件路径
CSV_PATH = r"./output_benchmark_all_multiscale/MBConv_E6_Bilinear/MBConv_E6_Bilinear_192x256_per-layer.csv"
# 输出图片路径 (自动根据 CSV 名称生成)
OUTPUT_DIR = os.path.dirname(CSV_PATH)  # 输出到 CSV 同目录


def smart_annotate(ax, x, y, text, fontsize, color='black', y_data=None):
    """
    智能标注：根据数据点位置和子图范围决定标注在上方还是下方
    Args:
        ax: matplotlib 轴对象
        x: 数据点 x 坐标
        y: 数据点 y 坐标
        text: 标注文本
        fontsize: 字体大小
        color: 字体颜色
        y_data: 整个 y 数据数组（用于计算范围）
    """
    if y_data is not None:
        y_range = np.max(y_data) - np.min(y_data)
        y_mid = (np.max(y_data) + np.min(y_data)) / 2
        # 如果点在上半部分，标注在下方；否则标注在上方
        if y > y_mid + y_range * 0.1:
            offset = (0, -12)  # 标注在下方
            va = 'top'
        else:
            offset = (0, 8)  # 标注在上方
            va = 'bottom'
    else:
        offset = (0, 8)
        va = 'bottom'
    
    ax.annotate(text, xy=(x, y), xytext=offset, textcoords='offset points',
                fontsize=fontsize, ha='center', va=va, color=color)


def load_and_plot(csv_path):
    """加载 CSV 并绘制四子图（竖向排列，共享 X 轴，每点都标注）"""
    
    # 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 提取需要的列 (注意 CSV 中可能有空格)
    df.columns = df.columns.str.strip()  # 去除列名空格
    
    # 定义需要的列名
    layer_col = 'TFLite_operator'  # 层名
    sram_col = 'SRAM Usage'  # SRAM 使用量
    sram_ac_col = 'SRAM AC'  # SRAM AC (另一个 SRAM 指标)
    sram_pct_col = 'Peak%'  # SRAM 占峰值的百分比
    cycles_col = 'Op Cycles'  # 操作周期数
    cycles_pct_col = 'Network%'  # 周期占网络的百分比 (第一个 Network%)
    mac_col = 'MAC Count'  # MAC 操作数
    mac_pct_col = 'Network%.1'  # MAC 占网络百分比 (第二个 Network%，pandas 自动重命名)
    util_col = 'Util%'  # 利用率
    
    # 处理重复列名问题 (pandas 自动添加 .1 后缀)
    if 'Network%.1' not in df.columns:
        network_cols = [c for c in df.columns if c.startswith('Network%')]
        if len(network_cols) >= 2:
            mac_pct_col = network_cols[1]
        else:
            mac_pct_col = network_cols[0] if network_cols else None
    
    # 提取数据
    layers = df[layer_col].astype(str)  # 层名转字符串
    sram = df[sram_col].values / 1024  # 转换为 KB
    sram_ac = df[sram_ac_col].values / 1024 if sram_ac_col in df.columns else None  # SRAM AC 转 KB
    sram_pct = df[sram_pct_col].values
    cycles = df[cycles_col].values / 1e6  # 转换为百万周期
    cycles_pct = df[cycles_pct_col].values if cycles_pct_col in df.columns else np.zeros(len(df))
    mac = df[mac_col].values / 1e6  # 转换为百万 MAC
    mac_pct = df[mac_pct_col].values if mac_pct_col and mac_pct_col in df.columns else np.zeros(len(df))
    util = df[util_col].values
    
    # 创建 X 轴索引
    x = np.arange(len(layers))
    
    # 创建图形 (4 个子图，竖向排列，共享 X 轴)
    fig, axes = plt.subplots(4, 1, figsize=(18, 18), sharex=True)
    fig.suptitle(f'Per-Layer Performance Analysis\n{os.path.basename(csv_path)}', fontsize=14, fontweight='bold')
    
    # 标注参数
    marker_size = 6
    annotation_fontsize = 8  # 放大字体
    
    # --- 子图 1: SRAM Usage + SRAM AC ---
    ax1 = axes[0]
    ax1.plot(x, sram, marker='o', markersize=marker_size, color='tab:blue', linewidth=1.5, label='SRAM Usage')
    if sram_ac is not None:
        ax1.plot(x, sram_ac, marker='x', markersize=marker_size, color='tab:red', linewidth=1.5, 
                linestyle='--', label='SRAM AC')
    ax1.set_ylabel('SRAM (KB)', fontsize=10)
    ax1.set_title('SRAM Usage & SRAM AC per Layer', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    # 每个点都标注值和百分比 (SRAM Usage)
    for i in range(len(x)):
        smart_annotate(ax1, x[i], sram[i], f'{sram[i]:.0f}\n({sram_pct[i]:.0f}%)', 
                      annotation_fontsize, color='tab:blue', y_data=sram)
    # SRAM AC 标注
    if sram_ac is not None:
        for i in range(len(x)):
            smart_annotate(ax1, x[i], sram_ac[i], f'{sram_ac[i]:.0f}', 
                          annotation_fontsize - 1, color='tab:red', y_data=sram_ac)
    
    # --- 子图 2: Op Cycles ---
    ax2 = axes[1]
    ax2.plot(x, cycles, marker='s', markersize=marker_size, color='tab:orange', linewidth=1.5)
    ax2.set_ylabel('Op Cycles (M)', fontsize=10)
    ax2.set_title('Op Cycles per Layer', fontsize=11)
    ax2.grid(True, alpha=0.3)
    # 每个点都标注值和百分比
    for i in range(len(x)):
        pct = cycles_pct[i] if cycles_pct[i] > 0 else 0
        smart_annotate(ax2, x[i], cycles[i], f'{cycles[i]:.1f}\n({pct:.1f}%)', 
                      annotation_fontsize, y_data=cycles)
    
    # --- 子图 3: MAC Count ---
    ax3 = axes[2]
    ax3.plot(x, mac, marker='^', markersize=marker_size, color='tab:green', linewidth=1.5)
    ax3.set_ylabel('MAC Count (M)', fontsize=10)
    ax3.set_title('MAC Count per Layer', fontsize=11)
    ax3.grid(True, alpha=0.3)
    # 每个点都标注值和百分比
    for i in range(len(x)):
        pct = mac_pct[i] if mac_pct[i] > 0 else 0
        smart_annotate(ax3, x[i], mac[i], f'{mac[i]:.1f}\n({pct:.1f}%)', 
                      annotation_fontsize, y_data=mac)
    
    # --- 子图 4: Utilization ---
    ax4 = axes[3]
    bars = ax4.bar(x, util, color='tab:purple', alpha=0.7)
    ax4.set_ylabel('Utilization (%)', fontsize=10)
    ax4.set_title('NPU Utilization per Layer', fontsize=11)
    ax4.axhline(y=np.mean(util), color='red', linestyle='--', linewidth=1, label=f'Avg: {np.mean(util):.1f}%')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    # 每个柱子都标注值
    for i, bar in enumerate(bars):
        smart_annotate(ax4, bar.get_x() + bar.get_width()/2, bar.get_height(), 
                      f'{util[i]:.1f}%', annotation_fontsize, y_data=util)
    
    # 设置共享 X 轴（只在最后一个子图显示层名）
    ax4.set_xlabel('Layer', fontsize=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)  # 倾斜 45 度避免遮挡
    
    # 调整布局，给 X 轴标签留出足够空间
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # 保存图片
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f'{base_name}_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[+] 图表已保存: {output_path}')
    
    # 显示图片
    plt.show()


def main():
    """主入口"""
    # 检查文件是否存在
    if not os.path.exists(CSV_PATH):
        print(f'[!] 错误: 找不到 CSV 文件: {CSV_PATH}')
        print('[*] 请修改脚本顶部的 CSV_PATH 变量')
        return
    
    print(f'[*] 正在分析: {CSV_PATH}')
    load_and_plot(CSV_PATH)


if __name__ == '__main__':
    main()
