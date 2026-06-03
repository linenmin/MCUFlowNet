#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vela Per-Layer CSV 可视化脚本
读取 Vela 生成的逐层性能 CSV，绘制 SRAM、Op Cycles、Util% 三个子图（竖向排列，共享 X 轴）
支持按 ResNet 子结构合并（ResBlock / mbconv_* / Conv* 等下一级目录为一组）。
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==================== 配置区 ====================
# 手动指定 CSV 文件路径（无命令行 --csv 时使用）
CSV_PATH = r"./output_benchmark_all_multiscale_33decoder/MBConv_E6_Bilinear/MBConv_E6_Bilinear_192x256_per-layer.csv"


def _show_figure(fig=None):
    """非交互后端（如 Agg）下不调用 plt.show，避免告警与阻塞。"""
    if plt.get_backend().lower() == 'agg':
        plt.close(fig) if fig is not None else plt.close('all')
    else:
        plt.show()


def smart_annotate(ax, x, y, text, fontsize, color='black', y_data=None):
    """
    智能标注：根据数据点位置和子图范围决定标注在上方还是下方
    """
    off = max(6, int(fontsize * 0.55))
    if y_data is not None:
        y_range = np.max(y_data) - np.min(y_data)
        y_mid = (np.max(y_data) + np.min(y_data)) / 2
        if y > y_mid + y_range * 0.1:
            offset = (0, -off)
            va = 'top'
        else:
            offset = (0, off)
            va = 'bottom'
    else:
        offset = (0, off)
        va = 'bottom'

    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords='offset points',
        fontsize=fontsize,
        ha='center',
        va=va,
        color=color,
    )


def merge_key_from_name(name: str) -> str:
    """
    在 ResNetBlock* 之后取「下一级」目录名为逻辑块，同块内多算子合并。
    适用于 ResBlock*、mbconv_enc_*、ConvBNReLUBlock*、ResizeBilinear* 等。
    """
    s = str(name).split(';')[0].strip()
    parts = [p for p in s.split('/') if p]
    for i, p in enumerate(parts):
        if p.startswith('ResNetBlock'):
            if i + 1 >= len(parts):
                return '/'.join(parts[: i + 1])
            return '/'.join(parts[: i + 2])
    return s


def short_group_label(merge_key: str, max_len: int = 40) -> str:
    """X 轴短标签：去掉 EncoderDecoderBlock0/ 前缀，过长则保留最后两段。"""
    k = merge_key.replace('EncoderDecoderBlock0/', '')
    if len(k) <= max_len:
        return k
    segs = k.split('/')
    if len(segs) >= 2:
        tail = '/'.join(segs[-2:])
        if len(tail) <= max_len:
            return tail
    return k[: max_len - 1] + '…'


def _aggregate_merged_groups(df: pd.DataFrame) -> pd.DataFrame:
    """按 Name 推导的 merge key 聚合；保持首次出现顺序。"""
    name_col = 'Name' if 'Name' in df.columns else None
    if name_col is None:
        return df

    work = df.copy()
    work['_mk'] = work[name_col].map(merge_key_from_name)
    work['_ord'] = np.arange(len(work), dtype=np.int64)

    cycles_col = 'Op Cycles'
    net_c_col = 'Network%'
    sram_col = 'SRAM Usage'
    sram_ac_col = 'SRAM AC'
    peak_col = 'Peak%'
    mac_col = 'MAC Count'
    util_col = 'Util%'

    mac_net_col = 'Network%.1'
    if mac_net_col not in work.columns:
        network_cols = [c for c in work.columns if c.startswith('Network%')]
        mac_net_col = network_cols[1] if len(network_cols) >= 2 else (network_cols[0] if network_cols else None)

    g_ord = work.groupby('_mk', sort=False)['_ord'].min()
    keys_in_order = g_ord.sort_values().index.tolist()

    rows = []
    for mk in keys_in_order:
        sub = work[work['_mk'] == mk]
        if util_col in sub.columns:
            wv = sub[cycles_col].values
            uv = sub[util_col].values
            m = wv > 0
            if m.any():
                w_util = float(np.average(uv[m], weights=wv[m]))
            else:
                w_util = float(np.mean(uv))
        else:
            w_util = 0.0
        row = {
            'TFLite_operator': short_group_label(mk),
            'Name': mk,
            cycles_col: sub[cycles_col].sum(),
            net_c_col: sub[net_c_col].sum(),
            sram_col: sub[sram_col].max(),
            peak_col: sub[peak_col].max(),
            mac_col: sub[mac_col].sum(),
            util_col: w_util,
        }
        if sram_ac_col in sub.columns:
            row[sram_ac_col] = sub[sram_ac_col].max()
        if mac_net_col and mac_net_col in sub.columns:
            row[mac_net_col] = sub[mac_net_col].sum()
        rows.append(row)

    return pd.DataFrame(rows)


def read_perlayer_arrays(csv_path, merge_logical_blocks=False):
    """读取 CSV 并返回绘图所需数组与层名。

    merge_logical_blocks: 为 True 时按 ResNetBlock 下一级子目录合并多行（点数变少、字号可更大）。
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if merge_logical_blocks and 'Name' in df.columns:
        df = _aggregate_merged_groups(df)

    layer_col = 'TFLite_operator'
    sram_col = 'SRAM Usage'
    sram_ac_col = 'SRAM AC'
    sram_pct_col = 'Peak%'
    cycles_col = 'Op Cycles'
    cycles_pct_col = 'Network%'
    mac_col = 'MAC Count'
    mac_pct_col = 'Network%.1'
    util_col = 'Util%'

    if 'Network%.1' not in df.columns:
        network_cols = [c for c in df.columns if c.startswith('Network%')]
        if len(network_cols) >= 2:
            mac_pct_col = network_cols[1]
        else:
            mac_pct_col = network_cols[0] if network_cols else None

    layers = df[layer_col].astype(str)
    sram = df[sram_col].values / 1024
    sram_ac = df[sram_ac_col].values / 1024 if sram_ac_col in df.columns else None
    sram_pct = df[sram_pct_col].values
    cycles = df[cycles_col].values / 1e6
    cycles_pct = df[cycles_pct_col].values if cycles_pct_col in df.columns else np.zeros(len(df))
    mac = df[mac_col].values / 1e6
    mac_pct = df[mac_pct_col].values if mac_pct_col and mac_pct_col in df.columns else np.zeros(len(df))
    util = df[util_col].values
    x = np.arange(len(layers))

    return {
        'layers': layers,
        'x': x,
        'sram': sram,
        'sram_ac': sram_ac,
        'sram_pct': sram_pct,
        'cycles': cycles,
        'cycles_pct': cycles_pct,
        'mac': mac,
        'mac_pct': mac_pct,
        'util': util,
    }


def _layer_list(layers) -> list:
    """统一为 str 列表。"""
    return [str(x) for x in layers]


def _x_axis_tick_params(use_numeric: bool, n: int, layers):
    """
    返回 (tick_label_strings, xlabel, rotation_deg, ha)。
    use_numeric 为 True 时 X 轴为 1..n。
    """
    if use_numeric:
        return [str(i + 1) for i in range(n)], 'Layer Num', 0, 'center'
    return _layer_list(layers), 'Layer', 45, 'right'


def print_layer_index_map(layers):
    """控制台打印序号与层名/合并键对照（便于 X 轴用数字时查阅）。"""
    names = _layer_list(layers)
    print('[*] 层序号 (1-based) 与名称对照:')
    for i, name in enumerate(names):
        disp = name if len(name) <= 120 else name[:117] + '...'
        print(f'  {i + 1}: {disp}')


def _annotation_fontsize_for_count(n, base_small=10, single_panel_base=22):
    """点数越多字号略降；合并后 n 变小，自动用更大标注。"""
    if n <= 8:
        return single_panel_base + 3
    if n <= 12:
        return single_panel_base + 1
    if n <= 20:
        return max(16, single_panel_base - 2)
    if n <= 28:
        return max(14, single_panel_base - 4)
    return max(11, min(base_small + 6, 560 // max(n, 1)))


def load_and_plot(csv_path, merge_logical_blocks=False, x_labels_numeric=False):
    """加载 CSV 并绘制三子图（竖向排列，共享 X 轴，每点都标注）"""
    d = read_perlayer_arrays(csv_path, merge_logical_blocks=merge_logical_blocks)
    layers, x = d['layers'], d['x']
    if x_labels_numeric:
        print_layer_index_map(layers)
    sram, sram_ac, sram_pct = d['sram'], d['sram_ac'], d['sram_pct']
    cycles, cycles_pct = d['cycles'], d['cycles_pct']
    util = d['util']

    fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
    n = len(x)
    fig.suptitle(
        f'Per-Layer Performance Analysis\n{os.path.basename(csv_path)}',
        fontsize=20,
        fontweight='bold',
    )

    marker_size = 6
    annotation_fontsize = max(10, min(15, _annotation_fontsize_for_count(n) - 2))
    axis_label_fs = 18
    subplot_title_fs = 17
    tick_label_fs = 16

    ax1 = axes[0]
    ax1.plot(x, sram, marker='o', markersize=marker_size, color='tab:blue', linewidth=1.5, label='SRAM Usage')
    if sram_ac is not None:
        ax1.plot(
            x,
            sram_ac,
            marker='x',
            markersize=marker_size,
            color='tab:red',
            linewidth=1.5,
            linestyle='--',
            label='SRAM AC',
        )
    ax1.set_ylabel('SRAM (KB)', fontsize=axis_label_fs, fontweight='bold')
    ax1.set_title('SRAM Usage & SRAM AC per Layer', fontsize=subplot_title_fs, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=14)
    ax1.tick_params(axis='y', labelsize=tick_label_fs)
    for i in range(len(x)):
        smart_annotate(
            ax1,
            x[i],
            sram[i],
            f'{sram[i]:.0f}\n({sram_pct[i]:.0f}%)',
            annotation_fontsize,
            color='tab:blue',
            y_data=sram,
        )
    if sram_ac is not None:
        for i in range(len(x)):
            smart_annotate(
                ax1,
                x[i],
                sram_ac[i],
                f'{sram_ac[i]:.0f}',
                annotation_fontsize - 1,
                color='tab:red',
                y_data=sram_ac,
            )

    ax2 = axes[1]
    ax2.plot(x, cycles, marker='s', markersize=marker_size, color='tab:orange', linewidth=1.5)
    ax2.set_ylabel('Op Cycles (M)', fontsize=axis_label_fs, fontweight='bold')
    ax2.set_title('Op Cycles per Layer', fontsize=subplot_title_fs, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelsize=tick_label_fs)
    for i in range(len(x)):
        pct = cycles_pct[i] if cycles_pct[i] > 0 else 0
        smart_annotate(
            ax2,
            x[i],
            cycles[i],
            f'{cycles[i]:.1f}\n({pct:.1f}%)',
            annotation_fontsize,
            y_data=cycles,
        )

    ax3 = axes[2]
    bars = ax3.bar(x, util, color='tab:purple', alpha=0.7)
    ax3.set_ylabel('Utilization (%)', fontsize=axis_label_fs, fontweight='bold')
    ax3.set_title('NPU Utilization per Layer', fontsize=subplot_title_fs, fontweight='bold')
    ax3.axhline(y=np.mean(util), color='red', linestyle='--', linewidth=1, label=f'Avg: {np.mean(util):.1f}%')
    ax3.legend(loc='upper right', fontsize=14)
    ax3.tick_params(axis='y', labelsize=tick_label_fs)
    ax3.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        smart_annotate(
            ax3,
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{util[i]:.1f}%',
            annotation_fontsize,
            y_data=util,
        )

    xticks, xlab, xrot, xha = _x_axis_tick_params(x_labels_numeric, len(x), layers)
    ax3.set_xlabel(xlab, fontsize=axis_label_fs, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(xticks, rotation=xrot, ha=xha, fontsize=tick_label_fs, fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    merged_tag = '_merged' if merge_logical_blocks else ''
    idx_tag = '_idx' if x_labels_numeric else ''
    output_path = os.path.join(output_dir, f'{base_name}{merged_tag}{idx_tag}_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[+] 图表已保存: {output_path}')
    _show_figure(fig)


def plot_op_cycles_only(
    csv_path,
    output_path=None,
    dpi=150,
    merge_logical_blocks=False,
    x_labels_numeric=False,
):
    """
    仅绘制原图中第二个子图：Op Cycles per Layer。
    单图放大字号，标注偏移随字号缩放，避免拥挤。
    """
    d = read_perlayer_arrays(csv_path, merge_logical_blocks=merge_logical_blocks)
    layers, x = d['layers'], d['x']
    if x_labels_numeric:
        print_layer_index_map(layers)
    cycles, cycles_pct = d['cycles'], d['cycles_pct']
    n = len(x)

    ann_fs = _annotation_fontsize_for_count(n)
    title_fs = ann_fs + 10
    ylab_fs = ann_fs + 8
    xlab_fs = ann_fs + 7
    tick_fs = max(ann_fs + 4, 20)
    y_tick_fs = max(ann_fs + 4, 20)

    fig_w = max(14.0, min(28.0, 0.55 * n + 8))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 8.2))
    marker_size = max(7, min(11, 130 // max(n, 1)))

    ax.plot(x, cycles, marker='s', markersize=marker_size, color='tab:orange', linewidth=2.2)
    ax.set_ylabel('Op Cycles (M)', fontsize=ylab_fs, fontweight='bold')
    ax.set_title('Op Cycles per Layer', fontsize=title_fs, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xmargin(0.02)
    ax.tick_params(axis='y', labelsize=y_tick_fs)

    for i in range(len(x)):
        pct = cycles_pct[i] if cycles_pct[i] > 0 else 0
        smart_annotate(
            ax,
            x[i],
            cycles[i],
            f'{cycles[i]:.1f}\n({pct:.1f}%)',
            ann_fs,
            y_data=cycles,
        )

    xticks, xlab, xrot, xha = _x_axis_tick_params(x_labels_numeric, n, layers)
    ax.set_xlabel(xlab, fontsize=xlab_fs, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation=xrot, ha=xha, fontsize=tick_fs, fontweight='bold')

    fig.suptitle(os.path.basename(csv_path), fontsize=title_fs + 2, y=1.03, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_dir = os.path.dirname(csv_path)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        merged_tag = '_merged' if merge_logical_blocks else ''
        idx_tag = '_idx' if x_labels_numeric else ''
        output_path = os.path.join(output_dir, f'{base_name}{merged_tag}{idx_tag}_op_cycles_only.png')

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f'[+] Op Cycles 单图已保存: {output_path}')
    _show_figure(fig)


def main():
    parser = argparse.ArgumentParser(description='Vela per-layer CSV 可视化')
    parser.add_argument('--csv', default=None, help='per-layer CSV 路径（默认使用脚本内 CSV_PATH）')
    parser.add_argument(
        '--panel',
        choices=('all', 'op_cycles'),
        default='all',
        help='all: 三子图完整分析图；op_cycles: 仅第二个子图（Op Cycles）',
    )
    parser.add_argument(
        '--out',
        default=None,
        help='输出 PNG 路径（仅 op_cycles 时可选；默认写到 CSV 同目录 *_op_cycles_only.png）',
    )
    parser.add_argument('--dpi', type=int, default=150, help='输出 DPI')
    parser.add_argument(
        '--merge-logical-blocks',
        action='store_true',
        help='按 Name 中 ResNetBlock* 的下一级子目录合并多算子（ResBlock/mbconv/Conv*/Resize 等），减少 X 轴点数',
    )
    parser.add_argument(
        '--x-labels-numeric',
        action='store_true',
        help='X 轴刻度改为 1、2、3…（控制台会打印序号与层名对照）；输出文件名带 _idx',
    )
    args = parser.parse_args()

    csv_path = args.csv or CSV_PATH
    if not os.path.exists(csv_path):
        print(f'[!] 错误: 找不到 CSV 文件: {csv_path}')
        print('[*] 请使用 --csv 或修改脚本顶部的 CSV_PATH')
        return

    flags = []
    if args.merge_logical_blocks:
        flags.append('merge logical blocks')
    if args.x_labels_numeric:
        flags.append('X = 1..n')
    extra = (' [' + ', '.join(flags) + ']') if flags else ''
    print(f'[*] 正在分析: {csv_path}{extra}')
    if args.panel == 'all':
        load_and_plot(
            csv_path,
            merge_logical_blocks=args.merge_logical_blocks,
            x_labels_numeric=args.x_labels_numeric,
        )
    else:
        plot_op_cycles_only(
            csv_path,
            output_path=args.out,
            dpi=args.dpi,
            merge_logical_blocks=args.merge_logical_blocks,
            x_labels_numeric=args.x_labels_numeric,
        )


if __name__ == '__main__':
    main()
