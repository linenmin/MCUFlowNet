"""
Generate figure for Ch3.6 Compute Bottleneck of the master's thesis.

Outputs:
  - fig_3p4_per_layer_cycles.pdf — per-layer share of total NPU cycles for baseline EdgeFlowNet at 156x208

Input:
  - output_benchmark_all_multiscale/ResNet_Transpose/ResNet_Transpose_156x208_per-layer.csv
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = r"D:\Dataset\MCUFlowNet\EdgeFlowNet\sramTest"
OUT_DIR      = r"D:\BaiduNetdiskWorkspace\Leuven\AI_Master_Thesis\thesis writing\Figure"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colour-blind-safe palette (same as ch3.5 figures for visual continuity)
COLOR_CONV       = "#4C72B0"
COLOR_TCONV_PEAK = "#C44E52"
COLOR_ADD        = "#9B9B9B"
COLOR_RESIZE     = "#55A868"
COLOR_SLICE      = "#CCB974"

OP_COLOR = {
    "CONV_2D":          COLOR_CONV,
    "TRANSPOSE_CONV":   COLOR_TCONV_PEAK,
    "ADD":              COLOR_ADD,
    "RESIZE_BILINEAR":  COLOR_RESIZE,
    "STRIDED_SLICE":    COLOR_SLICE,
}

csv_path = os.path.join(
    PROJECT_ROOT,
    "output_benchmark_all_multiscale", "ResNet_Transpose",
    "ResNet_Transpose_156x208_per-layer.csv",
)
df = pd.read_csv(csv_path)

# The 'Network%' column appears twice in the header (cycles % then macs %).
# pandas renames the duplicate to "Network%.1"; the first occurrence is cycles share.
cycles_pct = pd.to_numeric(df["Network%"], errors="coerce")
ops        = df["TFLite_operator"].tolist()
colors     = [OP_COLOR.get(op, "#777777") for op in ops]

# top-2 ops for annotation
top_two = cycles_pct.nlargest(2)
top_sum = float(top_two.sum())

fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.bar(range(len(df)), cycles_pct, color=colors, edgecolor="none", width=0.85)

# annotate top two; the absolute peak also gets its kernel/op tag
peak_idx_val, peak_pct_val = top_two.index[0], top_two.iloc[0]
second_idx_val, second_pct_val = top_two.index[1], top_two.iloc[1]

ax.annotate(
    f"{peak_pct_val:.1f}%\n$7\\times7$ Conv2D",
    xy=(peak_idx_val, peak_pct_val),
    xytext=(peak_idx_val, peak_pct_val + 2.2),
    ha="center", fontsize=9,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
)
ax.annotate(
    f"{second_pct_val:.1f}%",
    xy=(second_idx_val, second_pct_val),
    xytext=(second_idx_val, second_pct_val + 1.4),
    ha="center", fontsize=9,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
)

ax.set_xlabel("Operator (execution order)")
ax.set_ylabel("Share of total NPU cycles (\\%)")
ax.set_xticks([])
ax.set_ylim(0, cycles_pct.max() * 1.28)

handles = [
    mpatches.Patch(color=COLOR_CONV,        label="Conv2D"),
    mpatches.Patch(color=COLOR_TCONV_PEAK,  label="Transposed Conv"),
    mpatches.Patch(color=COLOR_RESIZE,      label="Bilinear Resize"),
    mpatches.Patch(color=COLOR_ADD,         label="Add"),
    mpatches.Patch(color=COLOR_SLICE,       label="Strided Slice"),
]
ax.legend(
    handles=handles,
    loc="upper center", bbox_to_anchor=(0.5, -0.10),
    frameon=False, ncol=5,
    handlelength=1.2, handleheight=0.9, columnspacing=1.6,
)

out_path = os.path.join(OUT_DIR, "fig_3p4_per_layer_cycles.pdf")
plt.tight_layout()
plt.savefig(out_path)
plt.close()
print(f"wrote {out_path}")
print(f"top-2 cycle share: {top_sum:.2f}%")
for idx, val in top_two.items():
    name = df.loc[idx, "Name"].split(";")[0].split("/")[-1]
    print(f"  - idx={idx} op={df.loc[idx, 'TFLite_operator']} cycles%={val:.2f}  short_name={name}")
