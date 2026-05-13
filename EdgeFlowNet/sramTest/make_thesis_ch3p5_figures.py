"""
Generate figures for Ch3.5 Memory Bottleneck of the master's thesis.

Outputs (academic-style PDF, dropped into ../Figure/):
  - fig_3p2_per_layer_sram.pdf       — per-layer SRAM of baseline EdgeFlowNet at 156x208 (multi-scale)
  - fig_3p3_sram_resolution_scan.pdf — baseline SRAM peak vs input resolution

Inputs (already on disk):
  - output_benchmark_all_multiscale/ResNet_Transpose/ResNet_Transpose_156x208_per-layer.csv
  - benchmark_comparison.csv  (filter: Original (Transpose))
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = r"D:\Dataset\MCUFlowNet\EdgeFlowNet\sramTest"
OUT_DIR      = r"D:\BaiduNetdiskWorkspace\Leuven\AI_Master_Thesis\thesis writing\Figure"

# ---------- academic-paper style ----------
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
    "pdf.fonttype": 42,        # embed fonts properly
    "ps.fonttype": 42,
})

# Colour-blind-safe palette
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

# ====================================================================
# Figure 3.2 : per-layer SRAM at 156x208, baseline (multi-scale)
# ====================================================================
csv_path = os.path.join(
    PROJECT_ROOT,
    "output_benchmark_all_multiscale", "ResNet_Transpose",
    "ResNet_Transpose_156x208_per-layer.csv",
)
df = pd.read_csv(csv_path)

sram_kib = df["SRAM Usage"].astype(float) / 1024.0
ops      = df["TFLite_operator"].tolist()
colors   = [OP_COLOR.get(op, "#777777") for op in ops]

peak_idx = sram_kib.idxmax()
peak_val = sram_kib.iloc[peak_idx]

fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.bar(range(len(df)), sram_kib, color=colors, edgecolor="none", width=0.85)

# arena cap line
ARENA_CAP_KIB = 1.4 * 1024
ax.axhline(ARENA_CAP_KIB, color="black", linestyle=":", linewidth=0.9)
ax.text(-0.5, ARENA_CAP_KIB + 25,
        "arena cap (1.4 MB)",
        ha="left", va="bottom", fontsize=8)

# annotate peak
ax.annotate(
    f"peak {peak_val:.0f} KiB",
    xy=(peak_idx, peak_val),
    xytext=(peak_idx, peak_val + 140),
    ha="center", fontsize=9,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
)

ax.set_xlabel("Operator (execution order)")
ax.set_ylabel("SRAM usage (KiB)")
ax.set_xticks([])
ax.set_ylim(0, peak_val * 1.18)

# legend BELOW the plot in a single row, avoids collision with peak annotation
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

out_path = os.path.join(OUT_DIR, "fig_3p2_per_layer_sram.pdf")
plt.tight_layout()
plt.savefig(out_path)
plt.close()
print(f"wrote {out_path}")

# ====================================================================
# Figure 3.3 : SRAM peak vs input resolution, baseline only
# ====================================================================
scan_path = os.path.join(PROJECT_ROOT, "benchmark_comparison.csv")
df_scan = pd.read_csv(scan_path)
df_orig = df_scan[df_scan["Type"] == "Original (Transpose)"].copy()
df_orig["Pixels"] = df_orig["Resolution"].apply(
    lambda r: int(r.split("x")[0]) * int(r.split("x")[1])
)
df_orig = df_orig.sort_values("Pixels").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.plot(df_orig["Resolution"], df_orig["SRAM (MB)"],
        marker="o", color=COLOR_TCONV_PEAK,
        linewidth=1.8, markersize=6, markerfacecolor=COLOR_TCONV_PEAK)

ARENA_CAP_MB = 1.4
CHIP_TOTAL_MB = 2.4
ax.axhline(ARENA_CAP_MB,  color="black", linestyle=":", linewidth=0.9)
ax.axhline(CHIP_TOTAL_MB, color="#6B6B6B", linestyle=":", linewidth=0.9)
# left-aligned labels placed just above each cap line; at small resolutions
# the data curve is well below 1.4 MB, leaving room on the left for labels.
ax.text(-0.4, ARENA_CAP_MB + 0.10, "arena cap (1.4 MB)",
        fontsize=8, ha="left", va="bottom")
ax.text(-0.4, CHIP_TOTAL_MB + 0.10, "chip total SRAM (2.4 MB)",
        fontsize=8, ha="left", va="bottom", color="#6B6B6B")

# annotate max deployable
dep = df_orig[df_orig["Resolution"] == "156x208"].iloc[0]
dep_idx = df_orig.index[df_orig["Resolution"] == "156x208"][0]
ax.annotate(
    "max deployable\n156×208 (1.40 MB)",
    xy=(dep_idx, dep["SRAM (MB)"]),
    xytext=(dep_idx, 4.2),
    fontsize=9, ha="center",
    arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
)

ax.set_xlabel("Input resolution")
ax.set_ylabel("SRAM peak (MB)")
ax.tick_params(axis="x", rotation=30)
ax.set_ylim(0, df_orig["SRAM (MB)"].max() * 1.1)

out_path = os.path.join(OUT_DIR, "fig_3p3_sram_resolution_scan.pdf")
plt.tight_layout()
plt.savefig(out_path)
plt.close()
print(f"wrote {out_path}")
