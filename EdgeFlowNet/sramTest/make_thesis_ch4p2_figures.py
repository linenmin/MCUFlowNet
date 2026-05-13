"""
Generate figures for Ch4.2 (L1a Bilinear Upsampling) of the master's thesis.

Outputs (academic-style PDF, dropped into ../Figure/):
  - fig_4p2_per_layer_sram_compare.pdf  -- per-layer SRAM A0 (transposed conv) vs A1 (bilinear) at 156x208
  - fig_4p3_sram_fps_scan.pdf           -- multi-res SRAM peak + FPS, two series (A0 / A1)

Style follows make_thesis_ch3p5_figures.py for cross-chapter coherence.

Inputs (already on disk):
  - output_benchmark_all_multiscale/ResNet_Transpose/ResNet_Transpose_156x208_per-layer.csv  (A0)
  - output_benchmark_all_multiscale/ResNet_Bilinear/ResNet_Bilinear_156x208_per-layer.csv    (A1)
  - benchmark_comparison.csv  (10 resolutions x {Original, Bilinear})
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = r"D:\Dataset\MCUFlowNet\EdgeFlowNet\sramTest"
OUT_DIR      = r"D:\BaiduNetdiskWorkspace\Leuven\AI_Master_Thesis\thesis writing\Figure"

# ---------- academic-paper style (matches Ch3) ----------
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

ARENA_CAP_KIB = 1.4 * 1024
ARENA_CAP_MB  = 1.4
CHIP_TOTAL_MB = 2.4

# ====================================================================
# Figure 4.2 : per-layer SRAM A0 vs A1 at 156x208 (two-row subplot)
# ====================================================================
csv_a0 = os.path.join(
    PROJECT_ROOT, "output_benchmark_all_multiscale", "ResNet_Transpose",
    "ResNet_Transpose_156x208_per-layer.csv",
)
csv_a1 = os.path.join(
    PROJECT_ROOT, "output_benchmark_all_multiscale", "ResNet_Bilinear",
    "ResNet_Bilinear_156x208_per-layer.csv",
)
df_a0 = pd.read_csv(csv_a0)
df_a1 = pd.read_csv(csv_a1)

a0_kib = df_a0["SRAM Usage"].astype(float) / 1024.0
a1_kib = df_a1["SRAM Usage"].astype(float) / 1024.0
a0_colors = [OP_COLOR.get(op, "#777777") for op in df_a0["TFLite_operator"]]
a1_colors = [OP_COLOR.get(op, "#777777") for op in df_a1["TFLite_operator"]]

a0_peak_idx = a0_kib.idxmax(); a0_peak_val = a0_kib.iloc[a0_peak_idx]
a1_peak_idx = a1_kib.idxmax(); a1_peak_val = a1_kib.iloc[a1_peak_idx]

# shared y-axis upper bound based on A0 peak (always larger)
y_top = max(a0_peak_val, ARENA_CAP_KIB) * 1.18

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7.0, 5.4), sharex=False,
    gridspec_kw=dict(height_ratios=[1, 1], hspace=0.45),
)

# --- top: A0 (Transposed conv decoder) ---
ax_top.bar(range(len(df_a0)), a0_kib, color=a0_colors, edgecolor="none", width=0.85)
ax_top.axhline(ARENA_CAP_KIB, color="black", linestyle=":", linewidth=0.9)
ax_top.text(-0.5, ARENA_CAP_KIB + 25, "arena cap (1.4 MB)", ha="left", va="bottom", fontsize=8)
ax_top.annotate(
    f"peak {a0_peak_val:.0f} KiB",
    xy=(a0_peak_idx, a0_peak_val),
    xytext=(a0_peak_idx, a0_peak_val + 140),
    ha="center", fontsize=9,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
)
ax_top.set_title("A0 baseline: transposed-convolution decoder", loc="left", fontsize=10)
ax_top.set_ylabel("SRAM usage (KiB)")
ax_top.set_xticks([])
ax_top.set_ylim(0, y_top)

# --- bottom: A1 (Bilinear decoder) ---
ax_bot.bar(range(len(df_a1)), a1_kib, color=a1_colors, edgecolor="none", width=0.85)
ax_bot.axhline(ARENA_CAP_KIB, color="black", linestyle=":", linewidth=0.9)
ax_bot.text(-0.5, ARENA_CAP_KIB + 25, "arena cap (1.4 MB)", ha="left", va="bottom", fontsize=8)
ax_bot.annotate(
    f"peak {a1_peak_val:.0f} KiB",
    xy=(a1_peak_idx, a1_peak_val),
    xytext=(a1_peak_idx, a1_peak_val + 140),
    ha="center", fontsize=9,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
)
ax_bot.set_title("A1 redesign: bilinear + standard convolution decoder", loc="left", fontsize=10)
ax_bot.set_xlabel("Operator (execution order)")
ax_bot.set_ylabel("SRAM usage (KiB)")
ax_bot.set_xticks([])
ax_bot.set_ylim(0, y_top)

# shared legend below
handles = [
    mpatches.Patch(color=COLOR_CONV,        label="Conv2D"),
    mpatches.Patch(color=COLOR_TCONV_PEAK,  label="Transposed Conv"),
    mpatches.Patch(color=COLOR_RESIZE,      label="Bilinear Resize"),
    mpatches.Patch(color=COLOR_ADD,         label="Add"),
    mpatches.Patch(color=COLOR_SLICE,       label="Strided Slice"),
]
fig.legend(
    handles=handles,
    loc="lower center", bbox_to_anchor=(0.5, -0.02),
    frameon=False, ncol=5,
    handlelength=1.2, handleheight=0.9, columnspacing=1.6,
)

out_path = os.path.join(OUT_DIR, "fig_4p2_per_layer_sram_compare.pdf")
plt.savefig(out_path)
plt.close()
print(f"wrote {out_path}")

# ====================================================================
# Figure 4.3 : multi-resolution SRAM peak + delta + FPS, A0 vs A1
# Filtered to deployment-relevant range (drop 72x96 and >= 300x400);
# adds 172x224 measurement and a delta SRAM panel.
# ====================================================================
scan_path = os.path.join(PROJECT_ROOT, "benchmark_comparison.csv")
df_scan = pd.read_csv(scan_path)

df_scan["Pixels"] = df_scan["Resolution"].apply(
    lambda r: int(r.split("x")[0]) * int(r.split("x")[1])
)

# keep deployment-relevant window: 96x128 .. 264x352 (drop 72x96 + 300x400 onward)
KEEP = {"96x128", "120x160", "156x208", "172x224", "192x256", "228x304", "264x352"}
df_scan = df_scan[df_scan["Resolution"].isin(KEEP)]

df_orig = df_scan[df_scan["Type"] == "Original (Transpose)"].sort_values("Pixels").reset_index(drop=True)
df_bili = df_scan[df_scan["Type"] == "Bilinear (Resize+Conv)"].sort_values("Pixels").reset_index(drop=True)

assert list(df_orig["Resolution"]) == list(df_bili["Resolution"]), "mismatched resolutions"
resolutions = list(df_orig["Resolution"])

fig, (axS, axF) = plt.subplots(
    2, 1, figsize=(7.0, 6.0), sharex=True,
    gridspec_kw=dict(hspace=0.18),
)

# --- top: SRAM peak vs resolution (with inline per-point labels) ---
axS.plot(
    resolutions, df_orig["SRAM (MB)"],
    marker="o", color=COLOR_TCONV_PEAK,
    linewidth=1.8, markersize=6, label="Transposed conv (A0)",
)
axS.plot(
    resolutions, df_bili["SRAM (MB)"],
    marker="s", color=COLOR_RESIZE,
    linewidth=1.8, markersize=6, label="Bilinear (A1)",
)
axS.axhline(ARENA_CAP_MB,  color="black",  linestyle=":", linewidth=0.9)
axS.axhline(CHIP_TOTAL_MB, color="#6B6B6B", linestyle=":", linewidth=0.9)
axS.text(0.05, ARENA_CAP_MB + 0.06, "arena cap (1.4 MB)",
         fontsize=8, ha="left", va="bottom")
axS.text(0.05, CHIP_TOTAL_MB + 0.06, "chip total SRAM (2.4 MB)",
         fontsize=8, ha="left", va="bottom", color="#6B6B6B")

# inline labels: A0 above each red point, A1 below each green point
for i, v in enumerate(df_orig["SRAM (MB)"]):
    axS.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 7), textcoords="offset points",
                 fontsize=7.5, ha="center", color=COLOR_TCONV_PEAK)
for i, v in enumerate(df_bili["SRAM (MB)"]):
    axS.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, -12), textcoords="offset points",
                 fontsize=7.5, ha="center", color=COLOR_RESIZE)

axS.set_ylabel("SRAM peak (MB)")
axS.legend(loc="upper left", frameon=False)
axS.set_ylim(0, max(df_orig["SRAM (MB)"].max(), CHIP_TOTAL_MB) * 1.18)

# --- bottom: FPS vs resolution (with inline per-point labels) ---
axF.plot(
    resolutions, df_orig["FPS"],
    marker="o", color=COLOR_TCONV_PEAK,
    linewidth=1.8, markersize=6, label="Transposed conv (A0)",
)
axF.plot(
    resolutions, df_bili["FPS"],
    marker="s", color=COLOR_RESIZE,
    linewidth=1.8, markersize=6, label="Bilinear (A1)",
)
# FPS panel inline labels: at each resolution, place the larger value above
# its point and the smaller value below the other point. This always separates
# the two labels by the inter-curve gap plus the label offsets.
for i, (vo, vb) in enumerate(zip(df_orig["FPS"], df_bili["FPS"])):
    if vo >= vb:
        axF.annotate(f"{vo:.1f}", xy=(i, vo), xytext=(0, 7), textcoords="offset points",
                     fontsize=7.5, ha="center", color=COLOR_TCONV_PEAK)
        axF.annotate(f"{vb:.1f}", xy=(i, vb), xytext=(0, -12), textcoords="offset points",
                     fontsize=7.5, ha="center", color=COLOR_RESIZE)
    else:
        axF.annotate(f"{vb:.1f}", xy=(i, vb), xytext=(0, 7), textcoords="offset points",
                     fontsize=7.5, ha="center", color=COLOR_RESIZE)
        axF.annotate(f"{vo:.1f}", xy=(i, vo), xytext=(0, -12), textcoords="offset points",
                     fontsize=7.5, ha="center", color=COLOR_TCONV_PEAK)

axF.set_xlabel("Input resolution")
axF.set_ylabel("Inference FPS (Vela predicted)")
axF.legend(loc="upper right", frameon=False)
axF.tick_params(axis="x", rotation=30)
axF.set_ylim(min(df_orig["FPS"].min(), df_bili["FPS"].min()) * 0.5,
             max(df_orig["FPS"].max(), df_bili["FPS"].max()) * 1.15)

out_path = os.path.join(OUT_DIR, "fig_4p3_sram_fps_scan.pdf")
plt.savefig(out_path)
plt.close()
print(f"wrote {out_path}")
