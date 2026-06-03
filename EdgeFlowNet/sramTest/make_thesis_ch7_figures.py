"""
Generate fig_7p1_epe_fps_compare.pdf for Ch7 (matched-hardware comparison).

Same-hardware (Grove Vision AI V2, Ethos-U55, 172x224, Vela) comparison of the
EdgeFlowNet baseline against the two retrained subnets in the (FPS, Sintel Final
EPE) plane. Data from Table 7.2 (tab:matched_hw):
  EdgeFlowNet baseline : 5.3 FPS, 6.31 EPE
  MCUFlowNet-L         : 5.3 FPS, 4.89 EPE  (-22% EPE at the same FPS)
  MCUFlowNet-S         : 9.1 FPS, 5.58 EPE  (-12% EPE, +72% FPS)
"""
from pathlib import Path
import os
import matplotlib.pyplot as plt

OUT = str(Path(__file__).resolve().parents[2] / "figures" / "fig_7p1_epe_fps_compare.pdf")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_BASE = "#555555"
C_L    = "#4C72B0"
C_S    = "#55A868"

base = (5.3, 6.31)
L    = (5.3, 4.89)
S    = (9.1, 5.58)

fig, ax = plt.subplots(figsize=(7.2, 4.6))

# improvement arrows (drawn under the markers)
ax.annotate("", xy=L, xytext=base,
            arrowprops=dict(arrowstyle="-|>", color=C_L, ls="--", lw=1.5, shrinkA=7, shrinkB=7))
ax.annotate("", xy=S, xytext=base,
            arrowprops=dict(arrowstyle="-|>", color=C_S, ls="--", lw=1.5, shrinkA=7, shrinkB=7))

# markers
ax.scatter(*base, marker="X", s=170, color=C_BASE, zorder=5, edgecolors="white", linewidths=0.8)
ax.scatter(*L,    marker="o", s=130, color=C_L,    zorder=5, edgecolors="white", linewidths=0.8)
ax.scatter(*S,    marker="D", s=120, color=C_S,    zorder=5, edgecolors="white", linewidths=0.8)

# point labels (placed in clear quadrants, no overlap with arrows)
ax.annotate("EdgeFlowNet baseline\n(5.3 FPS, 6.31 EPE)", base, xytext=(5.62, 6.55),
            ha="left", va="center", fontsize=9, color="#222222")
ax.annotate("MCUFlowNet-L\n(5.3 FPS, 4.89 EPE)", L, xytext=(5.62, 4.83),
            ha="left", va="center", fontsize=9, color=C_L, fontweight="bold")
ax.annotate("MCUFlowNet-S\n(9.1 FPS, 5.58 EPE)", S, xytext=(9.05, 5.30),
            ha="center", va="top", fontsize=9, color=C_S, fontweight="bold")

# delta labels along the arrows
ax.text(5.45, 5.62, "$-22\\%$ EPE", ha="left", va="center", fontsize=9, color=C_L, style="italic")
ax.text(7.0, 6.18, "$-12\\%$ EPE,  $+72\\%$ FPS", ha="center", va="center", fontsize=9, color=C_S, style="italic")

# "better" direction (clear upper-right corner, points down-right)
ax.annotate("", xy=(9.9, 6.05), xytext=(8.9, 6.55),
            arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=2.0))
ax.text(8.85, 6.62, "better", ha="left", va="bottom", fontsize=10, color="#C44E52", fontweight="bold")

ax.set_xlabel("Inference rate on Grove Vision AI V2 (FPS, Vela @ $172\\times224$)")
ax.set_ylabel("Sintel Final EPE (lower is better)")
ax.set_xlim(4.6, 10.5)
ax.set_ylim(4.5, 6.85)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.tight_layout()
plt.savefig(OUT)
plt.close()
print("wrote", OUT)
