"""
Generate fig_4p5_eca_gate_internals.pdf  -- side-by-side zoom-in showing the
internal operation chain of (a) Efficient Channel Attention and (b) the global
broadcast gate. Replaces the two display equations in 4.3.

Style follows make_thesis_ch4p3_figures.py for visual coherence.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = r"D:\BaiduNetdiskWorkspace\Leuven\AI_Master_Thesis\thesis writing\Figure"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_OP        = "#D9E2EC"  # operation box (light steel blue)
C_OP_EDGE   = "#2F4858"
C_TENSOR    = "#FAF3DD"  # tensor annotation box (warm cream)
C_TEN_EDGE  = "#B7990D"
C_ACCENT    = "#C44E52"  # mul / sigmoid accent
C_ARROW     = "#444444"

fig, (ax_e, ax_g) = plt.subplots(
    1, 2, figsize=(13.5, 5.6),
    gridspec_kw=dict(wspace=0.10),
)

OP_W = 4.0
OP_H = 0.75
TENSOR_W = 3.2
TENSOR_H = 0.55

def opbox(ax, x, y, label):
    rect = FancyBboxPatch(
        (x - OP_W/2, y - OP_H/2), OP_W, OP_H,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.0, edgecolor=C_OP_EDGE, facecolor=C_OP,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5)
    return (x, y)

def tensor_label(ax, x, y, label):
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9.5, fontstyle="italic", color="#444444")
    return (x, y)

def vert_arrow(ax, p_from, p_to, color=C_ARROW):
    ax.annotate(
        "", xy=(p_to[0], p_to[1] + OP_H/2 + 0.02),
        xytext=(p_from[0], p_from[1] - OP_H/2 - 0.02),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=color),
    )

def vert_arrow_label(ax, p_from, p_to, label, dx=0.10):
    ax.annotate(
        "", xy=(p_to[0], p_to[1] + 0.32),
        xytext=(p_from[0], p_from[1] - 0.32),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=C_ARROW),
    )
    ax.text(p_from[0] + dx, (p_from[1] + p_to[1]) / 2, label,
            ha="left", va="center", fontsize=9, color="#666666",
            fontstyle="italic")


# ========================================================================
# Panel (a): ECA  -- single-tensor chain  X -> GAP -> 1D conv -> sigmoid -> *X
# ========================================================================
ax_e.set_xlim(-0.6, 8.4)
ax_e.set_ylim(-0.4, 9.0)
ax_e.set_axis_off()
ax_e.set_aspect("equal")
ax_e.set_title("(a) Efficient Channel Attention (ECA)",
               fontsize=11, loc="left", pad=8, fontweight="bold")

cx = 3.0
side_x = cx + 4.0
ys = [8.0, 6.6, 5.2, 3.8, 2.4, 1.0]
# y0 input tensor
tensor_label(ax_e, cx, ys[0], r"input tensor $X$  ($H \times W \times C$)")
# y1 GAP op
op_gap_e = opbox(ax_e, cx, ys[1], "global average pool")
# y2 channel descriptor
tensor_label(ax_e, cx, ys[2], r"channel descriptor $y$  ($1 \times 1 \times C$)")
# y3 1D conv op
op_1d_e = opbox(ax_e, cx, ys[3], r"1D conv, kernel $k = 3$")
# y4 sigmoid op
op_sig_e = opbox(ax_e, cx, ys[4], "sigmoid")
# y5 multiply op
op_mul_e = opbox(ax_e, cx, ys[5], r"element-wise multiply")

# main chain arrows
vert_arrow(ax_e, (cx, ys[0]), op_gap_e)
vert_arrow(ax_e, op_gap_e, (cx, ys[2]))
vert_arrow(ax_e, (cx, ys[2]), op_1d_e)
vert_arrow(ax_e, op_1d_e, op_sig_e)
vert_arrow(ax_e, op_sig_e, op_mul_e)

# side path: input X flows directly to the multiply node (broadcast)
ax_e.add_patch(FancyArrowPatch(
    posA=(cx + OP_W / 2 + 0.25, ys[0]),
    posB=(cx + OP_W / 2 + 0.05, ys[5]),
    connectionstyle=f"arc3,rad=-0.45",
    arrowstyle="->", mutation_scale=10, color=C_ACCENT, lw=1.2,
))
ax_e.text(side_x, (ys[0] + ys[5]) / 2 + 0.2,
          r"broadcast $X$",
          ha="center", va="center", fontsize=9.5, color=C_ACCENT,
          fontstyle="italic",
          bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none"))

# output below mul
ax_e.annotate(
    "", xy=(cx, ys[5] - OP_H/2 - 0.55), xytext=(cx, ys[5] - OP_H/2 - 0.02),
    arrowprops=dict(arrowstyle="->", lw=1.1, color=C_ARROW),
)
ax_e.text(cx, ys[5] - OP_H/2 - 0.95, r"output $X'$  ($H \times W \times C$)",
          ha="center", va="center", fontsize=9.5, fontstyle="italic",
          color="#444444")


# ========================================================================
# Panel (b): Global broadcast gate -- two-input
#   context C -> GAP -> 1x1 conv -> sigmoid -> broadcast multiply with T
# ========================================================================
ax_g.set_xlim(-0.6, 8.4)
ax_g.set_ylim(-0.4, 9.0)
ax_g.set_axis_off()
ax_g.set_aspect("equal")
ax_g.set_title("(b) Global Broadcast Gate",
               fontsize=11, loc="left", pad=8, fontweight="bold")

cx2 = 3.0
target_x = 6.5
# context chain
tensor_label(ax_g, cx2, ys[0], r"context $C$  ($H_c \times W_c \times C_c$)")
op_gap_g = opbox(ax_g, cx2, ys[1], "global average pool")
tensor_label(ax_g, cx2, ys[2], r"context vector $c$  ($1 \times 1 \times C_c$)")
op_1x1_g = opbox(ax_g, cx2, ys[3], r"$1 \times 1$ conv, $C_c \to C_t$")
op_sig_g = opbox(ax_g, cx2, ys[4], "sigmoid")
op_mul_g = opbox(ax_g, cx2, ys[5], r"broadcast multiply")

# main chain arrows
vert_arrow(ax_g, (cx2, ys[0]), op_gap_g)
vert_arrow(ax_g, op_gap_g, (cx2, ys[2]))
vert_arrow(ax_g, (cx2, ys[2]), op_1x1_g)
vert_arrow(ax_g, op_1x1_g, op_sig_g)
vert_arrow(ax_g, op_sig_g, op_mul_g)

# target T enters the multiply from the right at the same y-level
tensor_label(ax_g, target_x + 0.4, ys[5] + 1.0,
             r"target $T$  ($H_t \times W_t \times C_t$)")
ax_g.add_patch(FancyArrowPatch(
    posA=(target_x + 0.4, ys[5] + 0.7),
    posB=(cx2 + OP_W/2 + 0.05, ys[5]),
    connectionstyle="arc3,rad=-0.30",
    arrowstyle="->", mutation_scale=10, color=C_ACCENT, lw=1.2,
))

# output below mul
ax_g.annotate(
    "", xy=(cx2, ys[5] - OP_H/2 - 0.55), xytext=(cx2, ys[5] - OP_H/2 - 0.02),
    arrowprops=dict(arrowstyle="->", lw=1.1, color=C_ARROW),
)
ax_g.text(cx2, ys[5] - OP_H/2 - 0.95,
          r"output $T'$  ($H_t \times W_t \times C_t$)",
          ha="center", va="center", fontsize=9.5, fontstyle="italic",
          color="#444444")


out_path = os.path.join(OUT_DIR, "fig_4p5_eca_gate_internals.pdf")
plt.savefig(out_path)
plt.close()
print(f"wrote {out_path}")
