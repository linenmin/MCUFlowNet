"""
Generate the L1b backbone schematic for Ch4.3.

Outputs:
  - fig_4p4_eca_gate_schematic.pdf  -- simplified architecture diagram showing
                                       ECA placement at the encoder bottleneck
                                       and the 1/4 global broadcast gate

Faithful to MultiScaleResNet_supernet_v3.py:
  Stem (E0 -> E1, /4) -> Encoder (EB0 /4, Down1+EB1 /8, Down2+DB0 /16)
  -> ECA at DB0 output (/16 bottleneck), bottleneck_context = ECA output
  -> Decoder (Up1+DB1 /8, Up2 /4)
  -> Global gate at /4 feature (target = net_high after Up2, context = bottleneck)
  -> Heads (H0 /4, H1 /2, H2 /1) all read from the gated /4 feature
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = r"D:\BaiduNetdiskWorkspace\Leuven\AI_Master_Thesis\thesis writing\Figure"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# colour palette (cool blues for encoder, warm peaches for decoder/head,
# highlight yellow for the ECA bottleneck)
C_STEM      = "#C8E6C9"   # mint green
C_ENC       = "#9FBFD6"   # steel blue
C_BOTTLE    = "#F6E27A"   # ECA highlight (yellow)
C_DEC       = "#D5C5E0"   # light purple
C_HEAD      = "#F5C9A8"   # coral
C_EDGE      = "#333333"
C_GATE      = "#C44E52"   # gate arrow / label colour (red, matches A0 in F4.3)

fig, ax = plt.subplots(figsize=(12.5, 4.6))
ax.set_xlim(0, 26)
ax.set_ylim(-2.0, 6.0)
ax.set_axis_off()
ax.set_aspect("equal")

BOX_W = 2.2
BOX_H = 0.9
GROUP_PAD = 0.8


def block(x, y, label, color, **kw):
    """Draw a rounded box with a centred single-line label."""
    rect = FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.0, edgecolor=C_EDGE, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=9)
    return (x, y)


def group_label(x_left, x_right, y, text, fontsize=10, color="#333333"):
    ax.text(
        (x_left + x_right) / 2, y, text,
        ha="center", va="top", fontsize=fontsize, fontweight="bold", color=color,
    )


def fwd_arrow(p1, p2):
    ax.annotate(
        "", xy=(p2[0] - BOX_W / 2, p2[1]),
        xytext=(p1[0] + BOX_W / 2, p1[1]),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=C_EDGE),
    )


main_y = 3.0


# ---- Stem ----
x = 1.5
e0 = block(x, main_y, "E0  (/2)", C_STEM); x += BOX_W + 0.5
e1 = block(x, main_y, "E1  (/4)", C_STEM)
group_label(0.5, x + BOX_W / 2, main_y - BOX_H / 2 - 0.35, "Stem")
fwd_arrow(e0, e1)


# ---- Encoder ----
x += BOX_W + GROUP_PAD + 0.3
eb0 = block(x, main_y, "EB0  (/4)", C_ENC); x += BOX_W + 0.5
eb1 = block(x, main_y, "EB1  (/8)", C_ENC)
group_label(eb0[0] - BOX_W / 2 - 0.2, x + BOX_W / 2, main_y - BOX_H / 2 - 0.35, "Encoder")
fwd_arrow(e1, eb0)
fwd_arrow(eb0, eb1)


# ---- Bottleneck (DB0 + ECA) ----
x += BOX_W + GROUP_PAD + 0.3
bot = block(x, main_y, "DB0  (/16)", C_BOTTLE)
ax.text(x, main_y + BOX_H / 2 + 0.18, "+ ECA",
        ha="center", va="bottom", fontsize=9, color=C_EDGE,
        fontweight="bold")
group_label(x - BOX_W / 2 - 0.1, x + BOX_W / 2 + 0.1,
            main_y - BOX_H / 2 - 0.35, "Bottleneck")
fwd_arrow(eb1, bot)


# ---- Decoder ----
x += BOX_W + GROUP_PAD + 0.3
db1 = block(x, main_y, "Up1+DB1  (/8)", C_DEC); x += BOX_W + 0.5
up2 = block(x, main_y, "Up2  (/4)", C_DEC)
group_label(db1[0] - BOX_W / 2 - 0.1, x + BOX_W / 2, main_y - BOX_H / 2 - 0.35, "Decoder")
fwd_arrow(bot, db1)
fwd_arrow(db1, up2)


# ---- Head branches ----
x += BOX_W + GROUP_PAD + 0.3
h0_y = main_y + 1.4
h1_y = main_y
h2_y = main_y - 1.4
h0 = block(x, h0_y, "H0  (out /4)",  C_HEAD)
h1 = block(x, h1_y, "H1  (out /2)",  C_HEAD)
h2 = block(x, h2_y, "H2  (out /1)",  C_HEAD)
group_label(x - BOX_W / 2 - 0.1, x + BOX_W / 2 + 0.1,
            h2_y - BOX_H / 2 - 0.35, "Heads")

# from up2 split to three heads
fwd_arrow(up2, h0)
fwd_arrow(up2, h1)
fwd_arrow(up2, h2)


# ---- Global broadcast gate: skip arc passing BELOW the main row, from the
# bottleneck output to the gated /4 feature (target = net_high after Up2).
gate_arc = FancyArrowPatch(
    posA=(bot[0], bot[1] - BOX_H / 2 - 0.05),
    posB=(up2[0], up2[1] - BOX_H / 2 - 0.05),
    connectionstyle="arc3,rad=0.32",
    arrowstyle="->", mutation_scale=14,
    color=C_GATE, lw=1.4,
)
ax.add_patch(gate_arc)
# label sits in the centre of the arc dip with a white background so it
# masks any residual crossing of the curve.
ax.text((bot[0] + up2[0]) / 2, main_y - BOX_H / 2 - 1.4,
        "1/4 Global Broadcast Gate",
        ha="center", va="center", fontsize=9,
        color=C_GATE, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="none"))

# small legend strip at the top
legend_handles = [
    mpatches.Patch(color=C_STEM,   label="Stem"),
    mpatches.Patch(color=C_ENC,    label="Encoder"),
    mpatches.Patch(color=C_BOTTLE, label="Bottleneck + ECA"),
    mpatches.Patch(color=C_DEC,    label="Decoder"),
    mpatches.Patch(color=C_HEAD,   label="Head"),
]
ax.legend(
    handles=legend_handles,
    loc="upper center", bbox_to_anchor=(0.5, 1.00),
    frameon=False, ncol=5, fontsize=9,
    handlelength=1.2, handleheight=0.9, columnspacing=1.4,
)

out_path = os.path.join(OUT_DIR, "fig_4p4_eca_gate_schematic.pdf")
plt.savefig(out_path)
plt.close()
print(f"wrote {out_path}")
