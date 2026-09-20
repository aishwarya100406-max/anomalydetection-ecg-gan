"""
Generates Figure 2: Hybrid quantum-classical GAN framework architecture,
as a publication-quality PNG for the IEEE Access manuscript.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG_W, FIG_H = 7.2, 6.6
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

C_DATA = "#DCE6F1"   # light blue - data stages
C_PROC = "#D9EAD3"   # light green - processing
C_QUANT = "#E4D9F2"  # light purple - quantum component
C_DISC = "#FCE4D6"   # light orange - discriminator
C_OUT = "#EFEFEF"    # grey - output
EDGE = "#333333"


def box(x, y, w, h, title, subtitle=None, fc=C_DATA, fs=9.5, sub_fs=8):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.6,rounding_size=1.2",
                       linewidth=1.0, edgecolor=EDGE, facecolor=fc, zorder=3)
    ax.add_patch(b)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center",
                fontsize=sub_fs, style="italic", color="#444444", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)


def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, linestyle=ls,
                        mutation_scale=13, linewidth=1.2, color=EDGE, zorder=2)
    ax.add_patch(a)


# ---- Stage 1: data acquisition ----
box(22, 88, 56, 9, "ECG Data Acquisition",
    "MIT-BIH (training / same-dataset test) + PTBDB (cross-dataset test)",
    fc=C_DATA, sub_fs=7.3)
arrow(50, 87.4, 50, 82.6)

# ---- Stage 2: preprocessing ----
box(22, 73, 56, 9, "Preprocessing & Partitioning",
    "normal-beat selection, 187-sample windows, 10/25/50/100% regimes",
    fc=C_PROC, sub_fs=7.3)
arrow(50, 72.4, 50, 66.6)

# ---- Adversarial training block (dashed container) ----
ax.add_patch(Rectangle((5, 34), 90, 32, fill=False, linestyle=(0, (5, 3)),
                       linewidth=1.1, edgecolor="#777777", zorder=1))
ax.text(50, 63.8, "Adversarial Training Loop", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#555555")

# Generator sub-block
ax.add_patch(Rectangle((8, 36.5), 48, 24.5, fill=False, linewidth=0.9,
                       edgecolor="#999999", zorder=1))
ax.text(32, 58.8, "Hybrid Generator", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#444444")

box(11, 52.0, 42, 5.0, "Classical pre-encoder  (187 \u2192 8)", fc=C_PROC, fs=8.0)
arrow(32, 51.4, 32, 49.4)
box(11, 44.8, 42, 5.0, "8-qubit Variational Quantum Circuit", fc=C_QUANT, fs=8.0)
arrow(32, 44.2, 32, 42.2)
box(11, 37.6, 42, 5.0, "Classical decoder  (8 \u2192 187)", fc=C_PROC, fs=8.0)

# Discriminator
box(68, 44.5, 24, 10, "Classical\nDiscriminator", "training-time only",
    fc=C_DISC, fs=8.8, sub_fs=7.0)

# Generator <-> Discriminator arrows
arrow(56.8, 52.4, 67.4, 52.4)
ax.text(62.1, 53.8, "reconstruction", ha="center", va="center", fontsize=6.4,
        color="#555555")
arrow(67.4, 47.2, 56.8, 47.2)
ax.text(62.1, 45.5, "adversarial loss", ha="center", va="center", fontsize=6.4,
        color="#555555")

arrow(50, 33.4, 50, 30.6)

# ---- Stage 4: anomaly scoring ----
box(20, 21, 60, 9, "Reconstruction-Error Scoring",
    "s(x) = (1/d) \u2016x \u2212 G(E(x))\u2016\u00b2   |   threshold \u03c4 = 95th pct. of normal training error",
    fc=C_PROC, sub_fs=6.9)
arrow(50, 20.4, 50, 15.6)

# ---- Stage 5: decision ----
box(27, 6, 46, 9, "Anomaly Decision",
    "s(x) > \u03c4  \u2192  anomalous beat;   otherwise normal",
    fc=C_OUT, sub_fs=7.3)

plt.tight_layout()
plt.savefig("/home/claude/figures/fig2_framework.png", dpi=220,
            bbox_inches="tight", facecolor="white")
print("saved fig2_framework.png")
