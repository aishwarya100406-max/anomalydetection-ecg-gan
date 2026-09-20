"""
Figure 2 (full architecture): hybrid quantum-classical GAN pipeline for ECG
anomaly detection, with real artifacts from the implemented pipeline embedded
at each stage (input beat, VQC ansatz, reconstruction overlay, error
distribution with the operating threshold).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

D = np.load("/home/claude/figures/_panel_data.npz")
orig, recon = D["orig"], D["recon"]
err_norm, err_anom, thr = D["err_norm"], D["err_anom"], float(D["thr"])
circuit_img = mpimg.imread("/home/claude/figures/fig_vqc_circuit_transparent.png")

C_DATA  = "#DCE6F1"
C_PROC  = "#D9EAD3"
C_QUANT = "#E4D9F2"
C_DISC  = "#FCE4D6"
C_OUT   = "#EFEFEF"
EDGE    = "#333333"

fig = plt.figure(figsize=(7.16, 8.6), dpi=240)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, fc, lw=1.0, ls="solid", ec=EDGE, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.5,rounding_size=1.0",
                 linewidth=lw, linestyle=ls, edgecolor=ec,
                 facecolor=fc, zorder=z))


def label(x, y, t, fs=8.6, bold=True, color="#111111", ha="center", style=None):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color,
            style=style, zorder=6)


def arrow(x1, y1, x2, y2, lw=1.15, color=EDGE, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=11, linewidth=lw, color=color,
                 linestyle=ls, zorder=5))


def inset(x, y, w, h):
    """x,y,w,h in 0-100 diagram units -> matplotlib axes in figure coords."""
    return fig.add_axes([x / 100, y / 100, w / 100, h / 100], zorder=7)


# ============================ STAGE 1: DATA ============================
box(4, 88.5, 92, 9.5, C_DATA)
label(50, 95.6, "ECG Data Acquisition", 9.4)
label(50, 92.6, "MIT-BIH Arrhythmia DB — training & same-dataset test    |    "
                "PTB Diagnostic DB — cross-dataset test", 6.9, bold=False,
      color="#444444", style="italic")
label(50, 90.0, "187-sample amplitude-normalized beat segments", 6.6,
      bold=False, color="#666666")

arrow(50, 88.2, 50, 85.6)

# ======================= STAGE 2: PREPROCESSING ========================
box(4, 75.5, 92, 10.0, C_PROC)
label(29, 83.3, "Preprocessing & Partitioning", 9.0)
label(29, 80.6, "normal-class beats only (72,471)", 6.8, bold=False,
      color="#444444", style="italic")
label(29, 78.1, "training regimes: 10% / 25% / 50% / 100%", 6.6,
      bold=False, color="#666666")

axi = inset(52, 76.6, 42, 7.6)
axi.plot(orig, color="#1f77b4", lw=0.9)
axi.set_title("representative normal input beat", fontsize=6.0, pad=1.5)
axi.tick_params(labelsize=5.0, length=1.6, pad=1)
axi.set_xlim(0, 186)
for s in axi.spines.values():
    s.set_linewidth(0.5)

arrow(50, 75.2, 50, 72.6)

# ================== STAGE 3: ADVERSARIAL TRAINING ======================
ax.add_patch(Rectangle((3, 33.0), 94, 39.4, fill=False,
             linestyle=(0, (5, 3)), linewidth=1.1, edgecolor="#777777",
             zorder=1))
label(50, 70.3, "Adversarial Training Loop", 9.2, color="#555555")

# ---- generator container ----
ax.add_patch(Rectangle((5.5, 35.0), 58, 33.2, fill=False, linewidth=0.9,
             edgecolor="#999999", zorder=1))
label(34.5, 66.6, "Hybrid Generator", 8.8, color="#444444")

# pre-encoder
box(8, 61.0, 53, 4.4, C_PROC)
label(34.5, 63.2, "Classical pre-encoder:  187 → 8  (dense + tanh)", 7.2)
arrow(34.5, 60.7, 34.5, 58.9)

# VQC block with the real circuit rendered inside
box(8, 45.6, 53, 13.0, C_QUANT)
label(34.5, 57.2, "8-qubit Variational Quantum Circuit", 7.4)
label(34.5, 54.9, "2 entangling layers · 32 trainable parameters", 6.2,
      bold=False, color="#4a4a4a", style="italic")
axc = inset(9.4, 46.2, 50.2, 7.7)
axc.imshow(circuit_img)
axc.axis("off")

arrow(34.5, 45.3, 34.5, 43.5)

# decoder
box(8, 37.0, 53, 6.2, C_PROC)
label(34.5, 41.7, "Classical decoder:  8 → 187  (dense + tanh)", 7.2)
axr = inset(9.8, 37.5, 49.4, 3.1)
axr.plot(orig, color="#1f77b4", lw=0.8, label="input")
axr.plot(recon, color="#2ca02c", lw=0.8, ls="--", label="reconstruction")
axr.set_xlim(0, 186)
axr.legend(fontsize=4.6, loc="upper right", frameon=False, handlelength=1.4)
axr.tick_params(labelsize=4.4, length=1.2, pad=0.8)
for s in axr.spines.values():
    s.set_linewidth(0.4)

# ---- discriminator ----
box(72, 46.5, 22, 12.0, C_DISC)
label(83, 55.4, "Classical", 8.4)
label(83, 52.8, "Discriminator", 8.4)
label(83, 50.2, "187 → 32 → 1", 6.6, bold=False, color="#444444")
label(83, 48.0, "training-time only", 6.4, bold=False, color="#666666",
      style="italic")

arrow(63.8, 54.6, 71.4, 54.6)
ax.text(67.6, 56.6, "reconstructed\nbeats", ha="center", va="center",
        fontsize=5.4, color="#555555", zorder=6, linespacing=1.05)
arrow(71.4, 49.2, 63.8, 49.2)
ax.text(67.6, 47.0, "adversarial\nloss", ha="center", va="center",
        fontsize=5.4, color="#555555", zorder=6, linespacing=1.05)

arrow(50, 32.7, 50, 30.3)

# ==================== STAGE 4: ANOMALY SCORING =========================
box(4, 13.0, 92, 17.0, C_PROC)
label(50, 28.2, "Reconstruction-Error Scoring", 9.2)
label(50, 25.6, r"$s(\mathbf{x}) = \frac{1}{d}\,\|\mathbf{x} - G(E(\mathbf{x}))\|^2$"
                "     |     threshold τ = 95th percentile of normal "
                "training-set error", 7.0, bold=False, color="#333333")

axh = inset(17, 15.0, 66, 8.6)
bins = np.linspace(0, np.percentile(np.concatenate([err_norm, err_anom]), 99), 60)
axh.hist(err_norm, bins=bins, color="#1f77b4", alpha=0.68, label="normal beats")
axh.hist(err_anom, bins=bins, color="#d62728", alpha=0.68, label="anomalous beats")
axh.axvline(thr, color="#111111", ls="--", lw=1.0)
axh.text(thr, axh.get_ylim()[1] * 0.86, "  τ", fontsize=6.4, fontweight="bold")
axh.set_xlabel("reconstruction error  s(x)", fontsize=5.8, labelpad=1.5)
axh.set_ylabel("count", fontsize=5.8, labelpad=1.5)
axh.tick_params(labelsize=5.0, length=1.6, pad=1)
axh.legend(fontsize=5.4, frameon=False, loc="upper right")
for s in axh.spines.values():
    s.set_linewidth(0.5)

arrow(50, 12.7, 50, 10.3)

# ======================== STAGE 5: DECISION ============================
box(20, 2.0, 60, 8.0, C_OUT)
label(50, 7.6, "Anomaly Decision", 9.2)
label(50, 4.6, "s(x) > τ  →  anomalous beat        s(x) ≤ τ  →  normal beat",
      7.2, bold=False, color="#333333")

fig.savefig("/home/claude/figures/fig2_full_architecture.png",
            bbox_inches="tight", facecolor="white")
print("saved fig2_full_architecture.png")
