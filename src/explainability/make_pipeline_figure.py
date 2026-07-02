"""
make_pipeline_figure.py — full pipeline figure (audio → V-A regression → Phase C explanation)
=============================================================================================
Clarifies that the system performs continuous-valued REGRESSION on arousal/valence
(not 4-way classification), and shows where the HybridLoss (MSE+CCC+Rank+SupCR)
attaches in training. Saves artifacts/pipeline_overview.png.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ART, exist_ok=True)

NAVY = "#1B3A6B"; TEAL = "#2E86AB"; LGRAY = "#F5F6FA"
DK = "#374151"; ACC = "#10B981"; AMBER = "#B45309"; RED = "#C0392B"

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc=LGRAY, ec=NAVY, fs=9, bold=False, tc=DK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    w_ = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight=w_, wrap=True)


def arrow(x1, y1, x2, y2, color=NAVY, label=None, lab_off=(0, 0)):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        lw=1.4, color=color)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2 + lab_off[0], (y1+y2)/2 + lab_off[1], label,
                ha="center", va="center", fontsize=7.5, color=color, style="italic")


# Title
ax.text(50, 97, "Music Emotion Recognition Pipeline — Audio → Continuous V-A Regression → Explanation",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)
ax.text(50, 93.5, "This system performs REGRESSION on continuous arousal & valence values "
                  "(it does NOT classify songs into discrete emotion classes)",
        ha="center", fontsize=9.5, style="italic", color=RED)

# Phase B header
ax.text(2, 87, "PHASE B — Training", fontsize=11, fontweight="bold", color=NAVY)
ax.plot([2, 98], [85.6, 85.6], color=NAVY, lw=0.6)

# Row 1: encoders (parallel branches)
box(2,  72, 14, 9, "Audio waveform\n(PMEmo chorus clip)", fc="#FFFFFF", fs=9)
box(20, 78, 16, 6, "MERT-v1-330M\n(frozen, 25 layers × 1024-d)", fc="#E0F7FA")
box(20, 70, 16, 6, "wav2vec2-base\n(frozen, 13 layers × 768-d)", fc="#FFF4E0")
box(20, 62, 16, 6, "Mel-spectrogram CNN\n(trainable, ~109K params)", fc="#FFE8E8")

arrow(16, 79, 20, 81)   # audio → MERT
arrow(16, 76, 20, 73)   # audio → w2v
arrow(16, 73, 20, 65)   # audio → mel-CNN

# Fusion column
box(40, 78, 14, 6, "WeightedLayerFusion\n(softmax over 25 layers → 1024-d)", fc="#E0F7FA", fs=8)
box(40, 70, 14, 6, "WeightedLayerFusion\n(softmax over 13 layers → 768-d)", fc="#FFF4E0", fs=8)
box(40, 62, 14, 6, "Shallow CNN\n→ 128-d", fc="#FFE8E8", fs=8)

arrow(36, 81, 40, 81)
arrow(36, 73, 40, 73)
arrow(36, 65, 40, 65)

# Concat
box(58, 70, 10, 12, "Concatenate\n(1024 + 768 + 128\n= 1920-d)", fc=LGRAY, bold=True, fs=8.5)
arrow(54, 81, 58, 80)
arrow(54, 73, 58, 76)
arrow(54, 65, 58, 72)

# Head + latent
box(72, 73, 12, 7, "Head\n1920 → 256 → 128", fc=LGRAY, fs=8.5)
arrow(68, 76, 72, 76)
box(72, 63, 12, 7, "L2-normalised\nlatent z (128-d)", fc=LGRAY, fs=8.5, bold=True)
arrow(78, 73, 78, 70)

# Regressor + outputs
box(88, 73, 10, 7, "Regressor\n(linear → 2)", fc=LGRAY, fs=8.5)
arrow(84, 76, 88, 76)
box(88, 63, 10, 7, "Arousal ∈ [0,1]\nValence ∈ [0,1]\n(continuous values)",
    fc="#FFFFFF", ec=ACC, fs=8.5, bold=True, tc=ACC)
arrow(93, 73, 93, 70)

# Loss block (highlighted as REGRESSION)
box(40, 50, 56, 8, "HybridLoss = MSE(1.0) + CCC(0.5) + Rank(0.3) + SupCR(0.1)        →   "
                   "REGRESSION objective on continuous A/V (no classification)",
    fc="#FFF8E1", ec=AMBER, fs=9.5, bold=True, tc=AMBER)
arrow(93, 63, 90, 58, color=AMBER, label="train signal", lab_off=(3, 0))

# Note
ax.text(2, 47, "Note: outputs are two continuous numbers in [0,1] — not 4 emotion classes. "
               "The 4 Russell quadrants used elsewhere are derived from V-A by thresholding at 0.5 only for "
               "analysis (per-quadrant R², Silhouette).",
        fontsize=8, color=DK, style="italic", wrap=True)

# Phase C header
ax.text(2, 41.5, "PHASE C — Inference & Explanation (uses the latent z from above)",
        fontsize=11, fontweight="bold", color=NAVY)
ax.plot([2, 98], [40, 40], color=NAVY, lw=0.6)

# Phase C flow
box(2, 28, 14, 9, "Query song\n(audio)", fc="#FFFFFF", fs=9)
box(20, 28, 16, 9, "Same encoder pipeline\n(re-uses Phase B model)\n→ latent z (128-d)", fc=LGRAY, fs=8.5)
arrow(16, 32.5, 20, 32.5)

box(40, 28, 16, 9, "Stored vector index\n(all 767 songs as 128-d latents\n+ ground-truth V-A + EDA)",
    fc="#E0F7FA", fs=8.5)
arrow(36, 32.5, 40, 32.5)

box(60, 28, 16, 9, "k-NN retrieval (cosine)\n+ contrastive foils\n+ 4-centroid profile", fc=LGRAY, fs=8.5)
arrow(56, 32.5, 60, 32.5)

box(80, 28, 17, 9, "Explanation:\n  L1 deterministic template\n  L2 LLM-generated prose",
    fc="#FFFFFF", ec=ACC, fs=8.5, bold=True, tc=ACC)
arrow(76, 32.5, 80, 32.5)

# Auxiliary annotations into Phase C
box(40, 14, 16, 9, "EDA features\n(7-dim, normalized)", fc="#FFF4E0", fs=8)
box(60, 14, 16, 9, "Music-theory annotator\n(librosa: key, tempo, timbre)\n— independent of SSL",
    fc="#FFE8E8", fs=8)
arrow(48, 23, 88, 28.5, color=DK)
arrow(68, 23, 88, 28.5, color=DK)

ax.text(50, 5, "Outputs of Phase C are explanations of the regression decision — not class labels. "
               "Ante-hoc 4-centroid profile is an interpretability readout (not an accurate classifier).",
        ha="center", fontsize=8, style="italic", color=DK)

plt.tight_layout()
out = os.path.join(ART, "pipeline_overview.png")
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved -> {out}")
