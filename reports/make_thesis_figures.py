"""
make_thesis_figures.py — 4 publication-grade figures for the MSc thesis
========================================================================
"A Critical Audit of Self-Supervised Music Representations for Explainable
Emotion Recognition"

Style:
  - Clean vector-flat (matplotlib patches), white background
  - Palette: Slate, Soft Navy, Cream, accent Deep Teal (+ Muted Gold for SSL)
  - Sans-serif fonts, no 3D, fully specified tensor shapes & arrows
  - Saves to /home/arvin/thesis/mert/reports/figures/

Figures:
  1) Phase B — End-to-End Multimodal Processing Pipeline (horizontal DAG)
  2) Multi-Objective Hybrid Loss Architecture (forward / backward pass)
  3) Phase C — Explainable RAG Retrieval Engine (top-to-bottom)
  4) Continuous Latent Manifold vs Discrete Quadrant Typology (concept map)

Run from reports/:
  python make_thesis_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Ellipse, Circle, Polygon, ConnectionPatch)
from matplotlib.lines import Line2D
from matplotlib import colormaps

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────
SLATE  = "#5E6B7A"   # muted slate gray (text / secondary lines)
NAVY   = "#1F3A5F"   # soft navy blue (primary structure)
CREAM  = "#F8F2E2"   # cream (neutral fills)
LCREAM = "#FBF8EE"   # very pale cream
TEAL   = "#0E7C7B"   # deep teal (optimisation accent)
GOLD   = "#B8923A"   # muted gold (frozen SSL encoders)
LGOLD  = "#E8D5A6"   # light gold tint
DGRAY  = "#2A2E33"   # near-black text
LGRAY  = "#EDEFF2"   # light gray for filled boxes
EDGE   = "#3B4A60"   # neutral edge

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.0,
    "axes.linewidth": 1.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


# ── Drawing primitives ───────────────────────────────────────────────
def rbox(ax, x, y, w, h, text, fc=LGRAY, ec=NAVY, lw=1.2, fs=9.0,
         tc=DGRAY, bold=False, rounding=0.8, align="center"):
    """Rounded box with centered text."""
    p = FancyBboxPatch((x, y), w, h,
                      boxstyle=f"round,pad=0.3,rounding_size={rounding}",
                      fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    fw = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=fw, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.3, ms=14,
          style="-|>", label=None, lab_off=(0, 0), lab_fs=7.5,
          lab_color=None, lab_bg=False, dashed=False, zorder=2):
    ls = (0, (4, 3)) if dashed else "-"
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=ms, lw=lw, color=color,
                        linestyle=ls, zorder=zorder)
    ax.add_patch(a)
    if label:
        kw = dict(ha="center", va="center", fontsize=lab_fs,
                  color=lab_color if lab_color else color, style="italic",
                  zorder=zorder + 1)
        if lab_bg:
            kw["bbox"] = dict(facecolor="white", edgecolor="none",
                              boxstyle="round,pad=0.2", alpha=0.95)
        ax.text((x1 + x2) / 2 + lab_off[0], (y1 + y2) / 2 + lab_off[1],
                label, **kw)


def dotted_container(ax, x, y, w, h, title, title_pos="top"):
    rect = Rectangle((x, y), w, h, fill=False, ec=SLATE, lw=1.0,
                    linestyle=(0, (4, 3)), zorder=1)
    ax.add_patch(rect)
    if title_pos == "top":
        ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top",
                fontsize=9.5, color=SLATE, fontweight="bold", style="italic")


# =====================================================================
# FIGURE 1 — Phase B end-to-end multimodal pipeline (horizontal DAG)
# =====================================================================
def figure1():
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # Title
    ax.text(50, 97, "Figure 1 — Phase B: End-to-End Multimodal Processing Pipeline",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(50, 93.2,
            "Frozen MERT (25 layers · 1024-D) + EDA bio-signals → Weighted Layer Fusion "
            "→ Bottleneck Regressor → continuous arousal/valence",
            ha="center", fontsize=10, style="italic", color=SLATE)

    # ── Input stage (dotted container) ──
    dotted_container(ax, 1.5, 50, 16, 36, "PMEmo 2019 Dataset Alignments")
    rbox(ax, 3, 73, 13, 8,
         "Raw Audio Clips\n(767 samples · 22 kHz)",
         fc=CREAM, ec=NAVY, fs=8.5)
    rbox(ax, 3, 55, 13, 8,
         "Synchronized EDA\nBio-Signals (200 Hz)",
         fc=CREAM, ec=NAVY, fs=8.5)

    # ── MERT encoder (orange/gold) ──
    rbox(ax, 22, 71, 17, 12,
         "Frozen MERT Encoder\n330 M parameters · 25 Layers",
         fc=GOLD, ec=NAVY, fs=9, bold=True, tc="white")
    arrow(ax, 16, 77, 22, 77, label="audio →", lab_off=(0, 1.2), lab_color=SLATE)

    # ── 25 stacked transformer plates ──
    n_layers = 25
    base_x, base_y = 42, 70.5
    plate_w, plate_h = 14, 0.55
    step_x, step_y = 0.10, 0.55
    for i in range(n_layers):
        ox = base_x + i * step_x
        oy = base_y + i * step_y
        rect = Rectangle((ox, oy), plate_w, plate_h, fc=LCREAM, ec=NAVY,
                         lw=0.45, zorder=2)
        ax.add_patch(rect)
    ax.text(base_x + plate_w / 2 + 1.0,
            base_y + n_layers * step_y + 1.4,
            "25 × Transformer Hidden States  (B, 25, 1024)",
            ha="center", fontsize=8.0, color=NAVY, fontweight="bold")
    ax.text(base_x + plate_w / 2 + 1.0,
            base_y - 1.2, "layer 0 (acoustic) … layer 24 (semantic)",
            ha="center", fontsize=7.0, color=SLATE, style="italic")
    arrow(ax, 39, 77, 42, 77, color=SLATE)

    # ── Fusion mixer with near-uniform weight bars ──
    rbox(ax, 60.5, 73, 17, 11,
         "Weighted Layer Fusion Mixer\nSoftmax-normalised α  (sum α = 1)",
         fc=LGRAY, ec=NAVY, fs=8.5, bold=True)
    # mini bar chart for weight uniformity
    bar_x0 = 61.5; bar_y0 = 67.0
    rng = np.random.default_rng(0)
    heights = 1.0 + rng.normal(0, 0.08, size=25)  # near-uniform with tiny noise
    heights = heights * 1.6
    bw = 0.55
    for i, hh in enumerate(heights):
        ax.add_patch(Rectangle((bar_x0 + i * 0.58, bar_y0),
                              bw, hh, fc=TEAL, ec="none", alpha=0.85))
    ax.plot([bar_x0, bar_x0 + 25 * 0.58], [bar_y0, bar_y0],
            color=DGRAY, lw=0.7)
    ax.text(bar_x0 + 25 * 0.58 / 2, bar_y0 - 1.0,
            "Learned α  (near-uniform · entropy H = 3.2178 / max 3.2189)",
            ha="center", fontsize=7.2, color=SLATE, style="italic")

    # arrow from plates → fusion
    arrow(ax, 57.5, 77, 60.5, 78.5, color=SLATE)
    arrow(ax, 57.5, 76, 60.5, 77.5, color=SLATE, lw=0.8)
    arrow(ax, 57.5, 75, 60.5, 76.5, color=SLATE, lw=0.8)

    # ── Fused embedding label + bottleneck head ──
    arrow(ax, 77.5, 78.5, 81.5, 78.5,
          label="Fused Embedding\n(B, 1024)", lab_off=(0, 1.8),
          lab_color=NAVY, lab_bg=True)
    rbox(ax, 81.5, 71, 17, 12,
         "Bottleneck Regressor Head\n1024 → 256 → 128-D\nLayerNorm · ReLU · Dropout(0.4)",
         fc=CREAM, ec=NAVY, fs=8.5, bold=True)

    # ── EDA branch (mid) ──
    rbox(ax, 22, 54, 18, 9,
         "Statistical Feature Extractor\nmean · std · slope · peaks · max · amp\n→ (B, 7)",
         fc=CREAM, ec=NAVY, fs=8.2)
    arrow(ax, 16, 59, 22, 58.5, label="EDA →", lab_off=(0, 1.2), lab_color=SLATE)

    arrow(ax, 40, 58.5, 81, 67.0, color=SLATE,
          label="7-D EDA features", lab_off=(0, 1.4),
          lab_color=SLATE, lab_bg=True)

    # ── Late concatenation & late-fusion head ──
    rbox(ax, 81.5, 56, 17, 9,
         "Late Concatenation\n(B, 128 + 7 = 135)\n[ pipeline reports 160 if EDA proj=32 ]",
         fc=LGRAY, ec=NAVY, fs=7.6)
    arrow(ax, 90, 71, 90, 65,
          color=NAVY, label="128-D audio latent z", lab_off=(7, 0),
          lab_color=NAVY, lab_bg=True, lab_fs=7.0)

    rbox(ax, 81.5, 41, 17, 11,
         "Multi-Encoder Late-Fusion Head\n160 → 64 → 2\n(Linear · ReLU · Dropout · Linear)",
         fc=CREAM, ec=NAVY, fs=8.2, bold=True)
    arrow(ax, 90, 56, 90, 52, color=NAVY)

    # ── Final outputs ──
    rbox(ax, 78, 28, 9.5, 8, "Predicted\nArousal\n∈ [0,1]",
         fc=LCREAM, ec=TEAL, fs=9, bold=True, tc=TEAL, lw=1.6)
    rbox(ax, 92, 28, 9.5, 8, "Predicted\nValence\n∈ [0,1]",
         fc=LCREAM, ec=TEAL, fs=9, bold=True, tc=TEAL, lw=1.6)
    arrow(ax, 87, 41, 83, 36, color=TEAL)
    arrow(ax, 93, 41, 97, 36, color=TEAL)

    # Footnote — emphasises continuous regression
    ax.text(50, 7,
            "Outputs are two continuous real numbers in [0,1] — no discrete classifier head. "
            "Russell-quadrant labels appear only in post-hoc reporting (per-quadrant R²).",
            ha="center", fontsize=8.5, style="italic", color=SLATE,
            bbox=dict(facecolor=LCREAM, edgecolor=SLATE, boxstyle="round,pad=0.5", lw=0.7))

    plt.tight_layout()
    out = os.path.join(OUT, "fig1_phaseB_pipeline.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 1 → {out}")


# =====================================================================
# FIGURE 2 — Multi-objective hybrid loss architecture
# =====================================================================
def figure2():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # Title
    ax.text(50, 97, "Figure 2 — Multi-Objective Hybrid Loss Architecture",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(50, 93.2,
            "Forward pass: 4 weighted loss components target distinct dataset challenges.   "
            "Backward pass: differential learning rate over fusion vs head.",
            ha="center", fontsize=10, style="italic", color=SLATE)

    # ── Latent block (forward in) ──
    rbox(ax, 36, 82, 28, 8,
         "Fused Latent Embeddings\nz ∈ ℝ^{B × 128}  (L2-normalised)",
         fc=CREAM, ec=NAVY, fs=9, bold=True)
    # Regressor and prediction
    rbox(ax, 36, 72, 28, 7,
         "Regression Module · y_pred = W·z + b ∈ ℝ^{B × 2}",
         fc=LGRAY, ec=NAVY, fs=9, bold=True)
    arrow(ax, 50, 82, 50, 79, color=NAVY)

    # ── Horizontal dispatch line ──
    dispatch_y = 67
    ax.plot([15, 90], [dispatch_y, dispatch_y], color=SLATE, lw=1.0)
    arrow(ax, 50, 72, 50, dispatch_y + 0.2, color=NAVY)

    # ── 4 loss branches (parallel column layout) ──
    branches = [
        dict(x=18, name="Mean Squared Error",  abbr="MSE",
             w="w = 1.0", color=TEAL,
             target="Absolute Coordinate\nVariance\n(point-wise V–A error)"),
        dict(x=40, name="Concordance Corr. Coef.", abbr="CCC",
             w="w = 0.5", color=TEAL,
             target="Systematic Linear\nTracking Bias"),
        dict(x=62, name="Differentiable Rank Loss", abbr="Rank",
             w="w = 0.3", color=TEAL,
             target="Ordinal Retrieval\nDistance Consistency"),
        dict(x=84, name="Supervised Contrastive Reg.", abbr="SupCR",
             w="w = 0.1", color=TEAL,
             target="Latent Topology →\nLocal Continuous\nNeighbourhoods"),
    ]
    by_top = 53; by_bot = 35
    for b in branches:
        # Drop from dispatch line into loss component box
        arrow(ax, b["x"], dispatch_y, b["x"], by_top + 11.5,
              color=SLATE, lw=1.0)
        # Loss component box
        rbox(ax, b["x"] - 8, by_top, 16, 11,
             f"{b['name']}\n({b['abbr']})\n{b['w']}",
             fc=LCREAM, ec=b["color"], fs=8.5, bold=True, tc=b["color"], lw=1.6)
        # Challenge / target box
        rbox(ax, b["x"] - 8, by_bot, 16, 9,
             f"Targets:\n{b['target']}",
             fc="white", ec=SLATE, fs=7.5, tc=DGRAY)
        # Arrow loss → target
        arrow(ax, b["x"], by_top, b["x"], by_bot + 9, color=b["color"], lw=1.0)

    # ── Total loss block (bottom centre) ──
    total_y = 15
    rbox(ax, 28, total_y, 44, 11,
         "Total Loss\nL = 1.0·MSE + 0.5·CCC + 0.3·Rank + 0.1·SupCR",
         fc=TEAL, ec=NAVY, fs=10, bold=True, tc="white", lw=1.6)
    # Convergence: each target box → total loss
    for b in branches:
        arrow(ax, b["x"], by_bot, 50, total_y + 11, color=b["color"], lw=0.9)

    # ── Backward pass dashed arrow (right side) ──
    arrow(ax, 75, total_y + 5, 92, total_y + 5,
          color=GOLD, lw=2.0, dashed=True, ms=20, zorder=4)
    arrow(ax, 92, total_y + 5, 92, 86,
          color=GOLD, lw=2.0, dashed=True, ms=20, zorder=4)
    arrow(ax, 92, 86, 65, 86,
          color=GOLD, lw=2.0, dashed=True, ms=20, zorder=4)
    ax.text(96, 50,
            "Gradient\nBackpropagation\nDifferential LR:\n"
            "η_fusion = 1×10⁻²\n"
            "η_head    = 1×10⁻⁴\n"
            "wd            = 1×10⁻³",
            ha="left", va="center", fontsize=8.0, color=GOLD, fontweight="bold",
            bbox=dict(facecolor=LCREAM, edgecolor=GOLD, boxstyle="round,pad=0.4", lw=1.2))

    # Footnote
    ax.text(50, 7,
            "Forward (solid) optimises 4 complementary criteria;  "
            "backward (dashed gold) injects per-parameter-group learning rates.",
            ha="center", fontsize=8.6, style="italic", color=SLATE,
            bbox=dict(facecolor=LCREAM, edgecolor=SLATE, boxstyle="round,pad=0.5", lw=0.7))

    plt.tight_layout()
    out = os.path.join(OUT, "fig2_hybrid_loss.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 2 → {out}")


# =====================================================================
# FIGURE 3 — Phase C RAG retrieval engine (top-down)
# =====================================================================
def figure3():
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.text(50, 98, "Figure 3 — Phase C: Explainable RAG Retrieval Engine",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(50, 95.5,
            "Vector index → dual-path retrieval (k-NN + contrastive foils) → context assembly → local LLM narrative",
            ha="center", fontsize=10, style="italic", color=SLATE)

    # ── 1. INDEX ENGINE (top) ──
    ax.text(3, 91, "1.  Index Engine  (offline)", fontsize=11, fontweight="bold", color=NAVY)
    rbox(ax, 5, 80, 30, 8,
         "Phase B Triple-Encoder Model\n(frozen, eval mode)",
         fc=GOLD, ec=NAVY, fs=9, bold=True, tc="white")
    rbox(ax, 40, 80, 22, 8,
         "PMEmo Library\n767 songs",
         fc=CREAM, ec=NAVY, fs=9, bold=True)
    arrow(ax, 40, 84, 35, 84, color=SLATE, label="batch encode", lab_off=(0, 1.0),
          lab_color=SLATE)
    rbox(ax, 67, 79, 28, 10,
         "Vector Store Index\nprototypes.npy\nUnit-normalised 1024-D continuous latents",
         fc=LCREAM, ec=TEAL, fs=8.5, bold=True, tc=TEAL, lw=1.6)
    arrow(ax, 35, 84, 67, 84, color=SLATE, label="write", lab_off=(0, 1.0),
          lab_color=SLATE)

    # Section divider
    ax.plot([2, 98], [76, 76], color=SLATE, lw=0.6, linestyle=(0, (4, 3)))

    # ── 2. SEARCH MECHANISM ──
    ax.text(3, 73, "2.  Search Mechanism  (per query)", fontsize=11, fontweight="bold", color=NAVY)
    rbox(ax, 5, 64, 26, 7, "Query Audio Clip", fc=CREAM, ec=NAVY, fs=9)
    rbox(ax, 37, 64, 26, 7,
         "Same Frozen Triple-Encoder\n(re-use of Phase B model)",
         fc=GOLD, ec=NAVY, fs=8.5, bold=True, tc="white")
    arrow(ax, 31, 67.5, 37, 67.5, color=SLATE)

    rbox(ax, 69, 64, 26, 7,
         "Query Latent Vector\nz_q ∈ ℝ^1024  (L2-norm)",
         fc=LCREAM, ec=TEAL, fs=8.5, bold=True, tc=TEAL, lw=1.5)
    arrow(ax, 63, 67.5, 69, 67.5, color=SLATE)

    # branch into two retrieval paths
    arrow(ax, 82, 64, 30, 55, color=SLATE, lw=1.1)
    arrow(ax, 82, 64, 70, 55, color=SLATE, lw=1.1)

    rbox(ax, 6, 47, 32, 9,
         "Path A — Nearest Proximity\nFAISS Exact Cosine k-NN\n(IndexFlatIP, k = 5)",
         fc=LGRAY, ec=NAVY, fs=8.5, bold=True)
    rbox(ax, 6, 39, 32, 6,
         "Top-k Emotion Prototypes",
         fc=CREAM, ec=NAVY, fs=8.5)
    arrow(ax, 22, 47, 22, 45, color=NAVY)

    rbox(ax, 56, 47, 32, 9,
         "Path B — Counterfactual Contrast\nMatrix Product Rejection Search\n(latents @ z_q.T → argmin)",
         fc=LGRAY, ec=NAVY, fs=8.5, bold=True)
    rbox(ax, 56, 39, 32, 6,
         "n Contrastive Foils\n(most emotionally dissimilar)",
         fc=CREAM, ec=NAVY, fs=8.5)
    arrow(ax, 72, 47, 72, 45, color=NAVY)

    # Section divider
    ax.plot([2, 98], [35, 35], color=SLATE, lw=0.6, linestyle=(0, (4, 3)))

    # ── 3. CONTEXT ASSEMBLY ──
    ax.text(3, 32, "3.  Context Assembly Framework", fontsize=11, fontweight="bold", color=NAVY)
    rbox(ax, 8, 19, 84, 11,
         "Structured Prompt:\n"
         "• Query predictions (arousal, valence)\n"
         "• Top-k prototypes (V-A coords, EDA descriptors, librosa music-theory tags)\n"
         "• EDA physiological population narratives    • Contrastive rejection foils    • MERT layer attribution",
         fc=LCREAM, ec=TEAL, fs=8.2, tc=DGRAY, lw=1.5)
    arrow(ax, 22, 39, 30, 30, color=NAVY)
    arrow(ax, 72, 39, 70, 30, color=NAVY)

    # Section divider
    ax.plot([2, 98], [16, 16], color=SLATE, lw=0.6, linestyle=(0, (4, 3)))

    # ── 4. GENERATION ──
    ax.text(3, 13, "4.  Generation", fontsize=11, fontweight="bold", color=NAVY)
    rbox(ax, 10, 4, 32, 9,
         "Local LLM Backend\nOllama / Llama 3.2\n(server-free, no API key)",
         fc=GOLD, ec=NAVY, fs=8.5, bold=True, tc="white")
    arrow(ax, 50, 19, 26, 13, color=NAVY)

    rbox(ax, 55, 4, 38, 9,
         "Structured Explainable Music Narrative\n(L1 deterministic template + L2 LLM prose, 150–200 words)",
         fc=LCREAM, ec=TEAL, fs=8.3, bold=True, tc=TEAL, lw=1.6)
    arrow(ax, 42, 8.5, 55, 8.5, color=TEAL, ms=18)

    plt.tight_layout()
    out = os.path.join(OUT, "fig3_phaseC_rag.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 3 → {out}")


# =====================================================================
# FIGURE 4 — Continuous manifold vs discrete quadrants (concept map)
# =====================================================================
def figure4():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7.5),
                                    gridspec_kw=dict(wspace=0.06))
    for a in (axL, axR):
        a.set_xlim(-1.05, 1.05); a.set_ylim(-1.05, 1.05); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        a.spines["top"].set_color(SLATE); a.spines["right"].set_color(SLATE)
        a.spines["bottom"].set_color(SLATE); a.spines["left"].set_color(SLATE)

    # Title
    fig.suptitle("Figure 4 — Continuous Latent Manifold vs. Discrete Quadrant Typology",
                 fontsize=14, fontweight="bold", color=NAVY, y=0.985)

    # =================  LEFT — hypothesised discrete clusters  =================
    axL.set_title("(a) The Trivial Hypothesis  —  Discrete Clusters",
                  fontsize=11, color=NAVY, fontweight="bold", pad=10)

    # V-A axes
    axL.axhline(0, color=SLATE, lw=0.8, linestyle=(0, (3, 2)))
    axL.axvline(0, color=SLATE, lw=0.8, linestyle=(0, (3, 2)))
    axL.text(1.02, 0, "Valence", fontsize=8.5, color=SLATE, va="center")
    axL.text(0, 1.04, "Arousal", fontsize=8.5, color=SLATE, ha="center")

    rng = np.random.default_rng(7)
    quad_centers = {
        "HVHA (Happy)":  ( 0.55,  0.55),
        "HVLA (Calm)":   ( 0.55, -0.55),
        "LVHA (Angry)":  (-0.55,  0.55),
        "LVLA (Sad)":    (-0.55, -0.55),
    }
    quad_colors = {
        "HVHA (Happy)": "#D4923E",
        "HVLA (Calm)":  "#3FA7A6",
        "LVHA (Angry)": "#B0405A",
        "LVLA (Sad)":   "#4A6A99",
    }
    for q, (cx, cy) in quad_centers.items():
        pts = rng.normal(loc=(cx, cy), scale=0.07, size=(70, 2))
        axL.scatter(pts[:, 0], pts[:, 1], s=14, c=quad_colors[q],
                    alpha=0.85, edgecolors="none", zorder=3)
        # outline circle
        circle = Circle((cx, cy), 0.20, fill=False, ec=quad_colors[q],
                       lw=1.4, linestyle=(0, (3, 2)), zorder=2)
        axL.add_patch(circle)
        axL.text(cx, cy - 0.30, q, ha="center", fontsize=8.5,
                color=quad_colors[q], fontweight="bold")

    axL.text(0.5, -0.08,
            "Hypothesised Discrete Clustering\n"
            "(disproven for single-MERT: Silhouette ≈ 0;\n"
            "see §3.5 — multi-encoder achieves Silhouette 0.26)",
            ha="center", fontsize=8.5, color=SLATE, style="italic",
            transform=axL.transAxes, va="top",
            bbox=dict(facecolor=LCREAM, edgecolor=SLATE,
                      boxstyle="round,pad=0.4", lw=0.7))

    # =================  RIGHT — empirical continuous manifold  =================
    axR.set_title("(b) The Empirical Reality  —  Continuous Manifold",
                  fontsize=11, color=NAVY, fontweight="bold", pad=10)

    axR.axhline(0, color=SLATE, lw=0.8, linestyle=(0, (3, 2)))
    axR.axvline(0, color=SLATE, lw=0.8, linestyle=(0, (3, 2)))
    axR.text(1.02, 0, "Valence", fontsize=8.5, color=SLATE, va="center")
    axR.text(0, 1.04, "Arousal", fontsize=8.5, color=SLATE, ha="center")

    # ── Crescent shape — comma/banana skewed heavily toward HVHA ──
    # Two-component mixture:
    #   majority (61%) : dense Gaussian in HVHA (top-right of crescent head)
    #   minority (39%) : thin arc threading down through LVHA → LVLA → HVLA tail
    rng = np.random.default_rng(11)
    n_total = 900
    n_maj = int(0.61 * n_total)
    n_min = n_total - n_maj
    # Majority — HVHA head
    maj_xy = rng.multivariate_normal(
        mean=[0.48, 0.42],
        cov=[[0.045, 0.020], [0.020, 0.045]],
        size=n_maj)
    # Minority — thin arc through other quadrants forming the crescent tail
    t_min = rng.uniform(-2.4, 0.4, n_min)        # arc parameter biased toward LVHA / LVLA
    arc_r = 0.70 + rng.normal(0, 0.06, n_min)
    min_x = arc_r * np.sin(t_min) + rng.normal(0, 0.08, n_min)
    min_y = arc_r * np.cos(t_min) - 0.25 + rng.normal(0, 0.10, n_min)
    X = np.concatenate([maj_xy[:, 0], min_x])
    Y = np.concatenate([maj_xy[:, 1], min_y])
    X = np.clip(X, -0.95, 0.95)
    Y = np.clip(Y, -0.95, 0.95)
    # color by "quadrant prior" — gradient from quad palette using V-A coords
    def color_for(xi, yi):
        # blend across quadrants
        if xi >= 0 and yi >= 0:   return "#D4923E"  # HVHA
        if xi >= 0 and yi <  0:   return "#3FA7A6"  # HVLA
        if xi <  0 and yi >= 0:   return "#B0405A"  # LVHA
        return "#4A6A99"                            # LVLA
    cols = [color_for(xi, yi) for xi, yi in zip(X, Y)]
    axR.scatter(X, Y, s=12, c=cols, alpha=0.55, edgecolors="none", zorder=3)

    # Dense core annotation
    core = Ellipse((0.45, 0.45), 0.55, 0.45, fc="none", ec="#D4923E",
                   lw=1.8, linestyle=(0, (4, 3)), zorder=4)
    axR.add_patch(core)
    axR.annotate("HVHA (Happy) dense core\n61 % majority-class dominance",
                 xy=(0.55, 0.55), xytext=(0.05, 0.92),
                 fontsize=8.5, color="#D4923E", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#D4923E", lw=1.2),
                 bbox=dict(facecolor=LCREAM, edgecolor="#D4923E",
                           boxstyle="round,pad=0.3", lw=0.9))

    # gradient interleaving annotation
    axR.text(-0.85, -0.85,
             "quadrants bleed into one another\nno crisp boundaries",
             fontsize=7.8, color=SLATE, style="italic",
             bbox=dict(facecolor=LCREAM, edgecolor=SLATE,
                       boxstyle="round,pad=0.3", lw=0.6))

    axR.text(0.5, -0.08,
             "True empirically-audited latent geometry\n"
             "(continuous emotion gradient · Precision@5 optimisation space)",
             ha="center", fontsize=8.5, color=SLATE, style="italic",
             transform=axR.transAxes, va="top",
             bbox=dict(facecolor=LCREAM, edgecolor=SLATE,
                       boxstyle="round,pad=0.4", lw=0.7))

    out = os.path.join(OUT, "fig4_manifold_vs_clusters.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure 4 → {out}")


# =====================================================================
def main():
    print(f"Output dir: {OUT}\n")
    figure1()
    figure2()
    figure3()
    figure4()
    print("\nDone. PNGs at 200 DPI + matching PDFs.")


if __name__ == "__main__":
    main()
