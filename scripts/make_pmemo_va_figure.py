#!/usr/bin/env python3
"""Thesis figure: hypothesised discrete clusters vs the real PMEmo annotation
distribution on the valence-arousal plane.

Panel (a) is an explicitly hypothetical schematic (synthetic blobs).
Panel (b) plots the REAL 767 matched PMEmo 2019 static annotations,
min-max normalised exactly as in src/utils/data_utils.load_pmemo_data,
quadrants thresholded at 0.5 (get_emotion_quadrant). Quadrant counts must
reproduce the canonical 469/67/64/167 (61% majority) or the script aborts.

Output: overleaf_draft/diagrams/pmemo_va_annotations.{pdf,png}
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

CSV = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"
OUT = "overleaf_draft/diagrams/pmemo_va_annotations"

NAVY, SLATE, LCREAM = "#1F3A5F", "#5A6B7A", "#FDF9EE"
COL = {"HVHA": "#D4923E", "HVLA": "#3FA7A6", "LVHA": "#B0405A", "LVLA": "#4A6A99"}

# ---- real data, processed exactly as the training pipeline does ----
df = pd.read_csv(CSV)
norm = lambda c: (c - c.min()) / (c.max() - c.min() + 1e-8)
V = norm(df["Valence(mean)"]).values
A = norm(df["Arousal(mean)"]).values
quad = np.where((V >= .5) & (A >= .5), "HVHA",
       np.where((V >= .5) & (A < .5), "HVLA",
       np.where((V < .5) & (A >= .5), "LVHA", "LVLA")))
counts = {q: int((quad == q).sum()) for q in COL}
assert counts == {"HVHA": 469, "HVLA": 67, "LVHA": 64, "LVLA": 167}, counts
assert len(df) == 767

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.2),
                               gridspec_kw=dict(wspace=0.10))

# ================= (a) hypothesised discrete clusters (schematic) ==========
axL.set_title("(a) Hypothesised geometry: four discrete clusters",
              fontsize=12, color=NAVY, fontweight="bold", pad=10)
axL.set_xlim(-1.05, 1.05); axL.set_ylim(-1.05, 1.05); axL.set_aspect("equal")
axL.set_xticks([]); axL.set_yticks([])
axL.axhline(0, color=SLATE, lw=0.8, ls=(0, (3, 2)))
axL.axvline(0, color=SLATE, lw=0.8, ls=(0, (3, 2)))
axL.text(0.99, 0.04, "Valence", fontsize=9, color=SLATE, ha="right", va="bottom")
axL.text(0.03, 0.99, "Arousal", fontsize=9, color=SLATE, ha="left", va="top", rotation=90)
rng = np.random.default_rng(7)
centers = {"HVHA": (0.55, 0.55), "HVLA": (0.55, -0.55),
           "LVHA": (-0.55, 0.55), "LVLA": (-0.55, -0.55)}
names = {"HVHA": "HVHA (happy)", "HVLA": "HVLA (calm)",
         "LVHA": "LVHA (tense)", "LVLA": "LVLA (sad)"}
for q, (cx, cy) in centers.items():
    pts = rng.normal((cx, cy), 0.07, (70, 2))
    axL.scatter(pts[:, 0], pts[:, 1], s=13, c=COL[q], alpha=.85, ec="none", zorder=3)
    axL.add_patch(Circle((cx, cy), 0.20, fill=False, ec=COL[q], lw=1.4,
                         ls=(0, (3, 2)), zorder=2))
    axL.text(cx, cy - 0.31, names[q], ha="center", fontsize=9,
             color=COL[q], fontweight="bold")
axL.text(0.5, -0.05, "schematic (not observed): separable clusters\n"
         "would give a Silhouette above 0.5;\nmeasured held-out values stay at 0.18--0.29",
         transform=axL.transAxes, ha="center", va="top", fontsize=9,
         color=SLATE, style="italic",
         bbox=dict(fc=LCREAM, ec=SLATE, boxstyle="round,pad=0.4", lw=0.7))

# ================= (b) real PMEmo annotations ===============================
axR.set_title("(b) Measured PMEmo 2019 annotations (n = 767)",
              fontsize=12, color=NAVY, fontweight="bold", pad=10)
axR.set_xlim(-0.03, 1.03); axR.set_ylim(-0.03, 1.03); axR.set_aspect("equal")
axR.set_xticks([0, 0.5, 1]); axR.set_yticks([0, 0.5, 1])
axR.tick_params(labelsize=8.5, colors=SLATE)
axR.axhline(0.5, color=SLATE, lw=0.8, ls=(0, (3, 2)))
axR.axvline(0.5, color=SLATE, lw=0.8, ls=(0, (3, 2)))
axR.set_xlabel("Valence (normalised)", fontsize=9.5, color=SLATE)
axR.set_ylabel("Arousal (normalised)", fontsize=9.5, color=SLATE)
for q in COL:
    m = quad == q
    axR.scatter(V[m], A[m], s=13, c=COL[q], alpha=.75, ec="none", zorder=3)
lab = {"HVHA": (0.985, 0.985, "right", "top",    "HVHA (happy)\n$n=469$ (61 %)"),
       "HVLA": (0.985, 0.015, "right", "bottom", "HVLA (calm)\n$n=67$"),
       "LVHA": (0.015, 0.985, "left",  "top",    "LVHA (tense)\n$n=64$"),
       "LVLA": (0.015, 0.015, "left",  "bottom", "LVLA (sad)\n$n=167$")}
for q, (x, y, ha, va, txt) in lab.items():
    axR.text(x, y, txt, ha=ha, va=va, fontsize=9.5, color=COL[q],
             fontweight="bold",
             bbox=dict(fc="white", ec=COL[q], alpha=.85, boxstyle="round,pad=0.3", lw=0.9))
for s in axR.spines.values():
    s.set_color(SLATE)
for s in axL.spines.values():
    s.set_color(SLATE)

fig.tight_layout()
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
print("saved", OUT + ".{pdf,png}", "counts:", counts)
