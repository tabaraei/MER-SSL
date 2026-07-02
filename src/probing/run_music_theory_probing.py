"""
run_music_theory_probing.py — Phase A: per-layer MERT probing + gap analysis
=============================================================================
For every music-theory feature, probes EACH of the 25 MERT transformer
layers (plus a mean-pooled-all-layers baseline) with a linear sklearn
probe — exactly the Phase A methodology (linear probing), no PyTorch
training loops, same split as probe_key.py / probe_tempo.py
(train_test_split test_size=0.2, random_state=42).

  Regression features (Ridge, metric=R²):
      chroma, tempo, rhythmic_stability, spectral_centroid,
      spectral_contrast, zcr
  Classification features (LogisticRegression, metric=accuracy):
      mode (binary), key (12-class)

Outputs:
  music_theory_probing_results.json   per-feature best layer + pooled score
  gap_analysis.json                   gap_features list (drives Phase B)
  plots/music_theory_probing_summary.png
  plots/music_theory_probing_heatmap.png

Uses the 25-layer embeddings at ../phaseB/pmemo_mert_all_layers.pt
(phaseA's own pmemo_mert_embeddings.pt is single-layer only).

Run from phaseA/:
    python run_music_theory_probing.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from configs.config import PATHS  # centralised config

MERT_PATH = str(PATHS.mert_features)
THEORY_PATH = str(PATHS.music_theory)
RESULTS_JSON = "music_theory_probing_results.json"
GAP_JSON = "gap_analysis.json"
PLOT_DIR = "plots"
N_LAYERS = 25

# task type per feature
REGRESSION = ["chroma", "tempo", "rhythmic_stability",
              "spectral_centroid", "spectral_contrast", "zcr"]
CLASSIFICATION = ["mode", "key"]
ALL_FEATURES = REGRESSION + CLASSIFICATION

R2_GAP_THRESHOLD = 0.40
ACC_GAP_THRESHOLD = 0.65


def _load_aligned():
    mert = torch.load(MERT_PATH, map_location="cpu", weights_only=False)
    theory = torch.load(THEORY_PATH, map_location="cpu", weights_only=False)
    ids = sorted(set(mert.keys()) & set(theory.keys()))
    if not ids:
        raise ValueError("No overlapping IDs between MERT embeddings and "
                         "music-theory ground truth. Run extract_music_theory.py first.")
    emb = np.stack([np.asarray(mert[i], dtype=np.float32) for i in ids])  # (N,25,1024)
    feats = {}
    for f in ALL_FEATURES:
        feats[f] = np.stack([theory[i][f].numpy().reshape(-1) for i in ids])  # (N,d)
    print(f"  Aligned {len(ids)} songs (MERT ∩ music-theory) | emb {emb.shape}")
    return emb, feats


def _probe(X, y, is_clf):
    """One linear probe on a fixed split. Returns scalar score."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if is_clf:
        model = LogisticRegression(max_iter=1000)
        model.fit(Xtr, ytr.ravel())
        return float(accuracy_score(yte.ravel(), model.predict(Xte)))
    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    return float(r2_score(yte, model.predict(Xte)))


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()  # enables --help without side effects
    emb, feats = _load_aligned()
    pooled = emb.mean(axis=1)  # (N,1024)

    results = {}
    score_grid = np.zeros((len(ALL_FEATURES), N_LAYERS))  # for heatmap

    for fi, feat in enumerate(ALL_FEATURES):
        is_clf = feat in CLASSIFICATION
        y = feats[feat]
        if is_clf:
            y = y.astype(int)

        layer_scores = []
        for L in range(N_LAYERS):
            layer_scores.append(_probe(emb[:, L, :], y, is_clf))
        layer_scores = np.array(layer_scores)
        score_grid[fi] = layer_scores

        best_layer = int(np.argmax(layer_scores))
        best_score = float(layer_scores[best_layer])
        pooled_score = _probe(pooled, y, is_clf)

        if is_clf:
            results[feat] = {"best_layer": best_layer,
                             "best_acc": round(best_score, 4),
                             "pooled_acc": round(pooled_score, 4)}
        else:
            results[feat] = {"best_layer": best_layer,
                             "best_r2": round(best_score, 4),
                             "pooled_r2": round(pooled_score, 4)}
        print(f"  {feat:<20} best L{best_layer:<2} "
              f"{'acc' if is_clf else 'R²'}={best_score:.4f} | "
              f"pooled={pooled_score:.4f}")

    with open(RESULTS_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n✅ Saved {RESULTS_JSON}")

    # ── A3: gap analysis ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}\n  GAP ANALYSIS  (R²<{R2_GAP_THRESHOLD} or acc<{ACC_GAP_THRESHOLD} = gap)\n{'=' * 60}")
    gap_features, gap_detail = [], {}
    for feat in ALL_FEATURES:
        is_clf = feat in CLASSIFICATION
        r = results[feat]
        score = r["best_acc"] if is_clf else r["best_r2"]
        thr = ACC_GAP_THRESHOLD if is_clf else R2_GAP_THRESHOLD
        is_gap = score < thr
        metric = "acc" if is_clf else "R²"
        verdict = "GAP — candidate for Phase B" if is_gap else "MERT captures well"
        print(f"  {feat:<20} {metric}={score:.2f} → [{verdict}]")
        gap_detail[feat] = {"metric": metric, "best_score": score,
                            "threshold": thr, "is_gap": is_gap}
        if is_gap:
            gap_features.append(feat)

    with open(GAP_JSON, "w") as fh:
        json.dump({"gap_features": gap_features, "detail": gap_detail}, fh, indent=2)
    print(f"\n  gap_features = {gap_features}")
    print(f"✅ Saved {GAP_JSON}  → drives Phase B extension")

    # ── plots ───────────────────────────────────────────────────────────────
    os.makedirs(PLOT_DIR, exist_ok=True)

    best_vals = [results[f].get("best_r2", results[f].get("best_acc")) for f in ALL_FEATURES]
    pooled_vals = [results[f].get("pooled_r2", results[f].get("pooled_acc")) for f in ALL_FEATURES]
    x = np.arange(len(ALL_FEATURES))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, best_vals, 0.4, label="Best single layer", color="#4C9BE8")
    ax.bar(x + 0.2, pooled_vals, 0.4, label="Mean-pooled all layers", color="#F5A623")
    ax.axhline(R2_GAP_THRESHOLD, ls="--", c="red", alpha=0.6, label=f"gap thr (R²={R2_GAP_THRESHOLD})")
    ax.set_xticks(x); ax.set_xticklabels(ALL_FEATURES, rotation=30, ha="right")
    ax.set_ylabel("Score (R² or accuracy)")
    ax.set_title("MERT Music-Theory Probing — Best Layer vs Mean-Pooled")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "music_theory_probing_summary.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(score_grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(ALL_FEATURES))); ax.set_yticklabels(ALL_FEATURES)
    ax.set_xticks(np.arange(N_LAYERS)); ax.set_xticklabels(range(N_LAYERS), fontsize=7)
    ax.set_xlabel("MERT Transformer Layer"); ax.set_title("Probing Score per Feature × Layer")
    fig.colorbar(im, ax=ax, label="Score (R² / accuracy)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "music_theory_probing_heatmap.png"), dpi=150)
    plt.close()
    print(f"✅ Saved plots → {PLOT_DIR}/")


if __name__ == "__main__":
    main()
