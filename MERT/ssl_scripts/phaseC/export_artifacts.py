"""
export_artifacts.py — Phase C thesis artifacts + naive baseline (read-only)
===========================================================================
Generates, without training or modifying anything:

  1. NAIVE BASELINE retrieval — average-pool the LAST MERT layer (no fine-tuning,
     no SupCR), L2-normalize, and run Precision@k + Silhouette on that raw space.
     Printed side-by-side with the SupCR-optimized indexes (prototypes*.npy).
  2. t-SNE comparison — baseline raw-MERT space vs fine-tuned latent space,
     coloured by Russell quadrant → artifacts/tsne_baseline_vs_finetuned.png
  3. WeightedLayerFusion bar chart — softmaxed learned layer weights, top-3
     annotated → artifacts/layer_fusion_weights.png

All outputs go to phaseC/artifacts/.

Run from phaseC/ with the venv active:
  python export_artifacts.py \
      --feat_path ../phaseB/pmemo_mert_all_layers.pt \
      --csv_path  /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
      --eda_dir   /datasets/emotions/PMEmo2019/EDA \
      --finetuned_index prototypes_dual.npy
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PHASEB = os.path.abspath(os.path.join(_THIS, "..", "phaseB"))
if os.path.isdir(_PHASEB):
    sys.path.insert(0, _PHASEB)

import torch                                       # noqa: E402
from data_utils import load_pmemo_data             # noqa: E402
from eda_loader import load_eda_for_ids            # noqa: E402
from evaluator import evaluate_retrieval           # noqa: E402

ART = os.path.join(_THIS, "artifacts")
QUAD_NAMES = {0: "HVHA (Happy)", 1: "HVLA (Calm)", 2: "LVHA (Angry)", 3: "LVLA (Sad)"}
QUAD_COLORS = {0: "#E84C4C", 1: "#4C9BE8", 2: "#9B4CE8", 3: "#4CB07A"}


def quad_of(a, v):
    if v >= 0.5 and a >= 0.5: return 0
    if v >= 0.5 and a < 0.5:  return 1
    if v < 0.5 and a >= 0.5:  return 2
    return 3


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description="Phase C artifacts + naive baseline")
    ap.add_argument("--feat_path", default="../phaseB/pmemo_mert_all_layers.pt")
    ap.add_argument("--csv_path",
                    default="/datasets/emotions/PMEmo2019/annotations/static_annotations.csv")
    ap.add_argument("--eda_dir", default="/datasets/emotions/PMEmo2019/EDA")
    ap.add_argument("--finetuned_index", default="prototypes_dual.npy")
    ap.add_argument("--mert_index", default="prototypes.npy")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    # ── load raw MERT all-layer features + labels ───────────────────────────
    X, Y, ids = load_pmemo_data(args.feat_path, args.csv_path)   # X:(N,25,1024)
    ids = [str(int(float(i))) for i in ids]
    arousal, valence = Y[:, 0].numpy(), Y[:, 1].numpy()
    quads = np.array([quad_of(a, v) for a, v in zip(arousal, valence)])

    eda = (load_eda_for_ids(ids, args.eda_dir)
           if args.eda_dir and os.path.isdir(args.eda_dir)
           else np.zeros((len(ids), 7), np.float32))

    # ── 1. NAIVE BASELINE: average-pooled LAST MERT layer, no training ──────
    naive = X[:, -1, :].numpy().astype(np.float32)               # (N,1024)
    naive /= np.linalg.norm(naive, axis=1, keepdims=True).clip(1e-8)
    naive_index = {
        "latents": naive, "music_ids": np.array([int(i) for i in ids], np.int32),
        "arousal": arousal, "valence": valence,
        "pred_arousal": arousal, "pred_valence": valence,
        "eda_feats": eda, "layer_weights": None,
    }
    print("\n" + "=" * 64)
    print("  1. NAIVE BASELINE — raw last-layer MERT (no fine-tuning, no SupCR)")
    print("=" * 64)
    naive_res = evaluate_retrieval(naive_index, k_values=(5, 10, 20), emotion_threshold=0.20)

    # trained indexes for comparison
    trained = {}
    for label, path in [("MERT (SupCR)", args.mert_index),
                        ("Dual-SSL (SupCR)", args.finetuned_index)]:
        if os.path.exists(path):
            idx = np.load(path, allow_pickle=True).item()
            print(f"\n  Re-scoring trained index: {label} ({path})")
            trained[label] = evaluate_retrieval(idx, k_values=(5, 10, 20), emotion_threshold=0.20)

    print("\n" + "=" * 64)
    print("  RETRIEVAL COMPARISON (Precision@k / Silhouette)")
    print("=" * 64)
    print(f"  {'Space':<22} {'P@5':>7} {'P@10':>7} {'P@20':>7} {'Silh':>8}")
    print(f"  {'-'*54}")
    rows = [("Naive last-layer MERT", naive_res)] + list(trained.items())
    for name, r in rows:
        print(f"  {name:<22} {r.get('Precision@5',0):>7.4f} {r.get('Precision@10',0):>7.4f} "
              f"{r.get('Precision@20',0):>7.4f} {r.get('Silhouette',0):>8.4f}")
    print("  → If trained ≈ naive, raw MERT already retrieves well; if trained >")
    print("    naive, SupCR fine-tuning measurably tightened the emotion space.")

    # ── 2. t-SNE: baseline raw MERT vs fine-tuned latents ───────────────────
    from sklearn.manifold import TSNE
    print("\n  2. Computing t-SNE (baseline vs fine-tuned)...")
    ft_path = args.finetuned_index if os.path.exists(args.finetuned_index) else args.mert_index
    ft_latents = np.load(ft_path, allow_pickle=True).item()["latents"]

    def tsne(x):
        return TSNE(n_components=2, perplexity=30, max_iter=1000,
                    random_state=42, init="pca").fit_transform(x)

    base_2d, ft_2d = tsne(naive), tsne(ft_latents)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, pts, title in [(axes[0], base_2d, "Baseline: raw last-layer MERT (untrained)"),
                           (axes[1], ft_2d, f"Fine-tuned latent space ({os.path.basename(ft_path)})")]:
        for q in range(4):
            m = quads == q
            ax.scatter(pts[m, 0], pts[m, 1], s=14, alpha=0.7,
                       c=QUAD_COLORS[q], label=QUAD_NAMES[q])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=9)
    plt.suptitle("t-SNE of MERT space before vs after SupCR fine-tuning "
                 "(coloured by Russell quadrant)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(ART, "tsne_baseline_vs_finetuned.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"     💾 {p}")

    # ── 3. WeightedLayerFusion bar chart (softmaxed, honest) ────────────────
    mert_idx = np.load(args.mert_index, allow_pickle=True).item() \
        if os.path.exists(args.mert_index) else None
    raw_w = mert_idx.get("layer_weights") if mert_idx else None
    if raw_w is not None:
        w = softmax(np.asarray(raw_w))                            # logits → proportions
        n = len(w)
        top3 = np.argsort(w)[::-1][:3]
        ent = float(-(w * np.log(w + 1e-12)).sum())
        colors = ["#E84C4C" if i in top3 else "#9DB7CF" for i in range(n)]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.bar(range(n), w, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(1.0 / n, ls="--", c="gray", alpha=0.7, label=f"uniform = 1/{n} = {1/n:.4f}")
        ax.set_xlabel("MERT transformer layer"); ax.set_ylabel("Softmax fusion weight")
        ax.set_xticks(range(n)); ax.set_xticklabels(range(n), fontsize=7)
        ax.set_title(f"Learned WeightedLayerFusion weights (single-MERT)  |  "
                     f"top-3 = {top3.tolist()}  |  entropy {ent:.4f} / max {np.log(n):.4f}",
                     fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        p = os.path.join(ART, "layer_fusion_weights.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"\n  3. Layer-fusion bar chart → {p}")
        print(f"     top-3 layers = {top3.tolist()} | entropy {ent:.4f}/{np.log(n):.4f} "
              f"(near-uniform → marginal dominance; fusion-collapse finding)")
    else:
        print("\n  3. ⚠️  No layer_weights in MERT index — skipped bar chart.")

    print(f"\n✅ Artifacts in {ART}/")


if __name__ == "__main__":
    main()
