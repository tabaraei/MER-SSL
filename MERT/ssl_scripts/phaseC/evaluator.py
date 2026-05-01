"""
evaluator.py — Retrieval Quality Evaluation
============================================
Computes two metrics over the full index:

  Precision@k  — fraction of top-k neighbors that are emotionally close
                 (V-A Euclidean distance < emotion_threshold).
                 Emotionally close = within a 0.20 radius in the [0,1]² V-A plane.

  Silhouette   — how well Russell's four quadrants cluster in latent space.
                 A score near +1 means quadrants are well separated;
                 near 0 means overlap; negative means misclassification.

These two metrics together assess both local (Precision@k) and global
(Silhouette) structure of the learned emotion-aware latent space.
"""

import numpy as np


def evaluate_retrieval(index, k_values=(5, 10, 20), emotion_threshold=0.20):
    """
    Args:
        index             : loaded index dict from VectorIndexBuilder
        k_values          : k values to evaluate Precision@k at
        emotion_threshold : V-A Euclidean distance threshold for a true positive

    Prints results and returns a dict of metric names → values.
    """
    from sklearn.metrics import silhouette_score

    latents = index["latents"]
    arousal = index["arousal"]
    valence = index["valence"]
    N       = len(latents)

    va      = np.stack([arousal, valence], axis=1)
    va_dist = np.linalg.norm(va[:, None, :] - va[None, :, :], axis=2)
    sim_mat = latents @ latents.T

    print("\n📐  Retrieval Evaluation")
    print("=" * 50)
    results = {}

    for k in k_values:
        precisions = []
        for i in range(N):
            row        = sim_mat[i].copy()
            row[i]     = -1
            top_k_idx  = np.argsort(row)[::-1][:k]
            true_pos   = (va_dist[i, top_k_idx] < emotion_threshold).mean()
            precisions.append(true_pos)
        avg_p = float(np.mean(precisions))
        results[f"Precision@{k}"] = avg_p
        print(f"  Precision@{k:2d} : {avg_p:.4f}")

    quads = []
    for a, v in zip(arousal, valence):
        mid = 0.5
        if   v >= mid and a >= mid: quads.append(0)  # HVHA
        elif v >= mid and a <  mid: quads.append(1)  # HVLA
        elif v <  mid and a >= mid: quads.append(2)  # LVHA
        else:                       quads.append(3)  # LVLA
    quads = np.array(quads)

    if len(np.unique(quads)) > 1:
        sil = silhouette_score(latents, quads, metric="cosine")
        results["Silhouette"] = float(sil)
        print(f"  Silhouette    : {sil:.4f}")
    print("=" * 50)
    return results
