"""
extra_metrics.py — prototype-activation accuracy + random-chance baseline (read-only)
=====================================================================================
Two quantitative checks that turn qualitative claims into hard numbers:

  1. Prototype-activation accuracy — % of songs whose best-match quadrant CENTROID
     equals their true ground-truth quadrant. Quantifies the ante-hoc 4-prototype
     classification feature. Centroid for a song's own quadrant is leave-one-out
     (the song never contributes to the centroid it is scored against).

  2. Random-chance Precision baseline — expected fraction of a random neighbour
     falling within the 0.20 V-A radius, to substantiate "above chance" claims.

Read-only: loads an existing index, trains/changes nothing.

  python extra_metrics.py --index_path prototypes_dual.npy
  python extra_metrics.py --index_path prototypes.npy
"""

import argparse
import numpy as np

QUAD = {0: "HVHA (Happy)", 1: "HVLA (Calm)", 2: "LVHA (Angry)", 3: "LVLA (Sad)"}


def quad_of(a, v):
    if v >= 0.5 and a >= 0.5: return 0
    if v >= 0.5 and a < 0.5:  return 1
    if v < 0.5 and a >= 0.5:  return 2
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", default="prototypes_dual.npy")
    ap.add_argument("--radius", type=float, default=0.20)
    args = ap.parse_args()

    idx = np.load(args.index_path, allow_pickle=True).item()
    lat = idx["latents"].astype(np.float64)            # already L2-normalized
    lat /= np.linalg.norm(lat, axis=1, keepdims=True).clip(1e-12)
    a, v = idx["arousal"], idx["valence"]
    q = np.array([quad_of(ai, vi) for ai, vi in zip(a, v)])
    N = len(lat)

    # per-quadrant sums/counts for leave-one-out centroids
    codes = [0, 1, 2, 3]
    sums = {c: lat[q == c].sum(0) for c in codes}
    counts = {c: int((q == c).sum()) for c in codes}

    correct = 0
    pq_tot = {c: 0 for c in codes}
    pq_cor = {c: 0 for c in codes}
    for i in range(N):
        qi = q[i]
        sims = {}
        for c in codes:
            if counts[c] == 0:
                continue
            if c == qi:                                 # leave-one-out
                cen = (sums[c] - lat[i]) / max(counts[c] - 1, 1)
            else:
                cen = sums[c] / counts[c]
            cen = cen / (np.linalg.norm(cen) + 1e-12)
            sims[c] = float(lat[i] @ cen)
        best = max(sims, key=sims.get)
        pq_tot[qi] += 1
        if best == qi:
            correct += 1
            pq_cor[qi] += 1

    acc = correct / N
    print(f"\n=== Prototype-Activation Accuracy ({args.index_path}) ===")
    print(f"  Overall: {acc:.4f}  ({correct}/{N} songs' best-match centroid = true quadrant)")
    print(f"  Majority-class (always-HVHA) baseline: {counts[0]/N:.4f}")
    print("  Per-quadrant recall:")
    for c in codes:
        if pq_tot[c]:
            print(f"    {QUAD[c]:<14} {pq_cor[c]}/{pq_tot[c]} = {pq_cor[c]/pq_tot[c]:.3f}")

    # random-chance Precision baseline
    va = np.stack([a, v], axis=1)
    d = np.linalg.norm(va[:, None, :] - va[None, :, :], axis=2)
    within = d < args.radius
    np.fill_diagonal(within, False)
    chance = (within.sum(1) / (N - 1)).mean()
    print(f"\n=== Random-chance Precision baseline (radius {args.radius}) ===")
    print(f"  Expected precision of a RANDOM neighbour: {chance:.4f}")
    print(f"  (compare to retrieval Precision@5 to substantiate 'above chance')")


if __name__ == "__main__":
    main()
