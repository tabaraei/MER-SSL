"""
analyze.py — Post-Training Analysis Script
==========================================
Run AFTER mainB.py to generate all thesis figures and analysis tables.

What this produces:
  1. Layer weight bar chart (for thesis Section: Hybrid Defense)
  2. Emotion quadrant confusion plot
  3. Embedding geometry comparison: baseline vs hybrid (silhouette scores)
  4. k-NN retrieval precision@k comparison
  5. Printed CCC score table for literature comparison

Usage:
    # Analyze a saved hybrid model checkpoint:
    python analyze.py --checkpoint hybrid_model.pt --feat_path pmemo_mert_all_layers.pt --csv_path /path/to/annotations.csv

    # Or run all analyses on fresh models (trains baseline + hybrid, then analyzes):
    python analyze.py --mode full --feat_path pmemo_mert_all_layers.pt --csv_path /path/to/annotations.csv
"""

import torch
import argparse
import numpy as np
import os
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from models import MERModel, analyze_layer_weights, plot_layer_weights
from losses import CCCLoss, HybridLoss
from data_utils import load_pmemo_data, quadrant_r2_breakdown, add_quadrant_labels


# =============================================================================
# 1. Embedding Geometry: Silhouette Score Comparison
# =============================================================================

def compare_embedding_geometry(
    baseline_model: MERModel,
    hybrid_model:   MERModel,
    X:              torch.Tensor,
    Y:              torch.Tensor,
    device:         torch.device,
):
    """
    Computes silhouette scores in baseline vs hybrid embedding spaces.
    Higher silhouette = better emotion cluster separation.

    This is your primary empirical defense for the Hybrid architecture.
    """
    quadrant_labels = add_quadrant_labels(Y).numpy()

    def get_latents(model, X):
        model.eval()
        latents = []
        loader = DataLoader(TensorDataset(X), batch_size=32)
        with torch.no_grad():
            for (bx,) in loader:
                _, lat = model(bx.to(device))
                latents.append(lat.cpu().numpy())
        return np.vstack(latents)

    baseline_latents = get_latents(baseline_model, X)
    hybrid_latents   = get_latents(hybrid_model, X)

    # Silhouette on quadrant labels
    sil_baseline = silhouette_score(baseline_latents, quadrant_labels)
    sil_hybrid   = silhouette_score(hybrid_latents,   quadrant_labels)

    print("\n📐  Embedding Geometry — Silhouette Score (higher = better clustering)")
    print(f"  Baseline (Layer 24)  : {sil_baseline:.4f}")
    print(f"  Hybrid (Fusion+SupCR): {sil_hybrid:.4f}")
    print(f"  Improvement          : {sil_hybrid - sil_baseline:+.4f}")

    return {
        "silhouette_baseline": sil_baseline,
        "silhouette_hybrid":   sil_hybrid,
    }


# =============================================================================
# 2. k-NN Retrieval Precision@k
# =============================================================================

def compare_retrieval_quality(
    baseline_model: MERModel,
    hybrid_model:   MERModel,
    X:              torch.Tensor,
    Y:              torch.Tensor,
    device:         torch.device,
    k_values:       list = [5, 10],
    emotion_threshold: float = 0.20,
):
    """
    Compares retrieval quality: for each query, retrieves k nearest neighbors
    in latent space, then checks if those neighbors are emotionally similar
    (within emotion_threshold in V-A space).

    Precision@k = fraction of retrieved neighbors that are emotionally similar.

    This directly validates Phase C: the hybrid space should retrieve better
    emotional neighbors than raw Layer 24 embeddings.
    """
    def get_latents(model, X):
        model.eval()
        latents = []
        loader = DataLoader(TensorDataset(X), batch_size=32)
        with torch.no_grad():
            for (bx,) in loader:
                _, lat = model(bx.to(device))
                latents.append(lat.cpu().numpy())
        return np.vstack(latents)

    baseline_latents = get_latents(baseline_model, X)
    hybrid_latents   = get_latents(hybrid_model, X)
    y_np = Y.numpy()

    # Ground-truth V-A distance matrix
    va_diff  = y_np[:, None, :] - y_np[None, :, :]   # (N, N, 2)
    va_dist  = np.linalg.norm(va_diff, axis=2)        # (N, N)

    results = {}

    for name, latents in [("Baseline (L24)", baseline_latents), ("Hybrid (SupCR)", hybrid_latents)]:
        knn = NearestNeighbors(n_neighbors=max(k_values) + 1, metric="cosine")
        knn.fit(latents)
        _, indices = knn.kneighbors(latents)

        for k in k_values:
            precisions = []
            for i in range(len(latents)):
                neighbors = indices[i, 1:k+1]  # exclude self
                # Check how many are emotionally close
                emo_close = (va_dist[i, neighbors] < emotion_threshold).mean()
                precisions.append(emo_close)
            avg_p = np.mean(precisions)
            key = f"{name}_P@{k}"
            results[key] = avg_p
            print(f"  {name:25s} | Precision@{k} = {avg_p:.4f}")

    print()
    for k in k_values:
        bl = results[f"Baseline (L24)_P@{k}"]
        hy = results[f"Hybrid (SupCR)_P@{k}"]
        print(f"  P@{k} improvement (Hybrid - Baseline): {hy - bl:+.4f}")

    return results


# =============================================================================
# 3. Full Evaluation Metrics Table
# =============================================================================

def full_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Model"):
    """
    Prints a complete metrics table:
    - R² per dimension
    - CCC per dimension
    - Quadrant breakdown
    """
    ccc_fn = CCCLoss()
    pred_t   = torch.tensor(y_pred, dtype=torch.float32)
    target_t = torch.tensor(y_true, dtype=torch.float32)
    ccc_scores = ccc_fn.compute_ccc_scores(pred_t, target_t)

    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    print(f"\n{'='*60}")
    print(f"  Full Metrics — {label}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Arousal':>10} {'Valence':>10}")
    print(f"  {'-'*45}")
    print(f"  {'R² Score':<25} {r2[0]:>10.4f} {r2[1]:>10.4f}")
    print(f"  {'CCC Score':<25} {ccc_scores['CCC_Arousal']:>10.4f} {ccc_scores['CCC_Valence']:>10.4f}")
    print(f"  {'-'*45}")

    breakdown = quadrant_r2_breakdown(y_true, y_pred)
    print(f"\n  Quadrant Breakdown:")
    for q_name, scores in breakdown.items():
        if scores["R2_Arousal"] is None:
            print(f"    {q_name:<30} n={scores['n']} (insufficient samples)")
        else:
            print(f"    {q_name:<30} A={scores['R2_Arousal']:.3f}  V={scores['R2_Valence']:.3f}  n={scores['n']}")
    print(f"{'='*60}\n")


# =============================================================================
# 4. Quadrant Visualization
# =============================================================================

def plot_va_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "va_scatter.png"):
    """
    Plots true vs predicted V-A coordinates, colored by quadrant.
    Useful for thesis visualization.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    quadrant_colors = ["#2196F3", "#4CAF50", "#F44336", "#9C27B0"]
    quadrant_names  = ["HVHA (Happy)", "HVLA (Calm)", "LVHA (Angry)", "LVLA (Sad)"]

    from data_utils import add_quadrant_labels
    quads = add_quadrant_labels(torch.tensor(y_true)).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in zip(axes, [y_true, y_pred], ["Ground Truth", "Predicted"]):
        for q_idx in range(4):
            mask = quads == q_idx
            ax.scatter(
                data[mask, 1], data[mask, 0],
                c=quadrant_colors[q_idx],
                label=quadrant_names[q_idx],
                alpha=0.6, s=20,
            )
        # Quadrant dividers
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Valence", fontsize=11)
        ax.set_ylabel("Arousal", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

    plt.suptitle("V-A Space: Ground Truth vs Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 V-A scatter plot saved to: {save_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_path",  type=str, default="pmemo_mert_all_layers.pt")
    parser.add_argument("--csv_path",   type=str,
                        default="/datasets/emotions/PMEmo2019/annotations/static_annotations.csv")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to saved hybrid model .pt checkpoint")
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Epochs to train fresh models (if no checkpoint)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻  Device: {device}")

    # Load data
    print("\n📂  Loading data...")
    X, Y = load_pmemo_data(args.feat_path, args.csv_path)

    # Train/test split for analysis (not k-fold here — we want fixed embeddings)
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)
    X_tr, X_te = X[tr_idx], X[te_idx]
    Y_tr, Y_te = Y[tr_idx], Y[te_idx]

    train_loader_bl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=32, shuffle=True)
    train_loader_hy = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=32, shuffle=True)
    test_loader     = DataLoader(TensorDataset(X_te, Y_te), batch_size=32)

    # ── Train or load models ──────────────────────────────────────────────────
    baseline_model = MERModel(mode="baseline").to(device)
    hybrid_model   = MERModel(mode="hybrid").to(device)

    criterion_bl = HybridLoss(w_mse=1.0, w_ccc=0.0, w_rank=0.0, w_supcr=0.0, use_supcr=False)
    criterion_hy = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n📦  Loading hybrid checkpoint from {args.checkpoint}")
        hybrid_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        # Still train baseline for fair comparison
        print("🏃  Training baseline for comparison...")
        opt_bl = torch.optim.Adam(baseline_model.parameters(), lr=1e-4, weight_decay=1e-2)
        for _ in range(args.epochs):
            for bx, by in train_loader_bl:
                opt_bl.zero_grad()
                p, lat = baseline_model(bx.to(device))
                loss, _ = criterion_bl(p, lat, by.to(device))
                loss.backward()
                opt_bl.step()
    else:
        print(f"\n🏃  Training baseline ({args.epochs} epochs)...")
        opt_bl = torch.optim.Adam(baseline_model.parameters(), lr=1e-4, weight_decay=1e-2)
        for ep in range(args.epochs):
            for bx, by in train_loader_bl:
                opt_bl.zero_grad()
                p, lat = baseline_model(bx.to(device))
                loss, _ = criterion_bl(p, lat, by.to(device))
                loss.backward()
                opt_bl.step()
            if ep % 10 == 0:
                print(f"  Baseline epoch {ep+1}/{args.epochs}")

        print(f"\n🏃  Training hybrid ({args.epochs} epochs)...")
        opt_hy = torch.optim.Adam(hybrid_model.parameters(), lr=1e-4, weight_decay=1e-2)
        sch_hy = torch.optim.lr_scheduler.CosineAnnealingLR(opt_hy, T_max=args.epochs)
        for ep in range(args.epochs):
            for bx, by in train_loader_hy:
                opt_hy.zero_grad()
                p, lat = hybrid_model(bx.to(device))
                loss, _ = criterion_hy(p, lat, by.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), 1.0)
                opt_hy.step()
            sch_hy.step()
            if ep % 10 == 0:
                print(f"  Hybrid epoch {ep+1}/{args.epochs}")

    # ── Evaluate and report ───────────────────────────────────────────────────
    def get_preds(model, loader):
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for bx, by in loader:
                p, _ = model(bx.to(device))
                all_p.append(p.cpu().numpy())
                all_y.append(by.numpy())
        return np.vstack(all_y), np.vstack(all_p)

    y_true_bl, y_pred_bl = get_preds(baseline_model, test_loader)
    y_true_hy, y_pred_hy = get_preds(hybrid_model,   test_loader)

    full_metrics_table(y_true_bl, y_pred_bl, "Baseline (Layer 24 + MSE)")
    full_metrics_table(y_true_hy, y_pred_hy, "Hybrid (Fusion + SupCR + CCC + Rank)")

    # ── Geometry comparison ───────────────────────────────────────────────────
    print("\n📐  Running embedding geometry analysis...")
    compare_embedding_geometry(baseline_model, hybrid_model, X, Y, device)

    # ── Retrieval comparison ──────────────────────────────────────────────────
    print("\n🔍  Running k-NN retrieval comparison...")
    compare_retrieval_quality(baseline_model, hybrid_model, X, Y, device)

    # ── Layer weights ─────────────────────────────────────────────────────────
    print("\n📊  Hybrid layer weight analysis...")
    analyze_layer_weights(hybrid_model, save_path="layer_weights_final.npy")
    plot_layer_weights(hybrid_model, save_path="layer_weights_final.png")

    # ── V-A scatter plots ─────────────────────────────────────────────────────
    plot_va_scatter(y_true_hy, y_pred_hy, save_path="va_scatter_hybrid.png")

    print("\n✅  All analyses complete.")
    print("    Generated files:")
    print("      layer_weights_final.png  — for thesis Section 'Hybrid Defense'")
    print("      va_scatter_hybrid.png    — for thesis Section 'MER Results'")
    print("      layer_weights_final.npy  — raw weights for further analysis")
