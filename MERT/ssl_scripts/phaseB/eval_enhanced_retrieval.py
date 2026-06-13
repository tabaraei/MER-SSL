"""
eval_enhanced_retrieval.py — held-out Precision@k for the Enhanced benchmark model
==================================================================================
The deployed Phase C system retrieves with the Enhanced model (MERT + wav2vec2 +
cyclic-key theory), but the reported Precision@5 ≈ 0.58 was measured on the
single-MERT / Dual latent space. This script measures the Enhanced model's
retrieval quality under the IDENTICAL out-of-sample protocol used for those
numbers (`phaseC/evaluate_latent_space.py`):

  5-fold KFold(random_state=42). Each fold: train Enhanced (cyclic key) on the
  train split, encode the HELD-OUT test split → test-fold latent (the song is
  encoded by a model that never trained on it). Assemble all 767 out-of-sample
  latents, then Precision@k = for each song, fraction of its top-k cosine
  neighbours within a 0.20 V-A Euclidean radius.

Comparable baselines (same protocol, from `04_results_and_sota.md` §3):
  Naive raw MERT (untrained) : 0.485
  MERT (SupCR)               : 0.576
  Dual-SSL (SupCR)           : 0.585
  random-chance baseline     : 0.276

Run from phaseB/:
  python eval_enhanced_retrieval.py
"""

import json
import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses import HybridLoss
from data_utils import get_emotion_quadrant
from models_enhanced import EnhancedDualSSLModel, gap_dim_of
from train_enhanced_dual import load_enhanced, GAP_JSON, THEORY_PATH

EPOCHS, LR, BATCH = 100, 1e-4, 32
RADIUS = 0.20
FEAT_PATH = "pmemo_mert_all_layers.pt"
W2V_PATH = "pmemo_wav2vec_all_layers.pt"
CSV_PATH = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"

BASELINES = {"Naive raw MERT (untrained)": 0.485, "MERT (SupCR)": 0.576,
             "Dual-SSL (SupCR)": 0.585, "random chance": 0.276}


def balanced_sampler(Y):
    q = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    w = 1.0 / (np.bincount(q, minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[i] for i in q], dtype=torch.float), len(q))


def get_optimizer(model, base_lr):
    return torch.optim.Adam([
        {"params": model.fusion_mert.parameters(),   "lr": 1e-2},
        {"params": model.fusion_w2v.parameters(),    "lr": 1e-2},
        {"params": model.theory_branch.parameters(), "lr": base_lr},
        {"params": model.head.parameters(),          "lr": base_lr},
        {"params": model.regressor.parameters(),     "lr": base_lr},
    ], weight_decay=1e-3)


def precision_at_k(latents, arousal, valence, k_values=(5, 10, 20), radius=RADIUS):
    """EXACT copy of phaseC/evaluator.evaluate_retrieval Precision@k."""
    va = np.stack([arousal, valence], axis=1)
    va_dist = np.linalg.norm(va[:, None, :] - va[None, :, :], axis=2)
    sim_mat = latents @ latents.T
    out = {}
    N = len(latents)
    for k in k_values:
        precisions = []
        for i in range(N):
            row = sim_mat[i].copy(); row[i] = -1
            top = np.argsort(row)[::-1][:k]
            precisions.append((va_dist[i, top] < radius).mean())
        out[k] = float(np.mean(precisions))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Enhanced held-out retrieval (out-of-sample test-fold latents)")
    with open(GAP_JSON) as fh:
        gap_features = json.load(fh).get("gap_features", [])
    Xm, Xw, Xt, Y = load_enhanced(FEAT_PATH, W2V_PATH, THEORY_PATH, CSV_PATH,
                                  gap_features, cyclic_key=True)
    gd = gap_dim_of(gap_features, cyclic_key=True)

    N = len(Y)
    oos_latents = np.zeros((N, 128), dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)

    for fold, (tr, te) in enumerate(kf.split(np.arange(N))):
        print(f"  -- fold {fold+1}/5 -- train {len(tr)} encode-test {len(te)}")
        ld = DataLoader(TensorDataset(Xm[tr], Xw[tr], Xt[tr], Y[tr]),
                        batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        model = EnhancedDualSSLModel(gap_dim=gd).to(device)
        opt = get_optimizer(model, LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        model.train()
        for _ in range(EPOCHS):
            for bm, bw, bt, by in ld:
                bm, bw, bt, by = bm.to(device), bw.to(device), bt.to(device), by.to(device)
                opt.zero_grad()
                preds, latent = model(bm, bw, bt)
                loss, _ = crit(preds, latent, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
        model.eval()
        with torch.no_grad():
            _, z = model(Xm[te].to(device), Xw[te].to(device), Xt[te].to(device))
        z = z.cpu().numpy()
        oos_latents[te] = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    arousal, valence = Y[:, 0].numpy(), Y[:, 1].numpy()
    P = precision_at_k(oos_latents, arousal, valence)

    # Silhouette (cosine, 4 quadrants) — consistency check vs canonical 0.26
    quads = np.array([get_emotion_quadrant(a, v) for a, v in zip(arousal, valence)])
    sil = silhouette_score(oos_latents, quads, metric="cosine")

    print(f"\n{'='*60}\n  ENHANCED — HELD-OUT RETRIEVAL (out-of-sample)\n{'='*60}")
    print(f"  Precision@5  : {P[5]:.4f}")
    print(f"  Precision@10 : {P[10]:.4f}")
    print(f"  Precision@20 : {P[20]:.4f}")
    print(f"  Silhouette (cosine, consistency check) : {sil:.4f}")
    print(f"\n  Comparison @ Precision@5 (same out-of-sample protocol):")
    for name, val in BASELINES.items():
        print(f"    {name:<28} {val:.3f}")
    print(f"    {'Enhanced (this run)':<28} {P[5]:.3f}")
    d = P[5] - BASELINES["Dual-SSL (SupCR)"]
    print(f"\n  Δ Enhanced − Dual = {d:+.4f}  |  Δ Enhanced − random = {P[5]-0.276:+.4f}")


if __name__ == "__main__":
    main()
