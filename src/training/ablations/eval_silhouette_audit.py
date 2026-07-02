"""
eval_silhouette_audit.py — resolve the Silhouette contradiction (Conflict 1)
=============================================================================
The reports quote Silhouette ≈ 0 (single-MERT, loss ablation, *Euclidean*) and
Silhouette = 0.255 (Enhanced, CE-sweep, *cosine*). These differ on TWO axes at
once — model AND metric — so the gap cannot be cleanly attributed.

This script computes Silhouette for BOTH models under BOTH metrics on the SAME
5-fold test splits, giving a clean 2×2 that says exactly how much of the gap is
model vs metric.

Both models trained identically to their canonical configs:
  - single-MERT : MERModel(hybrid, 25 layers), HybridLoss, balanced sampler,
                  differential optimizer (fusion 1e-2, head/reg 1e-4)
  - Enhanced    : EnhancedDualSSLModel (MERT + wav2vec2 + tempo/key), same recipe
Silhouette is computed on the held-out test-fold latents (128-d head output),
against Russell-quadrant labels, under metric ∈ {euclidean, cosine}.

Run from phaseB/:
  python eval_silhouette_audit.py
"""

from configs.config import PATHS, PHASE_B  # centralised config
import json
import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models.models import MERModel
from losses.losses import HybridLoss
from utils.data_utils import load_pmemo_data, get_emotion_quadrant
from models.models_enhanced import EnhancedDualSSLModel, gap_dim_of
from training.train_enhanced_dual import load_enhanced, GAP_JSON, THEORY_PATH

EPOCHS, LR, BATCH = PHASE_B.num_epochs, PHASE_B.learning_rate, PHASE_B.batch_size
FEAT_PATH = str(PATHS.mert_features)
W2V_PATH = str(PATHS.wav2vec_features)
CSV_PATH = str(PATHS.pmemo_annotations)


def quad_labels(Y):
    return np.array([get_emotion_quadrant(a, v) for a, v in Y.numpy()])


def balanced_sampler(Y):
    q = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    w = 1.0 / (np.bincount(q, minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[i] for i in q], dtype=torch.float), len(q))


def sil_both(latents, labels):
    """Silhouette under euclidean and cosine; guard degenerate single-class folds."""
    out = {}
    for metric in ("euclidean", "cosine"):
        try:
            out[metric] = silhouette_score(latents, labels, metric=metric)
        except Exception:
            out[metric] = float("nan")
    return out


def run_single_mert(X, Y, device):
    print("\n[single-MERT] training 5 folds …")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    eu, co = [], []
    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    for fold, (tr, te) in enumerate(kf.split(np.arange(len(X)))):
        tr_ld = DataLoader(TensorDataset(X[tr], Y[tr]), batch_size=BATCH,
                           sampler=balanced_sampler(Y[tr]))
        model = MERModel(mode="hybrid", n_layers=25, hidden_dim=1024).to(device)
        opt = torch.optim.Adam([
            {"params": model.fusion.parameters(),    "lr": 1e-2},
            {"params": model.head.parameters(),      "lr": LR},
            {"params": model.regressor.parameters(), "lr": LR},
        ], weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        model.train()
        for _ in range(EPOCHS):
            for bx, by in tr_ld:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                preds, latent = model(bx)
                loss, _ = crit(preds, latent, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
        model.eval()
        with torch.no_grad():
            _, lat = model(X[te].to(device))
        s = sil_both(lat.cpu().numpy(), quad_labels(Y[te]))
        print(f"  fold {fold+1}: euclidean={s['euclidean']:+.4f}  cosine={s['cosine']:+.4f}")
        eu.append(s["euclidean"]); co.append(s["cosine"])
    return np.array(eu), np.array(co)


def run_enhanced(Xm, Xw, Xt, Y, gd, device):
    print("\n[Enhanced] training 5 folds …")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    eu, co = [], []
    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        tr_ld = DataLoader(TensorDataset(Xm[tr], Xw[tr], Xt[tr], Y[tr]),
                           batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        model = EnhancedDualSSLModel(gap_dim=gd).to(device)
        opt = torch.optim.Adam([
            {"params": model.fusion_mert.parameters(),   "lr": 1e-2},
            {"params": model.fusion_w2v.parameters(),    "lr": 1e-2},
            {"params": model.theory_branch.parameters(), "lr": LR},
            {"params": model.head.parameters(),          "lr": LR},
            {"params": model.regressor.parameters(),     "lr": LR},
        ], weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        model.train()
        for _ in range(EPOCHS):
            for bm, bw, bt, by in tr_ld:
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
            _, lat = model(Xm[te].to(device), Xw[te].to(device), Xt[te].to(device))
        s = sil_both(lat.cpu().numpy(), quad_labels(Y[te]))
        print(f"  fold {fold+1}: euclidean={s['euclidean']:+.4f}  cosine={s['cosine']:+.4f}")
        eu.append(s["euclidean"]); co.append(s["cosine"])
    return np.array(eu), np.array(co)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Silhouette audit — model × metric (2×2)")

    X, Y, _ = load_pmemo_data(FEAT_PATH, CSV_PATH)
    sm_eu, sm_co = run_single_mert(X, Y, device)

    with open(GAP_JSON) as fh:
        gap_features = json.load(fh).get("gap_features", [])
    Xm, Xw, Xt, Ye = load_enhanced(FEAT_PATH, W2V_PATH, THEORY_PATH, CSV_PATH, gap_features)
    gd = gap_dim_of(gap_features)
    en_eu, en_co = run_enhanced(Xm, Xw, Xt, Ye, gd, device)

    print("\n" + "=" * 64)
    print("  SILHOUETTE 2×2 — model × metric (5-fold test-latent mean ± std)")
    print("=" * 64)
    print(f"{'Model':<16}{'Euclidean':>20}{'Cosine':>20}")
    print(f"{'single-MERT':<16}{sm_eu.mean():>10.4f} ± {sm_eu.std():.3f}"
          f"{en_co.mean()*0:>0}{sm_co.mean():>10.4f} ± {sm_co.std():.3f}")
    print(f"{'Enhanced':<16}{en_eu.mean():>10.4f} ± {en_eu.std():.3f}"
          f"{'':>0}{en_co.mean():>10.4f} ± {en_co.std():.3f}")
    print("\n  Interpretation keys:")
    print(f"   • model effect (cosine):  Enhanced {en_co.mean():.3f} − single-MERT {sm_co.mean():.3f} "
          f"= {en_co.mean()-sm_co.mean():+.3f}")
    print(f"   • metric effect (Enhanced): cosine {en_co.mean():.3f} − euclidean {en_eu.mean():.3f} "
          f"= {en_co.mean()-en_eu.mean():+.3f}")
    print(f"   • historic loss-ablation single-MERT euclidean ≈ 0 (analyze.py) — cross-check above")


if __name__ == "__main__":
    main()
