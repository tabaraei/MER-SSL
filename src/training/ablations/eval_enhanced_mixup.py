"""
eval_enhanced_mixup.py — feature-space mixup on the BEST multi-encoder config
=============================================================================
Tests whether a simple in-batch mixup augmentation (Zhang et al. 2017) on
already-extracted SSL features lifts the minority-quadrant floor that the
imbalance ablation (Step 17) attributed to too few minority-quadrant examples.

Identical setup to `train_enhanced_dual.py` (same HybridLoss, balanced sampler,
differential optimizer, 5-fold CV, 100 epochs, batch=32, KFold random_state=42).
ONLY difference: each training batch is mixup-augmented in feature space.

Mixup: for each batch, sample λ ~ Beta(α, α) (α=0.4 → U-shaped, most mixes
mild) and a random permutation of indices; mixed sample = λ·x_i + (1−λ)·x_π(i).
The same λ is applied to all three inputs (MERT, w2v2, theory) and labels.
No mixup at evaluation.

Pass mark (pre-registered):
  winner iff R² beats Enhanced fusion baseline by >1 fold-std on BOTH axes,
  OR by >2 fold-std on either axis, OR lifts minority per-quadrant R²
  from <0 to ≥0.

Run from phaseB/:
  python eval_enhanced_mixup.py
"""

from configs.config import PATHS, PHASE_B  # centralised config
import json
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses.losses import HybridLoss, CCCLoss
from utils.data_utils import get_emotion_quadrant, quadrant_r2_breakdown
from models.models_enhanced import EnhancedDualSSLModel, gap_dim_of
from training.train_enhanced_dual import load_enhanced, GAP_JSON, THEORY_PATH

EPOCHS, LR, BATCH = PHASE_B.num_epochs, PHASE_B.learning_rate, PHASE_B.batch_size
ALPHA = 0.4  # Beta(0.4, 0.4) — standard mixup default (U-shaped, mild mixes)
FEAT_PATH = str(PATHS.mert_features)
W2V_PATH = str(PATHS.wav2vec_features)
CSV_PATH = str(PATHS.pmemo_annotations)

ENHANCED_BASELINE = {"A": 0.7182, "V": 0.5686, "CCC_A": 0.8345, "CCC_V": 0.7259}


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


def mixup_batch(b_m, b_w, b_t, b_y, alpha=ALPHA):
    """Feature-space mixup. Same λ across all inputs and labels."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(b_m.size(0), device=b_m.device)
    b_m = lam * b_m + (1 - lam) * b_m[idx]
    b_w = lam * b_w + (1 - lam) * b_w[idx]
    b_t = lam * b_t + (1 - lam) * b_t[idx]
    b_y = lam * b_y + (1 - lam) * b_y[idx]
    return b_m, b_w, b_t, b_y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Enhanced + feature-space mixup (α={ALPHA})")

    with open(GAP_JSON) as fh:
        gap_features = json.load(fh).get("gap_features", [])
    if not gap_features:
        raise SystemExit("Phase A found no gaps; Enhanced not applicable.")

    X_mert, X_w2v, X_theory, Y = load_enhanced(
        FEAT_PATH, W2V_PATH, THEORY_PATH, CSV_PATH, gap_features)
    gd = gap_dim_of(gap_features)
    print(f"  X_mert={tuple(X_mert.shape)} | X_w2v={tuple(X_w2v.shape)} | "
          f"X_theory={tuple(X_theory.shape)} | gap={gap_features}")

    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, T_all, P_all = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"  ── Fold {fold+1}/5 ──")
        tr_ds = TensorDataset(X_mert[tr], X_w2v[tr], X_theory[tr], Y[tr])
        te_ds = TensorDataset(X_mert[te], X_w2v[te], X_theory[te], Y[te])
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

        model = EnhancedDualSSLModel(gap_dim=gd).to(device)
        opt = get_optimizer(model, LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        model.train()
        for _ in range(EPOCHS):
            for b_m, b_w, b_t, b_y in tr_ld:
                b_m, b_w, b_t, b_y = b_m.to(device), b_w.to(device), b_t.to(device), b_y.to(device)
                b_m, b_w, b_t, b_y = mixup_batch(b_m, b_w, b_t, b_y)
                opt.zero_grad()
                preds, latent = model(b_m, b_w, b_t)
                loss, _ = crit(preds, latent, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

        model.eval()
        P, T = [], []
        ccc_eval = CCCLoss()
        with torch.no_grad():
            for b_m, b_w, b_t, b_y in te_ld:
                p, _ = model(b_m.to(device), b_w.to(device), b_t.to(device))
                P.append(p.cpu()); T.append(b_y)
        y_pred, y_true = np.vstack(P), np.vstack(T)
        r2 = r2_score(y_true, y_pred, multioutput="raw_values")
        ccc = ccc_eval.compute_ccc_scores(torch.tensor(y_pred), torch.tensor(y_true))
        print(f"     R² A={r2[0]:.4f} V={r2[1]:.4f} | "
              f"CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f}")
        fold_r2.append(r2); fold_ccc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]])
        T_all.append(y_true); P_all.append(y_pred)

    fr = np.array(fold_r2); fc = np.array(fold_ccc)
    mr, sr = fr.mean(0), fr.std(0); mc = fc.mean(0)
    print(f"\n{'='*64}\n  ENHANCED + MIXUP (α={ALPHA}) — 5-FOLD AVERAGE\n{'='*64}")
    print(f"  R²  Arousal : {mr[0]:.4f} ± {sr[0]:.4f}")
    print(f"  R²  Valence : {mr[1]:.4f} ± {sr[1]:.4f}")
    print(f"  CCC Arousal : {mc[0]:.4f}")
    print(f"  CCC Valence : {mc[1]:.4f}")

    print("\n  Per-quadrant R²:")
    for q, s in quadrant_r2_breakdown(np.vstack(T_all), np.vstack(P_all)).items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f}  V={s['R2_Valence']:.3f}  n={s['n']}")

    print(f"\n  COMPARISON vs Enhanced fusion baseline:")
    print(f"  {'Model':<30} {'R² A':>10} {'R² V':>10} {'CCC A':>8} {'CCC V':>8}")
    print(f"  {'-'*68}")
    print(f"  {'Enhanced (no mixup)':<30} "
          f"{ENHANCED_BASELINE['A']:>10.4f} {ENHANCED_BASELINE['V']:>10.4f} "
          f"{ENHANCED_BASELINE['CCC_A']:>8.4f} {ENHANCED_BASELINE['CCC_V']:>8.4f}")
    print(f"  {'Enhanced + mixup':<30} "
          f"{mr[0]:>10.4f} {mr[1]:>10.4f} {mc[0]:>8.4f} {mc[1]:>8.4f}")
    print(f"  {'Δ (mixup − baseline)':<30} "
          f"{mr[0]-ENHANCED_BASELINE['A']:>+10.4f} {mr[1]-ENHANCED_BASELINE['V']:>+10.4f} "
          f"{mc[0]-ENHANCED_BASELINE['CCC_A']:>+8.4f} {mc[1]-ENHANCED_BASELINE['CCC_V']:>+8.4f}")
    print(f"  (fold-std A={sr[0]:.4f} V={sr[1]:.4f}; |Δ|>std → outside fold-noise on that axis)")


if __name__ == "__main__":
    main()
