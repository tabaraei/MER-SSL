"""
eval_enhanced_quadrant_ce.py — auxiliary quadrant-CE head trade-off ablation
============================================================================
Answers the viva question "can you make the t-SNE clusters look discrete?"
quantitatively: adds an auxiliary 4-way quadrant classification head to the
Enhanced model and sweeps the CE-loss weight λ ∈ {0, 0.1, 0.5, 1.0}.

Pre-registered hypothesis (from method-menu in chat):
  - Silhouette will RISE with λ — auxiliary CE forces cluster compactness.
  - R²/CCC will DROP — model optimises boundary-crossing instead of
    fine-grained V-A position.
  - Minority per-quadrant R² will NOT improve (data-floor invariant).

This ablation characterises the *trade-off* so the thesis can defend the
continuous representation as a deliberate choice rather than a limitation.

Architecture (inline wrapper, no changes to models_enhanced.py):
  EnhancedDualSSLModel → (preds, latent)
  + auxiliary head: Linear(128 → 4) on the SAME latent → quadrant logits
  Combined loss = HybridLoss(preds, latent, y) + λ · CrossEntropy(logits, q)

Same training recipe everywhere else: 5-fold KFold(42), 100 epochs,
batch=32, balanced sampler, differential optimizer.

Run from phaseB/:
  python eval_enhanced_quadrant_ce.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses import HybridLoss, CCCLoss
from data_utils import get_emotion_quadrant, quadrant_r2_breakdown
from models_enhanced import EnhancedDualSSLModel, gap_dim_of
from train_enhanced_dual import load_enhanced, GAP_JSON, THEORY_PATH

EPOCHS, LR, BATCH = 100, 1e-4, 32
LAMBDAS = [0.0, 0.1, 0.5, 1.0]
FEAT_PATH = "pmemo_mert_all_layers.pt"
W2V_PATH = "pmemo_wav2vec_all_layers.pt"
CSV_PATH = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"

ENHANCED_BASELINE = {"A": 0.7182, "V": 0.5686, "CCC_A": 0.8345, "CCC_V": 0.7259}


class EnhancedWithCE(nn.Module):
    """Enhanced + auxiliary 4-way quadrant classification head on the same latent."""
    def __init__(self, gap_dim, bottleneck=128, **kwargs):
        super().__init__()
        self.base = EnhancedDualSSLModel(gap_dim=gap_dim, **kwargs)
        self.ce_head = nn.Linear(bottleneck, 4)

    def forward(self, x_mert, x_w2v, x_theory):
        preds, latent = self.base(x_mert, x_w2v, x_theory)
        logits = self.ce_head(latent)
        return preds, latent, logits


def quadrant_labels(Y):
    return torch.tensor([get_emotion_quadrant(a, v) for a, v in Y.numpy()], dtype=torch.long)


def balanced_sampler(Y):
    q = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    w = 1.0 / (np.bincount(q, minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[i] for i in q], dtype=torch.float), len(q))


def get_optimizer(model, base_lr):
    return torch.optim.Adam([
        {"params": model.base.fusion_mert.parameters(),   "lr": 1e-2},
        {"params": model.base.fusion_w2v.parameters(),    "lr": 1e-2},
        {"params": model.base.theory_branch.parameters(), "lr": base_lr},
        {"params": model.base.head.parameters(),          "lr": base_lr},
        {"params": model.base.regressor.parameters(),     "lr": base_lr},
        {"params": model.ce_head.parameters(),            "lr": base_lr},
    ], weight_decay=1e-3)


def run_one(lam, X_mert, X_w2v, X_theory, Y, gap_dim, device):
    print(f"\n=== λ = {lam} ===")
    crit_reg = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    crit_ce  = nn.CrossEntropyLoss()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, fold_sil, T_all, P_all = [], [], [], [], []
    ccc_eval = CCCLoss()

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        Q_tr, Q_te = quadrant_labels(Y[tr]), quadrant_labels(Y[te])
        tr_ds = TensorDataset(X_mert[tr], X_w2v[tr], X_theory[tr], Y[tr], Q_tr)
        te_ds = TensorDataset(X_mert[te], X_w2v[te], X_theory[te], Y[te], Q_te)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

        model = EnhancedWithCE(gap_dim=gap_dim).to(device)
        opt = get_optimizer(model, LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        model.train()
        for _ in range(EPOCHS):
            for b_m, b_w, b_t, b_y, b_q in tr_ld:
                b_m, b_w, b_t = b_m.to(device), b_w.to(device), b_t.to(device)
                b_y, b_q = b_y.to(device), b_q.to(device)
                opt.zero_grad()
                preds, latent, logits = model(b_m, b_w, b_t)
                l_reg, _ = crit_reg(preds, latent, b_y)
                l_ce = crit_ce(logits, b_q) if lam > 0 else torch.tensor(0.0, device=device)
                loss = l_reg + lam * l_ce
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

        model.eval()
        P, T, Lat, Q = [], [], [], []
        with torch.no_grad():
            for b_m, b_w, b_t, b_y, b_q in te_ld:
                p, lat, _ = model(b_m.to(device), b_w.to(device), b_t.to(device))
                P.append(p.cpu()); T.append(b_y); Lat.append(lat.cpu()); Q.append(b_q)
        y_pred, y_true = np.vstack(P), np.vstack(T)
        latents = np.vstack(Lat); quads = np.concatenate(Q)
        r2 = r2_score(y_true, y_pred, multioutput="raw_values")
        ccc = ccc_eval.compute_ccc_scores(torch.tensor(y_pred), torch.tensor(y_true))
        try:
            sil = silhouette_score(latents, quads, metric="cosine")
        except Exception as e:
            sil = float("nan")
        print(f"  fold {fold+1}: R² A={r2[0]:.4f} V={r2[1]:.4f} | "
              f"CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f} | "
              f"Silhouette={sil:.4f}")
        fold_r2.append(r2); fold_ccc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]])
        fold_sil.append(sil); T_all.append(y_true); P_all.append(y_pred)

    fr = np.array(fold_r2); fc = np.array(fold_ccc); fs = np.array(fold_sil)
    pq = quadrant_r2_breakdown(np.vstack(T_all), np.vstack(P_all))
    print(f"  → λ={lam}: R² A {fr[:,0].mean():.4f}±{fr[:,0].std():.3f}  "
          f"V {fr[:,1].mean():.4f}±{fr[:,1].std():.3f} | "
          f"CCC A {fc[:,0].mean():.3f} V {fc[:,1].mean():.3f} | "
          f"Silhouette {fs.mean():.4f}±{fs.std():.4f}")
    return {
        "lambda": lam,
        "r2_A_mean": fr[:,0].mean(), "r2_A_std": fr[:,0].std(),
        "r2_V_mean": fr[:,1].mean(), "r2_V_std": fr[:,1].std(),
        "ccc_A": fc[:,0].mean(), "ccc_V": fc[:,1].mean(),
        "silhouette_mean": fs.mean(), "silhouette_std": fs.std(),
        "pq": pq,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Enhanced + auxiliary quadrant-CE head sweep")
    print(f"  Lambdas: {LAMBDAS}")

    with open(GAP_JSON) as fh:
        gap_features = json.load(fh).get("gap_features", [])
    if not gap_features:
        raise SystemExit("Phase A found no gaps; Enhanced not applicable.")

    X_mert, X_w2v, X_theory, Y = load_enhanced(
        FEAT_PATH, W2V_PATH, THEORY_PATH, CSV_PATH, gap_features)
    gd = gap_dim_of(gap_features)

    results = [run_one(lam, X_mert, X_w2v, X_theory, Y, gd, device) for lam in LAMBDAS]

    print("\n" + "="*88)
    print("  AUXILIARY QUADRANT-CE HEAD ABLATION — Enhanced model, 5-fold")
    print("="*88)
    print(f"{'λ':>6}  {'R² A':>16}  {'R² V':>16}  {'CCC A':>7}  {'CCC V':>7}  {'Silhouette':>16}")
    for r in results:
        print(f"{r['lambda']:>6.2f}  "
              f"{r['r2_A_mean']:.4f}±{r['r2_A_std']:.3f}  "
              f"{r['r2_V_mean']:.4f}±{r['r2_V_std']:.3f}  "
              f"{r['ccc_A']:>7.3f}  {r['ccc_V']:>7.3f}  "
              f"{r['silhouette_mean']:.4f}±{r['silhouette_std']:.4f}")

    print("\n  Per-quadrant R² breakdown (minority quadrants only):")
    for r in results:
        print(f"\n  λ={r['lambda']}")
        for q in ["HVHA (Happy)", "HVLA (Calm)", "LVHA (Angry)", "LVLA (Sad)"]:
            s = r['pq'].get(q, {})
            if s.get("R2_Arousal") is not None:
                print(f"     {q}: A={s['R2_Arousal']:+.3f}  V={s['R2_Valence']:+.3f}  n={s['n']}")

    print(f"\n  Reference: historic Enhanced (no CE) = "
          f"R² A {ENHANCED_BASELINE['A']} / V {ENHANCED_BASELINE['V']} | "
          f"CCC A {ENHANCED_BASELINE['CCC_A']} / V {ENHANCED_BASELINE['CCC_V']}")


if __name__ == "__main__":
    main()
