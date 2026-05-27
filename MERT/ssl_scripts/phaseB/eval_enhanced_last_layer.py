"""
eval_enhanced_last_layer.py — fusion ablation on the BEST multi-encoder config
==============================================================================
Ablation: does the WeightedLayerFusion (over 25 MERT + 13 wav2vec2 layers) help
the *Enhanced* model (MERT + wav2vec2 + music-theory gap features), or does
using just the last layer of each SSL encoder match it — as it did on MERT-only?

Identical setup to `train_enhanced_dual.py` (same HybridLoss, balanced sampler,
differential optimizer, 5-fold CV, 100 epochs, batch=32, KFold random_state=42).
ONLY difference: both SSL inputs are sliced to the last layer:
  - MERT branch:    (N, 25, 1024) → (N, 1, 1024)
  - wav2vec2 branch:(N, 13,  768) → (N, 1,  768)
  - theory branch:  unchanged

Model built with mert_layers=1, w2v_layers=1 — WeightedLayerFusion is then
effectively identity for the one layer.

Pass mark: same pre-registered rule as eval_imbalance_ablation.py — last-layer
"wins" iff it beats Enhanced fusion-baseline by >1 fold-std on both axes, or
>2 fold-std on either axis.

Run from phaseB/:
  python eval_enhanced_last_layer.py
"""

import json
import os
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses import HybridLoss, CCCLoss
from data_utils import get_emotion_quadrant, quadrant_r2_breakdown
from models_enhanced import EnhancedDualSSLModel, gap_dim_of
from train_enhanced_dual import load_enhanced, GAP_JSON, THEORY_PATH

EPOCHS, LR, BATCH = 100, 1e-4, 32
FEAT_PATH = "pmemo_mert_all_layers.pt"
W2V_PATH = "pmemo_wav2vec_all_layers.pt"
CSV_PATH = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"

# Cited Enhanced fusion-baseline (5-fold, train_enhanced_dual.py historic run).
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Enhanced LAST-LAYER-ONLY ablation (MERT + w2v2 + theory)")

    with open(GAP_JSON) as fh:
        gap = json.load(fh)
    gap_features = gap.get("gap_features", [])
    if not gap_features:
        raise SystemExit("Phase A found no gaps; Enhanced is not applicable.")

    X_mert, X_w2v, X_theory, Y = load_enhanced(
        FEAT_PATH, W2V_PATH, THEORY_PATH, CSV_PATH, gap_features)
    gd = gap_dim_of(gap_features)

    # ── slice both SSL inputs to last layer only ──
    X_mert = X_mert[:, -1:, :].contiguous()      # (N, 1, 1024)
    X_w2v  = X_w2v[:, -1:, :].contiguous()        # (N, 1,  768)
    print(f"  After slicing: X_mert={tuple(X_mert.shape)} | X_w2v={tuple(X_w2v.shape)} | "
          f"X_theory={tuple(X_theory.shape)} | gap_features={gap_features}")

    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, T_all, P_all = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"  ── Fold {fold+1}/5 ──")
        tr_ds = TensorDataset(X_mert[tr], X_w2v[tr], X_theory[tr], Y[tr])
        te_ds = TensorDataset(X_mert[te], X_w2v[te], X_theory[te], Y[te])
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

        # n_layers=1 for both SSL branches → WeightedLayerFusion is identity
        model = EnhancedDualSSLModel(
            gap_dim=gd,
            mert_layers=1, mert_dim=1024,
            w2v_layers=1,  w2v_dim=768,
        ).to(device)
        opt = get_optimizer(model, LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        model.train()
        for _ in range(EPOCHS):
            for b_m, b_w, b_t, b_y in tr_ld:
                b_m, b_w, b_t, b_y = b_m.to(device), b_w.to(device), b_t.to(device), b_y.to(device)
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
        print(f"     R² A={r2[0]:.4f} V={r2[1]:.4f} | CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f}")
        fold_r2.append(r2); fold_ccc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]])
        T_all.append(y_true); P_all.append(y_pred)

    fr = np.array(fold_r2); fc = np.array(fold_ccc)
    mr, sr = fr.mean(0), fr.std(0); mc = fc.mean(0)
    print(f"\n{'='*64}\n  ENHANCED LAST-LAYER-ONLY — 5-FOLD AVERAGE\n{'='*64}")
    print(f"  R²  Arousal : {mr[0]:.4f} ± {sr[0]:.4f}")
    print(f"  R²  Valence : {mr[1]:.4f} ± {sr[1]:.4f}")
    print(f"  CCC Arousal : {mc[0]:.4f}")
    print(f"  CCC Valence : {mc[1]:.4f}")

    print("\n  Per-quadrant R²:")
    for q, s in quadrant_r2_breakdown(np.vstack(T_all), np.vstack(P_all)).items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f}  V={s['R2_Valence']:.3f}  n={s['n']}")

    print(f"\n  COMPARISON")
    print(f"  {'Model':<32} {'A R²':>8} {'V R²':>8} {'CCC A':>8} {'CCC V':>8}")
    print(f"  {'-'*64}")
    print(f"  {'Enhanced (25-layer fusion)':<32} "
          f"{ENHANCED_BASELINE['A']:>8.4f} {ENHANCED_BASELINE['V']:>8.4f} "
          f"{ENHANCED_BASELINE['CCC_A']:>8.4f} {ENHANCED_BASELINE['CCC_V']:>8.4f}")
    print(f"  {'Enhanced (last-layer only)':<32} "
          f"{mr[0]:>8.4f} {mr[1]:>8.4f} {mc[0]:>8.4f} {mc[1]:>8.4f}")
    print(f"  {'Δ (last - fusion)':<32} "
          f"{mr[0]-ENHANCED_BASELINE['A']:>+8.4f} {mr[1]-ENHANCED_BASELINE['V']:>+8.4f} "
          f"{mc[0]-ENHANCED_BASELINE['CCC_A']:>+8.4f} {mc[1]-ENHANCED_BASELINE['CCC_V']:>+8.4f}")
    print(f"  (fold-std on R² A = {sr[0]:.4f}, R² V = {sr[1]:.4f}; "
          f"|Δ| > std → outside fold-noise on that axis)")


if __name__ == "__main__":
    main()
