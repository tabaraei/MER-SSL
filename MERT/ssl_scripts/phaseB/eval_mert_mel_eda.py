"""
eval_mert_mel_eda.py — NEW triple: MERT + Mel-CNN + EDA physiology
====================================================================
A third triple configuration to complete the Phase B SOTA table.
Mirrors the existing Triple (MERT + wav2vec2 + mel-CNN) but **swaps
wav2vec2 for the 7-d EDA physiological feature vector** — testing
whether physiological signal beats speech SSL as the second branch
beside MERT+mel.

Same training recipe as every other Phase B 5-fold run:
  HybridLoss (MSE 1.0 + CCC 0.5 + Rank 0.3 + SupCR 0.1), balanced
  sampler (inverse-quadrant freq), differential optimizer
  (fusion=1e-2, others=1e-4, wd=1e-3), 5-fold KFold(random_state=42),
  100 epochs, batch=32.

Architecture (inline — no changes to models.py / models_triple.py):
  Branch 1: frozen MERT     → WeightedLayerFusion(25, 1024) → 1024-d
  Branch 2: MelSpectrogramCNN (trainable)                   →  128-d
  Branch 3: EDA features (7-d) → Linear+ReLU+Dropout        →   32-d
  Concat (1024+128+32 = 1184) → head (256→128) → regressor → 2

Run from phaseB/:
  python eval_mert_mel_eda.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses import HybridLoss, CCCLoss
from data_utils import (_match_dict_to_csv, load_eda_features,
                        get_emotion_quadrant, quadrant_r2_breakdown)
from models import WeightedLayerFusion
from models_triple import MelSpectrogramCNN

EPOCHS, LR, BATCH = 100, 1e-4, 32
FEAT_PATH = "pmemo_mert_all_layers.pt"
MEL_PATH  = "pmemo_melspec.pt"
CSV_PATH  = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"
EDA_DIR   = "/datasets/emotions/PMEmo2019/EDA"


def load_mert_mel_eda(mert_path, mel_path, csv_path, eda_dir):
    """Intersect MERT ∩ mel ∩ CSV; load EDA features for matched IDs (zero-filled if missing)."""
    print(f"  Loading MERT embeddings:  {mert_path}")
    mert_data = torch.load(mert_path, map_location="cpu", weights_only=False)
    print(f"  Loading mel-spectrograms: {mel_path}")
    mel_data = torch.load(mel_path, map_location="cpu", weights_only=False)
    print(f"  Loading labels: {csv_path}")
    df = pd.read_csv(csv_path)

    ar_col = [c for c in df.columns if "arousal" in c.lower()][0]
    va_col = [c for c in df.columns if "valence" in c.lower()][0]
    id_col = [c for c in df.columns if any(x in c.lower() for x in ["music", "id"])][0]

    def _norm(col): return (col - col.min()) / (col.max() - col.min() + 1e-8)
    df[ar_col] = _norm(df[ar_col]); df[va_col] = _norm(df[va_col])

    mert_m = _match_dict_to_csv(mert_data, df, id_col)
    mel_m  = _match_dict_to_csv(mel_data,  df, id_col)
    common = [i for i in df.index if i in mert_m and i in mel_m]
    if not common:
        raise ValueError("No overlap MERT ∩ mel ∩ CSV.")

    X_mert = torch.stack([mert_m[i] for i in common]).float()
    X_mel  = torch.stack([mel_m[i]  for i in common]).float()
    df_m = df.loc[common]
    Y = torch.tensor(df_m[[ar_col, va_col]].values, dtype=torch.float32)
    ids = df_m[id_col].astype(str).tolist()

    print(f"  EDA features for {len(ids)} matched IDs:")
    X_eda_np = load_eda_features(eda_dir, ids)
    X_eda = torch.tensor(X_eda_np, dtype=torch.float32)
    print(f"  ✅ MERT+Mel+EDA matched {len(Y)} samples | "
          f"mert={tuple(X_mert.shape)} mel={tuple(X_mel.shape)} eda={tuple(X_eda.shape)}")
    return X_mert, X_mel, X_eda, Y


class MertMelEdaModel(nn.Module):
    def __init__(self, mert_layers=25, mert_dim=1024, mel_dim=128,
                 eda_dim=7, eda_proj_dim=32, bottleneck=128, dropout=0.4):
        super().__init__()
        self.fusion_mert = WeightedLayerFusion(mert_layers, mert_dim)
        self.mel_cnn = MelSpectrogramCNN(out_dim=mel_dim, dropout=dropout)
        self.eda_proj = nn.Sequential(
            nn.Linear(eda_dim, eda_proj_dim), nn.ReLU(), nn.Dropout(dropout))
        concat = mert_dim + mel_dim + eda_proj_dim
        self.head = nn.Sequential(
            nn.Linear(concat, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, bottleneck), nn.LayerNorm(bottleneck), nn.ReLU())
        self.regressor = nn.Linear(bottleneck, 2)

    def forward(self, x_mert, x_mel, x_eda):
        fm = self.fusion_mert(x_mert)
        fc = self.mel_cnn(x_mel)
        fe = self.eda_proj(x_eda)
        fused = torch.cat([fm, fc, fe], dim=1)
        latent = self.head(fused)
        latent_norm = F.normalize(latent, dim=1)
        return self.regressor(latent_norm), latent_norm


def balanced_sampler(Y):
    q = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    w = 1.0 / (np.bincount(q, minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[i] for i in q], dtype=torch.float), len(q))


def get_optimizer(model, base_lr):
    return torch.optim.Adam([
        {"params": model.fusion_mert.parameters(), "lr": 1e-2},
        {"params": model.mel_cnn.parameters(),     "lr": base_lr},
        {"params": model.eda_proj.parameters(),    "lr": base_lr},
        {"params": model.head.parameters(),        "lr": base_lr},
        {"params": model.regressor.parameters(),   "lr": base_lr},
    ], weight_decay=1e-3)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | NEW Triple: MERT + Mel-CNN + EDA")

    X_mert, X_mel, X_eda, Y = load_mert_mel_eda(FEAT_PATH, MEL_PATH, CSV_PATH, EDA_DIR)

    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, T_all, P_all = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"  ── Fold {fold+1}/5 ──")
        tr_ds = TensorDataset(X_mert[tr], X_mel[tr], X_eda[tr], Y[tr])
        te_ds = TensorDataset(X_mert[te], X_mel[te], X_eda[te], Y[te])
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

        model = MertMelEdaModel().to(device)
        opt = get_optimizer(model, LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        model.train()
        for _ in range(EPOCHS):
            for b_m, b_c, b_e, b_y in tr_ld:
                b_m, b_c, b_e, b_y = b_m.to(device), b_c.to(device), b_e.to(device), b_y.to(device)
                opt.zero_grad()
                preds, latent = model(b_m, b_c, b_e)
                loss, _ = crit(preds, latent, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

        model.eval()
        P, T = [], []
        ccc_eval = CCCLoss()
        with torch.no_grad():
            for b_m, b_c, b_e, b_y in te_ld:
                p, _ = model(b_m.to(device), b_c.to(device), b_e.to(device))
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
    print(f"\n{'='*64}\n  MERT + MEL-CNN + EDA — 5-FOLD AVERAGE\n{'='*64}")
    print(f"  R²  Arousal : {mr[0]:.4f} ± {sr[0]:.4f}")
    print(f"  R²  Valence : {mr[1]:.4f} ± {sr[1]:.4f}")
    print(f"  CCC Arousal : {mc[0]:.4f}")
    print(f"  CCC Valence : {mc[1]:.4f}")
    print("\n  Per-quadrant R²:")
    for q, s in quadrant_r2_breakdown(np.vstack(T_all), np.vstack(P_all)).items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f}  V={s['R2_Valence']:.3f}  n={s['n']}")
    print("\n  Reference comparisons (5-fold means):")
    print(f"    MERT-only             : A 0.6518 / V 0.5055")
    print(f"    MERT + EDA            : A 0.6738 / V 0.5075")
    print(f"    Spec-only (MERT+Mel)  : A 0.7069 / V 0.5709")
    print(f"    Triple (MERT+w2v+Mel) : A 0.7023 / V 0.5758")
    print(f"    Enhanced (MERT+w2v+TK): A 0.7182 / V 0.5686")
    print(f"    THIS RUN (MERT+Mel+EDA): A {mr[0]:.4f} / V {mr[1]:.4f}")


if __name__ == "__main__":
    main()
