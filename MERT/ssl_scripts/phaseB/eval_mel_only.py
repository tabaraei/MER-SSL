"""
eval_mel_only.py — single-encoder baseline: Mel-spectrogram CNN ALONE
======================================================================
Trains the trainable MelSpectrogramCNN on its own — no MERT, no
wav2vec2, no EDA, no music theory. Acts as a baseline isolating the
contribution of a non-SSL spectrogram encoder so the SOTA table has a
clean "what does a shallow CNN-on-spectrograms achieve?" reference.

Same training recipe as every other Phase B run (HybridLoss + balanced
sampler + 5-fold CV; lr=1e-4 throughout since there is no fusion module
to lift). 100 epochs, batch=32, KFold(random_state=42).

Architecture:
  MelSpectrogramCNN (~110K params) → 128-d embed → head (256→128) →
  L2-normalised latent → linear regressor → (arousal, valence)

Run from phaseB/:
  python eval_mel_only.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses import HybridLoss, CCCLoss
from data_utils import _match_dict_to_csv, get_emotion_quadrant, quadrant_r2_breakdown
from models_triple import MelSpectrogramCNN

EPOCHS, LR, BATCH = 100, 1e-4, 32
MEL_PATH = "pmemo_melspec.pt"
CSV_PATH = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"


def load_mel_only(mel_path, csv_path):
    print(f"  Loading mel-spectrograms: {mel_path}")
    mel_data = torch.load(mel_path, map_location="cpu", weights_only=False)
    df = pd.read_csv(csv_path)
    ar_col = [c for c in df.columns if "arousal" in c.lower()][0]
    va_col = [c for c in df.columns if "valence" in c.lower()][0]
    id_col = [c for c in df.columns if any(x in c.lower() for x in ["music", "id"])][0]
    def _norm(c): return (c - c.min()) / (c.max() - c.min() + 1e-8)
    df[ar_col] = _norm(df[ar_col]); df[va_col] = _norm(df[va_col])
    mel_m = _match_dict_to_csv(mel_data, df, id_col)
    common = [i for i in df.index if i in mel_m]
    X_mel = torch.stack([mel_m[i] for i in common]).float()
    df_m = df.loc[common]
    Y = torch.tensor(df_m[[ar_col, va_col]].values, dtype=torch.float32)
    print(f"  ✅ Mel ∩ CSV: {len(Y)} samples | mel shape {tuple(X_mel.shape)}")
    return X_mel, Y


class MelOnlyModel(nn.Module):
    def __init__(self, mel_dim=128, bottleneck=128, dropout=0.4):
        super().__init__()
        self.mel_cnn = MelSpectrogramCNN(out_dim=mel_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(mel_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, bottleneck), nn.LayerNorm(bottleneck), nn.ReLU())
        self.regressor = nn.Linear(bottleneck, 2)

    def forward(self, x_mel):
        fc = self.mel_cnn(x_mel)
        latent = self.head(fc)
        latent_norm = F.normalize(latent, dim=1)
        return self.regressor(latent_norm), latent_norm


def balanced_sampler(Y):
    q = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    w = 1.0 / (np.bincount(q, minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[i] for i in q], dtype=torch.float), len(q))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | MEL-CNN ALONE (no SSL, no EDA, no theory)")
    X_mel, Y = load_mel_only(MEL_PATH, CSV_PATH)

    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, T_all, P_all = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"  ── Fold {fold+1}/5 ──")
        tr_ld = DataLoader(TensorDataset(X_mel[tr], Y[tr]),
                           batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        te_ld = DataLoader(TensorDataset(X_mel[te], Y[te]), batch_size=BATCH, shuffle=False)
        model = MelOnlyModel().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
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
        P, T = [], []
        ccc_eval = CCCLoss()
        with torch.no_grad():
            for bx, by in te_ld:
                p, _ = model(bx.to(device))
                P.append(p.cpu()); T.append(by)
        y_pred, y_true = np.vstack(P), np.vstack(T)
        r2 = r2_score(y_true, y_pred, multioutput="raw_values")
        ccc = ccc_eval.compute_ccc_scores(torch.tensor(y_pred), torch.tensor(y_true))
        print(f"     R² A={r2[0]:.4f} V={r2[1]:.4f} | "
              f"CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f}")
        fold_r2.append(r2); fold_ccc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]])
        T_all.append(y_true); P_all.append(y_pred)

    fr = np.array(fold_r2); fc = np.array(fold_ccc)
    mr, sr = fr.mean(0), fr.std(0); mc = fc.mean(0)
    print(f"\n{'='*64}\n  MEL-CNN ALONE — 5-FOLD AVERAGE\n{'='*64}")
    print(f"  R²  Arousal : {mr[0]:.4f} ± {sr[0]:.4f}")
    print(f"  R²  Valence : {mr[1]:.4f} ± {sr[1]:.4f}")
    print(f"  CCC Arousal : {mc[0]:.4f}")
    print(f"  CCC Valence : {mc[1]:.4f}")
    print("\n  Per-quadrant R²:")
    for q, s in quadrant_r2_breakdown(np.vstack(T_all), np.vstack(P_all)).items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f}  V={s['R2_Valence']:.3f}  n={s['n']}")
    print("\n  Reference comparisons (5-fold means):")
    print(f"    wav2vec2-only         : A 0.6225 / V 0.4825 (speech SSL alone)")
    print(f"    MERT-only             : A 0.6518 / V 0.5055 (music SSL alone)")
    print(f"    THIS RUN (Mel-CNN alone): A {mr[0]:.4f} / V {mr[1]:.4f}")


if __name__ == "__main__":
    main()
