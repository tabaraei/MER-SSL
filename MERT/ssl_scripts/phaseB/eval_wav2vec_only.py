"""
eval_wav2vec_only.py — wav2vec2-only emotion prediction (apples-to-apples vs MERT-only)
=======================================================================================
Trains the SAME hybrid model used for MERT-only, but on wav2vec2 features instead
(13 layers x 768-dim). Same 4-part HybridLoss, differential optimizer, balanced
sampler, and 5-fold CV — so the only difference vs the MERT-only baseline is the
encoder. Reports R2 and CCC for valence and arousal.

Run from phaseB/ with the venv active:
  python eval_wav2vec_only.py
"""

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models import MERModel
from losses import HybridLoss, CCCLoss
from data_utils import load_pmemo_data, get_emotion_quadrant, quadrant_r2_breakdown

W2V_PATH = "pmemo_wav2vec_all_layers.pt"
CSV_PATH = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"
EPOCHS, LR, BATCH = 100, 1e-4, 32


def balanced_sampler(Y):
    quads = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    w = 1.0 / (np.bincount(quads, minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[q] for q in quads], dtype=torch.float), len(quads))


def get_optimizer(model, base_lr):
    return torch.optim.Adam([
        {"params": model.fusion.parameters(),    "lr": 1e-2},
        {"params": model.head.parameters(),      "lr": base_lr},
        {"params": model.regressor.parameters(), "lr": base_lr},
    ], weight_decay=1e-3)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | encoder=wav2vec2-only (13x768)")
    X, Y, _ = load_pmemo_data(W2V_PATH, CSV_PATH)          # X: (N,13,768)
    print(f"  Loaded {len(X)} songs | X={tuple(X.shape)}")

    criterion = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, T, P = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(X)))):
        print(f"  -- Fold {fold+1}/5 -- train={len(tr)} test={len(te)}")
        tr_ld = DataLoader(TensorDataset(X[tr], Y[tr]), batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        model = MERModel(mode="hybrid", n_layers=13, hidden_dim=768).to(device)
        opt = get_optimizer(model, LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        model.train()
        for _ in range(EPOCHS):
            for bx, by in tr_ld:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                preds, latent = model(bx)
                loss, _ = criterion(preds, latent, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
        model.eval()
        with torch.no_grad():
            preds, _ = model(X[te].to(device))
        yt, yp = Y[te].numpy(), preds.cpu().numpy()
        r2 = r2_score(yt, yp, multioutput="raw_values")
        ccc = CCCLoss().compute_ccc_scores(torch.tensor(yp), torch.tensor(yt))
        print(f"     R2 A={r2[0]:.4f} V={r2[1]:.4f} | CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f}")
        fold_r2.append(r2); fold_ccc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]])
        T.append(yt); P.append(yp)

    r2 = np.array(fold_r2); cc = np.array(fold_ccc)
    print(f"\n{'='*56}\n  WAV2VEC2-ONLY — 5-FOLD AVERAGE\n{'='*56}")
    print(f"  R2  Arousal : {r2[:,0].mean():.4f} +/- {r2[:,0].std():.4f}")
    print(f"  R2  Valence : {r2[:,1].mean():.4f} +/- {r2[:,1].std():.4f}")
    print(f"  CCC Arousal : {cc[:,0].mean():.4f}")
    print(f"  CCC Valence : {cc[:,1].mean():.4f}")
    print("\n  Per-quadrant R2:")
    for q, s in quadrant_r2_breakdown(np.vstack(T), np.vstack(P)).items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f} V={s['R2_Valence']:.3f} n={s['n']}")
    print(f"\n  COMPARISON | MERT-only: A 0.6518 / V 0.5055 (CCC 0.82/0.74)")
    print(f"             | wav2vec2 : A {r2[:,0].mean():.4f} / V {r2[:,1].mean():.4f} "
          f"(CCC {cc[:,0].mean():.2f}/{cc[:,1].mean():.2f})")


if __name__ == "__main__":
    main()
