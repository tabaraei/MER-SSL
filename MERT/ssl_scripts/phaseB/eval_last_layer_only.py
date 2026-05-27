"""
eval_last_layer_only.py — ablation: MERT LAST LAYER ONLY vs full WeightedLayerFusion
=====================================================================================
Tests whether the WeightedLayerFusion over all 25 MERT layers actually helps
emotion prediction, vs simply taking the last layer (layer 24, mean-pooled over
time, as in the MERT paper's default "use the last hidden state" recipe).

Identical setup to the MERT-only baseline (same HybridLoss, differential
optimizer, balanced sampler, 5-fold CV) — the ONLY difference is the input:
  - MERT-only (reference, ours): (N, 25, 1024) → softmax-fused → (N, 1024)
  - This run:                    (N,  1, 1024) → identity        → (N, 1024)

If R²/CCC drop here, the fusion is empirically justified.

Run from phaseB/:
  python eval_last_layer_only.py
"""

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models import MERModel
from losses import HybridLoss, CCCLoss
from data_utils import load_pmemo_data, get_emotion_quadrant, quadrant_r2_breakdown

FEAT_PATH = "pmemo_mert_all_layers.pt"
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
    print(f"Device: {device} | encoder = MERT LAST-LAYER ONLY (layer 24, 1024-d)")
    X, Y, _ = load_pmemo_data(FEAT_PATH, CSV_PATH)        # (N, 25, 1024)
    # Take only the last layer; keep the layer dim so MERModel input contract holds.
    X = X[:, -1:, :].contiguous()                          # (N, 1, 1024)
    print(f"  Loaded {len(X)} songs | X={tuple(X.shape)} (last layer only)")

    criterion = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, T, P = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(X)))):
        print(f"  -- Fold {fold+1}/5 -- train={len(tr)} test={len(te)}")
        tr_ld = DataLoader(TensorDataset(X[tr], Y[tr]), batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        # n_layers=1 → WeightedLayerFusion is effectively identity for a single layer
        model = MERModel(mode="hybrid", n_layers=1, hidden_dim=1024).to(device)
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
    print(f"\n{'='*56}\n  MERT LAST-LAYER ONLY — 5-FOLD AVERAGE\n{'='*56}")
    print(f"  R2  Arousal : {r2[:,0].mean():.4f} +/- {r2[:,0].std():.4f}")
    print(f"  R2  Valence : {r2[:,1].mean():.4f} +/- {r2[:,1].std():.4f}")
    print(f"  CCC Arousal : {cc[:,0].mean():.4f}")
    print(f"  CCC Valence : {cc[:,1].mean():.4f}")
    print("\n  Per-quadrant R2:")
    for q, s in quadrant_r2_breakdown(np.vstack(T), np.vstack(P)).items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f} V={s['R2_Valence']:.3f} n={s['n']}")
    print(f"\n  COMPARISON | All-25-layer fusion (MERT-only):  A 0.6518 / V 0.5055 (CCC 0.82 / 0.74)")
    print(f"             | Last-layer only (this run):      A {r2[:,0].mean():.4f} / V {r2[:,1].mean():.4f} "
          f"(CCC {cc[:,0].mean():.2f} / {cc[:,1].mean():.2f})")


if __name__ == "__main__":
    main()
