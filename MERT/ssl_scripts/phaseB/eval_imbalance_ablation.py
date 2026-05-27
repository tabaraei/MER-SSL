"""
eval_imbalance_ablation.py — does penalty-based imbalance handling beat the sampler?
====================================================================================
Baseline (Step 12 in updater.md) uses WeightedRandomSampler with inverse-quadrant
frequency. Per-quadrant R² on minority quadrants (HVLA n=67, LVHA n=64, LVLA n=167)
is *negative* — the model has no signal there. Question: can loss-level reweighting
(weighted-MSE, focal-MSE) and/or sampler+loss stacking push minority R² up?

4 configs on the MERT-only backbone (25-layer fusion, single encoder, 5-fold CV,
100 epochs, batch=32, differential optimizer, HybridLoss=MSE+CCC+Rank+SupCR):

  A) sampler-only         — current baseline (re-run for fold-matched comparison)
  B) weighted-MSE only    — no sampler; per-sample MSE × inv-quadrant-freq weight
  C) sampler + weighted-MSE — both, stacked
  D) focal-MSE γ=2 + sampler — hard-example focus, no quadrant prior

Pass mark: a treatment "wins" only if its R² beats baseline by > 1 fold-std on
both axes, OR by > 2 fold-std on either axis OR meaningfully improves minority
per-quadrant R² (from <0 to >=0).

Run from phaseB/:
  python eval_imbalance_ablation.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models import MERModel
from losses import CCCLoss, RankLoss, SupCRLoss
from data_utils import load_pmemo_data, get_emotion_quadrant, quadrant_r2_breakdown

FEAT_PATH = "pmemo_mert_all_layers.pt"
CSV_PATH = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"
EPOCHS, LR, BATCH = 100, 1e-4, 32


def quadrant_weights(Y):
    """Per-sample inverse-quadrant-frequency weight, normalized so mean(w)=1."""
    quads = np.array([get_emotion_quadrant(a, v) for a, v in Y.numpy()])
    cnt = np.bincount(quads, minlength=4) + 1e-6
    inv = 1.0 / cnt
    w_per_quad = inv / inv.mean()  # so balanced data → w=1 (no scale shift)
    w = np.array([w_per_quad[q] for q in quads], dtype=np.float32)
    return torch.tensor(w), quads


class FlexLoss(nn.Module):
    """HybridLoss with optional per-sample MSE weight and optional focal factor."""
    def __init__(self, w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, focal_gamma=0.0):
        super().__init__()
        self.w_mse, self.w_ccc, self.w_rank, self.w_supcr = w_mse, w_ccc, w_rank, w_supcr
        self.focal_gamma = focal_gamma
        self.ccc = CCCLoss(); self.rank = RankLoss(); self.supcr = SupCRLoss()

    def forward(self, pred, latent, target, sample_w=None):
        sq = (pred - target).pow(2).mean(dim=1)            # (B,)
        if self.focal_gamma > 0:
            with torch.no_grad():
                fw = sq.clamp(min=1e-8).pow(self.focal_gamma / 2.0)
                fw = fw / (fw.mean() + 1e-8)                # normalise so mean=1
            sq = sq * fw
        if sample_w is not None:
            sq = sq * sample_w
        l_mse = sq.mean()
        l_ccc = self.ccc(pred, target)
        l_rank = self.rank(pred, target)
        l_supcr = self.supcr(latent, target)
        return self.w_mse*l_mse + self.w_ccc*l_ccc + self.w_rank*l_rank + self.w_supcr*l_supcr


def get_optimizer(model, base_lr):
    return torch.optim.Adam([
        {"params": model.fusion.parameters(),    "lr": 1e-2},
        {"params": model.head.parameters(),      "lr": base_lr},
        {"params": model.regressor.parameters(), "lr": base_lr},
    ], weight_decay=1e-3)


def run_one(name, X, Y, use_sampler, use_loss_weight, focal_gamma=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {name} (sampler={use_sampler}, loss_w={use_loss_weight}, focal_gamma={focal_gamma}) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, T, P = [], [], [], []
    ccc_eval = CCCLoss()

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(X)))):
        Xtr, Ytr = X[tr], Y[tr]
        w_tr, _ = quadrant_weights(Ytr)

        ds = TensorDataset(Xtr, Ytr, w_tr)
        if use_sampler:
            sampler = WeightedRandomSampler(w_tr.double(), len(w_tr))
            ld = DataLoader(ds, batch_size=BATCH, sampler=sampler)
        else:
            ld = DataLoader(ds, batch_size=BATCH, shuffle=True)

        model = MERModel(mode="hybrid", n_layers=25, hidden_dim=1024).to(device)
        opt = get_optimizer(model, LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        criterion = FlexLoss(focal_gamma=focal_gamma)

        model.train()
        for _ in range(EPOCHS):
            for bx, by, bw in ld:
                bx, by, bw = bx.to(device), by.to(device), bw.to(device)
                opt.zero_grad()
                preds, latent = model(bx)
                loss = criterion(preds, latent, by, sample_w=bw if use_loss_weight else None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            preds, _ = model(X[te].to(device))
        yt, yp = Y[te].numpy(), preds.cpu().numpy()
        r2 = r2_score(yt, yp, multioutput="raw_values")
        ccc = ccc_eval.compute_ccc_scores(torch.tensor(yp), torch.tensor(yt))
        print(f"  fold {fold+1}: R2 A={r2[0]:.4f} V={r2[1]:.4f} | CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f}")
        fold_r2.append(r2); fold_ccc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]])
        T.append(yt); P.append(yp)

    r2 = np.array(fold_r2); cc = np.array(fold_ccc)
    pq = quadrant_r2_breakdown(np.vstack(T), np.vstack(P))
    print(f"  --> R²  A: {r2[:,0].mean():.4f} ± {r2[:,0].std():.4f}")
    print(f"  --> R²  V: {r2[:,1].mean():.4f} ± {r2[:,1].std():.4f}")
    print(f"  --> CCC A: {cc[:,0].mean():.4f} | CCC V: {cc[:,1].mean():.4f}")
    for q in ["HVHA", "HVLA", "LVHA", "LVLA"]:
        s = pq.get(q, {})
        if s.get("R2_Arousal") is not None:
            print(f"       {q} (n={s['n']:3d}): A={s['R2_Arousal']:+.3f} V={s['R2_Valence']:+.3f}")
    return {
        "name": name,
        "r2_A_mean": r2[:,0].mean(), "r2_A_std": r2[:,0].std(),
        "r2_V_mean": r2[:,1].mean(), "r2_V_std": r2[:,1].std(),
        "ccc_A":     cc[:,0].mean(), "ccc_V":     cc[:,1].mean(),
        "pq": pq,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Imbalance-ablation on MERT-only (25-layer fusion)")
    X, Y, _ = load_pmemo_data(FEAT_PATH, CSV_PATH)
    print(f"  Loaded {len(X)} songs | X={tuple(X.shape)}")
    _, quads_all = quadrant_weights(Y)
    print(f"  Quadrant counts: HVHA={int((quads_all==0).sum())} HVLA={int((quads_all==1).sum())} "
          f"LVHA={int((quads_all==2).sum())} LVLA={int((quads_all==3).sum())}")

    results = []
    results.append(run_one("A: sampler-only (baseline)",     X, Y, True,  False, 0.0))
    results.append(run_one("B: weighted-MSE only",           X, Y, False, True,  0.0))
    results.append(run_one("C: sampler + weighted-MSE",      X, Y, True,  True,  0.0))
    results.append(run_one("D: focal-MSE γ=2 + sampler",     X, Y, True,  False, 2.0))

    print("\n" + "="*78)
    print("  IMBALANCE ABLATION SUMMARY — MERT-only, 25-layer fusion, 5-fold")
    print("="*78)
    print(f"{'Treatment':<36}{'R² A':>14}{'R² V':>14}{'CCC A':>8}{'CCC V':>8}")
    for r in results:
        print(f"{r['name']:<36}"
              f"  {r['r2_A_mean']:.4f}±{r['r2_A_std']:.3f}"
              f"  {r['r2_V_mean']:.4f}±{r['r2_V_std']:.3f}"
              f"  {r['ccc_A']:.3f}  {r['ccc_V']:.3f}")

    print("\n  Minority per-quadrant R² (most diagnostic):")
    for r in results:
        print(f"\n  {r['name']}")
        for q in ["HVLA", "LVHA", "LVLA"]:
            s = r['pq'].get(q, {})
            if s.get("R2_Arousal") is not None:
                print(f"     {q:>5} (n={s['n']:3d}): A={s['R2_Arousal']:+.3f}  V={s['R2_Valence']:+.3f}")

    print("\n  Reference: published MERT-only = R² A 0.6518 / V 0.5055 (CCC 0.82 / 0.74)")


if __name__ == "__main__":
    main()
