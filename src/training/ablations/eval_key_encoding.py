"""
eval_key_encoding.py — raw-integer vs cyclic (sin/cos) key encoding (A/B)
=========================================================================
Phase A diagnosed `key` as a gap and the Enhanced model re-injected it — but
fed as a raw integer 0–11, which destroys the circular geometry (C=0 and B=11
are musically adjacent yet numerically maximally distant). Earlier result: key
helped arousal not at all and valence only +0.001 (null). Hypothesis: with a
*correct* cyclic encoding

    x_sin = sin(2π · key / 12),   x_cos = cos(2π · key / 12)

the key branch can finally use the feature, and valence should rise.

This is a pre-registered A/B: the SAME Enhanced model, SAME 5-fold splits, SAME
training recipe — only the key encoding differs (raw 1-d vs cyclic 2-d).

Pass mark (pre-registered): cyclic wins iff it beats raw-key by > 1 fold-std on
valence R² (the axis key should affect), or > 2 fold-std on either axis.

Run from phaseB/:
  python eval_key_encoding.py
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
FEAT_PATH = str(PATHS.mert_features)
W2V_PATH = str(PATHS.wav2vec_features)
CSV_PATH = str(PATHS.pmemo_annotations)

RAW_BASELINE = {"A": 0.7182, "V": 0.5686, "CCC_A": 0.8345, "CCC_V": 0.7259}


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


def run(label, Xm, Xw, Xt, Y, gd, device):
    print(f"\n=== {label} | gap_vec dim={Xt.shape[1]} ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    fr, fc, T, P = [], [], [], []
    ccc_eval = CCCLoss()
    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        tr_ld = DataLoader(TensorDataset(Xm[tr], Xw[tr], Xt[tr], Y[tr]),
                           batch_size=BATCH, sampler=balanced_sampler(Y[tr]))
        model = EnhancedDualSSLModel(gap_dim=gd).to(device)
        opt = get_optimizer(model, LR)
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
            preds, _ = model(Xm[te].to(device), Xw[te].to(device), Xt[te].to(device))
        yt, yp = Y[te].numpy(), preds.cpu().numpy()
        r2 = r2_score(yt, yp, multioutput="raw_values")
        ccc = ccc_eval.compute_ccc_scores(torch.tensor(yp), torch.tensor(yt))
        print(f"  fold {fold+1}: R² A={r2[0]:.4f} V={r2[1]:.4f} | "
              f"CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f}")
        fr.append(r2); fc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]]); T.append(yt); P.append(yp)
    fr = np.array(fr); fc = np.array(fc)
    print(f"  -> R² A {fr[:,0].mean():.4f}±{fr[:,0].std():.3f} | "
          f"V {fr[:,1].mean():.4f}±{fr[:,1].std():.3f} | "
          f"CCC A {fc[:,0].mean():.3f} V {fc[:,1].mean():.3f}")
    return dict(rA=fr[:,0].mean(), sA=fr[:,0].std(), rV=fr[:,1].mean(), sV=fr[:,1].std(),
                cA=fc[:,0].mean(), cV=fc[:,1].mean(),
                pq=quadrant_r2_breakdown(np.vstack(T), np.vstack(P)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Key-encoding A/B (raw integer vs cyclic sin/cos)")
    with open(GAP_JSON) as fh:
        gap_features = json.load(fh).get("gap_features", [])
    print(f"  gap_features = {gap_features}")

    # raw key (cyclic_key=False) — reproduces historic Enhanced
    Xm, Xw, Xt_raw, Y = load_enhanced(FEAT_PATH, W2V_PATH, THEORY_PATH, CSV_PATH,
                                      gap_features, cyclic_key=False)
    gd_raw = gap_dim_of(gap_features, cyclic_key=False)
    # cyclic key (sin/cos) — corrected geometry
    _, _, Xt_cyc, _ = load_enhanced(FEAT_PATH, W2V_PATH, THEORY_PATH, CSV_PATH,
                                    gap_features, cyclic_key=True)
    gd_cyc = gap_dim_of(gap_features, cyclic_key=True)

    raw = run("RAW key (integer 0-11)", Xm, Xw, Xt_raw, Y, gd_raw, device)
    cyc = run("CYCLIC key (sin/cos)",   Xm, Xw, Xt_cyc, Y, gd_cyc, device)

    print("\n" + "=" * 70)
    print("  KEY-ENCODING A/B — Enhanced model, 5-fold (same splits)")
    print("=" * 70)
    print(f"{'Encoding':<22}{'R² A':>16}{'R² V':>16}{'CCC A':>8}{'CCC V':>8}")
    print(f"{'RAW integer 0-11':<22}{raw['rA']:.4f}±{raw['sA']:.3f}   {raw['rV']:.4f}±{raw['sV']:.3f}"
          f"   {raw['cA']:.3f}  {raw['cV']:.3f}")
    print(f"{'CYCLIC sin/cos':<22}{cyc['rA']:.4f}±{cyc['sA']:.3f}   {cyc['rV']:.4f}±{cyc['sV']:.3f}"
          f"   {cyc['cA']:.3f}  {cyc['cV']:.3f}")
    dV = cyc['rV'] - raw['rV']; dA = cyc['rA'] - raw['rA']
    print(f"\n  Δ (cyclic − raw):  R² A {dA:+.4f}   R² V {dV:+.4f}")
    print(f"  valence fold-std (raw {raw['sV']:.3f}, cyclic {cyc['sV']:.3f}); "
          f"|ΔV| {'>' if abs(dV) > max(raw['sV'], cyc['sV']) else '≤'} fold-std "
          f"→ {'OUTSIDE' if abs(dV) > max(raw['sV'], cyc['sV']) else 'INSIDE'} noise")
    print(f"  reference historic raw-key Enhanced: A {RAW_BASELINE['A']} / V {RAW_BASELINE['V']}")


if __name__ == "__main__":
    main()
