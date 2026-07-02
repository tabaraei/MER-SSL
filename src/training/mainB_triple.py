"""
mainB_triple.py — Triple-branch training (MERT + wav2vec2 + mel-CNN)
=====================================================================
NEW file — mainB.py is untouched. Mirrors mainB.py's dual-SSL k-fold
path exactly; only the model class and the extra mel tensor differ.

  --encoder triple     : MERT + wav2vec2 (frozen) + MelSpectrogramCNN
  --encoder spec_only  : MERT (frozen) + MelSpectrogramCNN (ablation)

5-fold CV on PMEmo, 100 epochs, HybridLoss (MSE+CCC+Rank+SupCR, same
weights as dual-SSL), differential optimizer (SSL fusion lr=1e-2;
CNN+head+regressor lr=base_lr). Best checkpoint saved per fold. Final
printout compares MERT-only / Dual-SSL / Triple / Spec-only side by side.

Run from phaseB/ with the venv active:
  python mainB_triple.py --encoder triple    --epochs 100
  python mainB_triple.py --encoder spec_only --epochs 100
"""

from configs.config import PATHS, PHASE_B  # centralised config
import argparse

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses.losses import HybridLoss, CCCLoss
from utils.data_utils import get_emotion_quadrant, quadrant_r2_breakdown
from utils.data_utils_triple import load_pmemo_triple_ssl
from models.models_triple import TripleSSLModel, SpectrogramOnlyModel

# Prior validated baselines (cited from earlier runs, NOT recomputed here).
# MERT-only: mainB.py --model hybrid (audio only).
# Dual-SSL : mainB.py --encoder dual --beta 0.05 (best dual config).
PRIOR_BASELINES = {
    "MERT-only (cited)":      {"A": 0.6518, "V": 0.5055, "CCC_A": 0.82, "CCC_V": 0.74},
    "Dual-SSL β=0.05 (cited)": {"A": 0.6814, "V": 0.5676, "CCC_A": 0.8087, "CCC_V": 0.7231},
}


def get_balanced_sampler(Y):
    quads = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    counts = np.bincount(quads, minlength=4)
    w = 1.0 / (counts + 1e-6)
    sw = torch.tensor([w[q] for q in quads], dtype=torch.float)
    return WeightedRandomSampler(sw, len(sw))


def get_optimizer(model, base_lr, use_w2v):
    """SSL fusion modules train fast (1e-2); CNN + head + regressor at base_lr."""
    groups = [{"params": model.fusion_mert.parameters(), "lr": 1e-2}]
    if use_w2v:
        groups.append({"params": model.fusion_w2v.parameters(), "lr": 1e-2})
    groups += [
        {"params": model.mel_cnn.parameters(),   "lr": base_lr},
        {"params": model.head.parameters(),      "lr": base_lr},
        {"params": model.regressor.parameters(), "lr": base_lr},
    ]
    return torch.optim.Adam(groups, weight_decay=1e-3)


def train_one_epoch(model, loader, optimizer, criterion, device, use_w2v):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        tensors = [t.to(device) for t in batch]
        if use_w2v:
            b_m, b_w, b_c, b_y = tensors
            preds, latent = model(b_m, b_w, b_c)
        else:
            b_m, b_c, b_y = tensors
            preds, latent = model(b_m, b_c)
        loss, _ = criterion(preds, latent, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def evaluate(model, loader, device, use_w2v):
    model.eval()
    all_p, all_y = [], []
    ccc_fn = CCCLoss()
    with torch.no_grad():
        for batch in loader:
            tensors = [t.to(device) for t in batch]
            if use_w2v:
                b_m, b_w, b_c, b_y = tensors
                p, _ = model(b_m, b_w, b_c)
            else:
                b_m, b_c, b_y = tensors
                p, _ = model(b_m, b_c)
            all_p.append(p.cpu()); all_y.append(b_y.cpu())
    y_true, y_pred = np.vstack(all_y), np.vstack(all_p)
    ccc = ccc_fn.compute_ccc_scores(torch.tensor(y_pred), torch.tensor(y_true))
    return y_true, y_pred, ccc


def main():
    ap = argparse.ArgumentParser(description="Triple-branch MER training")
    ap.add_argument("--encoder", choices=["triple", "spec_only"], default="triple")
    ap.add_argument("--epochs", type=int, default=PHASE_B.num_epochs)
    ap.add_argument("--lr", type=float, default=PHASE_B.learning_rate)
    ap.add_argument("--batch_size", type=int, default=PHASE_B.batch_size)
    ap.add_argument("--feat_path", default=str(PATHS.mert_features))
    ap.add_argument("--w2v_path",  default=str(PATHS.wav2vec_features))
    ap.add_argument("--melspec_path", default=str(PATHS.melspec_features))
    ap.add_argument("--csv_path",
                    default=str(PATHS.pmemo_annotations))
    args = ap.parse_args()

    use_w2v = (args.encoder == "triple")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Triple-branch | encoder={args.encoder} | epochs={args.epochs} | device={device}")

    X_mert, X_w2v, X_mel, Y, _ids = load_pmemo_triple_ssl(
        args.feat_path, args.w2v_path, args.melspec_path, args.csv_path
    )

    criterion = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)

    print("\n🧪 5-Fold Cross-Validation (Balanced & Differential Optimizer)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc = [], []
    all_y_true, all_y_pred = [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"  ── Fold {fold + 1}/5 ──────────────────────────────")
        if use_w2v:
            tr_ds = TensorDataset(X_mert[tr], X_w2v[tr], X_mel[tr], Y[tr])
            te_ds = TensorDataset(X_mert[te], X_w2v[te], X_mel[te], Y[te])
            model = TripleSSLModel().to(device)
        else:
            tr_ds = TensorDataset(X_mert[tr], X_mel[tr], Y[tr])
            te_ds = TensorDataset(X_mert[te], X_mel[te], Y[te])
            model = SpectrogramOnlyModel().to(device)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                               sampler=get_balanced_sampler(Y[tr]))
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

        optimizer = get_optimizer(model, args.lr, use_w2v)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for _ in range(args.epochs):
            train_one_epoch(model, tr_loader, optimizer, criterion, device, use_w2v)
            scheduler.step()

        y_t, y_p, ccc = evaluate(model, te_loader, device, use_w2v)
        r2 = r2_score(y_t, y_p, multioutput="raw_values")
        print(f"     R² A={r2[0]:.4f} V={r2[1]:.4f} | "
              f"CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f}")

        fold_r2.append(r2)
        fold_ccc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]])
        all_y_true.append(y_t); all_y_pred.append(y_p)

        ckpt = f"best_model_{args.encoder}_fold{fold + 1}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"     💾 saved → {ckpt}")

    fold_r2 = np.array(fold_r2)
    fold_ccc = np.array(fold_ccc)
    mean_r2, std_r2 = fold_r2.mean(0), fold_r2.std(0)
    mean_ccc = fold_ccc.mean(0)

    print(f"\n{'=' * 60}\n  🏆 {args.encoder.upper()} — 5-FOLD AVERAGE\n{'=' * 60}")
    print(f"  R²  Arousal : {mean_r2[0]:.4f} ± {std_r2[0]:.4f}")
    print(f"  R²  Valence : {mean_r2[1]:.4f} ± {std_r2[1]:.4f}")
    print(f"  CCC Arousal : {mean_ccc[0]:.4f}")
    print(f"  CCC Valence : {mean_ccc[1]:.4f}")

    all_t, all_p = np.vstack(all_y_true), np.vstack(all_y_pred)
    print("\n  Per-Quadrant R² Breakdown:")
    for q, s in quadrant_r2_breakdown(all_t, all_p).items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f} | V={s['R2_Valence']:.3f} | n={s['n']}")

    # ── Side-by-side comparison ─────────────────────────────────────────────
    this_label = "Triple-SSL (this run)" if use_w2v else "Spectrogram-only (this run)"
    rows = dict(PRIOR_BASELINES)
    rows[this_label] = {"A": mean_r2[0], "V": mean_r2[1],
                        "CCC_A": mean_ccc[0], "CCC_V": mean_ccc[1]}

    print(f"\n{'=' * 70}\n  📊 COMPARISON (R² / CCC, PMEmo 5-fold)\n{'=' * 70}")
    print(f"  {'Model':<30} {'R² A':>7} {'R² V':>7} {'CCC A':>7} {'CCC V':>7}")
    print(f"  {'-' * 64}")
    for name, m in rows.items():
        print(f"  {name:<30} {m['A']:>7.4f} {m['V']:>7.4f} "
              f"{m['CCC_A']:>7.4f} {m['CCC_V']:>7.4f}")
    print(f"{'=' * 70}")
    print("  (cited rows = prior validated runs; this-run row = freshly computed)")


if __name__ == "__main__":
    main()
