"""
train_enhanced_dual.py — Phase B extension training (gap-conditional)
=====================================================================
NEW file — mirrors mainB_triple.py's dual-SSL k-fold path. Adds a third
branch that ingests ONLY the Phase A gap features (music theory MERT
fails to capture).

Self-skipping: reads ../phaseA/gap_analysis.json. If gap_features is
empty → prints a message and exits WITHOUT training (per spec: Phase B
extension is skipped automatically when Phase A finds no gaps).

5-fold CV, 100 epochs, HybridLoss + differential optimizer + balanced
sampler — identical config to dual-SSL. Final printed comparison vs the
cited Dual-SSL baseline.

Run from phaseB/ with the venv active:
    python train_enhanced_dual.py
"""

from configs.config import PATHS, PHASE_B  # centralised config
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses.losses import HybridLoss, CCCLoss
from utils.data_utils import _match_dict_to_csv, get_emotion_quadrant, quadrant_r2_breakdown
from models.models_enhanced import EnhancedDualSSLModel, gap_dim_of, build_gap_vector

GAP_JSON = str(PATHS.gap_analysis)
THEORY_PATH = str(PATHS.music_theory)

# Cited Dual-SSL baseline (prior validated run: mainB.py --encoder dual --beta 0.05).
DUAL_BASELINE = {"A": 0.6814, "V": 0.5676, "CCC_A": 0.8087, "CCC_V": 0.7231}


def _match_theory_to_csv(theory, df, id_col):
    """Like data_utils._match_dict_to_csv but keeps each value as the
    per-song feature dict (no tensor conversion — values ARE dicts)."""
    matched = {}
    for idx, raw_id in df[id_col].items():
        clean_id = str(int(raw_id)) if isinstance(raw_id, (int, float, np.number)) else str(raw_id)
        if clean_id in theory:
            matched[idx] = theory[clean_id]
    return matched


def load_enhanced(mert_path, w2v_path, theory_path, csv_path, gap_features,
                  cyclic_key: bool = False):
    """MERT ∩ wav2vec ∩ music-theory ∩ CSV, with the gap vector assembled
    in the exact gap_features order.

    cyclic_key=True encodes the `key` feature as [sin, cos] (corrected circular
    geometry); default False preserves the original raw-integer behaviour so
    that previously-reported runs reproduce exactly."""
    mert = torch.load(mert_path, map_location="cpu", weights_only=False)
    w2v = torch.load(w2v_path, map_location="cpu", weights_only=False)
    theory = torch.load(theory_path, map_location="cpu", weights_only=False)
    df = pd.read_csv(csv_path)

    ar_col = [c for c in df.columns if "arousal" in c.lower()][0]
    va_col = [c for c in df.columns if "valence" in c.lower()][0]
    id_col = [c for c in df.columns if any(x in c.lower() for x in ["music", "id"])][0]

    def _norm(c):
        return (c - c.min()) / (c.max() - c.min() + 1e-8)
    df[ar_col] = _norm(df[ar_col])
    df[va_col] = _norm(df[va_col])

    mert_m = _match_dict_to_csv(mert, df, id_col)
    w2v_m = _match_dict_to_csv(w2v, df, id_col)
    theory_m = _match_theory_to_csv(theory, df, id_col)

    common = [i for i in df.index if i in mert_m and i in w2v_m and i in theory_m]
    if not common:
        raise ValueError("No overlap MERT ∩ wav2vec ∩ music-theory ∩ CSV. "
                         "Run extract_music_theory.py first.")

    X_mert = torch.stack([mert_m[i] for i in common]).float()
    X_w2v = torch.stack([w2v_m[i] for i in common]).float()

    X_theory = torch.stack(
        [build_gap_vector(theory_m[i], gap_features, cyclic_key) for i in common]
    ).float()

    df_m = df.loc[common]
    Y = torch.tensor(df_m[[ar_col, va_col]].values, dtype=torch.float32)
    print(f"  ✅ Enhanced matched {len(Y)} songs | gap_vec dim={X_theory.shape[1]} "
          f"({gap_features})")
    return X_mert, X_w2v, X_theory, Y


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


def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    for b_m, b_w, b_t, b_y in loader:
        b_m, b_w, b_t, b_y = b_m.to(device), b_w.to(device), b_t.to(device), b_y.to(device)
        opt.zero_grad()
        preds, latent = model(b_m, b_w, b_t)
        loss, _ = crit(preds, latent, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()


def evaluate(model, loader, device):
    model.eval()
    P, T = [], []
    ccc = CCCLoss()
    with torch.no_grad():
        for b_m, b_w, b_t, b_y in loader:
            p, _ = model(b_m.to(device), b_w.to(device), b_t.to(device))
            P.append(p.cpu()); T.append(b_y)
    y_pred, y_true = np.vstack(P), np.vstack(T)
    return y_true, y_pred, ccc.compute_ccc_scores(torch.tensor(y_pred), torch.tensor(y_true))


def main():
    ap = argparse.ArgumentParser(description="Phase B enhanced (gap-conditional)")
    ap.add_argument("--epochs", type=int, default=PHASE_B.num_epochs)
    ap.add_argument("--lr", type=float, default=PHASE_B.learning_rate)
    ap.add_argument("--batch_size", type=int, default=PHASE_B.batch_size)
    ap.add_argument("--feat_path", default=str(PATHS.mert_features))
    ap.add_argument("--w2v_path", default=str(PATHS.wav2vec_features))
    ap.add_argument("--csv_path",
                    default=str(PATHS.pmemo_annotations))
    args = ap.parse_args()

    # ── B1: read gap analysis, self-skip if no gaps ─────────────────────────
    if not os.path.exists(GAP_JSON):
        raise SystemExit(f"❌ {GAP_JSON} not found. Run phaseA/run_music_theory_probing.py first.")
    with open(GAP_JSON) as fh:
        gap = json.load(fh)
    gap_features = gap.get("gap_features", [])

    if not gap_features:
        print("\n🟢 Phase A found NO gaps (gap_features is empty).")
        print("   MERT already captures all probed music-theory features.")
        print("   Phase B extension is SKIPPED by design — no model trained.")
        raise SystemExit(0)

    print(f"\n🚀 Enhanced Dual-SSL | gap_features={gap_features} | epochs={args.epochs}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_mert, X_w2v, X_theory, Y = load_enhanced(
        args.feat_path, args.w2v_path, THEORY_PATH, args.csv_path, gap_features)
    gd = gap_dim_of(gap_features)

    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_ccc, T_all, P_all = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(Y)))):
        print(f"  ── Fold {fold + 1}/5 ──────────────────────────────")
        tr_ds = TensorDataset(X_mert[tr], X_w2v[tr], X_theory[tr], Y[tr])
        te_ds = TensorDataset(X_mert[te], X_w2v[te], X_theory[te], Y[te])
        tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, sampler=balanced_sampler(Y[tr]))
        te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

        model = EnhancedDualSSLModel(gap_dim=gd).to(device)
        opt = get_optimizer(model, args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        for _ in range(args.epochs):
            train_one_epoch(model, tr_ld, opt, crit, device)
            sch.step()

        y_t, y_p, ccc = evaluate(model, te_ld, device)
        r2 = r2_score(y_t, y_p, multioutput="raw_values")
        print(f"     R² A={r2[0]:.4f} V={r2[1]:.4f} | "
              f"CCC A={ccc['CCC_Arousal']:.4f} V={ccc['CCC_Valence']:.4f}")
        fold_r2.append(r2)
        fold_ccc.append([ccc["CCC_Arousal"], ccc["CCC_Valence"]])
        T_all.append(y_t); P_all.append(y_p)
        ck = f"best_model_enhanced_fold{fold + 1}.pt"
        torch.save(model.state_dict(), ck)
        print(f"     💾 saved → {ck}")

    fr = np.array(fold_r2); fc = np.array(fold_ccc)
    mr, sr = fr.mean(0), fr.std(0)
    mc = fc.mean(0)
    print(f"\n{'=' * 60}\n  🏆 ENHANCED DUAL-SSL — 5-FOLD AVERAGE\n{'=' * 60}")
    print(f"  R²  Arousal : {mr[0]:.4f} ± {sr[0]:.4f}")
    print(f"  R²  Valence : {mr[1]:.4f} ± {sr[1]:.4f}")
    print(f"  CCC Arousal : {mc[0]:.4f}")
    print(f"  CCC Valence : {mc[1]:.4f}")

    print("\n  Per-Quadrant R² Breakdown:")
    for q, s in quadrant_r2_breakdown(np.vstack(T_all), np.vstack(P_all)).items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f} | V={s['R2_Valence']:.3f} | n={s['n']}")

    print(f"\n{'=' * 60}\n  📊 COMPARISON\n{'=' * 60}")
    print(f"  {'Model':<24} {'V R²':>7} {'A R²':>7} {'CCC V':>7} {'CCC A':>7}")
    print(f"  {'-' * 54}")
    print(f"  {'Dual-SSL (baseline)':<24} {DUAL_BASELINE['V']:>7.4f} "
          f"{DUAL_BASELINE['A']:>7.4f} {DUAL_BASELINE['CCC_V']:>7.4f} "
          f"{DUAL_BASELINE['CCC_A']:>7.4f}")
    print(f"  {'Enhanced Dual-SSL':<24} {mr[1]:>7.4f} {mr[0]:>7.4f} "
          f"{mc[1]:>7.4f} {mc[0]:>7.4f}")
    print(f"{'=' * 60}")
    print("  (baseline = cited prior run; enhanced = this run)")


if __name__ == "__main__":
    main()
