"""
evaluate_latent_space.py — Phase C Priority-1 evaluation (rigorous, test-fold)
=============================================================================
Proves the core Phase B claim: the Supervised Contrastive Regression (SupCR)
loss organizes the learned latent space into emotionally coherent neighborhoods.

Why a new script (not mainC --mode build): the current best_model.pt is the
DualSSL model, which does NOT load into mainC's single-encoder MERModel, so the
old build flow crashes. More importantly, mainC builds the index IN-SAMPLE (one
model over all songs). This script builds OUT-OF-SAMPLE (test-fold) latents —
every song is encoded by a model that never trained on it — which is the honest
way to prove the latent space is genuinely organized rather than memorized.

Two encoders:
  --encoder mert : MERModel (single MERT)              — "the MERT latent space"
  --encoder dual : DualSSLModel (MERT + wav2vec2)      — the best Phase B model
                   (use --beta 0.05 to match the headline dual config)

Pipeline (single command):
  1. Load features + ground-truth V-A.
  2. 5-fold CV (KFold shuffle, random_state=42 — same split as Phase B). Each
     fold: train the chosen model with HybridLoss (MSE+CCC+Rank+SupCR@0.1) on the
     train split, then encode the HELD-OUT test split → test-fold latents.
  3. Assemble per-song test-fold latents (L2-normalized) + V-A + EDA + IDs →
     prototypes.npy (same schema as index_builder).
  4. Silhouette (4 Russell quadrants, cosine) + Precision@k (k=5,10,20, within a
     0.20 V-A Euclidean radius) via the existing evaluator.evaluate_retrieval.

Run from phaseC/ with the venv active:
  # single MERT
  python evaluate_latent_space.py --encoder mert --epochs 100 --index_path prototypes_mert.npy
  # best dual model (MERT + wav2vec2)
  python evaluate_latent_space.py --encoder dual --beta 0.05 --epochs 100 \
      --feat_path ../phaseB/pmemo_mert_all_layers.pt \
      --w2v_path  ../phaseB/pmemo_wav2vec_all_layers.pt \
      --csv_path  /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
      --eda_dir   /datasets/emotions/PMEmo2019/EDA --index_path prototypes_dual.npy
"""

from configs.config import PATHS, PHASE_B  # centralised config
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ── resolve phaseB imports (same pattern as mainC.py) ───────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_PHASEB = os.path.abspath(os.path.join(_THIS, "..", "phaseB"))
if os.path.isdir(_PHASEB):
    sys.path.insert(0, _PHASEB)

from models.models import MERModel, DualSSLModel                          # noqa: E402
from losses.losses import HybridLoss, fusion_entropy_loss                 # noqa: E402
from utils.data_utils import (load_pmemo_data, load_pmemo_dual_ssl,      # noqa: E402
                        get_emotion_quadrant)

from explainability.eda_loader import load_eda_for_ids          # noqa: E402  (phaseC, already correct)
from explainability.evaluator import evaluate_retrieval         # noqa: E402  (phaseC, reused as-is)


def balanced_sampler(Y):
    quads = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    counts = np.bincount(quads, minlength=4)
    w = 1.0 / (counts + 1e-6)
    return WeightedRandomSampler(
        torch.tensor([w[q] for q in quads], dtype=torch.float), len(quads))


def make_optimizer(model, base_lr, is_dual):
    """Differential optimizer — same recipe as Phase B (fusion fast, head base)."""
    if is_dual:
        groups = [
            {"params": model.fusion_mert.parameters(), "lr": 1e-2},
            {"params": model.fusion_w2v.parameters(),  "lr": 1e-2},
            {"params": model.head.parameters(),        "lr": base_lr},
            {"params": model.regressor.parameters(),   "lr": base_lr},
        ]
    else:
        groups = [
            {"params": model.fusion.parameters(),    "lr": 1e-2},
            {"params": model.head.parameters(),      "lr": base_lr},
            {"params": model.regressor.parameters(), "lr": base_lr},
        ]
    return torch.optim.Adam(groups, weight_decay=1e-3)


def main():
    ap = argparse.ArgumentParser(description="Phase C test-fold latent-space evaluation")
    ap.add_argument("--encoder", choices=["mert", "dual"], default="mert")
    ap.add_argument("--feat_path", default="../phaseB/pmemo_mert_all_layers.pt")
    ap.add_argument("--w2v_path",  default="../phaseB/pmemo_wav2vec_all_layers.pt")
    ap.add_argument("--csv_path",
                    default=str(PATHS.pmemo_annotations))
    ap.add_argument("--eda_dir", default="/datasets/emotions/PMEmo2019/EDA")
    ap.add_argument("--index_path", default="prototypes.npy")
    ap.add_argument("--epochs", type=int, default=PHASE_B.num_epochs)
    ap.add_argument("--lr", type=float, default=PHASE_B.learning_rate)
    ap.add_argument("--batch_size", type=int, default=PHASE_B.batch_size)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="Entropy sharpening weight (use 0.05 to match the best dual model)")
    # Loss-ablation weights (defaults = full HybridLoss).
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_ccc", type=float, default=0.5)
    ap.add_argument("--w_rank", type=float, default=0.3)
    ap.add_argument("--w_supcr", type=float, default=0.1)
    args = ap.parse_args()

    is_dual = (args.encoder == "dual")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻  Device: {device} | encoder={args.encoder} | beta={args.beta}")

    # ── load features + labels ──────────────────────────────────────────────
    if is_dual:
        Xm, Xw, Y, ids = load_pmemo_dual_ssl(args.feat_path, args.w2v_path, args.csv_path)
        print(f"  Loaded {len(Y)} songs | MERT={tuple(Xm.shape)} wav2vec={tuple(Xw.shape)}")
    else:
        Xm, Y, ids = load_pmemo_data(args.feat_path, args.csv_path)
        Xw = None
        print(f"  Loaded {len(Y)} songs | MERT={tuple(Xm.shape)}")
    # load_pmemo_* return float-formatted ID strings ('1.0'); normalize to clean
    # integers so EDA lookup ('{id}_EDA.csv') and int() both work.
    ids = [str(int(float(i))) for i in ids]
    N = len(Y)

    criterion = HybridLoss(w_mse=args.w_mse, w_ccc=args.w_ccc, w_rank=args.w_rank,
                           w_supcr=args.w_supcr, use_supcr=(args.w_supcr > 0))
    print(f"  Loss weights: MSE={args.w_mse} CCC={args.w_ccc} Rank={args.w_rank} "
          f"SupCR={args.w_supcr}")

    def build_model():
        return (DualSSLModel() if is_dual else MERModel(mode="hybrid")).to(device)

    bottleneck = build_model().regressor.in_features
    test_latents = np.zeros((N, bottleneck), dtype=np.float32)
    pred_a = np.zeros(N, dtype=np.float32)
    pred_v = np.zeros(N, dtype=np.float32)
    filled = np.zeros(N, dtype=bool)
    last_layer_weights = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("\n🧪  5-fold CV — collecting TEST-FOLD (out-of-sample) latents...")
    for fold, (tr, te) in enumerate(kf.split(np.arange(N))):
        print(f"  ── Fold {fold + 1}/5 ── train={len(tr)} test={len(te)}")
        if is_dual:
            tr_ds = TensorDataset(Xm[tr], Xw[tr], Y[tr])
        else:
            tr_ds = TensorDataset(Xm[tr], Y[tr])
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=balanced_sampler(Y[tr]))

        model = build_model()
        opt = make_optimizer(model, args.lr, is_dual)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        model.train()
        for _ in range(args.epochs):
            for batch in tr_loader:
                if is_dual:
                    bm, bw, by = [t.to(device) for t in batch]
                    preds, latent = model(bm, bw)
                else:
                    bx, by = [t.to(device) for t in batch]
                    preds, latent = model(bx)
                loss, _ = criterion(preds, latent, by)
                if args.beta > 0.0:
                    loss = loss + fusion_entropy_loss(model) * args.beta
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            if is_dual:
                preds, latent = model(Xm[te].to(device), Xw[te].to(device))
            else:
                preds, latent = model(Xm[te].to(device))
        test_latents[te] = latent.cpu().numpy()
        pred_a[te] = preds[:, 0].cpu().numpy()
        pred_v[te] = preds[:, 1].cpu().numpy()
        filled[te] = True
        last_layer_weights = (model.fusion_mert if is_dual else model.fusion).layer_weights.detach().cpu().numpy()

    assert filled.all(), "some songs never appeared in a test fold"

    # L2-normalize (model already normalizes; idempotent + safe)
    norms = np.linalg.norm(test_latents, axis=1, keepdims=True)
    test_latents = test_latents / np.where(norms < 1e-8, 1.0, norms)

    # EDA (phaseC loader; {id}_EDA.csv — verified all 794 present)
    eda = None
    if args.eda_dir and os.path.isdir(args.eda_dir):
        print(f"\n  Loading EDA from {args.eda_dir}...")
        eda = load_eda_for_ids([str(i) for i in ids], args.eda_dir)

    index = {
        "latents":       test_latents.astype(np.float32),
        "music_ids":     np.array([int(i) for i in ids], dtype=np.int32),
        "arousal":       Y[:, 0].numpy(),
        "valence":       Y[:, 1].numpy(),
        "pred_arousal":  pred_a,
        "pred_valence":  pred_v,
        "eda_feats":     (eda if eda is not None else np.zeros((N, 7), np.float32)).astype(np.float32),
        "layer_weights": last_layer_weights,
    }
    np.save(args.index_path, index)
    print(f"\n💾  Test-fold index saved → {args.index_path} "
          f"(L2-normalized latents, IDs, ground-truth + predicted V-A, EDA)")

    # Sub-tasks 1.2 (Silhouette) + 1.3 (Precision@k) — reuse the existing evaluator
    print("\n" + "=" * 60)
    print(f"  PHASE C EVALUATION — {args.encoder} encoder, test-fold (out-of-sample)")
    print("=" * 60)
    results = evaluate_retrieval(index, k_values=(5, 10, 20), emotion_threshold=0.20)

    # CCC of the out-of-sample predictions (needed for the loss-ablation table)
    from losses.losses import CCCLoss
    ccc = CCCLoss().compute_ccc_scores(
        torch.tensor(np.stack([pred_a, pred_v], axis=1)),
        torch.tensor(np.stack([Y[:, 0].numpy(), Y[:, 1].numpy()], axis=1)))

    sil = results.get("Silhouette")
    print("\n🧠  Interpretation:")
    if sil is not None:
        verdict = ("POSITIVE → quadrants separate in latent space"
                   if sil > 0 else
                   "≤0 → quadrants overlap (emotion is a continuous gradient, not 4 blobs)")
        print(f"   Silhouette       = {sil:.4f}  [{verdict}]")
    for k in (5, 10, 20):
        p = results.get(f"Precision@{k}")
        if p is not None:
            print(f"   Precision@{k:<2}     = {p:.4f}")
    print(f"   CCC Arousal      = {ccc['CCC_Arousal']:.4f}")
    print(f"   CCC Valence      = {ccc['CCC_Valence']:.4f}")
    print(f"\n  ABLATION ROW | SupCR={args.w_supcr} CCC={args.w_ccc} Rank={args.w_rank} "
          f"MSE={args.w_mse} → Silhouette {sil:.4f} | CCC A {ccc['CCC_Arousal']:.4f} "
          f"V {ccc['CCC_Valence']:.4f} | P@5 {results.get('Precision@5',0):.4f}")


if __name__ == "__main__":
    main()
