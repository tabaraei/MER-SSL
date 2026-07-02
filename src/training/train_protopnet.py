"""
train_protopnet.py — train + evaluate the Audio ProtoPNet (learnable prototypes)
=================================================================================
Replaces the post-hoc 4-centroid quadrant classifier (`phaseC/extra_metrics.py`,
0.506 dual / 0.462 single, both below the 0.611 majority baseline) with a
ProtoPNet whose prototypes are learned DURING gradient descent and which
classifies by L2 distance to those prototypes.

Setup mirrors the rest of Phase B: MERT-only backbone (single encoder, for a
clean comparison to the single-MERT centroid 0.462), 5-fold KFold(42), 100
epochs, batch 32, balanced sampler (the quadrant task is 61% HVHA).

Pre-registered targets:
  • primary  : beat the post-hoc 4-centroid (single-MERT 0.462 raw accuracy)
  • stretch  : beat the majority-class baseline (0.611 raw accuracy)
  • fairness : report BALANCED accuracy (mean per-quadrant recall) too — raw
               accuracy is gameable by always predicting HVHA.

Run from phaseB/:
  python train_protopnet.py
"""

from configs.config import PATHS, PHASE_B  # centralised config
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from utils.data_utils import load_pmemo_data, get_emotion_quadrant, QUADRANT_NAMES
from models.models_protopnet import AudioProtoPNet

FEAT_PATH = str(PATHS.mert_features)
CSV_PATH = str(PATHS.pmemo_annotations)
EPOCHS, LR, BATCH = PHASE_B.num_epochs, PHASE_B.learning_rate, PHASE_B.batch_size
PROTOS_PER_CLASS = PHASE_B.protos_per_quadrant
LAMBDA_CLST, LAMBDA_SEP, LAMBDA_L1 = 0.8, 0.08, 1e-4

# Reference points (from phaseC/extra_metrics.py + report §3)
CENTROID_SINGLE = 0.462       # post-hoc 4-centroid, single-MERT, raw acc
CENTROID_DUAL = 0.506         # post-hoc 4-centroid, dual, raw acc
MAJORITY = 0.611              # always-HVHA baseline


def quad_labels(Y):
    return torch.tensor([get_emotion_quadrant(a, v) for a, v in Y.numpy()], dtype=torch.long)


def balanced_sampler(q):
    w = 1.0 / (np.bincount(q.numpy(), minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[i] for i in q], dtype=torch.float), len(q))


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()  # enables --help without side effects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Audio ProtoPNet — MERT backbone, {PROTOS_PER_CLASS} prototypes/class")
    X, Y, _ = load_pmemo_data(FEAT_PATH, CSV_PATH)
    Q = quad_labels(Y)
    print(f"  {len(X)} songs | quadrant counts: {np.bincount(Q.numpy(), minlength=4).tolist()} "
          f"(HVHA/HVLA/LVHA/LVLA)")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ce = nn.CrossEntropyLoss()
    fold_acc, fold_bal, all_true, all_pred = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(X)))):
        tr_ld = DataLoader(TensorDataset(X[tr], Q[tr]), batch_size=BATCH,
                           sampler=balanced_sampler(Q[tr]))
        model = AudioProtoPNet(protos_per_class=PROTOS_PER_CLASS).to(device)
        opt = torch.optim.Adam([
            {"params": model.fusion.parameters(),    "lr": 1e-2},
            {"params": model.head.parameters(),      "lr": LR},
            {"params": [model.prototypes],           "lr": 3e-3},
            {"params": model.last_layer.parameters(), "lr": LR},
        ], weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        model.train()
        for _ in range(EPOCHS):
            for bx, bq in tr_ld:
                bx, bq = bx.to(device), bq.to(device)
                opt.zero_grad()
                logits, dist, _ = model(bx)
                clst, sep = model.cluster_separation_costs(dist, bq)
                loss = ce(logits, bq) + LAMBDA_CLST * clst - LAMBDA_SEP * sep \
                       + LAMBDA_L1 * model.l1_offclass()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(X[te].to(device))
            pred = logits.argmax(1).cpu().numpy()
        true = Q[te].numpy()
        acc = accuracy_score(true, pred)
        bal = balanced_accuracy_score(true, pred)
        print(f"  fold {fold+1}: raw acc={acc:.4f}  balanced acc={bal:.4f}")
        fold_acc.append(acc); fold_bal.append(bal)
        all_true.append(true); all_pred.append(pred)

    acc = np.array(fold_acc); bal = np.array(fold_bal)
    yt, yp = np.concatenate(all_true), np.concatenate(all_pred)
    print(f"\n{'='*60}\n  AUDIO PROTOPNET — 5-FOLD AVERAGE\n{'='*60}")
    print(f"  Raw accuracy      : {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  Balanced accuracy : {bal.mean():.4f} ± {bal.std():.4f}")
    print("\n  Per-quadrant recall:")
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    for c in range(4):
        tot = cm[c].sum()
        rec = cm[c, c] / tot if tot else 0.0
        print(f"    {QUADRANT_NAMES[c]:<14} {cm[c,c]}/{tot} = {rec:.3f}")
    print("\n  Confusion matrix (rows=true, cols=pred; 0=HVHA 1=HVLA 2=LVHA 3=LVLA):")
    print("   " + "\n   ".join("  ".join(f"{v:4d}" for v in row) for row in cm))

    print(f"\n  {'='*56}\n  COMPARISON (raw accuracy)\n  {'='*56}")
    print(f"    Majority baseline (always-HVHA) : {MAJORITY:.3f}")
    print(f"    Post-hoc 4-centroid (single)    : {CENTROID_SINGLE:.3f}")
    print(f"    Post-hoc 4-centroid (dual)      : {CENTROID_DUAL:.3f}")
    print(f"    Audio ProtoPNet (this run)      : {acc.mean():.3f}  (balanced {bal.mean():.3f})")
    verdict_c = "BEATS" if acc.mean() > CENTROID_SINGLE else "does NOT beat"
    verdict_m = "BEATS" if acc.mean() > MAJORITY else "does NOT beat"
    print(f"\n    → ProtoPNet {verdict_c} the post-hoc centroid; {verdict_m} the majority baseline (raw acc).")
    print(f"    → On BALANCED accuracy (fair under 61% imbalance), ProtoPNet = {bal.mean():.3f} "
          f"vs majority-baseline balanced = 0.250.")


if __name__ == "__main__":
    main()
