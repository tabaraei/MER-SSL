"""
eval_protopnet_silhouette.py — does the OBJECTIVE determine the latent topology?
================================================================================
The regression models (single-MERT, Enhanced; HybridLoss = MSE+CCC+Rank+SupCR)
produce a latent space with WEAK-TO-MODERATE quadrant structure:
    single-MERT : Silhouette 0.193 Euclidean / 0.269 cosine (held-out)
    Enhanced    : Silhouette 0.182 Euclidean / 0.260 cosine (held-out)
— a continuous V-A gradient (Russell's circumplex), not discrete clusters.

Question: is that low Silhouette a *limitation of the architecture* or a
*consequence of the regression objective*? To find out we train the SAME encoder
(WeightedLayerFusion + head → 128-d latent) with a CLASSIFICATION + clustering
objective — the Audio ProtoPNet (cross-entropy + cluster + separation losses) —
and measure Silhouette on the held-out test-fold latents under the SAME protocol
as `eval_silhouette_audit.py`.

If the ProtoPNet latent shows substantially HIGHER Silhouette, the conclusion is:
the architecture CAN cluster; the regression model's continuum is a deliberate
consequence of optimising a continuous target, not an inability to separate.

Run from phaseB/:
  python eval_protopnet_silhouette.py
"""

from configs.config import PATHS, PHASE_B  # centralised config
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from utils.data_utils import load_pmemo_data, get_emotion_quadrant
from models.models_protopnet import AudioProtoPNet

FEAT_PATH = str(PATHS.mert_features)
CSV_PATH = str(PATHS.pmemo_annotations)
EPOCHS, LR, BATCH = PHASE_B.num_epochs, PHASE_B.learning_rate, PHASE_B.batch_size
PROTOS_PER_CLASS = PHASE_B.protos_per_quadrant
LAMBDA_CLST, LAMBDA_SEP, LAMBDA_L1 = 0.8, 0.08, 1e-4

# Held-out regression-latent Silhouettes (from eval_silhouette_audit.py) for contrast.
REG_SINGLE = dict(eu=0.1934, co=0.2691)
REG_ENHANCED = dict(eu=0.1815, co=0.2595)


def quad_labels(Y):
    return torch.tensor([get_emotion_quadrant(a, v) for a, v in Y.numpy()], dtype=torch.long)


def balanced_sampler(q):
    w = 1.0 / (np.bincount(q.numpy(), minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[i] for i in q], dtype=torch.float), len(q))


def sil_both(latents, labels):
    out = {}
    for metric in ("euclidean", "cosine"):
        try:
            out[metric] = silhouette_score(latents, labels, metric=metric)
        except Exception:
            out[metric] = float("nan")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | ProtoPNet latent Silhouette (objective-vs-topology test)")
    X, Y, _ = load_pmemo_data(FEAT_PATH, CSV_PATH)
    Q = quad_labels(Y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ce = nn.CrossEntropyLoss()
    eu, co, accs, bals = [], [], [], []

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(X)))):
        tr_ld = DataLoader(TensorDataset(X[tr], Q[tr]), batch_size=BATCH,
                           sampler=balanced_sampler(Q[tr]))
        model = AudioProtoPNet(protos_per_class=PROTOS_PER_CLASS).to(device)
        opt = torch.optim.Adam([
            {"params": model.fusion.parameters(),     "lr": 1e-2},
            {"params": model.head.parameters(),       "lr": LR},
            {"params": [model.prototypes],            "lr": 3e-3},
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
            logits, _, z = model(X[te].to(device))
        zt = z.cpu().numpy(); true = Q[te].numpy(); pred = logits.argmax(1).cpu().numpy()
        s = sil_both(zt, true)
        acc = accuracy_score(true, pred); bal = balanced_accuracy_score(true, pred)
        print(f"  fold {fold+1}: Silhouette euclidean={s['euclidean']:.4f} cosine={s['cosine']:.4f} "
              f"| acc={acc:.4f} bal={bal:.4f}")
        eu.append(s["euclidean"]); co.append(s["cosine"]); accs.append(acc); bals.append(bal)

    eu = np.array(eu); co = np.array(co); accs = np.array(accs); bals = np.array(bals)
    print(f"\n{'='*66}\n  PROTOPNET LATENT — 5-FOLD HELD-OUT SILHOUETTE\n{'='*66}")
    print(f"  Silhouette Euclidean : {eu.mean():.4f} ± {eu.std():.4f}")
    print(f"  Silhouette cosine    : {co.mean():.4f} ± {co.std():.4f}")
    print(f"  (sanity) raw acc {accs.mean():.4f} | balanced acc {bals.mean():.4f}")

    print(f"\n  THREE-WAY TOPOLOGY COMPARISON (held-out test latents, same protocol):")
    print(f"  {'Model (objective)':<34}{'Euclidean':>12}{'cosine':>12}")
    print(f"  {'single-MERT (regression)':<34}{REG_SINGLE['eu']:>12.4f}{REG_SINGLE['co']:>12.4f}")
    print(f"  {'Enhanced (regression)':<34}{REG_ENHANCED['eu']:>12.4f}{REG_ENHANCED['co']:>12.4f}")
    print(f"  {'ProtoPNet (classification+sep)':<34}{eu.mean():>12.4f}{co.mean():>12.4f}")
    print(f"\n  Δ cosine (ProtoPNet − Enhanced): {co.mean()-REG_ENHANCED['co']:+.4f}")
    print(f"  → Higher ProtoPNet Silhouette ⇒ the architecture CAN cluster; the regression")
    print(f"    continuum (≈0.26) is an objective-driven choice, not an architectural limit.")


if __name__ == "__main__":
    main()
