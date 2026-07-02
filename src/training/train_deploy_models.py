"""
train_deploy_models.py — final ALL-DATA checkpoints for Phase C deployment
===========================================================================
The Enhanced model and the Audio ProtoPNet were only ever 5-fold CV'd (per-fold
checkpoints). To deploy ONE model in the Phase C retrieval system we train each
on ALL 767 songs and save a single checkpoint:

  best_model_enhanced_final.pt   — EnhancedDualSSLModel (MERT + w2v2 + cyclic key)
  protopnet_final.pt             — AudioProtoPNet (learnable-prototype quadrant head)

IMPORTANT (honesty): these are *deployment* checkpoints trained on all data, with
NO held-out split. Their in-sample accuracy is NOT a generalization metric and must
NOT be reported as a result. The canonical reported numbers remain the 5-fold
held-out values (Enhanced R² A 0.7182 / V 0.5686; ProtoPNet 0.728 raw / 0.545
balanced). These checkpoints exist only so Phase C build + query share one model.

Run from phaseB/:
  python train_deploy_models.py
"""

from configs.config import PATHS, PHASE_B  # centralised config
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from losses.losses import HybridLoss
from utils.data_utils import get_emotion_quadrant
from models.models_enhanced import EnhancedDualSSLModel, gap_dim_of
from models.models_protopnet import AudioProtoPNet
from training.train_enhanced_dual import load_enhanced, GAP_JSON, THEORY_PATH

EPOCHS, LR, BATCH = PHASE_B.num_epochs, PHASE_B.learning_rate, PHASE_B.batch_size
FEAT_PATH = str(PATHS.mert_features)
W2V_PATH = str(PATHS.wav2vec_features)
CSV_PATH = str(PATHS.pmemo_annotations)
PROTOS_PER_CLASS = PHASE_B.protos_per_quadrant
L_CLST, L_SEP, L_L1 = 0.8, 0.08, 1e-4


def quad_labels(Y):
    return torch.tensor([get_emotion_quadrant(a, v) for a, v in Y.numpy()], dtype=torch.long)


def balanced_sampler(q):
    w = 1.0 / (np.bincount(q.numpy(), minlength=4) + 1e-6)
    return WeightedRandomSampler(torch.tensor([w[i] for i in q], dtype=torch.float), len(q))


def train_enhanced_final(Xm, Xw, Xt, Y, gd, device):
    print("\n[1/2] Final Enhanced (cyclic key) on ALL data …")
    model = EnhancedDualSSLModel(gap_dim=gd).to(device)
    opt = torch.optim.Adam([
        {"params": model.fusion_mert.parameters(),   "lr": 1e-2},
        {"params": model.fusion_w2v.parameters(),    "lr": 1e-2},
        {"params": model.theory_branch.parameters(), "lr": LR},
        {"params": model.head.parameters(),          "lr": LR},
        {"params": model.regressor.parameters(),     "lr": LR},
    ], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=True)
    ld = DataLoader(TensorDataset(Xm, Xw, Xt, Y), batch_size=BATCH,
                    sampler=balanced_sampler(quad_labels(Y)))
    model.train()
    for ep in range(EPOCHS):
        for bm, bw, bt, by in ld:
            bm, bw, bt, by = bm.to(device), bw.to(device), bt.to(device), by.to(device)
            opt.zero_grad()
            preds, latent = model(bm, bw, bt)
            loss, _ = crit(preds, latent, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    torch.save(model.state_dict(), "best_model_enhanced_final.pt")
    print("  ✅ saved best_model_enhanced_final.pt | gap_dim =", gd, "(cyclic key)")


def train_protopnet_final(Xm, Y, device):
    print("\n[2/2] Final Audio ProtoPNet on ALL data …")
    Q = quad_labels(Y)
    model = AudioProtoPNet(protos_per_class=PROTOS_PER_CLASS).to(device)
    opt = torch.optim.Adam([
        {"params": model.fusion.parameters(),     "lr": 1e-2},
        {"params": model.head.parameters(),       "lr": LR},
        {"params": [model.prototypes],            "lr": 3e-3},
        {"params": model.last_layer.parameters(), "lr": LR},
    ], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    ld = DataLoader(TensorDataset(Xm, Q), batch_size=BATCH, sampler=balanced_sampler(Q))
    model.train()
    for ep in range(EPOCHS):
        for bx, bq in ld:
            bx, bq = bx.to(device), bq.to(device)
            opt.zero_grad()
            logits, dist, _ = model(bx)
            clst, sep = model.cluster_separation_costs(dist, bq)
            loss = ce(logits, bq) + L_CLST*clst - L_SEP*sep + L_L1*model.l1_offclass()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    torch.save({"state_dict": model.state_dict(),
                "protos_per_class": PROTOS_PER_CLASS}, "protopnet_final.pt")
    print("  ✅ saved protopnet_final.pt | prototypes =", model.n_proto)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | training DEPLOYMENT checkpoints (all-data, no held-out)")
    with open(GAP_JSON) as fh:
        gap_features = json.load(fh).get("gap_features", [])
    Xm, Xw, Xt, Y = load_enhanced(FEAT_PATH, W2V_PATH, THEORY_PATH, CSV_PATH,
                                  gap_features, cyclic_key=True)
    gd = gap_dim_of(gap_features, cyclic_key=True)
    train_enhanced_final(Xm, Xw, Xt, Y, gd, device)
    train_protopnet_final(Xm, Y, device)
    print("\nDone. Deployment checkpoints ready for phaseC/build_index_unified.py.")
    print("NOTE: all-data checkpoints — do NOT report their in-sample accuracy; "
          "canonical metrics remain the 5-fold held-out values.")


if __name__ == "__main__":
    main()
