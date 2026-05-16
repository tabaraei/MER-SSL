"""
test_joint_smoke.py — wiring sanity check for IADS-E joint learning
====================================================================
No dataset / no GPU. Verifies:
  1. DualSSLDomainModel forward+backward on a MIXED-domain batch
  2. HybridLoss is finite and gradients contain no NaN/Inf
  3. build_joint_dataset works end-to-end on tiny dummy .pt + mini CSVs
     (incl. the IADS-E `{id}_2` → SoundID key normalization)

Usage:
    cd phaseB/
    python test_joint_smoke.py
"""

import csv
import os
import tempfile

import torch

from models import DualSSLDomainModel
from losses import HybridLoss
from data_utils import build_joint_dataset


def _forward_backward():
    torch.manual_seed(0)
    B = 8
    x_mert = torch.randn(B, 25, 1024)
    x_w2v = torch.randn(B, 13, 768)
    domain = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])  # mixed music/sound
    y = torch.rand(B, 2)

    model = DualSSLDomainModel()
    criterion = HybridLoss(use_supcr=True)

    preds, latent = model(x_mert, x_w2v, domain)
    assert preds.shape == (B, 2), preds.shape
    assert latent.shape == (B, 128), latent.shape

    loss, comp = criterion(preds, latent, y)
    loss.backward()
    assert torch.isfinite(loss), loss
    for n, prm in model.named_parameters():
        if prm.grad is not None:
            assert torch.isfinite(prm.grad).all(), f"non-finite grad in {n}"
    assert model.domain_emb.weight.grad is not None, "domain_emb got no gradient"
    print(f"  forward/backward OK | loss={loss.item():.4f}")


def _make_pt(path, ids, n_layers, dim):
    torch.save({i: torch.randn(n_layers, dim) for i in ids}, path)


def _build_joint():
    with tempfile.TemporaryDirectory() as d:
        # --- PMEmo dummy (ids 1..6, csv int ids) ---
        p_ids = [str(i) for i in range(1, 7)]
        _make_pt(os.path.join(d, "pm.pt"), p_ids, 25, 1024)
        _make_pt(os.path.join(d, "pw.pt"), p_ids, 13, 768)
        p_csv = os.path.join(d, "pmemo.csv")
        with open(p_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["musicId", "Arousal", "Valence"])
            for i in range(1, 7):
                w.writerow([i, 0.1 * i, 0.9 - 0.1 * i])

        # --- IADS-E dummy: .pt keyed by `{id}_2` stems, csv by SoundID ---
        s_ids = ["0001", "0002", "0003_b", "0004"]
        _make_pt(os.path.join(d, "im.pt"), [f"{s}_2" for s in s_ids], 25, 1024)
        _make_pt(os.path.join(d, "iw.pt"), [f"{s}_2" for s in s_ids], 13, 768)
        i_csv = os.path.join(d, "iadse.csv")
        with open(i_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sound_id", "valence", "arousal", "category"])
            for j, s in enumerate(s_ids):
                w.writerow([s, 2.0 + j, 7.0 - j, "Test"])

        Xm, Xw, Y, dom, tags = build_joint_dataset(
            os.path.join(d, "pm.pt"), os.path.join(d, "pw.pt"), p_csv,
            os.path.join(d, "im.pt"), os.path.join(d, "iw.pt"), i_csv,
            iadse_valence_col="valence", iadse_arousal_col="arousal",
            iadse_id_col="sound_id", k=1.0, p=1.0, seed=42,
        )
        assert Xm.shape == (10, 25, 1024), Xm.shape   # 6 PMEmo + 4 IADS-E
        assert Xw.shape == (10, 13, 768), Xw.shape
        assert Y.shape == (10, 2), Y.shape
        assert dom.tolist() == [0] * 6 + [1] * 4, dom.tolist()
        assert tags[0].startswith("pmemo:") and tags[-1].startswith("iadse:")
        print(f"  build_joint_dataset OK | N={len(Y)} tags[0]={tags[0]} tags[-1]={tags[-1]}")


def main():
    _forward_backward()
    _build_joint()
    print("✅ Joint SSL domain smoke test passed")


if __name__ == "__main__":
    main()
