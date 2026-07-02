"""
test_triple_smoke.py — wiring sanity check for the triple-branch models
========================================================================
No dataset / no GPU. Verifies:
  1. MelSpectrogramCNN parameter budget (< 200K) and length-robustness
  2. TripleSSLModel    forward+backward — grads reach all 3 branches
  3. SpectrogramOnlyModel forward+backward — grads reach MERT + CNN
  4. HybridLoss finite, no NaN/Inf grads

Usage:
    cd phaseB/
    python test_triple_smoke.py
"""

import torch

from losses.losses import HybridLoss
from models.models_triple import MelSpectrogramCNN, TripleSSLModel, SpectrogramOnlyModel


def _count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _cnn_budget():
    cnn = MelSpectrogramCNN()
    n = _count_params(cnn)
    assert n < 200_000, f"CNN has {n} params (budget 200K)"
    # length-robustness: two different T → same (B, 128) output
    o1 = cnn(torch.randn(4, 128, 1407))
    o2 = cnn(torch.randn(4, 128, 600))
    assert o1.shape == (4, 128) and o2.shape == (4, 128), (o1.shape, o2.shape)
    print(f"  MelSpectrogramCNN OK | params={n:,} (<200K) | length-robust ✓")


def _triple():
    torch.manual_seed(0)
    B, T = 8, 1407
    x_m = torch.randn(B, 25, 1024)
    x_w = torch.randn(B, 13, 768)
    x_c = torch.randn(B, 128, T)
    y = torch.rand(B, 2)

    model = TripleSSLModel()
    criterion = HybridLoss(use_supcr=True)
    preds, latent = model(x_m, x_w, x_c)
    assert preds.shape == (B, 2), preds.shape
    assert latent.shape == (B, 128), latent.shape

    loss, _ = criterion(preds, latent, y)
    loss.backward()
    assert torch.isfinite(loss), loss
    assert model.fusion_mert.layer_weights.grad is not None, "MERT fusion no grad"
    assert model.fusion_w2v.layer_weights.grad is not None,  "w2v fusion no grad"
    cnn_grad = any(p.grad is not None for p in model.mel_cnn.parameters())
    assert cnn_grad, "mel_cnn got no gradient"
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
    print(f"  TripleSSLModel OK | params={_count_params(model):,} | "
          f"loss={loss.item():.4f} | all 3 branches get grad ✓")


def _spec_only():
    torch.manual_seed(0)
    B, T = 8, 1407
    x_m = torch.randn(B, 25, 1024)
    x_c = torch.randn(B, 128, T)
    y = torch.rand(B, 2)

    model = SpectrogramOnlyModel()
    criterion = HybridLoss(use_supcr=True)
    preds, latent = model(x_m, x_c)
    assert preds.shape == (B, 2), preds.shape
    assert latent.shape == (B, 128), latent.shape

    loss, _ = criterion(preds, latent, y)
    loss.backward()
    assert torch.isfinite(loss), loss
    assert model.fusion_mert.layer_weights.grad is not None, "MERT fusion no grad"
    assert any(p.grad is not None for p in model.mel_cnn.parameters()), "mel_cnn no grad"
    print(f"  SpectrogramOnlyModel OK | params={_count_params(model):,} | "
          f"loss={loss.item():.4f} ✓")


def main():
    _cnn_budget()
    _triple()
    _spec_only()
    print("✅ Triple-branch smoke test passed")


if __name__ == "__main__":
    main()
