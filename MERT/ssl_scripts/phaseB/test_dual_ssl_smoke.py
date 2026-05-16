"""
test_dual_ssl_smoke.py — Wiring sanity check for the Dual-SSL pipeline
======================================================================
Does NOT touch the dataset or GPU. Runs one forward + one backward pass
through DualSSLModel with HybridLoss on random tensors of the correct
shapes and asserts output shapes + finite loss.

Usage:
    cd phaseB/
    python test_dual_ssl_smoke.py
"""

import torch

from models import DualSSLModel, analyze_dual_layer_weights
from losses import HybridLoss


def main():
    torch.manual_seed(0)
    B = 8

    # Dummy inputs matching real extractor output shapes
    x_mert = torch.randn(B, 25, 1024)   # MERT all-layers
    x_w2v  = torch.randn(B, 13,  768)   # wav2vec2 all-layers
    y      = torch.rand(B, 2)           # [arousal, valence] in [0, 1]

    model = DualSSLModel()
    criterion = HybridLoss(use_supcr=True)

    # Forward
    preds, latent = model(x_mert, x_w2v)
    assert preds.shape == (B, 2), f"preds shape {preds.shape} != ({B}, 2)"
    assert latent.shape == (B, 128), f"latent shape {latent.shape} != ({B}, 128)"

    # Backward
    loss, components = criterion(preds, latent, y)
    loss.backward()
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"

    # Gradients reach BOTH fusion modules
    assert model.fusion_mert.layer_weights.grad is not None, "no grad on MERT fusion"
    assert model.fusion_w2v.layer_weights.grad is not None, "no grad on wav2vec fusion"

    # Layer-weight accessor returns both encoders
    w = model.get_layer_weights()
    assert set(w.keys()) == {"mert", "w2v"}, f"unexpected keys {w.keys()}"
    assert w["mert"].shape == (25,) and w["w2v"].shape == (13,)

    print(f"  loss={loss.item():.4f} | components={ {k: round(v, 4) for k, v in components.items()} }")
    print("✅ DualSSL smoke test passed")


if __name__ == "__main__":
    main()
