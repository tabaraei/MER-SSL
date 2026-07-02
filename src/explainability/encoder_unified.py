"""
encoder_unified.py — single source of truth for Phase C encoding
=================================================================
Guarantees index/query MODEL SYMMETRY by construction: build and query both call
the SAME `UnifiedEnhancedEncoder.encode(...)`. There is no second model instance
that could silently fall back to a simple MERT baseline.

Architecture = the best multi-encoder Phase B model (EnhancedDualSSLModel):
    MERT (25×1024) ─┐
    wav2vec2 (13×768) ─┼─ WeightedLayerFusion ×2 + MusicTheoryBranch
    theory [tempo, sin(key), cos(key)] ─┘
        → concat → head (MLP bottleneck) → 128-D latent → L2-normalised

The cyclic key encoding is applied inside `encode()` via `build_gap_vector(
cyclic_key=True)`, and the 128-D output is L2-normalised — identically for every
song, whether it is a corpus entry (index build) or a runtime query.
"""

from configs.config import PATHS  # centralised config
import os
import sys

import numpy as np
import torch

_PHASEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phaseB")
if _PHASEB not in sys.path:
    sys.path.insert(0, _PHASEB)

from models.models_enhanced import (EnhancedDualSSLModel, gap_dim_of,    # noqa: E402
                             build_gap_vector)

DEFAULT_CKPT = str(PATHS.enhanced_ckpt)


class UnifiedEnhancedEncoder:
    """One encoder for the whole Phase C pipeline (build AND query)."""

    def __init__(self, ckpt_path=DEFAULT_CKPT, gap_features=("tempo", "key"),
                 cyclic_key=True, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gap_features = list(gap_features)
        self.cyclic_key = cyclic_key
        gd = gap_dim_of(self.gap_features, cyclic_key=cyclic_key)
        self.model = EnhancedDualSSLModel(gap_dim=gd).to(self.device)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Enhanced deployment checkpoint not found: {ckpt_path}\n"
                f"  Train it first:  cd ../phaseB && python train_deploy_models.py")
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # Strict load — raises if the checkpoint is NOT the Enhanced architecture,
        # so a wrong (e.g. single-MERT) checkpoint can never silently pass.
        self.model.load_state_dict(state)
        self.model.eval()
        self.latent_dim = self.model.regressor.in_features
        print(f"  ✅ UnifiedEnhancedEncoder | {os.path.basename(ckpt_path)} | "
              f"gap_dim={gd} (cyclic_key={cyclic_key}) | latent={self.latent_dim}-D")

    # ── the ONE encoding path used by build and query ──────────────────────────
    @torch.no_grad()
    def encode(self, x_mert, x_w2v, theory_dict):
        """Encode one song → (latent_dim,) float32, L2-normalised.

        x_mert      : (25, 1024) tensor    x_w2v : (13, 768) tensor
        theory_dict : {feature_name: tensor}  (raw key 0-11; cyclic applied here)
        """
        xm = self._as_batch(x_mert)
        xw = self._as_batch(x_w2v)
        xt = build_gap_vector(theory_dict, self.gap_features,
                              self.cyclic_key).unsqueeze(0).to(self.device)
        preds, latent = self.model(xm, xw, xt)          # head → 128-D, model L2-norms
        z = latent.squeeze(0).cpu().numpy().astype(np.float32)
        z = z / (np.linalg.norm(z) + 1e-12)             # explicit re-norm (idempotent)
        return z, float(preds[0, 0]), float(preds[0, 1])  # latent, pred_arousal, pred_valence

    @torch.no_grad()
    def encode_batch(self, X_mert, X_w2v, X_theory):
        """Vectorised build path — X_theory is the PRE-ASSEMBLED gap matrix
        (N, gap_dim) so it stays bit-identical to per-song `encode()`."""
        xm, xw, xt = (X_mert.float().to(self.device),
                      X_w2v.float().to(self.device),
                      X_theory.float().to(self.device))
        preds, latents = self.model(xm, xw, xt)
        z = latents.cpu().numpy().astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        return z, preds[:, 0].cpu().numpy(), preds[:, 1].cpu().numpy()

    def _as_batch(self, x):
        x = x if torch.is_tensor(x) else torch.as_tensor(x)
        return x.unsqueeze(0).float().to(self.device)
