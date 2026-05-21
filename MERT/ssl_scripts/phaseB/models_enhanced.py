"""
models_enhanced.py — Phase B extension: music-theory gap branch
================================================================
NEW file — existing models.py / models_triple.py are untouched.
Mirrors the models_triple.py convention (reuses WeightedLayerFusion
from models.py; same (preds, latent_norm) contract as DualSSLModel
so HybridLoss + SupCR work unchanged).

The spec referenced a `phaseB/models/` package directory; the actual
project convention is flat `models*.py` files, so this is a flat module
("mirror existing conventions exactly").

Classes:
  - MusicTheoryBranch    : tiny projection over the GAP features only
  - EnhancedDualSSLModel : DualSSL + MusicTheoryBranch (1024+768+32 = 1824)

FEATURE_DIMS is the single source of truth for per-feature dimensionality;
the training script imports it to assemble the gap vector consistently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import WeightedLayerFusion

# Per-feature dimensionality of the Phase A music-theory ground truth.
FEATURE_DIMS = {
    "chroma":             12,
    "spectral_contrast":  7,
    "mode":               1,
    "key":                1,
    "tempo":              1,
    "rhythmic_stability": 1,
    "spectral_centroid":  1,
    "zcr":                1,
}


def gap_dim_of(gap_features) -> int:
    """Total concatenated dimensionality of a gap-feature list."""
    return int(sum(FEATURE_DIMS[f] for f in gap_features))


# =============================================================================
# 1. MusicTheoryBranch — small projection over GAP features only
# =============================================================================

class MusicTheoryBranch(nn.Module):
    """
    Trainable projection for the concatenated gap-feature vector.

    Input  : (B, gap_dim)   only the features Phase A flagged as gaps
    Output : (B, 32)
    """
    def __init__(self, gap_dim: int, out_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gap_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_theory: torch.Tensor) -> torch.Tensor:
        return self.net(x_theory)


# =============================================================================
# 2. EnhancedDualSSLModel — DualSSL + music-theory gap branch
# =============================================================================

class EnhancedDualSSLModel(nn.Module):
    """
    Branch 1: frozen MERT     → WeightedLayerFusion → 1024-d
    Branch 2: frozen wav2vec2 → WeightedLayerFusion →  768-d
    Branch 3: MusicTheoryBranch (gap features only) →   32-d
    Concat → 1824-d → head → 128-d latent → regressor → 2

    forward(x_mert, x_w2v, x_theory) → (preds (B,2), latent_norm (B,128))
    Same contract as DualSSLModel.
    """
    def __init__(
        self,
        gap_dim: int,
        mert_layers: int = 25, mert_dim: int = 1024,
        w2v_layers:  int = 13, w2v_dim:  int = 768,
        theory_dim:  int = 32,
        bottleneck:  int = 128, dropout: float = 0.4,
    ):
        super().__init__()
        self.fusion_mert  = WeightedLayerFusion(mert_layers, mert_dim)
        self.fusion_w2v   = WeightedLayerFusion(w2v_layers,  w2v_dim)
        self.theory_branch = MusicTheoryBranch(gap_dim, theory_dim)

        concat_dim = mert_dim + w2v_dim + theory_dim  # 1024 + 768 + 32 = 1824

        self.head = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(bottleneck, 2)

    def forward(self, x_mert, x_w2v, x_theory):
        fm = self.fusion_mert(x_mert)                  # (B, 1024)
        fw = self.fusion_w2v(x_w2v)                    # (B,  768)
        ft = self.theory_branch(x_theory)              # (B,   32)
        fused = torch.cat([fm, fw, ft], dim=1)         # (B, 1824)
        latent = self.head(fused)                      # (B, 128)
        latent_norm = F.normalize(latent, dim=1)
        return self.regressor(latent_norm), latent_norm

    def get_layer_weights(self) -> dict:
        return {
            "mert": self.fusion_mert.get_weights(),
            "w2v":  self.fusion_w2v.get_weights(),
        }
