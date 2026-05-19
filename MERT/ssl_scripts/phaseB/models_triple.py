"""
models_triple.py — Third encoder branch (trainable mel-spectrogram CNN)
========================================================================
NEW file — existing models.py is untouched. Reuses WeightedLayerFusion
from models.py for the two frozen SSL branches.

Classes:
  - MelSpectrogramCNN   : shallow trainable CNN over a pre-extracted
                          mel-spectrogram (~110K params, < 200K budget)
  - TripleSSLModel      : MERT + wav2vec2 (frozen) + MelCNN (trainable)
  - SpectrogramOnlyModel: MERT (frozen) + MelCNN — ablation, no wav2vec2

NOTE on input: the mel-spectrogram transform has no learnable parameters,
so it is pre-extracted offline (extract_pmemo_melspec.py). These models
therefore consume a pre-computed mel-spectrogram tensor (B, 128, T) — NOT
a raw waveform. This is functionally identical to on-the-fly computation
and keeps the exact TensorDataset(.pt) pipeline used by the dual-SSL model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import WeightedLayerFusion


# =============================================================================
# 1. Trainable Mel-Spectrogram CNN  (input: pre-extracted mel-spec, NOT audio)
# =============================================================================

class MelSpectrogramCNN(nn.Module):
    """
    Shallow CNN over a pre-extracted log-mel spectrogram.

    Input  : (B, 128, T)   pre-computed mel-spectrogram (n_mels=128)
    Output : (B, 128)      learned embedding

    AdaptiveAvgPool2d((1, 1)) makes the output 128-d regardless of the
    time dimension T, so the branch is robust to clip-length differences.
    Kept shallow (~110K params) to avoid overfitting on ~600 train samples.
    """
    def __init__(self, out_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),                 # (B, 128)
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_mel: torch.Tensor) -> torch.Tensor:
        # (B, 128, T) → (B, 1, 128, T) channel dim for Conv2d
        x = x_mel.unsqueeze(1)
        x = self.conv(x)
        return self.proj(x)               # (B, out_dim)


# =============================================================================
# 2. TripleSSLModel — MERT + wav2vec2 (frozen) + MelCNN (trainable)
# =============================================================================

class TripleSSLModel(nn.Module):
    """
    Three-branch model:
      Branch 1: frozen MERT     → WeightedLayerFusion → 1024-d
      Branch 2: frozen wav2vec2 → WeightedLayerFusion →  768-d
      Branch 3: MelSpectrogramCNN (trainable)         →  128-d
      Concat → 1920-d → head → 128-d latent → regressor → 2

    forward(x_mert, x_w2v, x_mel) → (preds (B,2), latent_norm (B,128))
    Same (preds, latent) contract as DualSSLModel for HybridLoss/SupCR.
    """
    def __init__(
        self,
        mert_layers: int = 25, mert_dim: int = 1024,
        w2v_layers:  int = 13, w2v_dim:  int = 768,
        mel_dim:     int = 128,
        bottleneck:  int = 128, dropout: float = 0.4,
    ):
        super().__init__()
        self.fusion_mert = WeightedLayerFusion(mert_layers, mert_dim)
        self.fusion_w2v  = WeightedLayerFusion(w2v_layers,  w2v_dim)
        self.mel_cnn     = MelSpectrogramCNN(out_dim=mel_dim, dropout=dropout)

        concat_dim = mert_dim + w2v_dim + mel_dim  # 1024 + 768 + 128 = 1920

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

    def forward(self, x_mert, x_w2v, x_mel):
        fm = self.fusion_mert(x_mert)               # (B, 1024)
        fw = self.fusion_w2v(x_w2v)                 # (B,  768)
        fc = self.mel_cnn(x_mel)                    # (B,  128)
        fused = torch.cat([fm, fw, fc], dim=1)      # (B, 1920)
        latent = self.head(fused)                   # (B, 128)
        latent_norm = F.normalize(latent, dim=1)
        return self.regressor(latent_norm), latent_norm

    def get_layer_weights(self) -> dict:
        return {
            "mert": self.fusion_mert.get_weights(),
            "w2v":  self.fusion_w2v.get_weights(),
        }


# =============================================================================
# 3. SpectrogramOnlyModel — MERT (frozen) + MelCNN  (ablation, no wav2vec2)
# =============================================================================

class SpectrogramOnlyModel(nn.Module):
    """
    Ablation isolating the mel-spectrogram's marginal contribution:
      Branch 1: frozen MERT → WeightedLayerFusion → 1024-d
      Branch 3: MelSpectrogramCNN (trainable)     →  128-d
      Concat → 1152-d → head → 128-d latent → regressor → 2

    forward(x_mert, x_mel) → (preds (B,2), latent_norm (B,128))
    """
    def __init__(
        self,
        mert_layers: int = 25, mert_dim: int = 1024,
        mel_dim:     int = 128,
        bottleneck:  int = 128, dropout: float = 0.4,
    ):
        super().__init__()
        self.fusion_mert = WeightedLayerFusion(mert_layers, mert_dim)
        self.mel_cnn     = MelSpectrogramCNN(out_dim=mel_dim, dropout=dropout)

        concat_dim = mert_dim + mel_dim  # 1024 + 128 = 1152

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

    def forward(self, x_mert, x_mel):
        fm = self.fusion_mert(x_mert)               # (B, 1024)
        fc = self.mel_cnn(x_mel)                    # (B,  128)
        fused = torch.cat([fm, fc], dim=1)          # (B, 1152)
        latent = self.head(fused)
        latent_norm = F.normalize(latent, dim=1)
        return self.regressor(latent_norm), latent_norm

    def get_layer_weights(self) -> dict:
        return {"mert": self.fusion_mert.get_weights()}
