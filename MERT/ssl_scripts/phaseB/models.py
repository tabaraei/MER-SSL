"""
models.py — MER Model Architecture
====================================
Contains:
  - WeightedLayerFusion : Learnable softmax fusion over all 25 MERT layers
  - MERModel            : Full hybrid model (baseline or hybrid mode)
  - get_layer_weights   : Utility to extract and analyze learned weights (NEW)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# 1. Weighted Layer Fusion Module
# =============================================================================

class WeightedLayerFusion(nn.Module):
    """
    Learns a softmax distribution over all N MERT transformer layers.
    Output is a weighted sum of all layer embeddings.

    Args:
        n_layers    : Number of MERT layers (25 for MERT-v1-330M)
        hidden_dim  : Embedding dimension per layer (1024 for MERT-v1-330M)
    """
    def __init__(self, n_layers: int = 25, hidden_dim: int = 1024):
        super().__init__()
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        # Learnable scalar weight per layer — initialized uniformly
        self.layer_weights = nn.Parameter(torch.ones(n_layers))

    def forward(self, all_layers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_layers : (B, n_layers, hidden_dim)  stacked MERT layer outputs
        Returns:
            fused      : (B, hidden_dim)
        """
        weights = F.softmax(self.layer_weights, dim=0)       # (n_layers,)
        # Weighted sum across layer dimension
        fused = (all_layers * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return fused

    def get_weights(self) -> np.ndarray:
        """Returns the softmax-normalized weights as a numpy array for analysis."""
        with torch.no_grad():
            return F.softmax(self.layer_weights, dim=0).cpu().numpy()


# =============================================================================
# 2. MER Model — Baseline and Hybrid
# =============================================================================

class MERModel(nn.Module):
    """
    Music Emotion Recognition Model.

    Modes:
        'baseline' : Uses only the last MERT layer (Layer 24). Simple linear probe.
        'hybrid'   : Weighted fusion of all 25 layers + deeper regression head
                     with Dropout for regularization.

    Input shape  : (B, n_layers, hidden_dim)  — all layers stacked
    Output shape : (B, 2)  — [arousal, valence]
    Also returns : (B, bottleneck_dim) latent for SupCR loss
    """
    def __init__(
        self,
        mode:          str = 'hybrid',
        n_layers:      int = 25,
        hidden_dim:    int = 1024,
        bottleneck:    int = 128,
        dropout:       float = 0.4,
    ):
        super().__init__()
        self.mode = mode

        if mode == 'hybrid':
            self.fusion = WeightedLayerFusion(n_layers, hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, bottleneck),
                nn.LayerNorm(bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
            )
        else:  # baseline: take only layer 24
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck),
                nn.ReLU(),
            )

        self.regressor = nn.Linear(bottleneck, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, n_layers, hidden_dim)
        Returns:
            preds  : (B, 2)   valence-arousal predictions
            latent : (B, bottleneck_dim)  L2-normalized for contrastive loss
        """
        if self.mode == 'hybrid':
            fused = self.fusion(x)          # (B, hidden_dim)
        else:
            fused = x[:, -1, :]            # (B, hidden_dim) — last layer only

        latent = self.head(fused)           # (B, bottleneck)
        latent_norm = F.normalize(latent, dim=1)   # L2-normalize for SupCR
        preds  = self.regressor(latent_norm)
        return preds, latent_norm

    def get_layer_weights(self) -> np.ndarray:
        """
        Returns learned softmax layer weights (only valid for hybrid mode).
        Use for musicological analysis and thesis visualization.
        """
        if self.mode != 'hybrid':
            raise ValueError("Layer weights only available in hybrid mode.")
        return self.fusion.get_weights()


# =============================================================================
# 3. NEW: Layer Weight Analysis Utilities
# =============================================================================

def analyze_layer_weights(model: MERModel, save_path: str = None) -> dict:
    """
    Extracts and analyzes the learned layer weight distribution.

    Returns a dict with:
        - weights         : (25,) numpy array of softmax weights
        - top_layers      : indices of top-5 most important layers
        - early_mass      : total weight assigned to layers 0–7 (low-level)
        - mid_mass        : total weight assigned to layers 8–15 (mid-level)
        - late_mass       : total weight assigned to layers 16–24 (high-level)
        - entropy         : entropy of weight distribution (higher = more spread)

    Usage:
        analysis = analyze_layer_weights(model)
        print(analysis)
    """
    weights = model.get_layer_weights()           # (25,)
    n = len(weights)

    early_mass = weights[:8].sum()
    mid_mass   = weights[8:16].sum()
    late_mass  = weights[16:].sum()
    top_layers = np.argsort(weights)[::-1][:5].tolist()
    entropy    = float(-np.sum(weights * np.log(weights + 1e-10)))

    result = {
        "weights":     weights,
        "top_layers":  top_layers,
        "early_mass":  float(early_mass),
        "mid_mass":    float(mid_mass),
        "late_mass":   float(late_mass),
        "entropy":     entropy,
    }

    print("\n📊 Layer Weight Analysis")
    print(f"  Top-5 layers by importance : {top_layers}")
    print(f"  Early layers (0-7) mass    : {early_mass:.3f}")
    print(f"  Mid   layers (8-15) mass   : {mid_mass:.3f}")
    print(f"  Late  layers (16-24) mass  : {late_mass:.3f}")
    print(f"  Weight entropy             : {entropy:.4f}  (max={np.log(n):.4f})")

    if save_path:
        np.save(save_path, weights)
        print(f"  Weights saved to: {save_path}")

    return result


def plot_layer_weights(model: MERModel, save_path: str = "layer_weights.png"):
    """
    Plots the learned layer weight distribution for thesis visualization.
    Saves figure to save_path.

    Requires matplotlib.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    weights = model.get_layer_weights()
    n = len(weights)
    layers = np.arange(n)

    # Color-code by region
    colors = []
    for i in range(n):
        if i < 8:
            colors.append("#4C9BE8")    # blue = early (acoustic)
        elif i < 16:
            colors.append("#F5A623")    # orange = mid
        else:
            colors.append("#E84C4C")    # red = late (semantic)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(layers, weights, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("MERT Transformer Layer", fontsize=12)
    ax.set_ylabel("Learned Softmax Weight", fontsize=12)
    ax.set_title("Learned Layer Importance in WeightedLayerFusion", fontsize=13, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) for i in layers], fontsize=8)

    patches = [
        mpatches.Patch(color="#4C9BE8", label="Early (0–7): Low-level acoustic"),
        mpatches.Patch(color="#F5A623", label="Mid (8–15): Rhythmic/structural"),
        mpatches.Patch(color="#E84C4C", label="Late (16–24): Semantic/affective"),
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 Layer weight plot saved to: {save_path}")
