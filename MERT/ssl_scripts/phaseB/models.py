"""
models.py — MER Model Architecture
====================================
Contains:
  - WeightedLayerFusion : Learnable softmax fusion over all 25 MERT layers
  - MERModel            : Full hybrid model (baseline or hybrid mode)
  - get_layer_weights   : Utility to extract and analyze learned weights (NEW)
  - DualSSLModel        : MERT + wav2vec2 dual-encoder fusion model (NEW)
  - DualSSLDomainModel  : DualSSL + learned domain embedding (joint learning) (NEW)
  - analyze_dual_layer_weights : Side-by-side layer attribution for both encoders (NEW)
"""

import os
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
# 2b. NEW: Dual-SSL Model — MERT + wav2vec2
# =============================================================================

class DualSSLModel(nn.Module):
    """
    Dual self-supervised encoder model for Music Emotion Recognition.

    Fuses two complementary frozen SSL encoders via independent learnable
    layer-fusion heads, then concatenates the fused representations:
        - MERT-v1-330M  : music-pretrained  (25 layers, 1024-d)
        - wav2vec2-base  : speech-pretrained (13 layers,  768-d)

    Motivation (Simonetta et al. 2024): speech-pretrained representations
    capture complementary acoustic cues (voice timbre, articulation, energy
    dynamics) that music-only pretraining misses — especially useful for
    breaking the Valence ceiling.

    Input shapes  : x_mert (B, 25, 1024), x_w2v (B, 13, 768)
    Output shapes : preds (B, 2) [arousal, valence], latent_norm (B, bottleneck)
    """
    def __init__(
        self,
        mert_layers: int = 25,  mert_dim: int = 1024,
        w2v_layers:  int = 13,  w2v_dim:  int = 768,
        bottleneck:  int = 128, dropout:  float = 0.4,
    ):
        super().__init__()
        self.fusion_mert = WeightedLayerFusion(mert_layers, mert_dim)
        self.fusion_w2v  = WeightedLayerFusion(w2v_layers,  w2v_dim)

        concat_dim = mert_dim + w2v_dim  # 1024 + 768 = 1792

        self.head = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )
        self.regressor = nn.Linear(bottleneck, 2)

    def forward(self, x_mert: torch.Tensor, x_w2v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_mert : (B, 25, 1024)  stacked MERT layer outputs
            x_w2v  : (B, 13,  768)  stacked wav2vec2 layer outputs
        Returns:
            preds       : (B, 2)
            latent_norm : (B, bottleneck)  L2-normalized for SupCR
        """
        fused_m = self.fusion_mert(x_mert)              # (B, 1024)
        fused_w = self.fusion_w2v(x_w2v)                # (B,  768)
        fused   = torch.cat([fused_m, fused_w], dim=1)  # (B, 1792)
        latent  = self.head(fused)                      # (B, bottleneck)
        latent_norm = F.normalize(latent, dim=1)        # L2-normalize for SupCR
        preds   = self.regressor(latent_norm)
        return preds, latent_norm

    def get_layer_weights(self) -> dict:
        """Return both encoders' learned softmax weight distributions."""
        return {
            "mert": self.fusion_mert.get_weights(),
            "w2v":  self.fusion_w2v.get_weights(),
        }


# =============================================================================
# 2c. NEW: Dual-SSL + Domain Conditioning (Simonetta joint-learning, SSL-native)
# =============================================================================

class DualSSLDomainModel(nn.Module):
    """
    DualSSLModel + a learned domain embedding concatenated before the head.

    Conditions the emotion mapping on the source domain:
        domain 0 = music (PMEmo)   |   domain 1 = generalized sound (IADS-E)

    This is the SSL-native replication of Simonetta et al. (2024)'s shared
    emotional feature space: both domains are projected into one V-A latent,
    but the head can offset/condition on which domain a sample came from.
    """
    def __init__(
        self,
        mert_layers: int = 25, mert_dim: int = 1024,
        w2v_layers:  int = 13, w2v_dim:  int = 768,
        domain_emb_dim: int = 16, bottleneck: int = 128, dropout: float = 0.4,
    ):
        super().__init__()
        self.fusion_mert = WeightedLayerFusion(mert_layers, mert_dim)
        self.fusion_w2v  = WeightedLayerFusion(w2v_layers,  w2v_dim)
        self.domain_emb  = nn.Embedding(2, domain_emb_dim)

        concat_dim = mert_dim + w2v_dim + domain_emb_dim  # 1024 + 768 + 16 = 1808

        self.head = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )
        self.regressor = nn.Linear(bottleneck, 2)

    def forward(self, x_mert: torch.Tensor, x_w2v: torch.Tensor,
                domain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_mert : (B, 25, 1024)
            x_w2v  : (B, 13,  768)
            domain : (B,) long tensor in {0, 1}
        Returns:
            preds (B, 2), latent_norm (B, bottleneck)
        """
        fm = self.fusion_mert(x_mert)          # (B, 1024)
        fw = self.fusion_w2v(x_w2v)            # (B,  768)
        de = self.domain_emb(domain)           # (B,  16)
        fused = torch.cat([fm, fw, de], dim=1)  # (B, 1808)
        latent = self.head(fused)
        latent_norm = F.normalize(latent, dim=1)
        return self.regressor(latent_norm), latent_norm

    def get_layer_weights(self) -> dict:
        """Both encoders' learned softmax weight distributions."""
        return {
            "mert": self.fusion_mert.get_weights(),
            "w2v":  self.fusion_w2v.get_weights(),
        }


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


# =============================================================================
# 4. NEW: Dual-SSL Layer Weight Analysis
# =============================================================================

def _analyze_weight_vector(weights: np.ndarray, name: str, save_path: str = None) -> dict:
    """
    Shared analysis logic (mirrors analyze_layer_weights) applied to a single
    encoder's softmax weight vector. Region boundaries are proportional thirds
    so it generalizes across encoders with different layer counts.
    """
    n = len(weights)
    a, b = n // 3, 2 * n // 3

    early_mass = weights[:a].sum()
    mid_mass   = weights[a:b].sum()
    late_mass  = weights[b:].sum()
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

    print(f"\n📊 Layer Weight Analysis — {name} ({n} layers)")
    print(f"  Top-5 layers by importance : {top_layers}")
    print(f"  Early third (0-{a-1}) mass  : {early_mass:.3f}")
    print(f"  Mid   third ({a}-{b-1}) mass : {mid_mass:.3f}")
    print(f"  Late  third ({b}-{n-1}) mass : {late_mass:.3f}")
    print(f"  Weight entropy             : {entropy:.4f}  (max={np.log(n):.4f})")

    if save_path:
        np.save(save_path, weights)
        print(f"  Weights saved to: {save_path}")

    return result


def analyze_dual_layer_weights(model: DualSSLModel, save_dir: str = ".") -> dict:
    """
    Analyzes the learned layer-fusion weights of BOTH encoders in a
    DualSSLModel. Saves:
      - layer_weights_mert.npy
      - layer_weights_w2v.npy
      - dual_layer_weights.png  (side-by-side bar plot)

    Returns a dict {"mert": <analysis>, "w2v": <analysis>}.
    """
    w = model.get_layer_weights()
    res_mert = _analyze_weight_vector(
        w["mert"], "MERT", os.path.join(save_dir, "layer_weights_mert.npy")
    )
    res_w2v = _analyze_weight_vector(
        w["w2v"], "wav2vec2", os.path.join(save_dir, "layer_weights_w2v.npy")
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    wm = w["mert"]
    axes[0].bar(np.arange(len(wm)), wm, color="#4C9BE8", edgecolor="white", linewidth=0.5)
    axes[0].set_title("MERT-v1-330M — Learned Layer Weights", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("MERT Transformer Layer")
    axes[0].set_ylabel("Softmax Weight")
    axes[0].set_xticks(np.arange(len(wm)))
    axes[0].set_xticklabels([str(i) for i in range(len(wm))], fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)

    ww = w["w2v"]
    axes[1].bar(np.arange(len(ww)), ww, color="#E84C4C", edgecolor="white", linewidth=0.5)
    axes[1].set_title("wav2vec2-base — Learned Layer Weights", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("wav2vec2 Layer (0 = CNN feature projection)")
    axes[1].set_ylabel("Softmax Weight")
    axes[1].set_xticks(np.arange(len(ww)))
    axes[1].set_xticklabels([str(i) for i in range(len(ww))], fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "dual_layer_weights.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📈 Dual layer weight plot saved to: {out_path}")

    return {"mert": res_mert, "w2v": res_w2v}
