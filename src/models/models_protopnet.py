"""
models_protopnet.py — Audio ProtoPNet (Prototypical Part Network) for MER
==========================================================================
A learnable-prototype alternative to the post-hoc 4-centroid method
(`phaseC/extra_metrics.py`), which applied fixed K-means-style quadrant
centroids AFTER training and lost to the majority baseline (0.506 / 0.611).

ProtoPNet (Chen et al. 2019, "This Looks Like That", NeurIPS) instead learns
prototype vectors *during* gradient descent and classifies by L2 distance
between the encoded audio and each prototype. For audio there are no spatial
"parts", so a prototype is simply a learnable point in the 128-d latent space
(a prototypical *song-embedding* rather than a prototypical image patch).

Architecture:
    audio features ─▶ Encoder f  (MERT fusion + head)        ─▶ z ∈ ℝ^128 (L2-norm)
                     Prototype layer g (m prototypes/class)   ─▶ L2 dist  d_ij = ‖z_i − p_j‖²
                     similarity  s_ij = log((d_ij+1)/(d_ij+ε))
                     Linear h (n_proto → 4, no bias)          ─▶ quadrant logits

Losses (Chen et al. 2019):
    CE(logits, y)
  + λ_clst · mean_i  min_{j: class(j)=y_i}  d_ij        (pull to own-class proto)
  − λ_sep  · mean_i  min_{j: class(j)≠y_i}  d_ij        (push from other-class protos)
  + λ_l1   · ‖h_offclass‖₁                              (sparse, interpretable head)

The last layer is initialised with the ProtoPNet identity prior (+1 to a
prototype's own class, −0.5 elsewhere) so each prototype is, by construction,
evidence FOR its assigned quadrant — keeping the model ante-hoc interpretable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models import WeightedLayerFusion


class AudioProtoPNet(nn.Module):
    def __init__(self, n_layers=25, hidden_dim=1024, latent_dim=128,
                 n_classes=4, protos_per_class=5, dropout=0.4, eps=1e-4):
        super().__init__()
        self.n_classes = n_classes
        self.protos_per_class = protos_per_class
        self.n_proto = n_classes * protos_per_class
        self.latent_dim = latent_dim
        self.eps = eps

        # ── Encoder f: WeightedLayerFusion + head → 128-d L2-normalised latent ──
        self.fusion = WeightedLayerFusion(n_layers, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU(),
        )

        # ── Prototype layer g: learnable prototypes in latent space ──
        self.prototypes = nn.Parameter(torch.rand(self.n_proto, latent_dim))

        # fixed prototype→class identity (block structure): proto j belongs to
        # class j // protos_per_class
        ident = torch.zeros(self.n_proto, n_classes)
        for j in range(self.n_proto):
            ident[j, j // protos_per_class] = 1.0
        self.register_buffer("proto_class_identity", ident)         # (n_proto, n_classes)

        # ── Classification layer h: n_proto → n_classes, no bias ──
        self.last_layer = nn.Linear(self.n_proto, n_classes, bias=False)
        self._init_last_layer(pos=1.0, neg=-0.5)

    def _init_last_layer(self, pos=1.0, neg=-0.5):
        """ProtoPNet identity prior: +pos for a prototype's own class, neg elsewhere."""
        w = self.proto_class_identity.t() * (pos - neg) + neg            # (n_classes, n_proto)
        with torch.no_grad():
            self.last_layer.weight.copy_(w)

    def encode(self, x):
        """audio features (B, n_layers, hidden) → L2-normalised latent (B, latent_dim)."""
        fused = self.fusion(x)
        latent = self.head(fused)
        return F.normalize(latent, dim=1)

    def forward(self, x):
        z = self.encode(x)                                              # (B, D)
        # squared L2 distance to every prototype
        dist = torch.cdist(z, self.prototypes, p=2) ** 2                # (B, n_proto)
        # ProtoPNet bounded similarity (monotone decreasing in distance)
        sim = torch.log((dist + 1.0) / (dist + self.eps))               # (B, n_proto)
        logits = self.last_layer(sim)                                   # (B, n_classes)
        return logits, dist, z

    # ── ProtoPNet structural losses ─────────────────────────────────────────
    def cluster_separation_costs(self, dist, labels):
        """Returns (cluster_cost, separation_cost) for a batch.

        cluster    = mean_i  min over OWN-class prototypes of d_ij      (minimise)
        separation = mean_i  min over OTHER-class prototypes of d_ij    (maximise)
        """
        B = dist.size(0)
        # (B, n_proto) mask: 1 where prototype j is of sample i's class
        own = self.proto_class_identity[:, labels].t()                  # (B, n_proto)
        big = dist.max().detach() + 1.0
        own_dist = torch.where(own.bool(), dist, torch.full_like(dist, big))
        oth_dist = torch.where(own.bool(), torch.full_like(dist, big), dist)
        cluster = own_dist.min(dim=1).values.mean()
        separation = oth_dist.min(dim=1).values.mean()
        return cluster, separation

    def l1_offclass(self):
        """L1 on the off-class connections of the head (ProtoPNet sparsity prior)."""
        off = (1.0 - self.proto_class_identity.t())                     # (n_classes, n_proto)
        return (self.last_layer.weight * off).abs().sum()
