"""
index_builder.py — Vector Index Construction for Phase C Retrieval
===================================================================
Encodes all PMEmo songs through the Phase B model's latent space and
stores a structured NumPy index used by EmotionRetriever at query time.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class VectorIndexBuilder:
    """
    Builds and persists a latent-space index over the full dataset.

    Stored fields per song:
        latents      : (N, D)  L2-normalized latent vectors (FAISS input)
        music_ids    : (N,)    integer PMEmo music IDs
        arousal      : (N,)    ground-truth arousal
        valence      : (N,)    ground-truth valence
        pred_arousal : (N,)    model-predicted arousal
        pred_valence : (N,)    model-predicted valence
        eda_feats    : (N, 7)  EDA statistical features (zeros if unavailable)
        layer_weights: (L,)    learned MERT layer fusion weights (global, may be None)
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self.model.eval()

    def build(self, X, Y, music_ids, eda_features=None, layer_weights=None):
        print(f"\n🔨  Building vector index over {len(X)} songs...")
        loader  = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)
        latents, preds_a, preds_v = [], [], []

        with torch.no_grad():
            for (batch,) in loader:
                output, latent = self.model(batch.to(self.device))
                latents.append(latent.cpu())
                preds_a.append(output[:, 0].cpu())
                preds_v.append(output[:, 1].cpu())

        latents  = torch.cat(latents, dim=0).numpy()
        preds_a  = torch.cat(preds_a, dim=0).numpy()
        preds_v  = torch.cat(preds_v, dim=0).numpy()

        norms   = np.linalg.norm(latents, axis=1, keepdims=True)
        latents = latents / np.where(norms < 1e-8, 1.0, norms)

        eda = eda_features if eda_features is not None else np.zeros((len(X), 7), dtype=np.float32)

        index = {
            "latents":      latents.astype(np.float32),
            "music_ids":    np.array(music_ids, dtype=np.int32),
            "arousal":      Y[:, 0].numpy(),
            "valence":      Y[:, 1].numpy(),
            "pred_arousal": preds_a,
            "pred_valence": preds_v,
            "eda_feats":    eda.astype(np.float32),
            "layer_weights": layer_weights,
        }
        print(f"  ✅ Index built | latents: {latents.shape} | "
              f"EDA: {'loaded' if eda_features is not None else 'zeros'} | "
              f"layer weights: {'captured' if layer_weights is not None else 'none'}")
        return index

    @staticmethod
    def save(index, path):
        np.save(path, index)
        print(f"  💾 Index saved → {path}")

    @staticmethod
    def load(path):
        index = np.load(path, allow_pickle=True).item()
        print(f"  📦 Index loaded from {path} | {len(index['music_ids'])} songs")
        return index
