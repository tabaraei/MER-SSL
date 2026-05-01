"""
retriever.py — k-NN Emotion Retriever with Contrastive Foil Support
====================================================================
Performs cosine similarity search in the latent space index.
Supports both FAISS (fast) and sklearn (fallback) backends.

New: query_foils() returns the most DISSIMILAR songs, enabling
contrastive (XAI) explanations — a standard technique in explainable AI
(Miller, 2019, "Explanation in Artificial Intelligence").
"""

import numpy as np


class EmotionRetriever:

    def __init__(self, index):
        self.index   = index
        self.latents = index["latents"]
        self.n       = len(self.latents)
        self._backend = self._build_backend()

    def _build_backend(self):
        try:
            import faiss
            dim = self.latents.shape[1]
            idx = faiss.IndexFlatIP(dim)
            idx.add(self.latents)
            print(f"  🚀 FAISS backend ready | {self.n} vectors | dim={dim}")
            return ("faiss", idx)
        except ImportError:
            print("  ⚠️  FAISS not found — using sklearn cosine fallback (slower)")
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(self.n, 50),
                                  metric="cosine", algorithm="brute")
            nn.fit(self.latents)
            return ("sklearn", nn)

    def query(self, query_latent, k=5, exclude_self=True):
        """Returns the k most similar songs to query_latent."""
        q       = query_latent.reshape(1, -1).astype(np.float32)
        fetch_k = k + 1 if exclude_self else k

        backend, engine = self._backend
        if backend == "faiss":
            sims, idxs = engine.search(q, fetch_k)
            sims, idxs = sims[0], idxs[0]
        else:
            dists, idxs = engine.kneighbors(q, n_neighbors=fetch_k)
            sims  = 1.0 - dists[0]
            idxs  = idxs[0]

        results, rank = [], 1
        for sim, idx in zip(sims, idxs):
            if exclude_self and np.allclose(self.latents[idx], q, atol=1e-5):
                continue
            results.append(self._make_result(rank, int(idx), float(sim)))
            rank += 1
            if rank > k:
                break
        return results

    def query_foils(self, query_latent, n_foils=3):
        """
        Returns the n_foils most DISSIMILAR songs (lowest cosine similarity).
        Used to build contrastive explanations: these are songs that were
        explicitly NOT retrieved, grounding the explanation in what was rejected.
        """
        q    = query_latent.reshape(1, -1).astype(np.float32)
        sims = (self.latents @ q.T).flatten()
        worst_idxs = np.argsort(sims)[:n_foils]
        return [self._make_result(i + 1, int(idx), float(sims[idx]))
                for i, idx in enumerate(worst_idxs)]

    def _make_result(self, rank, idx, similarity):
        return {
            "rank":         rank,
            "music_id":     int(self.index["music_ids"][idx]),
            "similarity":   similarity,
            "arousal":      float(self.index["arousal"][idx]),
            "valence":      float(self.index["valence"][idx]),
            "pred_arousal": float(self.index["pred_arousal"][idx]),
            "pred_valence": float(self.index["pred_valence"][idx]),
            "eda_feats":    self.index["eda_feats"][idx],
        }
