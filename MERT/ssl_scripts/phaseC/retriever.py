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

    # Russell-quadrant codes → human-readable prototype names.
    QUAD_NAMES = {
        "HVHA": "Happy/Energetic",
        "HVLA": "Calm",
        "LVHA": "Tense/Angry",
        "LVLA": "Sad",
    }

    def __init__(self, index):
        self.index   = index
        self.latents = index["latents"]
        self.n       = len(self.latents)
        self._backend = self._build_backend()
        self.centroids = self._build_centroids()

    # ── Ante-hoc 4-prototype (quadrant centroid) profile ────────────────────
    @staticmethod
    def _quad_mask(arousal, valence, code):
        """Boolean mask of index songs in a given Russell quadrant
        (ground-truth V-A; same 0.5 cutoffs as the evaluator/explainer)."""
        hi_a, hi_v = arousal >= 0.5, valence >= 0.5
        return {
            "HVHA": hi_v & hi_a,
            "HVLA": hi_v & ~hi_a,
            "LVHA": ~hi_v & hi_a,
            "LVLA": ~hi_v & ~hi_a,
        }[code]

    def _build_centroids(self):
        """Mean latent vector per Russell quadrant (the 4 emotion prototypes),
        L2-normalized so cosine similarity is a dot product. Built from the
        index latents grouped by ground-truth V-A — computed once at init."""
        a, v = self.index["arousal"], self.index["valence"]
        centroids = {}
        for code in self.QUAD_NAMES:
            mask = self._quad_mask(a, v, code)
            if mask.sum() == 0:
                continue
            c = self.latents[mask].mean(axis=0)
            norm = np.linalg.norm(c)
            centroids[code] = (c / norm if norm > 1e-8 else c).astype(np.float32)
        print(f"  🎯 Built {len(centroids)} quadrant prototype centroids "
              f"({', '.join(centroids)})")
        return centroids

    def prototype_profile(self, query_latent):
        """Ante-hoc activation profile: cosine similarity of the query to each
        of the 4 quadrant prototype centroids.

        Returns:
            {
              "similarities": {code: cosine_sim, ...},   # all 4 quadrants
              "best": code,                              # highest-similarity quadrant
              "best_name": str,                          # human-readable prototype
            }
        """
        q = query_latent.reshape(-1).astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = {code: float(np.dot(q, c)) for code, c in self.centroids.items()}
        best = max(sims, key=sims.get) if sims else None
        return {
            "similarities": sims,
            "best": best,
            "best_name": self.QUAD_NAMES.get(best, "n/a"),
        }

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
