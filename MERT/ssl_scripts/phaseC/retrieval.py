"""
phaseC_retrieval.py — Explainable Music Emotion Retrieval (Phase C)
=====================================================================
Master's Thesis: Explainable & Perceptual Music Emotion Recognition
Student: Arvin Jafari Moghadam Fard

Pipeline:
  1. Load trained Phase B hybrid model (best_model.pt)
  2. Encode all PMEmo songs → 1024-D normalized latent vectors
  3. Build FAISS cosine-similarity index over the latent space
  4. For a query song, retrieve k nearest neighbors
  5. Generate a music-theoretic, EDA-grounded natural language explanation

Usage:
    # Step 1 — Build the index (run once):
    python retrieval.py --mode build \
        --model_path best_model.pt \
        --feat_path  pmemo_mert_all_layers.pt \
        --csv_path   /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
        --eda_dir    /datasets/emotions/PMEmo2019/EDA \
        --index_path prototypes.npy

    # Step 2 — Query by music ID:
    python retrieval.py --mode query \
        --query_id   760 \
        --index_path prototypes.npy \
        --feat_path  pmemo_mert_all_layers.pt \
        --csv_path   /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
        --eda_dir    /datasets/emotions/PMEmo2019/EDA \
        --model_path best_model.pt \
        --top_k      5

    # Step 3 — Evaluate retrieval quality (precision@k vs baseline):
    python retrieval.py --mode evaluate \
        --index_path prototypes.npy \
        --feat_path  pmemo_mert_all_layers.pt \
        --csv_path   /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
        --model_path best_model.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# ── Resolve phaseB directory so models.py / data_utils.py are importable ──────
# Default layout:  .../ssl_scripts/phaseC/phaseC_retrieval.py
#                  .../ssl_scripts/phaseB/models.py
# Override with --phaseb_dir if your folder layout differs.
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PHASEB_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'phaseB'))
if os.path.isdir(_PHASEB_DIR):
    sys.path.insert(0, _PHASEB_DIR)

from models import MERModel
from data_utils import load_pmemo_data
# ==============================================================================
# 1. Index Builder — encodes all songs and saves the vector index
# ==============================================================================

class VectorIndexBuilder:
    """
    Encodes every song in the dataset through the Phase B model's latent space
    and saves a structured index (prototypes.npy) for Phase C retrieval.

    The index stores:
        latents  : (N, 1024) L2-normalized MERT latent vectors  ← FAISS input
        music_ids: (N,)      integer music IDs for metadata lookup
        arousal  : (N,)      predicted arousal scores
        valence  : (N,)      predicted valence scores
        eda_feats: (N, 7)    EDA statistical features (zeros if unavailable)
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self.model.eval()

    def build(self, X, Y, music_ids, eda_features=None):
        """
        Args:
            X          : (N, 25, 1024) MERT all-layer embeddings
            Y          : (N, 2)        [arousal, valence] ground truth labels
            music_ids  : list of int   aligned music IDs
            eda_features: (N, 7) numpy or None

        Returns:
            index dict saved via np.save
        """
        print(f"\n🔨  Building vector index over {len(X)} songs...")
        loader   = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)
        latents  = []
        preds_a, preds_v = [], []

        with torch.no_grad():
            for (batch,) in loader:
                output, latent = self.model(batch.to(self.device))
                latents.append(latent.cpu())
                preds_a.append(output[:, 0].cpu())
                preds_v.append(output[:, 1].cpu())

        latents  = torch.cat(latents,  dim=0).numpy()   # (N, 1024)
        preds_a  = torch.cat(preds_a,  dim=0).numpy()
        preds_v  = torch.cat(preds_v,  dim=0).numpy()

        # L2-normalize for cosine similarity via dot product
        norms   = np.linalg.norm(latents, axis=1, keepdims=True)
        latents = latents / np.where(norms < 1e-8, 1.0, norms)

        eda = eda_features if eda_features is not None else np.zeros((len(X), 7), dtype=np.float32)

        index = {
            "latents":   latents.astype(np.float32),
            "music_ids": np.array(music_ids, dtype=np.int32),
            "arousal":   Y[:, 0].numpy(),
            "valence":   Y[:, 1].numpy(),
            "pred_arousal": preds_a,
            "pred_valence": preds_v,
            "eda_feats": eda.astype(np.float32),
        }
        print(f"  ✅ Index built | latents: {latents.shape}")
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


# ==============================================================================
# 2. Retrieval Engine — cosine k-NN with optional FAISS acceleration
# ==============================================================================

class EmotionRetriever:
    """
    k-NN retrieval in the emotion-aware latent space.

    Uses FAISS (IndexFlatIP on L2-normalized vectors = cosine similarity)
    with scikit-learn cosine fallback if FAISS is not installed.
    """

    def __init__(self, index):
        self.index    = index
        self.latents  = index["latents"]          # (N, 1024) float32
        self.n        = len(self.latents)
        self._backend = self._build_backend()

    def _build_backend(self):
        try:
            import faiss
            dim  = self.latents.shape[1]
            idx  = faiss.IndexFlatIP(dim)         # Inner Product on L2-norm = cosine
            idx.add(self.latents)
            print(f"  🚀 FAISS backend ready | {self.n} vectors | dim={dim}")
            return ("faiss", idx)
        except ImportError:
            print("  ⚠️  FAISS not found — using sklearn cosine fallback (slower)")
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
            nn.fit(self.latents)
            return ("sklearn", nn)

    def query(self, query_latent, k=5, exclude_self=True):
        """
        Args:
            query_latent : (1024,) numpy float32, L2-normalized
            k            : number of neighbors to return
            exclude_self : skip exact match (query is in the index)

        Returns:
            List of dicts, each containing:
                rank, music_id, similarity, arousal, valence,
                pred_arousal, pred_valence, eda_feats
        """
        q = query_latent.reshape(1, -1).astype(np.float32)
        fetch_k = k + 1 if exclude_self else k

        backend, engine = self._backend
        if backend == "faiss":
            sims, idxs = engine.search(q, fetch_k)
            sims, idxs = sims[0], idxs[0]
        else:
            dists, idxs = engine.kneighbors(q, n_neighbors=fetch_k)
            sims  = 1.0 - dists[0]
            idxs  = idxs[0]

        results = []
        rank    = 1
        for sim, idx in zip(sims, idxs):
            if exclude_self and np.allclose(self.latents[idx], q, atol=1e-5):
                continue
            results.append({
                "rank":         rank,
                "music_id":     int(self.index["music_ids"][idx]),
                "similarity":   float(sim),
                "arousal":      float(self.index["arousal"][idx]),
                "valence":      float(self.index["valence"][idx]),
                "pred_arousal": float(self.index["pred_arousal"][idx]),
                "pred_valence": float(self.index["pred_valence"][idx]),
                "eda_feats":    self.index["eda_feats"][idx],
            })
            rank += 1
            if rank > k:
                break

        return results


# ==============================================================================
# 3. Explanation Generator — RAG context + natural language output
# ==============================================================================

class ExplainableRAG:
    """
    Generates music-theoretic explanations for retrieved neighbors.

    Two explanation modes:
        'template' : rule-based, deterministic (used by default)
        'llm'      : sends context to an LLM via Ollama or Anthropic API (optional)

    The template mode is fully self-contained and produces thesis-quality
    explanations grounded in arousal/valence coordinates and EDA signals.
    """

    # Russell's quadrant labels
    QUADRANT = {
        (True,  True):  ("High Arousal / High Valence", "energetic and positive",   "HVHA"),
        (True,  False): ("High Arousal / Low Valence",  "intense and negative",     "LVHA"),
        (False, True):  ("Low Arousal / High Valence",  "calm and positive",        "HVLA"),
        (False, False): ("Low Arousal / Low Valence",   "melancholic and subdued",  "LVLA"),
    }

    def _quadrant(self, arousal, valence, mid=0.5):
        return self.QUADRANT[(arousal >= mid, valence >= mid)]

    def _eda_summary(self, eda_feats):
        """Converts 7-dim EDA feature vector into readable text."""
        if eda_feats is None or np.allclose(eda_feats, 0):
            return "no physiological data available"
        mean_eda, std_eda, slope, n_peaks, mean_amp, max_eda, high_ratio = eda_feats
        activity = "elevated" if mean_eda > 0.5 else "moderate" if mean_eda > 0.25 else "low"
        peaks_desc = f"{int(round(n_peaks * 10))} SCR peaks" if n_peaks > 0 else "minimal skin conductance responses"
        trend = "rising" if slope > 0.01 else "falling" if slope < -0.01 else "stable"
        return (f"{activity} electrodermal activity ({peaks_desc}, {trend} trend, "
                f"high-activity ratio={high_ratio:.0%})")

    def _arousal_description(self, arousal):
        if arousal >= 0.75: return "very high"
        if arousal >= 0.55: return "high"
        if arousal >= 0.45: return "moderate"
        if arousal >= 0.25: return "low"
        return "very low"

    def _valence_description(self, valence):
        if valence >= 0.75: return "very positive"
        if valence >= 0.55: return "positive"
        if valence >= 0.45: return "neutral"
        if valence >= 0.25: return "negative"
        return "very negative"

    def generate_template(self, query_id, query_arousal, query_valence,
                          query_eda, neighbors):
        """
        Produces a structured natural-language explanation.

        Args:
            query_id      : int
            query_arousal : float  (ground truth or predicted)
            query_valence : float
            query_eda     : (7,) numpy or None
            neighbors     : list of dicts from EmotionRetriever.query()

        Returns:
            str — human-readable explanation
        """
        q_label, q_desc, q_quad = self._quadrant(query_arousal, query_valence)
        q_eda_text = self._eda_summary(query_eda)

        lines = [
            "=" * 65,
            f"🎵  QUERY: Song ID {query_id}",
            "=" * 65,
            f"  Emotion Region  : {q_label} ({q_quad})",
            f"  Arousal         : {self._arousal_description(query_arousal)} ({query_arousal:.3f})",
            f"  Valence         : {self._valence_description(query_valence)} ({query_valence:.3f})",
            f"  Physiology      : {q_eda_text}",
            "",
            f"📋  RETRIEVAL EXPLANATION",
            "-" * 65,
            (f"  Songs were retrieved based on cosine similarity in a 1024-dimensional "
             f"emotion-aware latent space learned by a MERT-based Hybrid model "
             f"(Weighted Layer Fusion + Supervised Contrastive Regression). "
             f"The query song occupies the '{q_quad}' emotional region — "
             f"characterized by {q_desc} affective qualities. "
             f"Retrieved neighbors share proximity in this space, meaning their "
             f"audio representations were pulled together during contrastive training "
             f"due to similar valence-arousal coordinates."),
            "",
            f"🔍  TOP-{len(neighbors)} RETRIEVED SONGS",
            "-" * 65,
        ]

        for n in neighbors:
            n_label, n_desc, n_quad = self._quadrant(n["arousal"], n["valence"])
            n_eda_text = self._eda_summary(n["eda_feats"])

            # Compute dimensional delta for explanation
            da = n["arousal"] - query_arousal
            dv = n["valence"] - query_valence
            delta_str = []
            if abs(da) > 0.05:
                delta_str.append(f"arousal {'higher' if da > 0 else 'lower'} by {abs(da):.2f}")
            if abs(dv) > 0.05:
                delta_str.append(f"valence {'higher' if dv > 0 else 'lower'} by {abs(dv):.2f}")
            delta_text = (", ".join(delta_str) + " than the query") if delta_str else "nearly identical emotion coordinates"

            lines += [
                f"  ┌─ Rank {n['rank']} | Song ID {n['music_id']} "
                f"(cosine sim = {n['similarity']:.4f})",
                f"  │  Emotion       : {n_label} ({n_quad})",
                f"  │  Arousal       : {self._arousal_description(n['arousal'])} ({n['arousal']:.3f})",
                f"  │  Valence       : {self._valence_description(n['valence'])} ({n['valence']:.3f})",
                f"  │  Physiology    : {n_eda_text}",
                f"  │  Why retrieved : This prototype has {delta_text}. "
                f"Its {n_desc} character aligns with the query's {q_desc} profile. "
                f"The model's contrastive loss placed both songs within a {n['similarity']:.3f}-similarity "
                f"neighborhood in latent space.",
                f"  └─────────────────────────────────────────────────────────",
            ]

        lines += [
            "",
            "🧠  MUSIC-THEORETIC SUMMARY",
            "-" * 65,
            self._generate_summary(query_arousal, query_valence, neighbors),
            "=" * 65,
        ]

        return "\n".join(lines)

    def _generate_summary(self, q_arousal, q_valence, neighbors):
        avg_a = np.mean([n["arousal"] for n in neighbors])
        avg_v = np.mean([n["valence"] for n in neighbors])
        avg_sim = np.mean([n["similarity"] for n in neighbors])

        arousal_consistent = abs(avg_a - q_arousal) < 0.10
        valence_consistent = abs(avg_v - q_valence) < 0.10

        consistency = []
        if arousal_consistent: consistency.append("arousal")
        if valence_consistent: consistency.append("valence")

        if consistency:
            dim_text = " and ".join(consistency)
            return (f"The retrieved songs show strong consistency in {dim_text} "
                    f"(avg neighbor arousal={avg_a:.3f}, valence={avg_v:.3f}). "
                    f"Mean cosine similarity={avg_sim:.4f} indicates a well-structured "
                    f"emotion-aware latent space where the SupCR loss has successfully "
                    f"clustered perceptually similar tracks.")
        else:
            return (f"The retrieved songs are proximal in latent space "
                    f"(mean cosine sim={avg_sim:.4f}) but show some spread across "
                    f"the valence-arousal plane (avg A={avg_a:.3f}, V={avg_v:.3f}). "
                    f"This may reflect the known difficulty of valence prediction "
                    f"from audio features alone, a recognized ceiling in MER research.")

    def generate_rag_context(self, query_id, query_arousal, query_valence,
                             query_eda, neighbors):
        """
        Formats a structured context string for LLM consumption.
        Use this if you want to pass the context to an LLM for richer explanations.
        """
        ctx = {
            "query": {
                "music_id": query_id,
                "arousal":  round(query_arousal, 4),
                "valence":  round(query_valence, 4),
                "eda":      self._eda_summary(query_eda),
                "quadrant": self._quadrant(query_arousal, query_valence)[2],
            },
            "neighbors": [
                {
                    "rank":       n["rank"],
                    "music_id":   n["music_id"],
                    "similarity": round(n["similarity"], 4),
                    "arousal":    round(n["arousal"], 4),
                    "valence":    round(n["valence"], 4),
                    "eda":        self._eda_summary(n["eda_feats"]),
                    "quadrant":   self._quadrant(n["arousal"], n["valence"])[2],
                }
                for n in neighbors
            ],
        }

        prompt = (
            "You are a musicologist assistant in a Music Emotion Recognition system. "
            "Given the following retrieval context, provide a concise music-theoretic "
            "explanation for why these songs were retrieved together. "
            "Focus on emotional similarity, energy level, and mood.\n\n"
            f"QUERY SONG (ID {query_id}):\n"
            f"  Arousal: {ctx['query']['arousal']} | Valence: {ctx['query']['valence']}\n"
            f"  Emotional Region: {ctx['query']['quadrant']}\n"
            f"  Physiological Response: {ctx['query']['eda']}\n\n"
            "RETRIEVED SONGS:\n"
        )
        for n in ctx["neighbors"]:
            prompt += (
                f"  Rank {n['rank']} — ID {n['music_id']}: "
                f"A={n['arousal']}, V={n['valence']}, sim={n['similarity']}, "
                f"quadrant={n['quadrant']}, EDA={n['eda']}\n"
            )
        prompt += "\nExplanation:"
        return prompt, ctx


# ==============================================================================
# 4. Evaluation — Precision@k and Silhouette score
# ==============================================================================

def evaluate_retrieval(index, k_values=(5, 10), emotion_threshold=0.20):
    """
    Evaluates retrieval quality:
      - Precision@k: fraction of top-k neighbors that are emotionally close
      - Silhouette score: how well quadrants cluster in latent space

    Emotionally close = V-A Euclidean distance < emotion_threshold.

    Args:
        index             : loaded index dict
        k_values          : tuple of k values to evaluate
        emotion_threshold : max V-A distance to count as a true positive

    Prints and returns a results dict.
    """
    from sklearn.metrics import silhouette_score

    latents = index["latents"]
    arousal = index["arousal"]
    valence = index["valence"]
    N       = len(latents)

    # Ground-truth V-A distance matrix
    va  = np.stack([arousal, valence], axis=1)   # (N, 2)
    va_dist = np.linalg.norm(va[:, None, :] - va[None, :, :], axis=2)  # (N, N)

    # Cosine similarity matrix (dot product on L2-normalized vectors)
    sim_matrix = latents @ latents.T             # (N, N)

    print("\n📐  Retrieval Evaluation")
    print("=" * 50)
    results = {}

    for k in k_values:
        precisions = []
        for i in range(N):
            row  = sim_matrix[i].copy()
            row[i] = -1                           # exclude self
            top_k_idx = np.argsort(row)[::-1][:k]
            true_pos  = (va_dist[i, top_k_idx] < emotion_threshold).mean()
            precisions.append(true_pos)
        avg_p = float(np.mean(precisions))
        results[f"Precision@{k}"] = avg_p
        print(f"  Precision@{k:2d} : {avg_p:.4f}")

    # Quadrant labels for silhouette
    quads = []
    for a, v in zip(arousal, valence):
        mid = 0.5
        if   v >= mid and a >= mid: quads.append(0)
        elif v >= mid and a <  mid: quads.append(1)
        elif v <  mid and a >= mid: quads.append(2)
        else:                       quads.append(3)
    quads = np.array(quads)

    if len(np.unique(quads)) > 1:
        sil = silhouette_score(latents, quads, metric="cosine")
        results["Silhouette"] = float(sil)
        print(f"  Silhouette    : {sil:.4f}")
    print("=" * 50)

    return results


# ==============================================================================
# 5. EDA loader (standalone — no dependency on updated data_utils)
# ==============================================================================

def load_eda_for_ids(music_ids, eda_dir, sr=200):
    """
    Loads EDA features for a specific ordered list of music IDs.
    Returns (N, 7) float32 numpy array.
    """
    from scipy.signal import find_peaks

    def extract(eda):
        mean_eda = np.mean(eda)
        std_eda  = np.std(eda)
        t        = np.arange(len(eda)) / sr
        slope    = np.polyfit(t, eda, 1)[0] if len(eda) > 1 else 0.0
        peaks, props = find_peaks(eda, prominence=0.01, distance=sr)
        n_peaks  = len(peaks)
        mean_amp = props["prominences"].mean() if n_peaks > 0 else 0.0
        return np.array([mean_eda, std_eda, slope, n_peaks, mean_amp,
                         np.max(eda), np.mean(eda > mean_eda)], dtype=np.float32)

    features, missing = [], 0
    for mid in music_ids:
        path = os.path.join(eda_dir, f"{mid}_EDA.csv")
        if not os.path.exists(path):
            features.append(np.zeros(7, dtype=np.float32))
            missing += 1
            continue
        df  = pd.read_csv(path, header=None)
        eda = df.select_dtypes(include=[np.number]).iloc[:, 1:].mean(axis=1).values.astype(np.float32)
        features.append(extract(eda))

    arr = np.stack(features)
    mn, mx = arr.min(0, keepdims=True), arr.max(0, keepdims=True)
    arr = (arr - mn) / np.where(mx - mn < 1e-6, 1.0, mx - mn)
    if missing:
        print(f"  ⚠️  EDA missing for {missing}/{len(music_ids)} songs")
    return arr.astype(np.float32)


# ==============================================================================
# 6. Main entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase C — Explainable Music Retrieval")
    parser.add_argument("--mode",       choices=["build", "query", "evaluate"], required=True)
    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--feat_path",  type=str, default="pmemo_mert_all_layers.pt")
    parser.add_argument("--csv_path",   type=str,
                        default="/datasets/emotions/PMEmo2019/annotations/static_annotations.csv")
    parser.add_argument("--eda_dir",    type=str, default="",
                        help="PMEmo EDA directory (optional)")
    parser.add_argument("--index_path", type=str, default="prototypes.npy")
    parser.add_argument("--query_id",   type=int, default=None,
                        help="musicId to use as query (mode=query)")
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--model_mode", type=str, default="hybrid",
                        choices=["hybrid", "baseline"],
                        help="Must match the architecture of best_model.pt")
    parser.add_argument("--phaseb_dir", type=str, default="",
                        help="Override path to phaseB folder (contains models.py, data_utils.py)")
    parser.add_argument("--llm",        type=str, default="ollama",
                        choices=["ollama", "anthropic", "none"],
                        help="LLM backend for the musicological explanation")
    parser.add_argument("--llm_model",  type=str, default="",
                        help="Model name override (e.g. llama3.2 for ollama, claude-sonnet-4-6 for anthropic)")
    args = parser.parse_args()

    # Allow CLI override of phaseB path
    if args.phaseb_dir and os.path.isdir(args.phaseb_dir):
        sys.path.insert(0, os.path.abspath(args.phaseb_dir))
        print(f"  📁 phaseB path overridden → {args.phaseb_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻  Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    def load_model():
        model = MERModel(mode=args.model_mode).to(device)
        if not os.path.exists(args.model_path):
            print(f"❌ Model not found: {args.model_path}")
            sys.exit(1)
        state = torch.load(args.model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # MERModelWithEDA checkpoint: strip "base_model." prefix, drop eda/fusion_head keys
        if any(k.startswith("base_model.") for k in state):
            state = {k[len("base_model."):]: v for k, v in state.items() if k.startswith("base_model.")}
            print("  ℹ️  EDA checkpoint detected — loading base audio model weights only")
        model.load_state_dict(state)
        model.eval()
        print(f"  ✅ Model loaded from {args.model_path} (mode={args.model_mode})")
        return model

    # ── Load data ──────────────────────────────────────────────────────────────
    def load_data():
        X, Y, _ = load_pmemo_data(args.feat_path, args.csv_path)
        # Recover aligned music IDs in same order as X, Y
        data_dict  = torch.load(args.feat_path, map_location="cpu", weights_only=False)
        labels_df  = pd.read_csv(args.csv_path)
        ids = []
        for m_id in data_dict.keys():
            row = labels_df[labels_df["musicId"] == int(m_id)]
            if not row.empty:
                ids.append(int(m_id))
        return X, Y, ids

    # ==========================================================================
    # MODE: build
    # ==========================================================================
    if args.mode == "build":
        print("\n🔨  MODE: Build Index")
        model      = load_model()
        X, Y, ids  = load_data()

        eda_feats = None
        if args.eda_dir and os.path.isdir(args.eda_dir):
            print(f"  Loading EDA features from {args.eda_dir}...")
            eda_feats = load_eda_for_ids([str(i) for i in ids], args.eda_dir)

        builder = VectorIndexBuilder(model, device)
        index   = builder.build(X, Y, ids, eda_features=eda_feats)
        VectorIndexBuilder.save(index, args.index_path)
        print(f"\n✅  Index ready. Run with --mode query to search.")

    # ==========================================================================
    # MODE: query
    # ==========================================================================
    elif args.mode == "query":
        if args.query_id is None:
            print("❌  --query_id is required for query mode")
            sys.exit(1)

        print(f"\n🔍  MODE: Query | song_id={args.query_id} | top_k={args.top_k}")
        index     = VectorIndexBuilder.load(args.index_path)
        retriever = EmotionRetriever(index)
        explainer = ExplainableRAG()

        # Find query song in index
        ids_array = index["music_ids"]
        matches   = np.where(ids_array == args.query_id)[0]
        if len(matches) == 0:
            print(f"❌  Song ID {args.query_id} not found in index.")
            print(f"    Available IDs (first 10): {ids_array[:10].tolist()}")
            sys.exit(1)

        q_idx       = matches[0]
        q_latent    = index["latents"][q_idx]
        q_arousal   = float(index["arousal"][q_idx])
        q_valence   = float(index["valence"][q_idx])
        q_eda       = index["eda_feats"][q_idx]

        # Retrieve neighbors
        neighbors = retriever.query(q_latent, k=args.top_k, exclude_self=True)

        # Generate and print template explanation
        explanation = explainer.generate_template(
            query_id      = args.query_id,
            query_arousal = q_arousal,
            query_valence = q_valence,
            query_eda     = q_eda,
            neighbors     = neighbors,
        )
        print("\n" + explanation)

        # Also print RAG prompt (for LLM integration)
        rag_prompt, ctx = explainer.generate_rag_context(
            query_id      = args.query_id,
            query_arousal = q_arousal,
            query_valence = q_valence,
            query_eda     = q_eda,
            neighbors     = neighbors,
        )

        # Generate LLM explanation
        llm_explanation = ""
        if args.llm == "ollama":
            import requests as _req
            model_name = args.llm_model or "llama3.2"
            try:
                resp = _req.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": rag_prompt, "stream": False},
                    timeout=120,
                )
                resp.raise_for_status()
                llm_explanation = resp.json().get("response", "")
                print("\n🤖  LLM EXPLANATION (Ollama / " + model_name + "):")
                print("-" * 65)
                print(llm_explanation)
            except _req.exceptions.ConnectionError:
                print("\n⚠️  Ollama not running — start it with: ollama serve")
            except Exception as e:
                print(f"\n⚠️  Ollama error: {e}")

        elif args.llm == "anthropic":
            model_name = args.llm_model or "claude-sonnet-4-6"
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    print("\n⚠️  ANTHROPIC_API_KEY not set — skipping LLM explanation")
                else:
                    client = anthropic.Anthropic(api_key=api_key)
                    msg = client.messages.create(
                        model=model_name, max_tokens=512,
                        messages=[{"role": "user", "content": rag_prompt}],
                    )
                    llm_explanation = msg.content[0].text
                    print("\n🤖  LLM EXPLANATION (Anthropic / " + model_name + "):")
                    print("-" * 65)
                    print(llm_explanation)
            except ImportError:
                print("\n⚠️  anthropic library not installed — run: pip install anthropic")

        # Save results
        out_path = f"retrieval_query_{args.query_id}.txt"
        with open(out_path, "w") as f:
            f.write(explanation + "\n\n")
            f.write("RAG PROMPT:\n" + rag_prompt + "\n")
            if llm_explanation:
                f.write("\nLLM EXPLANATION:\n" + llm_explanation + "\n")
        print(f"\n💾  Results saved → {out_path}")

    # ==========================================================================
    # MODE: evaluate
    # ==========================================================================
    elif args.mode == "evaluate":
        print("\n📐  MODE: Evaluate Retrieval Quality")
        index = VectorIndexBuilder.load(args.index_path)
        results = evaluate_retrieval(index, k_values=(5, 10, 20))

        # Compare against a baseline (raw Layer 24 embeddings, no contrastive training)
        print("\n📊  To compare against baseline:")
        print("    1. Save a baseline model checkpoint (--model_mode baseline)")
        print("    2. Build a second index: --index_path prototypes_baseline.npy")
        print("    3. Run evaluate on both and compare Precision@k and Silhouette")


if __name__ == "__main__":
    main()
