"""
mainC.py — Phase C Entry Point: Explainable Music Emotion Retrieval
====================================================================
Master's Thesis: Explainable & Perceptual Music Emotion Recognition
Student: Arvin Jafari Moghadam Fard

Modes:
  build    — Encode all PMEmo songs into a latent-space index (run once)
  query    — Retrieve k similar songs + generate two-layer explanation
  evaluate — Compute Precision@k and Silhouette over the full index

Module layout:
  mainC.py         ← this file (CLI + orchestration)
  index_builder.py ← VectorIndexBuilder
  retriever.py     ← EmotionRetriever (k-NN + contrastive foils)
  explainer.py     ← ExplainableRAG (template + enriched RAG prompt)
  eda_loader.py    ← load_eda_for_ids
  evaluator.py     ← evaluate_retrieval

Usage:
    # Step 1 — Build index (once):
    python mainC.py --mode build \
        --model_path ../phaseB/best_model.pt \
        --feat_path  ../phaseB/pmemo_mert_all_layers.pt \
        --csv_path   /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
        --eda_dir    /datasets/emotions/PMEmo2019/EDA \
        --index_path prototypes.npy

    # Step 2 — Query:
    python mainC.py --mode query \
        --query_id   760 \
        --index_path prototypes.npy \
        --feat_path  ../phaseB/pmemo_mert_all_layers.pt \
        --csv_path   /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
        --model_path ../phaseB/best_model.pt \
        --eda_dir    /datasets/emotions/PMEmo2019/EDA \
        --top_k      5 --n_foils 3 --llm ollama --llm_model llama3.2

    # Step 3 — Evaluate:
    python mainC.py --mode evaluate --index_path prototypes.npy
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

# ── Resolve phaseB imports ─────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PHASEB_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'phaseB'))
if os.path.isdir(_PHASEB_DIR):
    sys.path.insert(0, _PHASEB_DIR)

from models import MERModel
from data_utils import load_pmemo_data

from index_builder import VectorIndexBuilder
from retriever     import EmotionRetriever
from explainer     import ExplainableRAG
from eda_loader    import load_eda_for_ids
from evaluator     import evaluate_retrieval


# ==============================================================================
# Helpers
# ==============================================================================

def get_layer_weights(model):
    """Extract learned MERT layer fusion weights from the model if available."""
    try:
        return model.fusion.layer_weights.detach().cpu().numpy()
    except AttributeError:
        return None


def load_model(args, device):
    model = MERModel(mode=args.model_mode).to(device)
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    state = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if any(k.startswith("base_model.") for k in state):
        state = {k[len("base_model."):]: v
                 for k, v in state.items() if k.startswith("base_model.")}
        print("  ℹ️  EDA checkpoint — loading base audio model weights only")
    model.load_state_dict(state)
    model.eval()
    print(f"  ✅ Model loaded from {args.model_path} (mode={args.model_mode})")
    return model


def load_data(args):
    X, Y, _ = load_pmemo_data(args.feat_path, args.csv_path)
    data_dict = torch.load(args.feat_path, map_location="cpu", weights_only=False)
    labels_df = pd.read_csv(args.csv_path)
    ids = [int(m_id) for m_id in data_dict.keys()
           if not labels_df[labels_df["musicId"] == int(m_id)].empty]
    return X, Y, ids


def call_llm(args, prompt):
    """Dispatches to Ollama or Anthropic based on --llm flag."""
    if args.llm == "none":
        return ""

    if args.llm == "ollama":
        import requests as _req
        model_name = args.llm_model or "llama3.2"
        try:
            resp = _req.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=180,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            print(f"\n🤖  LLM EXPLANATION (Ollama / {model_name}):")
            print("-" * 65)
            print(text)
            return text
        except _req.exceptions.ConnectionError:
            print("\n⚠️  Ollama not running — start with: ollama serve")
        except Exception as e:
            print(f"\n⚠️  Ollama error: {e}")
        return ""

    if args.llm == "anthropic":
        model_name = args.llm_model or "claude-sonnet-4-6"
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                print("\n⚠️  ANTHROPIC_API_KEY not set — skipping LLM explanation")
                return ""
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=model_name, max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            print(f"\n🤖  LLM EXPLANATION (Anthropic / {model_name}):")
            print("-" * 65)
            print(text)
            return text
        except ImportError:
            print("\n⚠️  anthropic not installed — pip install anthropic")
        return ""

    return ""


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase C — Explainable Music Emotion Retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",       required=True, choices=["build", "query", "evaluate"])
    parser.add_argument("--model_path", default="../phaseB/best_model.pt")
    parser.add_argument("--feat_path",  default="../phaseB/pmemo_mert_all_layers.pt")
    parser.add_argument("--csv_path",
                        default="/datasets/emotions/PMEmo2019/annotations/static_annotations.csv")
    parser.add_argument("--eda_dir",    default="",
                        help="Path to PMEmo EDA folder (files named {id}_EDA.csv)")
    parser.add_argument("--index_path", default="prototypes.npy")
    parser.add_argument("--query_id",   type=int,   default=None,
                        help="PMEmo musicId to query (required for mode=query)")
    parser.add_argument("--top_k",      type=int,   default=5,
                        help="Number of similar songs to retrieve")
    parser.add_argument("--n_foils",    type=int,   default=3,
                        help="Number of contrastive foil songs for XAI explanation")
    parser.add_argument("--model_mode", default="hybrid", choices=["hybrid", "baseline"])
    parser.add_argument("--phaseb_dir", default="",
                        help="Override phaseB path (if models.py is not auto-found)")
    parser.add_argument("--llm",        default="ollama", choices=["ollama", "anthropic", "none"],
                        help="LLM backend for humanized explanation")
    parser.add_argument("--llm_model",  default="",
                        help="Model override: e.g. llama3.2 (ollama) or claude-sonnet-4-6 (anthropic)")
    args = parser.parse_args()

    if args.phaseb_dir and os.path.isdir(args.phaseb_dir):
        sys.path.insert(0, os.path.abspath(args.phaseb_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻  Device: {device}")

    # ==========================================================================
    # MODE: build
    # ==========================================================================
    if args.mode == "build":
        print("\n🔨  MODE: Build Index")
        model         = load_model(args, device)
        X, Y, ids     = load_data(args)
        layer_weights = get_layer_weights(model)

        if layer_weights is not None:
            print(f"  ✅ Layer weights captured ({len(layer_weights)} MERT layers)")

        eda_feats = None
        if args.eda_dir and os.path.isdir(args.eda_dir):
            print(f"  Loading EDA features from {args.eda_dir}...")
            eda_feats = load_eda_for_ids([str(i) for i in ids], args.eda_dir)

        builder = VectorIndexBuilder(model, device)
        index   = builder.build(X, Y, ids,
                                eda_features=eda_feats,
                                layer_weights=layer_weights)
        VectorIndexBuilder.save(index, args.index_path)
        print(f"\n✅  Index ready. Run --mode query to search.")

    # ==========================================================================
    # MODE: query
    # ==========================================================================
    elif args.mode == "query":
        if args.query_id is None:
            print("❌  --query_id required for query mode")
            sys.exit(1)

        print(f"\n🔍  MODE: Query | song_id={args.query_id} | "
              f"top_k={args.top_k} | n_foils={args.n_foils} | llm={args.llm}")

        index     = VectorIndexBuilder.load(args.index_path)
        retriever = EmotionRetriever(index)
        explainer = ExplainableRAG()

        ids_array = index["music_ids"]
        matches   = np.where(ids_array == args.query_id)[0]
        if len(matches) == 0:
            print(f"❌  Song ID {args.query_id} not found in index.")
            print(f"    Available IDs (sample): {ids_array[:10].tolist()}")
            sys.exit(1)

        q_idx     = matches[0]
        q_latent  = index["latents"][q_idx]
        q_arousal = float(index["arousal"][q_idx])
        q_valence = float(index["valence"][q_idx])
        q_eda     = index["eda_feats"][q_idx]

        # Retrieve neighbors and contrastive foils
        neighbors = retriever.query(q_latent, k=args.top_k, exclude_self=True)
        foils     = retriever.query_foils(q_latent, n_foils=args.n_foils)

        # Ante-hoc 4-prototype activation profile (cosine sim to quadrant centroids)
        prototype_profile = retriever.prototype_profile(q_latent)

        # Get layer weights from index (captured at build time)
        layer_weights = index.get("layer_weights", None)

        # Layer 1: deterministic technical explanation
        template = explainer.generate_template(
            query_id=args.query_id, query_arousal=q_arousal,
            query_valence=q_valence, query_eda=q_eda, neighbors=neighbors,
            prototype_profile=prototype_profile,
        )
        print("\n" + template)

        # Layer 2: enriched RAG prompt → LLM
        rag_prompt, ctx = explainer.generate_rag_context(
            query_id=args.query_id, query_arousal=q_arousal,
            query_valence=q_valence, query_eda=q_eda,
            neighbors=neighbors, foils=foils, layer_weights=layer_weights,
        )
        llm_explanation = call_llm(args, rag_prompt)

        # Save full output
        out_path = f"retrieval_query_{args.query_id}.txt"
        with open(out_path, "w") as f:
            f.write(template + "\n\n")
            f.write("=" * 65 + "\n")
            f.write("RAG PROMPT (sent to LLM):\n")
            f.write("=" * 65 + "\n")
            f.write(rag_prompt + "\n")
            if llm_explanation:
                f.write("\n" + "=" * 65 + "\n")
                f.write("LLM EXPLANATION:\n")
                f.write("=" * 65 + "\n")
                f.write(llm_explanation + "\n")
        print(f"\n💾  Full output saved → {out_path}")

    # ==========================================================================
    # MODE: evaluate
    # ==========================================================================
    elif args.mode == "evaluate":
        print("\n📐  MODE: Evaluate Retrieval Quality")
        index = VectorIndexBuilder.load(args.index_path)
        results = evaluate_retrieval(index, k_values=(5, 10, 20))
        print("\n📊  Baseline comparison:")
        print("    1. Build baseline index: --model_mode baseline "
              "--index_path prototypes_baseline.npy")
        print("    2. Compare Precision@k and Silhouette between hybrid vs baseline")


if __name__ == "__main__":
    main()
