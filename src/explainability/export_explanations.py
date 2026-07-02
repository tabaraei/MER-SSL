"""
export_explanations.py — 5-song full explanation export for the results chapter
================================================================================
Runs the complete ExplainableRAG pipeline (Layer 1 deterministic template +
Layer 2 LLM-synthesized prose) for 5 distinct songs and writes everything to a
single text file: artifacts/explanations_5songs.txt.

By default the 5 songs are chosen to span the emotional space — the most
prototypical song of each Russell quadrant (nearest that quadrant's mean V-A)
plus the most boundary/ambiguous song (nearest the V-A centre) — so the results
chapter shows the system across all emotion regions. Override with --query_ids.

Read-only: loads the existing index, trains/changes nothing.

Run from phaseC/ (Layer 2 needs a local Ollama server or ANTHROPIC_API_KEY):
  python export_explanations.py --index_path prototypes_dual.npy --llm ollama --llm_model llama3.2
  # Layer-1-only (no LLM) still produces the file:
  python export_explanations.py --index_path prototypes_dual.npy --llm none
"""

import argparse
import os

import numpy as np

from explainability.index_builder import VectorIndexBuilder
from explainability.retriever import EmotionRetriever
from explainability.explainer import ExplainableRAG

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _quad(a, v):
    if v >= 0.5 and a >= 0.5: return 0
    if v >= 0.5 and a < 0.5:  return 1
    if v < 0.5 and a >= 0.5:  return 2
    return 3


def pick_query_ids(index):
    """One most-prototypical song per quadrant (nearest quadrant mean V-A) +
    the most ambiguous song (nearest the (0.5,0.5) centre)."""
    a, v = index["arousal"], index["valence"]
    ids = index["music_ids"]
    va = np.stack([a, v], axis=1)
    quads = np.array([_quad(ai, vi) for ai, vi in zip(a, v)])
    chosen = []
    for q in range(4):
        mask = quads == q
        if not mask.any():
            continue
        centre = va[mask].mean(axis=0)
        sub = np.where(mask)[0]
        chosen.append(int(ids[sub[np.argmin(np.linalg.norm(va[sub] - centre, axis=1))]]))
    # 5th: nearest the global centre (most ambiguous)
    chosen.append(int(ids[np.argmin(np.linalg.norm(va - 0.5, axis=1))]))
    # de-dup while preserving order
    seen, out = set(), []
    for c in chosen:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def _make_hf(llm_model):
    """Server-free, no-sudo LLM backend via HuggingFace transformers (reuses the
    existing venv). Model is loaded ONCE and reused for all songs. Default is an
    ungated Qwen2.5 instruct model that loads on transformers 4.38 (Llama-3.2 is
    gated and needs transformers >= 4.45, so it is not the default here)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = llm_model or "Qwen/Qwen2.5-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ⏳ Loading HF model {model_id} on {device} (one-time)...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model.eval()
    print(f"  ✅ {model_id} ready")

    def generate(prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return generate


def make_llm(llm, llm_model):
    """Returns a callable generate(prompt)->str, built ONCE. Empty string on any
    failure so Layer 1 output is always preserved."""
    if llm == "none":
        return lambda _p: ""

    if llm == "hf":
        try:
            return _make_hf(llm_model)
        except Exception as e:
            print(f"  ⚠️  HF backend failed to load ({e}) — Layer 2 left blank")
            return lambda _p: ""

    if llm == "ollama":
        import requests
        def gen(prompt):
            try:
                r = requests.post("http://localhost:11434/api/generate",
                                  json={"model": llm_model or "llama3.2",
                                        "prompt": prompt, "stream": False}, timeout=180)
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except Exception as e:
                print(f"  ⚠️  Ollama unavailable ({e}) — Layer 2 left blank")
                return ""
        return gen

    if llm == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("  ⚠️  ANTHROPIC_API_KEY not set — Layer 2 left blank")
            return lambda _p: ""
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        def gen(prompt):
            try:
                m = client.messages.create(model=llm_model or "claude-sonnet-4-6",
                                           max_tokens=600,
                                           messages=[{"role": "user", "content": prompt}])
                return m.content[0].text.strip()
            except Exception as e:
                print(f"  ⚠️  Anthropic error ({e}) — Layer 2 left blank")
                return ""
        return gen

    return lambda _p: ""


def main():
    ap = argparse.ArgumentParser(description="Export 5-song explanations")
    ap.add_argument("--index_path", default="prototypes_dual.npy")
    ap.add_argument("--query_ids", default="", help="comma-separated IDs (overrides auto-pick)")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--n_foils", type=int, default=3)
    ap.add_argument("--llm", default="hf", choices=["hf", "ollama", "anthropic", "none"],
                    help="hf = server-free HuggingFace transformers (no sudo, default)")
    ap.add_argument("--llm_model", default="",
                    help="hf default: Qwen/Qwen2.5-1.5B-Instruct; ollama default: llama3.2")
    ap.add_argument("--out", default=os.path.join(ART, "explanations_5songs.txt"))
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    index = VectorIndexBuilder.load(args.index_path)
    retriever = EmotionRetriever(index)
    explainer = ExplainableRAG()
    ids_arr = index["music_ids"]

    if args.query_ids.strip():
        query_ids = [int(x) for x in args.query_ids.split(",")]
    else:
        query_ids = pick_query_ids(index)
    print(f"  Query songs: {query_ids}")

    llm_fn = make_llm(args.llm, args.llm_model)   # built once (HF model loads once)
    blocks = []
    for qid in query_ids:
        matches = np.where(ids_arr == qid)[0]
        if len(matches) == 0:
            print(f"  ⚠️  {qid} not in index — skipped")
            continue
        qi = matches[0]
        q_lat = index["latents"][qi]
        q_a, q_v = float(index["arousal"][qi]), float(index["valence"][qi])
        q_eda = index["eda_feats"][qi]

        neighbors = retriever.query(q_lat, k=args.top_k, exclude_self=True)
        foils = retriever.query_foils(q_lat, n_foils=args.n_foils)
        profile = retriever.prototype_profile(q_lat)
        lw = index.get("layer_weights", None)

        template = explainer.generate_template(
            query_id=qid, query_arousal=q_a, query_valence=q_v,
            query_eda=q_eda, neighbors=neighbors, prototype_profile=profile)
        prompt, _ = explainer.generate_rag_context(
            query_id=qid, query_arousal=q_a, query_valence=q_v, query_eda=q_eda,
            neighbors=neighbors, foils=foils, layer_weights=lw)
        print(f"  → song {qid}: Layer 1 ✓  calling LLM ({args.llm})...")
        llm_text = llm_fn(prompt)

        block = ["#" * 70, f"# SONG {qid}", "#" * 70, "",
                 "----- LAYER 1: DETERMINISTIC TEMPLATE -----", template, ""]
        if llm_text:
            block += ["----- LAYER 2: LLM EXPLANATION -----", llm_text, ""]
        blocks.append("\n".join(block))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"Explainable MER — full pipeline output for {len(blocks)} songs\n")
        f.write(f"Index: {args.index_path} | LLM: {args.llm} ({args.llm_model})\n\n")
        f.write("\n".join(blocks))
    print(f"\n✅ Saved → {args.out}")


if __name__ == "__main__":
    main()
