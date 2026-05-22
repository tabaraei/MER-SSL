# 03 — Phase C: Explainable Retrieval

**Goal:** turn the Phase B emotion-aware latent space into a retrieval system that
not only returns emotionally similar songs but **explains why**, in scientifically
grounded and human-legible terms.

**Code:** `phaseC/index_builder.py`, `retriever.py`, `explainer.py`,
`evaluator.py`, `eda_loader.py`, `music_theory_annotator.py`,
`evaluate_latent_space.py`, `mainC.py`.

---

## 1. Why Explainability (the thesis contribution)

k-NN retrieval in a learned space is not novel; the contribution is the
**explainability layer**. Standard MER evaluation (R², CCC) shows a model predicts
labels but not that its representation is *meaningful*. Phase C is a complementary,
human-interpretable validation: if the latent space truly captures emotion, its
neighbors should be explainable by shared musical/physiological properties.

## 2. Prototype-Based Retrieval (ante-hoc by design)

The system answers a query by finding the real songs nearest to it in the latent
space — **the retrieved examples *are* the explanation** (case-based reasoning;
Aamodt & Plaza, 1994). It does not produce a black-box label and then rationalize
it; the decision is similarity to known examples. Backend: cosine k-NN
(`IndexFlatIP` / sklearn fallback) over L2-normalized latents.

## 3. The Explanation Channels

| Channel | What it adds | Grounding |
| :-- | :-- | :-- |
| **4-centroid prototype profile** | Cosine similarity of the query to each Russell-quadrant centroid + the best-match prototype | Russell (1980); ProtoPNet-style activation (Chen et al., 2019) |
| **Contrastive foils** | The most *dissimilar* songs — "why these and not those" | Miller (2019): contrastive explanation is most natural for humans |
| **EDA physiological narrative** | Body-response interpretation of the retrieved set | Thayer (1989): EDA indexes autonomic arousal |
| **MERT layer attribution** | Which acoustic feature levels drove retrieval | Mechanistic interpretability of the encoder |
| **Mood trajectory** | Energizing vs wind-down listening arc | Saarikallio & Erkkilä (2007): music for mood regulation |
| **Music-theory annotation** | Key, tempo, timbre, dominant pitches of the query | Krumhansl & Kessler (1982) key finding |

### 3a. 4-Centroid Ante-hoc Profile (the supervisor's requirement)
`EmotionRetriever` computes the mean latent of each Russell quadrant (the 4 emotion
prototypes, L2-normalized) once at init, then `prototype_profile(query)` returns the
query's cosine similarity to each and the best match. The Layer-1 explanation now
prints this 4-value activation profile and an explicit classification line. This is
the literal "if a prototype is more active, the song is classified to it" that the
supervisor asked for.

*Example (query 760, ground-truth HVHA):* HVHA 0.705 ◄ best · LVHA 0.700 · HVLA
0.633 · LVLA 0.629 → correctly classified HVHA.

**Quantitative validation — and it is honestly weak (`extra_metrics.py`).**
Prototype-activation accuracy = % of songs whose best-match centroid equals their
true quadrant (leave-one-out centroids):

| Index | Activation accuracy | Majority-class baseline (always HVHA) |
| :-- | :-: | :-: |
| Dual-SSL | 0.506 | **0.611** |
| MERT | 0.462 | 0.611 |

**The accuracy (50.6%) is *below* the trivial majority-class baseline (61.1%)** — a
classifier that always guesses "Happy" beats prototype-matching. Per-quadrant
recall: HVHA 0.63, HVLA 0.55, LVHA 0.41, **Sad 0.17**. **Honest conclusion:** the
4-centroid profile satisfies the supervisor's *form* (a per-prototype activation +
argmax, ante-hoc by construction) and is a useful **interpretability readout**, but
it is **not a competitive classifier** — it underperforms the majority baseline,
driven by the same class-imbalance and continuous-emotion limitations (near-zero
Silhouette, §5). It must be presented as an interpretability device, not an
accurate quadrant classifier.

### 3b. Two-Layer Explanation
- **Layer 1 — deterministic template:** V-A coordinates, similarities, per-neighbor
  deltas, EDA summary, the 4-centroid profile, mood trajectory. Reproducible, no
  hallucination — citable directly in the thesis.
- **Layer 2 — LLM-augmented (RAG):** the template feeds a structured prompt (plus a
  static, citable emotion knowledge base) to an LLM (Ollama local / Anthropic) which
  *translates* the deterministic facts into warm prose. The LLM synthesizes; it does
  not decide.

## 4. Ante-hoc vs Post-hoc (direct response to supervisor feedback)

The supervisor stressed that using a foundation model for embeddings loses
explainability, and that ante-hoc (intrinsic) is preferable to post-hoc.

| Component | Type | Explanation provided |
| :-- | :-- | :-- |
| WeightedLayerFusion weights | **Ante-hoc** | Which MERT layers drive prediction |
| SupCR latent organization | **Ante-hoc** | Emotion clusters formed *during* training |
| k-NN prototype retrieval | **Ante-hoc** | Decision = similarity to known examples |
| 4-centroid activation profile | **Ante-hoc** | "X% similar to each emotion prototype" |
| Contrastive foils | **Ante-hoc** | Counterfactual: what was rejected, why |
| EDA narrative | Post-hoc annotation | Physiological interpretation |
| Music-theory annotation (librosa) | Independent descriptive | Describes the song, not the model |
| LLM synthesis | Post-hoc synthesis | Human-language presentation only |

**The system is ante-hoc at its decision-making core**; post-hoc parts are the
presentation layer. **Remaining gap:** the 4 centroids are computed from data, not
*learned* parameters — a true ProtoPNet-style learned-prototype head is the honest
next step (see `05_limitations_future_work.md`).

**Data-provenance honesty (important).** The music-theory annotation is computed by
**librosa directly from the query audio — not from the SSL embeddings.** It
truthfully describes the song but is *not* a faithful account of the model's
internal computation (the model never saw those librosa features). It is independent
corroboration, distinct from the model-internal WeightedLayerFusion attribution.
`librosa.estimate_key()` does not exist → key/mode use Krumhansl–Schmuckler.

## 5. Quantitative Evaluation (test-fold, out-of-sample)

`evaluate_latent_space.py` builds **out-of-sample** latents: 5-fold CV
(`random_state=42`), each song encoded by the fold model that did not train on it —
stricter and more honest than encoding all songs with one model. (It also sidesteps
a real bug: `best_model.pt` is now DualSSL and will not load into `mainC`'s
single-encoder `MERModel`; and it fixed the EDA lookup, which failed because IDs
arrived float-formatted as `'1.0'` → `1.0_EDA.csv` missing.)

| Encoder | Precision@5 | Precision@10 | Precision@20 | Silhouette |
| :-- | :-: | :-: | :-: | :-: |
| MERT (single) | 0.5760 | 0.5687 | 0.5469 | −0.0293 |
| **Dual-SSL (MERT + wav2vec2)** | **0.5849** | 0.5613 | 0.5382 | **+0.0026** |

**Precision@k validates retrieval.** ~54–58% of each song's nearest latent
neighbors fall within a 0.20 V-A radius. The **random-chance baseline is 0.276**
(expected fraction of a random neighbour within that radius, `extra_metrics.py`),
so Precision@5 ≈ 0.58 is roughly **2× chance** — retrieval genuinely returns
emotionally similar songs. This is the load-bearing evidence and the foundation of
the example-based explanation.

**Silhouette ≈ 0 — reported honestly.** Both ≈ 0 (dual's +0.0026 is *not*
meaningful separation). Emotion is a **continuous** V-A gradient, not four discrete
clusters; Silhouette-by-quadrant imposes hard 0.5 cutoffs on a smooth manifold, so
boundary songs are legitimately close to the adjacent quadrant → near-zero even
when local structure is good (as Precision@k shows). Class imbalance compounds it.
**Lead with Precision@k; present Silhouette with this continuous-manifold caveat.**
The two encoders are statistically tied — consistent with the Phase B finding that
the second encoder adds little to latent *organization*.

**Definitions.** Precision@k = mean fraction of top-k neighbors within a 0.20 V-A
Euclidean radius (label consistency, not perceptual ground truth). Silhouette =
mean (b−a)/max(a,b) over songs partitioned by the 4 quadrants in cosine space.

### 5a. Naive baseline comparison (does the training help?)

To isolate the value of training, the test set was also retrieved in the **raw,
untrained** MERT space: average-pool the last MERT layer, L2-normalize, no
fine-tuning, no SupCR (`export_artifacts.py`).

| Space | P@5 | P@10 | P@20 | Silhouette |
| :-- | :-: | :-: | :-: | :-: |
| Naive raw last-layer MERT | 0.4847 | 0.4614 | 0.4553 | **0.1001** |
| MERT (SupCR) | 0.5760 | 0.5687 | 0.5469 | −0.0293 |
| Dual-SSL (SupCR) | 0.5849 | 0.5613 | 0.5382 | +0.0026 |

**SupCR fine-tuning lifts Precision@5 from 0.485 → 0.58 (+≈0.10, ~+19% relative)** —
clear, measurable evidence that training tightened the emotion-relevant retrieval
space (the example-based explanation is built on a genuinely better space, not raw
embeddings). **Honest nuance:** the *untrained* space has the higher Silhouette
(0.100 vs ≈0). Raw MERT retains coarse quadrant blobbiness (probably
genre/acoustic structure) but worse fine-grained V-A retrieval; SupCR reorganizes
around *continuous* V-A proximity, raising Precision@k while flattening the discrete
clusters. The metric that matters for retrieval (Precision@k) improves; the
discrete-cluster metric (Silhouette) is the wrong lens, as argued above.

### 5b. Exported thesis artifacts (`phaseC/artifacts/`)

- `tsne_baseline_vs_finetuned.png` — t-SNE of raw MERT (an undifferentiated blob,
  quadrant colours fully mixed) vs the SupCR latent space (clear structured
  filaments). Visual confirmation that training imposes organization. The
  fine-tuned clusters are a continuous manifold dominated by HVHA, not 4 clean
  quadrant blobs — consistent with §5.
- `layer_fusion_weights.png` — softmaxed WeightedLayerFusion weights (single-MERT).
  Top-3 layers = [14, 15, 16] (mid-to-late, supporting the qualitative claim) but
  the distribution is **near-uniform** (entropy 3.218 / max 3.219) — drawn honestly,
  consistent with the fusion-collapse finding in `02_phaseB_model.md`.
- `explanations_5songs.txt` — full Layer 1 + Layer 2 output for 5 quadrant-spanning
  songs (562, 706, 31, 894, 282), for the results chapter (`export_explanations.py`).
  Layer 2 was generated **server-free, without sudo**, via a HuggingFace
  `transformers` backend (default `Qwen/Qwen2.5-1.5B-Instruct`, runs in the existing
  venv on the GPU — Ollama/sudo not required; `--llm ollama|anthropic|hf` selectable).
  The model produces the intended 4-part structured explanation (Recommendation /
  Emotional Connection / Why-Not-The-Others / Listening Experience). **Caveat:** a
  1.5B local model is the quality floor; prose is coherent but model-dependent —
  Layer 1 remains the citable, deterministic artifact.

## 6. How to Run

```bash
cd MERT/ssl_scripts/phaseC
# evaluation (rebuilds index + Precision@k + Silhouette), dual = best model:
python evaluate_latent_space.py --encoder dual --beta 0.05 --epochs 100 \
    --w2v_path ../phaseB/pmemo_wav2vec_all_layers.pt --index_path prototypes_dual.npy
# a single explained query (uses the index only; no model load needed):
python mainC.py --mode query --query_id 760 --index_path prototypes_dual.npy --llm none
```
