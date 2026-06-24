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

### 2a. Unified encoder — index/query model symmetry (corrected pipeline)

A model-symmetry audit found that the original retrieval stack (`mainC.py`,
`retrieval.py`) instantiated only the single-encoder `MERModel` and that *query*
mode never re-encoded the query — it looked up a pre-stored latent by ID. The best
Phase B model (Enhanced) and the Audio ProtoPNet were not wired into retrieval at all.
This was corrected: a single `UnifiedEnhancedEncoder` (`phaseC/encoder_unified.py`)
now encodes **both** the database and the runtime query through one `encode()` call —
the **Enhanced** architecture (MERT + wav2vec2 + cyclic-key theory branch) → MLP
bottleneck → 128-D latent → L2-norm. Because build and query share one code path,
symmetry is guaranteed by construction.

**Symmetry proof** (`phaseC/build_index_unified.py --mode verify`): re-encoding every
corpus song via the query path reproduces the stored index vector to floating-point
precision — **cosine(query, stored) min = 1.000000, max ‖Δ‖ = 1.06×10⁻⁶**, all index
L2-norms = 1.0000, latent dim = 128. A new (unseen) query is now genuinely encodable,
not only an in-corpus lookup. The deployment checkpoint (`best_model_enhanced_final.pt`)
is trained on all 767 songs; its in-sample accuracy is **not** reported — canonical
metrics remain the 5-fold held-out values. The ProtoPNet ante-hoc profile
(`phaseC/protopnet_readout.py`, §3a) replaces the post-hoc 4-centroid readout.
Log: `reports/run_logs/unified_phaseC.log`.

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

The *post-hoc* 4-centroid accuracy (50.6%) is **below** the majority baseline (61.1%) —
fixed centroids computed after training underperform "always guess Happy". This
motivated a learnable-prototype upgrade.

**Upgrade — Audio ProtoPNet (learnable prototypes, ante-hoc by design).** Following
ProtoPNet (Chen et al. 2019), we learn the prototype vectors *during* gradient descent
(5/quadrant = 20 total) and classify by L2 distance, with cluster + separation losses
and a balanced sampler (`phaseB/models_protopnet.py`, `train_protopnet.py`):

| Quadrant classifier | Raw acc | Balanced acc | Sad recall |
| :-- | :-: | :-: | :-: |
| Majority (always-HVHA) | 0.611 | 0.250 | 0.00 |
| Post-hoc 4-centroid (MERT) | 0.462 | — | 0.17 |
| **Audio ProtoPNet (MERT, held-out)** | **0.728 ± 0.031** | **0.545 ± 0.041** | **0.69** |

**ProtoPNet beats both the post-hoc centroid (+0.22) and the majority baseline (+0.12)**,
on the more rigorous held-out protocol, and lifts Sad recall from 0.17 → 0.69. Learning
the prototypes jointly with the encoder (rather than freezing centroids after training)
places them where the quadrants are actually separable. **It stays ante-hoc and
interpretable** — each prototype is, by construction, evidence for one quadrant (identity
prior on the classification head). **Conclusion:** the ante-hoc prototype classifier the
supervisor requested now exists in *form and substance* — it is a genuine, accurate,
self-explaining classifier, not merely a readout. (The k-NN retrieval branch is separate
and still rests on the weakly-organized latent space, §5.)

### 3b. Two-Layer Explanation (refined)

**Layer 1 — deterministic template (cannot hallucinate).** A fixed, structured record of the
query: predicted V-A coordinates + quadrant; top-k retrieved neighbours with coordinates,
per-neighbour V-A deltas, and EDA summaries; the 4-prototype activation profile (§3a); librosa
tempo/key; mood trajectory; and the contrastive foils. Every field is a measured number or a
direct lookup → reproducible, citable directly in the thesis.

**Foil definition (exact).** Query latent `z` and database `{x_i}` are all L2-normalized;
retrieval ranks by cosine `s_i = zᵀx_i`. The foils are the `n_f` songs with the **lowest**
cosine similarity, `F = argmin^(n_f)_i (zᵀx_i)` (`retriever.query_foils`, literally
`np.argsort(sims)[:n_foils]`). Because `‖z−x_i‖² = 2(1−s_i)` on the unit hypersphere, lowest
cosine ⇔ largest Euclidean distance, so foils are the **globally most-distant** songs (in
practice the opposing quadrant). **NB (honesty):** these are *global "easy" negatives* — the
selection does **not** constrain foils to share the query's tempo/key, so they are **not**
metadata-matched "hard negatives". The thesis describes them as *most-dissimilar* contrastive
foils, matching the code.

**Layer 2 — LLM synthesis (Qwen2.5-3B-Instruct via HF `transformers`).** The template feeds a
structured prompt to a local instruction LLM that rewrites it into 4-part prose (Recommendation
/ Emotional Connection / Why-Not-The-Others / Listening Experience). The LLM **synthesizes; it
does not decide.** Standardized engine = **Qwen2.5-3B-Instruct**, served in the project venv via
the HuggingFace `transformers` backend (`--llm hf`) — no Ollama daemon, no sudo, no external API
(Ollama is not installed on the server). `--llm ollama` (llama3.2) and `--llm anthropic` remain
selectable but were **not** used for the reported export.

**Faithfulness metric (Layer-2 grounding precision).** Layer-1 owns *correctness* (already
measured by Precision@k, §5); Layer-2 is judged only on *faithfulness*. Define **grounding
precision** `GP = (# Layer-2 assertions supported by Layer-1) / (# Layer-2 assertions)` over two
assertion types: (i) **song IDs** named in the prose — supported iff `ID ∈ {retrieved ∪ foils ∪
query}`; (ii) **directional V-A claims** (e.g. "high arousal / high valence") — supported iff
consistent with the neighbours' mean-V-A sign. `GP < 1` ⇒ the model introduced content absent
from the evidence. *Methodology:* `eval_rag_faithfulness.py` recomputes top-5 and foils from
`prototypes_dual.npy`, parses `artifacts/explanations_5songs.txt`, and scores both assertion
types per song. Log: `reports/run_logs/rag_faithfulness.log`.

**Measured results (n=5, illustrative — not a headline quantitative claim):**
- ID-grounding precision: **1.00** (song 562 names IDs 296, 91, 821, 99, 704, 457 — all
  within {query ∪ top-5 ∪ foils}; the other 4 songs use generic references with no explicit IDs)
- Directional faithfulness: **4/5 songs fully consistent; 9/10 individual V/A axis-claims
  correct (0.90)**
- Single failure: song 282 (centre/ambiguous) — prose asserts "high arousal **and** high
  valence" while neighbours' mean valence = 0.438 (mildly negative, spanning 3 quadrants).
  That directional valence mismatch is exactly the over-generalisation GP flags.

Recomputed top-5 matches the printed export for all 5 songs (sanity check passed).

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

| Encoder | Precision@5 | Precision@10 | Precision@20 | Silhouette (pooled) |
| :-- | :-: | :-: | :-: | :-: |
| MERT (single) | 0.5760 | 0.5687 | 0.5469 | −0.0293 |
| Dual-SSL (MERT + wav2vec2) | 0.5849 | 0.5613 | 0.5382 | +0.0026 |
| **Enhanced (MERT + w2v2 + cyclic key)** — *deployed* | **0.5943** | **0.5761** | **0.5477** | 0.0039 |

**Precision@k validates retrieval — and the deployed Enhanced model is the best
retriever.** ~55–59% of each song's nearest latent neighbours fall within a 0.20 V-A
radius. The **random-chance baseline is 0.276**, so the **Enhanced** model's
Precision@5 = **0.594** is ~**2.15× chance** and the highest of all configurations
(+0.009 over Dual). This matters because Enhanced is the architecture the unified Phase C
pipeline actually deploys (§2a), so the reported retrieval number now matches the
deployed model. This is the load-bearing evidence and the foundation of the
example-based explanation. (`phaseB/eval_enhanced_retrieval.py`, same out-of-sample
protocol as the rows above.)

**Silhouette — pooled-models artifact, not the canonical magnitude.** The Silhouettes in
this table (untrained 0.100, MERT −0.029, Dual +0.003, Enhanced +0.004) are computed on the
retrieval index, which **pools test-fold latents from all 5 separately-trained fold models**;
those spaces are independently rotated, so a *global* Silhouette over the pool collapses to
≈0 — a pooling artifact, not a property of any model (Precision@k, a *local* metric, is
unaffected and stays ~0.58–0.59). The **canonical** Silhouette, computed **per-fold within a
single model**, is **≈ 0.19 Euclidean / ≈ 0.26 cosine** (single-MERT ≈ Enhanced,
`04_results_and_sota.md` §2a-sexies). Either way the score is *weak-to-moderate* and far
below the ≳0.5 of clean separable clusters:
emotion is a **continuous, organized** V-A gradient, not four discrete clusters.
Silhouette-by-quadrant imposes hard 0.5 cutoffs on a smooth manifold, so boundary
songs are legitimately close to the adjacent quadrant. **Lead with Precision@k**
(the protocol-robust metric); present Silhouette with this caveat. The two encoders
are statistically tied — consistent with the Phase B finding that the second encoder
adds little to latent *organization*.

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
embeddings). **Honest nuance (with the corrected magnitude):** within this *in-sample
index* measurement the untrained space scores a higher Silhouette than the trained one
(0.100 vs ≈0) — raw MERT retains coarse genre/acoustic blobbiness but worse
fine-grained V-A retrieval, and SupCR reorganizes around *continuous* V-A proximity,
raising Precision@k while flattening in-sample discrete clusters. Note, however, that
the *canonical* held-out Silhouette of the trained space is ≈0.26 cosine (§2a-sexies),
not ≈0 — so "training destroys clusters" overstates it; more precisely, training trades
a little in-sample quadrant blobbiness for substantially better V-A retrieval. The
metric that matters for retrieval (Precision@k) improves; the discrete-cluster metric
(Silhouette) is the secondary lens, as argued above.

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
  Layer 2 was generated **server-free, without sudo**, via the HuggingFace
  `transformers` backend; the exported run used **`Qwen/Qwen2.5-3B-Instruct`** (file
  header: `LLM: hf (Qwen/Qwen2.5-3B-Instruct)`), with `Qwen2.5-1.5B-Instruct` as the
  lighter default and `--llm ollama|anthropic` as selectable (unused) alternatives.
  Runs in the existing venv on the GPU — Ollama/sudo not required. The model produces
  the intended 4-part structured explanation (Recommendation / Emotional Connection /
  Why-Not-The-Others / Listening Experience). **Caveat:** a 1.5–3B local model is the
  quality floor; prose is coherent but model-dependent — Layer 1 remains the citable,
  deterministic artifact (see the song-282 grounding failure in §3b).

## 6. How to Run

```bash
cd MERT/ssl_scripts/phaseC
# evaluation (rebuilds index + Precision@k + Silhouette), dual = best model:
python evaluate_latent_space.py --encoder dual --beta 0.05 --epochs 100 \
    --w2v_path ../phaseB/pmemo_wav2vec_all_layers.pt --index_path prototypes_dual.npy
# a single explained query (uses the index only; no model load needed):
python mainC.py --mode query --query_id 760 --index_path prototypes_dual.npy --llm none
```
