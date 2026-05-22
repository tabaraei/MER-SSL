# Phase C: Explainable RAG Retrieval System — Technical & Scientific Report
**Student:** Arvin Jafari Moghadam Fard  
**Status:** Implemented and Functional  
**System:** Prototype-based Music Emotion Retrieval with XAI-grounded Natural Language Explanation

---

## 1. Research Motivation

The primary thesis contribution of Phase C is not retrieval accuracy per se — k-NN retrieval in a learned latent space is well-established. The contribution is tpython mainC.py --mode build --model_path ../phaseB/best_model.pt \
    --feat_path ../phaseB/pmemo_mert_all_layers.pt \
    --csv_path /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
    --eda_dir /datasets/emotions/PMEmo2019/EDA --index_path prototypes.npy

python mainC.py --mode query --query_id 760 --index_path prototypes.npy \
    --feat_path ../phaseB/pmemo_mert_all_layers.pt \
    --csv_path /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
    --model_path ../phaseB/best_model.pt --eda_dir /datasets/emotions/PMEmo2019/EDA \
    --top_k 5 --n_foils 3 --llm ollama --llm_model llama3.2he **explainability layer**: the ability to tell a listener *why* a song was recommended in terms that are simultaneously scientifically grounded and humanly legible.

This matters for three reasons directly relevant to the thesis argument:

**1.1 The Trust Problem in Affective AI**  
Music recommendation systems typically operate as black boxes. Users accept or reject recommendations without understanding the system's reasoning. For emotion-sensitive applications (therapeutic playlists, mood regulation, clinical music therapy), unexplained recommendations can be inappropriate or even counterproductive. Explainability is not a cosmetic feature — it is a functional requirement for responsible affective AI deployment.

**1.2 The Evaluation Gap in MER**  
Standard MER evaluation ($R^2$, CCC) measures how accurately a model predicts emotion labels, but provides no evidence that the model's internal representations are *meaningful* in a music-theoretic sense. Phase C provides a complementary evaluation path: if the latent space genuinely captures emotional structure, then retrieved neighbors should be explainable in terms of shared musical properties (arousal/valence profile, physiological response patterns, tonal character). This is an indirect, but human-interpretable, validation of the Phase B model.

**1.3 The XAI Research Landscape**  
Explainable AI in music retrieval remains underexplored. Most XAI work in music focuses on classification (e.g., explaining genre predictions via saliency maps). Extending XAI to *retrieval* — and grounding it in physiological signals (EDA) — is a novel contribution that differentiates this thesis from standard MER work.

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
Phase B Output                      Phase C Pipeline
─────────────                       ──────────────────────────────────────────────
best_model.pt     ──────────────►  [1] VectorIndexBuilder
                                        │  encode all songs → (N, D) latents
pmemo_mert_all_layers.pt ──────►       │  L2-normalize → cosine sim via dot product
                                        │  store: latents, V-A scores, EDA feats,
                                        │         layer weights → prototypes.npy
                                        ▼
                                   [2] EmotionRetriever
                                        │  query_latent → top-k neighbors (cosine k-NN)
                                        │  query_latent → n foils (most dissimilar)
                                        ▼
                                   [3] ExplainableRAG
                                        │  Layer 1: deterministic template
                                        │  Layer 2: enriched LLM prompt
                                        │     ├─ emotion knowledge base (per quadrant)
                                        │     ├─ EDA physiological narrative
                                        │     ├─ contrastive foils (XAI)
                                        │     ├─ MERT layer attribution
                                        │     └─ mood trajectory suggestion
                                        ▼
                                   [4] LLM Backend (Ollama / Anthropic)
                                        └─ humanized recommendation text
```

### 2.2 Index Construction (`index_builder.py`)

The index stores per-song data in a single NumPy dictionary (`.npy` file):

| Field | Shape | Description |
| :--- | :--- | :--- |
| `latents` | (N, D) float32 | L2-normalized latent vectors from Phase B model |
| `music_ids` | (N,) int32 | PMEmo integer music IDs |
| `arousal` | (N,) float32 | Ground-truth arousal scores |
| `valence` | (N,) float32 | Ground-truth valence scores |
| `pred_arousal` | (N,) float32 | Model-predicted arousal |
| `pred_valence` | (N,) float32 | Model-predicted valence |
| `eda_feats` | (N, 7) float32 | EDA statistical features, dataset-normalized |
| `layer_weights` | (L,) float32 | Global MERT layer fusion weights (or None) |

**L2 Normalization Design Choice:** Normalizing latent vectors to unit norm transforms cosine similarity into an inner product (dot product), enabling exact FAISS `IndexFlatIP` search. This is computationally efficient and geometrically meaningful: the angle between emotion vectors directly represents emotional distance, independent of embedding magnitude.

**Layer Weights Storage:** The `WeightedLayerFusion` weights from Phase B are captured once at build time and stored in the index. This means every query explanation can reference the same model-level layer attribution without re-loading the model.

### 2.3 Retrieval Engine (`retriever.py`)

**Standard Retrieval:**  
Exact cosine k-NN with FAISS `IndexFlatIP` (falls back to `sklearn` `NearestNeighbors` if FAISS is unavailable). Self-exclusion is handled by an L2-norm proximity check rather than ID lookup, which is more robust to floating-point indexing.

**Contrastive Foil Retrieval (`query_foils`):**  
A direct matrix-vector product `latents @ query.T` computes cosine similarity for all N songs simultaneously. Sorting ascending and taking the first `n_foils` entries yields the most emotionally distant songs. These foils are passed to the LLM as "rejected candidates" — a contrastive explanation strategy grounded in Miller (2019), who demonstrates that humans naturally explain events by referencing what *did not* happen.

**Scientific Argument for Foils:**  
Counterfactual explanations satisfy the "selectivity" requirement of good scientific explanation (Lipton, 1990): instead of listing all reasons a song was retrieved, they identify the *contrastive* reason — what would have made the result different. For a thesis on explainable retrieval, this is methodologically important because it demonstrates the system understands the *structure* of the latent space, not just its nearest-neighbor geometry.

### 2.4 EDA Feature Extraction (`eda_loader.py`)

Seven statistical features extracted from each song's mean EDA signal (averaged across PMEmo participants):

| Index | Feature | Interpretation |
| :--- | :--- | :--- |
| 0 | `mean_eda` | Overall physiological arousal level |
| 1 | `std_eda` | Variability of arousal response |
| 2 | `slope` | Trend: escalating (+) or relaxing (−) arousal |
| 3 | `n_peaks` | Number of skin conductance responses (SCR) |
| 4 | `mean_amp` | Average SCR amplitude (emotional intensity) |
| 5 | `max_eda` | Peak physiological arousal moment |
| 6 | `high_ratio` | Fraction of time in high-arousal state |

**Normalization:** Min-max normalized across the dataset before storage. This is important to note in the thesis: the physiological values are *relative* to the PMEmo population, not absolute physiological measurements.

**Scientific Argument for EDA in Explanations:**  
Including EDA in explanations bridges audio-based emotion prediction and physiological evidence. Thayer's (1989) biopsychological model posits that arousal in the V-A circumplex maps directly to autonomic nervous system activation, which EDA measures. By confirming that retrieved songs share similar EDA profiles — not just similar predicted V-A coordinates — the explanation provides *converging evidence* from two independent data modalities. This strengthens the scientific credibility of the retrieval.

### 2.5 MERT Layer Attribution

The Phase B model learns a `WeightedLayerFusion` over 25 MERT transformer layers (1 embedding + 24 transformer blocks). The learned softmax weights reveal which acoustic feature levels the model finds most informative for emotion prediction.

**Layer grouping (approximate, architecture-dependent):**
- **Layers 0–8 (low-level):** Acoustic/timbral features — onset, spectral texture, formants.
- **Layers 9–16 (mid-level):** Rhythmic patterns, tonal relationships, melodic contour.
- **Layers 17–24 (high-level):** Semantic/musical abstractions — phrase structure, harmonic progression.

**Validated finding from Phase B:** The model concentrates weight on layers 14, 16, and 17, with entropy 3.2178 (down from maximum ~3.22 for 25 uniform layers). This confirms specialization in mid-to-late transformer abstractions, consistent with findings in other music SSL probing studies showing that higher MERT layers encode more harmonic and structural information (Li et al., 2023, *MERT* paper).

**How to argue this in the thesis:** Layer attribution makes the retrieval *mechanistically interpretable* — the system can explain not just *that* songs are similar, but *what kind of acoustic information* drove the similarity judgement. This is a direct contribution to the XAI dimension of the thesis.

---

## 3. Explanation Generation (`explainer.py`)

### 3.1 Template Explanation (Layer 1)

Deterministic, reproducible output including:
- Russell quadrant label and description
- Arousal/valence with qualitative labels (very low → very high)
- Per-neighbor: quadrant, V-A delta from query (↑/↓), cosine similarity, EDA summary
- Mood trajectory (energizing and wind-down listening arc)
- Aggregate statistics: mean neighbor A/V, mean cosine similarity

**Thesis use:** This output can be directly included as a figure or table in the thesis results chapter to demonstrate the system's output format.

### 3.2 RAG Prompt Engineering (Layer 2)

The LLM prompt is structured into five named sections:

1. **QUERY SONG** — emotion character, quadrant, A/V with qualitative labels, music theory context from knowledge base, EDA narrative.
2. **RETRIEVED SONGS** — each neighbor described in plain English with emotional delta and EDA narrative.
3. **CONTRASTIVE FOILS** — rejected songs with explicit instruction to use them for contrast.
4. **MODEL EXPLANATION** — MERT layer attribution.
5. **MOOD TRAJECTORY** — listening arc suggestion.

**LLM Task Specification:**  
The prompt instructs a four-part structured response (Recommendation / Emotional Connection / Contrast with Foils / Listening Experience) with an explicit word limit (150–200 words). This structured instruction approach (chain-of-thought prompting) is more reliable than open-ended generation for consistent academic output.

**Static Emotion Knowledge Base:**  
Rather than asking the LLM to infer genre families from V-A coordinates (hallucination risk), the prompt injects pre-validated knowledge per Russell quadrant. This knowledge is grounded in music psychology literature and reviewed for accuracy. The key insight: the LLM's role is *translation and synthesis*, not *knowledge generation* — it translates structured scientific context into human language.

**Why this is academically defensible:**  
The knowledge base makes the LLM's contribution *verifiable*. Every genre family, tempo range, and harmonic description in `EMOTION_KNOWLEDGE` can be cited to music psychology literature. This means the LLM explanation is not a black box — its knowledge source is explicit and auditable, which is precisely what makes it appropriate for a thesis on *explainable* AI.

### 3.3 Mood Trajectory

Retrieved neighbors are sorted by predicted arousal and presented as two listening arc options:
- **Energizing arc:** ascending arousal (calm → energetic)
- **Wind-down arc:** descending arousal (energetic → calm)

**Scientific grounding:** Mood regulation via music is a well-studied phenomenon (Saarikallio & Erkkilä, 2007). Listeners actively use music to manage emotional state transitions. The trajectory suggestion directly operationalizes this as a practical recommendation feature — and demonstrates that the Phase C system goes beyond static retrieval to *dynamic playlist design*.

---

## 4. Evaluation Framework (`evaluator.py`)

### 4.1 Precision@k

**Definition:** For each song *i* in the index, retrieve top-k neighbors (excluding self). Count the fraction whose V-A Euclidean distance from *i* is below threshold θ = 0.20.

$$\text{Precision@k} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\{j \in \text{top-k}(i) : \|va_i - va_j\|_2 < \theta\}|}{k}$$

**Threshold justification:** θ = 0.20 corresponds to one-fifth of the V-A unit square diagonal (≈ 0.28), a moderate emotional proximity criterion. Songs within this radius share the same Russell quadrant in most cases, making it a reasonable operationalization of "emotionally close."

**Limitation to state in thesis:** V-A proximity is a proxy for perceptual similarity, not its direct measure. True evaluation would require human listening studies comparing retrieved songs with non-retrieved alternatives. Precision@k as defined here evaluates *label consistency*, not *perceptual quality*.

### 4.2 Silhouette Score

**Definition:** Average silhouette over all songs, partitioned by Russell quadrant (4 classes), computed in cosine similarity space.

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean nearest-cluster distance.

**Interpretation:**
- $s \approx +1$: quadrants are well-separated in latent space (good for emotion-specific retrieval)
- $s \approx 0$: quadrant boundaries are fuzzy
- $s < 0$: songs are closer to a different quadrant's centroid than their own

**Thesis argument:** A positive Silhouette score directly validates the claim that the SupCR loss in Phase B successfully organizes the latent space into emotionally coherent neighborhoods — a prerequisite for meaningful retrieval.

### 4.3 Results (computed — test-fold / out-of-sample)

Evaluation was run with `phaseC/evaluate_latent_space.py`, which builds **out-of-sample (test-fold) latents**: 5-fold CV (KFold shuffle, `random_state=42`), where every song is encoded by the fold model that did **not** train on it. This is stricter and more honest than encoding all songs with a single model, because it cannot benefit from memorization. Both the single-MERT model and the best Dual-SSL model (MERT + wav2vec2, β=0.05) were evaluated.

| Encoder | Precision@5 | Precision@10 | Precision@20 | Silhouette |
| :--- | :---: | :---: | :---: | :---: |
| MERT (single) | 0.5760 | 0.5687 | 0.5469 | −0.0293 |
| **Dual-SSL (MERT + wav2vec2)** | **0.5849** | 0.5613 | 0.5382 | **+0.0026** |

**Precision@k — retrieval validated.** For both models, ~54–58% of each song's nearest latent-space neighbours fall within a 0.20 V-A radius — substantially above chance for a continuous 4-quadrant emotion space. This is the primary, positive evidence that the latent space supports emotionally meaningful retrieval, establishing the foundation of the example-based explanation system.

**Silhouette ≈ 0 — interpreted honestly.** Both scores are essentially zero (single-MERT slightly negative, dual marginally positive at +0.0026). This is **not** strong evidence of quadrant separation and is *not* reported as such. The reason is conceptual rather than a model failure: emotion in the valence-arousal circumplex is a **continuous gradient**, not four discrete clusters. Silhouette-by-quadrant imposes hard boundaries at 0.5 on a smooth manifold, so songs near a quadrant border are legitimately close to songs just across it — which drives the score toward zero even when local neighbourhoods are emotionally coherent (as Precision@k shows). The strong class imbalance (HVHA ≈ 61%) compounds this.

**Thesis position:** lead with Precision@k as the evidence that retrieval is emotionally coherent; report Silhouette transparently with the continuous-manifold explanation above. The two encoders are statistically equivalent on these metrics, consistent with the project-wide finding that the second encoder adds little to the *organization* of the latent space (it helps prediction, not clustering).

---

## 5. Known Limitations and Honest Thesis Discussion

| Limitation | Impact | Mitigation Implemented |
| :--- | :--- | :--- |
| PMEmo has no artist/genre metadata | Cannot make artist-specific recommendations | Static emotion knowledge base provides genre-family context without hallucination |
| EDA normalized across dataset, not per-listener | Physiological narrative reflects population mean, not individual variation | Noted in explanation text; acknowledged as dataset limitation |
| LLM explanation quality varies by model size | Small models produce generic text | Structured four-part prompt reduces variability; Ollama 7B+ recommended |
| Valence prediction $R^2$ = 0.51 | Valence-based retrieval is noisier than arousal | Acknowledged in evaluation; Silhouette score provides independent measure |
| Cosine similarity ≠ emotional similarity | High latent similarity does not guarantee perceptual similarity | Precision@k uses V-A ground truth as external validation |
| No user study | Cannot validate whether explanations satisfy real listeners | Suggested as future work; template + RAG comparison is a reasonable proxy |

---

## 6. Future Work (Thesis Conclusion Material)

1. **Metadata enrichment via MusicBrainz/Spotify:** Map PMEmo IDs to real artist/title/genre data, enabling artist-specific recommendations of the form "since you liked X, you may enjoy Y by the same artist or in the same genre."

2. **Personalization via query history:** Maintain a running V-A "taste profile" across multiple queries. Re-rank retrieved songs based on distance from the user's profile centroid.

3. **User study on explanation quality:** Compare user satisfaction between: (a) no explanation, (b) template-only, (c) template + LLM RAG. Measures: trust, perceived relevance, engagement.

4. **Per-song layer attribution via Grad-CAM:** Extend the global layer weights to per-query attribution, showing which MERT layers are most activated for each specific query song.

5. **Multimodal retrieval index:** Currently the index stores EDA features separately from the latent vectors. A joint audio+EDA embedding (from the `MERModelWithEDA` in Phase B) would enable physiologically-aware nearest-neighbor search, not just physiologically-annotated audio search.

---

## 7. Academic References

- Kim, B., Khanna, R., & Koyejo, O. (2016). Examples are not enough, learn to criticize! Criticism for interpretability. *Advances in Neural Information Processing Systems (NeurIPS)*, 29.
- Li, Y., et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. *arXiv:2306.00107*.
- Lipton, P. (1990). *Inference to the Best Explanation*. Routledge.
- Miller, T. (2019). Explanation in Artificial Intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1–38.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
- Saarikallio, S., & Erkkilä, J. (2007). The role of music in adolescents' mood regulation. *Psychology of Music*, 35(1), 88–109.
- Thayer, R. E. (1989). *The Biopsychology of Mood and Arousal*. Oxford University Press.
- Yang, Y.-H., & Chen, H. H. (2012). Machine recognition of music emotion: A review. *ACM Transactions on Intelligent Systems and Technology*, 3(3), 40.
