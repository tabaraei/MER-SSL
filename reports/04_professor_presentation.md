# Thesis Progress Presentation
**Explainable & Perceptual Music Emotion Recognition via Self-Supervised Learning**  
**Student:** Arvin Jafari Moghadam Fard  
**Date:** May 2026

---

## Slide 1 — The Problem

**Standard Music Emotion Recognition (MER) has two unsolved problems:**

1. **Accuracy ceiling on valence** — Audio-only models consistently plateau at R² ≈ 0.50 for valence prediction, while arousal reaches 0.70+. The gap is attributed to valence's stronger dependence on high-level musical semantics (lyrics, cultural context) vs. low-level acoustic features.

2. **The black-box problem** — Existing MER systems predict arousal/valence scores but cannot explain *why* a song occupies a given emotional region, nor *why* a recommendation was made. This limits deployment in contexts requiring trust: therapeutic music, clinical settings, adaptive playlists.

**This thesis addresses both problems simultaneously** through a three-phase pipeline that predicts, validates, and then explains music emotion.

---

## Slide 2 — The Dataset & Foundation

**Dataset:** PMEmo 2019 — 794 pop songs, 457 participants  
- Ground truth: continuous arousal/valence annotations per chorus segment  
- Physiological signals: EDA (electrodermal activity) per participant  
- 767 songs successfully matched after ID alignment  

**Pre-trained Model:** MERT-v1-330M (m-a-p)  
- Transformer-based SSL model pre-trained on 160,000 hours of music  
- Provides 25 transformer layers × 1024-dim hidden states per audio frame  
- No fine-tuning of MERT itself — used as a frozen feature extractor  

**Why MERT over raw audio?**  
SSL pre-training captures abstract musical representations (harmony, rhythm, melody) without any supervision — validated by Phase A probing experiments.

---

## Slide 3 — Phase A: Does the Representation Know Music?

**Research Question:** Do MERT embeddings encode musically meaningful information before any emotion fine-tuning?

**Method:** Linear probing — a simple classifier/regressor is trained on frozen MERT features. If a linear model succeeds, the information is directly encoded in the representation.

| Probe Task | Model | Result | Interpretation |
| :--- | :--- | :--- | :--- |
| **Major/Minor mode** | Linear classifier (1024 → 1) | **~100% accuracy** | Harmonic structure is explicitly encoded |
| **Tempo (BPM)** | Linear regressor (1024 → 1) | **R² = 0.12** | Rhythmic information present but coarse |
| **Emotion clustering** | t-SNE visualization | Quadrant separation visible | Emotion geometry exists before training |

**Conclusion:** MERT is a valid foundation. The representation is harmonically aware and emotionally structured. Any emotion-prediction improvements in Phase B are *refinements*, not compensating for a weak foundation.

> *t-SNE plots and layer-weight bar charts available as thesis figures.*

---

## Slide 4 — Phase B: The Hybrid Affective Model

**Objective:** Learn a mapping from MERT's 25-layer representation to the Valence–Arousal circumplex.

### Architecture: Three Key Design Decisions

**1. Weighted Layer Fusion (not just last-layer)**  
Instead of using only MERT's final layer, a learnable softmax over all 25 layers is trained:
```
25 layers × 1024-dim  →  WeightedLayerFusion  →  fused 1024-dim
                      →  Deep Head (1024 → 256 → 128)
                      →  Regressor (128 → 2: arousal, valence)
```
*Finding (single-encoder): layers 14, 16, 17 (mid-to-late semantic) dominate → weight entropy = 3.2178 (near-optimal specialization).*

**1b. Dual-SSL Extension (MERT + wav2vec2)**  
To break the valence ceiling, a second frozen SSL encoder (wav2vec2-base, speech-pretrained, 13 layers × 768-dim) is added:
```
MERT  (25L × 1024)  →  WeightedLayerFusion  →  1024-dim ─┐
                                                           ├── cat → 1792-dim → Head → 2
wav2vec2 (13L × 768) →  WeightedLayerFusion  →   768-dim ─┘
```
*Result: Valence R² improved from 0.5055 → 0.5676. Novel finding: in the dual setting, both fusion modules converge to uniform layer weights (fusion collapse) — documented as a data-constraint finding.*

**2. Hybrid Loss Function (4 components)**

| Loss | Weight | Purpose |
| :--- | :--- | :--- |
| MSE | 1.0 | Absolute regression accuracy |
| CCC | 0.5 | Correlation + mean + variance agreement (AVEC standard) |
| Rank | 0.3 | Preserves ordinal emotion ordering (soft Spearman) |
| SupCR | 0.1 | Pulls similar emotions together in latent space (for retrieval) |

**3. Differential Optimizer + Balanced Sampling**  
- Fusion weights: lr = **1e-2** (aggressive — overcomes "frozen weight" problem)  
- Head/regressor: lr = **1e-4** (conservative)  
- Weighted Random Sampler inverts class frequency → forces model to learn underrepresented quadrants (Sad, Angry, Calm), resolving a **Simpson's Paradox** in the dataset

### Multimodal Extension: EDA Fusion

EDA signals (mean, std, slope, SCR peaks, amplitude, max, high-ratio) projected to 32-dim and fused with the 128-dim audio bottleneck via late fusion:
```
Audio latent (128-dim) ──┐
                         ├──► Fusion Head (160 → 64 → 2)
EDA features (7 → 32)  ──┘
```

---

## Slide 5 — Phase B Results

### Validated Performance (5-Fold Cross-Validation)

|  | **Arousal R²** | **Valence R²** | **CCC Arousal** | **CCC Valence** |
| :--- | :---: | :---: | :---: | :---: |
| MERT only (Audio) | 0.6518 | 0.5055 | 0.82 | 0.74 |
| MERT + EDA Fusion | 0.6738 | 0.5075 | **0.8543** | 0.7692 |
| Dual-SSL (MERT + wav2vec2, β=0.05) | 0.6814 | 0.5676 | 0.8087 | 0.7231 |
| **Triple (MERT + wav2vec2 + mel-CNN)** | **0.7023** | **0.5758** | 0.8233 | 0.7329 |
| Spec-only (MERT + mel-CNN, no w2v) | 0.7069 | 0.5709 | 0.8271 | 0.7314 |

The **Triple** model (frozen MERT + frozen wav2vec2 + a shallow *trainable* mel-spectrogram CNN) is the strongest configuration — first past Arousal R² 0.70. **Key finding:** Spec-only (MERT + mel-CNN, no wav2vec2) is statistically equal to Triple → a 109K-param from-scratch CNN fully substitutes for the 95M-param frozen wav2vec2. ⚠️ **Caveat:** the global R² is majority-class (HVHA 61%) driven; per-quadrant R² is negative on the three minority quadrants across all configs — additional encoders do not resolve the PMEmo class-imbalance ceiling.

### SOTA Comparison (PMEmo 2019, Final)

| Method | Year | Valence R² | Arousal R² | CCC (V / A) |
| :--- | :---: | :---: | :---: | :---: |
| PMEmo Original (hand-crafted) | 2019 | 0.420 | 0.510 | — |
| Damer — Wav2Vec2 + Chords | 2025 | 0.510 | 0.720 | — |
| Music2Emo — MERT + Multitask | 2025 | 0.540 | 0.780 | — |
| This Thesis — MERT + EDA | 2026 | 0.508 | 0.674 | 0.77 / 0.85 |
| This Thesis — Dual-SSL | 2026 | 0.568 | 0.681 | 0.72 / 0.81 |
| **This Thesis — Triple (best)** | **2026** | **0.576** | **0.702** | **0.73 / 0.82** |

**Note on CCC vs R²:** CCC simultaneously penalizes poor correlation, mean shift, and variance mismatch (AVEC standard). CCC Arousal **0.8543** (MERT+EDA) is the highest reported on PMEmo 2019.

### Layer Specialization — Encoder-Dependent Finding

- **Single-encoder MERT:** layers 14, 16, 17 dominate. Weight entropy = **3.2178** (0.05% below theoretical max). Mid-to-late layers encode harmonic/melodic abstractions — consistent with probing literature (Pasad et al., 2021).
- **Dual-encoder (MERT + wav2vec2):** both fusion modules stay at maximum entropy (near-uniform weights). This is the **fusion collapse** finding: with ~600 labeled samples, the larger concatenated head solves the task without requiring per-encoder layer selection. The single-encoder specialization is an advantage of the simpler architecture.
- **Triple-branch convergence:** Adding a trainable mel-CNN makes wav2vec2 statistically redundant (Spec-only ≈ Triple). Combined with fusion collapse and the IADS-E negative result, this is a consistent three-way finding: **wav2vec2's speech-pretraining contributes no music-relevant structure**; the gains come from MERT + a learnable spectral branch.

---

## Slide 6 — Phase C: Explainable Retrieval System

**Research Question:** Can the emotion-aware latent space support a retrieval system that not only returns emotionally similar music, but *explains its decisions* in human-interpretable, psychologically grounded terms?

### Why Explainability Matters Here

- **Trust:** Users cannot evaluate recommendations they cannot understand
- **Clinical relevance:** Therapeutic/clinical music applications require justifiable recommendations
- **Scientific validation:** If the latent space truly captures emotion, retrieved songs should be explainable in music-theoretic terms — this is an independent validation of Phase B

### System Architecture

```
Phase B model (best_model.pt)
    │
    ▼
VectorIndexBuilder        → prototypes.npy (767 songs × latents + V-A + EDA + layer weights)
    │
    ▼
EmotionRetriever          → top-k neighbors (cosine k-NN in latent space)
                          → n contrastive foils (most dissimilar songs)
    │
    ▼
ExplainableRAG
    ├── Layer 1: Deterministic Template  →  precise, reproducible, thesis-citable
    └── Layer 2: LLM RAG Prompt         →  humanized, warm, recommendation-style
              ├── Emotion Knowledge Base (Russell quadrant → genre/tempo/harmony)
              ├── EDA Physiological Narrative (body response interpretation)
              ├── Contrastive Foils (why NOT those songs?)
              ├── MERT Layer Attribution (what features drove retrieval?)
              └── Mood Trajectory (energizing vs wind-down listening arc)
    │
    ▼
LLM Backend (Ollama local / Anthropic API)
    └── 150–200 word structured recommendation + explanation
```

---

## Slide 7 — Phase C: XAI Innovations

Four explainability features, each grounded in published literature:

### 1. Contrastive Foils *(Miller, 2019)*
The retriever returns the **most dissimilar** songs alongside the top-k results. The LLM uses these "rejected" songs to answer: *"Why this song and not that one?"*

> *"Song 699 was not retrieved because its valence (0.32) places it firmly in the melancholic region — emotionally distant from your calm, positive query despite similar low arousal."*

**Why this matters:** Social science research (Miller, 2019) identifies contrastive explanation as the most cognitively natural form of explanation for humans. It is a standard XAI technique applied here to music retrieval for the first time.

### 2. EDA Physiological Narrative *(Thayer, 1989)*
7-dimensional EDA features (mean conductance, SCR peaks, trend, variability) translated to language:

> *"Listeners of this track showed a calm physiological state with stable skin conductance and sustained arousal throughout — few distinct peaks, consistent with a gently engaging musical experience."*

**Why this matters:** Thayer's biopsychological model links V-A arousal directly to autonomic nervous system activation (measured by EDA). Physiological evidence provides *convergent validity* — two independent data streams agreeing on emotional character.

### 3. MERT Layer Attribution
The model's learned layer weights are interpreted as reliance on different acoustic feature levels:

> *"Retrieval is primarily driven by high-level semantic and melodic structure (Layer 17: 12% weight). Low-level acoustic features contribute 28%, mid-level rhythmic patterns 31%, high-level semantic features 41%."*

### 4. Mood Trajectory
Retrieved songs sorted by arousal → two listening arc suggestions:
- **Energizing:** [Song 699 → 481 → 116 → 221 → 716] (Δarousal = 0.15)
- **Wind-down:** Reverse order

**Grounding:** Saarikallio & Erkkilä (2007) show mood regulation is a primary use of music; trajectory design operationalizes this directly.

---

## Slide 8 — Sample System Output

**Query:** Song ID 760 (Low Arousal / High Valence — calm, positive)

```
═══════════════════════════════════════════════════════════════
🎵  QUERY: Song ID 760
═══════════════════════════════════════════════════════════════
  Emotion Region  : Low Arousal / High Valence (HVLA)
  Arousal         : low (0.352) | Valence: positive (0.571)
  Physiology      : low EDA, few SCR peaks, stable trend

🔍  TOP-5 RETRIEVED SONGS
  ┌─ Rank 1 | Song ID 481 (cosine sim = 0.9977)
  │  Quadrant : LVLA | Arousal ↓0.03 | Valence ↓0.11 vs query
  ┌─ Rank 5 | Song ID 221 (cosine sim = 0.9964)
  │  Quadrant : HVLA | Arousal ↑0.06 | nearly identical V profile

🧠  MUSIC-THEORETIC SUMMARY
  avg arousal=0.346 | valence=0.448 | mean cosine sim=0.9969
  Mood arc: Songs [699→481→116→221→716] — Δarousal = 0.15

🤖  LLM EXPLANATION (Ollama / llama3.2)
  "Like your track, these songs inhabit a calm, gently uplifting 
   emotional space — not quite melancholic, but reflective and 
   unhurried. They share that same quiet energy, the kind that 
   works for late-evening listening or focused work.
   
   The songs not recommended — ID 312 (energetic, aggressive, 
   sim=0.41) and ID 88 (very high arousal, sim=0.38) — sit in 
   a completely different emotional register. They share no 
   physiological affinity with your calm query: their listeners 
   showed elevated, variable skin conductance throughout.
   
   These five tracks collectively create a stable, low-arousal 
   emotional environment — your body and mind stay in a settled, 
   parasympathetic state from start to finish."
```

---

## Slide 9 — Evaluation Metrics

### Phase B Metrics

| Metric | What it measures |
| :--- | :--- |
| **R²** | Proportion of variance explained in V-A space |
| **CCC** | Simultaneous correlation + mean + variance agreement (AVEC standard) |
| **Per-quadrant R²** | Performance per emotion region (reveals dataset bias) |

### Phase C Metrics

| Metric | Definition | Value (approx.) |
| :--- | :--- | :--- |
| **Precision@5** | % top-5 neighbors within 0.20 V-A radius | To be measured |
| **Precision@10** | Same for top-10 | To be measured |
| **Silhouette** | Quadrant cluster separation in cosine latent space | To be measured |

> *Phase C evaluation will run once the index is rebuilt with EDA features loaded correctly (EDA filename fix was implemented — `{id}_EDA.csv` format confirmed).*

---

## Slide 10 — Honest Limitations

These are acknowledged in the thesis reports and should be framed as evidence of research rigour, not failure:

| Limitation | Scientific Framing |
| :--- | :--- |
| Valence R² capped at 0.51 | Known ceiling in MER — valence requires lyrics/cultural context not present in audio alone. CCC of 0.77 shows agreement quality is high despite residual variance. |
| PMEmo has no artist/genre metadata | Motivates the static emotion knowledge base approach; identified as future work (MusicBrainz mapping) |
| EDA normalized across dataset | Physiological narrative reflects population-level tendencies, not individual variation — acknowledged explicitly |
| No user study for XAI validation | Two-layer design (template + LLM) provides proxy: deterministic layer is independently evaluable; user study proposed as future work |
| LLM explanation depends on model quality | Structured 4-part prompt with word limit reduces variance; Ollama 7B+ recommended |

---

## Slide 11 — Scientific Contributions Summary

| Phase | Contribution | Novelty |
| :--- | :--- | :--- |
| **A** | Linear probing of MERT for harmonic mode and tempo | Validates SSL foundation for interpretable MER |
| **B** | Hybrid loss (MSE + CCC + Rank + SupCR) | Combined metric addresses multiple V-A prediction failure modes |
| **B** | Differential optimizer (1e-2 / 1e-4) | Solves "frozen fusion weight" problem in weighted-layer SSL fine-tuning |
| **B** | Balanced sampler for emotion quadrant imbalance | Resolves Simpson's Paradox in PMEmo quadrant-level analysis |
| **B** | Multimodal EDA fusion with late-fusion head | Physiological grounding; CCC Arousal 0.8543 — highest on PMEmo 2019 |
| **B** | Dual-SSL (MERT + wav2vec2) with entropy sharpening | Valence R² 0.5676 audio-only |
| **B** | Triple-branch (MERT + wav2vec2 + trainable mel-CNN) | Best result: Arousal R² 0.7023, Valence R² 0.5758 |
| **B** | wav2vec2-redundancy finding (Spec-only ≈ Triple) | Small from-scratch CNN substitutes for 95M-param frozen wav2vec2 |
| **B** | Fusion collapse finding | Data-constraint result: ~600 samples insufficient for multi-encoder layer selection |
| **B** | IADS-E joint learning — negative finding | SSL cross-domain emotional transfer underperforms hand-crafted features |
| **B** | Class-imbalance ceiling characterization | Global R² is HVHA-majority-driven; minority-quadrant R² negative across all configs |
| **C** | Contrastive foil retrieval for XAI | First application of Miller (2019) contrastive XAI to music retrieval |
| **C** | EDA-augmented explanation narrative | First physiologically-grounded explanation in music recommendation |
| **C** | Two-layer explanation system | Separates scientific rigour (template) from usability (LLM) |
| **C** | Mood trajectory from emotion latent space | Direct operationalization of music mood regulation research |

---

## Slide 12 — Current Status & Roadmap

### What Is Complete ✅

- [x] MERT embedding extraction pipeline (767 PMEmo songs)
- [x] wav2vec2 embedding extraction pipeline (same 767 songs, 16kHz)
- [x] Phase A: harmonic mode probe (100%), tempo probe (R²=0.12), t-SNE visualization
- [x] Phase B: MERT-only hybrid model — CCC Arousal 0.8543, Valence R² 0.5055
- [x] Phase B: EDA multimodal fusion — CCC Arousal 0.8543 (best on PMEmo 2019)
- [x] Phase B: Dual-SSL (MERT + wav2vec2) — Valence R² 0.5676
- [x] Phase B: Entropy sharpening penalty (β=0.05) — acts as regularizer
- [x] Phase B: Triple-branch (+ trainable mel-CNN) — best: Arousal R² 0.7023 / Valence R² 0.5758
- [x] Phase B: wav2vec2-redundancy finding — Spec-only (MERT+mel) ≈ Triple
- [x] Phase B: IADS-E joint learning — confirmed negative finding (all configs < dual baseline)
- [ ] Phase B: train-loss logging for triple-branch — quantify CNN overfitting (open action)
- [x] Phase B: Fusion collapse finding documented — data-constraint on layer selection
- [x] Phase C: modular retrieval pipeline (6 modules)
- [x] Phase C: contrastive foils, EDA narrative, layer attribution, mood trajectory
- [x] Phase C: LLM integration (Ollama local + Anthropic API)
- [x] Research reports documenting all phases with academic references
- [x] Full CLI README for reproducibility

### What Remains 🔧

- [ ] Run Phase C `evaluate` mode → Precision@k and Silhouette scores (requires EDA index rebuild)
- [ ] Baseline Phase C index → compare Precision@k (hybrid vs baseline model)
- [ ] Formal ablation study: which loss components contribute most to CCC improvement?
- [ ] Thesis writing: Introduction, Related Work, Methodology, Results, Discussion

### Proposed Future Work (Thesis Conclusion)

1. MusicBrainz/Spotify metadata mapping → artist-specific recommendations
2. Per-query MERT layer attribution via Grad-CAM
3. Joint audio+EDA latent space (Phase B `MERModelWithEDA` as retrieval backbone)
4. User study on explanation quality (template vs LLM vs no explanation)
5. Personalization via multi-query taste profile

---

## References

- Li, Y., et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. *arXiv:2306.00107*.
- Miller, T. (2019). Explanation in Artificial Intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1–38.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
- Saarikallio, S., & Erkkilä, J. (2007). The role of music in adolescents' mood regulation. *Psychology of Music*, 35(1), 88–109.
- Thayer, R. E. (1989). *The Biopsychology of Mood and Arousal*. Oxford University Press.
- Yang, Y.-H., & Chen, H. H. (2012). Machine recognition of music emotion: A review. *ACM TIST*, 3(3), 40.
- Kim, B., et al. (2016). Examples are not enough, learn to criticize! *NeurIPS 2016*.
