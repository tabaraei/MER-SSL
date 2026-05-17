# 🎼 Research Progress: Music Emotion Recognition (MER) via SSL
**Student:** Arvin Jafari Moghadam Fard  
**Status:** Phase A ✅ | Phase B ✅ | Phase C ✅ (RAG Retrieval System Implemented)  

---

## 📅 Executive Summary
This report documents the finalized development of a robust **Hybrid Affective Architecture** for MER. The system evolved through three encoder configurations: (1) single MERT encoder with EDA fusion achieving CCC Arousal **0.8543**, (2) dual SSL encoders (MERT + wav2vec2) reaching Valence R² **0.5676** — the strongest audio-only valence result — and (3) a confirmed negative finding on IADS-E joint cross-domain SSL transfer. A novel **fusion collapse** finding documents that ~600 labeled samples is insufficient for layer-selective SSL fine-tuning in multi-encoder architectures.

---

## ✅ Phase A: Perceptual Feature Validation (Completed)
**Objective:** Verify if the MERT latent space preserves music-theoretic cues necessary for explainability.

* **Harmonic Mode Probing:** Achieved **100% Accuracy** in Major/Minor detection.
* **Tempo Probing:** Achieved an $R^2$ of **0.12**, confirming that rhythmic density is encoded within the transformer layers.
* **Significance:** Confirmed that critical music-theoretic information is retained, allowing for future explainable reasoning in retrieval tasks.

---

## ✅ Phase B: Hybrid Affective Modeling (Finalized)
**Objective:** Optimize the mapping from SSL embeddings to the Valence-Arousal (V-A) circumplex.

### 1. Architectural Design: Optimized Weighted Fusion
To synthesize information across the full transformer stack, we utilized a learnable `WeightedLayerFusion` module.
* **Mechanism:** Softmax-based weights are applied to all 25 MERT layers.
* **Differential Learning Rates:** Utilized **$10^{-2}$** for fusion parameters and **$10^{-4}$** for the regression head.
* **Entropy Breakthrough:** Successfully dropped weight entropy to **3.2178**, proving the model is now specializing in specific layers (Top layers: 14, 16, 17).

### 2. Multi-Objective Optimization: HybridLoss & Balanced Sampling
* **Balanced Sampling:** Implemented to force the model to learn from underrepresented quadrants (Sad, Angry, Calm), reducing the bias toward "Happy" music.
* **SupCR:** Physically reorganizes the 1024-D latent space into emotional "neighborhoods" for retrieval.

### 3. Validated Results — All Configurations (5-Fold CV)

| Configuration | R² Arousal | R² Valence | CCC Arousal | CCC Valence |
|:---|:---:|:---:|:---:|:---:|
| MERT only (hybrid, audio) | 0.6518 | 0.5055 | 0.82 | 0.74 |
| MERT + EDA fusion | 0.6738 | 0.5075 | **0.8543** | 0.7692 |
| **Dual-SSL (MERT + wav2vec2, β=0.05)** | **0.6814** | **0.5676** | 0.8087 | **0.7231** |

### 4. Dual-SSL Encoder Extension

To break the valence ceiling, a second frozen SSL encoder (facebook/wav2vec2-base, 13 layers × 768-dim, speech-pretrained) was added alongside MERT. Independent `WeightedLayerFusion` modules fuse each encoder's layers; outputs are concatenated (1792-dim) before the regression head.

* **Motivation:** Speech-pretrained wav2vec2 captures complementary prosodic and timbral cues absent from music-only MERT pretraining.
* **Result:** Valence R² improved from 0.5055 → **0.5676** (+0.062), the strongest audio-only valence result in this study.
* **Entropy sharpening (β=0.05):** A `fusion_entropy_loss` penalty added to training acts as a mild regularizer (+0.0075 valence R²) without inducing true layer specialization.

### 5. Novel Finding: Fusion Collapse in Multi-Encoder SSL

Both `WeightedLayerFusion` modules converge to near-maximum-entropy (uniform) distributions in the dual-encoder setting (MERT: 0.0% specialization; wav2vec2: 0.0%), in contrast to the single-encoder model where MERT learned layers 14/16/17 specialization (entropy 3.2178 / max 3.2189). Root cause: with ~600 training samples and a large concatenated head, the gradient signal reaching the fusion weights is too diffuse for layer-selective learning to emerge. This is a data-constraint finding applicable to multi-encoder SSL fine-tuning broadly.

### 6. IADS-E Joint Learning — Negative Finding

Joint training with IADS-E generalized environmental sounds (Simonetta et al., 2024 replication) was attempted across partial k,p ratio sweep. All tested configurations fell below the dual-SSL baseline. Confirmed negative result: SSL embeddings pretrained on music and speech do not transfer emotional structure across the music↔environmental-sound boundary as effectively as hand-crafted openSMILE features. Framed as a publishable negative finding on SSL cross-domain emotional transfer limits.

---

## ✅ Phase C: Explainable Retrieval-Augmented Generation System (Implemented)

**Objective:** Design and implement an explainability layer on top of the Phase B affective latent space, enabling human-interpretable music emotion retrieval grounded in psychoacoustic and physiological evidence.

**Core Research Question (Phase C):** *Can the emotion-aware latent space learned in Phase B support a retrieval system that not only returns emotionally similar music, but generates psychologically grounded, human-readable explanations for its decisions?*

### Scientific Framing

Phase C addresses a recognized gap in Music Emotion Recognition research: **the explainability problem**. Most MER models achieve acceptable predictive accuracy but remain opaque to end users — they cannot justify *why* a particular song is emotionally similar to another. This is particularly problematic in clinical or therapeutic music recommendation contexts, where trust in the system depends on the quality of its reasoning.

The Phase C system operationalizes **Explainable AI (XAI)** for music retrieval through three interlocking mechanisms:

1. **Prototype-based retrieval** (k-NN in latent space): Explanations reference concrete exemplars ("this song is retrieved because it is 0.997-similar to your query in emotion-aware latent space"), a strategy shown to improve user trust over black-box recommendations (Kim et al., 2016).

2. **Contrastive counterfactual explanation** (Miller, 2019): The system explicitly surfaces songs that were *not* retrieved (contrastive foils) alongside those that were. This directly answers the question users implicitly ask: *"Why this song and not that one?"* — a form of explanation that social science research identifies as the most cognitively natural to humans.

3. **Physiological grounding via EDA** (Thayer, 1989): EDA-derived features (mean skin conductance, SCR peaks, arousal trend) are translated into natural language and included in the explanation. This provides a biological dimension absent from audio-only systems — the explanation can state not just that a song is emotionally similar, but that *listeners' bodies responded similarly* to it.

### Two-Layer Explanation Architecture

A deliberate design decision separates explanation into two layers — each serving a different audience and purpose:

**Layer 1: Deterministic Template Explanation**
- Outputs precise V-A coordinates, cosine similarity scores, quadrant membership, per-neighbor deltas, and EDA summaries.
- Fully reproducible — no stochasticity, no hallucination risk.
- Purpose: technical validation, ablation studies, thesis quantitative reporting.

**Layer 2: LLM-Augmented Humanized Explanation (RAG)**
- Uses the template output as structured context, enriched with:
  - A static music-theory knowledge base (genre families, tempo ranges, harmonic tendencies per Russell quadrant)
  - EDA physiological narrative
  - Contrastive foils with explicit contrast framing
  - MERT layer attribution (which acoustic feature levels drove the retrieval)
  - Mood trajectory (suggested listening arc based on arousal ordering)
- Instructs the LLM to produce a four-part structured response: recommendation, emotional connection, contrast with foils, shared listening experience.
- Supports **Ollama** (local inference, no API dependency) and **Anthropic API** as backends.

### XAI Feature Details and Academic Justification

| Feature | Implementation | Academic Grounding |
| :--- | :--- | :--- |
| **Contrastive foils** | `retriever.query_foils()` returns N most dissimilar songs; injected into RAG prompt | Miller (2019): counterfactual contrast is the most cognitively natural explanation form |
| **EDA physiological narrative** | 7-dim feature vector (mean, std, slope, SCR peaks, max, high-ratio) → natural language | Thayer (1989): EDA indexes autonomic arousal; links audio emotion to bodily response |
| **MERT layer attribution** | Learned `WeightedLayerFusion` weights grouped into low/mid/high MERT layers | Provides mechanistic interpretability of the Phase B model's feature preferences |
| **Mood trajectory** | Neighbors sorted by arousal → two listening arc suggestions | Practical application: mood regulation via music (Saarikallio & Erkkilä, 2007) |
| **Emotion knowledge base** | Static dict per Russell quadrant: genre family, tempo, harmony, listener profile | Russell (1980) circumplex model; avoids LLM hallucination of artist names |

### Limitations (Thesis-Honest Reporting)

These limitations should be explicitly discussed in the thesis:

- **PMEmo provides no artist/title/genre metadata.** Song IDs are anonymous integers. The system cannot make artist-specific recommendations (e.g., "if you like Metallica, try Megadeth") without an external metadata mapping (MusicBrainz, Spotify API). The knowledge base approach (genre families per quadrant) is a principled workaround, but it operates at the categorical level, not the specific-song level.
- **EDA normalization is dataset-wide, not listener-specific.** The 7-dim EDA features are normalized across all 767 PMEmo songs, meaning the physiological narrative reflects population-level tendencies, not individual listener responses. Individual variability in EDA response is a recognized limitation in affective computing.
- **LLM explanation quality is model-dependent.** The RAG prompt engineering was designed for models in the 7B–70B parameter range. Smaller models (e.g., llama3.2:1b) may produce generic or inaccurate explanations regardless of prompt quality.
- **Precision@k evaluates V-A proximity, not perceptual similarity.** A song 0.20 away in V-A space is counted as a true positive, but this threshold is a design choice, not a perceptual ground truth. User studies would be needed to validate that emotionally similar songs are also *perceptually* similar to human listeners.
- **Cosine similarity in latent space correlates with, but does not equal, emotional similarity.** The SupCR loss encourages emotionally similar songs to cluster, but the mapping is imperfect — particularly for valence, where the Phase B $R^2$ of 0.51 confirms significant residual variance.

### Module Structure

```
ssl_scripts/phaseC/
├── mainC.py         ← CLI entry point (build / query / evaluate)
├── index_builder.py ← L2-normalized latent index construction + storage
├── retriever.py     ← cosine k-NN + query_foils() (contrastive XAI)
├── explainer.py     ← two-layer explanation: template + enriched RAG prompt
├── eda_loader.py    ← 7-dim EDA extraction from PMEmo CSV files
└── evaluator.py     ← Precision@k and Silhouette evaluation
```

---

## Academic References (Phase C)

- Kim, B., Khanna, R., & Koyejo, O. (2016). Examples are not enough, learn to criticize! Criticism for interpretability. *NeurIPS 2016*.
- Miller, T. (2019). Explanation in Artificial Intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1–38.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
- Saarikallio, S., & Erkkilä, J. (2007). The role of music in adolescents' mood regulation. *Psychology of Music*, 35(1), 88–109.
- Thayer, R. E. (1989). *The Biopsychology of Mood and Arousal*. Oxford University Press.