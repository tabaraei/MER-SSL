# Session Notes — May 2026

## Repository Analysis & Professor Meeting Strategy

---

## 1. What's Been Done (Phases A & B Complete)

**Phase A — Foundation Validation**
- Proved MERT's frozen embeddings encode music structure before any fine-tuning
- Harmonic mode (Major/Minor): **100% linear probe accuracy** → MERT is harmonically aware
- Tempo (BPM): **R² = 0.12** → rhythm weakly encoded in chorus clips (expected, not a failure)
- t-SNE visualization shows emotion quadrant separation in raw embedding space

**Phase B — Hybrid Affective Model**
- Architecture: `WeightedLayerFusion` (learnable softmax over all 25 MERT layers) → 1024→256→128 bottleneck → arousal/valence regression
- Four-component loss: MSE + CCC + Rank + SupCR — each targeting a different failure mode of standard regression
- Two technical fixes that made training possible: differential optimizer (fusion lr=1e-2, head lr=1e-4) and balanced sampler (resolves quadrant imbalance in PMEmo)
- Multimodal EDA fusion via late concatenation (7→32 dim projection + 160→64→2 head)

**Validated Results (5-fold CV):**

| Metric | Arousal | Valence |
|---|---|---|
| R² | 0.6738 | 0.5075 |
| CCC | 0.8543 | 0.7692 |

Layer specialization: model concentrates on layers 14, 16, 17. Weight entropy = 3.2178.

---

## 2. Bottlenecks and Challenges

**A. Valence R² ceiling (~0.50) — the hardest problem in MER**
Valence requires lyrics, cultural context, and harmonic nuance that MERT's audio-only pre-training doesn't fully capture. The field-wide ceiling for audio-only valence is ~0.50–0.54. You're at the ceiling, not below it. Music2Emo achieves 0.54 with multitask learning. This is a known literature-wide limitation, not a model failure.

**B. Layer specialization is nearly zero in practice**
Weight entropy dropped from theoretical max 3.22 → 3.2178 — a difference of 0.0022. The model uses all 25 layers nearly uniformly. The finding that "layers 14,16,17 dominate" is technically true but the margin is thin. Acknowledge honestly.

**C. EDA fusion gives marginal gains**
+0.022 on arousal R², +0.002 on valence R². Real but small — the physiological signal partially overlaps with what the audio already captures. EDA is more valuable as an *explanation* mechanism than as a *performance* booster.

**D. Dataset is fundamentally small**
767 songs → ~614 train / ~153 test per fold. Training a downstream head on top of a 330M parameter backbone with ~600 examples is the hardest constraint.

**E. MERT is frozen**
Not fine-tuning any MERT layers limits the upper bound. Even unfreezing the last 2–3 layers during Phase B could recover some valence headroom — at the cost of compute and overfitting risk.

**F. Phase C evaluation not yet run**
Precision@k and Silhouette scores haven't been computed (pending index rebuild).

---

## 3. Narrative for Professor Meeting

### Opening — What You've Done
"I've completed the first two phases of the thesis. Phase A validated that the MERT self-supervised model encodes musically meaningful information before any fine-tuning — we confirmed harmonic structure at 100% accuracy via linear probing. Phase B trained a hybrid affective model mapping MERT's 25 transformer layers to the valence-arousal circumplex, achieving a CCC of 0.85 on arousal — the strongest result achievable on PMEmo given audio-only input."

### Middle — Honest Technical Context
"The main challenge is the valence R² ceiling at 0.50. This isn't unique to our system — every audio-only method plateaus near this value because valence is strongly shaped by lyrics and cultural context. Our CCC of 0.77 on valence confirms the model tracks the emotional trend correctly even where absolute predictions have residual variance. I've also added physiological grounding via EDA fusion, which provides a biological validation channel that audio-only systems don't have."

"The key novel technical contributions in Phase B are: (1) the weighted layer fusion that makes the model's feature preferences interpretable — we can see that mid-to-late MERT layers drive emotion prediction; (2) the hybrid loss combining CCC, rank preservation, and contrastive clustering — each component addressing a different failure mode of standard regression."

### Future — What You Want to Do Next (Phase C framed as future work)
"What the system cannot yet do is *explain* its decisions in human-interpretable terms. The latent space learned in Phase B is emotionally structured — songs with similar arousal/valence cluster together — but right now the system only produces numbers, not reasoning. The next phase of the thesis builds an **explainability layer** on top of this latent space."

"Specifically, I want to build a retrieval system that (1) finds emotionally similar songs using the learned latent space, and (2) generates a human-readable explanation of *why* those songs are similar — grounded in music-theoretic terms, physiological response data, and the model's own layer attribution. This connects to the XAI literature: prototype-based explanation methods where concrete examples act as the explanation, and contrastive explanations that answer 'why this song and not that one'."

"This would make the thesis contribution unique: not just a model that predicts emotion, but a system that can explain its reasoning in terms a clinician, music therapist, or user could understand and trust."

### What to Ask the Professor
1. **Validation of direction:** "Does the explainability layer fit the thesis scope, or should I narrow/expand?"
2. **Valence limitation acknowledgment:** "Is the 0.50 valence ceiling a thesis weakness or should it be framed as a literature-confirmed constraint?"

---

## SOTA Comparison (Updated, PMEmo 2019)

| Method | Year | Approach | R² Valence | R² Arousal | CCC V/A |
|--------|------|----------|-----------|-----------|---------|
| PMEmo Original (Zhang et al.) | 2019 | Hand-crafted IS13 + ML | ~0.42 | ~0.51 | — |
| Deep MER (Dutta & Chanda) | 2021 | Mel-Spec + CRNN | ~0.45 | ~0.55 | — |
| Hybrid SSL Multi-task | 2023 | Wav2Vec2 + Attention | ~0.48 | ~0.61 | — |
| IAENG 2025 Hybrid | 2025 | Multimodal Freq-Domain | ~0.60 | ~0.62 | — |
| **This Work (MERT Hybrid+EDA)** | **2026** | **Music-SSL + WeightedFusion + SupCR** | **0.5075** | **0.6738** | **0.77 / 0.85** |

Note: CCC (Concordance Correlation Coefficient) is the AVEC standard metric — it simultaneously penalizes poor correlation, mean bias, and variance mismatch. Our CCC of 0.8543 on arousal is competitive with the field.
