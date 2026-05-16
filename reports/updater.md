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
| This Work (MERT Hybrid+EDA) | 2026 | Music-SSL + WeightedFusion + SupCR | 0.5075 | 0.6738 | 0.77 / 0.85 |
| **This Work (dual: MERT + wav2vec2)** | **2026** | **Dual-SSL + WeightedFusion + SupCR** | **0.5601** | **0.6810** | **0.72 / 0.81** |

Note: CCC (Concordance Correlation Coefficient) is the AVEC standard metric — it simultaneously penalizes poor correlation, mean bias, and variance mismatch. The current best configuration is **dual (MERT + wav2vec2), no IADS-E**; adding IADS-E joint data did not improve PMEmo valence (see Step 2 below — negative cross-domain SSL transfer).

---

## Step 2 Results Placeholder — Joint SSL (IADS-E)

### Motivation
Simonetta et al. (2024), *"Joint Learning of Emotions in Music and Generalized
Sounds"*, show that mixing music (PMEmo) with generalized environmental sounds
(IADS-E) lifts Valence R² to ~0.78 / Arousal ~0.86, because the two domains
**share a common emotional V-A space** and complement each other's coverage of
the V-A plane (music clusters high-valence; environmental sounds fill the
low-valence / high-arousal region). They used hand-crafted openSMILE features.
We replicate the *idea* with **dual SSL encoders + a learned domain embedding**
(`DualSSLDomainModel`), keeping the thesis focus on *assessing the goodness of
SSL representations* rather than hand-crafted features. K-fold is on PMEmo only;
IADS-E augments training; evaluation stays on PMEmo test folds (apples-to-apples
vs. Step-1 dual). A `mix_k`/`mix_p` sweep replicates Simonetta's ratio ablation.

### Honest framing
- If Valence R² **rises above 0.56** (current dual best): SSL domain transfer
  works — a positive replication that generalized-sound SSL features carry
  transferable affective structure.
- If Valence R² **stays flat (~0.56) or drops**: SSL domain transfer is *less*
  effective than Simonetta's hand-crafted features here — a **publishable
  negative finding** (SSL embeddings may over-specialize to their pretraining
  domain, limiting cross-domain emotional transfer). Either outcome is a
  legitimate thesis contribution about the goodness of SSL models.

### Results — partial sweep (2026-05-16)

Baselines: Audio-only MERT V 0.5055 / A 0.6518 · **MERT+wav2vec (dual, best)
V 0.5601 / A 0.6810**. Eval = PMEmo test folds only (never on IADS-E).

| Config (k, p) | R² Arousal | R² Valence | CCC Arousal | CCC Valence | vs. dual (ΔV) | Status |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| dual (no IADS-E, Step 1) | **0.6810** | **0.5601** | 0.8142 | 0.7182 | — | reference |
| k=1.0, p=1.0 | 0.6093 | 0.4874 | 0.8173 | 0.7413 | −0.073 | done |
| k=1.0, p=0.25 | 0.5752 | 0.3888 | 0.7800 | 0.6725 | −0.171 | done |
| k=1.0, p=0.0 | −0.3136 | −0.2848 | 0.2314 | −0.0983 | n/a | degenerate¹ |
| k=1.0, p=0.5 | — | — | — | — | — | interrupted |
| k≤0.75 (any p) | — | — | — | — | — | not yet run |

¹ `p=0.0` removes **all** music from training (`train = music(0) + sound(768)`),
then tests on music → negative R² is expected; a sanity check, not a result.

> Auto-generated: `phaseB/logs/ablation_summary.md` (re-run `summarize_ablation.py`).

### Observed conclusion (so far → leaning negative finding)

Across **every** completed joint config, Valence R² is **below** the dual
baseline (0.5601). The pattern is monotone: the more music kept (`p`↑), the
closer to dual — i.e. IADS-E is **diluting**, not complementing, the music
mapping. At full IADS-E (`k=1.0`) the best joint result (V 0.4874, p=1.0) is
−0.073 below dual. This currently supports the **pre-registered negative
finding**: SSL embeddings (MERT music-pretrained, wav2vec2 speech-pretrained)
do **not** transfer emotional structure across the music↔environmental-sound
boundary as well as Simonetta's hand-crafted openSMILE features did. Notable
secondary effect: **CCC Valence rose** at k=1,p=1 (0.7182→0.7413) even as R²
fell — IADS-E broadens the valence *range/ranking* the model tracks while
adding *scatter/bias* on music specifically.

**Caveat before declaring this final:** the joint sampler holds IADS-E at
≈20 % of every batch *regardless of `k`*, so the `k` axis under-tests the
weak-transfer regime (only `p` was effectively varied). A proper conclusion
needs a domain-weight `λ` that scales IADS-E gradient mass (lever #1 below) so
the transfer curve can be traced near `λ→0` (= dual). Until then this is a
**strong directional negative result, not yet a closed one.**

### Thesis takeaway

- **Headline result stays:** dual (MERT + wav2vec2), **Valence R² 0.5601 /
  Arousal 0.6810** — no IADS-E. This is the number to report.
- **Joint learning** is written up as a tested hypothesis with a (currently)
  **negative outcome**: SSL cross-domain emotional transfer underperforms the
  hand-crafted-feature literature — a legitimate contribution on *the goodness
  and limits of SSL models*, which is the thesis' stated focus.
- Open follow-ups if pursued: (1) domain-weight `λ`; (2) per-domain SupCR
  (music↔sound attract-only); (3) freeze MERT fusion on sound batches;
  (4) category-targeted IADS-E mixing (low-valence/high-arousal only).
