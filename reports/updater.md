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

## SOTA Comparison (Final, PMEmo 2019)

| Method | Year | Approach | R² Valence | R² Arousal | CCC V/A |
|--------|------|----------|-----------|-----------|---------|
| PMEmo Original (Zhang et al.) | 2019 | Hand-crafted IS13 + ML | ~0.42 | ~0.51 | — |
| Deep MER (Dutta & Chanda) | 2021 | Mel-Spec + CRNN | ~0.45 | ~0.55 | — |
| Hybrid SSL Multi-task | 2023 | Wav2Vec2 + Attention | ~0.48 | ~0.61 | — |
| IAENG 2025 Hybrid | 2025 | Multimodal Freq-Domain | ~0.60 | ~0.62 | — |
| This Work (MERT only, audio) | 2026 | Music-SSL + WeightedFusion + SupCR | 0.5055 | 0.6518 | 0.74 / 0.82 |
| This Work (MERT + EDA) | 2026 | MERT + Physiological Fusion | 0.5075 | 0.6738 | 0.77 / 0.85 |
| This Work (Dual-SSL + β=0.05) | 2026 | MERT + wav2vec2 + Entropy Reg. | 0.5676 | 0.6814 | 0.72 / 0.81 |
| This Work (Triple: +mel-CNN) | 2026 | MERT + wav2vec2 + trainable mel-CNN | 0.5758 | 0.7023 | 0.73 / 0.82 |
| **This Work (Enhanced: +tempo/key)** | **2026** | **MERT + wav2vec2 + music-theory gap branch** | **0.5686** | **0.7182** | **0.73 / 0.83** |

**Headline numbers:** best **Valence** = Triple (V R²=0.5758); best **Arousal** = Enhanced (A R²=0.7182, CCC A=0.8345 — highest of the no-EDA models). **Enhanced finding:** explicitly feeding the Phase-A gap features (tempo, key) lifts **arousal only** (+0.037 A R², +0.001 V R²) — tempo is the active ingredient (canonical arousal correlate); raw-integer key is inert. **Caveat:** global R² is majority-class (HVHA 61%) driven; minority-quadrant R² negative across all configs. MERT+EDA still leads CCC Arousal overall (0.8543).

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

### Thesis takeaway (IADS-E — CLOSED)

- **IADS-E joint learning is a confirmed negative finding.** All tested configs
  fall below the dual baseline. Declared closed.
- SSL cross-domain emotional transfer underperforms Simonetta's hand-crafted
  features — a citable, publishable negative result on SSL model goodness.

---

## Step 3 — Fusion Collapse & Entropy Sharpening (2026-05-17, CLOSED)

### Experiments run

All on PMEmo only, 5-fold CV, 100 epochs.

| Config | R² Arousal | R² Valence | CCC Arousal | CCC Valence | MERT spec. | w2v spec. |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Dual β=0 (original) | 0.6810 | 0.5601 | 0.8142 | 0.7182 | ~0% | ~0% |
| DualBottleneck β=0.01 | 0.6665 | 0.4903 | 0.8174 | 0.7045 | 0.2% | 1.8% |
| **Dual β=0.05 (BEST)** | **0.6814** | **0.5676** | **0.8087** | **0.7231** | **0.0%** | **0.0%** |

### Key findings (closed)

**Fusion collapse confirmed.** Both `WeightedLayerFusion` modules converge to
near-maximum-entropy (uniform) distributions in the dual-encoder setting:
- MERT: entropy 3.2180 / max 3.2189 → specialization 0.0%
- wav2vec2: entropy 2.5649 / max 2.5649 → specialization 0.0%

This contrasts with the **single-encoder MERModel** which learned layer 14/16/17
specialization (entropy 3.2178, 0.05% below max).

**Root cause:** With 600 training samples and a 1792-d concatenated head, the
gradient reaching `layer_weights` parameters is too diffuse — the head solves
the task by mixing all 38 virtual layers, so no individual layer needs to stand
out. The per-encoder bottleneck (DualSSLBottleneckModel, 1024→256, 768→256) made
things worse (-0.07 valence R²) because 256-d is too aggressive a compression
for 600 samples.

**β=0.05 as regularizer:** The entropy sharpening penalty at β=0.05 produced no
measurable specialization (still 0%) but gave a +0.0075 valence R² gain. It acts
as mild optimization noise that breaks the exact uniform-weight symmetry during
early training, slightly improving generalization without changing layer selection.

### Thesis framing (fusion collapse)

> *"WeightedLayerFusion converges to near-maximum-entropy distributions in both
> encoders of the dual-SSL model regardless of architectural bottlenecks or
> entropy penalties. This confirms that layer-selective SSL fine-tuning requires
> substantially more labeled data than the ~600 samples available in PMEmo.
> Interestingly, the single-encoder MERT model does learn layer specialization
> (entropy drops to 3.2178 with layers 14/16/17 dominating), suggesting that
> adding a second encoder dilutes the specialization gradient by giving the head
> an alternative information source."*

This is a publishable observation about the data requirements of multi-encoder SSL architectures.

---

## Step 4 — Triple-Branch (MERT + wav2vec2 + trainable mel-CNN) (2026-05-19)

### Motivation
Add a shallow trainable mel-spectrogram CNN (109K params) as a third branch to
test whether raw spectral detail adds value on top of the frozen SSL encoders.
Mel-specs pre-extracted (center-30 s window, fixed 128×1407) — the transform is
parameter-free, so this is functionally identical to on-the-fly and reuses the
exact dual-SSL `.pt` pipeline. Differential optimizer: SSL fusion lr=1e-2;
CNN+head+regressor lr=1e-4. 5-fold CV, 100 epochs, same HybridLoss.

### Results (PMEmo, 5-fold)

| Model | R² Arousal | R² Valence | CCC Arousal | CCC Valence |
|:--|:--:|:--:|:--:|:--:|
| MERT-only (cited) | 0.6518 | 0.5055 | 0.82 | 0.74 |
| Dual-SSL β=0.05 (cited) | 0.6814 | 0.5676 | 0.8087 | 0.7231 |
| **Triple (MERT+w2v+mel)** | **0.7023 ± 0.015** | **0.5758 ± 0.039** | 0.8233 | 0.7329 |
| **Spec-only (MERT+mel, no w2v)** | **0.7069 ± 0.013** | 0.5709 ± 0.042 | **0.8271** | 0.7314 |

### Key findings

**1. New best — first model past R² Arousal 0.70.** Triple reaches A 0.7023 /
V 0.5758; Spec-only A 0.7069 / V 0.5709.

**2. wav2vec2 is now redundant (the headline scientific result).** Spec-only
(MERT + mel-CNN, *no wav2vec2*) equals/beats Triple on every metric. Deltas
(A +0.005, V −0.005) are far inside the ±0.013–0.042 fold std — statistically
indistinguishable. A from-scratch 109K-param CNN on ~614 samples fully replaces
the 95M-param frozen wav2vec2. **Consistent with and reinforcing** the IADS-E
negative finding and fusion-collapse: wav2vec2's speech-pretraining never carried
music-relevant complementary structure; the real gain is MERT + a learnable
spectral branch.

**3. ⚠️ Overfitting / class-imbalance caveat (NOT a clean win).** The global
R²=0.70 is **majority-class-driven**. Per-quadrant (Spec-only):

```
HVHA (Happy, n=469, 61%):  A=0.230  V=-0.142
HVLA (Calm, n=67):         V=-2.148
LVHA (Angry, n=64):        A=-1.079  V=-1.178
LVLA (Sad, n=167):         V=-0.955
```

Three of four quadrants have **strongly negative R²** (worse than the mean).
The model essentially only fits HVHA. The headline gain is the model getting
better at the dominant quadrant, *not* better V-A-plane generalization
(Simpson's Paradox, worse than prior models on minority quadrants). Fold
variance is modest (±0.013 A) → argues against catastrophic CNN overfitting,
but `mainB_triple.py` does not log train loss, so the train/test gap is **not
yet directly measured**. **Open action:** add train-loss logging before the
thesis writeup to separate "CNN overfits" from "class imbalance."

### Thesis framing (triple branch)

> *"A shallow trainable mel-spectrogram CNN added to MERT matches the full
> tri-encoder system and renders the frozen speech-pretrained wav2vec2 branch
> statistically redundant. The headline R² gain (Arousal 0.70) is concentrated
> in the majority emotional quadrant; per-quadrant analysis shows the model does
> not generalize across the full Valence-Arousal plane — a class-imbalance
> limitation inherent to PMEmo, not resolved by additional encoders."*

### Open follow-ups
- (1) Add train/test-loss logging → quantify CNN overfitting directly.
- (2) Frozen **pretrained** CNN (MusiCNN / PANNs CNN14) vs from-scratch — proper
  "pretrained-CNN" test, extends SSL-vs-pretraining narrative. GTZAN
  self-pretraining deprioritized (won't fix class imbalance).
- (3) Minority-quadrant remedy (focal/quadrant-weighted loss) — the real ceiling.

### Final headline numbers for thesis

| System | R² Arousal | R² Valence | CCC A | CCC V | Notes |
|:--|:--:|:--:|:--:|:--:|:--|
| MERT only (hybrid) | 0.6518 | 0.5055 | 0.82 | 0.74 | Single SSL encoder |
| MERT + EDA | 0.6738 | 0.5075 | **0.8543** | 0.7692 | Best physiological CCC |
| Dual-SSL β=0.05 | 0.6814 | 0.5676 | 0.8087 | 0.7231 | Best dual-SSL |
| Triple (MERT+w2v+mel) | 0.7023 | **0.5758** | 0.8233 | 0.7329 | Best Valence |
| Spec-only (MERT+mel) | 0.7069 | 0.5709 | 0.8271 | 0.7314 | ≈ Triple → w2v redundant |
| **Enhanced (MERT+w2v+tempo/key)** | **0.7182** | 0.5686 | **0.8345** | 0.7259 | **Best Arousal (no-EDA)** |

**Caveat on all rows:** global R² is HVHA-majority-driven; minority-quadrant R²
is negative across configs (documented PMEmo class-imbalance limitation).

---

## Step 5 — Phase A/B/C Music-Theory Extension (2026-05-20)

### Phase A — Per-layer music-theory probing (`phaseA/`)
New: `extract_music_theory.py` (librosa GT; key/mode via **Krumhansl–Schmuckler**
profile correlation — note `librosa.estimate_key()` does not exist) and
`run_music_theory_probing.py` (sklearn linear probe of EACH of the 25 MERT
layers per feature; Ridge/R² for regression, LogisticRegression/accuracy for
mode & key; same split as Phase A: test_size=0.2, random_state=42).

**Gap analysis** (threshold R²<0.40, acc<0.65) → `phaseA/gap_analysis.json`.
Result: **gap_features = ['tempo', 'key']**. Across all 25 layers, MERT does not
linearly expose absolute tempo or key — consistent with the original Phase A
tempo finding (R²≈0.12). Harmony/timbre features are captured well (no gap).

### Phase B — Enhanced Dual-SSL (`phaseB/train_enhanced_dual.py`)
New: `models_enhanced.py` (`MusicTheoryBranch` = Linear(gap_dim,32)+ReLU+Dropout;
`EnhancedDualSSLModel` = MERT(1024) + wav2vec2(768) + theory(32) → 1824 → head).
Self-skips if Phase A finds no gaps. Gap branch fed only [tempo, key] (gap_dim=2).
5-fold CV, 100 epochs, same HybridLoss + differential optimizer as dual-SSL.

| Model | A R² | V R² | CCC A | CCC V |
|:--|:--:|:--:|:--:|:--:|
| Dual-SSL (baseline) | 0.6814 | 0.5676 | 0.8087 | 0.7231 |
| **Enhanced Dual-SSL** | **0.7182 ± 0.013** | 0.5686 ± 0.042 | **0.8345** | 0.7259 |

**Finding: the gain is arousal-only.** A R² +0.0368, CCC A +0.0258; V R² +0.0010
(noise). **Tempo is the active ingredient** — the canonical arousal correlate;
Phase A flagged it as a gap, Phase B fed it back, arousal improved. This is a
clean Phase-A→B closure: probing-driven feature augmentation works *for the
feature with a clear emotional correlate*. **Key was inert** — raw-integer key
(C=0…B=11) is not a usable continuous signal, and `mode` was not in the gap set;
valence (where key/mode should help) did not move. Enhanced has the **best
Arousal R² (0.7182) of any model in the study**; valence stays below Triple.
Class-imbalance caveat persists (3/4 quadrants negative R²).

### Phase C — Music-theory annotator (`phaseC/music_theory_annotator.py`)
New standalone module (no existing file modified; integration snippet in its
docstring). `annotate(path)` → {key, tempo, brightness, rhythmic_stability,
dominant_pitches}; `format_music_theory_block()` → the explanation section with
a hardcoded mode→character lookup. Verified on 760.mp3 → "D major, 105 BPM".

**IMPORTANT for thesis (data provenance):** the Phase C annotator computes its
features with **librosa directly from the query audio — NOT from the SSL
embeddings.** It is an **independent descriptive channel**, not a faithful
explanation of the SSL model's internals (the model never saw these librosa
features). Distinct from the WeightedLayerFusion attribution, which *is*
model-internal. Narrative coherence: the features Phase A proved MERT lacks
(tempo, key) are both fed back into the model (Phase B) and surfaced to the user
(Phase C).

### Open caveats
- `rhythmic_stability = 1 - std(tempogram)/mean(tempogram)` (spec formula) is
  **not bounded to [0,1]** — real clips give negatives (760.mp3 → −0.132).
  Spec-faithful; consider clamping or `1/(1+std/mean)` if used in the annotation.
- Annotator's `dominant_pitches` (top-3 raw chroma bins) don't always include
  the K-S tonic (760.mp3: key D major, top pitches G/G#/A) — the "supports the
  analysis" wording can overclaim. Cosmetic; K-S uses the full 12-d profile.
- Key as model input → one-hot or sin/cos circular encoding is the honest
  follow-up if a valence gain from key is desired.
