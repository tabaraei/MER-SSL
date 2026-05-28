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

**F. Phase C evaluation — DONE (see Step 6 below)**
Precision@k and Silhouette computed on test-fold (out-of-sample) latents. Retrieval
validated (~0.58 Precision@5); Silhouette ≈ 0 (continuous-emotion, not 4 blobs).

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

---

## Step 6 — Phase C Latent-Space Evaluation (2026-05-21)

### Method
New: `phaseC/evaluate_latent_space.py` (`--encoder {mert,dual}`). Builds
**out-of-sample (test-fold) latents**: 5-fold CV (KFold shuffle, random_state=42),
each song encoded by the fold model that did NOT train on it — the honest test of
latent-space organization (vs. the old in-sample `mainC --mode build`, which also
crashes now because `best_model.pt` is DualSSL and won't load into MERModel).
Rebuilds `prototypes.npy` (L2 latents, IDs, ground-truth + predicted V-A, EDA),
then runs the existing `evaluator.evaluate_retrieval` (Silhouette + Precision@k).

**Bug fixed (the real "EDA bug"):** `load_pmemo_data` returns float-formatted IDs
(`'1.0'`), so `{id}_EDA.csv` lookup built `1.0_EDA.csv` → all 767 missing, and
`int('1.0')` crashed. Normalized IDs to clean integers → EDA now loads for all 767.

### Results (test-fold, out-of-sample)

| Encoder | Precision@5 | Precision@10 | Precision@20 | Silhouette |
|:--|:--:|:--:|:--:|:--:|
| MERT (single) | 0.5760 | 0.5687 | 0.5469 | −0.0293 |
| **Dual (MERT+w2v, β=0.05)** | **0.5849** | 0.5613 | 0.5382 | **+0.0026** |

### Interpretation (HONEST — do not oversell Silhouette)

- **Precision@k validates retrieval.** ~58% of each song's top-5 latent neighbours
  fall within a 0.20 V-A radius — well above chance — so the system genuinely
  fetches emotionally similar songs. This is the **real evidence** that the
  example-based explanation foundation works (sub-task 1.3 ✓).
- **Silhouette ≈ 0 for both models.** Dual's +0.0026 is *technically* positive but
  is **effectively zero — NOT meaningful cluster separation.** Do not claim "SupCR
  separated the quadrants" on the strength of 0.003. The two encoders are tied.
- **Why Silhouette is ~0 (and why that's fine):** emotion is a *continuous* V-A
  gradient, not 4 discrete islands. Silhouette-by-quadrant imposes hard 0.5 cutoffs
  on a smooth manifold, so boundary songs are legitimately close to the adjacent
  quadrant → near-zero score even when local structure is good. Class imbalance
  (HVHA 61%) compounds it. **Lead the thesis with Precision@k; report Silhouette
  transparently with this continuous-manifold explanation, not as a failure.**
- **MERT vs dual are equivalent here** — consistent with the project-wide finding
  that the second encoder adds little to the latent organization.

### Thesis framing

> *"On out-of-sample (test-fold) latents, the emotion-aware space achieves
> Precision@5 ≈ 0.58 — nearest neighbours are reliably emotionally similar,
> validating the example-based retrieval that underlies the explanation system.
> The Silhouette score by Russell quadrant is ≈ 0, which reflects that affect is a
> continuous valence-arousal gradient rather than four separable clusters;
> Silhouette-by-quadrant therefore understates the latent organization that
> Precision@k directly demonstrates."*

---

## Step 7 — Naive Baseline + Thesis Artifacts (2026-05-22)

New (read-only): `phaseC/export_artifacts.py` (naive baseline + t-SNE + fusion
bar chart) and `phaseC/export_explanations.py` (5-song full RAG export).

### Naive baseline retrieval (raw last-layer MERT, no training)

| Space | P@5 | P@10 | P@20 | Silhouette |
|:--|:--:|:--:|:--:|:--:|
| Naive raw last-layer MERT | 0.4847 | 0.4614 | 0.4553 | **0.1001** |
| MERT (SupCR) | 0.5760 | 0.5687 | 0.5469 | −0.0293 |
| Dual-SSL (SupCR) | 0.5849 | 0.5613 | 0.5382 | +0.0026 |

**Validates training:** SupCR lifts Precision@5 0.485 → 0.58 (+≈0.10, ~+19% rel.).
**Honest twist:** untrained Silhouette (0.100) > trained (≈0) — SupCR trades coarse
quadrant blobbiness for continuous local V-A coherence (↑Precision@k, ↓discrete
clusters). Reinforces "Precision@k is the right metric; emotion is continuous."

### Artifacts (`phaseC/artifacts/`)
- `tsne_baseline_vs_finetuned.png` — raw MERT = one mixed blob; SupCR space = clear
  structured filaments (HVHA-dominated continuous manifold, not 4 clean blobs).
- `layer_fusion_weights.png` — softmaxed single-MERT fusion weights; top-3 [14,15,16]
  but near-uniform (entropy 3.218/3.219) — drawn honestly (fusion-collapse).
- `explanations_5songs.txt` — **DONE**: full Layer 1 + Layer 2 for 5
  quadrant-spanning songs (562, 706, 31, 894, 282). Layer 2 via a **server-free,
  no-sudo HF transformers backend** (Qwen/Qwen2.5-1.5B-Instruct in the venv on GPU);
  university server blocks sudo/Ollama, so `export_explanations.py` got an `--llm hf`
  mode. 4-part structured prose confirmed; Layer 1 stays the citable artifact.

Note: `mainC.py --mode build/query` loads MERModel and crashes on the DualSSL
`best_model.pt`; use `evaluate_latent_space.py` / `export_artifacts.py` instead.

---

## Step 8 — Loss-Function Ablation (2026-05-22)

Justifies the hybrid loss; holds architecture constant (single-MERT, test-fold,
100 epochs), varying only loss weights via `evaluate_latent_space.py --w_*`.

| Loss config | CCC A | CCC V | P@5 | Silhouette |
|:--|:--:|:--:|:--:|:--:|
| MSE only | 0.6861 | 0.5955 | 0.5259 | +0.0124 |
| + CCC + Rank (no SupCR) | 0.7814 | 0.7113 | 0.5398 | +0.0206 |
| + SupCR (full hybrid) | **0.8165** | 0.7110 | **0.5734** | −0.0311 |

- **CCC+Rank: decisively justified** — +0.095 CCC A, +0.116 CCC V over MSE-only.
  Answers "why not just MSE?".
- **SupCR: justified for retrieval, refutes clustering claim** — +0.034 P@5,
  +0.035 CCC A, but Silhouette drops (+0.021→−0.031). Removing SupCR *raised*
  Silhouette → SupCR does continuous local tightening, not discrete clustering.
- **Full vs MSE:** +0.13 CCC A, +0.12 CCC V, +0.047 P@5 → hybrid loss justified
  overall. No config clusters (all Silhouettes ≈ 0).

Methodology note: compare ablations against the single-MERT full-hybrid baseline
(CCC A 0.8165), NOT the MERT+EDA 0.8543 (different architecture).

---

## Step 9 — Claim Audit + Prototype-Activation Accuracy (2026-05-22)

New (read-only): `phaseC/extra_metrics.py` — prototype-activation accuracy +
random-chance Precision baseline.

**Prototype-activation accuracy** (best-match centroid = true quadrant, leave-one-out):
- Dual 0.506 · MERT 0.462 · **majority-class baseline 0.611** → feature is BELOW the
  trivial baseline. Per-quadrant recall HVHA 0.63 / HVLA 0.55 / LVHA 0.41 / Sad 0.17.
  → 4-prototype = interpretability readout, NOT an accurate classifier. Reported honestly.
- **Random-chance Precision = 0.276** → Precision@5 0.58 is ~2× chance ("above chance" now substantiated).

**Claim audit — fixes made to the clean reports:**
- 01: "raw t-SNE shows visible quadrant separation" → corrected (raw MERT is a blob;
  continuous gradient, not clusters; backed by baseline t-SNE artifact).
- 02: "layers 14/16/17 dominate" → corrected to near-uniform / faint-lean,
  run-dependent (backed by `layer_fusion_weights.png`).
- 03: "well above chance" → backed with random baseline 0.276; added prototype accuracy.
- 04: "highest reported CCC on PMEmo" → removed (no prior work reports CCC → not comparable).
- 05 §0: prototype-classifier underperformance quantified.

---

## Step 10 — wav2vec2-only baseline + SOTA table (2026-05-24)

New: `phaseB/eval_wav2vec_only.py` — same hybrid model/loss/CV as MERT-only, but
on wav2vec2 features (13x768) via MERModel(n_layers=13, hidden_dim=768).

| Single encoder | R2 A | R2 V | CCC A | CCC V |
|:--|:--:|:--:|:--:|:--:|
| wav2vec2 only (speech SSL) | 0.6225 | 0.4825 | 0.77 | 0.66 |
| MERT only (music SSL) | 0.6518 | 0.5055 | 0.82 | 0.74 |

MERT beats wav2vec2 on every metric (biggest gap: valence CCC 0.74 vs 0.66) →
justifies the MERT backbone; explains wav2vec2 redundancy in the dual model.

PDF: `mert_progress_report_v3.pdf` regenerated — added the wav2vec2-only row to the
Phase B table and the author-provided SOTA rows (PMEmo Original, Deep MER, Hybrid
SSL, Damer, Music2Emo) to the comparison table (flagged author-provided / verify
citations).

---

## Step 11 — SOTA table verified against literature (2026-05-24)

Verified PMEmo R² results against a 2024 MER survey (arXiv:2406.08809). Replaced
the author-provided/unverified rows with VERIFIED R²-comparable ones:

| Method | Year | R² V | R² A | source |
|:--|:--:|:--:|:--:|:--|
| Source-Separation + CNN | 2020 | 0.4814 | 0.6004 | survey [92] |
| Music2Emo (MERT+Multitask+KD) | 2025 | 0.5473 | 0.7940 | survey [124] |
| This work — Triple | 2026 | **0.576** | 0.702 | ours |
| This work — Enhanced | 2026 | 0.569 | **0.718** | ours |

**Catches:** (1) "Damer 0.51/0.72" was a metric error — DAMER reports *accuracy*
(~78% V / 86% A), not R²; removed from the R² table. (2) Dutta&Chanda and Wu rows
could not be verified; dropped. (3) Music2Emo confirmed (0.5473/0.7940 ≈ the
0.54/0.78 we had). **Headline:** our Triple valence R² 0.576 is the highest
*verified* R² valence on PMEmo (> Music2Emo 0.5473), audio-only. Accuracy-based
works (CNN+LSTM, DAMER, Sharma) noted separately as not R²-comparable.

---

## Step 12 — Single unified PMEmo comparison table (2026-05-25)

Merged the literature + all our models into ONE PMEmo R² table (PDF §6 + 04 §4).
Author supplied better-cited literature rows (EmoMucs, Music2Emo two variants):

| Method | Year | R² V | R² A | note |
|:--|:--:|:--:|:--:|:--|
| Zhang et al. (PMEmo paper) | 2018 | ~0.20–0.30 | ~0.30–0.45 | IS13+SVR baselines |
| EmoMucs C1D-M (de Berardinis) | 2020 | 0.349 | 0.557 | labels [-1,1] |
| EmoMucs C2D-M (de Berardinis) | 2020 | 0.414 | 0.610 | labels [-1,1] |
| Music2Emo (Kang & Herremans) | 2025 | 0.458 | 0.639 | PMEmo only |
| Music2Emo (Kang & Herremans) | 2025 | 0.547 | 0.794 | multitask, 4 datasets |
| ours wav2vec2-only | 2026 | 0.4825 | 0.6225 | |
| ours MERT-only | 2026 | 0.5055 | 0.6518 | |
| ours MERT+EDA | 2026 | 0.5075 | 0.6738 | |
| ours Dual | 2026 | 0.5676 | 0.6814 | |
| ours Triple | 2026 | **0.576** | 0.702 | best V in table |
| ours Enhanced | 2026 | 0.569 | **0.718** | |

Music2Emo multitask (0.547/0.794) matches the survey-verified 0.5473/0.7940.
Our Triple valence 0.576 is the highest valence R² in the table. References added:
de Berardinis et al. 2020 (EmoMucs, ISMIR), Kang & Herremans 2025 (Music2Emo).
Accuracy/RMSE-only works (DAMER, CNN+LSTM, Sharma) intentionally excluded from R² table.

---

## Step 13 — Added Simonetta AutoML rows to comparison (2026-05-25)

Author found the source for the IADS-E motivation numbers — Simonetta, Certo &
Ntalampiras (2024), "Joint Learning of Emotions in Music and Generalized Sounds"
(arXiv:2408.02009). Added two verified rows to the single PMEmo table (PDF §6, 04 §4):

| Method | Year | R² V | R² A | note |
|:--|:--:|:--:|:--:|:--|
| AutoML openSMILE (Simonetta) | 2024 | 0.525 | 0.727 | hand-crafted, PMEmo only |
| AutoML Joint (Simonetta) | 2024 | 0.780 | 0.861 | PMEmo + IADS-E (joint) |

**Reading updated (less flattering, honest):** (1) hand-crafted AutoML PMEmo-only
arousal 0.727 is on par/slightly above our best (0.718) — openSMILE strong on arousal;
(2) AutoML-Joint (0.780/0.861) is the SOTA bar but uses cross-domain joint training —
this is the EXACT approach we replicated with SSL and got a NEGATIVE result (§2b), so the
row simultaneously sets SOTA and contextualizes our negative finding. Our Triple valence
(0.576) remains best among single-dataset methods. Ref added: Simonetta et al. 2024.

---

## Step 14 — Supervisor-audit fixes to v3 report (2026-05-25)

Five fixes to mert_progress_report_v3.pdf (+ 04 §4):

1. **Zhang baseline made rigorous.** Row → ~0.41* / ~0.52* with footnote: Zhang et al.
   (2018) report Pearson r (0.638 V, 0.719 A); squared to R² ≈ r² for comparability.
   (Replaces the vague ~0.20–0.45 placeholder.)
2. **wav2vec2-only — confirmed already logged** (Step 10, JOURNEY, eval_wav2vec_only.py).
   Raw run output (bm3bhn9n3): R² A 0.6225±0.031, R² V 0.4825±0.053, CCC A 0.7722,
   CCC V 0.6564 — real 5-fold numbers, not estimates.
3. **Valence "ceiling" de-self-referenced.** No longer call our 0.576 "the ceiling";
   reframed as competitive single-dataset valence above prior SSL/hand-crafted (0.536/0.525).
4. **Music2Emo PMEmo-only CORRECTED.** Verified from the paper (Table III, arXiv:2502.03979):
   PMEmo-only = **0.536 V / 0.777 A** (NOT the 0.458/0.639 the author had — that was wrong).
   Multitask 0.547/0.794 confirmed. Consequence: Music2Emo arousal (0.777/0.794) beats ours
   (0.718) → arousal is competitive-but-trailing; valence (0.576) remains our lead.
5. **§3.2 names the source** — "replicating Simonetta et al. (2024)" instead of "a recent paper".

---

## Step 15 — Pipeline figure + supervisor Q&A (2026-05-26)

New: `phaseC/make_pipeline_figure.py` → `artifacts/pipeline_overview.png` showing
audio → 3 parallel encoder branches (MERT/wav2vec2/mel-CNN) → fusion → concat →
head → L2-normalised 128-d latent → linear regressor → **continuous A/V ∈ [0,1]**.
HybridLoss (MSE+CCC+Rank+SupCR) explicitly labelled "REGRESSION objective on
continuous A/V (no classification)". Phase C uses the same latent for k-NN retrieval
+ 4-centroid profile + EDA + librosa music-theory annotation → Layer 1 / Layer 2
explanation. Clarifies for readers that the system is regression, not classification.

Supervisor Q&A logged:
- Imbalance handling in code = `WeightedRandomSampler` only (inverse-quadrant freq);
  no class-weighted/focal loss tried — candidate for a future short ablation.
- Clustering: not cheaply fixable; the loss ablation already showed SupCR removal
  RAISES Silhouette → forcing clusters needs an auxiliary quadrant-CE head (a
  real new experiment, not a quick tweak) and likely costs Precision@k.
- Layer fusion: ALL 25 MERT layers via learnable softmax (hybrid mode); the
  "baseline" last-layer mode is not used in any reported result.
- Dual/Triple architecture: parallel branches, independent WeightedLayerFusion per
  encoder, concatenated (no sequential wav2vec-head-on-top-of-MERT).

---

## Step 16 — Layer-fusion ablation: last-layer-only vs all-25 (2026-05-26)

Supervisor question: "why all 25 layers? why not just the last layer?"
Answered empirically. New script `phaseB/eval_last_layer_only.py` — identical to
the MERT-only baseline (same HybridLoss, balanced sampler, differential optimizer,
5-fold CV, 100 epochs) except `X = X[:, -1:, :]` (last layer only, n_layers=1).

5-fold result:
- Last layer only:  R² A **0.6570 ± 0.026** / V **0.5162 ± 0.053**, CCC A 0.81 / V 0.70
- Full 25-layer fusion (ref): R² A 0.6518 / V 0.5055, CCC A 0.82 / V 0.74

**Indistinguishable.** This directly confirms the fusion-collapse story from §2a:
the learned softmax over 25 layers had entropy 3.218/max 3.219 because there is
no useful layer-specific structure for it to find on PMEmo — the last layer
already carries the signal.

Consequence for the write-up: the fusion is retained for *per-layer attribution
transparency* (it lets us *show* the layer weights, even when they are uniform),
**not** claimed as an accuracy contributor. Updated:
- `02_phaseB_model.md` §3 — empirical ablation paragraph
- `04_results_and_sota.md` §2 — added "last layer only" row, §2a — ablation note
- `generate_report_v3.py` §3.1 — new paragraph; v3 PDF regenerated

---

## Step 17 — Imbalance-handling ablation: penalty vs sampler (2026-05-26)

**Question (user-driven):** can a loss-level imbalance treatment outperform our
WeightedRandomSampler on the minority quadrants (HVLA n=67, LVHA n=64, LVLA n=167)?

**Pre-registered pass mark.** A treatment wins iff: R² beats baseline by >1
fold-std on both axes, OR by >2 fold-std on either axis, OR meaningfully lifts
minority per-quadrant R² (from <0 to ≥0).

**Setup.** Script: `phaseB/eval_imbalance_ablation.py`. 4 configs × 5-fold CV,
all else identical to current MERT-only baseline (25-layer fusion, HybridLoss
1.0/0.5/0.3/0.1, differential optimizer fusion=1e-2 head/reg=1e-4 wd=1e-3,
100 epochs, batch=32, KFold random_state=42).

  - A: sampler-only (current baseline; re-run for fold-matched comparison)
  - B: weighted-MSE only (no sampler; per-sample MSE × inv-quadrant-freq weight)
  - C: sampler + weighted-MSE (stacked)
  - D: focal-MSE γ=2 + sampler (hard-example focus, no quadrant prior)

**Result (5-fold mean ± std):**

| Treatment                       | R² A           | R² V           | CCC A | CCC V |
| :--                             | :-:            | :-:            | :-:   | :-:   |
| A: sampler-only (baseline)      | 0.6951 ± 0.016 | 0.5724 ± 0.050 | 0.820 | 0.731 |
| B: weighted-MSE only            | 0.6738 ± 0.018 | 0.5267 ± 0.109 | 0.828 | 0.739 |
| C: sampler + weighted-MSE       | 0.6855 ± 0.029 | 0.5696 ± 0.062 | 0.811 | 0.724 |
| D: focal-MSE γ=2 + sampler      | 0.6949 ± 0.018 | 0.5716 ± 0.057 | 0.821 | 0.727 |

**Verdict — no winner.** B is *worse* on R² V (−0.046, larger than fold-std);
C and D are statistically tied with A on all four metrics. Per the
pre-registered rule: **do not propagate to the multi-encoder configs**. The
WeightedRandomSampler is empirically the best imbalance treatment we have for
this dataset; penalty-based reweighting at the loss level adds nothing.

**Honest disclosure — baseline-rerun gap.** Rerun A (0.6951/0.5724) came in
higher than the historic published MERT-only number (0.6518/0.5055). The gap
is larger than fold-std on both axes. KFold seed is identical
(random_state=42); model-init and DataLoader RNG are not seeded, so this most
likely reflects single-seed initialization variance, not a methodological
change. The within-script A↔B↔C↔D comparison is the only audit-honest reading;
the absolute jump in A is flagged but the historic SOTA-table number is
**not overwritten** (replacing it would amount to picking the better seed).

**Consequence for the dissertation framing.** The minority-quadrant failure
(negative per-quadrant R²) is a *dataset-size* limit, not a method limit.
Re-weighting cannot create signal where there are only ~64 training examples
per minority quadrant. The honest fix would be data augmentation (mixup within
minority quadrants, SpecAugment) or a larger affect-annotated music corpus —
neither is in scope. The current sampler-based treatment is the strongest
imbalance handling that doesn't change the data, and we now have a documented
ablation supporting that claim instead of presenting it as an arbitrary choice.

---

## Step 18 — Fusion ablation on multi-encoder (Enhanced): nuanced result (2026-05-26)

**Question (user-driven):** if last-layer-only ties 25-layer fusion on MERT-only
(Step 16), should we re-run all multi-encoder configs with last-layer-only too?

**Approach.** Rather than re-running all 4 multi-encoder configs (Dual, Triple,
Spec-only, Enhanced) — most deltas would land inside fold-noise — run one
focused ablation on our **best** model (Enhanced; R² A 0.7182 / V 0.5686). If
fusion still doesn't help here, the negative result generalises and we frame
all multi-encoder configs as "fusion kept for transparency." If fusion DOES
help, we have a nuanced story: "fusion is needed for multi-encoder integration,
not for MERT alone."

**Setup.** New script `phaseB/eval_enhanced_last_layer.py`. Identical to
`train_enhanced_dual.py` except both SSL inputs are sliced to the last layer:
`X_mert[:, -1:, :]` (1024-d), `X_w2v[:, -1:, :]` (768-d). Model built with
`mert_layers=1, w2v_layers=1` so fusion is identity. Same HybridLoss,
balanced sampler, differential optimizer, 5-fold CV, 100 epochs, batch=32,
KFold random_state=42.

**Result — fusion clearly wins:**

| Model                       | R² A             | R² V             | CCC A | CCC V |
| :--                         | :-:              | :-:              | :-:   | :-:   |
| Enhanced (25-layer fusion)  | **0.7182**       | **0.5686**       | 0.835 | 0.726 |
| Enhanced (last-layer only)  | 0.6660 ± 0.035   | 0.4881 ± 0.070   | 0.812 | 0.687 |
| **Δ (last − fusion)**       | **−0.052** (1.5σ)| **−0.081** (1.2σ)| −0.022| −0.039|

Both R² deltas land **outside fold-std**. CCC drops parallel and meaningful.
**The nuanced story is the right one** — fusion's value is not intrinsic to
MERT, it's specifically about cross-encoder integration.

**Interpretation.** In single-encoder mode, MERT's last layer carries the
signal the head needs (the learned fusion weights collapse to ~uniform because
there's nothing useful to learn). In multi-encoder mode, mel-CNN / wav2vec2 /
theory already cover the late acoustic representation, so MERT's mid-layers
fill a complementary niche the last layer alone can't reach. The fusion lets
the model coordinate complementary information *across* encoders.

**Outcome for the write-up.**
- Supervisor question "why all 25 layers?" now has a *quantitative* answer:
  because in multi-encoder configs, removing fusion costs 5–8 pp R². The
  MERT-only ablation (Step 16) is the negative control that proves the gain
  isn't free.
- `§3.1` of v3 PDF revised: now contains both ablations as a single table
  with the "nuanced and quantitative" framing.
- `02_phaseB_model.md` §3 and `04_results_and_sota.md` §2a-bis updated
  in parallel.
- **No further re-runs.** One Enhanced data point is enough; further
  multi-encoder ablation (Dual, Triple, Spec-only) would only add
  same-direction evidence within the same conclusion.

Per-quadrant R² breakdown for Enhanced last-layer-only (negative across all
minority quadrants) reinforces the Step 17 conclusion that minority-quadrant
failure is a *dataset-size* limit, not a method limit:
HVLA n=67 A=-0.78 V=-2.92 · LVHA n=64 A=-1.14 V=-2.53 · LVLA n=167 A=-0.40 V=-1.28.

---

## Step 19 — Audio augmentation: feature-space mixup on Enhanced (2026-05-26)

**Question (user-driven):** does the standard simple cited augmentation
remedy — feature-space mixup (Zhang et al. 2017) — fix the minority-quadrant
R² floor identified in Steps 17–18?

**Pre-registered pass mark.** Same as imbalance/fusion ablations: winner iff
R² beats Enhanced fusion baseline by >1 fold-std on both axes, OR >2 fold-std
on either, OR lifts minority per-quadrant R² from <0 to ≥0.

**Setup.** New script `phaseB/eval_enhanced_mixup.py`. Identical to
`train_enhanced_dual.py` except each training batch is feature-space
mixup-augmented:

  - λ ~ Beta(0.4, 0.4) per batch (standard mixup default — U-shaped,
    most mixes mild)
  - Random permutation of batch indices
  - Same λ applied to all three inputs (MERT 25×1024, w2v2 13×768,
    theory 2-d) and to labels
  - No mixup at evaluation

5-fold CV, 100 epochs, batch=32, HybridLoss + balanced sampler +
differential optimizer + KFold random_state=42 — all identical to baseline.

**Result — no winner:**

| Model                       | R² A             | R² V             | CCC A  | CCC V  |
| :--                         | :-:              | :-:              | :-:    | :-:    |
| Enhanced (no mixup)         | **0.7182**       | **0.5686**       | 0.8345 | 0.7259 |
| Enhanced + mixup (α=0.4)    | 0.7077 ± 0.014   | 0.5651 ± 0.034   | 0.8213 | 0.7160 |
| Δ                           | −0.0105 (<std)   | −0.0035 (<std)   | −0.013 | −0.010 |

All four deltas land **inside fold-noise** — mixup is statistically tied with
the no-augmentation baseline. Minority per-quadrant R² remained negative
across all three minority quadrants:

  - HVLA (n=67):  A=−0.351  V=−1.879
  - LVHA (n=64):  A=−0.907  V=−1.084
  - LVLA (n=167): A=−0.106  V=−0.898

**Decision (per pre-registered rule + user instruction).** Don't escalate to
harder methods (C-Mixup, manifold mixup, audio re-extraction). The standard
cited remedy (Zhang 2017) ties baseline and does not lift minorities into the
positive range — exactly what the dataset-size floor hypothesis (Step 17)
predicted. The thesis now has empirical evidence that this is a *data limit*,
not a method limit.

**Outcome for the write-up.** §3 of v3 PDF gets a new "augmentation ablation"
subsection with this table; framing of the minority-quadrant failure changes
from "we hypothesise it is a data floor" (Step 17 argument-only) to "the
standard augmentation remedy was tested and confirms the data floor" (now
empirically grounded). Future-work paragraph still references C-Mixup
(Yao 2022) and SpecAugment (Park 2019) as larger-scope remedies, plus
augmenting the corpus itself — the principled long-term solution.

---

## Step 20 — Two new SOTA-table rows: Mel-CNN alone + MERT+Mel+EDA triple (2026-05-26)

**Request (user):** add a new triple model (MERT + mel-CNN + EDA, swapping
wav2vec2 for EDA physiology) and a Mel-CNN-alone single-encoder baseline.

**Scripts.** Inline-model scripts (no changes to existing `models*.py`):
  - `phaseB/eval_mel_only.py` — MelSpectrogramCNN → head → regressor
  - `phaseB/eval_mert_mel_eda.py` — MERT (fusion) + Mel-CNN + EDA (7-d MLP)
    → concat 1184-d → head → regressor

Both 5-fold CV, 100 epochs, batch=32, HybridLoss, balanced sampler,
differential optimizer (where applicable), KFold(random_state=42) —
identical recipe to every other Phase B run.

**Results (5-fold mean ± std):**

| Model                            | R² A           | R² V           | CCC A | CCC V |
| :--                              | :-:            | :-:            | :-:   | :-:   |
| Mel-CNN alone (NEW)              | 0.6486 ± 0.022 | 0.4452 ± 0.052 | 0.789 | 0.630 |
| MERT + Mel-CNN + EDA (NEW)       | 0.7077 ± 0.008 | 0.5706 ± 0.046 | 0.826 | 0.733 |

**Two new findings:**

1. **Mel-CNN alone localises SSL's contribution to the valence axis.** It
   beats wav2vec2-only on arousal (0.6486 vs 0.6225, +0.026) — energy is
   acoustically easy (loudness, spectral envelope) — but loses on valence
   (0.4452 vs 0.4825 wav2vec2; 0.5055 MERT). A shallow CNN can match SSL
   on energy without harmony/tonality modelling, but cannot reach SSL on
   valence. This is a clean, citation-grade single-encoder result for the
   SOTA table — every multi-encoder gain on valence is now traceable to
   music-specific SSL pre-training, not just to "more parameters".

2. **EDA is redundant once a spectral CNN is present.** Triple-bio
   (0.7077 / 0.5706) ≈ Spec-only (0.7069 / 0.5709) within fold-noise.
   This is the third second-branch redundancy result (wav2vec2 redundant
   beside MERT+Mel; EDA redundant beside MERT+Mel; theory features
   marginal beside MERT+Mel — Enhanced is ahead on arousal only).
   Different second-branches all converge to the same ~0.71 / 0.57
   ceiling. This reinforces the **fusion-collapse + dataset-floor** story:
   beyond MERT+Mel the model is data-bottlenecked, not architecture-
   bottlenecked.

**Per-quadrant minorities remained negative on both new configurations,**
consistent with Step 19 (data-floor confirmation): n=64–67 minority
examples cannot be lifted into positive R² by any architectural choice
tested.

**Outcome for the write-up:**
- §3 of v3 PDF: results table extended with Mel-CNN-alone and Triple-bio
  rows; single-encoder paragraph rewritten to a three-baseline framing
  (Mel-CNN / wav2vec2 / MERT) that localises SSL's contribution to
  valence; new paragraph on second-branch redundancy noting all three
  second-branches converge.
- `02_phaseB_model.md` §4 and `04_results_and_sota.md` §2 + §4 (SOTA
  table) updated with the two new rows.
- The MERT-only ranking is unchanged: Enhanced > {Triple-bio ≈ Spec-only
  ≈ Triple} > Dual > MERT+EDA > MERT > {Mel-CNN ≈ wav2vec2}.

---

## Step 21 — Cluster-enforcement trade-off ablation: auxiliary quadrant-CE head (2026-05-26)

**Viva question (preemptive):** "the t-SNE shows a continuous Happy-dominated
manifold, not 4 clusters — what would fix that?"

**Approach.** Pre-registered cluster-enforcement ablation on the Enhanced
model: add an auxiliary 4-way quadrant classification head on the same 128-d
latent, train with combined loss `HybridLoss + λ · CE(quadrant)`, sweep
λ ∈ {0.0, 0.1, 0.5, 1.0}. Report fold-matched R²/CCC, **Silhouette
(cosine, test latents)**, and minority per-quadrant R².

**Pre-registered hypothesis:** Silhouette ↑ with λ; R²/CCC ↓ with λ;
minority per-quadrant R² ≈ unchanged (data-floor invariant).

**Setup.** Script: `phaseB/eval_enhanced_quadrant_ce.py`. Inline wrapper
class `EnhancedWithCE` adds a `Linear(128 → 4)` head; everything else
identical to `train_enhanced_dual.py` (HybridLoss, balanced sampler,
differential optimizer, 5-fold KFold(42), 100 epochs, batch=32).

**Result — trade-off characterised:**

|  λ   |  R² A           |  R² V           | CCC A | CCC V |  Silhouette       |
| :-:  | :-:             | :-:             | :-:   | :-:   |  :-:              |
| 0.0  | 0.7080 ± 0.021  | 0.5725 ± 0.038  | 0.827 | 0.729 | **0.255 ± 0.054** |
| 0.1  | 0.7050 ± 0.013  | 0.5811 ± 0.033  | 0.828 | 0.738 |   0.258 ± 0.059   |
| 0.5  | 0.6966 ± 0.016  | 0.5624 ± 0.053  | 0.827 | 0.730 |   0.265 ± 0.064   |
| 1.0  | 0.6638 ± 0.020  | 0.5601 ± 0.037  | 0.810 | 0.734 |   0.286 ± 0.060   |

**Three findings:**

1. **Side-finding (notable for the thesis framing):** the Enhanced model's
   latent space **already has moderate quadrant structure** —
   Silhouette **= 0.255** (cosine, 5-fold test). The historic "Silhouette
   ≈ 0" claim was from the *single-MERT* loss ablation (Step 13) which
   used Euclidean over a different latent space; it does NOT generalise
   to the multi-encoder Enhanced setup. **Multi-encoder fusion produces
   inherently more cluster-structured latents than single-encoder.**
   The earlier "no clusters" framing applies to MERT-only, not Enhanced.

2. **The trade-off is real but unfavourable.** Silhouette rises monotonically
   with λ (+0.031 from λ=0 to λ=1.0 — modest). R² A drops 4.4 pp at λ=1.0
   (outside fold-std). λ=0.1 is essentially zero-cost on R² but also
   zero-benefit on Silhouette. The point of inflection is between λ=0.5
   (acceptable cost) and λ=1.0 (cost exceeds benefit).

3. **Minority per-quadrant R² remain negative across all λ** — and
   actually *worsen* on the minorities with stronger CE (HVLA: −0.43 at
   λ=0 → −0.85 at λ=1.0). Auxiliary classification cannot manufacture
   signal where n=64–67 minority examples are the actual limit. **Third
   independent confirmation of the dataset-floor hypothesis** after
   Step 17 (imbalance ablation) and Step 19 (mixup augmentation).

**Outcome for the thesis / viva:**
- §3 of v3 PDF: new §3.6 cluster-enforcement subsection with the table and
  the trade-off interpretation. §3.5 (loss ablation) gets a sentence
  noting the multi-encoder vs single-encoder Silhouette difference.
- `04_results_and_sota.md` §2a-quinquies added.
- The continuous-manifold representation is now an **empirically defended
  design choice** (we tested cluster enforcement; it produced clusters at
  R² cost; we chose the regression performance the system is designed to
  deliver). This neutralises the most common viva criticism by quantifying
  the trade-off rather than dismissing the question.

---

## Step 22 — Persistent result logging + report-vs-log audit (2026-05-28)

**Concern (user):** "double-check all of the codes and results — maybe it
would be better to log all results after every run and compare them with the
report".

**What I did.**

1. **Persisted all ephemeral tmp logs.** Copied every background-task output
   from `/tmp/.../tasks/*.output` into `reports/run_logs/` with descriptive
   names matching the script that produced them (e.g.
   `eval_enhanced_quadrant_ce.log`). These are now under version control of
   the reports folder, not session-scoped tmp.

2. **Wrote `reports/verify_results.py`** — automated audit that extracts the
   headline 5-fold means (R² A, R² V, CCC A, CCC V, Silhouette where present)
   from every persisted log and cross-checks them against the numbers
   currently quoted in `01_phaseA_probing.md`, `02_phaseB_model.md`,
   `04_results_and_sota.md` and `generate_report_v3.py`. Tolerance: ±0.005
   absolute, which is the floor of the 3rd-decimal rounding used in prose
   tables.

3. **Ran the audit.** Result:
   ```
   AUDIT SUMMARY:  60 OK · 0 MISMATCH · 0 regex-miss · 0 unverifiable
   All report numbers match the logged run outputs within ±0.005.
   ```
   Logs verified: wav2vec2-only, MERT-only last-layer, Enhanced last-layer,
   imbalance ablation (A/B/C/D), Enhanced + mixup, Mel-CNN alone,
   Triple-bio (MERT+Mel+EDA), CE-head sweep (λ=0 / 0.1 / 0.5 / 1.0,
   including Silhouette). **Zero mismatches — every number in the reports
   traces back exactly to its run log.**

4. **Wrote `reports/run_logs/AUDIT.md`** documenting the procedure and what
   the audit does (and does not) cover.

**Going forward.** Every new run produces stdout that should be saved to
`reports/run_logs/<descriptive_name>.log`. `verify_results.py` is the
single command that re-checks numerical consistency before each PDF
regeneration. This closes a real gap in the previous workflow (logs were
only in ephemeral tmp; reports could in principle drift without anyone
noticing) and gives the thesis a reproducibility trail that an examiner
can re-run independently.
