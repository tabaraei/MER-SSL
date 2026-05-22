# 02 — Phase B: Music Emotion Prediction

**Goal:** learn an accurate mapping from frozen SSL embeddings to the
valence-arousal circumplex, and organize the latent space so that emotionally
similar songs cluster (a prerequisite for Phase C retrieval).

**Code:** `phaseB/models.py`, `models_triple.py`, `models_enhanced.py`,
`losses.py`, `mainB.py`, `mainB_triple.py`, `train_enhanced_dual.py`.

---

## 1. Core Architecture

```
MERT hidden states: 25 layers × 1024-d × T  →  mean-pool over time → (25, 1024)
        │
  [WeightedLayerFusion]  learnable softmax α over layers → fused (1024,)
        │
  [Deep Head]  1024 → 256 → 128  (LayerNorm, ReLU, Dropout)   ← latent z
        │
  [Regressor]  128 → 2  (arousal, valence)
```

`WeightedLayerFusion` learns one softmax weight per layer, so the model — and the
reader — can see which layers it relies on. The 128-d latent `z` is L2-normalized
and reused by Phase C.

## 2. Four-Term Loss (`HybridLoss`)

A single error term cannot capture the several ways emotion prediction fails, so
the loss combines four, each targeting a distinct failure mode:

| Term | Weight | Purpose |
| :-- | :-: | :-- |
| MSE | 1.0 | Absolute regression accuracy |
| CCC | 0.5 | Correlation + mean + variance agreement (AVEC standard) |
| Rank | 0.3 | Preserves ordinal emotion ordering (soft Spearman) |
| SupCR | 0.1 | Pulls same-emotion songs together in latent space → enables retrieval |

The SupCR (Supervised Contrastive Regression) term was *intended* to make the
latent space emotionally organized — but the loss ablation (§2a) shows it improves
local retrieval, not clustering. See §2a before citing SupCR as a clustering method.

### 2a. Loss Ablation (single-MERT, test-fold, 100 epochs)

Holding the architecture constant and varying only the loss:

| Loss config | CCC A | CCC V | P@5 | Silhouette |
| :-- | :-: | :-: | :-: | :-: |
| MSE only | 0.6861 | 0.5955 | 0.5259 | +0.012 |
| + CCC + Rank (no SupCR) | 0.7814 | 0.7113 | 0.5398 | +0.021 |
| + SupCR (full hybrid) | **0.8165** | 0.7110 | **0.5734** | −0.031 |

- **CCC + Rank: decisively justified.** Adding them to MSE lifts CCC Arousal +0.095
  and CCC Valence +0.116 — this answers "why not just MSE?" (MSE alone loses ~0.10 CCC).
- **SupCR: justified for retrieval, NOT clustering.** It adds +0.034 Precision@5 and
  +0.035 CCC Arousal, but Silhouette goes the *wrong* way (+0.021 → −0.031): removing
  SupCR *raised* the cluster score. SupCR pulls songs together by continuous V-A
  proximity (tighter local neighbourhoods → better Precision@k) at the cost of
  discrete quadrant separation. **This directly refutes the original "SupCR organizes
  emotional clusters" hypothesis** while still justifying SupCR for retrieval.
- **Full hybrid vs MSE:** +0.13 CCC A, +0.12 CCC V, +0.047 P@5 — the complex loss is
  justified overall (for CCC and retrieval; no config produces real clustering —
  all Silhouettes ≈ 0).

## 3. Two Training Fixes

**Frozen fusion weights.** With a uniform learning rate, the fusion weights barely
moved (entropy stuck at maximum). **Fix:** a *differential optimizer* — fusion
params at lr = 1e-2, head/regressor at lr = 1e-4. This let the fusion weights move,
but **honestly the learned distribution stays near-uniform** (entropy ≈ 3.218 / max
3.219) with only a faint lean toward mid-to-late layers — and the exact top layer
varies by run (14, 15, 16 in one run; 10, 12 in others; see
`artifacts/layer_fusion_weights.png`). "Layers 14/16/17 dominate" is a marginal,
unstable argmax, **not** strong specialization — report it as a faint lean only.

**Class imbalance (Simpson's paradox).** PMEmo is ~61% high-valence/high-arousal
(HVHA). A model can post a high global score by over-predicting "happy." **Fix:** a
*balanced sampler* (sample weight = inverse quadrant frequency) so all four
quadrants are seen proportionally.

## 4. Encoder Configurations (all 5-fold CV)

| Model | Branches | A R² | V R² | CCC A | CCC V |
| :-- | :-- | :-: | :-: | :-: | :-: |
| MERT only | MERT | 0.6518 | 0.5055 | 0.82 | 0.74 |
| MERT + EDA | MERT + physiology | 0.6738 | 0.5075 | **0.8543** | 0.7692 |
| Dual-SSL (β=0.05) | MERT + wav2vec2 | 0.6814 | 0.5676 | 0.8087 | 0.7231 |
| Triple | MERT + wav2vec2 + mel-CNN | 0.7023 | **0.5758** | 0.8233 | 0.7329 |
| Spec-only | MERT + mel-CNN | 0.7069 | 0.5709 | 0.8271 | 0.7314 |
| **Enhanced** | MERT + wav2vec2 + tempo/key | **0.7182** | 0.5686 | 0.8345 | 0.7259 |

(Full tables, per-quadrant breakdowns, and SOTA comparison in `04_results_and_sota.md`.)

### 4a. EDA multimodal fusion
EDA (electrodermal activity) is a physiological arousal signal recorded from
listeners. A 7-dim EDA feature vector is projected (7→32) and late-fused with the
128-d audio latent. It produced the **best arousal CCC (0.8543)** — sensible, since
arousal is physical and the body reacts to it. EDA gives a body-based validation
channel independent of the audio.

### 4b. Dual-SSL (add wav2vec2)
wav2vec2-base is *speech*-pretrained, so it "hears" prosody/articulation/energy
rather than musical harmony. Adding it lifted Valence R² 0.5055 → 0.5676. An
entropy-sharpening penalty (β=0.05) acts as a mild regularizer (+0.0075 valence).

### 4c. Triple (add a trainable mel-CNN)
A small CNN (~109K params) over a pre-extracted center-30 s log-mel spectrogram
(the transform is parameter-free, so it is pre-extracted and the pipeline stays
identical to dual-SSL). This crossed **Arousal R² 0.70** for the first time.

### 4d. Enhanced (probing-driven augmentation)
A tiny branch fed *only* the Phase-A gap features [tempo, key] (Linear(2,32)),
concatenated with the two SSL fusions. Best **Arousal R² 0.7182**.

## 5. Findings (the scientific contributions of Phase B)

**Fusion collapse (multi-encoder SSL).** The single-encoder model learns layer
specialization, but in the dual model *both* fusion modules collapse to ~uniform
weights (≈0% specialization), and no intervention (256-d bottleneck — which *hurt*,
V 0.4903; or entropy penalty) restores it. **Root cause:** with ~600 training
songs and a large concatenated head, the gradient to the per-layer weights is too
diffuse — the head solves the task without needing to be selective. *Layer-selective
multi-encoder SSL fine-tuning needs substantially more labeled data than PMEmo has.*

**wav2vec2 redundancy.** Spec-only (MERT + mel-CNN, **no wav2vec2**) is
statistically tied with Triple on every metric (differences inside fold std). A
109K-param from-scratch CNN fully substitutes for the 95M-param frozen wav2vec2 —
i.e. wav2vec2's speech-pretraining carries no music-relevant complementary
structure here. Consistent with fusion collapse and the IADS-E result below.

**IADS-E joint learning — negative finding.** Simonetta et al. (2024) lifted
valence by mixing music with environmental sounds (hand-crafted features). The SSL
replication (`DualSSLDomainModel`, domain embedding, k/p mixing sweep) made valence
**worse** in every tested config. **Conclusion:** SSL embeddings do not transfer
emotional structure across the music↔environmental-sound boundary as hand-crafted
features did — a legitimate, publishable negative result.

**Probing-driven augmentation — arousal-only gain.** Re-injecting [tempo, key]
raised Arousal R² (+0.037) but not Valence (+0.001). **Tempo is the active
ingredient** — the canonical arousal correlate. **Key was inert**, because it was
fed as a raw integer 0–11 (keys are circular; this discards that) and `mode` was
not in the gap set. Honest lesson: supplying a diagnosed gap helps *only* when the
feature has a real link to the target *and* is encoded sensibly.

**Class-imbalance ceiling.** Across *all* configs, per-quadrant R² is negative for
the three minority quadrants (Calm/Angry/Sad); the high global R² is HVHA-driven.
This dataset property — not architecture — is the dominant remaining bottleneck.

**Valence ceiling.** No audio-only config exceeds V R² ≈ 0.58, matching the
field-wide ceiling (valence depends on lyrics/culture absent from audio).

## 6. Phase A → B Closure (why this hangs together)

Phase A diagnosed *what MERT cannot linearly expose* (tempo, key). Phase B fed
exactly those back (Enhanced model). The improvement landed precisely where theory
predicts (arousal, via tempo) and was honestly null where the encoding was poor
(valence, via raw key). The same diagnosed gaps also drive the Phase C
music-theory explanation — one coherent thread from probing → prediction →
explanation.
