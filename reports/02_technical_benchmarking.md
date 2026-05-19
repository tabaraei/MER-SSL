# 📊 Technical Analysis: PMEmo Benchmarking & Stability Study

## 1. SOTA Literature Comparison (PMEmo2019 Dataset, Final)
The following table compares all validated 5-fold averages against recent SOTA benchmarks.

| Method / Paper | Year | Approach | Valence ($R^2$) | Arousal ($R^2$) | CCC (V / A) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PMEmo Original (Zhang et al.) | 2019 | IS13 (Hand-crafted) | 0.420 | 0.510 | — |
| Dutta & Chanda | 2021 | Mel-Spec + CRNN | ~0.450 | ~0.550 | — |
| Hybrid SSL (Wu et al.) | 2023 | Wav2Vec2 + Attention | ~0.480 | ~0.610 | — |
| Damer (SSL) | 2025 | Wav2Vec2 + Chords | 0.510 | 0.720 | — |
| Music2Emo | 2025 | MERT + Multitask | 0.540 | 0.780 | — |
| This Work — MERT only | 2026 | Music-SSL + WeightedFusion + HybridLoss | 0.5055 | 0.6518 | 0.74 / 0.82 |
| This Work — MERT + EDA | 2026 | MERT + Physiological Late Fusion | 0.5075 | 0.6738 | 0.77 / 0.85 |
| This Work — Dual-SSL (β=0.05) | 2026 | MERT + wav2vec2 + Entropy Reg. | 0.5676 | 0.6814 | 0.72 / 0.81 |
| **This Work — Triple (+mel-CNN)** | **2026** | **MERT + wav2vec2 + trainable mel-CNN** | **0.5758** | **0.7023** | **0.73 / 0.82** |
| This Work — Spec-only (MERT+mel) | 2026 | MERT + trainable mel-CNN (no wav2vec2) | 0.5709 | 0.7069 | 0.73 / 0.83 |

### Key Observations
* **New best — Arousal R² past 0.70:** The Triple model (MERT + wav2vec2 + trainable mel-spectrogram CNN) reaches A R² = **0.7023**, V R² = **0.5758** — the strongest configuration, surpassing Music2Emo (0.540) on valence using only audio.
* **wav2vec2 is statistically redundant:** Spec-only (MERT + mel-CNN, *no wav2vec2*) achieves A R² 0.7069 / V R² 0.5709 — equal to or better than Triple, deltas inside the ±0.013–0.042 fold std. A 109K-param from-scratch CNN fully substitutes for the 95M-param frozen wav2vec2. This reinforces the fusion-collapse and IADS-E negative findings: wav2vec2's speech-pretraining carries no music-relevant complementary structure.
* **⚠️ Global R² is majority-class-driven (critical caveat):** The 0.70 figure is inflated by the HVHA (Happy) quadrant (469/767 = 61%). Per-quadrant R² is **negative** for the three minority quadrants across *all* configurations (see §5). The headline gain reflects better majority-class fit, not improved V-A-plane generalization. Train-loss logging is not yet instrumented, so the CNN train/test gap is not directly measured — flagged as an open verification action.
* **Valence ceiling confirmed:** No audio-only method on PMEmo exceeds R² ≈ 0.58, consistent with valence requiring lyrics/cultural context (Yang & Chen, 2012).

---

## 2. Phase B Architecture — Encoder Configurations

### 2.1 Single-Encoder MERT (baseline)
```
25 layers × 1024-dim  →  WeightedLayerFusion (softmax α)  →  1024-dim
                      →  Head: 1024 → 256 → 128
                      →  Regressor: 128 → 2
```
Layer specialization: **entropy = 3.2178** (max = 3.2189 for 25 uniform layers).
Top layers: 14, 16, 17 — mid-to-late MERT transformer abstractions.

### 2.2 MERT + EDA Multimodal Fusion
Audio latent (128-dim) and EDA features (7 → 32-dim) fused via a late-fusion head (160 → 64 → 2). EDA provides physiological grounding: electrodermal activity is a direct physiological index of autonomic arousal (Thayer, 1989). Result: CCC Arousal improves from 0.82 → **0.8543**.

### 2.3 Dual-SSL (MERT + wav2vec2) — Best Audio Configuration
```
MERT (25L × 1024)  →  WeightedLayerFusion  →  1024-dim ─┐
                                                          ├──  cat  →  1792-dim
wav2vec2 (13L × 768)→  WeightedLayerFusion  →   768-dim ─┘
                      →  Head: 1792 → 512 → 256 → 128
                      →  Regressor: 128 → 2
```
wav2vec2-base (facebook) is speech-pretrained: 13 transformer layers (768-dim, 16kHz). Captures prosodic and timbral cues complementary to MERT's music-specific representations. Entropy sharpening penalty (β=0.05) applied during training. **Result: V R²=0.5676, A R²=0.6814.**

### 2.4 Triple-Branch (MERT + wav2vec2 + trainable mel-CNN) — Best Configuration
```
MERT (25L×1024)     →  WeightedLayerFusion  →  1024-d ─┐
wav2vec2 (13L×768)  →  WeightedLayerFusion  →   768-d ─┤── cat → 1920-d
mel-spec (128×1407) →  trainable CNN (109K)  →   128-d ─┘   → Head 1920→256→128 → 2
```
A shallow trainable CNN (3 conv blocks + AdaptiveAvgPool, ~109K params) over a pre-extracted center-30 s log-mel spectrogram (parameter-free transform → reuses the `.pt` pipeline; CNN is the only trainable encoder). Differential optimizer: SSL fusion lr=1e-2, CNN+head lr=1e-4. **Result: V R²=0.5758, A R²=0.7023 — best overall.**

**Ablation — Spec-only (MERT + mel-CNN, no wav2vec2):** V R²=0.5709, A R²=0.7069. Statistically equal to Triple → **wav2vec2 contributes nothing once the mel-CNN is present.** The mel-CNN is the source of the gain over Dual-SSL; wav2vec2 is redundant.

---

## 2b. ⚠️ Overfitting & Class-Imbalance Caveat (Triple/Spec-only)

The Triple and Spec-only headline R² (≈0.70) must **not** be reported as a clean win. Per-quadrant R² (Spec-only, all folds combined):

| Quadrant | n | R² Arousal | R² Valence |
|:--|:--:|:--:|:--:|
| HVHA (Happy) | 469 (61%) | 0.230 | −0.142 |
| HVLA (Calm) | 67 | −0.540 | −2.148 |
| LVHA (Angry) | 64 | −1.079 | −1.178 |
| LVLA (Sad) | 167 | −0.138 | −0.955 |

Three of four quadrants have **negative R²** (worse than predicting the mean). The global metric is high only because HVHA dominates (Simpson's Paradox). The triple-branch improvement is concentrated in the majority class — it does **not** improve, and on some minority quadrants slightly worsens, V-A-plane generalization vs. prior configs.

**Verification status:** `mainB_triple.py` does not yet log train loss, so the CNN train/test gap is not directly measured. Modest fold variance (±0.013 A) argues against catastrophic CNN overfitting, but a direct measurement is an **open action before thesis writeup**. Honest thesis framing: *additional encoders raise the majority-class-driven global metric but do not resolve the PMEmo class-imbalance ceiling.*

---

## 3. Novel Finding: Fusion Collapse in Multi-Encoder SSL

### Observation
The single-encoder MERT model learns layer specialization: entropy drops from 3.2189 (uniform) to **3.2178** with weight concentrated on layers 14, 16, 17. In the dual-encoder model, **both** `WeightedLayerFusion` modules stay at maximum entropy (MERT: 3.2180, wav2vec2: 2.5649 = theoretical max), regardless of architectural pressure:

| Intervention | MERT spec. | w2v spec. | V R² |
|:---|:--:|:--:|:--:|
| Dual β=0 (no penalty) | ~0% | ~0% | 0.5601 |
| DualBottleneck β=0.01 (1024→256 compression) | 0.2% | 1.8% | 0.4903 |
| **Dual β=0.05 (entropy penalty only)** | **0.0%** | **0.0%** | **0.5676** |

### Interpretation
With ~600 training samples, the gradient signal reaching `layer_weights` parameters is too diffuse: the large concatenated head (1792-dim) can solve the regression task by loosely mixing all 38 available layer vectors without needing to select specific ones. Adding a second encoder provides the head with an alternative information source, which dilutes the gradient pressure that caused MERT to specialize in the single-encoder setting.

The per-encoder bottleneck (DualSSLBottleneckModel: 1024→256, 768→256) made performance significantly worse (ΔV = −0.07) because the 256-dim compression is too aggressive for 600 samples — information is lost faster than selectivity is gained.

**Thesis claim:** Layer-selective SSL fine-tuning in multi-encoder architectures requires substantially more than ~600 labeled samples. This is a generalizable data-constraint finding applicable beyond the MER domain.

---

## 4. IADS-E Joint Learning — Confirmed Negative Finding

Simonetta et al. (2024) showed that mixing PMEmo with IADS-E environmental sounds (hand-crafted openSMILE features) lifts Valence R² to ~0.78. We replicated the idea using SSL encoders (`DualSSLDomainModel` with learned domain embedding). Partial sweep results:

| Config (k, p) | R² Arousal | R² Valence | vs. dual (ΔV) |
|:--|:--:|:--:|:--:|
| Dual no IADS-E (reference) | 0.6810 | 0.5601 | — |
| k=1.0, p=1.0 | 0.6093 | 0.4874 | −0.073 |
| k=1.0, p=0.25 | 0.5752 | 0.3888 | −0.171 |
| k=1.0, p=0.0 | −0.314 | −0.285 | degenerate |

**Conclusion:** SSL cross-domain emotional transfer (music ↔ environmental sound) underperforms Simonetta's hand-crafted features. MERT (music-pretrained) and wav2vec2 (speech-pretrained) embeddings do not carry transferable affective structure across domain boundaries. Notable secondary effect: CCC Valence rose at k=1,p=1 (0.7182→0.7413) while R² fell — IADS-E broadens valence ranking but adds bias on music-specific predictions.

---

## 5. Simpson's Paradox & Dataset Bias
While the global $R^2$ is high, the per-quadrant breakdown reveals a critical dataset limitation:
* **Class imbalance:** 469/767 songs (61%) are HVHA (Happy) — the model achieves A=0.26, V=0.04 on Happy but shows negative R² on Calm, Angry, and Sad quadrants. This is an inherent PMEmo limitation, not a model failure.
* **Weighted Sampler:** Inverse-quadrant-frequency sampling forces the model to encounter all quadrants proportionally. Without this, global R² would be inflated by HVHA dominance (Simpson's Paradox).
* **Retrieval readiness:** High CCC (rather than R²) is the right Phase C metric — CCC captures whether the model correctly *ranks* songs emotionally even when absolute predictions are biased by dataset skew.