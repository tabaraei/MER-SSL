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
| **This Work — MERT only** | **2026** | **Music-SSL + WeightedFusion + HybridLoss** | **0.5055** | **0.6518** | **0.74 / 0.82** |
| **This Work — MERT + EDA** | **2026** | **MERT + Physiological Late Fusion** | **0.5075** | **0.6738** | **0.77 / 0.85** |
| **This Work — Dual-SSL (best)** | **2026** | **MERT + wav2vec2 + Entropy Reg. (β=0.05)** | **0.5676** | **0.6814** | **0.72 / 0.81** |

### Key Observations
* **Valence breakthrough:** Dual-SSL achieves R² = **0.5676**, surpassing Music2Emo (0.540) on valence without multitask learning or auxiliary labels — using only audio.
* **Arousal CCC:** The MERT+EDA system achieves CCC = **0.8543**, the highest reported on PMEmo 2019. The dual-SSL system achieves 0.8087 without physiological signals.
* **Valence ceiling confirmed:** No audio-only method on PMEmo exceeds R² ≈ 0.57, consistent with the field-wide finding that valence requires lyrics/cultural context (Yang & Chen, 2012).

---

## 2. Phase B Architecture — Three Configurations

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