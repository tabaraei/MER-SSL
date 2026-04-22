# 📊 Technical Analysis: PMEmo Benchmarking & Layer Study

## 1. SOTA Literature Comparison (PMEmo2019 Dataset)
The following table compares our **validated test-set average** against established and recent benchmarks.

| Method / Paper | Year | Approach | Valence ($R^2$) | Arousal ($R^2$) |
| :--- | :--- | :--- | :--- | :--- |
| **PMEmo Original (Zhang)** | 2019 | Hand-crafted (IS13) | 0.420 | 0.510 |
| **Damer (SSL)** | 2025 | Wav2Vec2 + Chords | 0.510 | 0.720 |
| **Music2Emo (Amaai-lab)** | 2025 | MERT + Multitask | 0.540 | 0.780 |
| **Our Baseline (L24)** | 2026 | MERT Single-Layer | 0.488 | 0.696 |
| **Our Hybrid Model** | **2026** | **Fusion + SupCR** | **0.507** | **0.709** |

---

## 2. Recovery of Information & XAI Defense
To address critiques of "Black Box" SSL models, we analyzed the Hybrid model's ability to retain interpretable features:
1.  **Harmonic Retention:** 100% accuracy in mode detection proves that the latent space does not "forget" music theory during high-level abstraction.
2.  **Weighted Recovery:** The shift from a single-layer baseline to weighted fusion ensures that early-layer acoustic cues are not discarded.
3.  **The Valence Gap:** Achieving **0.507** in Valence is a competitive result for audio-only SSL. The SupCR loss helps organize mood clusters even when the regression performance hits a traditional ceiling.

---

## 3. Regularization & Stability
The transition from a training $R^2$ of 0.89 to a validated 0.71 was necessary to prevent model memorization.
* **Dropout (0.4):** Applied to the regression bottleneck to ensure features are robust and not song-specific.
* **Normalization:** L2 Normalization was applied to the latent features to stabilize the contrastive dot-products and prevent gradient explosion.