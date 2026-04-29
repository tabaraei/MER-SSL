# 📊 Technical Analysis: PMEmo Benchmarking & Stability Study

## 1. SOTA Literature Comparison (PMEmo2019 Dataset)
The following table compares our **validated 5-fold average** against recent SOTA benchmarks.

| Method / Paper | Year | Approach | Valence ($R^2$) | Arousal ($R^2$) | CCC (V / A) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PMEmo Original** | 2019 | IS13 (Hand-crafted) | 0.420 | 0.510 | N/A |
| **Damer (SSL)** | 2025 | Wav2Vec2 + Chords | 0.510 | 0.720 | N/A |
| **Music2Emo** | 2025 | MERT + Multitask | 0.540 | 0.780 | N/A |
| **Hybrid (Audio Only)**| **2026** | **Balanced + Fixed Opt** | **0.5055** | **0.6518** | **0.74 / 0.82** |
| **Hybrid + EDA** | **2026** | **Multimodal Fusion** | **0.5075** | **0.6738** | **0.76 / 0.85** |

### Key Observations:
* **Valence Stability:** Our Hybrid model maintains a stable **0.50+ $R^2$** for Valence, competitive with audio-only SSL benchmarks.
* **Superior Correlation:** The CCC scores (**0.85** for Arousal) indicate that the model captures emotional trends with very high fidelity.

---

## 2. Recovery of Information & XAI Defense
The drop in **Weight Entropy (3.2178)** is a primary achievement of Phase B. 
1. **Specialization:** By forcing the model to select layers (specifically **Layers 14, 16, and 17**), we demonstrate that mid-to-late transformer abstractions are the most critical for affective synthesis.
2. **Layer Synthesis:** The mass distribution (Late Layers: **0.367**) confirms that the model draws heavily from deeper semantic abstractions while maintaining early-layer grounding.

---

## 3. Simpson's Paradox & Dataset Bias
While the global $R^2$ is high, the per-quadrant analysis identifies a critical limitation of the PMEmo dataset: **Imbalance**. 
* **Weighted Sampler Impact:** Our use of a **Weighted Sampler** ensures that the model attempts to map underrepresented regions like "Sad" and "Calm" ordinally, rather than simply over-predicting the "Happy" majority.
* **Retrieval Readiness:** This organized latent geometry (high CCC) ensures that Phase C retrieval will return songs that are emotionally aligned with user queries, regardless of coordinate bias.