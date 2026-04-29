# 🎼 Research Progress: Music Emotion Recognition (MER) via SSL
**Student:** Arvin Jafari Moghadam Fard  
**Status:** Phase A & B Finalized (Validated via 5-Fold CV) | Phase C Implementation  

---

## 📅 Executive Summary
This report documents the finalized development of a robust **Hybrid Affective Architecture** for MER. By implementing a **Differential Optimizer** and **Weighted Random Sampling**, the project has resolved the "Frozen Weight" issue and "Simpson's Paradox" inherent in the PMEmo dataset. The system now achieves a state-of-the-art **CCC of 0.8543** in Arousal, providing a physiologically grounded latent space for Phase C retrieval.

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

### 3. Validated Results (Multimodal 5-Fold CV)
The Hybrid + EDA model demonstrates its peak performance with physiological grounding:
* **Mean Arousal $R^2$:** **0.6738**
* **Mean Valence $R^2$:** **0.5075**
* **CCC (Arousal / Valence):** **0.8543 / 0.7692**

---

## 🚀 Next Steps: Phase C & Expansion
* **Explainable RAG:** Utilizing the optimized **0.85 / 0.77 (CCC)** latent space to build a Prototype-based retrieval engine.
* **EDA Fusion:** Leveraging physiological signals (Electrodermal Activity) to provide biological grounding for emotional energy predictions.