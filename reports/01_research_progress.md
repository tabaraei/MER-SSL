# 🎼 Research Progress: Music Emotion Recognition (MER) via SSL
**Student:** Arvin Jafari Moghadam Fard  
**Status:** Phase A & B Finalized (Validated via 5-Fold CV) | Phase C Implementation  

---

## 📅 Executive Summary
This report documents the transition from exploratory probing to a robust **Hybrid Affective Architecture**. The primary focus was overcoming the "Black Box" nature of Self-Supervised Learning (SSL) by implementing a multi-layer synthesis approach. While initial results showed extreme performance ($R^2 > 0.80$), subsequent diagnostics identified significant overfitting. The project has since moved to a **5-Fold Cross-Validation** framework, establishing a scientifically valid baseline for unseen music retrieval.

---

## ✅ Phase A: Perceptual Feature Validation (Completed)
**Objective:** Verify if the MERT latent space preserves music-theoretic cues necessary for explainability.

* **Harmonic Mode Probing:** Implemented a linear probe on the frozen embeddings. Result: **100% Accuracy** in Major/Minor detection.
* **Tempo Probing:** Achieved an $R^2$ of **0.12**. This suggests that while rhythmic density is present, it is likely encoded non-linearly or across multiple layers.
* **Significance:** This phase confirmed that "information loss" is not occurring at the encoding level, providing a solid foundation for XAI-driven retrieval.

---

## ✅ Phase B: Hybrid Affective Modeling (Finalized)
**Objective:** Optimize the mapping from SSL embeddings to the Valence-Arousal (V-A) circumplex.

### 1. Architectural Design: Weighted Layer Fusion
To address the supervisor's concern regarding the potential loss of low-level acoustic information in deep layers, we moved beyond single-layer probing (Layer 24).
* **Mechanism:** A learnable `WeightedLayerFusion` module was developed. It utilizes a `softmax` activation over 25 trainable parameters to assign an importance weight to every MERT layer.
* **Purpose:** This allows the model to dynamically capture rhythmic/acoustic features from early layers and semantic/affective features from the late layers simultaneously.

### 2. Optimization: Supervised Contrastive Regression (SupCR)
Standard MSE loss often fails to capture the subjective "neighborhoods" of emotion. We implemented a dual-loss objective: $Total Loss = Loss_{MSE} + \lambda Loss_{SupCR}$.
* **SupCR Objective:** Forces the model to cluster songs with similar emotional coordinates (within a **0.30 threshold**) while pushing dissimilar tracks apart.
* **Geometric Impact:** This reorganizes the 1024-dimensional latent space to be perceptually meaningful, which is a prerequisite for Phase C retrieval.

### 3. Scientific Validation & Overfitting Diagnostic
Initial results indicated $R^2$ values of **0.89 (Arousal)** and **0.80 (Valence)**. Diagnostic testing revealed these were **training-set artifacts** (memorization).
* **Action Taken:** Implemented **5-Fold Cross-Validation**, **Weight Decay (1e-2)**, and **Dropout (0.4)**.
* **Validated Results:** * **Mean Arousal $R^2$: 0.709**
    * **Mean Valence $R^2$: 0.507**
* **Conclusion:** These scores represent the model's **true generalization** on unseen data, matching contemporary 2025 SOTA benchmarks while ensuring academic integrity.

---

## 🚀 Next Steps: Phase C & Expansion
* **Explainable RAG:** Utilizing the contrastive latent space to retrieve "Prototypes" and generate human-readable reasoning for music recommendations.
* **EDA Fusion:** Integrating Electrodermal Activity (physiological) signals to ground the energy-based predictions in biological human response.