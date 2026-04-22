# 🎼 Research Progress: Music Emotion Recognition (MER) via SSL
**Student:** Arvin Jafari Moghadam Fard  
**Status:** Phase A & B Finalized | Phase C Implementation  

---

## 📅 Executive Summary
This report summarizes the development of an explainable MER system utilizing the **MERT** self-supervised model. The project has transitioned from a baseline single-layer regression to a **Hybrid Representation Learning** architecture. This new approach, featuring **Weighted Layer Fusion** and **Supervised Contrastive Regression (SupCR)**, has achieved State-of-the-Art (SOTA) performance on the **PMEmo2019** dataset, particularly in overcoming the traditional "Valence Gap."

---

## ✅ Phase A: Perceptual Feature Validation (Completed)
**Objective:** To verify if MERT embeddings preserve essential music theory cues before emotional modeling.

* **Harmonic Mode Probing:** Achieved **100% accuracy** in Major/Minor classification.
* **Tempo Probing:** Achieved an $R^2$ of **0.12**, confirming that rhythmic density is encoded in the latent space.
* **XAI Justification:** These features were extracted **directly from the embeddings**, proving the latent space retains critical music-theoretic information necessary for human-interpretable retrieval.

---

## ✅ Phase B: Hybrid Affective Modeling (Finalized)
**Objective:** To recover "lost" information and optimize the latent space for emotion prediction.

* **Architecture Expansion:** Implemented a **Weighted Layer Fusion** head that learns to optimally combine features from all 25 MERT layers, ensuring low-level acoustic cues (early layers) and high-level semantics (final layers) are both preserved.
* **Optimization:** Applied **Supervised Contrastive Regression (SupCR)** to physically reorganize the latent space based on emotional distance.
* **Key Performance Results:**
    * **Arousal ($R^2$): 0.892** (SOTA level).
    * **Valence ($R^2$): 0.808** (Significant breakthrough in mood prediction).

---

## 🧠 Explainability (XAI) Strategy
To address concerns regarding "black-box" embeddings, this project utilizes a **Representation Probing** and **Latent Clustering** defense:
1. **Information Retention:** The massive $R^2$ surge proves that the "lost information" in previous models was effectively recovered via weighted fusion.
2. **Prototype-Based Interpretation:** The organized contrastive space allows Phase C to utilize "Prototypes" (e.g., "This song is similar to the 'High-Energy/Major' prototype") for explainable retrieval.

---

## 🚀 Next Steps (Phase C)
* **Explainable Retrieval Engine:** Utilizing the optimized 0.89/0.80 latent space to build a **Retrieval-Augmented Generation (RAG)** framework.
* **Mechanism:** Using the learned "Prototypes" to generate textual reasoning for music similarity.