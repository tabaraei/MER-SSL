
# 🎼 Master's Thesis: Explainable & Perceptual Music Emotion Recognition
**Core Mission:** Moving from black-box emotion tagging toward interpretable and perceptually grounded music retrieval.

---

## 🔍 1. Research Gap

Recent self-supervised models such as MERT provide powerful audio representations and achieve strong performance on music tagging and classification tasks. However, several limitations remain in current Music Emotion Recognition (MER) and retrieval systems.

* **Limited Perceptual Alignment:** Similarity between tracks is often defined by genre or acoustic embeddings rather than perceptual musical factors such as rhythm, harmony, and expressive intensity.

* **Emotion Representation Gap:** While embeddings can predict valence–arousal labels, it remains unclear whether these representations capture interpretable musical cues that align with human emotional perception.

* **Lack of Explainability:** Most MIR retrieval systems operate as black boxes and cannot explain why two tracks are considered emotionally similar.

This thesis aims to address these issues by linking **self-supervised music representations with interpretable musical features and emotion perception**.

---

## 🛠️ 2. Proposed Approach

The thesis proposes a three-phase framework connecting representation analysis, emotion prediction, and explainable retrieval.

---

### Phase A — Perceptual Feature Validation

**Goal**

Evaluate whether self-supervised music embeddings preserve musically meaningful features that influence emotional perception.

**Method**

Perform probing experiments on MERT embeddings to predict interpretable musical attributes such as:

- Harmonic mode (Major / Minor)
- Tempo (BPM)

**Contribution**

This phase verifies that the learned representation contains **perceptually relevant musical cues**, forming a bridge between neural embeddings and music-theoretic features.

---

### Phase B — Music Emotion Recognition (MER)

**Goal**

Learn a mapping from SSL embeddings to emotional representations in the **valence–arousal space**.

**Method**

Train a lightweight regression head on top of MERT embeddings using the **PMEmo dataset**, which contains 794 music excerpts annotated with emotional ratings and physiological signals from human listeners. :contentReference[oaicite:0]{index=0}

**Contribution**

Evaluate whether SSL representations capture **musical affect**, rather than only genre or acoustic similarity.

---

### Phase C — Explainable Music Retrieval

**Goal**

Transform emotion prediction into an interpretable music retrieval framework.

**Method**

1. Encode songs using MERT embeddings.
2. Retrieve **k-nearest neighbors** in the emotion-aware embedding space.
3. Extract interpretable musical features (e.g., tempo, harmonic mode).
4. Generate a textual explanation describing the similarity between tracks.

Example explanation:

> “These tracks are similar because they share a fast tempo and major harmonic mode, producing comparable high-arousal emotional characteristics.”

**Contribution**

This phase connects **neural representation learning with explainable MIR systems**, enabling human-interpretable reasoning for music similarity.

---

## 📈 3. Evaluation Metrics

The success of the proposed framework will be evaluated along three complementary dimensions.

### 1. Perceptual Alignment
Measure whether embedding clusters correspond to musically meaningful factors such as harmonic mode and rhythmic intensity.

### 2. Emotion Prediction Performance
Evaluate MER performance using the **R² score** for valence and arousal regression.
V
### 3. Explainability Quality
Assess whether the system can provide interpretable music-theoretic explanations for similarity between retrieved tracks.

---

## 🎯 Expected Contributions

1. Empirical analysis of **musical information encoded in self-supervised music embeddings**.
2. Evaluation of SSL representations for **music emotion recognition**.
3. A prototype **explainable music retrieval framework** connecting neural embeddings with interpretable musical features.

This work aims to bridge the gap between **representation learning in MIR and human-interpretable music perception**.


-----------------------------
# 🎼 Master's Thesis: Explainable & Perceptual Music Emotion Recognition
**Core Mission:** Moving from black-box emotion tagging toward interpretable, human-aligned, and perceptually grounded music retrieval.

---

## 🔍 1. Research Gap: The Problem

While self-supervised models such as **MERT** provide powerful audio representations, several limitations remain in current Music Emotion Recognition (MER) and retrieval systems:

* **Temporal & Perceptual Neglect:** Current models often focus on broad genre or acoustic similarities, missing subtle temporal dynamics like rhythm and harmony.
* **Similarity Oversimplification:** Similarity is frequently defined by acoustic overlap rather than perceptual factors such as expressive intensity or mood.
* **The Black Box Problem:** Most MIR retrieval systems cannot explain **why** two tracks are considered emotionally similar or why a specific emotion was predicted.

> **Thesis Goal:** Linking self-supervised music representations with interpretable musical features and human-aligned emotion perception.

---

## 🛠️ 2. Proposed Three-Phase Framework

### 🟢 Phase A: Perceptual Feature Validation
**Status:** `COMPLETED` ✅

* **Goal:** Evaluate whether MERT embeddings preserve musically meaningful features that influence emotion perception.
* **Method:** Perform linear probing experiments on musical attributes.
* [cite_start]**Key Results[cite: 20, 23]:**
    | Feature | Metric | Result |
    | :--- | :--- | :--- |
    | **Harmonic Mode** | Accuracy | **100% (Major/Minor)** |
    | **Tempo (BPM)** | $R^{2}$ | **0.12** |
* **Contribution:** This phase verified that the representation contains **perceptually relevant cues**, bridging neural embeddings and music-theoretic features.

---

### 🔵 Phase B: Music Emotion Recognition (MER)
**Status:** `COMPLETED` ✅

* **Goal:** Map SSL embeddings to emotional representations in the **Valence–Arousal space**.
* **Method:** Train a lightweight regression head using the **PMEmo dataset** (794 music excerpts) and conduct layer-wise analysis.
* [cite_start]**Key Results[cite: 28, 29, 75]:**
    | Dimension | Metric | Result |
    | :--- | :--- | :--- |
    | **Arousal (Energy)** | $R^{2}$ | **0.70** |
    | **Valence (Mood)** | $R^{2}$ | **0.51** |
* **Observation:** Layer-wise probing revealed that affective information is consolidated in the final transformer layer (**Layer 24**)[cite: 75, 77].

---

### 🟡 Phase C: Explainable Music Retrieval
**Status:** `IN PROGRESS` 🚧

* **Goal:** Transform emotion prediction into an interpretable music retrieval framework.
* **Method:**
    1.  **Optimization:** Apply **Contrastive Learning** to refine the structure of emotion-aware embeddings.
    2.  **Retrieval:** Use $k$-nearest neighbors in the refined embedding space.
    3.  **Synthesis:** Integrate interpretable features (tempo, mode) to describe track similarity.
* **Example Explanation:**
    > *“These tracks are similar because they share a fast tempo and major harmonic mode, producing comparable high-arousal emotional characteristics”.*

---

## 📈 3. Evaluation Metrics

The success of the framework is evaluated along three dimensions:

1.  **Perceptual Alignment:** Correlation between embedding clusters and musical factors like harmonic mode.
2.  **Emotion Prediction Performance:** Accuracy measured by **$R^{2}$ score** for valence and arousal regression.
3.  **Explainability Quality:** The system's ability to provide valid music-theoretic reasoning for retrieved track similarity.

---

## 🎯 Expected Contributions

1.  **Empirical Analysis:** Investigation of musical information encoded across the layers of self-supervised music embeddings.
2.  **SSL Evaluation for MER:** Benchmarking of SSL representations for music emotion recognition beyond simple tagging.
3.  **Explainable Retrieval Framework:** A prototype connecting neural embeddings with interpretable musical features for human-aligned reasoning.