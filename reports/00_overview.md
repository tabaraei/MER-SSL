# 00 — Project Overview

**Thesis:** Explainable & Perceptual Music Emotion Recognition via Self-Supervised Learning
**Student:** Arvin Jafari Moghadam Fard
**Status:** Phase A ✅ · Phase B ✅ · Phase C ✅ (implemented + evaluated)

> This is the clean, phase-organized report set used for thesis writing. Each file
> documents one step of what was built and what was found. The chronological lab
> log lives in `updater.md`; superseded drafts are in `archive/`.
>
> - `00_overview.md` — this file (motivation, framework, headline results)
> - `01_phaseA_probing.md` — does MERT encode musical structure?
> - `02_phaseB_model.md` — the emotion-prediction models and findings
> - `03_phaseC_explainability.md` — the explainable retrieval system
> - `04_results_and_sota.md` — every results/SOTA table in one place
> - `05_limitations_future_work.md` — honest limitations + next steps

---

## 1. Motivation

Music Emotion Recognition (MER) maps audio to emotion, conventionally as two
continuous axes — **arousal** (energy: calm ↔ intense) and **valence**
(positivity: sad ↔ happy) — the Russell (1980) circumplex. Modern self-supervised
models predict these well but operate as **black boxes**: they output numbers
without explaining *why* two songs are emotionally similar. For trust-sensitive
uses (therapeutic playlists, mood regulation, clinical music therapy) that "why"
is a functional requirement, not a nicety.

## 2. Research Gap (from the proposal)

1. **Limited perceptual alignment** — similarity is usually defined by genre or
   raw acoustic embeddings, not perceptual factors (rhythm, harmony, intensity).
2. **Emotion-representation gap** — it is unclear whether SSL embeddings capture
   *interpretable* musical cues that align with human emotional perception.
3. **Lack of explainability** — most MIR retrieval systems cannot justify why two
   tracks are emotionally similar.

**Thesis aim:** link self-supervised music representations with interpretable
musical features and emotion perception, and deliver retrieval that can *explain*
itself.

## 3. Three-Phase Framework

| Phase | Question | Outcome |
| :-- | :-- | :-- |
| **A — Probing** | Does MERT encode musically meaningful structure before any fine-tuning? | Yes for harmony (mode 100%); weak for tempo (R²≈0.12); full per-layer probing identifies {tempo, key} as gaps. |
| **B — Prediction** | Can we map SSL embeddings to the V-A circumplex accurately? | Arousal R² up to **0.72**, Valence R² up to **0.58** (at the audio-only ceiling). Several rigorous negative/nuanced findings. |
| **C — Explanation** | Can the emotion-aware latent space support self-explaining retrieval? | Prototype-based, ante-hoc retrieval + multi-channel explanation; Precision@5 ≈ 0.58 validates it. |

## 4. Data & Models

- **PMEmo 2019** — 794 pop-song chorus clips (**767** matched across all sources),
  continuous V-A annotations + per-listener EDA (electrodermal activity). Clips are
  **variable length (≈13–77 s, not a fixed 45 s)** at 24 kHz.
- **IADS-E** — generalized environmental sounds with V-A labels (used in a
  cross-domain transfer experiment; see Phase B).
- **MERT-v1-330M** — music-pretrained SSL encoder, 25 transformer layers × 1024-d.
  Used **frozen** (feature extractor; not fine-tuned).
- **wav2vec2-base** — speech-pretrained SSL encoder, 13 layers × 768-d. Frozen.
- **Mel-spectrogram CNN** — small *trainable* branch (~109K params) over a
  pre-extracted log-mel spectrogram (added in the triple-branch experiment).

## 5. Headline Results (PMEmo, 5-fold CV)

| Configuration | Arousal R² | Valence R² | CCC A | CCC V |
| :-- | :-: | :-: | :-: | :-: |
| MERT only (hybrid) | 0.6518 | 0.5055 | 0.82 | 0.74 |
| MERT + EDA | 0.6738 | 0.5075 | **0.8543** | 0.7692 |
| Dual-SSL (MERT + wav2vec2, β=0.05) | 0.6814 | 0.5676 | 0.8087 | 0.7231 |
| **Triple (MERT + wav2vec2 + mel-CNN)** | 0.7023 | **0.5758** | 0.8233 | 0.7329 |
| Spec-only (MERT + mel-CNN) | 0.7069 | 0.5709 | 0.8271 | 0.7314 |
| **Enhanced (MERT + wav2vec2 + tempo/key)** | **0.7182** | 0.5686 | 0.8345 | 0.7259 |

- **Best Valence:** Triple (0.5758). **Best Arousal:** Enhanced (0.7182). **Best CCC Arousal:** MERT+EDA (0.8543).
- R² explained plainly: an Arousal R² of 0.72 means the model accounts for ~72% of the variation in how energetic listeners rated songs.
- ⚠️ **Caveat on every row:** the global R² is inflated by the majority HVHA (Happy) quadrant (61% of data); minority quadrants have negative R² across *all* configurations. This is a PMEmo class-imbalance ceiling, not a model defect. See `05_limitations_future_work.md`.
- ⚠️ **Read the honest self-critique first** (`05_limitations_future_work.md`, §0): the latent space is *not* cleanly emotion-clustered (Silhouette ≈ 0, lower than raw MERT), the multi-encoder program was mostly null/negative, and the layer-fusion "interpretability" is near-uniform. The defensible framing of this thesis is **a rigorous audit of SSL-for-affect**, where these negatives are the contribution — not a SOTA emotion-clustered system.

**Phase C evaluation (test-fold, out-of-sample):** Precision@5 ≈ **0.58** (retrieval returns emotionally similar songs); Silhouette ≈ **0** (emotion is a continuous gradient, not 4 separable clusters — reported transparently, not as failure).

## 6. Key Contributions

1. **Empirical probing** of musical information across all 25 MERT layers; identifies which musical attributes the representation does and does not linearly expose.
2. **Hybrid affective model** combining Weighted Layer Fusion + a four-term loss (MSE + CCC + Rank + SupCR), with a differential optimizer and balanced sampler that fix two concrete training failures.
3. **Rigorous negative/nuanced findings** that strengthen the scientific story: *fusion collapse* in multi-encoder SSL, *wav2vec2 redundancy*, *negative IADS-E cross-domain transfer*, and the *continuous-emotion* interpretation of a near-zero Silhouette.
4. **Probing-driven feature augmentation** (Phase A→B closure): re-injecting the diagnosed gap features (tempo, key) improves arousal specifically.
5. **Prototype-based, ante-hoc explainable retrieval** (Phase C): k-NN over an SupCR-organized latent space, a 4-centroid prototype activation profile, contrastive foils, physiological (EDA) grounding, MERT layer attribution, and librosa music-theory annotation — with an explicit, honest ante-hoc vs post-hoc accounting.

## 7. Reference Code Map

```
MERT/ssl_scripts/
  extract_pmemo.py / extract_pmemo_wav2vec.py / extract_pmemo_melspec.py  — feature extractors
  phaseA/  extract_music_theory.py, run_music_theory_probing.py           — Phase A
  phaseB/  models.py, models_triple.py, models_enhanced.py, losses.py,
           mainB.py, mainB_triple.py, train_enhanced_dual.py              — Phase B
  phaseC/  index_builder.py, retriever.py, explainer.py, evaluator.py,
           eda_loader.py, music_theory_annotator.py,
           evaluate_latent_space.py, mainC.py                             — Phase C
```
