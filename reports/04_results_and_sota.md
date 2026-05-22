# 04 — Results & SOTA (consolidated)

Every results table in one place — the reference for the thesis Results chapter.
All Phase B numbers are 5-fold cross-validation on PMEmo (767 matched songs).

> **Read every R²/CCC with one caveat:** global scores are inflated by the majority
> HVHA (Happy) quadrant (61% of PMEmo). Per-quadrant R² is negative for the three
> minority quadrants across *all* configurations (see §4 and `05_limitations_future_work.md`).

---

## 1. Phase A — Probing

| Probe | Target | Metric | Result |
| :-- | :-- | :-- | :-: |
| Harmonic mode | Major / Minor | accuracy | ~1.00 |
| Tempo | BPM | R² | 0.12 |
| Full per-layer sweep | 8 music-theory features × 25 layers | R²/acc | gaps = **{tempo, key}** |

Gap criterion: best-layer R² < 0.40 (regression) or accuracy < 0.65 (classification).
Key/mode via Krumhansl–Schmuckler. Details: `01_phaseA_probing.md`.

## 2. Phase B — Emotion Prediction (5-fold CV)

| Configuration | Arousal R² | Valence R² | CCC Arousal | CCC Valence |
| :-- | :-: | :-: | :-: | :-: |
| MERT only (hybrid) | 0.6518 | 0.5055 | 0.82 | 0.74 |
| MERT + EDA | 0.6738 | 0.5075 | **0.8543** | 0.7692 |
| Dual-SSL (MERT + wav2vec2, β=0.05) | 0.6814 | 0.5676 | 0.8087 | 0.7231 |
| Triple (MERT + wav2vec2 + mel-CNN) | 0.7023 | **0.5758** | 0.8233 | 0.7329 |
| Spec-only (MERT + mel-CNN) | 0.7069 | 0.5709 | 0.8271 | 0.7314 |
| **Enhanced (MERT + wav2vec2 + tempo/key)** | **0.7182** | 0.5686 | 0.8345 | 0.7259 |

- **Best Valence:** Triple (0.5758) · **Best Arousal:** Enhanced (0.7182) · **Best CCC Arousal:** MERT+EDA (0.8543).
- Enhanced ≈ Spec-only ≈ Triple on arousal → the extra encoder/feature adds little once a trainable spectral or theory branch is present.

### 2a. Fusion-collapse interventions (dual encoder)

| Intervention | MERT spec. | w2v spec. | Valence R² |
| :-- | :-: | :-: | :-: |
| Dual β=0 (no penalty) | ~0% | ~0% | 0.5601 |
| Dual + 256-d bottleneck, β=0.01 | 0.2% | 1.8% | 0.4903 |
| Dual + entropy penalty β=0.05 | 0.0% | 0.0% | 0.5676 |

Single-encoder MERT, for contrast: entropy 3.2178 / max 3.2189 (layers 14/16/17).
The bottleneck *hurt* (256-d too aggressive for ~600 samples); the penalty was a
mild regularizer that did not induce specialization.

### 2a-bis. Loss-function ablation (single-MERT, test-fold, 100 epochs)

| Loss config | CCC A | CCC V | P@5 | Silhouette |
| :-- | :-: | :-: | :-: | :-: |
| MSE only | 0.6861 | 0.5955 | 0.5259 | +0.0124 |
| + CCC + Rank (no SupCR) | 0.7814 | 0.7113 | 0.5398 | +0.0206 |
| + SupCR (full hybrid) | 0.8165 | 0.7110 | 0.5734 | −0.0311 |

- CCC+Rank vs MSE: **+0.095 CCC A, +0.116 CCC V** → the dominant contribution; justifies the non-MSE terms decisively.
- SupCR vs no-SupCR: +0.034 P@5, +0.035 CCC A (helps retrieval) but Silhouette **drops** (+0.021 → −0.031) → SupCR improves local retrieval, **not** clustering; it refutes the "SupCR creates emotional clusters" claim.
- Full vs MSE: +0.13 CCC A, +0.12 CCC V, +0.047 P@5 → the hybrid loss is justified overall. No config clusters (all Silhouettes ≈ 0).

### 2b. IADS-E joint learning (negative finding, partial k,p sweep)

| Config (k, p) | Arousal R² | Valence R² | ΔV vs dual |
| :-- | :-: | :-: | :-: |
| Dual, no IADS-E (reference) | 0.6810 | 0.5601 | — |
| k=1.0, p=1.0 | 0.6093 | 0.4874 | −0.073 |
| k=1.0, p=0.25 | 0.5752 | 0.3888 | −0.171 |
| k=1.0, p=0.0 | −0.314 | −0.285 | degenerate (no music in train) |

Every joint config underperforms the dual baseline → SSL cross-domain emotional
transfer (music ↔ environmental sound) underperforms hand-crafted features.

### 2c. Per-quadrant R² (class-imbalance evidence; representative: Spec-only)

| Quadrant | n | Arousal R² | Valence R² |
| :-- | :-: | :-: | :-: |
| HVHA (Happy) | 469 (61%) | 0.230 | −0.142 |
| HVLA (Calm) | 67 | −0.540 | −2.148 |
| LVHA (Angry) | 64 | −1.079 | −1.178 |
| LVLA (Sad) | 167 | −0.138 | −0.955 |

The global R² is carried by HVHA; minority quadrants are below the mean predictor.
This pattern holds across all configurations.

## 3. Phase C — Retrieval Evaluation (test-fold, out-of-sample)

| Space | Precision@5 | Precision@10 | Precision@20 | Silhouette |
| :-- | :-: | :-: | :-: | :-: |
| **Naive raw last-layer MERT** (untrained baseline) | 0.4847 | 0.4614 | 0.4553 | **0.1001** |
| MERT (single, SupCR) | 0.5760 | 0.5687 | 0.5469 | −0.0293 |
| Dual-SSL (MERT + wav2vec2, SupCR) | 0.5849 | 0.5613 | 0.5382 | +0.0026 |

Precision@k = mean fraction of top-k neighbors within a 0.20 V-A radius.
**Random-chance baseline = 0.276**, so Precision@5 ≈ 0.58 is ~2× chance (retrieval
works). Silhouette ≈ 0 (trained) = continuous emotion, not 4 clusters (not a
failure). Encoders statistically tied.

**Prototype-activation accuracy** (% of songs whose best-match quadrant centroid =
true quadrant, leave-one-out): **Dual 0.506 · MERT 0.462** — both **below** the
majority-class baseline **0.611** (always-HVHA). Per-quadrant recall: HVHA 0.63,
HVLA 0.55, LVHA 0.41, Sad 0.17. The ante-hoc 4-prototype feature is an
interpretability readout, **not** an accurate classifier (it loses to the trivial
baseline — class imbalance + continuous emotion). Reported honestly, not as a win.

**Naive-baseline result (validates the training):** SupCR fine-tuning lifts
Precision@5 from **0.485 → 0.58 (+≈0.10, ~+19% relative)** over raw average-pooled
MERT — the trained space measurably retrieves more emotionally-similar neighbours.
**Honest nuance:** the *untrained* space has a higher Silhouette (0.100 vs ≈0).
Raw MERT keeps coarse quadrant blobbiness (likely genre/acoustic), but worse
fine-grained V-A retrieval; SupCR reorganizes around *continuous* V-A proximity —
higher Precision@k, flatter discrete clusters. This reinforces that Precision@k is
the right metric and emotion is a continuous gradient. Details + interpretation:
`03_phaseC_explainability.md`, §5.

## 4. SOTA Comparison (PMEmo 2019)

| Method | Year | Approach | Valence R² | Arousal R² | CCC (V / A) |
| :-- | :-: | :-- | :-: | :-: | :-: |
| PMEmo Original (Zhang et al.) | 2019 | IS13 hand-crafted + ML | 0.42 | 0.51 | — |
| Dutta & Chanda | 2021 | Mel-Spec + CRNN | ~0.45 | ~0.55 | — |
| Hybrid SSL (Wu et al.) | 2023 | Wav2Vec2 + attention | ~0.48 | ~0.61 | — |
| Damer | 2025 | Wav2Vec2 + chords | 0.51 | 0.72 | — |
| Music2Emo | 2025 | MERT + multitask | 0.54 | 0.78 | — |
| **This work — Triple (best Valence)** | 2026 | MERT + wav2vec2 + mel-CNN | **0.576** | 0.702 | 0.73 / 0.82 |
| **This work — Enhanced (best Arousal)** | 2026 | MERT + wav2vec2 + tempo/key | 0.569 | **0.718** | 0.73 / 0.83 |

**Positioning:** Valence R² 0.576 surpasses Music2Emo (0.54) **audio-only, without
multitask supervision**. CCC Arousal 0.8543 (MERT+EDA) is strong, but a direct CCC
comparison to prior PMEmo work is **not possible** — the listed baselines do not
report CCC (column "—"), so "highest reported CCC" cannot be substantiated and is
*not* claimed. Valence stays ≤ 0.58 — the field-wide audio-only ceiling. CCC is the
AVEC standard (penalizes correlation, mean shift, and variance mismatch jointly).

## 5. One-Line Summary

> Audio-only emotion prediction reaches the field ceiling (Arousal R² 0.72,
> Valence R² 0.58); the latent space supports emotionally coherent retrieval
> (Precision@5 ≈ 0.58); and the gains, redundancies, and negative results are all
> reported with explicit, honest caveats.
