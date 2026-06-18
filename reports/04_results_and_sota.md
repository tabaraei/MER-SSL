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
| Harmonic mode | Major / Minor | accuracy | 0.673 |
| Tempo | BPM | R² | **−0.83** (best layer) / −2.12 (pooled) |
| Full per-layer sweep | 8 music-theory features × 25 layers | R²/acc | gaps = **{tempo, key}** |

Mode = **0.673** (Krumhansl–Schmuckler, best layer 11). The earlier "~100%" figure was a
**discarded labelling artifact** (degenerate mean-chroma threshold, `probe_key.py:40`) and
must not be cited as a result — see `01_phaseA_probing.md` §2.
Tempo = **negative R²** from the audited per-layer Ridge sweep (best layer −0.8307,
all-layers-pooled −2.1155; `music_theory_probing_results.json`): the probe does *worse than
predicting the mean tempo*, so tempo is not linearly recoverable anywhere in MERT. The earlier
"R² ≈ 0.12" is a **different probe** (Adam-trained `Linear(1024→1)` on the single pre-pooled
embedding, `probe_tempo.py`; reproduced 2026-06-17 = 0.1191,
`run_logs/phaseA_probe_tempo_single_embedding.log`) — a real but narrower, more favourable
setup (different input + implicit shrinkage from short Adam training). It is **superseded as the
headline** by the systematic all-layer negative value; both agree tempo is a gap.
Gap criterion: best-layer R² < 0.40 (regression) or accuracy < 0.65 (classification).
Key/mode via Krumhansl–Schmuckler. Details: `01_phaseA_probing.md`.

## 2. Phase B — Emotion Prediction (5-fold CV)

| Configuration | Arousal R² | Valence R² | CCC Arousal | CCC Valence |
| :-- | :-: | :-: | :-: | :-: |
| Mel-CNN only (shallow CNN, no SSL) | 0.6486 | 0.4452 | 0.7892 | 0.6302 |
| wav2vec2 only (speech SSL) | 0.6225 | 0.4825 | 0.77 | 0.66 |
| MERT — last layer only (no fusion) | 0.6570 | 0.5162 | 0.81 | 0.70 |
| MERT only (music SSL, 25-layer fusion) | 0.6518 | 0.5055 | 0.82 | 0.74 |
| MERT + EDA | 0.6738 | 0.5075 | **0.8543** | 0.7692 |
| Dual-SSL (MERT + wav2vec2, β=0.05) | 0.6814 | 0.5676 | 0.8087 | 0.7231 |
| Triple (MERT + wav2vec2 + mel-CNN) | 0.7023 | **0.5758** | 0.8233 | 0.7329 |
| Spec-only (MERT + mel-CNN) | 0.7069 | 0.5709 | 0.8271 | 0.7314 |
| Triple-bio (MERT + mel-CNN + EDA) | 0.7077 | 0.5706 | 0.8262 | 0.7325 |
| **Enhanced (MERT + wav2vec2 + tempo/key)** | **0.7182** | 0.5686 | 0.8345 | 0.7259 |

- **Best Valence:** Triple (0.5758) · **Best Arousal:** Enhanced (0.7182) · **Best CCC Arousal:** MERT+EDA (0.8543).
- **Honesty note on the valence lead:** Triple (0.5758), Spec-only (0.5709) and Triple-bio (0.5706) differ by **< 0.005**, which is *inside* the per-fold std (~0.04–0.05). Triple's "best valence" is therefore a **nominal lead within noise**, not a statistically separable win — the honest claim is "the MERT + mel-CNN family tops out at V R² ≈ 0.57, and Triple is the nominal best." Enhanced's arousal lead (0.7182 vs next-best 0.7077) is likewise small but consistent across folds.
- Enhanced ≈ Triple-bio ≈ Spec-only ≈ Triple on arousal (0.7023–0.7182) → second-branch identity barely matters once a trainable spectral branch is present; wav2vec2, EDA and music-theory features are all roughly interchangeable as the second/third branch beside MERT+Mel.
- **Single-encoder comparisons (three baselines):**
  - **MERT (music SSL):** 0.6518 / 0.5055 — best valence among single encoders, justifying the MERT backbone.
  - **wav2vec2 (speech SSL):** 0.6225 / 0.4825 — worst on valence; music-specific pre-training matters.
  - **Mel-CNN alone (shallow CNN, no SSL):** 0.6486 / 0.4452 — *beats* wav2vec2 on arousal (energy is acoustically easy: loudness, spectral envelope) but *loses* on valence (no harmony/tonality model). Cleanly localises SSL's value to the valence axis.
- **EDA contribution is conditional:** EDA helps when added to MERT alone (MERT+EDA valence CCC 0.7692 — best on that axis) but adds essentially nothing on top of MERT+Mel (Triple-bio 0.7077 / 0.5706 ≈ Spec-only 0.7069 / 0.5709, within fold-noise). EDA and the spectral CNN cover overlapping information — the same redundancy story as wav2vec2 once MERT is present.

### 2a. Fusion-collapse interventions (dual encoder)

| Intervention | MERT spec. | w2v spec. | Valence R² |
| :-- | :-: | :-: | :-: |
| Dual β=0 (no penalty) | ~0% | ~0% | 0.5601 |
| Dual + 256-d bottleneck, β=0.01 | 0.2% | 1.8% | 0.4903 |
| Dual + entropy penalty β=0.05 | 0.0% | 0.0% | 0.5676 |

Single-encoder MERT, for contrast: entropy 3.2178 / max 3.2189 (layers 14/16/17).
The bottleneck *hurt* (256-d too aggressive for ~600 samples); the penalty was a
mild regularizer that did not induce specialization.

**Ablation — last-layer-only vs full 25-layer fusion (two settings, 5-fold):**

| Setting | Variant | R² A | R² V | CCC A | CCC V |
| :-- | :-- | :-: | :-: | :-: | :-: |
| MERT-only (single)    | 25-layer fusion   | 0.6518 | 0.5055 | 0.82 | 0.74 |
| MERT-only (single)    | Last layer only   | 0.6570 ± 0.026 | 0.5162 ± 0.053 | 0.81 | 0.70 |
| MERT-only (single)    | Δ (last − fusion) | +0.005 (tied) | +0.011 (tied) | −0.01 | −0.04 |
| Enhanced (multi-enc.) | 25-layer fusion   | **0.7182** | **0.5686** | **0.8345** | **0.7259** |
| Enhanced (multi-enc.) | Last layer only   | 0.6660 ± 0.035 | 0.4881 ± 0.070 | 0.8124 | 0.6865 |
| Enhanced (multi-enc.) | Δ (last − fusion) | **−0.052** (loss) | **−0.081** (loss) | −0.022 | −0.039 |

**Nuanced, quantitative result.** On MERT-only, last-layer is statistically tied
with the 25-layer fusion (Δ inside fold-std on both axes) — consistent with the
near-uniform learned fusion weights (entropy 3.218 / max 3.219). On the
multi-encoder Enhanced model, removing fusion costs **5.2 pp R² A** and
**8.1 pp R² V** — outside fold-std on both axes (Δ/std ≈ 1.5 and 1.2).

**Interpretation.** The fusion's value is not intrinsic to MERT — it is
specifically about *cross-encoder integration*. In single-encoder mode the late
layer carries enough signal that the learnable softmax has nothing to add. In
multi-encoder mode, mel-CNN / wav2vec2 / theory already cover the late acoustic
representation, so MERT's mid-layers fill a complementary niche that the last
layer alone cannot reach. The architectural choice is empirically justified by
the Enhanced ablation; the MERT-only ablation is the negative control that
proves the gain is not free. Scripts: `phaseB/eval_last_layer_only.py`,
`phaseB/eval_enhanced_last_layer.py`.

### 2a-bis. Loss-function ablation (single-MERT, test-fold, 100 epochs)

| Loss config | CCC A | CCC V | P@5 | Silhouette |
| :-- | :-: | :-: | :-: | :-: |
| MSE only | 0.6861 | 0.5955 | 0.5259 | +0.0124 |
| + CCC + Rank (no SupCR) | 0.7814 | 0.7113 | 0.5398 | +0.0206 |
| + SupCR (full hybrid) | 0.8165 | 0.7110 | 0.5734 | −0.0311 |

- CCC+Rank vs MSE: **+0.095 CCC A, +0.116 CCC V** → the dominant contribution; justifies the non-MSE terms decisively.
- SupCR vs no-SupCR: +0.034 P@5, +0.035 CCC A (helps retrieval) but Silhouette **drops** (+0.021 → −0.031) → SupCR improves local retrieval, **not** clustering; it refutes the "SupCR creates emotional clusters" claim.
- Full vs MSE: +0.13 CCC A, +0.12 CCC V, +0.047 P@5 → the hybrid loss is justified overall. No config clusters (all Silhouettes ≈ 0) *(in-sample loss-ablation values; canonical held-out Silhouette is ≈0.19 Euclidean / ≈0.26 cosine, §2a-sexies)*.

### 2a-ter. Imbalance-handling ablation (MERT-only, 25-layer fusion, 5-fold)

**Pre-registered pass mark:** treatment wins iff R² beats baseline by >1 fold-std
on both axes, OR >2 fold-std on either axis, OR lifts minority quadrants from <0
to ≥0.

| Treatment | R² A | R² V | CCC A | CCC V |
| :-- | :-: | :-: | :-: | :-: |
| A: sampler-only (current baseline) | **0.6951 ± 0.016** | **0.5724 ± 0.050** | 0.820 | 0.731 |
| B: weighted-MSE only (no sampler) | 0.6738 ± 0.018 | 0.5267 ± 0.109 | 0.828 | 0.739 |
| C: sampler + weighted-MSE (stacked) | 0.6855 ± 0.029 | 0.5696 ± 0.062 | 0.811 | 0.724 |
| D: focal-MSE γ=2 + sampler | 0.6949 ± 0.018 | 0.5716 ± 0.057 | 0.821 | 0.727 |

- **No winner.** B is *worse* on R² V (−0.046, larger than fold-std). C and D are
  statistically tied with A. Per the pass mark: do not propagate to multi-encoder configs.
- The `WeightedRandomSampler` is empirically the best imbalance treatment we have at
  this dataset scale; gradient-level reweighting adds nothing measurable.
- **Honest disclosure:** within-script baseline rerun A (0.6951/0.5724) ran higher
  than the historic published MERT-only number (0.6518/0.5055). KFold seed is
  identical (random_state=42); model-init/DataLoader RNG are not seeded, so this
  most likely reflects single-seed init variance. The fold-matched A↔B↔C↔D
  comparison within this script is the only audit-honest reading; SOTA-table
  number is not overwritten. Script: `phaseB/eval_imbalance_ablation.py`.
- **Interpretation:** the minority-quadrant failure (negative per-quadrant R²) is
  a *dataset-size* limit (n=64–67 per minority quadrant), not a *method* limit.
  Re-weighting cannot create signal where there is none; the honest cure would be
  augmentation (mixup within minority quadrants, SpecAugment) or a larger
  affect-annotated corpus — neither in scope. We now have a documented ablation
  supporting the sampler choice instead of presenting it as an arbitrary default.

### 2a-quater. Augmentation ablation — feature-space mixup on Enhanced (5-fold)

**Pre-registered pass mark:** treatment wins iff R² beats baseline by >1 fold-std
on both axes, OR >2 fold-std on either, OR lifts minority quadrants from <0 to ≥0.

| Model | R² A | R² V | CCC A | CCC V |
| :-- | :-: | :-: | :-: | :-: |
| Enhanced (no augmentation) | **0.7182** | **0.5686** | **0.8345** | **0.7259** |
| Enhanced + feature-space mixup (α=0.4) | 0.7077 ± 0.014 | 0.5651 ± 0.034 | 0.8213 | 0.7160 |
| Δ (mixup − baseline) | −0.0105 (<std) | −0.0035 (<std) | −0.013 | −0.010 |

- **No winner.** All four deltas land inside fold-noise — mixup is statistically
  tied with the no-augmentation baseline.
- Minority per-quadrant R² remained negative across all three minority quadrants:
  HVLA n=67 A=−0.35 V=−1.88 · LVHA n=64 A=−0.91 V=−1.08 · LVLA n=167 A=−0.11 V=−0.90.
- **Conclusion: the dataset-size floor is empirically confirmed.** The standard
  cited augmentation remedy (Zhang et al. 2017, *mixup*) ties baseline and does
  not lift minorities into the positive range — exactly what the n=64–67
  hypothesis predicts. The minority-quadrant failure is a *data* limit, not a
  *method* limit. Long-term remedy is corpus expansion or task-aware augmentation
  (C-Mixup, Yao 2022; SpecAugment, Park 2019); out of scope for this thesis.
- Script: `phaseB/eval_enhanced_mixup.py`.

### 2a-quinquies. Cluster-enforcement trade-off — auxiliary quadrant-CE head sweep (Enhanced, 5-fold)

**Pre-registered hypothesis:** Silhouette ↑ with λ; R²/CCC ↓ with λ;
minority per-quadrant R² ≈ unchanged (data-floor invariant).

|  λ   |  R² A           |  R² V           | CCC A | CCC V |  Silhouette       |
| :-:  | :-:             | :-:             | :-:   | :-:   |  :-:              |
| 0.0  | 0.7080 ± 0.021  | 0.5725 ± 0.038  | 0.827 | 0.729 | **0.255 ± 0.054** |
| 0.1  | 0.7050 ± 0.013  | 0.5811 ± 0.033  | 0.828 | 0.738 |   0.258 ± 0.059   |
| 0.5  | 0.6966 ± 0.016  | 0.5624 ± 0.053  | 0.827 | 0.730 |   0.265 ± 0.064   |
| 1.0  | 0.6638 ± 0.020  | 0.5601 ± 0.037  | 0.810 | 0.734 |   0.286 ± 0.060   |

- **Side-finding (corrected — see §2a-sexies for the canonical measurement):**
  the Enhanced model's latent space has **weak-to-moderate** quadrant structure —
  Silhouette = 0.255 (cosine, 5-fold held-out). A dedicated matched audit
  (§2a-sexies) shows **single-MERT scores the same** (cosine 0.269) — so this
  structure is **not** specific to the multi-encoder setup, contrary to an earlier
  draft of this note. The "Silhouette ≈ 0" figures quoted elsewhere came from
  *in-sample* measurements on older saved indices and are superseded by the
  matched held-out protocol.
- **Trade-off is real but unfavourable.** Silhouette rises monotonically
  with λ (+0.031 from λ=0 to λ=1.0 — modest), but R² A drops 4.4 pp at
  λ=1.0 (outside fold-std). λ=0.1 is essentially zero-cost on R² and also
  zero-benefit on Silhouette.
- **Minority per-quadrant R² remain negative across all λ** — and
  *worsen* with stronger CE (HVLA −0.43 → −0.85). Third independent
  confirmation of the dataset-floor hypothesis (after §2a-ter imbalance
  and §2a-quater mixup).
- **Conclusion:** continuous representation is the empirically-defended
  design choice. Cluster enforcement is possible but pays for itself
  poorly. Script: `phaseB/eval_enhanced_quadrant_ce.py`.

### 2a-sexies. Silhouette — canonical measurement (resolves the "≈0 vs 0.255" contradiction)

The Silhouette score appeared inconsistently across earlier drafts (≈0 in some
places, 0.255 in others). A dedicated matched audit settles it: **both** single-MERT
and the multi-encoder Enhanced model were trained identically (HybridLoss, balanced
sampler, differential optimizer, 5-fold KFold(42)) and Silhouette was computed on
the **held-out test-fold latents** under **both** distance metrics:

| Model       | Silhouette (Euclidean) | Silhouette (cosine) |
| :--         | :-:                    | :-:                 |
| single-MERT | 0.1934 ± 0.043         | 0.2691 ± 0.064      |
| Enhanced    | 0.1815 ± 0.035         | 0.2595 ± 0.054      |

**What this resolves:**
1. **Model effect ≈ 0.** single-MERT and Enhanced are statistically
   indistinguishable (cosine Δ = −0.010, inside fold-std). Quadrant structure is
   **not** a multi-encoder property — the earlier "multi-encoder is more clustered"
   note (§2a-quinquies) is corrected.
2. **Metric effect ≈ +0.08.** cosine reads ~0.08 higher than Euclidean on the same
   latents — part of the historic "≈0 vs 0.255" gap was simply *metric choice*.
3. **Neither model is ≈0.** Under matched held-out evaluation the trained space
   carries **weak-to-moderate** quadrant structure (~0.19–0.27), confirmed
   independently by the CE-sweep λ=0 row (Enhanced cosine 0.255 ≈ 0.260 here).
4. **The historic "≈0" is superseded.** Those figures came from *in-sample*
   measurements on older saved retrieval indices (`analyze.py`, the Phase C
   `prototypes_dual.npy` index), not from this matched held-out protocol.

**Is the low Silhouette an *architectural* limit or an *intrinsic* property of emotion?**
To rule out "the architecture just can't separate quadrants", we trained the **same
encoder** with an explicit quadrant-**classification + separation** objective (the Audio
ProtoPNet, §3) and measured Silhouette on held-out latents under the identical protocol:

| Model (objective) | Silhouette (Euclidean) | Silhouette (cosine) |
| :-- | :-: | :-: |
| single-MERT (regression, SupCR) | 0.1934 ± 0.043 | 0.2691 ± 0.064 |
| Enhanced (regression, SupCR) | 0.1815 ± 0.035 | 0.2595 ± 0.054 |
| ProtoPNet (classification + separation) | 0.1181 ± 0.055 | 0.1778 ± 0.082 |
| *(ref)* CE-head sweep, λ=1.0 max | — | 0.286 ± 0.060 |

**The result is decisive — and counter-intuitive.** A model trained *explicitly* to
separate the four quadrants, and which *succeeds* at classification (74% raw / 0.57
balanced), produces a latent that is **no more quadrant-clustered than the regression
model — if anything slightly less** (cosine 0.178 vs 0.260). The reason: ProtoPNet uses
5 prototypes per class, so a quadrant can be recognised across several scattered regions
— accurate classification never required each quadrant to be one compact blob.
**Classifiability and cluster-separation are therefore decoupled.** Across *three*
independent objectives — regression+SupCR (0.26), an auxiliary quadrant-CE head pushed to
λ=1.0 (0.29 max), and full ProtoPNet classification (0.18) — Silhouette stays in a tight
**0.18–0.29 band, never approaching the ≳0.5 of clean clusters.**

**Canonical statement for the thesis:** *the trained latent space has weak-to-moderate
quadrant structure (Silhouette ≈ 0.19 Euclidean / ≈ 0.26 cosine, held-out, single-MERT
≈ Enhanced) — a structured continuum, not four discrete clusters and not a featureless
blob. This low-but-non-zero value is **intrinsic to the affective geometry, not an
architectural limitation**: even a model optimised to classify the quadrants does not
produce cleanly separated clusters (Silhouette 0.18 at 74% accuracy), and explicitly
forcing compactness (CE-head) raises Silhouette only marginally (0.26→0.29) at a cost to
regression accuracy. The four Russell quadrants are analytical bins on a continuous V-A
gradient, exactly as the circumplex model predicts.* Precision@5 ≈ 0.58 remains the
primary, protocol-robust evidence of emotional organization. Scripts:
`phaseB/eval_silhouette_audit.py`, `phaseB/eval_protopnet_silhouette.py`;
logs in `reports/run_logs/`.

### 2a-septies. Cyclic key encoding A/B (does fixing the geometry rescue valence?)

Phase A flagged `key` as a gap; the Enhanced model re-injected it, but as a raw
integer 0–11 — which destroys circular geometry (C=0 and B=11 are adjacent keys yet
numerically maximally distant). We long assumed this encoding was *why* key didn't
help valence (+0.001). We tested it directly: same Enhanced model, same 5-fold
splits, only the key encoding differs (raw 1-d vs cyclic `[sin(2πk/12), cos(2πk/12)]`
2-d).

| Key encoding | R² A | R² V | CCC A | CCC V |
| :-- | :-: | :-: | :-: | :-: |
| Raw integer 0–11 | 0.7049 ± 0.019 | 0.5777 ± 0.039 | 0.825 | 0.733 |
| Cyclic sin/cos | 0.7016 ± 0.010 | 0.5697 ± 0.040 | 0.826 | 0.732 |
| Δ (cyclic − raw) | −0.003 | **−0.008 (inside fold-std)** | +0.001 | −0.001 |

- **Result: null.** Cyclic encoding does **not** rescue valence — ΔV = −0.008 is inside
  the per-fold std (~0.04). **This falsifies our earlier hypothesis** that the raw-integer
  encoding was the bottleneck. The corrected geometry was necessary for correctness but
  was not the limiting factor.
- **Sharper, more defensible conclusion:** the key→valence link is itself too weak to
  exploit at this data scale, and/or MERT + mel-CNN already capture the harmonic
  information that `key` would supply (consistent with chroma/mode being *non-gaps* in
  Phase A). The bottleneck is the feature's relationship to valence and the dataset size
  — not the encoding. The cyclic encoder is retained as the *correct* implementation
  (`models_enhanced.build_gap_vector(..., cyclic_key=True)`); it simply doesn't move the
  needle here. Script: `phaseB/eval_key_encoding.py`.

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

| Space | Precision@5 | Precision@10 | Precision@20 | Silhouette (pooled) |
| :-- | :-: | :-: | :-: | :-: |
| **Naive raw last-layer MERT** (untrained baseline) | 0.4847 | 0.4614 | 0.4553 | **0.1001** |
| MERT (single, SupCR) | 0.5760 | 0.5687 | 0.5469 | −0.0293 |
| Dual-SSL (MERT + wav2vec2, SupCR) | 0.5849 | 0.5613 | 0.5382 | +0.0026 |
| **Enhanced (MERT + w2v2 + cyclic key)** — *benchmark / deployed* | **0.5943** | **0.5761** | **0.5477** | 0.0039 |

Precision@k = mean fraction of top-k neighbors within a 0.20 V-A radius.
**Random-chance baseline = 0.276.** The **Enhanced** model — the architecture actually
deployed in Phase C (§3 of `03_phaseC_explainability.md`) — is the **best retriever**
(Precision@5 = 0.594, +0.009 over Dual, ~2.15× chance), so the benchmark model is best at
*both* regression (arousal R² 0.7182) and retrieval. All rows use the identical 5-fold
out-of-sample protocol.

> **Silhouette caveat — this column is the POOLED-models artifact, NOT the canonical number.**
> The Silhouette here (untrained 0.100, MERT −0.029, Dual +0.003, Enhanced +0.004) is
> computed on the retrieval index, which **pools test-fold latents from all 5 separately-
> trained fold models** into one array. Those 5 models' latent spaces are independently
> rotated, so a *global* Silhouette over the pool collapses to ≈0 — a measurement artifact
> of pooling, not a property of any model. Precision@k is robust to this (it is a *local*
> V-A-neighbourhood metric) and stays ~0.58–0.59; Silhouette (a *global* cohesion/separation
> ratio) is not. The **canonical** Silhouette, computed **per-fold within a single model**
> (§2a-sexies), is **≈ 0.19 Euclidean / ≈ 0.26 cosine** (single-MERT ≈ Enhanced). This
> reconciles the long-standing "≈0 vs 0.26" tension: ≈0 = pooled-5-models retrieval index;
> 0.26 = per-model held-out. Either way it is far below the ≳0.5 of clean clusters, so the
> "continuous emotion, not 4 separable clusters" reading holds. Precision@5 ≈ 0.58–0.59
> remains the primary,
> protocol-robust evidence.

**Prototype-activation accuracy — post-hoc 4-centroid (superseded by ProtoPNet below).**
(% of songs whose best-match quadrant centroid = true quadrant, leave-one-out):
**Dual 0.506 · MERT 0.462** — both **below** the majority-class baseline **0.611**
(always-HVHA). Per-quadrant recall: HVHA 0.63, HVLA 0.55, LVHA 0.41, Sad 0.17. With
*fixed* centroids computed after training, the ante-hoc readout lost to the trivial
baseline. This motivated replacing it with a *learnable*-prototype network.

**Audio ProtoPNet — learnable prototypes (replaces the post-hoc centroid).**
A ProtoPNet (Chen et al. 2019) learns prototype vectors *during* gradient descent and
classifies by L2 distance to them (5 prototypes/quadrant = 20 total; cluster +
separation + L1 losses; balanced sampler; MERT backbone; 5-fold held-out). Result:

| Quadrant classifier | Raw acc | Balanced acc | Protocol |
| :-- | :-: | :-: | :-- |
| Majority (always-HVHA) | 0.611 | 0.250 | trivial baseline |
| Post-hoc 4-centroid (MERT) | 0.462 | — | in-sample LOO |
| Post-hoc 4-centroid (dual) | 0.506 | — | in-sample LOO |
| **Audio ProtoPNet (MERT)** | **0.728 ± 0.031** | **0.545 ± 0.041** | held-out 5-fold |

- **ProtoPNet beats both baselines** — +0.22 over the post-hoc centroid and **+0.12 over
  the majority baseline** (raw acc), under the *more rigorous* held-out protocol (vs the
  centroid's in-sample LOO). Balanced accuracy 0.545 is >2× the majority-balanced 0.250.
- **Minority quadrants improve dramatically:** Sad recall **0.17 → 0.69**, and Calm/Angry
  go from a fixed-centroid collapse to 0.36/0.27. Per-quadrant recall: HVHA 0.857,
  HVLA 0.358, LVHA 0.266, LVLA 0.689.
- **Why it works:** learning the prototypes jointly with the encoder (rather than freezing
  K-means-style centroids after training) lets the prototypes sit where the classes are
  *actually* separable, and the separation loss pushes them apart. It remains **ante-hoc
  interpretable** — each prototype is, by construction, evidence for one quadrant.
- Scripts: `phaseB/models_protopnet.py`, `phaseB/train_protopnet.py`. This upgrades the
  Phase C prototype-explanation component; it does not change the V-A regression numbers.

**Naive-baseline result (validates the training):** SupCR fine-tuning lifts
Precision@5 from **0.485 → 0.58 (+≈0.10, ~+19% relative)** over raw average-pooled
MERT — the trained space measurably retrieves more emotionally-similar neighbours.
**Honest nuance:** the *untrained* space has a higher Silhouette (0.100 vs ≈0).
Raw MERT keeps coarse quadrant blobbiness (likely genre/acoustic), but worse
fine-grained V-A retrieval; SupCR reorganizes around *continuous* V-A proximity —
higher Precision@k, flatter discrete clusters. This reinforces that Precision@k is
the right metric and emotion is a continuous gradient. Details + interpretation:
`03_phaseC_explainability.md`, §5.

## 4. PMEmo 2019 — Single Comparison Table (R²)

One unified table: prior work (top) + every model we built (bottom). R² is the common
metric across these papers. PMEmo works that report only classification accuracy or
RMSE (DAMER, CNN+LSTM, Sharma) are **not** R²-comparable and are omitted, not mixed in.
Confirm external numbers against primary sources before submission; EmoMucs used labels
scaled to [-1,1].

| Method / Architecture | Year | R² Valence | R² Arousal | Notes |
| :-- | :-: | :-: | :-: | :-- |
| Zhang et al. — IS13 + SVR/MLR (PMEmo paper) | 2018 | ~0.41* | ~0.52* | baselines (*r²-derived) |
| EmoMucs C1D-M (de Berardinis et al.) | 2020 | 0.349 | 0.557 | 1D CNN; labels [-1,1] |
| EmoMucs C2D-M (de Berardinis et al.) | 2020 | 0.414 | 0.610 | 2D CNN; labels [-1,1] |
| AutoML openSMILE (Simonetta et al.) | 2024 | 0.525 | 0.727 | hand-crafted; PMEmo only |
| AutoML Joint (Simonetta et al.) | 2024 | 0.780 | 0.861 | PMEmo + IADS-E (joint) |
| Music2Emo (Kang & Herremans) | 2025 | 0.536 | 0.777 | MERT+chord/key; PMEmo only |
| Music2Emo (Kang & Herremans) | 2025 | 0.547 | 0.794 | + multitask; 4 datasets |
| This work — Mel-CNN only | 2026 | 0.4452 | 0.6486 | shallow CNN, no SSL (single-encoder baseline) |
| This work — wav2vec2 only | 2026 | 0.4825 | 0.6225 | speech SSL, single encoder |
| This work — MERT only | 2026 | 0.5055 | 0.6518 | music SSL, single encoder |
| This work — MERT + EDA | 2026 | 0.5075 | 0.6738 | + physiology (CCC A 0.85) |
| This work — Dual (MERT+wav2vec2) | 2026 | 0.5676 | 0.6814 | dual SSL |
| **This work — Triple (+ mel-CNN)** | 2026 | **0.576** | 0.702 | best valence |
| This work — Spec-only (MERT+Mel) | 2026 | 0.5709 | 0.7069 | drops wav2vec2 from Triple |
| This work — Triple-bio (MERT+Mel+EDA) | 2026 | 0.5706 | 0.7077 | swaps wav2vec2 for EDA |
| **This work — Enhanced (+ tempo/key)** | 2026 | 0.569 | **0.718** | best arousal |

*\*Zhang et al. (2018) report Pearson's r; values are squared approximations (R² ≈ r²) from
their static baselines (r=0.638 valence → 0.41; r=0.719 arousal → 0.52).*

**Positioning (honest, with caveats):** our genuine strength is **valence** — our Triple
(0.576) is the best of all non-joint methods, above both Music2Emo variants (0.536 single,
0.547 multitask) and hand-crafted AutoML (0.525). On **arousal** we do **not** lead:
Music2Emo reaches 0.777 (single) / 0.794 (multitask) and AutoML 0.727 — all above our best
(0.718). Arousal is competitive-but-trailing, not a headline. The top row, Simonetta
AutoML-**Joint** (0.780/0.861, PMEmo+IADS-E), is the exact approach we replicated with SSL
(§2b) and it **did not transfer** (our SSL valence dropped) — it both sets the SOTA bar and
frames our key negative finding. Within our own models, music-pretrained MERT cleanly beats
speech-pretrained wav2vec2. **Verified:** Music2Emo PMEmo-only = 0.536/0.777 (paper Table III,
arXiv:2502.03979) — an earlier draft mis-listed 0.458/0.639; corrected.

**Not R²-comparable (accuracy/RMSE — do NOT tabulate as R²):** CNN+LSTM (2022) ≈79% V /
84% A acc; DAMER (2025) ≈78% V / 86% A acc; Sharma et al. (2020) ≈63% acc; original
Zhang et al. (2018) reports RMSE. A prior draft mis-listed DAMER as "R² 0.51/0.72" —
corrected (it is accuracy). No fair CCC comparison exists (prior work doesn't report CCC).

## 5. One-Line Summary

> Audio-only emotion prediction reaches the field ceiling (Arousal R² 0.72,
> Valence R² 0.58); the latent space supports emotionally coherent retrieval
> (Precision@5 ≈ 0.58); and the gains, redundancies, and negative results are all
> reported with explicit, honest caveats.
