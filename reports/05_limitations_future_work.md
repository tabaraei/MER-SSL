# 05 — Limitations & Future Work

Consolidated, thesis-honest limitations and the next steps each implies. These are
strengths to state plainly, not weaknesses to hide — much of the project's value is
in the rigor of these caveats.

---

## 0. Honest Self-Critique — What Did Not Work

This section states the project's genuine failures plainly. They are reported as
findings, not hidden. Recommended thesis framing: this work is best presented as a
**rigorous empirical audit of what SSL audio embeddings do and do not encode for
emotion** — under that framing the negatives below are the contribution. Presented
as "a SOTA explainable emotion-clustered retrieval system," the results do not
fully support the claim.

1. **The latent space is not emotionally clustered (core Phase C premise only
   weakly holds) — and the loss ablation proves SupCR is not the cause of any
   clustering.** Silhouette ≈ 0 for the trained space and *lower* than raw untrained
   MERT (0.10 → ≈0); the t-SNE (`artifacts/tsne_baseline_vs_finetuned.png`) shows
   clusters that mix all four emotions, dominated by HVHA. **The loss ablation is
   decisive: removing SupCR *raised* Silhouette (−0.031 → +0.021)** — i.e. SupCR
   actively lowers the quadrant-cluster score. SupCR produces *local* coherence
   (Precision@5 +0.034, CCC Arousal +0.035) but it pulls songs together by
   continuous V-A proximity, not into discrete clusters. The original "SupCR
   organizes the space into emotionally coherent clusters" hypothesis is therefore
   **refuted by our own ablation** — SupCR earns its place via retrieval quality, not
   clustering. This is the most important honest correction to the original claim.

2. **The multi-encoder program mostly produced null/negative results.** Dual added
   little (and collapsed the fusion); wav2vec2 was shown redundant; IADS-E transfer
   was negative. The real arousal gains (0.65 → 0.72) came from the trainable
   **mel-CNN and the explicit tempo feature** — not from the SSL-fusion ideas that
   frame the thesis. The genuine wins are narrower than the architecture suggests.

3. **The WeightedLayerFusion "interpretability" claim is undermined by its own
   data.** Learned weights are near-uniform (entropy 3.218 / max 3.219) and the
   "dominant" top layers vary across runs (10, 12, 14, 15, 16). "Layers 14/16/17
   dominate" is a marginal, unstable argmax — it should be reported as a *faint
   lean toward mid-to-late layers*, not a strong, interpretable preference.

4. **Class imbalance was never solved.** Per-quadrant R² is negative for
   Calm/Angry/Sad across *every* configuration; the model effectively only works on
   the majority Happy quadrant. This directly weakens the stated therapy/clinical
   motivation, which depends on the minority emotions. The balanced sampler is a
   partial mitigation, not a fix.

5. **The explainability is largely descriptive, not model-faithful.** librosa
   music-theory annotations describe the *song*, not the model's computation; the
   LLM layer is presentation; the 4 prototype centroids are computed from data, not
   learned parameters. The ante-hoc core (k-NN + centroids) rests on a latent space
   that is only weakly organized (point 1), so the XAI is weaker than
   "ante-hoc explainable MER" implies.

   **Quantified:** the 4-centroid prototype-activation accuracy is **0.506 (dual) /
   0.462 (MERT)** — *below* the trivial majority-class baseline of **0.611**. So the
   ante-hoc classifier the supervisor requested exists in *form* (per-prototype
   activation + argmax) but underperforms "always guess Happy"; Sad recall is only
   0.17. It is honestly an interpretability readout, not an accurate classifier.

6. **EDA fusion was marginal** (+0.02 arousal R²) — reported earlier with more
   weight than the effect size justifies.

**Net:** strong arousal prediction (R² 0.72, CCC 0.85), a working explanation
*pipeline*, and several clean, citable negative findings — but the emotion-clustering
and multi-encoder ambitions, and the deep-interpretability claim, were not met.

---

## 1. Limitations

### 1.1 Class imbalance is the dominant ceiling
PMEmo is ~61% HVHA (Happy). Global R²/CCC are carried by this majority; per-quadrant
R² is **negative** for Calm, Angry, and Sad across *every* configuration (see
`04_results_and_sota.md`, §2c). The balanced sampler mitigates but does not solve
this. **This is a dataset property, not a model defect** — and it is the single most
important limitation to foreground.

### 1.2 Valence ceiling (audio-only)
Valence R² ≤ ~0.58 for every audio-only configuration, matching the field-wide
ceiling. Valence depends on lyrics, cultural context, and subjectivity that are not
present in the audio signal (Yang & Chen, 2012).

### 1.3 Fusion collapse needs more data
Layer-selective fusion only specializes in the single-encoder model. In multi-encoder
setups it collapses to uniform weights regardless of intervention — a data-constraint
finding (~600 training songs is too few). Reported as a contribution, but it caps
what multi-encoder layer attribution can claim.

### 1.4 wav2vec2 adds little; second encoders are largely redundant
Spec-only (MERT + mel-CNN) ties Triple; dual ties single-MERT on Phase C metrics.
Speech-pretrained features carry no music-relevant complementary structure here.

### 1.5 Silhouette ≈ 0 (continuous emotion)
Quadrant Silhouette is ≈ 0 for both encoders. This reflects that affect is a
continuous V-A gradient, not 4 separable clusters — Silhouette-by-quadrant is a weak
lens. Precision@k is the valid evidence of latent organization. Do **not** report the
dual's +0.0026 as meaningful separation.

### 1.6 Key encoding is naive
The Enhanced model fed key as a raw integer 0–11, which discards the circular
relationship between keys. This is the likely reason key did not help valence. A
one-hot or sin/cos circular encoding is the honest fix to retest.

### 1.7 Explanation faithfulness is mixed
The retrieval/centroid core is ante-hoc, but the librosa music-theory annotation and
LLM synthesis are descriptive/post-hoc — they describe the song or translate the
decision, not the SSL model's internals. The 4 centroids are computed from data, not
*learned* parameters (not yet a true ProtoPNet).

### 1.8 EDA is population-normalized
EDA features are min-max normalized across the dataset, so the physiological
narrative reflects population-level tendencies, not individual listener response.

### 1.9 No user study; Precision@k is a proxy
Precision@k measures V-A label consistency within a 0.20 radius — a design choice,
not perceptual ground truth. Whether explanations satisfy real listeners is unvalidated.

### 1.10 Minor implementation caveats
- `rhythmic_stability = 1 − std(tempogram)/mean(tempogram)` is unbounded and can go
  negative on real clips (760.mp3 → −0.132); spec-faithful but consider clamping.
- The annotator's `dominant_pitches` (top-3 chroma bins) don't always include the
  K-S tonic — the "supports the analysis" wording can mildly overclaim.
- `mainC.py --mode build/query` loads the model as `MERModel`; the current
  `best_model.pt` is DualSSL and won't load. Use `evaluate_latent_space.py` (handles
  both encoders) for index building/evaluation.

## 2. Future Work

| Direction | Why | Effort |
| :-- | :-- | :-- |
| **Quadrant-weighted / focal loss** | Directly target the class-imbalance ceiling (the real bottleneck) | Medium |
| **Learned prototype vectors (ProtoPNet-style)** | Make the 4 prototypes trained parameters → fully ante-hoc classification, closing the XAI gap the supervisor named | Medium |
| **Circular key encoding (sin/cos)** | Re-test whether key can help valence once encoded sensibly | Low |
| **Pretrained music CNN (MusiCNN / PANNs)** | Replace the from-scratch mel-CNN; tests whether music-domain pretraining beats training on ~600 songs | Medium |
| **Lyrics / multimodal text** | The principled path past the valence ceiling | High |
| **Metadata enrichment (MusicBrainz/Spotify)** | PMEmo has no artist/title/genre → enables artist-specific recommendations | Medium |
| **User study on explanations** | Validate that template/LLM explanations actually build trust | High |
| **Per-query layer attribution (Grad-CAM)** | Move from global to per-song layer attribution | Medium |

## 3. Deadline-Aware Recommendation

The metrics are at the audio-only ceiling and the remaining bottleneck (class
imbalance) is not fixable by modeling in the available time. **The highest-value use
of the remaining week is writing, not more experiments.** The only sub-hour addition
worth doing is computing **prototype-activation accuracy** (% of songs whose
best-match centroid equals their true quadrant), which quantifies the ante-hoc
feature. Everything else is genuinely future work.
