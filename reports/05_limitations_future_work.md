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

1. **The latent space is not *cleanly* emotionally clustered (core Phase C premise
   holds only weakly) — and the loss ablation proves SupCR is not the source of any
   clustering.** The canonical held-out Silhouette is **≈ 0.19 Euclidean / ≈ 0.26
   cosine** (single-MERT ≈ Enhanced; `04_results_and_sota.md` §2a-sexies) —
   *weak-to-moderate*, well below the ≳0.5 that cleanly separable clusters would give.
   (Earlier "≈ 0" figures were in-sample index measurements and are superseded; see the
   §2a-sexies audit.) The t-SNE (`artifacts/tsne_baseline_vs_finetuned.png`) shows an
   organized but continuous manifold that mixes all four emotions, dominated by HVHA —
   a structured continuum, not four separable blobs. **The loss ablation remains
   decisive on the *relative* point: removing SupCR *raised* the (in-sample) Silhouette
   (−0.031 → +0.021)** — i.e. SupCR does not build discrete clusters. SupCR produces
   *local* coherence (Precision@5 +0.034, CCC Arousal +0.035) by pulling songs together
   along continuous V-A proximity, not into discrete clusters. The original "SupCR
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

5. **The explainability is largely descriptive, not model-faithful** (librosa
   music-theory annotations describe the *song*, not the model's computation; the
   LLM layer is presentation) — **but the prototype classifier is now learnable and
   accurate.** The original 4-centroid readout used *fixed* centroids computed after
   training and underperformed: accuracy **0.506 (dual) / 0.462 (MERT)**, *below* the
   majority baseline **0.611**, Sad recall only 0.17. **This has been addressed:** an
   Audio ProtoPNet (learnable prototypes optimised during gradient descent, L2-distance
   classification, cluster+separation losses) reaches **0.728 raw / 0.545 balanced
   accuracy** held-out — **beating both the post-hoc centroid (+0.22) and the majority
   baseline (+0.12)** — with Sad recall up to **0.69** (`04 §3`,
   `phaseB/train_protopnet.py`). The ante-hoc prototype explanation the supervisor asked
   for now exists in *form and substance*. (The k-NN retrieval core still rests on a
   weakly-organized latent space — point 1 — so this strengthens the prototype branch,
   not the retrieval branch.)

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

### 1.5 Silhouette is weak-to-moderate — and *intrinsic*, not an architectural limit
The canonical held-out quadrant Silhouette is **≈ 0.19 Euclidean / ≈ 0.26 cosine**
for both encoders (single-MERT ≈ Enhanced; §2a-sexies of `04_results_and_sota.md`) —
*weak-to-moderate* structure, far below the ≳0.5 of cleanly separable clusters but
clearly non-zero. **This is a property of the affective geometry, not a model defect:**
training the same encoder to *classify* quadrants (Audio ProtoPNet, classification +
separation losses) reaches 74% accuracy yet yields Silhouette 0.18 — no higher than the
regression model — and explicitly forcing compactness (the CE-head sweep) lifts it only
to 0.29 at a cost to regression accuracy. Across regression, auxiliary-CE, and full
classification objectives, Silhouette stays in 0.18–0.29. Affect is a **continuous,
organized** V-A gradient (Russell's circumplex); the four quadrants are analytical bins,
not natural clusters. Silhouette-by-quadrant is therefore a secondary lens; Precision@k
(≈0.58) is the protocol-robust evidence of latent organization. The older in-sample index
figures (≈0, dual +0.0026) are **superseded** by the matched held-out audit.

### 1.6 Key encoding was naive — now fixed and **tested** (encoding was not the bottleneck)
The Enhanced model originally fed key as a raw integer 0–11, discarding the circular
relationship between keys. We long assumed this was *why* key did not help valence.
**We implemented the proper cyclic encoding** (`[sin(2πk/12), cos(2πk/12)]`,
`models_enhanced.build_gap_vector(cyclic_key=True)`) and ran a pre-registered A/B
(`04 §2a-septies`): cyclic vs raw made **no difference** to valence (ΔV = −0.008,
inside fold-std). **This falsifies the encoding hypothesis** — the geometry fix was
necessary for correctness but was never the limiting factor. The real limit is the
weak key→valence relationship at this data scale (and MERT+mel already covering the
harmonic information). Honest, retested, closed.

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

**Completed since the first draft of this section:**
- ✅ **Learned prototype vectors (Audio ProtoPNet)** — done. Beats the post-hoc
  centroid and the majority baseline (0.728 raw / 0.545 balanced; `04 §3`). The
  XAI gap the supervisor named is closed.
- ✅ **Circular key encoding (sin/cos)** — done and tested; null effect (`04 §2a-septies`).
  Encoding was not the valence bottleneck.
- ✅ **Quadrant-weighted / focal loss** — tested in the imbalance ablation (`04 §2a-ter`);
  did not beat the WeightedRandomSampler.

| Direction (still open) | Why | Effort |
| :-- | :-- | :-- |
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
