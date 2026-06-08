# 01 — Phase A: Representation Probing

**Goal:** establish that frozen MERT embeddings preserve musically meaningful
information *before* any emotion fine-tuning — and identify exactly which musical
attributes the representation does and does not linearly expose. This grounds the
decision to use MERT as a feature extractor and motivates Phase B design choices.

**Code:** `phaseA/probe_key.py`, `phaseA/probe_tempo.py`,
`phaseA/extract_music_theory.py`, `phaseA/run_music_theory_probing.py`.

---

## 1. Method — Linear Probing

Linear probing attaches a *simple linear model* to frozen MERT features and tests
whether it can recover a known musical attribute. If a linear probe succeeds, the
information is **explicitly and linearly encoded** in the representation
(Alain & Bengio, 2017). All probes use the same split as the rest of the project:
`train_test_split(test_size=0.2, random_state=42)`.

Two probe families:
- **Regression** (continuous targets): `Ridge(alpha=1.0)`, scored by R².
- **Classification** (mode, key): `LogisticRegression`, scored by accuracy.

## 2. Initial Probes (tempo) — and a discarded early mode probe

| Probe | Target | Result | Reading |
| :-- | :-- | :-: | :-- |
| Tempo | BPM | **R² ≈ 0.12** | Speed is only weakly/coarsely encoded |

> **Discarded early result — do not cite.** An exploratory script (`probe_key.py`)
> reported "~100% accuracy" for major/minor mode. This number is **not valid** and is
> excluded from all findings: the probe's ground-truth label was a degenerate proxy —
> `1 if mean(chroma) > 0.2 else 0` (`probe_key.py:40`) — which thresholds overall chroma
> *magnitude*, not musical mode. Almost every song falls on one side of 0.2, so the label
> is near-constant and a trivial majority-class predictor scores ~100%. The honest mode
> result comes from the proper **Krumhansl–Schmuckler** estimation in §3 below:
> **0.673 accuracy** (best layer 11). The "~100%" was a labelling artifact, not evidence
> that MERT linearly exposes mode.

t-SNE of the raw embeddings shows a smooth gradient by *continuous* valence/arousal
(`mert_tsne_plot.png`, `mert_emotion_clusters.png`), but **not** clean discrete
quadrant clusters. The later quadrant-coloured baseline t-SNE
(`artifacts/tsne_baseline_vs_finetuned.png`, Phase C) confirms this: raw MERT is
one undifferentiated blob with the four quadrants fully mixed. So the honest claim
is "emotional information is present but continuously distributed pre-training,"
not "emotions form separable clusters."

## 3. Full Per-Layer Music-Theory Probing (extension)

To map *where* and *what* musical information lives, every one of MERT's 25 layers
was probed against eight librosa-extracted music-theory features. Ground truth is
saved by `extract_music_theory.py` to `phaseA/data/pmemo_music_theory.pt`.

**Key/mode estimation note:** the original spec referenced `librosa.estimate_key()`,
which **does not exist in any librosa version**. Key and mode are instead estimated
with the canonical **Krumhansl–Schmuckler** algorithm (Krumhansl & Kessler, 1982):
correlate the mean chroma vector against the 24 rotated major/minor key profiles
and take the best match.

**Features probed:** chroma (12-d), spectral_contrast (7-d), spectral_centroid,
zero-crossing rate, tempo, rhythmic_stability (all regression); mode (binary) and
key (12-class) classification.

**Gap criterion:** a feature is a "gap" if its *best-layer* score is below
R² = 0.40 (regression) or accuracy = 0.65 (classification).

### Result — full per-feature probing across all 25 MERT layers

All 8 librosa-extracted features were probed against every one of MERT's 25 layers.
**Best layer** = layer with the highest score; **pooled** = mean-of-all-layers
probe (one Ridge / LogReg trained on the per-layer-averaged representation).
Bold rows are the gaps under the threshold rule; these are the features the
Phase B *Enhanced* model re-injects via the music-theory branch.

| Feature | Probe type | Threshold | Best layer | Best score | Pooled score | Gap? |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: |
| chroma (12-d harmony profile) | Ridge R² | 0.40 | 0 | 0.6801 | 0.5769 | no |
| **tempo (BPM)** | Ridge R² | 0.40 | 0 | **−0.8307** | **−2.1155** | **yes** |
| rhythmic_stability | Ridge R² | 0.40 | 0 | 0.7346 | 0.5630 | no |
| spectral_centroid | Ridge R² | 0.40 | 1 | 0.9206 | 0.8663 | no |
| spectral_contrast (7-d) | Ridge R² | 0.40 | 3 | 0.7310 | 0.7270 | no |
| zero-crossing rate (ZCR) | Ridge R² | 0.40 | 0 | 0.9566 | 0.8471 | no |
| mode (major / minor, binary) | LogReg acc | 0.65 | 11 | 0.6730 | 0.6730 | no (marginal) |
| **key (12-class)** | LogReg acc | 0.65 | 2 | **0.5849** | **0.5786** | **yes** |

**Reading.**
- **Strongly encoded (effortless to probe linearly):** spectral centroid (0.92),
  ZCR (0.96), spectral contrast (0.73), chroma (0.68), rhythmic stability (0.73)
  — short-time spectral and chroma-style information dominates the *early* layers
  (best-layer = 0–3 for all five). This is consistent with the SSL literature
  (Pasad et al. 2021): low-level acoustic features live in early layers.
- **Captured but only marginally:** mode at 0.67 acc (binary classification, just
  above the 0.65 threshold) — MERT knows *major vs minor* a little better than
  chance, and the best layer for mode is 11 (mid layer), not an early acoustic one.
  **This 0.673 is the only valid mode number.** It supersedes the discarded "~100%"
  from the exploratory `probe_key.py` (§2), which used a degenerate proxy label
  (mean-chroma threshold, not actual mode) and therefore measured majority-class
  frequency rather than MERT's encoding of harmony. There is no contradiction —
  the "~100%" was never a real result.
- **Genuine gaps:** **tempo** (negative R² — strictly worse than predicting the
  mean) and **key** (0.58 acc, below the 12-class threshold of 0.65). These two
  features are *not linearly recoverable* anywhere in MERT's 25 layers, which is
  the exact motivation for re-injecting them as a hand-crafted branch in the
  Enhanced model.

The tempo result is consistent with the initial single-target tempo probe
(R² ≈ 0.12 on a separate split) — both confirm the same conclusion. The
per-layer × per-feature heatmap in `plots/music_theory_probing_heatmap.png`
shows the full 25 × 8 grid; the summary plot
`plots/music_theory_probing_summary.png` shows best-per-feature scores at a glance.

Outputs: `phaseA/music_theory_probing_results.json`, `phaseA/gap_analysis.json`,
`phaseA/plots/music_theory_probing_summary.png`,
`phaseA/plots/music_theory_probing_heatmap.png`.

## 4. What This Established

1. MERT is a valid foundation: harmonically aware and emotionally structured
   before any supervision — improvements in Phase B are refinements, not
   compensation for a weak base.
2. The representation is **selective**: it linearly exposes harmony/timbre but not
   absolute tempo/key. This is actionable — it directly drives the **Enhanced**
   model in Phase B, which re-injects exactly these gap features (see
   `02_phaseB_model.md`, §6) and the music-theory annotation in Phase C.

**Takeaway:** Phase A turns "do the embeddings know music?" into a precise,
evidence-based map of what they encode — and that map, not a guess, decided what
to add later.
