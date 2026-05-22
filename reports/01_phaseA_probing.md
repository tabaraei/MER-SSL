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

## 2. Initial Probes (mode + tempo)

| Probe | Target | Result | Reading |
| :-- | :-- | :-: | :-- |
| Harmonic mode | Major / Minor | **~100% accuracy** | Harmony is strongly, linearly encoded |
| Tempo | BPM | **R² ≈ 0.12** | Speed is only weakly/coarsely encoded |

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

### Result — gap_features = {tempo, key}

Across all 25 layers, MERT does **not** linearly expose absolute **tempo** or
**key**. Harmony/timbre features (chroma, spectral contrast/centroid, ZCR, mode,
rhythmic stability) are captured above threshold. The tempo gap is consistent with
the initial tempo probe (R² ≈ 0.12).

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
