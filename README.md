# Explainable Music Emotion Recognition via Self-Supervised Representations

**Master's Thesis** | MSc Computer Science — Music Information Retrieval
**University of Milan** | Supervisors: Prof. Stavros Ntalampiras, Ali Tabaraei

## Overview

This project audits **frozen self-supervised (SSL) audio representations** — primarily
[MERT](https://huggingface.co/m-a-p/MERT-v1-330M) — for **Music Emotion Recognition** on the
continuous valence–arousal plane, and adds an **ante-hoc explanation layer** so predictions
come with grounded, audible evidence rather than a black-box score. A frozen MERT backbone is
combined with a lightweight hybrid-loss regressor and a retrieval + prototype (Audio ProtoPNet)
explanation stage. The headline finding is that SSL features predict **arousal well but valence
up to an audio-only ceiling**, and several **negative results** (fusion collapse, encoder
redundancy, failed cross-domain transfer, a null key-encoding fix) are reported as first-class
contributions rather than hidden.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── conftest.py                  # puts repo root + src/ on sys.path
├── configs/
│   └── config.py                # single source of truth: hyperparameters + paths
├── data/
│   └── README.md                # PMEmo download + setup (dataset is gitignored)
├── src/
│   ├── extraction/              # frozen-backbone feature extractors (MERT / wav2vec2 / mel)
│   ├── models/                  # WeightedLayerFusion, MERModel, Dual/Triple/Enhanced, AudioProtoPNet
│   ├── losses/                  # HybridLoss = MSE + CCC + Rank + SupCR
│   ├── utils/                   # PMEmo / EDA data loading + helpers
│   ├── probing/                 # Phase A — linear probing of MERT layers
│   ├── training/                # Phase B — hybrid-loss V-A regressor
│   │   └── ablations/           # controlled ablations (fusion, imbalance, mixup, key, IADS-E, …)
│   └── explainability/          # Phase C — retrieval, ProtoPNet, RAG explanations
├── scripts/                     # reproduce_all.sh + report/figure tooling
├── tests/                       # smoke tests
├── examples/                    # sample explanation + retrieval outputs
└── results/                     # checkpoints & plots (gitignored)
```

## Installation

```bash
git clone https://github.com/robinjmf/Music-Emotion-Recognition-Self-Supervised-Learning.git
cd Music-Emotion-Recognition-Self-Supervised-Learning
pip install -r requirements.txt
```

Tested with: Python 3.10, PyTorch 2.5.1 (CUDA 12.1 build; CPU fallback supported).

The code uses an `src/` layout. Put the repo root and `src/` on the path before running
scripts directly (handled automatically by `scripts/reproduce_all.sh` and, for tests, by
`conftest.py`):

```bash
export PYTHONPATH="$PWD:$PWD/src"
```

## Dataset Setup

This work uses **PMEmo 2019** (Zhang et al., 2018, DOI:10.1145/3206025.3206037). See
[`data/README.md`](data/README.md) for the download and directory layout. In short: place the
dataset under `data/pmemo/` (or set `export PMEMO_ROOT=/path/to/PMEmo2019`), then extract the
frozen features:

```bash
python src/extraction/extract_pmemo.py          # MERT-v1-330M -> pmemo_mert_all_layers.pt
python src/extraction/extract_pmemo_wav2vec.py  # wav2vec2-base -> pmemo_wav2vec_all_layers.pt
python src/extraction/extract_pmemo_melspec.py  # mel-spectrogram -> pmemo_melspec.pt
```

MERT is loaded from HuggingFace as `m-a-p/MERT-v1-330M` (frozen; 25 layers × 1024-d, 24 kHz).

## Reproducing Experiments

Run everything with `bash scripts/reproduce_all.sh`, or phase by phase:

### Phase A — Probing SSL Representations
```bash
python src/probing/run_music_theory_probing.py
```
Outputs: per-layer linear-probe R² / accuracy for eight music-theory descriptors; harmonic
mode (Major/Minor) ≈ **67.3%**; **tempo and musical key identified as representational gaps**
(re-injected as explicit features in Phase B).

### Phase B — Hybrid Emotion Regression
```bash
python src/training/train_enhanced_dual.py   # Enhanced: MERT + wav2vec2 + tempo/key
python src/training/mainB_triple.py          # Triple:   MERT + wav2vec2 + mel-CNN
python src/training/mainB.py                 # single / dual-SSL baselines
python src/training/train_deploy_models.py   # train + save deployed checkpoints
```
Outputs: 5-fold cross-validated R² and CCC per configuration. Ablations live in
`src/training/ablations/`.

### Phase C — Explainable Retrieval
```bash
python src/explainability/build_index_unified.py   # build the latent index + prototypes
python src/explainability/mainC.py                 # retrieval + prototype + RAG explanation
```
Outputs: cosine k-NN retrieval index (128-d contrastive latent), **Precision@5 = 0.594**
(vs. random baseline 0.276, ≈2.15×), Audio ProtoPNet prototype explanations, and natural-language
explanation samples (see [`examples/`](examples/)).

## Key Results

| Model Configuration | R²-V | R²-A | CCC-V | CCC-A |
|---|---|---|---|---|
| Zhang et al. 2018 (IS13+SVR) — baseline | 0.41 | 0.52 | — | — |
| EmoMucs (de Berardinis et al. 2020) | 0.414 | 0.610 | — | — |
| Music2Emo single-dataset (Kang & Herremans 2025) | 0.536 | 0.777 | — | — |
| MERT + EDA (this work) | 0.508 | 0.674 | 0.769 | 0.854 |
| **Triple: MERT+Wav2Vec2+mel-CNN (this work)** | **0.576** | 0.702 | — | — |
| **Enhanced: MERT+mel-CNN+tempo/key (this work)** | 0.569 | **0.718** | — | — |

Among single-dataset, audio-only PMEmo methods, this work leads on the harder **valence** axis
while adding an explanation layer none of the baselines provide.

**Explainability (Phase C):**
- Retrieval **Precision@5 = 0.594** (≈2.15× the 0.276 random baseline).
- **Audio ProtoPNet** (5 prototypes/quadrant, 20 total): **0.728 raw / 0.545 balanced accuracy**,
  beating the majority-class baseline (0.611) and lifting Sad-quadrant recall 0.17 → 0.69.
- Latent geometry: Silhouette **≈0.19 (Euclidean) / ≈0.26 (cosine)** — a continuous valence–arousal
  gradient (Russell's circumplex), *not* four discrete clusters. Precision@k, not Silhouette, is the
  faithful measure.

**Selected negative findings (treated as contributions):**
- WeightedLayerFusion weights collapse to near-uniform at N ≈ 600 (entropy 3.218 vs. max 3.219) —
  insufficient data for layer selection.
- Wav2Vec2 (speech-SSL) is largely redundant once stacked with MERT (music-SSL) plus a small
  spectrogram/tempo branch.
- Cross-domain transfer from the IADS-E environmental-sound corpus degrades valence — SSL features
  discard the low-level acoustics such transfer depends on.
- Correcting the musical-key encoding (cyclic sin/cos) is a **null result** (ΔR²-V = −0.008): the
  valence limit comes from the data, not the feature encoding.
- SupCR improves retrieval Precision@5 (≈+0.03) but does **not** produce globally separable clusters
  (Silhouette 0.18–0.29) — consistent with a continuous emotion manifold.

## Citation

```bibtex
@mastersthesis{jafari2026mer,
  author  = {Arvin Jafari Moghadam Fard},
  title   = {An Empirical Audit of Self-Supervised Learning for
             Explainable Music Emotion Recognition},
  school  = {University of Milan},
  year    = {2026},
  type    = {Master's Thesis, MSc Computer Science -- Music Information Retrieval}
}
```

## Acknowledgements

Supervised by Prof. Stavros Ntalampiras and Ali Tabaraei, University of Milan.
Language model tools were used for prose editing assistance during thesis writing,
consistent with university policy on AI-assisted academic work.
