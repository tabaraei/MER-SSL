# Explainable Music Emotion Recognition — SSL Scripts
**Thesis:** Explainable & Perceptual Music Emotion Recognition via Self-Supervised Learning  
**Student:** Arvin Jafari Moghadam Fard

This directory contains the full three-phase experimental pipeline. Each phase builds on the previous one — run them in order.

---

## Prerequisites

```bash
# Activate the project virtual environment
source ../.venv/bin/activate   # or: conda activate MERT

# Required packages (already in pyproject.toml)
# torch, transformers, librosa, scikit-learn, pandas, numpy, scipy, tqdm
```

**Dataset paths assumed throughout:**
```
/datasets/emotions/PMEmo2019/
├── chorus/                         # .mp3 audio files
├── annotations/static_annotations.csv
└── EDA/                            # {id}_EDA.csv files
```

---

## Overview

```
ssl_scripts/
├── extract_pmemo.py        ← Step 0: extract MERT embeddings from audio
│
├── phaseA/                 ← Phase A: perceptual probing
│   ├── probe_key.py            probe major/minor mode
│   ├── probe_tempo.py          probe tempo (BPM)
│   └── visualize_embeddings.py t-SNE plot coloured by emotion quadrant
│
├── phaseB/                 ← Phase B: affective model training
│   ├── mainB.py                train the hybrid MER model
│   ├── analyze.py              post-training analysis & thesis figures
│   ├── models.py               MERModel architecture
│   ├── losses.py               HybridLoss, CCCLoss, SupCRLoss
│   └── data_utils.py           PMEmo data loading utilities
│
└── phaseC/                 ← Phase C: explainable retrieval
    ├── mainC.py                entry point (build / query / evaluate)
    ├── index_builder.py        latent-space index construction
    ├── retriever.py            cosine k-NN + contrastive foils
    ├── explainer.py            two-layer explanation engine
    ├── eda_loader.py           EDA feature extraction
    └── evaluator.py            Precision@k + Silhouette evaluation
```

---

## Step 0 — Extract MERT Embeddings

Extract all-layer MERT embeddings from the PMEmo chorus clips. Run **once** — output is reused by all three phases.

```bash
cd ssl_scripts/
python extract_pmemo.py
```

**Output:** `pmemo_mert_embeddings.pt` (single last-layer), `phaseB/pmemo_mert_all_layers.pt` (all 25 layers)

> The all-layers file is required by Phase B and C. It stores a dictionary `{music_id: tensor(25, 1024)}` for all matched songs.

---

## Phase A — Perceptual Probing

Validates that the MERT latent space encodes music-theoretically meaningful information before any fine-tuning. Answers: *does the pre-trained SSL model already understand music structure?*

### A1 — Harmonic Mode Probe (Major/Minor)

```bash
cd phaseA/
python probe_key.py
```

Trains a linear classifier on top of frozen MERT embeddings to predict major vs minor mode (extracted via `librosa`). A high accuracy confirms that harmonic structure is encoded.

**Expected result:** ~100% accuracy → MERT embeddings are harmonically aware.

---

### A2 — Tempo Probe

```bash
python probe_tempo.py
```

Trains a linear regressor to predict BPM from frozen embeddings.

**Expected result:** R² ≈ 0.12 → tempo is weakly but significantly encoded (expected: rhythm is harder to capture from chorus clips alone).

---

### A3 — Emotion Cluster Visualization

```bash
python visualize_embeddings.py
```

Runs t-SNE on the MERT embeddings coloured by Russell quadrant (HVHA / HVLA / LVHA / LVLA).

**Output:** `mert_tsne_plot.png`, `mert_emotion_clusters.png`

---

## Phase B — Hybrid Affective Model Training

Trains the emotion regression model (arousal + valence) on top of the MERT all-layer embeddings. The architecture uses learnable `WeightedLayerFusion` across all 25 MERT layers and a `HybridLoss` combining MSE + CCC + Rank + SupCR.

### B1 — Train

```bash
cd phaseB/

# Recommended: 5-fold cross-validation with balanced sampler + differential optimizer
python mainB.py \
    --model  hybrid \
    --mode   kfold \
    --epochs 100 \
    --lr     1e-4 \
    --feat_path pmemo_mert_all_layers.pt \
    --csv_path  /datasets/emotions/PMEmo2019/annotations/static_annotations.csv
```

**Key arguments:**

| Argument | Options | Description |
| :--- | :--- | :--- |
| `--model` | `hybrid` / `baseline` | Hybrid = WeightedFusion + SupCR; Baseline = mean pool |
| `--mode` | `kfold` / `simple` | 5-fold CV (thesis) or quick 80/20 split |
| `--epochs` | int | 100 recommended for convergence |
| `--lr` | float | Base learning rate (fusion gets 1e-2 automatically) |
| `--use_eda` | flag | Enable EDA multimodal fusion |
| `--eda_dir` | path | Required if `--use_eda` is set |

**With EDA fusion (multimodal):**
```bash
python mainB.py \
    --model   hybrid \
    --mode    kfold \
    --use_eda \
    --eda_dir /datasets/emotions/PMEmo2019/EDA \
    --feat_path pmemo_mert_all_layers.pt \
    --csv_path  /datasets/emotions/PMEmo2019/annotations/static_annotations.csv
```

**Output:**
- `best_model.pt` — saved weights of the last fold's model
- `layer_weights.npy` — learned MERT layer fusion weights
- `layer_weights_kfold.png` — layer weight visualization

**Validated results (5-fold, Hybrid + EDA):**

| Metric | Arousal | Valence |
| :--- | :--- | :--- |
| R² | 0.6738 | 0.5075 |
| CCC | 0.8543 | 0.7692 |

---

### B2 — Post-Training Analysis

```bash
python analyze.py
```

Generates all thesis figures from the saved model and layer weights:
1. Layer weight bar chart (which MERT layers the model specializes on)
2. Emotion quadrant confusion plot
3. Silhouette score comparison: baseline vs hybrid latent geometry
4. Precision@k comparison (preview of Phase C retrieval quality)

> Run this after `mainB.py` completes. Reads `best_model.pt` and `layer_weights.npy` automatically.

---

## Phase C — Explainable Retrieval System

Builds a prototype-based retrieval system on top of the Phase B latent space and generates two-layer explanations: a deterministic technical report and an LLM-powered humanized recommendation.

### C1 — Build the Index

Encode all PMEmo songs into a searchable latent-space index. Run **once** after Phase B.

```bash
cd phaseC/

python mainC.py --mode build \
    --model_path ../phaseB/best_model.pt \
    --feat_path  ../phaseB/pmemo_mert_all_layers.pt \
    --csv_path   /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
    --eda_dir    /datasets/emotions/PMEmo2019/EDA \
    --index_path prototypes.npy
```

**Output:** `prototypes.npy` — stores latents, V-A scores, EDA features, and layer weights for all 767 songs.

---

### C2 — Query: Retrieve + Explain

Retrieve the k most emotionally similar songs to a query and generate a full explanation.

```bash
python mainC.py --mode query \
    --query_id   760 \
    --index_path prototypes.npy \
    --feat_path  ../phaseB/pmemo_mert_all_layers.pt \
    --csv_path   /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
    --model_path ../phaseB/best_model.pt \
    --eda_dir    /datasets/emotions/PMEmo2019/EDA \
    --top_k      5 \
    --n_foils    3 \
    --llm        ollama \
    --llm_model  llama3.2
```

**Key arguments:**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--query_id` | required | PMEmo music ID to query |
| `--top_k` | 5 | Number of similar songs to retrieve |
| `--n_foils` | 3 | Contrastive rejected songs (for XAI explanation) |
| `--llm` | `ollama` | LLM backend: `ollama` / `anthropic` / `none` |
| `--llm_model` | `llama3.2` | Model name passed to the backend |

**Output:**
- Terminal: template explanation + LLM humanized recommendation
- `retrieval_query_{id}.txt` — full saved output (template + RAG prompt + LLM response)

**LLM backends:**

```bash
# Option A — Local (Ollama, no API key needed)
ollama serve                     # start in another terminal
ollama pull llama3.2             # or llama3.1:8b for better quality
--llm ollama --llm_model llama3.2

# Option B — Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...
--llm anthropic --llm_model claude-sonnet-4-6

# Option C — No LLM (template only)
--llm none
```

---

### C3 — Evaluate Retrieval Quality

```bash
python mainC.py --mode evaluate --index_path prototypes.npy
```

Computes two metrics over the full 767-song index:

| Metric | Definition |
| :--- | :--- |
| **Precision@k** (k=5,10,20) | Fraction of top-k neighbors within 0.20 V-A Euclidean distance |
| **Silhouette** | Russell quadrant cluster separation in cosine latent space (+1 = perfect) |

**Baseline comparison:**
```bash
# Build a baseline index first
python mainC.py --mode build --model_mode baseline \
    --index_path prototypes_baseline.npy [... same paths ...]

# Then compare both
python mainC.py --mode evaluate --index_path prototypes.npy
python mainC.py --mode evaluate --index_path prototypes_baseline.npy
```

---

## What Each Explanation Contains

A Phase C query produces a **two-layer** output:

```
═══════════════════════════════════════════════════════════════
🎵  QUERY: Song ID 760
═══════════════════════════════════════════════════════════════
  Emotion Region  : Low Arousal / High Valence (HVLA)
  Arousal         : low (0.352) | Valence: positive (0.571)
  Physiology      : low EDA, few SCR peaks, stable trend

🔍  TOP-5 RETRIEVED SONGS
  ┌─ Rank 1 | Song ID 481 (cosine sim = 0.9977)
  │  Quadrant : LVLA | Arousal ↓0.03 | Valence ↓0.11 vs query
  └────────────────────────────────────────────

🧠  MUSIC-THEORETIC SUMMARY
  avg arousal=0.346 | valence=0.448 | mean sim=0.9969
  Mood arc: energizing [699→481→116→221→716] | wind-down [reverse]

🤖  LLM EXPLANATION (Ollama / llama3.2)
  "If you've been drawn to this calm, gently uplifting track, these
   five songs share that same unhurried emotional warmth..."
```

**Layer 1 (template):** precise, deterministic, citable — V-A coordinates, cosine similarities, per-neighbor deltas, EDA summaries, mood trajectory.

**Layer 2 (LLM RAG):** humanized recommendation grounded in: emotion knowledge base (genre families, music theory), EDA physiological narrative, contrastive foils (rejected songs), and MERT layer attribution.

---

## Troubleshooting

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `⚠️ EDA missing for 767/767 songs` | Wrong filename format | Files must be named `{id}_EDA.csv` |
| `RuntimeError: Missing key(s) in state_dict` | EDA model loaded into base model | Handled automatically — keys are stripped |
| `unrecognized arguments: --eda_dir` | `\ ` in single-line command | Remove `\` when running on one line |
| `Ollama not running` | Server not started | Run `ollama serve` in a separate terminal |
| `FAISS not found` | FAISS not installed | `pip install faiss-gpu` or use sklearn fallback |
