"""Centralised configuration for the Explainable Music Emotion Recognition project.

Single source of truth for hyperparameters and file paths.

IMPORTANT
---------
* Every hyperparameter below is copied **verbatim** from the original scripts.
  No value has been changed — only relocated here (Stage-4 config consolidation).
* Dataset / embedding / checkpoint locations are centralised as env-overridable
  roots so the code is portable. On the working server, point the roots at the
  existing data with environment variables (see below); on a fresh checkout,
  follow ``data/README.md`` to place PMEmo under ``data/pmemo/``.

Environment overrides
---------------------
* ``PMEMO_ROOT``       — root of the PMEmo 2019 dataset
                         (default: ``<repo>/data/pmemo``).
* ``MER_FEATURES_DIR`` — directory holding the pre-extracted feature tensors and
                         trained checkpoints
                         (default: ``<repo>/MERT/ssl_scripts/phaseB``).

Example (server)::

    export PMEMO_ROOT=/datasets/emotions/PMEmo2019
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent

# Dataset root (override with $PMEMO_ROOT on machines where PMEmo lives elsewhere).
PMEMO_ROOT = Path(os.environ.get("PMEMO_ROOT", REPO_ROOT / "data" / "pmemo"))

# Pre-extracted feature tensors + trained checkpoints (left physically in place).
FEATURES_DIR = Path(os.environ.get("MER_FEATURES_DIR", REPO_ROOT / "MERT" / "ssl_scripts" / "phaseB"))
PHASEA_DIR = Path(os.environ.get("MER_PHASEA_DIR", REPO_ROOT / "MERT" / "ssl_scripts" / "phaseA"))
PHASEC_DIR = Path(os.environ.get("MER_PHASEC_DIR", REPO_ROOT / "MERT" / "ssl_scripts" / "phaseC"))
SSL_SCRIPTS_DIR = REPO_ROOT / "MERT" / "ssl_scripts"


@dataclass(frozen=True)
class Paths:
    """All data / feature / checkpoint locations used across the three phases."""

    # PMEmo 2019 dataset root + annotations (continuous valence--arousal)
    pmemo_root: Path = PMEMO_ROOT
    pmemo_annotations: Path = PMEMO_ROOT / "annotations" / "static_annotations.csv"

    # Pre-extracted per-layer embeddings
    mert_features: Path = FEATURES_DIR / "pmemo_mert_all_layers.pt"          # 25 x 1024
    wav2vec_features: Path = FEATURES_DIR / "pmemo_wav2vec_all_layers.pt"    # 13 x 768
    melspec_features: Path = FEATURES_DIR / "pmemo_melspec.pt"
    mert_pooled_embeddings: Path = SSL_SCRIPTS_DIR / "pmemo_mert_embeddings.pt"  # single-vector (Phase A)

    # Music-theory gap features (Phase A -> Phase B re-injection)
    gap_analysis: Path = PHASEA_DIR / "gap_analysis.json"
    music_theory: Path = PHASEA_DIR / "data" / "pmemo_music_theory.pt"

    # IADS-E cross-domain corpus (Phase B transfer experiment)
    iadse_mert: Path = FEATURES_DIR / "iadse_mert_all_layers.pt"
    iadse_wav2vec: Path = FEATURES_DIR / "iadse_wav2vec_all_layers.pt"
    iadse_labels: Path = FEATURES_DIR / "iadse_labels.csv"

    # Trained checkpoints
    enhanced_ckpt: Path = FEATURES_DIR / "best_model_enhanced_final.pt"
    protopnet_ckpt: Path = FEATURES_DIR / "protopnet_final.pt"
    baseline_ckpt: Path = FEATURES_DIR / "best_model.pt"

    # Phase C index artefacts
    prototypes: Path = PHASEC_DIR / "prototypes.npy"
    prototypes_enhanced: Path = PHASEC_DIR / "prototypes_enhanced.npy"


PATHS = Paths()

# --------------------------------------------------------------------------- #
# Frozen self-supervised backbones (loaded from HuggingFace)
# --------------------------------------------------------------------------- #
MERT_MODEL = "m-a-p/MERT-v1-330M"        # frozen, 25 x 1024-d, 24 kHz
WAV2VEC_MODEL = "facebook/wav2vec2-base"  # frozen, 13 x 768-d, 16 kHz
MERT_SR = 24000
WAV2VEC_SR = 16000


# --------------------------------------------------------------------------- #
# Phase A — representation probing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhaseAConfig:
    ridge_alpha: float = 1.0            # Ridge regression probe
    r2_gap_threshold: float = 0.40     # regression "representational gap" cut-off
    acc_gap_threshold: float = 0.65    # classification gap cut-off
    # Superseded single-target linear probe (kept for reproducibility)
    probe_input_dim: int = 1024
    probe_lr: float = 0.001
    probe_epochs: int = 200


# --------------------------------------------------------------------------- #
# Phase B — multi-encoder affective regressor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhaseBConfig:
    learning_rate: float = 1e-4        # base LR (head / regressor / projection)
    fusion_lr: float = 1e-2            # differential LR for Weighted Layer Fusion weights
    prototype_lr: float = 3e-3         # Audio ProtoPNet prototype LR
    weight_decay: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    n_folds: int = 5
    seed: int = 42
    grad_clip_norm: float = 1.0
    embedding_dim: int = 128           # L2-normalised latent dimension
    loss_weights: dict = field(
        default_factory=lambda: {"mse": 1.0, "ccc": 0.5, "rank": 0.3, "supcr": 0.1}
    )
    protos_per_quadrant: int = 5       # Audio ProtoPNet
    n_prototypes: int = 20             # 5 per quadrant x 4 quadrants


# --------------------------------------------------------------------------- #
# Phase C — explainable retrieval
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhaseCConfig:
    faiss_index_type: str = "IndexFlatIP"  # inner product on L2-normed vectors = cosine
    k_neighbours: int = 5                  # top-k for Precision@k / retrieval
    k_values: tuple = (5, 10, 20)
    va_radius: float = 0.20                # emotional-closeness radius in [0,1]^2 V-A plane
    n_foils: int = 3                       # contrastive "why-not" foils
    protos_per_quadrant: int = 5
    n_prototypes: int = 20
    llm_backend: str = "hf"                # hf (Qwen2.5-3B, deployed) | ollama | anthropic | none


PHASE_A = PhaseAConfig()
PHASE_B = PhaseBConfig()
PHASE_C = PhaseCConfig()
