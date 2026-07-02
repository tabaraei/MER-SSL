#!/usr/bin/env bash
# Reproduce the three experimental phases end-to-end.
# Requires the pre-extracted feature tensors (see data/README.md) and, for
# fresh extraction, the PMEmo 2019 dataset under $PMEMO_ROOT.
set -euo pipefail

# --- make `configs` (repo root) and the src/ layout importable ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# If PMEmo is not at ./data/pmemo, point PMEMO_ROOT at it, e.g.:
#   export PMEMO_ROOT=/path/to/PMEmo2019

echo "==================== Phase A — probing SSL representations ===================="
python src/probing/run_music_theory_probing.py

echo "==================== Phase B — hybrid emotion regression ======================"
python src/training/train_enhanced_dual.py     # Enhanced: MERT + wav2vec2 + tempo/key
python src/training/mainB_triple.py            # Triple:   MERT + wav2vec2 + mel-CNN
python src/training/train_deploy_models.py     # train + save deployed checkpoints

echo "==================== Phase C — explainable retrieval =========================="
python src/explainability/build_index_unified.py
python src/explainability/mainC.py

echo "Done."
