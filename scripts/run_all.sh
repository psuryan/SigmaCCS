#!/usr/bin/env bash
# Run all SigmaCCS baseline experiments (Exp A–E).
# Activates the venv, then runs run_experiment.py sequentially for every split.
#
# Usage:
#   bash scripts/run_all.sh
#
# Override DATA_DIR or REPO_ROOT if needed:
#   DATA_DIR=/other/path bash scripts/run_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/../GraphCCS/data}"
VENV="${VENV:-$HOME/.venvs/SigmaCCS}"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"
SCRIPT="${REPO_ROOT}/scripts/run_experiment.py"
DATA_CSV="${DATA_DIR}/data.csv"
CONFORMERS="${REPO_ROOT}/data/conformers.pkl"

# Activate virtual environment
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "============================================================"
echo " SigmaCCS Baseline — Full Experiment Suite"
echo " data: ${DATA_CSV}"
echo " out:  ${EXPERIMENTS_DIR}"
echo "============================================================"

# ----------------------------------------------------------------
# Exp A — Random split (full)
# ----------------------------------------------------------------
echo ""
echo "[Exp A] Random split (full)"
python "${SCRIPT}" \
    --data "${DATA_CSV}" \
    --conformers "${CONFORMERS}" \
    --split "${DATA_DIR}/splits/random/split.json" \
    --out "${EXPERIMENTS_DIR}/random" \
    --seeds 0 1 2 3 4 \
    --epochs 500 \
    --batch-size 14 \
    --patience 20 \
    --label random

# ----------------------------------------------------------------
# Exp B — Scaffold split
# ----------------------------------------------------------------
echo ""
echo "[Exp B] Scaffold split"
python "${SCRIPT}" \
    --data "${DATA_CSV}" \
    --conformers "${CONFORMERS}" \
    --split "${DATA_DIR}/splits/scaffold/split.json" \
    --out "${EXPERIMENTS_DIR}/scaffold" \
    --seeds 0 1 2 3 4 \
    --epochs 500 \
    --batch-size 14 \
    --patience 20 \
    --label scaffold

# ----------------------------------------------------------------
# Exp C — Adduct-sensitive split
# ----------------------------------------------------------------
echo ""
echo "[Exp C] Adduct-sensitive split"
python "${SCRIPT}" \
    --data "${DATA_CSV}" \
    --conformers "${CONFORMERS}" \
    --split "${DATA_DIR}/splits/adduct_sensitive/split.json" \
    --out "${EXPERIMENTS_DIR}/adduct_sensitive" \
    --seeds 0 1 2 3 4 \
    --epochs 500 \
    --batch-size 14 \
    --patience 20 \
    --label adduct_sensitive

# ----------------------------------------------------------------
# Exp D — Random learning curve (5 fractions + full)
# ----------------------------------------------------------------
echo ""
echo "[Exp D] Random learning curve"

for frac in 0.1 0.2 0.4 0.6 0.8; do
    label="frac_${frac}"
    echo "  -> ${label}"
    python "${SCRIPT}" \
        --data "${DATA_CSV}" \
        --conformers "${CONFORMERS}" \
        --split "${DATA_DIR}/splits/random_frac/split_${frac}.json" \
        --out "${EXPERIMENTS_DIR}/random_frac/${label}" \
        --seeds 0 1 2 3 4 \
        --epochs 500 \
        --batch-size 14 \
        --patience 20 \
        --label "${label}"
done

# full fraction reuses Exp A output — skip re-training, just symlink
echo "  -> frac_full (reusing Exp A)"
ln -sfn "${EXPERIMENTS_DIR}/random" "${EXPERIMENTS_DIR}/random_frac/full"

# ----------------------------------------------------------------
# Exp E — Adduct-sensitive learning curve (optional)
# ----------------------------------------------------------------
echo ""
echo "[Exp E] Adduct-sensitive learning curve"

for frac in 0.1 0.2 0.4 0.6 0.8; do
    label="frac_${frac}"
    echo "  -> ${label}"
    python "${SCRIPT}" \
        --data "${DATA_CSV}" \
        --conformers "${CONFORMERS}" \
        --split "${DATA_DIR}/splits/adduct_sensitive_frac/split_${frac}.json" \
        --out "${EXPERIMENTS_DIR}/adduct_sensitive_frac/${label}" \
        --seeds 0 1 2 3 4 \
        --epochs 500 \
        --batch-size 14 \
        --patience 20 \
        --label "adduct_sensitive_${label}"
done

echo "  -> frac_full (reusing Exp C)"
ln -sfn "${EXPERIMENTS_DIR}/adduct_sensitive" "${EXPERIMENTS_DIR}/adduct_sensitive_frac/full"

echo ""
echo "============================================================"
echo " All experiments complete.  Run analysis:"
echo "   python scripts/run_analysis.py --exp-dir ${EXPERIMENTS_DIR}"
echo "============================================================"
