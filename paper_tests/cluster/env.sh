#!/bin/bash
# Sourced by every script here, on Snellius. Loads the modules the H100 jobs need
# (the same three lines as ~/blanket_optimization/slurm_util/modules_load.sh),
# activates the study's own venv when it exists, and sets the XLA env the study
# documents (paper_tests/README.md, "Batching").
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.9.1

VENV="${VENV:-$HOME/cottaxvenv}"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# XLA takes 75 % of the card otherwise; the README's runs were measured with this.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The harness's converged-reference cache and batching.py's cross-backend diff arrays
# (~2.5 GB of .npy) land here; the default is the same path.
export FP_HARNESS_CACHE_DIR="${FP_HARNESS_CACHE_DIR:-$HOME/.cache/functional_process}"
