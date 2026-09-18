#!/bin/bash
#SBATCH --job-name=ouu5_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:40:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#
# `--evidence` and `--sweep-te` re-run with `G_TOL = 2e-3` (PROCESS's own converged
# point is feasible only to that tolerance) for the B2 and D designs. Submit after
# job_ouu4's D tasks have written their JSON.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1
ALPHAS=(0.75 0.9 0.95 0.9   0.75 0.9 0.95 0.9)
TAGS=(_B2 _B2 _B2 _B2_start   _D _D _D _D16)
I="$SLURM_ARRAY_TASK_ID"
ALPHA="${ALPHAS[$I]}"; TAG="${TAGS[$I]}"
NAME="ouu2_alpha${ALPHA}${TAG}"
LOG="paper_tests/cluster/logs/${NAME}_h100.log"
export JAX_PLATFORMS=cuda
export PYTHONPATH="$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"
{
    echo "== $(date '+%F %T') re-evidence (G_TOL 2e-3) and sweep for $NAME"
    python -u paper_tests/ouu.py --evidence --robust "paper_tests/out/$NAME.json" --alpha "$ALPHA" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
    python -u paper_tests/ouu.py --sweep-te --robust "paper_tests/out/$NAME.json" --points 16
    echo "== $(date '+%F %T') exit $?"
} >> "$LOG" 2>&1
