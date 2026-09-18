#!/bin/bash
#SBATCH --job-name=ouu7_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#
# D continued (see job_ouu4.sh): task 0 is D at 0.95 started from D's 0.9 design (from
# the deterministic start SLSQP spent its 400 iterations without a feasible success);
# task 1 is D at 0.9 restarted from its own final point with a fresh iteration budget.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1
ALPHAS=(0.95 0.9)
TAGS=(_D_cont _D_cont)
STARTS=(paper_tests/out/ouu2_alpha0.9_D.json paper_tests/out/ouu2_alpha0.9_D.json)
I="$SLURM_ARRAY_TASK_ID"
ALPHA="${ALPHAS[$I]}"; TAG="${TAGS[$I]}"; START="${STARTS[$I]}"
NAME="ouu2_alpha${ALPHA}${TAG}"
LOG="paper_tests/cluster/logs/${NAME}_h100.log"
export JAX_PLATFORMS=cuda
export PYTHONPATH="$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"
{
    echo "== $(date '+%F %T') D continued: alpha $ALPHA from $START"
    python -u paper_tests/ouu.py --smoke --alpha "$ALPHA" --pairing one --table build --objective levelised \
        --te-recourse 5 --te-range 5,15 --chunk 2 --n 8192 --start "$START" \
        --jac fwd --max-iter 400 --evidence --name "$NAME" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
