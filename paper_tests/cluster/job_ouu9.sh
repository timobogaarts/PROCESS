#!/bin/bash
#SBATCH --job-name=ouu9_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#
# The operating point as a plain first-stage design variable -- no recourse, no
# closure of it. T_e joins the build vector, so one machine is chosen AND one
# operating point, and what varies across scenarios is what the plant actually
# delivers. In flexibility-analysis terms this is the design with no control
# variable: the feasible fraction is then the stochastic flexibility of a fully
# specified operating policy, and (since fixing an operating variable first-stage
# shrinks the feasible set) the objective is an upper bound on the recourse answer.
#
#   task 0: alpha 0.9, lifted, T_e first stage      <- the comparison run
#   task 1: the same with the 5-point T_e recourse  <- the old formulation, for the gap
#
# Submit: cd ~/PROCESS/paper_tests/cluster && sbatch job_ouu9.sh
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

TAGS=(_te_design _te_recourse)
EXTRA=("" "--te-recourse 5 --te-range 5,15")
I="$SLURM_ARRAY_TASK_ID"
TAG="${TAGS[$I]}"; ARG="${EXTRA[$I]}"
NAME="ouu9_alpha0.9${TAG}"
LOG="paper_tests/cluster/logs/${NAME}_h100.log"
export JAX_PLATFORMS=cuda
export PYTHONPATH="$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"
{
    echo "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "== $(date '+%F %T') $NAME"
    # shellcheck disable=SC2086
    python -u paper_tests/ouu.py --smoke --n 8192 --alpha 0.9 \
        --pairing one --table build --objective levelised --closure bracketed \
        --lifts winding_pack --chunk 2 $ARG \
        --jac fwd --max-iter 400 --evidence --name "$NAME" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
