#!/bin/bash
#SBATCH --job-name=ouu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/%x-%A_%a.out
#
# The OUU alpha sweep (paper_tests/ouu.py --smoke, N = 16384, jacfwd, one H100 per
# task): tasks 0-4 are alpha in (0.5, 0.75, 0.9, 0.95, 0.99) with the `mean` objective
# and the physics inputs (hfact sigma 0.10); task 5 is alpha 0.9 with every input
# (`--inputs all`); task 6 is alpha 0.9, physics, hfact sigma 0.15. Tasks 7-13 repeat
# the seven with the `median` objective, because the conditional mean of coe is
# dominated by samples whose net power is barely positive (coe ~ 1 / kwh there) and
# SLSQP may not converge on it -- so a converged sweep exists either way. Every task
# ends with `--evidence` (both designs on the fixed and a fresh Sobol' set).
# The harness's reference run (PROCESS's own VMCON, ~2 min on the CPU) is primed once
# under a lock before the tasks build, since concurrent first builds would race on the
# cache file's `.partial` sibling.
# Submit: cd ~/PROCESS/paper_tests/cluster && mkdir -p logs && sbatch job_ouu.sh
# Results: ~/PROCESS/paper_tests/out/ouu_alpha<alpha><tag>.json and
# ouu_evidence_<alpha><tag>.{json,npz}; logs: logs/ouu_alpha<alpha><tag>_h100.log.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

ALPHAS=(0.5 0.75 0.9 0.95 0.99 0.9 0.9   0.5 0.75 0.9 0.95 0.99 0.9 0.9)
INPUTS=(physics physics physics physics physics all physics   physics physics physics physics physics all physics)
SIGMAS=(0.10 0.10 0.10 0.10 0.10 0.10 0.15   0.10 0.10 0.10 0.10 0.10 0.10 0.15)
TAGS=("" "" "" "" "" "_all" "_s015"   "_median" "_median" "_median" "_median" "_median" "_all_median" "_s015_median")
OBJECTIVES=(mean mean mean mean mean mean mean   median median median median median median median)
I="$SLURM_ARRAY_TASK_ID"
ALPHA="${ALPHAS[$I]}"; INP="${INPUTS[$I]}"; SIGMA="${SIGMAS[$I]}"; TAG="${TAGS[$I]}"; OBJ="${OBJECTIVES[$I]}"
N="${N:-16384}"
MAX_ITER="${MAX_ITER:-400}"
NAME="ouu_alpha${ALPHA}${TAG}"
LOG="paper_tests/cluster/logs/${NAME}_h100.log"
export JAX_PLATFORMS=cuda
export PYTHONPATH="$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"

{
    echo "job $SLURM_JOB_ID ($SLURM_ARRAY_JOB_ID[$I]) on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "== $(date '+%F %T') prime the harness cache (flock)"
    mkdir -p "$FP_HARNESS_CACHE_DIR"
    JAX_PLATFORMS=cpu flock "$FP_HARNESS_CACHE_DIR/ouu_prime.lock" \
        python -c "import sys; sys.path.insert(0, 'paper_tests'); import uq; uq.build(); print('primed')"
    echo "== $(date '+%F %T') exit $?"
    echo "== $(date '+%F %T') ouu.py --smoke --n $N --alpha $ALPHA --objective $OBJ --inputs $INP --hfact-sigma $SIGMA --jac fwd --max-iter $MAX_ITER --evidence --name $NAME --tag $TAG"
    python -u paper_tests/ouu.py --smoke --n "$N" --alpha "$ALPHA" --objective "$OBJ" --inputs "$INP" \
        --hfact-sigma "$SIGMA" --jac fwd --max-iter "$MAX_ITER" --evidence --name "$NAME" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
