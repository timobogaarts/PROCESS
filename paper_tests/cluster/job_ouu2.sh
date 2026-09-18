#!/bin/bash
#SBATCH --job-name=ouu2_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=01:30:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x-%A_%a.out
#
# The two-stage OUU sweep (paper_tests/ouu.py --smoke, N = 16384, jacfwd, one H100 per
# task), every task on the `build` belief table (the four build-scatter / margin leaves
# at their nominal, so every sample has one build -- paper_tests/stage_check.py):
#   B      tasks 0-2: alpha 0.75 / 0.9 / 0.95, the density closes the power balance per
#          sample, T_e and the alpha fraction shared operating set-points, c16 (net
#          electric power) a CVaR constraint, objective `rated` (expected annual cost
#          per rated MWh).
#   B_start  task 3: B at 0.9 from the earlier robust design (a second local start).
#   Bfree    task 4: B at 0.9 without c16 (net power free), objective `nominal` -- the
#          earlier sweep's formulation on the build table.
#   A / Amean tasks 5-6: c16 closed per sample by the alpha fraction (`--pairing two`),
#          objective `nominal` / `mean`.
#   C      task 7: c16 closed per sample by T_e (`--pairing te`), objective `rated`.
#   B_new  task 8: B at 0.9 on the 26-leaf `new` table (build leaves sampled), for the
#          effect of fixing them.
# Every task ends with `--evidence` (both designs on the fixed and a fresh Sobol' set)
# and `--sweep-te` (per-sample best T_e at the robust design, where T_e is a design place).
# Submit: cd ~/PROCESS/paper_tests/cluster && mkdir -p logs && sbatch job_ouu2.sh
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

ALPHAS=(0.75 0.9 0.95 0.9 0.9 0.9 0.9 0.9 0.9)
TAGS=(_B _B _B _B_start _Bfree _A _Amean _C _B_new)
ARGS=(
  "--pairing one --table build --with-c16 --objective rated"
  "--pairing one --table build --with-c16 --objective rated"
  "--pairing one --table build --with-c16 --objective rated"
  "--pairing one --table build --with-c16 --objective rated --start paper_tests/out/ouu_alpha0.9_median.json"
  "--pairing one --table build --objective nominal"
  "--pairing two --table build --objective nominal"
  "--pairing two --table build --objective mean"
  "--pairing te --table build --objective rated"
  "--pairing one --table new --with-c16 --objective rated"
)
I="$SLURM_ARRAY_TASK_ID"
ALPHA="${ALPHAS[$I]}"; TAG="${TAGS[$I]}"; ARG="${ARGS[$I]}"
N="${N:-16384}"
MAX_ITER="${MAX_ITER:-400}"
NAME="ouu2_alpha${ALPHA}${TAG}"
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
    echo "== $(date '+%F %T') ouu.py --smoke --n $N --alpha $ALPHA $ARG --jac fwd --max-iter $MAX_ITER --evidence --name $NAME --tag $TAG"
    # shellcheck disable=SC2086
    python -u paper_tests/ouu.py --smoke --n "$N" --alpha "$ALPHA" $ARG \
        --jac fwd --max-iter "$MAX_ITER" --evidence --name "$NAME" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
    echo "== $(date '+%F %T') ouu.py --sweep-te --robust paper_tests/out/$NAME.json"
    python -u paper_tests/ouu.py --sweep-te --robust "paper_tests/out/$NAME.json" --points 16
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
