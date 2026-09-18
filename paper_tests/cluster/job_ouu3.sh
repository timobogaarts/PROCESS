#!/bin/bash
#SBATCH --job-name=ouu3_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=01:30:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x-%A_%a.out
#
# Second two-stage sweep (see job_ouu2.sh): `c16` as a CVaR at the physics limits' alpha
# was infeasible (SLSQP could not leave the deterministic start, and from the earlier
# robust design R sits on its 30 m bound with the worst decile 180 MW short), so the
# shortfall is priced instead of bounded:
#   B2   tasks 0-2: alpha 0.75 / 0.9 / 0.95, no c16, objective `levelised`
#        (E[annual cost] / E[energy]: a shortfall raises the coe).
#   B2r  task 3: alpha 0.9, no c16, objective `rated`.
#   B3   tasks 4-5: alpha 0.9, `levelised`, c16 kept as a CVaR at its own level
#        `--alpha16` 0.5 / 0.75 (the worst half / quarter on average at the rated power).
#   B2s  task 6: B2 at 0.9 from the earlier robust design (a second local start).
# Submit: cd ~/PROCESS/paper_tests/cluster && sbatch job_ouu3.sh
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

ALPHAS=(0.75 0.9 0.95 0.9 0.9 0.9 0.9)
TAGS=(_B2 _B2 _B2 _B2r _B3a50 _B3a75 _B2_start)
ARGS=(
  "--pairing one --table build --objective levelised"
  "--pairing one --table build --objective levelised"
  "--pairing one --table build --objective levelised"
  "--pairing one --table build --objective rated"
  "--pairing one --table build --objective levelised --with-c16 --alpha16 0.5"
  "--pairing one --table build --objective levelised --with-c16 --alpha16 0.75"
  "--pairing one --table build --objective levelised --start paper_tests/out/ouu_alpha0.9_median.json"
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
    echo "== $(date '+%F %T') ouu.py --smoke --n $N --alpha $ALPHA $ARG --jac fwd --max-iter $MAX_ITER --evidence --name $NAME --tag $TAG"
    # shellcheck disable=SC2086
    python -u paper_tests/ouu.py --smoke --n "$N" --alpha "$ALPHA" $ARG \
        --jac fwd --max-iter "$MAX_ITER" --evidence --name "$NAME" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
    echo "== $(date '+%F %T') ouu.py --sweep-te --robust paper_tests/out/$NAME.json"
    python -u paper_tests/ouu.py --sweep-te --robust "paper_tests/out/$NAME.json" --points 16
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
