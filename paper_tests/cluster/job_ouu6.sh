#!/bin/bash
#SBATCH --job-name=ouu6_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#
# D again (see job_ouu4.sh) with 9 temperatures, `--gtol 1e-2` (the grid recourse's
# CVaRs jump as samples switch points, so 1e-4 is never met) and shorter outer calls:
#   D    tasks 0-2: alpha 0.75 / 0.9 / 0.95, `levelised`, no c16, **T_e as recourse**
#        (`--te-recourse 5 --te-range 5,15`: per sample the cheapest feasible of 5
#        temperatures between 5 and 15 keV, the outer derivative at the chosen point),
#        N = 8192 and 2 checkpointed chunks: the batch's Newton runs to the slowest
#        (sample, T_e) row, so the full 3-15 keV grid of 8 at N = 16384 was 17 s a call.
#   D16  task 3: D at 0.9 with c16 kept as a CVaR at `--alpha16 0.5`.
#   tasks 4-7: `--evidence` and `--sweep-te` re-run for the B2 designs (0.75, 0.9,
#        0.95, 0.9 from the second start) with the per-sample feasibility test at
#        `G_TOL` (the active radial build sat at +1e-9 and read as violated everywhere).
# Submit: cd ~/PROCESS/paper_tests/cluster && sbatch job_ouu4.sh
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

ALPHAS=(0.75 0.9 0.95 0.9   0.75 0.9 0.95 0.9)
TAGS=(_D9 _D9 _D9 _D9_16   _B2 _B2 _B2 _B2_start)
ARGS=(
  "--pairing one --table build --objective levelised --te-recourse 9 --te-range 5,15 --chunk 4 --n 8192 --gtol 1e-2 --max-iter 150"
  "--pairing one --table build --objective levelised --te-recourse 9 --te-range 5,15 --chunk 4 --n 8192 --gtol 1e-2 --max-iter 150"
  "--pairing one --table build --objective levelised --te-recourse 9 --te-range 5,15 --chunk 4 --n 8192 --gtol 1e-2 --max-iter 150"
  "--pairing one --table build --objective levelised --te-recourse 9 --te-range 5,15 --chunk 4 --n 8192 --gtol 1e-2 --max-iter 150 --with-c16 --alpha16 0.5"
  "" "" "" ""
)
I="$SLURM_ARRAY_TASK_ID"
ALPHA="${ALPHAS[$I]}"; TAG="${TAGS[$I]}"; ARG="${ARGS[$I]}"
N="${N:-16384}"
MAX_ITER="${MAX_ITER:-400}"
NAME="ouu2_alpha${ALPHA}${TAG}"
LOG="paper_tests/cluster/logs/${NAME}_h100.log"
export JAX_PLATFORMS=cuda
export PYTHONPATH="$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"

if [ "$I" -le 3 ]; then
{
    echo "job $SLURM_JOB_ID ($SLURM_ARRAY_JOB_ID[$I]) on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "== $(date '+%F %T') ouu.py --smoke --alpha $ALPHA $ARG --jac fwd --max-iter $MAX_ITER --evidence --name $NAME --tag $TAG"
    # shellcheck disable=SC2086
    python -u paper_tests/ouu.py --smoke --alpha "$ALPHA" $ARG \
        --jac fwd --max-iter "$MAX_ITER" --evidence --name "$NAME" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
else
{
    echo "== $(date '+%F %T') re-evidence and sweep for $NAME"
    python -u paper_tests/ouu.py --evidence --robust "paper_tests/out/$NAME.json" --alpha "$ALPHA" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
    python -u paper_tests/ouu.py --sweep-te --robust "paper_tests/out/$NAME.json" --points 16
    echo "== $(date '+%F %T') exit $?"
} >> "$LOG" 2>&1
fi
