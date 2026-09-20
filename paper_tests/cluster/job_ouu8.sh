#!/bin/bash
#SBATCH --job-name=ouu8_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x-%A_%a.out
#
# Variant D under the **bracketed** closure, and the winding-pack lift at scale.
# Both are the 2026-09-19 hand-off's open items, at N = 8192 rather than the N = 64
# the two were first measured at on a CPU.
#
#   D_br   tasks 0-2: D (pairing one, build table, levelised, T_e recourse on 5 points)
#          at alpha 0.75 / 0.9 / 0.95 with `--closure bracketed` -- the density's root
#          find keeping its one unknown, the cycle's Picards nested inside it, a
#          bracket then a safeguarded Newton, so a far trial design converges instead
#          of stalling (commit e8888787; the flattened Newton is what every earlier
#          D run used). Started from PROCESS's own design, **no `--start`**: the
#          earlier D runs' continuation is what this is meant to re-verify without.
#   wp     tasks 3-6: the winding-pack experiment, two arms at alpha 0.75 and 0.9.
#          `wp_base` is D itself; `wp_lift` adds `--lifts winding_pack`, which makes
#          the pack width a design variable (ixc 140) and `j_tf_wp <= f j_c` (icc 33)
#          a chance constraint, and releases `f_j_tf_wp_critical_max` from the build
#          table's held rows -- that row was held *because* the pack is re-sized per
#          sample. The difference in the robust objective is the coil's share of the
#          price of robustness. On CPU at N = 64 (commit db9e5a5a) it was +3.0 %, with
#          the lifted inequality active at the answer; this is the same at scale.
#
# Submit: cd ~/PROCESS/paper_tests/cluster && mkdir -p logs && sbatch job_ouu8.sh
# Results: ~/PROCESS/paper_tests/out/ouu8_alpha<alpha><tag>.json and
# ouu_evidence_<alpha><tag>.{json,npz}; logs: logs/ouu8_alpha<alpha><tag>_h100.log.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

ALPHAS=(0.75 0.9 0.95   0.75 0.75 0.9 0.9)
TAGS=(_D_br _D_br _D_br   _wp_base _wp_lift _wp_base _wp_lift)
EXTRA=("" "" ""   "" "--lifts winding_pack" "" "--lifts winding_pack")
COMMON="--pairing one --table build --objective levelised --closure bracketed --te-recourse 5 --te-range 5,15 --chunk 2"

I="$SLURM_ARRAY_TASK_ID"
ALPHA="${ALPHAS[$I]}"; TAG="${TAGS[$I]}"; ARG="${EXTRA[$I]}"
N="${N:-8192}"
MAX_ITER="${MAX_ITER:-400}"
NAME="ouu8_alpha${ALPHA}${TAG}"
LOG="paper_tests/cluster/logs/${NAME}_h100.log"
export JAX_PLATFORMS=cuda
export PYTHONPATH="$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"

{
    echo "job $SLURM_JOB_ID ($SLURM_ARRAY_JOB_ID[$I]) on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "== $(date '+%F %T') ouu.py --smoke --n $N --alpha $ALPHA $COMMON $ARG --jac fwd --max-iter $MAX_ITER --evidence --name $NAME --tag $TAG"
    # shellcheck disable=SC2086
    python -u paper_tests/ouu.py --smoke --n "$N" --alpha "$ALPHA" $COMMON $ARG \
        --jac fwd --max-iter "$MAX_ITER" --evidence --name "$NAME" --tag "$TAG"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
