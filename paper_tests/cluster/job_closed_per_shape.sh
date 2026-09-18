#!/bin/bash
#SBATCH --job-name=closed_shape_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#
# job_closed.sh, one shape per process: a job array over plain / nested / closed /
# predicted, each on an H100 of its own, so a shape's ladder starts on an empty card.
# In one process (job 26848174) the device memory an earlier shape's ladder held was
# never given back, and `closed` OOMed at N=65536 asking 4.8 GiB, `predicted` at 16384
# asking 1.2 GiB. The script has no wall limit; an OOM at a size is a status row that
# ends the ladder, so sizes past what the card holds cost one row each.
# Submit: cd ~/PROCESS/paper_tests/cluster && sbatch job_closed_per_shape.sh
# Results: ~/PROCESS/paper_tests/out/close_conditions_batch_gpu_<shape>.{csv,tex}
# (one file per shape, never the four-shape file); log: logs/closed_<shape>_h100.log.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

SHAPES=(plain nested closed predicted)
SHAPE="${SHAPES[$SLURM_ARRAY_TASK_ID]}"
SIZES="${SIZES:-1,4,16,64,256,1024,4096,16384,65536,262144,1048576}"
LOG="paper_tests/cluster/logs/closed_${SHAPE}_h100.log"
export JAX_PLATFORMS=cuda      # close_conditions.py reads the backend from the env

{
    echo "job $SLURM_JOB_ID ($SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]) on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "== $(date '+%F %T') close_conditions.py --batch --shapes $SHAPE --sizes $SIZES"
    python -u paper_tests/close_conditions.py --batch --shapes "$SHAPE" --sizes "$SIZES"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
