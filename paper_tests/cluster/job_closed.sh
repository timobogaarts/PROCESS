#!/bin/bash
#SBATCH --job-name=closed_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# The batched closed MDA (paper_tests/close_conditions.py --batch): the four shapes
# (plain / nested / closed / predicted) vmapped over N points on one H100. The script
# has no wall limit; an OOM (or any failure) at a size is written as a status row and
# ends that shape's ladder, so sizes past what the card holds cost one row, not the
# job. N need not be a square here.
# Submit: cd ~/PROCESS/paper_tests/cluster && sbatch job_closed.sh
# Results: ~/PROCESS/paper_tests/out/close_conditions_batch_gpu.{csv,tex} (rows carry
# `backend` only, no device -- keep them in out_cluster/); log: logs/closed_h100.log.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

SIZES="${SIZES:-1,4,16,64,256,1024,4096,16384,65536,262144}"
LOG="paper_tests/cluster/logs/closed_h100.log"
export JAX_PLATFORMS=cuda      # close_conditions.py reads the backend from the env

{
    echo "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "== $(date '+%F %T') close_conditions.py --batch --sizes $SIZES"
    python -u paper_tests/close_conditions.py --batch --sizes "$SIZES"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
