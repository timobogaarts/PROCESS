#!/bin/bash
#SBATCH --job-name=batching_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#
# The batched-MDA scaling study (paper_tests/batching.py) on one H100:
#   1. the CPU ladder on this node's CPUs -- it includes N=64 and N=1024, the float64
#      reference arrays the GPU rows diff themselves against (`*_vs_cpu_f64`), which
#      live under $FP_HARNESS_CACHE_DIR/batching.arrays and are empty on the cluster;
#   2. the GPU ladder, extended past the RTX 3080's OOM point (80 GB here);
#   3. --hardware (matmul / elementwise / SVD / solve f64:f32 on the H100);
#   4. --render (csv + tex from out/batching.json) and plot_batching.py (the pngs).
# Submit from this directory: cd ~/PROCESS/paper_tests/cluster && sbatch job_batching.sh
# Results: ~/PROCESS/paper_tests/out/batching.{json,csv,tex,png}; log: logs/batching_h100.log.
#
# Knobs (environment, e.g. `SIZES=1,4,16 sbatch job_batching.sh`):
#   SIZES      GPU ladder; every N of the `mda` shape must be a square
#   CPU_SIZES  CPU ladder; the minimum the diff cache needs is 64,1024
#   REPEATS    warm calls per row (batching.py's default is 3)
#   BATCHING_WALL_LIMIT  seconds of warm wall past which a shape's ladder stops;
#              batching.py's constant is 60, which would end the tokamak MDA ladder
#              around N=65536 (1.9 ms/pt); run_batching.py lifts it
#   SHAPES / INPUTS  extra `--shape` / `--input` arguments passed through unchanged
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

SIZES="${SIZES:-1,4,16,64,256,1024,4096,16384,65536,262144}"
CPU_SIZES="${CPU_SIZES:-1,4,16,64,256,1024,4096}"
REPEATS="${REPEATS:-3}"
export BATCHING_WALL_LIMIT="${BATCHING_WALL_LIMIT:-900}"
EXTRA=(${SHAPES:+--shape $SHAPES} ${INPUTS:+--input $INPUTS})
LOG="paper_tests/cluster/logs/batching_h100.log"
RUN="python -u paper_tests/cluster/run_batching.py"

run() {  # one step; a failed step is logged and the next one still runs
    echo "== $(date '+%F %T') $*"
    "$@"
    echo "== $(date '+%F %T') exit $?"
}
{
    echo "job $SLURM_JOB_ID on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    run $RUN --backend cpu --sizes "$CPU_SIZES" --repeats "$REPEATS" "${EXTRA[@]}"
    run $RUN --backend gpu --sizes "$SIZES" --repeats "$REPEATS" "${EXTRA[@]}"
    run $RUN --backend gpu --hardware
    run $RUN --render
    run python -u paper_tests/plot_batching.py
} > "$LOG" 2>&1
