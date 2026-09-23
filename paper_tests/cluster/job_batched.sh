#!/bin/bash
#SBATCH --job-name=arch_batched
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x-%A_%a.out
#
# The batched table (paper_tests/architectures/batched.py): the optimiser's block over
# N designs at once, per design, one machine per array task. The partition decides the
# platform, and is given on the command line:
#
#     sbatch -p genoa    --cpus-per-task=24          job_batched.sh   # CPU, the serial table's cores
#     sbatch -p gpu_h100 --gpus=1 --cpus-per-task=16 job_batched.sh   # one H100 (float64 at half rate)
#
# The same N on both platforms, so the two are compared at every N; the GPU goes on to
# the batches only it can take. Past --chunk designs the batch is lax.map'ped over
# chunks of one vmap.
set -u
# The mirror to run from: ~/arch_run/PROCESS for these jobs (sync_to_cluster.sh with
# REMOTE_HOME=~/arch_run), not the flexibility study's ~/PROCESS.
ROOT="${ROOT:-$HOME/arch_run/PROCESS}"
source "$ROOT/paper_tests/cluster/env.sh"
source "$ROOT/paper_tests/cluster/pins.env"
cd "$ROOT" || exit 1

NAMES=(stellarator_helias helias_5b large_tokamak_nof large_tokamak_eval
       low_aspect_ratio_DEMO spherical_tokamak_eval st_regression)
NAME="${NAMES[${SLURM_ARRAY_TASK_ID:-0}]}"
SCHEME="${SCHEME:-minimal}"

export PYTHONPATH="$COTTAX_SRC:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1
if [ -n "${SLURM_GPUS:-}${SLURM_GPUS_ON_NODE:-}${CUDA_VISIBLE_DEVICES:-}" ]; then
    export JAX_PLATFORMS=cuda
    BATCHES="${BATCHES:-1 16 256 4096 16384 65536 262144}"
else
    export JAX_PLATFORMS=cpu
    BATCHES="${BATCHES:-1 16 256 4096 16384}"
fi
CHUNK="${CHUNK:-4096}"
LOG="paper_tests/cluster/logs/batched_${JAX_PLATFORMS}_${NAME}.log"
mkdir -p paper_tests/cluster/logs

{
    echo "job ${SLURM_JOB_ID:-none} on $(hostname): $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) cores"
    [ "$JAX_PLATFORMS" = cuda ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "cottax $COMMIT_COTTAX ($COTTAX_SRC), PROCESS $COMMIT_PROCESS"
    python -c "import cottax, jax; print('cottax', cottax.__file__, 'jax', jax.__version__, jax.devices())"
    # One process per arm: every N is two compiles, and XLA's CPU JIT runs out of
    # section memory ("LLVM ERROR: Unable to allocate section memory") once some tens
    # of programs are compiled in one process. batched.py merges each arm's rows in.
    for ARM in ${ARMS:-MDF IDF SAND}; do
        echo "== $(date '+%F %T') batched.py $NAME $ARM N=$BATCHES chunk=$CHUNK"
        # shellcheck disable=SC2086
        python -u paper_tests/architectures/batched.py --scheme "$SCHEME" --configurations "$NAME" \
            --arms "$ARM" --batches $BATCHES --chunk "$CHUNK"
        rc=$?   # before the echo: inside it, $? would be the $(date ...)'s
        echo "== $(date '+%F %T') exit $rc"
    done
} > "$LOG" 2>&1
