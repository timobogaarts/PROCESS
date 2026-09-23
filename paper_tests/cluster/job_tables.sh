#!/bin/bash
#SBATCH --job-name=arch_tables
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x-%A_%a.out
#
# The serial architecture table (paper_tests/architectures/README.md), one machine per
# array task, on 1/8 of a genoa node (24 cores, the smallest share the partition
# gives). Everything it times is one thread of work -- the models are scalar, the
# drivers are Python loops -- so most of the share idles; it is taken so that every
# machine, and PROCESS itself, is timed on the same kind of core.
#
# Per machine, in the order run_all.sh uses: structure, iteration, solve (vmcon, then
# slsqp), native (PROCESS's own run). Each is its own process (XLA's JIT runs out of
# section memory once enough programs are compiled in one).
#
# Submit (after sync_to_cluster.sh and pin.sh):
#     cd ~/arch_run/PROCESS/paper_tests/cluster && mkdir -p logs && sbatch job_tables.sh
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
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
# PROCESS's numpy and SLSQP's scipy on one BLAS thread, as the laptop's rows effectively
# were: OpenBLAS's reduction order depends on the thread count, and the SQPs see the
# last bits (ulp.py).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="paper_tests/cluster/logs/tables_${NAME}.log"
mkdir -p paper_tests/cluster/logs

{
    echo "job ${SLURM_JOB_ID:-none} on $(hostname): $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) cores"
    echo "cottax $COMMIT_COTTAX ($COTTAX_SRC), PROCESS $COMMIT_PROCESS"
    python -c "import cottax, jax; print('cottax', cottax.__file__, 'jax', jax.__version__)"
    for script in "structure.py --scheme $SCHEME" "iteration.py --scheme $SCHEME" \
                  "solve.py --scheme $SCHEME --optimiser vmcon" "solve.py --scheme $SCHEME --optimiser slsqp" \
                  "native.py"; do
        echo "== $(date '+%F %T') $script $NAME"
        python -u paper_tests/architectures/$script --configurations "$NAME"
        echo "== $(date '+%F %T') exit $?"
    done
} > "$LOG" 2>&1
