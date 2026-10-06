#!/bin/bash
#SBATCH --job-name=arch_pin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#
# The port's pin (functional_process/tests/test_architectures.py -m tier4) on the
# pinned cottax, on the tables' kind of core: before the tables run, that the cluster
# reproduces what the laptop does.
#
#     cd ~/arch_run/PROCESS/paper_tests/cluster && mkdir -p logs && sbatch job_pin.sh
set -u
ROOT="${ROOT:-$HOME/arch_run/PROCESS}"
source "$ROOT/paper_tests/cluster/env.sh"
source "$ROOT/paper_tests/cluster/pins.env"
cd "$ROOT" || exit 1

export PYTHONPATH="$COTTAX_SRC:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="paper_tests/cluster/logs/pin.log"
mkdir -p paper_tests/cluster/logs

{
    echo "job ${SLURM_JOB_ID:-none} on $(hostname): $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) cores"
    echo "cottax $COMMIT_COTTAX ($COTTAX_SRC), PROCESS $COMMIT_PROCESS"
    python -c "import cottax, jax; print('cottax', cottax.__file__, 'jax', jax.__version__)"
    python -m pytest functional_process/tests/test_architectures.py -m tier4 -q -p no:cacheprovider
    echo "== exit $?"
} > "$LOG" 2>&1
