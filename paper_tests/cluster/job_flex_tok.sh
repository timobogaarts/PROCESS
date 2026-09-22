#!/bin/bash
#SBATCH --job-name=flex_tok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=rome
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#
# Flexibility analysis of the deterministic *tokamak* design (large_tokamak_nof): the
# copy of job_flex.sh for `flexibility_tokamak.py` and its `flextok_` outputs. The
# build (twelve build ixc, q95, the current-drive share, the installed H&CD power) is
# fixed at PROCESS's converged design; per belief draw the operator re-optimises the
# density, the temperature and the xenon seeding over the closed MDA (c2 by the heating
# power, c62 by the helium fraction, c1 by the beta), in two phases:
#
#   1.  psi = min_z max_j g_j(z)  (epigraph)   -- is the machine operable in this world?
#   2.  min coe s.t. g <= max(tol, psi), warm-started from 1 -- what does it cost there?
#
# Measured locally (Ryzen, 2026-09-22): ~0.12 s a draw per core after a ~60 s model
# build and ~70 s compile per worker; peak RSS 3.6 GB per worker locally and 4.5 GB on
# rome. 64 workers were OOM-killed on the node (2026-09-22, 64 x 4.5 GB against 224 GB
# -- unlike job_flex.sh's workers, little of it is shared); 32 workers ran N = 2048 in
# 100 s. 40 is the default: ~16 min of solving at N = 262144.
#
#   task 0: hfact sigma 0.10     the headline belief
#   task 1: hfact sigma 0.05
#   task 2: hfact sigma 0.02
#
# Submit: cd ~/PROCESS/paper_tests/cluster && mkdir -p logs && sbatch job_flex_tok.sh
#         ORDER=hfact sbatch job_flex_tok.sh
#         N=65536 WORKERS=64 sbatch job_flex_tok.sh
#         DROP_C16=1 sbatch job_flex_tok.sh        ("can it be run at all")
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

N="${N:-262144}"
WORKERS="${WORKERS:-40}"
ORDER="${ORDER:-hfact}"
DROP_C16="${DROP_C16:-0}"
SIGMAS=(0.10 0.05 0.02)
I="${SLURM_ARRAY_TASK_ID:-0}"
SIGMA="${SIGMAS[$I]}"
NAME="flextok_n${N}_s${SIGMA}_${ORDER}"
[ "$DROP_C16" = 0 ] || NAME="${NAME}_noc16"
LOG="paper_tests/cluster/logs/${NAME}.log"
OUT="paper_tests/out/${NAME}.json"

# The port tracks cottax 6488a1c (`cottax.mdao_architectures`, `cottax.pytree`); the
# cluster's ~/jaxgraph may have moved, so a read-only export goes first on PYTHONPATH:
#   mkdir -p ~/cottax_6488a1c && git -C ~/jaxgraph archive 6488a1c src | tar -x -C ~/cottax_6488a1c
export PYTHONPATH="$HOME/cottax_6488a1c/src:$HOME/PROCESS:$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/flextok_jit"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" paper_tests/cluster/logs paper_tests/out

EXTRA=()
[ "$DROP_C16" = 0 ] || EXTRA+=(--drop-c16)
{
    echo "job ${SLURM_JOB_ID:-none} on $(hostname); $(nproc) cores, $(free -g | awk '/Mem:/{print $2}') GB"
    echo "== $(date '+%F %T') $NAME  N=$N workers=$WORKERS"
    python -u paper_tests/flexibility_tokamak.py \
        --n "$N" --sigma "$SIGMA" --workers "$WORKERS" --chunks-per-worker 4 \
        --order "$ORDER" "${EXTRA[@]}" \
        --jit-cache "$JAX_COMPILATION_CACHE_DIR" --out "$OUT"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
