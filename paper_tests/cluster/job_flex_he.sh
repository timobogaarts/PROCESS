#!/bin/bash
#SBATCH --job-name=flex_he
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=rome
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#
# `job_flex.sh`, for the 2026-09-22 closure: the belief table pruned to PLASMA + LIMITS
# (`configurations.kinds.BELIEFS`) and the helium fraction closed by `c62` inside the
# MDA alongside `c2` by the density (`--pairing he`), instead of being a free second
# operator knob. The operator now re-optimises **one** set-point (T_e) per belief draw,
# in the same two phases as before:
#
#   1.  psi = min_z max_j g_j(z)  (epigraph)   -- is the machine operable in this world?
#   2.  min coe s.t. g <= 0, warm-started from 1 -- and what does it cost there?
#
# Same resources as `job_flex.sh` (96 workers on one exclusive `rome` node, ~7 min at
# N = 262144 once compiled): closing two conditions together rather than one changes
# what the closing Newton solves for, not how many samples a worker walks per second.
#
#   task 0: hfact sigma 0.10     the headline belief
#   task 1: hfact sigma 0.05
#   task 2: hfact sigma 0.02
#
# Submit: cd ~/PROCESS/paper_tests/cluster && mkdir -p logs && sbatch job_flex_he.sh
#         ORDER=hfact sbatch job_flex_he.sh       (the order-corrected economics)
#         N=65536 WORKERS=64 sbatch job_flex_he.sh
#
# Output lands under the `flex_he_` prefix (`TAG=he`), so it never collides with the
# tracked `flex_n262144_s*_hfact*` files `job_flex.sh` writes.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

N="${N:-262144}"
WORKERS="${WORKERS:-96}"
ORDER="${ORDER:-index}"
REQUIRE_NET="${REQUIRE_NET:-0}"
RESTARTS="${RESTARTS:-0}"
PAIRING="${PAIRING:-he}"
TABLE="${TABLE:-flex}"
TAG="${TAG:-he}"
SIGMAS=(0.10 0.05 0.02)
I="${SLURM_ARRAY_TASK_ID:-0}"
SIGMA="${SIGMAS[$I]}"
NAME="flex_${TAG}_n${N}_s${SIGMA}"
[ "$ORDER" = index ] || NAME="${NAME}_${ORDER}"
[ "$REQUIRE_NET" = 0 ] || NAME="${NAME}_req${REQUIRE_NET}"
[ "$RESTARTS" = 0 ] || NAME="${NAME}_r${RESTARTS}"
LOG="paper_tests/cluster/logs/${NAME}.log"
OUT="paper_tests/out/${NAME}.json"

# `job_flex.sh` prepends a `$HOME/cottax_head` archive of cottax 6b1d540 ahead of the
# venv's editable install, because CLAUDE.md's pin predates a later re-port. As of
# 2026-09-22 that pin is stale in the *other* direction: 6b1d540 has no `cottax.pytree`
# subpackage at all (`cottax/__init__.py` is flat there -- `cottax.graph`, `cottax.problem`,
# ... -- `git -C ~/jaxgraph show 6b1d540:src/cottax` lists no `pytree/`), while every
# architecture module this run imports (`closing.py`, `mdf.py`, `sand.py`, `lift.py`, ...)
# now imports from `cottax.pytree.*`, which only exists on the cluster's editable
# `~/jaxgraph` (`~/cottaxvenv` resolves it from there per `setup_venv.sh`). Importing
# through the archived pin fails outright (`ModuleNotFoundError: No module named
# 'cottax.pytree'`), so this job does NOT export `$HOME/cottax_head` onto `PYTHONPATH` --
# it lets the venv's own editable cottax resolve, which is what actually has the package
# this codebase now needs.
export PYTHONPATH="$HOME/PROCESS:$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/flex_jit"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" paper_tests/cluster/logs paper_tests/out

{
    echo "job ${SLURM_JOB_ID:-none} on $(hostname); $(nproc) cores, $(free -g | awk '/Mem:/{print $2}') GB"
    echo "== $(date '+%F %T') $NAME  N=$N workers=$WORKERS pairing=$PAIRING table=$TABLE"
    python -u paper_tests/flexibility.py \
        --n "$N" --which deterministic --sigma "$SIGMA" --knobs te \
        --pairing "$PAIRING" --table "$TABLE" --tag "$TAG" \
        --workers "$WORKERS" --chunks-per-worker 4 \
        --order "$ORDER" --restarts "$RESTARTS" --require-net "$REQUIRE_NET" \
        --jit-cache "$JAX_COMPILATION_CACHE_DIR" --out "$OUT" --condense
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
