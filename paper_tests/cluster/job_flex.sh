#!/bin/bash
#SBATCH --job-name=flex_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=rome
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#
# Flexibility analysis of the deterministic stellarator design, at a sample count the
# laptop cannot reach. The build is fixed; per belief draw the operator re-optimises
# the two set-points (T_e, the helium fraction) over the closed MDA, in two phases:
#
#   1.  psi = min_z max_j g_j(z)  (epigraph)   -- is the machine operable in this world?
#   2.  min coe s.t. g <= 0, warm-started from 1 -- and what does it cost there?
#
# The draws are independent, so this is `multiprocessing` over the node's cores; each
# worker traces and compiles the per-sample MDA once (~130 s cold on a rome core, less
# against JAX's persistent compilation cache) and then walks contiguous slices at a
# **measured 156 ms a draw per core** (srun smoke, N=4096, 96 workers, 2026-09-19;
# the same draw costs ~55 ms on a Ryzen 3700X core).
#
# 96 workers, not 128: a worker's peak RSS is 2.5 GB on this node, and 128 of them
# would be 320 GB against the node's 224 GB (251 GB with --exclusive). At 96 the
# nominal 240 GB is only survivable because most of that RSS is shared library text,
# which the smoke run confirmed.
#
# Wall, from the measured rate: N * 0.156 s / 96 workers of solving -- ~7 min at
# N = 262144 -- plus ~40 s of model build per worker and one compile. 3 h is slack.
#
#   task 0: hfact sigma 0.10     the headline belief
#   task 1: hfact sigma 0.05     the operability fraction is stable across these; the
#   task 2: hfact sigma 0.02     *sensitivity ranking* is what N is being bought for
#
# Submit: cd ~/PROCESS/paper_tests/cluster && mkdir -p logs && sbatch job_flex.sh
#         ORDER=hfact sbatch job_flex.sh          (the order-corrected economics)
#         N=65536 WORKERS=64 sbatch job_flex.sh   (anything smaller)
# Measured 2026-09-19: N=262144 x 3 sigmas, 96 workers on one exclusive rome node
# each, 6 min 20 s wall per task (22 s model build, 33 s compile against a warm JAX
# cache, ~250 s of solving at 92 ms a draw per core), 2.25 GB peak RSS per worker.
set -u
source "$HOME/PROCESS/paper_tests/cluster/env.sh"
cd "$HOME/PROCESS" || exit 1

N="${N:-262144}"
WORKERS="${WORKERS:-96}"
# `index` walks the draws in Sobol' order, which is what flex.py did and what the
# N=1024 reference reproduces. `hfact` sorts them by the dominant belief's quantile so
# each draw's warm start comes from a near neighbour. That is worth ~10 % of the
# per-draw cost, and far more than that in *quality*: phase 2 is a local solve of a
# non-convex problem, and in index order it strands ~5 % of the operable draws at a
# near-zero-net-power point (coe p95 795 against 310, N=1024). RESTARTS=1|2 adds
# further phase-2 starts; measured, it buys nothing once the draws are ordered.
ORDER="${ORDER:-index}"
# c16 is not in the constraint set. 0 = "can it be run at all"; set it to the
# PORT's nominal (982.4) for "can it be run at its rated power" -- not the
# file's 1000.0, which the port misses at the nominal by its own divergence.
REQUIRE_NET="${REQUIRE_NET:-0}"
RESTARTS="${RESTARTS:-0}"
SIGMAS=(0.10 0.05 0.02)
I="${SLURM_ARRAY_TASK_ID:-0}"
SIGMA="${SIGMAS[$I]}"
NAME="flex_n${N}_s${SIGMA}"
[ "$ORDER" = index ] || NAME="${NAME}_${ORDER}"
[ "$REQUIRE_NET" = 0 ] || NAME="${NAME}_req${REQUIRE_NET}"
[ "$RESTARTS" = 0 ] || NAME="${NAME}_r${RESTARTS}"
LOG="paper_tests/cluster/logs/${NAME}.log"
OUT="paper_tests/out/${NAME}.json"

# The port tracks cottax 6b1d540; the cluster's ~/jaxgraph has moved past it, and the
# venv resolves cottax editable from there. PYTHONPATH precedes site-packages, so the
# read-only export wins -- the same trick CLAUDE.md documents for a worktree locally.
export PYTHONPATH="$HOME/cottax_head/src:$HOME/PROCESS:$HOME/PROCESS/paper_tests${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
# One worker per core: every library that would start a thread pool of its own is
# pinned to one thread, or 96 processes each spawning ~128 threads thrash the node.
# OpenBLAS is also the reason to pin it rather than leave it: its reduction order
# depends on the thread count, and SLSQP's iterates are sensitive to the last bits.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/flex_jit"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" paper_tests/cluster/logs paper_tests/out

{
    echo "job ${SLURM_JOB_ID:-none} on $(hostname); $(nproc) cores, $(free -g | awk '/Mem:/{print $2}') GB"
    echo "== $(date '+%F %T') $NAME  N=$N workers=$WORKERS"
    python -u paper_tests/flexibility.py \
        --n "$N" --which deterministic --sigma "$SIGMA" --knobs both \
        --workers "$WORKERS" --chunks-per-worker 4 \
        --order "$ORDER" --restarts "$RESTARTS" --require-net "$REQUIRE_NET" \
        --jit-cache "$JAX_COMPILATION_CACHE_DIR" --out "$OUT"
    echo "== $(date '+%F %T') exit $?"
} > "$LOG" 2>&1
