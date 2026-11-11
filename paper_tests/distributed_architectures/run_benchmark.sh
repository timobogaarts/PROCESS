#!/bin/bash
# AAO, sequential and ALC on Helias, one process, cases one after the other (benchmark.py):
# `--runs` warm solves of each after its cold one. ALC is ~4 min a run on one CPU, so the
# default (5 runs) takes ~30 min. Everything is logged to out/benchmark.log.
#
#   paper_tests/distributed_architectures/run_benchmark.sh [--runs 5] [--cases aao sequential alc]
#   PY=... COTTAX=... paper_tests/distributed_architectures/run_benchmark.sh
set -u
PY=${PY:-~/anaconda3/envs/func_PROCESS_env/bin/python}
COTTAX=${COTTAX:-~/projects/jaxgraph/src}
cd "$(dirname "$0")/../.."                          # the configurations load by relative path
HERE=paper_tests/distributed_architectures
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$COTTAX:.:$HERE
# As the architecture-variant sweep and the Snellius jobs: single-threaded BLAS, which SLSQP
# calls, so its path does not depend on the thread count.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}} XDG_CACHE_HOME=${XDG_CACHE_HOME:-${TMPDIR:-/tmp}}
mkdir -p $HERE/out
$PY $HERE/benchmark.py "$@" 2>&1 | grep --line-buffered -v -i -E "B matrix|^E0921|LLVM" | tee $HERE/out/benchmark.log
