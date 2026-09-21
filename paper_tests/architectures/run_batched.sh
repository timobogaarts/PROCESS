#!/bin/bash
# The batched table on the CPU, then on the GPU, one process per machine. ~30 min.
set -u
cd "$(dirname "$0")/../.."
export JAX_ENABLE_X64=1 PYTHONPATH=~/PROCESS:~/jaxgraph/src XLA_PYTHON_CLIENT_PREALLOCATE=false
NAMES=${NAMES:-"stellarator_helias helias_5b large_tokamak_nof large_tokamak_eval low_aspect_ratio_DEMO spherical_tokamak_eval st_regression"}
for name in $NAMES; do
    echo "== cpu $name"
    JAX_PLATFORMS=cpu ~/miniconda3/envs/process_port/bin/python paper_tests/architectures/batched.py \
        --configurations $name --batches 1 16 256 4096 "$@" 2>&1 | grep -v "Warning\|warn\|^E0921\|LLVM"
done
for name in $NAMES; do
    echo "== gpu $name"
    JAX_PLATFORMS=cuda ~/miniconda3/envs/process_port_gpu/bin/python paper_tests/architectures/batched.py \
        --configurations $name --batches 1 256 1024 4096 16384 "$@" 2>&1 | grep -v "Warning\|warn\|^E0921\|^W0921\|Source Location\|^external\|allocat\|^$"
done
