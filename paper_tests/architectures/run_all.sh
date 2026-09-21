#!/bin/bash
# Every table, every machine, in order. ~1-2 h on one CPU.
set -u
PY=${PY:-~/miniconda3/envs/process_port/bin/python}
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:~/jaxgraph/src
cd "$(dirname "$0")/../.."
SCHEME=${SCHEME:-minimal}
for script in "structure.py --scheme $SCHEME" "iteration.py --scheme $SCHEME" \
              "solve.py --scheme $SCHEME --optimiser vmcon" "solve.py --scheme $SCHEME --optimiser slsqp" \
              "native.py"; do
    echo "== $script"
    $PY paper_tests/architectures/$script "$@" 2>&1 | grep -v "Warning\|warn\|B matrix"
done
