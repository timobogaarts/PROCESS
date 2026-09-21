#!/bin/bash
# Every table, every machine, in order. ~1-2 h on one CPU.
set -u
PY=${PY:-~/miniconda3/envs/process_port/bin/python}
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:~/jaxgraph/src
cd "$(dirname "$0")/../.."
SCHEME=${SCHEME:-minimal}
NAMES=${NAMES:-"stellarator_helias helias_5b large_tokamak_nof large_tokamak_eval low_aspect_ratio_DEMO spherical_tokamak_eval st_regression"}
# One process per machine and script: XLA's JIT on this box runs out of section memory
# once enough programs have been compiled in one process.
for script in "structure.py --scheme $SCHEME" "iteration.py --scheme $SCHEME" \
              "solve.py --scheme $SCHEME --optimiser vmcon" "solve.py --scheme $SCHEME --optimiser slsqp" \
              "native.py"; do
    for name in $NAMES; do
        echo "== $script $name"
        $PY paper_tests/architectures/$script --configurations $name "$@" 2>&1 | grep -v "Warning\|warn\|B matrix\|^E0921\|LLVM"
    done
done
