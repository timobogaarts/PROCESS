#!/bin/bash
# Run ONCE on a Snellius login node, after sync_to_cluster.sh:
#     bash ~/PROCESS/paper_tests/cluster/setup_venv.sh
# Builds $HOME/cottaxvenv (a venv of the module Python 3.13) holding jax with CUDA 12
# wheels pinned to the version the local process_port_gpu env runs, cottax (editable,
# with its `solvers` extra) and PROCESS (editable), then smoke-checks both imports.
#
# PROCESS is pure Python (hatchling, no compiled sources; hatch-vcs falls back to
# version 0.0.0 without .git), and every dependency has a Linux wheel for 3.13
# (CoolProp via its cp312-abi3 wheel), so no cmake / gfortran and no build flags. The
# full `process` package IS needed: functional_process imports process.core.model
# and friends throughout, and the harness runs PROCESS's own VMCON (SingleRun) once
# per input file to build its converged-reference cache.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HOME/cottaxvenv}"

# The versions in ~/miniconda/envs/process_port_gpu on 2026-09-17 (jax 0.11.1 +
# CUDA 12 plugin; the solver stack cottax's drivers run on). Exact pins so the cluster
# rows are measured on the same numerics as the RTX 3080 rows.
JAX_VERSION="${JAX_VERSION:-0.11.1}"
EQUINOX_VERSION="${EQUINOX_VERSION:-0.13.8}"
OPTIMISTIX_VERSION="${OPTIMISTIX_VERSION:-0.1.0}"
LINEAX_VERSION="${LINEAX_VERSION:-0.1.1}"

# shellcheck disable=SC1091
source "$HERE/env.sh"           # modules only: the venv does not exist yet
if [ -d "$VENV" ]; then
    echo "$VENV exists; remove it first to rebuild (rm -rf $VENV)"
    exit 1
fi
python -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# jax first, so nothing below decides the jaxlib.
pip install "jax[cuda12]==$JAX_VERSION"
pip install "equinox==$EQUINOX_VERSION" "optimistix==$OPTIMISTIX_VERSION" "lineax==$LINEAX_VERSION"
# pytest is a *runtime* import of the port (functional_process/cottax/_harness/__init__.py
# -> contracts.py), so it is installed without PROCESS's whole [test] extra (jupyter).
pip install pytest
pip install -e "$HOME/jaxgraph[solvers]"
pip install -e "$HOME/PROCESS"
# Re-assert the jax pin last (the README's recipe installs jax[cuda] last for this
# reason): a no-op if nothing above moved it.
pip install "jax[cuda12]==$JAX_VERSION"
pip check || echo "pip check reported conflicts (see above); the smoke check decides"

mkdir -p "$HERE/logs"

echo "== smoke check"
# functional_process resolves its reference input file relative to the working
# directory at import (indat.py -> tests/regression/input_files/...), as the jobs do.
cd "$HOME/PROCESS" || exit 1
python - <<'PY'
import jax, jaxlib, cottax, equinox, optimistix
print("python", __import__("sys").version.split()[0])
print("jax", jax.__version__, "jaxlib", jaxlib.__version__,
      "equinox", equinox.__version__, "optimistix", optimistix.__version__)
print("devices", jax.devices(), "(CpuDevice on a login node; the H100 appears inside a job)")
print("cottax", cottax.__file__)
import process
print("process", process.__file__)
import functional_process.cottax.mda as mda
print("functional_process.cottax.mda", mda.__file__)
PY
echo "== ok: $VENV"
