"""What the UQ drivers share: the environment pinned, PROCESS's own converged design,
and where the reference run is cached.

The drivers (`flexibility.py`, `flexibility_dsm.py`, `ga.py`) are thin command lines
over the port (`functional_process.cottax.architectures`, `functional_process
.configurations`): the physics, the graph operations and the statistics live there.
What lives here is the one thing the port deliberately does not carry -- PROCESS
itself in the loop (`process_reference`): one converged VMCON run per input file,
which is the deterministic optimum every study starts from and is compared against.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CACHE_DIR = Path(
    os.environ.get("FP_HARNESS_CACHE_DIR", Path.home() / ".cache/functional_process")
)
"""Where PROCESS's reference run lands -- the same directory the deleted harness used,
so a cluster's `env.sh` still points at it."""

sys.path.insert(0, str(ROOT))


def stem(path: str) -> str:
    """`.../helias_5b.IN.DAT` -> `helias_5b`; a configuration name is its own stem."""
    return Path(path).name.replace(".IN.DAT", "")


def input_file(name: str) -> Path:
    """The regression `IN.DAT` a configuration name or path refers to, absolute."""
    path = Path(name)
    if path.suffix == ".DAT" or path.exists():
        return path if path.is_absolute() else (ROOT / path).resolve()
    return (ROOT / "tests/regression/input_files" / f"{stem(name)}.IN.DAT").resolve()


# ---------------------------------------------------------------- PROCESS itself

_REFERENCE_VERSION = "process-reference-v1"


def process_reference(name: str = "stellarator_helias", use_cache: bool = True) -> dict:
    """PROCESS's own converged answer for one regression file: `x` (`{spelling:
    value}` over its `ixc`, the port's spellings), `iterations` (VMCON's count),
    `outputs` (net electric power, coe, concost, fusion power, the density) and
    `seconds`. Run in-process (`process.main.SingleRun`, ~1-2 min for the
    stellarator) and cached as JSON under `CACHE_DIR`, keyed by the file's content.

    The port's `session` is deliberately PROCESS-free; a comparison column against
    PROCESS is this directory's business, so the run lives here.
    """
    path = input_file(name)
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    cached = CACHE_DIR / f"{_REFERENCE_VERSION}-{digest}.json"
    if use_cache and cached.exists():
        return json.loads(cached.read_text())
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from functional_process.cottax.architectures.sand import (  # noqa: PLC0415
        iteration_variable_path,
    )
    from process.core.solver.iteration_variables import (  # noqa: PLC0415
        ITERATION_VARIABLES,
    )
    from process.main import SingleRun  # noqa: PLC0415

    scratch = Path(tempfile.mkdtemp())
    shutil.copy(path, scratch / path.name)
    companion = path.with_name(f"{stem(str(path))}.stella_conf.json")
    if companion.exists():
        shutil.copy(companion, scratch / companion.name)
    run = SingleRun(str(scratch / path.name), "vmcon")
    began = time.perf_counter()
    run.run()
    seconds = time.perf_counter() - began
    data = run.data
    n = int(data.numerics.n_iteration_variables)
    ixc = [int(i) for i in data.numerics.ixc[:n]]

    def value(i):
        iteration_variable = ITERATION_VARIABLES[i]
        area = getattr(data, iteration_variable.module)
        field = getattr(area, iteration_variable.target_name or iteration_variable.name)
        if iteration_variable.array_index is None:
            return float(field)
        return float(field[iteration_variable.array_index])

    result = {
        "file": str(path),
        "ixc": ixc,
        "x": {iteration_variable_path(i).spelling: value(i) for i in ixc},
        "iterations": int(data.numerics.n_solver_iterations),
        "convergence_parameter": float(data.globals.convergence_parameter),
        "outputs": {
            ".heat_transport.p_plant_electric_net_mw": float(
                data.heat_transport.p_plant_electric_net_mw
            ),
            ".costs.coe": float(data.costs.coe),
            ".costs.concost": float(data.costs.concost),
            ".physics.p_fusion_total_mw": float(data.physics.p_fusion_total_mw),
            ".physics.nd_plasma_electrons_vol_avg": float(
                data.physics.nd_plasma_electrons_vol_avg
            ),
            "^cond.numerics.objf": float(data.costs.coe) / 100.0,
        },
        "seconds": seconds,
    }
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(result, indent=1))
    return result


def deterministic_values(live, pairing: str = "one") -> tuple[list, dict]:
    """PROCESS's converged design for `live`'s configuration, as `ouu.two_stage` /
    `closing.seed` take it: `design_values` (one per design place kept, `ixc` order
    minus the closing variables of `kinds.PAIRINGS[pairing]`) and `closing_values`
    (`{closing variable: value}`). The deterministic optimum every study here starts
    from or is compared against.
    """
    from functional_process.configurations import kinds  # noqa: PLC0415
    from functional_process.cottax.architectures.sand import (  # noqa: PLC0415
        iteration_variable_path,
    )

    process = process_reference(live.name)
    closing = set(kinds.PAIRINGS[pairing].values())
    design = [iteration_variable_path(i).spelling for i in live.reference.ixc]
    return (
        [process["x"][v] for v in design if v not in closing],
        {v: process["x"][v] for v in closing},
    )
