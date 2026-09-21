"""Table 4 -- PROCESS itself on the same machines, the same CPU, the same tolerance:
what the port's numbers are compared against.

Per machine, at the file's own design: one pass of the models (`Caller._call_models_once`,
what one sweep of the port's Gauss-Seidel corresponds to), the idempotence loop PROCESS
actually evaluates per optimiser call (`call_models`: passes until the objective and the
constraints stop moving), and the finite-difference gradient VMCON takes (`fcnvmc2`:
two passes per design variable); then the full run (`SingleRun.run`) with `epsvmc` set
to `bench.TOLERANCE`, its iterations and its answer.

    $PY paper_tests/architectures/native.py [--configurations ...]

PROCESS writes its output files next to the input, so each run gets a scratch copy.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

from process.core.caller import Caller  # noqa: E402
from process.core.solver import constraints  # noqa: E402
from process.core.solver.evaluators import Evaluators  # noqa: E402
from process.core.solver.iteration_variables import load_iteration_variables, load_scaled_bounds  # noqa: E402
from process.core.solver.objectives import objective_function  # noqa: E402
from process.main import SingleRun  # noqa: E402

INPUTS = bench.ROOT / "tests" / "regression" / "input_files"


def scratch_copy(name: str) -> Path:
    src = INPUTS / f"{name}.IN.DAT"
    folder = Path(tempfile.mkdtemp(prefix=f"process_{name}_"))
    shutil.copy(src, folder / src.name)
    companion = INPUTS / f"{name}.stella_conf.json"
    if companion.exists():
        shutil.copy(companion, folder / companion.name)
    return folder / src.name


def evaluation_row(name: str, repeats: int) -> dict:
    """One pass, the idempotence loop and the FD gradient at the file's design."""
    run = SingleRun(str(scratch_copy(name)), "vmcon")
    data = run.data
    load_iteration_variables(data)
    load_scaled_bounds(data)
    n = int(data.numerics.n_iteration_variables)
    m = int(data.numerics.n_equality_constraints + data.numerics.n_inequality_constraints)
    x = np.array(data.numerics.xcm[:n])
    caller = Caller(run.models, data)
    evaluators = Evaluators(run.models, data, x)
    caller._call_models_once(x)                                  # warm the models once
    one_pass = bench.median_seconds(lambda: caller._call_models_once(x), repeats)
    loop = bench.median_seconds(lambda: caller.call_models(x, m), repeats)
    gradient = bench.median_seconds(lambda: evaluators.fcnvmc2(n, m, x, m), max(1, repeats // 2))
    return {
        "design": n,
        "constraints": m,
        "pass_ms": 1e3 * one_pass,
        "loop_ms": 1e3 * loop,
        "gradient_ms": 1e3 * gradient,
    }


def solve_row(name: str) -> dict:
    """The full run at `bench.TOLERANCE`."""
    run = SingleRun(str(scratch_copy(name)), "vmcon")
    data = run.data
    data.numerics.epsvmc = bench.TOLERANCE
    _, seconds = bench.timed(run.run)
    m = int(data.numerics.n_equality_constraints + data.numerics.n_inequality_constraints)
    meq = int(data.numerics.n_equality_constraints)
    conf, *_ = constraints.constraint_eqns(m, -1, data)
    conf = np.asarray(conf, dtype=float)
    optimising = int(data.numerics.ioptimz) >= 0 if hasattr(data.numerics, "ioptimz") else True
    return {
        "iterations": int(data.numerics.n_solver_iterations),
        "objf": float(objective_function(data.numerics.i_figure_merit, data)) if optimising else float("nan"),
        "max_eq": float(np.max(np.abs(conf[:meq]))) if meq else 0.0,
        "min_ie": float(np.min(conf[meq:])) if m > meq else float("nan"),
        "convergence": float(data.globals.convergence_parameter),
        "solve_s": seconds,
    }


def main():
    args = bench.arguments(__doc__, optimiser=False)
    rows = []
    for name in args.configurations:
        row = {"configuration": name, **evaluation_row(name, args.repeats), **solve_row(name)}
        rows.append(row)
        print(row)
    bench.write("native.py", rows, "native")


if __name__ == "__main__":
    main()
