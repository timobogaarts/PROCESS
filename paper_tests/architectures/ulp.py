"""The solve's stability under last-bit perturbation of its start: every design
variable moved by `k` ulp (`x -> x * (1 + k * eps)`, `eps` double precision's), the
whole solve rerun, and what moved -- the optimiser's iterations, its verdict, its
objective. A solve whose iteration count or answer swings on the last bit of its
start is telling you how much of its path is noise, and a finite-difference gradient
at `epsfcn = 1e-3` is where that noise enters: the quotient subtracts two values that
agree to three digits.

The port under each of `bench.OPTIMISERS` (`vmcon`, `vmcon-fd`, `slsqp`, `slsqp-fd`)
and each arm, and with `--native`, PROCESS itself on the same file with the same
perturbation of the same variables (`data.<module>.<name>`, the places
`load_iteration_variables` reads its start from) -- at `bench.TOLERANCE`, as
`native.py` runs it. `k = 0` is the unperturbed solve, so the row it makes should be
the table's.

    $PY paper_tests/architectures/ulp.py --configurations stellarator_helias
    $PY paper_tests/architectures/ulp.py --configurations stellarator_helias --native --optimisers

Writes `out/ulp/<configuration>.csv`, one row per (optimiser, arm, k). The port's
rows are ~1.5 s each warm; a PROCESS row is a full PROCESS run (minutes).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

from process.core.solver.iteration_variables import ITERATION_VARIABLES  # noqa: E402

EPS = float(np.finfo(float).eps)
ULPS = (0, 1, -1, 2, -2, 4, -4, 16, -16)


def perturbed_cold(reference, k: int):
    """`reference.cold` with every design variable at `x * (1 + k * eps)`."""
    cold = copy.deepcopy(reference.cold)
    for i in reference.ixc:
        variable = ITERATION_VARIABLES[int(i)]
        area, name = variable.module, variable.target_name or variable.name
        value = cold.values[(area, name)]
        if variable.array_index is not None:
            value = np.array(value, dtype=float)
            value[variable.array_index] *= 1.0 + k * EPS
        else:
            value = float(value) * (1.0 + k * EPS)
        cold.values[(area, name)] = value
        cold.areas[area].values[name] = value
    return cold


def port_rows(name: str, optimisers, arms, ulps) -> list[dict]:
    rows = []
    for optimiser in optimisers:
        args = type("Args", (), {"scheme": "minimal", "optimiser": optimiser})()
        live = bench.open_session(name, args)
        for arm in arms:
            if arm not in live.arms:
                continue
            live.solve(arm)                                    # compile, off the clock
            for k in ulps:
                result, seconds = bench.timed(live.solve, arm, perturbed_cold(live.reference, k))
                rows.append({
                    "configuration": name, "optimiser": optimiser, "arm": arm, "k": k,
                    "status": result["status"], "iterations": result["iterations"],
                    "objf": result["objf"], "max_eq": result["max_eq"], "min_ie": result["min_ie"],
                    "solve_s": seconds,
                })
                print(rows[-1])
    return rows


def native_rows(name: str, ulps) -> list[dict]:
    """PROCESS itself, its start perturbed at the same places by the same factor."""
    import native  # noqa: PLC0415
    from process.core.solver import constraints  # noqa: PLC0415
    from process.core.solver.objectives import objective_function  # noqa: PLC0415
    from process.main import SingleRun  # noqa: PLC0415

    rows = []
    for k in ulps:
        run = SingleRun(str(native.scratch_copy(name)), "vmcon")
        data = run.data
        n = int(data.numerics.n_iteration_variables)
        for i in data.numerics.ixc[:n]:
            variable = ITERATION_VARIABLES[int(i)]
            module = getattr(data, variable.module)
            field = variable.target_name or variable.name
            if variable.array_index is not None:
                getattr(module, field)[variable.array_index] *= 1.0 + k * EPS
            else:
                setattr(module, field, float(getattr(module, field)) * (1.0 + k * EPS))
        data.numerics.epsvmc = bench.TOLERANCE
        _, seconds = bench.timed(run.run)
        m = int(data.numerics.n_equality_constraints + data.numerics.n_inequality_constraints)
        meq = int(data.numerics.n_equality_constraints)
        conf = np.asarray(constraints.constraint_eqns(m, -1, data)[0], dtype=float)
        converged = float(data.globals.convergence_parameter) <= bench.TOLERANCE
        rows.append({
            "configuration": name, "optimiser": "PROCESS", "arm": "PROCESS", "k": k,
            "status": "converged" if converged else "stopped",
            "iterations": int(data.numerics.n_solver_iterations),
            "objf": float(objective_function(data.numerics.i_figure_merit, data)),
            "max_eq": float(np.max(np.abs(conf[:meq]))) if meq else 0.0,
            "min_ie": float(np.min(conf[meq:])) if m > meq else float("nan"),
            "solve_s": seconds,
        })
        print(rows[-1])
    return rows


def main():
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configurations", nargs="*", default=["stellarator_helias"], metavar="NAME")
    p.add_argument("--optimisers", nargs="*", default=["vmcon", "vmcon-fd", "slsqp", "slsqp-fd"],
                   choices=sorted(bench.OPTIMISERS), metavar="OPT")
    p.add_argument("--arms", nargs="*", default=["MDF", "IDF", "SAND"], metavar="ARM")
    p.add_argument("--ulps", nargs="*", type=int, default=list(ULPS), metavar="K")
    p.add_argument("--native", action="store_true", help="also PROCESS itself, one full run per k")
    args = p.parse_args()
    for name in args.configurations:
        rows = port_rows(name, args.optimisers, args.arms, args.ulps)
        if args.native:
            rows += native_rows(name, args.ulps)
        bench.write("ulp.py", rows, "ulp")


if __name__ == "__main__":
    main()
