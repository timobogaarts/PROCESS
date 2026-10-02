"""Table 6 -- the same MDF under OpenMDAO, through cottax's own export
(`study.openmdao`, `cottax.export.openmdao`) of the optimiser's cycle, what each
iteration re-runs, with what it reads from outside run once and held (no partial is
taken in it): one jitted component per model, its partials by `jax.jacfwd` where the
jaxpr says they are nonzero; a group per nested solve
(`NonlinearBlockGS` for a fixed point, `NewtonSolver` for a root find, `DirectSolver` in
each); SLSQP as the driver, in units of the design's starts.

What it measures is the cost of *a call per component*: the models' arithmetic, the
blocks and the problem are the port's, so the difference between this table and
`iteration.py` is the framework's execution -- data transfers, vector bookkeeping and
one Python call per component per sweep -- against one program for the whole block.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu $PY paper_tests/architectures/om_mdf.py --configurations stellarator_helias

Per machine: the components and solver groups, the sweeps one MDA takes from the
starts (`sweeps`, per group `kind:count` in `block_sweeps`) and warm (`warm_sweeps`), one
converged MDA warm (`model_ms`, `run_model` at the same design with the groups' start
guess switched off) and restarted from the starts (`restart_model_ms`, the export's own
`run_model`: each group sets its unknowns to their starts as its solver begins), one
total Jacobian warm (`totals_ms`), and the full SLSQP run (`iterations` SLSQP's major
iterations, `evaluations` OpenMDAO's model runs).

Dropped: `zeroed_partials`, `dense_components`, `zeroed_sinks`, which recorded what the
port's own converter patched; the export patches nothing.

`--once`: every solver `NonlinearRunOnce` -- one pass over the components, warm, into
`out/openmdao_once/`.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import openmdao.api as om
from cottax.relational.consumers.openmdao_solvers import ProblemComponent
from openmdao.core.component import Component

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

from functional_process import configurations  # noqa: E402
from functional_process.cottax import study  # noqa: E402

KINDS = {om.NonlinearBlockGS: "fixed", om.NewtonSolver: "root"}
"""The solver of a group the export iterates, by the kind of problem it answers."""


def problem_of(configuration, args, once=False) -> om.Problem:
    """The MDF of `configuration` as the export's `om.Problem`, set up; `once` swaps every
    solver for `NonlinearRunOnce` before its first run."""
    problem = study.openmdao(configuration, "MDF", rule=bench.SCHEMES[args.scheme]).problem()
    if once:
        for group in problem.model.system_iter(recurse=True, include_self=True, typ=om.Group):
            group.nonlinear_solver = om.NonlinearRunOnce()
            group.linear_solver = om.LinearRunOnce()
    return problem


def components(problem) -> int:
    return sum(1 for _ in problem.model.system_iter(recurse=True, typ=Component))


def groups(problem) -> list:
    """The export's solver groups, one per nested solve, outermost first."""
    return list(problem.model.system_iter(recurse=True, typ=om.Group))


def sweeps(problem) -> tuple[int, str]:
    """The last run's sweeps: their total, and per group `kind:count`."""
    per = [(KINDS[type(g.nonlinear_solver)], g.nonlinear_solver._iter_count)
           for g in groups(problem) if type(g.nonlinear_solver) in KINDS]
    return sum(n for _, n in per), "|".join(f"{k}:{n}" for k, n in per)


@contextmanager
def warm(problem):
    """Every group left where its last run converged: the start guess switched off."""
    solves = list(problem.model.system_iter(recurse=True, typ=ProblemComponent))
    for c in solves:
        c.guess_nonlinear = lambda *a: None
    try:
        yield
    finally:
        for c in solves:
            del c.guess_nonlinear


def once(args):
    """`--once`: one pass over the components, warm -- `out/openmdao_once/`."""
    rows = []
    for name_ in args.configurations:
        configuration = configurations.load(name_)
        if bench.is_root_find(configuration):
            continue
        prob, setup_s = bench.timed(problem_of, configuration, args, once=True)
        _, cold_s = bench.timed(prob.run_model)
        pass_s = bench.median_seconds(prob.run_model, args.repeats)
        rows.append({
            "configuration": name_, "arm": "one pass (OpenMDAO)",
            "components": components(prob), "groups": len(groups(prob)),
            "setup_s": setup_s, "cold_pass_s": cold_s, "pass_ms": 1e3 * pass_s,
        })
        print(rows[-1])
    if rows:
        bench.write("om_mdf.py", rows, "openmdao_once")


def main():
    args = bench.arguments(__doc__, optimiser=False)
    if args.once:
        return once(args)
    rows = []
    for name_ in args.configurations:
        configuration = configurations.load(name_)
        if bench.is_root_find(configuration):
            continue
        prob, setup_s = bench.timed(problem_of, configuration, args)
        _, cold_s = bench.timed(prob.run_model)
        total, by_group = sweeps(prob)
        restart_s = bench.median_seconds(prob.run_model, args.repeats)
        with warm(prob):
            model_s = bench.median_seconds(prob.run_model, args.repeats)
            warm_sweeps, _ = sweeps(prob)
            totals_s = bench.median_seconds(lambda: prob.compute_totals(), args.repeats)
        began = time.perf_counter()
        prob.run_driver()
        driver_s = time.perf_counter() - began
        result, driver = prob.driver._scipy_optimize_result, prob.driver
        (objf,) = (float(np.ravel(v)[0])
                   for v in driver.get_objective_values(driver_scaling=False).values())
        cons = driver.get_constraint_values(driver_scaling=False)
        equality = {k: driver._cons[k]["equals"] is not None for k in cons}
        eqs = [np.max(np.abs(v)) for k, v in cons.items() if equality[k] and np.size(v)]
        ies = [np.min(-np.asarray(v)) for k, v in cons.items() if not equality[k] and np.size(v)]
        rows.append({
            "configuration": name_, "arm": "MDF (OpenMDAO)",
            "components": components(prob), "groups": len(groups(prob)),
            "design": len(driver._designvars),
            "sweeps": total, "block_sweeps": by_group, "warm_sweeps": warm_sweeps,
            "setup_s": setup_s, "cold_model_s": cold_s,
            "model_ms": 1e3 * model_s, "restart_model_ms": 1e3 * restart_s, "totals_ms": 1e3 * totals_s,
            "iterations": int(result.nit), "evaluations": driver.iter_count,
            "status": "converged" if result.success else "failed",
            "driver_s": driver_s, "objf": objf,
            "max_eq": float(max(eqs)) if eqs else 0.0,
            "min_ie": float(min(ies)) if ies else float("nan"),
        })
        print(rows[-1])
    if rows:
        bench.write("om_mdf.py", rows, "openmdao_mdf")


if __name__ == "__main__":
    main()
