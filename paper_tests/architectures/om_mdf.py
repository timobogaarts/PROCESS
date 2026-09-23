"""Table 6 -- the same MDF under OpenMDAO, **on the port's own partition**: the
optimiser block the port's `SlsqpDriver` answers (`Session.assemble("MDF").drive`),
converted by `to_openmdao` -- one `ExplicitComponent` per model (its `compute` the
port's jax body, jitted; its partials by `jax.jacfwd`, sparse and coloured), one
`om.Group` per coupled block the port's schedule drives (a fixed point under
`NonlinearBlockGS`, a root find under `NewtonSolver`, `DirectSolver` in each, at the
port driver's tolerances), everything else feed-forward in the schedule's order
(`NonlinearRunOnce`), the block's context set once as inputs, SLSQP outside at the
port's design scaling and tolerance.

What it measures is the cost of *a call per component*: the models' arithmetic, the
blocks and the problem are the port's, so the difference between this table and
`iteration.py` is the framework's execution -- data transfers, vector bookkeeping and
one Python call per component per sweep -- against one program for the whole block.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu $PY paper_tests/architectures/om_mdf.py --configurations stellarator_helias

Per machine: the components and solver groups, the sweeps one MDA takes from the port's
starts (`sweeps`, per block in `block_sweeps`) and warm (`warm_sweeps`), one converged
MDA warm (`model_ms`, `run_model` at the same design) and restarted from the port's
starts (`restart_model_ms`, what the port's evaluation does), one total Jacobian warm
(`totals_ms`), and the full SLSQP run (`iterations` are SLSQP's major iterations, as the
port's `SlsqpDriver` counts them; `evaluations` OpenMDAO's model runs).

`--once`: every solver `NonlinearRunOnce` -- one pass over the components, warm, into
`out/openmdao_once/`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from to_openmdao import name, to_openmdao  # noqa: E402

from functional_process.cottax.architectures.evaluate import (  # noqa: E402
    ground_truth,
    mda_env,
    seed_block,
)
from functional_process.cottax.architectures.mda import seed_starts  # noqa: E402

__all__ = ["name", "problem_of", "values_of"]


def values_of(live, build) -> dict:
    """A value at every input of the MDF block (`drive.inputs`), as the port's own
    solve is seeded (`iteration.optimiser_row`): the MDA's at the file's design, else
    the seeded one, else the file's, else zero."""
    drive, schedule = build.drive, build.solve_schedule
    cold = live.reference.cold
    stage = mda_env(live.reference, graph=live.machine_graph, data=cold, scheme=live.scheme)[1]
    seeded, _ = seed_block(schedule, drive, cold, stage, design=set())
    seeded.update(seed_starts(schedule, stage))

    def value(v):
        if v in stage:
            return stage[v]
        if v in seeded:
            return seeded[v]
        try:
            return jnp.asarray(ground_truth(cold, v))
        except (AttributeError, KeyError):
            return jnp.asarray(0.0)

    return {v: value(v) for v in drive.inputs}


def problem_of(live, once=False):
    """`(conversion, build)`: the port's MDF block as an OpenMDAO problem."""
    build = live.assemble("MDF")
    return to_openmdao(build.drive, values_of(live, build), once=once), build


def open_session(name_, args):
    """The session with the port's SLSQP on its optimiser, whatever `--optimiser`
    says: the converted problem takes its scaling, tolerance and budget from it."""
    args.optimiser = "slsqp"
    return bench.open_session(name_, args)


def sweeps(conv) -> tuple[int, str]:
    """The last run's sweeps: their total, and per block `kind:count`."""
    per = conv.sweeps()
    kinds = {path: kind for path, kind, _ in conv.groups}
    return sum(per.values()), "|".join(f"{kinds[p].split()[0]}:{n}" for p, n in per.items())


def once(args):
    """`--once`: one pass over the components, warm -- `out/openmdao_once/`."""
    rows = []
    for name_ in args.configurations:
        live = open_session(name_, args)
        if live.root_find:
            continue
        (conv, _build), setup_s = bench.timed(problem_of, live, once=True)
        _, cold_s = bench.timed(conv.problem.run_model)
        pass_s = bench.median_seconds(conv.problem.run_model, args.repeats)
        rows.append({
            "configuration": name_, "arm": "one pass (OpenMDAO)",
            "components": conv.components, "groups": len(conv.groups),
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
        live = open_session(name_, args)
        if live.root_find:
            continue
        (conv, build), setup_s = bench.timed(problem_of, live)
        prob = conv.problem
        conv.restart()
        _, cold_s = bench.timed(prob.run_model)
        total, by_block = sweeps(conv)

        def restarted():
            conv.restart()
            prob.run_model()

        restart_s = bench.median_seconds(restarted, args.repeats)
        prob.run_model()
        model_s = bench.median_seconds(prob.run_model, args.repeats)
        warm_sweeps, _ = sweeps(conv)                      # the timed, warm MDA's
        totals_s = bench.median_seconds(lambda: prob.compute_totals(), args.repeats)
        conv.restart()
        began = time.perf_counter()
        prob.run_driver()
        driver_s = time.perf_counter() - began
        result = prob.driver._scipy_optimize_result
        scalar = lambda n: float(np.asarray(prob.get_val(n)).reshape(-1)[0])  # noqa: E731
        eqs = [abs(scalar(c)) for c in conv.equalities]
        ies = [-scalar(c) for c in conv.inequalities]
        rows.append({
            "configuration": name_, "arm": "MDF (OpenMDAO)",
            "components": conv.components, "groups": len(conv.groups), "design": len(conv.design),
            "sweeps": total, "block_sweeps": by_block, "warm_sweeps": warm_sweeps,
            "setup_s": setup_s, "cold_model_s": cold_s,
            "model_ms": 1e3 * model_s, "restart_model_ms": 1e3 * restart_s, "totals_ms": 1e3 * totals_s,
            "iterations": int(result.nit), "evaluations": prob.driver.iter_count,
            "status": "converged" if result.success else "failed",
            "driver_s": driver_s, "objf": scalar(conv.objective),
            "max_eq": max(eqs) if eqs else 0.0, "min_ie": min(ies) if ies else float("nan"),
            "zeroed_partials": len(conv.nonfinite), "dense_components": len(conv.dense),
            "zeroed_sinks": len(conv.held),
        })
        print(rows[-1])
    if rows:
        bench.write("om_mdf.py", rows, "openmdao_mdf")


if __name__ == "__main__":
    main()
