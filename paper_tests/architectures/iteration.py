"""Table 2 -- the cost of one optimiser iteration, on the CPU, warm: one evaluation of
the problem the optimiser holds (its objective and constraints at a design) and one
Jacobian of it, each compiled once and then timed as the median of `--repeats` calls.

For MDF that evaluation *is* the MDA converged inside the optimiser's iteration and the
Jacobian goes through the converged solve (implicit differentiation); for IDF and SAND
it is one pass of the models, the coupling copies being the optimiser's own. The `MDA`
row is the analysis alone, run once from the file's design. `serial_us` is one evaluation with no dispatch
in it: `SERIAL_K` evaluations chained inside one program, each design fed by the last
result, per evaluation. The compile time of each
program is reported beside it, since a cold run pays it once; `block_nodes` is how much
of the graph the unknowns reach -- what one iteration re-runs -- the rest being context
computed once.

    $PY paper_tests/architectures/iteration.py [--scheme minimal] [--repeats 5]
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from cottax.pytree.names import PathMap  # noqa: E402

from cottax.execution.schedule import Schedule  # noqa: E402
from functional_process.cottax.architectures.evaluate import (  # noqa: E402
    ground_truth,
    mda_env,
    mda_schedule,
    seed_block,
    seed_env,
)
from functional_process.cottax.architectures.host_cache import bind  # noqa: E402
from functional_process.cottax.architectures.mda import seed_starts  # noqa: E402


def mda_row(live, args) -> dict:
    """The analysis alone: one run of the MDA schedule from the file's own values."""
    driven, runnable, schedule, run = mda_schedule(live.machine_graph, bench.SCHEMES[args.scheme])
    env = seed_env(live.reference.data, schedule, runnable, None)
    values = PathMap(env)
    _, compile_seconds = bench.timed(run, values)          # traced and compiled here
    return {
        "evaluate_ms": 1e3 * bench.median_seconds(lambda: run(values), args.repeats),
        "serial_us": float("nan"),
        "jacobian_ms": float("nan"),
        "compile_s": compile_seconds,
        "unknowns": 0,
        "block_nodes": len(driven.nodes),
    }


def context_value(var, stage, seeded, cold):
    """As `session.solve_block` closes the drive over its context: the MDA's value,
    else the seeded one, else the file's, else zero (PROCESS's own default for a
    quantity nothing has written)."""
    if var in stage:
        return stage[var]
    if var in seeded:
        return seeded[var]
    try:
        return jnp.asarray(ground_truth(cold, var))
    except (AttributeError, KeyError):
        return jnp.asarray(0.0)


SERIAL_K = 100


def in_program_serial(cm, unravel, x):
    """`SERIAL_K` evaluations one after another **inside one program**, each design
    depending on the last evaluation's every condition, so nothing can be batched or
    hoisted: the cost of one evaluation with no dispatch in it -- what an optimiser
    embedded in the same program pays per iteration."""
    def f(flat):
        return ravel_pytree(cm(*unravel(flat)))[0]

    def step(xk, _):
        y = f(xk)
        return xk * (1 + 1e-12 * jnp.tanh(y).sum() / y.size), y

    run = jax.jit(lambda x0: jax.lax.scan(step, x0, None, length=SERIAL_K)[1])
    run(x)
    return run


def optimiser_row(live, arm, build, args) -> dict:
    """One evaluation and one Jacobian of the arm's optimiser problem at its start."""
    if live.root_find:
        drive, schedule = build.in_graph.drive, Schedule(build.in_graph.blocking)
    else:
        drive, schedule = build.drive, build.solve_schedule
    cold = live.reference.cold
    stage = mda_env(live.reference, graph=live.machine_graph, data=cold, scheme=bench.SCHEMES[args.scheme])[1]
    seeded, _ = seed_block(schedule, drive, cold, stage, design=set())
    seeded.update(seed_starts(schedule, stage))
    context = PathMap({v: context_value(v, stage, seeded, cold) for v in drive.context})
    x, unravel = ravel_pytree(tuple(jnp.asarray(seeded[u]) for u in drive.unknowns))
    cm = drive.condition_map(context)
    values, jacobian, _ = bind(cm, unravel)
    _, compile_values = bench.timed(values, x)
    _, compile_jacobian = bench.timed(jacobian, x)
    serial = in_program_serial(cm, unravel, x)
    return {
        "evaluate_ms": 1e3 * bench.median_seconds(lambda: values(x), args.repeats),
        "serial_us": 1e6 * bench.median_seconds(lambda: serial(x), args.repeats) / SERIAL_K,
        "jacobian_ms": 1e3 * bench.median_seconds(lambda: jacobian(x), args.repeats),
        "compile_s": compile_values + compile_jacobian,
        "unknowns": int(np.size(x)),
        "block_nodes": len(drive.nodes),       # what the unknowns reach; the rest is context
    }


def main():
    args = bench.arguments(__doc__, optimiser=False)
    rows = []
    for name in args.configurations:
        live = bench.open_session(name, args)
        for arm in live.arms:
            build = live.assemble(arm)
            row = mda_row(live, args) if arm == "MDA" else optimiser_row(live, arm, build, args)
            rows.append({"configuration": name, "arm": arm, **row})
            print(rows[-1])
    bench.write("iteration.py", rows, f"iteration_{args.scheme}")


if __name__ == "__main__":
    main()
