"""Table 2 -- the cost of one optimiser iteration, on the CPU, warm: one evaluation of
the problem the optimiser holds (its objective and constraints at a design) and one
Jacobian of it, each compiled once and then timed as the median of `--repeats` calls.

For MDF that evaluation *is* the MDA converged inside the optimiser's iteration and the
Jacobian goes through the converged solve (implicit differentiation); for IDF and SAND
it is one pass of the models, the coupling copies being the optimiser's own. The `MDA`
row is the analysis alone, run once from the file's design. The compile time of each
program is reported beside it, since a cold run pays it once; `block_nodes` is how much
of the graph the unknowns reach -- what one iteration re-runs -- the rest being context
computed once.

    $PY paper_tests/architectures/iteration.py [--scheme minimal] [--repeats 5]
"""

from __future__ import annotations

import sys
from pathlib import Path

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
        "jacobian_ms": float("nan"),
        "compile_s": compile_seconds,
        "unknowns": 0,
        "block_nodes": len(driven.nodes),
    }


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
    context = PathMap({
        v: (stage[v] if v in stage else seeded.get(v, jnp.asarray(ground_truth(cold, v))))
        for v in drive.context
    })
    x, unravel = ravel_pytree(tuple(jnp.asarray(seeded[u]) for u in drive.unknowns))
    values, jacobian, _ = bind(drive.condition_map(context), unravel)
    _, compile_values = bench.timed(values, x)
    _, compile_jacobian = bench.timed(jacobian, x)
    return {
        "evaluate_ms": 1e3 * bench.median_seconds(lambda: values(x), args.repeats),
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
