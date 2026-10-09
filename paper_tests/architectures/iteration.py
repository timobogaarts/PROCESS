"""Table 2 -- the cost of one optimiser iteration, on the CPU, warm: one evaluation of
the problem the optimiser holds (its objective and constraints at a design) and one
Jacobian of it, each compiled once and then timed as the median of `--repeats` calls.

For MDF that evaluation *is* the MDA converged inside the optimiser's iteration and the
Jacobian goes through the converged solve (implicit differentiation); for IDF and SAND
it is one pass of the models, the coupling copies being the optimiser's own. The `MDA`
row is the analysis alone (`study.mda_graph`), run once from the file's design. `serial_us` is one evaluation with no dispatch
in it (`serial_jacobian_us` one Jacobian): `SERIAL_K` evaluations chained inside one program, each design fed by the last
result, per evaluation. `eager_walk_ms` and `eager_ms` are the same evaluation with
nothing compiled across nodes (`eager.eager_graph`): every body compiled on its own and
cottax walking the graph in Python, the nested solves compiled (`eager_walk_ms`) or
stepped from Python as well (`eager_ms`, the same optimistix solver and tolerances, its
steps in `eager_steps`); each is checked against the compiled evaluation. The compile time of each
program is reported beside it, since a cold run pays it once; `block_nodes` is how much
of the graph the unknowns reach -- what one iteration re-runs -- the rest being context
computed once.

`--epsfcn` adds the other way of getting the same Jacobian: PROCESS's own central
difference, `x * (1 +/- epsfcn)` per coordinate, `2n` value-only evaluations of the
*same* compiled block (`fd_jacobian_ms`), and how far its entries land from `jacfwd`'s
(`fd_rel_median`, `fd_rel_max`, over the `fd_entries` that are not negligible
beside the largest in their own row).
It writes `out/iteration_<scheme>_fd/` so the autodiff table is not overwritten.

The problem is `study.assemble`'s arm, posed as its driver is handed it
(`study.posed`) and read as gaps (`bench.block`), at the arm's own start.

    $PY paper_tests/architectures/iteration.py [--scheme minimal] [--repeats 5]
    $PY paper_tests/architectures/iteration.py --epsfcn --configurations stellarator_helias
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import eager  # noqa: E402

from functional_process import configurations  # noqa: E402
from functional_process.cottax import study  # noqa: E402
from functional_process.cottax.architectures.drivers import finite_difference_jacobian  # noqa: E402
from functional_process.cottax.architectures.host_cache import bind  # noqa: E402


NEGLIGIBLE = 1e-6
"""Below this fraction of its row's largest entry, a Jacobian entry cannot move that
condition, and `fd_columns` does not count it as a disagreement."""


def finite_difference(values, x, epsfcn):
    """`drivers.finite_difference_jacobian` on the bound block: `2n` value-only calls,
    PROCESS's derivative on the port's block -- the one thing `VmconDriver(epsfcn=...)`
    does differently from `jax.jacfwd`, taken out on its own to be timed and compared.
    """
    return finite_difference_jacobian(
        lambda flat: np.asarray(values(jnp.asarray(flat))), np.asarray(x, dtype=float), epsfcn
    )


def fd_columns(values, jacobian, x, args) -> dict:
    """The finite-difference Jacobian's cost and its distance from `jacfwd`'s, or the
    same keys as `nan` where there is nothing to differentiate (the `MDA` row).
    """
    if not getattr(args, "epsfcn", None):
        return {}
    if x is None:
        return dict.fromkeys(
            ("fd_jacobian_ms", "fd_calls", "fd_rel_median", "fd_rel_max", "fd_entries"),
            float("nan"),
        )
    exact = np.asarray(jacobian(x), dtype=float)
    approx = finite_difference(values, x, args.epsfcn)
    # **Per row, against that row's own largest entry.** A plain `exact != 0.0` mask
    # reports a relative error of 3e+03 on `d(c24)/d(nd_plasma_electrons_vol_avg)`,
    # where `jacfwd` says `-2.6e-37` and the difference says `-8.3e-34`: both are zero
    # as far as any SQP step is concerned, and the statistic is then about rounding,
    # not about the derivative. An entry below `NEGLIGIBLE` of the largest in its own
    # row cannot move that condition, so it is excluded rather than allowed to set
    # `fd_rel_max`.
    row_scale = np.max(np.abs(exact), axis=1, keepdims=True)
    live = np.abs(exact) > NEGLIGIBLE * row_scale
    relative = np.abs(approx[live] - exact[live]) / np.abs(exact[live])
    return {
        "fd_jacobian_ms": 1e3 * bench.median_seconds(
            lambda: finite_difference(values, x, args.epsfcn), args.repeats
        ),
        "fd_calls": 2 * int(np.size(x)),
        "fd_rel_median": float(np.median(relative)) if relative.size else float("nan"),
        "fd_rel_max": float(np.max(relative)) if relative.size else float("nan"),
        "fd_entries": int(live.sum()),
    }


def mda_row(assembled, args) -> dict:
    """The analysis alone: one run of the MDA from the file's own values."""
    graph, inputs = assembled.graph, assembled.inputs
    jax.clear_caches()                          # `assemble` ran it once: compile it again
    _, compile_seconds = bench.timed(study.solve, graph, inputs)
    return {
        "evaluate_ms": 1e3 * bench.median_seconds(lambda: study.solve(graph, inputs), args.repeats),
        "serial_us": float("nan"),
        "jacobian_ms": float("nan"),
        "serial_jacobian_us": float("nan"),
        "compile_s": compile_seconds,
        "unknowns": 0,
        "block_nodes": len(graph.nodes),
        **fd_columns(None, None, None, args),
        **eager_columns(None, args),
    }


SERIAL_K = 100


def eager_columns(assembled, args) -> dict:
    """The evaluation at the arm's start with nothing compiled across nodes: the walk
    eager with the nested solves compiled, then the solves stepped from Python too. Each
    is checked to give the compiled evaluation's every condition."""
    if assembled is None:
        return {"eager_walk_ms": float("nan"), "eager_ms": float("nan"), "eager_steps": "-"}
    gaps, x, unravel = bench.block(assembled)
    reference = np.concatenate([np.ravel(v) for v in jax.tree.leaves(
        jax.jit(lambda x: gaps(*unravel(x)))(x))])
    out = {}
    for column, solvers in (("eager_walk_ms", False), ("eager_ms", True)):
        graph = eager.eager_graph(assembled.graph, solvers=solvers)
        e_gaps, _, e_unravel = bench.block(dataclasses.replace(assembled, graph=graph))
        evaluate = lambda: e_gaps(*e_unravel(x))  # noqa: E731
        got = np.concatenate([np.ravel(v) for v in jax.tree.leaves(evaluate())])
        scale = float(np.max(np.abs(reference)))
        if not np.allclose(got, reference, rtol=1e-9, atol=1e-9 * scale):
            raise AssertionError(f"{column}: the eager evaluation is not the compiled one")
        eager.STEPS.clear()
        evaluate()
        steps = "|".join(f"{kind}:{n}" for kind, n in eager.STEPS) or "-"
        out[column] = 1e3 * bench.median_seconds(evaluate, args.repeats)
    return {**out, "eager_steps": steps}


def in_program_serial(gaps, unravel, x, jacobian=False):
    """`SERIAL_K` evaluations one after another **inside one program**, each design
    depending on the last evaluation's every condition, so nothing can be batched or
    hoisted: the cost of one evaluation with no dispatch in it -- what an optimiser
    embedded in the same program pays per iteration. With `jacobian`, the same chain
    over `jax.jacfwd` of the evaluation: one Jacobian with no dispatch in it."""
    def f(flat):
        return ravel_pytree(gaps(*unravel(flat)))[0]

    g = jax.jacfwd(f) if jacobian else f

    def step(xk, _):
        y = g(xk)
        return xk * (1 + 1e-12 * jnp.tanh(y).sum() / y.size), y

    run = jax.jit(lambda x0: jax.lax.scan(step, x0, None, length=SERIAL_K)[1])
    run(x)
    return run


def optimiser_row(assembled, args) -> dict:
    """One evaluation and one Jacobian of the arm's optimiser problem at its start."""
    gaps, x, unravel = bench.block(assembled)
    values, jacobian, _ = bind(gaps, unravel)
    _, compile_values = bench.timed(values, x)
    _, compile_jacobian = bench.timed(jacobian, x)
    serial = in_program_serial(gaps, unravel, x)
    serial_jacobian = in_program_serial(gaps, unravel, x, jacobian=True)
    return {
        "evaluate_ms": 1e3 * bench.median_seconds(lambda: values(x), args.repeats),
        "serial_us": 1e6 * bench.median_seconds(lambda: serial(x), args.repeats) / SERIAL_K,
        "jacobian_ms": 1e3 * bench.median_seconds(lambda: jacobian(x), args.repeats),
        "serial_jacobian_us": 1e6 * bench.median_seconds(lambda: serial_jacobian(x), args.repeats) / SERIAL_K,
        "compile_s": compile_values + compile_jacobian,
        "unknowns": int(np.size(x)),
        "block_nodes": bench.block_nodes(assembled.graph),   # what the unknowns reach
        **fd_columns(values, jacobian, x, args),
        **eager_columns(assembled, args),
    }


def main():
    args = bench.arguments(__doc__, optimiser=False, epsfcn=True)
    rows = []
    for name in args.configurations:
        configuration = configurations.load(name)
        cold = study.cold_state(configuration)
        for arm in study.arms(configuration):
            assembled = study.assemble(configuration, arm, cold, **bench.how(args))
            row = mda_row(assembled, args) if arm == "MDA" else optimiser_row(assembled, args)
            rows.append({"configuration": name, "arm": arm, **row})
            print(rows[-1])
    folder = f"iteration_{args.scheme}" + ("_fd" if args.epsfcn else "")
    bench.write("iteration.py", rows, folder)


if __name__ == "__main__":
    main()
