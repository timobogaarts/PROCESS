"""Consistency solved where it arises: the two **equality** constraints of
`stellarator_helias` closed by a root find over one design variable each, inside the
MDA, and the optimiser left with the objective, the twelve inequalities and the six
design variables that remain.

The architecture is the port's `architectures.closing` (`close`, `seed`,
`nested_blocking`, `describe`, `cycle_of`, `place_for`, the `SafeguardedNewtonDriver`);
this script is the three measurements over it:

`--table` -- every equality against every design variable and every scalar boundary
input it is structurally sensitive to: the cycle `RootFind((cond,), (var,))` creates
on the raw and on the cut graph (`closing.cycle_of`), and the sensitivity
`d cond / d var` through the converged MDA (`jax.jacfwd` of `mdf.condition_map`), so
that a small cycle with no leverage (`t_tf_superconductor_quench` -> `c16`: six
nodes, scaled sensitivity 8e-5) is not mistaken for a candidate.

`--solve` -- the chosen pairings (`kinds.HISTORIC_PAIRING` by default: the power
balance `c2` by `hfact`, the smallest cycle, and the net electric power `c16` by the
thermal alpha fraction) closed with `closing.close`, then VMCON and SLSQP over the six
remaining design variables from the same cold start `mdf.solve` uses, cold then warm,
the Newton steps at every outer iterate read back, and the plain MDF (same cut, same
optimisers) and PROCESS's own answer (`common.process_reference`) beside them.

`--batch` -- one MDA evaluation, plain and with the root finds inside, single and
`jax.vmap`ped over N design points on whichever backend jax is on. Four shapes:
`plain`; `nested` (the cut fixed points nested inside the root finds, exact Newton);
`closed` (flattened into one square problem per root find, Broyden -- `close`'s
default); `predicted` (`closed`, started from the first-order prediction of the roots
by one `jacfwd` at the centre).

    $PY paper_tests/close_conditions.py --table            # ~2 min
    $PY paper_tests/close_conditions.py --solve            # ~5 min
    $PY paper_tests/close_conditions.py --solve --pairing one           # kinds.PAIRINGS
    $PY paper_tests/close_conditions.py --solve --pair16 .physics.nd_plasma_electrons_vol_avg
    $PY paper_tests/close_conditions.py --solve --driver newton   # optimistix's undamped Newton
    $PY paper_tests/close_conditions.py --solve --driver exact    # safeguarded, exact Jacobian per step
    $PY paper_tests/close_conditions.py --solve --nested          # fixed points nested, not combined
    $PY paper_tests/close_conditions.py --batch [--sizes 1,64,1024] [--shapes closed,predicted]
    JAX_PLATFORMS=cuda $G paper_tests/close_conditions.py --batch --sizes 1,4,16,64,256,1024,4096,16384

Outputs: `out/close_conditions.csv` (the pairing table), `out/close_conditions.tex`
(design-variable rows plus the ten smallest non-design cycles per equality),
`out/close_conditions_solve[<suffix>].{csv,tex,json}` and
`out/close_conditions_batch_<backend>[_<shapes>].{csv,tex}`.
"""

from __future__ import annotations

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from common import (  # noqa: E402
    fmt,
    process_reference,
    sci,
    write_csv,
    write_json,
    write_tex,
)
from cottax.names import PathMap  # noqa: E402
from cottax.problem import Converged, Steps  # noqa: E402

from functional_process.configurations import kinds  # noqa: E402
from functional_process.cottax.architectures import (  # noqa: E402
    closing,
    mdf,
    sand,
    session,
)
from functional_process.cottax.architectures.drivers import SlsqpDriver  # noqa: E402
from functional_process.cottax.architectures.evaluate import (  # noqa: E402
    run_schedule,
    without_excluded,
)
from functional_process.cottax.architectures.recipes import recipe  # noqa: E402
from functional_process.cottax.input.indat import graph_for  # noqa: E402

NAME = "stellarator_helias"
CUT = "gauss_seidel_minimal"

PAIRINGS = dict(kinds.HISTORIC_PAIRING)
"""Which design variable closes which equality -- the choice `--table` justifies:
`hfact` is the smallest cycle for the power balance (3 nodes, scaled sensitivity 1.0,
and the power balance is what `hfact` exists to close in PROCESS's own usage). For the
net electric power the smallest cycle among the design variables is
`t_tf_superconductor_quench` (6 nodes) but its scaled sensitivity is 8e-5, so it is not
a candidate; the next is `f_nd_alpha_thermal_electron` (26 nodes, 0.42). The density
(27 nodes, 1.7) is the alternative `--pair16` solves. `kinds.PAIRINGS` (`--pairing`)
are the OUU study's choices, where `hfact` is a belief and the density closes `c2`."""

DRIVERS = {
    "safeguarded": closing.safeguarded,
    "newton": closing.newton,
    "exact": closing.safeguarded_newton,
}


def open_live(cut_name: str = CUT):
    """The configuration under the cut the closed graph is built on."""
    return session.open_session(NAME, cut=recipe(cut_name))


def raw_graph(live):
    return without_excluded(
        live.machine_graph if live.machine_graph is not None else graph_for()
    )


def with_conditions(live, graph):
    """`graph` plus the constraint and objective nodes, and `mdf_graph`'s report."""
    ref = live.reference
    inserted, _conditions, _n, report = mdf.mdf_graph(
        graph, ref.icc, ref.n_equality, ref.i_figure_merit, live.switch_values
    )
    return inserted, report


def design_of(live) -> tuple:
    return tuple(sand.iteration_variable_path(i) for i in live.reference.ixc)


# ---------------------------------------------------------------- the table


def pairing_table(live) -> list[dict]:
    """Every equality against every design variable and every scalar boundary input
    it is structurally sensitive to.
    """
    ref = live.reference
    raw = raw_graph(live)
    cut = live.cut(raw)
    raw_c, report = with_conditions(live, raw)
    cut_c, _ = with_conditions(live, cut)
    design = design_of(live)
    equalities = tuple(report["equalities"])

    # The plain MDF at the primed cold point, for the sensitivities.
    build = session.build_mdf(ref, live.machine_graph, live.switch_values, cut=live.cut)
    problem = build.problem
    env = mdf.seed(problem, ref.cold)
    env, _out = mdf.prime(problem, env)
    inputs = set(problem.eager.inputs)
    condition_index = {c: i for i, c in enumerate(problem.conditions)}

    # Which boundary inputs reach which equality, structurally.
    deps = raw_c.graph
    sensitive: dict = {}
    for cond in equalities:
        ancestors = set(deps.ancestors([deps.owners[cond]]))
        for var in deps.boundary_inputs:
            if any(r in ancestors for r in deps.readers.get(var, ())):
                sensitive.setdefault(var, set()).add(cond)
    candidates = [v for v in deps.boundary_inputs if v in sensitive]
    scalar = [
        v for v in candidates
        if v in inputs and np.size(np.asarray(env[v])) == 1
        and np.issubdtype(np.asarray(env[v]).dtype, np.floating)
    ]
    # One jacfwd over every scalar candidate at once: `MdfConditionMap` with those as
    # its unknowns and everything else (the design included) closed over.
    context = {v: x for v, x in env.items() if v in inputs and v not in set(scalar)}
    roles = mdf.condition_map(problem, env).roles
    cmap = mdf.MdfConditionMap(
        body=problem.traceable.subgraph,
        unknowns=tuple(scalar),
        conditions=problem.conditions,
        roles=roles,
        context=PathMap(context.items()),
        schedule=problem.traceable,
    )
    start = tuple(jnp.asarray(env[v]) for v in scalar)

    def stacked(*x):
        return jnp.stack([jnp.asarray(c).reshape(()) for c in cmap(*x)])

    began = time.perf_counter()
    columns = jax.jit(jax.jacfwd(stacked, argnums=tuple(range(len(scalar)))))(*start)
    jacobian = np.stack([np.asarray(c).reshape(-1) for c in columns], axis=1)
    print(f"jacobian over {len(scalar)} scalar inputs: {time.perf_counter() - began:.1f} s")
    column = {v: j for j, v in enumerate(scalar)}

    rows = []
    for cond in equalities:
        for var in candidates:
            if cond not in sensitive[var]:
                continue
            raw_cycle = closing.cycle_of(raw_c, cond, var)
            cut_cycle = closing.cycle_of(cut_c, cond, var)
            row = {
                "condition": cond.spelling,
                "variable": var.spelling,
                "design": var in design,
                "nodes_raw": len(raw_cycle),
                "nodes_cut": len(cut_cycle),
                "problems_in_cycle": sum(
                    1 for n in cut_cycle if n.spelling.startswith("^problem")
                ),
                "value": None,
                "d_cond_d_var": None,
                "scaled": None,
            }
            if var in column:
                x = float(np.asarray(env[var]))
                d = float(jacobian[condition_index[cond], column[var]])
                row.update(value=x, d_cond_d_var=d, scaled=x * d)
            else:
                row["note"] = "array-valued, integer or not a schedule input; no sensitivity"
            rows.append(row)
    rows.sort(key=lambda r: (r["condition"], r["nodes_raw"], r["variable"]))
    return rows


def render_table(rows: list[dict]) -> None:
    write_csv("close_conditions.py", rows)
    header = ["equality", "variable", "design", "cycle (raw)", "cycle (cut)",
              r"$\partial c/\partial x$", r"$x\,\partial c/\partial x$"]
    body = []
    for cond in sorted({r["condition"] for r in rows}):
        of = [r for r in rows if r["condition"] == cond]
        shown = [r for r in of if r["design"]] + [
            r for r in of
            if not r["design"] and r["scaled"] is not None and abs(r["scaled"]) > 1e-6
        ][:10]
        for r in shown:
            body.append([
                cond.split(".")[-1].replace("c", "icc\\,"),
                r"\texttt{" + r["variable"].replace("_", r"\_") + "}",
                "yes" if r["design"] else "",
                r["nodes_raw"], r["nodes_cut"],
                sci(r["d_cond_d_var"]), sci(r["scaled"]),
            ])
    write_tex(
        "close_conditions.py", header, body, align="llcrrrr",
        caption_note=(
            "cycle = nodes of the SCC that RootFind(cond <- var) creates, the root "
            "find included; sensitivities by jacfwd through the converged MDA "
            f"({CUT} cut) at the primed cold start; the residuals are PROCESS's "
            "normalised ones, so the condition scale is 1. Design rows, then the ten "
            "smallest non-design cycles with a non-zero sensitivity."
        ),
    )


# ---------------------------------------------------------------- the solve


def solve(built: closing.Closed, optimiser=None, label="vmcon") -> dict:
    """VMCON (or `optimiser`) over the kept design, from the cold start; then a warm
    repeat.
    """
    live, problem = built.session, built.problem
    ref = live.reference
    bounds = tuple(b for b in ref.bounds if b[0] in set(built.design))
    began = time.perf_counter()
    env = closing.seed(built, ref.cold)
    env, primed = mdf.prime(problem, env)
    prime_seconds = time.perf_counter() - began
    row = {"optimiser": label, "prime_s": prime_seconds}
    row["primed_root_finds"] = {c.spelling: r for c, r in built.root_find_reports(primed).items()}
    results = []
    for _attempt in ("cold", "warm"):
        trace: list = []
        iterates: list = []
        record = session.recorder(trace)

        def recording(i, result, x, convergence, record=record, iterates=iterates):
            record(i, result, x, convergence)
            iterates.append([float(v) for v in np.asarray(x)])

        began = time.perf_counter()
        x, out, seconds = mdf.solve(
            problem, dict(env), bounds=bounds, callback=recording,
            tolerance=session.MDF_TOLERANCE, max_iter=session.MDF_MAX_ITER,
            **({} if optimiser is None else {"optimiser": optimiser}),
        )
        total = time.perf_counter() - began
        iterations, objf, _max_eq, min_ie = session.trace_tail(trace)
        results.append({
            "iterations": iterations, "objf": objf, "min_ie": min_ie,
            "status": session._status(trace, session.MDF_TOLERANCE, session.MDF_MAX_ITER),
            "driver_status": mdf.verdict(out, mdf.Status),
            "solve_s": seconds, "total_s": total,
            "x": {v.spelling: float(np.asarray(xx)) for v, xx in zip(built.design, x, strict=True)},
            "closed": {c.spelling: float(np.asarray(out[v])) for c, v in built.pairings.items()},
            "equalities": {c.spelling: float(np.asarray(out[c])) for c in built.pairings},
            "root_finds": {c.spelling: r for c, r in built.root_find_reports(out).items()},
            "trace": trace,
            "_iterates": iterates,
        })
    cold, warm = results
    row.update(
        iterations=cold["iterations"], objf=cold["objf"], status=cold["status"],
        driver_status=cold["driver_status"], min_ie=cold["min_ie"],
        max_eq=max(abs(v) for v in cold["equalities"].values()),
        cold_s=cold["total_s"], warm_s=warm["total_s"],
        x=cold["x"], closed=cold["closed"], root_finds=cold["root_finds"],
        trace=cold["trace"], _iterates=cold["_iterates"],
    )
    return row


def newton_along_trace(built: closing.Closed, row: dict) -> list[dict]:
    """The nested Newtons at every outer iterate the optimiser recorded: the schedule
    re-run at each `x`, its `Steps`/`Converged` reports read back.
    """
    problem = built.problem
    env = closing.seed(built, built.session.reference.cold)
    env, _ = mdf.prime(problem, env)
    inputs = mdf._inputs_only(problem, env)
    out_rows = []
    for x in row["_iterates"]:
        at = dict(inputs)
        at.update(zip(built.design, [jnp.asarray(v) for v in x], strict=True))
        out = run_schedule(problem.eager, at)
        out_rows.append(built.root_find_reports(out))
    return out_rows


def shape_of(built: closing.Closed, nested) -> dict:
    """The blocks the picture shows: the optimiser's, each root find's, the top."""
    graph = nested.graph
    return {
        "optimise_block": next(
            len(c) for c in graph.graph.components if closing.OPTIMISE in c
        ),
        "root_find_blocks": {
            c.spelling: next(len(b) for b in built.graph.graph.components if p in b)
            for c, p in built.places.items()
        },
        "graph_nodes": len(graph.nodes),
        "top_blocks": len(graph.graph.components),
    }


def run_solve(pairings=None, suffix="", driver=None, flatten=True) -> list[dict]:
    live = open_live()
    built = closing.close(
        live, pairings or PAIRINGS, flatten=flatten,
        driver=None if driver is None else driver(),
    )
    print("\n".join(closing.describe(built.graph)))
    nested = closing.nested_blocking(built, driver=None if driver is None else driver())
    print("with the Optimise stated:")
    print("\n".join(closing.describe(nested)))
    rows = []
    for label, optimiser in (("vmcon", None), ("slsqp", SlsqpDriver)):
        row = solve(built, optimiser, label)
        print(
            f"[{label}] {row['status']} in {row['iterations']} it, objf {row['objf']!r}, "
            f"max|eq| {row['max_eq']:.2e}, min ie {row['min_ie']:.2e}, cold {row['cold_s']:.1f} s, "
            f"warm {row['warm_s']:.2f} s, driver status {row['driver_status']}"
        )
        print("   root finds at the answer:", row["root_finds"])
        along = newton_along_trace(built, row)
        row["newton_steps_along_trace"] = [max(r[0] for r in at.values()) for at in along]
        row["unconverged_iterates"] = [i for i, at in enumerate(along) if not all(r[1] for r in at.values())]
        row["max_newton_steps"] = max(row["newton_steps_along_trace"], default=0)
        print(f"   Newton steps per outer iterate (max over the closures): {row['newton_steps_along_trace']}")
        print(f"   outer iterates where a closure did NOT converge: {row['unconverged_iterates']}")
        rows.append(row)
    # PROCESS's own answer and the MDF's (same cut, same optimisers, cold then warm),
    # for the comparison columns.
    process = process_reference(NAME)
    reference = {"process_x": process["x"], "process_iterations": process["iterations"]}
    build = session.build_mdf(live.reference, live.machine_graph, live.switch_values, cut=live.cut)
    for label, optimiser in (("vmcon", None), ("slsqp", SlsqpDriver)):
        cold = session.solve_mdf(build, live.reference, live.reference.cold, optimiser=optimiser)
        warm = session.solve_mdf(build, live.reference, live.reference.cold, optimiser=optimiser)
        row = {
            "optimiser": f"mdf-{label}", "iterations": cold["iterations"], "objf": cold["objf"],
            "status": cold["status"], "max_eq": cold["max_eq"], "min_ie": cold["min_ie"],
            "cold_s": cold["seconds"], "warm_s": warm["seconds"],
            "x": dict(zip([v.spelling for v in build.problem.design], cold["x"], strict=True)),
            "root_finds": {}, "max_newton_steps": None, "unconverged_iterates": [],
        }
        print(f"[mdf-{label}] {row['status']} in {row['iterations']} it, objf {row['objf']!r}, "
              f"max|eq| {row['max_eq']:.2e}, cold {row['cold_s']:.1f} s, warm {row['warm_s']:.2f} s")
        rows.append(row)
    for row in rows:
        row["worst_dx_vs_process"] = max(
            abs(v - process["x"][k]) / abs(process["x"][k]) for k, v in row["x"].items()
        )
    shape = shape_of(built, nested)
    payload = {"rows": rows, "reference": reference, "shape": shape,
               "structure": closing.describe(nested),
               "pairings": {c.spelling: v.spelling for c, v in built.pairings.items()},
               "flattened": built.flat, "cut": CUT}
    write_json("close_conditions.py", payload, name=f"close_conditions_solve{suffix}")
    hidden = ("trace", "_iterates", "x", "closed", "root_finds", "primed_root_finds",
              "newton_steps_along_trace", "unconverged_iterates")
    flat = [{k: v for k, v in r.items() if k not in hidden} for r in rows]
    write_csv("close_conditions.py", flat, name=f"close_conditions_solve{suffix}")
    header = ["optimiser", "it", "verdict", "objective", r"$\max|h|$", r"$\min g$", "cold s", "warm s",
              "Newton steps (answer / max)", "iterates w/o root"]
    body = [[r["optimiser"].upper(), r["iterations"], r["status"], fmt(r["objf"], 8), sci(r["max_eq"]),
             sci(r["min_ie"]), fmt(r["cold_s"], 1), fmt(r["warm_s"], 2),
             ("/".join(str(v[0]) for v in r["root_finds"].values()) + f" / {r['max_newton_steps']}")
             if r["root_finds"] else "--",
             len(r["unconverged_iterates"]) if r["root_finds"] else "--"] for r in rows]
    write_tex("close_conditions.py", header, body, name=f"close_conditions_solve{suffix}", align="lrlrrrrrrr",
              caption_note=f"equalities closed inside the MDA: {payload['pairings']}; the optimiser keeps "
                           f"{len(built.design)} design variables, the objective and "
                           f"{built.problem.n_inequality} inequalities")
    return rows


# ---------------------------------------------------------------- the batch angle


def predicted_starts(built: closing.Closed, point: PathMap, values: dict) -> dict:
    """`{guess port: [N] array}` -- the closing unknowns' roots to first order in the
    batch's design offsets: `u*(x0) + du*/dx (x_i - x0)`, the sensitivities by one
    `jax.jacfwd` of the unbatched closed MDA at the batch's centre (through the root
    finds' `custom_root`). Computed once per batch, outside the timed call; a scan
    around a design has this derivative at hand anyway. The +-1 % offsets leave a
    second-order residual (~1e-4), which Newton or Broyden closes in two steps.
    """
    problem = built.problem
    design = tuple(built.design)
    unknowns = []
    for cond in built.places:
        unknowns += [u for u in built.unknowns(cond) if u not in unknowns]
    ports = {u: built.start_port(u) for u in unknowns}
    single = jax.jit(problem.traceable.run)

    def roots(xs):
        env = dict(point.items())
        env.update(zip(design, xs, strict=True))
        out = single(PathMap(env))
        return jnp.stack([out[u] for u in unknowns])

    x0 = tuple(jnp.asarray(point[v]) for v in design)
    sensitivity = jnp.stack(list(jax.jacfwd(roots)(x0)), axis=1)  # [unknowns, design]
    u0 = roots(x0)
    dx = jnp.stack([values[v] - point[v] for v in design], axis=1)  # [N, design]
    predicted = u0[None, :] + dx @ sensitivity.T
    return {ports[u]: predicted[:, i] for i, u in enumerate(unknowns)}


def batch_rows(sizes=(1, 4, 16, 64, 256, 1024, 4096), repeats=3, only=None, pairings=None) -> list[dict]:
    """The four shapes (`plain`, `nested`, `closed`, `predicted`), single and `vmap`,
    on whichever backend jax is on (`JAX_PLATFORMS`); rows say which. `only`: a subset
    of the labels (`--shapes`), written to `close_conditions_batch_<backend>_<labels>`
    so one process per shape -- the device memory an earlier ladder held is never given
    back -- does not overwrite the others' rows.
    """
    backend = jax.default_backend()
    live = open_live()
    pairings = pairings or PAIRINGS
    plain = session.build_mdf(live.reference, live.machine_graph, live.switch_values, cut=live.cut).problem
    nested = closing.close(live, pairings, flatten=False, driver=closing.safeguarded_newton())
    flat = closing.close(live, pairings)
    rows = []
    shapes = (
        ("plain", plain, plain.design, None, False),
        ("nested", nested.problem, nested.design, nested, False),
        ("closed", flat.problem, flat.design, flat, False),
        ("predicted", flat.problem, flat.design, flat, True),
    )
    if only is not None:
        if unknown := set(only) - {s[0] for s in shapes}:
            raise ValueError(f"no such shape: {sorted(unknown)}")
        shapes = tuple(s for s in shapes if s[0] in only)
    name = f"close_conditions_batch_{backend}" + ("" if only is None else "_" + "_".join(s[0] for s in shapes))
    for label, problem, design, built, predict in shapes:
        env = mdf.seed(problem, live.reference.cold) if built is None else closing.seed(built, live.reference.cold)
        env, _ = mdf.prime(problem, env)
        inputs = mdf._inputs_only(problem, env)
        point = PathMap(inputs.items())
        varied = tuple(design)
        single = jax.jit(problem.traceable.run)
        for n in sizes:
            rng = np.random.default_rng(0)
            values = dict(point.items())
            for v in varied:
                base = np.asarray(point[v])
                values[v] = jnp.asarray(base * (1 + 0.01 * rng.uniform(-1, 1, size=n)))
            batched = set(varied)
            if predict and n > 1:
                began = time.perf_counter()
                starts = predicted_starts(built, point, values)
                jax.block_until_ready(list(starts.values()))
                predict_s = time.perf_counter() - began
                values.update(starts)
                batched |= set(starts)
            axes = PathMap((v, 0 if v in batched else None) for v in point)
            fn = jax.jit(jax.vmap(problem.traceable.run, in_axes=(axes,))) if n > 1 else single
            arg = PathMap(values) if n > 1 else point
            out = None  # the previous size's output is not kept alive across this call
            try:
                began = time.perf_counter()
                out = jax.block_until_ready(fn(arg))
                first = time.perf_counter() - began
                walls = []
                for _ in range(repeats):
                    began = time.perf_counter()
                    out = jax.block_until_ready(fn(arg))
                    walls.append(time.perf_counter() - began)
            except Exception as failure:  # noqa: BLE001 -- an OOM is a row, not an exit
                rows.append({"shape": label, "backend": backend, "N": n, "vmap": n > 1,
                             "status": f"{type(failure).__name__}: {str(failure)[:80]}"})
                print(rows[-1])
                break
            row = {
                "shape": label, "backend": backend, "N": n, "vmap": n > 1, "first_call_s": first,
                "warm_s": min(walls), "us_per_point": 1e6 * min(walls) / n,
            }
            if built is not None:
                eq = {c.spelling: np.asarray(out[c]) for c in built.pairings}
                places = list(dict.fromkeys(built.places.values()))
                steps = [np.asarray(out[Steps.name_for(p)]).reshape(-1) for p in places]
                row.update({
                    "max_abs_eq": max(float(np.max(np.abs(v))) for v in eq.values()),
                    "newton_steps_max": max(int(np.max(st)) for st in steps),
                    "newton_steps_mean": float(np.mean(np.concatenate(steps))),
                    "all_converged": bool(all(np.all(np.asarray(out[Converged.name_for(p)])) for p in places)),
                })
            if predict and n > 1:
                row["predict_s"] = predict_s
            rows.append(row)
            print(rows[-1])
    write_csv("close_conditions.py", rows, name=name)
    header = ["MDA", "N", "compile s", "warm s", r"$\mu$s/point", r"$\max|h|$", "Newton steps (max / mean)"]
    body = [[r["shape"], r["N"], fmt(r["first_call_s"], 1), fmt(r["warm_s"], 4), fmt(r["us_per_point"], 1),
             sci(r["max_abs_eq"]) if r.get("max_abs_eq") is not None else "--",
             f"{r['newton_steps_max']} / {r['newton_steps_mean']:.2f}" if r.get("newton_steps_max") is not None else "--"]
            if "status" not in r else [r["shape"], r["N"], r["status"], "", "", "", ""] for r in rows]
    write_tex("close_conditions.py", header, body, name=name, align="lrrrrrr",
              caption_note=(
                  f"{backend}; N > 1 is jax.vmap over N design points, each design entry perturbed "
                  "by +-1 %. nested: the cut fixed points nested inside the root finds, exact Newton; "
                  "closed: flattened into one square problem per root find, Broyden; predicted: "
                  "closed, started from the first-order prediction of the roots (sensitivities by "
                  "one jacfwd at the centre, outside the timed call)"
              ))
    return rows


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv or not any(a in argv for a in ("--table", "--solve", "--batch")):
        print(__doc__)
        return 0
    pairings, suffix = None, ""
    if "--pairing" in argv:
        key = argv[argv.index("--pairing") + 1]
        pairings = dict(kinds.PAIRINGS[key]) if key != "historic" else dict(kinds.HISTORIC_PAIRING)
        suffix = "" if key == "historic" else f"_{key}"
    if "--table" in argv:
        rows = pairing_table(open_live())
        render_table(rows)
        for r in rows:
            if r["design"] or r["nodes_raw"] <= 8:
                print(f"{r['condition']:24} {r['variable']:55} {'D' if r['design'] else ' '} raw {r['nodes_raw']:3d} cut {r['nodes_cut']:3d} scaled {sci(r['scaled'])}")
    if "--solve" in argv:
        driver = None
        if "--pair16" in argv:
            var = argv[argv.index("--pair16") + 1]
            pairings = {**(pairings or PAIRINGS), kinds.C16: var}
            suffix += "_" + var.split(".")[-1]
        if "--driver" in argv:
            name = argv[argv.index("--driver") + 1]
            driver = DRIVERS[name]
            suffix += "" if name == "safeguarded" else f"_{name}"
        flatten = "--nested" not in argv
        if not flatten:
            suffix += "_nested"
        run_solve(pairings, suffix, driver, flatten)
    if "--batch" in argv:
        sizes = tuple(int(x) for x in argv[argv.index("--sizes") + 1].split(",")) if "--sizes" in argv else (1, 4, 16, 64, 256, 1024, 4096)
        only = tuple(argv[argv.index("--shapes") + 1].split(",")) if "--shapes" in argv else None
        batch_rows(sizes, only=only, pairings=pairings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
