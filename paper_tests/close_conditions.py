"""Consistency solved where it arises: the two **equality** constraints of
`stellarator_helias` closed by a root find over one design variable each, nested inside
the MDA, and the optimiser left with the objective, the twelve inequalities and the six
design variables that remain.

MDF hands every condition to the optimiser, which owns every design variable, so the
`Optimise` closes one cycle over most of the machine (`dsms.py`, `uncut_optimiser`: 121
nodes here). An equality need not be closed that way. `Determine(condition, unknown)` --
here `RootFind(conditions=(cond,), unknowns=(var,))` inserted into the graph -- closes a
cycle of its own, and the size of that cycle depends on the variable chosen: the power
balance (`^cond.constraints.c2`) closed by `hfact` is **three nodes**, since `hfact`
enters the confinement time and nothing else; closed by `rmajor` it is 24. The table
this script writes (`--table`) is that enumeration -- every equality against every
design variable and every scalar boundary input it is structurally sensitive to: the
cycle's size on the raw and on the cut graph, and the sensitivity `d cond / d var`
through the converged MDA (`jax.jacfwd` of `mdf.condition_map`), so that a small cycle
with no leverage (`t_tf_superconductor_quench` -> `c16`: six nodes, scaled sensitivity
8e-5) is not mistaken for a candidate.

The solve (`--solve`) takes the chosen pairings (`PAIRINGS`), builds

    raw -> recipe cut -> constraint and objective nodes (`mdf.mdf_graph`)
        -> Insert RootFind per equality -> NestInside(each root find)
        -> drivers: Picard on the cut fixed points, Newton on the root finds

and drives VMCON (and SLSQP) over the six remaining design variables from the same cold
start `mdf.solve` uses, through the same `mdf.Mdf` / `mdf.condition_map` machinery: the
`Mdf` record is built by hand with `n_equality = 0`, so nothing in `mdf.py` had to
change. The optimiser is stated as a node too (`nested_blocking`) for the structural
picture -- the blocking with the `Optimise` around everything it iterates and the two
root finds nested inside it -- but the solve runs VMCON from outside the graph, as
`mdf.assemble` does.

    $PY paper_tests/close_conditions.py --table            # ~2 min
    $PY paper_tests/close_conditions.py --solve            # ~5 min
    $PY paper_tests/close_conditions.py --solve --pair16 .physics.nd_plasma_electrons_vol_avg
    $PY paper_tests/close_conditions.py --solve --driver newton   # optimistix's undamped Newton
    $PY paper_tests/close_conditions.py --batch [--sizes 1,64,1024]   # vmap sweep, CPU
    JAX_PLATFORMS=cuda $G paper_tests/close_conditions.py --batch      # the same on the GPU

Outputs: `out/close_conditions.csv` (the pairing table), `out/close_conditions.tex`
(design-variable rows plus the ten smallest non-design cycles per equality),
`out/close_conditions_solve.{csv,tex}` and `out/close_conditions_batch.{csv,tex}`.
"""

from __future__ import annotations

import dataclasses
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # before any array: PROCESS is float64

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from common import (  # noqa: E402
    ROOT,
    fmt,
    sci,
    write_csv,
    write_json,
    write_tex,
)
from cottax.blocking import Blocking
from cottax.evaluation.schedule import Schedule
from cottax.names import PathMap
from cottax.plan import Insert, Plan
from cottax.problem import Converged, Optimise, RootFind, Start, Steps
from cottax.rewrites import NestInside
from cottax.spec import NodePath, VarPath
from jax.tree_util import GetAttrKey

from functional_process.cottax import mdf, sand, session
from functional_process.cottax.core.solver.drivers import SlsqpDriver
from functional_process.cottax.indat import graph_for
from functional_process.cottax.mda import assign_drivers, default_drivers
from functional_process.cottax.mda_harness import _without_excluded
from functional_process.cottax.recipes import recipe
from functional_process.cottax.run_cold_matrix import (
    MDF_MAX_ITER,
    MDF_TOLERANCE,
    _recorder,
    _status,
    _trace_tail,
    build_mdf,
    solve_mdf,
)
from functional_process.cottax.sand_harness import reference_run, run_schedule

PATH = "tests/regression/input_files/stellarator_helias.IN.DAT"
CUT = "gauss_seidel_minimal"

PAIRINGS = {
    "^cond.constraints.c2": ".physics.hfact",
    "^cond.constraints.c16": ".physics.f_nd_alpha_thermal_electron",
}
"""Which design variable closes which equality -- the choice `--table` justifies:
`hfact` is the smallest cycle for the power balance (3 nodes, scaled sensitivity 1.0,
and the power balance is what `hfact` exists to close in PROCESS's own usage). For the
net electric power the smallest cycle among the design variables is
`t_tf_superconductor_quench` (6 nodes) but its scaled sensitivity is 8e-5 -- a 1 %
change in the quench time moves the residual by 1e-6 -- so it is not a candidate. The
next is `f_nd_alpha_thermal_electron` (26 nodes, scaled sensitivity 0.42), whose root
exists at the cold start (0.0198; the residual runs from -0.10 at `f_nd_alpha -> 0` to
+0.37 at the file's 0.1) and, measured, at every outer iterate of both optimisers. The
density (27 nodes, 1.7) is the alternative `--pair16` solves: one node more, four times
the leverage, and VMCON's trajectory with it passes through eleven iterates where the
power balance has no root in `hfact` (`te` at its lower bound, 3 keV), which costs it
129 iterations against 16."""

NEWTON_TOL = 1e-10
"""`rtol = atol` of the nested Newton on the two root finds. The port's default
(`SeededNewtonDriver`, 1e-4) would leave the equalities four orders looser than the
MDF's own `max|eq|` of 9e-11."""

PLACE = NodePath((GetAttrKey("Close"),))
"""Where a `RootFind` closing an equality binds: `.Close.c2`, `.Close.c16`."""


def open_live(cut_name: str = CUT):
    return session.open_session(PATH, cut=recipe(cut_name))


def raw_graph(live):
    return _without_excluded(
        live.machine_graph if live.machine_graph is not None else graph_for()
    )


def with_conditions(live, graph):
    """`graph` plus the constraint and objective nodes, and `mdf_graph`'s report."""
    ref = live.reference
    inserted, _conditions, _n, report = mdf.mdf_graph(
        graph, ref.icc, ref.n_equality, ref.i_figure_merit, live.switch_values
    )
    return inserted, report


def design_of(live) -> tuple[VarPath, ...]:
    return tuple(sand.iteration_variable_path(i) for i in live.reference.ixc)


def place_for(cond: VarPath) -> NodePath:
    return NodePath((*PLACE.segments, GetAttrKey(cond.spelling.split(".")[-1])))


def cycle_of(graph, cond: VarPath, var: VarPath) -> tuple[NodePath, ...]:
    """The component `Determine(cond, var)` creates: a `RootFind` over `var` reading
    `cond`, inserted, and the SCC it lands in.
    """
    place = place_for(cond)
    with_problem = (
        Plan(graph) + Insert(PathMap(((place, RootFind((cond,), (var,))),)))
    ).graph
    return next(c for c in with_problem.components if place in c)


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
    build = build_mdf(ref, live.machine_graph, live.switch_values, cut=live.cut)
    problem = build.problem
    env = mdf.seed(problem, live.cold)
    env, _out = mdf.prime(problem, env)
    inputs = set(problem.eager.inputs)
    condition_index = {c: i for i, c in enumerate(problem.conditions)}

    # Which boundary inputs reach which equality, structurally.
    sensitive: dict[VarPath, set[VarPath]] = {}
    for cond in equalities:
        ancestors = set(raw_c.ancestors([raw_c.owners[cond]]))
        for var in raw_c.boundary_inputs:
            if any(r in ancestors for r in raw_c.readers.get(var, ())):
                sensitive.setdefault(var, set()).add(cond)
    candidates = [v for v in raw_c.boundary_inputs if v in sensitive]
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
    began = time.perf_counter()
    jacobian, _compile, _ms = mdf.jacobian(cmap, start)
    print(f"jacobian over {len(scalar)} scalar inputs: {time.perf_counter() - began:.1f} s")
    column = {v: j for j, v in enumerate(scalar)}

    rows = []
    for cond in equalities:
        for var in candidates:
            if cond not in sensitive[var]:
                continue
            raw_cycle = cycle_of(raw_c, cond, var)
            cut_cycle = cycle_of(cut_c, cond, var)
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


# ---------------------------------------------------------------- the closed graph


@dataclasses.dataclass(frozen=True)
class Closed:
    """The architecture: an `Mdf` whose schedule carries the two root finds."""

    live: object
    problem: mdf.Mdf
    pairings: dict
    """`{condition: design variable}` -- what closes what."""
    places: dict
    """`{condition: NodePath}` -- where each root find binds."""
    design: tuple[VarPath, ...]
    """The six the optimiser keeps."""
    blocking: Blocking
    """`Blocking.scc` of the driven graph, root finds nested, no `Optimise`."""

    def root_find_reports(self, out) -> dict:
        """`{condition: (steps, converged, residual, unknown)}` out of a run's env."""
        return {
            cond: (
                int(np.asarray(out[Steps.name_for(place)])),
                bool(np.asarray(out[Converged.name_for(place)])),
                float(np.asarray(out[cond])),
                float(np.asarray(out[self.pairings[cond]])),
            )
            for cond, place in self.places.items()
        }


def newton(**kwargs):
    """The driver on a closing root find: optimistix's Newton, reporting its verdict
    (`Steps`, `Converged`, `Status`) instead of raising, at `NEWTON_TOL`.
    """
    return mdf.MdfNewtonDriver(**{"rtol": NEWTON_TOL, "atol": NEWTON_TOL, **kwargs})


class SafeguardedNewtonDriver(mdf.MdfNewtonDriver):
    """`MdfNewtonDriver`'s contract (Newton, reports its verdict), with the two
    safeguards VMCON's line search forced: a relative step cap and backtracking on the
    residual norm, in coordinates scaled by the starting guess.

    An undamped Newton on a residual with **no root** runs away: at VMCON's eighth trial
    point of this problem (`te` pushed to its lower bound, 3 keV) the power balance
    tends to -0.64 as `hfact -> inf`, `optimistix.Newton` reached `hfact = 5e211` in 256
    steps, the derivative of an inequality downstream came back non-finite and
    `VmconDriver` refused the whole solve (`VMCON_NON_FINITE`, the start returned
    untouched). `optimistix`'s damped least-squares solvers (`LevenbergMarquardt`,
    `Dogleg`) were tried first and raise from inside `lineax` on this block even at the
    root -- a trial step of theirs sends the nested fusion-rate Picard non-finite and
    lineax's `error_if` turns that into an exception rather than a `nan` -- while
    `GaussNewton` is undamped and no better than Newton. So the loop is written out:
    `jax.lax.custom_root` for the implicit derivative, a `while_loop` of capped,
    backtracked Newton steps inside. Where a root exists it converges like Newton;
    where none exists it stops at a finite point with a non-zero residual (`Converged`
    false), so the outer line search sees a finite, infeasible merit and backs off.
    """

    max_steps: int = 40
    cap: float = 0.5
    """Largest step, relative to the current iterate."""
    halvings: int = 12
    """Backtracking budget per Newton step. **Only an active point may spend it**
    (`worse` below is masked by `norm(r) > tol`): under `jax.vmap` a `while_loop` runs
    its body for every point until the last one's predicate is false, with `select` on
    the carry -- so a point the outer loop has already finished still evaluates a
    Newton step, and at a root that step is noise, `norm(r_new) >= norm(r)` half the
    time, and its halving loop runs to the 12-step cap, each halving one evaluation
    of the whole cycle **for the whole batch**. Measured on the GPU at N = 4096 before
    the mask: no active point ever needed a halving, the batch-max halvings per outer
    step were [0, 0, 12, 12] once 25 % of the points had converged, and the driver
    was 852 ms against 186 ms with them masked (the same `u`, the same step counts).
    For a single point the mask changes nothing: the outer `go_on` implies it."""

    def __call__(self, conditions, data) -> tuple:
        from jax import lax  # noqa: PLC0415
        from jax.flatten_util import ravel_pytree  # noqa: PLC0415

        start = data.get(Start)
        flat_guess, unravel = ravel_pytree(start)
        scale = jnp.where(flat_guess == 0.0, 1.0, flat_guess)
        tol, cap, halvings, max_steps = self.rtol, self.cap, self.halvings, self.max_steps

        def residual(u):
            out, _ = ravel_pytree(conditions(*unravel(u * scale)))
            return out

        def norm(r):
            return jnp.max(jnp.abs(r))

        def solve(f, u0):
            def step(state):
                u, r, k, _stalled = state
                jacobian = jax.jacfwd(f)(u)
                du = -jnp.linalg.solve(jacobian, r)
                bound = cap * jnp.maximum(jnp.abs(u), 1e-3)
                du = jnp.clip(du, -bound, bound)

                active = norm(r) > tol  # false where the outer loop is done (vmap)

                def worse(bs):
                    _t, r_new, j = bs
                    bad = ~jnp.all(jnp.isfinite(r_new)) | (norm(r_new) >= norm(r))
                    return bad & (j < halvings) & active

                def halve(bs):
                    t, _r_new, j = bs
                    t = 0.5 * t
                    return t, f(u + t * du), j + 1

                t, r_new, _j = lax.while_loop(worse, halve, (1.0, f(u + du), 0))
                accepted = jnp.all(jnp.isfinite(r_new)) & (norm(r_new) < norm(r))
                u_new = jnp.where(accepted, u + t * du, u)
                r_new = jnp.where(accepted, r_new, r)
                return u_new, r_new, k + 1, ~accepted

            def go_on(state):
                _u, r, k, stalled = state
                return (norm(r) > tol) & (k < max_steps) & ~stalled

            r0 = f(u0)
            u, r, k, stalled = lax.while_loop(go_on, step, (u0, r0, 0, False))
            # As floats: `custom_root`'s aux must carry a float tangent.
            return u, jnp.stack([k, norm(r) <= tol, stalled]).astype(float)

        def tangent_solve(g, y):
            return jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)

        u, aux = lax.custom_root(
            residual, jnp.ones_like(flat_guess), solve, tangent_solve, has_aux=True
        )
        aux = jax.lax.stop_gradient(aux)
        steps, converged, stalled = aux[0].astype(int), aux[1] > 0.5, aux[2] > 0.5
        status = jnp.where(converged, 0, jnp.where(stalled, 2, 1))
        return (*unravel(u * scale), steps, converged, status)


def safeguarded(**kwargs):
    return SafeguardedNewtonDriver(**{"rtol": NEWTON_TOL, "atol": NEWTON_TOL, **kwargs})


def closed(live, pairings=None, driver=safeguarded) -> Closed:
    """Build the graph and the `Mdf` record for `pairings` (spellings, or `PAIRINGS`)."""
    raw = raw_graph(live)
    graph, report = with_conditions(live, live.cut(raw))
    design = design_of(live)
    by_spelling = {v.spelling: v for v in design}
    conds = {c.spelling: c for c in report["equalities"]}
    chosen = {
        conds[c]: by_spelling[v] for c, v in (pairings or PAIRINGS).items()
    }
    places = {}
    for cond, var in chosen.items():
        place = place_for(cond)
        places[cond] = place
        graph = (
            Plan(graph) + Insert(PathMap(((place, RootFind((cond,), (var,))),)))
        ).graph
    components = {
        cond: next(c for c in graph.components if p in c) for cond, p in places.items()
    }
    seen: set = set()
    for cond, component in components.items():
        if seen & set(component):
            raise ValueError(
                f"the root find for {cond.spelling} shares a cycle with another -- "
                f"pick pairings whose cycles are disjoint, or nest one in the other"
            )
        seen |= set(component)
    # Whatever declared problem sits on a root find's cycle (a cut fixed point) is
    # answered inside its iteration.
    for place in places.values():
        graph = (Plan(graph) + NestInside(place)).graph
    drivers = default_drivers(graph)
    for place in places.values():
        drivers[place] = driver()
    assigned = assign_drivers(graph, drivers)
    blocking = Blocking.scc(assigned)
    schedule = Schedule(blocking)
    kept = tuple(v for v in design if v not in set(chosen.values()))
    report = dict(
        report,
        blocks=len(blocking.blocks),
        driven_blocks=sum(1 for t in blocking.problem_types if t is not None),
    )
    problem = mdf.Mdf(
        graph=graph,
        eager=schedule,
        traceable=schedule,
        design=kept,
        conditions=(report["objective"], *report["inequalities"]),
        n_equality=0,
        n_inequality=len(report["inequalities"]),
        report=report,
        raw=raw,
    )
    return Closed(
        live=live, problem=problem, pairings=chosen, places=places, design=kept,
        blocking=blocking,
    )


def nested_blocking(built: Closed) -> Blocking:
    """The architecture as structure: the `Optimise` over the six inserted and
    `NestInside` it, so the root finds and the MDA's fixed points are answered inside
    its iteration; `VmconDriver` assigned for the picture.
    """
    graph = built.problem.graph
    report = built.problem.report
    place = NodePath((GetAttrKey("Opt"),))
    node = Optimise(
        objective=report["objective"],
        unknowns=built.design,
        equalities=(),
        inequalities=tuple(report["inequalities"]),
    )
    with_problem = (Plan(graph) + Insert(PathMap(((place, node),)))).graph
    with_problem = (Plan(with_problem) + NestInside(place)).graph
    drivers = default_drivers(with_problem)
    for rf in built.places.values():
        drivers[rf] = safeguarded()
    return Blocking.scc(assign_drivers(with_problem, drivers))


def describe(blocking: Blocking, depth: int = 0) -> list[str]:
    lines = []
    for i, block in enumerate(blocking.blocks):
        problem = blocking.problems[i]
        if len(block) == 1 and problem is None:
            continue
        label = "run" if problem is None else f"{blocking.problem_types[i]} {problem.spelling}"
        lines.append("  " * depth + f"block {i}: {len(block)} nodes, {label}")
        if blocking.inner[i] is not None:
            lines.extend(describe(blocking.inner[i], depth + 1))
    return lines


# ---------------------------------------------------------------- the solve


def solve(built: Closed, optimiser=None, label="vmcon") -> dict:
    """VMCON (or `optimiser`) over the six, from the cold start; then a warm repeat."""
    live, problem = built.live, built.problem
    ref = live.reference
    bounds = tuple(b for b in ref.bounds if b[0] in set(built.design))
    began = time.perf_counter()
    env = mdf.seed(problem, live.cold)
    env, primed = mdf.prime(problem, env)
    prime_seconds = time.perf_counter() - began
    row = {"optimiser": label, "prime_s": prime_seconds}
    row["primed_root_finds"] = {c.spelling: r for c, r in built.root_find_reports(primed).items()}
    results = []
    for _attempt in ("cold", "warm"):
        trace: list = []
        iterates: list = []
        record = _recorder(trace)

        def recording(i, result, x, convergence, record=record, iterates=iterates):
            record(i, result, x, convergence)
            iterates.append([float(v) for v in np.asarray(x)])

        began = time.perf_counter()
        x, out, seconds = mdf.solve(
            problem, dict(env), bounds=bounds, callback=recording,
            tolerance=MDF_TOLERANCE, max_iter=MDF_MAX_ITER,
            **({} if optimiser is None else {"optimiser": optimiser}),
        )
        total = time.perf_counter() - began
        iterations, objf, _max_eq, min_ie = _trace_tail(trace)
        results.append({
            "iterations": iterations, "objf": objf, "min_ie": min_ie,
            "status": _status(trace, MDF_TOLERANCE, MDF_MAX_ITER),
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


def newton_along_trace(built: Closed, row: dict) -> list[dict]:
    """The nested Newtons at every outer iterate VMCON recorded: the schedule re-run
    at each `x`, its `Steps`/`Converged` reports read back.
    """
    problem = built.problem
    env = mdf.seed(problem, built.live.cold)
    env, _ = mdf.prime(problem, env)
    inputs = mdf._inputs_only(problem, env)
    out_rows = []
    for x in row["_iterates"]:
        at = dict(inputs)
        at.update(zip(built.design, [jnp.asarray(v) for v in x], strict=True))
        out = run_schedule(problem.eager, at)
        out_rows.append(built.root_find_reports(out))
    return out_rows


def process_answer() -> dict:
    """PROCESS's own converged `x` and iteration count, out of the cached reference
    run (`sand_harness.reference_run` runs PROCESS once and pickles it).
    """
    run = reference_run(str(ROOT / PATH))
    return {
        "x": {sand.iteration_variable_path(i).spelling: v for i, v in run.converged.items()},
        "iterations": run.solver_iterations,
    }


def run_solve(pairings=None, suffix="", driver=safeguarded) -> list[dict]:
    live = open_live()
    built = closed(live, pairings, driver=driver)
    print("\n".join(describe(built.blocking)))
    nested = nested_blocking(built)
    print("with the Optimise stated:")
    print("\n".join(describe(nested)))
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
        row["max_newton_steps"] = max(row["newton_steps_along_trace"])
        print(f"   Newton steps per outer iterate (max over the two): {row['newton_steps_along_trace']}")
        print(f"   outer iterates where a closure did NOT converge: {row['unconverged_iterates']}")
        rows.append(row)
    # PROCESS's own answer and the MDF's (same cut, same optimisers, cold then warm),
    # for the comparison columns.
    process = process_answer()
    reference = {"process_x": process["x"], "process_iterations": process["iterations"]}
    for label, optimiser in (("vmcon", None), ("slsqp", SlsqpDriver)):
        cold = live.mdf() if optimiser is None else solve_mdf(live.mdf_build, live.reference, live.cold, optimiser=optimiser)
        warm = solve_mdf(live.mdf_build, live.reference, live.cold, optimiser=optimiser)
        row = {
            "optimiser": f"mdf-{label}", "iterations": cold["iterations"], "objf": cold["objf"],
            "status": cold["status"], "max_eq": cold["max_eq"], "min_ie": cold["min_ie"],
            "cold_s": cold["_seconds_total"], "warm_s": warm["_seconds_total"],
            "x": dict(zip([v.spelling for v in live.mdf_build.problem.design], cold["_x"], strict=True)),
            "root_finds": {}, "max_newton_steps": None, "unconverged_iterates": [],
        }
        print(f"[mdf-{label}] {row['status']} in {row['iterations']} it, objf {row['objf']!r}, max|eq| {row['max_eq']:.2e}, cold {row['cold_s']:.1f} s, warm {row['warm_s']:.2f} s")
        rows.append(row)
    for row in rows:
        row["worst_dx_vs_process"] = max(
            abs(v - process["x"][k]) / abs(process["x"][k]) for k, v in row["x"].items()
        )
    top = Blocking.scc(nested.graph)
    shape = {
        "optimise_block": next(len(b) for i, b in enumerate(nested.blocks) if nested.problems[i] is not None and nested.problems[i].spelling == ".Opt"),
        "root_find_blocks": {c.spelling: len(built.blocking.block_of(p)) for c, p in built.places.items()},
        "graph_nodes": len(nested.graph.nodes),
        "top_blocks": len(top.blocks),
    }
    payload = {"rows": rows, "reference": reference, "shape": shape,
               "structure": describe(nested), "pairings": {c.spelling: v.spelling for c, v in built.pairings.items()}}
    write_json("close_conditions.py", payload, name=f"close_conditions_solve{suffix}")
    flat = [{k: v for k, v in r.items() if k not in ("trace", "_iterates", "x", "closed", "root_finds", "primed_root_finds", "newton_steps_along_trace", "unconverged_iterates")} for r in rows]
    write_csv("close_conditions.py", flat, name=f"close_conditions_solve{suffix}")
    header = ["optimiser", "it", "verdict", "objective", r"$\max|h|$", r"$\min g$", "cold s", "warm s", "Newton steps (answer / max)", "iterates w/o root"]
    body = [[r["optimiser"].upper(), r["iterations"], r["status"], fmt(r["objf"], 8), sci(r["max_eq"]), sci(r["min_ie"]),
             fmt(r["cold_s"], 1), fmt(r["warm_s"], 2),
             ("/".join(str(v[0]) for v in r["root_finds"].values()) + f" / {r['max_newton_steps']}") if r["root_finds"] else "--",
             len(r["unconverged_iterates"]) if r["root_finds"] else "--"] for r in rows]
    write_tex("close_conditions.py", header, body, name=f"close_conditions_solve{suffix}", align="lrlrrrrrrr",
              caption_note=f"equalities closed inside the MDA: {payload['pairings']}; the optimiser keeps 6 design variables, the objective and 12 inequalities")
    return rows


# ---------------------------------------------------------------- the batch angle


def batch_rows(sizes=(1, 4, 16, 64, 256, 1024, 4096), repeats=3) -> list[dict]:
    """One MDA evaluation, plain and with the root finds inside: single and `vmap`,
    on whichever backend jax is on (`JAX_PLATFORMS`); rows say which."""
    backend = jax.default_backend()
    live = open_live()
    built = closed(live)
    plain = build_mdf(live.reference, live.machine_graph, live.switch_values, cut=live.cut).problem
    rows = []
    for label, problem, design in (("plain", plain, plain.design), ("closed", built.problem, built.design)):
        env = mdf.seed(problem, live.cold)
        env, _ = mdf.prime(problem, env)
        inputs = mdf._inputs_only(problem, env)
        point = PathMap(inputs.items())
        varied = tuple(design)
        axes = PathMap((v, 0 if v in set(varied) else None) for v in point)
        single = jax.jit(problem.traceable.run)
        batched = jax.jit(jax.vmap(problem.traceable.run, in_axes=(axes,)))
        for n in sizes:
            rng = np.random.default_rng(0)
            values = dict(point.items())
            for v in varied:
                base = np.asarray(point[v])
                values[v] = jnp.asarray(base * (1 + 0.01 * rng.uniform(-1, 1, size=n)))
            arg = PathMap(values)
            fn = batched if n > 1 else single
            arg = arg if n > 1 else point
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
            eq = {c.spelling: np.asarray(out[c]) for c in built.pairings} if label == "closed" else {}
            rows.append({
                "shape": label, "backend": backend, "N": n, "vmap": n > 1, "first_call_s": first,
                "warm_s": min(walls), "us_per_point": 1e6 * min(walls) / n,
                "max_abs_eq": max(float(np.max(np.abs(v))) for v in eq.values()) if eq else None,
                "newton_steps_max": max(int(np.max(np.asarray(out[Steps.name_for(p)]))) for p in built.places.values()) if label == "closed" else None,
                "all_converged": bool(all(np.all(np.asarray(out[Converged.name_for(p)])) for p in built.places.values())) if label == "closed" else None,
            })
            print(rows[-1])
    write_csv("close_conditions.py", rows, name=f"close_conditions_batch_{backend}")
    header = ["MDA", "N", "compile s", "warm s", r"$\mu$s/point", r"$\max|h|$", "max Newton steps"]
    body = [[r["shape"], r["N"], fmt(r["first_call_s"], 1), fmt(r["warm_s"], 4), fmt(r["us_per_point"], 1),
             sci(r["max_abs_eq"]) if r.get("max_abs_eq") is not None else "--",
             r["newton_steps_max"] if r.get("newton_steps_max") is not None else "--"]
            if "status" not in r else [r["shape"], r["N"], r["status"], "", "", "", ""] for r in rows]
    write_tex("close_conditions.py", header, body, name=f"close_conditions_batch_{backend}", align="lrrrrrr",
              caption_note=f"{backend}; N > 1 is jax.vmap over N design points, each design entry perturbed by +-1 %")
    return rows


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--table" in argv:
        rows = pairing_table(open_live())
        render_table(rows)
        for r in rows:
            if r["design"] or r["nodes_raw"] <= 8:
                print(f"{r['condition']:24} {r['variable']:55} {'D' if r['design'] else ' '} raw {r['nodes_raw']:3d} cut {r['nodes_cut']:3d} scaled {sci(r['scaled'])}")
    if "--solve" in argv:
        pairings, suffix, driver = None, "", safeguarded
        if "--pair16" in argv:
            var = argv[argv.index("--pair16") + 1]
            pairings = {**PAIRINGS, "^cond.constraints.c16": var}
            suffix = "_" + var.split(".")[-1]
        if "--driver" in argv:
            name = argv[argv.index("--driver") + 1]
            driver = {"newton": newton, "safeguarded": safeguarded}[name]
            suffix += "" if name == "safeguarded" else f"_{name}"
        run_solve(pairings, suffix, driver)
    if "--batch" in argv:
        sizes = tuple(int(x) for x in argv[argv.index("--sizes") + 1].split(",")) if "--sizes" in argv else (1, 4, 16, 64, 256, 1024, 4096)
        batch_rows(sizes)
    if not any(a in argv for a in ("--table", "--solve", "--batch")):
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
