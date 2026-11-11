"""Each variant solved by SLSQP: first from the file's own start, as
`../architectures/solve.py` does it (assembly, the cold solve, `--repeats` warm ones,
every call the optimiser makes into the model counted and timed by that script's own
`CALLS`), then from `--starts` random starts, the same ones for every variant of a
machine.

**The starts.** One generator per machine, seeded by its name (`seed_of`), draws a
factor `U(1 - spread, 1 + spread)` per design variable and start; each design variable
starts at its file value times the factor, clipped to its bounds (a root find's
unknowns have none). The coupling copies and every other start come, as for the
file's start, from the MDA solved at that design (`study.mda_graph`, compiled once);
a start whose MDA is not finite is `invalid` for every variant and counts in no rate.
The graph and its condition scaling stay the file start's, so no start recompiles.

**Success.** `converged` is SLSQP's own verdict; `success` is converged *and* at the
reference optimum: an objective within `F_TOLERANCE` (relative) of the paper table's
MDF SLSQP objective on that machine (`../architectures/out/solve_minimal_slsqp`), or, for
a root find, unknowns within `X_TOLERANCE` (relative) of the variant's own answer from
the file's start. Every start's objective, residuals and design are stored, so either
criterion can be redrawn from the csv.

Writes `out/starts_minimal_slsqp/<machine>/<label>.csv` (one row per start), then the
variant's row in `out/solve_minimal_slsqp/<machine>.csv`; the second is written last,
so a variant with a row is finished.

    $PY paper_tests/architecture_variants/solve.py --configurations NAME --variants LABEL [--starts 100]
"""

from __future__ import annotations

import dataclasses
import time
import zlib

import jax
import jax.numpy as jnp
import numpy as np

import common
from common import bench

from cottax.core.frozen import OrderedMap  # noqa: E402
from functional_process import configurations  # noqa: E402
from functional_process.configurations import Optimisation  # noqa: E402
from functional_process.cottax import study  # noqa: E402
from functional_process.cottax.architectures.drivers import SlsqpDriver  # noqa: E402
from functional_process.cottax.problem import TOP  # noqa: E402

import variants as V  # noqa: E402

import equinox as eqx  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from functional_process.cottax.architectures import drivers  # noqa: E402
from functional_process.cottax.architectures.contracts import Start, Status  # noqa: E402
from functional_process.cottax.architectures.host_cache import flat_answer  # noqa: E402
from functional_process.cottax.architectures.reading import Gaps  # noqa: E402

arch_solve = common.load("architectures_solve", common.ARCH / "solve.py")   # patches the driver's calls
CALLS = arch_solve.CALLS

SLSQP_MESSAGES = {
    0: "Optimization terminated successfully",
    1: "Function evaluation required",
    2: "More equality constraints than independent variables",
    3: "More than 3*n iterations in LSQ subproblem",
    4: "Inequality constraints incompatible",
    5: "Singular matrix E in LSQ subproblem",
    6: "Singular matrix C in LSQ subproblem",
    7: "Rank-deficient equality constraint subproblem HFTI",
    8: "Positive directional derivative for linesearch",
    9: "Iteration limit reached",
}
"""scipy's SLSQP exit modes: `SlsqpDriver` reports `result.status` as its `Status`."""


# ---------------------------------------------------------------- what the timed run leaves behind


RECORD: dict = {"on": False, "calls": []}
"""While `on`, every call the optimiser makes into the model: `(kind, flat_x, raw)`, the
design unscaled (`flat_x`, the block's own coordinates) and, for a value call, the
values unscaled (`raw`, what `host_cache.flat_answer` stacks). Kept on the host, after the
call and outside the time `CALLS` takes of it: the compiled program is not touched."""


def _recording(scaled_problem):
    def recorded(*a, **kw):
        # The reading the driver poses: its `order` is the relations as it partitions them
        # (equalities first), which `flat_answer` stacks in.
        reading = a[1] if len(a) > 1 else kw.get("conditions")
        RECORD["order"] = getattr(reading, "order", ())
        evaluate, jacobian, both, unravel, scale, condition_scale, *rest = scaled_problem(*a, **kw)

        def wrap(kind, fn):
            def call(x):
                out = fn(x)
                if RECORD["on"]:
                    flat = np.array(x, dtype=float) / scale
                    raw = None
                    if kind != "jacobians":
                        raw = np.asarray(out[0] if kind == "fused" else out, dtype=float) / condition_scale
                    RECORD["calls"].append((kind, flat, raw))
                return out
            return call

        return (wrap("values", evaluate), wrap("jacobians", jacobian), wrap("fused", both),
                unravel, scale, condition_scale, *rest)
    return recorded


drivers.scaled_problem = _recording(drivers.scaled_problem)     # outside `CALLS`' own wrap

LAST: dict = {}
"""The last `study.solve`'s graph and values: where `study.answer` leaves the top's
reports, `Status` among them, which it does not hand back."""
_study_solve = study.solve


def _keeping(graph, inputs):
    out = _study_solve(graph, inputs)
    LAST["graph"], LAST["out"] = graph, out
    return out


study.solve = _keeping


def exit_code(assembled):
    """SLSQP's exit mode for the last solve of `assembled`, or `None` (a root find's
    Newton reports none; a solve that raised left nothing)."""
    if LAST.get("graph") is not assembled.graph:
        return None
    top = assembled.graph[TOP]
    slots = dict(zip(top.contract.reports, top.reports, strict=True))
    if Status not in slots:
        return None
    return int(np.asarray(LAST["out"][slots[Status][0]]))


# ---------------------------------------------------------------- the inner work


class InnerCounter:
    """The inner sweeps behind every model call of a solve, counted afterwards.

    A call into the top's problem runs each nested solve to convergence -- a block's
    MDA (Picard), an implicit model's Newton -- and `CALLS` counts it once whatever
    they took. This re-evaluates the problem at every point the optimiser called it at
    (`RECORD`), on a copy of the variant's graph that differs only in its nested
    problems reporting their `Steps` (`variants.driven_reporting_steps`; optimistix's
    `num_steps`), and reads those. The same program gives the conditions back as well,
    and they are checked against the values the timed call returned (`count_check`, the
    largest relative difference): the counts are of the run that was timed.

    A nested solve starts from its `Start` in the top's context, the same at every
    call, so re-running it at a point repeats exactly what the call did.

    `model_evals` turns the counts into model evaluations: per call, every model on the
    top's cycle once, each nested problem's own `size` models once per step instead.
    Nested problems are disjoint (each nested directly in the top) on every variant
    here; `nested_depth` says if one is not."""

    def __init__(self, configuration, variant, assembled):
        self.graph = V.variant_plan(configuration, variant, assembled.env, [],
                                    drive=V.driven_reporting_steps).graph
        self.inner = V.inner_problems(self.graph)
        self.top_models = V.top_models(self.graph)
        within = dict(self.graph.within)

        def depth(p):
            d = 0
            while p != TOP and p in within:
                p, d = within[p], d + 1
            return d

        self.nested_depth = max((depth(i.label) for i in self.inner), default=0)
        self._program = None
        self._unravel = None

    def names(self) -> list[str]:
        return [f"{i.kind}:{i.label}" for i in self.inner]

    def _prepare(self, inputs, order):
        posed, context, data = study.posed(self.graph, inputs)
        _, unravel = ravel_pytree(tuple(data[Start]))
        context = jax.tree.map(lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, context)
        gaps = Gaps(posed, context, tuple(order))
        steps_map = dataclasses.replace(posed.conditions, conditions=tuple(i.steps for i in self.inner))
        if self._program is None:
            self._unravel = unravel

            def program(gaps, steps_map, flat):
                u = self._unravel(flat)
                return flat_answer(gaps(*u)), tuple(steps_map(u, gaps.context))

            self._program = eqx.filter_jit(program)
        return gaps, steps_map

    def counts(self, inputs, calls, order=()) -> dict:
        if not calls:
            return {}
        gaps, steps_map = self._prepare(inputs, order)
        seen, worst = {}, 0.0
        totals = np.zeros(len(self.inner), dtype=np.int64)
        values_totals = np.zeros(len(self.inner), dtype=np.int64)
        evals = evals_values = 0
        outside = self.top_models - sum(i.size for i in self.inner)
        sizes = np.array([i.size for i in self.inner], dtype=np.int64)
        for kind, flat, raw in calls:
            key = flat.tobytes()
            if key not in seen:
                values, steps = self._program(gaps, steps_map, jnp.asarray(flat))
                seen[key] = (np.asarray(values, dtype=float), np.array([int(np.asarray(s)) for s in steps], dtype=np.int64))
            values, steps = seen[key]
            if raw is not None:
                scale = np.maximum(np.abs(raw), 1e-30)
                worst = max(worst, float(np.max(np.abs(values - raw) / np.maximum(scale, np.max(np.abs(raw)) * 1e-12))))
            n = outside + int(steps @ sizes)
            totals += steps
            evals += n
            if kind != "jacobians":
                values_totals += steps
                evals_values += n
        out = {
            "points": len(seen),
            "inner_steps": int(totals.sum()),
            "inner_steps_values": int(values_totals.sum()),
            "model_evals": evals,
            "model_evals_values": evals_values,
            "count_check": worst,
        }
        out.update({f"steps:{name}": int(t) for name, t in zip(self.names(), totals)})
        return out

FOLDER = f"solve_{common.SCHEME}_{common.OPTIMISER}"
STARTS = f"starts_{common.SCHEME}_{common.OPTIMISER}"
F_TOLERANCE = 1e-3
X_TOLERANCE = 1e-3


def seed_of(name: str) -> int:
    return zlib.crc32(name.encode())


def reference_objective(name: str) -> float | None:
    """The paper table's MDF SLSQP objective on `name`, from the Snellius csv."""
    for r in common.read_rows(common.ARCH / "out" / FOLDER / f"{name}.csv"):
        if r["arm"] == "MDF" and r["objf"] not in ("", "None"):
            return float(r["objf"])
    return None


def finite(tree) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree.leaves(tree)
               if np.issubdtype(np.asarray(leaf).dtype, np.floating))


def same_value(got, want) -> bool:
    """Numbers to 1e-9 (relative), every leaf of a pytree; anything else (a name, an
    integer table) equal."""
    a, b = jax.tree.leaves(got), jax.tree.leaves(want)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.dtype.kind in "fc" or ya.dtype.kind in "fc":
            if xa.shape != ya.shape or not np.allclose(xa.astype(float), ya.astype(float),
                                                       rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
        elif not np.array_equal(xa, ya):
            return False
    return True


def as_like(value, like):
    like = jnp.asarray(like)
    return jnp.asarray(value, dtype=like.dtype).reshape(like.shape)


class Starts:
    """The inputs of one variant at each random start: the design perturbed, the MDA
    solved there, `study.block_values` from it, the design starts set."""

    def __init__(self, configuration, assembled, variant):
        self.configuration, self.assembled, self.variant = configuration, assembled, variant
        self.state = study.state_of(configuration)
        self.model = study.machine_model(configuration)
        self.mda = study.mda_graph(self.model, study.RULE)
        self.cold = study.cold_state(configuration)
        self.mda_nominal = study.values(self.mda, self.state, self.model.given, self.cold)
        problem = configuration.problem
        self.design = tuple(problem.design)
        self.bounds = dict(problem.design) if isinstance(problem, Optimisation) else {}
        self.block_given = study.problem_model(configuration).given
        started = study.starts(assembled.graph)
        top = assembled.graph[TOP]
        top_starts = top.data[top.contract.takes.index(study.Start)]
        self.design_starts = {s: started[s] for s in top_starts if started[s] in set(self.design)}
        self.x0 = {u: float(np.asarray(assembled.inputs[s])) for s, u in self.design_starts.items()}

    def mda_inputs(self, x: dict):
        """The MDA's inputs at the file's values, the design places among them (each a
        boundary input of its own, an array element included) set to `x`."""
        inputs = dict(self.mda_nominal)
        hit = set()
        for var in inputs:
            for u in (var, study._place(var)):
                if u in x:
                    inputs[var] = as_like(x[u], inputs[var])
                    hit.add(u)
        missing = set(x) - hit
        if missing:
            raise KeyError(f"design variables not found among the MDA's inputs: {sorted(map(str, missing))}")
        return OrderedMap(inputs)

    def inputs_at(self, x: dict):
        """`(inputs, env)` at design `x`, or `(None, None)` where the MDA there raises or
        leaves an input of the solve not finite. Not the whole MDA: a place PROCESS never
        writes is NaN by design (`truth.UNWRITTEN_BY_PROCESS`; `.physics.nu_star` on
        `helias_5b`), and the solve does not read it."""
        try:
            env = study.solve(self.mda, self.mda_inputs(x))
        except Exception:  # noqa: BLE001 -- an MDA that fails is an invalid start
            return None, None
        inputs = dict(study.block_values(self.assembled.graph, self.state, self.block_given, env,
                                         set(self.design), nested_given=V.nested_given(self.variant)))
        for s, u in self.design_starts.items():
            inputs[s] = as_like(x[u], inputs[s])
        if not finite(list(inputs.values())):
            return None, None
        return OrderedMap(inputs), env

    def draw(self, n: int, spread: float):
        rng = np.random.default_rng(seed_of(self.configuration.name))
        factors = rng.uniform(1 - spread, 1 + spread, size=(n, len(self.design)))
        for k in range(n):
            x = {}
            for u, f in zip(self.design, factors[k]):
                value = self.x0[u] * f
                if u in self.bounds:
                    lo, hi = self.bounds[u]
                    value = min(max(value, lo), hi)
                x[u] = value
            yield k, x

    def check_nominal(self):
        """The machinery at factor 1 gives back the file start's own inputs."""
        inputs, _ = self.inputs_at(dict(self.x0))
        if inputs is None:
            raise AssertionError("the file's own design is an invalid start")
        for var, want in self.assembled.inputs.items():
            if not same_value(inputs[var], want):
                raise AssertionError(f"start machinery at the file's design differs at {var}")


def solved(configuration, assembled, record=False):
    """`study.answer`, with the optimiser's model calls counted (`CALLS`), SLSQP's exit
    mode read off the run, and -- with `record` -- the points it called the model at
    kept for `InnerCounter` (`calls["recorded"]`)."""
    CALLS.clear()
    LAST.clear()
    RECORD["calls"], RECORD["on"] = [], record
    try:
        out = study.answer(configuration, assembled)
        code = exit_code(assembled)
    except Exception as e:  # noqa: BLE001 -- a failed solve is a result here
        out = {"status": f"error:{type(e).__name__}", "iterations": None, "objf": None,
               "max_eq": None, "min_ie": None, "seconds": None, "x": ()}
        code = None
    finally:
        RECORD["on"] = False
    calls = {k: CALLS[k] for k in ("values", "jacobians", "fused")}
    calls["model_s"] = CALLS["values_s"] + CALLS["jacobians_s"] + CALLS["fused_s"]
    calls["exit_code"] = code
    calls["exit_message"] = SLSQP_MESSAGES.get(code, "") if code is not None else ""
    calls["recorded"] = RECORD["calls"] if record else []
    calls["order"] = RECORD.get("order", ())
    return out, calls


def inner_columns(counter, inputs, calls) -> dict:
    """`counter.counts` at the points of `calls`, the recorded points dropped from the row."""
    recorded, order = calls.pop("recorded", []), calls.pop("order", ())
    return counter.counts(inputs, recorded, order) if counter is not None else {}


def success(configuration, out, f_ref, x_ref) -> bool:
    if out["status"] != "converged":
        return False
    if isinstance(configuration.problem, Optimisation):
        return f_ref is not None and out["objf"] is not None and \
            abs(out["objf"] - f_ref) <= F_TOLERANCE * max(abs(f_ref), 1e-12)
    x, r = np.asarray(out["x"], dtype=float), np.asarray(x_ref, dtype=float)
    return x.shape == r.shape and bool(np.all(np.abs(x - r) <= X_TOLERANCE * np.maximum(np.abs(r), 1e-12)))


def run_variant(configuration, v, cold, args) -> dict:
    name = configuration.name
    row = {"configuration": name, "label": v.label, "preset": v.preset, "optimiser": common.OPTIMISER}
    try:
        assembled, assemble_s = bench.timed(V.assemble_variant, configuration, v, cold, optimiser=SlsqpDriver)
    except Exception as e:  # noqa: BLE001 -- a refusal is a result here
        return {**row, "status": f"refused:{type(e).__name__}", "refused": str(e).splitlines()[0][:300]}
    began = time.perf_counter()
    first, _ = solved(configuration, assembled)              # compiles, then solves
    cold_s = time.perf_counter() - began
    warm_times = []
    for k in range(args.repeats):                            # the last one recorded: same solve
        began = time.perf_counter()
        warm, calls = solved(configuration, assembled, record=k == args.repeats - 1)
        warm_times.append(time.perf_counter() - began)
    row.update({
        "status": warm["status"], "iterations": warm["iterations"], "objf": warm["objf"],
        "max_eq": warm["max_eq"], "min_ie": warm["min_ie"],
        "assemble_s": assemble_s, "cold_s": cold_s, "warm_s": sorted(warm_times)[len(warm_times) // 2],
        "solve_s": warm["seconds"], "refused": "",
    })
    # The inner work, afterwards and untimed: a root find's Newton runs inside the
    # graph, so it makes no host-side calls to count at.
    counter = None
    if isinstance(configuration.problem, Optimisation):
        began = time.perf_counter()
        counter = InnerCounter(configuration, v, assembled)
        row.update({"inner": "|".join(counter.names()), "inner_sizes": "|".join(str(i.size) for i in counter.inner),
                    "top_models": counter.top_models, "nested_depth": counter.nested_depth})
    inner = inner_columns(counter, assembled.inputs, calls)      # takes the recorded points out of `calls`
    row.update({**calls, **inner})
    if counter is not None:
        row["count_s"] = time.perf_counter() - began
    print({k: row.get(k) for k in ("label", "status", "exit_message", "iterations", "objf", "cold_s", "warm_s",
                                    "inner_steps", "model_evals", "count_check")}, flush=True)

    starts = Starts(configuration, assembled, v)
    starts.check_nominal()
    f_ref = reference_objective(name)
    x_ref = warm["x"]
    records, began_all = [], time.perf_counter()
    for k, x in starts.draw(args.starts, args.spread):
        rec = {"configuration": name, "label": v.label, "start": k,
               **{f"x0:{u}": x[u] for u in starts.design}}
        inputs, _ = starts.inputs_at(x)
        if inputs is None:
            rec.update({"valid": False, "status": "invalid-start"})
        else:
            began = time.perf_counter()
            out, calls = solved(configuration, dataclasses.replace(assembled, inputs=inputs), record=True)
            wall = time.perf_counter() - began
            rec.update({
                "valid": True, "status": out["status"], "iterations": out["iterations"],
                "objf": out["objf"], "max_eq": out["max_eq"], "min_ie": out["min_ie"],
                "wall_s": wall,
                "converged": out["status"] == "converged",
                "success": success(configuration, out, f_ref, x_ref),
                **{f"x:{u}": xi for u, xi in zip(starts.design, out["x"] or ())},
            })
            inner = inner_columns(counter, inputs, calls)
            rec.update({**calls, **inner})
        records.append(rec)
        if (k + 1) % 10 == 0:
            ok = sum(bool(r.get("success")) for r in records)
            print(f"  {v.label}: {k + 1}/{args.starts} starts, {ok} successful, "
                  f"{time.perf_counter() - began_all:.0f}s", flush=True)
    common.write_all("solve.py", common.OUT / STARTS / name / f"{v.label}.csv", records)

    valid = [r for r in records if r["valid"]]
    its = [r["iterations"] for r in valid if r.get("success")]
    good = [r for r in valid if r.get("success")]
    for column in ("inner_steps", "model_evals"):
        found = [r[column] for r in good if r.get(column) not in (None, "")]
        row[f"starts_{column}_median"] = float(np.median(found)) if found else None
    checks = [r["count_check"] for r in valid if r.get("count_check") not in (None, "")]
    row["starts_count_check_max"] = max(checks) if checks else None
    exits: dict = {}
    for r in valid:
        if r.get("exit_code") is not None:
            exits[r["exit_code"]] = exits.get(r["exit_code"], 0) + 1
    row["starts_exit_codes"] = "|".join(f"{c}:{n}" for c, n in sorted(exits.items()))
    row.update({
        "starts": len(records), "starts_valid": len(valid),
        "starts_converged": sum(bool(r["converged"]) for r in valid),
        "starts_success": sum(bool(r["success"]) for r in valid),
        "success_rate": (sum(bool(r["success"]) for r in valid) / len(valid)) if valid else None,
        "starts_iterations_median": float(np.median(its)) if its else None,
        "starts_s": time.perf_counter() - began_all,
        "spread": args.spread, "seed": seed_of(name),
        "f_reference": f_ref, "f_tolerance": F_TOLERANCE,
    })
    return row


def main():
    args = common.arguments(__doc__, starts=True)
    for name in args.configurations:
        configuration = configurations.load(name)
        todo = [v for v in V.variants(configuration) if args.variants is None or v.label in args.variants]
        todo = [v for v in todo if args.force or not common.done(FOLDER, name, v.label)]
        if not todo:
            continue
        cold = study.cold_state(configuration)
        for v in todo:
            row = run_variant(configuration, v, cold, args)
            common.upsert("solve.py", FOLDER, row)


if __name__ == "__main__":
    main()
