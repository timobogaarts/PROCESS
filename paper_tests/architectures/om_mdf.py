"""Table 6 -- the same models under OpenMDAO, as MDF: one `ExplicitComponent` per
model (its `compute` the port's jax body, jitted; its partials by `jax.jacfwd`, jitted),
an `ImplicitComponent` per declared solve, `NonlinearBlockGS` over the coupling in the
models' own order, `DirectSolver` for the total derivatives, SLSQP outside.

What it measures is the cost of *a call per component*: the models' arithmetic is the
same compiled code as in the port, so the difference between this table and
`iteration.py` is the framework's dispatch -- data transfers, vector bookkeeping and
one Python call per component per sweep -- against one program for the whole block.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu $PY paper_tests/architectures/om_mdf.py --configurations stellarator_helias

Per machine: the components, the NLBGS sweeps one MDA takes, one converged MDA warm
(`run_model`), one total Jacobian warm (`compute_totals`), and the full SLSQP run.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import openmdao.api as om

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from cottax.pytree.problem import ConditionalNode, is_fixed_point, is_root_find  # noqa: E402

from functional_process.cottax.architectures import mdf, sand  # noqa: E402
from functional_process.cottax.architectures.evaluate import ground_truth, mda_env, without_excluded  # noqa: E402


def name(path) -> str:
    """A spelling OpenMDAO accepts: `^cond.constraints.c5` -> `cond__constraints__c5`."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", path.spelling.lstrip("^.")).replace("_", "__", 0) or "root"


# ---------------------------------------------------------------- the components

NONFINITE: set = set()
"""`(output, input)` partials that came back non-finite and were zeroed."""

DENSE: set = set()
"""Components whose colouring did not reproduce their dense Jacobian and keep dense partials."""

DROPPED: list = []
"""Per machine, the unread non-finite outputs held at zero."""

UNREAD: set = set()
"""Their names."""


def greedy_colouring(pattern: np.ndarray) -> np.ndarray:
    """A colour per column of a sparsity `pattern`, two columns sharing a row never the
    same colour: what one forward tangent may carry at once (distance-1 colouring)."""
    n_out, n_in = pattern.shape
    colour = -np.ones(n_in, dtype=int)
    rows_of = [np.flatnonzero(pattern[:, j]) for j in range(n_in)]
    taken: list[np.ndarray] = []          # per colour, the rows already used
    for j in range(n_in):
        for c, rows in enumerate(taken):
            if not rows[rows_of[j]].any():
                colour[j] = c
                rows[rows_of[j]] = True
                break
        else:
            used = np.zeros(n_out, dtype=bool)
            used[rows_of[j]] = True
            taken.append(used)
            colour[j] = len(taken) - 1
    return colour


class Model(om.ExplicitComponent):
    """One `ImplementedFunction`: `compute` is its body, jitted; its partials are
    **sparse and coloured** -- the sparsity pattern found once from the body's dense
    Jacobian at two nearby points, the columns coloured, one forward tangent per colour
    -- which is what OpenMDAO's own jax component does, and what a user of it would
    get. OpenMDAO's vectors are float64, so an integer input (a switch, an index) is
    cast back to its own dtype on the way in and is not differentiated."""

    def initialize(self):
        self.options.declare("node")
        self.options.declare("values")
        self.options.declare("varying")

    def setup(self):
        d, values, varying = self.options["node"], self.options["values"], self.options["varying"]
        self._dtype = {v: np.asarray(values[v]).dtype for v in d.reads}
        self._shape = {v: np.shape(values[v]) for v in d.reads}
        # Differentiated only with respect to what the design reaches: a partial with
        # respect to a constant (an impurity table, a cost coefficient) is never used by
        # a total derivative, and the port's block never forms one either.
        self._real = [j for j, v in enumerate(d.reads) if np.issubdtype(self._dtype[v], np.floating) and v in varying]
        self._ints = [j for j in range(len(d.reads)) if j not in self._real]
        for v in d.reads:
            self.add_input(name(v), val=np.asarray(values[v], dtype=float))
        for v in d.owns:
            self.add_output(name(v), val=0.0 if name(v) in UNREAD else np.asarray(values[v], dtype=float))
        fn, n_out = d.fn, len(d.owns)
        sizes = [int(np.size(values[d.reads[j]])) for j in self._real]
        splits = np.cumsum(sizes)[:-1]
        out_sizes = [int(np.size(values[o])) for o in d.owns]

        def outs(*args):
            out = fn(*args)
            return tuple(out) if n_out > 1 else (out,)

        def flat_outs(flat, ints):
            args = [None] * len(d.reads)
            for j, piece in zip(self._real, jnp.split(flat, splits), strict=True):
                args[j] = piece.reshape(self._shape[d.reads[j]])
            for j, value in zip(self._ints, ints, strict=True):
                args[j] = value
            return jnp.concatenate([jnp.ravel(o) for o in outs(*args)]) if n_out else jnp.zeros(0)

        self._value_fn = jax.jit(outs)
        self._flat = flat_outs
        self._jvp = self._dense = None
        if not self._real or not d.owns:
            return
        x0 = np.concatenate([np.ravel(np.asarray(values[d.reads[j]], dtype=float)) for j in self._real])
        ints0 = tuple(jnp.asarray(values[d.reads[j]]) for j in self._ints)
        dense = jax.jit(jax.jacfwd(flat_outs))
        J0 = np.asarray(dense(jnp.asarray(x0), ints0))
        # The pattern: the dense Jacobian at the start and at random points around it,
        # a non-finite entry counted as present. Then the colouring is *checked* against
        # the dense Jacobian at the start, and a component it does not reproduce keeps
        # dense partials -- so a partial is never wrong, only sometimes dear.
        rng = np.random.default_rng(0)
        pattern = (J0 != 0) | ~np.isfinite(J0)
        for _ in range(3):
            x = x0 * (1 + 0.3 * rng.uniform(-1, 1, x0.shape)) + 1e-6 * rng.uniform(-1, 1, x0.shape)
            J = np.asarray(dense(jnp.asarray(x), ints0))
            pattern |= (J != 0) | ~np.isfinite(J)
        self._colour = greedy_colouring(pattern)
        n_colours, n_rows = int(self._colour.max()) + 1, int(pattern.shape[0])
        if n_rows < n_colours:
            # Fewer outputs than colours: reverse mode, one cotangent per output row,
            # gives the dense Jacobian outright (`best_partial_deriv_direction`).
            self._colour = np.arange(x0.size)
            cotangents = jnp.eye(n_rows)

            def compressed(flat, ints):
                _, vjp = jax.vjp(lambda x: flat_outs(x, ints), flat)
                return jax.vmap(lambda c: vjp(c)[0])(cotangents).T          # inputs x outputs
        else:
            tangents = jnp.asarray(np.eye(n_colours)[self._colour].T)  # colours x inputs

            def compressed(flat, ints):
                return jax.vmap(lambda t: jax.jvp(lambda x: flat_outs(x, ints), (flat,), (t,))[1])(tangents)

        C0 = np.asarray(jax.jit(compressed)(jnp.asarray(x0), ints0))
        rebuilt = np.where(pattern, C0[self._colour, :].T, 0.0)
        finite = np.isfinite(J0)
        if not np.allclose(np.where(finite, rebuilt, 0), np.where(finite, J0, 0), rtol=1e-9, atol=1e-12):
            DENSE.add(self.name)
            self._jvp, self._dense = None, dense
            for o in d.owns:
                for j in self._real:
                    self.declare_partials(name(o), name(d.reads[j]))
            self._blocks = [(i, k, j) for i in range(n_out) for k, j in enumerate(self._real)]
            self._offsets = (np.concatenate([[0], np.cumsum(out_sizes)]), np.concatenate([[0], np.cumsum(sizes)]))
            return
        self._dense = None
        self._jvp = jax.jit(compressed)
        # the sparse blocks: per (output, input), which rows and columns, and where each
        # entry sits in the compressed jacobian
        self._blocks = []
        r0 = 0
        for i, o in enumerate(d.owns):
            c0 = 0
            for k, j in enumerate(self._real):
                r = d.reads[j]
                block = pattern[r0:r0 + out_sizes[i], c0:c0 + sizes[k]]
                rows, cols = np.nonzero(block)
                if rows.size:
                    self.declare_partials(name(o), name(r), rows=rows, cols=cols)
                    self._blocks.append((name(o), name(r), r0 + rows, self._colour[c0 + cols]))
                c0 += sizes[k]
            r0 += out_sizes[i]

    def _args(self, inputs):
        d = self.options["node"]
        return tuple(np.asarray(inputs[name(v)]).reshape(self._shape[v]).astype(self._dtype[v]) for v in d.reads)

    def compute(self, inputs, outputs):
        d = self.options["node"]
        for v, value in zip(d.owns, self._value_fn(*self._args(inputs)), strict=True):
            outputs[name(v)] = 0.0 if name(v) in UNREAD else np.asarray(value, dtype=float)

    def compute_partials(self, inputs, partials):
        if self._jvp is None and self._dense is None:
            return
        d = self.options["node"]
        args = self._args(inputs)
        flat = np.concatenate([np.ravel(args[j]) for j in self._real])
        ints = tuple(args[j] for j in self._ints)
        if self._dense is not None:
            J = np.asarray(self._dense(flat, ints))
            ro, co = self._offsets
            for i, k, j in self._blocks:
                block = J[ro[i]:ro[i + 1], co[k]:co[k + 1]]
                partials[name(d.owns[i]), name(d.reads[j])] = self._finite(name(d.owns[i]), name(d.reads[j]), block)
            return
        C = np.asarray(self._jvp(flat, ints))                       # colours x outputs
        for o, r, rows, colours in self._blocks:
            partials[o, r] = self._finite(o, r, C[colours, rows])

    @staticmethod
    def _finite(o, r, entries):
        if np.all(np.isfinite(entries)):
            return entries
        # A partial the design never reaches, at a point where the model's derivative
        # is undefined (a root at zero, a power at zero). The whole-block Jacobian never
        # forms it; an assembled one must hold something, and zero is what a path the
        # design does not reach means.
        NONFINITE.add((o, r))
        return np.nan_to_num(entries, nan=0.0, posinf=0.0, neginf=0.0)


class Solve(om.ImplicitComponent):
    """One declared statement: a fixed point `u = c` (residual `u - c`, one Picard step
    as its own solve, which is what a Gauss-Seidel sweep through it does) or a root
    find `r = 0` (residual `r`; solved by the Newton of the group around it)."""

    def initialize(self):
        self.options.declare("node")
        self.options.declare("values")

    def setup(self):
        d, values = self.options["node"], self.options["values"]
        for v in d.conditions:
            self.add_input(name(v), val=np.asarray(values[v], dtype=float))
        for v in d.unknowns:
            self.add_output(name(v), val=np.asarray(values[v], dtype=float))
        self.declare_partials("*", "*")

    def apply_nonlinear(self, inputs, outputs, residuals):
        d = self.options["node"]
        for u, c in zip(d.unknowns, d.conditions, strict=True):
            residuals[name(u)] = (outputs[name(u)] - inputs[name(c)]) if is_fixed_point(d) else inputs[name(c)]

    def solve_nonlinear(self, inputs, outputs):
        d = self.options["node"]
        if is_fixed_point(d):
            for u, c in zip(d.unknowns, d.conditions, strict=True):
                outputs[name(u)] = inputs[name(c)]

    def linearize(self, inputs, outputs, partials):
        d = self.options["node"]
        for u, c in zip(d.unknowns, d.conditions, strict=True):
            n = outputs[name(u)].size
            partials[name(u), name(c)] = -np.eye(n) if is_fixed_point(d) else np.eye(n)
            partials[name(u), name(u)] = np.eye(n) if is_fixed_point(d) else np.zeros((n, n))


# ---------------------------------------------------------------- the problem

def values_of(live, graph, report):
    """A value for every variable of `graph`: the MDA's at the file's design, the
    file's own, or the producer evaluated once (the constraint and objective nodes)."""
    cold = live.reference.cold
    stage = mda_env(live.reference, graph=live.machine_graph, data=cold)[1]
    memo = {}

    def value(v):
        if v in memo:
            return memo[v]
        if v in stage:
            out = np.asarray(stage[v])
        else:
            try:
                out = np.asarray(ground_truth(cold, v))
            except (AttributeError, KeyError):
                owner = graph.graph.owners.get(v)
                if owner is None:
                    out = np.asarray(0.0)
                else:
                    d = graph[owner]
                    produced = d.fn(*(jnp.asarray(value(r)) for r in d.reads))
                    produced = tuple(produced) if len(d.owns) > 1 else (produced,)
                    for o, p in zip(d.owns, produced, strict=True):
                        memo[o] = np.asarray(p)
                    out = memo[v]
        memo[v] = out
        return out

    return {v: value(v) for v in graph.graph.variables}


def problem_of(live):
    """The MDF problem: every model a component in binding order, each root find with
    its body in a Newton group, the whole under NLBGS, SLSQP over the design."""
    ref = live.reference
    raw = without_excluded(live.machine_graph)
    graph, _conditions, _n, report = mdf.mdf_graph(raw, ref.icc, ref.n_equality, ref.i_figure_merit, live.switch_values)
    values = values_of(live, graph, report)
    design = [sand.iteration_variable_path(i) for i in ref.ixc]
    # What moves: the design, every declared statement's unknowns (the assembled
    # system needs their columns whether or not the design reaches them), and all
    # they reach.
    moving = list(design) + [u for n in graph.nodes if isinstance(graph[n], ConditionalNode) for u in graph[n].unknowns]
    varying = set(moving) | {o for n in graph.graph.reach(moving) for o in graph[n].owns}
    # An output read by nothing and not finite at the file's design (a tokamak
    # quantity on a stellarator file) would stop NLBGS's convergence check, which
    # cottax never looks at; it is a number nobody uses, so it is held at zero.
    dead = [o for n in graph.nodes if not isinstance(graph[n], ConditionalNode)
            for o in graph[n].owns if not graph.graph.readers.get(o) and not np.all(np.isfinite(values[o]))]
    DROPPED.append(len(dead))
    UNREAD.clear(); UNREAD.update(name(o) for o in dead)

    prob = om.Problem()
    top = prob.model
    top.nonlinear_solver = om.NonlinearBlockGS(maxiter=500, atol=1e-10, rtol=1e-10, iprint=0)
    top.linear_solver = om.DirectSolver()
    driven_bodies = {graph.graph.owners[c]: n for n in graph.nodes if isinstance(graph[n], ConditionalNode) and is_root_find(graph[n]) for c in graph[n].conditions}
    placed = set()
    for n in graph.nodes:
        if n in placed:
            continue
        d = graph[n]
        if isinstance(d, ConditionalNode):
            if is_root_find(d):
                continue                                     # placed with its body below
            top.add_subsystem(name(n), Solve(node=d, values=values), promotes=["*"])
        elif n in driven_bodies:
            h = driven_bodies[n]
            group = om.Group()
            group.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=50, atol=1e-10, rtol=1e-10, iprint=0)
            group.linear_solver = om.DirectSolver()
            group.add_subsystem(name(n), Model(node=d, values=values, varying=varying), promotes=["*"])
            group.add_subsystem(name(h), Solve(node=graph[h], values=values), promotes=["*"])
            top.add_subsystem(f"solve__{name(n)}", group, promotes=["*"])
            placed.add(h)
        else:
            top.add_subsystem(name(n), Model(node=d, values=values, varying=varying), promotes=["*"])
        placed.add(n)

    bounds = {v: (lo, hi) for v, lo, hi in ref.bounds}
    for v in design:
        lo, hi = bounds[v]
        top.add_design_var(name(v), lower=lo, upper=hi, ref=abs(float(values[v])) or 1.0)
    top.add_objective(name(report["objective"]))
    for c in report["equalities"]:
        top.add_constraint(name(c), equals=0.0)
    for c in report["inequalities"]:
        top.add_constraint(name(c), upper=0.0)
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=bench.TOLERANCE, maxiter=300, disp=False)
    prob.setup()
    for v in graph.graph.boundary_inputs:
        prob.set_val(name(v), np.asarray(values[v], dtype=float))
    return prob, graph, report


def main():
    args = bench.arguments(__doc__, optimiser=False)
    rows = []
    for name_ in args.configurations:
        live = bench.open_session(name_, args)
        if live.root_find:
            continue
        (prob, graph, report), setup_s = bench.timed(problem_of, live)
        _, cold_s = bench.timed(prob.run_model)
        sweeps = prob.model.nonlinear_solver._iter_count
        model_s = bench.median_seconds(prob.run_model, args.repeats)
        totals_s = bench.median_seconds(lambda: prob.compute_totals(), args.repeats)
        began = time.perf_counter()
        result = prob.run_driver()
        driver_s = time.perf_counter() - began
        scalar = lambda v: float(np.asarray(prob.get_val(name(v))).reshape(-1)[0])  # noqa: E731
        objf = scalar(report["objective"])
        eqs = [abs(scalar(c)) for c in report["equalities"]]
        ies = [-scalar(c) for c in report["inequalities"]]
        rows.append({
            "configuration": name_, "arm": "MDF (OpenMDAO)",
            "components": len(graph.nodes), "design": len(live.reference.ixc),
            "sweeps": sweeps, "setup_s": setup_s, "cold_model_s": cold_s,
            "model_ms": 1e3 * model_s, "totals_ms": 1e3 * totals_s,
            "iterations": prob.driver.iter_count, "status": "converged" if result.success else "failed",
            "driver_s": driver_s, "objf": objf,
            "max_eq": max(eqs) if eqs else 0.0, "min_ie": min(ies) if ies else float("nan"),
            "zeroed_partials": len(NONFINITE), "dense_components": len(DENSE), "zeroed_sinks": DROPPED[-1],
        })
        NONFINITE.clear(); DENSE.clear()
        print(rows[-1])
    if rows:
        bench.write("om_mdf.py", rows, "openmdao_mdf")


if __name__ == "__main__":
    main()
