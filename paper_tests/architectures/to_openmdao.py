"""A cottax optimiser block as an OpenMDAO problem, **on the block's own partition**.

A stand-in for a future cottax feature ("export to OpenMDAO"): it reads nothing but a
`cottax.interfaces.Drive` -- the step a `Schedule` executes for an `Optimise` statement
-- and the values at its inputs, and it builds nothing the schedule does not already
say. So what differs between the port running that `Drive` and OpenMDAO running what
this returns is OpenMDAO's execution (a Python call per component, host vectors, its
solvers' own stopping tests) and nothing else: the bodies are the same functions, the
blocks the same blocks, the problem the same problem.

The mapping, step by step:

| the schedule                          | OpenMDAO                                             |
|---------------------------------------|------------------------------------------------------|
| the `Optimise` `Drive`                | the `Problem`: its unknowns the design variables, its objectives and relations the objective and constraints, `ScipyOptimizeDriver` |
| its body, a `Schedule`                | the model: `NonlinearRunOnce` + `LinearRunOnce`, one subsystem per step in the schedule's order |
| a `Call` step                         | one `Model` (`ExplicitComponent`) per node, in the block's topological order |
| a `Drive` of a fixed point            | an `om.Group`: a `Solve` for the statement, then its body; `NonlinearBlockGS` + `DirectSolver` |
| a `Drive` of a root find              | the same group under `NewtonSolver` + `DirectSolver` |
| the `Drive`'s context                 | unconnected inputs (OpenMDAO's auto-IVC), set to the values handed in, never re-run |

A fixed point's `Solve` sits first in its group and sets `u <- rhs`, so one Gauss-Seidel
sweep of the group is one Picard iterate of the port's `PicardDriver` (`u_{k+1} =
g(u_k)`, the body run once in its order). The tolerances are the port driver's own
(`atol`, `rtol`, its step budget); the *norms* are OpenMDAO's (the 2-norm of the group's
change, relative to its first sweep's), which is part of what is being compared.

**Scaling is the optimiser driver's.** `scaled` (the default) is `x / x_start` per
design coordinate, `ref=x_start` here -- the port's `drivers.design_scale`, floor and
all; `condition_scale` multiplies the named condition, `scaler=` here; `bounds` are the
driver's own. The SQP is `ScipyOptimizeDriver(optimizer="SLSQP")` at the driver's
`tolerance` and `max_iter`, so iteration counts compare with the port's `SlsqpDriver`.

What it needs of cottax, beyond `cottax.interfaces`: `Start`
(`cottax.execution.drivers.kinds`, the name a driver reads its starting value under) and
three *port* driver fields read by duck type -- `bounds`, `scaled`, `condition_scale`,
the optimiser's settings, which cottax's own `SLSQPDriver` does not state the same way.
And the solver settings of a nested driver (`atol`, `rtol`, `max_steps`), read the same
way. An export inside cottax would want those as declared driver data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import openmdao.api as om
from cottax.execution.drivers.kinds import Start
from cottax.interfaces import (
    Call,
    Drive,
    Le,
    PathMap,
    Schedule,
    is_equalities,
    is_fixed_point,
    is_optimise,
)

# ---------------------------------------------------------------- names


def name(path) -> str:
    """A spelling OpenMDAO accepts, one-to-one on the port's names: `.` -> `__`, a
    mint's `^` -> `M_`, anything else not an identifier character -> `_`.
    `.constraints.c5` -> `constraints__c5`, `^hat.physics.x` -> `M_hat__physics__x`."""
    s = path.spelling.replace("^", "M_").replace(".", "__")
    return re.sub(r"[^0-9a-zA-Z_]", "_", s).lstrip("_") or "root"


# ---------------------------------------------------------------- the record


@dataclass
class Conversion:
    """What `to_openmdao` built, and what it had to do to build it."""

    problem: om.Problem
    drive: Drive
    values: dict
    """Every variable of the block at the start: the inputs handed in and the body's
    outputs from one run of it."""
    design: tuple = ()
    objective: object = None
    equalities: tuple = ()
    inequalities: tuple = ()
    groups: list = field(default_factory=list)
    """`(OpenMDAO pathname, "fixed point" | "root find", problem)` per driven block,
    outermost first."""
    starts: dict = field(default_factory=dict)
    """`{unknown: start}` for every nested block: where the port's driver starts it."""
    components: int = 0
    nonfinite: set = field(default_factory=set)
    """`(output, input)` partials that came back non-finite and were zeroed."""
    dense: set = field(default_factory=set)
    """Components whose colouring did not reproduce their dense Jacobian."""
    held: set = field(default_factory=set)
    """Outputs read by nothing and non-finite at the start, held at zero."""

    def sweeps(self) -> dict:
        """`{group: iterations}` of each block's solver in the last run."""
        return {path: self.problem.model._get_subsystem(path).nonlinear_solver._iter_count
                for path, _kind, _p in self.groups}

    def restart(self) -> None:
        """Every nested block's unknowns back at the port driver's start, so the next
        `run_model` iterates from where the port's evaluation does, not warm."""
        for u, value in self.starts.items():
            self.problem.set_val(name(u), np.asarray(value, dtype=float))


# ---------------------------------------------------------------- the components


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


_CALLS = ("jaxpr", "call_jaxpr", "fun_jaxpr")


def _flow(jaxpr, given):
    """Per outvar of `jaxpr`, the set of argument indices it depends on, `given` the
    sets of its invars: a union over every equation, into sub-jaxprs of calls (`pjit`,
    `custom_jvp_call`, ...) where their arity lines up, and over the whole equation
    otherwise (a `cond`, a `while`: every output from every input) -- never less than
    the truth, sometimes more."""
    env = dict(zip(jaxpr.invars, given, strict=True))
    for v in jaxpr.constvars:
        env[v] = frozenset()

    def read(v):
        return env.get(v, frozenset()) if not hasattr(v, "val") else frozenset()

    for eqn in jaxpr.eqns:
        ins = [read(v) for v in eqn.invars]
        sub = next((eqn.params[k] for k in _CALLS if k in eqn.params), None)
        sub = getattr(sub, "jaxpr", sub)
        if sub is not None and hasattr(sub, "eqns") and len(sub.invars) == len(ins) \
                and len(sub.outvars) == len(eqn.outvars):
            outs = _flow(sub, ins)
        else:
            union = frozenset().union(*ins)
            outs = [union] * len(eqn.outvars)
        env.update(zip(eqn.outvars, outs, strict=True))
    return [read(v) for v in jaxpr.outvars]


def structural_dependence(fn, args) -> list[frozenset]:
    """Per output of `fn(*args)` (a tuple), which argument indices it can depend on."""
    closed = jax.make_jaxpr(fn)(*args)
    flat = _flow(closed.jaxpr, [frozenset({i}) for i in range(len(args))])
    return flat


class Model(om.ExplicitComponent):
    """One node with a body: `compute` is its `implementation`, jitted; its partials are
    **sparse and coloured** -- the sparsity pattern found once from the body's dense
    Jacobian at the start and three points around it, the columns coloured, one forward
    tangent per colour (reverse, one cotangent per row, where there are fewer rows than
    colours) -- which is what OpenMDAO's own jax component does. The colouring is
    checked against the dense Jacobian at the start and a component it does not
    reproduce keeps dense partials. OpenMDAO's vectors are float64, so an integer input
    (a switch, an index) is cast back to its own dtype on the way in and is not
    differentiated; nor is anything not in `varying` (a constant: its partial is never
    part of a total derivative)."""

    def initialize(self):
        self.options.declare("node")
        self.options.declare("values")
        self.options.declare("varying")
        self.options.declare("record")

    def setup(self):
        d, values, varying = self.options["node"], self.options["values"], self.options["varying"]
        record = self.options["record"]
        self._dtype = {v: np.asarray(values[v]).dtype for v in d.reads}
        self._shape = {v: np.shape(values[v]) for v in d.reads}
        self._real = [j for j, v in enumerate(d.reads) if np.issubdtype(self._dtype[v], np.floating) and v in varying]
        self._ints = [j for j in range(len(d.reads)) if j not in self._real]
        for v in d.reads:
            self.add_input(name(v), val=np.asarray(values[v], dtype=float))
        for v in d.owns:
            self.add_output(name(v), val=0.0 if v in record.held else np.asarray(values[v], dtype=float))
        fn, n_out = d.implementation, len(d.owns)
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
        rng = np.random.default_rng(0)
        pattern = (J0 != 0) | ~np.isfinite(J0)
        for _ in range(3):
            x = x0 * (1 + 0.3 * rng.uniform(-1, 1, x0.shape)) + 1e-6 * rng.uniform(-1, 1, x0.shape)
            J = np.asarray(dense(jnp.asarray(x), ints0))
            pattern |= (J != 0) | ~np.isfinite(J)
        # Sampling alone is not enough: an entry that is zero at every sampled point
        # and not elsewhere (an interpolation whose bracket moves with the design --
        # `stellarator.coils.intersect` reads one element of a table, a different one
        # per point) would be missed, and its partial silently zero. So every
        # (output, input) pair the body *structurally* connects is a dense block too.
        depends = structural_dependence(outs, [jnp.asarray(values[r]) for r in d.reads])
        r0 = 0
        for i in range(n_out):
            c0 = 0
            for k, j in enumerate(self._real):
                if j in depends[i]:
                    pattern[r0:r0 + out_sizes[i], c0:c0 + sizes[k]] = True
                c0 += sizes[k]
            r0 += out_sizes[i]
        self._pattern = pattern
        self._colour = greedy_colouring(pattern)
        n_colours, n_rows = int(self._colour.max()) + 1, int(pattern.shape[0])
        if n_rows < n_colours:
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
            record.dense.add(self.pathname)
            self._dense = dense
            for o in d.owns:
                for j in self._real:
                    self.declare_partials(name(o), name(d.reads[j]))
            self._blocks = [(i, k, j) for i in range(n_out) for k, j in enumerate(self._real)]
            self._offsets = (np.concatenate([[0], np.cumsum(out_sizes)]), np.concatenate([[0], np.cumsum(sizes)]))
            return
        self._jvp = jax.jit(compressed)
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
        d, held = self.options["node"], self.options["record"].held
        for v, value in zip(d.owns, self._value_fn(*self._args(inputs)), strict=True):
            outputs[name(v)] = 0.0 if v in held else np.asarray(value, dtype=float)

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

    def _finite(self, o, r, entries):
        if np.all(np.isfinite(entries)):
            return entries
        # A partial at a point where the model's derivative is undefined (a root at
        # zero, a power at zero), on a path the whole-block Jacobian never forms. An
        # assembled Jacobian must hold something, and zero is what that path means.
        self.options["record"].nonfinite.add((o, r))
        return np.nan_to_num(entries, nan=0.0, posinf=0.0, neginf=0.0)


def _sides(statement):
    """`[(lhs, rhs)]` per relation, `None` for a side not written."""
    return [(r.lhs, r.rhs) for r in statement.relations]


class Solve(om.ImplicitComponent):
    """One driven statement's unknowns, and its relations as residuals: relation `i`,
    `lhs - rhs` (a side not written is zero), is unknown `i`'s residual -- the port's
    `GapDriver` reading, positionally. A fixed point (`u = g`, its left side the
    unknown) also *solves* itself, `u <- g`: the port's `IterateDriver` reading, so a
    Gauss-Seidel sweep through it is one Picard iterate."""

    def initialize(self):
        self.options.declare("statement")
        self.options.declare("values")
        self.options.declare("start")

    def setup(self):
        st, values = self.options["statement"], self.options["values"]
        self._unknowns = tuple(st.unknowns)
        self._sides = _sides(st)
        if len(self._sides) != len(self._unknowns):
            raise NotImplementedError(f"{st!r}: {len(self._sides)} relations for {len(self._unknowns)} unknowns")
        self._fixed = is_fixed_point(st)
        if self._fixed and any(lhs != u for (lhs, _), u in zip(self._sides, self._unknowns, strict=True)):
            raise NotImplementedError(f"{st!r}: a fixed point whose left sides are not its unknowns")
        own = set(self._unknowns)
        self._inputs = tuple(dict.fromkeys(v for pair in self._sides for v in pair if v is not None and v not in own))
        for v in self._inputs:
            self.add_input(name(v), val=np.asarray(values[v], dtype=float))
        for u, s in zip(self._unknowns, self.options["start"], strict=True):
            self.add_output(name(u), val=np.asarray(s, dtype=float))
        self._sign = {}
        for u, (lhs, rhs) in zip(self._unknowns, self._sides, strict=True):
            n = int(np.size(values[u]))
            for v, sign in ((lhs, 1.0), (rhs, -1.0)):
                if v is None:
                    continue
                if int(np.size(values[v])) != n:
                    raise NotImplementedError(f"{u.spelling}: a side of size {np.size(values[v])} for an unknown of size {n}")
                self._sign[u, v] = self._sign.get((u, v), 0.0) + sign
        for (u, v), _ in self._sign.items():
            n = int(np.size(values[u]))
            self.declare_partials(name(u), name(v), rows=np.arange(n), cols=np.arange(n))

    def _value(self, v, inputs, outputs):
        if v is None:
            return 0.0
        return outputs[name(v)] if v in set(self._unknowns) else inputs[name(v)]

    def apply_nonlinear(self, inputs, outputs, residuals):
        for u, (lhs, rhs) in zip(self._unknowns, self._sides, strict=True):
            residuals[name(u)] = self._value(lhs, inputs, outputs) - self._value(rhs, inputs, outputs)

    def solve_nonlinear(self, inputs, outputs):
        if self._fixed:
            for u, (_lhs, rhs) in zip(self._unknowns, self._sides, strict=True):
                outputs[name(u)] = inputs[name(rhs)]

    def linearize(self, inputs, outputs, partials):
        for (u, v), sign in self._sign.items():
            partials[name(u), name(v)] = np.full(int(np.size(outputs[name(u)])), sign)


class Gap(om.ExplicitComponent):
    """A relation stated between two computed sides at the optimiser, `lhs - rhs`: what
    the port's `Gaps` reading hands its SQP, as one output the problem can constrain."""

    def initialize(self):
        self.options.declare("relation")
        self.options.declare("values")
        self.options.declare("out")

    def setup(self):
        (lhs, rhs), values = self.options["relation"], self.options["values"]
        self._terms = [(v, s) for v, s in ((lhs, 1.0), (rhs, -1.0)) if v is not None]
        n = int(np.size(values[self._terms[0][0]]))
        for v, _ in self._terms:
            self.add_input(name(v), val=np.asarray(values[v], dtype=float))
        self.add_output(self.options["out"], val=np.zeros(np.shape(values[self._terms[0][0]])))
        for v, s in self._terms:
            self.declare_partials(self.options["out"], name(v), rows=np.arange(n), cols=np.arange(n), val=s)

    def compute(self, inputs, outputs):
        outputs[self.options["out"]] = sum(s * inputs[name(v)] for v, s in self._terms)


# ---------------------------------------------------------------- the walk


def _driver_data(drive, v) -> bool:
    """Whether `v` is read only as some nested driver's data (a `Start`)."""
    return all(n in _problems(drive) for n in drive.subgraph.graph.readers.get(v, ()))


def _problems(drive):
    return {step.problem for step in _nested(drive.body)} | {drive.problem}


def _nested(schedule):
    """Every `Drive` inside `schedule`, at any depth, outermost first."""
    for step in getattr(schedule, "steps", ()):
        if isinstance(step, Drive):
            yield step
            yield from _nested(step.body)


def _solvers(step, once, inner):
    """The nonlinear and linear solver of a driven block, by what its statement is."""
    if once:
        return om.NonlinearRunOnce(), om.DirectSolver()
    d = step.driver
    atol, rtol = getattr(d, "atol", 1e-10), getattr(d, "rtol", 1e-10)
    maxiter = int(getattr(d, "max_steps", 256))
    if inner is not None:
        atol, rtol, maxiter = inner(step) if callable(inner) else inner
    if is_fixed_point(step.statement):
        nl = om.NonlinearBlockGS(maxiter=maxiter, atol=atol, rtol=rtol, iprint=-1, err_on_non_converge=False)
    elif is_equalities(step.statement):
        nl = om.NewtonSolver(solve_subsystems=True, max_sub_solves=maxiter, maxiter=maxiter,
                             atol=atol, rtol=rtol, iprint=-1, err_on_non_converge=False)
    else:
        raise NotImplementedError(f"{step.problem.spelling}: no OpenMDAO solver for {step.statement!r}")
    return nl, om.DirectSolver()


def _add_steps(group, prefix, body, conv, varying, once, inner, env):
    steps = body.steps if isinstance(body, Schedule) else (body,)
    for step in steps:
        if isinstance(step, Drive):
            sub = om.Group()
            sub.nonlinear_solver, sub.linear_solver = _solvers(step, once, inner)
            start = step.role_data(env).get(Start) if Start in step.driver.requires else None
            if start is None:
                start = tuple(env[u] for u in step.unknowns)
            conv.starts.update(zip(step.unknowns, start, strict=True))
            sub.add_subsystem(name(step.problem), Solve(statement=step.statement, values=conv.values, start=start),
                              promotes=["*"])
            path = f"{prefix}solve__{name(step.problem)}"
            kind = "fixed point" if is_fixed_point(step.statement) else "root find"
            conv.groups.append((path, kind, step.problem))
            _add_steps(sub, f"{path}.", step.body, conv, varying, once, inner, env)
            group.add_subsystem(path.rsplit(".", 1)[-1], sub, promotes=["*"])
        elif isinstance(step, Call):
            for n in step.subgraph.graph.topological_order:
                group.add_subsystem(name(n), Model(node=step.subgraph[n], values=conv.values, varying=varying,
                                                   record=conv), promotes=["*"])
                conv.components += 1
        else:
            raise NotImplementedError(f"a {type(step).__name__} step")


def to_openmdao(drive, values, *, bounds=None, once=False, inner=None, optimizer="SLSQP",
                tolerance=None, max_iter=None, hold_nonfinite_sinks=True) -> Conversion:
    """`drive` -- the `Drive` of an `Optimise` statement, as a `Schedule` executes it --
    as an OpenMDAO `Problem` on exactly its partition, set up and seeded.

    `values` holds a value for every input of `drive` (`drive.inputs`: its context and
    its driver's data, the starting design among it). The body is run once from there
    (`drive.body`, jitted), which gives every other variable its starting value, its
    shape and its dtype.

    `bounds`: `((VarPath, lower, upper), ...)`, default the driver's own. `once`: every
    solver `NonlinearRunOnce` -- one pass over the components, no block converged.
    `inner`: `(atol, rtol, maxiter)` for every block instead of its driver's own.
    `tolerance` / `max_iter`: the SQP's, default the driver's (`1e-8`, `100`).
    `hold_nonfinite_sinks`: an output nothing reads that is not finite at the start (a
    tokamak quantity on a stellarator file) is held at zero, since OpenMDAO's
    convergence norms would read it and the port's never do.
    """
    if not is_optimise(drive.statement):
        raise ValueError(f"{drive.problem.spelling} states {drive.statement!r}, not an Optimise")
    env = {v: jnp.asarray(values[v]) for v in drive.inputs}
    start = drive.role_data(env).get(Start) if Start in drive.driver.requires else None
    if start is None:
        start = tuple(env[u] for u in drive.unknowns)
    given = PathMap([*((v, env[v]) for v in drive.context), *zip(drive.unknowns, start, strict=True)])
    ran = jax.jit(drive.body.run)(given) if isinstance(drive.body, Schedule) else drive.body(dict(given.items()))
    env.update({v: jnp.asarray(x) for v, x in dict(ran.items()).items()})
    conv = Conversion(problem=om.Problem(), drive=drive, values={v: np.asarray(x) for v, x in env.items()})

    graph = drive.subgraph.graph
    spelt: dict = {}
    for v in graph.variables:
        if spelt.setdefault(name(v), v) != v:
            raise ValueError(f"{v.spelling} and {spelt[name(v)].spelling} spell the same OpenMDAO name {name(v)}")

    design = tuple(drive.unknowns)
    nested = [u for step in _nested(drive.body) for u in step.unknowns]
    varying = set(design) | set(nested) | {o for n in graph.reach([*design, *nested]) for o in drive.subgraph[n].owns}
    if hold_nonfinite_sinks:
        conv.held = {o for o in graph.owned_variables
                     if not graph.readers.get(o) and o in conv.values and not np.all(np.isfinite(conv.values[o]))}
        for o in conv.held:
            conv.values[o] = np.zeros_like(np.asarray(conv.values[o], dtype=float))

    prob = conv.problem
    top = prob.model
    top.nonlinear_solver, top.linear_solver = om.NonlinearRunOnce(), om.LinearRunOnce()
    _add_steps(top, "", drive.body, conv, varying, once, inner, env)

    # The problem, as the driver reads it.
    d = drive.driver
    limits = {v: (lo, hi) for v, lo, hi in (getattr(d, "bounds", ()) if bounds is None else bounds)}
    scaled = getattr(d, "scaled", True)
    by_place = {v: float(f) for v, f in getattr(d, "condition_scale", ())}
    for u, first in zip(design, start, strict=True):
        x0 = np.asarray(first, dtype=float)
        ref = np.where(np.abs(x0) > 1e-12, x0, 1.0) if scaled else np.ones_like(x0)   # drivers.design_scale
        if np.any(ref < 0):
            raise NotImplementedError(f"{u.spelling} starts negative: the port swaps its bounds, OpenMDAO would not")
        lo, hi = limits.get(u, (None, None))
        top.add_design_var(name(u), lower=lo, upper=hi, ref=ref)
    st = drive.statement
    (objective,) = st.objectives
    top.add_objective(name(objective), scaler=by_place.get(objective))
    equalities, inequalities = [], []
    for i, r in enumerate(st.relations):
        place = r.lhs if r.lhs is not None else r.rhs
        if r.rhs is None:
            response = name(r.lhs)
        else:
            response = f"gap{i}__{name(place)}"
            top.add_subsystem(f"gap{i}", Gap(relation=(r.lhs, r.rhs), values=conv.values, out=response),
                              promotes=["*"])
        kind = {"upper": 0.0} if r.op is Le else {"equals": 0.0}
        top.add_constraint(response, scaler=by_place.get(place), **kind)
        (inequalities if "upper" in kind else equalities).append(response)
    conv.design, conv.objective = tuple(name(u) for u in design), name(objective)
    conv.equalities, conv.inequalities = tuple(equalities), tuple(inequalities)
    prob.driver = om.ScipyOptimizeDriver(
        optimizer=optimizer,
        tol=getattr(d, "tolerance", 1e-8) if tolerance is None else tolerance,
        maxiter=getattr(d, "max_iter", 100) if max_iter is None else max_iter,
        disp=False,
    )
    prob.setup()
    # The context OpenMDAO reads -- a nested driver's own start (`^guess.*`) is driver
    # data the port reads and OpenMDAO does not: its solvers start from the outputs.
    read = {v for n in graph.nodes for v in drive.subgraph[n].reads}
    for v in (*drive.context, *design):
        if v in read and not _driver_data(drive, v):
            prob.set_val(name(v), np.asarray(conv.values[v], dtype=float))
    return conv
