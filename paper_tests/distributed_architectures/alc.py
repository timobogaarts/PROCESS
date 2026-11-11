"""Augmented Lagrangian coordination (ALC) of build, operation and economics.

Tosserams, Etman & Rooda's ALC (2008), in its non-hierarchical form, on Helias. The
file's one problem (min COE over eight variables, 2 equalities, 12 inequalities) is
split by design context into three disciplines (`partition.disciplines`):

- **Build** sizes the machine: `b_t`, `rmajor`, `tdmptf`, `fcutfsu`, the coil limits
  (c32, c34, c35, c65, c82, c83), and no objective of its own.
- **Operation** finds the operating point: `te`, `dene`, `hfact`, `f_alpha`, its own
  copy of the six build quantities the plasma reads, the physics limits (c2, c8, c17,
  c18, c24, c62, c67), and no objective of its own.
- **Economics** prices the plant: its own copy of the 25 build and 12 operation
  quantities the plant models read, c16 (net electric), and COE. Tosserams would call
  it the master problem; it is solved in turn with the others, not above them.

Each copy is held equal to what it copies by a consistency constraint `c = (copy -
value) / |reference|`, 43 in all, none redundant: a quantity both operation and
economics read is checked against build twice, never operation against economics.
Every consistency constraint is relaxed by `phi(c) = v'c + |w o c|^2`. Each discipline
minimises its share of the objective (COE for economics, nothing for the others) plus
the `phi` terms of the constraints touching its own variables, subject to its own
constraints. At `c = 0` the three problems' optimality conditions combine into the
file's problem's, with `v` as the multipliers of its consistency constraints; that holds
without convexity. Whether the iteration gets there is proven only for convex problems:
here it is empirical.

**The graph.** A recipe on the AAO base (`recipe`): the three disciplines, and over them
the coordination problem, `Coordination`. It owns the copies the upstream disciplines
read of the downstream ones, the multipliers `v`, the weights `w` and the last COE. It
holds `copy = value`, `merit_seen = merit` and the consistency `c = 0`. The three
disciplines are nested in it.

**The driver.** `Coordinator`, ALC's own (contract `AugmentedLagrangian`). Like any
driver it reads the model's outputs each iteration (one sweep of the disciplines) and
decides the next point from them: the copies move to what they copy; `v` and `w` follow
the method of multipliers on `c`, once the copies settle. Each nested SLSQP starts from
the same start in every sweep, not the last answer (FR-1).

**Three choices depart from Tosserams' stated algorithm** (`alc_report.md`,
"Departures"):
1. the inner loop ends when the copies settle, not when the relaxed objective does,
   with a fixed tolerance;
2. the stopping test adds the multiplier step, the copies' movement and COE's to the
   paper's two tests on `c`;
3. `beta = 1`: the weights are never raised, below the paper's `beta > 1`.

What else convergence took, run by run, is in `alc_report.md`: SLSQP instead of VMCON,
each objective divided by its penalty curvature, and no weight raised below tolerance.

No derivative is taken through any optimiser: each subproblem differentiates its own
models in its own variables, the others' variables fixed, and the coordination takes
none.

    python paper_tests/distributed_architectures/alc.py [--max-sweeps 3000] [--initial-weight 1]
        [--beta 2.2] [--inner-tolerance 1e-3] [--no-aao]
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.tree_util import GetAttrKey

import cottax.assemblies.jax as cx
from cottax.assemblies.jax import Scope, V
from cottax.core import Function, Mint, Plan
from cottax.core.driven import Start, Steps
from cottax.core.maths.plan import Combine, Cut, Delete, Determine, Nest, Undetermine
from cottax.core.realisation import ImplementedFunction

from cottax.core import is_problem
from cottax.core.driven import DriverContract
from functional_process.cottax.architectures.drivers import SlsqpDriver
from functional_process.cottax.architectures.reading import PortDriver, Reading

import partition as C

BUILD, OPERATION, ECONOMICS = C.BUILD, C.OPERATION, C.ECONOMICS

# ---------------------------------------------------------------- the names ALC adds

ALC = Scope(V.alc, GetAttrKey)

OBJECTIVE = {k: ALC.place(f"objective_{k}") for k in C.DISCIPLINES}
"""What each discipline minimises: its `phi` terms, plus COE for economics."""

MULTIPLIER = ALC.place("multiplier")
WEIGHT = ALC.place("weight")
INCONSISTENCY = ALC.place("inconsistency")
"""`c`, one entry per consistency constraint, at the end of a sweep."""
SEEN_MERIT = ALC.place("merit")
"""COE at the end of a sweep, carried by the coordination: its stopping test reads it."""


# WORKAROUND(FR-5): a mint per discipline for its own copies, written by hand.
class OperationCopy(Mint):
    """Operation's own copy of a quantity build computes."""


class EconomicsCopy(Mint):
    """Economics' own copy of a quantity build or operation computes."""


COPY = {OPERATION: OperationCopy(), ECONOMICS: EconomicsCopy()}

PROBLEM = {BUILD: C.node("OptBuild"), OPERATION: C.node("OptOperation"),
           ECONOMICS: C.node("OptEconomics")}
PENALTY = {k: C.node(f"objective_{k}") for k in C.DISCIPLINES}
CONSISTENCY = C.node("consistency")
COORDINATION = C.node("Coordination")

BETA = 1.0
"""How much a weight grows where its inconsistency did not fall enough. 1: the weights
stay at `INITIAL_WEIGHT` and the multipliers alone close the gaps. Tosserams' 2.2 raised
them to 23 in the first 300 sweeps, after which each sweep moved the design along the
prices by ~1 / (2 w^2) and it crept: +1.7e-5 on COE after 3000 sweeps, `f_alpha` 31 %
off (`alc_report.md`, R6 against R7)."""
GAMMA = 0.4
"""How much an inconsistency must fall per sweep for its weight to stay."""
INITIAL_WEIGHT = 1.0
"""Every weight at the start, on the scaled inconsistencies: a 1 % inconsistency costs
1e-4, against COE ~ 1.2 and its sensitivities of order 0.1 per 1 %. Stiffer starts
(5.6) settle the copies sooner but leave the prices further off."""
TOLERANCE = 1e-4
"""`max |c|` at which the coordination stops: every copy within 0.01 % of its
reference. Below it no weight is raised."""
INNER_TOLERANCE = 1e-3
"""The largest relative change of a copy over a sweep at which the copies count as
settled and the multipliers are updated; `inf` for alternating direction."""
DUAL_TOLERANCE = 1e-3
"""The largest multiplier step `2 w o w o c` at which the prices count as settled: in
COE per unit relative inconsistency, so 0.1 % of COE ~ 1.2."""
MAX_INNER = 25
"""Sweeps at fixed multipliers before an update is taken regardless."""
MAX_SWEEPS = 3000


# ---------------------------------------------------------------- the consistency constraints


@dataclasses.dataclass(frozen=True)
class Consistency:
    """The 43 consistency constraints, in `Disciplines.links` order: each a copy, what it
    copies, and the reference magnitude scaling it."""

    copies: tuple
    values: tuple
    reference: np.ndarray

    def __len__(self) -> int:
        return len(self.copies)

    def touching(self, discipline, links) -> np.ndarray:
        """The constraints with `discipline` at either end: its `phi` terms."""
        return np.asarray([j for j, k in enumerate(links)
                           if discipline in (k.source, k.holder)], dtype=int)

    def pairs(self, index) -> tuple:
        return tuple(p for j in index for p in (self.copies[j], self.values[j]))


def consistency(d: C.Disciplines, env, floor=1e-6) -> Consistency:
    copies = tuple(COPY[k.holder](k.place) for k in d.links)
    values = tuple(k.place for k in d.links)
    reference = np.asarray([max(abs(float(np.asarray(env[k.place]))), floor) for k in d.links])
    return Consistency(copies, values, reference)


# WORKAROUND(FR-10): plain jax bodies for the inserted functions.
def _mismatch(flat, scale):
    copy = jnp.stack([jnp.asarray(x, dtype=float) for x in flat[0::2]])
    value = jnp.stack([jnp.asarray(x, dtype=float) for x in flat[1::2]])
    return (copy - value) * scale


def penalty_body(index, reference, plus_merit=False):
    """`phi` over the constraints at `index` (and COE before them for economics), divided
    by `2 mean(w^2)` over them: `body(v, w, copy_0, value_0, ...)`, or `body(merit, v, w,
    ...)`.

    The division changes no subproblem's minimiser (`w` is fixed during its solve). It
    puts the penalty's curvature near 1 in the SQP's scaled coordinates (`x / x_start`,
    and `c` relative to a reference near the start), which an SQP started from the
    identity Hessian needs: unscaled, the first step is `2 w^2 c` start-magnitudes long,
    takes the unbounded copies through zero, and lands on a non-finite model."""
    idx, scale = np.asarray(index), 1.0 / np.asarray(reference)[np.asarray(index)]

    def phi(v, w, flat):
        c = _mismatch(flat, scale)
        return jnp.sum(v[idx] * c) + jnp.sum((w[idx] * c) ** 2)

    def curvature(w):
        return 2.0 * jnp.mean(w[idx] ** 2)

    if plus_merit:
        return lambda merit, v, w, *flat: (merit + phi(v, w, flat)) / curvature(w)
    return lambda v, w, *flat: phi(v, w, flat) / curvature(w)


def consistency_body(reference):
    scale = 1.0 / np.asarray(reference)
    return lambda *flat: _mismatch(flat, scale)


def function(body, reads, owns):
    """A function node's definition (not a graph operation). WORKAROUND(FR-10)."""
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), body)


insert = C.insert


# ---------------------------------------------------------------- the model and the recipe


def recipe(b: C.Base, d: C.Disciplines, cons: Consistency) -> tuple:
    """ALC as a recipe on the AAO base, one op at a time and none composed of others, in
    the three steps of jaxgraph's `examples/sellar_plans.py`:

    1. **Remove.** Every cut first, while `Opt` still determines the design: each
       reader of another discipline's quantity onto its own discipline's copy, then
       AAO's own cuts of the cycles inside operation. Then `Opt` and the implicit
       model's solve undetermined, and `Opt` deleted: its relations are split between
       the disciplines, which no op can do to a node in place (`partition.Base`).
    2. **Assemble.** Per discipline, in binding order, its problem (a condition node
       determining nothing: its objective, and `Opt`'s bounds of its design and its
       constraints, unchanged) and its objective function; the consistency at the end
       of a sweep; AAO's closures. Then each requirement
       combined into the problem owning its variables: the closures into operation,
       the coil solve into build.
    3. **Add.** Each discipline determined (its design, the unknowns of what was
       combined into it, its copies). Then the coordination over them: the upstream
       objectives' reads of the downstream copies cut (Gauss-Seidel, build first); the
       coordination problem inserted (`copy = value`, `merit_seen = merit`, `c = 0`),
       determined (the swept copies, `merit_seen`, `v`, `w`), and the three nested in
       it. Its driver, not the graph, updates `v` and `w` (`Coordinator`).

    Inside each discipline this is exactly what AAO's SAND does to the whole problem:
    the same cuts, the same closures, the same solve absorbed."""
    links = d.links
    owner = {c.at: b.owner(d.nodes, c.at) for c in b.closures}
    owner |= {s: b.owner(d.nodes, s) for s in b.solves}

    # 1. Remove
    ops = [Cut(k.place, k.readers, COPY[k.holder]) for k in links]
    ops += [cut for c in b.closures for cut in c.cuts]
    ops.append(Undetermine(C.TOP))
    ops += [Undetermine(s) for s in b.solves]
    ops.append(Delete((C.TOP,)))

    # 2. Assemble
    for k in C.DISCIPLINES:
        relations = b.share(d.designs[k], C.DISCIPLINE_CONSTRAINTS[k])
        ops.append(insert(PROBLEM[k], C.condition(OBJECTIVE[k], relations,
                                                  C.held_parameters(b, relations))))
        index = cons.touching(k, links)
        if k == ECONOMICS:
            body = penalty_body(index, cons.reference, plus_merit=True)
            reads = (C.MERIT, MULTIPLIER, WEIGHT, *cons.pairs(index))
        else:
            body = penalty_body(index, cons.reference)
            reads = (MULTIPLIER, WEIGHT, *cons.pairs(index))
        ops.append(insert(PENALTY[k], function(body, reads, (OBJECTIVE[k],))))
    ops.append(insert(CONSISTENCY, function(consistency_body(cons.reference),
                                            cons.pairs(range(len(cons))), (INCONSISTENCY,))))
    ops += [insert(c.at, C.requirement_of(c)) for c in b.closures]
    for k in C.DISCIPLINES:
        if mine := tuple(label for label, o in owner.items() if o == k):
            ops.append(Combine(PROBLEM[k], (PROBLEM[k], *mine), cx.RelationalCondition))

    # 3. Add
    held = {h: tuple(COPY[h](k.place) for k in d.copies_held_by(h)) for h in C.DISCIPLINES}
    for k in C.DISCIPLINES:
        absorbed = (*(u for s in b.solves if owner[s] == k for u in b.graph[s].unknowns),
                    *(m for c in b.closures if owner[c.at] == k for m in c.copies))
        ops.append(Determine(PROBLEM[k], (*d.designs[k], *absorbed, *held[k])))
    rank = {k: i for i, k in enumerate(C.DISCIPLINES)}
    swept = []
    for j, k in enumerate(links):
        readers = tuple(PENALTY[o] for o in C.DISCIPLINES
                        if rank[o] < rank[k.holder] and o in (k.source, k.holder))
        if readers:
            ops.append(Cut(cons.copies[j], readers, cx.Copy()))
            swept.append(cons.copies[j])
    moved = (*(cx.Relation(cx.Copy()(x), x, cx.Eq) for x in swept),
             cx.Relation(SEEN_MERIT, C.MERIT, cx.Eq))
    relations = (*moved, cx.Relation(INCONSISTENCY, None, cx.Eq))
    ops.append(insert(COORDINATION, C.condition(relations=relations)))
    ops.append(Determine(COORDINATION, (*(r.lhs for r in moved), MULTIPLIER, WEIGHT)))
    ops.extend(Nest(PROBLEM[k], COORDINATION) for k in C.DISCIPLINES)
    return tuple(ops)


def plan(configuration, d, cons) -> tuple:
    """`(base model, plan)`: the AAO base and the recipe on it."""
    b = C.base(configuration)
    p = Plan(b.graph)
    for op in recipe(b, d, cons):
        p = p + op
    return b.model, p


# ---------------------------------------------------------------- the coordination's driver


def is_coordination(node) -> bool:
    """A coordination: no objective; some unknowns each on the left of one equality,
    moved to its right-hand side every sweep; the others (multipliers and weights) on
    none; every other relation an equality held at zero (the consistency)."""
    if not is_problem(node) or node.objectives:
        return False
    unknowns = set(node.unknowns)
    moved = [r.lhs for r in node.relations if r.lhs in unknowns]
    held = [r for r in node.relations if r.lhs not in unknowns]
    return (all(r.op is cx.Eq for r in node.relations) and len(set(moved)) == len(moved)
            and bool(held) and all(r.rhs is None for r in held))


class AugmentedLagrangian(DriverContract):
    """Answers a coordination (`is_coordination`) from a `Start`, reporting `reports`:
    the moved unknowns to their right-hand sides, and the held conditions to zero by
    the unknowns on no relation, which the driver updates from the held values.
    ALC's own; not one of cottax's or the port's standard contracts."""

    accepts = staticmethod(is_coordination)
    reports: tuple = ()

    @property
    def takes(self) -> tuple:
        return (Start,)


class Sweep(Reading):
    """A coordination read at one point: `(moved, held)`, where each unknown goes next
    (its relation's right-hand side, or where it is for an unknown on no relation), and
    the values the held relations hold at zero. Each call walks the interior once: one
    sweep of the disciplines."""

    def __call__(self, *unknowns):
        at = self.values(unknowns)
        to = {r.lhs: r.rhs for r in self.relations if r.lhs in set(self.unknowns)}
        moved = tuple(at[to[u]] if u in to else value
                      for u, value in zip(self.unknowns, unknowns, strict=True))
        held = tuple(at[r.lhs] for r in self.relations if r.lhs not in set(self.unknowns))
        return moved, held


class Coordinator(PortDriver[AugmentedLagrangian]):
    """ALC's driver (Tosserams, Etman & Rooda 2008, Section 5), on the coordination
    problem: it owns the copies, the multipliers `v` and the weights `w`, and holds the
    consistency `c = 0`. Each iteration is one sweep (every discipline solved in turn at
    the current `v`, `w`), which moves the copies and returns `c`. The driver then
    updates `v <- v + 2 w o w o c` (Eq. 24) and `w <- beta w` where `|c|` fell by less
    than `gamma` since the last update and is still above `tolerance` (Eq. 25 with a
    guard), as the method of multipliers does.

    Three choices depart from the paper's stated algorithm (`alc_report.md`, section
    "Departures"):

    1. *The inner loop.* The update is taken once the copies have settled, their largest
       relative change over a sweep at most `inner_tolerance`, or after `max_inner`
       sweeps without, not when the relaxed objective stops changing (Eq. 28). Its
       tolerance is fixed; `inner_tolerance = inf` is alternating direction.
    2. *The stopping test.* At an update with `max |c| <= tolerance`, the multiplier step
       `max |2 w o w o c| <= dual_tolerance`, the copies moved by at most
       `settled_tolerance` and COE (carried at `merit`) by at most `merit_tolerance`
       over the sweep; not Eqs. 26-27 alone, which held at non-optimal points.
    3. *`beta`*, default 1 (weights never raised), below the paper's `beta > 1`.

    Traced (`lax.while_loop`), so it runs under `study.solve`'s `jit`; the nested
    SLSQP solves are host callbacks inside it."""

    multiplier: object
    weight: object
    merit: object
    beta: float = BETA
    gamma: float = GAMMA
    tolerance: float = TOLERANCE
    dual_tolerance: float = 1e-3
    inner_tolerance: float = 1e-3
    settled_tolerance: float = 1e-6
    merit_tolerance: float = 1e-7
    max_inner: int = 25
    max_sweeps: int = 400
    callback: object = None
    """`f(sweep, max_abs_c, settled, updated, max_weight, max_abs_dv, merit) -> None` on
    the host after each sweep."""

    needs = frozenset({Start})
    provides = frozenset({Steps})
    computes = (Steps,)

    def read(self, posed, context) -> Sweep:
        return Sweep(posed, context)

    def solve(self, sweep: Sweep, data) -> tuple:
        start = tuple(jnp.asarray(u, dtype=float) for u in data[Start])
        names = list(sweep.unknowns)
        v_at, w_at, m_at = (names.index(p) for p in (self.multiplier, self.weight, self.merit))
        copies = [i for i in range(len(names)) if i not in {v_at, w_at, m_at}]

        def cond(state):
            k, _j, _u, _previous, done = state
            return (k < self.max_sweeps) & ~done

        def body(state):
            k, j, u, previous, _done = state
            moved, held = sweep(*u)
            new = tuple(jnp.asarray(a, dtype=b.dtype).reshape(b.shape)
                        for a, b in zip(moved, u, strict=True))
            c = jnp.concatenate([jnp.ravel(jnp.asarray(h, dtype=float)) for h in held])
            v, w = u[v_at], u[w_at]
            worst = jnp.max(jnp.abs(c))
            settled = jnp.max(jnp.stack([
                jnp.max(jnp.abs(new[i] - u[i]) / jnp.maximum(jnp.abs(u[i]), 1e-12))
                for i in copies]))
            update = (settled <= self.inner_tolerance) | (j + 1 >= self.max_inner)
            step = 2.0 * w * w * c
            grow = (jnp.abs(c) > self.gamma * jnp.abs(previous)) & (jnp.abs(c) > self.tolerance)
            v_next = jnp.where(update, v + step, v)
            w_next = jnp.where(update, jnp.where(grow, self.beta * w, w), w)
            kept = tuple(v_next if i == v_at else w_next if i == w_at else a
                         for i, a in enumerate(new))
            dv = jnp.max(jnp.abs(step))
            merit_moved = jnp.abs(new[m_at] - u[m_at]) / jnp.abs(new[m_at])
            if self.callback is not None:
                jax.debug.callback(self.callback, k + 1, worst, settled, update,
                                   jnp.max(w_next), dv, new[m_at])
            done = (update & (worst <= self.tolerance) & (dv <= self.dual_tolerance)
                    & (settled <= self.settled_tolerance)
                    & (merit_moved <= self.merit_tolerance))
            return (k + 1, jnp.where(update, 0, j + 1), kept,
                    jnp.where(update, c, previous), done)

        none_yet = jnp.full(jnp.shape(start[v_at]), jnp.inf)   # one per multiplier
        k, _j, u, _previous, _done = lax.while_loop(
            cond, body, (jnp.asarray(0), jnp.asarray(0), start, none_yet, jnp.asarray(False)))
        return (*u, k)


# ---------------------------------------------------------------- run


def starts(cons: Consistency, initial_weight=INITIAL_WEIGHT) -> dict:
    """Where the coordination starts what it owns besides the copies (which start from
    the MDA at the file's design): no multipliers, every weight at `initial_weight`, no
    COE yet."""
    return {MULTIPLIER: np.zeros(len(cons)), WEIGHT: np.full(len(cons), float(initial_weight)),
            SEEN_MERIT: np.inf}


def driven(configuration, d, cons, env, traces: dict, *, max_sweeps=MAX_SWEEPS,
           tolerance=TOLERANCE, dual_tolerance=DUAL_TOLERANCE, inner_tolerance=INNER_TOLERANCE,
           max_inner=MAX_INNER, beta=BETA, gamma=GAMMA, max_iter=C.study.MAX_ITER, log=None) -> tuple:
    """`(model, graph)`: the recipe applied, each discipline answered by SLSQP and the
    coordination by `Coordinator`."""
    m, p = plan(configuration, d, cons)
    coordination = (COORDINATION, AugmentedLagrangian(reports=(Steps,)),
                    Coordinator(MULTIPLIER, WEIGHT, SEEN_MERIT, beta=beta, gamma=gamma,
                                tolerance=tolerance, dual_tolerance=dual_tolerance,
                                inner_tolerance=inner_tolerance, max_inner=max_inner,
                                max_sweeps=max_sweeps, callback=log))
    return m, C.drive(p, tuple(PROBLEM.values()), env, traces, max_iter=max_iter,
                      optimiser=SlsqpDriver, also=(coordination,)).graph


def run(configuration, *, max_sweeps=MAX_SWEEPS, tolerance=TOLERANCE,
        dual_tolerance=DUAL_TOLERANCE, inner_tolerance=INNER_TOLERANCE, max_inner=MAX_INNER, beta=BETA, gamma=GAMMA,
        initial_weight=INITIAL_WEIGHT, max_iter=C.study.MAX_ITER, verbose=True) -> dict:
    d = C.disciplines(configuration)
    state, env = C.mda_env(configuration)
    cons = consistency(d, env)
    sweeps: list = []

    def log(k, worst, settled, updated, wmax, dv, coe):
        sweeps.append((int(k), float(worst), float(settled), bool(updated), float(wmax),
                       float(dv), float(coe), time.perf_counter()))
        if verbose and (updated or int(k) % 25 == 0):
            print(f"  sweep {int(k):4d}  COE {float(coe):.7f}  max|c| {float(worst):.2e}  "
                  f"copies moved {float(settled):.1e}  {'update' if updated else '      '}  "
                  f"max w {float(wmax):.3g}  max|dv| {float(dv):.1e}", flush=True)

    traces: dict = {}
    m, graph = driven(configuration, d, cons, env, traces, max_sweeps=max_sweeps,
                      tolerance=tolerance, dual_tolerance=dual_tolerance,
                      inner_tolerance=inner_tolerance,
                      max_inner=max_inner, beta=beta, gamma=gamma, max_iter=max_iter, log=log)
    start = starts(cons, initial_weight)
    w0 = start[WEIGHT]
    inputs = C.values(graph, state, m.given, env, set(d.design), start=start)
    out, seconds = C.timed(graph, inputs)
    rows = {k: C.report(graph, out, PROBLEM[k], traces[PROBLEM[k]]) for k in C.DISCIPLINES}
    c = np.asarray(out[INCONSISTENCY])
    return {"graph": graph, "out": out, "seconds": seconds, "rows": rows, "disciplines": d,
            "consistency": cons, "c": c, "sweeps": sweeps, "w0": w0,
            "updates": sum(1 for row in sweeps if row[3]),
            "steps": int(np.asarray(out[_report(graph, COORDINATION, Steps)])),
            "design": {C.short(x): C.number(out, x) for x in d.design},
            "coe": C.number(out, C.MERIT)}


def _report(graph, problem, role):
    node = graph[problem]
    return dict(zip(node.contract.reports, node.reports, strict=True))[role][0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-sweeps", type=int, default=MAX_SWEEPS)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--dual-tolerance", type=float, default=DUAL_TOLERANCE)
    ap.add_argument("--inner-tolerance", type=float, default=INNER_TOLERANCE,
                    help="inf: alternating direction, one sweep per multiplier update")
    ap.add_argument("--max-inner", type=int, default=MAX_INNER)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--initial-weight", type=float, default=INITIAL_WEIGHT)
    ap.add_argument("--no-aao", action="store_true")
    args = ap.parse_args(argv)
    configuration = C.load(C.CONFIGURATION)
    d = C.disciplines(configuration)
    for k in C.DISCIPLINES:
        print(f"{k:10s} {len(d.nodes[k]):3d} models  design {[C.short(x) for x in d.designs[k]]}"
              f"  constraints {[C.cid(c) for c in d.constraints[k]]}"
              f"  copies {len(d.copies_held_by(k))}")
    for s, h in ((BUILD, OPERATION), (BUILD, ECONOMICS), (OPERATION, ECONOMICS)):
        print(f"  {s} -> {h}: {[C.short(k.place) for k in d.links_between(s, h)]}")
    print(f"  read without a copy (constant): {[C.short(p) for p in d.constant]}\n")

    result = run(configuration, max_sweeps=args.max_sweeps, tolerance=args.tolerance,
                 dual_tolerance=args.dual_tolerance,
                 inner_tolerance=args.inner_tolerance, max_inner=args.max_inner,
                 beta=args.beta, gamma=args.gamma, initial_weight=args.initial_weight)
    print(f"\nsolved in {result['seconds']:.1f} s (cold: includes compilation), "
          f"{result['steps']} sweeps, {result['updates']} multiplier updates, "
          f"max|c| {np.max(np.abs(result['c'])):.2e}")
    for name, row in result["rows"].items():
        print(f"  {name:10s} last solve: {row['status']:10s} status {row['vmcon_status']} "
              f"iterations {row['iterations']:3d}")
    print("  design:", {k: round(v, 4) for k, v in result["design"].items()})
    print(f"  COE {result['coe']:.6f}")
    worst = np.argsort(-np.abs(result["c"]))[:5]
    print("  largest inconsistencies:", [(C.short(result["consistency"].values[j]),
                                          f"{result['c'][j]:+.1e}") for j in worst])
    if not args.no_aao:
        ref = C.aao(configuration)
        print(f"\nAAO (SAND) from the file's start: {ref['status']}, {ref['iterations']} "
              f"iterations, COE {ref['objf']:.7f}")
        print(f"ALC: COE {result['coe']:.7f} ({result['coe'] / ref['objf'] - 1:+.2e} relative)")
        start = {x: C.number(result["out"], x) for x in result["disciplines"].design}
        there = C.aao(configuration, start=start)
        print(f"AAO (SAND) from ALC's answer: {there['status']}, {there['iterations']} "
              f"iterations, COE {there['objf']:.7f}: ALC's answer is "
              f"{'AAO' if abs(there['objf'] / result['coe'] - 1) < 1e-6 else 'not AAO'}'s minimum")
        result["aao"], result["aao_from_alc"] = ref, there
    return result


if __name__ == "__main__":
    main()
