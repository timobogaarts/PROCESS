"""The `Optimise` layer, assembled the other way round: **MDF** (Multidisciplinary
Feasible).

`sand.py` puts the whole coupled system into one solver -- the eight iteration variables
*and* every coupling variable are design unknowns, and every coupling equation is an
equality constraint (`_audit/optimise_design.md` §10). This module does what **PROCESS
itself does**: the optimiser sees only the eight iteration variables, and every
evaluation of the objective and the fourteen constraints first drives the whole MDA to
convergence. `process/core/caller.py`'s `Caller.call_models` re-runs the entire model
pipeline up to ten times per objective/constraint evaluation and stops when the answer
stops changing; that is a Gauss-Seidel MDA nested inside every function call, i.e. MDF
with an undeclared inner solver. Here the inner solve is the graph's own `Schedule` --
each genuinely coupled block driven by the algorithm it declares, everything else run
once in dependency order.

What MDF *is*, in cottax terms
------------------------------
An `Optimise` whose block's **interior is itself blocked and scheduled**. cottax states
that exactly and cannot yet run it, and the boundary between those two is sharp:

- **Structure: expressible today.** Insert the `Optimise` (`sand.optimise_graph` does it)
  and the design variables reach every node while the conditions come back from them, so
  the optimiser and the whole MDA collapse into **one** SCC -- 131 of this graph's 174
  nodes, holding twelve declared problems. `Graph.problem_type` refuses that ("one driver
  answers one problem"), and `Blocking.nest(problem)` is cottax's own answer to that
  refusal: `Blocking.scc(graph).nest(opt)` records the `Optimise` as answered at the
  outer level with the remaining 130 nodes blocked into 111 inner blocks, 11 of them
  driven. That is MDF, stated. `nested_blocking()` below builds it, and
  `test_mdf.py::test_cottax_states_mdf_structurally` pins that it builds.
- **Evaluation: not expressible today.** `Schedule.steps` derives one `Call`/`Drive` per
  block from `blocking.subgraphs` and **never reads `blocking.inner`**, so `schedule_for`
  on the nested blocking builds a `Drive` over the entire 131-node block and dies on the
  same `problem_type` refusal. `~/jaxgraph`'s own `CLAUDE.md` says so without hedging:
  *"nothing builds a nested one yet: the steps are derived from the blocking, so a
  sub-schedule has no slot to sit in. Nesting lands on `Drive.body`, which becomes a
  `Step` rather than a `Graph`. Until then ... `schedule_for` still refuses MDF."*

So the missing upstream piece is **one type**: `Drive.body` is a `Graph` (run by
`_run_acyclic`, therefore acyclic) where MDF needs it to be a `Step`. `MdfConditionMap`
below is that one change, made locally and only for the outer problem: a `ConditionMap`
whose body is run by a `Schedule` instead of by `_run_acyclic`. Nothing else about the
driver interface moves -- `VmconDriver` is handed this object **unchanged** and cannot
tell the difference, which is the evidence that the gap really is that narrow. See §
"What this module does not do" for what stays out of scope.

Why bother, given SAND works
----------------------------
Two reasons.

1. **Like-for-like.** MDF's design vector *is* PROCESS's `ixc` and its conditions *are*
   PROCESS's `icc` plus the figure of merit, so the Jacobian this module differentiates
   is `d(conditions)/d(design)` -- **the same derivative `Evaluators.fcnvmc2`
   finite-differences**, with no Schur complement in between. `sand_harness.
   reduce_jacobian` exists precisely because a SAND Jacobian is *not* that derivative
   (it holds the coupling variables fixed) and has to be reduced onto the design columns
   before any cell-by-cell comparison means anything. MDF needs no reduction, no
   equilibration and no `np.linalg.solve`: the comparison is direct.
2. **The cold start, and what it turned out to be about.** SAND cold-started at *zero*
   SQP iterations when this module was begun: 20 of its 24 conditions were `nan` at the
   cold point, because a SAND condition map holds the coupling variables *fixed* at
   whatever guess it was given and a cold `DataStructure` has `0.0` in every
   model-computed field (`_audit/optimise_design.md` §10.6). An MDF condition map has no
   coupling unknowns to guess: it *computes* every one of them from the design point by
   running the MDA, so the question does not arise.

   That gap has since been closed on the SAND side too -- by seeding its coupling
   unknowns from a completed MDA run at the same design, which
   `run_sand_harness._seed`'s own docstring describes as *"exactly what MDF would hand
   iteration 0"*. Which is the point worth keeping: SAND cold-starts **because** an MDA
   run is available to seed it. MDF does not need the seeding rule to be got right,
   because the MDA is the formulation.

What MDF does not need, and it is a long list
---------------------------------------------
Every one of these is machinery `sand.py` needs and this module does not, because the
coupling never enters the optimiser's problem:

- **`Residualise` / `Combine`** -- the `FixedPoint`s stay `FixedPoint`s and are driven by
  `PicardDriver`, exactly as `mda.schedule()` already drives them.
- **`degenerate_fixed_points`** -- an identity fixed point is a well-posed Picard problem
  (it converges in one step) and only became a rank-deficient *equality* because SAND
  turned it into one. Nothing is dropped here. What it does mean, and it is the same
  fact wearing different clothes, is that such an unknown keeps **whatever `prime()`
  seeded it with**: `.heat_transport.eta_turbine` comes out at `0.375` because that is
  what went in. SAND says so structurally (it deletes the problem and the unknown reverts
  to a boundary input); MDF says it quietly, so `sand.degenerate_fixed_points` is still
  the thing to run when the question is "does the graph determine this at all".
- **`residual_condition_scales`** -- every condition is one of PROCESS's own normalised
  residuals (`value/bound - 1`, O(1) by construction) or the figure of merit. There are
  no residuals in physical units to rescale, so `VmconDriver.condition_scale` stays
  empty and the iterates stay directly comparable with PROCESS's.
- **starting guesses for coupling variables** -- PROCESS never had to supply them; nor
  does this.

What it needs instead is that the **inner solve converge at every trial point**, which
SAND buys off by refusing to solve it exactly. That is the real trade, and
`inner_residuals()` below is how it is checked rather than assumed.

The starting guess for the inner solve is frozen, deliberately
--------------------------------------------------------------
`prime()` runs the schedule **once**, eagerly, from a `DataStructure` and keeps its
converged values as the starting guess for every inner unknown. Those guesses then sit
in the condition map's `context` and do not move as the outer optimiser steps. That is a
choice, and the alternative -- warm-starting each inner solve from the previous outer
iterate -- is the one to avoid here: it makes the conditions a function of the
*optimiser's history* rather than of `x`, so two evaluations at the same `x` can differ
and the Jacobian stops being the Jacobian of the function VMCON thinks it is stepping on.
A frozen guess keeps the map a pure function of `x`, which is what both the SQP's line
search and `jax.jacfwd` assume.

Differentiating through the inner solve
---------------------------------------
`jax.jacfwd` of an `MdfConditionMap` differentiates **through** thirteen driven blocks:
twelve `PicardDriver`s (`jax.lax.while_loop`) and one `SeededNewtonDriver`
(`optimistix.root_find`). Forward mode is what makes that legal -- `lax.while_loop` has
a JVP rule and no transpose rule, so `jacfwd` works and `jacrev`/`grad` do not, and
`optimistix` supplies an implicit-function derivative for the root find. With eight
design variables and fifteen conditions, forward mode is also the cheaper direction, so
nothing is given up.

Two consequences worth stating rather than discovering:

- A `while_loop`'s JVP differentiates **the iteration that actually ran**, not the exact
  fixed point. Those agree to the extent the iteration converged, which is why
  `inner_residuals()` reports how far each block got. The derivative is nonetheless the
  exact derivative of the function the optimiser is *actually* evaluating, which is the
  property an SQP needs; `run_mdf_harness.py` checks it against a central difference of
  that same map.
- **A `SeededNewtonDriver` carrying a `seed` cannot be traced at all**, and this is a
  defect in `core/solver/drivers.py` rather than a fact about MDF: `_usable(start)` calls
  `np.asarray(ravel_pytree(start)[0])`, and inside *any* JAX trace `ravel_pytree` returns
  a tracer even for a concrete argument, so the guard raises
  `TracerArrayConversionError` before the seed is ever needed. `traceable_drivers()`
  works around it by clearing `seed` (the start is supplied explicitly by `prime()`
  instead), and the one-line upstream fix is recorded there.

What this module does not do
----------------------------
It does not make cottax able to *run* a nested blocking, and does not try to:
the outer `Drive` is performed by calling the driver directly (`solve()`), which is what
`Drive.__call__` would do. Nothing here is a `Step`, so an MDF cannot be nested inside
anything else, and that limit is the same one upstream has.
"""

import dataclasses
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import ConditionMap, Drive, Schedule, schedule_for
from cottax.graph import Graph
from cottax.plan import Insert, Plan
from cottax.problem import FixedPoint
from cottax.spec import VarPath
from cottax.tools.path import path_map

from functional_process import sand
from functional_process.core.solver.drivers import SeededNewtonDriver, VmconDriver
from functional_process.mda import default_drivers, driven_graph
from functional_process.mda_harness import _without_excluded
from functional_process.sand_harness import ground_truth
from functional_process.total_process import graph_for


@dataclasses.dataclass(frozen=True)
class Mdf:
    """One assembled MDF problem: the graph, the two schedules, and the problem's shape.

    `eager` and `traceable` run the **same** blocking with the same drivers bar one
    field (see `traceable_drivers`), so a value computed by one is computed by the other;
    they exist separately only because the seed fallback that makes a cold start possible
    is not traceable and the traced map does not need it.
    """

    graph: Graph
    """The MDA graph plus one `CallableNode` per active constraint and one for the
    objective. **No `Optimise` node**: inserting one fuses the whole graph into a single
    SCC that `schedule_for` cannot run (this module's docstring). The problem is stated
    by `design`/`conditions` below and answered by `solve`."""
    eager: Schedule
    traceable: Schedule
    design: tuple[VarPath, ...]
    """The run's `ixc`, in PROCESS's own order."""
    conditions: tuple[VarPath, ...]
    """`(objective, *equalities, *inequalities)` -- `Optimise.inputs`' own order, which
    is the positional contract `VmconDriver` reads its split from."""
    n_equality: int
    n_inequality: int
    report: dict


def mdf_graph(graph, icc, n_equality, i_figure_merit, switch_values=None, omit=()):
    """`graph` with `sand.constraint_nodes`' and `sand.objective_node`'s nodes inserted.

    Reused wholesale from `sand.py` and deliberately not re-derived: which constraints
    are active, which of them are equalities (positional, from `n_equality`), which
    parameters are static switches and which resolve to `VarPath`s, and the objective's
    folded-in sign are all properties of *the run*, not of the formulation. An MDF
    assembly that answered any of them differently from the SAND one would make the two
    incomparable, which is the whole point of building this beside it.

    The one thing this does *not* do is insert the `Optimise`. See `nested_blocking`.

    Returns
    -------
    :
        `(graph, conditions, n_inequality, report)`.
    """
    nodes, equalities, inequalities, omitted = sand.constraint_nodes(
        graph, icc, n_equality, switch_values, omit
    )
    objective_name, objective_definition, objective = sand.objective_node(
        graph, i_figure_merit, switch_values
    )
    nodes[objective_name] = objective_definition
    inserted = (Plan(graph) + Insert(path_map(nodes.items()))).graph
    return (
        inserted,
        (objective, *equalities, *inequalities),
        len(inequalities),
        {
            "equalities": equalities,
            "inequalities": inequalities,
            "objective": objective,
            "omitted": omitted,
        },
    )


def traceable_drivers(drivers):
    """`drivers` with every `SeededNewtonDriver`'s `seed` cleared.

    **This is a workaround for a defect, not a design choice.**
    `SeededNewtonDriver.__call__` opens with `if self.seed is not None and not
    _usable(start)`, and `_usable` does `np.asarray(ravel_pytree(start)[0])`. Inside a
    JAX trace `ravel_pytree` stages its `concatenate` out and returns a tracer **even for
    a concrete argument**, so that line raises `TracerArrayConversionError` before it can
    decide anything -- which means no `Schedule` containing a seeded `SeededNewtonDriver`
    can be `jit`ted or differentiated at all. Measured directly, not inferred: with the
    seed in place `jax.jit(condition_map)` fails at
    `drivers.py:_usable`; with it cleared the whole 152-block schedule traces, compiles
    and differentiates.

    The upstream fix is one line in a file this module does not own -- `_usable` should
    return `False` (or `True`, but not raise) for a traced start, e.g. by testing
    `isinstance(flat, jax.core.Tracer)` first, since a tracer is by construction not the
    `0.0` placeholder the guard exists to catch. **Reported, not applied.**

    Clearing the seed is safe *here* and only here: `prime()` supplies a converged
    starting value for every inner unknown, so the fallback would never fire, and
    `optimistix`' implicit-function derivative does not depend on where the root find
    started. A caller with no primed env would get `SeededNewtonDriver`'s own
    "needs a starting value" error instead of a silent bad guess.
    """
    return {
        problem: (
            dataclasses.replace(driver, seed=None)
            if isinstance(driver, SeededNewtonDriver)
            else driver
        )
        for problem, driver in drivers.items()
    }


def assemble(
    ixc,
    icc,
    n_equality,
    i_figure_merit,
    graph=None,
    switch_values=None,
    omit=(),
):
    """The whole MDF assembly: cut the raw cycles, add the conditions, build both
    schedules.

    `graph` defaults to `graph_for()` -- the reference configuration -- with
    `mda_harness.EXCLUDED_NODE_NAMES` deleted, which is exactly what
    `sand_harness.mda_env` feeds SAND. The exclusion (`DuctDiameterRootFind`, whose
    `VarPath`s no `DataStructure` field backs) is kept **for comparability, not because
    MDF needs it**: dropping it changes which graph is being optimised, and the two
    formulations have to optimise the same one.

    Raises
    ------
    ValueError
        Via `sand.constraint_nodes`, on any active constraint that cannot be assembled.
        Same policy and same reason as SAND's: an `Optimise` over 12 of PROCESS's 14
        active constraints is a different problem.
    """
    driven = driven_graph(_without_excluded(graph if graph is not None else graph_for()))
    graph, conditions, n_inequality, report = mdf_graph(
        driven, icc, n_equality, i_figure_merit, switch_values, omit
    )
    blocking = Blocking.scc(graph)
    drivers = default_drivers(blocking)
    design = tuple(sand.iteration_variable_path(i) for i in ixc)
    eager = schedule_for(blocking, drivers)
    missing = [d for d in design if d not in eager.inputs]
    if missing:
        raise ValueError(
            f"design variable(s) {[d.path_str() for d in missing]} are not boundary "
            f"inputs of the MDA graph -- a node already produces them, so the optimiser "
            f"cannot own them (see `sand.optimise_graph` on the same conflict)"
        )
    report["blocks"] = len(blocking.blocks)
    report["driven_blocks"] = sum(1 for t in blocking.problem_types if t is not None)
    return Mdf(
        graph=graph,
        eager=eager,
        traceable=schedule_for(blocking, traceable_drivers(drivers)),
        design=design,
        conditions=conditions,
        n_equality=n_equality,
        n_inequality=n_inequality,
        report=report,
    )


def seed(mdf: Mdf, data, design_values=None):
    """Every schedule input and every inner unknown, read off `data`.

    `ground_truth` is `sand_harness`'s -- `mda_harness._ground_truth`'s rule made public,
    handling the minted `VarPath`s that have no `DataStructure` field of their own. It is
    a general seeding rule with nothing SAND-specific about it and it lives in a harness
    module; importing it from there rather than restating it is the lesser of the two
    wrongs, but it does belong one layer down (reported, not moved -- `sand_harness.py`
    is not this session's file).

    A field with no value at all falls back to `0.0`, which for a cold `DataStructure` is
    most of them. That is exactly the state `prime()` then resolves.

    **Which `data` this is given decides which problem gets solved**, and on this run
    that is not a formality. Of the MDA's 360 boundary inputs, a converged
    `DataStructure` and a cold one differ in exactly **two** --
    `.physics.nd_plasma_pedestal_electron` (`0` against `4e19`) and
    `.physics.nd_plasma_separatrix_electron` (`0` against `3e19`) -- and at one fixed
    design point that pair moves the objective from `1.2150` to `1.2491` and `c2` from
    `-1.4e-07` to `-7.6e-03`. The cause is a **missing producer**, not a seeding choice:
    `PlasmaProfile.run` resets six fields to L-mode values when the profile is not a
    pedestal one (`process/models/physics/plasma_profiles.py:110-117`, zeroing both of
    these), and the port has no node for that reset, so both are boundary inputs and a
    cold seed carries the input file's own values into
    `plasma_profiles.calculate_parabolic_profile_values`' `prn1`. Emulating the reset by
    hand takes the cold solve from a median `1.4e-02` distance to PROCESS's converged `x`
    to `8.6e-03`. Reported rather than patched -- registering the node is
    `total_process.py`'s business. It is the same defect class as
    `_audit/optimise_design.md` §10.5a/§10.5c (a missing producer no value test can see),
    found by a third method: a cold start rather than a gradient.
    """
    env = {}
    for var in list(mdf.eager.inputs) + list(mdf.eager.unknowns):
        try:
            env[var] = jnp.asarray(ground_truth(data, var))
        except (AttributeError, KeyError):  # noqa: PERF203 -- per-variable by nature
            env[var] = jnp.asarray(0.0)
    if design_values is not None:
        values = [jnp.asarray(v) for v in design_values]
        env.update(zip(mdf.design, values, strict=True))
    return env


def prime(mdf: Mdf, env):
    """Run the MDA once, eagerly, and keep its answer as the inner starting guess.

    Returns `(env, out)`: `env` is `env` with every inner unknown replaced by the value
    the schedule converged it to, and `out` the full output env of that run.

    Why it exists: two of this graph's inner solvers cannot start from the `0.0` a cold
    `DataStructure` supplies. `Intersect`'s residual is exactly flat below `x ~ 0.1`, so
    Newton from `0.0` cannot move (`SeededNewtonDriver`'s own docstring records the
    measurement), and `PlasmaComposition` branches on `fusden_alpha_total < 1e-6` as a
    "not yet calculated" bootstrap, so a Picard iterate started at zero drives a *branch
    predicate*. Running the schedule once with the seeded drivers -- eagerly, where the
    fallback guesses work -- resolves both, and every subsequent traced evaluation starts
    from a converged point.

    This is also, exactly, what an MDF architecture would hand iteration 0 anyway
    (`run_sand_harness._seed` says as much about SAND's coupling guesses); here it is the
    architecture rather than a borrowing.

    One pass, not a loop
    --------------------
    The MDA needs **no outer fixed-point iteration** around the per-block drivers, and
    that follows from the blocking rather than from luck: `Blocking.__check_init__`
    refuses any block that reads what a later one owns, so the block order is a
    topological order over the condensation, and each cyclic block is driven to its own
    problem's answer before the pass leaves it. PROCESS's `Caller.call_models` re-runs
    its whole pipeline up to ten times and checks idempotence because it has no
    condensation to order by -- the loop is the price of the missing structure, not a
    property of the physics.

    Checked, not asserted, by `test_mdf.py::test_one_pass_of_the_schedule_is_idempotent`,
    which applies `check_agreement`'s own `rtol = 1e-6` to one pass against two.
    """
    out = mdf.eager(dict(env))
    primed = dict(env)
    for unknown in mdf.eager.unknowns:
        primed[unknown] = out[unknown]
    return primed, out


class MdfConditionMap(ConditionMap):
    """`f(*design) -> conditions`, with a whole converged MDA inside every call.

    A `cottax.evaluate.ConditionMap` in every respect the driver can observe -- same
    `unknowns`/`conditions`/`context` fields, same `f(*unknowns) -> tuple` contract, so
    `VmconDriver` takes it unchanged and unmodified -- differing in **one** thing:
    `ConditionMap.__call__` runs its body with `_run_acyclic`, which requires the body to
    be acyclic, and this runs it with a `Schedule`, which does not.

    That single difference is the whole of the upstream gap this module documents. cottax
    already says where the fix belongs: *"Nesting lands on `Drive.body`, which becomes a
    `Step` rather than a `Graph`"*. `body` is kept here as the schedule's own graph in
    run order, so it stays a truthful answer to "what does this map compute", and
    `schedule` carries how.

    Subclassing rather than duck-typing is deliberate: `isinstance(x, ConditionMap)`
    stays true, so if a driver ever checks -- `core/solver/drivers.py` is being edited
    by another session as this is written -- this does not silently stop being
    acceptable.
    """

    schedule: Schedule

    def __call__(self, *design):
        """The conditions at `design`, with the MDA driven to convergence there.

        Raises
        ------
        TypeError
            If the number of design values does not match `unknowns`.
        """
        if len(design) != len(self.unknowns):
            raise TypeError(
                f"MDF condition map takes {len(self.unknowns)} design variable(s) "
                f"({', '.join(v.path_str() for v in self.unknowns)}), got {len(design)}"
            )
        env = dict(self.context)
        env.update(zip(self.unknowns, design, strict=True))
        at = self.schedule(env)
        return tuple(at[condition] for condition in self.conditions)


def condition_map(mdf: Mdf, env, traceable=True) -> MdfConditionMap:
    """`f(*design) -> conditions` for `mdf`, everything else in `env` closed over.

    `env` must be a **primed** env (`prime`): it supplies the starting guess for every
    inner unknown, frozen for the whole solve (this module's docstring says why).
    """
    context = {var: value for var, value in env.items() if var not in set(mdf.design)}
    return MdfConditionMap(
        body=mdf.traceable.subgraph,
        unknowns=mdf.design,
        conditions=mdf.conditions,
        context=path_map(context.items()),
        schedule=mdf.traceable if traceable else mdf.eager,
    )


def driver(mdf: Mdf, bounds=(), callback=None, **kwargs) -> VmconDriver:
    """`VmconDriver` for `mdf`, its equality/inequality counts read off the assembly.

    The same rule `mda.default_drivers` applies to a real `Optimise` node -- the counts
    come from what was assembled, never from a caller counting conditions.
    `condition_scale` is deliberately not offered: MDF has no residual conditions (see
    this module's docstring), so every row is one of PROCESS's own normalised residuals
    and scaling any of them would break comparability with PROCESS's iterates.
    """
    return VmconDriver(
        n_equality=mdf.n_equality,
        n_inequality=mdf.n_inequality,
        bounds=bounds,
        callback=callback,
        **kwargs,
    )


def solve(mdf: Mdf, env, bounds=(), callback=None, optimiser=None, **kwargs):
    """Drive the outer `Optimise`, then re-run the MDA at the answer.

    Exactly what `cottax.evaluate.Drive.__call__` does -- call the driver with a
    condition map and a start, put the answer back into the env, re-run the body so the
    block's internals land there too -- written out because the `Drive` this would be
    cannot be constructed (this module's docstring).

    `optimiser` overrides the `VmconDriver` this would otherwise build -- any
    `AbstractDriver` whose `drives` is `Optimise` will do, which is how the same MDF
    problem can be handed to a second SQP as a controlled comparison.

    Returns
    -------
    :
        `(x, out, seconds)` -- the design values in `mdf.design` order, the full output
        env of the MDA at that point, and the wall clock for the whole solve.
    """
    conditions = condition_map(mdf, env)
    start = tuple(jnp.asarray(env[var]) for var in mdf.design)
    optimiser = optimiser or driver(mdf, bounds=bounds, callback=callback, **kwargs)
    started = time.perf_counter()
    x = optimiser(conditions, start)
    elapsed = time.perf_counter() - started
    at = dict(env)
    at.update(zip(mdf.design, x, strict=True))
    return tuple(x), mdf.eager(at), elapsed


def evaluation(conditions: MdfConditionMap, start, repeats=5):
    """`(values, compile seconds, jitted median milliseconds)` -- what one MDF condition
    evaluation costs.

    The number the "MDF is more expensive per call than SAND" expectation is about: one
    call here converges the whole MDA, where a SAND call runs an acyclic body once. It is
    measured rather than assumed because the expectation turns out to be wrong at this
    size -- the inner solvers are a handful of small `lax.while_loop`s inside a single
    compiled program, and XLA does not care that they are loops.

    Compare it with `Evaluators.fcnvmc1`'s cost on the PROCESS side, which is one
    `Caller.call_models` -- up to ten full pipeline passes -- per call.
    """

    def flat(*design):
        return jnp.stack([jnp.asarray(v) for v in conditions(*design)])

    compiled = eqx.filter_jit(flat)
    began = time.perf_counter()
    values = compiled(*start)
    jax.block_until_ready(values)
    compile_seconds = time.perf_counter() - began
    timings = []
    for _ in range(repeats):
        began = time.perf_counter()
        values = compiled(*start)
        jax.block_until_ready(values)
        timings.append(time.perf_counter() - began)
    return (
        np.asarray(values, dtype=float),
        compile_seconds,
        float(np.median(timings) * 1e3),
    )


def jacobian(conditions: MdfConditionMap, start, repeats=5):
    """`(J, compile seconds, jitted median milliseconds)` -- `d(conditions)/d(design)`.

    `jax.jacfwd`, not `jacrev`: the inner `PicardDriver`s are `jax.lax.while_loop`s,
    which have a JVP rule and no transpose rule, so reverse mode is not available at any
    price. Forward mode is also the right direction at 8 design variables against 15
    conditions.

    Timed under `eqx.filter_jit` after one warm-up call, the same convention
    `sand_harness.port_jacobian` uses and for the same reason: a single unjitted call is
    trace plus compile plus execute, and reporting that as the cost of a Jacobian
    overstates it by three orders of magnitude.

    **This matrix needs no reduction to be compared with PROCESS's.** It is already
    `d(objective, constraints)/d(ixc)` with the MDA converged at every point, which is
    what `Evaluators.fcnvmc2` finite-differences. Only the two exact conversions of
    `sand_harness.to_process_spelling` apply (PROCESS differentiates the *scaled*
    variable, and its `cc` is `-normalised_residual`).
    """

    def flat(*design):
        return jnp.stack([jnp.asarray(v) for v in conditions(*design)])

    compiled = eqx.filter_jit(jax.jacfwd(flat, argnums=tuple(range(len(start)))))
    began = time.perf_counter()
    columns = compiled(*start)
    jax.block_until_ready(columns[0])
    compile_seconds = time.perf_counter() - began
    timings = []
    for _ in range(repeats):
        began = time.perf_counter()
        columns = compiled(*start)
        jax.block_until_ready(columns[0])
        timings.append(time.perf_counter() - began)
    full = np.stack([np.asarray(c, dtype=float) for c in columns], axis=1)
    return full, compile_seconds, float(np.median(timings) * 1e3)


def central_difference(conditions: MdfConditionMap, start, relative_step=1e-5):
    """`d(conditions)/d(design)` by central differences **of this same map**.

    The control for `jacobian`: it differentiates the function the optimiser actually
    evaluates, inner solve and all, without any autodiff. If `jacfwd` and this disagree,
    the fault is in differentiating *through* the inner solve -- a `while_loop` JVP that
    does not match the loop it ran, or an implicit-function derivative attached to a root
    find that did not converge -- and not in the model, because both see the same model.
    Agreement is the evidence that MDF's gradient is right; agreement with PROCESS's own
    finite differences is a separate and weaker claim, since PROCESS perturbs by 1 % and
    re-converges its own loop only to `rtol = 1e-6`.

    A `while_loop` whose trip count changes between the two perturbed points makes its
    column meaningless rather than merely noisy; that shows up as one bad column, not as
    scatter, which is why this is reported per cell.

    Jitted, like `jacobian` and for the same reason: one uncompiled call of this map
    costs ~1.5 s against ~0.4 ms compiled, so `2n` of them dominate everything else in
    the harness. It is the same program either way -- measured agreement between the
    compiled and uncompiled map is `9e-16` -- and no derivative machinery is involved,
    which is the whole point of this control.
    """

    def flat(*design):
        return jnp.stack([jnp.asarray(v) for v in conditions(*design)])

    compiled = eqx.filter_jit(flat)
    columns = []
    for index, value in enumerate(start):
        step = relative_step * max(abs(float(np.asarray(value))), 1.0)
        forward = list(start)
        backward = list(start)
        forward[index] = jnp.asarray(float(np.asarray(value)) + step)
        backward[index] = jnp.asarray(float(np.asarray(value)) - step)
        high = np.asarray(compiled(*forward), dtype=float)
        low = np.asarray(compiled(*backward), dtype=float)
        columns.append((high - low) / (2.0 * step))
    return np.stack(columns, axis=1)


def inner_residuals(schedule: Schedule, env):
    """How well each driven block is actually solved at `env`, as
    `[(block, unknown, residual, relative)]`.

    An MDF answer is only as good as its inner solve, and `PicardDriver` stops at
    `max_iter = 20` whether or not it converged -- silently, because `lax.while_loop`
    cannot raise. So this re-evaluates every driven block's own condition map at the
    values `env` holds and reports the gap: `g(u) - u` for a `FixedPoint` (whose
    condition *is* the next iterate), the condition itself for a `RootFind` (whose
    condition is already a residual).

    Not a check the schedule performs for itself, and it should be read as the answer to
    "did the MDA converge here", which is the one question MDF's correctness rests on and
    SAND answers structurally instead (its residuals are equalities the SQP drives to
    zero).
    """
    rows = []
    for step in schedule.steps:
        if not isinstance(step, Drive):
            continue
        values = step.condition_map(env)(*[env[u] for u in step.unknowns])
        fixed_point = issubclass(step.problem_type, FixedPoint)
        for unknown, value in zip(step.unknowns, values, strict=True):
            current = float(np.asarray(env[unknown]))
            residual = float(np.asarray(value)) - (current if fixed_point else 0.0)
            rows.append((
                step.problem,
                unknown,
                residual,
                abs(residual) / max(abs(current), 1e-30),
            ))
    return rows


def nested_blocking(ixc, icc, n_equality, i_figure_merit, graph=None, **kwargs):
    """MDF **stated as structure**: `Blocking.scc(graph + Optimise).nest(the Optimise)`.

    Nothing runs this -- `schedule_for` refuses it, see this module's docstring -- and
    that is exactly why it is here: it is the difference between "cottax cannot express
    MDF" (false) and "cottax cannot evaluate a nesting it can express" (true), and the
    two have completely different consequences for the rewrite.

    Returns
    -------
    :
        `(blocking, problem_name, report)` -- `blocking.inner[i]` at the `Optimise`'s
        block is the MDA, blocked in its own right.
    """
    driven = driven_graph(_without_excluded(graph if graph is not None else graph_for()))
    with_problem, problem_name, report = sand.optimise_graph(
        driven, ixc, icc, n_equality, i_figure_merit, **kwargs
    )
    return Blocking.scc(with_problem).nest(problem_name), problem_name, report


def mdf_shape(mdf: Mdf) -> dict:
    """The assembly's shape, for reporting -- the counterpart of `sand.sand_shape`.

    The contrast is the point: SAND reports one `Drive` holding most of the graph, with
    the coupling variables among its unknowns. MDF reports a design vector the size of
    PROCESS's `ixc` and an inner schedule the size of the whole MDA.
    """
    return {
        "nodes": len(mdf.graph.nodes),
        "design": len(mdf.design),
        "conditions": len(mdf.conditions),
        "equalities": mdf.n_equality,
        "inequalities": mdf.n_inequality,
        "inner_blocks": mdf.report["blocks"],
        "inner_driven": mdf.report["driven_blocks"],
        "inner_unknowns": len(mdf.eager.unknowns),
        "inner_inputs": len(mdf.eager.inputs),
    }


__all__ = [
    "Mdf",
    "MdfConditionMap",
    "assemble",
    "central_difference",
    "condition_map",
    "driver",
    "evaluation",
    "inner_residuals",
    "jacobian",
    "mdf_graph",
    "mdf_shape",
    "nested_blocking",
    "prime",
    "seed",
    "solve",
    "traceable_drivers",
]
