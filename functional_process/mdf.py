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
  the optimiser and the whole MDA collapse into **one** SCC -- 123 of this graph's 165
  nodes on the stellarator reference configuration, holding five declared problems.
  `Graph.problem_type` refuses that ("one driver answers one problem"), and
  `Blocking.nest(problem)` is cottax's own answer to that refusal:
  `Blocking.scc(graph).nest(opt)` records the `Optimise` as answered at the outer level
  with the remaining 122 nodes blocked into 112 inner blocks, 4 of them driven. That is
  MDF, stated. `nested_blocking()` below builds it, and
  `test_mdf.py::test_cottax_states_mdf_structurally` pins that it builds. (These counts
  were 174/131/twelve/111/11 before `1db889f6` dissolved ten `FixedPoint`s into switch
  slots; `run_mdf_harness.py` prints the live ones, and `_audit/optimise_design.md` §14
  measures what the change cost the outer solve.)
- **Evaluation: expressible too, since `~/jaxgraph` `33af0a5`.** This paragraph used to
  say the opposite, and the correction is `_audit/in_graph_rootfind.md` §1: `Drive` now
  carries `(subgraph, problem, body : Step)`, `ConditionMap.body` is a `Step` run by
  calling it, and `Schedule.steps` builds `Schedule(held)` for a block whose
  `blocking.inner` entry is not `None`. So `schedule_for` on a nested blocking builds a
  nested `Drive`, and the *one type* this section named as the missing upstream piece is
  no longer missing. **Measured, not read** (`_audit/in_graph_rootfind.md` §1):
  `Schedule(Blocking.scc(assign_drivers(g)).nest(opt))` on the stellarator reference
  builds a 47-step schedule whose `Drive` over the 123-node block has a 112-block
  `Schedule` for a body.

  What the old claim's own evidence turned out to be: `nested_blocking()` returns a graph
  with **no drivers assigned at all** (`cut_graph` is structure only and
  `sand.optimise_graph` attaches nothing by default), so `schedule_for` refused it with
  *"carries no driver ... `Assign` is how the algorithm is said"* -- a refusal about the
  missing `Assign`, not about the nesting. `test_mdf.py`'s regex matched that message and
  the test passed for the wrong reason; §1 records both.

`MdfConditionMap` below is therefore **redundant** and is kept for one release anyway:
removing it is a second change, and one change at a time is what makes a moved answer
attributable (`_audit/in_graph_rootfind.md` §7).

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

Two shapes of outer solve, and the second is the one to prefer
--------------------------------------------------------------
`assemble` + `solve` is the **outer-driver** shape: the problem is not in the graph, the
outer `Drive` is performed by calling the driver directly, and `MdfConditionMap` re-runs
the *entire* MDA schedule on every residual evaluation.

`in_graph_root_find` + `in_graph_solve` is the **in-graph** shape, for the two files
whose `i_process_run_mode` is `-2`: the `RootFind` is a node of the graph, `Blocking.scc`
decides what it drives, and the driven block is the SCC the problem actually closes --
39 of 259 nodes on `spherical_tokamak_eval`, 35 of 271 on `large_tokamak_eval`
[measured]. Everything upstream of the design variables runs **once**, before the solve;
everything downstream of the constraints runs **once**, after it. The two coupled blocks
that genuinely fall inside the loop become the driven block's `Blocking.inner`, so one
outer Newton step is one run of that interior's own `Schedule`.

`_audit/in_graph_rootfind.md` carries the block sizes, the answer agreement and the cost.
"""

import dataclasses
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import ConditionMap, Drive, Schedule
from cottax.graph import Graph
from cottax.plan import Insert, Plan
from cottax.problem import (
    Converged,
    DriverOut,
    Equality,
    FixedPoint,
    Inequality,
    Objective,
    Optimise,
    Residual,
    RootFind,
    Start,
    Steps,
)
from cottax.spec import In, NodePath, Out, VarPath
from cottax.tools.path import path_map
from jax.flatten_util import ravel_pytree
from jax.tree_util import GetAttrKey

from functional_process import sand
from functional_process.core.solver.drivers import (
    SeededNewtonDriver,
    # `Status` was written here, for `MdfNewtonDriver`, and moved to `drivers` when
    # `VmconDriver` and `SlsqpDriver` started reporting one too: a port kind belongs
    # beside the drivers that write it, not beside the one assembly that first read it.
    # Re-exported (`__all__`) so `mdf.Status` still resolves for every existing caller.
    Status,
    VmconDriver,
)
from functional_process.indat import graph_for
from functional_process.mda import (
    assign_drivers,
    cut_graph,
    default_drivers,
    given_start,
    guess_sources,
)
from functional_process.mda_harness import _without_excluded
from functional_process.sand_harness import ground_truth, run_schedule


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
    """What the outer driver is handed.

    For an `Optimise`: `(objective, *equalities, *inequalities)` -- `Optimise.inputs`'
    own order, which is the positional contract `VmconDriver` reads its split from.
    For a `RootFind`: the equalities alone, one per design variable.
    """
    n_equality: int
    n_inequality: int
    report: dict
    problem_type: type = Optimise
    """Which problem this file states -- `Optimise` or `RootFind`. See `assemble`."""
    reported: tuple[VarPath, ...] = ()
    """Conditions assembled but **not driven**: a `RootFind`'s inequalities.

    PROCESS does the same and it is not an afterthought there either: `_Fsolve.solve`
    root-finds `evaluate_eq_cons` (equalities only, `fcnvmc1(n, self.meq, ...)`) and then
    evaluates all `m` constraints once at the answer, so the inequalities are *reported*
    at a point they had no vote in choosing. Empty for an `Optimise`, whose inequalities
    are in `conditions`.
    """


def mdf_graph(graph, icc, n_equality, i_figure_merit, switch_values=None, omit=()):
    """`graph` with `sand.constraint_nodes`' and `sand.objective_node`'s nodes inserted.

    Reused wholesale from `sand.py` and deliberately not re-derived: which constraints
    are active, which of them are equalities (positional, from `n_equality`), which
    parameters are static switches and which resolve to `VarPath`s, and the objective's
    folded-in sign are all properties of *the run*, not of the formulation. An MDF
    assembly that answered any of them differently from the SAND one would make the two
    incomparable, which is the whole point of building this beside it.

    The one thing this does *not* do is insert the `Optimise`. See `nested_blocking`.

    **`i_figure_merit=None` mints no objective node at all**, and that is the honest
    shape for a file PROCESS runs in evaluation mode: `_Fsolve.solve` ends with
    `self.objf = None` and the output writer skips the figure-of-merit line entirely
    (`solver_handler.py:190-191`). A node computing PROCESS's `numerics.py:154` default
    of `7` would be inventing a quantity PROCESS never forms.

    Returns
    -------
    :
        `(graph, conditions, n_inequality, report)`. `conditions` is
        `(objective, *equalities, *inequalities)`, the objective absent when there is
        none.
    """
    nodes, equalities, inequalities, omitted = sand.constraint_nodes(
        graph, icc, n_equality, switch_values, omit
    )
    objective = None
    if i_figure_merit is not None:
        objective_name, objective_definition, objective = sand.objective_node(
            graph, i_figure_merit, switch_values
        )
        nodes[objective_name] = objective_definition
    inserted = (Plan(graph) + Insert(path_map(nodes.items()))).graph
    return (
        inserted,
        ((objective,) if objective is not None else ()) + (*equalities, *inequalities),
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
    root_find=False,
):
    """The whole MDF assembly: cut the raw cycles, add the conditions, build both
    schedules.

    `graph` defaults to `graph_for()` -- the reference configuration -- with
    `mda_harness.EXCLUDED_NODE_NAMES` deleted, which is exactly what
    `sand_harness.mda_env` feeds SAND. The exclusion (`DuctDiameterRootFind`, whose
    `VarPath`s no `DataStructure` field backs) is kept **for comparability, not because
    MDF needs it**: dropping it changes which graph is being optimised, and the two
    formulations have to optimise the same one.

    `root_find`: state a `RootFind` over the equalities instead of an `Optimise`
    -----------------------------------------------------------------------------
    **This is a different problem type, not a different tolerance**, and it is the one
    PROCESS states for a file whose `i_process_run_mode` is `-2`
    (`importer.Problem.is_evaluation`). PROCESS answers that mode by replacing VMCON with
    `scipy.optimize.fsolve` over `evaluate_eq_cons` -- the equalities *alone*
    (`fcnvmc1(n, self.meq, ...)`) -- forming no objective (`self.objf = None`) and
    letting the inequalities be evaluated once, at the answer, with no vote in choosing
    it.

    The port built an `Optimise` for those files anyway, and the cost was measured before
    this arm existed: `large_tokamak_eval`'s 2x2 square system reached VMCON as a
    2-variable design with 1 objective, 2 equalities and 23 inequalities, several of them
    infeasible at every point of the 2-dimensional feasible set (§21.3's
    "inequality-infeasible by construction" note -- which is this, not a quirk of
    that file), and `pyvmcon`'s first QP had none, so the row read `no-step`.

    `n_equality` must equal `len(ixc)` for a root find, and that is checked rather than
    assumed: squareness is a *consequence* of PROCESS's evaluation mode (the mode is
    what makes it a root find) and a file that states the mode without being square is
    stating something PROCESS's own `fsolve` call could not answer either.

    Raises
    ------
    ValueError
        Via `sand.constraint_nodes`, on any active constraint that cannot be assembled.
        Same policy and same reason as SAND's: an `Optimise` over 12 of PROCESS's 14
        active constraints is a different problem. Also when `root_find` is asked of a
        problem that is not square.
    """
    if root_find and n_equality != len(ixc):
        raise ValueError(
            f"a root find needs one equality per iteration variable, and this file "
            f"states {n_equality} equality constraint(s) against {len(ixc)} iteration "
            f"variable(s) -- PROCESS's own `fsolve` over `evaluate_eq_cons` would be "
            f"the same non-square system, so there is nothing to root-find here"
        )
    driven = cut_graph(_without_excluded(graph if graph is not None else graph_for()))
    graph, conditions, n_inequality, report = mdf_graph(
        driven,
        icc,
        n_equality,
        None if root_find else i_figure_merit,
        switch_values,
        omit,
    )
    drivers = default_drivers(graph)
    # Two algorithms over one structure means **two graphs** now, not two driver maps:
    # `Assign` puts the algorithm in the graph, so the eager and traceable variants are
    # separate objects. `reassign_drivers` is not needed -- neither graph carries a
    # driver yet, since `cut_graph` is structure only.
    eager_graph = assign_drivers(graph, drivers)
    blocking = Blocking.scc(eager_graph)
    design = tuple(sand.iteration_variable_path(i) for i in ixc)
    eager = Schedule(blocking)
    missing = [d for d in design if d not in eager.inputs]
    if missing:
        raise ValueError(
            f"design variable(s) {[d.path_str() for d in missing]} are not boundary "
            f"inputs of the MDA graph -- a node already produces them, so the optimiser "
            f"cannot own them (see `sand.optimise_graph` on the same conflict)"
        )
    report["blocks"] = len(blocking.blocks)
    report["driven_blocks"] = sum(1 for t in blocking.problem_types if t is not None)
    driven_conditions, reported = conditions, ()
    if root_find:
        # The equalities alone are driven; the inequalities stay in the graph so they
        # can be read at the answer, exactly as PROCESS reads them there.
        driven_conditions = tuple(report["equalities"])
        reported = tuple(report["inequalities"])
    return Mdf(
        graph=graph,
        eager=eager,
        traceable=Schedule(
            Blocking.scc(assign_drivers(graph, traceable_drivers(drivers)))
        ),
        design=design,
        conditions=driven_conditions,
        n_equality=n_equality,
        n_inequality=0 if root_find else n_inequality,
        report=report,
        problem_type=RootFind if root_find else Optimise,
        reported=reported,
    )


def guess_ports(mdf: Mdf) -> dict:
    """`{guess_port: unknown}` over every problem in `mdf.graph`.

    A driven problem's starting value is read from its `Start` port (`^guess.<place>`),
    not from the unknown's own name -- so both seeding (`seed`) and re-seeding (`prime`)
    have to write *there*, and what they write is the value of the unknown that port
    starts. Without this the ports fall to `ground_truth`'s `0.0` fallback, since no
    `DataStructure` field is spelled `^guess.*`, and the inner solvers start from the
    cold point `prime` exists to get them off.
    """
    # Asked of the **eager schedule's** graph, not `mdf.graph`. Since `Assign` mints the
    # `^guess.*` ports from the driver's own `requires`, a graph with no drivers on it has
    # no start ports at all -- and `mdf.graph` is the cut graph plus the `Optimise`,
    # deliberately undriven so that `Combine` can still join its problems. Asking that one
    # returns nothing, every port falls to `ground_truth`'s `0.0`, and the inner solves
    # start from exactly the cold point `prime` exists to get them off.
    return guess_sources(mdf.eager.blocking.graph)


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
    the model tree's business (`models/*/namespace.py`). It is the same defect class as
    `_audit/optimise_design.md` §10.5a/§10.5c (a missing producer no value test can see),
    found by a third method: a cold start rather than a gradient.
    """
    env = {}
    starts = guess_ports(mdf)
    for var in list(mdf.eager.inputs) + list(mdf.eager.unknowns):
        # A `^guess.*` port is grounded from the unknown it starts, not from its own
        # name -- there is no `DataStructure` field spelled that way.
        source = starts.get(var, var)
        try:
            grounded = ground_truth(data, source)
        except (AttributeError, KeyError):
            grounded = 0.0
        # A `^guess.*` port may be *given* its value instead of read off `data` --
        # `mda.GIVEN_STARTS` for which, and why a cold dataclass default is not a
        # starting guess. Guess ports only; an ordinary input is the machine's own number.
        if var in starts:
            grounded = given_start(source, grounded)
        env[var] = jnp.asarray(grounded)
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
    # Jitted, driver by driver: `Mdf.eager`'s `SeededNewtonDriver`s cannot be traced,
    # so the schedule is walked with every `Call` run and every `Drive` body fused and
    # the drivers left eager. 49.3 s / 978 XLA compiles -> 16.4 s / 32 on
    # `large_tokamak_nof` (`_audit/next_steps.md` §24.11).
    out = run_schedule(mdf.eager, _inputs_only(mdf, env))
    primed = dict(env)
    for guess, unknown in guess_ports(mdf).items():
        # Written to the `Start` port, which is where the driver reads it. Writing the
        # unknown's own name would set the *answer* and leave the guess cold.
        primed[guess] = out[unknown]
    return primed, out


def _inputs_only(mdf: Mdf, env):
    """`env` restricted to what the schedule may be handed: its own inputs.

    A `seed` env is also a value store -- it grounds the inner unknowns at their own
    names so callers can read a design point off it -- but a `Schedule` refuses a value
    at a name it owns (cottax's owned-name guard: an owned value could only be clobbered
    unread or, under an ordering bug, read stale in silence). So every schedule call
    filters at the door, and the store keeps its extra names for its other readers.
    """
    inputs = set(mdf.eager.inputs)
    return {var: value for var, value in env.items() if var in inputs}


def restart(mdf: Mdf, out):
    """The env a next pass starts from: the run's own inputs, every `Start` port
    re-seeded from the unknown its driver converged.

    What `prime` does to a seed env, applied to a full output env -- and the only way to
    hand a schedule its own output, since a `Schedule` refuses values at owned names.
    """
    env = {var: out[var] for var in mdf.eager.inputs if var in out}
    for guess, unknown in guess_ports(mdf).items():
        env[guess] = out[unknown]
    return env


class MdfConditionMap(ConditionMap):
    """`f(*design) -> conditions`, with a whole converged MDA inside every call.

    **Redundant since `~/jaxgraph` `33af0a5`, and kept for one more change anyway.**
    `ConditionMap.body` is a `Step` upstream now and `__call__` does
    `at = self.body(env)` -- which is what this subclass was written to do -- so a
    `ConditionMap` whose body is a `Schedule` is cottax's own object and needs no
    subclass. `_audit/in_graph_rootfind.md` §1 measures that; §7 says why removing it is
    a *separate* pass: a restructuring and a deletion landing together makes a moved
    answer unattributable.

    What it was: a `cottax.evaluate.ConditionMap` in every respect the driver can observe
    -- same `unknowns`/`conditions`/`context` fields, same `f(*unknowns) -> tuple`
    contract, so `VmconDriver` takes it unchanged -- differing in **one** thing, that
    `ConditionMap.__call__` used to run its body with `_run_acyclic`, which requires an
    acyclic body, and this runs it with a `Schedule`, which does not. `body` is kept here
    as the schedule's own graph in run order, so it stays a truthful answer to "what does
    this map compute", and `schedule` carries how.

    Note what this object is *not*, and what `in_graph_root_find` is for: its `schedule`
    is the **whole** MDA, so every residual evaluation re-runs every node of the graph,
    including the ~85 % that are either upstream of the design variables (constant during
    the solve) or downstream of the constraints (no vote in choosing `x`). A `Drive` over
    a block `Blocking.scc` chose runs only what is coupled.

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
        at = self.schedule.run(path_map(env))
        return tuple(at[condition] for condition in self.conditions)


def condition_map(mdf: Mdf, env, traceable=True) -> MdfConditionMap:
    """`f(*design) -> conditions` for `mdf`, everything else in `env` closed over.

    `env` must be a **primed** env (`prime`): it supplies the starting guess for every
    inner unknown, frozen for the whole solve (this module's docstring says why).
    """
    # Restricted to the schedule's own inputs (minus the design, supplied per call):
    # the primed env is also a value store carrying the inner unknowns at their own
    # names, and a `Schedule` refuses a value at an owned name (`_inputs_only`).
    design = set(mdf.design)
    context = {
        var: value for var, value in _inputs_only(mdf, env).items() if var not in design
    }
    # `roles` is cottax's own answer to what this module worked around with
    # `VmconDriver.n_equality`/`n_inequality`: the condition map now carries what each
    # condition *is*, parallel to `conditions`, so the split travels on the driver seam
    # instead of beside it (`_audit/optimise_design.md` §8, closed upstream). MDF's
    # order is the one `mdf_graph` assembles -- objective, equalities, inequalities --
    # and it is spelled here rather than counted by anyone.
    if issubclass(mdf.problem_type, RootFind):
        # Every condition vanishes at the answer, and none of them is an objective or a
        # one-sided bound -- which is precisely `RootFind.condition_roles`.
        roles = (Residual,) * len(mdf.conditions)
    else:
        n_equality = len(mdf.conditions) - 1 - mdf.n_inequality
        roles = (
            (Objective,) + (Equality,) * n_equality + (Inequality,) * mdf.n_inequality
        )
    return MdfConditionMap(
        body=mdf.traceable.subgraph,
        unknowns=mdf.design,
        conditions=mdf.conditions,
        roles=roles,
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


class MdfNewtonDriver(SeededNewtonDriver):
    """`SeededNewtonDriver` that reports optimistix's verdict instead of raising it.

    The algorithm is unchanged -- `optx.Newton` on the raveled residual, the same one
    that answers every `RootFind` inside the MDA -- and only two things move, both for
    the same reason: on this table a solve that did not converge has to be a *row*, not
    an exception (`run_cold_matrix`'s "a failure is a row, not an exit"). So
    `throw=False`, and `stats` is **reported**: `reports` names `Steps`, `Converged` and
    `Status`, so the verdict comes back through the ports `Assign` mints for them and
    lands in the env like every other value.

    **This class is where the `DriverOut` design was bought, so what it replaced is worth
    stating.** The three numbers used to leave through a `jax.debug.callback` writing
    into a mutable `Outcome` dict held as a driver field. That worked and was wrong twice
    over: it was a host effect smuggled through a traced program to carry what is plainly
    data, and the sink had to be a *field* to be reachable, which made the whole
    `Schedule` unhashable unless the dict was hashed by identity. Before that was
    understood, asking for a step count silently forfeited the whole-program jit --
    measured at 856 XLA compiles against 5. None of it is needed: a report is an ordinary
    output of this node, and a tracer is a perfectly good value to return.

    **`Start` is required and never fallen back on.** `SeededNewtonDriver`'s `seed`
    exists for the inner coil island, whose cold guess is a structural `0.0`; the outer
    design vector is seeded from the input file's own `ixc` values, which are real
    numbers a user chose, so there is nothing to fall back to and nothing that should.

    Bounds are **not** honoured, deliberately: `scipy.optimize.fsolve` (MINPACK `hybrd`)
    takes none either, so a bounded Newton here would be answering a different question
    from the one PROCESS answers in this mode. A root that leaves `boundl`/`boundu` is
    therefore visible in the design table rather than clipped out of it.
    """

    max_steps: int = 256

    @property
    def reports(self):
        """`(Steps, Converged, Status)` -- what `__call__` returns after the design.

        A property and not a `ClassVar` because that is what `AbstractDriver.reports` is:
        what an algorithm needs is a property of the algorithm, what is worth saying is a
        property of this use of it. This driver always says all three; a driver whose
        report costs something (a padded per-iteration buffer) makes it conditional on
        its own fields, which a class attribute could not express.
        """
        return (Steps, Converged, Status)

    def __call__(self, conditions: ConditionMap, data) -> tuple:
        """The root of `conditions` started from `data[Start]`, then the verdict.

        Returns the design values positionally, then `steps`, `converged` and the
        `optimistix.RESULTS` code -- `AbstractDriver.__call__`'s contract, which is the
        unknowns followed by one value per kind in `reports`.

        All three are jax arrays and stay so: converting them here (`int(...)`,
        `bool(...)`, `str(...)`) is what used to raise `TracerArrayConversionError` under
        a trace and cost this driver its jit. A caller renders them.

        Raises
        ------
        ValueError
            If no `Start` was supplied. Unlike `SeededNewtonDriver` there is no
            fallback: the outer design vector is seeded from the input file's own
            `ixc` values, so there is nothing to fall back to.
        """
        import optimistix as optx  # noqa: PLC0415 -- only this arm needs it

        start = data.get(Start)
        if start is None:
            raise ValueError(
                f"MdfNewtonDriver needs a starting value for every design variable "
                f"({', '.join(v.path_str() for v in conditions.unknowns)})"
            )
        flat_guess, unravel = ravel_pytree(start)

        def residual(flat, args=None):
            out, _ = ravel_pytree(conditions(*unravel(flat)))
            return out

        solution = optx.root_find(
            residual,
            optx.Newton(rtol=self.rtol, atol=self.atol),
            flat_guess,
            throw=False,
            max_steps=self.max_steps,
        )
        return (
            *unravel(solution.value),
            solution.stats["num_steps"],
            solution.result == optx.RESULTS.successful,
            getattr(solution.result, "_value", jnp.asarray(-1)),
        )


def root_find_driver(mdf: Mdf, **kwargs) -> MdfNewtonDriver:
    """The driver for a `RootFind` MDF -- `mdf.problem_type` decides, not the caller.

    `rtol`/`atol` default to `SeededNewtonDriver`'s `1e-4` on the *residual*, and the
    residuals here are PROCESS's own normalised constraint residuals, so `1e-4` is
    already a fraction of a percent on each equality. PROCESS's `fsolve` uses MINPACK's
    default `xtol = 1.49e-8` on the *step*, which is a tighter but incomparable rule;
    the honest comparison is the answer, and that is what the matrix column measures.

    The verdict needs no argument here any more: this driver `reports` it, so
    `solve` binds it into the env it returns and `verdict` reads it back out. There is
    no results sink to hand in, and therefore nothing that has to be a driver field.

    Raises
    ------
    TypeError
        If `mdf` states an `Optimise` -- `mdf.driver` is the one to build for that.
    """
    if not issubclass(mdf.problem_type, RootFind):
        raise TypeError(
            f"this MDF states an {mdf.problem_type.__name__}, not a RootFind -- "
            f"`mdf.driver` is the one to build"
        )
    return MdfNewtonDriver(**kwargs)


def solve(mdf: Mdf, env, bounds=(), callback=None, optimiser=None, **kwargs):
    """Drive the outer problem, then re-run the MDA at the answer.

    Exactly what `cottax.evaluate.Drive.__call__` does -- call the driver with a
    condition map and a start, put the answer back into the env, re-run the body so the
    block's internals land there too -- written out because the `Drive` this would be
    cannot be constructed (this module's docstring).

    `optimiser` overrides the `VmconDriver` this would otherwise build -- any
    `AbstractDriver` whose `drives` is `Optimise` will do, which is how the same MDF
    problem can be handed to a second SQP as a controlled comparison.

    **What the driver reports lands in `out`**, under the names `Assign` would mint for
    it at `IN_GRAPH_PLACE` -- so the outer-`solve` shape and the in-graph shape report
    the same verdict under the same keys and are directly comparable. `reported` is how
    a caller reads one; nothing here has to know which kinds a given driver names.

    Returns
    -------
    :
        `(x, out, seconds)` -- the design values in `mdf.design` order, the full output
        env of the MDA at that point (plus the driver's own reports), and the wall clock
        for the whole solve.
    """
    conditions = condition_map(mdf, env)
    start = tuple(jnp.asarray(env[var]) for var in mdf.design)
    if optimiser is None and issubclass(mdf.problem_type, RootFind):
        # A `RootFind` takes neither an objective nor a bound nor a per-iterate callback:
        # `bounds` and `callback` are dropped here rather than forwarded, so that a
        # caller passing the `Optimise` arm's arguments gets PROCESS's unbounded
        # `fsolve` semantics and not a silently different problem. The step count comes
        # back through the driver's own reports, in the env this returns (`verdict`).
        optimiser = root_find_driver(mdf, **kwargs)
    optimiser = optimiser or driver(mdf, bounds=bounds, callback=callback, **kwargs)
    started = time.perf_counter()
    # The driver is called directly here, not through a `Drive`, so the driver-data
    # mapping `Drive.role_data` would have built has to be built by hand: `Start` is
    # what `VmconDriver.requires` names, and the design values are what starts it.
    answered = optimiser(conditions, {Start: start})
    elapsed = time.perf_counter() - started
    # `Drive.__call__`'s own split, written out for the same reason the rest of this
    # function is: the driver returns its unknowns and then one value per kind in
    # `reports`, so the design is the first `len(mdf.design)` and the verdict is the
    # rest.
    x, verdict = answered[: len(mdf.design)], answered[len(mdf.design) :]
    at = dict(env)
    at.update(zip(mdf.design, x, strict=True))
    # Through `run_schedule`, not a bare `mdf.eager(...)`, for the reason `prime` a few
    # lines above already goes that way: a direct `Schedule.__call__` dispatches every
    # primitive eagerly and XLA compiles each one as its own module.
    # `_audit/optimise_design.md` §24.8 measured that tail at 268 compiles in an
    # MDF-only process and §24.9 left it alone, to avoid moving a second thing in
    # §24.1's pass. Measured here on a full `run_cold_matrix --native` row
    # (`stellarator_helias`, MDF then SAND, persistent cache OFF): **336 -> 277** XLA
    # compiles for the row, and this call itself **60 -> 1**. The gap to §24.8's 268 is
    # that a row runs SAND too, so jax's in-process cache already holds most of the
    # one-primitive programs by the time this line is reached -- inference from the
    # counts, not separately verified.
    # The row's result is unchanged to the table's precision (`objf 1.21775747` MDF /
    # `1.21775743` SAND, 108/169 iterations, identical `max|eq|`/`min ie`). NOT checked
    # bitwise: `sand_harness.mda_env`'s docstring records the jitted schedule differing
    # from the eager one at `1.1e-13` on 254 of 831 keys, so the reported env may move
    # in its last bits even where the table does not.
    out = run_schedule(mdf.eager, _inputs_only(mdf, at))
    out.update(
        zip(
            (kind.name_for(IN_GRAPH_PLACE) for kind in optimiser.reports),
            verdict,
            strict=True,
        )
    )
    return tuple(x), out, elapsed


def verdict(out, kind: type[DriverOut], place: NodePath = None):
    """What a driver said about its own run, out of the env a solve returned.

    Named `verdict` and not `reported`: `InGraphRootFind.reported` already means the
    inequalities evaluated at the answer, and one word for two things in one module is
    how a reader ends up debugging the wrong one.

    `None` where this run's driver did not report that kind -- a `VmconDriver` that
    names no `reports` leaves nothing behind, and asking is not an error.

    `place` defaults to `IN_GRAPH_PLACE`, which is where both shapes bind the outer
    problem: `in_graph_root_find` really does bind it there, and `solve` -- which drives
    by hand and has no node -- writes its reports under the same name so the two are
    read the same way.
    """
    return out.get(kind.name_for(IN_GRAPH_PLACE if place is None else place))


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

    **One row per unknown, reduced over its elements.** An inner unknown need not be a
    scalar -- the tokamak PF-coil ring is cut at `.pf_coil.n_pf_coil_turns` (a per-coil
    vector) and `.pf_coil.ind_pf_cs_plasma_mutual` (a circuit-by-circuit matrix), so
    `^problem.times.t_plant_pulse_burn.cycle` owns an array on every tokamak. This used
    to `float(np.asarray(env[unknown]))`, which raises `TypeError: only 0-dimensional
    arrays can be converted to Python scalars` on exactly those, so the one instrument
    for *"did the MDA converge"* had **never run on any tokamak configuration** -- the
    same defect class as `sand_harness.reference_run`'s array-element `ixc` crash
    (`d2890d90`), found the same way, by pointing an existing instrument at a second
    machine. The reduction keeps the element with the **worst relative** gap, since that
    is the element a caller sorting on the fourth column is asking about, and reports
    that element's signed gap beside it so the two columns describe the same number.
    A zero-sized unknown contributes no row: there is nothing to be unconverged about.
    """
    rows = []
    for step in schedule.steps:
        if not isinstance(step, Drive):
            continue
        values = step.condition_map(env)(*[env[u] for u in step.unknowns])
        fixed_point = issubclass(step.problem_type, FixedPoint)
        for unknown, value in zip(step.unknowns, values, strict=True):
            current = np.asarray(env[unknown], dtype=float)
            gap = np.asarray(value, dtype=float) - (current if fixed_point else 0.0)
            gap, current = np.broadcast_arrays(gap, current)
            if gap.size == 0:
                continue
            relative = np.abs(gap) / np.maximum(np.abs(current), 1e-30)
            worst = int(np.argmax(relative))
            rows.append((
                step.problem,
                unknown,
                float(gap.reshape(-1)[worst]),
                float(relative.reshape(-1)[worst]),
            ))
    return rows


def nested_blocking(ixc, icc, n_equality, i_figure_merit, graph=None, **kwargs):
    """MDF **stated as structure**: `Blocking.scc(graph + Optimise).nest(the Optimise)`.

    The returned graph carries **no drivers** -- `cut_graph` is structure only and
    `sand.optimise_graph` attaches none by default, because `Combine` refuses to join two
    problems that already carry an algorithm. So `schedule_for` on this blocking refuses
    it, and the refusal is *"carries no driver ... `Assign` is how the algorithm is
    said"*: a missing `Assign`, **not** a limit on nesting.

    That distinction is the whole of `_audit/in_graph_rootfind.md` §1. This function's
    docstring used to say "nothing runs this", and `test_mdf.py` pinned it with a regex
    loose enough to match the driver message -- so the claim survived the upstream change
    that falsified it. `assign_drivers(blocking.graph, default_drivers(...))` and then
    `Blocking.scc(...).nest(...)` schedules fine [measured].

    Returns
    -------
    :
        `(blocking, problem_name, report)` -- `blocking.inner[i]` at the `Optimise`'s
        block is the MDA, blocked in its own right.
    """
    driven = cut_graph(_without_excluded(graph if graph is not None else graph_for()))
    with_problem, problem_name, report = sand.optimise_graph(
        driven, ixc, icc, n_equality, i_figure_merit, **kwargs
    )
    return Blocking.scc(with_problem).nest(problem_name), problem_name, report


IN_GRAPH_PLACE = NodePath((GetAttrKey("RootFind"),))
"""Where the in-graph problem binds.

A plain `NodePath`, the same shape `sand.optimise_graph` binds its `Optimise` at
(`.Opt`), and deliberately **not** one of `rewrites`' minted problem namespaces:
`^problem.<var>` names a problem after the *one* variable it owns, and a root find over
two or three `ixc` entries has no single variable to be named from. `in_graph_root_find`
takes `place` so a caller assembling two of them in one graph can say where each goes.
"""


@dataclasses.dataclass(frozen=True)
class InGraphRootFind:
    """PROCESS's evaluation-mode root find, **stated as a node of the graph**.

    The contrast with `Mdf` + `solve` is the whole point, and it is structural rather
    than numerical (`_audit/in_graph_rootfind.md` §3 measures that the answer does not
    move):

    | | outer driver (`solve`) | in-graph (`in_graph_solve`) |
    |---|---|---|
    | where the problem is | nowhere: `Mdf.design`/`conditions` | a `RootFind` node |
    | what decides the loop | the caller, handing over the schedule | `Blocking.scc` |
    | nodes per residual | every node of the graph | the block, one problem in |
    | the coupled blocks inside | re-driven by the flat schedule | `Blocking.inner` |
    | upstream / downstream | re-evaluated per residual | run once, before / after |

    `mdf` is kept whole rather than unpacked because every downstream reader wants
    something from it -- `design`, `conditions`, `reported`, `report`, and `eager` for
    `prime` -- and a second copy of those fields would be a second thing to keep in step.
    """

    mdf: Mdf
    """The assembly this states -- a `RootFind` one (`assemble(root_find=True)`)."""
    graph: Graph
    """`mdf.graph` plus the `RootFind`, with every problem's driver `Assign`ed on."""
    blocking: Blocking
    """`Blocking.scc(graph).nest(problem)`: the SCC blocking, nested at the problem."""
    schedule: Schedule
    problem: NodePath

    def verdict(self, out, kind: type[DriverOut]):
        """What the outer driver said about its own run, out of a run's env.

        **A question about a run, not a field on the assembly**, which is the shape the
        `DriverOut` change buys. It used to be `self.outcome.get(...)` -- a mutable dict
        the driver wrote into through a `jax.debug.callback`, held as a field here
        because a `Drive` returns an env and the solver's diagnosis had nowhere else to
        go. It has somewhere now: the outer `MdfNewtonDriver` owns
        `^driver_out.<kind>.<problem>` and `Drive` binds it like any other output, so the
        verdict is in the env the run returns and this is a lookup.

        That also removes the reason `InGraphRootFind` could only be used once: an
        assembly is now a value with no run-state on it, so two runs of one assembly
        report their own step counts instead of overwriting one sink.
        """
        return out.get(kind.name_for(self.problem))

    def steps(self, out) -> int | None:
        """How many Newton steps the outer solve took, from that run's env."""
        return self.verdict(out, Steps)

    def successful(self, out) -> bool | None:
        """Optimistix's own verdict on the outer solve, from that run's env."""
        return self.verdict(out, Converged)

    @property
    def index(self) -> int:
        """Which block of `blocking` the root find is answered at."""
        return self.blocking.index[self.problem]

    @property
    def drive(self) -> Drive:
        """The outer `Drive` step -- the surface a caller measures a residual on."""
        return self.schedule.steps[self.index]

    @property
    def block(self) -> tuple:
        """The nodes `Blocking.scc` put in the driven block, the problem included."""
        return self.blocking.blocks[self.index]

    @property
    def interior(self) -> Blocking:
        """How that block is blocked one level down -- the block minus the root find."""
        return self.blocking.inner[self.index]

    @property
    def design(self) -> tuple[VarPath, ...]:
        """The run's `ixc`, in PROCESS's own order -- what the root find owns."""
        return self.mdf.design

    @property
    def conditions(self) -> tuple[VarPath, ...]:
        """The equalities alone -- what the root find reads, one per design variable."""
        return self.mdf.conditions

    @property
    def reported(self) -> tuple[VarPath, ...]:
        """The inequalities: in the graph, evaluated once at the answer, never driven."""
        return self.mdf.reported


def root_find_node(mdf: Mdf) -> RootFind:
    """The `RootFind` `mdf` states, as a cottax node: owns `design`, reads `conditions`.

    One condition per unknown, paired by declaration order -- which
    `Square.__check_init__` enforces and `assemble(root_find=True)` already guaranteed by
    refusing a non-square file. No `Start` port is declared here: driver data belongs to
    the driver, so `Assign` mints `^guess.<place>` when one is attached
    (`cottax.problem._check_no_driver_ports`).

    Raises
    ------
    TypeError
        If `mdf` states an `Optimise`. See the message.
    """
    if not issubclass(mdf.problem_type, RootFind):
        raise TypeError(
            f"this MDF states an {mdf.problem_type.__name__}, and only the `RootFind` "
            f"arm is stated in-graph here -- an `Optimise` nests just as well "
            f"(`_audit/in_graph_rootfind.md` §1 measures it), but its outer driver is a "
            f"`VmconDriver`, which does not trace, so that is a separate change"
        )
    return RootFind(
        inputs=tuple(In(c) for c in mdf.conditions),
        outputs=tuple(Out(v) for v in mdf.design),
    )


def in_graph_root_find(
    mdf: Mdf,
    place: NodePath = IN_GRAPH_PLACE,
    driver=None,
    traceable: bool = True,
    **kwargs,
) -> InGraphRootFind:
    """State `mdf`'s root find inside the graph and let `Blocking.scc` decide what it
    drives.

    Four steps, and none of them is a new mechanism -- every one is cottax's own:

    1. `Insert` the `RootFind` (`root_find_node`). It owns the `ixc` design variables,
       which were boundary inputs of `mdf.graph` (`assemble` refuses an assembly where
       they are not), so registering it turns free inputs into owned variables and
       changes nothing else.
    2. `Assign` a driver onto every problem, the new one included (`assign_drivers`).
       This is the step `nested_blocking` omits, and omitting it is what made
       `schedule_for` look like it refused the nesting.
    3. `Blocking.scc` -- **the finest blocking, and no caller decision at all.** What
       lands in the driven block is what the graph says is coupled to the root find, and
       that is the measurement this whole formulation is about.
    4. `.nest(place)` -- the root find answered at its own level, everything else in its
       block blocked again by `Blocking.scc`. The two coupled blocks that fall inside the
       loop become inner `Drive`s, so one outer Newton step is one run of the interior's
       `Schedule`.

    `traceable` clears every inner `SeededNewtonDriver`'s `seed` (`traceable_drivers`),
    matching what `condition_map` hands the outer driver today, so the two shapes
    differentiate the same function. It is kept as a switch rather than hardcoded because
    the defect that motivated it is fixed upstream (`core/solver/drivers._usable` opens
    with an `isinstance(flat, jax.core.Tracer)` test, `_audit/next_steps.md` §24.11); the
    default stays `True` only so that this change moves the structure and not the seeds.

    The verdict takes no argument: `MdfNewtonDriver` `reports` its step count, its
    convergence and its `RESULTS` code, so `Assign` mints `^driver_out.*.<place>` for
    them, `Drive` binds them into the env, and `built.steps(out)` reads one back. The
    schedule stays hashable because nothing mutable is a driver field any more, so the
    step count and `run_schedule`'s single jit were never really alternatives
    (`_audit/in_graph_rootfind.md` §6 measured the old trade at 856 compiles against 5).

    `**kwargs` reach `MdfNewtonDriver` (`rtol`, `atol`, `max_steps`); `driver` overrides
    it outright, for a caller comparing two algorithms over one structure.

    Raises
    ------
    ValueError
        If `place` already names a node of `mdf.graph`.
    """
    node = root_find_node(mdf)
    if place in mdf.graph.definitions:
        raise ValueError(
            f"{place!r} is already a node of this graph -- pass `place` to bind the "
            f"root find somewhere else"
        )
    with_problem = (Plan(mdf.graph) + Insert(path_map([(place, node)]))).graph
    drivers = default_drivers(with_problem)
    if traceable:
        drivers = traceable_drivers(drivers)
    # `default_drivers` already put a `SeededNewtonDriver` on the new `RootFind` -- it
    # dispatches on the problem type and cannot know this one is the outer solve. It is
    # replaced rather than skipped: `MdfNewtonDriver` reports optimistix's verdict
    # instead of raising it, which is what makes a non-converged outer solve a row.
    drivers[place] = driver or MdfNewtonDriver(**kwargs)
    assigned = assign_drivers(with_problem, drivers)
    blocking = Blocking.scc(assigned).nest(place)
    return InGraphRootFind(
        mdf=mdf,
        graph=assigned,
        blocking=blocking,
        schedule=Schedule(blocking),
        problem=place,
    )


def in_graph_inputs(built: InGraphRootFind, env):
    """The env `built.schedule` is handed, out of an `Mdf` seed/prime env.

    Two things move between the two shapes, and both are the same fact from opposite
    sides: the design variables are **owned** here rather than supplied, so their values
    have to arrive at the outer problem's `Start` ports (`^guess.<place>`, minted by
    `Assign`) instead of at their own names -- exactly the rule `seed`/`prime` already
    apply to every inner unknown, now applied to the outer one too. Everything else is
    carried across unchanged.

    A `Schedule` refuses a value at a name its own nodes own, so this filters to
    `schedule.inputs` at the door rather than handing the store on whole
    (`_inputs_only`'s reason, one level up).

    Raises
    ------
    KeyError
        If `env` answers no value for some schedule input, its own name and its
        unknown's both.
    """
    starts = guess_sources(built.graph)
    out, missing = {}, []
    for var in built.schedule.inputs:
        if var in env:
            out[var] = env[var]
        elif var in starts and starts[var] in env:
            # A `^guess.*` port is grounded from the unknown it starts. For the outer
            # root find that unknown is a design variable, whose value `seed` read off
            # the input file's own `ixc`.
            out[var] = env[starts[var]]
        else:
            missing.append(var)
    if missing:
        raise KeyError(
            f"no value for schedule input(s) "
            f"{[v.path_str() for v in missing]} -- `seed` then `prime` is what fills "
            f"this env, and a `^guess.*` port is filled from the unknown it starts"
        )
    return out


def _hashable(value) -> bool:
    """Whether `value` can be a key -- asked, never inferred from what it holds.

    `run_schedule` hashes the schedule it is given, and a `Schedule` reaches every
    driver field through `Graph.__hash__`, so the question is about the whole object and
    not about any one field a caller could enumerate.
    """
    try:
        hash(value)
    except TypeError:
        return False
    return True


def in_graph_solve(built: InGraphRootFind, env, whole=None):
    """Run the whole schedule: upstream once, the root find, downstream once.

    The counterpart of `solve`, and shorter for a structural reason rather than a
    stylistic one -- `solve` writes out by hand what `Drive.__call__` does, because the
    `Drive` it would be could not be constructed. Here it can, so the outer solve is one
    step of an ordinary `Schedule` walk and there is nothing to write out.

    **Which walk it is matters more than the restructuring does, at this size**, and the
    two are independent: run step by step, a `Schedule` re-enters Python once per node
    and XLA compiles once per `jnp` primitive (`_audit/optimise_design.md` §18.6), so the
    ~215 ordinary `Call` steps either side of the driven block dominate everything the
    restructuring saved. `run_schedule` is the fix and it is not this module's
    (`_audit/next_steps.md` §24.11): one jit for the whole run, the outer Newton in it.
    `whole` is forwarded to it; `whole=False` skips its single-jit probe.

    **The path is chosen on whether the schedule can actually be hashed**, asked by
    hashing it. `run_schedule` keys its whole-jit verdict and its runner groups on the
    schedule, so an unhashable one fails several frames in with a `TypeError` naming
    `dict`. Every driver this module builds is hashable today, and now trivially so --
    the verdict travels through `DriverOut` ports, so there is no mutable results sink on
    a driver to make a schedule unhashable in the first place. This stays as a guard
    against a hand-built driver carrying some other unhashable field, not a trade-off any
    caller has to make; it is *asked* rather than inferred, because inferring it from one
    particular field is what made the step count and the single jit look mutually
    exclusive when only that field ever made them so.

    Returns
    -------
    :
        `(x, out, seconds)` -- the design values in `built.design` order, the full output
        env, and the wall clock of the whole run. `built.steps(out)` and
        `built.successful(out)` read the driver's own verdict back out of that env.
    """
    inputs = in_graph_inputs(built, env)
    started = time.perf_counter()
    if _hashable(built.schedule):
        out = run_schedule(built.schedule, inputs, whole=whole)
    else:
        # The walk computes the same values by the same nodes in the same order -- it is
        # the cost that differs, not the answer (`_audit/in_graph_rootfind.md` §6).
        out = dict(built.schedule.run(path_map(inputs)))
    return (
        tuple(out[var] for var in built.design),
        out,
        time.perf_counter() - started,
    )


def in_graph_shape(built: InGraphRootFind) -> dict:
    """The assembly's shape -- `mdf_shape` plus what the blocking decided.

    `block` against `nodes` is the measurement the formulation exists for: how much of
    the graph a residual evaluation actually touches. `inner_blocks`/`inner_driven` here
    are the *driven block's* interior, not the whole MDA's -- `mdf_shape`'s are that.
    """
    interior = built.interior
    return {
        **mdf_shape(built.mdf),
        "graph_nodes": len(built.graph.nodes),
        "outer_blocks": len(built.blocking.blocks),
        "block": len(built.block),
        "body": len(built.drive.body.nodes),
        "interior_blocks": len(interior.blocks),
        "interior_driven": sum(1 for t in interior.problem_types if t is not None),
        "upstream": built.index,
        "downstream": len(built.blocking.blocks) - built.index - 1,
    }


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
    "IN_GRAPH_PLACE",
    "InGraphRootFind",
    "Mdf",
    "MdfConditionMap",
    "MdfNewtonDriver",
    "Status",
    "assemble",
    "central_difference",
    "condition_map",
    "driver",
    "evaluation",
    "in_graph_inputs",
    "in_graph_root_find",
    "in_graph_shape",
    "in_graph_solve",
    "inner_residuals",
    "jacobian",
    "mdf_graph",
    "mdf_shape",
    "nested_blocking",
    "prime",
    "root_find_driver",
    "root_find_node",
    "seed",
    "solve",
    "traceable_drivers",
    "verdict",
]
