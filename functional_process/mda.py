"""Turning `indat.GRAPH` into something that can actually be run.

Most of `indat.GRAPH`'s SCCs already declare a problem and need only a driver:
the structural `FixedPointFunction`/`ImplicitFunction` self-loop pairs, plus the coil
island (`Intersect`/`WindingPackIntersectInputs`/`WindingPackTotalSizePost`). Two are
**raw cross-node cycles with no declared problem at all** -- `Blocking` finds them;
nobody has said what solves them. A `Drive` refuses such a block outright
(`cottax.evaluate.Drive.__check_init__`: *"block ... declares no problem: it is run,
not driven"*), so the graph is not runnable until something says what closes them.

The two are `Divertor`/`AFwTotalWithPowerflow` (`ipowerflow != 0` only) and the
density/fusion/composition cycle around `DensityProfile`/`FusionRates`/
`PlasmaComposition`/`ParabolicOnAxisDensities`.

This module does two things: cut those raw cycles into declared `FixedPoint` problems
(via `cottax.rewrites.FixedPointCut`, using `Graph.closing_readers` to find the cut and
an empirical check that the cut set actually breaks the whole cycle -- neither is
guesswork, both are computed), and assign a driver to every block automatically, by
problem type.

**Deliberately no node/SCC/cut counts here.** They moved on every porting wave, and a
docstring stating last wave's number is worse than one stating none -- five different
node counts, all present tense, once coexisted in a single audit document. The cycles
and their cuts are named in `CUTS` below and re-derived by
`test_mda.py::test_each_raw_cycle_is_fully_broken_by_its_own_cuts_and_no_fewer`, which
fails if either membership changes.
"""

from cottax.blocking import Blocking
from cottax.evaluate import Schedule, schedule_for
from cottax.interfaces.pytree_namespace_module import resolve
from cottax.problem import (
    Driven,
    FixedPoint,
    Optimise,
    RootFind,
    Start,
    driver_vars,
)
from cottax.rewrites import Assign, Cut, FixedPointCut, Undrive
from cottax.graph import Graph
from cottax.spec import DeclaredNode, NodePath, VarPath
from cottax.tools.path import written
from jax.tree_util import GetAttrKey

from functional_process.core.solver.drivers import (
    PicardDriver,
    SeededNewtonDriver,
    VmconDriver,
)
from functional_process.indat import GRAPH
from functional_process.paths import fwbs, physics

CUTS = (
    resolve(physics.proton_rate_density, VarPath),
    resolve(physics.fusden_alpha_total, VarPath),
    resolve(fwbs.f_ster_div_single, VarPath),
)
"""The variables cut to turn each raw cross-node cycle into a declared `FixedPoint`.

**`fusden_alpha_total` is the density/fusion cycle's *second* cut**, added when
`FusionTotalsNoBeam` gave `.physics.fusden_total`/`.fusden_alpha_total`/`.p_dt_total_mw`
their first producers (`_audit/boundary_inputs_audit.md` §4c (b7)/(b8)). That edge
(`FusionRates -> FusionTotalsNoBeam -> PlasmaComposition`) runs parallel to the one
`proton_rate_density` already cut, so one cut no longer breaks the cycle: `Blocking`
raised *"still cyclic with its problem(s) removed"* until this was added. Which second
cut to use was **measured, not chosen** -- of all 42 variables owned inside the enlarged
6-node cycle, `.physics.fusden_alpha_total` is the only one that makes the cycle acyclic
when paired with `proton_rate_density`, and no single variable does it alone.

**Watch this one on a cold start.** `PlasmaComposition` branches on
`fusden_alpha_total < 1e-6` as a "not yet calculated" bootstrap
(`composition.py:203-210`), so cutting it makes a Picard iterate drive a
*branch predicate*, not just a value. Seeded from a converged run (every harness here)
the branch never flips; from a cold `DataStructure` it starts on the other side.

`proton_rate_density` and `f_ster_div_single` were each, when added, the *only*
single-variable cut (out of every variable owned inside their own cycle) that made that
cycle's subgraph fully acyclic on its own -- checked directly, not assumed, by cutting
each candidate in turn and checking `.is_acyclic` on the result. `proton_rate_density`
(owned by `FusionRates`, read by `PlasmaComposition`) was sufficient for the
density/fusion/pedestal/composition loop as it stood then, before `FusionTotalsNoBeam`
enlarged it; `f_ster_div_single` (owned by `Divertor`, read by
`AFwTotalWithPowerflow`) still is, on its own, for the divertor/first-wall loop -- see
`_audit/next_steps.md` §5 for the cycle's own discovery. Neither is a `Feasibility`/
`Optimise` question: both are genuine "PROCESS iterates this to a fixed point"
couplings, so `FixedPointCut` (not `RootFindCut`) is the right closure -- matching
PROCESS's own `Caller.call_models`, which re-runs its whole pipeline up to 10 times
and checks idempotence, the same shape as a Picard iteration over these two blocks.

**The second cycle is `ipowerflow != 0`-only** (`next_steps.md` §5:
`AFwTotalWithPowerflow` is the `ipowerflow != 0` arm; `AFwTotalNoPowerflow` -- the
`ipowerflow == 0` arm -- does not read `.fwbs.f_ster_div_single` at all, so there is
no cycle to cut there). `driven_graph` below only attempts a cut whose variable
actually has closing readers in the `graph` it was given, so a graph built with
`ipowerflow == 0` skips this cut cleanly rather than raising `Cut`'s own "no readers"
refusal. The default `GRAPH` (this module's default argument) has `ipowerflow == 1`,
so both cuts apply there.
"""


def cut_graph(graph=GRAPH):
    """`graph` (default: `indat.GRAPH`, the default-configuration graph), with
    every raw cycle in `CUTS` that actually exists in `graph` cut into a declared
    `FixedPoint` problem. Every remaining multi-node SCC now declares exactly one
    problem -- confirmed below, not assumed, by `driven_graph()`'s own doctest-free
    but exercised construction (`schedule()` calling `Blocking.scc` on this graph
    would raise if not).

    Takes `graph` rather than always using the module's default so a caller checking
    this against a specific real `IN.DAT` can pass `indat.graph_for(
    configuration_matching_that_file)` instead -- which nodes exist at all (and
    therefore which of `CUTS` even applies -- see that tuple's own docstring on the
    `ipowerflow`-gated second cycle) can differ by configuration, so this should be
    run against whichever graph is actually being checked, not silently default to
    the wrong one.
    """
    # Cuts are grouped by the cycle they break, and each group becomes **one**
    # `FixedPointCut` -- i.e. one `FixedPoint` problem over however many unknowns that
    # cycle needed. Applying them one at a time instead mints one problem per cut, and
    # `Blocking` then refuses the block outright: *"declares 2 problems -- one driver
    # answers one problem, so `Combine` them into a single problem over every unknown,
    # or nest one inside the other. Which of those is a modelling decision"*. It is,
    # and this is the decision: PROCESS iterates its whole pipeline to idempotence, so
    # the two cut variables of the density/fusion cycle are two unknowns of one Picard
    # iteration, not two nested loops.
    #
    # Every `closing_readers` call is made on the **uncut** graph, before any of the
    # group is applied, so the readers a cut re-routes are the ones the original cycle
    # had rather than ones a sibling cut already moved.
    by_cycle: dict = {}
    cycles = [frozenset(c) for c in graph.cycles]
    for var in CUTS:
        readers = graph.closing_readers(var)
        if not readers:
            continue  # this cycle does not exist in this configuration
        owner = graph.owners[var]
        key = next((i for i, c in enumerate(cycles) if owner in c), var)
        by_cycle.setdefault(key, []).append(Cut(var=var, readers=readers))
    for cuts in by_cycle.values():
        # One cut keeps its historical name (`^problem.physics.proton_rate_density`);
        # several need an explicit `place`, since no single variable names what closes
        # them. Named after the first cut's own place with a `.cycle` component, which
        # is unique (a variable is cut at most once) and reads as what it is:
        # `^problem.physics.proton_rate_density.cycle`.
        place = (
            None
            if len(cuts) == 1
            else NodePath((*cuts[0].var.keys, GetAttrKey("cycle")))
        )
        graph = FixedPointCut(tuple(cuts), place=place).apply(graph)

    # Every problem gets `Start` ports, one per unknown, read from `^guess.<place>`.
    #
    # `cottax.evaluate.AbstractDriver` takes its starting values as *declared driver
    # data* rather than reading them off the unknowns' own names: `Drive.role_data`
    # walks the driver's `requires` and looks up the ports the problem declares, and
    # `Drive.__check_init__` refuses both directions -- a driver requiring a kind the
    # problem lacks, and a kind declared but not consumed. Every driver this port
    # **The driver is part of the graph now.** `Assign` retypes each problem into a
    # `Driven` -- problem plus algorithm -- and *mints* the ports that algorithm needs
    # from its own `requires`: a Newton wants a `Start`, so `^guess.<place>` appears per
    # unknown; a Picard wants nothing and nothing appears. That is one op where this used
    # to need two (`Initialise` to declare the ports, then a separate `{problem: driver}`
    # map handed to `schedule_for`), and it removes the failure mode between them --
    # ports declared before the algorithm was known could be required-but-undeclared or
    # declared-but-unconsumed, and both are now unrepresentable rather than refused.
    #
    # It stays here rather than in `schedule()` because the minted ports are real
    # boundary inputs: a caller measuring this graph's boundary, or drawing it, must see
    # them. Assigning is a modelling decision and is recorded in `Plan.ops` like any
    # other, so it survives `subgraph`/`prune` without a side table.
    return graph


def starts_for(graph, problem):
    """`(unknown, guess_port)` pairs for `problem`, in `owns` order.

    A driven problem's starting values are no longer read off the unknowns' own names:
    `Initialise` (applied by `driven_graph`) gives every problem one `Start` port per
    unknown, named `^guess.<place>`, and `Drive.role_data` reads the start from *there*.
    So a caller seeding a run writes `^guess.physics.temp_plasma_ion_vol_avg_kev`, not
    `.physics.temp_plasma_ion_vol_avg_kev` -- the latter is the answer, and writing it
    would be seeding the output.

    The pairing is read off the node rather than re-derived with `Initialise.start_of`:
    `Start`s pair with `owns` by declaration order (`cottax.problem._check_starts`), so
    the node itself is the authority on which guess belongs to which unknown, and
    `strict=True` fails loudly if that ever stops being true.
    """
    node = graph[problem]
    starts = driver_vars(node, Start)
    if not starts:
        # **No driver, or a driver that needs no start: no ports, and that is legitimate
        # now.** `Assign` mints driver data from the algorithm's own `requires`, so a
        # problem that has not been assigned one has no `Start` to pair with -- where the
        # old `Initialise` gave every problem a port before any algorithm was chosen, and
        # a missing port could only mean a bug. `strict=True` below still catches the
        # real error, a partially-ported problem.
        return ()
    return tuple(zip(node.owns, starts, strict=True))


def guess_sources(graph) -> dict:
    """`{guess_port: unknown}` over every problem in `graph`.

    The inverse lookup every seeding site needs: given a `^guess.*` input, which
    unknown's value belongs in it. Nothing in a `DataStructure` is spelled `^guess.*`,
    so a seeder that grounds these ports by their own name silently writes `0.0` into
    every starting guess -- which is not a slow solve but an impossible one, and is
    exactly the failure `run_sand_harness._seed`'s own docstring describes.
    """
    return {
        guess: unknown
        for problem in graph.declared
        for unknown, guess in starts_for(graph, problem)
    }


ROOT_FIND_SEEDS = {
    ".stellarator.wp_width_r_min": lambda context: (
        (context[_var(context, ".stellarator.r_coil_minor")] / 10.0) ** 2,
    ),
    # PROCESS's own starting value, `d = np.full(4, 1e-6)`
    # (`process/models/vacuum.py:379`) -- a flat constant there, so a flat constant
    # here. Every `VarPath` of this node is minted, so cold or warm there is nothing in
    # `data` to seed it from: this is its *only* starting guess, not a fallback.
    ".vacuum.d_duct": lambda context: (1.0e-6,),
}
"""Fallback starting guesses for `RootFind` unknowns, as `f(context) -> tuple`, used
only when the value seeded from `data` is unusable (see `SeededNewtonDriver`).

`wp_width_r_min`'s entry is **PROCESS's own guess**, not a fitted constant:
`winding_pack_pre_intersect` computes
`(r_coil_minor / (20 if i_tf_sc_mat == 6 else 10)) ** 2` at `coils/calculate.py:737`
and hands it to `intersect` as `xin`. The `20` arm is `i_tf_sc_mat == 6` (REBCO), which
`WindingPackIntersectInputs` is not registered with in any configuration here; if that
changes, this needs the same branch, and `switch_audit` will not catch it because a
driver carries no static switch kwarg. Recorded rather than guarded, because a wrong
starting guess is a slower solve, not a wrong answer -- unlike a wrong switch.

Keyed by `path_str()` rather than by `VarPath` so the table reads as a table; resolved
against the block's own context at use, which is also the check that the block really
does close over what the guess needs.
"""


def _var(context, path_str):
    """The `VarPath` in `context` spelled `path_str`.

    Raises
    ------
    KeyError
        If the block does not close over it -- a seed that silently fell back would be
        indistinguishable from no seed at all.
    """
    for var in context:
        if var.path_str() == path_str:
            return var
    raise KeyError(
        f"{path_str} is not in this block's context, so no starting guess can be "
        f"derived from it"
    )


def _root_find_seed(problem):
    """The seed for the block whose problem is `problem`, or `None`.

    Matched on the problem's own name, which for a `FixedPointCut`/`ImplicitFunction`
    is minted from the unknown's place -- so `^problem['Intersect']` is matched by the
    *unknown* it owns, resolved from the context at call time.
    """

    def seed(conditions):
        for var in conditions.unknowns:
            entry = ROOT_FIND_SEEDS.get(var.path_str())
            if entry is not None:
                return entry(conditions.context)
        raise KeyError(
            f"no starting guess for {written(conditions.unknowns)}, and the one seeded "
            f"from `data` was unusable -- add an entry to `ROOT_FIND_SEEDS`"
        )

    return seed


def driven_graph(graph=GRAPH, **driver_options):
    """`cut_graph` with an algorithm attached to every problem: the runnable graph.

    **Split from `cut_graph` because the two are different decisions and one caller needs
    only the first.** Cutting a cycle is structure -- it says these nodes are coupled and
    this variable closes the loop. Assigning a driver is an algorithm choice. They used
    to be one function because a driver lived in a side map and could be chosen last;
    now it lives *in the graph*, and `Combine` refuses to join two problems that carry
    one (*"combining two problems discards the algorithm answering each -- `Undrive`
    first"*). `sand` joins its `FixedPoint`s into one `Optimise`, so it must build on the
    cut graph and assign afterwards. That refusal is what made the seam visible; it was
    always there.
    """
    graph = cut_graph(graph)
    return assign_drivers(graph, default_drivers(graph, **driver_options))


def assign_drivers(graph: Graph, drivers: dict) -> Graph:
    """`Assign` each driver onto its problem, returning the graph that carries them.

    The two-line idiom every call site needs now that `schedule_for` takes no drivers:
    choose (`default_drivers`), then attach. Kept as a function rather than inlined
    because *re*-assigning is a different operation -- see `reassign_drivers` -- and a
    caller that conflates them silently replaces one algorithm with another.
    """
    for problem, driver in drivers.items():
        graph = Assign(problem, driver).apply(graph)
    return graph


def reassign_drivers(graph: Graph, drivers: dict) -> Graph:
    """Replace the algorithm on problems that already carry one: `Undrive`, then `Assign`.

    `Assign` refuses a problem that is already `Driven`, deliberately -- *"replacing one
    algorithm with another silently is not a rewrite"* -- so swapping a driver is two
    recorded ops, not an overwrite. `mdf` needs exactly this: it runs one blocking under
    two algorithms (a seeded eager solve and a traceable one), which used to be two
    `{problem: driver}` maps over one graph and is now two graphs, because the algorithm
    is part of the graph.
    """
    for problem, driver in drivers.items():
        if isinstance(graph[problem], Driven):
            graph = Undrive(problem).apply(graph)
        graph = Assign(problem, driver).apply(graph)
    return graph


def default_drivers(
    graph: Graph, bounds=(), callback=None, condition_scale=()
) -> dict:
    """One driver per **problem**, chosen mechanically by problem type

    Takes a `Graph` rather than a `Blocking`: since `Assign` puts the driver *in* the
    graph, the choice has to be made before there is a blocking to speak of -- and it
    never needed one, because the problem's own type is what decides. A node that already
    carries a driver (`Driven`) is skipped rather than re-assigned, which `Assign` would
    refuse anyway: replacing one algorithm with another silently is not a rewrite.

    Historic shape -- a Newton
    for every `RootFind` (`Intersect`, `DuctDiameterRootFind`), a Picard for every
    `FixedPoint` (the 8 structural self-loops plus the two cuts above), and a
    `VmconDriver` for an `Optimise` (only `functional_process.sand` registers one). No
    bespoke per-block choice: every block in this graph is one of exactly these shapes.

    **The `Optimise` arm is where the equality/inequality split is settled.**
    `ConditionMap` carries a flat condition tuple with no type information
    (`~/jaxgraph/src/cottax/evaluate.py:135-178`), so a driver cannot ask which of its
    conditions is the objective. It does not have to guess either: this function has the
    problem's own definition in hand, so the counts are **read off `Optimise.equalities`/
    `Optimise.inequalities`** and never counted by a caller. That is the whole of
    `_audit/optimise_design.md` §4.1's "positional contract" worry, removed by asking the
    node instead of the reader.

    `bounds`/`callback`/`condition_scale` are forwarded to the `VmconDriver` and ignored
    by the others -- all three are algorithm choices with no home on `Optimise` (see
    `VmconDriver`'s docstring on why bounds are not extra inequality constraints, and on
    why the residual equalities need scaling that PROCESS's own constraints must not
    get).

    Raises
    ------
    TypeError
        If a block declares a problem type none of the three answers -- e.g. a
        `Feasibility`, which this graph does not currently register any of.
    """
    drivers = {}
    for problem, definition in graph.definitions.items():
        if not isinstance(definition, DeclaredNode) or isinstance(definition, Driven):
            continue
        if isinstance(definition, RootFind):
            drivers[problem] = SeededNewtonDriver(seed=_root_find_seed(problem))
        elif isinstance(definition, FixedPoint):
            drivers[problem] = PicardDriver()
        elif isinstance(definition, Optimise):
            drivers[problem] = VmconDriver(
                n_equality=len(definition.equalities),
                n_inequality=len(definition.inequalities),
                bounds=bounds,
                callback=callback,
                condition_scale=condition_scale,
            )
        else:
            raise TypeError(
                f"{problem!r} declares a {type(definition).__name__}, and "
                f"default_drivers has no default driver for that problem type -- "
                f"assign one explicitly"
            )
    return drivers


def schedule(graph=GRAPH) -> Schedule:
    """`graph` (default: `indat.GRAPH`), block by block, every cyclic block
    driven by its default driver. `Drive`/`Schedule`'s own construction is what
    checks this is actually runnable -- a `Schedule` that builds is a `Schedule` that
    can be called. See `driven_graph`'s own docstring for why `graph` is a parameter,
    not always the module default.
    """
    driven = driven_graph(graph)
    blocking = Blocking.scc(driven)
    return schedule_for(blocking)
