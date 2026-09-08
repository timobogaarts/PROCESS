"""Turning `indat.GRAPH` into something that can actually be run."""

from cottax.blocking import Blocking
from cottax.evaluate import Schedule
from cottax.interfaces.pytree_namespace_module import resolve
from cottax.problem import (
    Driven,
    FixedPoint,
    Optimise,
    RootFind,
    Start,
    driver_vars,
    unknowns_of,
    is_fixed_point, is_optimise, is_root_find,
)
from cottax.rewrites import Assign, Cut, FixedPointCut, Supply, Undrive
from cottax.graph import Graph
from cottax.spec import ConditionNode, NodePath, VarPath
from cottax.tools.path import path_map, written
import jax.numpy as jnp
from jax.tree_util import GetAttrKey

from functional_process.cottax.core.solver.drivers import (
    PicardDriver,
    SeededNewtonDriver,
    VmconDriver,
)
from functional_process.cottax.indat import GRAPH
from functional_process.cottax.paths import fwbs, pf_coil, physics, tfcoil, times

CUTS = (
    resolve(physics.proton_rate_density, VarPath),
    resolve(physics.fusden_alpha_total, VarPath),
    resolve(physics.f_temp_plasma_electron_density_vol_avg, VarPath),
    resolve(fwbs.f_ster_div_single, VarPath),
    resolve(tfcoil.dx_tf_wp_primary_toroidal, VarPath),
    resolve(times.t_plant_pulse_burn, VarPath),
    resolve(pf_coil.ind_pf_cs_plasma_mutual, VarPath),
    resolve(pf_coil.n_pf_coil_turns, VarPath),
    resolve(tfcoil.dr_tf_plasma_case, VarPath),
)
"""The variables cut to turn each raw cross-node cycle into a declared `FixedPoint`."""


def cut_graph(graph=GRAPH):
    """`graph` (default: `indat.GRAPH`, the default-configuration graph), with every raw
    cycle in `CUTS` that actually exists in `graph` cut into a declared `FixedPoint`
    problem.
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
    declared = frozenset(graph.declared)
    for var in CUTS:
        if var not in graph.owners:
            # Not produced in this configuration at all -- `closing_readers` refuses
            # an unowned variable outright, and unowned is the strongest form of "no
            # cycle to cut here": `.times.t_plant_pulse_burn` is a *plain boundary
            # input* of the stellarator graph (its producer, `.tokamak.pulse.
            # burn_time`, is a tokamak node), where `dx_tf_wp_primary_toroidal` is
            # merely acyclic there.
            continue
        readers = graph.closing_readers(var)
        if not readers:
            continue  # this cycle does not exist in this configuration
        owner = graph.owners[var]
        key = next((i for i, c in enumerate(cycles) if owner in c), var)
        if key is not var and any(n in declared for n in cycles[key]):
            # **The SCC already declares its own problem, so it needs no cut.**
            # `Blocking` allows a block exactly one problem -- *"one driver answers one
            # problem, so `Combine` them into a single problem over every unknown, or
            # nest one inside the other"* -- and `cut_graph`'s whole job is to give a
            # problem to an SCC that has none. Where a `FixedPointFunction`'s declared
            # self-loop already sits inside the SCC, that job is done: the self-loop's
            # driver re-runs every other node of the block on each iterate, which is
            # exactly what a cut here would buy. Adding one anyway mints a *second*
            # problem in the same block and `Blocking` refuses it outright.
            #
            # This is the same shape as the `closing_readers` skip above -- a cut
            # applies where the cycle it names actually needs breaking -- and it is what
            # lets `.tfcoil.dr_tf_plasma_case` be one table entry serving every machine:
            # on `st_regression.IN.DAT` the TF case slot is `DrTfPlasmaCaseFromFraction`
            # (an `ExplicitFunction`, no loop) and the three-node SCC has no problem, so
            # the cut lands; on `large_tokamak_nof`/`_eval`/`low_aspect_ratio_DEMO` the
            # slot is `DrTfPlasmaCaseFromInput` (a `FixedPointFunction`) and its
            # `^problem.tokamak.cicc_superconducting_tf_coil.dr_tf_plasma_case` is in the
            # SCC, so the cut is skipped and those three graphs are bit-for-bit what they
            # were. Measured: without this guard all three raise *"declares 2
            # problems"*.
            continue
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
    """`(unknown, guess_port)` pairs for `problem`, in `owns` order."""
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
    return tuple(
        (unknown, start)
        for unknown, start in zip(unknowns_of(node), starts, strict=True)
        if start not in graph.owners
    )


def guess_sources(graph) -> dict:
    """`{guess_port: unknown}` over every problem in `graph`."""
    return {
        guess: unknown
        for problem in graph.declared
        for unknown, guess in starts_for(graph, problem)
    }


SUPPLIED_STARTS = {
    ".stellarator.wp_width_r_min": ".stellarator.wp_width_r_min_guess",
}
"""`unknown -> the graph-owned variable that is its starting guess`, applied by
`supply_starts` as a `cottax.rewrites.Supply` on the problem's `Start` port.
"""

GIVEN_STARTS = {
    ".pf_coil.ind_pf_cs_plasma_mutual": 1.0,
    ".pf_coil.n_pf_coil_turns": 100.0,
}
"""`unknown -> the value its `^guess.*` port is **given**`, applied by `given_start`."""


def given_start(unknown, fallback):
    """`GIVEN_STARTS`' value for `unknown`, shaped like `fallback`, or `fallback`."""
    from cottax.tools.minting import unminted  # noqa: PLC0415

    # Keyed on the **quantity**, not on the minted copy. A `FixedPointCut`'s unknown is
    # `^hat.pf_coil.n_pf_coil_turns`; the number PROCESS writes is for
    # `.pf_coil.n_pf_coil_turns`, and a table that had to spell the mint would be a table
    # about cottax's naming rather than about the machine. `unminted` is the same
    # normalisation `mda_harness._ground_truth` already applies for the same reason.
    given = GIVEN_STARTS.get(unminted(unknown).path_str())
    if given is None:
        return fallback
    return jnp.full_like(jnp.asarray(fallback, dtype=float), given)


ROOT_FIND_SEEDS = {
    # PROCESS's own starting value, `d = np.full(4, 1e-6)`
    # (`process/models/vacuum.py:379`) -- a flat constant there, so a flat constant
    # here. Every `VarPath` of this node is minted, so cold or warm there is nothing in
    # `data` to seed it from: this is its *only* starting guess, not a fallback.
    ".vacuum.d_duct": lambda context: (1.0e-6,),
}
"""Fallback starting guesses for `RootFind` unknowns, as `f(context) -> tuple`, used
only when the value seeded from `data` is unusable (see `SeededNewtonDriver`).
"""


def _var(context, path_str):
    """The `VarPath` in `context` spelled `path_str`."""
    for var in context:
        if var.path_str() == path_str:
            return var
    raise KeyError(
        f"{path_str} is not in this block's context, so no starting guess can be "
        f"derived from it"
    )


def _root_find_seed(conditions):
    """A `SeededNewtonDriver`'s starting guess for whichever block it is driving."""
    for var in conditions.unknowns:
        entry = ROOT_FIND_SEEDS.get(var.path_str())
        if entry is not None:
            return entry(conditions.context)
    raise KeyError(
        f"no starting guess for {written(conditions.unknowns)}, and the one seeded "
        f"from `data` was unusable -- add an entry to `ROOT_FIND_SEEDS`"
    )


def driven_graph(graph=GRAPH, **driver_options):
    """`cut_graph` with an algorithm attached to every problem: the runnable graph."""
    graph = cut_graph(graph)
    return assign_drivers(graph, default_drivers(graph, **driver_options))


def supply_starts(graph: Graph) -> Graph:
    """Point every `Start` port `SUPPLIED_STARTS` names at the node that computes it."""
    for problem in tuple(graph.declared):
        node = graph[problem]
        if not isinstance(node, Driven):
            continue
        onto = {}
        for unknown, start in starts_for(graph, problem):
            target = SUPPLIED_STARTS.get(unknown.path_str())
            if target is None:
                continue
            producer = next(
                (
                    (var, owner)
                    for var, owner in graph.owners.items()
                    if var.path_str() == target
                ),
                None,
            )
            if producer is None:
                continue  # no occupant of that slot produces it in this machine
            var, owner = producer
            if owner in graph.descendants([problem]):
                continue  # inside the block -- see this function's own docstring
            onto[start] = var
        if onto:
            graph = Supply(problem, path_map(onto)).apply(graph)
    return graph


def assign_drivers(graph: Graph, drivers: dict) -> Graph:
    """`Assign` each driver onto its problem, then `supply_starts`."""
    for problem, driver in drivers.items():
        graph = Assign(problem, driver).apply(graph)
    return supply_starts(graph)


def reassign_drivers(graph: Graph, drivers: dict) -> Graph:
    """Replace the algorithm on problems that already carry one: `Undrive`, then
    `Assign`.
    """
    for problem, driver in drivers.items():
        if isinstance(graph[problem], Driven):
            graph = Undrive(problem).apply(graph)
        graph = Assign(problem, driver).apply(graph)
    return supply_starts(graph)


def default_drivers(
    graph: Graph,
    bounds=(),
    callback=None,
    condition_scale=(),
    max_iter=None,
    optimiser=VmconDriver,
) -> dict:
    """One driver per **problem**, chosen mechanically by problem type Takes a `Graph`
    rather than a `Blocking`: since `Assign` puts the driver *in* the graph, the choice
    has to be made before there is a blocking to speak of -- and it never needed one,
    because the problem's own type is what decides.
    """
    drivers = {}
    for problem, definition in graph.definitions.items():
        if not isinstance(definition, ConditionNode) or isinstance(definition, Driven):
            continue
        if is_root_find(definition):
            drivers[problem] = SeededNewtonDriver(seed=_root_find_seed)
        elif is_fixed_point(definition):
            drivers[problem] = PicardDriver()
        elif is_optimise(definition):
            # Not passed as `max_iter=max_iter`: `None` here means *say nothing*, and
            # `VmconDriver.max_iter` is an `int` field with a default it would then be
            # handed instead of keeping.
            said = {} if max_iter is None else {"max_iter": max_iter}
            drivers[problem] = optimiser(
                n_equality=len(definition.equalities),
                n_inequality=len(definition.inequalities),
                bounds=bounds,
                callback=callback,
                condition_scale=condition_scale,
                **said,
            )
        else:
            raise TypeError(
                f"{problem!r} declares a {type(definition).__name__}, and "
                f"default_drivers has no default driver for that problem type -- "
                f"assign one explicitly"
            )
    return drivers


def schedule(graph=GRAPH) -> Schedule:
    """`graph` (default: `indat.GRAPH`), block by block, every cyclic block driven by
    its default driver.
    """
    driven = driven_graph(graph)
    blocking = Blocking.scc(driven)
    return Schedule(blocking)
