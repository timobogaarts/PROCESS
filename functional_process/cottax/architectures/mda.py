"""Turning `indat.GRAPH` into something that can actually be run."""

import jax.numpy as jnp
from cottax.execution import RunnableGraph
from cottax.execution.schedule import Schedule
from cottax.pytree.graph import Graph
from cottax.pytree.names import PathMap
from cottax.pytree.problem import (
    ConditionalNode,
    Driven,
    Start,
    driver_vars,
    is_fixed_point,
    is_optimise,
    is_root_find,
    unknowns_of,
)
from cottax.mdao_architectures import GaussSeidelMinimal
from cottax.pytree.plan import Plan
from cottax.pytree.rewrites import Assign, Supply, Undrive

from functional_process.cottax.architectures.drivers import (
    PicardDriver,
    SeededNewtonDriver,
    VmconDriver,
)
from functional_process.cottax.input.indat import GRAPH
from functional_process.cottax.paths import written
from functional_process.cottax.queries import declared

SCHEME = GaussSeidelMinimal()
"""How the raw graph's cycles are opened: a Gauss-Seidel sweep of each cycle in the
order that cuts the fewest variables, one consistency statement per cycle
(`cottax.mdao_architectures`). One Picard iterate of a cycle is then one sweep of it
in that order, which is how PROCESS's own idempotence loop runs."""


def cut_graph(graph=GRAPH, scheme=SCHEME):
    """`graph` (default: `indat.GRAPH`, the default-configuration graph) with every
    cycle opened by `scheme` and closed by a consistency statement of its own -- a
    `FixedPoint` over the copies, under `^mda`, with no driver yet. `Plan(graph) +
    scheme` is the same graph with the op recorded.
    """
    return (Plan(graph) + scheme).graph


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
        if start not in graph.graph.owners
    )


def guess_sources(graph) -> dict:
    """`{guess_port: unknown}` over every problem in `graph`."""
    return {
        guess: unknown
        for problem in declared(graph)
        for unknown, guess in starts_for(graph, problem)
    }


SUPPLIED_STARTS = {
    ".stellarator.wp_width_r_min": ".stellarator.wp_width_r_min_guess",
}
"""`unknown -> the graph-owned variable that is its starting guess`, applied by
`supply_starts` as a `cottax.pytree.rewrites.Supply` on the problem's `Start` port.
"""

GIVEN_STARTS = {
    ".pf_coil.ind_pf_cs_plasma_mutual": 1.0,
    ".pf_coil.n_pf_coil_turns": 100.0,
}
"""`unknown -> the value its `^guess.*` port is **given**`, applied by `given_start`."""


def given_start(unknown, fallback):
    """`GIVEN_STARTS`' value for `unknown`, shaped like `fallback`, or `fallback`."""
    from cottax.pytree.names import unminted  # noqa: PLC0415

    # Keyed on the **quantity**, not on the minted copy. A `FixedPointCut`'s unknown is
    # `^hat.pf_coil.n_pf_coil_turns`; the number PROCESS writes is for
    # `.pf_coil.n_pf_coil_turns`, and a table that had to spell the mint would be a table
    # about cottax's naming rather than about the machine. `unminted` is the same
    # normalisation `evaluate.ground_truth` already applies for the same reason.
    given = GIVEN_STARTS.get(unminted(unknown).spelling)
    if given is None:
        return fallback
    return jnp.full_like(jnp.asarray(fallback, dtype=float), given)


def seed_starts(schedule, env, exclude=()) -> dict:
    """`{start_port: value}` for every `Start` input of `schedule` whose unknown `env`
    holds, bar the unknowns in `exclude` (a solve's design variables), `given_start`
    applied. What a *nested* solve's start needs: its unknowns are not the outer drive's,
    so `evaluate.seed_block` leaves them at the cold value, and a Picard or Newton
    started from a cold zero does not converge.
    """
    exclude = set(exclude)
    guesses = guess_sources(schedule.executable.graph)
    return {
        port: given_start(unknown, env[unknown])
        for port in schedule.inputs
        if (unknown := guesses.get(port)) is not None
        and unknown not in exclude
        and unknown in env
    }


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
        if var.spelling == path_str:
            return var
    raise KeyError(
        f"{path_str} is not in this block's context, so no starting guess can be "
        f"derived from it"
    )


def _root_find_seed(conditions):
    """A `SeededNewtonDriver`'s starting guess for whichever block it is driving."""
    for var in conditions.unknowns:
        entry = ROOT_FIND_SEEDS.get(var.spelling)
        if entry is not None:
            return entry(conditions.context)
    raise KeyError(
        f"no starting guess for {written(conditions.unknowns)}, and the one seeded "
        f"from `data` was unusable -- add an entry to `ROOT_FIND_SEEDS`"
    )


def driven_graph(graph=GRAPH, scheme=SCHEME, **driver_options):
    """`cut_graph` with an algorithm attached to every problem: the runnable graph."""
    graph = cut_graph(graph, scheme)
    return assign_drivers(graph, default_drivers(graph, **driver_options))


def supply_starts(graph: Graph) -> Graph:
    """Point every `Start` port `SUPPLIED_STARTS` names at the node that computes it."""
    for problem in tuple(declared(graph)):
        node = graph[problem]
        if not isinstance(node, Driven):
            continue
        onto = {}
        for unknown, start in starts_for(graph, problem):
            target = SUPPLIED_STARTS.get(unknown.spelling)
            if target is None:
                continue
            producer = next(
                (
                    (var, owner)
                    for var, owner in graph.graph.owners.items()
                    if var.spelling == target
                ),
                None,
            )
            if producer is None:
                continue  # no occupant of that slot produces it in this machine
            var, owner = producer
            if owner in graph.graph.descendants([problem]):
                continue  # inside the block -- see this function's own docstring
            onto[start] = var
        if onto:
            graph = Supply(problem, PathMap(onto)).apply(graph)
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
    rather than an `ExecutableGraph`: since `Assign` puts the driver *in* the graph, the
    choice has to be made before there is a blocking to speak of -- and it never needed
    one, because the problem's own type is what decides.
    """
    drivers = {}
    for problem, definition in graph.definitions.items():
        if not isinstance(definition, ConditionalNode) or isinstance(definition, Driven):
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
    return Schedule(RunnableGraph(driven))
