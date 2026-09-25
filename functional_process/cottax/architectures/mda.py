"""Turning `indat.GRAPH` into something that can actually be run."""

import jax.numpy as jnp
from cottax.execution.drivers.kinds import Start
from cottax.interfaces import (
    Assign,
    Graph,
    Le,
    Plan,
    Rename,
    RunnableGraph,
    Schedule,
    Undrive,
    is_driven,
    is_fixed_point,
    is_optimise,
    is_problem,
    is_equalities,
)
from cottax.mdao_architectures import (At, CloseFunctionalCycles, Closing, CutRule,
                                        GaussSeidelMinimal)
from cottax.pytree.mint import Minted
from cottax.pytree.path import NodePath, node_of
from jax.tree_util import GetAttrKey
from cottax.pytree.mint import unminted

from functional_process.cottax.architectures.drivers import (
    PicardDriver,
    SeededNewtonDriver,
    VmconDriver,
)
from functional_process.cottax.input.indat import GRAPH
from functional_process.cottax.paths import written
from functional_process.cottax.queries import declared

HAT, MDA = Minted("hat"), Minted("mda")
"""Where a cut copy lives (`^hat.<place>`) and where a closure binds (`^mda.<place>`)."""


class AsBefore(At):
    """How a closure is named: after the cut variable's own place for one cut
    (`^mda.<place>`), after the first cut's place under `.mda` for several
    (`^mda.<place>.mda`) -- the names every pin and every `^mda` lookup in this port
    were written against."""

    def __call__(self, group):
        var = next(iter(group))
        return MDA(
            node_of(var)
            if len(group) == 1
            else NodePath((*var.segments, GetAttrKey("mda")))
        )

    def __repr__(self):
        return "as_before"


CLOSING = Closing(HAT, AsBefore())
"""How a group of cuts is closed: copies under `^hat`, the closure named `AsBefore`."""


MAX_EXACT_ORDER = 18
"""The largest cycle `GaussSeidelMinimal` searches exactly; beyond it the scheme refuses
rather than guessing."""


def scheme(rule: CutRule) -> CloseFunctionalCycles:
    """`rule` as a cutting scheme of this port: every cycle of the bodies closed per
    cycle, copies under `^hat`, closures named `AsBefore`. An architecture is handed
    its `rule` and `closing`."""
    return CloseFunctionalCycles(rule, CLOSING)


SCHEME = scheme(GaussSeidelMinimal(MAX_EXACT_ORDER))
"""How the raw graph's cycles are opened: a Gauss-Seidel sweep of each cycle in the
order that cuts the fewest variables, one consistency problem per cycle
(`cottax.mdao_architectures`). One Picard iterate of a cycle is then one sweep of it
in that order, which is how PROCESS's own idempotence loop runs. An architecture is
handed its `rule` and `closing` and cuts inside itself."""


def cut_graph(graph=GRAPH, scheme=SCHEME):
    """`graph` (default: `indat.GRAPH`, the default-configuration graph) with every
    cycle opened by `scheme` and closed by a consistency statement of its own -- a
    fixed point over the copies, under `^mda`, with no driver yet. `Plan(graph) +
    scheme` is the same graph with the op recorded.
    """
    return (Plan(graph) + scheme).graph


def starts_for(graph, problem):
    """`(unknown, start place)` pairs for `problem`, in `unknowns` order.

    Empty where nothing drives the statement or its driver asks for no `Start`: the
    places a driver reads for itself are derived from `driver.requires`, so a statement
    nobody has assigned an algorithm to has none.
    """
    node = graph[problem]
    if not is_driven(node) or Start not in node.driver.requires:
        return ()
    slot = node.data[node.driver.requires.index(Start)]
    return tuple(
        (unknown, start)
        for unknown, start in zip(node.unknowns, slot, strict=True)
        if start not in graph.graph.owners
    )


def guess_sources(graph) -> dict:
    """`{start place: unknown}` over every problem in `graph`."""
    return {
        start: unknown
        for problem in declared(graph)
        for unknown, start in starts_for(graph, problem)
    }


SUPPLIED_STARTS = {
    ".stellarator.wp_width_r_min": ".stellarator.wp_width_r_min_guess",
}
"""`unknown -> the graph-owned variable that is its starting guess`, applied by
`supply_starts` as a `cottax.core.plan.Rename` of the problem's `Start` datum.
"""

GIVEN_STARTS = {
    ".pf_coil.ind_pf_cs_plasma_mutual": 1.0,
    ".pf_coil.n_pf_coil_turns": 100.0,
}
"""`unknown -> the value its `^guess.*` place is **given**`, applied by `given_start`."""


def given_start(unknown, fallback):
    """`GIVEN_STARTS`' value for `unknown`, shaped like `fallback`, or `fallback`."""
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
    """`{start place: value}` for every `Start` input of `schedule` whose unknown `env`
    holds, bar the unknowns in `exclude` (a solve's design variables), `given_start`
    applied. What a *nested* solve's start needs: its unknowns are not the outer drive's,
    so `evaluate.seed_block` leaves them at the cold value, and a Picard or Newton
    started from a cold zero does not converge.
    """
    exclude = set(exclude)
    guesses = guess_sources(schedule.executable.graph)
    return {
        place: given_start(unknown, env[unknown])
        for place in schedule.inputs
        if (unknown := guesses.get(place)) is not None
        and unknown not in exclude
        and unknown in env
    }


ROOT_FIND_SEEDS = {
    # `.vacuum.d_duct` had the only entry here -- PROCESS's own `d = np.full(4, 1e-6)`
    # (`process/models/vacuum.py:379`), needed because every `VarPath` of
    # `DuctDiameterRootFind` is minted and there was nothing in `data` to seed it from.
    # That node is no longer registered in any machine (`vacuum/namespace.py`), so the
    # entry went with it. `test_vacuum.py`'s test-only driver supplies its own start.
}
"""Fallback starting guesses for root-find unknowns, as `f(context) -> tuple`, used
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
    """Point every `Start` datum `SUPPLIED_STARTS` names at the node that computes it.

    After `Assign`, never before: the datum is a place the *driver's* naming derives, so
    it exists only on the `Driven`, and `Rename` refuses a place the node neither reads
    nor owns.
    """
    for problem in tuple(declared(graph)):
        node = graph[problem]
        if not is_driven(node):
            continue
        onto = []
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
                continue  # inside the block -- a driver reads its data before it runs
            onto.append((start, var))
        if onto:
            graph = Rename(problem, tuple(onto)).apply(graph)
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
        if is_driven(graph[problem]):
            graph = Undrive(problem).apply(graph)
        graph = Assign(problem, driver).apply(graph)
    return supply_starts(graph)


def relation_counts(definition) -> tuple[int, int]:
    """`(equalities, inequalities)` a statement holds, read off its relations' symbols.
    """
    inequalities = sum(1 for r in definition.relations if r.op is Le)
    return len(definition.relations) - inequalities, inequalities


def default_drivers(
    graph: Graph,
    bounds=(),
    callback=None,
    condition_scale=(),
    max_iter=None,
    optimiser=VmconDriver,
) -> dict:
    """One driver per **problem**, chosen mechanically by what makes an answer right.

    Takes a `Graph` rather than an `ExecutableGraph`: since `Assign` puts the driver
    *in* the graph, the choice has to be made before there is a proof to speak of --
    and it never needed one, because the statement's own shape is what decides. A
    requirement gets none: it determines nothing, and an architecture is what answers
    it.
    """
    drivers = {}
    for problem, definition in graph.definitions.items():
        if not is_problem(definition) or is_driven(definition):
            continue
        if is_optimise(definition):
            n_equality, n_inequality = relation_counts(definition)
            # Not passed as `max_iter=max_iter`: `None` here means *say nothing*, and
            # `VmconDriver.max_iter` is an `int` field with a default it would then be
            # handed instead of keeping.
            said = {} if max_iter is None else {"max_iter": max_iter}
            drivers[problem] = optimiser(
                n_equality=n_equality,
                n_inequality=n_inequality,
                bounds=bounds,
                callback=callback,
                condition_scale=condition_scale,
                **said,
            )
        elif is_fixed_point(definition):
            drivers[problem] = PicardDriver()
        elif is_equalities(definition):
            drivers[problem] = SeededNewtonDriver(seed=_root_find_seed)
        else:
            raise TypeError(
                f"{problem!r} states {definition!r}, and default_drivers has no "
                f"default algorithm for that shape -- assign one explicitly"
            )
    return drivers


def schedule(graph=GRAPH) -> Schedule:
    """`graph` (default: `indat.GRAPH`), block by block, every cyclic block driven by
    its default driver.
    """
    driven = driven_graph(graph)
    return Schedule(RunnableGraph(driven))
