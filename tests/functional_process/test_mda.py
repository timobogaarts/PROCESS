"""`mda.py` builds a runnable `Schedule` for the whole registered graph.

Not a numerical test (nothing here checks a value against PROCESS -- that's the
block-by-block comparison harness's job, a separate, larger piece) -- this only pins
the structural claim: every one of `indat.GRAPH`'s 11 SCCs, after the two
`FixedPointCut`s, declares exactly one problem, and every driven block gets a driver.
Both facts fail loudly (a `Schedule`/`Drive`/`Blocking` construction error) if either
stops being true -- see `Schedule`'s own docstring: "a `Schedule` that exists is
runnable."
"""

from cottax.blocking import Blocking
from cottax.problem import Driven, FixedPoint, RootFind
from cottax.rewrites import Cut

from functional_process.core.solver.drivers import PicardDriver, SeededNewtonDriver
from functional_process.indat import GRAPH
from functional_process.mda import (
    CUTS,
    ROOT_FIND_SEEDS,
    default_drivers,
    driven_graph,
    schedule,
)


def test_each_raw_cycle_is_fully_broken_by_its_own_cuts_and_no_fewer():
    """Every raw cycle's `CUTS` entries break it completely, and dropping any one of
    them leaves it cyclic -- i.e. the cut set is sufficient *and* minimal.

    Pinned so a future edit to a cycle's membership (a new node reading or owning
    something in it) is forced to re-check rather than silently keep a now-partial cut.
    That is not hypothetical: registering `FusionTotalsNoBeam` added a second
    `FusionRates -> PlasmaComposition` path and made the single
    `proton_rate_density` cut insufficient, which `Blocking` caught only because it
    refuses a block that is *"still cyclic with its problem(s) removed"*. This test
    now states the property directly. It re-derives `mda.CUTS`'s own claim rather than
    trusting the docstring.
    """
    for cycle in GRAPH.cycles:
        names = {n.path_str() for n in cycle}
        if any(n.startswith("^problem") for n in names):
            continue  # already a declared block, not a raw cycle to cut
        sub = GRAPH.subgraph(cycle)
        cutting_vars = [v for v in CUTS if v in sub.owners]
        assert cutting_vars, (
            f"cycle {sorted(names)} has no CUTS entry among its owned variables"
        )

        def cut_all(vars_, sub=sub):
            out = sub
            for v in vars_:
                readers = out.closing_readers(v)
                if not readers:
                    return None
                out = Cut(var=v, readers=readers).apply(out)
            return out

        assert cut_all(cutting_vars).is_acyclic, (
            f"cycle {sorted(names)} is still cyclic after cutting "
            f"{[v.path_str() for v in cutting_vars]}"
        )
        for dropped in cutting_vars:
            rest = [v for v in cutting_vars if v != dropped]
            without = cut_all(rest)
            assert without is None or not without.is_acyclic, (
                f"{dropped.path_str()} is redundant: cycle {sorted(names)} is already "
                f"acyclic without it, so it should not be in CUTS"
            )


def test_driven_graph_has_no_raw_cycles_left():
    """After both cuts, every genuinely *cyclic* block (more than one node) declares
    exactly one problem -- `Blocking.scc` would raise `Graph.problem_type`'s refusal
    otherwise. Singleton, acyclic blocks are expected to have `problem_type is None`
    ("run, not driven") -- most of the graph's 98 nodes are ordinary acyclic ones.
    """
    graph = driven_graph()
    blocking = Blocking.scc(graph)
    for block, problem_type in zip(blocking.blocks, blocking.problem_types, strict=True):
        if len(block) > 1:
            assert problem_type is not None, (
                f"cyclic block {block!r} declares no problem"
            )


def test_default_drivers_assigns_newton_to_root_find_and_picard_to_fixed_point():
    """Every driven block gets the driver matching its own declared problem type --
    still assigned by type, not per block. The `RootFind` driver is
    `SeededNewtonDriver` (a Newton with a fallback starting guess, see its docstring);
    which *guess* it falls back to is per-unknown, but the driver choice is not.
    """
    graph = driven_graph()
    blocking = Blocking.scc(graph)
    # Read off the graph, not from `default_drivers`: `driven_graph` has already
    # `Assign`ed every driver, and `default_drivers` skips a problem that carries one --
    # so asking it again would return an empty map. The driver is a property of the node
    # now, which is the whole point of the change.
    drivers = {
        name: node.driver
        for name, node in graph.definitions.items()
        if isinstance(node, Driven)
    }

    for problem, problem_type in zip(
        blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None:
            continue
        driver = drivers[problem]
        if issubclass(problem_type, RootFind):
            assert isinstance(driver, SeededNewtonDriver)
        elif issubclass(problem_type, FixedPoint):
            assert isinstance(driver, PicardDriver)


def test_every_root_find_unknown_has_a_fallback_starting_guess():
    """Every `RootFind` block in the graph can name a starting guess without `data`.

    Pinned because the failure it prevents is not local: seeded from a cold
    `DataStructure`, `Intersect`'s unknown is `0.0`, the residual is exactly flat there,
    and `optimistix` aborts -- taking the **whole schedule** down, not just its own
    block. A new `RootFind` with no `ROOT_FIND_SEEDS` entry would reintroduce that the
    moment anyone ran the port cold.
    """
    graph = driven_graph()
    blocking = Blocking.scc(graph)
    for problem, problem_type in zip(
        blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None or not issubclass(problem_type, RootFind):
            continue
        unknowns = graph[problem].owns
        assert any(u.path_str() in ROOT_FIND_SEEDS for u in unknowns), (
            f"{problem.path_str()} solves for "
            f"{[u.path_str() for u in unknowns]}, none of which has a "
            f"`ROOT_FIND_SEEDS` entry -- it would fail from a cold start"
        )


def test_schedule_builds_for_the_whole_graph():
    """The actual point of this module: one `Schedule` answering every block in
    `indat.GRAPH`, not just a hand-picked slice.
    """
    s = schedule()
    assert len(s.blocking.blocks) > 0
