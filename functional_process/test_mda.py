"""`mda.py` builds a runnable `Schedule` for the whole registered graph.

Not a numerical test (nothing here checks a value against PROCESS -- that's the
block-by-block comparison harness's job, a separate, larger piece) -- this only pins
the structural claim: every one of `total_process.GRAPH`'s 11 SCCs, after the two
`FixedPointCut`s, declares exactly one problem, and every driven block gets a driver.
Both facts fail loudly (a `Schedule`/`Drive`/`Blocking` construction error) if either
stops being true -- see `Schedule`'s own docstring: "a `Schedule` that exists is
runnable."
"""

from cottax.blocking import Blocking
from cottax.drivers import NewtonDriver
from cottax.problem import FixedPoint, RootFind
from cottax.rewrites import Cut

from functional_process.core.solver.drivers import PicardDriver
from functional_process.mda import CUTS, default_drivers, driven_graph, schedule
from functional_process.total_process import GRAPH


def test_both_cuts_are_the_only_variable_that_fully_breaks_their_own_cycle():
    """Pinned so a future edit to either cycle's membership (a new node reading/
    owning something in it) is forced to re-check this rather than silently keep a
    now-partial cut. Re-derives the claim `mda.CUTS`'s own docstring makes, rather
    than trusting it.
    """
    for cycle in GRAPH.cycles:
        names = {n.path_str() for n in cycle}
        if any(n.startswith("^problem") for n in names):
            continue  # already a declared block, not a raw cycle to cut
        sub = GRAPH.subgraph(cycle)
        cutting_vars = [v for v in CUTS if v in sub.owners]
        assert len(cutting_vars) == 1, (
            f"cycle {sorted(names)} should have exactly one CUTS entry among its "
            f"owned variables, found {len(cutting_vars)}"
        )
        (var,) = cutting_vars
        cut = Cut(var=var, readers=sub.closing_readers(var))
        assert cut.apply(sub).is_acyclic


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
    no bespoke per-block choice, since every block in this graph is one of exactly
    these two shapes.
    """
    graph = driven_graph()
    blocking = Blocking.scc(graph)
    drivers = default_drivers(blocking)

    for problem, problem_type in zip(
        blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None:
            continue
        driver = drivers[problem]
        if issubclass(problem_type, RootFind):
            assert isinstance(driver, NewtonDriver)
        elif issubclass(problem_type, FixedPoint):
            assert isinstance(driver, PicardDriver)


def test_schedule_builds_for_the_whole_graph():
    """The actual point of this module: one `Schedule` answering every block in
    `total_process.GRAPH`, not just a hand-picked slice.
    """
    s = schedule()
    assert len(s.blocking.blocks) > 0
