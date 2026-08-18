"""Turning `total_process.GRAPH` into something that can actually be run.

`total_process.GRAPH` is 96 nodes and 11 SCCs, but 2 of those 11 are still raw
cross-node cycles with no declared problem at all (`Blocking` found them; nobody has
said what solves them) -- `Divertor`/`AFwTotalWithPowerflow` and
`DensityProfile`/`FusionRates`/`PedestalOnAxisDensities`/`PlasmaComposition`. A
`Drive` refuses to run a block that declares zero problems (`cottax.evaluate.Drive.
__check_init__`: *"block ... declares no problem: it is run, not driven"*), so those
two are not runnable as-is -- everything else (the 8 structural `FixedPointFunction`/
`ImplicitFunction` self-loop pairs, plus `Intersect`/`WindingPackIntersectInputs`/
`WindingPackTotalSizePost`) already declares exactly one problem per block and needs
only a driver.

This module does two things: cut the two raw cycles into declared `FixedPoint`
problems (via `cottax.rewrites.FixedPointCut`, using `Graph.closing_readers` to find
the minimal cut and an empirical single-variable check that it actually breaks the
whole cycle -- neither is guesswork, both are computed), and assign a driver to every
block automatically, by problem type.
"""

from cottax.blocking import Blocking
from cottax.drivers import NewtonDriver
from cottax.evaluate import Schedule, schedule_for
from cottax.interfaces.pytree_namespace_module import path_of
from cottax.problem import FixedPoint, RootFind
from cottax.rewrites import Cut, FixedPointCut
from cottax.spec import VarPath

from functional_process.core.solver.drivers import PicardDriver
from functional_process.total_process import GRAPH

CUTS = (
    path_of(lambda s: s.physics.proton_rate_density, VarPath),
    path_of(lambda s: s.fwbs.f_ster_div_single, VarPath),
)
"""The one variable cut per raw cross-node cycle. Both are the *only* single-variable
cut (out of every variable owned inside each cycle) that makes that cycle's own
subgraph fully acyclic on its own -- checked directly, not assumed, by cutting each
candidate in turn and checking `.is_acyclic` on the result. `proton_rate_density`
(owned by `FusionRates`, read by `PlasmaComposition`) closes the 4-node density/
fusion/pedestal/composition loop; `f_ster_div_single` (owned by `Divertor`, read by
`AFwTotalWithPowerflow`) closes the 2-node divertor/first-wall loop -- see
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


def driven_graph(graph=GRAPH):
    """`graph` (default: `total_process.GRAPH`, the default-configuration graph), with
    every raw cycle in `CUTS` that actually exists in `graph` cut into a declared
    `FixedPoint` problem. Every remaining multi-node SCC now declares exactly one
    problem -- confirmed below, not assumed, by `driven_graph()`'s own doctest-free
    but exercised construction (`schedule()` calling `Blocking.scc` on this graph
    would raise if not).

    Takes `graph` rather than always using the module's default so a caller checking
    this against a specific real `IN.DAT` can pass `total_process.graph_for(
    configuration_matching_that_file)` instead -- which nodes exist at all (and
    therefore which of `CUTS` even applies -- see that tuple's own docstring on the
    `ipowerflow`-gated second cycle) can differ by configuration, so this should be
    run against whichever graph is actually being checked, not silently default to
    the wrong one.
    """
    for var in CUTS:
        readers = graph.closing_readers(var)
        if not readers:
            continue  # this cycle does not exist in this configuration
        graph = FixedPointCut((Cut(var=var, readers=readers),)).apply(graph)
    return graph


def default_drivers(blocking: Blocking) -> dict:
    """One driver per driven block, assigned mechanically by problem type -- a Newton
    for every `RootFind` (`Intersect`, `DuctDiameterRootFind`), a Picard for every
    `FixedPoint` (the 8 structural self-loops plus the two cuts above). No bespoke
    per-block choice: every block in this graph is one of exactly these two shapes.

    Raises
    ------
    TypeError
        If a block declares a problem type that is neither -- e.g. an `Optimise` or
        `Feasibility`, which this graph does not currently register any of.
    """
    drivers = {}
    for block, problem, problem_type in zip(
        blocking.blocks, blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None:
            continue
        if issubclass(problem_type, RootFind):
            drivers[problem] = NewtonDriver()
        elif issubclass(problem_type, FixedPoint):
            drivers[problem] = PicardDriver()
        else:
            raise TypeError(
                f"block {block!r} declares a {problem_type.__name__}, and "
                f"default_drivers has no default driver for that problem type -- "
                f"assign one explicitly"
            )
    return drivers


def schedule(graph=GRAPH) -> Schedule:
    """`graph` (default: `total_process.GRAPH`), block by block, every cyclic block
    driven by its default driver. `Drive`/`Schedule`'s own construction is what
    checks this is actually runnable -- a `Schedule` that builds is a `Schedule` that
    can be called. See `driven_graph`'s own docstring for why `graph` is a parameter,
    not always the module default.
    """
    driven = driven_graph(graph)
    blocking = Blocking.scc(driven)
    return schedule_for(blocking, default_drivers(blocking))
