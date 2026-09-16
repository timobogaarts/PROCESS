"""`functional_process.cottax.recipes`: the paper's cut-and-determine recipe, on the
stellarator.

Three claims, in the order the recipe is built:

- **Structure.** Each recipe cuts what the paper says it cuts -- every coupling read
  (Jacobi), the backward reads in an order (Gauss-Seidel), the fewest of those over any
  order (Gauss-Seidel minimal, checked against brute force) -- and leaves every cyclic
  block with exactly one problem, so the result schedules.
- **The MDA agrees.** From one cold state, every recipe's Picard reaches the fixed
  point the hand-measured `mda.CUTS` reach, on every variable, to the Picard tolerance.
- **The arms agree** (`tier4`, ~15 s each). MDF and SAND assembled on a recipe cut
  converge to the hand cut's optimum. SAND under Jacobi is the one that carries an
  array-valued unknown (the 201-point density profile), which is what
  `drivers.condition_sizes` and `host_cache.flat_values` exist for.

`helias_5b` is the second stellarator and the tokamaks are the other five
configurations; `paper_tests/architectures.py` runs the recipes across all of them
and is where a cross-configuration claim is measured, not here.
"""

import itertools

import jax
import numpy as np
import pytest
from cottax.blocking import Blocking
from cottax.evaluation.schedule import Schedule
from cottax.names import PathMap
from cottax.problem import ConditionNode

from functional_process.cottax import recipes, session
from functional_process.cottax.core.solver.drivers import PicardDriver
from functional_process.cottax.indat import graph_for
from functional_process.cottax.mda import assign_drivers, cut_graph, default_drivers
from functional_process.cottax.mda_harness import _without_excluded
from functional_process.cottax.sand_harness import _mda_runner, cold_state, mda_env, seed_env

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

jax.config.update("jax_enable_x64", True)

STELLARATOR = "tests/regression/input_files/stellarator_helias.IN.DAT"

# The census `recipes.py`'s docstring describes, as numbers: (nodes, variables cut,
# reads cut) per cyclic component of the stellarator graph, in component order.
EXPECTED = {
    "jacobi": ((2, 0, 0), (6, 12, 12), (2, 0, 0), (2, 2, 2), (2, 0, 0)),
    "gauss_seidel": ((2, 0, 0), (6, 7, 7), (2, 0, 0), (2, 1, 1), (2, 0, 0)),
    "gauss_seidel_minimal": ((2, 0, 0), (6, 2, 2), (2, 0, 0), (2, 1, 1), (2, 0, 0)),
}


@pytest.fixture(scope="module")
def live():
    return session.open_session(STELLARATOR)


@pytest.fixture(scope="module")
def raw(live):
    return _without_excluded(live.machine_graph if live.machine_graph is not None else graph_for())


@pytest.mark.parametrize("name", recipes.RECIPES)
def test_each_recipe_cuts_what_the_paper_says(raw, name):
    graph, records = recipes.recipe(name).cut(raw)
    got = tuple((len(r.component), r.n_variables, r.n_reads) for r in records)
    assert got == EXPECTED[name]
    # Every cyclic block holds exactly one problem, and the whole thing schedules.
    blocking = Blocking.scc(graph)
    for block in blocking.blocks:
        if len(block) > 1:
            assert sum(isinstance(graph[n], ConditionNode) for n in block) == 1
    Schedule(Blocking.scc(assign_drivers(graph, default_drivers(graph))))


def test_the_minimal_order_is_minimal_by_brute_force(raw):
    """The subset DP finds what enumerating every order of the six-node component finds."""
    component = next(c for c in raw.cycles if len(c) == 6)
    reads = recipes.coupling_reads(raw, component)
    best = min(
        sum(1 for var, readers in reads.items()
            if any(order.index(r) < order.index(raw.owners[var]) for r in readers))
        for order in itertools.permutations(component)
    )
    found = recipes.backward_reads(raw, component, recipes.minimal_feedback_order(raw, component))
    assert len(found) == best == 2


def test_jacobi_reads_nothing_current(raw):
    """After a Jacobi cut no function node of a coupled block reads another's output:
    the block body is one parallel layer over the previous iterate."""
    graph, records = recipes.recipe("jacobi").cut(raw)
    for record in records:
        if record.problem is None:
            continue
        body = [n for n in record.component if not isinstance(graph[n], ConditionNode)]
        owned = {v for n in body for v in graph[n].owns}
        for node in body:
            stale = [v for v in graph[node].reads if v in owned and graph.owners[v] != node]
            # A node may still read a *solver's* unknown its own solve is driving.
            stale = [v for v in stale if not isinstance(graph[graph.owners[v]], ConditionNode)]
            assert not stale, (node.spelling, [v.spelling for v in stale])


@pytest.mark.parametrize("name", recipes.RECIPES)
def test_the_recipe_mda_reaches_the_hand_cut_fixed_point(live, name):
    _, base = mda_env(live.reference, graph=live.machine_graph)
    _, env = mda_env(live.reference, graph=live.machine_graph, cut=recipes.recipe(name))
    worst = 0.0
    compared = 0
    for var, value in env.items():
        if var.spelling.startswith("^") or var not in base:
            continue
        a, b = np.asarray(value, float), np.asarray(base[var], float)
        if a.shape != b.shape or not a.size:
            continue
        compared += 1
        if np.all(a == 0) and np.all(b == 0):
            continue
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))))
    assert compared > 800
    assert worst < 1e-5, worst


def test_picard_steps_are_reported_per_block(live, raw):
    """`PicardDriver(report_steps=True)` puts one `Steps` per fixed point in the env."""
    graph = recipes.recipe("gauss_seidel_minimal")(raw)
    drivers = default_drivers(graph)
    for problem in drivers:
        if isinstance(drivers[problem], PicardDriver):
            drivers[problem] = PicardDriver(report_steps=True)
    runnable = assign_drivers(graph, drivers)
    schedule = Schedule(Blocking.scc(runnable))
    env = seed_env(live.reference.data, schedule, runnable, cold_state(live.reference.data, live.machine_graph))
    out = dict(_mda_runner(schedule)(PathMap(env)))
    steps = {v.spelling: int(out[v]) for v in out if v.spelling.startswith("^driver_out.steps")}
    assert len(steps) == 4
    assert all(1 <= s <= 256 for s in steps.values()), steps


@pytest.mark.tier4
@pytest.mark.parametrize("name", recipes.RECIPES)
@pytest.mark.parametrize("arm", ["mdf", "sand"])
def test_the_arms_converge_to_the_hand_optimum(name, arm):
    hand = session.open_session(STELLARATOR)
    expected = getattr(hand, arm)()
    assert expected["status"] == "converged"
    live = session.open_session(STELLARATOR, cut=recipes.recipe(name))
    row = getattr(live, arm)()
    assert row["status"] == "converged", row.get("note")
    assert row["objf"] == pytest.approx(expected["objf"], rel=1e-7)
    jax.clear_caches()
