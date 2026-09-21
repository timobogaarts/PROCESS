"""The `Optimise` layer as **IDF** (Individual Discipline Feasible).

SAND (`sand.sand_graph`) residualises *every* fixed point and folds every problem into
the optimiser. IDF lifts only the **coupling**: the fixed points `mda.cut_graph` minted
to open the cycles become residual equalities of the optimiser, while the problems the
models declare themselves (a coil's root find, the ion-temperature iteration) stay as
they are and are converged inside each optimiser iterate. The optimiser therefore
owns the design variables and one copy per cut, and every discipline is feasible at
every iterate -- which is what the name says.

The recipe, in cottax's ops:

1. `cut` (default `mda.cut_graph`): a `FixedPointCut` per loop-carried variable.
2. `sand.optimise_graph`: the constraint and objective nodes, and the `Optimise`.
3. `Residualise` each minted fixed point: `x = g(^hat.x)` becomes `g(^hat.x) - x = 0`.
4. `Combine` the optimiser with those residuals into `^problem.idf`, the optimiser's
   unknowns leading.
5. `nested_inside(^problem.idf)`: every other problem on the cycle is solved inside it.

`sand.sand_schedule` then assigns the drivers, exactly as for SAND.
"""

from __future__ import annotations

from cottax.pytree.plan import Plan
from cottax.pytree.problem import is_fixed_point
from cottax.pytree.rewrites import Combine, Residualise
from cottax.pytree.spec import NodePath
from jax.tree_util import GetAttrKey

from functional_process.cottax.architectures.evaluate import without_excluded
from functional_process.cottax.architectures.mda import cut_graph
from functional_process.cottax.architectures.sand import optimise_graph
from functional_process.cottax.queries import declared, nested_inside

IDF = NodePath((GetAttrKey("idf"),))
"""Where the combined problem is bound before minting: `^problem.idf`."""


def idf_graph(
    graph,
    ixc,
    icc,
    n_equality,
    i_figure_merit,
    switch_values=None,
    omit=(),
    cut=cut_graph,
):
    """`(graph, problem, report)`: the IDF graph, the combined problem's name, and
    `sand.optimise_graph`'s report with the lifted fixed points under `"coupling"`.
    """
    raw = without_excluded(graph)
    driven = cut(raw)
    with_problem, optimiser, report = optimise_graph(
        driven,
        ixc,
        icc,
        n_equality,
        i_figure_merit,
        switch_values=switch_values,
        omit=omit,
    )
    internal = set(declared(raw))
    coupling = tuple(
        p
        for p in declared(with_problem)
        if p not in internal and p != optimiser and is_fixed_point(with_problem[p])
    )
    plan = Plan(with_problem)
    for problem in coupling:
        plan += Residualise(problem)
    combine = Combine(IDF, (optimiser, *coupling))
    plan += combine
    report["coupling"] = coupling
    return nested_inside(plan.graph, combine.problem), combine.problem, report
