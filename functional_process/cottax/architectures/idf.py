"""The optimisation problem as **IDF** (Individual Discipline Feasible).

SAND (`sand.assemble`) folds every statement on the optimiser's cycle into the
optimiser. IDF lifts only the **coupling**: the consistency statements the scheme
minted to open the cycles become equalities of the optimiser, while the statements the
models declare themselves (a coil's root find, the ion-temperature iteration) stay as
they are and are converged inside each optimiser iterate. The optimiser therefore owns
the design variables and one copy per cut, and every discipline is feasible at every
iterate -- which is what the name says.

The recipe, in cottax's ops: `mda.cut_graph` (the scheme), `sand.problem_graph` (the
constraint, requirement and objective nodes, and the `Optimise`), then
`cottax.mdao_architectures.IDF` -- absorb the `^mda` statements, nest the models' own
inside the optimiser. `sand.sand_schedule` then assigns the drivers, exactly as for
SAND.
"""

from __future__ import annotations

from cottax.interfaces import Absorb, Plan
from cottax.mdao_architectures import IDF, Global

from functional_process.cottax.architectures.evaluate import without_excluded
from functional_process.cottax.architectures.mda import SCHEME, cut_graph
from functional_process.cottax.architectures.sand import problem_graph


def idf_graph(
    graph,
    ixc,
    icc,
    n_equality,
    i_figure_merit,
    switch_values=None,
    omit=(),
    scheme=SCHEME,
):
    """`(graph, problem, report)`: the IDF graph, the optimiser's name, and
    `sand.problem_graph`'s report with the lifted statements under `"coupling"`.
    """
    driven = cut_graph(without_excluded(graph), scheme)
    with_problem, optimiser, report = problem_graph(
        driven,
        ixc,
        icc,
        n_equality,
        i_figure_merit,
        switch_values=switch_values,
        omit=omit,
    )
    architecture = IDF(optimiser=optimiser)
    # `placed` reads the optimiser's cycle, and a statement the design reaches only
    # through a requirement is on that cycle only once the requirements are absorbed --
    # which is the architecture's own first op. So the report asks after it.
    required = architecture.requirements_of(with_problem)
    absorbed = (
        Absorb(optimiser, required).apply(with_problem) if required else with_problem
    )
    report["coupling"] = architecture.placed(absorbed)[Global]
    return (Plan(with_problem) + architecture).graph, optimiser, report
