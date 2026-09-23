"""The optimisation problem as **IDF** (Individual Discipline Feasible).

SAND (`sand.assemble`) folds every statement on the optimiser's cycle into the
optimiser. IDF lifts only the **coupling**: the consistency statements the scheme
minted to open the cycles become equalities of the optimiser, while the statements the
models declare themselves (a coil's root find, the ion-temperature iteration) stay as
they are and are converged inside each optimiser iterate. The optimiser therefore owns
the design variables and one copy per cut, and every discipline is feasible at every
iterate -- which is what the name says.

The recipe, in cottax's ops: `mda.cut_graph` (the scheme), `sand.problem_graph` (the
constraint declarations -- each a body and the requirement beside it -- the objective
node and the `Optimise`), then
`cottax.mdao_architectures.IDF` -- absorb the `^mda` statements, nest the models' own
inside the optimiser. `sand.sand_schedule` then assigns the drivers, exactly as for
SAND.
"""

from __future__ import annotations

from cottax.interfaces import Plan
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
    # `resolved` is `placed` read off the graph the architecture's own first op leaves
    # -- a statement the design reaches only through a requirement is on the
    # optimiser's cycle once the requirements are absorbed and not before. The
    # architecture answers that itself, so the report does not repeat the op here.
    report["coupling"] = architecture.resolved(with_problem)[Global]
    return (Plan(with_problem) + architecture).graph, optimiser, report
