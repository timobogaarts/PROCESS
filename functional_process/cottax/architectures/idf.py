"""The optimisation problem as **IDF** (Individual Discipline Feasible).

SAND (`sand.assemble`) folds every statement on the optimiser's cycle into the
optimiser. IDF lifts only the **coupling**: the consistency statements the scheme
minted to open the cycles become equalities of the optimiser, while the statements the
models declare themselves (a coil's root find, the ion-temperature iteration) stay as
they are and are converged inside each optimiser iterate. The optimiser therefore owns
the design variables and one copy per cut, and every discipline is feasible at every
iterate -- which is what the name says.

The recipe, in cottax's ops: `sand.problem_graph` on the raw graph (the constraint
declarations -- each a body and the requirement beside it -- the objective node and
the `Optimise`), then `cottax.mdao_architectures.IDF` handed `mda.SCHEME`'s rule and closing: it cuts the
cycles itself, absorbs the closures it bound and nests the models' own problems inside
the optimiser. A closure is what the architecture's own cutting binds, so the graph is
**not** cut beforehand -- a closure cut earlier would be a declared problem to it, and
nested rather than lifted. `sand.sand_schedule` then assigns the drivers, exactly as
for SAND.
"""

from __future__ import annotations

from cottax.mdao_architectures import IDF

from functional_process.cottax.architectures.evaluate import without_excluded
from functional_process.cottax.architectures.mda import SCHEME
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
    with_problem, optimiser, report = problem_graph(
        without_excluded(graph),
        ixc,
        icc,
        n_equality,
        i_figure_merit,
        switch_values=switch_values,
        omit=omit,
    )
    # The lifted problems are the closures the architecture's cutting binds: read off
    # its own derivation, applied once, so the report says what the op did.
    resolution = IDF(scheme.rule, scheme.closing, optimiser).resolution(with_problem)
    report["coupling"] = tuple(c.at for c in resolution.closures)
    return resolution.graph, optimiser, report
