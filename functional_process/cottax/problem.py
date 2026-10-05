"""A configuration's problem as a model: its machine, the constraints and merit its
problem names, and at `TOP` the optimisation (or the root find) over them. Nothing is
chosen here: the configuration names every declaration.
"""

import cottax.relational_pytree as cx

from functional_process.configurations import Configuration, Optimisation
from functional_process.cottax.models.constraints import held

TOP = cx.node(cx.N["Opt"])
"""Where the problem is bound: the optimisation, or the root find."""


def problem_model(configuration: Configuration):
    """`configuration` as one model: the machine, each constraint and the merit at its
    own slot, and the problem at `TOP`."""
    problem = configuration.problem
    if isinstance(problem, Optimisation):
        constraints = (*problem.equalities, *problem.inequalities)
        requirements = tuple(
            cx.requirement(held(c), cx.Eq if c in problem.equalities else cx.Le)
            for c in constraints
        )
        merit = problem.minimise or problem.maximise
        sense = "minimise" if problem.minimise is not None else "maximise"
        slots = {"merit": merit}
        top = cx.optimise(
            design=problem.design,
            subject_to=requirements,
            **{sense: merit.function_outputs[0]},
        )
    else:
        constraints = (*problem.equalities, *problem.evaluated)
        slots = {}
        top = cx.satisfy(
            design=problem.design,
            subject_to=[cx.requirement(held(c), cx.Eq) for c in problem.equalities],
        )
    bodies = {held(c).segments[-1].name: c for c in constraints}
    return cx.declare(configuration.machine, bodies | slots | {"Opt": top})


__all__ = ["TOP", "problem_model"]
