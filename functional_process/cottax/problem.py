"""A configuration's problem as a model: its machine, and over it the problem the input
file states -- the design variables (`ixc`) within their bounds, the active constraints
(`icc`, the first `n_equality` held as equalities) and the figure of merit
(`i_figure_merit`, negative to maximise). An architecture then places everything the
machine solves around it.
"""

import cottax.interfaces as cx
from cottax.pytree.path import VarPath
from jax.tree_util import GetAttrKey, SequenceKey

from functional_process.configurations import Configuration
from functional_process.cottax.models.constraints import (
    MERIT,
    NON_INPUT_FIELDS,
    area_of,
    constraint,
    constraint_place,
    merit,
)
from functional_process.cottax.input.native import layered_values
from functional_process.vocabulary import ITERATION_VARIABLES

TOP = cx.node(cx.N["Opt"])
"""Where the problem is bound: the optimisation, or in a root-find run the root find."""


def design_place(ixc_id: int) -> VarPath:
    """Where PROCESS iteration variable `ixc_id` lives."""
    variable = ITERATION_VARIABLES[ixc_id]
    keys = (
        GetAttrKey(variable.module),
        GetAttrKey(variable.target_name or variable.name),
    )
    if variable.array_index is not None:
        keys = (*keys, SequenceKey(variable.array_index))
    return VarPath(keys)


def problem_model(configuration: Configuration, omit=()):
    """`(model, omitted)`: `configuration`'s machine and the problem its file states,
    at `TOP`, and why each constraint left out is (`{id: reason}`).

    A constraint that reads nothing the design reaches is constant over it, so no
    optimiser can move it: an inequality is left out, an equality refused.

    Raises
    ------
    ValueError
        If an equality reads nothing the design reaches, a root-find run is not square,
        a switch disagrees with the machine's value of it, or a read resolved through
        `NON_INPUT_FIELDS` is owned by no node.
    """
    problem = configuration.problem
    values, _sources = layered_values(configuration.values)
    _check_switches(problem.switches, configuration.values)
    design = tuple(design_place(i) for i in problem.ixc)
    equal = set(problem.icc[: problem.n_equality])
    omitted = {cid: "omitted by the caller" for cid in problem.icc if cid in omit}
    bodies = {
        cid: constraint(cid, problem.switches) for cid in problem.icc if cid not in omit
    }

    if problem.root_find:
        if len(equal & bodies.keys()) != len(design):
            raise ValueError(
                f"a root find needs one equality per iteration variable: this file "
                f"holds {len(equal & bodies.keys())} against {len(design)}"
            )
    else:
        machine = cx.to_graph(configuration.machine).graph
        reached = {p for n in machine.reach(design) for p in machine[n].owns}
        reached |= set(design)
        external = [c for c, b in bodies.items() if not reached & set(b.reads.values())]
        if refused := [c for c in external if c in equal]:
            raise ValueError(
                f"equality constraint(s) {refused} read nothing the design reaches: "
                f"leaving one out changes what feasible means"
            )
        for cid in external:
            del bodies[cid]
            omitted[cid] = "reads nothing the design reaches"

    slots = {f"Constraint{cid}": b for cid, b in bodies.items()}
    equalities = [constraint_place(c) for c in bodies if c in equal]
    if problem.root_find:
        top = cx.RootFind(equalities, design)
    else:
        held = {
            f"Require{cid}": cx.Requirement(
                constraint_place(cid), cx.Eq if cid in equal else cx.Le
            )
            for cid in bodies
        }
        slots |= held | {"Merit": merit(problem.i_figure_merit, problem.switches)}
        sense = "maximise" if problem.i_figure_merit < 0 else "minimise"
        top = cx.Optimise(
            design={
                u: problem.bounds[i] for u, i in zip(design, problem.ixc, strict=True)
            },
            subject_to=tuple(held.values()),
            **{sense: MERIT},
        )
    model = cx.declare(configuration.machine, slots | {"Opt": top})
    _check_computed_reads(model.graph, values, bodies.values())
    return model, omitted


def _area(name: str) -> str:
    return area_of(name).place(name).segments[0].name


def _check_switches(switches, stated) -> None:
    """A switch the problem is stated under and the configuration also states for the
    machine must agree."""
    for name, value in switches.items():
        held = stated.get(_area(name), {}).get(name)
        if held is not None and int(held) != int(value):
            raise ValueError(
                f"switch {name!r} is {value} for the problem and {held} for the machine"
            )


def _check_computed_reads(graph, values, bodies) -> None:
    """Every read resolved through `NON_INPUT_FIELDS` is computed by some model, or
    given a value by the configuration: a stale entry would otherwise read nothing."""
    owners = graph.graph.owners
    stale = {
        p.spelling
        for body in bodies
        for p in body.reads.values()
        if (name := p.segments[-1].name) in NON_INPUT_FIELDS
        and p not in owners
        and (_area(name), name) not in values
    }
    if stale:
        raise ValueError(
            f"{sorted(stale)} are read as computed values (`NON_INPUT_FIELDS`), but no "
            f"node of this machine computes them and the configuration gives none"
        )


__all__ = ["TOP", "design_place", "problem_model"]
