"""`cottax.problem` and the converter's choice of declarations: every configuration's
problem builds, each constraint and merit the converter chose is the branch PROCESS's
own dispatch takes under the file's switches, and every read meant as a computed value
is one."""

import inspect

import numpy as np
import pytest

from functional_process import configurations
from functional_process.configurations import Optimisation
from functional_process.cottax.input.indat import (
    _as_imported,
    problem_from_indat,
    switch_values_from_indat,
)
from functional_process.cottax.input.native import layered_values
from functional_process.cottax.models.constraints import NON_INPUT_FIELDS
from functional_process.cottax.problem import TOP, problem_model
from functional_process.models import constraints, objectives
from functional_process.vocabulary import FiguresOfMerit

NAMES = configurations.NAMES


def _file(name):
    return f"tests/regression/input_files/{name}.IN.DAT"


@pytest.mark.parametrize("name", NAMES)
def test_every_configuration_states_its_problem(name):
    configuration = configurations.load(name)
    top = problem_model(configuration).graph[TOP]
    assert top.unknowns
    assert bool(top.objectives) == isinstance(configuration.problem, Optimisation)


def _chosen(name):
    """`(PROCESS's dispatching function, the declaration the converter chose)` for every
    constraint and the merit of `name`'s file."""
    problem = configurations.load(name).problem
    stated = problem_from_indat(_as_imported(_file(name)))
    chosen = [
        *problem.equalities,
        *getattr(problem, "inequalities", ()),
        *getattr(problem, "evaluated", ()),
    ]
    by_id = {c.function_outputs[1].segments[-1].name: c for c in chosen}
    pairs = [
        (getattr(constraints, f"constraint_{cid}"), by_id[f"c{cid}"])
        for cid in stated.icc
        if f"c{cid}" in by_id
    ]
    if isinstance(problem, Optimisation):
        merit = FiguresOfMerit(abs(int(stated.i_figure_merit or 7)))
        pairs.append((
            objectives.OBJECTIVE_METRICS[merit],
            problem.minimise or problem.maximise,
        ))
    return pairs


@pytest.mark.parametrize("name", NAMES)
def test_each_declaration_is_the_branch_processes_switches_take(name):
    """The declaration, fed at the places it reads, agrees with PROCESS's dispatch
    called with the file's switches: the converter chose the right variant.

    Every parameter of the dispatch gets a value by name; a declaration reading some
    parameter at another's place (`Constraint5Greenwald` reads the line density as
    `nd_plasma_electron_limited`) is read at that place's name.
    """
    switches = switch_values_from_indat(_file(name))
    rng = np.random.default_rng(0)
    for dispatch, declaration in _chosen(name):
        arguments = {
            p: switches[p] if p in switches else float(rng.uniform(0.5, 2.0))
            for p in inspect.signature(dispatch).parameters
        }
        values = []
        parameters = inspect.signature(declaration.fn).parameters
        for parameter, read in zip(parameters, declaration.function_reads, strict=True):
            place = read.place
            leaf = place.segments[-1].name
            values.append(arguments[leaf] if leaf in arguments else arguments[parameter])
        np.testing.assert_allclose(
            np.asarray(declaration.fn(*values), dtype=float),
            np.asarray(dispatch(**arguments), dtype=float),
            rtol=1e-12,
            err_msg=f"{declaration.fn.__name__} is not {dispatch.__name__}'s branch",
        )


@pytest.mark.parametrize("name", NAMES)
def test_every_computed_read_is_computed_or_given(name):
    """A read resolved through `NON_INPUT_FIELDS` is computed by some node, or given a
    value by the configuration: a stale entry would otherwise read nothing."""
    configuration = configurations.load(name)
    graph = problem_model(configuration).graph
    values, _sources = layered_values(configuration.values)
    owners = graph.graph.owners
    stale = {
        p.spelling
        for p in graph.graph.variables
        if len(p.segments) == 2
        and (leaf := p.segments[-1].name) in NON_INPUT_FIELDS
        and p not in owners
        and (p.segments[0].name, leaf) not in values
    }
    assert not stale
