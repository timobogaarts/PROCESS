"""`cottax.problem`: every configuration's problem builds, and a switch picks the branch
PROCESS's own dispatch takes."""

import inspect
import itertools

import numpy as np
import pytest
from cottax.interfaces import Scope

from functional_process import configurations
from functional_process.cottax.models import constraints as layer
from functional_process.cottax.problem import TOP, problem_model
from functional_process.models import constraints, objectives


@pytest.mark.parametrize("name", configurations.NAMES)
def test_every_configuration_states_its_problem(name):
    configuration = configurations.load(name)
    model, omitted = problem_model(configuration)
    top = model.graph[TOP]
    assert top.unknowns and not omitted
    assert bool(top.objectives) != configuration.problem.root_find


def _switched():
    """Every switched function a configuration uses, with that configuration's switches."""
    seen = set()
    for name in configurations.NAMES:
        problem = configurations.load(name).problem
        used = [getattr(constraints, f"constraint_{c}") for c in problem.icc]
        used.append(
            objectives.OBJECTIVE_METRICS[
                layer.FiguresOfMerit(abs(problem.i_figure_merit))
            ]
        )
        for fn in used:
            if fn.__name__ not in layer.VARIANTS:
                continue
            taken = [
                p
                for p in inspect.signature(fn).parameters
                if p in layer.SWITCH_PARAMETER_NAMES
            ]
            switches = {p: problem.switches[p] for p in taken}
            key = (fn.__name__, tuple(switches.items()))
            if key not in seen:
                seen.add(key)
                yield pytest.param(
                    fn,
                    switches,
                    id="-".join(map(str, itertools.chain(*key[1:]))) or fn.__name__,
                )


@pytest.mark.parametrize(("fn", "switches"), list(_switched()))
def test_a_variant_is_the_branch_its_switches_take(fn, switches):
    """The variant, fed at the places it reads, agrees with the dispatcher."""
    rng = np.random.default_rng(0)
    arguments = {
        p: switches[p] if p in switches else float(rng.uniform(0.5, 2.0))
        for p in inspect.signature(fn).parameters
    }
    at = {
        layer.area_of(p).place(p): v for p, v in arguments.items() if p not in switches
    }
    variant, chosen = layer.variant(fn, switches)
    reads = layer._reads(variant, chosen)
    places = [r.place(p) if isinstance(r, Scope) else r for p, r in reads.items()]
    got = variant(*(at[place] for place in places))
    np.testing.assert_allclose(np.asarray(got), np.asarray(fn(**arguments)), rtol=1e-12)
