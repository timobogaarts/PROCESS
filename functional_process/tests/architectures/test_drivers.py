"""The port's drivers: a fixed point declared through the model surface under the port's
contract, answered by cottax's `Picard`, end to end through `run`; VMCON's design
scaling; and the SQP drivers' refusal of an objective the design cannot move.
"""

import cottax.cx as cx
import jax.numpy as jnp
import numpy as np
import pytest
from cottax.core.frozen import OrderedMap
from cottax.core import Plan
from cottax.consumers.execution import CallableGraph, run
from cottax.core import GraphWithValues
from cottax.relational.consumers.execution.jax_solvers import Picard
from cottax.cx import N, Scope, V
from cottax.relational import Eq, Relation
from jax.tree_util import GetAttrKey

from functional_process.cottax.architectures.contracts import (
    PortFixedPointIteration,
    Start,
)
from functional_process.cottax.architectures.drivers import _refuse_inert_objective

toy = Scope(V.toy, GetAttrKey)
"""A synthetic area for the contraction toy problem -- not a `DataStructure` area, so
it is a bare `Scope` rather than one off `functional_process.cottax.paths`."""


def relax(u, k):
    """`u = 0.5u + k`: a fixed point that genuinely closes its own loop.

    **Not a model of the port, and that is a fact about the port**: no fixed point in
    `functional_process/cottax/models/` reads its own output any more (the
    availability `CplifeAvailSt*` arms recompute `.costs.cplife` from scratch). The
    port's real ones are minted by `FixedPointCut` on a configuration's cycles, which is
    the architecture tests' subject and not a driver unit's.
    """
    return 0.5 * u + k


Relax = toy.fixed_point(relax, "u")


def test_picard_driver_drives_a_fixed_point_node_end_to_end():
    """The driver through everything the port actually runs: the model surface, the
    contract assigned, the driver implemented, cottax's `run`. `u = 0.5u + 3` has the
    exact fixed point `u = 6`, so the answer is hand-computable and the test is about
    the wiring, not about a model.
    """
    graph = cx.to_graph({"relax": Relax})
    problem = cx.minted.problem(N["relax"])
    driven = (
        Plan(graph)
        + cx.assign(graph, problem, PortFixedPointIteration())
        + cx.implement({problem: Picard(rtol=1e-12, atol=1e-14)})
    ).graph
    # The start is read at the problem's own `Start` place (`^start.toy.u`), not at the
    # unknown's name: writing the latter would be seeding the answer.
    (start,) = driven[problem].data[driven[problem].driver.takes.index(Start)]
    out = run(
        GraphWithValues(
            CallableGraph(driven),
            OrderedMap({start: jnp.asarray(0.0), toy.place("k"): jnp.asarray(3.0)}),
        )
    )
    assert float(out[toy.place("u")]) == pytest.approx(6.0, abs=1e-6)


# ---------------------------------------------------------------- design scaling


@pytest.mark.parametrize(
    ("value", "scaled"),
    [
        (2.0, True),
        (-4.0, True),
        (1e-11, True),  # above PROCESS's floor: still conditions its own coordinate
        (0.0, False),
        (1e-13, False),
        (-3.8e-27, False),  # `.power.qac` after a solve -- the value that broke VMCON
    ],
)
def test_a_start_below_processes_own_floor_is_left_unscaled(value, scaled):
    """`1 / x_start` conditioning needs a floor, and PROCESS supplies the number.

    `check_iteration_variable` rejects `abs(value) <= 1e-12` outright, so below that
    PROCESS does not believe a value can condition anything. Testing `!= 0.0` instead --
    as this did -- lets a numerically-zero start through: `.power.qac` is exactly `0.0`
    on a seeded env but `-3.8e-27` after a solve, which produced a scale of `-2.6e+26`
    and killed the QP. Only a restart from a solved point ever reached it.
    """
    import numpy as np

    from functional_process.cottax.architectures.drivers import (
        UNSCALABLE_BELOW,
        design_scale,
    )

    factor = float(design_scale(np.array([value], dtype=float))[0])
    if scaled:
        assert factor == pytest.approx(1.0 / value)
    else:
        assert factor == 1.0
        assert abs(value) <= UNSCALABLE_BELOW


def test_scaling_leaves_a_workable_problem_when_a_coordinate_is_unscalable():
    """The floor degrades to the unscaled problem in that coordinate, not to a
    divide-by-zero and not to an error: every factor stays finite and non-zero.
    """
    import numpy as np

    from functional_process.cottax.architectures.drivers import design_scale

    scale = design_scale(np.array([1e-13, 2.0, 0.0, -4.0, -3.8e-27]))
    assert np.all(np.isfinite(scale))
    assert np.all(scale != 0.0)


# ================================================ the inert-objective refusal (§26)


class _Rows:
    """A stand-in for a `Gaps` carrying only what `_refuse_inert_objective` reads
    through `condition_places`: the objective, the relations read and the unknowns.
    """

    def __init__(self, objective, conditions, unknowns):
        self.objectives = (objective,) if objective is not None else ()
        self.relations = tuple(Relation(c, None, Eq) for c in conditions)
        self.unknowns = tuple(unknowns)


def _rows():
    return _Rows(
        toy.place("objf"),
        (toy.place("c1"), toy.place("c2")),
        (toy.place("u"), toy.place("v")),
    )


def test_a_steerable_objective_is_not_refused():
    _refuse_inert_objective([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], _rows())


def test_an_objective_row_of_zeros_is_refused_and_the_message_says_why():
    """`st_regression`'s shape: the objective reads one path this graph does not own,
    so its whole gradient row is zero and the SQP solves the feasibility problem that
    remains -- reporting `converged` while doing it (`_audit/optimise_design.md` §26).
    """
    with pytest.raises(ValueError, match=r"identically zero gradient") as caught:
        _refuse_inert_objective([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], _rows())
    message = str(caught.value)
    assert "feasibility" in message
    assert "MISSING PRODUCER" in message
    assert toy.place("u").spelling in message  # the design variables, named


def test_other_zero_rows_are_named_and_not_refused_on():
    """A constraint the design cannot steer is common and often intended -- refusing on
    one would fail `helias_5b`, whose equality 11 compares a radial build against
    `.physics.rmajor` on a file with three iteration variables that do not move it.
    """
    _refuse_inert_objective([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], _rows())
    with pytest.raises(ValueError, match=r"identically zero gradient") as caught:
        _refuse_inert_objective([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], _rows())
    assert toy.place("c1").spelling in str(caught.value)
    assert toy.place("c2").spelling not in str(caught.value)


def test_an_empty_jacobian_is_not_refused():
    """A problem with no conditions is a different defect and has its own report."""
    _refuse_inert_objective(np.zeros((0, 0)), _Rows(None, (), ()))
