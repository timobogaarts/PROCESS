"""Tests for `drivers.PicardDriver`.

Two levels: a synthetic contraction mapping (exact, hand-computable fixed point --
proves the iteration mechanics themselves, independent of any real node), and a real
`FixedPointFunction` already registered in this codebase, driven end to end through
`cottax.execution.schedule.Drive` (proves genuine integration, not just the driver in isolation).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from cottax.execution.drivers.kinds import Start
from cottax.interfaces import Assign, Eq, RunnableGraph, Schedule
from cottax.interfaces.pytree_namespace_module import (
    FixedPointFunction,
    From,
    OutputInto,
    area,
    resolve,
    to_graph,
)
from cottax.pytree.path import PathMap, VarPath

from functional_process.cottax.architectures.drivers import (
    PicardDriver,
    _refuse_inert_objective,
)
from functional_process.cottax.queries import declared

toy = area("toy")
"""A synthetic area for the contraction toy problem -- not a `DataStructure` area, so
it is built with cottax's bare `area()` rather than off `functional_process.cottax.paths`."""


def vpath(where):
    """`resolve(where, VarPath)`, short enough to write at every call site below."""
    return resolve(where, VarPath)


class _Contraction:
    """A minimal stand-in for `ConditionMap`: `PicardDriver` is an `IterateDriver`, so
    cottax wraps this in a `NextValues` and the driver only ever calls *that*
    positionally; `.symbols` and `.unknowns` are what the level and the missing-`Start`
    message read, so a plain callable with that much is enough to test the iteration in
    isolation from any real graph.

    A call gives `(objectives, sides)` -- the statement as stated, `u = 0.5u + 3` as
    the pair -- and the reading takes the right side as the next iterate. Nothing here
    subtracts.
    """

    unknowns = (vpath(toy.u),)
    symbols = (Eq,)
    objectives = ()

    def __call__(self, u):
        # Fixed point at u = 6.0 (u = 0.5u + 3 => u = 6), |derivative| = 0.5 < 1, so
        # Picard converges geometrically from any start.
        return (), ((u, 0.5 * u + 3.0),)


def test_picard_driver_converges_on_a_contraction_mapping():
    """`u = 0.5u + 3` has the exact fixed point `u = 6`; Picard must find it."""
    driver = PicardDriver(rtol=1e-10, atol=1e-12, max_steps=100)
    (result,) = driver(_Contraction(), {Start: (jnp.asarray(0.0),)})
    assert float(result) == pytest.approx(6.0, abs=1e-8)


def test_picard_driver_converges_regardless_of_starting_point():
    """A genuine contraction (|derivative| < 1) reaches the same fixed point from
    any starting guess, not just a convenient one.
    """
    driver = PicardDriver(rtol=1e-10, atol=1e-12, max_steps=100)
    for start in (0.0, 100.0, -50.0):
        (result,) = driver(_Contraction(), {Start: (jnp.asarray(start),)})
        assert float(result) == pytest.approx(6.0, abs=1e-8)


def test_picard_driver_requires_a_start():
    """Same requirement and same reasoning as `NewtonDriver`: no shape to guess a
    pytree from, so a missing start is a clear error, not a silent default.

    Spelled as an empty driver-data mapping rather than the old `start=None`, which is
    what "no start supplied" now looks like: `Drive.role_data` builds this mapping from
    the driver's `requires`, so a driver called directly with `{}` is the one path that
    still reaches the refusal.
    """
    driver = PicardDriver()
    with pytest.raises(ValueError, match="needs a starting value"):
        driver(_Contraction(), {})


class Relax(FixedPointFunction):
    """`u = 0.5u + k`, declared through the surface a model is declared through.

    **Not a registered model node, and that is a fact about the port**: no
    `FixedPointFunction` in `functional_process/cottax/models/` reads its own output any
    more (`CplifeAvailSt`'s four arms recompute `.costs.cplife` from scratch, and
    `df050df3` made `DeltaEtaStep` -- which this test used to drive -- an ordinary
    `ExplicitFunction`, its entering `.power.delta_eta` being PROCESS's incoming field
    value rather than a previous iterate). A fixed point with no cycle is not
    executable, so the end-to-end check needs a declaration that genuinely closes its
    own loop; the port's real ones are minted by `FixedPointCut` on a configuration's
    cycles, which is the architecture tests' subject and not a driver unit's.
    """

    u = OutputInto(toy)

    def step(self, u=From(toy), k=From(toy)):
        return 0.5 * u + k


def test_picard_driver_drives_a_fixed_point_node_end_to_end():
    """The driver through everything the port actually runs: `to_graph`, `Assign`,
    `RunnableGraph`, `Schedule`, `Drive`, the `ConditionMap`. `u = 0.5u + 3` has the
    exact fixed point `u = 6`, so the answer is hand-computable and the test is about
    the wiring, not about a model.
    """
    node = Relax()
    graph = to_graph(node)
    # `Assign` attaches the driver *and* derives the places that driver's own `requires`
    # names -- one `^guess.<place>` per unknown here. The starting guess is supplied at
    # `^guess.*` rather than at the unknown's own name; writing the latter would be
    # seeding the answer. `mda.driven_graph` does exactly this to every problem in the
    # real graph.
    (problem,) = declared(graph)
    graph = Assign(problem, PicardDriver(rtol=1e-12, atol=1e-14)).apply(graph)
    # Built through the schedule rather than by constructing a `Drive` directly: the
    # schedule is what the port actually runs, and it assembles the `Drive` itself, so
    # this test does not restate `Drive`'s constructor signature.
    schedule = Schedule(RunnableGraph(graph))
    # The guess port is read off the problem rather than spelled out, the same way
    # `mda.starts_for` does it: the node is the authority on where its start is read.
    driven = graph[problem]
    (guess,) = driven.data[driven.driver.requires.index(Start)]

    out = schedule.run(
        PathMap({guess: jnp.asarray(0.0), vpath(toy.k): jnp.asarray(3.0)})
    )

    assert float(out[vpath(toy.u)]) == pytest.approx(6.0, abs=1e-6)


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


class _Stated:
    """The statement under a reading -- where `condition_places` finds the sides."""

    def __init__(self, relations):
        self.relations = relations


class _Rows:
    """A stand-in for a `Gaps` carrying only what `_refuse_inert_objective` reads
    through `condition_places`: the objective, the statement's relations and the
    unknowns.
    """

    def __init__(self, objective, conditions, unknowns):
        self.objectives = (objective,) if objective is not None else ()
        self.conditions = _Stated(tuple((c, None, Eq) for c in conditions))
        self.unknowns = tuple(unknowns)


def _rows():
    return _Rows(
        vpath(toy.objf), (vpath(toy.c1), vpath(toy.c2)), (vpath(toy.u), vpath(toy.v))
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
    assert vpath(toy.u).spelling in message  # the design variables, named


def test_other_zero_rows_are_named_and_not_refused_on():
    """A constraint the design cannot steer is common and often intended -- refusing on
    one would fail `helias_5b`, whose equality 11 compares a radial build against
    `.physics.rmajor` on a file with three iteration variables that do not move it.
    """
    _refuse_inert_objective([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], _rows())
    with pytest.raises(ValueError, match=r"identically zero gradient") as caught:
        _refuse_inert_objective([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], _rows())
    assert vpath(toy.c1).spelling in str(caught.value)
    assert vpath(toy.c2).spelling not in str(caught.value)


def test_an_empty_jacobian_is_not_refused():
    """A problem with no conditions is a different defect and has its own report."""
    _refuse_inert_objective(np.zeros((0, 0)), _Rows(None, (), ()))
