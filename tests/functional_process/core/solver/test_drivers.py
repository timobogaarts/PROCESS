"""Tests for `drivers.PicardDriver`.

Two levels: a synthetic contraction mapping (exact, hand-computable fixed point --
proves the iteration mechanics themselves, independent of any real node), and a real
`FixedPointFunction` already registered in this codebase, driven end to end through
`cottax.evaluate.Drive` (proves genuine integration, not just the driver in isolation).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from cottax.tools.path import path_map
from cottax.blocking import Blocking
from cottax.evaluate import Schedule
from cottax.interfaces.pytree_namespace_module import area, resolve, to_graph
from cottax.problem import Start, driver_vars
from cottax.rewrites import Assign
from cottax.spec import VarPath

from functional_process.core.solver.drivers import PicardDriver, _refuse_inert_objective
from functional_process.models.power.thermal_cryo import CryoQNucStep
from functional_process.paths import fwbs
from functional_process.models.switch_enums import CoilNuclearHeatingModel
from process.models.tfcoil.base import TFConductorModel

toy = area("toy")
"""A synthetic area for the contraction toy problem -- not a `DataStructure` area, so
it is built with cottax's bare `area()` rather than off `functional_process.paths`."""


def vpath(where):
    """`resolve(where, VarPath)`, short enough to write at every call site below."""
    return resolve(where, VarPath)


class _Contraction:
    """A minimal stand-in for `ConditionMap`: `PicardDriver` only ever calls it
    positionally and reads `.unknowns` in its missing-`Start` error message, so a
    plain callable with that much is enough to test the iteration in isolation from
    any real graph.
    """

    unknowns = (vpath(toy.u),)

    def __call__(self, u):
        # Fixed point at u = 6.0 (u = 0.5u + 3 => u = 6), |derivative| = 0.5 < 1, so
        # Picard converges geometrically from any start.
        return (0.5 * u + 3.0,)


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


def test_picard_driver_drives_a_real_fixed_point_function_node():
    """`CryoQNucStep` at `inuclear = FRANCES_FOX` with a superconducting TF coil has
    `d(qnuc_next)/d(qnuc) == 0` (`test_thermal_cryo.py`'s own gradient test pins this):
    PROCESS recomputes `.fwbs.qnuc` from `.fwbs.p_tf_nuclear_heat_mw` alone, so the
    fixed point does not depend on the starting guess and Picard reaches it in exactly
    one step. Ground truth is one direct call to `step`, not a hardcoded number, since
    the point of this regime is that any entering value gives the same answer.

    **This test used to drive `TempTurbineCoolantInStep`, which no longer exists.**
    That node was a `FixedPointFunction` only because
    `calculate_plant_thermal_efficiency` passes the entering value through on some arms
    of `i_thermal_electric_conversion` x `i_blanket_type` x `secondary_cycle_liq`;
    splitting those switches into occupants showed the pass-through arms are "the field
    is an input" and the computing arms read nothing they own, so the fixed point was an
    artefact of the switch (`_audit/next_steps.md` §14.2). `CryoQNucStep` is the nearest
    surviving instance of the same shape and the same zero self-gradient.
    """
    node = CryoQNucStep(
        i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        inuclear=CoilNuclearHeatingModel.FRANCES_FOX,
    )
    p_tf_nuclear_heat_mw = 0.045
    expected = node.step(
        qnuc=0.0,  # arbitrary -- ignored in this regime
        p_tf_nuclear_heat_mw=p_tf_nuclear_heat_mw,
    )

    graph = to_graph(node)
    # `Assign` attaches the driver *and* mints the ports that driver's own `requires`
    # names -- one `^guess.<place>` per unknown here. The starting guess is supplied at
    # `^guess.*` rather than at the unknown's own name; writing the latter would be
    # seeding the answer. `mda.driven_graph` does exactly this to every problem in the
    # real graph.
    (problem,) = graph.declared
    graph = Assign(problem, PicardDriver()).apply(graph)
    # Built through `schedule_for` rather than by constructing a `Drive` directly: the
    # schedule is what the port actually runs, and it assembles the `Drive` itself, so
    # this test does not restate `Drive`'s constructor signature.
    schedule = Schedule(Blocking.scc(graph))
    # The guess port is read off the problem rather than spelled out, the same way
    # `mda.starts_for` does it: the node is the authority on where its start is read.
    (guess,) = driver_vars(graph[problem], Start)
    env = {
        guess: jnp.asarray(300.0),
        vpath(fwbs.p_tf_nuclear_heat_mw): jnp.asarray(p_tf_nuclear_heat_mw),
    }

    out = schedule.run(path_map(env))

    got = out[vpath(fwbs.qnuc)]
    assert float(got) == pytest.approx(float(expected), abs=1e-6)


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

    from functional_process.core.solver.drivers import UNSCALABLE_BELOW, design_scale

    factor = float(design_scale(np.array([value], dtype=float))[0])
    if scaled:
        assert factor == pytest.approx(1.0 / value)
    else:
        assert factor == 1.0
        assert abs(value) <= UNSCALABLE_BELOW


def test_scaling_leaves_a_workable_problem_when_a_coordinate_is_unscalable():
    """The floor degrades to the unscaled problem in that coordinate, not to a
    divide-by-zero and not to an error: every factor stays finite and non-zero."""
    import numpy as np

    from functional_process.core.solver.drivers import design_scale

    scale = design_scale(np.array([1e-13, 2.0, 0.0, -4.0, -3.8e-27]))
    assert np.all(np.isfinite(scale))
    assert np.all(scale != 0.0)


# ================================================ the inert-objective refusal (§26)


class _Rows:
    """A stand-in for `ConditionMap` carrying only what `_refuse_inert_objective`
    reads: the condition names (objective first) and the unknowns.
    """

    def __init__(self, conditions, unknowns):
        self.conditions = tuple(conditions)
        self.unknowns = tuple(unknowns)


def _rows():
    return _Rows(
        (vpath(toy.objf), vpath(toy.c1), vpath(toy.c2)), (vpath(toy.u), vpath(toy.v))
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
    assert vpath(toy.u).path_str() in message  # the design variables, named


def test_other_zero_rows_are_named_and_not_refused_on():
    """A constraint the design cannot steer is common and often intended -- refusing on
    one would fail `helias_5b`, whose equality 11 compares a radial build against
    `.physics.rmajor` on a file with three iteration variables that do not move it.
    """
    _refuse_inert_objective([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], _rows())
    with pytest.raises(ValueError, match=r"identically zero gradient") as caught:
        _refuse_inert_objective([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], _rows())
    assert vpath(toy.c1).path_str() in str(caught.value)
    assert vpath(toy.c2).path_str() not in str(caught.value)


def test_an_empty_jacobian_is_not_refused():
    """A problem with no conditions is a different defect and has its own report."""
    _refuse_inert_objective(np.zeros((0, 0)), _Rows((), ()))
