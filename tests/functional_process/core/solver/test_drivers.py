"""Tests for `drivers.PicardDriver`.

Two levels: a synthetic contraction mapping (exact, hand-computable fixed point --
proves the iteration mechanics themselves, independent of any real node), and a real
`FixedPointFunction` already registered in this codebase, driven end to end through
`cottax.evaluate.Drive` (proves genuine integration, not just the driver in isolation).
"""

import jax.numpy as jnp
import pytest
from cottax.evaluate import Drive
from cottax.interfaces.pytree_namespace_module import area, resolve, to_graph
from cottax.spec import VarPath

from functional_process.core.solver.drivers import PicardDriver
from functional_process.models.power_B_thermal_cryo import TempTurbineCoolantInStep
from functional_process.paths import fwbs, heat_transport
from process.data_structure.blanket_variables import BlktModelTypes
from process.models.power import ElectricConversionModelTypes

toy = area("toy")
"""A synthetic area for the contraction toy problem -- not a `DataStructure` area, so
it is built with cottax's bare `area()` rather than off `functional_process.paths`."""


def vpath(where):
    """`resolve(where, VarPath)`, short enough to write at every call site below."""
    return resolve(where, VarPath)


class _Contraction:
    """A minimal stand-in for `ConditionMap`: `PicardDriver` only ever calls it
    positionally and reads `.unknowns` in its `start is None` error message, so a
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
    driver = PicardDriver(rtol=1e-10, atol=1e-12, max_iter=100)
    (result,) = driver(_Contraction(), start=(jnp.asarray(0.0),))
    assert float(result) == pytest.approx(6.0, abs=1e-8)


def test_picard_driver_converges_regardless_of_starting_point():
    """A genuine contraction (|derivative| < 1) reaches the same fixed point from
    any starting guess, not just a convenient one.
    """
    driver = PicardDriver(rtol=1e-10, atol=1e-12, max_iter=100)
    for start in (0.0, 100.0, -50.0):
        (result,) = driver(_Contraction(), start=(jnp.asarray(start),))
        assert float(result) == pytest.approx(6.0, abs=1e-8)


def test_picard_driver_requires_a_start():
    """Same requirement and same reasoning as `NewtonDriver`: no shape to guess a
    pytree from, so a missing `start` is a clear error, not a silent default.
    """
    driver = PicardDriver()
    with pytest.raises(ValueError, match="needs a starting value"):
        driver(_Contraction(), start=None)


def test_picard_driver_drives_a_real_fixed_point_function_node():
    """`TempTurbineCoolantInStep` in the `STEAM_RANKINE_CYCLE`/`secondary_cycle_liq=2`
    regime has `d(temp_turbine_coolant_in_next)/d(temp_turbine_coolant_in) == 0`
    (`test_power_B_thermal_cryo.py`'s own gradient test already pins this): the first
    stage overwrites the entering value from `temp_blkt_coolant_out` unconditionally,
    so the fixed point does not depend on the starting guess at all and Picard
    iteration reaches it in exactly one step. Ground truth is computed the same way
    `test_power_B_thermal_cryo.py` does -- one direct call to `step` -- not a
    hardcoded number, since the whole point of this regime is that any entering value
    gives the same answer.
    """
    node = TempTurbineCoolantInStep(
        i_thermal_electric_conversion=ElectricConversionModelTypes.STEAM_RANKINE_CYCLE,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
        secondary_cycle_liq=ElectricConversionModelTypes.USER_INPUT,
    )
    temp_blkt_coolant_out = 700.0
    outlet_temp_liq = 700.0
    expected = node.step(
        temp_turbine_coolant_in=0.0,  # arbitrary -- ignored in this regime
        temp_blkt_coolant_out=temp_blkt_coolant_out,
        outlet_temp_liq=outlet_temp_liq,
    )

    graph = to_graph(node)
    drive = Drive(subgraph=graph, driver=PicardDriver())
    env = {
        vpath(heat_transport.temp_turbine_coolant_in): jnp.asarray(300.0),
        vpath(fwbs.temp_blkt_coolant_out): jnp.asarray(temp_blkt_coolant_out),
        vpath(fwbs.outlet_temp_liq): jnp.asarray(outlet_temp_liq),
    }

    out = drive(env)

    got = out[vpath(heat_transport.temp_turbine_coolant_in)]
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
