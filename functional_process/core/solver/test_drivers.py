"""Tests for `drivers.PicardDriver`.

Two levels: a synthetic contraction mapping (exact, hand-computable fixed point --
proves the iteration mechanics themselves, independent of any real node), and a real
`FixedPointFunction` already registered in this codebase, driven end to end through
`cottax.evaluate.Drive` (proves genuine integration, not just the driver in isolation).
"""

import jax.numpy as jnp
import pytest
from cottax.evaluate import Drive
from cottax.interfaces.pytree_namespace_module import path_of, to_graph
from cottax.spec import VarPath

from functional_process.core.solver.drivers import PicardDriver
from functional_process.models.power_B_thermal_cryo import TempTurbineCoolantInStep
from process.data_structure.blanket_variables import BlktModelTypes
from process.models.power import ElectricConversionModelTypes


def vpath(where):
    """`path_of(where, VarPath)`, short enough to write at every call site below."""
    return path_of(where, VarPath)


class _Contraction:
    """A minimal stand-in for `ConditionMap`: `PicardDriver` only ever calls it
    positionally and reads `.unknowns` in its `start is None` error message, so a
    plain callable with that much is enough to test the iteration in isolation from
    any real graph.
    """

    unknowns = (vpath(lambda s: s.toy.u),)

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
        i_thermal_electric_conversion=int(
            ElectricConversionModelTypes.STEAM_RANKINE_CYCLE
        ),
        i_blanket_type=int(BlktModelTypes.CCFE_HCPB),
        secondary_cycle_liq=2,
    )
    temp_blkt_coolant_out = 700.0
    outlet_temp_liq = 700.0
    (expected,) = node.step(
        temp_turbine_coolant_in=0.0,  # arbitrary -- ignored in this regime
        temp_blkt_coolant_out=temp_blkt_coolant_out,
        outlet_temp_liq=outlet_temp_liq,
    )

    graph = to_graph(node)
    drive = Drive(subgraph=graph, driver=PicardDriver())
    env = {
        vpath(lambda s: s.heat_transport.temp_turbine_coolant_in): jnp.asarray(300.0),
        vpath(lambda s: s.fwbs.temp_blkt_coolant_out): jnp.asarray(
            temp_blkt_coolant_out
        ),
        vpath(lambda s: s.fwbs.outlet_temp_liq): jnp.asarray(outlet_temp_liq),
    }

    out = drive(env)

    got = out[vpath(lambda s: s.heat_transport.temp_turbine_coolant_in)]
    assert float(got) == pytest.approx(float(expected), abs=1e-6)
