"""The PF/CS volt-second accounting: per-turn current waveforms and `PFCoil.vsec`."""

from cottax.interfaces.pytree_namespace_module import From, OutputInto

from functional_process.cottax.paths import pf_coil, physics
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.pfcoil.volt_seconds import (
    calculate_pf_coil_turn_currents,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_turn_currents_no_central_solenoid,
    calculate_pf_coil_turn_currents_reference,
    calculate_pf_cs_volt_seconds,
    calculate_pf_volt_seconds_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_volt_seconds_no_central_solenoid_bound,
)


class PFCoilTurnCurrents(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.turn_currents`."""

    fn = calculate_pf_coil_turn_currents_reference

    f_c_pf_cs_peak_time_array = From(pf_coil)
    c_pf_coil_turn_peak_input = From(pf_coil)
    c_pf_cs_coils_peak_ma = From(pf_coil)
    plasma_current = From(physics)

    c_pf_coil_turn = OutputInto(pf_coil)


class PFCoilTurnCurrentsNoCentralSolenoid(PFCoilTurnCurrents):
    """cottax node: `.tokamak.pf_coil.turn_currents`, the `iohcl = 0` occupant.

    Same reads and output as `PFCoilTurnCurrents` -- only which row is the plasma's
    differs, and that lives in `fn`'s baked topology.
    """

    fn = calculate_pf_coil_turn_currents_no_central_solenoid


class PFCoilVoltSeconds(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.volt_seconds`."""

    fn = calculate_pf_cs_volt_seconds

    ind_pf_cs_plasma_mutual = From(pf_coil)
    c_pf_coil_turn = From(pf_coil)

    vs_cs_pf_total_burn = OutputInto(pf_coil)
    vs_cs_pf_total_pulse = OutputInto(pf_coil)


class PFCoilVoltSecondsNoCentralSolenoid(PFCoilVoltSeconds):
    """cottax node: `.tokamak.pf_coil.volt_seconds`, the `iohcl = 0` occupant.

    Same reads as `PFCoilVoltSeconds`; only `fn` differs, and it is redeclared here
    (rather than left inherited) so `WrapsFunction` resynthesises `__call__` against
    *this* class's own target -- see `wraps.py`'s `__init_subclass__` docstring for the
    silent-wrong-function risk that guards against relying on an inherited `fn` instead.
    """

    fn = calculate_pf_volt_seconds_no_central_solenoid_bound
