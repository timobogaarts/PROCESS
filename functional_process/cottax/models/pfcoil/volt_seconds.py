"""The PF/CS volt-second accounting: per-turn current waveforms and `PFCoil.vsec`."""

from functional_process.cottax.paths import pf_coil
from functional_process.models.pfcoil.volt_seconds import (
    calculate_pf_coil_turn_currents,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_turn_currents_no_central_solenoid,
    calculate_pf_coil_turn_currents_reference,
    calculate_pf_cs_volt_seconds,
    calculate_pf_volt_seconds_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_volt_seconds_no_central_solenoid_bound,
)

# cottax node: `.tokamak.pf_coil.turn_currents`.
PFCoilTurnCurrents = pf_coil.function(
    calculate_pf_coil_turn_currents_reference,
    "c_pf_coil_turn",
)


# cottax node: `.tokamak.pf_coil.turn_currents`, the `iohcl = 0` occupant.
#
# Same reads and output as `PFCoilTurnCurrents` -- only which row is the plasma's
# differs, and that lives in `fn`'s baked topology.
PFCoilTurnCurrentsNoCentralSolenoid = pf_coil.function(
    calculate_pf_coil_turn_currents_no_central_solenoid,
    "c_pf_coil_turn",
)


# cottax node: `.tokamak.pf_coil.volt_seconds`.
PFCoilVoltSeconds = pf_coil.function(
    calculate_pf_cs_volt_seconds, "vs_cs_pf_total_burn", "vs_cs_pf_total_pulse"
)


# cottax node: `.tokamak.pf_coil.volt_seconds`, the `iohcl = 0` occupant.
#
# Same reads as `PFCoilVoltSeconds`; only `fn` differs, and it is redeclared here
# (rather than left inherited) so `WrapsFunction` resynthesises `__call__` against
# *this* class's own target -- see `wraps.py`'s `__init_subclass__` docstring for the
# silent-wrong-function risk that guards against relying on an inherited `fn` instead.
PFCoilVoltSecondsNoCentralSolenoid = pf_coil.function(
    calculate_pf_volt_seconds_no_central_solenoid_bound,
    "vs_cs_pf_total_burn",
    "vs_cs_pf_total_pulse",
)
