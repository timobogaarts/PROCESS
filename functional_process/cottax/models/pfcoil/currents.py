"""What current each PF coil group carries, at each of the three time points."""

from cottax.relational_pytree import V

from functional_process.cottax.models.pfcoil import REFERENCE_TOPOLOGY, PFLocation
from functional_process.cottax.paths import pf_coil
from functional_process.models.pfcoil.currents import (
    calculate_cs_flux_swing,  # noqa: F401 -- re-exported for tests
    calculate_cs_flux_swing_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_cs_flux_swing_reference,
    calculate_efc_currents,  # noqa: F401 -- re-exported for tests
    calculate_equilibrium_currents,  # noqa: F401 -- re-exported for tests
    calculate_equilibrium_currents_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_equilibrium_currents_no_central_solenoid,
    calculate_equilibrium_currents_reference,
    calculate_j_cs_pulse_start,
    calculate_plasma_initiation_currents,  # noqa: F401 -- re-exported for tests
    calculate_plasma_initiation_currents_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_plasma_initiation_currents_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_plasma_initiation_currents_no_central_solenoid_bound,
    calculate_plasma_initiation_currents_no_central_solenoid_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_plasma_initiation_currents_reference,
    calculate_time_point_currents,  # noqa: F401 -- re-exported for tests
    calculate_time_point_currents_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_time_point_currents_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_time_point_currents_no_central_solenoid_bound,
    calculate_time_point_currents_no_central_solenoid_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_time_point_currents_reference,
)

FIXED_CURRENT_GROUPS = REFERENCE_TOPOLOGY.groups_at(PFLocation.ABOVE_TF)
"""The `i_pf_location = 2` divertor-coil groups of the reference topology, whose current
PROCESS fixes analytically and then hands to the equilibrium solve as fixed-current
filaments (`pfcoil.py:485-511`).
"""

EQUILIBRIUM_GROUPS = REFERENCE_TOPOLOGY.groups_at(PFLocation.OUTSIDE_TF)
"""The reference topology's `i_pf_location = 3` groups, whose current the SVD solves
for, in `pcls0`'s order (`pfcoil.py:519-532`).
"""


# cottax node: `.tokamak.cs_coil.current_density_pulse_start`.
CSCurrentDensityPulseStart = pf_coil.function(
    calculate_j_cs_pulse_start, "j_cs_pulse_start"
)


# cottax node: `.tokamak.pf_coil.initiation_currents`.
PFCoilInitiationCurrents = pf_coil.function(
    calculate_plasma_initiation_currents_reference,
    "ssq0",
    "ccl0",
)


# cottax node: `.tokamak.pf_coil.initiation_currents`, the `iohcl = 0` occupant.
#
# Not a subclass of `PFCoilInitiationCurrents`: this occupant's read set is a strict
# subset (no CS to place a filament at), and `WrapsFunction` reads accumulate over the
# MRO rather than being retractable, so sharing the base would leave four reads
# declared that `fn` no longer takes.
PFCoilInitiationCurrentsNoCentralSolenoid = pf_coil.function(
    calculate_plasma_initiation_currents_no_central_solenoid_bound,
    "ssq0",
    "ccl0",
)


# cottax node: `.tokamak.pf_coil.equilibrium_currents`.
PFCoilEquilibriumCurrents = pf_coil.function(
    calculate_equilibrium_currents_reference,
    "ccls",
    V.physics.b_plasma_vertical_required,
)


# cottax node: `.tokamak.pf_coil.equilibrium_currents`, the `iohcl = 0` occupant.
#
# Same reads and outputs as `PFCoilEquilibriumCurrents` -- only `fn`'s baked topology
# differs -- so this is the inheriting arm rather than a standalone declaration.
PFCoilEquilibriumCurrentsNoCentralSolenoid = pf_coil.function(
    calculate_equilibrium_currents_no_central_solenoid,
    "ccls",
    V.physics.b_plasma_vertical_required,
)


# cottax node: `.tokamak.cs_coil.flux_swing`.
#
# Necessarily a topology *with* a central solenoid -- this node is the solenoid's
# flux-swing balance, and `iohcl = 0` deletes it rather than changing it, so `fn` is
# bound to `REFERENCE_TOPOLOGY` unconditionally with no second arm.
CSFluxSwing = pf_coil.function(
    calculate_cs_flux_swing_reference,
    "f_j_cs_start_end_flat_top",
)


# cottax node: `.tokamak.pf_coil.time_point_currents`.
PFCoilTimePointCurrents = pf_coil.function(
    calculate_time_point_currents_reference,
    "c_pf_cs_coil_pulse_start_ma",
    "c_pf_cs_coil_flat_top_ma",
    "c_pf_cs_coil_pulse_end_ma",
    "ccl0_ma",
    "ccls_ma",
)


# cottax node: `.tokamak.pf_coil.time_point_currents`, the `iohcl = 0` occupant.
#
# Not a subclass of `PFCoilTimePointCurrents`: this occupant reads three of its six
# inputs (no `f_j_cs_start_end_flat_top` in -- it is this occupant's own output, not a
# read) and `WrapsFunction` reads cannot be retracted by inheritance.
PFCoilTimePointCurrentsNoCentralSolenoid = pf_coil.function(
    calculate_time_point_currents_no_central_solenoid_bound,
    "c_pf_cs_coil_pulse_start_ma",
    "c_pf_cs_coil_flat_top_ma",
    "c_pf_cs_coil_pulse_end_ma",
    "ccl0_ma",
    "ccls_ma",
    "f_j_cs_start_end_flat_top",
)
