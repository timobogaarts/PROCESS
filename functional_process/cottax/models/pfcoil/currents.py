"""What current each PF coil group carries, at each of the three time points."""

from cottax.interfaces.pytree_namespace_module import From, OutputInto

from functional_process.cottax.models.pfcoil import REFERENCE_TOPOLOGY, PFLocation
from functional_process.cottax.paths import build, pf_coil, physics
from functional_process.cottax.wraps import WrapsFunction
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


class CSCurrentDensityPulseStart(WrapsFunction):
    """cottax node: `.tokamak.cs_coil.current_density_pulse_start`."""

    fn = calculate_j_cs_pulse_start

    j_cs_flat_top_end = From(pf_coil)
    f_j_cs_start_pulse_end_flat_top = From(pf_coil)

    j_cs_pulse_start = OutputInto(pf_coil)


class PFCoilInitiationCurrents(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.initiation_currents`."""

    fn = calculate_plasma_initiation_currents_reference

    rmajor = From(physics)
    rminor = From(physics)
    r_pf_coil_middle_group_array = From(pf_coil)
    z_pf_coil_middle_group_array = From(pf_coil)
    r_cs_middle = From(pf_coil)
    dz_cs_full = From(pf_coil)
    a_cs_poloidal = From(pf_coil)
    j_cs_flat_top_end = From(pf_coil)
    f_j_cs_start_pulse_end_flat_top = From(pf_coil)
    alfapf = From(pf_coil)

    ssq0 = OutputInto(pf_coil)
    ccl0 = OutputInto(pf_coil)


class PFCoilInitiationCurrentsNoCentralSolenoid(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.initiation_currents`, the `iohcl = 0` occupant.

    Not a subclass of `PFCoilInitiationCurrents`: this occupant's read set is a strict
    subset (no CS to place a filament at), and `WrapsFunction` reads accumulate over the
    MRO rather than being retractable, so sharing the base would leave four reads
    declared that `fn` no longer takes.
    """

    fn = calculate_plasma_initiation_currents_no_central_solenoid_bound

    rmajor = From(physics)
    rminor = From(physics)
    r_pf_coil_middle_group_array = From(pf_coil)
    z_pf_coil_middle_group_array = From(pf_coil)
    alfapf = From(pf_coil)

    ssq0 = OutputInto(pf_coil)
    ccl0 = OutputInto(pf_coil)


class PFCoilEquilibriumCurrents(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.equilibrium_currents`."""

    fn = calculate_equilibrium_currents_reference

    rmajor = From(physics)
    rminor = From(physics)
    kappa = From(physics)
    aspect = From(physics)
    plasma_current = From(physics)
    beta_poloidal_vol_avg = From(physics)
    ind_plasma_internal_norm = From(physics)
    r_pf_coil_middle_group_array = From(pf_coil)
    z_pf_coil_middle_group_array = From(pf_coil)
    alfapf = From(pf_coil)

    ccls = OutputInto(pf_coil)
    b_plasma_vertical_required = OutputInto(physics)


class PFCoilEquilibriumCurrentsNoCentralSolenoid(PFCoilEquilibriumCurrents):
    """cottax node: `.tokamak.pf_coil.equilibrium_currents`, the `iohcl = 0` occupant.

    Same reads and outputs as `PFCoilEquilibriumCurrents` -- only `fn`'s baked topology
    differs -- so this is the inheriting arm rather than a standalone declaration.
    """

    fn = calculate_equilibrium_currents_no_central_solenoid


class CSFluxSwing(WrapsFunction):
    """cottax node: `.tokamak.cs_coil.flux_swing`.

    Necessarily a topology *with* a central solenoid -- this node is the solenoid's
    flux-swing balance, and `iohcl = 0` deletes it rather than changing it, so `fn` is
    bound to `REFERENCE_TOPOLOGY` unconditionally with no second arm.
    """

    fn = calculate_cs_flux_swing_reference

    ccls = From(pf_coil)
    ind_pf_cs_plasma_mutual = From(pf_coil)
    n_pf_coil_turns = From(pf_coil)
    vs_plasma_ramp_required = From(physics)
    dr_cs_bore = From(build)
    dr_cs = From(build)
    dz_cs_full = From(pf_coil)
    a_cs_poloidal = From(pf_coil)
    j_cs_flat_top_end = From(pf_coil)
    f_j_cs_start_pulse_end_flat_top = From(pf_coil)

    f_j_cs_start_end_flat_top = OutputInto(pf_coil)


class PFCoilTimePointCurrents(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.time_point_currents`."""

    fn = calculate_time_point_currents_reference

    ccl0 = From(pf_coil)
    ccls = From(pf_coil)
    a_cs_poloidal = From(pf_coil)
    j_cs_flat_top_end = From(pf_coil)
    f_j_cs_start_pulse_end_flat_top = From(pf_coil)
    f_j_cs_start_end_flat_top = From(pf_coil)

    c_pf_cs_coil_pulse_start_ma = OutputInto(pf_coil)
    c_pf_cs_coil_flat_top_ma = OutputInto(pf_coil)
    c_pf_cs_coil_pulse_end_ma = OutputInto(pf_coil)
    ccl0_ma = OutputInto(pf_coil)
    ccls_ma = OutputInto(pf_coil)


class PFCoilTimePointCurrentsNoCentralSolenoid(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.time_point_currents`, the `iohcl = 0` occupant.

    Not a subclass of `PFCoilTimePointCurrents`: this occupant reads three of its six
    inputs (no `f_j_cs_start_end_flat_top` in -- it is this occupant's own output, not a
    read) and `WrapsFunction` reads cannot be retracted by inheritance.
    """

    fn = calculate_time_point_currents_no_central_solenoid_bound

    ccl0 = From(pf_coil)
    ccls = From(pf_coil)
    f_j_cs_start_pulse_end_flat_top = From(pf_coil)

    c_pf_cs_coil_pulse_start_ma = OutputInto(pf_coil)
    c_pf_cs_coil_flat_top_ma = OutputInto(pf_coil)
    c_pf_cs_coil_pulse_end_ma = OutputInto(pf_coil)
    ccl0_ma = OutputInto(pf_coil)
    ccls_ma = OutputInto(pf_coil)
    f_j_cs_start_end_flat_top = OutputInto(pf_coil)
