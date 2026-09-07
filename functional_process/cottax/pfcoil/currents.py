"""What current each PF coil group carries, at each of the three time points."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.pfcoil import (
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
    PFLocation,
)
from functional_process.cottax.paths import build, pf_coil, physics
from functional_process.models.pfcoil.currents import (
    calculate_cs_flux_swing,  # noqa: F401 -- re-exported for tests
    calculate_cs_flux_swing_for_topology,
    calculate_efc_currents,  # noqa: F401 -- re-exported for tests
    calculate_equilibrium_currents,  # noqa: F401 -- re-exported for tests
    calculate_equilibrium_currents_for_topology,
    calculate_plasma_initiation_currents,  # noqa: F401 -- re-exported for tests
    calculate_plasma_initiation_currents_for_topology,
    calculate_plasma_initiation_currents_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_plasma_initiation_currents_no_central_solenoid_for_topology,
    calculate_time_point_currents,  # noqa: F401 -- re-exported for tests
    calculate_time_point_currents_for_topology,
    calculate_time_point_currents_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_time_point_currents_no_central_solenoid_for_topology,
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


class CSCurrentDensityPulseStart(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.current_density_pulse_start`."""

    j_cs_pulse_start = OutputInto(pf_coil)

    def __call__(
        self,
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
    ):
        return j_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top


class PFCoilInitiationCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.initiation_currents`."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static."""

    ssq0 = OutputInto(pf_coil)
    ccl0 = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        r_cs_middle=From(pf_coil),
        dz_cs_full=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
        alfapf=From(pf_coil),
    ):
        return calculate_plasma_initiation_currents_for_topology(
            rmajor=rmajor,
            rminor=rminor,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array,
            r_cs_middle=r_cs_middle,
            dz_cs_full=dz_cs_full,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            alfapf=alfapf,
            topology=self.topology,
        )


class PFCoilInitiationCurrentsNoCentralSolenoid(PFCoilInitiationCurrents):
    """cottax node: `.tokamak.pf_coil.initiation_currents`, the `iohcl = 0` occupant."""

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        alfapf=From(pf_coil),
    ):
        return calculate_plasma_initiation_currents_no_central_solenoid_for_topology(
            rmajor=rmajor,
            rminor=rminor,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array,
            alfapf=alfapf,
            topology=self.topology,
        )


class PFCoilEquilibriumCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.equilibrium_currents`."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static."""

    ccls = OutputInto(pf_coil)
    b_plasma_vertical_required = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
        aspect=From(physics),
        plasma_current=From(physics),
        beta_poloidal_vol_avg=From(physics),
        ind_plasma_internal_norm=From(physics),
        r_pf_coil_middle_group_array=From(pf_coil),
        z_pf_coil_middle_group_array=From(pf_coil),
        alfapf=From(pf_coil),
    ):
        return calculate_equilibrium_currents_for_topology(
            rmajor=rmajor,
            rminor=rminor,
            kappa=kappa,
            aspect=aspect,
            plasma_current=plasma_current,
            beta_poloidal_vol_avg=beta_poloidal_vol_avg,
            ind_plasma_internal_norm=ind_plasma_internal_norm,
            r_pf_coil_middle_group_array=r_pf_coil_middle_group_array,
            z_pf_coil_middle_group_array=z_pf_coil_middle_group_array,
            alfapf=alfapf,
            topology=self.topology,
        )


class CSFluxSwing(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.flux_swing`."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static, and necessarily a topology *with* a central solenoid -- this node is the
    solenoid's flux-swing balance, and `iohcl = 0` deletes it rather than changing it.
    """

    f_j_cs_start_end_flat_top = OutputInto(pf_coil)

    def __call__(
        self,
        ccls=From(pf_coil),
        ind_pf_cs_plasma_mutual=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        vs_plasma_ramp_required=From(physics),
        dr_cs_bore=From(build),
        dr_cs=From(build),
        dz_cs_full=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
    ):
        return calculate_cs_flux_swing_for_topology(
            ccls=ccls,
            ind_pf_cs_plasma_mutual=ind_pf_cs_plasma_mutual,
            n_pf_coil_turns=n_pf_coil_turns,
            vs_plasma_ramp_required=vs_plasma_ramp_required,
            dr_cs_bore=dr_cs_bore,
            dr_cs=dr_cs,
            dz_cs_full=dz_cs_full,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            topology=self.topology,
        )


class PFCoilTimePointCurrents(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.time_point_currents`."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static. Which slot each coil's three currents land in."""

    c_pf_cs_coil_pulse_start_ma = OutputInto(pf_coil)
    c_pf_cs_coil_flat_top_ma = OutputInto(pf_coil)
    c_pf_cs_coil_pulse_end_ma = OutputInto(pf_coil)
    ccl0_ma = OutputInto(pf_coil)
    ccls_ma = OutputInto(pf_coil)

    def __call__(
        self,
        ccl0=From(pf_coil),
        ccls=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
        j_cs_flat_top_end=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
        f_j_cs_start_end_flat_top=From(pf_coil),
    ):
        return calculate_time_point_currents_for_topology(
            ccl0=ccl0,
            ccls=ccls,
            a_cs_poloidal=a_cs_poloidal,
            j_cs_flat_top_end=j_cs_flat_top_end,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            f_j_cs_start_end_flat_top=f_j_cs_start_end_flat_top,
            topology=self.topology,
        )


class PFCoilTimePointCurrentsNoCentralSolenoid(PFCoilTimePointCurrents):
    """cottax node: `.tokamak.pf_coil.time_point_currents`, the `iohcl = 0` occupant."""

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    f_j_cs_start_end_flat_top = OutputInto(pf_coil)

    def __call__(
        self,
        ccl0=From(pf_coil),
        ccls=From(pf_coil),
        f_j_cs_start_pulse_end_flat_top=From(pf_coil),
    ):
        return calculate_time_point_currents_no_central_solenoid_for_topology(
            ccl0=ccl0,
            ccls=ccls,
            f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
            topology=self.topology,
        )
