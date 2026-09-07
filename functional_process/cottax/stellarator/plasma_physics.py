"""Pure-functional port of `Stellarator.st_phys`'s genuinely-new sub-computations (chunk
1B of unit #1).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.safe_math import (
    safe_sqrt,  # noqa: F401
)
from functional_process.cottax.paths import (
    constraints,
    current_drive,
    first_wall,
    fwbs,
    physics,
    stellarator,
)
from functional_process.models.stellarator.plasma_physics import (
    calculate_clipped_radiation_powers,
    calculate_fusion_power_totals_mw,
    calculate_fusion_totals_no_beam,
    calculate_heating_and_radiation_power,  # noqa: F401
    calculate_heating_and_radiation_power_ignited,
    calculate_heating_and_radiation_power_non_ignited,
    calculate_neutron_wall_load,  # noqa: F401
    calculate_neutron_wall_load_first_wall_area_comprehensive_2014,
    calculate_neutron_wall_load_first_wall_area_pre_2014,
    calculate_neutron_wall_load_scaled_plasma_surface,
    calculate_poloidal_field_from_rotational_transform,
    calculate_radiated_wall_load_and_fraction,  # noqa: F401
    calculate_radiated_wall_load_first_wall_area_comprehensive_2014,
    calculate_radiated_wall_load_first_wall_area_pre_2014,
    calculate_radiated_wall_load_scaled_plasma_surface,
    calculate_stellarator_beta_and_rho_star,
    calculate_thermal_energy_totals,
    calculate_total_field,
    select_stellarator_beta_and_stored_energy,
)
from functional_process.vocabulary import (
    constants,  # noqa: F401
)


class TotalField(ExplicitFunction):
    """cottax node: `calculate_total_field`, ports declared."""

    b_plasma_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average
        )


class PoloidalFieldFromRotationalTransform(ExplicitFunction):
    """cottax node: `calculate_poloidal_field_from_rotational_transform`, ports
    declared.
    """

    b_plasma_surface_poloidal_average = OutputInto(physics)

    def __call__(
        self,
        rminor=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        rmajor=From(physics),
        iotabar=From(stellarator),
    ):
        return calculate_poloidal_field_from_rotational_transform(
            rminor, b_plasma_toroidal_on_axis, rmajor, iotabar
        )


class StellaratorBetaAndRhoStar(ExplicitFunction):
    """cottax node: `calculate_stellarator_beta_and_rho_star`, ports declared."""

    beta_total_vol_avg = OutputInto(physics)
    e_plasma_beta = OutputInto(physics)
    rho_star = OutputInto(physics)

    def __call__(
        self,
        beta_fast_alpha=From(physics),
        beta_beam=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        b_plasma_total=From(physics),
        vol_plasma=From(physics),
        m_ions_total_amu=From(physics),
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        eps=From(physics),
        rmajor=From(physics),
    ):
        return calculate_stellarator_beta_and_rho_star(
            beta_fast_alpha,
            beta_beam,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
            b_plasma_total,
            vol_plasma,
            m_ions_total_amu,
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            eps,
            rmajor,
        )


class FusionPowerTotalsMw(ExplicitFunction):
    """cottax node: `calculate_fusion_power_totals_mw`, ports declared."""

    p_plasma_dt_mw = OutputInto(physics)
    p_dhe3_total_mw = OutputInto(physics)
    p_dd_total_mw = OutputInto(physics)

    def __call__(
        self,
        dt_power_density_plasma=From(physics),
        dhe3_power_density=From(physics),
        dd_power_density=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_fusion_power_totals_mw(
            dt_power_density_plasma, dhe3_power_density, dd_power_density, vol_plasma
        )


class FusionTotalsNoBeam(ExplicitFunction):
    """cottax node: `calculate_fusion_totals_no_beam`, ports declared."""

    fusden_total = OutputInto(physics)
    fusden_alpha_total = OutputInto(physics)
    p_dt_total_mw = OutputInto(physics)

    def __call__(
        self,
        fusden_plasma=From(physics),
        fusden_plasma_alpha=From(physics),
        p_plasma_dt_mw=From(physics),
    ):
        return calculate_fusion_totals_no_beam(
            fusden_plasma, fusden_plasma_alpha, p_plasma_dt_mw
        )


class ClippedRadiationPowers(ExplicitFunction):
    """cottax node: `calculate_clipped_radiation_powers`, ports declared."""

    pden_plasma_core_rad_mw = OutputInto(physics)
    pden_plasma_outer_rad_mw = OutputInto(physics)
    p_plasma_inner_rad_mw = OutputInto(physics)
    p_plasma_outer_rad_mw = OutputInto(physics)

    def __call__(
        self,
        pden_plasma_core_rad_mw_unclipped=From(physics),
        pden_plasma_outer_rad_mw_unclipped=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_clipped_radiation_powers(
            pden_plasma_core_rad_mw_unclipped,
            pden_plasma_outer_rad_mw_unclipped,
            vol_plasma,
        )


class NeutronWallLoad(ExplicitFunction):
    """The `calculate_neutron_wall_load` family -- one occupant per arm of the
    `.physics.i_pflux_fw_neutron` x `.heat_transport.ipowerflow` dispatch.
    """

    pflux_fw_neutron_mw = OutputInto(physics)


class NeutronWallLoadScaledPlasmaSurface(NeutronWallLoad):
    """`i_pflux_fw_neutron == SCALED_PLASMA_SURFACE_AREA` (1) -- PROCESS's own default
    (`physics_variables.py:1006`) and the reference run's.
    """

    def __call__(
        self,
        ffwal=From(physics),
        p_neutron_total_mw=From(physics),
        a_plasma_surface=From(physics),
    ):
        return calculate_neutron_wall_load_scaled_plasma_surface(
            ffwal, p_neutron_total_mw, a_plasma_surface
        )


class NeutronWallLoadFirstWallAreaPre2014(NeutronWallLoad):
    """`i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow == PRE_2014` (0)."""

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        fhole=From(fwbs),
        a_fw_total=From(first_wall),
    ):
        return calculate_neutron_wall_load_first_wall_area_pre_2014(
            p_neutron_total_mw, fhole, a_fw_total
        )


class NeutronWallLoadFirstWallAreaComprehensive2014(NeutronWallLoad):
    """`i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow ==
    COMPREHENSIVE_2014` (1) -- PROCESS's own `ipowerflow` default.
    """

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        fhole=From(fwbs),
        a_fw_total=From(first_wall),
        f_a_fw_outboard_hcd=From(fwbs),
        f_ster_div_single=From(fwbs),
    ):
        return calculate_neutron_wall_load_first_wall_area_comprehensive_2014(
            p_neutron_total_mw,
            fhole,
            a_fw_total,
            f_a_fw_outboard_hcd,
            f_ster_div_single,
        )


class HeatingAndRadiationPower(ExplicitFunction):
    """The `calculate_heating_and_radiation_power` family -- one occupant per
    `.physics.i_plasma_ignited` value.
    """

    p_plasma_rad_mw = OutputInto(physics)
    psolradmw = OutputInto(physics)
    p_plasma_separatrix_mw = OutputInto(physics)
    p_fw_alpha_mw = OutputInto(physics)


class HeatingAndRadiationPowerIgnited(HeatingAndRadiationPower):
    """`i_plasma_ignited == IGNITED` (1) -- the reference run's."""

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        pden_plasma_rad_mw=From(physics),
        vol_plasma=From(physics),
        f_rad=From(stellarator),
    ):
        return calculate_heating_and_radiation_power_ignited(
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            pden_plasma_rad_mw,
            vol_plasma,
            f_rad,
        )


class HeatingAndRadiationPowerNonIgnited(HeatingAndRadiationPower):
    """`i_plasma_ignited == NON_IGNITED` (0) -- PROCESS's own default
    (`physics_variables.py:881`).
    """

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        pden_plasma_rad_mw=From(physics),
        vol_plasma=From(physics),
        f_rad=From(stellarator),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_heating_and_radiation_power_non_ignited(
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            pden_plasma_rad_mw,
            vol_plasma,
            f_rad,
            p_hcd_injected_total_mw,
        )


class RadiatedWallLoadAndFraction(ExplicitFunction):
    """The `calculate_radiated_wall_load_and_fraction` family -- the same three arms as
    `NeutronWallLoad`, applied to `.physics.p_plasma_rad_mw`.
    """

    pflux_fw_rad_mw = OutputInto(physics)
    pflux_fw_rad_max_mw = OutputInto(constraints)
    rad_fraction_total = OutputInto(physics)


class RadiatedWallLoadScaledPlasmaSurface(RadiatedWallLoadAndFraction):
    """`i_pflux_fw_neutron == SCALED_PLASMA_SURFACE_AREA` (1) -- the reference run's."""

    def __call__(
        self,
        ffwal=From(physics),
        a_plasma_surface=From(physics),
        p_plasma_rad_mw=From(physics),
        f_fw_rad_max=From(constraints),
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_radiated_wall_load_scaled_plasma_surface(
            ffwal,
            a_plasma_surface,
            p_plasma_rad_mw,
            f_fw_rad_max,
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            p_hcd_injected_total_mw,
        )


class RadiatedWallLoadFirstWallAreaPre2014(RadiatedWallLoadAndFraction):
    """`i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow == PRE_2014` (0)."""

    def __call__(
        self,
        fhole=From(fwbs),
        a_fw_total=From(first_wall),
        p_plasma_rad_mw=From(physics),
        f_fw_rad_max=From(constraints),
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_radiated_wall_load_first_wall_area_pre_2014(
            fhole,
            a_fw_total,
            p_plasma_rad_mw,
            f_fw_rad_max,
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            p_hcd_injected_total_mw,
        )


class RadiatedWallLoadFirstWallAreaComprehensive2014(RadiatedWallLoadAndFraction):
    """`i_pflux_fw_neutron == FIRST_WALL_AREA` (2) with `ipowerflow ==
    COMPREHENSIVE_2014` (1).
    """

    def __call__(
        self,
        fhole=From(fwbs),
        a_fw_total=From(first_wall),
        f_a_fw_outboard_hcd=From(fwbs),
        f_ster_div_single=From(fwbs),
        p_plasma_rad_mw=From(physics),
        f_fw_rad_max=From(constraints),
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_plasma_ohmic_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
    ):
        return calculate_radiated_wall_load_first_wall_area_comprehensive_2014(
            fhole,
            a_fw_total,
            f_a_fw_outboard_hcd,
            f_ster_div_single,
            p_plasma_rad_mw,
            f_fw_rad_max,
            f_p_alpha_plasma_deposited,
            p_alpha_total_mw,
            p_non_alpha_charged_mw,
            p_plasma_ohmic_mw,
            p_hcd_injected_total_mw,
        )


class ThermalEnergyTotals(ExplicitFunction):
    """cottax node: `calculate_thermal_energy_totals`, ports declared."""

    eden_plasma_thermal_vol_avg = OutputInto(physics)
    e_plasma_thermal_total = OutputInto(physics)

    def __call__(
        self,
        eden_plasma_electrons_thermal_vol_avg=From(physics),
        eden_plasma_ions_thermal_vol_avg=From(physics),
        e_plasma_electrons_thermal=From(physics),
        e_plasma_ions_thermal=From(physics),
    ):
        return calculate_thermal_energy_totals(
            eden_plasma_electrons_thermal_vol_avg,
            eden_plasma_ions_thermal_vol_avg,
            e_plasma_electrons_thermal,
            e_plasma_ions_thermal,
        )


def select_stellarator_beta_and_stored_energy(
    beta_fast_alpha,
    beta_beam,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_ion_density_weighted_kev,
    b_plasma_total,
    vol_plasma,
    m_ions_total_amu,
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    eps,
    rmajor,
):
    """`(beta_total_vol_avg, e_plasma_beta)` half of
    `calculate_stellarator_beta_and_rho_star` -- `rho_star` is `DimensionlessPlasma
    Parameters`' output (see `StellaratorBetaAndStoredEnergy`'s own docstring for why
    this class must not also produce it), so it is discarded here exactly as the
    declaration already discarded it.
    """
    beta_total_vol_avg, e_plasma_beta, _rho_star = (
        calculate_stellarator_beta_and_rho_star(
            beta_fast_alpha,
            beta_beam,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
            b_plasma_total,
            vol_plasma,
            m_ions_total_amu,
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            eps,
            rmajor,
        )
    )
    return beta_total_vol_avg, e_plasma_beta


class StellaratorBetaAndStoredEnergy(ExplicitFunction):
    """cottax node: `calculate_stellarator_beta_and_rho_star` minus its `rho_star`
    output -- the registerable form of `StellaratorBetaAndRhoStar` above.
    """

    beta_total_vol_avg = OutputInto(physics)
    e_plasma_beta = OutputInto(physics)

    def __call__(
        self,
        beta_fast_alpha=From(physics),
        beta_beam=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        b_plasma_total=From(physics),
        vol_plasma=From(physics),
        m_ions_total_amu=From(physics),
        nd_plasma_electron_line=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        eps=From(physics),
        rmajor=From(physics),
    ):
        return select_stellarator_beta_and_stored_energy(
            beta_fast_alpha,
            beta_beam,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_density_weighted_kev,
            nd_plasma_ions_total_vol_avg,
            temp_plasma_ion_density_weighted_kev,
            b_plasma_total,
            vol_plasma,
            m_ions_total_amu,
            nd_plasma_electron_line,
            b_plasma_toroidal_on_axis,
            eps,
            rmajor,
        )
