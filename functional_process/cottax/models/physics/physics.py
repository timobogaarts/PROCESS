"""Pure-functional port of the **tokamak arm** of `process/models/physics/physics.py`.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import current_drive, physics, times
from functional_process.models.physics.physics import (
    calculate_beta_limit_from_norm,
    calculate_beta_norm_max_wesson,
    calculate_continuous_plant_ramp_times,
    calculate_coulomb_logarithm_ion_electron,
    calculate_pflux_plasma_surface_neutron_avg_mw,
    calculate_plasma_energy_from_beta,
    calculate_poloidal_beta,
    calculate_pulsed_plant_ramp_times,
    calculate_separatrix_power,
    calculate_surface_averaged_poloidal_field_amperes,
    calculate_thermal_beta,
    calculate_toroidal_beta,
    calculate_total_radiation_power,
    calculate_unclipped_radiation_powers,
    force_positive_separatrix_power,
    plasma_ohmic_heating,
)


class SurfaceAveragedPoloidalField(ExplicitFunction):
    """The family that owns `.physics.b_plasma_surface_poloidal_average`."""


class SurfaceAveragedPoloidalFieldAmperes(SurfaceAveragedPoloidalField):
    """`i_plasma_current != PENG_DIVERTOR_SCALING`: Ampere's law over the perimeter."""

    b_plasma_surface_poloidal_average = OutputInto(physics)

    def __call__(
        self,
        plasma_current=From(physics),
        len_plasma_poloidal=From(physics),
    ):
        return calculate_surface_averaged_poloidal_field_amperes(
            plasma_current,
            len_plasma_poloidal,
        )


class UnclippedRadiationPowers(ExplicitFunction):
    """cottax node: `calculate_unclipped_radiation_powers`, ports declared."""

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
        return calculate_unclipped_radiation_powers(
            pden_plasma_core_rad_mw_unclipped,
            pden_plasma_outer_rad_mw_unclipped,
            vol_plasma,
        )


class TotalRadiationPower(ExplicitFunction):
    """cottax node: `calculate_total_radiation_power`, ports declared."""

    p_plasma_rad_mw = OutputInto(physics)

    def __call__(
        self,
        pden_plasma_rad_mw=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_total_radiation_power(pden_plasma_rad_mw, vol_plasma)


class SeparatrixPower(ExplicitFunction):
    """The family that owns `.physics.p_plasma_separatrix_mw_raw`."""


class SeparatrixPowerNonIgnited(SeparatrixPower):
    """`i_plasma_ignited == NON_IGNITED`: injected heating crosses the separatrix."""

    p_plasma_separatrix_mw_raw = OutputInto(physics)

    def __call__(
        self,
        f_p_alpha_plasma_deposited=From(physics),
        p_alpha_total_mw=From(physics),
        p_non_alpha_charged_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        p_plasma_ohmic_mw=From(physics),
        p_plasma_rad_mw=From(physics),
    ):
        return calculate_separatrix_power(
            f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
            p_alpha_total_mw=p_alpha_total_mw,
            p_non_alpha_charged_mw=p_non_alpha_charged_mw,
            p_hcd_injected_total_mw=p_hcd_injected_total_mw,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
            p_plasma_rad_mw=p_plasma_rad_mw,
        )


class PositiveSeparatrixPower(ExplicitFunction):
    """cottax node: `force_positive_separatrix_power`, ports declared."""

    p_plasma_separatrix_mw = OutputInto(physics)

    def __call__(
        self,
        p_plasma_separatrix_mw_raw=From(physics),
    ):
        return force_positive_separatrix_power(p_plasma_separatrix_mw_raw)


class PulseRampTimes(ExplicitFunction):
    """The family that owns the plasma-current ramp times, `physics.py:463-498`."""


class PulseRampTimesPulsedDefault(PulseRampTimes):
    """`i_pulsed_plant == 1` and `pulsetimings == 0` -- `large_tokamak_eval`'s arm."""

    t_plant_pulse_plasma_current_ramp_up = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_down = OutputInto(times)

    def __call__(
        self,
        plasma_current=From(physics),
    ):
        return calculate_pulsed_plant_ramp_times(plasma_current)


class PulseRampTimesContinuousDefault(PulseRampTimes):
    """`i_pulsed_plant != 1` and `i_t_current_ramp_up == 0` -- the spherical tokamaks'.
    """

    t_plant_pulse_plasma_current_ramp_up = OutputInto(times)
    t_plant_pulse_coil_precharge = OutputInto(times)
    t_plant_pulse_plasma_current_ramp_down = OutputInto(times)

    def __call__(
        self,
        plasma_current=From(physics),
    ):
        return calculate_continuous_plant_ramp_times(plasma_current)


class PlasmaEnergyFromBeta(ExplicitFunction):
    """cottax node: `calculate_plasma_energy_from_beta`, total-beta binding."""

    e_plasma_beta = OutputInto(physics)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        b_plasma_total=From(physics),
        vol_plasma=From(physics),
    ):
        return calculate_plasma_energy_from_beta(
            beta_total_vol_avg,
            b_plasma_total,
            vol_plasma,
        )


class PlasmaOhmicHeating(ExplicitFunction):
    """cottax node: `plasma_ohmic_heating`."""

    pden_plasma_ohmic_mw = OutputInto(physics)
    p_plasma_ohmic_mw = OutputInto(physics)
    f_res_plasma_neo = OutputInto(physics)
    res_plasma = OutputInto(physics)

    def __call__(
        self,
        f_c_plasma_inductive=From(physics),
        kappa95=From(physics),
        plasma_current=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        vol_plasma=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        plasma_res_factor=From(physics),
    ):
        return plasma_ohmic_heating(
            f_c_plasma_inductive=f_c_plasma_inductive,
            kappa95=kappa95,
            plasma_current=plasma_current,
            rmajor=rmajor,
            rminor=rminor,
            temp_plasma_electron_density_weighted_kev=(
                temp_plasma_electron_density_weighted_kev
            ),
            vol_plasma=vol_plasma,
            zeff=n_charge_plasma_effective_vol_avg,
            plasma_res_factor=plasma_res_factor,
        )


class CoulombLogarithmIonElectron(ExplicitFunction):
    """cottax node: `calculate_coulomb_logarithm_ion_electron`."""

    dlamie = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
    ):
        return calculate_coulomb_logarithm_ion_electron(
            nd_plasma_electrons_vol_avg, temp_plasma_electron_vol_avg_kev
        )


class PlasmaSurfaceNeutronFlux(ExplicitFunction):
    """cottax node: `calculate_pflux_plasma_surface_neutron_avg_mw`."""

    pflux_plasma_surface_neutron_avg_mw = OutputInto(physics)

    def __call__(
        self,
        p_neutron_total_mw=From(physics),
        a_plasma_surface=From(physics),
    ):
        return calculate_pflux_plasma_surface_neutron_avg_mw(
            p_neutron_total_mw, a_plasma_surface
        )


class BetaNormMaxWesson(ExplicitFunction):
    """cottax node: `calculate_beta_norm_max_wesson`."""

    beta_norm_max = OutputInto(physics)

    def __call__(self, ind_plasma_internal_norm=From(physics)):
        return calculate_beta_norm_max_wesson(ind_plasma_internal_norm)


class BetaLimitFromNorm(ExplicitFunction):
    """cottax node: `calculate_beta_limit_from_norm`. Unswitched."""

    beta_vol_avg_max = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        beta_norm_max=From(physics),
        plasma_current=From(physics),
        rminor=From(physics),
    ):
        return calculate_beta_limit_from_norm(
            b_plasma_toroidal_on_axis,
            beta_norm_max,
            plasma_current,
            rminor,
        )


class ToroidalBeta(ExplicitFunction):
    """cottax node: `calculate_toroidal_beta`. Unswitched."""

    beta_toroidal_vol_avg = OutputInto(physics)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        b_plasma_total=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
    ):
        return calculate_toroidal_beta(
            beta_total_vol_avg,
            b_plasma_total,
            b_plasma_toroidal_on_axis,
        )


class PoloidalBeta(ExplicitFunction):
    """cottax node: `calculate_poloidal_beta`."""

    beta_poloidal_vol_avg = OutputInto(physics)

    def __call__(
        self,
        b_plasma_total=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
        beta_total_vol_avg=From(physics),
    ):
        return calculate_poloidal_beta(
            b_plasma_total,
            b_plasma_surface_poloidal_average,
            beta_total_vol_avg,
        )


class ThermalBeta(ExplicitFunction):
    """cottax node: `calculate_thermal_beta`."""

    beta_thermal_vol_avg = OutputInto(physics)

    def __call__(
        self,
        beta_total_vol_avg=From(physics),
        beta_fast_alpha=From(physics),
        beta_beam=From(physics),
    ):
        return calculate_thermal_beta(beta_total_vol_avg, beta_fast_alpha, beta_beam)
