"""Pure-functional port of `process/models/physics/fusion_reactions.py`."""

from functional_process.cottax.paths import physics
from functional_process.models.physics.fusion_reactions import (
    alpha_power_beam,
    beam_fusion_cross_section,
    beam_slowing_down_state,
    beam_target_reaction_rate,
    bosch_hale_reactivity,
    calculate_deuterium_branching_trit,
    calculate_fusion_rates,
    fast_ion_pressure_integral,
    fusion_rates_from_profiles,
    hot_beam_fusion_reaction_rate_integrand,
    set_fusion_powers,
)

__all__ = [
    "alpha_power_beam",
    "beam_fusion_cross_section",
    "beam_slowing_down_state",
    "beam_target_reaction_rate",
    "bosch_hale_reactivity",
    "calculate_deuterium_branching_trit",
    "calculate_fusion_rates",
    "fast_ion_pressure_integral",
    "hot_beam_fusion_reaction_rate_integrand",
]


# cottax node: `calculate_fusion_rates`, fusing all three in-scope
# `FusionReactionRate` methods (`.deuterium_branching()`, `.calculate_fusion_rates()`,
# `.set_physics_variables()`) -- see the audit record's "cottax node" section for why.
def _fusion_rates(
    radius_plasma_profile_norm,
    temp_plasma_electron_profile_kev,
    nd_plasma_electron_profile,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    f_plasma_fuel_deuterium,
    f_plasma_fuel_tritium,
    f_plasma_fuel_helium3,
    nd_plasma_fuel_ions_vol_avg,
    nd_plasma_electrons_vol_avg,
):
    return fusion_rates_from_profiles(
        radius_plasma_profile_norm,
        temp_plasma_electron_profile_kev,
        nd_plasma_electron_profile,
        temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev,
        f_plasma_fuel_deuterium,
        f_plasma_fuel_tritium,
        f_plasma_fuel_helium3,
        nd_plasma_fuel_ions_vol_avg,
        nd_plasma_electrons_vol_avg,
    )


FusionRates = physics.function(
    _fusion_rates,
    "pden_plasma_alpha_mw",
    "pden_non_alpha_charged_mw",
    "pden_plasma_neutron_mw",
    "fusden_plasma",
    "fusden_plasma_alpha",
    "proton_rate_density",
    "sigmav_dt_average",
    "dt_power_density_plasma",
    "dhe3_power_density",
    "dd_power_density",
    "f_dd_branching_trit",
    "fusrat_plasma_dt_profile",
    "fusrat_plasma_dhe3_profile",
    "fusrat_plasma_dd_helion_profile",
    "fusrat_plasma_dd_triton_profile",
)


# cottax node: `set_fusion_powers`, unchanged, ports declared.
def _set_fusion_powers(
    f_alpha_electron,
    f_alpha_ion,
    p_beam_alpha_mw,
    pden_non_alpha_charged_mw,
    pden_plasma_neutron_mw,
    vol_plasma,
    pden_plasma_alpha_mw,
    f_p_alpha_plasma_deposited,
):
    return set_fusion_powers(
        f_alpha_electron,
        f_alpha_ion,
        p_beam_alpha_mw,
        pden_non_alpha_charged_mw,
        pden_plasma_neutron_mw,
        vol_plasma,
        pden_plasma_alpha_mw,
        f_p_alpha_plasma_deposited,
    )


SetFusionPowers = physics.function(
    _set_fusion_powers,
    "pden_neutron_total_mw",
    "p_plasma_alpha_mw",
    "p_alpha_total_mw",
    "p_plasma_neutron_mw",
    "p_neutron_total_mw",
    "p_non_alpha_charged_mw",
    "pden_alpha_total_mw",
    "f_pden_alpha_electron_mw",
    "f_pden_alpha_ions_mw",
    "p_charged_particle_mw",
    "p_fusion_total_mw",
)
