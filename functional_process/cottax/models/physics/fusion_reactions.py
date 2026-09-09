"""Pure-functional port of `process/models/physics/fusion_reactions.py`."""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

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


class FusionRates(ExplicitFunction):
    """cottax node: `calculate_fusion_rates`, fusing all three in-scope
    `FusionReactionRate` methods (`.deuterium_branching()`, `.calculate_fusion_rates()`,
    `.set_physics_variables()`) -- see the audit record's "cottax node" section for why.
    """

    pden_plasma_alpha_mw = OutputInto(physics)
    pden_non_alpha_charged_mw = OutputInto(physics)
    pden_plasma_neutron_mw = OutputInto(physics)
    fusden_plasma = OutputInto(physics)
    fusden_plasma_alpha = OutputInto(physics)
    proton_rate_density = OutputInto(physics)
    sigmav_dt_average = OutputInto(physics)
    dt_power_density_plasma = OutputInto(physics)
    dhe3_power_density = OutputInto(physics)
    dd_power_density = OutputInto(physics)
    f_dd_branching_trit = OutputInto(physics)
    fusrat_plasma_dt_profile = OutputInto(physics)
    fusrat_plasma_dhe3_profile = OutputInto(physics)
    fusrat_plasma_dd_helion_profile = OutputInto(physics)
    fusrat_plasma_dd_triton_profile = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        nd_plasma_electron_profile=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        f_plasma_fuel_deuterium=From(physics),
        f_plasma_fuel_tritium=From(physics),
        f_plasma_fuel_helium3=From(physics),
        nd_plasma_fuel_ions_vol_avg=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
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


class SetFusionPowers(ExplicitFunction):
    """cottax node: `set_fusion_powers`, unchanged, ports declared."""

    pden_neutron_total_mw = OutputInto(physics)
    p_plasma_alpha_mw = OutputInto(physics)
    p_alpha_total_mw = OutputInto(physics)
    p_plasma_neutron_mw = OutputInto(physics)
    p_neutron_total_mw = OutputInto(physics)
    p_non_alpha_charged_mw = OutputInto(physics)
    pden_alpha_total_mw = OutputInto(physics)
    f_pden_alpha_electron_mw = OutputInto(physics)
    f_pden_alpha_ions_mw = OutputInto(physics)
    p_charged_particle_mw = OutputInto(physics)
    p_fusion_total_mw = OutputInto(physics)

    def __call__(
        self,
        f_alpha_electron=From(physics),
        f_alpha_ion=From(physics),
        p_beam_alpha_mw=From(physics),
        pden_non_alpha_charged_mw=From(physics),
        pden_plasma_neutron_mw=From(physics),
        vol_plasma=From(physics),
        pden_plasma_alpha_mw=From(physics),
        f_p_alpha_plasma_deposited=From(physics),
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
