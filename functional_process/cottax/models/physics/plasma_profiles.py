"""Pure-functional port of `process/models/physics/plasma_profiles.py`."""

from cottax.cx import V

from functional_process.cottax.paths import physics
from functional_process.cottax.stated import states
from functional_process.models.physics.plasma_profiles import (
    calculate_ion_vol_avg_temperature,
    calculate_parabolic_gradient_lengths,
    calculate_parabolic_profile_values,
    calculate_pedestal_profile_values,
    calculate_profile_factors,
    lmode_profile_reset,
)

__all__ = [
    "lmode_profile_reset",
]


# cottax node: `calculate_profile_factors`, unchanged, ports declared.
ProfileFactors = physics.function(
    calculate_profile_factors,
    "pres_plasma_thermal_on_axis",
    "pres_plasma_electron_profile",
    "pres_plasma_ion_total_profile",
    "pres_plasma_thermal_total_profile",
    "pres_plasma_fuel_profile",
    "alphap",
    "pres_plasma_thermal_vol_avg",
    "j_plasma_on_axis",
    reads={
        "ne_profile_y": "nd_plasma_electron_profile",
        "te_profile_y": "temp_plasma_electron_profile_kev",
    },
)


# cottax node: `calculate_parabolic_gradient_lengths`, unchanged, ports declared.
ParabolicGradientLengths = physics.function(
    calculate_parabolic_gradient_lengths, "gradient_length_te", "gradient_length_ne"
)


# cottax node: `calculate_ion_vol_avg_temperature`.
#
# **A plain node, not a fixed point.** The incumbent `temp_plasma_ion_vol_avg_kev`
# this selects between is PROCESS's *input* value -- the field as the IN.DAT left it,
# which is exactly what `f_temp_plasma_ion_electron <= 0` means by "use the input
# directly" -- and not a previous iterate of this node. Read as
# `.physics.temp_plasma_ion_vol_avg_kev_in`, a free place `evaluate.KNOWN_MINT_VALUES`
# resolves back to the same PROCESS field, the self-read is gone and with it the
# minted problem, the cut and the driver: PROCESS stores the input and the result in
# one field, and only that made this look like a solve.
#
# `u = g(u)` is a fixed point worth the name only when the value read can be this
# node's own earlier output. It never could be here: on the `> 0` arm `g` does not
# depend on `u` at all, and on the other arm `g` is the identity.
IonVolAvgTemperature = physics.function(
    calculate_ion_vol_avg_temperature,
    "temp_plasma_ion_vol_avg_kev",
    reads={"temp_plasma_ion_vol_avg_kev": "temp_plasma_ion_vol_avg_kev_in"},
)


# cottax node: `calculate_parabolic_profile_values`, the `i_plasma_pedestal == 0`
# arm of `parameterise_plasma`'s line-average/density-weighted tail.
ParabolicProfileValues = physics.function(
    calculate_parabolic_profile_values,
    "f_temp_plasma_electron_density_vol_avg",
    "nd_plasma_electron_line",
    "temp_plasma_electron_line_avg_kev",
    "temp_plasma_electron_density_weighted_kev",
    "temp_plasma_ion_density_weighted_kev",
)


# cottax node: `lmode_profile_reset`, the `i_plasma_pedestal == 0` arm's
# input-validation reset, as the producer it always was.
LModeProfileReset = states(
    physics,
    "radius_plasma_pedestal_temp_norm",
    "radius_plasma_pedestal_density_norm",
    "temp_plasma_pedestal_kev",
    "temp_plasma_separatrix_kev",
    "nd_plasma_pedestal_electron",
    "nd_plasma_separatrix_electron",
    "tbeta",
)


# cottax node: `calculate_pedestal_profile_values`, the `i_plasma_pedestal == 1`
# arm of `parameterise_plasma`'s line-average/density-weighted tail --
# `ParabolicProfileValues`' pedestal-arm counterpart.
PedestalProfileValues = physics.function(
    calculate_pedestal_profile_values,
    "temp_plasma_electron_density_weighted_kev",
    "temp_plasma_ion_density_weighted_kev",
    "f_temp_plasma_electron_density_vol_avg",
    "nd_plasma_electron_line",
    "temp_plasma_electron_line_avg_kev",
    V.divertor.prn1,
    reads={
        "profile_x": "radius_plasma_profile_norm",
        "ne_profile_y": "nd_plasma_electron_profile",
        "te_profile_y": "temp_plasma_electron_profile_kev",
        "ne_profile_integ": "nd_plasma_electron_profile_integral",
        "te_profile_integ": "temp_plasma_electron_profile_integral_kev",
    },
)
