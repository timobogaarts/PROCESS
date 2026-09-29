"""Pure-functional port of `process/models/physics/plasma_profiles.py`."""

from cottax.interfaces import V

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
def _profile_factors(
    nd_plasma_electron_profile,
    temp_plasma_electron_profile_kev,
    nd_plasma_electron_on_axis,
    temp_plasma_electron_on_axis_kev,
    nd_plasma_ions_on_axis,
    temp_plasma_ion_on_axis_kev,
    nd_plasma_ions_total_vol_avg,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    f_temp_plasma_ion_electron,
    temp_plasma_electron_density_weighted_kev,
    temp_plasma_ion_density_weighted_kev,
    alphan,
    alphat,
    alphaj,
    plasma_current,
    a_plasma_poloidal,
):
    return calculate_profile_factors(
        nd_plasma_electron_profile,
        temp_plasma_electron_profile_kev,
        nd_plasma_electron_on_axis,
        temp_plasma_electron_on_axis_kev,
        nd_plasma_ions_on_axis,
        temp_plasma_ion_on_axis_kev,
        nd_plasma_ions_total_vol_avg,
        nd_plasma_electrons_vol_avg,
        nd_plasma_fuel_ions_vol_avg,
        f_temp_plasma_ion_electron,
        temp_plasma_electron_density_weighted_kev,
        temp_plasma_ion_density_weighted_kev,
        alphan,
        alphat,
        alphaj,
        plasma_current,
        a_plasma_poloidal,
    )


ProfileFactors = physics.function(
    _profile_factors,
    "pres_plasma_thermal_on_axis",
    "pres_plasma_electron_profile",
    "pres_plasma_ion_total_profile",
    "pres_plasma_thermal_total_profile",
    "pres_plasma_fuel_profile",
    "alphap",
    "pres_plasma_thermal_vol_avg",
    "j_plasma_on_axis",
)


# cottax node: `calculate_parabolic_gradient_lengths`, unchanged, ports declared.
def _parabolic_gradient_lengths(
    alphat, alphan, temp_plasma_electron_on_axis_kev, nd_plasma_electron_on_axis, rminor
):
    return calculate_parabolic_gradient_lengths(
        alphat,
        alphan,
        temp_plasma_electron_on_axis_kev,
        nd_plasma_electron_on_axis,
        rminor,
    )


ParabolicGradientLengths = physics.function(
    _parabolic_gradient_lengths, "gradient_length_te", "gradient_length_ne"
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
def _ion_vol_avg_temperature(
    f_temp_plasma_ion_electron,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev_in,
):
    return calculate_ion_vol_avg_temperature(
        f_temp_plasma_ion_electron,
        temp_plasma_electron_vol_avg_kev,
        temp_plasma_ion_vol_avg_kev_in,
    )


IonVolAvgTemperature = physics.function(
    _ion_vol_avg_temperature, "temp_plasma_ion_vol_avg_kev"
)


# cottax node: `calculate_parabolic_profile_values`, the `i_plasma_pedestal == 0`
# arm of `parameterise_plasma`'s line-average/density-weighted tail.
def _parabolic_profile_values(
    alphan,
    alphat,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
):
    return calculate_parabolic_profile_values(
        alphan,
        alphat,
        nd_plasma_electrons_vol_avg,
        temp_plasma_electron_vol_avg_kev,
        temp_plasma_ion_vol_avg_kev,
    )


ParabolicProfileValues = physics.function(
    _parabolic_profile_values,
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
def _pedestal_profile_values(
    radius_plasma_profile_norm,
    nd_plasma_electron_profile,
    temp_plasma_electron_profile_kev,
    nd_plasma_electron_profile_integral,
    temp_plasma_electron_profile_integral_kev,
    temp_plasma_ion_vol_avg_kev,
    temp_plasma_electron_vol_avg_kev,
    nd_plasma_separatrix_electron,
    nd_plasma_electrons_vol_avg,
):
    return calculate_pedestal_profile_values(
        radius_plasma_profile_norm,
        nd_plasma_electron_profile,
        temp_plasma_electron_profile_kev,
        nd_plasma_electron_profile_integral,
        temp_plasma_electron_profile_integral_kev,
        temp_plasma_ion_vol_avg_kev,
        temp_plasma_electron_vol_avg_kev,
        nd_plasma_separatrix_electron,
        nd_plasma_electrons_vol_avg,
    )


PedestalProfileValues = physics.function(
    _pedestal_profile_values,
    "temp_plasma_electron_density_weighted_kev",
    "temp_plasma_ion_density_weighted_kev",
    "f_temp_plasma_electron_density_vol_avg",
    "nd_plasma_electron_line",
    "temp_plasma_electron_line_avg_kev",
    V.divertor.prn1,
)
