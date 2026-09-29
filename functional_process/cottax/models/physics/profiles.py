"""Pure-functional port of `process/models/physics/profiles.py`."""

from functional_process.cottax.paths import physics
from functional_process.models.physics.profiles import (
    calculate_density_profile,
    calculate_greenwald_density_fractions,
    calculate_parabolic_on_axis_densities,
    calculate_parabolic_on_axis_temperatures,
    calculate_parabolic_temperature_profile,
    calculate_pedestal_on_axis_densities,
    calculate_pedestal_on_axis_temperatures,
    calculate_pedestal_separatrix_densities,
    calculate_pedestal_temperature_profile,
    calculate_profile_grid,
    integrate_profile_y,
    ncore,
    tcore,
)

__all__ = [
    "ncore",
    "tcore",
]


# cottax node: `calculate_profile_grid`, unchanged, ports declared.
# `n_plasma_profile_elements`: the grid's resolution, read at
# `.physics.n_plasma_profile_elements` (PROCESS's own field, 201).
ProfileGrid = physics.function(
    calculate_profile_grid,
    "radius_plasma_profile_norm",
    "dradius_plasma_profile_norm",
)


# cottax node: `integrate_profile_y` on the **density** profile.
def _ne_profile_integral(nd_plasma_electron_profile, radius_plasma_profile_norm):
    return integrate_profile_y(nd_plasma_electron_profile, radius_plasma_profile_norm)


NeProfileIntegral = physics.function(
    _ne_profile_integral, "nd_plasma_electron_profile_integral"
)


# cottax node: `integrate_profile_y` on the **temperature** profile.
def _te_profile_integral(temp_plasma_electron_profile_kev, radius_plasma_profile_norm):
    return integrate_profile_y(
        temp_plasma_electron_profile_kev, radius_plasma_profile_norm
    )


TeProfileIntegral = physics.function(
    _te_profile_integral, "temp_plasma_electron_profile_integral_kev"
)


# cottax node: `calculate_density_profile`, unchanged, ports declared.
def _density_profile(
    radius_plasma_profile_norm,
    radius_plasma_pedestal_density_norm,
    nd_plasma_electron_on_axis,
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    alphan,
):
    return calculate_density_profile(
        radius_plasma_profile_norm,
        radius_plasma_pedestal_density_norm,
        nd_plasma_electron_on_axis,
        nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron,
        alphan,
    )


DensityProfile = physics.function(_density_profile, "nd_plasma_electron_profile")


# cottax node: `calculate_parabolic_temperature_profile`.
def _parabolic_temperature_profile(
    radius_plasma_profile_norm, temp_plasma_electron_on_axis_kev, alphat
):
    return calculate_parabolic_temperature_profile(
        radius_plasma_profile_norm, temp_plasma_electron_on_axis_kev, alphat
    )


ParabolicTemperatureProfile = physics.function(
    _parabolic_temperature_profile, "temp_plasma_electron_profile_kev"
)


# cottax node: `calculate_pedestal_temperature_profile`.
def _pedestal_temperature_profile(
    radius_plasma_profile_norm,
    radius_plasma_pedestal_temp_norm,
    temp_plasma_electron_on_axis_kev,
    temp_plasma_pedestal_kev,
    temp_plasma_separatrix_kev,
    alphat,
    tbeta,
):
    return calculate_pedestal_temperature_profile(
        radius_plasma_profile_norm,
        radius_plasma_pedestal_temp_norm,
        temp_plasma_electron_on_axis_kev,
        temp_plasma_pedestal_kev,
        temp_plasma_separatrix_kev,
        alphat,
        tbeta,
    )


PedestalTemperatureProfile = physics.function(
    _pedestal_temperature_profile, "temp_plasma_electron_profile_kev"
)


# cottax node: `calculate_parabolic_on_axis_densities`.
def _parabolic_on_axis_densities(
    nd_plasma_electrons_vol_avg, nd_plasma_ions_total_vol_avg, alphan
):
    return calculate_parabolic_on_axis_densities(
        nd_plasma_electrons_vol_avg, nd_plasma_ions_total_vol_avg, alphan
    )


ParabolicOnAxisDensities = physics.function(
    _parabolic_on_axis_densities, "nd_plasma_electron_on_axis", "nd_plasma_ions_on_axis"
)


# cottax node: `calculate_pedestal_on_axis_densities`.
def _pedestal_on_axis_densities(
    radius_plasma_pedestal_density_norm,
    nd_plasma_pedestal_electron,
    nd_plasma_separatrix_electron,
    nd_plasma_electrons_vol_avg,
    nd_plasma_ions_total_vol_avg,
    alphan,
):
    return calculate_pedestal_on_axis_densities(
        radius_plasma_pedestal_density_norm,
        nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron,
        nd_plasma_electrons_vol_avg,
        nd_plasma_ions_total_vol_avg,
        alphan,
    )


PedestalOnAxisDensities = physics.function(
    _pedestal_on_axis_densities, "nd_plasma_electron_on_axis", "nd_plasma_ions_on_axis"
)


# cottax node: `calculate_parabolic_on_axis_temperatures`.
def _parabolic_on_axis_temperatures(
    temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev, alphat
):
    return calculate_parabolic_on_axis_temperatures(
        temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev, alphat
    )


ParabolicOnAxisTemperatures = physics.function(
    _parabolic_on_axis_temperatures,
    "temp_plasma_electron_on_axis_kev",
    "temp_plasma_ion_on_axis_kev",
)


# cottax node: `calculate_pedestal_on_axis_temperatures`.
def _pedestal_on_axis_temperatures(
    radius_plasma_pedestal_temp_norm,
    temp_plasma_pedestal_kev,
    temp_plasma_separatrix_kev,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
    alphat,
    tbeta,
):
    return calculate_pedestal_on_axis_temperatures(
        radius_plasma_pedestal_temp_norm,
        temp_plasma_pedestal_kev,
        temp_plasma_separatrix_kev,
        temp_plasma_electron_vol_avg_kev,
        temp_plasma_ion_vol_avg_kev,
        alphat,
        tbeta,
    )


PedestalOnAxisTemperatures = physics.function(
    _pedestal_on_axis_temperatures,
    "temp_plasma_electron_on_axis_kev",
    "temp_plasma_ion_on_axis_kev",
)


# cottax node: `calculate_greenwald_density_fractions`.
def _greenwald_density_fractions(
    nd_plasma_pedestal_electron, nd_plasma_separatrix_electron, plasma_current, rminor
):
    return calculate_greenwald_density_fractions(
        nd_plasma_pedestal_electron,
        nd_plasma_separatrix_electron,
        plasma_current,
        rminor,
    )


GreenwaldDensityFractions = physics.function(
    _greenwald_density_fractions,
    "f_nd_plasma_pedestal_greenwald",
    "f_nd_plasma_separatrix_greenwald",
)


# cottax node: `calculate_pedestal_separatrix_densities`.
def _pedestal_separatrix_densities(
    f_nd_plasma_pedestal_greenwald,
    f_nd_plasma_separatrix_greenwald,
    plasma_current,
    rminor,
):
    return calculate_pedestal_separatrix_densities(
        f_nd_plasma_pedestal_greenwald,
        f_nd_plasma_separatrix_greenwald,
        plasma_current,
        rminor,
    )


PedestalSeparatrixDensities = physics.function(
    _pedestal_separatrix_densities,
    "nd_plasma_pedestal_electron",
    "nd_plasma_separatrix_electron",
)
