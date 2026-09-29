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
NeProfileIntegral = physics.function(
    integrate_profile_y,
    "nd_plasma_electron_profile_integral",
    reads={
        "profile_y": "nd_plasma_electron_profile",
        "profile_x": "radius_plasma_profile_norm",
    },
)


# cottax node: `integrate_profile_y` on the **temperature** profile.
TeProfileIntegral = physics.function(
    integrate_profile_y,
    "temp_plasma_electron_profile_integral_kev",
    reads={
        "profile_y": "temp_plasma_electron_profile_kev",
        "profile_x": "radius_plasma_profile_norm",
    },
)


# cottax node: `calculate_density_profile`, unchanged, ports declared.
DensityProfile = physics.function(
    calculate_density_profile,
    "nd_plasma_electron_profile",
    reads={"rho": "radius_plasma_profile_norm"},
)


# cottax node: `calculate_parabolic_temperature_profile`.
ParabolicTemperatureProfile = physics.function(
    calculate_parabolic_temperature_profile,
    "temp_plasma_electron_profile_kev",
    reads={"rho": "radius_plasma_profile_norm"},
)


# cottax node: `calculate_pedestal_temperature_profile`.
PedestalTemperatureProfile = physics.function(
    calculate_pedestal_temperature_profile,
    "temp_plasma_electron_profile_kev",
    reads={"rho": "radius_plasma_profile_norm"},
)


# cottax node: `calculate_parabolic_on_axis_densities`.
ParabolicOnAxisDensities = physics.function(
    calculate_parabolic_on_axis_densities,
    "nd_plasma_electron_on_axis",
    "nd_plasma_ions_on_axis",
)


# cottax node: `calculate_pedestal_on_axis_densities`.
PedestalOnAxisDensities = physics.function(
    calculate_pedestal_on_axis_densities,
    "nd_plasma_electron_on_axis",
    "nd_plasma_ions_on_axis",
)


# cottax node: `calculate_parabolic_on_axis_temperatures`.
ParabolicOnAxisTemperatures = physics.function(
    calculate_parabolic_on_axis_temperatures,
    "temp_plasma_electron_on_axis_kev",
    "temp_plasma_ion_on_axis_kev",
)


# cottax node: `calculate_pedestal_on_axis_temperatures`.
PedestalOnAxisTemperatures = physics.function(
    calculate_pedestal_on_axis_temperatures,
    "temp_plasma_electron_on_axis_kev",
    "temp_plasma_ion_on_axis_kev",
)


# cottax node: `calculate_greenwald_density_fractions`.
GreenwaldDensityFractions = physics.function(
    calculate_greenwald_density_fractions,
    "f_nd_plasma_pedestal_greenwald",
    "f_nd_plasma_separatrix_greenwald",
)


# cottax node: `calculate_pedestal_separatrix_densities`.
PedestalSeparatrixDensities = physics.function(
    calculate_pedestal_separatrix_densities,
    "nd_plasma_pedestal_electron",
    "nd_plasma_separatrix_electron",
)
