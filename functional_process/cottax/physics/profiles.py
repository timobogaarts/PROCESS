"""Pure-functional port of `process/models/physics/profiles.py`."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

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


class ProfileGrid(ExplicitFunction):
    """cottax node: `calculate_profile_grid`, unchanged, ports declared."""

    n_plasma_profile_elements: int = eqx.field(static=True)

    radius_plasma_profile_norm = OutputInto(physics)
    dradius_plasma_profile_norm = OutputInto(physics)

    def __call__(self):
        return calculate_profile_grid(self.n_plasma_profile_elements)


class NeProfileIntegral(ExplicitFunction):
    """cottax node: `integrate_profile_y` on the **density** profile."""

    nd_plasma_electron_profile_integral = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_profile=From(physics),
        radius_plasma_profile_norm=From(physics),
    ):
        return integrate_profile_y(
            nd_plasma_electron_profile, radius_plasma_profile_norm
        )


class TeProfileIntegral(ExplicitFunction):
    """cottax node: `integrate_profile_y` on the **temperature** profile."""

    temp_plasma_electron_profile_integral_kev = OutputInto(physics)

    def __call__(
        self,
        temp_plasma_electron_profile_kev=From(physics),
        radius_plasma_profile_norm=From(physics),
    ):
        return integrate_profile_y(
            temp_plasma_electron_profile_kev, radius_plasma_profile_norm
        )


class DensityProfile(ExplicitFunction):
    """cottax node: `calculate_density_profile`, unchanged, ports declared."""

    nd_plasma_electron_profile = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        radius_plasma_pedestal_density_norm=From(physics),
        nd_plasma_electron_on_axis=From(physics),
        nd_plasma_pedestal_electron=From(physics),
        nd_plasma_separatrix_electron=From(physics),
        alphan=From(physics),
    ):
        return calculate_density_profile(
            radius_plasma_profile_norm,
            radius_plasma_pedestal_density_norm,
            nd_plasma_electron_on_axis,
            nd_plasma_pedestal_electron,
            nd_plasma_separatrix_electron,
            alphan,
        )


class ParabolicTemperatureProfile(ExplicitFunction):
    """cottax node: `calculate_parabolic_temperature_profile`."""

    temp_plasma_electron_profile_kev = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        alphat=From(physics),
    ):
        return calculate_parabolic_temperature_profile(
            radius_plasma_profile_norm, temp_plasma_electron_on_axis_kev, alphat
        )


class PedestalTemperatureProfile(ExplicitFunction):
    """cottax node: `calculate_pedestal_temperature_profile`."""

    temp_plasma_electron_profile_kev = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        radius_plasma_pedestal_temp_norm=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        temp_plasma_pedestal_kev=From(physics),
        temp_plasma_separatrix_kev=From(physics),
        alphat=From(physics),
        tbeta=From(physics),
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


class ParabolicOnAxisDensities(ExplicitFunction):
    """cottax node: `calculate_parabolic_on_axis_densities`."""

    nd_plasma_electron_on_axis = OutputInto(physics)
    nd_plasma_ions_on_axis = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        alphan=From(physics),
    ):
        return calculate_parabolic_on_axis_densities(
            nd_plasma_electrons_vol_avg, nd_plasma_ions_total_vol_avg, alphan
        )


class PedestalOnAxisDensities(ExplicitFunction):
    """cottax node: `calculate_pedestal_on_axis_densities`."""

    nd_plasma_electron_on_axis = OutputInto(physics)
    nd_plasma_ions_on_axis = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_pedestal_density_norm=From(physics),
        nd_plasma_pedestal_electron=From(physics),
        nd_plasma_separatrix_electron=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        alphan=From(physics),
    ):
        return calculate_pedestal_on_axis_densities(
            radius_plasma_pedestal_density_norm,
            nd_plasma_pedestal_electron,
            nd_plasma_separatrix_electron,
            nd_plasma_electrons_vol_avg,
            nd_plasma_ions_total_vol_avg,
            alphan,
        )


class ParabolicOnAxisTemperatures(ExplicitFunction):
    """cottax node: `calculate_parabolic_on_axis_temperatures`."""

    temp_plasma_electron_on_axis_kev = OutputInto(physics)
    temp_plasma_ion_on_axis_kev = OutputInto(physics)

    def __call__(
        self,
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        alphat=From(physics),
    ):
        return calculate_parabolic_on_axis_temperatures(
            temp_plasma_electron_vol_avg_kev, temp_plasma_ion_vol_avg_kev, alphat
        )


class PedestalOnAxisTemperatures(ExplicitFunction):
    """cottax node: `calculate_pedestal_on_axis_temperatures`."""

    temp_plasma_electron_on_axis_kev = OutputInto(physics)
    temp_plasma_ion_on_axis_kev = OutputInto(physics)

    def __call__(
        self,
        radius_plasma_pedestal_temp_norm=From(physics),
        temp_plasma_pedestal_kev=From(physics),
        temp_plasma_separatrix_kev=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        alphat=From(physics),
        tbeta=From(physics),
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


class GreenwaldDensityFractions(ExplicitFunction):
    """cottax node: `calculate_greenwald_density_fractions`."""

    f_nd_plasma_pedestal_greenwald = OutputInto(physics)
    f_nd_plasma_separatrix_greenwald = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_pedestal_electron=From(physics),
        nd_plasma_separatrix_electron=From(physics),
        plasma_current=From(physics),
        rminor=From(physics),
    ):
        return calculate_greenwald_density_fractions(
            nd_plasma_pedestal_electron,
            nd_plasma_separatrix_electron,
            plasma_current,
            rminor,
        )


class PedestalSeparatrixDensities(ExplicitFunction):
    """cottax node: `calculate_pedestal_separatrix_densities`."""

    nd_plasma_pedestal_electron = OutputInto(physics)
    nd_plasma_separatrix_electron = OutputInto(physics)

    def __call__(
        self,
        f_nd_plasma_pedestal_greenwald=From(physics),
        f_nd_plasma_separatrix_greenwald=From(physics),
        plasma_current=From(physics),
        rminor=From(physics),
    ):
        return calculate_pedestal_separatrix_densities(
            f_nd_plasma_pedestal_greenwald,
            f_nd_plasma_separatrix_greenwald,
            plasma_current,
            rminor,
        )
