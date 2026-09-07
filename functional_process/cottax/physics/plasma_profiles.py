"""Pure-functional port of `process/models/physics/plasma_profiles.py`."""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    From,
    OutputInto,
)

from functional_process.cottax.stated import StatesValues
from functional_process.cottax.paths import divertor, physics
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


class ProfileFactors(ExplicitFunction):
    """cottax node: `calculate_profile_factors`, unchanged, ports declared."""

    pres_plasma_thermal_on_axis = OutputInto(physics)
    pres_plasma_electron_profile = OutputInto(physics)
    pres_plasma_ion_total_profile = OutputInto(physics)
    pres_plasma_thermal_total_profile = OutputInto(physics)
    pres_plasma_fuel_profile = OutputInto(physics)
    alphap = OutputInto(physics)
    pres_plasma_thermal_vol_avg = OutputInto(physics)
    j_plasma_on_axis = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_profile=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        nd_plasma_electron_on_axis=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        nd_plasma_ions_on_axis=From(physics),
        temp_plasma_ion_on_axis_kev=From(physics),
        nd_plasma_ions_total_vol_avg=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        nd_plasma_fuel_ions_vol_avg=From(physics),
        f_temp_plasma_ion_electron=From(physics),
        temp_plasma_electron_density_weighted_kev=From(physics),
        temp_plasma_ion_density_weighted_kev=From(physics),
        alphan=From(physics),
        alphat=From(physics),
        alphaj=From(physics),
        plasma_current=From(physics),
        a_plasma_poloidal=From(physics),
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


class ParabolicGradientLengths(ExplicitFunction):
    """cottax node: `calculate_parabolic_gradient_lengths`, unchanged, ports declared.
    """

    gradient_length_te = OutputInto(physics)
    gradient_length_ne = OutputInto(physics)

    def __call__(
        self,
        alphat=From(physics),
        alphan=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        nd_plasma_electron_on_axis=From(physics),
        rminor=From(physics),
    ):
        return calculate_parabolic_gradient_lengths(
            alphat,
            alphan,
            temp_plasma_electron_on_axis_kev,
            nd_plasma_electron_on_axis,
            rminor,
        )


class IonVolAvgTemperature(FixedPointFunction):
    """cottax node: `calculate_ion_vol_avg_temperature`, as a fixed point."""

    temp_plasma_ion_vol_avg_kev = OutputInto(physics)

    def step(
        self,
        f_temp_plasma_ion_electron=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
    ):
        return calculate_ion_vol_avg_temperature(
            f_temp_plasma_ion_electron,
            temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
        )


class ParabolicProfileValues(ExplicitFunction):
    """cottax node: `calculate_parabolic_profile_values`, the `i_plasma_pedestal == 0`
    arm of `parameterise_plasma`'s line-average/density-weighted tail.
    """

    f_temp_plasma_electron_density_vol_avg = OutputInto(physics)
    nd_plasma_electron_line = OutputInto(physics)
    temp_plasma_electron_line_avg_kev = OutputInto(physics)
    temp_plasma_electron_density_weighted_kev = OutputInto(physics)
    temp_plasma_ion_density_weighted_kev = OutputInto(physics)

    def __call__(
        self,
        alphan=From(physics),
        alphat=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
    ):
        return calculate_parabolic_profile_values(
            alphan,
            alphat,
            nd_plasma_electrons_vol_avg,
            temp_plasma_electron_vol_avg_kev,
            temp_plasma_ion_vol_avg_kev,
        )


class LModeProfileReset(StatesValues):
    """cottax node: `lmode_profile_reset`, the `i_plasma_pedestal == 0` arm's
    input-validation reset, as the producer it always was.
    """

    radius_plasma_pedestal_temp_norm = OutputInto(physics)
    radius_plasma_pedestal_density_norm = OutputInto(physics)
    temp_plasma_pedestal_kev = OutputInto(physics)
    temp_plasma_separatrix_kev = OutputInto(physics)
    nd_plasma_pedestal_electron = OutputInto(physics)
    nd_plasma_separatrix_electron = OutputInto(physics)
    tbeta = OutputInto(physics)


class PedestalProfileValues(ExplicitFunction):
    """cottax node: `calculate_pedestal_profile_values`, the `i_plasma_pedestal == 1`
    arm of `parameterise_plasma`'s line-average/density-weighted tail --
    `ParabolicProfileValues`' pedestal-arm counterpart.
    """

    temp_plasma_electron_density_weighted_kev = OutputInto(physics)
    temp_plasma_ion_density_weighted_kev = OutputInto(physics)
    f_temp_plasma_electron_density_vol_avg = OutputInto(physics)
    nd_plasma_electron_line = OutputInto(physics)
    temp_plasma_electron_line_avg_kev = OutputInto(physics)
    prn1 = OutputInto(divertor)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        nd_plasma_electron_profile=From(physics),
        temp_plasma_electron_profile_kev=From(physics),
        nd_plasma_electron_profile_integral=From(physics),
        temp_plasma_electron_profile_integral_kev=From(physics),
        temp_plasma_ion_vol_avg_kev=From(physics),
        temp_plasma_electron_vol_avg_kev=From(physics),
        nd_plasma_separatrix_electron=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
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
