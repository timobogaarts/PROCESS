"""Pure-functional port of `Physics.plasma_composition` and
`Physics.calculate_effective_charge_ionisation_profiles`
(`process/models/physics/physics.py`).
"""

import functools

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    Output,
    OutputInto,
)

from functional_process.cottax.paths import current_drive, impurity_radiation, physics
from functional_process.models.physics.composition import (
    H_INDEX,
    HE_INDEX,
    calculate_effective_charge_ionisation_profiles,
    effective_charge_ionisation_profiles_from_indexed_impurities,
    plasma_composition,
    plasma_composition_ignited,
    plasma_composition_non_ignited,
)

__all__ = [
    "calculate_effective_charge_ionisation_profiles",
    "plasma_composition",
]


class PlasmaComposition(ExplicitFunction):
    """cottax node: `plasma_composition`'s outputs."""

    nd_plasma_alphas_thermal_vol_avg = OutputInto(physics)
    nd_plasma_protons_vol_avg = OutputInto(physics)
    nd_beam_ions = OutputInto(physics)
    nd_plasma_fuel_ions_vol_avg = OutputInto(physics)
    f_nd_impurity_electron_array_h = Output(
        impurity_radiation.f_nd_impurity_electron_array[H_INDEX]
    )
    """`f_nd_impurity_electron_array[0]` (PROCESS display label
    `f_nd_impurity_electrons(01)` per `naming_convention.md` § "Array elements" --
    record both, they are not the same thing).
    """
    f_nd_impurity_electron_array_he = Output(
        impurity_radiation.f_nd_impurity_electron_array[HE_INDEX]
    )
    """`f_nd_impurity_electron_array[1]` (display label `f_nd_impurity_electrons(02)`).
    """
    nd_plasma_impurities_vol_avg = OutputInto(physics)
    nd_plasma_ions_total_vol_avg = OutputInto(physics)
    f_nd_plasma_carbon_electron = OutputInto(physics)
    f_nd_plasma_oxygen_electron = OutputInto(physics)
    f_nd_plasma_iron_argon_electron = OutputInto(physics)
    n_charge_plasma_effective_vol_avg = OutputInto(physics)
    f_alpha_electron = OutputInto(physics)
    f_alpha_ion = OutputInto(physics)
    m_fuel_amu = OutputInto(physics)
    m_beam_amu = OutputInto(physics)
    m_ions_total_amu = OutputInto(physics)
    n_charge_plasma_effective_mass_weighted_vol_avg = OutputInto(physics)

    def _composition(
        self,
        arm,
        nd_plasma_electrons_vol_avg,
        f_nd_alpha_thermal_electron,
        fusden_alpha_total,
        f_nd_protium_electrons,
        proton_rate_density,
        f_nd_impurity_electron_array_2,
        f_nd_impurity_electron_array_3,
        f_nd_impurity_electron_array_4,
        f_nd_impurity_electron_array_5,
        f_nd_impurity_electron_array_6,
        f_nd_impurity_electron_array_7,
        f_nd_impurity_electron_array_8,
        f_nd_impurity_electron_array_9,
        f_nd_impurity_electron_array_10,
        f_nd_impurity_electron_array_11,
        f_nd_impurity_electron_array_12,
        f_nd_impurity_electron_array_13,
        temp_plasma_electron_vol_avg_kev,
        temp_impurity_keV_array,
        impurity_arr_zav,
        f_plasma_fuel_deuterium,
        f_plasma_fuel_tritium,
        f_plasma_fuel_helium3,
        f_temp_plasma_electron_density_vol_avg,
        f_beam_tritium,
        m_impurity_amu_array,
    ):
        """The array assembly and result reshaping both `i_plasma_ignited` occupants
        share, given the arm function that occupant is for.
        """
        # `plasma_composition` (the pure function) is unchanged -- still one 14-array
        # parameter, physics untouched. Indices 0/1 are placeholders: the function
        # never reads the *old* values there (see the class docstring), only
        # overwrites them via `.at[H_INDEX].set(...)`/`.at[HE_INDEX].set(...)`, so
        # zeros are numerically exact, not an approximation.
        placeholder = jnp.zeros_like(f_nd_impurity_electron_array_2)
        f_nd_impurity_electron_array = jnp.stack([
            placeholder,  # index 0 (H_) -- owned by this node's own Output, not read
            placeholder,  # index 1 (He) -- ditto
            f_nd_impurity_electron_array_2,
            f_nd_impurity_electron_array_3,
            f_nd_impurity_electron_array_4,
            f_nd_impurity_electron_array_5,
            f_nd_impurity_electron_array_6,
            f_nd_impurity_electron_array_7,
            f_nd_impurity_electron_array_8,
            f_nd_impurity_electron_array_9,
            f_nd_impurity_electron_array_10,
            f_nd_impurity_electron_array_11,
            f_nd_impurity_electron_array_12,
            f_nd_impurity_electron_array_13,
        ])

        results = arm(
            nd_plasma_electrons_vol_avg,
            f_nd_alpha_thermal_electron,
            fusden_alpha_total,
            f_nd_protium_electrons,
            proton_rate_density,
            f_nd_impurity_electron_array,
            temp_plasma_electron_vol_avg_kev,
            temp_impurity_keV_array,
            impurity_arr_zav,
            f_plasma_fuel_deuterium,
            f_plasma_fuel_tritium,
            f_plasma_fuel_helium3,
            f_temp_plasma_electron_density_vol_avg,
            f_beam_tritium,
            m_impurity_amu_array,
        )
        # `results[4]` is the post-update 14-array (index 4 of `plasma_composition`'s
        # return tuple, see its own docstring); this node owns its two updated entries
        # individually (`f_nd_impurity_electron_array_h`/`_he`) rather than the whole
        # array. Order must match the `Output` declarations above: results[:4] (4), the
        # two extracted H_/He_ entries (2), results[5:] (12) -- 18 total.
        return (
            *results[:4],
            results[4][H_INDEX],
            results[4][HE_INDEX],
            *results[5:],
        )


class PlasmaCompositionIgnited(PlasmaComposition):
    """`i_plasma_ignited == IGNITED` (1) -- the reference run's."""

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        f_nd_alpha_thermal_electron=From(physics),
        fusden_alpha_total=From(physics),
        f_nd_protium_electrons=From(physics),
        proton_rate_density=From(physics),
        f_nd_impurity_electron_array_2=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_impurity_keV_array=From(impurity_radiation),
        impurity_arr_zav=From(impurity_radiation),
        f_plasma_fuel_deuterium=From(physics),
        f_plasma_fuel_tritium=From(physics),
        f_plasma_fuel_helium3=From(physics),
        f_temp_plasma_electron_density_vol_avg=From(physics),
        f_beam_tritium=From(current_drive),
        m_impurity_amu_array=From(impurity_radiation),
    ):
        return self._composition(
            plasma_composition_ignited,
            nd_plasma_electrons_vol_avg,
            f_nd_alpha_thermal_electron,
            fusden_alpha_total,
            f_nd_protium_electrons,
            proton_rate_density,
            f_nd_impurity_electron_array_2,
            f_nd_impurity_electron_array_3,
            f_nd_impurity_electron_array_4,
            f_nd_impurity_electron_array_5,
            f_nd_impurity_electron_array_6,
            f_nd_impurity_electron_array_7,
            f_nd_impurity_electron_array_8,
            f_nd_impurity_electron_array_9,
            f_nd_impurity_electron_array_10,
            f_nd_impurity_electron_array_11,
            f_nd_impurity_electron_array_12,
            f_nd_impurity_electron_array_13,
            temp_plasma_electron_vol_avg_kev,
            temp_impurity_keV_array,
            impurity_arr_zav,
            f_plasma_fuel_deuterium,
            f_plasma_fuel_tritium,
            f_plasma_fuel_helium3,
            f_temp_plasma_electron_density_vol_avg,
            f_beam_tritium,
            m_impurity_amu_array,
        )


class PlasmaCompositionNonIgnited(PlasmaComposition):
    """`i_plasma_ignited == NON_IGNITED` (0) -- PROCESS's own default
    (`physics_variables.py:881`) and the conventional tokamak's.
    """

    def __call__(
        self,
        nd_plasma_electrons_vol_avg=From(physics),
        f_nd_alpha_thermal_electron=From(physics),
        fusden_alpha_total=From(physics),
        f_nd_protium_electrons=From(physics),
        proton_rate_density=From(physics),
        f_nd_beam_electron=From(physics),
        f_nd_impurity_electron_array_2=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_plasma_electron_vol_avg_kev=From(physics),
        temp_impurity_keV_array=From(impurity_radiation),
        impurity_arr_zav=From(impurity_radiation),
        f_plasma_fuel_deuterium=From(physics),
        f_plasma_fuel_tritium=From(physics),
        f_plasma_fuel_helium3=From(physics),
        f_temp_plasma_electron_density_vol_avg=From(physics),
        f_beam_tritium=From(current_drive),
        m_impurity_amu_array=From(impurity_radiation),
    ):
        return self._composition(
            functools.partial(
                plasma_composition_non_ignited,
                f_nd_beam_electron=f_nd_beam_electron,
            ),
            nd_plasma_electrons_vol_avg,
            f_nd_alpha_thermal_electron,
            fusden_alpha_total,
            f_nd_protium_electrons,
            proton_rate_density,
            f_nd_impurity_electron_array_2,
            f_nd_impurity_electron_array_3,
            f_nd_impurity_electron_array_4,
            f_nd_impurity_electron_array_5,
            f_nd_impurity_electron_array_6,
            f_nd_impurity_electron_array_7,
            f_nd_impurity_electron_array_8,
            f_nd_impurity_electron_array_9,
            f_nd_impurity_electron_array_10,
            f_nd_impurity_electron_array_11,
            f_nd_impurity_electron_array_12,
            f_nd_impurity_electron_array_13,
            temp_plasma_electron_vol_avg_kev,
            temp_impurity_keV_array,
            impurity_arr_zav,
            f_plasma_fuel_deuterium,
            f_plasma_fuel_tritium,
            f_plasma_fuel_helium3,
            f_temp_plasma_electron_density_vol_avg,
            f_beam_tritium,
            m_impurity_amu_array,
        )


class CalculateEffectiveChargeIonisationProfiles(ExplicitFunction):
    """cottax node: `calculate_effective_charge_ionisation_profiles`, ports declared."""

    n_charge_plasma_effective_profile = OutputInto(physics)
    n_charge_impurity_profile = OutputInto(impurity_radiation)

    def __call__(
        self,
        temp_plasma_electron_profile_kev=From(physics),
        f_nd_impurity_electron_array_0=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[0]
        ),
        f_nd_impurity_electron_array_1=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[1]
        ),
        f_nd_impurity_electron_array_2=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[2]
        ),
        f_nd_impurity_electron_array_3=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[3]
        ),
        f_nd_impurity_electron_array_4=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[4]
        ),
        f_nd_impurity_electron_array_5=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[5]
        ),
        f_nd_impurity_electron_array_6=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[6]
        ),
        f_nd_impurity_electron_array_7=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[7]
        ),
        f_nd_impurity_electron_array_8=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[8]
        ),
        f_nd_impurity_electron_array_9=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[9]
        ),
        f_nd_impurity_electron_array_10=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[10]
        ),
        f_nd_impurity_electron_array_11=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[11]
        ),
        f_nd_impurity_electron_array_12=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[12]
        ),
        f_nd_impurity_electron_array_13=FromExactly(
            impurity_radiation.f_nd_impurity_electron_array[13]
        ),
        temp_impurity_keV_array=From(impurity_radiation),
        impurity_arr_zav=From(impurity_radiation),
    ):
        return effective_charge_ionisation_profiles_from_indexed_impurities(
            temp_plasma_electron_profile_kev,
            f_nd_impurity_electron_array_0,
            f_nd_impurity_electron_array_1,
            f_nd_impurity_electron_array_2,
            f_nd_impurity_electron_array_3,
            f_nd_impurity_electron_array_4,
            f_nd_impurity_electron_array_5,
            f_nd_impurity_electron_array_6,
            f_nd_impurity_electron_array_7,
            f_nd_impurity_electron_array_8,
            f_nd_impurity_electron_array_9,
            f_nd_impurity_electron_array_10,
            f_nd_impurity_electron_array_11,
            f_nd_impurity_electron_array_12,
            f_nd_impurity_electron_array_13,
            temp_impurity_keV_array,
            impurity_arr_zav,
        )
