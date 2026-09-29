"""Pure-functional port of `Physics.plasma_composition` and
`Physics.calculate_effective_charge_ionisation_profiles`
(`process/models/physics/physics.py`).
"""

import jax.numpy as jnp
from cottax.interfaces import V

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


# The PlasmaComposition family:
# cottax node: `plasma_composition`'s outputs.


def _impurity_array(
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
):
    """The 14-array `plasma_composition` takes, from the entries this node reads.

    Indices 0/1 are placeholders: the function never reads the *old* values there, only
    overwrites them via `.at[H_INDEX].set(...)`/`.at[HE_INDEX].set(...)`, so zeros are
    numerically exact, not an approximation.
    """
    placeholder = jnp.zeros_like(f_nd_impurity_electron_array_2)
    return jnp.stack([
        placeholder,  # index 0 (H_) -- owned by this node's own output, not read
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


def _composition_outputs(results):
    """`results[4]` is the post-update 14-array; the node owns its two updated entries
    individually rather than the whole array: results[:4], the H_/He_ entries,
    results[5:] -- 18 total, in the order of `_COMPOSITION_OUTPUTS`.
    """
    return (
        *results[:4],
        results[4][H_INDEX],
        results[4][HE_INDEX],
        *results[5:],
    )


def _plasma_composition_ignited(
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
    return _composition_outputs(
        plasma_composition_ignited(
            nd_plasma_electrons_vol_avg,
            f_nd_alpha_thermal_electron,
            fusden_alpha_total,
            f_nd_protium_electrons,
            proton_rate_density,
            _impurity_array(
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
            ),
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
    )


def _plasma_composition_non_ignited(
    nd_plasma_electrons_vol_avg,
    f_nd_alpha_thermal_electron,
    fusden_alpha_total,
    f_nd_protium_electrons,
    proton_rate_density,
    f_nd_beam_electron,
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
    return _composition_outputs(
        plasma_composition_non_ignited(
            nd_plasma_electrons_vol_avg,
            f_nd_alpha_thermal_electron,
            fusden_alpha_total,
            f_nd_protium_electrons,
            proton_rate_density,
            _impurity_array(
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
            ),
            temp_plasma_electron_vol_avg_kev,
            temp_impurity_keV_array,
            impurity_arr_zav,
            f_plasma_fuel_deuterium,
            f_plasma_fuel_tritium,
            f_plasma_fuel_helium3,
            f_temp_plasma_electron_density_vol_avg,
            f_beam_tritium,
            m_impurity_amu_array,
            f_nd_beam_electron=f_nd_beam_electron,
        )
    )


_COMPOSITION_OUTPUTS = (
    "nd_plasma_alphas_thermal_vol_avg",
    "nd_plasma_protons_vol_avg",
    "nd_beam_ions",
    "nd_plasma_fuel_ions_vol_avg",
    V.impurity_radiation.f_nd_impurity_electron_array[H_INDEX],
    V.impurity_radiation.f_nd_impurity_electron_array[HE_INDEX],
    "nd_plasma_impurities_vol_avg",
    "nd_plasma_ions_total_vol_avg",
    "f_nd_plasma_carbon_electron",
    "f_nd_plasma_oxygen_electron",
    "f_nd_plasma_iron_argon_electron",
    "n_charge_plasma_effective_vol_avg",
    "f_alpha_electron",
    "f_alpha_ion",
    "m_fuel_amu",
    "m_beam_amu",
    "m_ions_total_amu",
    "n_charge_plasma_effective_mass_weighted_vol_avg",
)

_COMPOSITION_READS = {
    "f_nd_impurity_electron_array_2": V.impurity_radiation.f_nd_impurity_electron_array[
        2
    ],
    "f_nd_impurity_electron_array_3": V.impurity_radiation.f_nd_impurity_electron_array[
        3
    ],
    "f_nd_impurity_electron_array_4": V.impurity_radiation.f_nd_impurity_electron_array[
        4
    ],
    "f_nd_impurity_electron_array_5": V.impurity_radiation.f_nd_impurity_electron_array[
        5
    ],
    "f_nd_impurity_electron_array_6": V.impurity_radiation.f_nd_impurity_electron_array[
        6
    ],
    "f_nd_impurity_electron_array_7": V.impurity_radiation.f_nd_impurity_electron_array[
        7
    ],
    "f_nd_impurity_electron_array_8": V.impurity_radiation.f_nd_impurity_electron_array[
        8
    ],
    "f_nd_impurity_electron_array_9": V.impurity_radiation.f_nd_impurity_electron_array[
        9
    ],
    "f_nd_impurity_electron_array_10": V.impurity_radiation.f_nd_impurity_electron_array[
        10
    ],
    "f_nd_impurity_electron_array_11": V.impurity_radiation.f_nd_impurity_electron_array[
        11
    ],
    "f_nd_impurity_electron_array_12": V.impurity_radiation.f_nd_impurity_electron_array[
        12
    ],
    "f_nd_impurity_electron_array_13": V.impurity_radiation.f_nd_impurity_electron_array[
        13
    ],
    "temp_impurity_keV_array": impurity_radiation,
    "impurity_arr_zav": impurity_radiation,
    "f_beam_tritium": current_drive,
    "m_impurity_amu_array": impurity_radiation,
}


# `i_plasma_ignited == IGNITED` (1) -- the reference run's.
PlasmaCompositionIgnited = physics.function(
    _plasma_composition_ignited, *_COMPOSITION_OUTPUTS, reads=_COMPOSITION_READS
)


# `i_plasma_ignited == NON_IGNITED` (0) -- PROCESS's own default
# (`physics_variables.py:881`) and the conventional tokamak's.
PlasmaCompositionNonIgnited = physics.function(
    _plasma_composition_non_ignited, *_COMPOSITION_OUTPUTS, reads=_COMPOSITION_READS
)


# cottax node: `calculate_effective_charge_ionisation_profiles`, ports declared.
def _calculate_effective_charge_ionisation_profiles(
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


CalculateEffectiveChargeIonisationProfiles = physics.function(
    _calculate_effective_charge_ionisation_profiles,
    "n_charge_plasma_effective_profile",
    V.impurity_radiation.n_charge_impurity_profile,
    reads={
        "f_nd_impurity_electron_array_0": V.impurity_radiation.f_nd_impurity_electron_array[
            0
        ],
        "f_nd_impurity_electron_array_1": V.impurity_radiation.f_nd_impurity_electron_array[
            1
        ],
        "f_nd_impurity_electron_array_2": V.impurity_radiation.f_nd_impurity_electron_array[
            2
        ],
        "f_nd_impurity_electron_array_3": V.impurity_radiation.f_nd_impurity_electron_array[
            3
        ],
        "f_nd_impurity_electron_array_4": V.impurity_radiation.f_nd_impurity_electron_array[
            4
        ],
        "f_nd_impurity_electron_array_5": V.impurity_radiation.f_nd_impurity_electron_array[
            5
        ],
        "f_nd_impurity_electron_array_6": V.impurity_radiation.f_nd_impurity_electron_array[
            6
        ],
        "f_nd_impurity_electron_array_7": V.impurity_radiation.f_nd_impurity_electron_array[
            7
        ],
        "f_nd_impurity_electron_array_8": V.impurity_radiation.f_nd_impurity_electron_array[
            8
        ],
        "f_nd_impurity_electron_array_9": V.impurity_radiation.f_nd_impurity_electron_array[
            9
        ],
        "f_nd_impurity_electron_array_10": V.impurity_radiation.f_nd_impurity_electron_array[
            10
        ],
        "f_nd_impurity_electron_array_11": V.impurity_radiation.f_nd_impurity_electron_array[
            11
        ],
        "f_nd_impurity_electron_array_12": V.impurity_radiation.f_nd_impurity_electron_array[
            12
        ],
        "f_nd_impurity_electron_array_13": V.impurity_radiation.f_nd_impurity_electron_array[
            13
        ],
    },
)
