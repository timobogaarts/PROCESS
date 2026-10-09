"""Pure-functional port of `process/models/physics/radiation_power.py`."""

from cottax.assemblies.jax import V

from functional_process.cottax.paths import impurity_radiation, physics
from functional_process.models.physics.radiation_power import (
    calculate_impurity_radiation_power_density,
    calculate_impurity_radiation_totals,
    calculate_radiation_powers,
    combine_radiation_powers,
    impurity_radiation_totals_from_indexed_impurities,
    psync_albajar_fidone,
)

__all__ = [
    "calculate_impurity_radiation_power_density",
    "calculate_impurity_radiation_totals",
    "calculate_radiation_powers",
]


# cottax node: `psync_albajar_fidone`, unchanged, ports declared.
SynchrotronRadiationPower = physics.function(psync_albajar_fidone, "pden_plasma_sync_mw")


# cottax node: `calculate_impurity_radiation_totals`, ports declared.
# Reassembles the fourteen individually-addressed fractions and selects `imp_indices`
# (which of the 14 species are present, read at `.impurity_radiation.imp_indices`)
# before forwarding.
ImpurityRadiationTotals = impurity_radiation.function(
    impurity_radiation_totals_from_indexed_impurities,
    "pden_impurity_rad_total_mw",
    "pden_impurity_core_rad_total_mw",
    reads={
        "radius_plasma_profile_norm": physics.place("radius_plasma_profile_norm"),
        "nd_plasma_electron_profile": physics.place("nd_plasma_electron_profile"),
        "temp_plasma_electron_profile_kev": physics.place(
            "temp_plasma_electron_profile_kev"
        ),
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


# cottax node: `combine_radiation_powers`, unchanged, ports declared.
PlasmaRadiationPowers = physics.function(
    combine_radiation_powers,
    "pden_plasma_core_rad_mw_unclipped",
    "pden_plasma_outer_rad_mw_unclipped",
    "pden_plasma_rad_mw",
    reads={
        "pden_impurity_rad_total_mw": impurity_radiation.place(
            "pden_impurity_rad_total_mw"
        ),
        "pden_impurity_core_rad_total_mw": impurity_radiation.place(
            "pden_impurity_core_rad_total_mw"
        ),
    },
)
