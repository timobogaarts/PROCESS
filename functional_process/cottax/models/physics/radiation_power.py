"""Pure-functional port of `process/models/physics/radiation_power.py`."""

from cottax.interfaces import V

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
def _synchrotron_radiation_power(
    nd_plasma_electron_on_axis,
    rminor,
    b_plasma_toroidal_on_axis,
    aspect,
    alphan,
    alphat,
    tbeta,
    temp_plasma_electron_on_axis_kev,
    f_sync_reflect,
    rmajor,
    kappa,
    vol_plasma,
):
    return psync_albajar_fidone(
        nd_plasma_electron_on_axis,
        rminor,
        b_plasma_toroidal_on_axis,
        aspect,
        alphan,
        alphat,
        tbeta,
        temp_plasma_electron_on_axis_kev,
        f_sync_reflect,
        rmajor,
        kappa,
        vol_plasma,
    )


SynchrotronRadiationPower = physics.function(
    _synchrotron_radiation_power, "pden_plasma_sync_mw"
)


# cottax node: `calculate_impurity_radiation_totals`, ports declared.
# Reassembles the fourteen individually-addressed fractions and selects `imp_indices`
# (which of the 14 species are present, read at `.impurity_radiation.imp_indices`)
# before forwarding.
def _impurity_radiation_totals(
    imp_indices,
    radius_plasma_profile_norm,
    nd_plasma_electron_profile,
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
    pden_impurity_lz_nd_temp_array,
    radius_plasma_core_norm,
    f_p_plasma_core_rad_reduction,
):
    return impurity_radiation_totals_from_indexed_impurities(
        imp_indices,
        radius_plasma_profile_norm,
        nd_plasma_electron_profile,
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
        pden_impurity_lz_nd_temp_array,
        radius_plasma_core_norm,
        f_p_plasma_core_rad_reduction,
    )


ImpurityRadiationTotals = impurity_radiation.function(
    _impurity_radiation_totals,
    "pden_impurity_rad_total_mw",
    "pden_impurity_core_rad_total_mw",
    reads={
        "radius_plasma_profile_norm": physics,
        "nd_plasma_electron_profile": physics,
        "temp_plasma_electron_profile_kev": physics,
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
def _plasma_radiation_powers(
    pden_impurity_rad_total_mw, pden_impurity_core_rad_total_mw, pden_plasma_sync_mw
):
    return combine_radiation_powers(
        pden_impurity_rad_total_mw,
        pden_impurity_core_rad_total_mw,
        pden_plasma_sync_mw,
    )


PlasmaRadiationPowers = physics.function(
    _plasma_radiation_powers,
    "pden_plasma_core_rad_mw_unclipped",
    "pden_plasma_outer_rad_mw_unclipped",
    "pden_plasma_rad_mw",
    reads={
        "pden_impurity_rad_total_mw": impurity_radiation,
        "pden_impurity_core_rad_total_mw": impurity_radiation,
    },
)
