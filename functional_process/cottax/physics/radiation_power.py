"""Pure-functional port of `process/models/physics/radiation_power.py`."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

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


class SynchrotronRadiationPower(ExplicitFunction):
    """cottax node: `psync_albajar_fidone`, unchanged, ports declared."""

    pden_plasma_sync_mw = OutputInto(physics)

    def __call__(
        self,
        nd_plasma_electron_on_axis=From(physics),
        rminor=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        aspect=From(physics),
        alphan=From(physics),
        alphat=From(physics),
        tbeta=From(physics),
        temp_plasma_electron_on_axis_kev=From(physics),
        f_sync_reflect=From(physics),
        rmajor=From(physics),
        kappa=From(physics),
        vol_plasma=From(physics),
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


class ImpurityRadiationTotals(ExplicitFunction):
    """cottax node: `calculate_impurity_radiation_totals`, ports declared."""

    imp_indices: tuple[int, ...] = eqx.field(static=True)
    """Which of the 14 species are present -- a graph-assembly-time fact."""

    pden_impurity_rad_total_mw = OutputInto(impurity_radiation)
    pden_impurity_core_rad_total_mw = OutputInto(impurity_radiation)

    def __call__(
        self,
        radius_plasma_profile_norm=From(physics),
        nd_plasma_electron_profile=From(physics),
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
        pden_impurity_lz_nd_temp_array=From(impurity_radiation),
        radius_plasma_core_norm=From(impurity_radiation),
        f_p_plasma_core_rad_reduction=From(impurity_radiation),
    ):
        """Reassembles the fourteen individually-addressed fractions and selects
        `imp_indices` before forwarding.
        """
        return impurity_radiation_totals_from_indexed_impurities(
            self.imp_indices,
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


class PlasmaRadiationPowers(ExplicitFunction):
    """cottax node: `combine_radiation_powers`, unchanged, ports declared."""

    pden_plasma_core_rad_mw_unclipped = OutputInto(physics)
    pden_plasma_outer_rad_mw_unclipped = OutputInto(physics)
    pden_plasma_rad_mw = OutputInto(physics)

    def __call__(
        self,
        pden_impurity_rad_total_mw=From(impurity_radiation),
        pden_impurity_core_rad_total_mw=From(impurity_radiation),
        pden_plasma_sync_mw=From(physics),
    ):
        return combine_radiation_powers(
            pden_impurity_rad_total_mw,
            pden_impurity_core_rad_total_mw,
            pden_plasma_sync_mw,
        )
