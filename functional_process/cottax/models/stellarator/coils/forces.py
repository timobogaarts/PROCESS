"""Pure-functional port of `process/models/stellarator/coils/forces.py` (registry #11).
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    stellarator,
    stellarator_config,
    tfcoil,
)
from functional_process.models.stellarator.coils.forces import (
    calculate_centering_force_avg_mn,  # noqa: F401
    calculate_centering_force_max_mn,  # noqa: F401
    calculate_centering_force_min_mn,  # noqa: F401
    calculate_max_force_density,
    calculate_max_force_density_mnm,  # noqa: F401
    calculate_max_lateral_force_density,  # noqa: F401
    calculate_max_radial_force_density,  # noqa: F401
    calculate_maximum_stress,
)


class MaxForceDensity(ExplicitFunction):
    """cottax node: `calculate_max_force_density`."""

    max_force_density = OutputInto(tfcoil)

    def __call__(
        self,
        a_tf_wp_no_insulation=From(tfcoil),
        stella_config_max_force_density=From(stellarator_config),
        f_st_i_total=From(stellarator),
        f_st_n_coils=From(stellarator),
        b_tf_inboard_peak_symmetric=From(tfcoil),
        stella_config_wp_bmax=From(stellarator_config),
        stella_config_wp_area=From(stellarator_config),
    ):
        return calculate_max_force_density(
            a_tf_wp_no_insulation,
            stella_config_max_force_density,
            f_st_i_total,
            f_st_n_coils,
            b_tf_inboard_peak_symmetric,
            stella_config_wp_bmax,
            stella_config_wp_area,
        )


class MaximumStress(ExplicitFunction):
    """cottax node: `calculate_maximum_stress`."""

    sig_tf_wp = OutputInto(tfcoil)

    def __call__(
        self,
        max_force_density=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
    ):
        return calculate_maximum_stress(max_force_density, dr_tf_wp_with_insulation)
