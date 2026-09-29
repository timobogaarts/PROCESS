"""Pure-functional port of `process/models/stellarator/coils/forces.py`.

Registry #11.
"""

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

# cottax node: `calculate_max_force_density`.
MaxForceDensity = tfcoil.function(
    calculate_max_force_density,
    "max_force_density",
    reads={
        "stella_config_max_force_density": stellarator_config,
        "f_st_i_total": stellarator,
        "f_st_n_coils": stellarator,
        "stella_config_wp_bmax": stellarator_config,
        "stella_config_wp_area": stellarator_config,
    },
)


# cottax node: `calculate_maximum_stress`.
MaximumStress = tfcoil.function(calculate_maximum_stress, "sig_tf_wp")
