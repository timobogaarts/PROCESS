"""Pure-functional port of `Stellarator.st_new_config`/`st_geom`.

Chunk 1C of unit #1.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces import V

from functional_process.cottax.paths import (
    physics,
    stellarator,
    stellarator_config,
)
from functional_process.models.stellarator.geometry import (
    calculate_default_aspect_ratio,
    calculate_stellarator_plasma_geometry,
    calculate_stellarator_scaling_factors,
)

# cottax node: `calculate_default_aspect_ratio`, ports declared.
DefaultAspectRatio = physics.function(
    calculate_default_aspect_ratio,
    "aspect",
    reads={
        "stella_config_aspect_ref": stellarator_config,
    },
)


# cottax node: `calculate_stellarator_scaling_factors`, ports declared.
StellaratorScalingFactors = physics.function(
    calculate_stellarator_scaling_factors,
    "rminor",
    "eps",
    V.tfcoil.n_tf_coils,
    V.stellarator.f_st_rmajor,
    V.stellarator.f_st_rminor,
    V.stellarator.f_st_aspect,
    V.stellarator.f_st_n_coils,
    V.stellarator.f_st_b,
    V.stellarator.r_coil_major,
    V.stellarator.r_coil_minor,
    V.stellarator.f_coil_shape,
    reads={
        "f_st_coil_aspect": stellarator,
        "stella_config_coilspermodule": stellarator_config,
        "stella_config_symmetry": stellarator_config,
        "stella_config_rmajor_ref": stellarator_config,
        "stella_config_rminor_ref": stellarator_config,
        "stella_config_aspect_ref": stellarator_config,
        "stella_config_bt_ref": stellarator_config,
        "stella_config_coil_rmajor": stellarator_config,
        "stella_config_coil_rminor": stellarator_config,
        "stella_config_min_plasma_coil_distance": stellarator_config,
    },
)


# cottax node: `calculate_stellarator_plasma_geometry`, ports declared.
StellaratorPlasmaGeometry = physics.function(
    calculate_stellarator_plasma_geometry,
    "vol_plasma",
    "a_plasma_surface",
    "a_plasma_poloidal",
    "a_plasma_surface_outboard",
    reads={
        "f_st_rmajor": stellarator,
        "f_st_rminor": stellarator,
        "stella_config_vol_plasma": stellarator_config,
        "stella_config_plasma_surface": stellarator_config,
    },
)
