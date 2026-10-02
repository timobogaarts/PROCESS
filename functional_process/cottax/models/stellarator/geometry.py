"""Pure-functional port of `Stellarator.st_new_config`/`st_geom`.

Chunk 1C of unit #1.
"""

import jax.numpy as jnp  # noqa: F401
from cottax.cx import V

from functional_process.cottax.paths import (
    physics,
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
)


# cottax node: `calculate_stellarator_plasma_geometry`, ports declared.
StellaratorPlasmaGeometry = physics.function(
    calculate_stellarator_plasma_geometry,
    "vol_plasma",
    "a_plasma_surface",
    "a_plasma_poloidal",
    "a_plasma_surface_outboard",
)
