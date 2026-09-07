"""Pure-functional port of `Stellarator.st_new_config`/`st_geom` (chunk 1C of unit #1).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    physics,
    stellarator,
    stellarator_config,
    tfcoil,
)
from functional_process.models.stellarator.geometry import (
    calculate_default_aspect_ratio,
    calculate_stellarator_plasma_geometry,
    calculate_stellarator_scaling_factors,
)


class DefaultAspectRatio(ExplicitFunction):
    """cottax node: `calculate_default_aspect_ratio`, ports declared."""

    aspect = OutputInto(physics)

    def __call__(
        self,
        stella_config_aspect_ref=From(stellarator_config),
    ):
        return calculate_default_aspect_ratio(stella_config_aspect_ref)


class StellaratorScalingFactors(ExplicitFunction):
    """cottax node: `calculate_stellarator_scaling_factors`, ports declared."""

    rminor = OutputInto(physics)
    eps = OutputInto(physics)
    n_tf_coils = OutputInto(tfcoil)
    f_st_rmajor = OutputInto(stellarator)
    f_st_rminor = OutputInto(stellarator)
    f_st_aspect = OutputInto(stellarator)
    f_st_n_coils = OutputInto(stellarator)
    f_st_b = OutputInto(stellarator)
    r_coil_major = OutputInto(stellarator)
    r_coil_minor = OutputInto(stellarator)
    f_coil_shape = OutputInto(stellarator)

    def __call__(
        self,
        rmajor=From(physics),
        aspect=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        f_st_coil_aspect=From(stellarator),
        stella_config_coilspermodule=From(stellarator_config),
        stella_config_symmetry=From(stellarator_config),
        stella_config_rmajor_ref=From(stellarator_config),
        stella_config_rminor_ref=From(stellarator_config),
        stella_config_aspect_ref=From(stellarator_config),
        stella_config_bt_ref=From(stellarator_config),
        stella_config_coil_rmajor=From(stellarator_config),
        stella_config_coil_rminor=From(stellarator_config),
        stella_config_min_plasma_coil_distance=From(stellarator_config),
    ):
        return calculate_stellarator_scaling_factors(
            rmajor,
            aspect,
            b_plasma_toroidal_on_axis,
            f_st_coil_aspect,
            stella_config_coilspermodule,
            stella_config_symmetry,
            stella_config_rmajor_ref,
            stella_config_rminor_ref,
            stella_config_aspect_ref,
            stella_config_bt_ref,
            stella_config_coil_rmajor,
            stella_config_coil_rminor,
            stella_config_min_plasma_coil_distance,
        )


class StellaratorPlasmaGeometry(ExplicitFunction):
    """cottax node: `calculate_stellarator_plasma_geometry`, ports declared."""

    vol_plasma = OutputInto(physics)
    a_plasma_surface = OutputInto(physics)
    a_plasma_poloidal = OutputInto(physics)
    a_plasma_surface_outboard = OutputInto(physics)

    def __call__(
        self,
        f_st_rmajor=From(stellarator),
        f_st_rminor=From(stellarator),
        rminor=From(physics),
        stella_config_vol_plasma=From(stellarator_config),
        stella_config_plasma_surface=From(stellarator_config),
    ):
        return calculate_stellarator_plasma_geometry(
            f_st_rmajor,
            f_st_rminor,
            rminor,
            stella_config_vol_plasma,
            stella_config_plasma_surface,
        )
