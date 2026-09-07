"""Pure-functional port of `load_stellarator_config` (registry unit #8)."""

import json  # noqa: F401

import equinox as eqx
import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    OutputInto,
)

from functional_process.cottax.paths import (
    stellarator_config,
)
from functional_process.models.stellarator.preset_config import (
    STELLA_CONFIG_DEFAULT,  # noqa: F401
    STELLA_CONFIG_SCALAR_FIELDS,  # noqa: F401
    STELLARATOR_MACHINE_PRESETS,  # noqa: F401
    dropped_config_keys,  # noqa: F401
    machine_config_for_istell,  # noqa: F401
    read_stellarator_config_file,  # noqa: F401
    select_stellarator_config_scalars,
)
from functional_process.vocabulary import (
    HELIAS3,  # noqa: F401
    HELIAS4,  # noqa: F401
    HELIAS5B,  # noqa: F401
    W7X30,  # noqa: F401
    W7X50,  # noqa: F401
)


class StellaratorMachineConfig(ExplicitFunction):
    """cottax node: the selected machine config's 34 scalars, owned by the graph."""

    machine_config: tuple = eqx.field(static=True)

    stella_config_symmetry = OutputInto(stellarator_config)
    stella_config_coilspermodule = OutputInto(stellarator_config)
    stella_config_rmajor_ref = OutputInto(stellarator_config)
    stella_config_rminor_ref = OutputInto(stellarator_config)
    stella_config_coil_rmajor = OutputInto(stellarator_config)
    stella_config_coil_rminor = OutputInto(stellarator_config)
    stella_config_aspect_ref = OutputInto(stellarator_config)
    stella_config_bt_ref = OutputInto(stellarator_config)
    stella_config_wp_area = OutputInto(stellarator_config)
    stella_config_wp_bmax = OutputInto(stellarator_config)
    stella_config_i0 = OutputInto(stellarator_config)
    stella_config_a1 = OutputInto(stellarator_config)
    stella_config_a2 = OutputInto(stellarator_config)
    stella_config_dmin = OutputInto(stellarator_config)
    stella_config_inductance = OutputInto(stellarator_config)
    stella_config_coilsurface = OutputInto(stellarator_config)
    stella_config_coillength = OutputInto(stellarator_config)
    stella_config_max_portsize_width = OutputInto(stellarator_config)
    stella_config_maximal_coil_height = OutputInto(stellarator_config)
    stella_config_min_plasma_coil_distance = OutputInto(stellarator_config)
    stella_config_derivative_min_lcfs_coils_dist = OutputInto(stellarator_config)
    stella_config_vol_plasma = OutputInto(stellarator_config)
    stella_config_plasma_surface = OutputInto(stellarator_config)
    stella_config_wp_ratio = OutputInto(stellarator_config)
    stella_config_max_force_density = OutputInto(stellarator_config)
    stella_config_max_force_density_mnm = OutputInto(stellarator_config)
    stella_config_min_bend_radius = OutputInto(stellarator_config)
    stella_config_epseff = OutputInto(stellarator_config)
    stella_config_max_lateral_force_density = OutputInto(stellarator_config)
    stella_config_max_radial_force_density = OutputInto(stellarator_config)
    stella_config_centering_force_max_mn = OutputInto(stellarator_config)
    stella_config_centering_force_min_mn = OutputInto(stellarator_config)
    stella_config_centering_force_avg_mn = OutputInto(stellarator_config)
    stella_config_neutron_peakfactor = OutputInto(stellarator_config)

    def __call__(self):
        return select_stellarator_config_scalars(dict(self.machine_config))
