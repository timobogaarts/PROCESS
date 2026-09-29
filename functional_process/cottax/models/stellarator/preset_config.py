"""Pure-functional port of `load_stellarator_config` (registry unit #8)."""

import json  # noqa: F401

import jax.numpy as jnp  # noqa: F401

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


def _machine_config_scalars(machine_config):
    return select_stellarator_config_scalars(dict(machine_config))


# cottax node: the selected machine config's 34 scalars, owned by the graph.
# `.stellarator_config.machine_config` is the selected machine's `(key, value)` pairs
# (`machine_config_for_istell`), read as a setting.
StellaratorMachineConfig = stellarator_config.function(
    _machine_config_scalars,
    "stella_config_symmetry",
    "stella_config_coilspermodule",
    "stella_config_rmajor_ref",
    "stella_config_rminor_ref",
    "stella_config_coil_rmajor",
    "stella_config_coil_rminor",
    "stella_config_aspect_ref",
    "stella_config_bt_ref",
    "stella_config_wp_area",
    "stella_config_wp_bmax",
    "stella_config_i0",
    "stella_config_a1",
    "stella_config_a2",
    "stella_config_dmin",
    "stella_config_inductance",
    "stella_config_coilsurface",
    "stella_config_coillength",
    "stella_config_max_portsize_width",
    "stella_config_maximal_coil_height",
    "stella_config_min_plasma_coil_distance",
    "stella_config_derivative_min_lcfs_coils_dist",
    "stella_config_vol_plasma",
    "stella_config_plasma_surface",
    "stella_config_wp_ratio",
    "stella_config_max_force_density",
    "stella_config_max_force_density_mnm",
    "stella_config_min_bend_radius",
    "stella_config_epseff",
    "stella_config_max_lateral_force_density",
    "stella_config_max_radial_force_density",
    "stella_config_centering_force_max_mn",
    "stella_config_centering_force_min_mn",
    "stella_config_centering_force_avg_mn",
    "stella_config_neutron_peakfactor",
)
