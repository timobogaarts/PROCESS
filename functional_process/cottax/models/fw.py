"""Pure-functional port of `process/models/fw.py` (`FirstWall`, `.tokamak.first_wall`)
-- the minimal closure for `.first_wall.a_fw_total`, `.physics.p_fw_alpha_mw` and
`.physics.pflux_fw_neutron_mw`, `tokamak_boundary.md`'s three reads of this slot.
"""

from cottax.relational_pytree import V

from functional_process.cottax.paths import (
    build,
    first_wall,
    physics,
)
from functional_process.models.engineering.ivc_functions import (
    dshellarea,  # noqa: F401
    eshellarea,  # noqa: F401
)
from functional_process.models.fw import (
    apply_first_wall_coverage_factors,  # noqa: F401
    apply_first_wall_coverage_factors_double_null,  # noqa: F401
    calculate_dshaped_first_wall_areas,  # noqa: F401
    calculate_elliptical_first_wall_areas,  # noqa: F401
    calculate_first_wall_half_height,  # noqa: F401
    calculate_first_wall_half_height_double_null,  # noqa: F401
    calculate_first_wall_outputs,
    calculate_first_wall_outputs_double_null,
    calculate_first_wall_outputs_dshaped_double_null,
    calculate_p_fw_alpha_mw,  # noqa: F401
    calculate_pflux_fw_neutron_mw_ffwal,  # noqa: F401
    calculate_radiated_wall_load_scaled_plasma_surface,
    set_fw_geometry,
)

# The FirstWall family:
# The family that occupies `.tokamak.first_wall`: one occupant per cell of the
# shape x divertor-count grid, all at `.physics.i_pflux_fw_neutron == 1`.


# cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 1`, elliptical --
# the combination live on `large_tokamak_eval.IN.DAT` (see module docstring).
FirstWallSingleNull = first_wall.function(
    calculate_first_wall_outputs,
    "a_fw_inboard",
    "a_fw_outboard",
    "a_fw_total",
    V.physics.p_fw_alpha_mw,
    V.physics.pflux_fw_neutron_mw,
)


# cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 2`, elliptical.
FirstWallDoubleNull = first_wall.function(
    calculate_first_wall_outputs_double_null,
    "a_fw_inboard",
    "a_fw_outboard",
    "a_fw_total",
    V.physics.p_fw_alpha_mw,
    V.physics.pflux_fw_neutron_mw,
)


# cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 2` **and** the
# D-shaped shape arm -- the configuration live on `spherical_tokamak_eval.IN.DAT` and
# `st_regression.IN.DAT` (`i_single_null = 0`; `itart = 1` and `i_fw_blkt_vv_shape =
# 1`, either of which alone selects the D-shaped arm).
FirstWallDShapedDoubleNull = first_wall.function(
    calculate_first_wall_outputs_dshaped_double_null,
    "a_fw_inboard",
    "a_fw_outboard",
    "a_fw_total",
    V.physics.p_fw_alpha_mw,
    V.physics.pflux_fw_neutron_mw,
)


# cottax node: `.tokamak.first_wall_geometry`.
FirstWallGeometry = build.function(
    set_fw_geometry,
    "dr_fw_inboard",
    "dr_fw_outboard",
)


# cottax node: `calculate_radiated_wall_load_scaled_plasma_surface`, ports
# declared.
RadiatedWallLoad = physics.function(
    calculate_radiated_wall_load_scaled_plasma_surface,
    "pflux_fw_rad_mw",
    V.constraints.pflux_fw_rad_max_mw,
)
