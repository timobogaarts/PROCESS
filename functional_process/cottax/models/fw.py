"""Pure-functional port of `process/models/fw.py` (`FirstWall`, `.tokamak.first_wall`)
-- the minimal closure for `.first_wall.a_fw_total`, `.physics.p_fw_alpha_mw` and
`.physics.pflux_fw_neutron_mw`, `tokamak_boundary.md`'s three reads of this slot.
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.paths import (
    build,
    constraints,
    divertor,
    first_wall,
    fwbs,
    physics,
)
from functional_process.cottax.wraps import WrapsFunction
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


class FirstWall(ExplicitFunction):
    """The family that occupies `.tokamak.first_wall`: one occupant per cell of the
    shape x divertor-count grid, all at `.physics.i_pflux_fw_neutron == 1`.
    """


class FirstWallSingleNull(FirstWall, WrapsFunction):
    """cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 1`, elliptical --
    the combination live on `large_tokamak_eval.IN.DAT` (see module docstring).
    """

    fn = calculate_first_wall_outputs

    z_plasma_xpoint_lower = From(build)
    dz_xpoint_divertor = From(build)
    dz_divertor = From(divertor)
    dz_blkt_upper = From(build)
    z_plasma_xpoint_upper = From(build)
    dz_fw_plasma_gap = From(build)
    dr_fw_inboard = From(build)
    dr_fw_outboard = From(build)
    rmajor = From(physics)
    rminor = From(physics)
    triang = From(physics)
    dr_fw_plasma_gap_inboard = From(build)
    dr_fw_plasma_gap_outboard = From(build)
    f_ster_div_single = From(fwbs)
    f_a_fw_outboard_hcd = From(fwbs)
    p_alpha_total_mw = From(physics)
    f_p_alpha_plasma_deposited = From(physics)
    ffwal = From(physics)
    pflux_plasma_surface_neutron_avg_mw = From(physics)

    a_fw_inboard = OutputInto(first_wall)
    a_fw_outboard = OutputInto(first_wall)
    a_fw_total = OutputInto(first_wall)
    p_fw_alpha_mw = OutputInto(physics)
    pflux_fw_neutron_mw = OutputInto(physics)


class FirstWallDoubleNull(FirstWall, WrapsFunction):
    """cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 2`, elliptical."""

    fn = calculate_first_wall_outputs_double_null

    z_plasma_xpoint_lower = From(build)
    dz_xpoint_divertor = From(build)
    dz_divertor = From(divertor)
    dz_blkt_upper = From(build)
    dr_fw_inboard = From(build)
    dr_fw_outboard = From(build)
    rmajor = From(physics)
    rminor = From(physics)
    triang = From(physics)
    dr_fw_plasma_gap_inboard = From(build)
    dr_fw_plasma_gap_outboard = From(build)
    f_ster_div_single = From(fwbs)
    f_a_fw_outboard_hcd = From(fwbs)
    p_alpha_total_mw = From(physics)
    f_p_alpha_plasma_deposited = From(physics)
    ffwal = From(physics)
    pflux_plasma_surface_neutron_avg_mw = From(physics)

    a_fw_inboard = OutputInto(first_wall)
    a_fw_outboard = OutputInto(first_wall)
    a_fw_total = OutputInto(first_wall)
    p_fw_alpha_mw = OutputInto(physics)
    pflux_fw_neutron_mw = OutputInto(physics)


class FirstWallDShapedDoubleNull(FirstWall, WrapsFunction):
    """cottax node: `.tokamak.first_wall` at `.divertor.n_divertors == 2` **and** the
    D-shaped shape arm -- the configuration live on `spherical_tokamak_eval.IN.DAT` and
    `st_regression.IN.DAT` (`i_single_null = 0`; `itart = 1` and `i_fw_blkt_vv_shape =
    1`, either of which alone selects the D-shaped arm).
    """

    fn = calculate_first_wall_outputs_dshaped_double_null

    z_plasma_xpoint_lower = From(build)
    dz_xpoint_divertor = From(build)
    dz_divertor = From(divertor)
    dz_blkt_upper = From(build)
    dr_fw_inboard = From(build)
    dr_fw_outboard = From(build)
    rmajor = From(physics)
    rminor = From(physics)
    dr_fw_plasma_gap_inboard = From(build)
    dr_fw_plasma_gap_outboard = From(build)
    f_ster_div_single = From(fwbs)
    f_a_fw_outboard_hcd = From(fwbs)
    p_alpha_total_mw = From(physics)
    f_p_alpha_plasma_deposited = From(physics)
    ffwal = From(physics)
    pflux_plasma_surface_neutron_avg_mw = From(physics)

    a_fw_inboard = OutputInto(first_wall)
    a_fw_outboard = OutputInto(first_wall)
    a_fw_total = OutputInto(first_wall)
    p_fw_alpha_mw = OutputInto(physics)
    pflux_fw_neutron_mw = OutputInto(physics)


class FirstWallGeometry(WrapsFunction):
    """cottax node: `.tokamak.first_wall_geometry`."""

    fn = set_fw_geometry

    radius_fw_channel = From(fwbs)
    dr_fw_wall = From(fwbs)

    dr_fw_inboard = OutputInto(build)
    dr_fw_outboard = OutputInto(build)


class RadiatedWallLoad(WrapsFunction):
    """cottax node: `calculate_radiated_wall_load_scaled_plasma_surface`, ports
    declared.
    """

    fn = calculate_radiated_wall_load_scaled_plasma_surface

    ffwal = From(physics)
    p_plasma_rad_mw = From(physics)
    a_plasma_surface = From(physics)
    f_fw_rad_max = From(constraints)

    pflux_fw_rad_mw = OutputInto(physics)
    pflux_fw_rad_max_mw = OutputInto(constraints)
