"""Pure-functional port of `process/models/stellarator/build.py`'s `st_build` (unit #2).
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    first_wall,
    fwbs,
    physics,
    stellarator,
    stellarator_config,
)
from functional_process.models.stellarator.build import (
    calculate_a_fw_total_no_powerflow,
    calculate_a_fw_total_with_powerflow,
    calculate_blktmodel_blanket_thickness,
    calculate_build,
)


class BlktmodelBlanketThickness(ExplicitFunction):
    """cottax node: `calculate_blktmodel_blanket_thickness`, ports declared."""

    dr_blkt_inboard = OutputInto(build)
    dr_blkt_outboard = OutputInto(build)
    dz_shld_upper = OutputInto(build)

    def __call__(
        self,
        blbuith=From(build),
        blbmith=From(build),
        blbpith=From(build),
        blbuoth=From(build),
        blbmoth=From(build),
        blbpoth=From(build),
        dr_shld_inboard=From(build),
        dr_shld_outboard=From(build),
    ):
        return calculate_blktmodel_blanket_thickness(
            blbuith,
            blbmith,
            blbpith,
            blbuoth,
            blbmoth,
            blbpoth,
            dr_shld_inboard,
            dr_shld_outboard,
        )


class Build(ExplicitFunction):
    """cottax node: `calculate_build`, ports declared."""

    dz_blkt_upper = OutputInto(build)
    dr_fw_inboard = OutputInto(build)
    dr_fw_outboard = OutputInto(build)
    dr_bore = OutputInto(build)
    rbld = OutputInto(build)
    required_radial_space = OutputInto(build)
    available_radial_space = OutputInto(build)
    r_shld_inboard_inner = OutputInto(build)
    r_shld_outboard_outer = OutputInto(build)
    dr_tf_outboard = OutputInto(build)
    dr_shld_vv_gap_outboard = OutputInto(build)
    r_tf_outboard_mid = OutputInto(build)
    rspo = OutputInto(build)
    # Invented intermediate, not a real PROCESS field -- see module docstring.
    a_fw_total_unadjusted = OutputInto(first_wall)

    def __call__(
        self,
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
        radius_fw_channel=From(fwbs),
        dr_fw_wall=From(fwbs),
        rmajor=From(physics),
        rminor=From(physics),
        dr_cs=From(build),
        dr_cs_tf_gap=From(build),
        dr_tf_inboard=From(build),
        dr_shld_vv_gap_inboard=From(build),
        dr_vv_inboard=From(build),
        dr_shld_inboard=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        r_coil_minor=From(stellarator),
        f_coil_shape=From(stellarator),
        stella_config_derivative_min_lcfs_coils_dist=From(stellarator_config),
        f_st_rmajor=From(stellarator),
        stella_config_rminor_ref=From(stellarator_config),
        dr_fw_plasma_gap_outboard=From(build),
        dr_shld_outboard=From(build),
        gapomin=From(build),
        dr_vv_outboard=From(build),
        a_plasma_surface=From(physics),
    ):
        return calculate_build(
            dr_blkt_inboard,
            dr_blkt_outboard,
            radius_fw_channel,
            dr_fw_wall,
            rmajor,
            rminor,
            dr_cs,
            dr_cs_tf_gap,
            dr_tf_inboard,
            dr_shld_vv_gap_inboard,
            dr_vv_inboard,
            dr_shld_inboard,
            dr_fw_plasma_gap_inboard,
            r_coil_minor,
            f_coil_shape,
            stella_config_derivative_min_lcfs_coils_dist,
            f_st_rmajor,
            stella_config_rminor_ref,
            dr_fw_plasma_gap_outboard,
            dr_shld_outboard,
            gapomin,
            dr_vv_outboard,
            a_plasma_surface,
        )


class AFwTotalNoPowerflow(ExplicitFunction):
    """cottax node: `calculate_a_fw_total_no_powerflow`. Instantiate iff `ipowerflow == 0`."""

    a_fw_total = OutputInto(first_wall)

    def __call__(
        self,
        a_fw_total_unadjusted=From(first_wall),
        fhole=From(fwbs),
    ):
        return calculate_a_fw_total_no_powerflow(a_fw_total_unadjusted, fhole)


class AFwTotalWithPowerflow(ExplicitFunction):
    """cottax node: `calculate_a_fw_total_with_powerflow`. Instantiate iff `ipowerflow != 0`."""

    a_fw_total = OutputInto(first_wall)

    def __call__(
        self,
        a_fw_total_unadjusted=From(first_wall),
        fhole=From(fwbs),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
    ):
        return calculate_a_fw_total_with_powerflow(
            a_fw_total_unadjusted, fhole, f_ster_div_single, f_a_fw_outboard_hcd
        )
