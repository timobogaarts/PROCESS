"""Harness cases for the ported stellarator radial build (unit #2).

No existing PROCESS unit test covers `st_build` (`tests/unit/models/stellarator/` has
no `test_build.py`), so there is no golden legacy point to lift -- every sample here is
`fuzz`, drawn from physically-plausible ranges rather than a validated operating point.

`st_build` is a module-level function taking `(stellarator, f_output, data)`; `stellarator`
is only dereferenced inside the `if f_output:` branch (for `.outfile`), so `None` stands
in for it here, same reasoning as `test_structure.py`'s use of `None` for
unused injected sub-models.

A full `DataStructure()` has `rminor = 0.0` by default (`physics_variables.py`), which
`st_build` divides by -- every reference adapter below sets a complete, sane baseline
first and overrides only the fields its own port signature covers, so a field the port
doesn't take is still non-degenerate for the parts of `st_build` that run regardless.
"""

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax.stellarator.build import (
    calculate_a_fw_total_no_powerflow,
    calculate_a_fw_total_with_powerflow,
    calculate_blktmodel_blanket_thickness,
    calculate_build,
)
from process.core.model import DataStructure
from process.models.stellarator.build import st_build


def _baseline_data():
    """A `DataStructure` with every field `st_build` reads set to a sane, nonzero value."""
    data = DataStructure()

    data.build.blbuith = 0.365
    data.build.blbmith = 0.17
    data.build.blbpith = 0.30
    data.build.blbuoth = 0.465
    data.build.blbmoth = 0.27
    data.build.blbpoth = 0.35
    data.build.dr_shld_inboard = 0.30
    data.build.dr_shld_outboard = 0.30

    # `blktmodel <= 0`, so `dr_blkt_inboard`/`dr_blkt_outboard` are plain inputs, matching
    # `calculate_build`'s own assumption (see build.py's module docstring).
    data.fwbs.blktmodel = 0
    data.build.dr_blkt_inboard = 0.835
    data.build.dr_blkt_outboard = 1.085

    data.fwbs.radius_fw_channel = 0.006
    data.fwbs.dr_fw_wall = 0.003
    data.physics.rmajor = 22.0
    data.physics.rminor = 1.78
    data.build.dr_cs = 0.0
    data.build.dr_cs_tf_gap = 0.0
    data.build.dr_tf_inboard = 0.6
    data.build.dr_shld_vv_gap_inboard = 0.02
    data.build.dr_vv_inboard = 0.3
    data.build.dr_fw_plasma_gap_inboard = 0.2
    data.stellarator.r_coil_minor = 2.0
    data.stellarator.f_coil_shape = 1.0
    data.stellarator_config.stella_config_derivative_min_lcfs_coils_dist = 0.1
    data.stellarator.f_st_rmajor = 0.99
    data.stellarator_config.stella_config_rminor_ref = 1.0
    data.build.dr_fw_plasma_gap_outboard = 0.2
    data.build.gapomin = 0.02
    data.build.dr_vv_outboard = 0.3
    data.physics.a_plasma_surface = 1500.0

    data.heat_transport.ipowerflow = 0
    data.fwbs.fhole = 0.0
    data.fwbs.f_ster_div_single = 0.0
    data.fwbs.f_a_fw_outboard_hcd = 0.0

    return data


def _reference_blktmodel_blanket_thickness(
    blbuith,
    blbmith,
    blbpith,
    blbuoth,
    blbmoth,
    blbpoth,
    dr_shld_inboard,
    dr_shld_outboard,
):
    """Call PROCESS's `st_build` (`blktmodel = 1` branch) through the port's signature."""
    data = _baseline_data()
    data.fwbs.blktmodel = 1
    data.build.blbuith = blbuith
    data.build.blbmith = blbmith
    data.build.blbpith = blbpith
    data.build.blbuoth = blbuoth
    data.build.blbmoth = blbmoth
    data.build.blbpoth = blbpoth
    data.build.dr_shld_inboard = dr_shld_inboard
    data.build.dr_shld_outboard = dr_shld_outboard

    st_build(None, False, data)
    return (
        data.build.dr_blkt_inboard,
        data.build.dr_blkt_outboard,
        data.build.dz_shld_upper,
    )


def _reference_build(
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
):
    """Call PROCESS's `st_build` (`blktmodel = 0`, `ipowerflow = 0`, `fhole = 0`).

    `ipowerflow = 0` and `fhole = 0` make the final `.first_wall.a_fw_total` PROCESS
    writes numerically identical to `calculate_build`'s unadjusted intermediate, which
    PROCESS itself never stores -- see build.py's module docstring.
    """
    data = _baseline_data()
    data.build.dr_blkt_inboard = dr_blkt_inboard
    data.build.dr_blkt_outboard = dr_blkt_outboard
    data.fwbs.radius_fw_channel = radius_fw_channel
    data.fwbs.dr_fw_wall = dr_fw_wall
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.build.dr_cs = dr_cs
    data.build.dr_cs_tf_gap = dr_cs_tf_gap
    data.build.dr_tf_inboard = dr_tf_inboard
    data.build.dr_shld_vv_gap_inboard = dr_shld_vv_gap_inboard
    data.build.dr_vv_inboard = dr_vv_inboard
    data.build.dr_shld_inboard = dr_shld_inboard
    data.build.dr_fw_plasma_gap_inboard = dr_fw_plasma_gap_inboard
    data.stellarator.r_coil_minor = r_coil_minor
    data.stellarator.f_coil_shape = f_coil_shape
    data.stellarator_config.stella_config_derivative_min_lcfs_coils_dist = (
        stella_config_derivative_min_lcfs_coils_dist
    )
    data.stellarator.f_st_rmajor = f_st_rmajor
    data.stellarator_config.stella_config_rminor_ref = stella_config_rminor_ref
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.build.dr_shld_outboard = dr_shld_outboard
    data.build.gapomin = gapomin
    data.build.dr_vv_outboard = dr_vv_outboard
    data.physics.a_plasma_surface = a_plasma_surface

    st_build(None, False, data)
    return (
        data.build.dz_blkt_upper,
        data.build.dr_fw_inboard,
        data.build.dr_fw_outboard,
        data.build.dr_bore,
        data.build.rbld,
        data.build.required_radial_space,
        data.build.available_radial_space,
        data.build.r_shld_inboard_inner,
        data.build.r_shld_outboard_outer,
        data.build.dr_tf_outboard,
        data.build.dr_shld_vv_gap_outboard,
        data.build.r_tf_outboard_mid,
        data.build.rspo,
        data.first_wall.a_fw_total,
    )


def _reference_a_fw_total_no_powerflow(a_fw_total_unadjusted, fhole):
    """`(1 - fhole) * a_fw_total_unadjusted` -- PROCESS's `ipowerflow == 0` branch,
    reproduced directly (there is no PROCESS entry point taking the unadjusted value
    as an argument -- see build.py's module docstring for why it is invented)."""
    return (1.0 - fhole) * a_fw_total_unadjusted


def _reference_a_fw_total_with_powerflow(
    a_fw_total_unadjusted, fhole, f_ster_div_single, f_a_fw_outboard_hcd
):
    """PROCESS's `ipowerflow != 0` branch, reproduced directly (same reasoning)."""
    return (
        1.0 - fhole - f_ster_div_single - f_a_fw_outboard_hcd
    ) * a_fw_total_unadjusted


class TestBlktmodelBlanketThickness(Tier1Contract):
    """`st_build` (`blktmodel > 0` branch) -> `calculate_blktmodel_blanket_thickness`."""

    audit_record = "models/stellarator/build.md"
    reference = _reference_blktmodel_blanket_thickness
    ported = calculate_blktmodel_blanket_thickness

    fuzz_bounds = {
        "blbuith": (0.05, 1.0),
        "blbmith": (0.05, 1.0),
        "blbpith": (0.05, 1.0),
        "blbuoth": (0.05, 1.0),
        "blbmoth": (0.05, 1.0),
        "blbpoth": (0.05, 1.0),
        "dr_shld_inboard": (0.05, 1.0),
        "dr_shld_outboard": (0.05, 1.0),
    }


class TestBuild(Tier1Contract):
    """`st_build` (unconditional body) -> `calculate_build`."""

    audit_record = "models/stellarator/build.md"
    reference = _reference_build
    ported = calculate_build

    fuzz_bounds = {
        "dr_blkt_inboard": (0.2, 2.0),
        "dr_blkt_outboard": (0.2, 2.0),
        "radius_fw_channel": (0.001, 0.02),
        "dr_fw_wall": (0.001, 0.01),
        "rmajor": (5.0, 30.0),
        "rminor": (0.5, 3.0),
        "dr_cs": (0.0, 1.0),
        "dr_cs_tf_gap": (0.0, 0.5),
        "dr_tf_inboard": (0.1, 1.5),
        "dr_shld_vv_gap_inboard": (0.01, 0.2),
        "dr_vv_inboard": (0.1, 1.0),
        "dr_shld_inboard": (0.05, 1.0),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "r_coil_minor": (0.5, 5.0),
        "f_coil_shape": (0.5, 2.0),
        "stella_config_derivative_min_lcfs_coils_dist": (0.01, 1.0),
        "f_st_rmajor": (0.5, 1.5),
        "stella_config_rminor_ref": (0.5, 2.0),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "dr_shld_outboard": (0.05, 1.0),
        "gapomin": (0.01, 0.2),
        "dr_vv_outboard": (0.1, 1.0),
        "a_plasma_surface": (100.0, 5000.0),
    }


class TestAFwTotalNoPowerflow(Tier1Contract):
    """`st_build` (`ipowerflow == 0` branch) -> `calculate_a_fw_total_no_powerflow`."""

    audit_record = "models/stellarator/build.md"
    reference = _reference_a_fw_total_no_powerflow
    ported = calculate_a_fw_total_no_powerflow

    fuzz_bounds = {
        "a_fw_total_unadjusted": (10.0, 5000.0),
        "fhole": (0.0, 0.5),
    }


class TestAFwTotalWithPowerflow(Tier1Contract):
    """`st_build` (`ipowerflow != 0` branch) -> `calculate_a_fw_total_with_powerflow`."""

    audit_record = "models/stellarator/build.md"
    reference = _reference_a_fw_total_with_powerflow
    ported = calculate_a_fw_total_with_powerflow

    fuzz_bounds = {
        "a_fw_total_unadjusted": (10.0, 5000.0),
        "fhole": (0.0, 0.3),
        "f_ster_div_single": (0.0, 0.2),
        "f_a_fw_outboard_hcd": (0.0, 0.2),
    }
