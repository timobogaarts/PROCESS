"""Harness cases for the ported `coils/calculate.py` functions (registry unit #9).

No `legacy_sample` for the tier-1 functions below: unlike `density_limits.py`/
`stellarator_D`/`stellarator_F`, none of the ten has a matching test in
`tests/unit/models/stellarator/` (checked by grep -- there is no dedicated
stellarator-coils test file in PROCESS's own suite), so there is no literal point to
lift. Coverage is `fuzz_bounds`-only; agreement is still checked live against the real
PROCESS function at every sampled point, so this is not weaker evidence, just a
different provenance.

`winding_pack_total_size` (tier-2, `Tier2Contract`) and `st_coil` (tier-3, plain
end-to-end value comparison -- see its own test function's docstring for why it isn't a
`Contract` subclass) are added below the tier-1 classes. Both also have no PROCESS unit
test to lift a literal point from, so their base points are built by hand from
`load_stellarator_config`'s real HELIAS5B preset (`istell=1`) plus realistic material
constants -- verified while writing this port to converge well away from
`winding_pack_total_size`'s internal turn-size clamp (see both test docstrings).
"""

import inspect
from types import MappingProxyType

import equinox as eqx
import jax
import numpy as np
import optimistix as optx
import pytest
from cottax.evaluate import AbstractDriver, schedule_for
from cottax.interfaces.pytree_namespace_module import path_of, to_graph
from cottax.problem import RootFind
from cottax.spec import VarPath

from functional_process._harness import Sample, Tier1Contract, Tier2Contract
from functional_process.models.stellarator.coils.calculate import (
    WindingPackIntersectInputs,
    WindingPackTotalSizePost,
    calculate_casing,
    calculate_coil_coil_toroidal_gap,
    calculate_coil_cross_sectional_area,
    calculate_coil_half_widths,
    calculate_coil_radial_thickness,
    calculate_coil_toroidal_thickness,
    calculate_coils_summary_variables,
    calculate_current,
    calculate_horizontal_ports,
    calculate_inductance,
    calculate_plasma_facing_coil_area,
    calculate_stored_magnetic_energy,
    calculate_vertical_ports,
    calculate_winding_pack_geometry,
    calculate_z_tf_inside_half,
    st_coil,
    winding_pack_curves,
    winding_pack_total_size,
)
from functional_process.models.stellarator.coils.coils import (
    Intersect,
    intersect_residual,
)
from process.core.model import DataStructure
from process.models.stellarator.coils import calculate as process_calculate
from process.models.stellarator.preset_config import load_stellarator_config


def _reference_coil_toroidal_thickness(
    dx_tf_wp_primary_toroidal, dx_tf_side_case_min, dx_tf_wp_insulation
):
    data = DataStructure()
    data.tfcoil.dx_tf_wp_primary_toroidal = dx_tf_wp_primary_toroidal
    data.tfcoil.dx_tf_side_case_min = dx_tf_side_case_min
    data.tfcoil.dx_tf_wp_insulation = dx_tf_wp_insulation
    process_calculate.calculate_coil_toroidal_thickness(data)
    return data.tfcoil.dx_tf_inboard_out_toroidal


class TestCoilToroidalThickness(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_coil_toroidal_thickness
    ported = calculate_coil_toroidal_thickness

    fuzz_bounds = {
        "dx_tf_wp_primary_toroidal": (0.05, 2.0),
        "dx_tf_side_case_min": (0.01, 0.3),
        "dx_tf_wp_insulation": (0.001, 0.05),
    }


def _reference_coil_radial_thickness(
    dr_tf_nose_case, dr_tf_wp_with_insulation, dr_tf_plasma_case, dx_tf_wp_insulation
):
    data = DataStructure()
    data.tfcoil.dr_tf_nose_case = dr_tf_nose_case
    data.tfcoil.dr_tf_wp_with_insulation = dr_tf_wp_with_insulation
    data.tfcoil.dr_tf_plasma_case = dr_tf_plasma_case
    data.tfcoil.dx_tf_wp_insulation = dx_tf_wp_insulation
    process_calculate.calculate_coil_radial_thickness(data)
    return data.build.dr_tf_inboard


class TestCoilRadialThickness(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_coil_radial_thickness
    ported = calculate_coil_radial_thickness

    fuzz_bounds = {
        "dr_tf_nose_case": (0.01, 0.5),
        "dr_tf_wp_with_insulation": (0.1, 2.0),
        "dr_tf_plasma_case": (0.01, 0.3),
        "dx_tf_wp_insulation": (0.001, 0.05),
    }


def _reference_coil_cross_sectional_area(
    a_tf_wp_with_insulation, dr_tf_inboard, dx_tf_inboard_out_toroidal
):
    data = DataStructure()
    data.build.dr_tf_inboard = dr_tf_inboard
    data.tfcoil.dx_tf_inboard_out_toroidal = dx_tf_inboard_out_toroidal
    process_calculate.calculate_coil_cross_sectional_area(a_tf_wp_with_insulation, data)
    return data.tfcoil.a_tf_leg_outboard, data.tfcoil.a_tf_coil_inboard_case


class TestCoilCrossSectionalArea(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_coil_cross_sectional_area
    ported = calculate_coil_cross_sectional_area

    fuzz_bounds = {
        "a_tf_wp_with_insulation": (0.01, 2.0),
        "dr_tf_inboard": (0.1, 3.0),
        "dx_tf_inboard_out_toroidal": (0.05, 2.0),
    }


def _reference_coil_half_widths(dx_tf_inboard_out_toroidal):
    data = DataStructure()
    data.tfcoil.dx_tf_inboard_out_toroidal = dx_tf_inboard_out_toroidal
    process_calculate.calculate_coil_half_widths(data)
    return data.tfcoil.tfocrn, data.tfcoil.tficrn


class TestCoilHalfWidths(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_coil_half_widths
    ported = calculate_coil_half_widths

    fuzz_bounds = {"dx_tf_inboard_out_toroidal": (0.05, 2.0)}


def _reference_plasma_facing_coil_area(
    n_tf_coils, dx_tf_inboard_out_toroidal, len_tf_coil
):
    data = DataStructure()
    data.tfcoil.n_tf_coils = n_tf_coils
    data.tfcoil.dx_tf_inboard_out_toroidal = dx_tf_inboard_out_toroidal
    data.tfcoil.len_tf_coil = len_tf_coil
    process_calculate.calculate_plasma_facing_coil_area(data)
    return data.tfcoil.tfsai, data.tfcoil.tfsao


class TestPlasmaFacingCoilArea(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_plasma_facing_coil_area
    ported = calculate_plasma_facing_coil_area

    fuzz_bounds = {
        "n_tf_coils": (1.0, 100.0),
        "dx_tf_inboard_out_toroidal": (0.05, 2.0),
        "len_tf_coil": (10.0, 5000.0),
    }


def _reference_coil_coil_toroidal_gap(
    stella_config_dmin,
    r_coil_major,
    r_coil_minor,
    stella_config_coil_rmajor,
    stella_config_coil_rminor,
    dx_tf_inboard_out_toroidal,
):
    data = DataStructure()
    data.stellarator_config.stella_config_dmin = stella_config_dmin
    data.stellarator_config.stella_config_coil_rmajor = stella_config_coil_rmajor
    data.stellarator_config.stella_config_coil_rminor = stella_config_coil_rminor
    data.tfcoil.dx_tf_inboard_out_toroidal = dx_tf_inboard_out_toroidal
    return process_calculate.calculate_coil_coil_toroidal_gap(
        r_coil_major, r_coil_minor, data
    )


class TestCoilCoilToroidalGap(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_coil_coil_toroidal_gap
    ported = calculate_coil_coil_toroidal_gap

    fuzz_bounds = {
        "stella_config_dmin": (0.05, 2.0),
        "r_coil_major": (5.0, 30.0),
        "r_coil_minor": (0.5, 3.0),
        # rmajor > rminor kept comfortably apart from the plasma-side pair above so the
        # denominator (rmajor - rminor) doesn't approach zero under fuzzing.
        "stella_config_coil_rmajor": (10.0, 40.0),
        "stella_config_coil_rminor": (0.1, 1.0),
        "dx_tf_inboard_out_toroidal": (0.05, 2.0),
    }


def _reference_z_tf_inside_half(
    stella_config_maximal_coil_height, r_coil_minor, stella_config_coil_rminor
):
    """`0.5 * stella_config_maximal_coil_height * (r_coil_minor /
    stella_config_coil_rminor)` -- `st_coil`'s own inline formula
    (`process/models/stellarator/coils/calculate.py:80`), reproduced directly rather
    than invoked through the full `st_coil` orchestrator: there is no standalone real
    PROCESS entry point for this one line (same reasoning `test_build.py`'s
    `_reference_a_fw_total_no_powerflow` gives for its own inline-formula case). The
    full `st_coil` orchestrator's own end-to-end test below (`checks["z_tf_inside_half"]`)
    already exercises this same formula against real PROCESS at realistic operating
    points -- this class adds fuzz coverage and a direct node-level check, not a first
    check of correctness.
    """
    return (
        0.5 * stella_config_maximal_coil_height * (r_coil_minor / stella_config_coil_rminor)
    )


class TestZTfInsideHalf(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_z_tf_inside_half
    ported = calculate_z_tf_inside_half

    fuzz_bounds = {
        "stella_config_maximal_coil_height": (1.0, 20.0),
        "r_coil_minor": (0.5, 3.0),
        "stella_config_coil_rminor": (0.1, 1.0),
    }


def test_z_tf_inside_half_node_assembles_and_owns_the_right_varpath():
    """`ZTfInsideHalf` -- the actual point of extracting this formula into its own
    node: it must assemble via `to_graph` and own `.build.z_tf_inside_half`, not
    `Build` (which used to, until the block-by-block MDA-vs-PROCESS comparison
    harness found `.build.z_tf_inside_half` has two independent real PROCESS
    producers and `Build`'s formula was the wrong one to keep -- see `build.py`'s
    `calculate_build`/`Build` docstrings and this node's own).
    """
    from functional_process.models.stellarator.coils.calculate import ZTfInsideHalf

    node = ZTfInsideHalf()
    graph = to_graph(node)
    assert graph.definitions

    owned = {out.var for out in node.outputs}
    z_tf_inside_half_path = path_of(lambda s: s.build.z_tf_inside_half, VarPath)
    assert z_tf_inside_half_path in owned


def _reference_coils_summary_variables(
    n_tf_coils, a_tf_leg_outboard, coilcurrent, r_coil_major, r_coil_minor, awp_rad
):
    data = DataStructure()
    data.tfcoil.n_tf_coils = n_tf_coils
    data.tfcoil.a_tf_leg_outboard = a_tf_leg_outboard
    process_calculate.calculate_coils_summary_variables(
        coilcurrent, r_coil_major, r_coil_minor, awp_rad, data
    )
    return (
        data.tfcoil.a_tf_inboard_total,
        data.tfcoil.c_tf_total,
        data.tfcoil.j_tf_coil_full_area,
        data.tfcoil.r_b_tf_inboard_peak_symmetric,
    )


class TestCoilsSummaryVariables(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_coils_summary_variables
    ported = calculate_coils_summary_variables

    fuzz_bounds = {
        "n_tf_coils": (1.0, 100.0),
        "a_tf_leg_outboard": (0.01, 5.0),
        "coilcurrent": (0.1, 50.0),
        "r_coil_major": (5.0, 30.0),
        "r_coil_minor": (0.5, 3.0),
        "awp_rad": (0.01, 1.0),
    }


def _reference_inductance(
    stella_config_inductance,
    f_st_rmajor,
    r_coil_minor,
    stella_config_coil_rminor,
    f_st_n_coils,
):
    data = DataStructure()
    data.stellarator_config.stella_config_inductance = stella_config_inductance
    data.stellarator.f_st_rmajor = f_st_rmajor
    data.stellarator_config.stella_config_coil_rminor = stella_config_coil_rminor
    data.stellarator.f_st_n_coils = f_st_n_coils
    return process_calculate.calculate_inductance(r_coil_minor, data)


class TestInductance(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_inductance
    ported = calculate_inductance

    fuzz_bounds = {
        "stella_config_inductance": (0.1, 10.0),
        "f_st_rmajor": (0.5, 1.5),
        "r_coil_minor": (0.5, 3.0),
        "stella_config_coil_rminor": (0.1, 1.0),
        "f_st_n_coils": (1.0, 100.0),
    }


def _reference_stored_magnetic_energy(
    stella_config_inductance,
    f_st_rmajor,
    r_coil_minor,
    stella_config_coil_rminor,
    f_st_n_coils,
    c_tf_total,
    n_tf_coils,
):
    data = DataStructure()
    data.stellarator_config.stella_config_inductance = stella_config_inductance
    data.stellarator.f_st_rmajor = f_st_rmajor
    data.stellarator_config.stella_config_coil_rminor = stella_config_coil_rminor
    data.stellarator.f_st_n_coils = f_st_n_coils
    data.tfcoil.c_tf_total = c_tf_total
    data.tfcoil.n_tf_coils = n_tf_coils
    process_calculate.calculate_stored_magnetic_energy(r_coil_minor, data)
    return data.tfcoil.e_tf_magnetic_stored_total_gj


class TestStoredMagneticEnergy(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_stored_magnetic_energy
    ported = calculate_stored_magnetic_energy

    fuzz_bounds = {
        "stella_config_inductance": (0.1, 10.0),
        "f_st_rmajor": (0.5, 1.5),
        "r_coil_minor": (0.5, 3.0),
        "stella_config_coil_rminor": (0.1, 1.0),
        "f_st_n_coils": (1.0, 100.0),
        "c_tf_total": (1.0e6, 1.0e9),
        "n_tf_coils": (1.0, 100.0),
    }


def _reference_winding_pack_geometry(
    dx_tf_turn_general, dx_tf_turn_steel, dx_tf_turn_insulation
):
    data = DataStructure()
    data.tfcoil.dx_tf_turn_general = dx_tf_turn_general
    data.tfcoil.dx_tf_turn_steel = dx_tf_turn_steel
    data.tfcoil.dx_tf_turn_insulation = dx_tf_turn_insulation
    process_calculate.calculate_winding_pack_geometry(data)
    return data.tfcoil.a_tf_turn_cable_space_no_void, data.tfcoil.a_tf_turn_steel


class TestWindingPackGeometry(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_winding_pack_geometry
    ported = calculate_winding_pack_geometry

    # Kept comfortably in the positive-cable-space regime (the source logs, but does
    # not raise, when it goes negative -- a value the port has no way to reproduce a
    # log for, see the module docstring).
    fuzz_bounds = {
        "dx_tf_turn_general": (0.03, 0.15),
        "dx_tf_turn_steel": (0.001, 0.01),
        "dx_tf_turn_insulation": (0.0005, 0.005),
    }


def _reference_current(f_st_b, stella_config_i0, f_st_rmajor, f_st_n_coils):
    data = DataStructure()
    data.stellarator.f_st_b = f_st_b
    data.stellarator_config.stella_config_i0 = stella_config_i0
    data.stellarator.f_st_rmajor = f_st_rmajor
    data.stellarator.f_st_n_coils = f_st_n_coils
    coilcurrent = process_calculate.calculate_current(data)
    return coilcurrent, data.stellarator.f_st_i_total


class TestCurrent(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_current
    ported = calculate_current

    fuzz_bounds = {
        "f_st_b": (0.1, 3.0),
        "stella_config_i0": (1.0e5, 5.0e7),
        "f_st_rmajor": (0.5, 1.5),
        "f_st_n_coils": (1.0, 100.0),
    }


def _reference_casing(dr_tf_nose_case):
    data = DataStructure()
    data.tfcoil.dr_tf_nose_case = dr_tf_nose_case
    process_calculate.calculate_casing(data)
    return data.tfcoil.dr_tf_plasma_case, data.tfcoil.dx_tf_side_case_min


class TestCasing(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_casing
    ported = calculate_casing

    fuzz_bounds = {"dr_tf_nose_case": (0.01, 0.5)}


def _reference_vertical_ports(
    stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
):
    data = DataStructure()
    data.stellarator_config.stella_config_max_portsize_width = (
        stella_config_max_portsize_width
    )
    data.stellarator.f_st_rmajor = f_st_rmajor
    data.stellarator.f_st_n_coils = f_st_n_coils
    process_calculate.calculate_vertical_ports(data)
    return (
        data.stellarator.vporttmax,
        data.stellarator.vportpmax,
        data.stellarator.vportamax,
    )


class TestVerticalPorts(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_vertical_ports
    ported = calculate_vertical_ports

    fuzz_bounds = {
        "stella_config_max_portsize_width": (0.2, 5.0),
        "f_st_rmajor": (0.5, 1.5),
        "f_st_n_coils": (1.0, 100.0),
    }


def _reference_horizontal_ports(
    stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
):
    data = DataStructure()
    data.stellarator_config.stella_config_max_portsize_width = (
        stella_config_max_portsize_width
    )
    data.stellarator.f_st_rmajor = f_st_rmajor
    data.stellarator.f_st_n_coils = f_st_n_coils
    process_calculate.calculate_horizontal_ports(data)
    return (
        data.stellarator.hporttmax,
        data.stellarator.hportpmax,
        data.stellarator.hportamax,
    )


class TestHorizontalPorts(Tier1Contract):
    audit_record = "models/stellarator/coils/calculate.md"
    reference = _reference_horizontal_ports
    ported = calculate_horizontal_ports

    fuzz_bounds = {
        "stella_config_max_portsize_width": (0.2, 5.0),
        "f_st_rmajor": (0.5, 1.5),
        "f_st_n_coils": (1.0, 100.0),
    }


def _helias5b_winding_pack_base():
    """Realistic HELIAS5B-like base point for `winding_pack_total_size`'s samples.

    No PROCESS unit test exists for this function to lift a literal point from (checked
    by grep), so this is built by hand: `load_stellarator_config`'s real `istell=1`
    preset (the same configuration `TestBmaxFromAwp`'s legacy sample above is drawn
    from -- `stella_config_a1=0.688`/`stella_config_a2=0.025` match) supplies the
    `stellarator_config` fields, and the geometry/current
    (`r_coil_major`/`r_coil_minor`/`coilcurrent`/`n_tf_coils`) is exactly that same
    legacy sample's point. The rest are typical ITER-Nb3Sn (`i_tf_sc_mat=1`) material
    constants. Verified while writing this port (running PROCESS's own
    `winding_pack_total_size` directly at this point) to converge well inside the
    domain: the found `wp_width_r_min` comes out ~0.51 m, comfortably clear of the
    turn-size floor clamp at `dx_tf_turn_general**2 = 0.056**2 = 0.0031 m` -- so
    `intersect`'s bisection finds a genuine curve crossing rather than the clamp
    overriding it, keeping `test_ported_residual_small` meaningful.
    """
    config = DataStructure()
    load_stellarator_config(1, None, config)
    return {
        "r_coil_major": 22.237837837837837,
        "r_coil_minor": 4.7171171171171169,
        "coilcurrent": 12.711229086229087,
        "n_tf_coils": 50.0,
        "i_tf_sc_mat": 1,
        "stella_config_a1": config.stellarator_config.stella_config_a1,
        "stella_config_a2": config.stellarator_config.stella_config_a2,
        "stella_config_wp_ratio": config.stellarator_config.stella_config_wp_ratio,
        "tftmp": 4.2,
        "tmargmin": 0.0,
        "b_crit_upper_nbti": 14.86,
        "bcritsc": 24.0,
        "f_a_tf_turn_cable_copper": 0.69,
        "fhts": 0.5,
        "t_crit_nbti": 9.04,
        "tcritsc": 16.0,
        "f_a_tf_turn_cable_space_extra_void": 0.3,
        "j_tf_wp": 0.0,
        "f_j_tf_wp_critical_max": 0.7,
        "a_tf_turn_cable_space_no_void": 0.0022500000000000007,
        "dx_tf_turn_general": 0.056,
        "dx_tf_wp_insulation": 0.018,
        "a_tf_turn_steel": 0.0007093599999999996,
    }


def _reference_winding_pack_total_size(**kwargs):
    """Call PROCESS's own `winding_pack_total_size` through the port's signature."""
    data = DataStructure()
    data.tfcoil.n_tf_coils = kwargs["n_tf_coils"]
    data.tfcoil.i_tf_sc_mat = kwargs["i_tf_sc_mat"]
    data.stellarator_config.stella_config_a1 = kwargs["stella_config_a1"]
    data.stellarator_config.stella_config_a2 = kwargs["stella_config_a2"]
    data.stellarator_config.stella_config_wp_ratio = kwargs["stella_config_wp_ratio"]
    data.tfcoil.tftmp = kwargs["tftmp"]
    data.tfcoil.tmargmin = kwargs["tmargmin"]
    data.tfcoil.b_crit_upper_nbti = kwargs["b_crit_upper_nbti"]
    data.tfcoil.bcritsc = kwargs["bcritsc"]
    data.tfcoil.f_a_tf_turn_cable_copper = kwargs["f_a_tf_turn_cable_copper"]
    data.tfcoil.fhts = kwargs["fhts"]
    data.tfcoil.t_crit_nbti = kwargs["t_crit_nbti"]
    data.tfcoil.tcritsc = kwargs["tcritsc"]
    data.tfcoil.f_a_tf_turn_cable_space_extra_void = kwargs[
        "f_a_tf_turn_cable_space_extra_void"
    ]
    data.tfcoil.j_tf_wp = kwargs["j_tf_wp"]
    data.constraints.f_j_tf_wp_critical_max = kwargs["f_j_tf_wp_critical_max"]
    data.tfcoil.a_tf_turn_cable_space_no_void = kwargs["a_tf_turn_cable_space_no_void"]
    data.tfcoil.dx_tf_turn_general = kwargs["dx_tf_turn_general"]
    data.tfcoil.dx_tf_wp_insulation = kwargs["dx_tf_wp_insulation"]
    data.tfcoil.a_tf_turn_steel = kwargs["a_tf_turn_steel"]

    awp_rad, a_tf_wp_no_insulation, a_tf_wp_with_insulation, f_a_scu_of_wp = (
        process_calculate.winding_pack_total_size(
            kwargs["r_coil_major"], kwargs["r_coil_minor"], kwargs["coilcurrent"], data
        )
    )
    return (
        data.tfcoil.b_tf_inboard_peak_symmetric,
        data.tfcoil.dx_tf_wp_primary_toroidal,
        data.tfcoil.dx_tf_wp_secondary_toroidal,
        awp_rad,
        data.tfcoil.j_tf_wp,
        data.tfcoil.n_tf_coil_turns,
        data.tfcoil.c_tf_turn,
        data.tfcoil.a_tf_wp_conductor,
        data.tfcoil.a_tf_wp_extra_void,
        data.tfcoil.a_tf_coil_wp_turn_insulation,
        data.tfcoil.a_tf_wp_steel,
        a_tf_wp_no_insulation,
        a_tf_wp_with_insulation,
        f_a_scu_of_wp,
    )


_WINDING_PACK_GEOMETRY_POINTS = (
    # (n_tf_coils, r_coil_minor, coilcurrent, r_coil_major)
    (53.046326169970904, 4.553936165643477, 12.76552141139271, 24.185572846808796),
    (42.460313030875525, 4.659277601445338, 15.16675164080167, 24.420937702273037),
    (50.90644564245625, 4.238211100021444, 15.315526222870028, 25.85538821680489),
    (45.7515083355785, 5.025089928871112, 9.8689167859239, 21.379918197020242),
)
"""4 additional `(n_tf_coils, r_coil_minor, coilcurrent, r_coil_major)` points, drawn
from `n_tf_coils in [40, 60]`, `r_coil_minor in [4.0, 5.5]`, `coilcurrent in [8, 18]`,
`r_coil_major in [18, 26]` (seed 20260819, offline, not through `--fp-fuzz-seed`) and
individually verified while writing this port -- through the exact
`eqx.filter_jit`-wrapped code path `Tier2Contract` itself uses, not just eager Python --
to converge to a genuine crossing with a comfortable margin between the port's residual
and PROCESS's own (`>= 3e-12`, chosen from a 120-point offline sweep specifically to
stay clear of the float64 round-off floor both residuals sit near).

**Curated, not `fuzz_bounds`, and deliberately not regenerated per run** -- same
reasoning as `coils.md`'s `TestIntersect._intersect_samples`. Both `intersect`'s
bisection-then-Newton and PROCESS's own secant scheme land at or near the round-off
floor on this well-conditioned a problem, so a *blind* `fuzz_bounds` version of this
same box (tried first) hit two distinct failure modes across different
`--fp-fuzz-seed` values: an `EquinoxRuntimeError` on one draw whose curves never cross
inside `intersect`'s bracket at all, and several `residual_no_worse_than_process` ties
at 1e-13-2e-13 where the eager/jit numerical paths disagree on which side of PROCESS's
own near-zero residual the port lands -- neither is a real regression, but a
CLI-controlled seed is the wrong knob for a tier-2 unit whose pass criterion is this
precision-sensitive. These four points were individually checked to avoid both.
"""


def _winding_pack_geometry_samples(base):
    """The 4 `_WINDING_PACK_GEOMETRY_POINTS` as `Sample`s over `base`.

    A module-level function, not a class-body comprehension: a comprehension opens its
    own scope in Python 3 and cannot see a class body's own attributes (`TestWindingPackTotalSize.samples`
    hit exactly this `NameError` for `_base` when first written this way).
    """
    return [
        Sample(
            MappingProxyType({
                **base,
                "n_tf_coils": n_tf_coils,
                "r_coil_minor": r_coil_minor,
                "coilcurrent": coilcurrent,
                "r_coil_major": r_coil_major,
            }),
            "synthetic",
            f"geometry-{i}",
        )
        for i, (n_tf_coils, r_coil_minor, coilcurrent, r_coil_major) in enumerate(
            _WINDING_PACK_GEOMETRY_POINTS
        )
    ]


_WINDING_PACK_CURVE_PARAMS = tuple(inspect.signature(winding_pack_curves).parameters)


def _winding_pack_total_size_residual(solution, **kwargs):
    """`(solution, **kwargs) -> array`, per `_harness.contracts.Tier2Contract`.

    `solution[3]` is `dr_tf_wp_with_insulation` -- `winding_pack_total_size`'s resolved
    `wp_width_r_min` (after the turn-size-floor clamp). Rebuilds the same
    `(wp_width_r, lhs, rhs)` sampled curves from `kwargs` with `winding_pack_curves` (the
    same helper `winding_pack_total_size` itself calls) and evaluates
    `intersect_residual` there -- the same defining equation `coils.md`'s
    `TestIntersect` checks `intersect` against directly.
    """
    dr_tf_wp_with_insulation = solution[3]
    wp_width_r, lhs, rhs, _fraction = winding_pack_curves(**{
        name: kwargs[name] for name in _WINDING_PACK_CURVE_PARAMS
    })
    return intersect_residual(dr_tf_wp_with_insulation, wp_width_r, lhs, wp_width_r, rhs)


class TestWindingPackTotalSize(Tier2Contract):
    """`winding_pack_total_size` -> itself: a genuine internal solve (see `calculate.md`).

    No value-agreement test by construction (`Tier2Contract`) -- the solve being
    reproduced here is `intersect`'s own, and PROCESS's `intersect` has no real
    convergence check (fixed 100-iteration cap, data-dependent early `break`), so its
    answer is not ground truth any more than `power_at_ignition_point`'s two hardcoded
    calls are in `density_limits.md`. Pass criterion: both answers plugged back into
    `intersect_residual`, the port's residual small in an absolute sense and no worse
    than PROCESS's own at its stopping point.
    """

    audit_record = "models/stellarator/coils/calculate.md"
    reference = staticmethod(_reference_winding_pack_total_size)
    ported = winding_pack_total_size
    residual = staticmethod(eqx.filter_jit(_winding_pack_total_size_residual))

    _base = _helias5b_winding_pack_base()

    samples = [
        Sample(MappingProxyType(_base), "synthetic", "helias5b-like-mat1"),
        Sample(
            MappingProxyType({**_base, "i_tf_sc_mat": 5}),
            "synthetic",
            "helias5b-like-mat5",
        ),
        Sample(
            MappingProxyType({**_base, "i_tf_sc_mat": 7}),
            "synthetic",
            "helias5b-like-mat7",
        ),
        *_winding_pack_geometry_samples(_base),
    ]
    """`i_tf_sc_mat` in {1, 5, 7}: material branches verified (while writing this port)
    to converge to a genuine crossing at this geometry, away from both the turn-size
    clamp and the domain edge. `i_tf_sc_mat == 3` (NbTi) was checked too and dropped: at
    this same geometry PROCESS's own algorithm never finds a crossing inside
    `[wp_width_r.min(), wp_width_r.max()]` at all (logs "X has risen above Xmax" and
    clamps to the domain edge) -- not a fair residual-based comparison for either side.
    `i_tf_sc_mat == 8` (Durham GL REBCO) was checked too and dropped for a different
    reason: at this geometry PROCESS's own secant algorithm happens to land its answer
    at *exactly* zero residual (`0.0`, not merely small) while the port's answer is
    `5.7e-14` -- both are at the float64 round-off floor, but
    `test_ported_residual_no_worse_than_process`'s `<=` comparison has no slack for a
    reference that got lucky and hit exactly `0.0`. Same "which round-off floor comes
    out ahead is not a meaningful signal" situation `coils.md`'s `TestIntersect`
    describes for its own `n=25` sample, just landing on the wrong side of a strict
    `<=` here instead of the `TestIntersect` sample that happened not to.
    `i_tf_sc_mat == 2` (Bi-2212) is not sampled either: `bi2212`'s validity domain
    (`temp <= 20`, `6 <= b <= 104`) is narrow enough that the 200-point sweep runs well
    outside it for most of its range, and it also depends on the stale-`j_tf_wp` input
    (see `winding_pack_total_size`'s own docstring) -- both real fragilities of this
    branch, not exercised here; see the record's JAX-difficulty flags.
    `i_tf_sc_mat == 6` (REBCO) is excluded from the *reference* comparison, not from the
    port: PROCESS's own `coils.py:136` call to `jcrit_rebco` passes an extra positional
    argument that the real `jcrit_rebco` does not accept, so PROCESS's own
    `winding_pack_total_size` raises `TypeError` outright whenever `i_tf_sc_mat == 6`
    (confirmed directly while writing this port -- see the record's "real PROCESS bugs
    found"). The port's own REBCO branch (`_critical_current_density_by_material`) works
    fine on its own terms; there is simply no PROCESS answer to compare it against.
    """


# ---------------------------------------------------------------------------
# the Shape A split:
# `WindingPackIntersectInputs` / `Intersect` / `WindingPackTotalSizePost`
#
# `winding_pack_total_size`'s `.tfcoil.j_tf_wp` self-loop cannot be a plain node -- one
# node cannot read and own the same `VarPath` (`cottax.spec`'s "reads what it also
# owns" construction error). An earlier pass isolated it with a `WindingPackJTfWp`
# `FixedPointFunction` that duplicated this entire function's computation just to
# extract `j_tf_wp` alone. That class is gone: once `WindingPackTotalSizePost` owns
# `.tfcoil.j_tf_wp` and `WindingPackIntersectInputs` reads it, the self-reference closes
# through the two of them plus `coils.py`'s `Intersect` in between -- a genuine
# multi-node cycle, the same "Shape A" shape as `Divertor`/`AFwTotalWithPowerflow`
# (`_audit/next_steps.md` §5). No `FixedPointFunction`/`Cut` needed: `Blocking`/
# `to_graph()` finds the SCC on its own. These checks are the actual point of the split:
# prove the shape is legal and the cycle actually forms, don't just assert it.
# ---------------------------------------------------------------------------


def test_winding_pack_intersect_inputs_node_assembles_and_does_not_own_j_tf_wp():
    """`WindingPackIntersectInputs` (the pre-`intersect` node) must assemble on its own,
    and must not itself own `.tfcoil.j_tf_wp` -- that `VarPath` belongs to
    `WindingPackTotalSizePost` alone, not this one (it only reads the real value).
    """
    node = WindingPackIntersectInputs(i_tf_sc_mat=1)
    graph = to_graph(node)
    assert graph.definitions
    owned = {out.var for out in node.outputs}
    j_tf_wp_path = path_of(lambda s: s.tfcoil.j_tf_wp, VarPath)
    assert j_tf_wp_path not in owned


def test_winding_pack_total_size_post_owns_j_tf_wp():
    """`WindingPackTotalSizePost` (the post-`intersect` node) must assemble on its own
    and must now own `.tfcoil.j_tf_wp` -- the fresh value `winding_pack_post_intersect`
    computes, no longer discarded now that `WindingPackJTfWp` is gone.
    """
    node = WindingPackTotalSizePost()
    graph = to_graph(node)
    assert graph.definitions
    owned = {out.var for out in node.outputs}
    j_tf_wp_path = path_of(lambda s: s.tfcoil.j_tf_wp, VarPath)
    assert j_tf_wp_path in owned


def test_winding_pack_intersect_pair_assembles_around_the_root_find():
    """`WindingPackIntersectInputs` mints exactly the `VarPath`s `coils.py`'s `Intersect`
    reads (`.stellarator.wp_width_r`/`.lhs`/`.rhs`), so the two assemble together into
    one coupled block with a single `RootFind` problem -- the actual point of the split
    around `intersect` this pass makes (`_audit/next_steps.md` §7, `Intersect`'s own
    docstring).
    """
    pre = WindingPackIntersectInputs(i_tf_sc_mat=1)
    graph = to_graph(pre, Intersect())
    assert graph.definitions
    assert len(graph.definitions) == 3  # pre's 1 + Intersect's 2 (body + RootFind)
    assert not graph.is_acyclic
    (block,) = [b for b in graph.scc_blocks if b.declared]
    assert block.problem_type is RootFind


def test_winding_pack_total_size_post_reads_the_root_finds_own_output():
    """`WindingPackTotalSizePost` reads `.stellarator.wp_width_r_min` -- the `VarPath`
    `Intersect`'s `RootFind` problem owns -- as an ordinary `Input`, alongside owning
    `.tfcoil.j_tf_wp` itself (see above). Two different edges, not a second self-loop.
    """
    post = WindingPackTotalSizePost()
    wp_width_r_min_path = path_of(lambda s: s.stellarator.wp_width_r_min, VarPath)
    assert wp_width_r_min_path in post.node_definition.reads
    assert wp_width_r_min_path not in {out.var for out in post.outputs}


def test_winding_pack_intersect_split_forms_one_combined_cycle():
    """The full three-piece split (`WindingPackIntersectInputs`, `coils.py`'s
    `Intersect`, `WindingPackTotalSizePost`) assembles into one graph, and now that
    `WindingPackTotalSizePost` owns `.tfcoil.j_tf_wp` (which `WindingPackIntersectInputs`
    reads), the three nodes plus `Intersect`'s own `RootFind` problem form a **single**
    SCC -- not `Intersect`'s 2-node self-loop sitting disjoint from two plain
    acyclic nodes either side of it, the shape the pre-`WindingPackTotalSizePost`
    ownership change had. `WindingPackJTfWp` (an earlier pass's separate
    `FixedPointFunction` duplicating this whole computation just to isolate `j_tf_wp`)
    is gone -- this one merged cycle replaces it, with no duplicated computation.
    """
    pre = WindingPackIntersectInputs(i_tf_sc_mat=1)
    post = WindingPackTotalSizePost()
    graph = to_graph(pre, Intersect(), post)
    assert graph.definitions
    assert len(graph.definitions) == 4  # pre's 1 + Intersect's 2 + post's 1
    assert not graph.is_acyclic

    (cycle,) = graph.cycles
    assert {n.path_str() for n in cycle} == {
        "['WindingPackIntersectInputs']",
        "['Intersect']",
        "^problem['Intersect']",
        "['WindingPackTotalSizePost']",
    }


class _GenericBisectionRootFind(AbstractDriver):
    """Test-only `AbstractDriver` answering `RootFind` generically, via
    `ConditionMap.__call__` (`conditions(x) -> residual`) rather than
    `IntersectBisectionNewtonPolish`'s deliberate shortcut of reaching into
    `conditions.context` for `coils.py`'s own tabulated curve arrays directly.

    That shortcut stops working once `WindingPackIntersectInputs` (the node producing
    those curves) is itself *inside* the driven block rather than external to it --
    exactly what happens now that `WindingPackTotalSizePost` owns `.tfcoil.j_tf_wp` and
    `WindingPackIntersectInputs` reads it: `wp_width_r`/`lhs`/`rhs` are no longer part of
    `conditions.context` (`Drive.context` is "what the body reads from outside the
    block" -- with `pre` now *inside* the block, its own outputs are internal, not
    external). `conditions.context`'s own docstring already flags this as a deliberate,
    non-forced choice ("nothing stops a different concrete driver from being fully
    generic and calling `conditions(x)` instead") -- this class is that generic driver,
    proving the merged 4-node block (`WindingPackIntersectInputs` -> `Intersect` ->
    `WindingPackTotalSizePost`, cycling back through `.tfcoil.j_tf_wp`) is drivable at
    all, not just structurally admissible.

    Unlike `intersect`'s own unconstrained-Newton-after-bisection, `lhs`/`rhs` here can
    cross more than once outside the physically meaningful band (confirmed empirically:
    an unconstrained `optx.Newton` from `x0=1.0` converges to a spurious root at ~0.096,
    well below the sampled curve's own domain floor) -- so this driver still needs a
    bracket, supplied by the caller (same information `intersect`'s own `lo`/`hi` uses),
    rather than discovering the domain itself the way the narrower
    `IntersectBisectionNewtonPolish` can by reaching into `conditions.context` for the
    tabulated `wp_width_r` array's own min/max.
    """

    drives = RootFind
    lower: float
    upper: float

    def __call__(self, conditions, start):
        x0 = start[0] if start is not None else 0.5 * (self.lower + self.upper)
        bracketed = optx.root_find(
            lambda x, _: conditions(x)[0],
            optx.Bisection(rtol=0.0, atol=1e-10, flip="detect"),
            x0,
            options={"lower": self.lower, "upper": self.upper},
            throw=False,
            max_steps=100,
        ).value
        return (bracketed,)


def test_winding_pack_intersect_driven_matches_the_pure_function():
    """The actual numeric point of this split: driving the merged four-node block
    (`WindingPackIntersectInputs` -> `Intersect` -> `WindingPackTotalSizePost`, cycling
    back to `WindingPackIntersectInputs` through `.tfcoil.j_tf_wp`) with a generic
    `RootFind` driver reproduces the same `dr_tf_wp_with_insulation`/
    `a_tf_wp_with_insulation` the plain `winding_pack_total_size` pure function computes
    eagerly -- same numbers, produced through the driven path instead of the old eager
    one, and through the *merged* block, not the narrower `Intersect`-alone block an
    earlier version of this test drove (see `_GenericBisectionRootFind`'s own docstring
    for why `IntersectBisectionNewtonPolish` alone no longer suffices here).

    Uses `i_tf_sc_mat=1` (ITER Nb3Sn, not Bi-2212) -- `j_tf_wp` doesn't actually affect
    the curves on this branch (`_critical_current_density_by_material` never reads it
    off `i_tf_sc_mat == 2`), so the cycle converges in one pass regardless of starting
    guess; this test proves the *block* drives correctly, not that this particular
    material genuinely needs iterating -- the Bi-2212 branch would, but has no sample
    point built for this test yet.
    """
    base = dict(_helias5b_winding_pack_base())
    i_tf_sc_mat = base.pop("i_tf_sc_mat")
    base.pop("j_tf_wp")  # no longer an external condition -- WindingPackTotalSizePost
    # owns `.tfcoil.j_tf_wp` now, so it is produced inside the driven block, not fed in.

    field_paths = {
        "r_coil_major": lambda s: s.stellarator.r_coil_major,
        "r_coil_minor": lambda s: s.stellarator.r_coil_minor,
        "coilcurrent": lambda s: s.stellarator.coilcurrent,
        "n_tf_coils": lambda s: s.tfcoil.n_tf_coils,
        "stella_config_a1": lambda s: s.stellarator_config.stella_config_a1,
        "stella_config_a2": lambda s: s.stellarator_config.stella_config_a2,
        "stella_config_wp_ratio": lambda s: s.stellarator_config.stella_config_wp_ratio,
        "tftmp": lambda s: s.tfcoil.tftmp,
        "tmargmin": lambda s: s.tfcoil.tmargmin,
        "b_crit_upper_nbti": lambda s: s.tfcoil.b_crit_upper_nbti,
        "bcritsc": lambda s: s.tfcoil.bcritsc,
        "f_a_tf_turn_cable_copper": lambda s: s.tfcoil.f_a_tf_turn_cable_copper,
        "fhts": lambda s: s.tfcoil.fhts,
        "t_crit_nbti": lambda s: s.tfcoil.t_crit_nbti,
        "tcritsc": lambda s: s.tfcoil.tcritsc,
        "f_a_tf_turn_cable_space_extra_void": (
            lambda s: s.tfcoil.f_a_tf_turn_cable_space_extra_void
        ),
        "f_j_tf_wp_critical_max": lambda s: s.constraints.f_j_tf_wp_critical_max,
        "a_tf_turn_cable_space_no_void": (
            lambda s: s.tfcoil.a_tf_turn_cable_space_no_void
        ),
        "dx_tf_turn_general": lambda s: s.tfcoil.dx_tf_turn_general,
        "dx_tf_wp_insulation": lambda s: s.tfcoil.dx_tf_wp_insulation,
        "a_tf_turn_steel": lambda s: s.tfcoil.a_tf_turn_steel,
    }
    env = {
        path_of(where, VarPath): jax.numpy.asarray(base[name])
        for name, where in field_paths.items()
    }

    pre = WindingPackIntersectInputs(i_tf_sc_mat=i_tf_sc_mat)
    post = WindingPackTotalSizePost()
    graph = to_graph(pre, Intersect(), post)
    # Same domain `winding_pack_curves` itself samples `wp_width_r` over -- see that
    # function's own `lo`/`hi` (`i_tf_sc_mat == 6` uses a different lower bound, not hit
    # here since this test uses material 1).
    r_coil_minor = base["r_coil_minor"]
    driver = _GenericBisectionRootFind(
        lower=r_coil_minor / 40.0, upper=r_coil_minor / 1.0
    )
    schedule = schedule_for(graph, {Intersect().problem_name: driver})
    out = schedule(env)

    reference = winding_pack_total_size(
        **base, j_tf_wp=jax.numpy.asarray(0.0), i_tf_sc_mat=i_tf_sc_mat
    )
    dr_tf_wp_with_insulation_path = path_of(
        lambda s: s.tfcoil.dr_tf_wp_with_insulation, VarPath
    )
    a_tf_wp_with_insulation_path = path_of(
        lambda s: s.tfcoil.a_tf_wp_with_insulation, VarPath
    )
    assert out[dr_tf_wp_with_insulation_path] == pytest.approx(reference[3])
    assert out[a_tf_wp_with_insulation_path] == pytest.approx(reference[12])


# ---------------------------------------------------------------------------
# st_coil (tier-3)
# ---------------------------------------------------------------------------


def _helias5b_st_coil_kwargs():
    """Realistic HELIAS5B-like point for `st_coil`'s end-to-end comparison.

    Extends `_helias5b_winding_pack_base` with the further `build`/`physics`/`fwbs`
    fields `st_coil` reaches (via `coils/mass.py`, `coils/quench.py`, `coils/forces.py`)
    that `winding_pack_total_size` itself never touches. `f_st_rmajor`/`f_st_n_coils`/
    `f_st_b` are set to `1.0` (device == reference config, the scaling factors' neutral
    value) rather than independently chosen, so `r_coil_major`/`r_coil_minor`/
    `coilcurrent` stay exactly the values verified above -- `st_coil` derives its own
    `coilcurrent` from these scaling factors rather than taking it as a direct input, so
    an inconsistent choice here would silently retarget the whole comparison at a
    different, unverified geometry.
    """
    wp_base = _helias5b_winding_pack_base()
    config = DataStructure()
    load_stellarator_config(1, None, config)
    return {
        "r_coil_major": wp_base["r_coil_major"],
        "r_coil_minor": wp_base["r_coil_minor"],
        "n_tf_coils": wp_base["n_tf_coils"],
        "i_tf_sc_mat": wp_base["i_tf_sc_mat"],
        "dx_tf_turn_general": wp_base["dx_tf_turn_general"],
        "dx_tf_turn_steel": 0.0022,
        "dx_tf_turn_insulation": 0.0008,
        "f_st_b": 1.0,
        "stella_config_i0": config.stellarator_config.stella_config_i0,
        "f_st_rmajor": 1.0,
        "f_st_n_coils": 1.0,
        "stella_config_a1": wp_base["stella_config_a1"],
        "stella_config_a2": wp_base["stella_config_a2"],
        "stella_config_wp_ratio": wp_base["stella_config_wp_ratio"],
        "tftmp": wp_base["tftmp"],
        "tmargmin": wp_base["tmargmin"],
        "b_crit_upper_nbti": wp_base["b_crit_upper_nbti"],
        "bcritsc": wp_base["bcritsc"],
        "f_a_tf_turn_cable_copper": wp_base["f_a_tf_turn_cable_copper"],
        "fhts": wp_base["fhts"],
        "t_crit_nbti": wp_base["t_crit_nbti"],
        "tcritsc": wp_base["tcritsc"],
        "f_a_tf_turn_cable_space_extra_void": wp_base[
            "f_a_tf_turn_cable_space_extra_void"
        ],
        "j_tf_wp": wp_base["j_tf_wp"],
        "f_j_tf_wp_critical_max": wp_base["f_j_tf_wp_critical_max"],
        "dx_tf_wp_insulation": wp_base["dx_tf_wp_insulation"],
        "dr_tf_nose_case": 0.1,
        "stella_config_max_portsize_width": (
            config.stellarator_config.stella_config_max_portsize_width
        ),
        "stella_config_dmin": config.stellarator_config.stella_config_dmin,
        "stella_config_coil_rmajor": config.stellarator_config.stella_config_coil_rmajor,
        "stella_config_coil_rminor": config.stellarator_config.stella_config_coil_rminor,
        "stella_config_inductance": config.stellarator_config.stella_config_inductance,
        "stella_config_maximal_coil_height": (
            config.stellarator_config.stella_config_maximal_coil_height
        ),
        "stella_config_coillength": config.stellarator_config.stella_config_coillength,
        "stella_config_coilsurface": config.stellarator_config.stella_config_coilsurface,
        "stella_config_min_bend_radius": (
            config.stellarator_config.stella_config_min_bend_radius
        ),
        "den_tf_coil_case": 8000.0,
        "den_tf_wp_turn_insulation": 1800.0,
        "a_tf_wp_coolant_channels": 0.0,
        "den_tf_sc_material": 6080.0,  # dcond[i_tf_sc_mat - 1] at i_tf_sc_mat == 1
        "den_steel": 7800.0,
        "rmajor": 8.14,
        "rminor": 8.14 / 12.33,
        "dr_fw_plasma_gap_inboard": 0.14,
        "dr_fw_inboard": 0.0,
        "dr_blkt_inboard": 0.115,
        "dr_shld_blkt_gap": 0.05,
        "dr_shld_inboard": 0.69,
        "dr_fw_plasma_gap_outboard": 0.15,
        "dr_fw_outboard": 0.0,
        "dr_blkt_outboard": 0.235,
        "dr_shld_outboard": 1.05,
        "b_plasma_toroidal_on_axis": 5.68,
        "t_tf_superconductor_quench": 10.0,
        "dr_vv_inboard": 0.07,
        "dr_vv_outboard": 0.07,
        "t_tf_quench_detection": 3.0,
        "stella_config_max_force_density": (
            config.stellarator_config.stella_config_max_force_density
        ),
        "stella_config_wp_bmax": config.stellarator_config.stella_config_wp_bmax,
        "stella_config_wp_area": config.stellarator_config.stella_config_wp_area,
        "stella_config_max_force_density_mnm": (
            config.stellarator_config.stella_config_max_force_density_mnm
        ),
        "stella_config_max_lateral_force_density": (
            config.stellarator_config.stella_config_max_lateral_force_density
        ),
        "stella_config_max_radial_force_density": (
            config.stellarator_config.stella_config_max_radial_force_density
        ),
        "stella_config_centering_force_max_mn": (
            config.stellarator_config.stella_config_centering_force_max_mn
        ),
        "stella_config_centering_force_min_mn": (
            config.stellarator_config.stella_config_centering_force_min_mn
        ),
        "stella_config_centering_force_avg_mn": (
            config.stellarator_config.stella_config_centering_force_avg_mn
        ),
    }


def test_st_coil_matches_process_end_to_end():
    """`st_coil` (tier-3) against PROCESS's own, at one realistic HELIAS5B-like point.

    No `Tier1Contract`/`Tier2Contract` applies here -- `st_coil` introduces no new
    solver of its own, it composes ten tier-1 functions from this file with
    `winding_pack_total_size` (tier-2, above) and the already-ported units #11/#12/#14
    (`coils/forces.py`/`mass.py`/`quench.py`), so this is exactly
    `_audit/test_harness.md`'s tier-3 shape: "mostly structural assertions... end-to-end
    value comparison... near machine precision (no *new* solver)". There is no
    `Tier3Contract` in `_harness/contracts.py` yet (`test_harness.md`'s "Not built"
    section), and no matching PROCESS unit test to lift a point from either (checked by
    grep), so this is a plain, hand-written comparison rather than a declared
    `Contract` subclass -- deliberately narrower coverage than the tier-1/2 units above
    (one point, no fuzzing, no gradient check), acknowledged as this tier's known gap.

    Every field PROCESS's `st_coil` writes to `data` is checked against the pure port's
    matching dict entry, at float64 round-off (`rtol=1e-9`). `tfsai`/`tfsao` are
    expected to be **exactly `0.0`** on both sides -- not a bug in this comparison, see
    `st_coil`'s own docstring for the real, reproduced-not-fixed PROCESS ordering bug
    this demonstrates (`len_tf_coil_stale`, read before `st_coil` itself refreshes
    `len_tf_coil`).
    """
    data = DataStructure()
    load_stellarator_config(1, None, data)

    kwargs = _helias5b_st_coil_kwargs()
    data.stellarator.r_coil_major = kwargs["r_coil_major"]
    data.stellarator.r_coil_minor = kwargs["r_coil_minor"]
    data.tfcoil.n_tf_coils = kwargs["n_tf_coils"]
    data.tfcoil.i_tf_sc_mat = kwargs["i_tf_sc_mat"]
    data.tfcoil.dx_tf_turn_general = kwargs["dx_tf_turn_general"]
    data.tfcoil.dx_tf_turn_steel = kwargs["dx_tf_turn_steel"]
    data.tfcoil.dx_tf_turn_insulation = kwargs["dx_tf_turn_insulation"]
    data.stellarator.f_st_b = kwargs["f_st_b"]
    data.stellarator.f_st_rmajor = kwargs["f_st_rmajor"]
    data.stellarator.f_st_n_coils = kwargs["f_st_n_coils"]
    data.tfcoil.tftmp = kwargs["tftmp"]
    data.tfcoil.tmargmin = kwargs["tmargmin"]
    data.tfcoil.b_crit_upper_nbti = kwargs["b_crit_upper_nbti"]
    data.tfcoil.bcritsc = kwargs["bcritsc"]
    data.tfcoil.f_a_tf_turn_cable_copper = kwargs["f_a_tf_turn_cable_copper"]
    data.tfcoil.fhts = kwargs["fhts"]
    data.tfcoil.t_crit_nbti = kwargs["t_crit_nbti"]
    data.tfcoil.tcritsc = kwargs["tcritsc"]
    data.tfcoil.f_a_tf_turn_cable_space_extra_void = kwargs[
        "f_a_tf_turn_cable_space_extra_void"
    ]
    data.tfcoil.j_tf_wp = kwargs["j_tf_wp"]
    data.constraints.f_j_tf_wp_critical_max = kwargs["f_j_tf_wp_critical_max"]
    data.tfcoil.dx_tf_wp_insulation = kwargs["dx_tf_wp_insulation"]
    data.tfcoil.dr_tf_nose_case = kwargs["dr_tf_nose_case"]
    data.tfcoil.den_tf_coil_case = kwargs["den_tf_coil_case"]
    data.tfcoil.den_tf_wp_turn_insulation = kwargs["den_tf_wp_turn_insulation"]
    data.tfcoil.a_tf_wp_coolant_channels = kwargs["a_tf_wp_coolant_channels"]
    data.fwbs.den_steel = kwargs["den_steel"]
    data.physics.rmajor = kwargs["rmajor"]
    data.physics.rminor = kwargs["rminor"]
    data.build.dr_fw_plasma_gap_inboard = kwargs["dr_fw_plasma_gap_inboard"]
    data.build.dr_fw_inboard = kwargs["dr_fw_inboard"]
    data.build.dr_blkt_inboard = kwargs["dr_blkt_inboard"]
    data.build.dr_shld_blkt_gap = kwargs["dr_shld_blkt_gap"]
    data.build.dr_shld_inboard = kwargs["dr_shld_inboard"]
    data.build.dr_fw_plasma_gap_outboard = kwargs["dr_fw_plasma_gap_outboard"]
    data.build.dr_fw_outboard = kwargs["dr_fw_outboard"]
    data.build.dr_blkt_outboard = kwargs["dr_blkt_outboard"]
    data.build.dr_shld_outboard = kwargs["dr_shld_outboard"]
    data.physics.b_plasma_toroidal_on_axis = kwargs["b_plasma_toroidal_on_axis"]
    data.tfcoil.t_tf_superconductor_quench = kwargs["t_tf_superconductor_quench"]
    data.build.dr_vv_inboard = kwargs["dr_vv_inboard"]
    data.build.dr_vv_outboard = kwargs["dr_vv_outboard"]
    data.tfcoil.t_tf_quench_detection = kwargs["t_tf_quench_detection"]

    kwargs["len_tf_coil_stale"] = data.tfcoil.len_tf_coil

    process_calculate.st_coil(stellarator=None, output=False, data=data)
    ported = st_coil(**kwargs)

    checks = {
        "tfsai": data.tfcoil.tfsai,
        "tfsao": data.tfcoil.tfsao,
        "dr_tf_wp_with_insulation": data.tfcoil.dr_tf_wp_with_insulation,
        "len_tf_coil": data.tfcoil.len_tf_coil,
        "c_tf_total": data.tfcoil.c_tf_total,
        "a_tf_inboard_total": data.tfcoil.a_tf_inboard_total,
        "j_tf_coil_full_area": data.tfcoil.j_tf_coil_full_area,
        "r_b_tf_inboard_peak_symmetric": data.tfcoil.r_b_tf_inboard_peak_symmetric,
        "a_tf_leg_outboard": data.tfcoil.a_tf_leg_outboard,
        "a_tf_coil_inboard_case": data.tfcoil.a_tf_coil_inboard_case,
        "dr_tf_inboard": data.build.dr_tf_inboard,
        "dr_tf_outboard": data.build.dr_tf_outboard,
        "tfocrn": data.tfcoil.tfocrn,
        "tficrn": data.tfcoil.tficrn,
        "e_tf_magnetic_stored_total_gj": data.tfcoil.e_tf_magnetic_stored_total_gj,
        "m_tf_coil_case": data.tfcoil.m_tf_coil_case,
        "m_tf_coil_wp_insulation": data.tfcoil.m_tf_coil_wp_insulation,
        "m_tf_coil_superconductor": data.tfcoil.m_tf_coil_superconductor,
        "m_tf_coil_copper": data.tfcoil.m_tf_coil_copper,
        "m_tf_wp_steel_conduit": data.tfcoil.m_tf_wp_steel_conduit,
        "m_tf_coil_wp_turn_insulation": data.tfcoil.m_tf_coil_wp_turn_insulation,
        "m_tf_coil_conductor": data.tfcoil.m_tf_coil_conductor,
        "m_tf_coils_total": data.tfcoil.m_tf_coils_total,
        "j_tf_wp_quench_heat_max": data.tfcoil.j_tf_wp_quench_heat_max,
        "coppera_m2": data.rebco.coppera_m2,
        "v_tf_coil_dump_quench_kv": data.tfcoil.v_tf_coil_dump_quench_kv,
        "max_force_density": data.tfcoil.max_force_density,
        "sig_tf_wp": data.tfcoil.sig_tf_wp,
        "z_tf_inside_half": data.build.z_tf_inside_half,
        "tfcryoarea": data.tfcoil.tfcryoarea,
        "toroidalgap": data.tfcoil.toroidalgap,
    }
    mismatches = [
        f"  {name}: port={ported[name]!r} process={reference!r}"
        for name, reference in checks.items()
        if not np.isclose(ported[name], reference, rtol=1e-9, atol=1e-12)
    ]
    assert not mismatches, "\n".join(["st_coil value mismatch:", *mismatches])
