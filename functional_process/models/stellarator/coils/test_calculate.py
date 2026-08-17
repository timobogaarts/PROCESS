"""Harness cases for the ported `coils/calculate.py` tier-1 functions (registry unit #9).

No `legacy_sample` here: unlike `density_limits.py`/`stellarator_D`/`stellarator_F`,
none of these ten functions has a matching test in `tests/unit/models/stellarator/`
(checked by grep -- there is no dedicated stellarator-coils test file in PROCESS's own
suite), so there is no literal point to lift. Coverage is `fuzz_bounds`-only; agreement
is still checked live against the real PROCESS function at every sampled point, so this
is not weaker evidence, just a different provenance.
"""

from functional_process._harness import Tier1Contract
from functional_process.models.stellarator.coils.calculate import (
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
)
from process.core.model import DataStructure
from process.models.stellarator.coils import calculate as process_calculate


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


def _reference_plasma_facing_coil_area(n_tf_coils, dx_tf_inboard_out_toroidal, len_tf_coil):
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
    stella_config_inductance, f_st_rmajor, r_coil_minor, stella_config_coil_rminor, f_st_n_coils
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


def _reference_vertical_ports(stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils):
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


def _reference_horizontal_ports(stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils):
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
