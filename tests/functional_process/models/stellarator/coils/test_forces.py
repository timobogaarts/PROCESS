"""Harness cases for the ported `coils/forces.py` (registry #11).

No matching PROCESS unit test exists for any of these 7 functions (grepped
`tests/unit/models/stellarator/test_stellarator.py`), so every case here is fuzz-only —
still a real check (both port and reference run PROCESS's own arithmetic on the same
random points), just with no independently-validated legacy point to anchor it.
"""

from functional_process._harness import Tier1Contract
from functional_process.models.stellarator.coils.forces import (
    calculate_centering_force_avg_mn,
    calculate_centering_force_max_mn,
    calculate_centering_force_min_mn,
    calculate_max_force_density,
    calculate_max_force_density_mnm,
    calculate_max_lateral_force_density,
    calculate_max_radial_force_density,
    calculate_maximum_stress,
)
from process.core.model import DataStructure
from process.models.stellarator.coils import forces as _process_forces

_FORCE_DENSITY_FUZZ = {
    "a_tf_wp_no_insulation": (0.01, 5.0),
    "f_st_i_total": (0.1, 20.0),
    "f_st_n_coils": (1.0, 100.0),
    "b_tf_inboard_peak_symmetric": (1.0, 20.0),
}
_WP_BMAX_AREA_FUZZ = {
    "stella_config_wp_bmax": (1.0, 20.0),
    "stella_config_wp_area": (0.01, 5.0),
}
_CENTERING_FUZZ = {
    "f_st_i_total": (0.1, 20.0),
    "f_st_n_coils": (1.0, 100.0),
    "b_tf_inboard_peak_symmetric": (1.0, 20.0),
    "stella_config_wp_bmax": (1.0, 20.0),
    "stella_config_coillength": (1.0, 5000.0),
    "n_tf_coils": (1.0, 100.0),
    "len_tf_coil": (1.0, 5000.0),
}


def _reference_max_force_density(
    a_tf_wp_no_insulation,
    stella_config_max_force_density,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_wp_area,
):
    data = DataStructure()
    data.stellarator_config.stella_config_max_force_density = (
        stella_config_max_force_density
    )
    data.stellarator.f_st_i_total = f_st_i_total
    data.stellarator.f_st_n_coils = f_st_n_coils
    data.tfcoil.b_tf_inboard_peak_symmetric = b_tf_inboard_peak_symmetric
    data.stellarator_config.stella_config_wp_bmax = stella_config_wp_bmax
    data.stellarator_config.stella_config_wp_area = stella_config_wp_area
    _process_forces.calculate_max_force_density(a_tf_wp_no_insulation, data)
    return data.tfcoil.max_force_density


def _reference_max_force_density_mnm(
    stella_config_max_force_density_mnm,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
):
    data = DataStructure()
    data.stellarator_config.stella_config_max_force_density_mnm = (
        stella_config_max_force_density_mnm
    )
    data.stellarator.f_st_i_total = f_st_i_total
    data.stellarator.f_st_n_coils = f_st_n_coils
    data.tfcoil.b_tf_inboard_peak_symmetric = b_tf_inboard_peak_symmetric
    data.stellarator_config.stella_config_wp_bmax = stella_config_wp_bmax
    return _process_forces.calculate_max_force_density_mnm(data)


def _reference_maximum_stress(max_force_density, dr_tf_wp_with_insulation):
    data = DataStructure()
    data.tfcoil.max_force_density = max_force_density
    data.tfcoil.dr_tf_wp_with_insulation = dr_tf_wp_with_insulation
    _process_forces.calculate_maximum_stress(data)
    return data.tfcoil.sig_tf_wp


def _reference_max_lateral_force_density(
    a_tf_wp_no_insulation,
    stella_config_max_lateral_force_density,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_wp_area,
):
    data = DataStructure()
    data.stellarator_config.stella_config_max_lateral_force_density = (
        stella_config_max_lateral_force_density
    )
    data.stellarator.f_st_i_total = f_st_i_total
    data.stellarator.f_st_n_coils = f_st_n_coils
    data.tfcoil.b_tf_inboard_peak_symmetric = b_tf_inboard_peak_symmetric
    data.stellarator_config.stella_config_wp_bmax = stella_config_wp_bmax
    data.stellarator_config.stella_config_wp_area = stella_config_wp_area
    return _process_forces.calculate_max_lateral_force_density(
        a_tf_wp_no_insulation, data
    )


def _reference_max_radial_force_density(
    a_tf_wp_no_insulation,
    stella_config_max_radial_force_density,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_wp_area,
):
    data = DataStructure()
    data.stellarator_config.stella_config_max_radial_force_density = (
        stella_config_max_radial_force_density
    )
    data.stellarator.f_st_i_total = f_st_i_total
    data.stellarator.f_st_n_coils = f_st_n_coils
    data.tfcoil.b_tf_inboard_peak_symmetric = b_tf_inboard_peak_symmetric
    data.stellarator_config.stella_config_wp_bmax = stella_config_wp_bmax
    data.stellarator_config.stella_config_wp_area = stella_config_wp_area
    return _process_forces.calculate_max_radial_force_density(
        a_tf_wp_no_insulation, data
    )


def _centering_data(
    field_name,
    value,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_coillength,
    n_tf_coils,
    len_tf_coil,
):
    data = DataStructure()
    setattr(data.stellarator_config, field_name, value)
    data.stellarator.f_st_i_total = f_st_i_total
    data.stellarator.f_st_n_coils = f_st_n_coils
    data.tfcoil.b_tf_inboard_peak_symmetric = b_tf_inboard_peak_symmetric
    data.stellarator_config.stella_config_wp_bmax = stella_config_wp_bmax
    data.stellarator_config.stella_config_coillength = stella_config_coillength
    data.tfcoil.n_tf_coils = n_tf_coils
    data.tfcoil.len_tf_coil = len_tf_coil
    return data


def _reference_centering_force_max_mn(
    stella_config_centering_force_max_mn,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_coillength,
    n_tf_coils,
    len_tf_coil,
):
    data = _centering_data(
        "stella_config_centering_force_max_mn",
        stella_config_centering_force_max_mn,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_coillength,
        n_tf_coils,
        len_tf_coil,
    )
    return _process_forces.calculate_centering_force_max_mn(data)


def _reference_centering_force_min_mn(
    stella_config_centering_force_min_mn,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_coillength,
    n_tf_coils,
    len_tf_coil,
):
    data = _centering_data(
        "stella_config_centering_force_min_mn",
        stella_config_centering_force_min_mn,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_coillength,
        n_tf_coils,
        len_tf_coil,
    )
    return _process_forces.calculate_centering_force_min_mn(data)


def _reference_centering_force_avg_mn(
    stella_config_centering_force_avg_mn,
    f_st_i_total,
    f_st_n_coils,
    b_tf_inboard_peak_symmetric,
    stella_config_wp_bmax,
    stella_config_coillength,
    n_tf_coils,
    len_tf_coil,
):
    data = _centering_data(
        "stella_config_centering_force_avg_mn",
        stella_config_centering_force_avg_mn,
        f_st_i_total,
        f_st_n_coils,
        b_tf_inboard_peak_symmetric,
        stella_config_wp_bmax,
        stella_config_coillength,
        n_tf_coils,
        len_tf_coil,
    )
    return _process_forces.calculate_centering_force_avg_mn(data)


class TestMaxForceDensity(Tier1Contract):
    audit_record = "models/stellarator/coils/forces.md"
    reference = _reference_max_force_density
    ported = calculate_max_force_density
    fuzz_bounds = {
        **_FORCE_DENSITY_FUZZ,
        "stella_config_max_force_density": (0.1, 50.0),
        "stella_config_wp_bmax": _WP_BMAX_AREA_FUZZ["stella_config_wp_bmax"],
        "stella_config_wp_area": _WP_BMAX_AREA_FUZZ["stella_config_wp_area"],
    }


class TestMaxForceDensityMnm(Tier1Contract):
    audit_record = "models/stellarator/coils/forces.md"
    reference = _reference_max_force_density_mnm
    ported = calculate_max_force_density_mnm
    fuzz_bounds = {
        "stella_config_max_force_density_mnm": (0.1, 50.0),
        "f_st_i_total": _FORCE_DENSITY_FUZZ["f_st_i_total"],
        "f_st_n_coils": _FORCE_DENSITY_FUZZ["f_st_n_coils"],
        "b_tf_inboard_peak_symmetric": _FORCE_DENSITY_FUZZ["b_tf_inboard_peak_symmetric"],
        "stella_config_wp_bmax": _WP_BMAX_AREA_FUZZ["stella_config_wp_bmax"],
    }


class TestMaximumStress(Tier1Contract):
    audit_record = "models/stellarator/coils/forces.md"
    reference = _reference_maximum_stress
    ported = calculate_maximum_stress
    fuzz_bounds = {
        "max_force_density": (0.01, 100.0),
        "dr_tf_wp_with_insulation": (0.01, 2.0),
    }


class TestMaxLateralForceDensity(Tier1Contract):
    audit_record = "models/stellarator/coils/forces.md"
    reference = _reference_max_lateral_force_density
    ported = calculate_max_lateral_force_density
    fuzz_bounds = {
        **_FORCE_DENSITY_FUZZ,
        "stella_config_max_lateral_force_density": (0.1, 50.0),
        "stella_config_wp_bmax": _WP_BMAX_AREA_FUZZ["stella_config_wp_bmax"],
        "stella_config_wp_area": _WP_BMAX_AREA_FUZZ["stella_config_wp_area"],
    }


class TestMaxRadialForceDensity(Tier1Contract):
    audit_record = "models/stellarator/coils/forces.md"
    reference = _reference_max_radial_force_density
    ported = calculate_max_radial_force_density
    fuzz_bounds = {
        **_FORCE_DENSITY_FUZZ,
        "stella_config_max_radial_force_density": (0.1, 50.0),
        "stella_config_wp_bmax": _WP_BMAX_AREA_FUZZ["stella_config_wp_bmax"],
        "stella_config_wp_area": _WP_BMAX_AREA_FUZZ["stella_config_wp_area"],
    }


class TestCenteringForceMaxMn(Tier1Contract):
    audit_record = "models/stellarator/coils/forces.md"
    reference = _reference_centering_force_max_mn
    ported = calculate_centering_force_max_mn
    fuzz_bounds = {
        **_CENTERING_FUZZ,
        "stella_config_centering_force_max_mn": (0.1, 50.0),
    }


class TestCenteringForceMinMn(Tier1Contract):
    audit_record = "models/stellarator/coils/forces.md"
    reference = _reference_centering_force_min_mn
    ported = calculate_centering_force_min_mn
    fuzz_bounds = {
        **_CENTERING_FUZZ,
        "stella_config_centering_force_min_mn": (0.1, 50.0),
    }


class TestCenteringForceAvgMn(Tier1Contract):
    audit_record = "models/stellarator/coils/forces.md"
    reference = _reference_centering_force_avg_mn
    ported = calculate_centering_force_avg_mn
    fuzz_bounds = {
        **_CENTERING_FUZZ,
        "stella_config_centering_force_avg_mn": (0.1, 50.0),
    }
