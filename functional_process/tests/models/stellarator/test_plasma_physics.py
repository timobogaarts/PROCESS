"""Harness cases for chunk 1B's genuinely-new sub-computations (`st_phys`'s own body,
not delegated to another model). See `plasma_physics.py`'s module docstring for
which pieces these are and why the rest of `st_phys` is not ported here.

None of these eight expressions is separately callable in PROCESS -- each lives inline
in `Stellarator.st_phys`'s ~570-line body, never isolated into its own method -- so
there is no PROCESS function to call as a reference (the same situation
`structure.py`'s `calculate_intercoil_mass_scaling_reference` was in).
Each reference below is an independent, from-source transcription of the relevant
`stellarator.py` lines (cited per class), not a copy of the port's own expression --
that is what keeps this test able to catch a transcription mistake in the port rather
than just checking the port agrees with itself.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.stellarator.plasma_physics import (
    calculate_clipped_radiation_powers,
    calculate_fusion_power_totals_mw,
    calculate_fusion_totals_no_beam,
    calculate_heating_and_radiation_power,
    calculate_neutron_wall_load,
    calculate_poloidal_field_from_rotational_transform,
    calculate_radiated_wall_load_and_fraction,
    calculate_stellarator_beta_and_rho_star,
    calculate_thermal_energy_totals,
    calculate_total_field,
)
from process.core import constants


def _reference_total_field(b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average):
    """`stellarator.py:1916-1919`."""
    return np.sqrt(b_plasma_toroidal_on_axis**2 + b_plasma_surface_poloidal_average**2)


class TestTotalField(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_total_field
    ported = calculate_total_field

    samples = FROM_FILE


def _reference_poloidal_field_from_rotational_transform(
    rminor, b_plasma_toroidal_on_axis, rmajor, iotabar
):
    """`stellarator.py:1971-1976`."""
    return rminor * b_plasma_toroidal_on_axis / rmajor * iotabar


class TestPoloidalFieldFromRotationalTransform(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_poloidal_field_from_rotational_transform
    ported = calculate_poloidal_field_from_rotational_transform

    samples = FROM_FILE


def _reference_stellarator_beta_and_rho_star(
    beta_fast_alpha,
    beta_beam,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_ion_density_weighted_kev,
    b_plasma_total,
    vol_plasma,
    m_ions_total_amu,
    nd_plasma_electron_line,
    b_plasma_toroidal_on_axis,
    eps,
    rmajor,
):
    """`stellarator.py:1930-1968`."""
    beta_total_vol_avg = (
        beta_fast_alpha
        + beta_beam
        + 2.0e3
        * constants.RMU0
        * constants.ELECTRON_CHARGE
        * (
            nd_plasma_electrons_vol_avg * temp_plasma_electron_density_weighted_kev
            + nd_plasma_ions_total_vol_avg * temp_plasma_ion_density_weighted_kev
        )
        / b_plasma_total**2
    )
    e_plasma_beta = (
        1.5e0
        * beta_total_vol_avg
        * b_plasma_total
        * b_plasma_total
        / (2.0e0 * constants.RMU0)
        * vol_plasma
    )
    rho_star = np.sqrt(
        2.0e0
        * constants.PROTON_MASS
        * m_ions_total_amu
        * e_plasma_beta
        / (3.0e0 * vol_plasma * nd_plasma_electron_line)
    ) / (constants.ELECTRON_CHARGE * b_plasma_toroidal_on_axis * eps * rmajor)
    return beta_total_vol_avg, e_plasma_beta, rho_star


class TestStellaratorBetaAndRhoStar(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_stellarator_beta_and_rho_star
    ported = calculate_stellarator_beta_and_rho_star

    samples = FROM_FILE


def _reference_fusion_power_totals_mw(
    dt_power_density_plasma, dhe3_power_density, dd_power_density, vol_plasma
):
    """`stellarator.py:1991-2001`."""
    return (
        dt_power_density_plasma * vol_plasma,
        dhe3_power_density * vol_plasma,
        dd_power_density * vol_plasma,
    )


class TestFusionPowerTotalsMw(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_fusion_power_totals_mw
    ported = calculate_fusion_power_totals_mw

    samples = FROM_FILE


def _reference_fusion_totals_no_beam(fusden_plasma, fusden_plasma_alpha, p_plasma_dt_mw):
    """`stellarator.py:2045-2054`, the `else` (no-beam) arm."""
    return fusden_plasma, fusden_plasma_alpha, p_plasma_dt_mw


class TestFusionTotalsNoBeam(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_fusion_totals_no_beam
    ported = calculate_fusion_totals_no_beam

    samples = FROM_FILE


def _reference_clipped_radiation_powers(
    pden_plasma_core_rad_mw_unclipped, pden_plasma_outer_rad_mw_unclipped, vol_plasma
):
    """`stellarator.py:2152-2166`, transcribed from source: two `max(..., 0.0)` clips
    and the two products formed from the clipped values.
    """
    core = max(pden_plasma_core_rad_mw_unclipped, 0.0)
    outer = max(pden_plasma_outer_rad_mw_unclipped, 0.0)
    return core, outer, core * vol_plasma, outer * vol_plasma


class TestClippedRadiationPowers(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_clipped_radiation_powers
    ported = calculate_clipped_radiation_powers

    samples = FROM_FILE


def _reference_neutron_wall_load(
    i_pflux_fw_neutron,
    ipowerflow,
    ffwal,
    p_neutron_total_mw,
    a_plasma_surface,
    fhole,
    a_fw_total,
    f_a_fw_outboard_hcd,
    f_ster_div_single,
):
    """`stellarator.py:2095-2117`."""
    if i_pflux_fw_neutron == 1:
        return ffwal * p_neutron_total_mw / a_plasma_surface
    if ipowerflow == 0:
        return (1.0 - fhole) * p_neutron_total_mw / a_fw_total
    return (
        (1.0 - fhole - f_a_fw_outboard_hcd - f_ster_div_single)
        * p_neutron_total_mw
        / a_fw_total
    )


_NEUTRON_WALL_LOAD_FUZZ_BOUNDS = {
    "ffwal": (0.5, 1.0),
    "p_neutron_total_mw": (100.0, 3000.0),
    "a_plasma_surface": (500.0, 4000.0),
    "fhole": (0.0, 0.2),
    "a_fw_total": (500.0, 4000.0),
    "f_a_fw_outboard_hcd": (0.0, 0.1),
    "f_ster_div_single": (0.0, 0.2),
}


class TestNeutronWallLoadDirect(Tier1Contract):
    """`i_pflux_fw_neutron == 1` branch."""

    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_neutron_wall_load
    ported = calculate_neutron_wall_load
    static_argnames = ("i_pflux_fw_neutron", "ipowerflow")

    samples = FROM_FILE


class TestNeutronWallLoadSimplePowerflow(TestNeutronWallLoadDirect):
    """`i_pflux_fw_neutron == 0`, `ipowerflow == 0` branch."""

    samples = FROM_FILE


class TestNeutronWallLoadDetailedPowerflow(TestNeutronWallLoadDirect):
    """`i_pflux_fw_neutron == 0`, `ipowerflow != 0` branch."""

    samples = FROM_FILE


def _reference_heating_and_radiation_power(
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    pden_plasma_rad_mw,
    vol_plasma,
    i_plasma_ignited,
    p_hcd_injected_total_mw,
    f_rad,
):
    """`stellarator.py:2175-2220`."""
    powht = (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_plasma_ohmic_mw
        - pden_plasma_rad_mw * vol_plasma
    )
    powht = max(0.00001, powht)

    if i_plasma_ignited == 0:
        powht += p_hcd_injected_total_mw

    p_plasma_rad_mw = max(0.0, pden_plasma_rad_mw * vol_plasma)

    psolradmw = f_rad * powht
    p_plasma_separatrix_mw = powht - psolradmw

    p_plasma_rad_mw += psolradmw

    p_plasma_separatrix_mw = max(0.001, p_plasma_separatrix_mw)

    p_fw_alpha_mw = p_alpha_total_mw * (1.0 - f_p_alpha_plasma_deposited)

    return p_plasma_rad_mw, psolradmw, p_plasma_separatrix_mw, p_fw_alpha_mw


_HEATING_AND_RADIATION_POWER_FUZZ_BOUNDS = {
    "f_p_alpha_plasma_deposited": (0.5, 1.0),
    "p_alpha_total_mw": (10.0, 800.0),
    "p_non_alpha_charged_mw": (0.0, 100.0),
    "p_plasma_ohmic_mw": (0.0, 5.0),
    "pden_plasma_rad_mw": (0.0, 1.0),
    "vol_plasma": (100.0, 3000.0),
    "p_hcd_injected_total_mw": (0.0, 100.0),
    "f_rad": (0.1, 0.95),
}


class TestHeatingAndRadiationPowerNonIgnited(Tier1Contract):
    """`i_plasma_ignited == 0` (NON_IGNITED): auxiliary power is added to `powht`."""

    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_heating_and_radiation_power
    ported = calculate_heating_and_radiation_power
    static_argnames = ("i_plasma_ignited",)

    samples = FROM_FILE


class TestHeatingAndRadiationPowerIgnited(TestHeatingAndRadiationPowerNonIgnited):
    """`i_plasma_ignited == 1` (IGNITED): no auxiliary power added."""

    samples = FROM_FILE


def _reference_radiated_wall_load_and_fraction(
    i_pflux_fw_neutron,
    ipowerflow,
    ffwal,
    p_plasma_rad_mw,
    a_plasma_surface,
    fhole,
    a_fw_total,
    f_a_fw_outboard_hcd,
    f_ster_div_single,
    f_fw_rad_max,
    f_p_alpha_plasma_deposited,
    p_alpha_total_mw,
    p_non_alpha_charged_mw,
    p_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
):
    """`stellarator.py:2223-2257`."""
    if i_pflux_fw_neutron == 1:
        pflux_fw_rad_mw = ffwal * p_plasma_rad_mw / a_plasma_surface
    elif ipowerflow == 0:
        pflux_fw_rad_mw = (1.0 - fhole) * p_plasma_rad_mw / a_fw_total
    else:
        pflux_fw_rad_mw = (
            (1.0 - fhole - f_a_fw_outboard_hcd - f_ster_div_single)
            * p_plasma_rad_mw
            / a_fw_total
        )

    pflux_fw_rad_max_mw = pflux_fw_rad_mw * f_fw_rad_max

    rad_fraction_total = p_plasma_rad_mw / (
        f_p_alpha_plasma_deposited * p_alpha_total_mw
        + p_non_alpha_charged_mw
        + p_plasma_ohmic_mw
        + p_hcd_injected_total_mw
    )

    return pflux_fw_rad_mw, pflux_fw_rad_max_mw, rad_fraction_total


_RADIATED_WALL_LOAD_FUZZ_BOUNDS = {
    "ffwal": (0.5, 1.0),
    "p_plasma_rad_mw": (50.0, 1000.0),
    "a_plasma_surface": (500.0, 4000.0),
    "fhole": (0.0, 0.2),
    "a_fw_total": (500.0, 4000.0),
    "f_a_fw_outboard_hcd": (0.0, 0.1),
    "f_ster_div_single": (0.0, 0.2),
    "f_fw_rad_max": (1.0, 5.0),
    "f_p_alpha_plasma_deposited": (0.5, 1.0),
    "p_alpha_total_mw": (10.0, 800.0),
    "p_non_alpha_charged_mw": (0.0, 100.0),
    "p_plasma_ohmic_mw": (0.0, 5.0),
    "p_hcd_injected_total_mw": (0.0, 100.0),
}


class TestRadiatedWallLoadAndFractionDirect(Tier1Contract):
    """`i_pflux_fw_neutron == 1` branch."""

    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_radiated_wall_load_and_fraction
    ported = calculate_radiated_wall_load_and_fraction
    static_argnames = ("i_pflux_fw_neutron", "ipowerflow")

    samples = FROM_FILE


class TestRadiatedWallLoadAndFractionSimplePowerflow(
    TestRadiatedWallLoadAndFractionDirect
):
    """`i_pflux_fw_neutron == 0`, `ipowerflow == 0` branch."""

    samples = FROM_FILE


class TestRadiatedWallLoadAndFractionDetailedPowerflow(
    TestRadiatedWallLoadAndFractionDirect
):
    """`i_pflux_fw_neutron == 0`, `ipowerflow != 0` branch."""

    samples = FROM_FILE


def _reference_thermal_energy_totals(
    eden_plasma_electrons_thermal_vol_avg,
    eden_plasma_ions_thermal_vol_avg,
    e_plasma_electrons_thermal,
    e_plasma_ions_thermal,
):
    """`stellarator.py:2282-2290`."""
    return (
        eden_plasma_electrons_thermal_vol_avg + eden_plasma_ions_thermal_vol_avg,
        e_plasma_electrons_thermal + e_plasma_ions_thermal,
    )


class TestThermalEnergyTotals(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_thermal_energy_totals
    ported = calculate_thermal_energy_totals

    samples = FROM_FILE
