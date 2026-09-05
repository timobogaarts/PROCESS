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

from functional_process._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.cottax.stellarator.plasma_physics import (
    calculate_fusion_power_totals_mw,
    calculate_clipped_radiation_powers,
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

    samples = [
        legacy_sample(
            "typical-helias",
            b_plasma_toroidal_on_axis=5.5,
            b_plasma_surface_poloidal_average=0.6,
        ),
        *fuzz_samples(
            {
                "b_plasma_toroidal_on_axis": (2.0, 12.0),
                "b_plasma_surface_poloidal_average": (0.05, 2.0),
            },
            count=5,
            seed=0,
        ),
    ]


def _reference_poloidal_field_from_rotational_transform(
    rminor, b_plasma_toroidal_on_axis, rmajor, iotabar
):
    """`stellarator.py:1971-1976`."""
    return rminor * b_plasma_toroidal_on_axis / rmajor * iotabar


class TestPoloidalFieldFromRotationalTransform(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_poloidal_field_from_rotational_transform
    ported = calculate_poloidal_field_from_rotational_transform

    samples = [
        legacy_sample(
            "typical-helias",
            rminor=1.7842660178426601,
            b_plasma_toroidal_on_axis=5.5,
            rmajor=22.0,
            iotabar=1.0,
        ),
        *fuzz_samples(
            {
                "rminor": (0.3, 5.0),
                "b_plasma_toroidal_on_axis": (2.0, 12.0),
                "rmajor": (3.0, 30.0),
                "iotabar": (0.1, 2.0),
            },
            count=5,
            seed=0,
        ),
    ]


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

    samples = [
        legacy_sample(
            "typical-helias",
            beta_fast_alpha=0.001,
            beta_beam=0.0005,
            nd_plasma_electrons_vol_avg=7.5e19,
            temp_plasma_electron_density_weighted_kev=13.0,
            nd_plasma_ions_total_vol_avg=6.6e19,
            temp_plasma_ion_density_weighted_kev=13.0,
            b_plasma_total=5.5327,
            vol_plasma=1400.0,
            m_ions_total_amu=2.5,
            nd_plasma_electron_line=2.357822619799476e20,
            b_plasma_toroidal_on_axis=5.5,
            eps=0.0811,
            rmajor=22.0,
        ),
        *fuzz_samples(
            {
                "beta_fast_alpha": (0.0001, 0.05),
                "beta_beam": (0.0, 0.02),
                "nd_plasma_electrons_vol_avg": (2.0e19, 1.0e21),
                "temp_plasma_electron_density_weighted_kev": (1.0, 30.0),
                "nd_plasma_ions_total_vol_avg": (1.0e19, 1.0e21),
                "temp_plasma_ion_density_weighted_kev": (1.0, 30.0),
                "b_plasma_total": (1.0, 15.0),
                "vol_plasma": (100.0, 3000.0),
                "m_ions_total_amu": (1.0, 3.0),
                "nd_plasma_electron_line": (2.0e19, 1.0e21),
                "b_plasma_toroidal_on_axis": (2.0, 12.0),
                "eps": (0.02, 0.3),
                "rmajor": (3.0, 30.0),
            },
            count=5,
            seed=0,
        ),
    ]


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

    samples = [
        legacy_sample(
            "typical-helias",
            dt_power_density_plasma=0.5,
            dhe3_power_density=0.0,
            dd_power_density=0.01,
            vol_plasma=1400.0,
        ),
        *fuzz_samples(
            {
                "dt_power_density_plasma": (0.0, 2.0),
                "dhe3_power_density": (0.0, 1.0),
                "dd_power_density": (0.0, 0.5),
                "vol_plasma": (100.0, 3000.0),
            },
            count=5,
            seed=0,
        ),
    ]


def _reference_fusion_totals_no_beam(fusden_plasma, fusden_plasma_alpha, p_plasma_dt_mw):
    """`stellarator.py:2045-2054`, the `else` (no-beam) arm."""
    return fusden_plasma, fusden_plasma_alpha, p_plasma_dt_mw


class TestFusionTotalsNoBeam(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_fusion_totals_no_beam
    ported = calculate_fusion_totals_no_beam

    samples = [
        legacy_sample(
            "typical-helias",
            fusden_plasma=1.2e18,
            fusden_plasma_alpha=1.1e18,
            p_plasma_dt_mw=700.0,
        ),
        *fuzz_samples(
            {
                "fusden_plasma": (1.0e16, 1.0e20),
                "fusden_plasma_alpha": (1.0e16, 1.0e20),
                "p_plasma_dt_mw": (0.0, 3000.0),
            },
            count=5,
            seed=0,
        ),
    ]


def _reference_clipped_radiation_powers(
    pden_plasma_core_rad_mw_unclipped, pden_plasma_outer_rad_mw_unclipped, vol_plasma
):
    """`stellarator.py:2152-2166`, transcribed from source: two `max(..., 0.0)` clips
    and the two products formed from the clipped values."""
    core = max(pden_plasma_core_rad_mw_unclipped, 0.0)
    outer = max(pden_plasma_outer_rad_mw_unclipped, 0.0)
    return core, outer, core * vol_plasma, outer * vol_plasma


class TestClippedRadiationPowers(Tier1Contract):
    audit_record = "models/stellarator/plasma_physics.md"
    reference = _reference_clipped_radiation_powers
    ported = calculate_clipped_radiation_powers

    samples = [
        legacy_sample(
            "clip-inactive-helias",
            pden_plasma_core_rad_mw_unclipped=0.057544135593658154,
            pden_plasma_outer_rad_mw_unclipped=0.05525606,
            vol_plasma=2475.6886164316024,
        ),
        # The clip's *active* side, which this run never reaches. Kept because the whole
        # reason this block is its own node is that PROCESS clips here and does not clip
        # at `calculate_radiation_powers`'s other call site -- an arm that is only ever
        # exercised by a sample.
        legacy_sample(
            "clip-active-negative-core",
            pden_plasma_core_rad_mw_unclipped=-0.01,
            pden_plasma_outer_rad_mw_unclipped=-0.002,
            vol_plasma=1400.0,
        ),
        *fuzz_samples(
            {
                "pden_plasma_core_rad_mw_unclipped": (-0.05, 0.5),
                "pden_plasma_outer_rad_mw_unclipped": (-0.05, 0.5),
                "vol_plasma": (100.0, 3000.0),
            },
            count=5,
            seed=0,
        ),
    ]


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

    samples = [
        legacy_sample(
            "direct-branch",
            i_pflux_fw_neutron=1,
            ipowerflow=1,
            ffwal=0.9,
            p_neutron_total_mw=1500.0,
            a_plasma_surface=1925.3641313657533,
            fhole=0.0,
            a_fw_total=1918.87696696527,
            f_a_fw_outboard_hcd=0.0,
            f_ster_div_single=0.115,
        ),
        *fuzz_samples(
            _NEUTRON_WALL_LOAD_FUZZ_BOUNDS,
            count=5,
            seed=0,
            fixed={"i_pflux_fw_neutron": 1, "ipowerflow": 1},
        ),
    ]


class TestNeutronWallLoadSimplePowerflow(TestNeutronWallLoadDirect):
    """`i_pflux_fw_neutron == 0`, `ipowerflow == 0` branch."""

    samples = [
        legacy_sample(
            "simple-powerflow-branch",
            i_pflux_fw_neutron=0,
            ipowerflow=0,
            ffwal=0.9,
            p_neutron_total_mw=1500.0,
            a_plasma_surface=1925.3641313657533,
            fhole=0.0,
            a_fw_total=1918.87696696527,
            f_a_fw_outboard_hcd=0.0,
            f_ster_div_single=0.115,
        ),
        *fuzz_samples(
            _NEUTRON_WALL_LOAD_FUZZ_BOUNDS,
            count=5,
            seed=1,
            fixed={"i_pflux_fw_neutron": 0, "ipowerflow": 0},
        ),
    ]


class TestNeutronWallLoadDetailedPowerflow(TestNeutronWallLoadDirect):
    """`i_pflux_fw_neutron == 0`, `ipowerflow != 0` branch."""

    samples = [
        legacy_sample(
            "detailed-powerflow-branch",
            i_pflux_fw_neutron=0,
            ipowerflow=1,
            ffwal=0.9,
            p_neutron_total_mw=1500.0,
            a_plasma_surface=1925.3641313657533,
            fhole=0.0,
            a_fw_total=2120.685245576686,
            f_a_fw_outboard_hcd=0.0,
            f_ster_div_single=0.021924555536480182,
        ),
        *fuzz_samples(
            _NEUTRON_WALL_LOAD_FUZZ_BOUNDS,
            count=5,
            seed=2,
            fixed={"i_pflux_fw_neutron": 0, "ipowerflow": 1},
        ),
    ]


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

    samples = [
        legacy_sample(
            "non-ignited",
            f_p_alpha_plasma_deposited=0.95,
            p_alpha_total_mw=400.0,
            p_non_alpha_charged_mw=20.0,
            p_plasma_ohmic_mw=0.0,
            pden_plasma_rad_mw=0.3,
            vol_plasma=1400.0,
            i_plasma_ignited=0,
            p_hcd_injected_total_mw=50.0,
            f_rad=0.85,
        ),
        *fuzz_samples(
            _HEATING_AND_RADIATION_POWER_FUZZ_BOUNDS,
            count=5,
            seed=0,
            fixed={"i_plasma_ignited": 0},
        ),
    ]


class TestHeatingAndRadiationPowerIgnited(TestHeatingAndRadiationPowerNonIgnited):
    """`i_plasma_ignited == 1` (IGNITED): no auxiliary power added."""

    samples = [
        legacy_sample(
            "ignited",
            f_p_alpha_plasma_deposited=0.95,
            p_alpha_total_mw=400.0,
            p_non_alpha_charged_mw=20.0,
            p_plasma_ohmic_mw=0.0,
            pden_plasma_rad_mw=0.3,
            vol_plasma=1400.0,
            i_plasma_ignited=1,
            p_hcd_injected_total_mw=0.0,
            f_rad=0.85,
        ),
        *fuzz_samples(
            _HEATING_AND_RADIATION_POWER_FUZZ_BOUNDS,
            count=5,
            seed=1,
            fixed={"i_plasma_ignited": 1},
        ),
    ]


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

    samples = [
        legacy_sample(
            "direct-branch",
            i_pflux_fw_neutron=1,
            ipowerflow=1,
            ffwal=0.9,
            p_plasma_rad_mw=350.0,
            a_plasma_surface=1925.3641313657533,
            fhole=0.0,
            a_fw_total=1918.87696696527,
            f_a_fw_outboard_hcd=0.0,
            f_ster_div_single=0.115,
            f_fw_rad_max=3.33,
            f_p_alpha_plasma_deposited=0.95,
            p_alpha_total_mw=400.0,
            p_non_alpha_charged_mw=20.0,
            p_plasma_ohmic_mw=0.0,
            p_hcd_injected_total_mw=50.0,
        ),
        *fuzz_samples(
            _RADIATED_WALL_LOAD_FUZZ_BOUNDS,
            count=5,
            seed=0,
            fixed={"i_pflux_fw_neutron": 1, "ipowerflow": 1},
        ),
    ]


class TestRadiatedWallLoadAndFractionSimplePowerflow(
    TestRadiatedWallLoadAndFractionDirect
):
    """`i_pflux_fw_neutron == 0`, `ipowerflow == 0` branch."""

    samples = [
        legacy_sample(
            "simple-powerflow-branch",
            i_pflux_fw_neutron=0,
            ipowerflow=0,
            ffwal=0.9,
            p_plasma_rad_mw=350.0,
            a_plasma_surface=1925.3641313657533,
            fhole=0.0,
            a_fw_total=1918.87696696527,
            f_a_fw_outboard_hcd=0.0,
            f_ster_div_single=0.115,
            f_fw_rad_max=3.33,
            f_p_alpha_plasma_deposited=0.95,
            p_alpha_total_mw=400.0,
            p_non_alpha_charged_mw=20.0,
            p_plasma_ohmic_mw=0.0,
            p_hcd_injected_total_mw=50.0,
        ),
        *fuzz_samples(
            _RADIATED_WALL_LOAD_FUZZ_BOUNDS,
            count=5,
            seed=1,
            fixed={"i_pflux_fw_neutron": 0, "ipowerflow": 0},
        ),
    ]


class TestRadiatedWallLoadAndFractionDetailedPowerflow(
    TestRadiatedWallLoadAndFractionDirect
):
    """`i_pflux_fw_neutron == 0`, `ipowerflow != 0` branch."""

    samples = [
        legacy_sample(
            "detailed-powerflow-branch",
            i_pflux_fw_neutron=0,
            ipowerflow=1,
            ffwal=0.9,
            p_plasma_rad_mw=350.0,
            a_plasma_surface=1925.3641313657533,
            fhole=0.0,
            a_fw_total=2120.685245576686,
            f_a_fw_outboard_hcd=0.0,
            f_ster_div_single=0.021924555536480182,
            f_fw_rad_max=3.33,
            f_p_alpha_plasma_deposited=0.95,
            p_alpha_total_mw=400.0,
            p_non_alpha_charged_mw=20.0,
            p_plasma_ohmic_mw=0.0,
            p_hcd_injected_total_mw=50.0,
        ),
        *fuzz_samples(
            _RADIATED_WALL_LOAD_FUZZ_BOUNDS,
            count=5,
            seed=2,
            fixed={"i_pflux_fw_neutron": 0, "ipowerflow": 1},
        ),
    ]


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

    samples = [
        legacy_sample(
            "typical-helias",
            eden_plasma_electrons_thermal_vol_avg=2.34e5,
            eden_plasma_ions_thermal_vol_avg=2.0e5,
            e_plasma_electrons_thermal=3.28e8,
            e_plasma_ions_thermal=2.8e8,
        ),
        *fuzz_samples(
            {
                "eden_plasma_electrons_thermal_vol_avg": (1.0e4, 1.0e6),
                "eden_plasma_ions_thermal_vol_avg": (1.0e4, 1.0e6),
                "e_plasma_electrons_thermal": (1.0e6, 1.0e10),
                "e_plasma_ions_thermal": (1.0e6, 1.0e10),
            },
            count=5,
            seed=0,
        ),
    ]
