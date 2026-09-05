"""Harness cases for `neoclassics.py`'s twelve pure functions.

The two scalar-argument ones (`calculate_profile_values`,
`calculate_effective_thermal_diffusivity`) carry legacy points lifted from PROCESS's own
unit tests. The other ten are **fuzz-only**, deliberately: every one of them takes a
pipeline intermediate (`densities`, `temperatures`, `kt`, `vd`, `nu`, `d11_mono`, ...)
that PROCESS's unit tests never write down as an input literal -- they only ever appear
as `init_neoclassics`'s internal state -- so there is no literal point to lift, and
manufacturing one by running the pipeline would be a generated point wearing a legacy
label. Their `fuzz_bounds` are instead centred on the magnitudes that pipeline actually
produces at the helias5b point (recorded per contract below), which is what makes the
random draws land in the physical region rather than in an arbitrary one.

Reference adapters bind `self.data` for the PROCESS side; the `.neoclassics.*` fields
each method reads implicitly (`neoclassics.md`'s implicit-io rows) are exactly the
arguments the port takes explicitly, so writing the adapter is what tests that
classification rather than asserting it.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.stellarator.neoclassics import (
    calculate_collision_frequency,
    calculate_drift_velocity,
    calculate_effective_thermal_diffusivity,
    calculate_gamma_flux,
    calculate_integrated_radial_transport_coefficient,
    calculate_kt,
    calculate_monoenergetic_transport_coefficient,
    calculate_normalized_collision_frequency,
    calculate_normalized_collision_frequency_from_temperature,
    calculate_plateau_transport_coefficient,
    calculate_profile_values,
    calculate_q_flux,
)
from process.core.model import DataStructure
from process.models.stellarator.neoclassics import Neoclassics

_KEV = 1.602176634e-16
"""J per keV. Local, so a bound written in keV is readable next to a field held in J."""


def _neoclassics():
    n = Neoclassics()
    n.data = DataStructure()
    return n


def _process_quadrature():
    """PROCESS's own Gauss-Laguerre roots and weights.

    They are literals inside `init_neoclassics` rather than module constants, so the only
    way to get PROCESS's copy is to run that method once. Taken from PROCESS and never
    from the port's `ROOTS`/`WEIGHTS`: a transcription slip in the port's copy has to show
    up as a value mismatch, not cancel out on both sides. The point it is run at is
    irrelevant -- only the two literal arrays are kept.
    """
    n = _neoclassics()
    p = n.data.physics
    p.temp_plasma_electron_on_axis_kev = 13.2418
    p.temp_plasma_ion_on_axis_kev = 12.57971
    p.alphat = 1.2
    p.alphan = 0.35
    p.nd_plasma_electron_on_axis = 2.795661e20
    p.f_plasma_fuel_deuterium = 0.5
    p.nd_plasma_ions_on_axis = 2.393085816e20
    p.nd_plasma_alphas_thermal_vol_avg = 2.9820384e19
    p.rminor = 1.7993820274145451
    p.rmajor = 22.16
    p.b_plasma_toroidal_on_axis = 5.24
    p.temp_plasma_electron_vol_avg_kev = 6.019
    p.temp_plasma_ion_vol_avg_kev = 5.71805
    p.nd_plasma_electrons_vol_avg = 2.07086e20
    p.nd_plasma_fuel_ions_vol_avg = 1.47415411616e20
    n.init_neoclassics(0.6, 0.01464553, 0.9)
    return (
        np.asarray(n.data.neoclassics.roots, dtype=float),
        np.asarray(n.data.neoclassics.weights, dtype=float),
    )


_ROOTS, _WEIGHTS = _process_quadrature()
_NO_ROOTS = _ROOTS.size


def _with_quadrature():
    """A `Neoclassics` whose quadrature is set, which every grid method needs."""
    n = _neoclassics()
    n.data.neoclassics.roots = _ROOTS
    n.data.neoclassics.weights = _WEIGHTS
    return n


def _shaped(low, high, shape):
    """Bounds of `shape`, both ends the same for every component."""
    return (np.full(shape, low), np.full(shape, high))


# Species order is (electron, deuterium, tritium, alpha) everywhere in this file.
_SPECIES = (4,)
_GRID = (4, _NO_ROOTS)


def _reference_profile_values(
    rho,
    temp_plasma_electron_on_axis_kev,
    temp_plasma_ion_on_axis_kev,
    alphat,
    nd_plasma_electron_on_axis,
    f_plasma_fuel_deuterium,
    nd_plasma_ions_on_axis,
    nd_plasma_alphas_thermal_vol_avg,
    alphan,
    rminor,
):
    """Call PROCESS's `Neoclassics.init_profile_values_from_PROCESS`."""
    n = _neoclassics()
    p = n.data.physics
    p.temp_plasma_electron_on_axis_kev = temp_plasma_electron_on_axis_kev
    p.temp_plasma_ion_on_axis_kev = temp_plasma_ion_on_axis_kev
    p.alphat = alphat
    p.nd_plasma_electron_on_axis = nd_plasma_electron_on_axis
    p.f_plasma_fuel_deuterium = f_plasma_fuel_deuterium
    p.nd_plasma_ions_on_axis = nd_plasma_ions_on_axis
    p.nd_plasma_alphas_thermal_vol_avg = nd_plasma_alphas_thermal_vol_avg
    p.alphan = alphan
    p.rminor = rminor

    return n.init_profile_values_from_PROCESS(rho)


def _reference_effective_thermal_diffusivity(
    vol_plasma,
    f_st_rmajor,
    radius_plasma_core_norm,
    rminor,
    stella_config_rminor_ref,
    a_plasma_surface,
    f_p_alpha_plasma_deposited,
    pden_alpha_total_mw,
    pden_plasma_core_rad_mw,
    nd_plasma_electron_on_axis,
    temp_plasma_electron_on_axis_kev,
    alphat,
    alphan,
):
    """Call PROCESS's `Neoclassics.st_calc_eff_chi`."""
    n = _neoclassics()
    p = n.data.physics
    p.vol_plasma = vol_plasma
    p.a_plasma_surface = a_plasma_surface
    p.rminor = rminor
    p.f_p_alpha_plasma_deposited = f_p_alpha_plasma_deposited
    p.pden_alpha_total_mw = pden_alpha_total_mw
    p.pden_plasma_core_rad_mw = pden_plasma_core_rad_mw
    p.nd_plasma_electron_on_axis = nd_plasma_electron_on_axis
    p.temp_plasma_electron_on_axis_kev = temp_plasma_electron_on_axis_kev
    p.alphat = alphat
    p.alphan = alphan
    n.data.stellarator.f_st_rmajor = f_st_rmajor
    n.data.impurity_radiation.radius_plasma_core_norm = radius_plasma_core_norm
    n.data.stellarator_config.stella_config_rminor_ref = stella_config_rminor_ref

    return n.st_calc_eff_chi()


def _reference_kt(temperatures):
    """Call PROCESS's `Neoclassics.neoclassics_calc_KT`."""
    n = _with_quadrature()
    n.data.neoclassics.temperatures = np.asarray(temperatures)
    return n.neoclassics_calc_KT()


def _reference_collision_frequency(densities, temperatures):
    """Call PROCESS's `Neoclassics.neoclassics_calc_nu`."""
    n = _with_quadrature()
    n.data.neoclassics.densities = np.asarray(densities)
    n.data.neoclassics.temperatures = np.asarray(temperatures)
    return n.neoclassics_calc_nu()


def _reference_normalized_collision_frequency(temperatures, nu, iota, rmajor):
    """Call PROCESS's `Neoclassics.neoclassics_calc_nu_star`."""
    n = _with_quadrature()
    n.data.neoclassics.temperatures = np.asarray(temperatures)
    n.data.neoclassics.nu = np.asarray(nu)
    n.data.neoclassics.iota = iota
    n.data.physics.rmajor = rmajor
    return n.neoclassics_calc_nu_star()


def _reference_normalized_collision_frequency_from_temperature(
    iota,
    temp_plasma_electron_vol_avg_kev,
    temp_plasma_ion_vol_avg_kev,
    nd_plasma_electrons_vol_avg,
    nd_plasma_fuel_ions_vol_avg,
    f_plasma_fuel_deuterium,
    nd_plasma_alphas_thermal_vol_avg,
    rmajor,
):
    """Call PROCESS's `Neoclassics.neoclassics_calc_nu_star_fromT`.

    `iota` here is the method's own argument, *not* `.neoclassics.iota` -- the two are
    different values under one name in the source, see `neoclassics.md`'s open question 2.
    """
    n = _neoclassics()
    p = n.data.physics
    p.temp_plasma_electron_vol_avg_kev = temp_plasma_electron_vol_avg_kev
    p.temp_plasma_ion_vol_avg_kev = temp_plasma_ion_vol_avg_kev
    p.nd_plasma_electrons_vol_avg = nd_plasma_electrons_vol_avg
    p.nd_plasma_fuel_ions_vol_avg = nd_plasma_fuel_ions_vol_avg
    p.f_plasma_fuel_deuterium = f_plasma_fuel_deuterium
    p.nd_plasma_alphas_thermal_vol_avg = nd_plasma_alphas_thermal_vol_avg
    p.rmajor = rmajor
    return n.neoclassics_calc_nu_star_fromT(iota)


def _reference_drift_velocity(temperatures, rmajor, b_plasma_toroidal_on_axis):
    """Call PROCESS's `Neoclassics.neoclassics_calc_vd`."""
    n = _with_quadrature()
    n.data.neoclassics.temperatures = np.asarray(temperatures)
    n.data.physics.rmajor = rmajor
    n.data.physics.b_plasma_toroidal_on_axis = b_plasma_toroidal_on_axis
    return n.neoclassics_calc_vd()


def _reference_plateau_transport_coefficient(kt, vd, rmajor, iota):
    """Call PROCESS's `Neoclassics.neoclassics_calc_D11_plateau`."""
    n = _with_quadrature()
    n.data.neoclassics.kt = np.asarray(kt)
    n.data.neoclassics.vd = np.asarray(vd)
    n.data.physics.rmajor = rmajor
    n.data.neoclassics.iota = iota
    return n.neoclassics_calc_D11_plateau()


def _reference_monoenergetic_transport_coefficient(eps_eff, vd, nu):
    """Call PROCESS's `Neoclassics.neoclassics_calc_d11_mono`."""
    n = _with_quadrature()
    n.data.neoclassics.vd = np.asarray(vd)
    n.data.neoclassics.nu = np.asarray(nu)
    return n.neoclassics_calc_d11_mono(eps_eff)


def _reference_integrated_radial_transport_coefficient(d11_mono, index):
    """Call PROCESS's `Neoclassics.calc_integrated_radial_transport_coeffs`."""
    n = _with_quadrature()
    n.data.neoclassics.d11_mono = np.asarray(d11_mono)
    return n.calc_integrated_radial_transport_coeffs(index=index)


def _reference_gamma_flux(
    densities, temperatures, dr_densities, dr_temperatures, d111, d112, er
):
    """Call PROCESS's `Neoclassics.neoclassics_calc_gamma_flux`."""
    n = _neoclassics()
    n.data.neoclassics.d111 = np.asarray(d111)
    n.data.neoclassics.d112 = np.asarray(d112)
    n.data.neoclassics.er = er
    return n.neoclassics_calc_gamma_flux(
        np.asarray(densities),
        np.asarray(temperatures),
        np.asarray(dr_densities),
        np.asarray(dr_temperatures),
    )


def _reference_q_flux(
    densities, temperatures, dr_densities, dr_temperatures, d112, d113, er
):
    """Call PROCESS's `Neoclassics.neoclassics_calc_q_flux`.

    Unlike `gamma_flux`, this one takes no arguments at all in the source -- every one of
    the port's seven is an implicit `.neoclassics.*` read.
    """
    n = _neoclassics()
    nc = n.data.neoclassics
    nc.densities = np.asarray(densities)
    nc.temperatures = np.asarray(temperatures)
    nc.dr_densities = np.asarray(dr_densities)
    nc.dr_temperatures = np.asarray(dr_temperatures)
    nc.d112 = np.asarray(d112)
    nc.d113 = np.asarray(d113)
    nc.er = er
    return n.neoclassics_calc_q_flux()


class TestProfileValues(Tier1Contract):
    """`init_profile_values_from_PROCESS` -> `calculate_profile_values`."""

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_profile_values
    ported = calculate_profile_values

    # tests/unit/models/stellarator/test_neoclassics.py::test_init_neoclassics,
    # generated from helias5b.IN.DAT. `rho` is the `r_effin` field there (0.6) -- not
    # the separately-monkeypatched `r_eff` field, which that test leaves stale.
    samples = [
        legacy_sample(
            "init-profile-values-helias5b",
            rho=0.59999999999999998,
            temp_plasma_electron_on_axis_kev=13.241800000000001,
            temp_plasma_ion_on_axis_kev=12.579710000000002,
            alphat=1.2,
            nd_plasma_electron_on_axis=2.7956610000000002e20,
            f_plasma_fuel_deuterium=0.5,
            nd_plasma_ions_on_axis=2.3930858160000005e20,
            nd_plasma_alphas_thermal_vol_avg=2.9820384000000004e19,
            alphan=0.35000000000000003,
            rminor=1.7993820274145451,
        ),
    ]

    fuzz_bounds = {
        "rho": (0.0, 0.99),
        "temp_plasma_electron_on_axis_kev": (1.0, 50.0),
        "temp_plasma_ion_on_axis_kev": (1.0, 50.0),
        "alphat": (0.1, 3.0),
        "nd_plasma_electron_on_axis": (1.0e19, 1.0e21),
        "f_plasma_fuel_deuterium": (0.1, 0.9),
        "nd_plasma_ions_on_axis": (1.0e19, 1.0e21),
        "nd_plasma_alphas_thermal_vol_avg": (1.0e17, 1.0e20),
        "alphan": (0.1, 3.0),
        "rminor": (0.5, 5.0),
    }


class TestEffectiveThermalDiffusivity(Tier1Contract):
    """`st_calc_eff_chi` -> `calculate_effective_thermal_diffusivity`."""

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_effective_thermal_diffusivity
    ported = calculate_effective_thermal_diffusivity

    # tests/unit/models/stellarator/test_stellarator.py::test_st_calc_eff_chi, generated
    # from stellarator_helias.IN.DAT (two operating points).
    samples = [
        legacy_sample(
            "st-calc-eff-chi-helias-0",
            temp_plasma_electron_on_axis_kev=19.108573496973477,
            nd_plasma_electron_on_axis=3.4479000000000007e20,
            f_p_alpha_plasma_deposited=0.95000000000000007,
            pden_alpha_total_mw=1.2629524018077414,
            pden_plasma_core_rad_mw=0.10762698429338043,
            alphan=0.35000000000000003,
            alphat=1.2,
            vol_plasma=1385.8142655379029,
            a_plasma_surface=1926.0551116585129,
            rminor=1.7863900994187722,
            radius_plasma_core_norm=0.60000000000000009,
            stella_config_rminor_ref=1.80206932,
            f_st_rmajor=0.99129932482229,
        ),
        legacy_sample(
            "st-calc-eff-chi-helias-1",
            temp_plasma_electron_on_axis_kev=17.5,
            nd_plasma_electron_on_axis=3.4479000000000007e20,
            f_p_alpha_plasma_deposited=0.95000000000000007,
            pden_alpha_total_mw=1.0570658694225301,
            pden_plasma_core_rad_mw=0.1002475669217598,
            alphan=0.35000000000000003,
            alphat=1.2,
            vol_plasma=1385.8142655379029,
            a_plasma_surface=1926.0551116585129,
            rminor=1.7863900994187722,
            radius_plasma_core_norm=0.60000000000000009,
            stella_config_rminor_ref=1.80206932,
            f_st_rmajor=0.99129932482229,
        ),
    ]

    fuzz_bounds = {
        "vol_plasma": (100.0, 5000.0),
        "f_st_rmajor": (0.5, 1.5),
        "radius_plasma_core_norm": (0.1, 0.9),
        "rminor": (0.5, 5.0),
        "stella_config_rminor_ref": (0.5, 5.0),
        "a_plasma_surface": (100.0, 5000.0),
        "f_p_alpha_plasma_deposited": (0.5, 1.0),
        "pden_alpha_total_mw": (0.1, 5.0),
        "pden_plasma_core_rad_mw": (0.01, 1.0),
        "nd_plasma_electron_on_axis": (1.0e19, 1.0e21),
        "temp_plasma_electron_on_axis_kev": (1.0, 50.0),
        "alphat": (0.1, 3.0),
        "alphan": (0.1, 3.0),
    }


class TestKt(Tier1Contract):
    """`neoclassics_calc_KT` -> `calculate_kt`.

    Species temperatures at the helias5b point are all within a few percent of
    1.2e-15 J (7.5 keV); the bound spans 1-30 keV around that.
    """

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_kt
    ported = calculate_kt

    fuzz_bounds = {"temperatures": _shaped(1.0 * _KEV, 30.0 * _KEV, _SPECIES)}


class TestCollisionFrequency(Tier1Contract):
    """`neoclassics_calc_nu` -> `calculate_collision_frequency`.

    Densities are bounded per species: the alpha population is two decades below the
    electron and fuel-ion ones at the helias5b point (3e19 against 2e20), which a single
    shared bound would flatten.
    """

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_collision_frequency
    ported = calculate_collision_frequency

    fuzz_bounds = {
        "densities": (
            np.array([1.0e18, 1.0e18, 1.0e18, 1.0e16]),
            np.array([1.0e21, 1.0e21, 1.0e21, 1.0e20]),
        ),
        "temperatures": _shaped(1.0 * _KEV, 30.0 * _KEV, _SPECIES),
    }


class TestNormalizedCollisionFrequency(Tier1Contract):
    """`neoclassics_calc_nu_star` -> `calculate_normalized_collision_frequency`.

    `nu` spans 0.5 to 4e6 over the grid at the helias5b point (it falls off steeply with
    energy), hence a log-uniform bound over eight decades rather than a narrow one.
    """

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_normalized_collision_frequency
    ported = calculate_normalized_collision_frequency

    fuzz_bounds = {
        "temperatures": _shaped(1.0 * _KEV, 30.0 * _KEV, _SPECIES),
        "nu": _shaped(1.0e-1, 1.0e7, _GRID),
        "iota": (0.5, 1.5),
        "rmajor": (5.0, 30.0),
    }


class TestNormalizedCollisionFrequencyFromTemperature(Tier1Contract):
    """`neoclassics_calc_nu_star_fromT` -> the same, from volume-averaged T/n.

    All-scalar, unlike the rest of this group -- it is here rather than with the two
    legacy-sampled contracts because PROCESS's unit test for it exercises it only through
    `init_neoclassics`, with no separately recorded input/output pair to lift.
    """

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_normalized_collision_frequency_from_temperature
    ported = calculate_normalized_collision_frequency_from_temperature

    fuzz_bounds = {
        "iota": (0.5, 1.5),
        "temp_plasma_electron_vol_avg_kev": (1.0, 30.0),
        "temp_plasma_ion_vol_avg_kev": (1.0, 30.0),
        "nd_plasma_electrons_vol_avg": (1.0e18, 1.0e21),
        "nd_plasma_fuel_ions_vol_avg": (1.0e18, 1.0e21),
        "f_plasma_fuel_deuterium": (0.1, 0.9),
        "nd_plasma_alphas_thermal_vol_avg": (1.0e16, 1.0e20),
        "rmajor": (5.0, 30.0),
    }


class TestDriftVelocity(Tier1Contract):
    """`neoclassics_calc_vd` -> `calculate_drift_velocity`."""

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_drift_velocity
    ported = calculate_drift_velocity

    fuzz_bounds = {
        "temperatures": _shaped(1.0 * _KEV, 30.0 * _KEV, _SPECIES),
        "rmajor": (5.0, 30.0),
        "b_plasma_toroidal_on_axis": (2.0, 12.0),
    }


class TestPlateauTransportCoefficient(Tier1Contract):
    """`neoclassics_calc_D11_plateau` -> `calculate_plateau_transport_coefficient`.

    `kt` carries PROCESS's own `roots / KEV` scaling (0.3 to 800 over the grid at the
    helias5b point, not an energy in J); `vd` runs 1 to 7e3. Both are bounded a decade
    wide of that.
    """

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_plateau_transport_coefficient
    ported = calculate_plateau_transport_coefficient

    fuzz_bounds = {
        "kt": _shaped(1.0e-2, 1.0e4, _GRID),
        "vd": _shaped(1.0e-1, 1.0e5, _GRID),
        "rmajor": (5.0, 30.0),
        "iota": (0.5, 1.5),
    }


class TestMonoenergeticTransportCoefficient(Tier1Contract):
    """`neoclassics_calc_d11_mono` -> `calculate_monoenergetic_transport_coefficient`."""

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_monoenergetic_transport_coefficient
    ported = calculate_monoenergetic_transport_coefficient

    fuzz_bounds = {
        "eps_eff": (1.0e-4, 1.0e-1),
        "vd": _shaped(1.0e-1, 1.0e5, _GRID),
        "nu": _shaped(1.0e-1, 1.0e7, _GRID),
    }


class _IntegratedRadialTransportCoefficient(Tier1Contract):
    """`calc_integrated_radial_transport_coeffs` -> the port, one moment at a time.

    `index` is static (which of `d111`/`d112`/`d113` you get), so it is `fuzz_fixed` and
    excluded from differentiation -- and it is why this is three contracts rather than
    one: the source calls the same function three times, and a contract that fuzzed
    `index` would be differentiating a switch.

    Not collected itself (no `Test` prefix); the three finals below are.
    """

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_integrated_radial_transport_coefficient
    ported = calculate_integrated_radial_transport_coefficient
    static_argnames = ("index",)

    fuzz_bounds = {"d11_mono": _shaped(1.0e-10, 1.0e6, _GRID)}


class TestIntegratedRadialTransportCoefficientD111(
    _IntegratedRadialTransportCoefficient
):
    """The `index=1` moment, `.neoclassics.d111`."""

    fuzz_fixed = {"index": 1}


class TestIntegratedRadialTransportCoefficientD112(
    _IntegratedRadialTransportCoefficient
):
    """The `index=2` moment, `.neoclassics.d112`."""

    fuzz_fixed = {"index": 2}


class TestIntegratedRadialTransportCoefficientD113(
    _IntegratedRadialTransportCoefficient
):
    """The `index=3` moment, `.neoclassics.d113`."""

    fuzz_fixed = {"index": 3}


class TestGammaFlux(Tier1Contract):
    """`neoclassics_calc_gamma_flux` -> `calculate_gamma_flux`.

    `er` is bounded away from zero on both sides: PROCESS's relative perturbation
    degenerates at exactly zero (`ZeroPerturbationError`), and `.neoclassics.er`'s default
    *is* zero, so a bound straddling it would silently skip the one argument whose
    producer is unresolved (`neoclassics.md`, open question 1).
    """

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_gamma_flux
    ported = calculate_gamma_flux

    fuzz_bounds = {
        "densities": (
            np.array([1.0e18, 1.0e18, 1.0e18, 1.0e16]),
            np.array([1.0e21, 1.0e21, 1.0e21, 1.0e20]),
        ),
        "temperatures": _shaped(1.0 * _KEV, 30.0 * _KEV, _SPECIES),
        "dr_densities": _shaped(-1.0e21, -1.0e18, _SPECIES),
        "dr_temperatures": _shaped(-5.0e-15, -1.0e-16, _SPECIES),
        "d111": _shaped(1.0e-4, 1.0e2, _SPECIES),
        "d112": _shaped(1.0e-4, 1.0e2, _SPECIES),
        "er": (1.0e2, 1.0e4),
    }


class TestQFlux(Tier1Contract):
    """`neoclassics_calc_q_flux` -> `calculate_q_flux`. Same shape as `TestGammaFlux`."""

    audit_record = "models/stellarator/neoclassics.md"
    reference = _reference_q_flux
    ported = calculate_q_flux

    fuzz_bounds = {
        "densities": (
            np.array([1.0e18, 1.0e18, 1.0e18, 1.0e16]),
            np.array([1.0e21, 1.0e21, 1.0e21, 1.0e20]),
        ),
        "temperatures": _shaped(1.0 * _KEV, 30.0 * _KEV, _SPECIES),
        "dr_densities": _shaped(-1.0e21, -1.0e18, _SPECIES),
        "dr_temperatures": _shaped(-5.0e-15, -1.0e-16, _SPECIES),
        "d112": _shaped(1.0e-4, 1.0e2, _SPECIES),
        "d113": _shaped(1.0e-4, 1.0e2, _SPECIES),
        "er": (1.0e2, 1.0e4),
    }
