"""Harness cases for the ported tokamak arm of `process/models/physics/physics.py`.

Audit record: `functional_process/_audit/units/models/physics/physics.md`.

Three of the eight ported functions have a PROCESS `@staticmethod`/method to diff
against directly (`PlasmaFields.calculate_surface_averaged_poloidal_field`,
`PlasmaExhaust.calculate_separatrix_power`,
`PlasmaBeta.calculate_plasma_energy_from_beta`). The other five are arithmetic PROCESS
writes inline inside `Physics.run`, so their reference is transcribed from the source
here in `numpy`, with the source lines named -- the same convention
`test_plasma_physics.py`'s `_reference_clipped_radiation_powers` established for
`st_phys`'s inline blocks.

**Sample provenance.** Two of the units have real legacy points lifted from
`tests/unit/models/physics/test_physics.py` (marked in each case). The rest have no
PROCESS unit test at all, so their `legacy_sample`s are hand-built at
`large_tokamak_eval`-scale operating values and the real coverage is the fuzz draws --
recorded as this unit's weak point in the audit record's "tier signal", not papered
over.
"""

import functools

import numpy as np

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.physics.physics import (
    calculate_beta_limit_from_norm,
    calculate_beta_norm_max_wesson,
    calculate_continuous_plant_ramp_times,
    calculate_coulomb_logarithm_ion_electron,
    calculate_pflux_plasma_surface_neutron_avg_mw,
    calculate_plasma_energy_from_beta,
    calculate_pulsed_plant_ramp_times,
    calculate_separatrix_power,
    calculate_surface_averaged_poloidal_field_amperes,
    calculate_thermal_beta,
    calculate_toroidal_beta,
    calculate_total_radiation_power,
    calculate_unclipped_radiation_powers,
    force_positive_separatrix_power,
    plasma_ohmic_heating,
)
from process.models.physics.exhaust import PlasmaExhaust
from process.models.physics.physics import Physics, PlasmaBeta
from process.models.physics.plasma_fields import PlasmaFields

_FIELDS = PlasmaFields()

# `PlasmaCurrentModel.IPDG89_SCALING`, `large_tokamak_eval.IN.DAT`'s value. Any value
# other than `2` (`PENG_DIVERTOR_SCALING`) selects the same arm -- see the port's
# `SurfaceAveragedPoloidalFieldAmperes` docstring.
_I_PLASMA_CURRENT_AMPERES = 4


_reference_surface_averaged_poloidal_field_amperes = functools.partial(
    _FIELDS.calculate_surface_averaged_poloidal_field,
    i_plasma_current=_I_PLASMA_CURRENT_AMPERES,
    q95=3.5,
    aspect=3.0,
    b_plasma_toroidal_on_axis=5.7,
    kappa=1.85,
    triang=0.5,
)


class TestSurfaceAveragedPoloidalFieldAmperes(Tier1Contract):
    """`<Bp(a)> = mu_0 * Ip / L_pol`, `plasma_fields.py:83-84`.

    Both legacy points are lifted verbatim from
    `tests/unit/models/physics/test_physics.py::test_calculate_surface_averaged_poloidal_field`
    -- its `i_plasma_current = 3` and `i_plasma_current = 4` rows, the two of its four
    parametrisations that take this arm.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_surface_averaged_poloidal_field_amperes
    ported = calculate_surface_averaged_poloidal_field_amperes

    samples = FROM_FILE


def _reference_unclipped_radiation_powers(
    pden_plasma_core_rad_mw_unclipped,
    pden_plasma_outer_rad_mw_unclipped,
    vol_plasma,
):
    """`physics.py:751-752` and `:758-763`, transcribed from source.

    The point of the transcription is what is *absent*: `stellarator.py:2153-2158`
    wraps both densities in `max(..., 0.0)` before forming the products and this call
    site does not. The `negative-core-density` sample below is the point where the two
    references disagree, and it is the only reason this function is a node of its own.
    """
    core = pden_plasma_core_rad_mw_unclipped
    outer = pden_plasma_outer_rad_mw_unclipped
    return core, outer, core * vol_plasma, outer * vol_plasma


class TestUnclippedRadiationPowers(Tier1Contract):
    """The tokamak's radiation-density assignments: no clip at zero."""

    audit_record = "models/physics/physics.md"
    reference = _reference_unclipped_radiation_powers
    ported = calculate_unclipped_radiation_powers

    samples = FROM_FILE


def _reference_total_radiation_power(pden_plasma_rad_mw, vol_plasma):
    """`physics.py:764-766`, transcribed from source."""
    return pden_plasma_rad_mw * vol_plasma


class TestTotalRadiationPower(Tier1Contract):
    """`P_rad = pden_rad * V`."""

    audit_record = "models/physics/physics.md"
    reference = _reference_total_radiation_power
    ported = calculate_total_radiation_power

    samples = FROM_FILE


class TestSeparatrixPower(Tier1Contract):
    """`PlasmaExhaust.calculate_separatrix_power`, `exhaust.py:88-127`, unchanged.

    The reference is PROCESS's own `@staticmethod`, called directly -- it takes no
    `self.data` access, so no adapter is needed. `p_hcd_injected_total_mw` is a live
    argument here rather than the `0.0` the ignited arm passes, which is the whole
    content of the `i_plasma_ignited` split this node's occupant declares.
    """

    audit_record = "models/physics/physics.md"
    reference = staticmethod(PlasmaExhaust.calculate_separatrix_power)
    ported = calculate_separatrix_power

    samples = FROM_FILE


def _reference_force_positive_separatrix_power(p_plasma_separatrix_mw_raw):
    """`physics.py:839-845`, transcribed from source."""
    return p_plasma_separatrix_mw_raw / (1 - np.exp(-p_plasma_separatrix_mw_raw))


class TestForcePositiveSeparatrixPower(Tier1Contract):
    """PROCESS's own "KLUDGE" positivity transform, `physics.py:839-845`.

    Three samples, one per regime: far above the transform's scale (where it is the
    identity to machine precision), inside it, and below zero -- the case the transform
    exists for, where a negative separatrix power is mapped to a small positive one.
    `x == 0` is *not* sampled: PROCESS evaluates `0.0/0.0` there and returns `nan`, and
    the port reproduces that rather than inventing the limit (which is `1.0`).
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_force_positive_separatrix_power
    ported = force_positive_separatrix_power

    samples = FROM_FILE


def _reference_pulsed_plant_ramp_times(plasma_current):
    """`physics.py:476-483`, the `i_pulsed_plant == 1, pulsetimings == 0` arm."""
    t_plant_pulse_plasma_current_ramp_up = plasma_current / 1.0e5
    return (
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_up,
    )


class TestPulsedPlantRampTimes(Tier1Contract):
    """The one arm of `pulsetimings` this port supports.

    `pulsetimings` is read at `physics.py:476` and nowhere else in all of
    `process/models/**`, so this two-line function is the whole of that decision's
    computation.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_pulsed_plant_ramp_times
    ported = calculate_pulsed_plant_ramp_times

    samples = FROM_FILE


def _reference_continuous_plant_ramp_times(plasma_current):
    """`physics.py:465-474`, the `i_pulsed_plant != 1, i_t_current_ramp_up == 0` arm."""
    t_plant_pulse_plasma_current_ramp_up = plasma_current / 5.0e5
    return (
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_plasma_current_ramp_up,
    )


class TestContinuousPlantRampTimes(Tier1Contract):
    """The continuous-plant arm of the ramp-time family, `physics.py:465-474`.

    Like `TestPulsedPlantRampTimes` the writes are inline in `Physics.run` with no
    PROCESS `@staticmethod` to call, so the reference is transcribed from the source.
    Three outputs where the pulsed arm has two: this arm also owns
    `.times.t_plant_pulse_coil_precharge`. The legacy point is the Sauter bootstrap
    fixture's plasma current (`tests/unit/models/physics/test_physics.py:392`), a
    validated PROCESS unit-test input.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_continuous_plant_ramp_times
    ported = calculate_continuous_plant_ramp_times

    samples = FROM_FILE


class TestPlasmaEnergyFromBeta(Tier1Contract):
    """`PlasmaBeta.calculate_plasma_energy_from_beta`, `physics.py:4153-4176`.

    The legacy point is `tests/unit/models/physics/test_physics.py::
    test_calculate_plasma_energy_from_beta`'s, verbatim.
    """

    audit_record = "models/physics/physics.md"
    reference = staticmethod(PlasmaBeta.calculate_plasma_energy_from_beta)
    ported = calculate_plasma_energy_from_beta

    samples = FROM_FILE


def _reference_plasma_ohmic_heating(
    f_c_plasma_inductive,
    kappa95,
    plasma_current,
    rmajor,
    rminor,
    temp_plasma_electron_density_weighted_kev,
    vol_plasma,
    n_charge_plasma_effective_vol_avg,
    plasma_res_factor,
):
    """`Physics.plasma_ohmic_heating` through the port's signature. `zeff` is the
    staticmethod's spelling of `.physics.n_charge_plasma_effective_vol_avg`
    (`Physics.run:782`); `aspect` is read only by the negative-resistance
    `logger.error` and is passed as PROCESS's own definition, `rmajor / rminor`.
    """
    return Physics.plasma_ohmic_heating(
        f_c_plasma_inductive=f_c_plasma_inductive,
        kappa95=kappa95,
        plasma_current=plasma_current,
        rmajor=rmajor,
        rminor=rminor,
        temp_plasma_electron_density_weighted_kev=(
            temp_plasma_electron_density_weighted_kev
        ),
        vol_plasma=vol_plasma,
        zeff=n_charge_plasma_effective_vol_avg,
        plasma_res_factor=plasma_res_factor,
        aspect=rmajor / rminor,
    )


def _ported_plasma_ohmic_heating(
    f_c_plasma_inductive,
    kappa95,
    plasma_current,
    rmajor,
    rminor,
    temp_plasma_electron_density_weighted_kev,
    vol_plasma,
    n_charge_plasma_effective_vol_avg,
    plasma_res_factor,
):
    return plasma_ohmic_heating(
        f_c_plasma_inductive=f_c_plasma_inductive,
        kappa95=kappa95,
        plasma_current=plasma_current,
        rmajor=rmajor,
        rminor=rminor,
        temp_plasma_electron_density_weighted_kev=(
            temp_plasma_electron_density_weighted_kev
        ),
        vol_plasma=vol_plasma,
        zeff=n_charge_plasma_effective_vol_avg,
        plasma_res_factor=plasma_res_factor,
    )


class TestPlasmaOhmicHeating(Tier1Contract):
    """`plasma_ohmic_heating` -> `Physics.plasma_ohmic_heating`
    (`physics.py:1605-1697`), added 2026-08-27 (`cold_boundary.md` producer 3).

    The legacy point is `large_tokamak_eval` at convergence, read off a live
    `SingleRun` -- where PROCESS's chained-comparison defect (`2.5 >= A <= 4.0`, i.e.
    `A <= 2.5`) takes the *enhancement* arm at `A = 3.0` and lands on
    `f_res_plasma_neo = 2.5`, `res_plasma = 4.0496e-9` (the converged value
    `cold_boundary.md` records). The fuzz draws straddle the defect's real kink at
    `A = 2.5` from both sides (`rmajor/rminor` spans ~1.1 to 40 over the bounds), so
    both arms of the reproduced `jnp.where` are exercised against PROCESS.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_plasma_ohmic_heating
    ported = _ported_plasma_ohmic_heating

    samples = FROM_FILE

    fuzz = True


# =========================================================== `PlasmaBeta.run`'s limits
#
# The constraint-24 trio and the normalised limit that feeds it, added 2026-08-27 for
# `optimise_design.md` §11.5. Two of the four have a PROCESS `@staticmethod` to diff
# against; the other two are inline assignments in `PlasmaBeta.run` and are transcribed
# here in `numpy` with their source lines named, the same convention
# `_reference_unclipped_radiation_powers` above uses.


def _reference_toroidal_beta(
    beta_total_vol_avg, b_plasma_total, b_plasma_toroidal_on_axis
):
    """`physics.py:3818-3822`, transcribed."""
    return beta_total_vol_avg * b_plasma_total**2 / b_plasma_toroidal_on_axis**2


def _reference_thermal_beta(beta_total_vol_avg, beta_fast_alpha, beta_beam):
    """`physics.py:3831-3835`, transcribed."""
    return beta_total_vol_avg - beta_fast_alpha - beta_beam


def _reference_coulomb_logarithm_ion_electron(
    nd_plasma_electrons_vol_avg, temp_plasma_electron_vol_avg_kev
):
    """`physics.py:279-283`, transcribed."""
    return (
        31.3
        - (np.log(nd_plasma_electrons_vol_avg) / 2.0)
        + np.log(temp_plasma_electron_vol_avg_kev * 1000.0)
    )


def _reference_pflux_plasma_surface_neutron_avg_mw(p_neutron_total_mw, a_plasma_surface):
    """`physics.py:835-837`, transcribed."""
    return p_neutron_total_mw / a_plasma_surface


class TestBetaNormMaxWesson(Tier1Contract):
    """`beta_N_max = 4 * l_i`, `physics.py:3941-3974`.

    The legacy point is `large_tokamak_eval` at convergence
    (`ind_plasma_internal_norm = 1.2568...`, giving PROCESS's own
    `beta_norm_max = 5.0273`). The function is linear, so the value and gradient
    checks are exact by construction and what they actually pin is the *binding* --
    that the port reads `.physics.ind_plasma_internal_norm` and not one of the four
    sibling scalings `get_beta_norm_max_value` selects among.
    """

    audit_record = "models/physics/physics.md"
    reference = staticmethod(PlasmaBeta.calculate_beta_norm_max_wesson)
    ported = calculate_beta_norm_max_wesson

    samples = FROM_FILE

    fuzz_bounds = {"ind_plasma_internal_norm": (0.5, 2.0)}


class TestBetaLimitFromNorm(Tier1Contract):
    """Constraint 24's bound, `physics.py:4180-4235`.

    Legacy point: `large_tokamak_eval` at convergence, where the four inputs give
    PROCESS's own `beta_vol_avg_max = 0.05703975985`.
    """

    audit_record = "models/physics/physics.md"
    reference = staticmethod(PlasmaBeta.calculate_beta_limit_from_norm)
    ported = calculate_beta_limit_from_norm

    samples = FROM_FILE

    fuzz_bounds = {
        "b_plasma_toroidal_on_axis": (1.0, 15.0),
        "beta_norm_max": (1.0, 8.0),
        "plasma_current": (1.0e6, 3.0e7),
        "rminor": (0.5, 5.0),
    }


class TestToroidalBeta(Tier1Contract):
    """`beta_tor = beta * B_tot^2 / B_tor^2`, `physics.py:3818-3822`."""

    audit_record = "models/physics/physics.md"
    reference = _reference_toroidal_beta
    ported = calculate_toroidal_beta

    samples = FROM_FILE

    fuzz_bounds = {
        "beta_total_vol_avg": (0.005, 0.15),
        "b_plasma_total": (1.0, 15.0),
        "b_plasma_toroidal_on_axis": (1.0, 15.0),
    }


class TestThermalBeta(Tier1Contract):
    """`beta_th = beta - beta_alpha - beta_beam`, `physics.py:3831-3835`.

    The legacy point has `beta_beam = 0`, which is `large_tokamak_eval`'s value and not
    an accident: there is no neutral beam, so `beam_fusion` never writes it. The fuzz
    draws give it a nonzero range anyway, because the port declares the read and a
    beam-heated file would exercise it.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_thermal_beta
    ported = calculate_thermal_beta

    samples = FROM_FILE

    fuzz_bounds = {
        "beta_total_vol_avg": (0.005, 0.15),
        "beta_fast_alpha": (0.0, 0.02),
        "beta_beam": (0.0, 0.02),
    }


class TestCoulombLogarithmIonElectron(Tier1Contract):
    """`ln(Lambda)_ie = 31.3 - ln(n_e)/2 + ln(T_e[eV])`, `physics.py:279-283`.

    Added 2026-08-30 with the `.physics.dlamie` producer. The reference is transcribed
    rather than called because PROCESS writes these two lines inline in `Physics.run`
    with no staticmethod around them -- which is precisely why the field had no producer
    here (see the port function's docstring).

    The legacy point is `large_tokamak_eval` at convergence, and it reproduces PROCESS's
    own `dlamie = 17.834316405099152` on that run.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_coulomb_logarithm_ion_electron
    ported = calculate_coulomb_logarithm_ion_electron

    samples = FROM_FILE

    fuzz_bounds = {
        "nd_plasma_electrons_vol_avg": (1.0e19, 5.0e20),
        "temp_plasma_electron_vol_avg_kev": (1.0, 40.0),
    }


class TestPfluxPlasmaSurfaceNeutronAvgMw(Tier1Contract):
    """`pflux = p_neutron_total_mw / a_plasma_surface`, `physics.py:835-837`.

    Added 2026-08-30 with the producer. The legacy point is `large_tokamak_eval` at
    convergence and reproduces PROCESS's `1.0911547345980364` there; the value the pin
    quotes (`0.71479842`) is `large_tokamak_nof`'s, a different run of a different file.

    The fuzz bounds keep `a_plasma_surface` well away from zero: the division has no
    guard in PROCESS and none here, and a surface area of zero is not a plasma.
    """

    audit_record = "models/physics/physics.md"
    reference = _reference_pflux_plasma_surface_neutron_avg_mw
    ported = calculate_pflux_plasma_surface_neutron_avg_mw

    samples = FROM_FILE

    fuzz_bounds = {
        "p_neutron_total_mw": (100.0, 3000.0),
        "a_plasma_surface": (100.0, 3000.0),
    }
