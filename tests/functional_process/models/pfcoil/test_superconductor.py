"""Harness cases for `functional_process/models/pfcoil/superconductor.py`.

Audit record: `functional_process/_audit/units/models/pfcoil/fields.md` § "2026-08-27 --
the CS chain" (this module is owed a registry row of its own; see its own docstring).

Three tier-1 contracts, all against PROCESS's real `superconpf` -- a module-level
function with no `self` and no `data`, so the adapters are thin: they call it with the
arm's `isumat`, take the two of its four returns `ohcalc` uses, and apply `ohcalc`'s own
cable-space-to-cross-section scaling.

**The adapters run PROCESS's temperature-margin solve and throw the answer away.**
`superconpf` always finishes with a `scipy.optimize.newton` secant iteration for
`tmarg` (`process/models/pfcoil.py:4894-4921`), whatever the caller wants. The port does
not own that output on this pass (see `CSCriticalCurrentDensitiesIterNb3Sn`), so the
oracle discards it -- but it is *run*, which is the point: the reference here is the
whole real function, not a hand-copied subset of it. `disp=False` means a
non-convergence returns rather than raises, so a fuzz point the solve cannot reach still
produces the two critical current densities this unit is about.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.pfcoil.superconductor import (
    calculate_cs_critical_current_density_iter_nb3sn,
    calculate_cs_critical_current_density_wst_nb3sn,
    calculate_cs_strand_critical_current_density,
)
from process.models.pfcoil import superconpf
from process.models.superconductors import SuperconductorModel

# Read off a converged, in-process PROCESS run of
# `tests/regression/input_files/large_tokamak_eval.IN.DAT`.
_B_CS_PEAK_FLAT_TOP_END = 14.040951087341834
_B_CS_PEAK_PULSE_START = 13.978498492871925
_F_A_CS_VOID = 0.3
_FCUOHSU = 0.7
_STR_CS_CON_RES = -0.005
_TEMP_CS_OPERATING = 4.75
_A_CS_CABLE_SPACE = 4.463921187651396
_A_CS_POLOIDAL = 8.679505454534445
_J_CS_CONDUCTOR_CRITICAL_FLAT_TOP_END = 350002855.2


def _around(base, fraction):
    """`(lower, upper)` at `+-fraction` of `base`, sign-safe."""
    base = np.asarray(base, dtype=float)
    low = base * (1.0 - fraction)
    high = base * (1.0 + fraction)
    return np.minimum(low, high), np.maximum(low, high)


def _reference_superconpf(isumat):
    """An adapter over PROCESS's `superconpf` for one `isumat`, plus `ohcalc`'s scaling.

    The eight arguments the two ported arms never read are held at
    `large_tokamak_eval`'s own values rather than at zeros, so that a regression in the
    branch selection shows up as a value mismatch and not as a division by zero:
    `j_pf_wp` (read only by `BI2212`), `fhts` (only by `BI2212`), `bcritsc`/`tcritsc`
    (only by `USER_DEFINED_NB3SN`), `b_crit_upper_nbti`/`t_crit_nbti` (only by
    `DURHAM_NBTI`) and the three HTS tape dimensions (only by `HAZELTON_ZHAI_REBCO`).
    """

    def reference(
        b_cs_peak,
        f_a_cs_void,
        fcuohsu,
        strain,
        temp_cs_superconductor_operating,
        a_cs_cable_space,
        a_cs_poloidal,
    ):
        jcritwp, _j_crit_cable, j_crit_sc, _tmarg = superconpf(
            b_pf_peak=float(b_cs_peak),
            fhe=float(f_a_cs_void),
            fcu=float(fcuohsu),
            j_pf_wp=2.0e7,
            isumat=isumat,
            fhts=0.5,
            strain=float(strain),
            temp_pf_peak_field=float(temp_cs_superconductor_operating),
            bcritsc=24.0,
            tcritsc=16.0,
            b_crit_upper_nbti=14.86,
            t_crit_nbti=9.04,
            dr_hts_tape=4.0e-3,
            dx_hts_tape_rebco=1.0e-6,
            dx_hts_tape_total=6.5e-5,
        )
        return (
            jcritwp * float(a_cs_cable_space) / float(a_cs_poloidal),
            j_crit_sc,
        )

    return reference


def _reference_strand_critical_current_density(
    j_cs_conductor_critical_flat_top_end, fcuohsu
):
    """`ohcalc:3626-3628`'s `else` arm, transcribed -- three lines with no callable."""
    return float(j_cs_conductor_critical_flat_top_end) * (1.0 - float(fcuohsu))


class TestCSCriticalCurrentDensityIterNb3Sn(Tier1Contract):
    """`..._iter_nb3sn` -> `superconpf(isumat=1)` + `ohcalc`'s scaling.

    Two legacy points, the two `ohcalc` actually evaluates: the peak field at the end of
    flat-top (constraint 26's bound, PROCESS's converged `3.780e7` A/m^2) and at the
    beginning of pulse (constraint 27's, `3.841e7`).
    """

    audit_record = "models/pfcoil/superconductor.md"
    reference = _reference_superconpf(SuperconductorModel.ITER_NB3SN)
    ported = calculate_cs_critical_current_density_iter_nb3sn

    samples = [
        legacy_sample(
            "large-tokamak-converged-flat-top-end",
            b_cs_peak=_B_CS_PEAK_FLAT_TOP_END,
            f_a_cs_void=_F_A_CS_VOID,
            fcuohsu=_FCUOHSU,
            strain=_STR_CS_CON_RES,
            temp_cs_superconductor_operating=_TEMP_CS_OPERATING,
            a_cs_cable_space=_A_CS_CABLE_SPACE,
            a_cs_poloidal=_A_CS_POLOIDAL,
        ),
        legacy_sample(
            "large-tokamak-converged-pulse-start",
            b_cs_peak=_B_CS_PEAK_PULSE_START,
            f_a_cs_void=_F_A_CS_VOID,
            fcuohsu=_FCUOHSU,
            strain=_STR_CS_CON_RES,
            temp_cs_superconductor_operating=_TEMP_CS_OPERATING,
            a_cs_cable_space=_A_CS_CABLE_SPACE,
            a_cs_poloidal=_A_CS_POLOIDAL,
        ),
    ]

    fuzz_bounds = {
        "b_cs_peak": (6.0, 18.0),
        "f_a_cs_void": (0.15, 0.45),
        "fcuohsu": (0.4, 0.85),
        "strain": (-0.01, 0.0),
        "temp_cs_superconductor_operating": (4.0, 6.0),
        "a_cs_cable_space": _around(_A_CS_CABLE_SPACE, 0.25),
        "a_cs_poloidal": _around(_A_CS_POLOIDAL, 0.25),
    }


class TestCSCriticalCurrentDensityWstNb3Sn(Tier1Contract):
    """`..._wst_nb3sn` -> `superconpf(isumat=5)` + `ohcalc`'s scaling.

    `low_aspect_ratio_DEMO.IN.DAT`'s CS conductor. Its own converged numbers are not
    read off that machine -- the port does not assemble it end to end yet -- so both
    legacy points are the reference machine's *fields* with this arm's fit, which is a
    legitimate point for a pure function and is what the fuzz draws extend.
    """

    audit_record = "models/pfcoil/superconductor.md"
    reference = _reference_superconpf(SuperconductorModel.WST_NB3SN)
    ported = calculate_cs_critical_current_density_wst_nb3sn

    samples = [
        legacy_sample(
            "reference-fields-flat-top-end",
            b_cs_peak=_B_CS_PEAK_FLAT_TOP_END,
            f_a_cs_void=_F_A_CS_VOID,
            fcuohsu=_FCUOHSU,
            strain=_STR_CS_CON_RES,
            temp_cs_superconductor_operating=_TEMP_CS_OPERATING,
            a_cs_cable_space=_A_CS_CABLE_SPACE,
            a_cs_poloidal=_A_CS_POLOIDAL,
        ),
    ]

    fuzz_bounds = {
        "b_cs_peak": (6.0, 18.0),
        "f_a_cs_void": (0.15, 0.45),
        "fcuohsu": (0.4, 0.85),
        "strain": (-0.01, 0.0),
        "temp_cs_superconductor_operating": (4.0, 6.0),
        "a_cs_cable_space": _around(_A_CS_CABLE_SPACE, 0.25),
        "a_cs_poloidal": _around(_A_CS_POLOIDAL, 0.25),
    }


class TestCSStrandCriticalCurrentDensity(Tier1Contract):
    """`calculate_cs_strand_critical_current_density` -> `ohcalc:3626-3628`'s `else`."""

    audit_record = "models/pfcoil/superconductor.md"
    reference = _reference_strand_critical_current_density
    ported = calculate_cs_strand_critical_current_density

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            j_cs_conductor_critical_flat_top_end=(_J_CS_CONDUCTOR_CRITICAL_FLAT_TOP_END),
            fcuohsu=_FCUOHSU,
        ),
    ]

    fuzz_bounds = {
        "j_cs_conductor_critical_flat_top_end": _around(
            _J_CS_CONDUCTOR_CRITICAL_FLAT_TOP_END, 0.40
        ),
        "fcuohsu": (0.4, 0.85),
    }
