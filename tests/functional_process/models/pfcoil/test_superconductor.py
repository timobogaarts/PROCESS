"""Harness cases for `functional_process/cottax/pfcoil/superconductor.py`.

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
from functional_process.cottax.pfcoil.superconductor import (
    calculate_cs_critical_current_density_iter_nb3sn,
    calculate_cs_critical_current_density_wst_nb3sn,
    calculate_cs_strand_critical_current_density,
    calculate_cs_temperature_margin_iter_nb3sn,
    calculate_cs_temperature_margin_wst_nb3sn,
    calculate_pf_strand_critical_current_density,
    calculate_pf_strand_critical_current_density_hazelton_zhai_rebco,
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

# The two the temperature margin needs on top of those, off the same run --
# `large_tokamak_nof`, where the margin was the missing producer. `superconpf`'s
# `j_pf_wp` is `|c_pf_cs_coils_peak_ma[6]| / a_cs_cable_space * 1e6`
# (`ohcalc:3597-3606`),
# and `abs()` is why the CS's negative peak current is carried as written.
_C_PF_CS_COILS_PEAK_MA_CS = -191.60361
_TEMP_CS_SUPERCONDUCTOR_MARGIN = 3.4208032
"""PROCESS's own `.pf_coil.temp_cs_superconductor_margin` on that run, against the
port's frozen `0.0` before `.tokamak.cs_coil.temperature_margin` landed. Recorded here
as the number this contract exists to keep, not asserted directly -- the contract
compares against `superconpf` itself."""

# The PF coils' own `superconpf` call, at the **last** of the six coils -- the one whose
# value survives `pfcoil()`'s loop (`pfcoil.py:900-904`, no index on the assignment).
# Off the same `large_tokamak_nof` run.
_B_PF_COIL_PEAK_LAST = 2.5773536152742245
_BPF2_LAST = 1.5851124381765633
_TEMP_PF_PEAK_FIELD = 4.75
_FCUPFSU = 0.69
_F_A_PF_COIL_VOID_LAST = 0.3
_J_PF_COIL_WP_PEAK_LAST = 8.0e6
_J_CRIT_STR_PF = 1101789868.9092896
"""PROCESS's own `.pf_coil.j_crit_str_pf` on that run, against the port's frozen `0.0`
before `.tokamak.pf_coil.strand_critical_current` landed -- the missing producer that
kept `.costs.c2222` off the graph (`_audit/cost_boundary_inputs.md` §13.2). Recorded,
not asserted: the contract compares against `superconpf` itself."""


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


def _reference_cs_temperature_margin(isumat):
    """An adapter over `superconpf`'s **fourth** return, `min`ed over the two fields.

    The sibling adapter above runs this same root find and throws the answer away
    (`superconpf` always finishes with it, whatever the caller wants). Here it is the
    only return kept, and the two calls are `ohcalc`'s own two -- end of flat-top and
    beginning of pulse (`pfcoil.py:3586-3618`, `:3636-3665`) -- combined by
    `min(tmarg1, tmarg2)` (`:3679`).

    Unlike the critical-current adapter, `j_pf_wp` is **not** a held constant here: the
    margin is the one consumer that reads it, so it is rebuilt from
    `c_pf_cs_coils_peak_ma` and `a_cs_cable_space` exactly as `ohcalc` does.
    """

    def reference(
        b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start,
        c_pf_cs_coils_peak_ma,
        a_cs_cable_space,
        f_a_cs_void,
        fcuohsu,
        strain,
        temp_cs_superconductor_operating,
    ):
        j_pf_wp = abs(float(c_pf_cs_coils_peak_ma)) / float(a_cs_cable_space) * 1.0e6
        margins = []
        for b_cs_peak in (b_cs_peak_flat_top_end, b_cs_peak_pulse_start):
            *_, tmarg = superconpf(
                b_pf_peak=float(b_cs_peak),
                fhe=float(f_a_cs_void),
                fcu=float(fcuohsu),
                j_pf_wp=j_pf_wp,
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
            margins.append(tmarg)
        return min(margins)

    return reference


def _reference_strand_critical_current_density(
    j_cs_conductor_critical_flat_top_end, fcuohsu
):
    """`ohcalc:3626-3628`'s `else` arm, transcribed -- three lines with no callable."""
    return float(j_cs_conductor_critical_flat_top_end) * (1.0 - float(fcuohsu))


def _reference_pf_strand_critical_current_density(
    b_pf_coil_peak, bpf2, temp_pf_peak_field, fcupfsu
):
    """`pfcoil()`'s `:871-904` for one coil: `bmax`, `superconpf(isumat=3)`, the `else`
    strand branch.

    The **whole** real function again, not a subset: the NbTi arm's `j_crit_sc` is the
    third of `superconpf`'s four returns and the temperature margin is still solved and
    discarded, exactly as `pfcoil()` discards it here (`_tmarg`, `:877`).

    `fhe`, `j_pf_wp` and the eight arguments this arm never reads are held at
    `large_tokamak_nof`'s own values rather than at zeros, for `_reference_superconpf`'s
    reason: a regression in the branch selection must show up as a value mismatch and not
    as a division by zero. On this arm they reach only `j_crit_cable` (the second return,
    discarded) and the root find (the fourth, also discarded), so they cannot move the
    number under test -- which is the point the fuzz draws below check by varying
    `fcupfsu` against them.
    """
    _jcritwp, _j_crit_cable, j_crit_sc, _tmarg = superconpf(
        b_pf_peak=max(abs(float(b_pf_coil_peak)), abs(float(bpf2))),
        fhe=_F_A_PF_COIL_VOID_LAST,
        fcu=float(fcupfsu),
        j_pf_wp=_J_PF_COIL_WP_PEAK_LAST,
        isumat=SuperconductorModel.OLD_LUBELL_NBTI,
        fhts=0.5,
        strain=-0.005,
        temp_pf_peak_field=float(temp_pf_peak_field),
        bcritsc=24.0,
        tcritsc=16.0,
        b_crit_upper_nbti=14.86,
        t_crit_nbti=9.04,
        dr_hts_tape=4.0e-3,
        dx_hts_tape_rebco=1.0e-6,
        dx_hts_tape_total=6.5e-5,
    )
    return j_crit_sc * (1.0 - float(fcupfsu))


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


class TestCSTemperatureMarginIterNb3Sn(Tier1Contract):
    """`..._iter_nb3sn` -> `min` over `superconpf(isumat=1)`'s two `tmarg` returns.

    **Tier 1 despite a root find on both sides**, which is the decision worth naming:
    the port replicates `scipy.optimize.newton`'s secant branch step for step
    (`models/tfcoil/superconducting.py::solve_current_sharing_temperature`, imported
    rather than re-derived), so the two iterations take the same steps from the same
    two starting points and stop on the same rule. What is being compared is therefore
    the same quantity on both sides, not two answers to the same question -- which is
    what tier 2 exists for. Measured agreement on the legacy point: 2 ulp.

    The one legacy point is `large_tokamak_nof`'s converged state, where this producer
    was missing: PROCESS computes `3.4208032` K and the port had `0.0`, so constraint 60
    was comparing a frozen zero against `.tfcoil.temp_cs_superconductor_margin_min`.
    """

    audit_record = "models/pfcoil/superconductor.md"
    reference = _reference_cs_temperature_margin(SuperconductorModel.ITER_NB3SN)
    ported = calculate_cs_temperature_margin_iter_nb3sn

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            b_cs_peak_flat_top_end=_B_CS_PEAK_FLAT_TOP_END,
            b_cs_peak_pulse_start=_B_CS_PEAK_PULSE_START,
            c_pf_cs_coils_peak_ma=_C_PF_CS_COILS_PEAK_MA_CS,
            a_cs_cable_space=_A_CS_CABLE_SPACE,
            f_a_cs_void=_F_A_CS_VOID,
            fcuohsu=_FCUOHSU,
            strain=_STR_CS_CON_RES,
            temp_cs_superconductor_operating=_TEMP_CS_OPERATING,
        ),
    ]

    fuzz_bounds = {
        # Narrower than the critical-current contracts' `(6, 18)` T on purpose: the
        # solve is for the temperature at which the critical current density falls to
        # the operating one, and far outside the machine's own operating box the two
        # curves need not cross at all -- `disp=False` then returns a non-root and the
        # comparison would be between two arbitrary iterates rather than two roots.
        "b_cs_peak_flat_top_end": (10.0, 16.0),
        "b_cs_peak_pulse_start": (10.0, 16.0),
        "c_pf_cs_coils_peak_ma": (-260.0, -130.0),
        "a_cs_cable_space": _around(_A_CS_CABLE_SPACE, 0.15),
        "f_a_cs_void": (0.25, 0.35),
        "fcuohsu": (0.6, 0.8),
        "strain": (-0.008, -0.002),
        "temp_cs_superconductor_operating": (4.2, 5.2),
    }


class TestCSTemperatureMarginWstNb3Sn(Tier1Contract):
    """`..._wst_nb3sn` -> the same, with `superconpf(isumat=5)`.

    `low_aspect_ratio_DEMO.IN.DAT`'s conductor, evaluated at the reference machine's
    fields for the same reason `TestCSCriticalCurrentDensityWstNb3Sn` is: the port does
    not assemble that machine end to end yet, and a legitimate point for a pure function
    is a legitimate point whichever file its numbers came from.
    """

    audit_record = "models/pfcoil/superconductor.md"
    reference = _reference_cs_temperature_margin(SuperconductorModel.WST_NB3SN)
    ported = calculate_cs_temperature_margin_wst_nb3sn

    samples = TestCSTemperatureMarginIterNb3Sn.samples
    fuzz_bounds = TestCSTemperatureMarginIterNb3Sn.fuzz_bounds


class TestPFStrandCriticalCurrentDensity(Tier1Contract):
    """`calculate_pf_strand_critical_current_density` -> `pfcoil.py:871-904`.

    One legacy point, the one that survives `pfcoil()`'s loop: the last of the six PF
    coils on `large_tokamak_nof`, where PROCESS's `.pf_coil.j_crit_str_pf` is
    `1.1017899e9` A/m^2 and the port read a frozen `0.0` until this unit landed.

    **`b_pf_coil_peak` and `bpf2` are fuzzed independently**, which is what exercises the
    `max(|.|, |.|)` PROCESS takes of them: the legacy point has the inner edge above the
    outer, and the draws below cross over.
    """

    audit_record = "models/pfcoil/superconductor.md"
    reference = _reference_pf_strand_critical_current_density
    ported = calculate_pf_strand_critical_current_density

    samples = [
        legacy_sample(
            "large-tokamak-nof-last-pf-coil",
            b_pf_coil_peak=_B_PF_COIL_PEAK_LAST,
            bpf2=_BPF2_LAST,
            temp_pf_peak_field=_TEMP_PF_PEAK_FIELD,
            fcupfsu=_FCUPFSU,
        ),
    ]

    fuzz_bounds = {
        "b_pf_coil_peak": (0.5, 6.0),
        "bpf2": (0.5, 6.0),
        "temp_pf_peak_field": (4.0, 6.0),
        "fcupfsu": (0.4, 0.85),
    }


# ---------------------------------------------------------------------------------
# The REBCO-tape PF arm (`i_pf_superconductor = 9`), 2026-08-31.
# ---------------------------------------------------------------------------------
#
# The last of the eight PF coils on `test_masses.py::TestPFCoilChainSphericalTokamak`'s
# legacy point -- the one that survives `pfcoil()`'s loop, since the assignment at
# `pfcoil.py:900-904` carries no index. Read off that contract's own reference chain.
_ST_B_PF_COIL_PEAK_LAST = 1.3945960069778585
_ST_BPF2_LAST = 0.977502159795704
_ST_TEMP_PF_PEAK_FIELD = 20.0
"""`.tfcoil.tftmp` on both tracked spherical tokamaks
(`spherical_tokamak_eval.IN.DAT:363`, `st_regression.IN.DAT:1332`) -- 20 K, not the
4.75 K of the NbTi machines, which is the point of an HTS coil and is why this
contract's temperature bounds sit where they do."""
_ST_FCUPFSU = 0.69
"""`pfcoil_variables.py:217`'s default. Neither ST file sets it -- `st_regression.IN.DAT`
carries the line commented out."""
_ST_DR_HTS_TAPE = 6.2886208094437651e-3
_ST_DX_HTS_TAPE_REBCO = 1.0e-6
_ST_DX_HTS_TAPE_TOTAL = 2.11e-4
"""The tape geometry, and **it is the TF coil's, not the PF coil's** --
`pfcoil.py:892-894` passes `.superconducting_tfcoil.dr_tf_hts_tape` /
`.dx_tf_hts_tape_rebco` /
`.dx_tf_hts_tape_total` straight into the PF call, because `pfcoil_variables.py` has no
tape geometry of its own. The width and total thickness are the CroCo turn's own outputs
(`tests/functional_process/models/tfcoil/test_croco.py:509-511`, the same two numbers);
`dx_tf_hts_tape_rebco` is `1.0e-6` on both ST files explicitly
(`spherical_tokamak_eval.IN.DAT:73`, `st_regression.IN.DAT:787`) and by default."""


def _reference_pf_strand_critical_current_density_hazelton_zhai_rebco(
    b_pf_coil_peak,
    bpf2,
    temp_pf_peak_field,
    fcupfsu,
    dr_hts_tape,
    dx_hts_tape_rebco,
    dx_hts_tape_total,
):
    """`pfcoil()`'s `:871-904` for one coil at `i_pf_superconductor = 9`.

    The whole real `superconpf` again, `HAZELTON_ZHAI_REBCO` arm: `bmax`, the third of
    its four returns, and the `else` strand branch. `bc20m = 138` / `tc0m = 92` are
    `superconpf`'s own literals (`pfcoil.py:4853-4854`) and are therefore *not*
    arguments on either side -- the port carries them as module constants for the same
    reason.

    **The three tape dimensions are real inputs here**, unlike in
    `_reference_superconpf`/`_reference_pf_strand_critical_current_density`, where they
    are held constants because the NbTi and Nb3Sn arms never read them. This is the arm
    they exist for.

    `fhe`, `j_pf_wp` and the six arguments this arm does not read are held at the
    spherical tokamak's own values rather than at zeros, for the reason the sibling
    adapters give: a regression in the branch selection must surface as a value mismatch
    and not as a division by zero. They reach only `j_crit_cable` (the second return,
    discarded) and the temperature-margin root find (the fourth, also discarded, and run
    anyway because `superconpf` always runs it).
    """
    _jcritwp, _j_crit_cable, j_crit_sc, _tmarg = superconpf(
        b_pf_peak=max(abs(float(b_pf_coil_peak)), abs(float(bpf2))),
        fhe=0.3,
        fcu=float(fcupfsu),
        j_pf_wp=1.1e7,
        isumat=SuperconductorModel.HAZELTON_ZHAI_REBCO,
        fhts=0.5,
        strain=-0.005,
        temp_pf_peak_field=float(temp_pf_peak_field),
        bcritsc=24.0,
        tcritsc=16.0,
        b_crit_upper_nbti=14.86,
        t_crit_nbti=9.04,
        dr_hts_tape=float(dr_hts_tape),
        dx_hts_tape_rebco=float(dx_hts_tape_rebco),
        dx_hts_tape_total=float(dx_hts_tape_total),
    )
    return j_crit_sc * (1.0 - float(fcupfsu))


class TestPFStrandCriticalCurrentDensityHazeltonZhaiRebco(Tier1Contract):
    """`calculate_pf_strand_critical_current_density_hazelton_zhai_rebco` ->
    `superconpf(isumat=9)`'s third return times `1 - fcupfsu`.

    Owed since 2026-08-30 (`_audit/next_steps.md` §20.5 item 2). The REBCO-tape PF arm
    both spherical tokamaks set (`spherical_tokamak_eval.IN.DAT:235`,
    `st_regression.IN.DAT:1670`), and the sibling of
    `TestPFStrandCriticalCurrentDensity`, which is the same block of `pfcoil()` with the
    `OLD_LUBELL_NBTI` arm.

    **Distinct from `TestHijcRebco`**, which covers the same fit in
    `tests/functional_process/models/physics/test_superconductors.py`. That contract
    tests `hijc_rebco` against `process.models.superconductors.hijc_rebco` -- the
    critical surface alone, called directly with its own arguments. This one tests the
    *PF block around it*: which of `superconpf`'s four returns is taken, that
    `bc20m = 138` / `tc0m = 92` are the constants this arm passes, that `bmax` is
    `max(|b_pf_coil_peak|, |bpf2|)`, and that the strand branch is the `else` one
    (`* (1 - fcupfsu)`, not the `/ (1 - fcupfsu)` of the `{2, 6, 8}` arm). None of that
    is reachable from the material fit's own contract.

    `b_pf_coil_peak` and `bpf2` are fuzzed independently so the draws cross the
    `max(|.|, |.|)` over: the legacy point has the inner edge above the outer.
    """

    audit_record = "models/pfcoil/superconductor.md"
    reference = _reference_pf_strand_critical_current_density_hazelton_zhai_rebco
    ported = calculate_pf_strand_critical_current_density_hazelton_zhai_rebco

    samples = [
        legacy_sample(
            "spherical-tokamak-plausible-last-pf-coil",
            b_pf_coil_peak=_ST_B_PF_COIL_PEAK_LAST,
            bpf2=_ST_BPF2_LAST,
            temp_pf_peak_field=_ST_TEMP_PF_PEAK_FIELD,
            fcupfsu=_ST_FCUPFSU,
            dr_hts_tape=_ST_DR_HTS_TAPE,
            dx_hts_tape_rebco=_ST_DX_HTS_TAPE_REBCO,
            dx_hts_tape_total=_ST_DX_HTS_TAPE_TOTAL,
        ),
    ]

    fuzz_bounds = {
        "b_pf_coil_peak": (0.5, 6.0),
        "bpf2": (0.5, 6.0),
        # Well under `tc0m = 92 K`: above it PROCESS's `(1 - T/T_c0) ** 1.4` returns a
        # *complex* and the port returns `nan`, a type-level disagreement neither side
        # is trying to reproduce (`TestHijcRebco`'s docstring, and `hijc_rebco`'s).
        "temp_pf_peak_field": (4.0, 40.0),
        "fcupfsu": (0.4, 0.85),
        "dr_hts_tape": (2.0e-3, 8.0e-3),
        "dx_hts_tape_rebco": (5.0e-7, 2.0e-6),
        "dx_hts_tape_total": (5.0e-5, 3.0e-4),
    }
