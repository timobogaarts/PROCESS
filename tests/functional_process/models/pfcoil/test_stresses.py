"""Harness cases for `functional_process/models/pfcoil/stresses.py`.

Audit record: `functional_process/_audit/units/models/pfcoil/fields.md` § "2026-08-27 --
the CS chain" (this module is owed a registry row of its own; see its own docstring).

Six tier-1 contracts, and **none of them needs an adapter**: every PROCESS function this
module ports is either a bare `@staticmethod` on `CSCoil` or a module-level function in
`process/models/engineering/materials.py`, and none touches `self.data`. The whole
`ohcalc` block is diffed as one function too, against those same PROCESS callables
composed in PROCESS's order.

**The elliptic integrals are the point of two of these cases.** `_ellipk`/`_ellipe`
replace `scipy.special.ellipk`/`ellipe`, which is what makes the axial stress traceable
at all -- so they are diffed against scipy directly, over the whole open unit interval,
rather than only through the stress that consumes them. A fit that agreed at the
reference machine's `m = 0.94` and drifted at `m -> 1` would pass every other case here.
"""

import numpy as np
from scipy.special import ellipe as scipy_ellipe
from scipy.special import ellipk as scipy_ellipk

from functional_process._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.models.pfcoil.stresses import (
    _ellipe,
    _ellipk,
    calculate_cs_hoop_stress,
    calculate_cs_radial_stress,
    calculate_cs_self_peak_midplane_axial_stress,
    calculate_cs_stresses,
    calculate_tresca_stress,
    calculate_von_mises_stress,
)
from process.models.engineering.materials import (
    calculate_tresca_stress as process_tresca,
)
from process.models.engineering.materials import (
    calculate_von_mises_stress as process_von_mises,
)
from process.models.pfcoil import CSCoil

# Every literal below is read off a converged, in-process PROCESS run of
# `tests/regression/input_files/large_tokamak_eval.IN.DAT`.
_R_CS_INNER = 2.003843190236783
_R_CS_OUTER = 2.550659784225536
_R_CS_MIDDLE = 2.2772514872311596
_DZ_CS_FULL = 15.87279089542949
_A_CS_TOROIDAL = 7.824066771971542
_J_CS_PULSE_START = 20047872.417147826
_B_CS_PEAK_PULSE_START = 13.978498492871925
_C_CS_PEAK_MA = -186.11980298805437
_POISSON_STEEL = 0.3
_F_A_CS_TURN_STEEL = 0.4856940627

# The three principal stresses at that same point, for the two criterion functions.
_STRESS_HOOP = 748367494.3
_STRESS_AXIAL = -416318065.4
_STRESS_RADIAL = -13874803.95


def _reference_ellipk(m):
    """`scipy.special.ellipk`, wrapped.

    The bare ufunc takes no keyword arguments and the harness calls every reference by
    name, so a one-line Python wrapper is the adapter here.
    """
    return float(scipy_ellipk(float(m)))


def _reference_ellipe(m):
    """`scipy.special.ellipe`, wrapped -- see `_reference_ellipk`."""
    return float(scipy_ellipe(float(m)))


def _around(base, fraction):
    """Elementwise `(lower, upper)` at `+-fraction` of `base`, sign-safe."""
    base = np.asarray(base, dtype=float)
    low = base * (1.0 - fraction)
    high = base * (1.0 + fraction)
    return np.minimum(low, high), np.maximum(low, high)


def _reference_cs_stresses(
    r_cs_inner,
    r_cs_outer,
    r_cs_middle,
    dz_cs_full,
    a_cs_toroidal,
    j_cs_pulse_start,
    b_cs_peak_pulse_start,
    c_cs_peak_ma,
    f_poisson_cs_structure,
    f_a_cs_turn_steel,
):
    """`ohcalc`'s superconducting stress block, composed from PROCESS's own callables.

    Five calls in PROCESS's order (`pfcoil.py:3406-3521`), with PROCESS's own choice of
    radius at each -- the inner radius for the hoop stress, the mean radius for the peak
    radial stress, the inner radius again for the "inner" radial stress. No
    `DataStructure` and no adapter: all five are `@staticmethod`s or module functions.
    """
    hoop = CSCoil.calculate_cs_hoop_stress(
        r_stress_point=float(r_cs_inner),
        r_cs_inner=float(r_cs_inner),
        r_cs_outer=float(r_cs_outer),
        j_cs=float(j_cs_pulse_start),
        b_cs_inner=float(b_cs_peak_pulse_start),
        f_poisson_cs_structure=float(f_poisson_cs_structure),
        f_a_cs_turn_steel=float(f_a_cs_turn_steel),
    )
    axial, force = CSCoil.calculate_cs_self_peak_midplane_axial_stress(
        r_cs_outer=float(r_cs_outer),
        dz_cs_half=float(dz_cs_full) / 2.0,
        c_cs_peak=float(c_cs_peak_ma) * 1.0e6,
        a_cs_toroidal=float(a_cs_toroidal),
    )
    radial_peak = CSCoil.calculate_cs_radial_stress(
        r_stress_point=float(r_cs_middle),
        r_cs_inner=float(r_cs_inner),
        r_cs_outer=float(r_cs_outer),
        j_cs=float(j_cs_pulse_start),
        b_cs_inner=float(b_cs_peak_pulse_start),
        f_poisson_cs_structure=float(f_poisson_cs_structure),
    )
    radial_inner = CSCoil.calculate_cs_radial_stress(
        r_stress_point=float(r_cs_inner),
        r_cs_inner=float(r_cs_inner),
        r_cs_outer=float(r_cs_outer),
        j_cs=float(j_cs_pulse_start),
        b_cs_inner=float(b_cs_peak_pulse_start),
        f_poisson_cs_structure=float(f_poisson_cs_structure),
    )
    return (
        hoop,
        axial,
        force,
        radial_peak,
        radial_inner,
        process_tresca(stress_x=hoop, stress_y=axial, stress_z=radial_peak),
        process_von_mises(
            stress_x=hoop,
            stress_y=axial,
            stress_z=radial_peak,
            stress_shear_xy=0.0,
            stress_shear_yz=0.0,
            stress_shear_zx=0.0,
        ),
    )


class TestEllipk(Tier1Contract):
    """`_ellipk` (AGM) -> `scipy.special.ellipk`, the function PROCESS actually calls.

    Not a PROCESS function, and it is here anyway: `ohcalc` evaluates
    `scipy.special.ellipk`, so scipy *is* the reference the port has to reproduce, and
    substituting an approximation for it would be a divergence dressed as a port. The
    fuzz range spans most of `[0, 1)` rather than the neighbourhood of the reference
    machine's `m = 0.94`, because the AGM's failure mode -- if it had one -- would be at
    the endpoints.

    **The upper bound is `0.998` and not `1`, and the limit is the oracle's, not the
    port's.** `test_gradient_agreement` differentiates the reference by finite
    difference with PROCESS's own `epsfcn = 1e-3`, a *relative* step -- so at
    `m = 0.9999` the forward point is `1.00089`, outside `K`'s domain, and scipy returns
    `nan`. The port is fine there (measured against scipy at `m = 0.9999`: `1.5e-16` for
    `K`, `2.2e-16` for `E`); it is the finite-difference reference that cannot be
    evaluated. Bounded rather than special-cased, because a sample whose gradient check
    is vacuous is worse than one that is not drawn.
    """

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_ellipk
    ported = _ellipk

    samples = [
        legacy_sample("reference-machine-kb2", m=0.9390581),
        legacy_sample("near-zero", m=1.0e-8),
        legacy_sample("half", m=0.5),
        legacy_sample("near-one", m=0.999),
        *fuzz_samples({"m": (1.0e-6, 0.998)}, count=8, seed=1),
    ]

    fuzz_bounds = {"m": (1.0e-6, 0.998)}


class TestEllipe(Tier1Contract):
    """`_ellipe` (AGM) -> `scipy.special.ellipe`. See `TestEllipk`."""

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_ellipe
    ported = _ellipe

    samples = [
        legacy_sample("reference-machine-kb2", m=0.9390581),
        legacy_sample("near-zero", m=1.0e-8),
        legacy_sample("half", m=0.5),
        legacy_sample("near-one", m=0.999),
        *fuzz_samples({"m": (1.0e-6, 0.998)}, count=8, seed=2),
    ]

    fuzz_bounds = {"m": (1.0e-6, 0.998)}


class TestCalculateCSHoopStress(Tier1Contract):
    """`calculate_cs_hoop_stress` -> the `@staticmethod` it ports."""

    audit_record = "models/pfcoil/fields.md"
    reference = staticmethod(CSCoil.calculate_cs_hoop_stress)
    ported = calculate_cs_hoop_stress

    samples = [
        legacy_sample(
            "large-tokamak-converged-inner-radius",
            r_stress_point=_R_CS_INNER,
            r_cs_inner=_R_CS_INNER,
            r_cs_outer=_R_CS_OUTER,
            j_cs=_J_CS_PULSE_START,
            b_cs_inner=_B_CS_PEAK_PULSE_START,
            f_poisson_cs_structure=_POISSON_STEEL,
            f_a_cs_turn_steel=_F_A_CS_TURN_STEEL,
        ),
        legacy_sample(
            "off-the-inner-radius",
            r_stress_point=_R_CS_MIDDLE,
            r_cs_inner=_R_CS_INNER,
            r_cs_outer=_R_CS_OUTER,
            j_cs=_J_CS_PULSE_START,
            b_cs_inner=_B_CS_PEAK_PULSE_START,
            f_poisson_cs_structure=_POISSON_STEEL,
            f_a_cs_turn_steel=_F_A_CS_TURN_STEEL,
        ),
    ]

    fuzz_bounds = {
        "r_stress_point": _around(_R_CS_MIDDLE, 0.10),
        "r_cs_inner": _around(_R_CS_INNER, 0.10),
        "r_cs_outer": _around(_R_CS_OUTER, 0.10),
        "j_cs": _around(_J_CS_PULSE_START, 0.20),
        "b_cs_inner": _around(_B_CS_PEAK_PULSE_START, 0.20),
        "f_poisson_cs_structure": (0.2, 0.4),
        "f_a_cs_turn_steel": (0.2, 0.8),
    }


class TestCalculateCSRadialStress(Tier1Contract):
    """`calculate_cs_radial_stress` -> the `@staticmethod` it ports.

    **The first legacy point is the snap-to-zero case.** At `r_stress_point =
    r_cs_inner` both of PROCESS's shape terms are algebraically zero, PROCESS evaluates
    them in floating point and then applies `np.isclose(..., 0.0)`, and the port
    reproduces that with a `jnp.where` on the same threshold. The output is exactly
    `0.0` on both sides -- which is what `.pf_coil.stress_radial_cs_inner` is at the
    converged machine, and would be ~1e-12 without the guard.

    **That point's gradient is not checked against the finite difference, and the reason
    is a real finding rather than a tolerance problem.** `np.isclose(x, 0.0)` snaps a
    window of half-width `1e-8` in the shape terms to a *constant*, so inside it the
    function's true derivative w.r.t. the two radii is zero -- and the port's autodiff
    says zero. PROCESS's own finite difference says `-9.9e7`, because `epsfcn = 1e-3`
    steps clean out of a `1e-8`-wide window and measures the slope on the far side. The
    FD is not a valid oracle for a feature narrower than its own step, so
    `diff_argnames` drops the two arguments the window is a function of, at that sample
    only. Every other argument at that sample, and every argument at every other sample,
    is still checked -- including `r_stress_point` at the mean radius, where the snap
    does not fire and the two agree.

    Recorded, not worked around: **PROCESS's snap makes the CS's inner radial stress
    exactly derivative-free in the two radii**, and any optimiser that reaches for that
    quantity's sensitivity will get zero from the port and a step-size-dependent number
    from a finite difference.
    """

    audit_record = "models/pfcoil/fields.md"
    reference = staticmethod(CSCoil.calculate_cs_radial_stress)
    ported = calculate_cs_radial_stress

    _SNAPPED = "inner-radius-the-snapped-zero"
    _SNAPPED_ARGS = ("r_stress_point", "r_cs_inner")

    @classmethod
    def diff_argnames(cls, sample):
        """The base rule, minus the snap window's two arguments at the snapped sample.

        See the class docstring. Matched on `Sample.label`, which is the name the
        `legacy_sample` call gives it; `Sample.id` adds the provenance prefix and is
        what the test node is named after.
        """
        names = super().diff_argnames(sample)
        if sample.label == cls._SNAPPED:
            return tuple(n for n in names if n not in cls._SNAPPED_ARGS)
        return names

    samples = [
        legacy_sample(
            "inner-radius-the-snapped-zero",
            r_stress_point=_R_CS_INNER,
            r_cs_inner=_R_CS_INNER,
            r_cs_outer=_R_CS_OUTER,
            j_cs=_J_CS_PULSE_START,
            b_cs_inner=_B_CS_PEAK_PULSE_START,
            f_poisson_cs_structure=_POISSON_STEEL,
        ),
        legacy_sample(
            "mean-radius-the-peak",
            r_stress_point=_R_CS_MIDDLE,
            r_cs_inner=_R_CS_INNER,
            r_cs_outer=_R_CS_OUTER,
            j_cs=_J_CS_PULSE_START,
            b_cs_inner=_B_CS_PEAK_PULSE_START,
            f_poisson_cs_structure=_POISSON_STEEL,
        ),
    ]

    fuzz_bounds = {
        "r_stress_point": _around(_R_CS_MIDDLE, 0.08),
        "r_cs_inner": _around(_R_CS_INNER, 0.08),
        "r_cs_outer": _around(_R_CS_OUTER, 0.08),
        "j_cs": _around(_J_CS_PULSE_START, 0.20),
        "b_cs_inner": _around(_B_CS_PEAK_PULSE_START, 0.20),
        "f_poisson_cs_structure": (0.2, 0.4),
    }


class TestCalculateCSSelfPeakMidplaneAxialStress(Tier1Contract):
    """`calculate_cs_self_peak_midplane_axial_stress` -> the `@staticmethod` it ports.

    This is the case that closes the `scipy.special` blocker: PROCESS's side calls
    `ellipk`/`ellipe` and the port's calls the AGM, and the two agree at the value and
    gradient level over the whole fuzz range.
    """

    audit_record = "models/pfcoil/fields.md"
    reference = staticmethod(CSCoil.calculate_cs_self_peak_midplane_axial_stress)
    ported = calculate_cs_self_peak_midplane_axial_stress

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            r_cs_outer=_R_CS_OUTER,
            dz_cs_half=_DZ_CS_FULL / 2.0,
            c_cs_peak=_C_CS_PEAK_MA * 1.0e6,
            a_cs_toroidal=_A_CS_TOROIDAL,
        ),
        # A squat coil, where `kb2` moves far from the reference machine's 0.94.
        legacy_sample(
            "squat-cs",
            r_cs_outer=_R_CS_OUTER,
            dz_cs_half=0.5,
            c_cs_peak=_C_CS_PEAK_MA * 1.0e6,
            a_cs_toroidal=_A_CS_TOROIDAL,
        ),
    ]

    fuzz_bounds = {
        "r_cs_outer": _around(_R_CS_OUTER, 0.20),
        "dz_cs_half": (0.5, 12.0),
        "c_cs_peak": _around(_C_CS_PEAK_MA * 1.0e6, 0.25),
        "a_cs_toroidal": _around(_A_CS_TOROIDAL, 0.20),
    }


class TestCalculateTrescaStress(Tier1Contract):
    """`calculate_tresca_stress` -> `materials.calculate_tresca_stress`."""

    audit_record = "models/pfcoil/fields.md"
    reference = staticmethod(process_tresca)
    ported = calculate_tresca_stress

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            stress_x=_STRESS_HOOP,
            stress_y=_STRESS_AXIAL,
            stress_z=_STRESS_RADIAL,
        ),
        *fuzz_samples(
            {
                "stress_x": (-1.0e9, 1.0e9),
                "stress_y": (-1.0e9, 1.0e9),
                "stress_z": (-1.0e9, 1.0e9),
            },
            count=6,
            seed=72,
        ),
    ]

    fuzz_bounds = {
        "stress_x": (-1.0e9, 1.0e9),
        "stress_y": (-1.0e9, 1.0e9),
        "stress_z": (-1.0e9, 1.0e9),
    }


class TestCalculateVonMisesStress(Tier1Contract):
    """`calculate_von_mises_stress` -> `materials.calculate_von_mises_stress`."""

    audit_record = "models/pfcoil/fields.md"
    reference = staticmethod(process_von_mises)
    ported = calculate_von_mises_stress

    samples = [
        legacy_sample(
            "large-tokamak-converged-no-shear",
            stress_x=_STRESS_HOOP,
            stress_y=_STRESS_AXIAL,
            stress_z=_STRESS_RADIAL,
            stress_shear_xy=0.0,
            stress_shear_yz=0.0,
            stress_shear_zx=0.0,
        ),
        *fuzz_samples(
            {
                "stress_x": (-1.0e9, 1.0e9),
                "stress_y": (-1.0e9, 1.0e9),
                "stress_z": (-1.0e9, 1.0e9),
                "stress_shear_xy": (-1.0e8, 1.0e8),
                "stress_shear_yz": (-1.0e8, 1.0e8),
                "stress_shear_zx": (-1.0e8, 1.0e8),
            },
            count=6,
            seed=73,
        ),
    ]

    fuzz_bounds = {
        "stress_x": (-1.0e9, 1.0e9),
        "stress_y": (-1.0e9, 1.0e9),
        "stress_z": (-1.0e9, 1.0e9),
        "stress_shear_xy": (-1.0e8, 1.0e8),
        "stress_shear_yz": (-1.0e8, 1.0e8),
        "stress_shear_zx": (-1.0e8, 1.0e8),
    }


class TestCalculateCSStresses(Tier1Contract):
    """The whole `ohcalc` stress block -> the same five PROCESS callables, composed."""

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_cs_stresses
    ported = calculate_cs_stresses

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            r_cs_inner=_R_CS_INNER,
            r_cs_outer=_R_CS_OUTER,
            r_cs_middle=_R_CS_MIDDLE,
            dz_cs_full=_DZ_CS_FULL,
            a_cs_toroidal=_A_CS_TOROIDAL,
            j_cs_pulse_start=_J_CS_PULSE_START,
            b_cs_peak_pulse_start=_B_CS_PEAK_PULSE_START,
            c_cs_peak_ma=_C_CS_PEAK_MA,
            f_poisson_cs_structure=_POISSON_STEEL,
            f_a_cs_turn_steel=_F_A_CS_TURN_STEEL,
        ),
    ]

    fuzz_bounds = {
        "r_cs_inner": _around(_R_CS_INNER, 0.08),
        "r_cs_outer": _around(_R_CS_OUTER, 0.08),
        "r_cs_middle": _around(_R_CS_MIDDLE, 0.05),
        "dz_cs_full": _around(_DZ_CS_FULL, 0.20),
        "a_cs_toroidal": _around(_A_CS_TOROIDAL, 0.20),
        "j_cs_pulse_start": _around(_J_CS_PULSE_START, 0.20),
        "b_cs_peak_pulse_start": _around(_B_CS_PEAK_PULSE_START, 0.20),
        "c_cs_peak_ma": _around(_C_CS_PEAK_MA, 0.25),
        "f_poisson_cs_structure": (0.2, 0.4),
        "f_a_cs_turn_steel": (0.2, 0.8),
    }
