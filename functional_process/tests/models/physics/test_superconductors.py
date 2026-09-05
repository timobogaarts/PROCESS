"""Harness cases for the ported superconductor material models (registry unit #22).

All 7 in-scope functions (plus the shared `bottura_scaling` helper) are already plain,
`self.data`-free functions in the source, so every `reference` below is PROCESS's own
module-level function called directly through the port's (unchanged) signature -- no
`data`-backdoor adapter needed, unlike most units in this registry. Legacy samples are
lifted from `tests/unit/models/test_superconductors.py`'s own parametrised cases.
"""

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.models.physics.superconductors import (
    bi2212,
    bottura_scaling,
    gl_nbti,
    gl_rebco,
    hijc_rebco,
    itersc,
    jcrit_nbti,
    jcrit_rebco,
    western_superconducting_nb3sn,
)
from process.core.exceptions import ProcessValueError
from process.models import superconductors as ref


class TestJcritRebco(Tier1Contract):
    """`jcrit_rebco` -> the same, unchanged.

    Legacy sample is `test_jcrit_rebco`'s case verbatim. Fuzz bounds span both the
    `temp_conductor < temp_c0max` / `>=` and `b_conductor < birr` / `>=` branches, and
    both in/out-of-validity-range points -- PROCESS does not raise on invalidity here
    (only logs), so no `reference_domain_errors` is needed.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.jcrit_rebco)
    ported = jcrit_rebco

    samples = [
        legacy_sample("jcrit-rebco-reference", temp_conductor=4.75, b_conductor=7.0),
    ]

    fuzz_bounds = {
        "temp_conductor": (1.0, 85.0),
        "b_conductor": (0.5, 25.0),
    }


class TestBotturaScaling(Tier1Contract):
    """`bottura_scaling` -> the same, unchanged.

    No direct PROCESS unit test exists (it's only exercised indirectly, through
    `itersc`/`western_superconducting_nb3sn`) -- sample is a constructed point using
    `itersc`'s own ITER fitting constants, verified by hand against
    `reactions.bottura_scaling` while writing the port. Fuzz bounds span both the
    `temp_critical` normal/abnormal branch and the inside/outside-critical-surface
    branch selecting `j_scaling`.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.bottura_scaling)
    ported = bottura_scaling

    static_argnames = ("csc", "p", "q", "c_a1", "c_a2", "epsilon_0a")

    samples = [
        legacy_sample(
            "bottura-scaling-iter-constants",
            csc=19922.0,
            p=0.63,
            q=2.1,
            c_a1=44.48,
            c_a2=0.0,
            epsilon_0a=0.00256,
            temp_conductor=4.75,
            b_conductor=13.008974843466492,
            epsilon=0.001601605753441172,
            b_c20max=32.97,
            temp_c0max=16.06,
        ),
    ]

    fuzz_bounds = {
        "temp_conductor": (1.0, 15.5),
        "b_conductor": (0.5, 32.0),
        "epsilon": (-0.005, 0.0),
        "b_c20max": (25.0, 35.0),
        "temp_c0max": (14.0, 18.0),
    }
    fuzz_fixed = {
        "csc": 19922.0,
        "p": 0.63,
        "q": 2.1,
        "c_a1": 44.48,
        "c_a2": 0.0,
        "epsilon_0a": 0.00256,
    }


class TestItersc(Tier1Contract):
    """`itersc` -> the same, unchanged.

    Both legacy samples are `test_itersc`'s parametrised cases verbatim -- genuinely
    legacy, generated from `large_tokamak_nof.IN.DAT`.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.itersc)
    ported = itersc

    samples = [
        legacy_sample(
            "itersc-large-tokamak-1",
            temp_conductor=4.75,
            b_conductor=13.008974843466492,
            strain=0.001601605753441172,
            b_c20max=32.969999999999999,
            temp_c0max=16.059999999999999,
        ),
        legacy_sample(
            "itersc-large-tokamak-2",
            temp_conductor=6.2510000000000003,
            b_conductor=13.008974843466492,
            strain=0.001601605753441172,
            b_c20max=32.969999999999999,
            temp_c0max=16.059999999999999,
        ),
    ]

    fuzz_bounds = {
        "temp_conductor": (1.0, 15.5),
        "b_conductor": (0.5, 32.0),
        "strain": (-0.005, 0.0),
        "b_c20max": (25.0, 35.0),
        "temp_c0max": (14.0, 18.0),
    }


class TestJcritNbti(Tier1Contract):
    """`jcrit_nbti` -> the same, unchanged.

    Both legacy samples are `test_jcrit_nbti`'s parametrised cases verbatim. Fuzz bounds
    span both the `bratio < 1` / `>= 1` branches.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.jcrit_nbti)
    ported = jcrit_nbti

    samples = [
        legacy_sample(
            "jcrit-nbti-large-tokamak-1",
            temp_conductor=4.75,
            b_conductor=8.0517923638507547,
            c0=10000000000,
            b_c20max=15,
            temp_c0max=9.3000000000000007,
        ),
        legacy_sample(
            "jcrit-nbti-large-tokamak-2",
            temp_conductor=6,
            b_conductor=8.0517923638507547,
            c0=10000000000,
            b_c20max=15,
            temp_c0max=9.3000000000000007,
        ),
    ]

    fuzz_bounds = {
        "temp_conductor": (1.0, 9.0),
        "b_conductor": (0.5, 14.0),
        "c0": (1.0e9, 2.0e10),
        "b_c20max": (12.0, 18.0),
        "temp_c0max": (7.0, 11.0),
    }


class TestBi2212(Tier1Contract):
    """`bi2212` -> the same, unchanged.

    Legacy sample is `test_bi2212`'s case verbatim. PROCESS raises `ProcessValueError`
    outside the fit's validity range (`temp_conductor > 20.0`, `b_conductor < 6.0`, or
    a computed `b > 104.0`) -- the fuzz range deliberately spans both sides of that
    boundary (verified: ~17% of 3000 draws over these bounds raise, none
    unexpectedly), and `reference_domain_errors` asserts the port returns non-finite
    exactly there.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.bi2212)
    ported = bi2212

    reference_domain_errors = (ProcessValueError,)

    samples = [
        legacy_sample(
            "bi2212-reference",
            b_conductor=7.0,
            jstrand=2.0e7,
            temp_conductor=4.75,
            f_strain=0.2,
        ),
        # Constructed to land outside the validity range (temp_conductor > 20.0),
        # exercising the `reference_domain_errors` path -- no PROCESS unit test does.
        legacy_sample(
            "bi2212-out-of-range",
            b_conductor=3.0,
            jstrand=2.0e7,
            temp_conductor=25.0,
            f_strain=0.2,
        ),
    ]

    fuzz_bounds = {
        "b_conductor": (6.0, 20.0),
        "jstrand": (1.0e6, 5.0e7),
        "temp_conductor": (1.0, 20.0),
        "f_strain": (0.1, 1.0),
    }


class TestGlNbti(Tier1Contract):
    """`gl_nbti` -> the same, unchanged.

    Legacy sample is `test_gl_nbti`'s case verbatim (note: that test's own unpacking
    order, `jcrit, tcrit, bcrit = gl_nbti(...)`, is a naming mismatch in the *test* --
    `gl_nbti` actually returns `(j_critical, b_critical, t_critical)`, matched here
    and in the port's own docstring). Fuzz bounds span both `b_reduced <= 1` / `> 1`
    branches.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.gl_nbti)
    ported = gl_nbti

    samples = [
        legacy_sample(
            "gl-nbti-reference",
            temp_conductor=4.75,
            b_conductor=7.0,
            strain=2,
            b_c20max=9.5,
            t_c0=13.75,
        ),
    ]

    fuzz_bounds = {
        "temp_conductor": (1.0, 9.0),
        "b_conductor": (0.5, 12.0),
        "strain": (0.5, 3.0),
        "b_c20max": (7.0, 12.0),
        "t_c0": (10.0, 16.0),
    }


class TestGlRebco(Tier1Contract):
    """`gl_rebco` -> the same, unchanged.

    Legacy sample is `test_gl_rebco`'s case verbatim. Fuzz bounds are kept inside the
    physically valid `b_reduced < 1` region -- see the audit record's JAX-difficulty
    flags for why: the source itself has no branch guarding `b_reduced > 1`, and returns
    a *complex* number there (Python's `**` on a negative base with a non-integer
    exponent), which `jnp`'s `**` would instead turn into `nan` -- a type-level
    disagreement neither side of this port is trying to reproduce.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.gl_rebco)
    ported = gl_rebco

    samples = [
        legacy_sample(
            "gl-rebco-reference",
            temp_conductor=4.75,
            b_conductor=7.0,
            strain=2,
            b_c20max=30.0,
            t_c0=25.0,
        ),
    ]

    fuzz_bounds = {
        "temp_conductor": (1.0, 8.0),
        "b_conductor": (0.5, 3.0),
        "strain": (0.5, 3.0),
        "b_c20max": (28.0, 32.0),
        "t_c0": (24.0, 28.0),
    }


class TestWesternSuperconductingNb3Sn(Tier1Contract):
    """`western_superconducting_nb3sn` -> the same, unchanged.

    Legacy sample is `test_wstsc`'s case verbatim.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.western_superconducting_nb3sn)
    ported = western_superconducting_nb3sn

    samples = [
        legacy_sample(
            "western-nb3sn-reference",
            temp_conductor=4.75,
            b_conductor=27.0,
            strain=0.001,
            b_c20max=30.0,
            temp_c0max=25.0,
        ),
    ]

    fuzz_bounds = {
        "temp_conductor": (1.0, 15.0),
        "b_conductor": (0.5, 26.0),
        "strain": (-0.005, 0.005),
        "b_c20max": (25.0, 35.0),
        "temp_c0max": (14.0, 18.0),
    }


class TestHijcRebco(Tier1Contract):
    """`hijc_rebco` -> the same, unchanged.

    Legacy sample is `test_hijc_rebco`'s case verbatim. The ninth material, ported
    2026-08-30 as `models/tfcoil/croco.py`'s critical surface -- `i_tf_sc_mat == 9` is
    what both tracked spherical tokamaks set, and nothing on the cable-in-conduit side
    ever reaches this fit.

    **PROCESS's two-arm `cur_critical` is one expression in the port** and the fuzz range
    exercises both sides of the seam it replaces: `b_conductor` spans below and above
    `b_critical`, which is the branch PROCESS writes as two formulas differing only in
    the sign that keeps `|1 - B/B_c|`'s base non-negative. Bounds stay under `t_c0` for
    the same reason `TestGlRebco`'s do: above it PROCESS returns a *complex* number from
    `(1 - T/T_c0) ** 1.4` and the port returns `nan`, a type-level disagreement neither
    side is trying to reproduce.
    """

    audit_record = "models/physics/superconductors.md"
    reference = staticmethod(ref.hijc_rebco)
    ported = hijc_rebco

    samples = [
        legacy_sample(
            "hijc-rebco-reference",
            temp_conductor=4.75,
            b_conductor=7.0,
            b_c20max=30.0,
            t_c0=25.0,
            dr_hts_tape=4.0e-3,
            dx_hts_tape_rebco=1.0e-6,
            dx_hts_tape_total=6.5e-5,
        ),
        legacy_sample(
            "hijc-rebco-croco-tape",
            temp_conductor=4.75,
            b_conductor=11.7,
            b_c20max=138.0,
            t_c0=92.0,
            dr_hts_tape=6.28e-3,
            dx_hts_tape_rebco=1.0e-6,
            dx_hts_tape_total=2.11e-4,
        ),
    ]

    fuzz_bounds = {
        "temp_conductor": (4.0, 60.0),
        "b_conductor": (2.0, 30.0),
        "b_c20max": (120.0, 150.0),
        "t_c0": (85.0, 95.0),
        "dr_hts_tape": (2.0e-3, 8.0e-3),
        "dx_hts_tape_rebco": (5.0e-7, 2.0e-6),
        "dx_hts_tape_total": (5.0e-5, 3.0e-4),
    }
