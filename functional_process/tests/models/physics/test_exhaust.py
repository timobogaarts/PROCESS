"""Harness cases for the ported plasma radiation fraction (registry unit #11).

No PROCESS unit test exists for `calculate_radiation_fraction` (fuzz-only, same
situation `build.md`/several `coils/*.md` units already recorded for their own units).
`PlasmaExhaust.calculate_radiation_fraction` is called directly as the reference -- it
takes no `self.data` access at all, so no adapter is needed.
"""

from functional_process.cottax._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.cottax.physics.exhaust import (
    calculate_eu_demo_re_attachment_metric,
    calculate_psep_over_r_metric,
    calculate_radiation_fraction,
)
from process.models.physics.exhaust import PlasmaExhaust


class TestRadiationFraction(Tier1Contract):
    """`calculate_radiation_fraction` -> the same, unchanged.

    `reference_domain_errors` is not used: PROCESS returns a real, finite `0.0` (plus a
    logged warning) at `p_plasma_heating_mw == 0` rather than raising, so this is an
    ordinary `jnp.where` branch, not a traced-domain-error case. The
    `zero-heating-power` sample exercises exactly that branch, both value and gradient
    (checking the safe-denominator trick actually avoids a NaN gradient leak -- see the
    port's docstring).
    """

    audit_record = "models/physics/exhaust.md"
    reference = staticmethod(PlasmaExhaust.calculate_radiation_fraction)
    ported = calculate_radiation_fraction

    samples = [
        legacy_sample(
            "typical-operating-point", p_plasma_rad_mw=250.0, p_plasma_heating_mw=450.0
        ),
        legacy_sample(
            "radiation-exceeds-heating-power",
            p_plasma_rad_mw=500.0,
            p_plasma_heating_mw=450.0,
        ),
        legacy_sample(
            "zero-heating-power", p_plasma_rad_mw=250.0, p_plasma_heating_mw=0.0
        ),
    ]

    fuzz_bounds = {
        "p_plasma_rad_mw": (0.0, 1000.0),
        "p_plasma_heating_mw": (1.0, 1000.0),
    }


class TestEuDemoReAttachmentMetric(Tier1Contract):
    """`P_sep*B_t / (q95*A*R0)`, `exhaust.py:150-192`. No adapter -- the static takes no
    `self.data` access either.

    The legacy point is `large_tokamak_eval` at PROCESS's own solution, and it is worth
    naming what it is: **the point at which constraint 68 is violated**. PROCESS's
    converged `p_div_bt_q_aspect_rmajor_mw = 10.4949` against the file's bound of 10,
    normalised residual `+4.949e-02` -- one of the two inequalities the evaluation-mode
    run reports and does not enforce (`optimise_design.md` §11.1). So the value test
    here is a direct diff against the number the SAND Stage A comparison uses.

    No PROCESS unit test exists for this static, so the rest is fuzz. `q95`, `aspect`
    and `rmajor` are bounded away from zero: the source divides by their product with
    no guard and PROCESS would produce an `inf` too, which is faithful but not an
    interesting sample.
    """

    audit_record = "models/physics/exhaust.md"
    reference = staticmethod(PlasmaExhaust.calculate_eu_demo_re_attachment_metric)
    ported = calculate_eu_demo_re_attachment_metric

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged-and-violating",
            p_plasma_separatrix_mw=176.83927443617988,
            b_plasma_toroidal_on_axis=5.318322174646137,
            q95=3.7339078191146117,
            aspect=3.0,
            rmajor=8.0,
        ),
        *fuzz_samples(
            {
                "p_plasma_separatrix_mw": (10.0, 500.0),
                "b_plasma_toroidal_on_axis": (1.0, 15.0),
                "q95": (2.0, 10.0),
                "aspect": (1.5, 5.0),
                "rmajor": (2.0, 20.0),
            },
            count=5,
            seed=68,
        ),
    ]

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "b_plasma_toroidal_on_axis": (1.0, 15.0),
        "q95": (2.0, 10.0),
        "aspect": (1.5, 5.0),
        "rmajor": (2.0, 20.0),
    }


class TestPsepOverRMetric(Tier1Contract):
    """`P_sep / R0`, `exhaust.py:127-147`. No adapter -- another bare static.

    **Both legacy points are PROCESS's own converged answer on a file where
    constraint 56 is active, and both are worth naming.** `st_regression` sits at
    `39.99999999988` against a bound of `40` -- the constraint is *active*, exactly on
    its bound, and it is the single most binding constraint of that problem;
    `spherical_tokamak_eval` reads `40.2816` against the same bound, i.e. PROCESS
    **violates** it at its own answer (evaluation mode, so nothing enforces it). Until
    2026-09-01 this port had no producer for the path at all and read a frozen `0.0`,
    which the `leq` reported as satisfied with the whole of its margin to spare
    (`optimise_design.md` §26.3 ranks 2 and 3, §29 for what porting it moved).

    The `p_plasma_separatrix_mw` written down here is PROCESS's *converged* field --
    i.e. post-KLUDGE. The node reads the pre-KLUDGE mint
    `.physics.p_plasma_separatrix_mw_raw` instead, and at 180 MW the two differ by
    ~1e-79; see `PsepOverRMetric`'s docstring for why the distinction is kept anyway.
    No test in this port can see it, and this case does not pretend to.

    `rmajor` is bounded away from zero in the fuzz box: PROCESS divides by it with no
    guard and would produce an `inf` too, which is faithful but uninteresting.
    """

    audit_record = "models/physics/exhaust.md"
    reference = staticmethod(PlasmaExhaust.calculate_psep_over_r_metric)
    ported = calculate_psep_over_r_metric

    samples = [
        legacy_sample(
            "st_regression-converged-and-active",
            p_plasma_separatrix_mw=179.99999999946084,
            rmajor=4.5,
        ),
        legacy_sample(
            "spherical_tokamak_eval-converged-and-violating",
            p_plasma_separatrix_mw=181.2672921059313,
            rmajor=4.5,
        ),
        *fuzz_samples(
            {
                "p_plasma_separatrix_mw": (10.0, 500.0),
                "rmajor": (2.0, 20.0),
            },
            count=5,
            seed=56,
        ),
    ]

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "rmajor": (2.0, 20.0),
    }
