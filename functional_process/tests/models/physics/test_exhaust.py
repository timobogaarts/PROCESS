"""Harness cases for the ported plasma radiation fraction (registry unit #11).

No PROCESS unit test exists for `calculate_radiation_fraction` (fuzz-only, same
situation `build.md`/several `coils/*.md` units already recorded for their own units).
`PlasmaExhaust.calculate_radiation_fraction` is called directly as the reference -- it
takes no `self.data` access at all, so no adapter is needed.
"""

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.sample_store import FROM_FILE
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

    samples = FROM_FILE

    fuzz = True


class TestEuDemoReAttachmentMetric(Tier1Contract):
    """`P_sep*B_t / (q95*A*R0)`, `exhaust.py:150-192`. No adapter -- the static takes no
    `self.data` access either.

    No PROCESS unit test exists for this static, so the rest is fuzz. `q95`, `aspect`
    and `rmajor` are bounded away from zero: the source divides by their product with
    no guard and PROCESS would produce an `inf` too, which is faithful but not an
    interesting sample.
    """

    audit_record = "models/physics/exhaust.md"
    reference = staticmethod(PlasmaExhaust.calculate_eu_demo_re_attachment_metric)
    ported = calculate_eu_demo_re_attachment_metric

    samples = FROM_FILE

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "b_plasma_toroidal_on_axis": (1.0, 15.0),
        "q95": (2.0, 10.0),
        "aspect": (1.5, 5.0),
        "rmajor": (2.0, 20.0),
    }


class TestPsepOverRMetric(Tier1Contract):
    """`P_sep / R0`, `exhaust.py:127-147`. No adapter -- another bare static.

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

    samples = FROM_FILE

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (10.0, 500.0),
        "rmajor": (2.0, 20.0),
    }
