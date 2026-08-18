"""Harness cases for the ported plasma radiation fraction (registry unit #11).

No PROCESS unit test exists for `calculate_radiation_fraction` (fuzz-only, same
situation `build.md`/several `coils/*.md` units already recorded for their own units).
`PlasmaExhaust.calculate_radiation_fraction` is called directly as the reference -- it
takes no `self.data` access at all, so no adapter is needed.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.physics.exhaust import calculate_radiation_fraction
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
