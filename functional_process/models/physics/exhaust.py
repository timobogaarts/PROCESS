"""Pure-functional port of `process/models/physics/exhaust.py`.

Registry unit #11. Audit record: `functional_process/models/physics/exhaust.md`.

In scope: `calculate_radiation_fraction` only (the registry's stated method). The other
three `PlasmaExhaust` statics (`calculate_separatrix_power`,
`calculate_psep_over_r_metric`, `calculate_eu_demo_re_attachment_metric`) are already
pure and self-contained but out of the registry's stated scope for this unit; not
ported here.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FromExactly,
    Output,
)


def calculate_radiation_fraction(p_plasma_rad_mw, p_plasma_heating_mw):
    """Radiation fraction of the plasma. Ports `PlasmaExhaust.calculate_
    radiation_fraction`, `process/models/physics/exhaust.py:194-220`.

    Source returns `0.0` (and logs a warning) when `p_plasma_heating_mw == 0`, rather
    than raising -- a real domain guard, not PROCESS-signals-invalid-input-by-raising.
    Ported as a genuine `jnp.where` branch (not `reference_domain_errors`): the safe
    denominator avoids the classic "NaN through the untaken branch" trap
    (`_audit/test_harness.md`'s `test_gradient_finite`) that a bare
    `p_plasma_rad_mw / p_plasma_heating_mw` inside the `where` would otherwise leak
    into the gradient even on the taken (nonzero) branch.
    """
    zero_heating = p_plasma_heating_mw == 0
    safe_denominator = jnp.where(zero_heating, 1.0, p_plasma_heating_mw)
    return jnp.where(zero_heating, 0.0, p_plasma_rad_mw / safe_denominator)


class RadiationFraction(ExplicitFunction):
    """cottax node: `calculate_radiation_fraction`, ports declared."""

    f_p_plasma_separatrix_rad = Output(lambda s: s.physics.f_p_plasma_separatrix_rad)

    def __call__(
        self,
        p_plasma_rad_mw=FromExactly(lambda s: s.physics.p_plasma_rad_mw),
        p_plasma_heating_mw=FromExactly(lambda s: s.physics.p_plasma_heating_total_mw),
    ):
        return calculate_radiation_fraction(p_plasma_rad_mw, p_plasma_heating_mw)
