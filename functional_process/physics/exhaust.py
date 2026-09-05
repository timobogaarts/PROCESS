"""Pure physics functions extracted from `models/physics/exhaust.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/exhaust.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp


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


def calculate_eu_demo_re_attachment_metric(
    p_plasma_separatrix_mw,
    b_plasma_toroidal_on_axis,
    q95,
    aspect,
    rmajor,
):
    """The EU-DEMO divertor re-attachment metric, P_sep*B_t / (q95*A*R0) [MW T / m].

    Ports `PlasmaExhaust.calculate_eu_demo_re_attachment_metric`,
    `process/models/physics/exhaust.py:150-192`, unchanged.

    Parameters
    ----------
    p_plasma_separatrix_mw :
        Power crossing the separatrix (MW).
    b_plasma_toroidal_on_axis :
        Toroidal field on the plasma axis (T).
    q95 :
        Safety factor at the 95% flux surface.
    aspect :
        Plasma aspect ratio.
    rmajor :
        Plasma major radius (m).

    Returns
    -------
    :
        The re-attachment metric (MW T / m).
    """
    return (p_plasma_separatrix_mw * b_plasma_toroidal_on_axis) / (q95 * aspect * rmajor)


def calculate_psep_over_r_metric(p_plasma_separatrix_mw, rmajor):
    """Power crossing the separatrix per unit major radius, P_sep / R0 [MW / m].

    Ports `PlasmaExhaust.calculate_psep_over_r_metric`,
    `process/models/physics/exhaust.py:127-147`, unchanged.

    Parameters
    ----------
    p_plasma_separatrix_mw :
        Power crossing the separatrix (MW).
    rmajor :
        Plasma major radius (m).

    Returns
    -------
    :
        `P_sep / R0` (MW/m). `.physics.p_plasma_separatrix_rmajor_mw`.
    """
    return p_plasma_separatrix_mw / rmajor
