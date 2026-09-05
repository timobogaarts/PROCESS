"""Pure physics functions extracted from
`functional_process.cottax.stellarator.heating`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""

import jax.numpy as jnp

_ZERO_DIVISION_GUARD_MW = 1e-8
"""Below this, the source treats injected beam power as exactly zero (line 112)."""


_Q_DEGENERATE_GUARD_MW = 1e-6
"""Below this denominator, the source reports `big_q_plasma = 1e18` rather than divide."""


_Q_DEGENERATE_VALUE = 1e18


def calculate_ecrh_heating(p_hcd_primary_extra_heat_mw, eta_ecrh_injector_wall_plug):
    """Auxiliary heating power split for `isthtr == 1` (ECRH).

    Ports the `isthtr == 1` branch of `st_heat`.

    Parameters
    ----------
    p_hcd_primary_extra_heat_mw :
        Auxiliary power supplied to the plasma (MW). `.current_drive.p_hcd_primary_extra_heat_mw`.
    eta_ecrh_injector_wall_plug :
        ECRH injector wall-plug efficiency. `.current_drive.eta_ecrh_injector_wall_plug`.

    Returns
    -------
    :
        `(p_hcd_ecrh_injected_total_mw, p_hcd_injected_ions_mw,
        p_hcd_injected_electrons_mw, eta_hcd_primary_injector_wall_plug,
        p_hcd_electric_total_mw)`.
    """
    p_hcd_ecrh_injected_total_mw = p_hcd_primary_extra_heat_mw
    p_hcd_injected_ions_mw = 0.0
    p_hcd_injected_electrons_mw = p_hcd_ecrh_injected_total_mw
    eta_hcd_primary_injector_wall_plug = eta_ecrh_injector_wall_plug
    p_hcd_electric_total_mw = (
        p_hcd_injected_ions_mw + p_hcd_injected_electrons_mw
    ) / eta_hcd_primary_injector_wall_plug

    return (
        p_hcd_ecrh_injected_total_mw,
        p_hcd_injected_ions_mw,
        p_hcd_injected_electrons_mw,
        eta_hcd_primary_injector_wall_plug,
        p_hcd_electric_total_mw,
    )


def calculate_lowhyb_heating(p_hcd_primary_extra_heat_mw, eta_lowhyb_injector_wall_plug):
    """Auxiliary heating power split for `isthtr == 2` (lower hybrid).

    Ports the `isthtr == 2` branch of `st_heat`. Same shape as
    `calculate_ecrh_heating`, mints a differently-named total.

    Parameters
    ----------
    p_hcd_primary_extra_heat_mw :
        Auxiliary power supplied to the plasma (MW). `.current_drive.p_hcd_primary_extra_heat_mw`.
    eta_lowhyb_injector_wall_plug :
        Lower-hybrid injector wall-plug efficiency. `.current_drive.eta_lowhyb_injector_wall_plug`.

    Returns
    -------
    :
        `(p_hcd_lowhyb_injected_total_mw, p_hcd_injected_ions_mw,
        p_hcd_injected_electrons_mw, eta_hcd_primary_injector_wall_plug,
        p_hcd_electric_total_mw)`.
    """
    p_hcd_lowhyb_injected_total_mw = p_hcd_primary_extra_heat_mw
    p_hcd_injected_ions_mw = 0.0
    p_hcd_injected_electrons_mw = p_hcd_lowhyb_injected_total_mw
    eta_hcd_primary_injector_wall_plug = eta_lowhyb_injector_wall_plug
    p_hcd_electric_total_mw = (
        p_hcd_injected_ions_mw + p_hcd_injected_electrons_mw
    ) / eta_hcd_primary_injector_wall_plug

    return (
        p_hcd_lowhyb_injected_total_mw,
        p_hcd_injected_ions_mw,
        p_hcd_injected_electrons_mw,
        eta_hcd_primary_injector_wall_plug,
        p_hcd_electric_total_mw,
    )


def calculate_injected_power_total(p_hcd_injected_electrons_mw, p_hcd_injected_ions_mw):
    """Total injected heating/current-drive power.

    Ports `st_heat`'s "Total injected power" step (lines 105-108), common to all three
    `isthtr` branches.

    Parameters
    ----------
    p_hcd_injected_electrons_mw, p_hcd_injected_ions_mw :
        Injected electron/ion power (MW). `.current_drive.p_hcd_injected_electrons_mw`,
        `.current_drive.p_hcd_injected_ions_mw`.

    Returns
    -------
    :
        `p_hcd_injected_total_mw` (MW).
    """
    return p_hcd_injected_electrons_mw + p_hcd_injected_ions_mw


def calculate_beam_current(p_hcd_beam_injected_total_mw, e_beam_kev):
    """Neutral beam current.

    Ports `st_heat`'s "Calculate neutral beam current" step (lines 112-119). For
    ECRH/lower-hybrid, callers pass `p_hcd_beam_injected_total_mw=0.0` (see module
    docstring), which the `jnp.where` below already sends to `0.0` explicitly rather
    than relying on that value's magnitude.

    Parameters
    ----------
    p_hcd_beam_injected_total_mw :
        Injected neutral beam power (MW). `.current_drive.p_hcd_beam_injected_total_mw`.
    e_beam_kev :
        Neutral beam energy (keV). `.current_drive.e_beam_kev`.

    Returns
    -------
    :
        `c_beam_total`, the neutral beam current (A).
    """
    return jnp.where(
        jnp.abs(p_hcd_beam_injected_total_mw) > _ZERO_DIVISION_GUARD_MW,
        1e-3 * (p_hcd_beam_injected_total_mw * 1e6) / e_beam_kev,
        0.0,
    )


def calculate_fusion_gain(
    p_fusion_total_mw,
    p_hcd_injected_total_mw,
    p_beam_orbit_loss_mw,
    p_plasma_ohmic_mw,
):
    """Fusion gain factor Q.

    Ports `st_heat`'s "Ratio of fusion to input power" step (lines 123-137). For
    ECRH/lower-hybrid, callers pass `p_beam_orbit_loss_mw=0.0` (see module docstring).

    Parameters
    ----------
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.
    p_hcd_injected_total_mw :
        Total injected heating/current-drive power (MW).
        `.current_drive.p_hcd_injected_total_mw`.
    p_beam_orbit_loss_mw :
        Neutral beam orbit loss power (MW). `.current_drive.p_beam_orbit_loss_mw`.
    p_plasma_ohmic_mw :
        Ohmic heating power (MW). `.physics.p_plasma_ohmic_mw`.

    Returns
    -------
    :
        `big_q_plasma`, the fusion gain factor.
    """
    denominator = p_hcd_injected_total_mw + p_beam_orbit_loss_mw + p_plasma_ohmic_mw
    degenerate = jnp.abs(denominator) < _Q_DEGENERATE_GUARD_MW
    # The *double* `jnp.where`, and the inner one is load-bearing. A single select still
    # evaluates `p_fusion / 0` on the untaken arm: `inf` in value and `-inf` in
    # `d/d denominator`. Forward mode multiplies and then selects, so it discards both;
    # the transposed select runs first and then multiplies, and `0 * inf` is `nan`. So
    # the single spelling is finite under `jacfwd` and non-finite under `jacrev` at the
    # same point (`_audit/optimise_design.md` §33). Substituting `1.0` changes no
    # value: the arm it appears on is the one the outer select discards.
    safe_denominator = jnp.where(degenerate, 1.0, denominator)
    return jnp.where(
        degenerate,
        _Q_DEGENERATE_VALUE,
        p_fusion_total_mw / safe_denominator,
    )
