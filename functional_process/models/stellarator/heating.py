"""Pure-functional port of `process/models/stellarator/heating.py` (registry unit #5).

Audit record: `functional_process/models/stellarator/heating.md`. Only the `isthtr`
branches that are self-contained (call no other model) are ported: ECRH (`isthtr == 1`)
and lower-hybrid (`isthtr == 2`). The NBI branch (`isthtr == 3`) calls
`stellarator.current_drive.culnbi()` -- a not-yet-audited model -- so it stays audit-only
per the record.

`calculate_ecrh_heating`/`calculate_lowhyb_heating` are **mutually exclusive
alternatives**: both write the same four downstream fields
(`p_hcd_injected_ions_mw`, `p_hcd_injected_electrons_mw`,
`eta_hcd_primary_injector_wall_plug`, `p_hcd_electric_total_mw`), because `isthtr`
selects exactly one of the three source branches to run -- never more than one node from
{`EcrhHeating`, `LowhybHeating`, an eventual NBI node} belongs in an assembled graph at
once. This is the same "switch picks which node exists" shape as `i_tf_sup` in
`stellarator_F_tf_nuclear_heating.py`, not a naming collision to fix.

`calculate_beam_current`/`calculate_fusion_gain` (the source's common tail, after the
`isthtr` branch) read `p_hcd_beam_injected_total_mw`/`p_beam_orbit_loss_mw`, which only
the NBI branch ever assigns -- for ECRH/lowhyb, the source relies on these fields already
being (or defaulting to) `0.0` on `data`, an implicit-io dependency on external
initialisation rather than a value this file produces. Ported as explicit arguments
(caller supplies `0.0` for the non-beam case) rather than silently modelled as always
zero -- see the record's open questions.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import current_drive, heat_transport, physics

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
    return jnp.where(
        jnp.abs(denominator) < _Q_DEGENERATE_GUARD_MW,
        _Q_DEGENERATE_VALUE,
        p_fusion_total_mw / denominator,
    )


class EcrhHeating(ExplicitFunction):
    """cottax node: `calculate_ecrh_heating`, unchanged, ports declared.

    See module docstring -- mutually exclusive with `LowhybHeating` (and an eventual
    NBI node): `isthtr` selects at most one at graph-assembly time.
    """

    p_hcd_ecrh_injected_total_mw = OutputInto(current_drive)
    p_hcd_injected_ions_mw = OutputInto(current_drive)
    p_hcd_injected_electrons_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_ecrh_injector_wall_plug=From(current_drive),
    ):
        return calculate_ecrh_heating(
            p_hcd_primary_extra_heat_mw, eta_ecrh_injector_wall_plug
        )


class LowhybHeating(ExplicitFunction):
    """cottax node: `calculate_lowhyb_heating`, unchanged, ports declared.

    See module docstring -- mutually exclusive with `EcrhHeating`.
    """

    p_hcd_lowhyb_injected_total_mw = OutputInto(current_drive)
    p_hcd_injected_ions_mw = OutputInto(current_drive)
    p_hcd_injected_electrons_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_lowhyb_injector_wall_plug=From(current_drive),
    ):
        return calculate_lowhyb_heating(
            p_hcd_primary_extra_heat_mw, eta_lowhyb_injector_wall_plug
        )


class InjectedPowerTotal(ExplicitFunction):
    """cottax node: `calculate_injected_power_total`, unchanged, ports declared."""

    p_hcd_injected_total_mw = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_injected_electrons_mw=From(current_drive),
        p_hcd_injected_ions_mw=From(current_drive),
    ):
        return calculate_injected_power_total(
            p_hcd_injected_electrons_mw, p_hcd_injected_ions_mw
        )


class BeamCurrent(ExplicitFunction):
    """cottax node: `calculate_beam_current`, unchanged, ports declared.

    Not yet registered in `total_process.py`: for ECRH/lower-hybrid, its
    `p_hcd_beam_injected_total_mw` input has no producing node in the current graph
    (only the not-yet-ported NBI branch would write it) -- see module docstring and the
    audit record's open questions.
    """

    c_beam_total = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_beam_injected_total_mw=From(current_drive),
        e_beam_kev=From(current_drive),
    ):
        return calculate_beam_current(p_hcd_beam_injected_total_mw, e_beam_kev)


class FusionGain(ExplicitFunction):
    """cottax node: `calculate_fusion_gain`, unchanged, ports declared.

    Not yet registered in `total_process.py` -- same reason as `BeamCurrent`.
    """

    big_q_plasma = OutputInto(current_drive)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        p_beam_orbit_loss_mw=From(current_drive),
        p_plasma_ohmic_mw=From(physics),
    ):
        return calculate_fusion_gain(
            p_fusion_total_mw,
            p_hcd_injected_total_mw,
            p_beam_orbit_loss_mw,
            p_plasma_ohmic_mw,
        )
