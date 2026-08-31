"""Pure-functional port of `process/models/engineering/pumping.py` (partial).

The coolant-hydraulics leaf formulas: Reynolds number, the Haaland friction-factor
approximation, and the Gnielinski heat-transfer correlation that composes the two.

**Filed here, beside `ivc_functions.py`, for that unit's own reason**: these three have
two callers in two different subtrees -- `process/models/fw.py::FirstWall.fw_temp`
(`eurofer97_thermal_conductivity` + `gnielinski_heat_transfer_coefficient`) and
`process/models/blankets/blanket_library.py::BlanketLibrary.
coolant_friction_pressure_drop` (`calculate_reynolds_number` +
`darcy_friction_haaland`) -- and the wave-1 dispatch's convention is that a helper
needed by two or more units is ported once under `models/engineering/`, mirroring
PROCESS, rather than duplicated privately into each caller's module.

**This registers nothing and unblocks no configuration today.** Both call sites sit
behind `.fwbs.i_p_coolant_pumping == 2` (`MECHANICAL`), and no tracked regression input
selects it -- `large_tokamak_eval.IN.DAT:172` sets `3`. `indat.py`'s refusal of that arm
is keyed on **CoolProp** (`_audit/next_steps.md` §5's unresolved wrapping policy), not
on these formulas, and porting them does not lift it: `fw_temp` and
`thermo_hydraulic_model` still call `FluidProperties` for every one of the density,
viscosity, heat-capacity and thermal-conductivity arguments these three consume. What
this file buys is that when that policy is settled, the arithmetic behind it is already
validated -- arbitrary-`IN.DAT` support, not a machine that assembles.

Not ported from the same PROCESS file: `CoolantType`, an `IntEnum` carrying the CoolProp
fluid name. It is a lookup key for the external library, not arithmetic; nothing here
consumes it.

Audit record: `functional_process/_audit/units/models/engineering/pumping.md`.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_pow, safe_sqrt

__all__ = [
    "calculate_reynolds_number",
    "darcy_friction_haaland",
    "gnielinski_heat_transfer_coefficient",
]


def calculate_reynolds_number(*, den_coolant, vel_coolant, radius_channel, visc_coolant):
    """Reynolds number for flow in a pipe. Ports
    `process/models/engineering/pumping.py:175-201`, unchanged.

    Parameters
    ----------
    den_coolant :
        Coolant density, averaged over inlet and outlet (kg/m^3).
    vel_coolant :
        Coolant velocity in a single channel (m/s).
    radius_channel :
        Coolant pipe radius (m). For a non-circular duct the callers pass half the
        hydraulic diameter, which is why this is a radius and not a diameter.
    visc_coolant :
        Coolant dynamic viscosity, averaged over inlet and outlet (Pa.s).

    Returns
    -------
    :
        Reynolds number (dimensionless).
    """
    diameter = 2.0 * radius_channel
    return den_coolant * vel_coolant * diameter / visc_coolant


def darcy_friction_haaland(*, reynolds, roughness_channel, radius_channel):
    """Darcy friction factor from the Haaland equation. Ports
    `process/models/engineering/pumping.py:46-78`, unchanged.

    Parameters
    ----------
    reynolds :
        Reynolds number.
    roughness_channel :
        Absolute roughness of the coolant channel wall (m).
    radius_channel :
        Radius of the coolant channel (m).

    Returns
    -------
    :
        Darcy friction factor (dimensionless).

    Notes
    -----
    An explicit approximation to the implicit Colebrook-White equation; valid for
    turbulent flow. PROCESS applies no domain guard, and neither does this port.

    No `safe_pow` here, and the reason is worth recording rather than leaving to a
    reader's guess. The two exponents are `1.11` and `-2`, and neither is in
    `safe_math`'s `0 < p < 1` window: `x ** 1.11` has derivative `1.11 * x ** 0.11`,
    which is `0` and finite at `x == 0` (the smooth-pipe limit `roughness_channel == 0`,
    which is a physically reachable input), and the `-2` acts on `1.8 * log10(bracket)`,
    which is singular only at `bracket == 1` -- a Reynolds/roughness combination far
    outside any turbulent-flow domain, and an ordinary domain restriction rather than
    the zero-derivative trap `safe_pow` exists for.
    """
    # Bracketed term in the Haaland equation.
    bracket = (roughness_channel / radius_channel / 3.7) ** 1.11 + 6.9 / reynolds

    return (1.8 * jnp.log10(bracket)) ** (-2)


def gnielinski_heat_transfer_coefficient(
    *,
    mflux_coolant,
    den_coolant,
    radius_channel,
    heatcap_coolant,
    visc_coolant,
    thermcond_coolant,
    roughness_channel,
):
    """Heat transfer coefficient from the Gnielinski correlation. Ports
    `process/models/engineering/pumping.py:81-172`.

    Parameters
    ----------
    mflux_coolant :
        Coolant mass flux in a single channel (kg/m^2/s).
    den_coolant :
        Coolant density, averaged over inlet and outlet (kg/m^3).
    radius_channel :
        Coolant pipe radius (m).
    heatcap_coolant :
        Coolant specific heat capacity at constant pressure, averaged over inlet and
        outlet (J/kg/K).
    visc_coolant :
        Coolant dynamic viscosity, averaged over inlet and outlet (Pa.s).
    thermcond_coolant :
        Coolant thermal conductivity, averaged over inlet and outlet (W/m/K).
    roughness_channel :
        Absolute roughness of the coolant channel wall (m).

    Returns
    -------
    :
        Heat transfer coefficient (W/m^2/K).

    Notes
    -----
    Valid for `3000 < Re < 5e6` and `0.5 < Pr < 2000`. **PROCESS's three range checks
    are dropped, and dropping them changes no value**: all three are `logger.error`
    calls placed *after* `heat_transfer_coefficient` is computed and before it is
    returned unmodified (`pumping.py:159-172`). They neither clamp nor raise, so they
    are diagnostics on a value already fixed -- there is nothing here for `jnp.where`
    to express, and a traced body cannot emit them anyway. The bounds are stated above
    so the information survives the port; the fuzz box in this unit's harness case is
    drawn inside them for the same reason.

    `radius_channel=diameter / 2` in PROCESS's internal call to
    `calculate_reynolds_number` is `radius_channel` written the long way round -- kept
    verbatim so the two files diff line for line.
    """
    # Pipe diameter (m).
    diameter = 2.0 * radius_channel

    # Flow velocity (m/s).
    vel_coolant = mflux_coolant / den_coolant

    reynolds = calculate_reynolds_number(
        den_coolant=den_coolant,
        vel_coolant=vel_coolant,
        radius_channel=diameter / 2,
        visc_coolant=visc_coolant,
    )

    # Prandtl number.
    pr = heatcap_coolant * visc_coolant / thermcond_coolant

    f = darcy_friction_haaland(
        reynolds=reynolds,
        roughness_channel=roughness_channel,
        radius_channel=radius_channel,
    )

    # Nusselt number. `safe_sqrt`/`safe_pow` are the two `0 < p < 1` sites in this
    # file: `sqrt(f / 8)` at a vanishing friction factor, and `pr ** (2/3)` at a
    # vanishing Prandtl number. Both are value-identical away from zero and merely
    # replace an `inf` derivative with `0` at it (see `models/safe_math.py`).
    nusselt = (
        (f / 8.0)
        * (reynolds - 1000.0)
        * pr
        / (1.0 + 12.7 * safe_sqrt(f / 8.0) * (safe_pow(pr, 2.0 / 3.0) - 1.0))
    )

    return nusselt * thermcond_coolant / (2.0 * radius_channel)
