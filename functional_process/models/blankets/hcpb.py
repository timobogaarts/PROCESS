"""Pure-functional port of `process/models/blankets/hcpb.py`'s in-scope methods.

Registry unit #13. In scope: `nuclear_heating_blanket`, `nuclear_heating_shield`,
`nuclear_heating_magnets` -- see `functional_process/models/blankets/hcpb.md` (read it
first), which found all three self-contained tier-1 (no calls into other models, no
internal iteration) and ported all three. This was the sole blocker on `st_fwbs`'s S2
sub-computation (`stellarator_E_fwbs_synthesis.md`); the record also flags a live
call-site bug in `stellarator.py`'s `blanket_neutronics()` that S2's port will hit next
(see the record's "open questions" #1) -- not a blocker for this unit, since the port
targets the staticmethods' own declared signatures, the ones `CCFE_HCPB.run()`'s own call
sites already use correctly.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import build, ccfe_hcpb, fwbs, physics, tfcoil
from process.core import constants


def nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw):
    """Nuclear heating in the blanket (MW). Ports the `@staticmethod` of the same name
    verbatim -- already pure in the source, no `self` to close.

    The source's `logger.error(...)` diagnostic (gated on
    `p_blkt_nuclear_heat_total_mw < 1`) is dropped -- see `hcpb.md`'s JAX-difficulty
    flags: a Python-level side effect conditioned on a traced value, with no effect on
    the return value.

    Parameters
    ----------
    m_blkt_total :
        Total mass of the blanket (kg). `.fwbs.m_blkt_total`.
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.

    Returns
    -------
    tuple
        `(p_blkt_nuclear_heat_total_mw, exp_blanket)`.
    """
    a = 0.764
    b = 2.476e-3  # 1/tonne

    m_blkt_total_tonnes = m_blkt_total / 1000

    exp_blanket = 1 - jnp.exp(-b * m_blkt_total_tonnes)
    p_blkt_nuclear_heat_total_mw = p_fusion_total_mw * a * exp_blanket

    return p_blkt_nuclear_heat_total_mw, exp_blanket


def nuclear_heating_shield(
    itart,
    dr_shld_outboard,
    dr_shld_inboard,
    shield_density,
    whtshld,
    x_blanket,
    p_fusion_total_mw,
):
    """Nuclear heating in the shield (MW). Ports the `@staticmethod` of the same name
    verbatim -- already pure in the source, no `self` to close.

    The `itart` branch (source: `if itart == 1: ... else: ...`, selecting whether the
    average shield thickness uses the outboard value alone or the inboard/outboard mean)
    becomes `jnp.where` -- see `hcpb.md`'s "switches touched" for why this stays a plain
    traced argument rather than a topology split: the source itself already unifies both
    branches in one function.

    Parameters
    ----------
    itart :
        Spherical-tokamak indicator (1 if ST, else 0). `.physics.itart`.
    dr_shld_outboard, dr_shld_inboard :
        Outboard/inboard shield thickness (m). `.build.dr_shld_outboard`/`_inboard`.
    shield_density :
        Shield smeared density (kg/m^3). `.ccfe_hcpb.shield_density` --
        `NuclearHeatingMagnets`'s output at the real call site.
    whtshld :
        Shield mass (kg). `.fwbs.whtshld`.
    x_blanket :
        Blanket line density (tonne/m^2). `.ccfe_hcpb.x_blanket` --
        `NuclearHeatingMagnets`'s output at the real call site.
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.

    Returns
    -------
    tuple
        `(p_shld_nuclear_heat_mw, exp_shield1, exp_shield2, shld_u_nuc_heating)`.
    """
    f = 6.88e2  # Shield nuclear heating coefficient (W/kg/W)
    g = 2.723  # Shield nuclear heating exponent (m^2/tonne)
    h = 0.798  # Shield nuclear heating exponent (m^2/tonne)

    is_st = itart == 1
    dr_shld_average = jnp.where(
        is_st, dr_shld_outboard, 0.5 * (dr_shld_outboard + dr_shld_inboard)
    )

    # Decay length (m^-2)
    y = (shield_density / 1000) * dr_shld_average

    exp_shield1 = jnp.exp(-g * x_blanket)
    exp_shield2 = jnp.exp(-h * y)
    shld_u_nuc_heating = whtshld * f * exp_shield1 * exp_shield2

    p_shld_nuclear_heat_mw = shld_u_nuc_heating * (p_fusion_total_mw / 1000) / 1.0e6

    return p_shld_nuclear_heat_mw, exp_shield1, exp_shield2, shld_u_nuc_heating


def calculate_nuclear_heating_magnets(
    radius_fw_channel,
    dx_fw_module,
    dr_fw_inboard,
    dr_fw_outboard,
    den_steel,
    m_blkt_total,
    vol_blkt_total,
    whtshld,
    vol_shld_total,
    dr_vv_inboard,
    dr_vv_outboard,
    m_vv,
    vol_vv,
    itart,
    dr_blkt_outboard,
    dr_blkt_inboard,
    dr_shld_outboard,
    dr_shld_inboard,
    fw_armour_thickness,
    whttflgs,
    m_tf_coils_total,
    p_fusion_total_mw,
):
    """Nuclear heating in the magnets. Closes `nuclear_heating_magnets`'s `self.data`
    back-door -- unlike its two siblings above, the source method is `self`-bound, not a
    `@staticmethod`, so this is a genuine extraction (`calculate_` prefix per
    `_audit/naming_convention.md`, same as `calculate_density_limit` for an instance
    method with no pre-existing pure core).

    Two `jnp.where`s ported from the source's Python `if`s:

    - `vv_density`'s divide-by-`vol_vv`, guarded so the untaken branch (`d_vv_all <=
      1e-6`, where `vol_vv` may legitimately be `0.0`) cannot leak a NaN through the
      gradient -- the source's `if` short-circuits the division entirely in that case,
      which a traced `jnp.where` cannot do (see `hcpb.md`'s JAX-difficulty flags).
    - the `itart` branch, applied twice (`th_blanket_av`/`th_shield_av`, and the
      TF-coil-mass source for `tfc_nuc_heating`) -- same switch, same treatment as
      `nuclear_heating_shield`'s own `itart` branch.

    Parameters
    ----------
    radius_fw_channel, dx_fw_module, dr_fw_inboard, dr_fw_outboard :
        FW coolant channel geometry (m). `.fwbs.radius_fw_channel`, `.fwbs.dx_fw_module`,
        `.build.dr_fw_inboard`, `.build.dr_fw_outboard`.
    den_steel :
        Steel density (kg/m^3). `.fwbs.den_steel`.
    m_blkt_total, vol_blkt_total :
        Blanket mass (kg) / volume (m^3). `.fwbs.m_blkt_total`, `.fwbs.vol_blkt_total`.
    whtshld, vol_shld_total :
        Shield mass (kg) / volume (m^3). `.fwbs.whtshld`, `.fwbs.vol_shld_total`.
    dr_vv_inboard, dr_vv_outboard, m_vv, vol_vv :
        Vacuum vessel thickness (m) / mass (kg) / volume (m^3). `.build.dr_vv_inboard`,
        `.build.dr_vv_outboard`, `.fwbs.m_vv`, `.fwbs.vol_vv`.
    itart :
        Spherical-tokamak indicator (1 if ST, else 0). `.physics.itart`.
    dr_blkt_outboard, dr_blkt_inboard :
        Blanket thickness (m). `.build.dr_blkt_outboard`, `.build.dr_blkt_inboard`.
    dr_shld_outboard, dr_shld_inboard :
        Shield thickness (m). `.build.dr_shld_outboard`, `.build.dr_shld_inboard`.
    fw_armour_thickness :
        FW armour thickness (m). `.fwbs.fw_armour_thickness`.
    whttflgs :
        TF coil outboard leg mass (kg), ST only. `.tfcoil.whttflgs`.
    m_tf_coils_total :
        Total TF coil mass (kg), non-ST. `.tfcoil.m_tf_coils_total`.
    p_fusion_total_mw :
        Total fusion power (MW). `.physics.p_fusion_total_mw`.

    Returns
    -------
    tuple
        `(f_a_fw_coolant_inboard, f_a_fw_coolant_outboard, armour_density, fw_density,
        blanket_density, shield_density, vv_density, x_blanket, x_shield,
        tfc_nuc_heating, p_tf_nuclear_heat_mw)`.
    """
    # Model factors and coefficients
    a = 2.830  # Exponential factor (m2/tonne)
    b = 0.583  # Exponential factor (m2/tonne)
    e = 9.062  # Pre-factor (1/kg)

    # First wall void fractions
    f_a_fw_coolant_inboard = (
        jnp.pi * radius_fw_channel**2 / (dx_fw_module * dr_fw_inboard)
    )
    f_a_fw_coolant_outboard = f_a_fw_coolant_inboard
    vffwm = f_a_fw_coolant_inboard

    # Smeared densities of blanket sections
    armour_density = constants.DEN_TUNGSTEN * (1.0 - vffwm)
    fw_density = den_steel * (1.0 - vffwm)
    blanket_density = m_blkt_total / vol_blkt_total
    shield_density = whtshld / vol_shld_total

    # Picking the largest value for VV thickness
    d_vv_all = jnp.maximum(dr_vv_inboard, dr_vv_outboard)
    vv_density = jnp.where(
        d_vv_all > 1.0e-6,
        m_vv / jnp.where(vol_vv == 0.0, 1.0, vol_vv),  # noqa: RUF069
        0.0,
    )

    # Average blanket/shield thickness (m)
    is_st = itart == 1
    th_blanket_av = jnp.where(
        is_st, dr_blkt_outboard, 0.5 * (dr_blkt_outboard + dr_blkt_inboard)
    )
    th_shield_av = jnp.where(
        is_st, dr_shld_outboard, 0.5 * (dr_shld_outboard + dr_shld_inboard)
    )

    # Exponents (tonne/m2)
    x_blanket = (
        armour_density * fw_armour_thickness
        + fw_density * (dr_fw_inboard + dr_fw_outboard) / 2.0
        + blanket_density * th_blanket_av
    ) / 1000.0

    x_shield = (
        shield_density * th_shield_av
        + vv_density * (dr_vv_inboard + dr_vv_outboard) / 2.0
    ) / 1000.0

    # ST: outboard TF coil legs mass only. Non-ST: total TF coil mass.
    tf_coil_mass = jnp.where(is_st, whttflgs, m_tf_coils_total)
    tfc_nuc_heating = e * jnp.exp(-a * x_blanket) * jnp.exp(-b * x_shield) * tf_coil_mass

    # Total heating (MW)
    p_tf_nuclear_heat_mw = tfc_nuc_heating * (p_fusion_total_mw / 1000.0) / 1.0e6

    return (
        f_a_fw_coolant_inboard,
        f_a_fw_coolant_outboard,
        armour_density,
        fw_density,
        blanket_density,
        shield_density,
        vv_density,
        x_blanket,
        x_shield,
        tfc_nuc_heating,
        p_tf_nuclear_heat_mw,
    )


class NuclearHeatingBlanket(ExplicitFunction):
    """cottax node: `nuclear_heating_blanket`, unchanged."""

    p_blkt_nuclear_heat_total_mw = OutputInto(fwbs)
    exp_blanket = OutputInto(ccfe_hcpb)

    def __call__(
        self,
        m_blkt_total=From(fwbs),
        p_fusion_total_mw=From(physics),
    ):
        return nuclear_heating_blanket(m_blkt_total, p_fusion_total_mw)


class NuclearHeatingShield(ExplicitFunction):
    """cottax node: `nuclear_heating_shield`, unchanged.

    `shield_density`/`x_blanket` are `NuclearHeatingMagnets`'s own outputs at the real
    call site -- an ordinary graph edge, magnets before shield, matching the call order
    both `CCFE_HCPB.run()` and `stellarator.py`'s `blanket_neutronics()` use.
    """

    p_shld_nuclear_heat_mw = OutputInto(fwbs)
    exp_shield1 = OutputInto(ccfe_hcpb)
    exp_shield2 = OutputInto(ccfe_hcpb)
    shld_u_nuc_heating = OutputInto(ccfe_hcpb)

    def __call__(
        self,
        itart=From(physics),
        dr_shld_outboard=From(build),
        dr_shld_inboard=From(build),
        shield_density=From(ccfe_hcpb),
        whtshld=From(fwbs),
        x_blanket=From(ccfe_hcpb),
        p_fusion_total_mw=From(physics),
    ):
        return nuclear_heating_shield(
            itart,
            dr_shld_outboard,
            dr_shld_inboard,
            shield_density,
            whtshld,
            x_blanket,
            p_fusion_total_mw,
        )


class NuclearHeatingMagnets(ExplicitFunction):
    """cottax node: `calculate_nuclear_heating_magnets`.

    See `hcpb.md`'s data-footprint table for the full 21-input, 11-output ledger. Two
    reused `VarPath`s round-trip through `NuclearHeatingShield`: `shield_density` and
    `x_blanket`.
    """

    f_a_fw_coolant_inboard = OutputInto(fwbs)
    f_a_fw_coolant_outboard = OutputInto(fwbs)
    armour_density = OutputInto(ccfe_hcpb)
    fw_density = OutputInto(ccfe_hcpb)
    blanket_density = OutputInto(ccfe_hcpb)
    shield_density = OutputInto(ccfe_hcpb)
    vv_density = OutputInto(ccfe_hcpb)
    x_blanket = OutputInto(ccfe_hcpb)
    x_shield = OutputInto(ccfe_hcpb)
    tfc_nuc_heating = OutputInto(ccfe_hcpb)
    p_tf_nuclear_heat_mw = OutputInto(fwbs)

    def __call__(
        self,
        radius_fw_channel=From(fwbs),
        dx_fw_module=From(fwbs),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        den_steel=From(fwbs),
        m_blkt_total=From(fwbs),
        vol_blkt_total=From(fwbs),
        whtshld=From(fwbs),
        vol_shld_total=From(fwbs),
        dr_vv_inboard=From(build),
        dr_vv_outboard=From(build),
        m_vv=From(fwbs),
        vol_vv=From(fwbs),
        itart=From(physics),
        dr_blkt_outboard=From(build),
        dr_blkt_inboard=From(build),
        dr_shld_outboard=From(build),
        dr_shld_inboard=From(build),
        fw_armour_thickness=From(fwbs),
        whttflgs=From(tfcoil),
        m_tf_coils_total=From(tfcoil),
        p_fusion_total_mw=From(physics),
    ):
        return calculate_nuclear_heating_magnets(
            radius_fw_channel,
            dx_fw_module,
            dr_fw_inboard,
            dr_fw_outboard,
            den_steel,
            m_blkt_total,
            vol_blkt_total,
            whtshld,
            vol_shld_total,
            dr_vv_inboard,
            dr_vv_outboard,
            m_vv,
            vol_vv,
            itart,
            dr_blkt_outboard,
            dr_blkt_inboard,
            dr_shld_outboard,
            dr_shld_inboard,
            fw_armour_thickness,
            whttflgs,
            m_tf_coils_total,
            p_fusion_total_mw,
        )
