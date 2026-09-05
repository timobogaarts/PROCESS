"""Pure physics functions extracted from
`functional_process.cottax.stellarator.tf_nuclear_heating`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""

import jax.numpy as jnp

_FACT = (8.0, 8.0, 6.0, 4.0, 4.0)


_COEF = (10.3, 11.6, 7.08e5, 2.19e18, 3.33e-7)


_DECAY = (10.05, 17.61, 13.82, 13.24, 14.31, 13.26, 13.25)


_SECONDS_PER_FULL_POWER_YEAR = 3.154e7


def calculate_sc_tf_coil_nuclear_heating(
    dr_shld_inboard,
    dr_fw_inboard,
    dr_blkt_inboard,
    dr_shld_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_tf_wp_with_insulation,
    dx_tf_wp_insulation,
    pflux_fw_neutron_mw,
    tfsai,
    tfsao,
    dr_tf_plasma_case,
    f_t_plant_available,
    life_plant,
):
    """Nuclear heating in a superconducting TF coil (1990 ITER exponential-attenuation
    estimate).

    Ports the SUPERCONDUCTING branch of `sc_tf_coil_nuclear_heating_iter90`.

    Parameters
    ----------
    dr_shld_inboard, dr_fw_inboard, dr_blkt_inboard :
        Inboard shield/first-wall/blanket thicknesses (m). `.build.dr_shld_inboard` etc.
    dr_shld_outboard, dr_fw_outboard, dr_blkt_outboard :
        Outboard shield/first-wall/blanket thicknesses (m). `.build.dr_shld_outboard` etc.
    dr_tf_wp_with_insulation :
        TF winding pack radial thickness including groundwall insulation (m).
        `.tfcoil.dr_tf_wp_with_insulation`.
    dx_tf_wp_insulation :
        TF winding pack insulation thickness (m). `.tfcoil.dx_tf_wp_insulation`.
    pflux_fw_neutron_mw :
        Neutron wall load (MW/m2). `.physics.pflux_fw_neutron_mw`.
    tfsai, tfsao :
        Inboard/outboard TF coil surface areas (m2). `.tfcoil.tfsai`, `.tfcoil.tfsao`.
    dr_tf_plasma_case :
        TF coil plasma-side case thickness (m). `.tfcoil.dr_tf_plasma_case`.
    f_t_plant_available :
        Plant availability factor. `.costs.f_t_plant_available`.
    life_plant :
        Plant lifetime (years). `.costs.life_plant`.

    Returns
    -------
    :
        `(coilhtmx, dpacop, htheci, flu_tf_neutron_fast_peak, pheci, pheco, ptfiwp,
        ptfowp, raddose, p_tf_nuclear_heat_mw)` -- peak magnet heating (MW/m3), copper
        stabiliser displacements/atom, peak TF coil case heating (MW/m3), maximum
        neutron fluence (n/m2), inboard/outboard coil case heating (MW), inboard/
        outboard winding pack heating (MW), insulator dose (rad), and total TF coil
        nuclear heating (MW).
    """
    dshieq = dr_shld_inboard + dr_fw_inboard + dr_blkt_inboard
    dshoeq = dr_shld_outboard + dr_fw_outboard + dr_blkt_outboard

    wpthk = dr_tf_wp_with_insulation + 2.0 * dx_tf_wp_insulation

    coilhtmx = (
        _FACT[0]
        * pflux_fw_neutron_mw
        * _COEF[0]
        * jnp.exp(-_DECAY[5] * (dshieq + dr_tf_plasma_case))
    )

    ptfiwp = coilhtmx * tfsai * (1.0 - jnp.exp(-_DECAY[0] * wpthk)) / _DECAY[0]
    ptfowp = (
        _FACT[0]
        * pflux_fw_neutron_mw
        * _COEF[0]
        * jnp.exp(-_DECAY[5] * (dshoeq + dr_tf_plasma_case))
        * tfsao
        * (1.0 - jnp.exp(-_DECAY[0] * wpthk))
        / _DECAY[0]
    )

    htheci = _FACT[1] * pflux_fw_neutron_mw * _COEF[1] * jnp.exp(-_DECAY[6] * dshieq)
    pheci = htheci * tfsai * (1.0 - jnp.exp(-_DECAY[1] * dr_tf_plasma_case)) / _DECAY[1]
    pheco = (
        _FACT[1]
        * pflux_fw_neutron_mw
        * _COEF[1]
        * jnp.exp(-_DECAY[6] * dshoeq)
        * tfsao
        * (1.0 - jnp.exp(-_DECAY[1] * dr_tf_plasma_case))
        / _DECAY[1]
    )

    p_tf_nuclear_heat_mw = (ptfiwp + pheci) + (ptfowp + pheco)

    fpydt = f_t_plant_available * life_plant
    fpsdt = fpydt * _SECONDS_PER_FULL_POWER_YEAR

    raddose = (
        _COEF[2]
        * fpsdt
        * _FACT[2]
        * pflux_fw_neutron_mw
        * jnp.exp(-_DECAY[2] * (dshieq + dr_tf_plasma_case))
    )

    flu_tf_neutron_fast_peak = (
        fpsdt
        * _FACT[3]
        * pflux_fw_neutron_mw
        * _COEF[3]
        * jnp.exp(-_DECAY[3] * (dshieq + dr_tf_plasma_case))
    )

    dpacop = (
        fpsdt
        * _FACT[4]
        * pflux_fw_neutron_mw
        * _COEF[4]
        * jnp.exp(-_DECAY[4] * (dshieq + dr_tf_plasma_case))
    )

    return (
        coilhtmx,
        dpacop,
        htheci,
        flu_tf_neutron_fast_peak,
        pheci,
        pheco,
        ptfiwp,
        ptfowp,
        raddose,
        p_tf_nuclear_heat_mw,
    )
