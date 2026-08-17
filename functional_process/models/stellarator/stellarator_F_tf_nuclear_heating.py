"""Pure-functional port of `Stellarator.sc_tf_coil_nuclear_heating_iter90` (chunk 1F).

Audit record: `functional_process/models/stellarator/stellarator_F_tf_nuclear_heating.md`.
Ports only the SUPERCONDUCTING branch of the source's `i_tf_sup` switch -- per
`core/solver/switches.md`'s `i_tf_sup` entry (split, high confidence, three independent
data points including this unit) and `naming_convention.md`'s "switches are not ports":
the resistive branch takes no inputs and always returns ten zeros, so it is not a
computation to port, it is the absence of this node in the graph once `i_tf_sup` selects
a non-superconducting coil (see the record's open questions).

`ishmat` (source: "stainless steel coil casing is assumed") is hardcoded to the stainless
column of `coef`/`decay`; the unused tungsten column is dropped entirely rather than kept
as a dead second index, per the record's note.

`ScTfCoilNuclearHeating` below is the `cottax` node. Its output `VarPath`s are
best-effort, not existing PROCESS storage: the source never writes 8 of its 10 return
values to `self.data` (see the record's "cottax node" section) -- `.fwbs.*` is inferred
from the two fields that *are* stored elsewhere in `st_fwbs`, flagged there for whoever
audits 1E1/1E2 to confirm.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

# TF coil nuclear heating coefficients (stainless-steel shield only -- see module
# docstring). `fact[i]`/`coef[i]`/`decay[i]` index the same physical quantity as the
# source's `fact`/`coef[:, ishmat]`/`decay[:, ishmat]`.
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

    ptfiwp = (
        coilhtmx * tfsai * (1.0 - jnp.exp(-_DECAY[0] * wpthk)) / _DECAY[0]
    )
    ptfowp = (
        _FACT[0]
        * pflux_fw_neutron_mw
        * _COEF[0]
        * jnp.exp(-_DECAY[5] * (dshoeq + dr_tf_plasma_case))
        * tfsao
        * (1.0 - jnp.exp(-_DECAY[0] * wpthk))
        / _DECAY[0]
    )

    htheci = (
        _FACT[1] * pflux_fw_neutron_mw * _COEF[1] * jnp.exp(-_DECAY[6] * dshieq)
    )
    pheci = (
        htheci
        * tfsai
        * (1.0 - jnp.exp(-_DECAY[1] * dr_tf_plasma_case))
        / _DECAY[1]
    )
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


class ScTfCoilNuclearHeating(ExplicitFunction):
    """cottax node: `calculate_sc_tf_coil_nuclear_heating`, unchanged, ports declared."""

    coilhtmx = Output(lambda s: s.fwbs.coilhtmx)
    dpacop = Output(lambda s: s.fwbs.dpacop)
    htheci = Output(lambda s: s.fwbs.htheci)
    flu_tf_neutron_fast_peak = Output(lambda s: s.fwbs.flu_tf_neutron_fast_peak)
    pheci = Output(lambda s: s.fwbs.pheci)
    pheco = Output(lambda s: s.fwbs.pheco)
    ptfiwp = Output(lambda s: s.fwbs.ptfiwp)
    ptfowp = Output(lambda s: s.fwbs.ptfowp)
    raddose = Output(lambda s: s.fwbs.raddose)
    p_tf_nuclear_heat_mw = Output(lambda s: s.fwbs.p_tf_nuclear_heat_mw)

    def __call__(
        self,
        dr_shld_inboard=Input(lambda s: s.build.dr_shld_inboard),
        dr_fw_inboard=Input(lambda s: s.build.dr_fw_inboard),
        dr_blkt_inboard=Input(lambda s: s.build.dr_blkt_inboard),
        dr_shld_outboard=Input(lambda s: s.build.dr_shld_outboard),
        dr_fw_outboard=Input(lambda s: s.build.dr_fw_outboard),
        dr_blkt_outboard=Input(lambda s: s.build.dr_blkt_outboard),
        dr_tf_wp_with_insulation=Input(lambda s: s.tfcoil.dr_tf_wp_with_insulation),
        dx_tf_wp_insulation=Input(lambda s: s.tfcoil.dx_tf_wp_insulation),
        pflux_fw_neutron_mw=Input(lambda s: s.physics.pflux_fw_neutron_mw),
        tfsai=Input(lambda s: s.tfcoil.tfsai),
        tfsao=Input(lambda s: s.tfcoil.tfsao),
        dr_tf_plasma_case=Input(lambda s: s.tfcoil.dr_tf_plasma_case),
        f_t_plant_available=Input(lambda s: s.costs.f_t_plant_available),
        life_plant=Input(lambda s: s.costs.life_plant),
    ):
        return calculate_sc_tf_coil_nuclear_heating(
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
        )
