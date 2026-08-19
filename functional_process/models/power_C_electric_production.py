"""Pure-functional port of the AC/electric-production sub-unit of
`process/models/power.py` (registry unit #14, chunk C).

Audit record: `functional_process/models/power_C_electric_production.md`. Covers
`Power.acpow` (696-813), `Power.power_profiles_over_time` (2632-2825) and
`Power.plant_electric_production` (1631-1772) -- see the audit record's data-footprint
table for the full trace.

All three are tier-1: no internal iteration, no calls into any other model. The time
axis `power_profiles_over_time` builds is always exactly 7 points (`PulseTimings.
total_pulse_cumulative` is `len()` of a fixed 7-tuple of cumulative sums over the six
phase durations, never data-dependent in length) -- so every array in this chunk has
a static, compile-time-known shape; there is no dynamic-shape difficulty here at all.
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

from process.models.power import PumpingPowerModelTypes


def calculate_acpow(
    p_tf_electric_supplies_mw,
    srcktpm,
    peakmva,
    i_pf_energy_storage_source,
    p_hcd_electric_total_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_coolant_pump_elec_total_mw,
    p_tritium_plant_electric_mw,
    p_plant_electric_base_total_mw,
    fmgdmw,
):
    """`Power.acpow`.

    Ports `process/models/power.py:696-753` verbatim (the output section, 755-812,
    is excluded from scope). `bdvmw` ("Power to divertor coil supplies") is a
    hardcoded `0.0` in the source -- dropped as a dead `+ 0.0` term, same convention
    chunk A used for `tfreacmw`.

    Parameters
    ----------
    p_tf_electric_supplies_mw :
        `.heat_transport.p_tf_electric_supplies_mw`.
    srcktpm :
        PF coil circuit total peak MVA. `.pf_power.srcktpm`.
    peakmva :
        Peak MVA requirement. `.heat_transport.peakmva`.
    i_pf_energy_storage_source :
        Static switch (`2` = motor-generator flywheels not used as PF energy store).
        `.pf_power.i_pf_energy_storage_source`.
    p_hcd_electric_total_mw :
        `.heat_transport.p_hcd_electric_total_mw`.
    p_cryo_plant_electric_mw :
        `.heat_transport.p_cryo_plant_electric_mw`.
    vachtmw :
        `.heat_transport.vachtmw`.
    p_coolant_pump_elec_total_mw :
        `.heat_transport.p_coolant_pump_elec_total_mw`.
    p_tritium_plant_electric_mw :
        `.heat_transport.p_tritium_plant_electric_mw`.
    p_plant_electric_base_total_mw :
        `.heat_transport.p_plant_electric_base_total_mw`.
    fmgdmw :
        Flywheel motor-generator power (MW). `.heat_transport.fmgdmw`.

    Returns
    -------
    :
        `(pacpmw, tlvpmw)` -- `.heat_transport.pacpmw`, `.heat_transport.tlvpmw`.
    """
    ptfmw = p_tf_electric_supplies_mw
    ppfmw = 1.0e-3 * srcktpm
    if i_pf_energy_storage_source == 2:
        ppfmw = ppfmw + peakmva

    pheatingmw = p_hcd_electric_total_mw
    crymw = p_cryo_plant_electric_mw

    pacpmw = (
        ppfmw
        + ptfmw
        + crymw
        + vachtmw
        + p_coolant_pump_elec_total_mw
        + p_tritium_plant_electric_mw
        + pheatingmw
    )
    if i_pf_energy_storage_source != 2:
        pacpmw = pacpmw + fmgdmw

    tlvpmw = (
        p_plant_electric_base_total_mw
        + p_tritium_plant_electric_mw
        + p_coolant_pump_elec_total_mw
        + vachtmw
        + 0.5e0 * (crymw + ppfmw)
    )

    return pacpmw, tlvpmw


def power_profiles_over_time(
    p_plant_electric_base_total_mw,
    p_cryo_plant_electric_mw,
    p_tritium_plant_electric_mw,
    vachtmw,
    p_tf_electric_supplies_mw,
    p_pf_electric_supplies_mw,
    p_coolant_pump_elec_total_mw,
    p_hcd_electric_total_mw,
    p_fusion_total_mw,
    p_plant_electric_gross_mw,
    p_plant_electric_net_mw,
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_dwell,
):
    """`Power.power_profiles_over_time` (already a `@staticmethod` in PROCESS).

    Ports `process/models/power.py:2632-2825`. PROCESS's own `PulseTimings` dataclass
    (`process/models/pulse.py`) is not reused -- it is a plain namespace object over
    the same six `.times.t_plant_pulse_*` fields with no `self.data` access of its
    own, so its two properties this function needs
    (`total_pulse_cumulative`/`n_pulse_points_total`) are inlined here directly rather
    than wrapped. `n_pulse_points_total` is always `7`
    (`len(total_pulse_cumulative)`, and `total_pulse_cumulative` is always a 7-tuple
    regardless of the six durations' values) -- a true compile-time constant, not
    data-dependent, so every array below has a static shape `(7,)`.

    PROCESS's own diagnostic check (`np.isclose(p_plant_electric_net_profile_mw[3],
    p_plant_electric_net_mw)`, logged as an error on mismatch) is a side-effect-only
    read with no effect on any returned value -- dropped, same convention as every
    other diagnostic-only log call in this codebase's ported units.

    Parameters
    ----------
    p_plant_electric_base_total_mw, p_cryo_plant_electric_mw,
    p_tritium_plant_electric_mw, vachtmw, p_tf_electric_supplies_mw,
    p_pf_electric_supplies_mw, p_coolant_pump_elec_total_mw, p_hcd_electric_total_mw,
    p_fusion_total_mw, p_plant_electric_gross_mw, p_plant_electric_net_mw :
        Steady-state powers (MW) to spread over the pulse profile -- see
        `power_C_electric_production.md` for each field's `VarPath`.
    t_plant_pulse_coil_precharge, t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp, t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down, t_plant_pulse_dwell :
        Phase durations (s). `.times.t_plant_pulse_*`.

    Returns
    -------
    :
        `(energy_made_kwh, energy_made_mj, p_plant_electric_base_total_profile_mw,
        p_plant_electric_gross_profile_mw, p_plant_electric_net_profile_mw,
        p_hcd_electric_total_profile_mw, p_coolant_pump_elec_total_profile_mw,
        p_tf_electric_supplies_profile_mw, p_pf_electric_supplies_profile_mw,
        vachtmw_profile_mw, p_tritium_plant_electric_profile_mw,
        p_cryo_plant_electric_profile_mw, p_fusion_total_profile_mw)`, each profile a
        length-7 array, matching `Power.plant_electric_production`'s unpacking order
        (which puts the two energy scalars first, `kwh` before `mj`).
    """
    t0 = 0.0e0
    t1 = t0 + t_plant_pulse_coil_precharge
    t2 = t1 + t_plant_pulse_plasma_current_ramp_up
    t3 = t2 + t_plant_pulse_fusion_ramp
    t4 = t3 + t_plant_pulse_burn
    t5 = t4 + t_plant_pulse_plasma_current_ramp_down
    t6 = t5 + t_plant_pulse_dwell
    total_pulse_cumulative = jnp.stack([t0, t1, t2, t3, t4, t5, t6])

    def _plateau(value, active):
        """Length-7 profile: `value` where `active` is `True`, `0.0` elsewhere.

        `jnp.where` broadcasts the scalar `value` against the length-7 boolean mask,
        so this needs no explicit shape bookkeeping.
        """
        return jnp.where(jnp.asarray(active), value, 0.0e0)

    fusion_active = (False, False, True, True, True, False, False)
    p_fusion_total_profile_mw = _plateau(p_fusion_total_mw, fusion_active)

    always_active = (True,) * 7
    p_plant_electric_base_total_profile_mw = _plateau(
        -p_plant_electric_base_total_mw, always_active
    )
    p_cryo_plant_electric_profile_mw = _plateau(-p_cryo_plant_electric_mw, always_active)
    p_tritium_plant_electric_profile_mw = _plateau(
        -p_tritium_plant_electric_mw, always_active
    )
    vachtmw_profile_mw = _plateau(-vachtmw, always_active)
    p_tf_electric_supplies_profile_mw = _plateau(
        -p_tf_electric_supplies_mw, always_active
    )

    ramp_and_burn = (False, True, True, True, True, False, False)
    ramp_and_burn_only = (False, False, True, True, True, False, False)

    p_pf_electric_supplies_profile_mw = _plateau(
        -p_pf_electric_supplies_mw, ramp_and_burn
    )
    p_coolant_pump_elec_total_profile_mw = _plateau(
        -p_coolant_pump_elec_total_mw, ramp_and_burn_only
    )
    p_hcd_electric_total_profile_mw = _plateau(
        -p_hcd_electric_total_mw, ramp_and_burn_only
    )
    p_plant_electric_gross_profile_mw = _plateau(
        p_plant_electric_gross_mw, ramp_and_burn_only
    )

    p_plant_electric_net_profile_mw = (
        p_plant_electric_gross_profile_mw
        + p_plant_electric_base_total_profile_mw
        + p_cryo_plant_electric_profile_mw
        + p_tritium_plant_electric_profile_mw
        + vachtmw_profile_mw
        + p_tf_electric_supplies_profile_mw
        + p_pf_electric_supplies_profile_mw
        + p_coolant_pump_elec_total_profile_mw
        + p_hcd_electric_total_profile_mw
    )

    energy_made_mj = jnp.trapezoid(
        p_plant_electric_net_profile_mw, total_pulse_cumulative
    )
    energy_made_kwh = energy_made_mj / 3.6e0

    return (
        energy_made_kwh,
        energy_made_mj,
        p_plant_electric_base_total_profile_mw,
        p_plant_electric_gross_profile_mw,
        p_plant_electric_net_profile_mw,
        p_hcd_electric_total_profile_mw,
        p_coolant_pump_elec_total_profile_mw,
        p_tf_electric_supplies_profile_mw,
        p_pf_electric_supplies_profile_mw,
        vachtmw_profile_mw,
        p_tritium_plant_electric_profile_mw,
        p_cryo_plant_electric_profile_mw,
        p_fusion_total_profile_mw,
    )


def calculate_plant_electric_production(
    itart,
    i_tf_sup,
    p_cp_coolant_pump_elec,
    p_plant_electric_base,
    a_plant_floor_effective,
    pflux_plant_floor_electric,
    p_cryo_plant_electric_mw,
    p_tf_electric_supplies_mw,
    p_tritium_plant_electric_mw,
    vachtmw,
    p_pf_electric_supplies_mw,
    p_hcd_electric_loss_mw,
    p_coolant_pump_loss_total_mw,
    p_div_secondary_heat_mw,
    p_shld_secondary_heat_mw,
    p_hcd_secondary_heat_mw,
    p_tf_nuclear_heat_mw,
    ireactor,
    i_blkt_dual_coolant,
    i_p_coolant_pumping,
    p_plant_primary_heat_mw,
    p_blkt_liquid_breeder_heat_deposited_mw,
    eta_turbine,
    etath_liq,
    p_hcd_electric_total_mw,
    p_coolant_pump_elec_total_mw,
    p_plant_electric_gross_mw,
    p_turbine_loss_mw,
    p_plant_electric_recirc_mw,
    p_plant_electric_net_mw,
    f_p_plant_electric_recirc,
    p_fusion_total_mw,
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_dwell,
):
    """`Power.plant_electric_production`.

    Ports `process/models/power.py:1631-1772`. Composes `power_profiles_over_time`
    internally, in the same call PROCESS makes at the end of this method.

    **`p_plant_electric_gross_mw`/`p_turbine_loss_mw`/`p_plant_electric_recirc_mw`/
    `p_plant_electric_net_mw`/`f_p_plant_electric_recirc` are a conditional-ownership
    pass-through** on the static switch `ireactor` (`.costs.ireactor`, unit #18's
    field, not yet ported -- an ordinary `Input`, not a call into unit #18's
    methods): PROCESS only computes these five when `ireactor == 1`; otherwise they
    keep whatever value they entered this call with. `power_profiles_over_time` is
    then called **unconditionally**, using `p_plant_electric_gross_mw`/
    `p_plant_electric_net_mw` regardless of whether this call just computed them or
    they are carried over stale/default -- ported faithfully, not resolved, same
    treatment `power_B_thermal_cryo.md` gives `.power.delta_eta`.

    Parameters
    ----------
    itart, i_tf_sup :
        Static switches gating `p_cp_coolant_pump_elec_mw`'s ownership (only owned
        when `itart == 1 and i_tf_sup == 0`, a tight-aspect-ratio/resistive-centrepost
        configuration; always `0.0` otherwise, no other producer). `.physics.itart`,
        `.tfcoil.i_tf_sup`.
    ireactor, i_blkt_dual_coolant, i_p_coolant_pumping :
        Static switches -- see docstring body above and
        `power_C_electric_production.md`.
    (all other parameters) :
        See `power_C_electric_production.md`'s data-footprint table for each
        field's `VarPath`.

    Returns
    -------
    :
        `(p_cp_coolant_pump_elec_mw, p_plant_electric_base_total_mw, fachtmw,
        p_plant_core_systems_elec_mw, p_plant_secondary_heat_mw,
        p_plant_electric_gross_mw, p_turbine_loss_mw, p_plant_electric_recirc_mw,
        p_plant_electric_net_mw, f_p_plant_electric_recirc,
        e_plant_net_electric_pulse_kwh, e_plant_net_electric_pulse_mj,
        p_plant_electric_base_total_profile_mw, p_plant_electric_gross_profile_mw,
        p_plant_electric_net_profile_mw, p_hcd_electric_total_profile_mw,
        p_coolant_pump_elec_total_profile_mw, p_tf_electric_supplies_profile_mw,
        p_pf_electric_supplies_profile_mw, vachtmw_profile_mw,
        p_tritium_plant_electric_profile_mw, p_cryo_plant_electric_profile_mw,
        p_fusion_total_profile_mw)`.
    """
    if itart == 1 and i_tf_sup == 0:
        p_cp_coolant_pump_elec_mw = 1.0e-6 * p_cp_coolant_pump_elec
    else:
        p_cp_coolant_pump_elec_mw = 0.0e0

    p_plant_electric_base_total_mw = (
        p_plant_electric_base * 1.0e-6
        + a_plant_floor_effective * (pflux_plant_floor_electric * 1.0e-3) / 1000.0e0
    )
    fachtmw = p_plant_electric_base_total_mw

    p_plant_core_systems_elec_mw = (
        p_cryo_plant_electric_mw
        + fachtmw
        + p_cp_coolant_pump_elec_mw
        + p_tf_electric_supplies_mw
        + p_tritium_plant_electric_mw
        + vachtmw
        + p_pf_electric_supplies_mw
    )

    p_plant_secondary_heat_mw = (
        p_plant_core_systems_elec_mw
        + p_hcd_electric_loss_mw
        + p_coolant_pump_loss_total_mw
        + p_div_secondary_heat_mw
        + p_shld_secondary_heat_mw
        + p_hcd_secondary_heat_mw
        + p_tf_nuclear_heat_mw
    )

    if ireactor == 1:
        if (
            i_blkt_dual_coolant > 0
            and i_p_coolant_pumping == PumpingPowerModelTypes.MECHANICAL
        ):
            p_plant_electric_gross_mw = (
                p_plant_primary_heat_mw - p_blkt_liquid_breeder_heat_deposited_mw
            ) * eta_turbine + p_blkt_liquid_breeder_heat_deposited_mw * etath_liq
        else:
            p_plant_electric_gross_mw = p_plant_primary_heat_mw * eta_turbine

        p_turbine_loss_mw = p_plant_primary_heat_mw * (1 - eta_turbine)
        p_plant_electric_recirc_mw = (
            p_plant_core_systems_elec_mw
            + p_hcd_electric_total_mw
            + p_coolant_pump_elec_total_mw
        )
        p_plant_electric_net_mw = p_plant_electric_gross_mw - p_plant_electric_recirc_mw
        f_p_plant_electric_recirc = (
            p_plant_electric_gross_mw - p_plant_electric_net_mw
        ) / p_plant_electric_gross_mw

    (
        e_plant_net_electric_pulse_kwh,
        e_plant_net_electric_pulse_mj,
        p_plant_electric_base_total_profile_mw,
        p_plant_electric_gross_profile_mw,
        p_plant_electric_net_profile_mw,
        p_hcd_electric_total_profile_mw,
        p_coolant_pump_elec_total_profile_mw,
        p_tf_electric_supplies_profile_mw,
        p_pf_electric_supplies_profile_mw,
        vachtmw_profile_mw,
        p_tritium_plant_electric_profile_mw,
        p_cryo_plant_electric_profile_mw,
        p_fusion_total_profile_mw,
    ) = power_profiles_over_time(
        p_plant_electric_base_total_mw,
        p_cryo_plant_electric_mw,
        p_tritium_plant_electric_mw,
        vachtmw,
        p_tf_electric_supplies_mw,
        p_pf_electric_supplies_mw,
        p_coolant_pump_elec_total_mw,
        p_hcd_electric_total_mw,
        p_fusion_total_mw,
        p_plant_electric_gross_mw,
        p_plant_electric_net_mw,
        t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_fusion_ramp,
        t_plant_pulse_burn,
        t_plant_pulse_plasma_current_ramp_down,
        t_plant_pulse_dwell,
    )

    return (
        p_cp_coolant_pump_elec_mw,
        p_plant_electric_base_total_mw,
        fachtmw,
        p_plant_core_systems_elec_mw,
        p_plant_secondary_heat_mw,
        p_plant_electric_gross_mw,
        p_turbine_loss_mw,
        p_plant_electric_recirc_mw,
        p_plant_electric_net_mw,
        f_p_plant_electric_recirc,
        e_plant_net_electric_pulse_kwh,
        e_plant_net_electric_pulse_mj,
        p_plant_electric_base_total_profile_mw,
        p_plant_electric_gross_profile_mw,
        p_plant_electric_net_profile_mw,
        p_hcd_electric_total_profile_mw,
        p_coolant_pump_elec_total_profile_mw,
        p_tf_electric_supplies_profile_mw,
        p_pf_electric_supplies_profile_mw,
        vachtmw_profile_mw,
        p_tritium_plant_electric_profile_mw,
        p_cryo_plant_electric_profile_mw,
        p_fusion_total_profile_mw,
    )


class Acpow(ExplicitFunction):
    """cottax node: `calculate_acpow`."""

    i_pf_energy_storage_source: int = eqx.field(static=True)

    pacpmw = Output(lambda s: s.heat_transport.pacpmw)
    tlvpmw = Output(lambda s: s.heat_transport.tlvpmw)

    def __call__(
        self,
        p_tf_electric_supplies_mw=Input(
            lambda s: s.heat_transport.p_tf_electric_supplies_mw
        ),
        srcktpm=Input(lambda s: s.pf_power.srcktpm),
        peakmva=Input(lambda s: s.heat_transport.peakmva),
        p_hcd_electric_total_mw=Input(
            lambda s: s.heat_transport.p_hcd_electric_total_mw
        ),
        p_cryo_plant_electric_mw=Input(
            lambda s: s.heat_transport.p_cryo_plant_electric_mw
        ),
        vachtmw=Input(lambda s: s.heat_transport.vachtmw),
        p_coolant_pump_elec_total_mw=Input(
            lambda s: s.heat_transport.p_coolant_pump_elec_total_mw
        ),
        p_tritium_plant_electric_mw=Input(
            lambda s: s.heat_transport.p_tritium_plant_electric_mw
        ),
        p_plant_electric_base_total_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_base_total_mw
        ),
        fmgdmw=Input(lambda s: s.heat_transport.fmgdmw),
    ):
        return calculate_acpow(
            p_tf_electric_supplies_mw,
            srcktpm,
            peakmva,
            self.i_pf_energy_storage_source,
            p_hcd_electric_total_mw,
            p_cryo_plant_electric_mw,
            vachtmw,
            p_coolant_pump_elec_total_mw,
            p_tritium_plant_electric_mw,
            p_plant_electric_base_total_mw,
            fmgdmw,
        )


class PowerProfilesOverTime(ExplicitFunction):
    """cottax node: `power_profiles_over_time`."""

    e_plant_net_electric_pulse_kwh = Output(
        lambda s: s.power.e_plant_net_electric_pulse_kwh
    )
    e_plant_net_electric_pulse_mj = Output(
        lambda s: s.power.e_plant_net_electric_pulse_mj
    )
    p_plant_electric_base_total_profile_mw = Output(
        lambda s: s.power.p_plant_electric_base_total_profile_mw
    )
    p_plant_electric_gross_profile_mw = Output(
        lambda s: s.power.p_plant_electric_gross_profile_mw
    )
    p_plant_electric_net_profile_mw = Output(
        lambda s: s.power.p_plant_electric_net_profile_mw
    )
    p_hcd_electric_total_profile_mw = Output(
        lambda s: s.power.p_hcd_electric_total_profile_mw
    )
    p_coolant_pump_elec_total_profile_mw = Output(
        lambda s: s.power.p_coolant_pump_elec_total_profile_mw
    )
    p_tf_electric_supplies_profile_mw = Output(
        lambda s: s.power.p_tf_electric_supplies_profile_mw
    )
    p_pf_electric_supplies_profile_mw = Output(
        lambda s: s.power.p_pf_electric_supplies_profile_mw
    )
    vachtmw_profile_mw = Output(lambda s: s.power.vachtmw_profile_mw)
    p_tritium_plant_electric_profile_mw = Output(
        lambda s: s.power.p_tritium_plant_electric_profile_mw
    )
    p_cryo_plant_electric_profile_mw = Output(
        lambda s: s.power.p_cryo_plant_electric_profile_mw
    )
    p_fusion_total_profile_mw = Output(lambda s: s.power.p_fusion_total_profile_mw)

    def __call__(
        self,
        p_plant_electric_base_total_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_base_total_mw
        ),
        p_cryo_plant_electric_mw=Input(
            lambda s: s.heat_transport.p_cryo_plant_electric_mw
        ),
        p_tritium_plant_electric_mw=Input(
            lambda s: s.heat_transport.p_tritium_plant_electric_mw
        ),
        vachtmw=Input(lambda s: s.heat_transport.vachtmw),
        p_tf_electric_supplies_mw=Input(
            lambda s: s.heat_transport.p_tf_electric_supplies_mw
        ),
        p_pf_electric_supplies_mw=Input(lambda s: s.pf_coil.p_pf_electric_supplies_mw),
        p_coolant_pump_elec_total_mw=Input(
            lambda s: s.heat_transport.p_coolant_pump_elec_total_mw
        ),
        p_hcd_electric_total_mw=Input(
            lambda s: s.heat_transport.p_hcd_electric_total_mw
        ),
        p_fusion_total_mw=Input(lambda s: s.physics.p_fusion_total_mw),
        p_plant_electric_gross_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_gross_mw
        ),
        p_plant_electric_net_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_net_mw
        ),
        t_plant_pulse_coil_precharge=Input(
            lambda s: s.times.t_plant_pulse_coil_precharge
        ),
        t_plant_pulse_plasma_current_ramp_up=Input(
            lambda s: s.times.t_plant_pulse_plasma_current_ramp_up
        ),
        t_plant_pulse_fusion_ramp=Input(lambda s: s.times.t_plant_pulse_fusion_ramp),
        t_plant_pulse_burn=Input(lambda s: s.times.t_plant_pulse_burn),
        t_plant_pulse_plasma_current_ramp_down=Input(
            lambda s: s.times.t_plant_pulse_plasma_current_ramp_down
        ),
        t_plant_pulse_dwell=Input(lambda s: s.times.t_plant_pulse_dwell),
    ):
        return power_profiles_over_time(
            p_plant_electric_base_total_mw,
            p_cryo_plant_electric_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_tf_electric_supplies_mw,
            p_pf_electric_supplies_mw,
            p_coolant_pump_elec_total_mw,
            p_hcd_electric_total_mw,
            p_fusion_total_mw,
            p_plant_electric_gross_mw,
            p_plant_electric_net_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProduction(ExplicitFunction):
    """cottax node: `calculate_plant_electric_production`."""

    itart: int = eqx.field(static=True)
    i_tf_sup: int = eqx.field(static=True)
    ireactor: int = eqx.field(static=True)
    i_blkt_dual_coolant: int = eqx.field(static=True)
    i_p_coolant_pumping: int = eqx.field(static=True)

    p_cp_coolant_pump_elec_mw = Output(lambda s: s.power.p_cp_coolant_pump_elec_mw)
    p_plant_electric_base_total_mw = Output(
        lambda s: s.heat_transport.p_plant_electric_base_total_mw
    )
    fachtmw = Output(lambda s: s.heat_transport.fachtmw)
    p_plant_core_systems_elec_mw = Output(lambda s: s.power.p_plant_core_systems_elec_mw)
    p_plant_secondary_heat_mw = Output(
        lambda s: s.heat_transport.p_plant_secondary_heat_mw
    )
    p_plant_electric_gross_mw = Output(
        lambda s: s.heat_transport.p_plant_electric_gross_mw
    )
    p_turbine_loss_mw = Output(lambda s: s.power.p_turbine_loss_mw)
    p_plant_electric_recirc_mw = Output(
        lambda s: s.heat_transport.p_plant_electric_recirc_mw
    )
    p_plant_electric_net_mw = Output(lambda s: s.heat_transport.p_plant_electric_net_mw)
    f_p_plant_electric_recirc = Output(
        lambda s: s.heat_transport.f_p_plant_electric_recirc
    )
    e_plant_net_electric_pulse_kwh = Output(
        lambda s: s.power.e_plant_net_electric_pulse_kwh
    )
    e_plant_net_electric_pulse_mj = Output(
        lambda s: s.power.e_plant_net_electric_pulse_mj
    )
    p_plant_electric_base_total_profile_mw = Output(
        lambda s: s.power.p_plant_electric_base_total_profile_mw
    )
    p_plant_electric_gross_profile_mw = Output(
        lambda s: s.power.p_plant_electric_gross_profile_mw
    )
    p_plant_electric_net_profile_mw = Output(
        lambda s: s.power.p_plant_electric_net_profile_mw
    )
    p_hcd_electric_total_profile_mw = Output(
        lambda s: s.power.p_hcd_electric_total_profile_mw
    )
    p_coolant_pump_elec_total_profile_mw = Output(
        lambda s: s.power.p_coolant_pump_elec_total_profile_mw
    )
    p_tf_electric_supplies_profile_mw = Output(
        lambda s: s.power.p_tf_electric_supplies_profile_mw
    )
    p_pf_electric_supplies_profile_mw = Output(
        lambda s: s.power.p_pf_electric_supplies_profile_mw
    )
    vachtmw_profile_mw = Output(lambda s: s.power.vachtmw_profile_mw)
    p_tritium_plant_electric_profile_mw = Output(
        lambda s: s.power.p_tritium_plant_electric_profile_mw
    )
    p_cryo_plant_electric_profile_mw = Output(
        lambda s: s.power.p_cryo_plant_electric_profile_mw
    )
    p_fusion_total_profile_mw = Output(lambda s: s.power.p_fusion_total_profile_mw)

    def __call__(
        self,
        p_cp_coolant_pump_elec=Input(lambda s: s.tfcoil.p_cp_coolant_pump_elec),
        p_plant_electric_base=Input(lambda s: s.heat_transport.p_plant_electric_base),
        a_plant_floor_effective=Input(lambda s: s.buildings.a_plant_floor_effective),
        pflux_plant_floor_electric=Input(
            lambda s: s.heat_transport.pflux_plant_floor_electric
        ),
        p_cryo_plant_electric_mw=Input(
            lambda s: s.heat_transport.p_cryo_plant_electric_mw
        ),
        p_tf_electric_supplies_mw=Input(
            lambda s: s.heat_transport.p_tf_electric_supplies_mw
        ),
        p_tritium_plant_electric_mw=Input(
            lambda s: s.heat_transport.p_tritium_plant_electric_mw
        ),
        vachtmw=Input(lambda s: s.heat_transport.vachtmw),
        p_pf_electric_supplies_mw=Input(lambda s: s.pf_coil.p_pf_electric_supplies_mw),
        p_hcd_electric_loss_mw=Input(lambda s: s.heat_transport.p_hcd_electric_loss_mw),
        p_coolant_pump_loss_total_mw=Input(
            lambda s: s.heat_transport.p_coolant_pump_loss_total_mw
        ),
        p_div_secondary_heat_mw=Input(
            lambda s: s.heat_transport.p_div_secondary_heat_mw
        ),
        p_shld_secondary_heat_mw=Input(
            lambda s: s.heat_transport.p_shld_secondary_heat_mw
        ),
        p_hcd_secondary_heat_mw=Input(
            lambda s: s.heat_transport.p_hcd_secondary_heat_mw
        ),
        p_tf_nuclear_heat_mw=Input(lambda s: s.fwbs.p_tf_nuclear_heat_mw),
        p_plant_primary_heat_mw=Input(
            lambda s: s.heat_transport.p_plant_primary_heat_mw
        ),
        p_blkt_liquid_breeder_heat_deposited_mw=Input(
            lambda s: s.power.p_blkt_liquid_breeder_heat_deposited_mw
        ),
        eta_turbine=Input(lambda s: s.heat_transport.eta_turbine),
        etath_liq=Input(lambda s: s.heat_transport.etath_liq),
        p_hcd_electric_total_mw=Input(
            lambda s: s.heat_transport.p_hcd_electric_total_mw
        ),
        p_coolant_pump_elec_total_mw=Input(
            lambda s: s.heat_transport.p_coolant_pump_elec_total_mw
        ),
        p_plant_electric_gross_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_gross_mw
        ),
        p_turbine_loss_mw=Input(lambda s: s.power.p_turbine_loss_mw),
        p_plant_electric_recirc_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_recirc_mw
        ),
        p_plant_electric_net_mw=Input(
            lambda s: s.heat_transport.p_plant_electric_net_mw
        ),
        f_p_plant_electric_recirc=Input(
            lambda s: s.heat_transport.f_p_plant_electric_recirc
        ),
        p_fusion_total_mw=Input(lambda s: s.physics.p_fusion_total_mw),
        t_plant_pulse_coil_precharge=Input(
            lambda s: s.times.t_plant_pulse_coil_precharge
        ),
        t_plant_pulse_plasma_current_ramp_up=Input(
            lambda s: s.times.t_plant_pulse_plasma_current_ramp_up
        ),
        t_plant_pulse_fusion_ramp=Input(lambda s: s.times.t_plant_pulse_fusion_ramp),
        t_plant_pulse_burn=Input(lambda s: s.times.t_plant_pulse_burn),
        t_plant_pulse_plasma_current_ramp_down=Input(
            lambda s: s.times.t_plant_pulse_plasma_current_ramp_down
        ),
        t_plant_pulse_dwell=Input(lambda s: s.times.t_plant_pulse_dwell),
    ):
        return calculate_plant_electric_production(
            self.itart,
            self.i_tf_sup,
            p_cp_coolant_pump_elec,
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            self.ireactor,
            self.i_blkt_dual_coolant,
            self.i_p_coolant_pumping,
            p_plant_primary_heat_mw,
            p_blkt_liquid_breeder_heat_deposited_mw,
            eta_turbine,
            etath_liq,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_plant_electric_gross_mw,
            p_turbine_loss_mw,
            p_plant_electric_recirc_mw,
            p_plant_electric_net_mw,
            f_p_plant_electric_recirc,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )


class PlantElectricProductionReactor(ExplicitFunction):
    """cottax node: `calculate_plant_electric_production` at `ireactor == 1`.

    **Why this exists as a separate class, and why `PlantElectricProduction` above is
    not registerable.** `PlantElectricProduction` declares
    `p_plant_electric_gross_mw` / `p_turbine_loss_mw` / `p_plant_electric_recirc_mw` /
    `p_plant_electric_net_mw` / `f_p_plant_electric_recirc` as both `Output`s and
    `Input`s, so `to_graph` refuses it outright (*"reads [...], which it also owns"*,
    confirmed directly -- `total_process.py`'s own comment records the same refusal).
    That is not a modelling cycle: it is PROCESS's conditional-ownership
    pass-through, and it exists **only on the `ireactor == 0` arm**. Read
    `calculate_plant_electric_production`'s body: all five are assigned inside
    `if ireactor == 1:`, before `power_profiles_over_time` consumes
    `p_plant_electric_gross_mw`/`p_plant_electric_net_mw`, so on that arm not one of
    the five entering values is ever read. `.costs.ireactor` is a static switch
    (`process/main.py` resolves the cost model once per run; it is neither an
    iteration variable nor a scan variable), so which arm is live is a
    graph-assembly-time fact -- exactly `configuration.py`'s category. This class is
    the `ireactor == 1` arm with the five dead reads simply not declared, which makes
    it an ordinary acyclic node owning all 23 fields.

    **The five dead parameters are passed as `jnp.nan`, deliberately.** They are
    provably overwritten before use on this arm, so any value would do; `nan` is the
    one that makes a future edit which *starts* reading them fail loudly (a `nan`
    reaching `.heat_transport.p_plant_electric_net_mw` is caught by the very first
    comparison in `mda_harness.compare`) instead of silently substituting a zero.

    `ireactor` is therefore **not** a static field here -- it is structural, spent by
    picking this class over the `ireactor == 0` arm, which is
    `PowerProfilesOverTime` (whose 13 outputs are a strict subset of this node's, and
    which reads the two carried-over values as boundary inputs, exactly as PROCESS's
    `ireactor == 0` run does).

    Registering this closes `.heat_transport.p_plant_electric_net_mw`'s producer gap,
    which matters beyond its own value: `CostOfElectricity` reads that field, so
    without a producer `.costs.coe` -- this run's own objective -- was a function of a
    *boundary input* rather than of the design variables along that whole path, and
    constraint 16 (net electric power, an equality in
    `stellarator_helias.IN.DAT`) had no live argument at all.
    """

    itart: int = eqx.field(static=True)
    i_tf_sup: int = eqx.field(static=True)
    i_blkt_dual_coolant: int = eqx.field(static=True)
    i_p_coolant_pumping: int = eqx.field(static=True)

    p_cp_coolant_pump_elec_mw = Output(lambda s: s.power.p_cp_coolant_pump_elec_mw)
    p_plant_electric_base_total_mw = Output(
        lambda s: s.heat_transport.p_plant_electric_base_total_mw
    )
    fachtmw = Output(lambda s: s.heat_transport.fachtmw)
    p_plant_core_systems_elec_mw = Output(lambda s: s.power.p_plant_core_systems_elec_mw)
    p_plant_secondary_heat_mw = Output(
        lambda s: s.heat_transport.p_plant_secondary_heat_mw
    )
    p_plant_electric_gross_mw = Output(
        lambda s: s.heat_transport.p_plant_electric_gross_mw
    )
    p_turbine_loss_mw = Output(lambda s: s.power.p_turbine_loss_mw)
    p_plant_electric_recirc_mw = Output(
        lambda s: s.heat_transport.p_plant_electric_recirc_mw
    )
    p_plant_electric_net_mw = Output(lambda s: s.heat_transport.p_plant_electric_net_mw)
    f_p_plant_electric_recirc = Output(
        lambda s: s.heat_transport.f_p_plant_electric_recirc
    )
    e_plant_net_electric_pulse_kwh = Output(
        lambda s: s.power.e_plant_net_electric_pulse_kwh
    )
    e_plant_net_electric_pulse_mj = Output(
        lambda s: s.power.e_plant_net_electric_pulse_mj
    )
    p_plant_electric_base_total_profile_mw = Output(
        lambda s: s.power.p_plant_electric_base_total_profile_mw
    )
    p_plant_electric_gross_profile_mw = Output(
        lambda s: s.power.p_plant_electric_gross_profile_mw
    )
    p_plant_electric_net_profile_mw = Output(
        lambda s: s.power.p_plant_electric_net_profile_mw
    )
    p_hcd_electric_total_profile_mw = Output(
        lambda s: s.power.p_hcd_electric_total_profile_mw
    )
    p_coolant_pump_elec_total_profile_mw = Output(
        lambda s: s.power.p_coolant_pump_elec_total_profile_mw
    )
    p_tf_electric_supplies_profile_mw = Output(
        lambda s: s.power.p_tf_electric_supplies_profile_mw
    )
    p_pf_electric_supplies_profile_mw = Output(
        lambda s: s.power.p_pf_electric_supplies_profile_mw
    )
    vachtmw_profile_mw = Output(lambda s: s.power.vachtmw_profile_mw)
    p_tritium_plant_electric_profile_mw = Output(
        lambda s: s.power.p_tritium_plant_electric_profile_mw
    )
    p_cryo_plant_electric_profile_mw = Output(
        lambda s: s.power.p_cryo_plant_electric_profile_mw
    )
    p_fusion_total_profile_mw = Output(lambda s: s.power.p_fusion_total_profile_mw)

    def __call__(
        self,
        p_cp_coolant_pump_elec=Input(lambda s: s.tfcoil.p_cp_coolant_pump_elec),
        p_plant_electric_base=Input(lambda s: s.heat_transport.p_plant_electric_base),
        a_plant_floor_effective=Input(lambda s: s.buildings.a_plant_floor_effective),
        pflux_plant_floor_electric=Input(
            lambda s: s.heat_transport.pflux_plant_floor_electric
        ),
        p_cryo_plant_electric_mw=Input(
            lambda s: s.heat_transport.p_cryo_plant_electric_mw
        ),
        p_tf_electric_supplies_mw=Input(
            lambda s: s.heat_transport.p_tf_electric_supplies_mw
        ),
        p_tritium_plant_electric_mw=Input(
            lambda s: s.heat_transport.p_tritium_plant_electric_mw
        ),
        vachtmw=Input(lambda s: s.heat_transport.vachtmw),
        p_pf_electric_supplies_mw=Input(lambda s: s.pf_coil.p_pf_electric_supplies_mw),
        p_hcd_electric_loss_mw=Input(lambda s: s.heat_transport.p_hcd_electric_loss_mw),
        p_coolant_pump_loss_total_mw=Input(
            lambda s: s.heat_transport.p_coolant_pump_loss_total_mw
        ),
        p_div_secondary_heat_mw=Input(
            lambda s: s.heat_transport.p_div_secondary_heat_mw
        ),
        p_shld_secondary_heat_mw=Input(
            lambda s: s.heat_transport.p_shld_secondary_heat_mw
        ),
        p_hcd_secondary_heat_mw=Input(
            lambda s: s.heat_transport.p_hcd_secondary_heat_mw
        ),
        p_tf_nuclear_heat_mw=Input(lambda s: s.fwbs.p_tf_nuclear_heat_mw),
        p_plant_primary_heat_mw=Input(
            lambda s: s.heat_transport.p_plant_primary_heat_mw
        ),
        p_blkt_liquid_breeder_heat_deposited_mw=Input(
            lambda s: s.power.p_blkt_liquid_breeder_heat_deposited_mw
        ),
        eta_turbine=Input(lambda s: s.heat_transport.eta_turbine),
        etath_liq=Input(lambda s: s.heat_transport.etath_liq),
        p_hcd_electric_total_mw=Input(
            lambda s: s.heat_transport.p_hcd_electric_total_mw
        ),
        p_coolant_pump_elec_total_mw=Input(
            lambda s: s.heat_transport.p_coolant_pump_elec_total_mw
        ),
        p_fusion_total_mw=Input(lambda s: s.physics.p_fusion_total_mw),
        t_plant_pulse_coil_precharge=Input(
            lambda s: s.times.t_plant_pulse_coil_precharge
        ),
        t_plant_pulse_plasma_current_ramp_up=Input(
            lambda s: s.times.t_plant_pulse_plasma_current_ramp_up
        ),
        t_plant_pulse_fusion_ramp=Input(lambda s: s.times.t_plant_pulse_fusion_ramp),
        t_plant_pulse_burn=Input(lambda s: s.times.t_plant_pulse_burn),
        t_plant_pulse_plasma_current_ramp_down=Input(
            lambda s: s.times.t_plant_pulse_plasma_current_ramp_down
        ),
        t_plant_pulse_dwell=Input(lambda s: s.times.t_plant_pulse_dwell),
    ):
        dead = jnp.nan  # see the class docstring: provably overwritten at ireactor == 1
        return calculate_plant_electric_production(
            self.itart,
            self.i_tf_sup,
            p_cp_coolant_pump_elec,
            p_plant_electric_base,
            a_plant_floor_effective,
            pflux_plant_floor_electric,
            p_cryo_plant_electric_mw,
            p_tf_electric_supplies_mw,
            p_tritium_plant_electric_mw,
            vachtmw,
            p_pf_electric_supplies_mw,
            p_hcd_electric_loss_mw,
            p_coolant_pump_loss_total_mw,
            p_div_secondary_heat_mw,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            p_tf_nuclear_heat_mw,
            1,  # ireactor -- structural, see the class docstring
            self.i_blkt_dual_coolant,
            self.i_p_coolant_pumping,
            p_plant_primary_heat_mw,
            p_blkt_liquid_breeder_heat_deposited_mw,
            eta_turbine,
            etath_liq,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            dead,  # p_plant_electric_gross_mw
            dead,  # p_turbine_loss_mw
            dead,  # p_plant_electric_recirc_mw
            dead,  # p_plant_electric_net_mw
            dead,  # f_p_plant_electric_recirc
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )
