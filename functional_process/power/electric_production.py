"""Pure-functional port of the AC/electric-production sub-unit of
`process/models/power.py` (registry unit #14, chunk C).

Audit record: `functional_process/_audit/units/models/power/electric_production.md`.
Covers `Power.acpow` (696-813), `Power.power_profiles_over_time` (2632-2825) and
`Power.plant_electric_production` (1631-1772) -- see the audit record's data-footprint
table for the full trace.

All three are tier-1: no internal iteration, no calls into any other model. The time
axis `power_profiles_over_time` builds is always exactly 7 points (`PulseTimings.
total_pulse_cumulative` is `len()` of a fixed 7-tuple of cumulative sums over the six
phase durations, never data-dependent in length) -- so every array in this chunk has
a static, compile-time-known shape; there is no dynamic-shape difficulty here at all.
"""

import jax.numpy as jnp

from functional_process.vocabulary import PumpingPowerModelTypes


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
    if i_pf_energy_storage_source == 2:
        return calculate_acpow_line(
            p_tf_electric_supplies_mw,
            srcktpm,
            peakmva,
            p_hcd_electric_total_mw,
            p_cryo_plant_electric_mw,
            vachtmw,
            p_coolant_pump_elec_total_mw,
            p_tritium_plant_electric_mw,
            p_plant_electric_base_total_mw,
        )
    return calculate_acpow_motor_generator_flywheel(
        p_tf_electric_supplies_mw,
        srcktpm,
        p_hcd_electric_total_mw,
        p_cryo_plant_electric_mw,
        vachtmw,
        p_coolant_pump_elec_total_mw,
        p_tritium_plant_electric_mw,
        p_plant_electric_base_total_mw,
        fmgdmw,
    )


def calculate_acpow_line(
    p_tf_electric_supplies_mw,
    srcktpm,
    peakmva,
    p_hcd_electric_total_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_coolant_pump_elec_total_mw,
    p_tritium_plant_electric_mw,
    p_plant_electric_base_total_mw,
):
    """`i_pf_energy_storage_source == LINE` (2) -- all pulsed power from the line, the
    reference run's.

    The PF power draw carries the peak MVA (`.heat_transport.peakmva`) and **no**
    motor-generator-flywheel term, so `.heat_transport.fmgdmw` is not read at all. Its
    sibling is the exact mirror: `fmgdmw` and no `peakmva`. One edge each way, and the
    two are complementary, which is why `switch_kwarg_survey.md` band (b3) called this
    the cleanest conversion in the survey.
    """
    return _acpow(
        1.0e-3 * srcktpm + peakmva,
        0.0,
        p_tf_electric_supplies_mw,
        p_hcd_electric_total_mw,
        p_cryo_plant_electric_mw,
        vachtmw,
        p_coolant_pump_elec_total_mw,
        p_tritium_plant_electric_mw,
        p_plant_electric_base_total_mw,
    )


def calculate_acpow_motor_generator_flywheel(
    p_tf_electric_supplies_mw,
    srcktpm,
    p_hcd_electric_total_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_coolant_pump_elec_total_mw,
    p_tritium_plant_electric_mw,
    p_plant_electric_base_total_mw,
    fmgdmw,
):
    """`i_pf_energy_storage_source == MGF` (1) -- all power from motor-generator
    flywheel units.

    Reads `.heat_transport.fmgdmw` and **not** `.heat_transport.peakmva`.
    """
    return _acpow(
        1.0e-3 * srcktpm,
        fmgdmw,
        p_tf_electric_supplies_mw,
        p_hcd_electric_total_mw,
        p_cryo_plant_electric_mw,
        vachtmw,
        p_coolant_pump_elec_total_mw,
        p_tritium_plant_electric_mw,
        p_plant_electric_base_total_mw,
    )


def _acpow(
    ppfmw,
    p_flywheel_mw,
    p_tf_electric_supplies_mw,
    p_hcd_electric_total_mw,
    p_cryo_plant_electric_mw,
    vachtmw,
    p_coolant_pump_elec_total_mw,
    p_tritium_plant_electric_mw,
    p_plant_electric_base_total_mw,
):
    """The two sums both `i_pf_energy_storage_source` arms share, given the arm's own
    PF power draw and its flywheel contribution (a literal `0.0` on the line arm).
    """
    ptfmw = p_tf_electric_supplies_mw
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
    pacpmw = pacpmw + p_flywheel_mw

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
        `electric_production.md` for each field's `VarPath`.
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
    field, not yet ported -- an ordinary `FromExactly`, not a call into unit #18's
    methods): PROCESS only computes these five when `ireactor == 1`; otherwise they
    keep whatever value they entered this call with. `power_profiles_over_time` is
    then called **unconditionally**, using `p_plant_electric_gross_mw`/
    `p_plant_electric_net_mw` regardless of whether this call just computed them or
    they are carried over stale/default -- ported faithfully, not resolved, same
    treatment `thermal_cryo.md` gives `.power.delta_eta`.

    Parameters
    ----------
    itart, i_tf_sup :
        Static switches gating `p_cp_coolant_pump_elec_mw`'s ownership (only owned
        when `itart == 1 and i_tf_sup == 0`, a tight-aspect-ratio/resistive-centrepost
        configuration; always `0.0` otherwise, no other producer). `.physics.itart`,
        `.tfcoil.i_tf_sup`.
    ireactor, i_blkt_dual_coolant, i_p_coolant_pumping :
        Static switches -- see docstring body above and
        `electric_production.md`.
    (all other parameters) :
        See `electric_production.md`'s data-footprint table for each
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
        p_cp_coolant_pump_elec_mw = centrepost_coolant_pump_power_resistive(
            p_cp_coolant_pump_elec
        )
    else:
        p_cp_coolant_pump_elec_mw = centrepost_coolant_pump_power_absent()

    if ireactor != 1:
        return _plant_electric_production_carried_over(
            p_cp_coolant_pump_elec_mw,
            p_plant_electric_gross_mw,
            p_turbine_loss_mw,
            p_plant_electric_recirc_mw,
            p_plant_electric_net_mw,
            f_p_plant_electric_recirc,
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
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        )

    if (
        i_blkt_dual_coolant > 0
        and i_p_coolant_pumping == PumpingPowerModelTypes.MECHANICAL
    ):
        p_plant_electric_gross_mw = gross_electric_power_liquid_breeder(
            p_plant_primary_heat_mw,
            eta_turbine,
            p_blkt_liquid_breeder_heat_deposited_mw,
            etath_liq,
        )
    else:
        p_plant_electric_gross_mw = gross_electric_power_single_coolant(
            p_plant_primary_heat_mw, eta_turbine
        )
    return calculate_plant_electric_production_reactor(
        p_cp_coolant_pump_elec_mw,
        p_plant_electric_gross_mw,
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
        p_plant_primary_heat_mw,
        eta_turbine,
        p_hcd_electric_total_mw,
        p_coolant_pump_elec_total_mw,
        p_fusion_total_mw,
        t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_fusion_ramp,
        t_plant_pulse_burn,
        t_plant_pulse_plasma_current_ramp_down,
        t_plant_pulse_dwell,
    )


def calculate_plant_electric_production_reactor(
    p_cp_coolant_pump_elec_mw,
    p_plant_electric_gross_mw,
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
    p_plant_primary_heat_mw,
    eta_turbine,
    p_hcd_electric_total_mw,
    p_coolant_pump_elec_total_mw,
    p_fusion_total_mw,
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_dwell,
):
    """`Power.plant_electric_production` at `ireactor == 1`, given the two quantities
    the four `(itart, i_tf_sup)` x `(i_blkt_dual_coolant, i_p_coolant_pumping)` arms
    disagree about: the centrepost coolant pump's electric power and the gross electric
    power.

    Both are **data**, not switches: which law produced each follows from which
    occupant of `.availability.electric_production` is assembled
    (`indat.py`'s `_centrepost_coolant_pump_arm` / `_gross_electric_power_arm`). That is
    what lets the conventional/single-coolant occupant declare neither
    `.tfcoil.p_cp_coolant_pump_elec` nor `.heat_transport.etath_liq` nor
    `.power.p_blkt_liquid_breeder_heat_deposited_mw` -- three reads no such machine
    makes (`_audit/next_steps.md` §14.2).
    """
    (
        p_plant_electric_base_total_mw,
        fachtmw,
        p_plant_core_systems_elec_mw,
        p_plant_secondary_heat_mw,
    ) = _plant_core_and_secondary(
        p_cp_coolant_pump_elec_mw,
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
        p_plant_primary_heat_mw,
        eta_turbine,
        p_hcd_electric_total_mw,
        p_coolant_pump_elec_total_mw,
        p_fusion_total_mw,
        t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_fusion_ramp,
        t_plant_pulse_burn,
        t_plant_pulse_plasma_current_ramp_down,
        t_plant_pulse_dwell,
    )

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
        *_plant_electric_profiles(
            p_plant_electric_base_total_mw,
            p_plant_electric_gross_mw,
            p_plant_electric_net_mw,
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
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        ),
    )


def _plant_electric_production_carried_over(
    p_cp_coolant_pump_elec_mw,
    p_plant_electric_gross_mw,
    p_turbine_loss_mw,
    p_plant_electric_recirc_mw,
    p_plant_electric_net_mw,
    f_p_plant_electric_recirc,
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
    p_plant_primary_heat_mw,
    eta_turbine,
    p_hcd_electric_total_mw,
    p_coolant_pump_elec_total_mw,
    p_fusion_total_mw,
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_dwell,
):
    """`Power.plant_electric_production` at `ireactor != 1`: the five electric-power
    fields keep whatever value they entered with (`power.py:1686`'s `if`), and
    `power_profiles_over_time` is called with them regardless.

    Not reachable from any occupant in this port -- `.availability.electric_production`
    holds `PowerProfilesOverTime` on that arm, which is the same 13 profile outputs and
    nothing else -- but kept so the composite above stays PROCESS's whole method, which
    is what the harness diffs against.
    """
    (
        p_plant_electric_base_total_mw,
        fachtmw,
        p_plant_core_systems_elec_mw,
        p_plant_secondary_heat_mw,
    ) = _plant_core_and_secondary(
        p_cp_coolant_pump_elec_mw,
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
        p_plant_primary_heat_mw,
        eta_turbine,
        p_hcd_electric_total_mw,
        p_coolant_pump_elec_total_mw,
        p_fusion_total_mw,
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
        *_plant_electric_profiles(
            p_plant_electric_base_total_mw,
            p_plant_electric_gross_mw,
            p_plant_electric_net_mw,
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
            p_plant_primary_heat_mw,
            eta_turbine,
            p_hcd_electric_total_mw,
            p_coolant_pump_elec_total_mw,
            p_fusion_total_mw,
            t_plant_pulse_coil_precharge,
            t_plant_pulse_plasma_current_ramp_up,
            t_plant_pulse_fusion_ramp,
            t_plant_pulse_burn,
            t_plant_pulse_plasma_current_ramp_down,
            t_plant_pulse_dwell,
        ),
    )


def _plant_core_and_secondary(
    p_cp_coolant_pump_elec_mw,
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
    p_plant_primary_heat_mw,
    eta_turbine,
    p_hcd_electric_total_mw,
    p_coolant_pump_elec_total_mw,
    p_fusion_total_mw,
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_dwell,
):
    """The base, core-systems and secondary-heat sums, which every arm shares
    (`power.py:1652-1684`).
    """
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
    return (
        p_plant_electric_base_total_mw,
        fachtmw,
        p_plant_core_systems_elec_mw,
        p_plant_secondary_heat_mw,
    )


def _plant_electric_profiles(
    p_plant_electric_base_total_mw,
    p_plant_electric_gross_mw,
    p_plant_electric_net_mw,
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
    p_plant_primary_heat_mw,
    eta_turbine,
    p_hcd_electric_total_mw,
    p_coolant_pump_elec_total_mw,
    p_fusion_total_mw,
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_dwell,
):
    """`power_profiles_over_time`, called unconditionally at the end of the method --
    the thirteen profile outputs, in PROCESS's own order.
    """
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


def centrepost_coolant_pump_power_resistive(p_cp_coolant_pump_elec):
    """`itart == SPHERICAL_TOKAMAK` with `i_tf_sup == COPPER`: the centrepost coolant
    pump draws real electric power (`power.py:1648`), read from
    `.tfcoil.p_cp_coolant_pump_elec`.
    """
    return 1.0e-6 * p_cp_coolant_pump_elec


def centrepost_coolant_pump_power_absent():
    """Every other `(itart, i_tf_sup)` pair: there is no resistive centrepost to cool,
    so PROCESS writes a literal `0.0` and **`.tfcoil.p_cp_coolant_pump_elec` is not read
    at all** (`power.py:1650`).
    """
    return 0.0e0


def gross_electric_power_liquid_breeder(
    p_plant_primary_heat_mw,
    eta_turbine,
    p_blkt_liquid_breeder_heat_deposited_mw,
    etath_liq,
):
    """`i_blkt_dual_coolant > 0` with `i_p_coolant_pumping == MECHANICAL`: the liquid
    breeder's heat goes through its own turbine efficiency (`power.py:1690-1693`).

    Reads `.power.p_blkt_liquid_breeder_heat_deposited_mw` and
    `.heat_transport.etath_liq`, which the single-coolant law does not.
    """
    return (
        p_plant_primary_heat_mw - p_blkt_liquid_breeder_heat_deposited_mw
    ) * eta_turbine + p_blkt_liquid_breeder_heat_deposited_mw * etath_liq


def gross_electric_power_single_coolant(p_plant_primary_heat_mw, eta_turbine):
    """Every other `(i_blkt_dual_coolant, i_p_coolant_pumping)` pair: one coolant, one
    turbine efficiency (`power.py:1695`).
    """
    return p_plant_primary_heat_mw * eta_turbine


def calculate_plant_electric_production_resistive_centrepost_liquid_breeder(
    p_cp_coolant_pump_elec,
    p_blkt_liquid_breeder_heat_deposited_mw,
    etath_liq,
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
    p_plant_primary_heat_mw,
    eta_turbine,
    p_hcd_electric_total_mw,
    p_coolant_pump_elec_total_mw,
    p_fusion_total_mw,
    t_plant_pulse_coil_precharge,
    t_plant_pulse_plasma_current_ramp_up,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    t_plant_pulse_plasma_current_ramp_down,
    t_plant_pulse_dwell,
):
    """`PlantElectricProductionResistiveCentrepostLiquidBreeder`'s own composition of
    `centrepost_coolant_pump_power_resistive` and `gross_electric_power_liquid_breeder`
    -- the two quantities the arm needs before the 23-output call -- moved out of the
    declaration and into a named function (`_audit/formulas_split.md` step 1). Calls
    `calculate_plant_electric_production_reactor` directly rather than through
    `PlantElectricProductionReactor._production`: that method is itself pure
    delegation to the same function, and a module-level function has no `self` to call
    it through.
    """
    return calculate_plant_electric_production_reactor(
        centrepost_coolant_pump_power_resistive(p_cp_coolant_pump_elec),
        gross_electric_power_liquid_breeder(
            p_plant_primary_heat_mw,
            eta_turbine,
            p_blkt_liquid_breeder_heat_deposited_mw,
            etath_liq,
        ),
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
        p_plant_primary_heat_mw,
        eta_turbine,
        p_hcd_electric_total_mw,
        p_coolant_pump_elec_total_mw,
        p_fusion_total_mw,
        t_plant_pulse_coil_precharge,
        t_plant_pulse_plasma_current_ramp_up,
        t_plant_pulse_fusion_ramp,
        t_plant_pulse_burn,
        t_plant_pulse_plasma_current_ramp_down,
        t_plant_pulse_dwell,
    )
