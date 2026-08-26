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

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CostOfElectricityModel,
    SphericalTokamakModel,
)
from functional_process.paths import (
    buildings,
    fwbs,
    heat_transport,
    pf_coil,
    pf_power,
    physics,
    power,
    tfcoil,
    times,
)
from process.models.power import PumpingPowerModelTypes
from process.models.tfcoil.base import TFConductorModel


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


class Acpow(ExplicitFunction):
    """The `calculate_acpow` family -- one occupant per
    `.pf_power.i_pf_energy_storage_source` value.

    **The switch was an `eqx.field(static=True)` here and is gone**
    (`_audit/next_steps.md` §14.2). The two arms' reads are **complementary**: the line
    arm reads `.heat_transport.peakmva` and not `fmgdmw`, the flywheel arm the reverse.
    One node carrying the switch declared both, so exactly one edge was invented either
    way -- the smallest and cleanest case in `switch_kwarg_survey.md` band (b3).
    """

    pacpmw = OutputInto(heat_transport)
    tlvpmw = OutputInto(heat_transport)


class AcpowLine(Acpow):
    """`i_pf_energy_storage_source == LINE` (2) -- the reference run's.

    **One read leaves with this occupant**: `.heat_transport.fmgdmw`.
    """

    def __call__(
        self,
        p_tf_electric_supplies_mw=From(heat_transport),
        srcktpm=From(pf_power),
        peakmva=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        p_plant_electric_base_total_mw=From(heat_transport),
    ):
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


class AcpowMotorGeneratorFlywheel(Acpow):
    """`i_pf_energy_storage_source == MGF` (1) -- all power from motor-generator
    flywheel units, PROCESS's own default (`pf_power_variables.py:18`).

    **Reads `.heat_transport.fmgdmw` and not `.heat_transport.peakmva`.**
    """

    def __call__(
        self,
        p_tf_electric_supplies_mw=From(heat_transport),
        srcktpm=From(pf_power),
        p_hcd_electric_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        p_plant_electric_base_total_mw=From(heat_transport),
        fmgdmw=From(heat_transport),
    ):
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


class PowerProfilesOverTime(ExplicitFunction):
    """cottax node: `power_profiles_over_time`."""

    e_plant_net_electric_pulse_kwh = OutputInto(power)
    e_plant_net_electric_pulse_mj = OutputInto(power)
    p_plant_electric_base_total_profile_mw = OutputInto(power)
    p_plant_electric_gross_profile_mw = OutputInto(power)
    p_plant_electric_net_profile_mw = OutputInto(power)
    p_hcd_electric_total_profile_mw = OutputInto(power)
    p_coolant_pump_elec_total_profile_mw = OutputInto(power)
    p_tf_electric_supplies_profile_mw = OutputInto(power)
    p_pf_electric_supplies_profile_mw = OutputInto(power)
    vachtmw_profile_mw = OutputInto(power)
    p_tritium_plant_electric_profile_mw = OutputInto(power)
    p_cryo_plant_electric_profile_mw = OutputInto(power)
    p_fusion_total_profile_mw = OutputInto(power)

    def __call__(
        self,
        p_plant_electric_base_total_mw=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        p_plant_electric_gross_mw=From(heat_transport),
        p_plant_electric_net_mw=From(heat_transport),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
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

    itart: SphericalTokamakModel = eqx.field(static=True)
    i_tf_sup: TFConductorModel = eqx.field(static=True)
    ireactor: CostOfElectricityModel = eqx.field(static=True)
    i_blkt_dual_coolant: BlanketDualCoolantModel = eqx.field(static=True)
    i_p_coolant_pumping: PumpingPowerModelTypes = eqx.field(static=True)

    p_cp_coolant_pump_elec_mw = OutputInto(power)
    p_plant_electric_base_total_mw = OutputInto(heat_transport)
    fachtmw = OutputInto(heat_transport)
    p_plant_core_systems_elec_mw = OutputInto(power)
    p_plant_secondary_heat_mw = OutputInto(heat_transport)
    p_plant_electric_gross_mw = OutputInto(heat_transport)
    p_turbine_loss_mw = OutputInto(power)
    p_plant_electric_recirc_mw = OutputInto(heat_transport)
    p_plant_electric_net_mw = OutputInto(heat_transport)
    f_p_plant_electric_recirc = OutputInto(heat_transport)
    e_plant_net_electric_pulse_kwh = OutputInto(power)
    e_plant_net_electric_pulse_mj = OutputInto(power)
    p_plant_electric_base_total_profile_mw = OutputInto(power)
    p_plant_electric_gross_profile_mw = OutputInto(power)
    p_plant_electric_net_profile_mw = OutputInto(power)
    p_hcd_electric_total_profile_mw = OutputInto(power)
    p_coolant_pump_elec_total_profile_mw = OutputInto(power)
    p_tf_electric_supplies_profile_mw = OutputInto(power)
    p_pf_electric_supplies_profile_mw = OutputInto(power)
    vachtmw_profile_mw = OutputInto(power)
    p_tritium_plant_electric_profile_mw = OutputInto(power)
    p_cryo_plant_electric_profile_mw = OutputInto(power)
    p_fusion_total_profile_mw = OutputInto(power)

    def __call__(
        self,
        p_cp_coolant_pump_elec=From(tfcoil),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        p_blkt_liquid_breeder_heat_deposited_mw=From(power),
        eta_turbine=From(heat_transport),
        etath_liq=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_plant_electric_gross_mw=From(heat_transport),
        p_turbine_loss_mw=From(power),
        p_plant_electric_recirc_mw=From(heat_transport),
        p_plant_electric_net_mw=From(heat_transport),
        f_p_plant_electric_recirc=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
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
    """The `calculate_plant_electric_production` family at `ireactor == 1` -- one
    occupant per `(itart, i_tf_sup)` x `(i_blkt_dual_coolant,
    i_p_coolant_pumping)` arm pair.

    **Why this exists as a separate class, and why `PlantElectricProduction` above is
    not registerable.** `PlantElectricProduction` declares
    `p_plant_electric_gross_mw` / `p_turbine_loss_mw` / `p_plant_electric_recirc_mw` /
    `p_plant_electric_net_mw` / `f_p_plant_electric_recirc` as both `Output`s and
    `FromExactly`s, so `to_graph` refuses it outright (*"reads [...], which it also owns"*,
    confirmed directly -- `total_process.py`'s own comment records the same refusal).
    That is not a modelling cycle: it is PROCESS's conditional-ownership
    pass-through, and it exists **only on the `ireactor == 0` arm**. Read
    `calculate_plant_electric_production`'s body: all five are assigned inside
    `if ireactor == 1:`, before `power_profiles_over_time` consumes
    `p_plant_electric_gross_mw`/`p_plant_electric_net_mw`, so on that arm not one of
    the five entering values is ever read. `.costs.ireactor` is a static switch
    (`process/main.py` resolves the cost model once per run; it is neither an
    iteration variable nor a scan variable), so which arm is live is a
    graph-assembly-time fact -- exactly `machine_from_indat`'s category. This class is
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

    **All four switches were `eqx.field(static=True)`s here and none is now**
    (`_audit/next_steps.md` §14.2). They gate two things and nothing else: whether the
    centrepost coolant pump draws electric power (`itart == 1 and i_tf_sup == 0`) and
    whether the liquid breeder has its own turbine efficiency
    (`i_blkt_dual_coolant > 0 and i_p_coolant_pumping == MECHANICAL`). Both conditions
    are joint, so this is two arm indices rather than four switches, and the four
    occupants below are their product.

    **Three reads leave with the conventional/single-coolant occupant**:
    `.tfcoil.p_cp_coolant_pump_elec`, `.heat_transport.etath_liq` and
    `.power.p_blkt_liquid_breeder_heat_deposited_mw`. The first is the `.tfcoil ->
    .power` edge `switch_kwarg_survey.md` §3 records as `live (1)` for `itart`; the
    other two are its `live (2)` for `i_blkt_dual_coolant`. Both were reported "(joint)"
    there because neither switch decides them alone -- which is exactly why the two arm
    indices are joint here too.
    """

    p_cp_coolant_pump_elec_mw = OutputInto(power)
    p_plant_electric_base_total_mw = OutputInto(heat_transport)
    fachtmw = OutputInto(heat_transport)
    p_plant_core_systems_elec_mw = OutputInto(power)
    p_plant_secondary_heat_mw = OutputInto(heat_transport)
    p_plant_electric_gross_mw = OutputInto(heat_transport)
    p_turbine_loss_mw = OutputInto(power)
    p_plant_electric_recirc_mw = OutputInto(heat_transport)
    p_plant_electric_net_mw = OutputInto(heat_transport)
    f_p_plant_electric_recirc = OutputInto(heat_transport)
    e_plant_net_electric_pulse_kwh = OutputInto(power)
    e_plant_net_electric_pulse_mj = OutputInto(power)
    p_plant_electric_base_total_profile_mw = OutputInto(power)
    p_plant_electric_gross_profile_mw = OutputInto(power)
    p_plant_electric_net_profile_mw = OutputInto(power)
    p_hcd_electric_total_profile_mw = OutputInto(power)
    p_coolant_pump_elec_total_profile_mw = OutputInto(power)
    p_tf_electric_supplies_profile_mw = OutputInto(power)
    p_pf_electric_supplies_profile_mw = OutputInto(power)
    vachtmw_profile_mw = OutputInto(power)
    p_tritium_plant_electric_profile_mw = OutputInto(power)
    p_cryo_plant_electric_profile_mw = OutputInto(power)
    p_fusion_total_profile_mw = OutputInto(power)

    def _production(
        self,
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
        """The twenty-three outputs, given the two quantities the four arms
        disagree about.

        Not a port surface: `_params` reads `__call__`'s signature only
        (`ExplicitFunction._signature_of`), so what each occupant declares is
        still its own parameter list.
        """
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


class PlantElectricProductionSingleCoolant(PlantElectricProductionReactor):
    """No resistive centrepost, one coolant -- the reference run's and the
    conventional tokamak's (`itart = 0`, `i_blkt_dual_coolant = 0`).

    Declares **none** of `.tfcoil.p_cp_coolant_pump_elec`,
    `.power.p_blkt_liquid_breeder_heat_deposited_mw`,
    `.heat_transport.etath_liq`.
    """

    def __call__(
        self,
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_absent(),
            gross_electric_power_single_coolant(p_plant_primary_heat_mw, eta_turbine),
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


class PlantElectricProductionLiquidBreeder(PlantElectricProductionReactor):
    """No resistive centrepost; liquid breeder with its own turbine efficiency
    (`i_blkt_dual_coolant > 0` and `i_p_coolant_pumping == MECHANICAL`).
    """

    def __call__(
        self,
        p_blkt_liquid_breeder_heat_deposited_mw=From(power),
        etath_liq=From(heat_transport),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_absent(),
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


class PlantElectricProductionResistiveCentrepostSingleCoolant(
    PlantElectricProductionReactor
):
    """Resistive centrepost (`itart == 1` and `i_tf_sup == 0`), one coolant.

    Reads `.tfcoil.p_cp_coolant_pump_elec`, which the two conventional
    occupants do not.
    """

    def __call__(
        self,
        p_cp_coolant_pump_elec=From(tfcoil),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
            centrepost_coolant_pump_power_resistive(p_cp_coolant_pump_elec),
            gross_electric_power_single_coolant(p_plant_primary_heat_mw, eta_turbine),
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


class PlantElectricProductionResistiveCentrepostLiquidBreeder(
    PlantElectricProductionReactor
):
    """Resistive centrepost and liquid breeder -- both extra reads at once."""

    def __call__(
        self,
        p_cp_coolant_pump_elec=From(tfcoil),
        p_blkt_liquid_breeder_heat_deposited_mw=From(power),
        etath_liq=From(heat_transport),
        p_plant_electric_base=From(heat_transport),
        a_plant_floor_effective=From(buildings),
        pflux_plant_floor_electric=From(heat_transport),
        p_cryo_plant_electric_mw=From(heat_transport),
        p_tf_electric_supplies_mw=From(heat_transport),
        p_tritium_plant_electric_mw=From(heat_transport),
        vachtmw=From(heat_transport),
        p_pf_electric_supplies_mw=From(pf_coil),
        p_hcd_electric_loss_mw=From(heat_transport),
        p_coolant_pump_loss_total_mw=From(heat_transport),
        p_div_secondary_heat_mw=From(heat_transport),
        p_shld_secondary_heat_mw=From(heat_transport),
        p_hcd_secondary_heat_mw=From(heat_transport),
        p_tf_nuclear_heat_mw=From(fwbs),
        p_plant_primary_heat_mw=From(heat_transport),
        eta_turbine=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_coolant_pump_elec_total_mw=From(heat_transport),
        p_fusion_total_mw=From(physics),
        t_plant_pulse_coil_precharge=From(times),
        t_plant_pulse_plasma_current_ramp_up=From(times),
        t_plant_pulse_fusion_ramp=From(times),
        t_plant_pulse_burn=From(times),
        t_plant_pulse_plasma_current_ramp_down=From(times),
        t_plant_pulse_dwell=From(times),
    ):
        return self._production(
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
