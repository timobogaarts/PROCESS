"""Pure-functional port of the thermal-power-balance and cryogenics sub-unit of
`process/models/power.py` (registry unit #14, chunk B).

Audit record: `functional_process/models/power_B_thermal_cryo.md`. Covers
`Power.component_thermal_powers` (814-1036), `Power.plant_thermal_efficiency`
(1935-2071), `Power.plant_thermal_efficiency_2` (2073-2116),
`Power.calculate_cryo_loads` (1037-1118) and `Power.cryo` (1773-1852) -- see the
audit record's data-footprint table for the full trace and, in particular, the
`.power.delta_eta` self-loop finding (§ Real findings #1), which is the reason
`calculate_component_thermal_powers` is not a fully acyclic node.

All five source methods are tier-1 once ported: no internal iteration anywhere in
this chunk (the `cryo`/`calculate_cryo_loads` pair looks like it could be tier-2 from
its name, but it is a single straight-line evaluation, not a solve).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

from process.core import constants
from process.core.exceptions import ProcessValueError
from process.data_structure.blanket_variables import BlktModelTypes
from process.data_structure.pfcoil_variables import PFConductorModel
from process.models.power import ElectricConversionModelTypes, PumpingPowerModelTypes


def calculate_plant_thermal_efficiency(
    eta_turbine,
    delta_eta,
    temp_blkt_coolant_out,
    temp_turbine_coolant_in,
    i_thermal_electric_conversion,
    i_blanket_type,
):
    """`Power.plant_thermal_efficiency`.

    Ports `process/models/power.py:1935-2071`. `i_thermal_electric_conversion` and
    `i_blanket_type` are static configuration switches (per
    `_audit/naming_convention.md`), not `Input`s -- ordinary Python `if`/`elif`, not
    `jnp.where`, since each of the five values reads/writes a genuinely different set
    of fields (not the "identical reads-set" exception `FastAlphaBeta`'s
    `i_beta_fast_alpha` uses), and only one is ever selected for a given run.

    **Real finding, ported around, not reproduced**: PROCESS's own source
    (`power.py:2038`) compares against
    `ElectricConversionModelTypes.SUPERCRITICAL_CO2_CYCLE`, an attribute that does not
    exist on the enum (the real member is `SUPERCRITICAL_CO2_BRAYTON_CYCLE`, value 4)
    -- calling `Power.plant_thermal_efficiency` with
    `i_thermal_electric_conversion == 4` therefore raises `AttributeError` in
    PROCESS today, unconditionally, before any physics runs. This port uses the
    correct member name (`SUPERCRITICAL_CO2_BRAYTON_CYCLE`) since the intent is
    unambiguous from the branch's own body (a working supercritical-CO2 correlation),
    but the two are **not equivalent for testing purposes**: the harness's samples
    cannot include `i_thermal_electric_conversion == 4` and diff against PROCESS's
    reference, because PROCESS crashes there. See `power_B_thermal_cryo.md`.

    Parameters
    ----------
    eta_turbine :
        Thermal-to-electric conversion efficiency, in/out.
        `.heat_transport.eta_turbine`.
    delta_eta :
        Loss in efficiency from low-grade divertor heat collection.
        `.power.delta_eta`. Read here as the value **already in `data`** -- see
        `power_B_thermal_cryo.md`'s self-loop finding; this function does not compute
        it (`calculate_component_thermal_powers` does, later in the same PROCESS
        call, from *this* call's `eta_turbine`/`f_p_div_primary_heat` outputs).
    temp_blkt_coolant_out :
        Blanket coolant outlet temperature (K). `.fwbs.temp_blkt_coolant_out`.
    temp_turbine_coolant_in :
        Turbine coolant inlet temperature (K), in/out.
        `.heat_transport.temp_turbine_coolant_in`. Two of the five branches write it;
        the other three pass it through unchanged (see `power_B_thermal_cryo.md`'s
        finding on this field's write ordering against `plant_thermal_efficiency_2`).
    i_thermal_electric_conversion :
        Static switch, `ElectricConversionModelTypes`. `.fwbs.i_thermal_electric_conversion`.
    i_blanket_type :
        Static switch, `BlktModelTypes`. `.fwbs.i_blanket_type`.

    Returns
    -------
    :
        `(eta_turbine, temp_turbine_coolant_in)`.
    """
    if i_thermal_electric_conversion == ElectricConversionModelTypes.CCFE_HCPB_VALUE:
        if i_blanket_type == BlktModelTypes.CCFE_HCPB:
            eta_turbine = 0.411e0
        return eta_turbine, temp_turbine_coolant_in

    if (
        i_thermal_electric_conversion
        == ElectricConversionModelTypes.CCFE_HCPB_VALUE_WITH_DIVERTOR
    ):
        if i_blanket_type == BlktModelTypes.CCFE_HCPB:
            eta_turbine = 0.411e0 - delta_eta
        return eta_turbine, temp_turbine_coolant_in

    if i_thermal_electric_conversion == ElectricConversionModelTypes.USER_INPUT:
        return eta_turbine, temp_turbine_coolant_in

    if (
        i_thermal_electric_conversion
        == ElectricConversionModelTypes.STEAM_RANKINE_CYCLE
    ):
        if i_blanket_type == BlktModelTypes.CCFE_HCPB:
            temp_turbine_coolant_in = temp_blkt_coolant_out - 20.0e0
            eta_turbine = (
                0.1802e0 * jnp.log(temp_turbine_coolant_in) - 0.7823e0 - delta_eta
            )
        return eta_turbine, temp_turbine_coolant_in

    if (
        i_thermal_electric_conversion
        == ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
    ):
        temp_turbine_coolant_in = temp_blkt_coolant_out - 20.0e0
        eta_turbine = 0.4347e0 * jnp.log(temp_turbine_coolant_in) - 2.5043e0
        return eta_turbine, temp_turbine_coolant_in

    return eta_turbine, temp_turbine_coolant_in


def calculate_plant_thermal_efficiency_2(
    etath_liq,
    outlet_temp_liq,
    temp_turbine_coolant_in,
    secondary_cycle_liq,
):
    """`Power.plant_thermal_efficiency_2`.

    Ports `process/models/power.py:2073-2116`. `secondary_cycle_liq` is a static
    switch; PROCESS raises `ProcessValueError` for any value other than `2`/`4`
    (`power.py:2112-2115`) -- kept as a plain Python `raise` here since the switch is
    resolved outside any traced computation, not routed through the "return
    non-finite" convention `Tier1Contract.reference_domain_errors` exists for
    (that convention is for *continuous* out-of-domain inputs a traced function
    cannot reject; a static config value is rejected in ordinary Python before
    tracing begins, same as `i_thermal_electric_conversion`'s branch selection
    above).

    Parameters
    ----------
    etath_liq :
        Liquid-breeder thermal-to-electric efficiency, in/out. `.heat_transport.etath_liq`.
    outlet_temp_liq :
        Liquid-breeder outlet temperature (K). `.fwbs.outlet_temp_liq`.
    temp_turbine_coolant_in :
        Turbine coolant inlet temperature (K), in/out.
        `.heat_transport.temp_turbine_coolant_in` -- the *same* field
        `calculate_plant_thermal_efficiency` may also write; whichever of the two
        calls happens last in `calculate_component_thermal_powers` determines the
        final value (see `power_B_thermal_cryo.md`).
    secondary_cycle_liq :
        Static switch (`2` user input / `4` supercritical CO2). `.fwbs.secondary_cycle_liq`.

    Returns
    -------
    :
        `(etath_liq, temp_turbine_coolant_in)`.

    Raises
    ------
    ProcessValueError
        If `secondary_cycle_liq` is not `2` or `4`.
    """
    if secondary_cycle_liq == 2:
        return etath_liq, temp_turbine_coolant_in

    if secondary_cycle_liq == 4:
        temp_turbine_coolant_in = outlet_temp_liq - 20.0e0
        etath_liq = 0.4347e0 * jnp.log(temp_turbine_coolant_in) - 2.5043e0
        return etath_liq, temp_turbine_coolant_in

    raise ProcessValueError(
        f"secondary_cycle_liq ={secondary_cycle_liq} is an invalid option."
    )


def calculate_p_fw_blkt_coolant_pump_mw(
    i_p_coolant_pumping,
    p_fw_coolant_pump_mw,
    p_blkt_coolant_pump_mw,
    p_fw_blkt_coolant_pump_mw,
):
    """The conditional-ownership pass-through, extracted verbatim from
    `calculate_component_thermal_powers` (`power.py:815-820`) so both it and
    `DeltaEtaStep` (the `FixedPointFunction` isolating the `.power.delta_eta`
    self-loop, below) share one body instead of two hand-copies of the same branch.
    See `calculate_component_thermal_powers`'s own docstring for the
    conditional-ownership finding this implements.
    """
    if i_p_coolant_pumping not in (
        PumpingPowerModelTypes.MECHANICAL,
        PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP,
    ):
        return p_fw_coolant_pump_mw + p_blkt_coolant_pump_mw
    return p_fw_blkt_coolant_pump_mw


def calculate_p_fw_blkt_heat_deposited_mw(
    i_blkt_dual_coolant,
    p_fw_nuclear_heat_total_mw,
    p_fw_rad_total_mw,
    p_blkt_nuclear_heat_total_mw,
    p_blkt_breeder_pump_mw,
    p_fw_blkt_coolant_pump_mw,
    p_beam_orbit_loss_mw,
    p_fw_alpha_mw,
    p_beam_shine_through_mw,
):
    """Extracted verbatim from `calculate_component_thermal_powers`
    (`power.py:857-878`) -- see that function's docstring, reused by `DeltaEtaStep`
    below."""
    if i_blkt_dual_coolant in (1, 2):
        return (
            p_fw_nuclear_heat_total_mw
            + p_fw_rad_total_mw
            + p_blkt_nuclear_heat_total_mw
            + p_blkt_breeder_pump_mw
            + p_fw_blkt_coolant_pump_mw
            + p_beam_orbit_loss_mw
            + p_fw_alpha_mw
            + p_beam_shine_through_mw
        )
    return (
        p_fw_nuclear_heat_total_mw
        + p_fw_rad_total_mw
        + p_blkt_nuclear_heat_total_mw
        + p_fw_blkt_coolant_pump_mw
        + p_beam_orbit_loss_mw
        + p_fw_alpha_mw
        + p_beam_shine_through_mw
    )


def calculate_p_shld_heat_deposited_mw(
    p_cp_shield_nuclear_heat_mw, p_shld_nuclear_heat_mw, p_shld_coolant_pump_mw
):
    """Extracted verbatim from `calculate_component_thermal_powers`
    (`power.py:895-897`), reused by `DeltaEtaStep` below."""
    return p_cp_shield_nuclear_heat_mw + p_shld_nuclear_heat_mw + p_shld_coolant_pump_mw


def calculate_p_div_heat_deposited_mw(
    p_plasma_separatrix_mw,
    p_div_nuclear_heat_total_mw,
    p_div_rad_total_mw,
    p_div_coolant_pump_mw,
):
    """Extracted verbatim from `calculate_component_thermal_powers`
    (`power.py:898-902`), reused by `DeltaEtaStep` below."""
    return (
        p_plasma_separatrix_mw
        + (p_div_nuclear_heat_total_mw + p_div_rad_total_mw)
        + p_div_coolant_pump_mw
    )


def calculate_delta_eta(
    p_fw_blkt_heat_deposited_mw,
    i_shld_primary_heat,
    p_shld_heat_deposited_mw,
    p_div_heat_deposited_mw,
    i_thermal_electric_conversion,
):
    """The primary/secondary heat split and the divertor-heat-fraction efficiency
    correction it feeds -- extracted verbatim from `calculate_component_thermal_powers`
    (`power.py:1005-1014`).

    This is the block that *produces* `.power.delta_eta`; `calculate_component_thermal_powers`
    also *consumes* the entering value of the same field earlier in the same call (via
    `calculate_plant_thermal_efficiency`) -- see `power_B_thermal_cryo.md`'s "The
    `delta_eta` self-loop" section. Isolated here so `DeltaEtaStep` (the
    `FixedPointFunction` node, below) can compute the same next-iterate without
    duplicating this logic by hand.

    **Note the entering `delta_eta` value plays no part in this computation at all** --
    confirmed by inspection (no parameter here is derived from `eta_turbine`/
    `temp_turbine_coolant_in`, the only two quantities `delta_eta` can influence via
    `calculate_plant_thermal_efficiency`) and pinned by
    `test_delta_eta_step_gradient_is_exactly_zero_wrt_delta_eta` in
    `test_power_B_thermal_cryo.py`.

    Returns
    -------
    :
        `(p_plant_primary_heat_mw, p_div_secondary_heat_mw, i_div_primary_heat,
        f_p_div_primary_heat, delta_eta)`.
    """
    if i_thermal_electric_conversion == ElectricConversionModelTypes.CCFE_HCPB_VALUE:
        p_plant_primary_heat_mw = (
            p_fw_blkt_heat_deposited_mw
            + i_shld_primary_heat * p_shld_heat_deposited_mw
        )
        p_div_secondary_heat_mw = p_div_heat_deposited_mw
        i_div_primary_heat = 0
    else:
        p_plant_primary_heat_mw = (
            p_fw_blkt_heat_deposited_mw
            + i_shld_primary_heat * p_shld_heat_deposited_mw
            + p_div_heat_deposited_mw
        )
        p_div_secondary_heat_mw = 0.0e0
        i_div_primary_heat = 1

    f_p_div_primary_heat = p_div_heat_deposited_mw / p_plant_primary_heat_mw
    delta_eta = 0.339e0 * f_p_div_primary_heat

    return (
        p_plant_primary_heat_mw,
        p_div_secondary_heat_mw,
        i_div_primary_heat,
        f_p_div_primary_heat,
        delta_eta,
    )


def calculate_component_thermal_powers(
    i_p_coolant_pumping,
    p_fw_coolant_pump_mw,
    p_blkt_coolant_pump_mw,
    p_fw_blkt_coolant_pump_mw,
    eta_coolant_pump_electric,
    p_shld_coolant_pump_mw,
    p_div_coolant_pump_mw,
    p_blkt_breeder_pump_mw,
    p_hcd_electric_total_mw,
    p_hcd_injected_total_mw,
    i_blkt_dual_coolant,
    p_blkt_nuclear_heat_total_mw,
    f_nuc_pow_bz_liq,
    p_fw_nuclear_heat_total_mw,
    p_fw_rad_total_mw,
    p_beam_orbit_loss_mw,
    p_fw_alpha_mw,
    p_beam_shine_through_mw,
    p_cp_shield_nuclear_heat_mw,
    p_shld_nuclear_heat_mw,
    p_plasma_separatrix_mw,
    p_div_nuclear_heat_total_mw,
    p_div_rad_total_mw,
    p_fw_div_heat_deposited_mw,
    p_fw_hcd_nuclear_heat_mw,
    p_fw_hcd_rad_total_mw,
    i_shld_primary_heat,
    i_thermal_electric_conversion,
    i_blanket_type,
    eta_turbine,
    etath_liq,
    delta_eta,
    temp_blkt_coolant_out,
    outlet_temp_liq,
    temp_turbine_coolant_in,
    secondary_cycle_liq,
):
    """`Power.component_thermal_powers`.

    Ports `process/models/power.py:814-1036`. Composes
    `calculate_plant_thermal_efficiency`/`calculate_plant_thermal_efficiency_2`
    internally, in the same order PROCESS calls them (`plant_thermal_efficiency`
    first, `plant_thermal_efficiency_2` second) -- see those functions' docstrings
    for `.heat_transport.temp_turbine_coolant_in`'s order-dependence.

    Three static switches select formula variants with genuinely different
    reads/writes-sets (`i_p_coolant_pumping`, `i_blkt_dual_coolant`,
    `i_thermal_electric_conversion` -- the last used here as a **binary** split,
    `CCFE_HCPB_VALUE` vs. everything else, unlike `plant_thermal_efficiency`'s own
    5-way use of the same field -- see `power_B_thermal_cryo.md`).

    **`p_fw_blkt_coolant_pump_mw` is a conditional-ownership pass-through**: PROCESS
    only computes it here when `i_p_coolant_pumping` is `USER_INPUT`/
    `FRACTION_OF_HEAT`; for `MECHANICAL`/`MECHANICAL_WITH_PRESSURE_DROP` it is
    produced elsewhere (`process/models/blankets/hcpb.py`,
    `process/models/blankets/blanket_library.py` -- not yet ported, registry unit
    #13). This function takes the *entering* value as an ordinary parameter and only
    overwrites it internally for the two switch values that own it, exactly mirroring
    PROCESS's own conditional write.

    **`p_fw_div_heat_deposited_mw` is a similar pass-through**, but on a *different*
    partition of the same switch (`!= MECHANICAL_WITH_PRESSURE_DROP` here, vs. `not in
    {MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}` for `p_fw_blkt_coolant_pump_mw`
    above) -- confirmed these are genuinely different conditions in the source, not a
    copy-paste of the same guard. Its only other producer anywhere in `process/` is
    `models/ife.py` (IFE devices, out of scope entirely).

    Returns
    -------
    :
        `(p_fw_blkt_coolant_pump_mw, p_fw_blkt_coolant_pump_elec_mw,
        p_shld_coolant_pump_elec_mw, p_div_coolant_pump_elec_mw,
        p_blkt_breeder_pump_elec_mw, p_coolant_pump_total_mw,
        p_coolant_pump_elec_total_mw, p_coolant_pump_loss_total_mw,
        p_hcd_electric_loss_mw, p_blkt_liquid_breeder_heat_deposited_mw,
        p_fw_blkt_heat_deposited_mw, p_fw_heat_deposited_mw, p_blkt_heat_deposited_mw,
        p_shld_heat_deposited_mw, p_div_heat_deposited_mw,
        p_fw_div_heat_deposited_mw, eta_turbine, etath_liq, temp_turbine_coolant_in,
        p_plant_primary_heat_mw, p_div_secondary_heat_mw, i_div_primary_heat,
        f_p_div_primary_heat, delta_eta, p_shld_secondary_heat_mw,
        p_hcd_secondary_heat_mw, n_primary_heat_exchangers)`.
    """
    p_fw_blkt_coolant_pump_mw = calculate_p_fw_blkt_coolant_pump_mw(
        i_p_coolant_pumping,
        p_fw_coolant_pump_mw,
        p_blkt_coolant_pump_mw,
        p_fw_blkt_coolant_pump_mw,
    )

    p_fw_blkt_coolant_pump_elec_mw = (
        p_fw_blkt_coolant_pump_mw / eta_coolant_pump_electric
    )
    p_shld_coolant_pump_elec_mw = p_shld_coolant_pump_mw / eta_coolant_pump_electric
    p_div_coolant_pump_elec_mw = p_div_coolant_pump_mw / eta_coolant_pump_electric
    p_blkt_breeder_pump_elec_mw = p_blkt_breeder_pump_mw / eta_coolant_pump_electric

    p_coolant_pump_total_mw = (
        p_fw_blkt_coolant_pump_mw
        + p_blkt_breeder_pump_mw
        + p_shld_coolant_pump_mw
        + p_div_coolant_pump_mw
    )
    p_coolant_pump_elec_total_mw = (
        p_fw_blkt_coolant_pump_elec_mw
        + p_blkt_breeder_pump_elec_mw
        + p_shld_coolant_pump_elec_mw
        + p_div_coolant_pump_elec_mw
    )
    p_coolant_pump_loss_total_mw = (
        p_coolant_pump_elec_total_mw - p_coolant_pump_total_mw
    )

    p_hcd_electric_loss_mw = p_hcd_electric_total_mw - p_hcd_injected_total_mw

    if i_blkt_dual_coolant == 2:
        p_blkt_liquid_breeder_heat_deposited_mw = (
            p_blkt_nuclear_heat_total_mw * f_nuc_pow_bz_liq
        ) + p_blkt_breeder_pump_mw
    elif i_blkt_dual_coolant == 1:
        p_blkt_liquid_breeder_heat_deposited_mw = p_blkt_breeder_pump_mw
    else:
        p_blkt_liquid_breeder_heat_deposited_mw = 0.0e0

    p_fw_blkt_heat_deposited_mw = calculate_p_fw_blkt_heat_deposited_mw(
        i_blkt_dual_coolant,
        p_fw_nuclear_heat_total_mw,
        p_fw_rad_total_mw,
        p_blkt_nuclear_heat_total_mw,
        p_blkt_breeder_pump_mw,
        p_fw_blkt_coolant_pump_mw,
        p_beam_orbit_loss_mw,
        p_fw_alpha_mw,
        p_beam_shine_through_mw,
    )

    p_fw_heat_deposited_mw = (
        p_fw_nuclear_heat_total_mw
        + p_fw_rad_total_mw
        + p_fw_coolant_pump_mw
        + p_beam_orbit_loss_mw
        + p_fw_alpha_mw
        + p_beam_shine_through_mw
    )
    p_blkt_heat_deposited_mw = p_blkt_nuclear_heat_total_mw + p_blkt_coolant_pump_mw
    p_shld_heat_deposited_mw = calculate_p_shld_heat_deposited_mw(
        p_cp_shield_nuclear_heat_mw, p_shld_nuclear_heat_mw, p_shld_coolant_pump_mw
    )
    p_div_heat_deposited_mw = calculate_p_div_heat_deposited_mw(
        p_plasma_separatrix_mw,
        p_div_nuclear_heat_total_mw,
        p_div_rad_total_mw,
        p_div_coolant_pump_mw,
    )

    if i_p_coolant_pumping != PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP:
        p_fw_div_heat_deposited_mw = p_fw_heat_deposited_mw + p_div_heat_deposited_mw

    eta_turbine, temp_turbine_coolant_in = calculate_plant_thermal_efficiency(
        eta_turbine,
        delta_eta,
        temp_blkt_coolant_out,
        temp_turbine_coolant_in,
        i_thermal_electric_conversion,
        i_blanket_type,
    )
    etath_liq, temp_turbine_coolant_in = calculate_plant_thermal_efficiency_2(
        etath_liq,
        outlet_temp_liq,
        temp_turbine_coolant_in,
        secondary_cycle_liq,
    )

    (
        p_plant_primary_heat_mw,
        p_div_secondary_heat_mw,
        i_div_primary_heat,
        f_p_div_primary_heat,
        delta_eta,
    ) = calculate_delta_eta(
        p_fw_blkt_heat_deposited_mw,
        i_shld_primary_heat,
        p_shld_heat_deposited_mw,
        p_div_heat_deposited_mw,
        i_thermal_electric_conversion,
    )

    p_shld_secondary_heat_mw = p_shld_heat_deposited_mw * (1 - i_shld_primary_heat)
    p_hcd_secondary_heat_mw = p_fw_hcd_nuclear_heat_mw + p_fw_hcd_rad_total_mw
    n_primary_heat_exchangers = jnp.ceil(p_plant_primary_heat_mw / 1000.0e0)

    return (
        p_fw_blkt_coolant_pump_mw,
        p_fw_blkt_coolant_pump_elec_mw,
        p_shld_coolant_pump_elec_mw,
        p_div_coolant_pump_elec_mw,
        p_blkt_breeder_pump_elec_mw,
        p_coolant_pump_total_mw,
        p_coolant_pump_elec_total_mw,
        p_coolant_pump_loss_total_mw,
        p_hcd_electric_loss_mw,
        p_blkt_liquid_breeder_heat_deposited_mw,
        p_fw_blkt_heat_deposited_mw,
        p_fw_heat_deposited_mw,
        p_blkt_heat_deposited_mw,
        p_shld_heat_deposited_mw,
        p_div_heat_deposited_mw,
        p_fw_div_heat_deposited_mw,
        eta_turbine,
        etath_liq,
        temp_turbine_coolant_in,
        p_plant_primary_heat_mw,
        p_div_secondary_heat_mw,
        i_div_primary_heat,
        f_p_div_primary_heat,
        delta_eta,
        p_shld_secondary_heat_mw,
        p_hcd_secondary_heat_mw,
        n_primary_heat_exchangers,
    )


def calculate_cryo(
    i_tf_sup,
    inuclear,
    tfcryoarea,
    coldmass,
    p_tf_nuclear_heat_mw,
    ensxpfm,
    t_plant_pulse_plasma_present,
    c_tf_turn,
    n_tf_coils,
    qnuc,
):
    """`Power.cryo`.

    Ports `process/models/power.py:1773-1852` verbatim. `i_tf_sup` and `inuclear` are
    static configuration switches, ordinary Python `if`.

    **`.fwbs.qnuc` is a conditional-ownership read/write** (per
    `_audit/schema.md`'s `conditional-ownership-by-run-config` category, same shape as
    `.physics.aspect` in `stellarator_build.md`): PROCESS only computes it here when
    `inuclear == 0 and i_tf_sup == 1` ("Issue #511: if inuclear = 1: qnuc is input");
    otherwise it is an ordinary `InputVariable`, supplied from outside this function.
    `qnuc` is therefore both an input (the entering value, used whenever this
    function does not own it) and an output (the value actually used in `qmisc`/
    `helpow`, which is either the entering value or the freshly-computed one).

    Parameters
    ----------
    i_tf_sup :
        Static switch: TF coil conductor type (0 resistive / 1 superconducting /
        2 aluminium). `.tfcoil.i_tf_sup`.
    inuclear :
        Static switch: whether TF coil nuclear heating is computed (`0`) or supplied
        as an input (`1`). `.fwbs.inuclear`.
    tfcryoarea :
        Surface area of toroidal shells covering TF coils (m2). `.tfcoil.tfcryoarea`.
    coldmass :
        Mass of cold (cryogenic) components (kg). `.structure.coldmass`.
    p_tf_nuclear_heat_mw :
        Nuclear heating in TF coils (MW). `.fwbs.p_tf_nuclear_heat_mw`.
    ensxpfm :
        Maximum PF coil stored energy (MJ). `.pf_power.ensxpfm`.
    t_plant_pulse_plasma_present :
        Pulse length of cycle (s). `.times.t_plant_pulse_plasma_present`.
    c_tf_turn :
        Current per turn in TF coils (A). `.tfcoil.c_tf_turn`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    qnuc :
        Entering value of `.fwbs.qnuc` (W), used whenever this function does not own
        it (see above).

    Returns
    -------
    :
        `(helpow, qss, qac, qcl, qmisc, qnuc)` -- `helpow` is `Power.cryo`'s return
        value (`.heat_transport.helpow` once written by the caller); the rest are the
        `.power.*`/`.fwbs.qnuc` fields `cryo` writes directly.
    """
    qss = 4.3e-4 * coldmass
    if i_tf_sup == 1:
        qss = qss + 2.0e0 * tfcryoarea

    if inuclear == 0 and i_tf_sup == 1:
        qnuc = 1.0e6 * p_tf_nuclear_heat_mw

    qac = 1.0e3 * ensxpfm / t_plant_pulse_plasma_present

    if i_tf_sup == 1:
        qcl = 13.6e-3 * n_tf_coils * c_tf_turn
    else:
        qcl = 0.0e0

    qmisc = 0.45e0 * (qss + qnuc + qac + qcl)
    helpow = jnp.maximum(0.0e0, qmisc + qss + qnuc + qac + qcl)

    return helpow, qss, qac, qcl, qmisc, qnuc


def calculate_cryo_loads(
    i_tf_sup,
    i_pf_conductor,
    inuclear,
    tfcryoarea,
    coldmass,
    p_tf_nuclear_heat_mw,
    ensxpfm,
    t_plant_pulse_plasma_present,
    c_tf_turn,
    n_tf_coils,
    qnuc,
    eff_tf_cryo,
    temp_tf_cryo,
    p_cp_resistive,
    p_tf_leg_resistive,
    p_tf_joints_resistive,
    pnuc_cp_tf,
    temp_cp_coolant_inlet,
    qss,
    qac,
    qcl,
    qmisc,
):
    """`Power.calculate_cryo_loads`.

    Ports `process/models/power.py:1037-1118`. `i_tf_sup`/`i_pf_conductor` gate
    whether `calculate_cryo` runs at all (a whole-function conditional call, same
    shape as `vacuum.py`'s branch dispatch, not a per-field `jnp.where`) -- when
    neither TF nor PF conductor is superconducting, PROCESS never calls `cryo()`, so
    `.power.qss`/`qac`/`qcl`/`qmisc`/`.fwbs.qnuc` are left untouched by this method
    entirely; `qss`/`qac`/`qcl`/`qmisc` are therefore ordinary pass-through
    parameters here, exactly like `qnuc` is for `calculate_cryo` itself. `helpow` and
    `p_cryo_plant_electric_mw` *are* unconditionally (re-)initialised to `0.0` at the
    top of the PROCESS method regardless of the guard (`power.py:1049-1050`), so
    those two are not pass-throughs.

    **`.heat_transport.helpow_cryal` has no producer anywhere in `process/` other
    than this function's own `i_tf_sup == 2` branch** (confirmed by grep), and its
    dataclass default is `0.0`
    (`process/data_structure/heat_transport_variables.py:67`) -- so, unlike
    `qss`/`qac`/`qcl`/`qmisc`/`qnuc`, it is safe to compute unconditionally here
    (`0.0` when `i_tf_sup != 2`) rather than threading it through as a pass-through;
    doing so is observationally identical to PROCESS's own behaviour on every input
    this port can reach, since nothing else ever writes it.

    Parameters
    ----------
    i_tf_sup, i_pf_conductor, inuclear :
        Static switches. `.tfcoil.i_tf_sup`, `.pf_coil.i_pf_conductor`
        (`PFConductorModel`), `.fwbs.inuclear`.
    tfcryoarea, coldmass, p_tf_nuclear_heat_mw, ensxpfm, t_plant_pulse_plasma_present,
    c_tf_turn, n_tf_coils, qnuc :
        Forwarded to `calculate_cryo` -- see that function's docstring.
    eff_tf_cryo :
        Cryogenic plant Carnot efficiency fraction. `.tfcoil.eff_tf_cryo`.
    temp_tf_cryo :
        TF coil cryogenic temperature (K). `.tfcoil.temp_tf_cryo`.
    p_cp_resistive, p_tf_leg_resistive, p_tf_joints_resistive :
        Resistive powers (W), aluminium-TF-coil branch only.
        `.tfcoil.p_cp_resistive`, `.tfcoil.p_tf_leg_resistive`,
        `.tfcoil.p_tf_joints_resistive`.
    pnuc_cp_tf :
        Nuclear heating on the centrepost (MW), aluminium-TF-coil branch only.
        `.fwbs.pnuc_cp_tf`.
    temp_cp_coolant_inlet :
        Centrepost coolant inlet temperature (K), aluminium-TF-coil branch only.
        `.tfcoil.temp_cp_coolant_inlet`. Always a legitimate positive input in
        practice (default `313.15`, never written by this chunk) -- `cryo_cool_req`'s
        formula divides by it unconditionally, same as PROCESS's own source, and this
        is not a `0/0` risk the way `.tfcoil.res_tf_leg` was in chunk A (see
        `power_A_tf_coil_power.md`).
    qss, qac, qcl, qmisc :
        Entering values of `.power.qss`/`qac`/`qcl`/`qmisc`, used only when
        `calculate_cryo` does not run (see above).

    Returns
    -------
    :
        `(helpow, p_cryo_plant_electric_mw, helpow_cryal, cryo_cool_req, qss, qac,
        qcl, qmisc, qnuc)`.
    """
    if i_tf_sup == 1 or i_pf_conductor == PFConductorModel.SUPERCONDUCTING:
        helpow, qss, qac, qcl, qmisc, qnuc = calculate_cryo(
            i_tf_sup,
            inuclear,
            tfcryoarea,
            coldmass,
            p_tf_nuclear_heat_mw,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )
        p_cryo_plant_electric_mw = (
            1.0e-6
            * (constants.TEMP_ROOM - temp_tf_cryo)
            / (eff_tf_cryo * temp_tf_cryo)
            * helpow
        )
    else:
        helpow = 0.0e0
        p_cryo_plant_electric_mw = 0.0e0

    if i_tf_sup == 2:
        helpow_cryal = (
            p_cp_resistive
            + p_tf_leg_resistive
            + p_tf_joints_resistive
            + pnuc_cp_tf * 1.0e6
        )
        p_tf_cryoal_cryo = (
            1.0e-6
            * (constants.TEMP_ROOM - temp_cp_coolant_inlet)
            / (eff_tf_cryo * temp_cp_coolant_inlet)
            * helpow_cryal
        )
        p_cryo_plant_electric_mw = p_cryo_plant_electric_mw + p_tf_cryoal_cryo
    else:
        helpow_cryal = 0.0e0

    cryo_cool_req = (
        helpow * ((293 / temp_tf_cryo) - 1) / ((293 / 4.5) - 1)
        + helpow_cryal * ((293 / temp_cp_coolant_inlet) - 1) / ((293 / 4.5) - 1)
    ) / 1.0e3

    return (
        helpow,
        p_cryo_plant_electric_mw,
        helpow_cryal,
        cryo_cool_req,
        qss,
        qac,
        qcl,
        qmisc,
        qnuc,
    )


class PlantThermalEfficiency(ExplicitFunction):
    """cottax node: `calculate_plant_thermal_efficiency`."""

    i_thermal_electric_conversion: int = eqx.field(static=True)
    i_blanket_type: int = eqx.field(static=True)

    eta_turbine = Output(lambda s: s.heat_transport.eta_turbine)
    temp_turbine_coolant_in = Output(lambda s: s.heat_transport.temp_turbine_coolant_in)

    def __call__(
        self,
        eta_turbine=Input(lambda s: s.heat_transport.eta_turbine),
        delta_eta=Input(lambda s: s.power.delta_eta),
        temp_blkt_coolant_out=Input(lambda s: s.fwbs.temp_blkt_coolant_out),
        temp_turbine_coolant_in=Input(
            lambda s: s.heat_transport.temp_turbine_coolant_in
        ),
    ):
        return calculate_plant_thermal_efficiency(
            eta_turbine,
            delta_eta,
            temp_blkt_coolant_out,
            temp_turbine_coolant_in,
            self.i_thermal_electric_conversion,
            self.i_blanket_type,
        )


class PlantThermalEfficiency2(ExplicitFunction):
    """cottax node: `calculate_plant_thermal_efficiency_2`."""

    secondary_cycle_liq: int = eqx.field(static=True)

    etath_liq = Output(lambda s: s.heat_transport.etath_liq)
    temp_turbine_coolant_in = Output(lambda s: s.heat_transport.temp_turbine_coolant_in)

    def __call__(
        self,
        etath_liq=Input(lambda s: s.heat_transport.etath_liq),
        outlet_temp_liq=Input(lambda s: s.fwbs.outlet_temp_liq),
        temp_turbine_coolant_in=Input(
            lambda s: s.heat_transport.temp_turbine_coolant_in
        ),
    ):
        return calculate_plant_thermal_efficiency_2(
            etath_liq,
            outlet_temp_liq,
            temp_turbine_coolant_in,
            self.secondary_cycle_liq,
        )


class ComponentThermalPowers(ExplicitFunction):
    """cottax node: `calculate_component_thermal_powers`."""

    i_p_coolant_pumping: int = eqx.field(static=True)
    i_blkt_dual_coolant: int = eqx.field(static=True)
    i_thermal_electric_conversion: int = eqx.field(static=True)
    i_blanket_type: int = eqx.field(static=True)
    secondary_cycle_liq: int = eqx.field(static=True)

    p_fw_blkt_coolant_pump_mw = Output(lambda s: s.primary_pumping.p_fw_blkt_coolant_pump_mw)
    p_fw_blkt_coolant_pump_elec_mw = Output(lambda s: s.power.p_fw_blkt_coolant_pump_elec_mw)
    p_shld_coolant_pump_elec_mw = Output(lambda s: s.power.p_shld_coolant_pump_elec_mw)
    p_div_coolant_pump_elec_mw = Output(lambda s: s.power.p_div_coolant_pump_elec_mw)
    p_blkt_breeder_pump_elec_mw = Output(lambda s: s.power.p_blkt_breeder_pump_elec_mw)
    p_coolant_pump_total_mw = Output(lambda s: s.power.p_coolant_pump_total_mw)
    p_coolant_pump_elec_total_mw = Output(lambda s: s.heat_transport.p_coolant_pump_elec_total_mw)
    p_coolant_pump_loss_total_mw = Output(lambda s: s.heat_transport.p_coolant_pump_loss_total_mw)
    p_hcd_electric_loss_mw = Output(lambda s: s.heat_transport.p_hcd_electric_loss_mw)
    p_blkt_liquid_breeder_heat_deposited_mw = Output(
        lambda s: s.power.p_blkt_liquid_breeder_heat_deposited_mw
    )
    p_fw_blkt_heat_deposited_mw = Output(lambda s: s.power.p_fw_blkt_heat_deposited_mw)
    p_fw_heat_deposited_mw = Output(lambda s: s.power.p_fw_heat_deposited_mw)
    p_blkt_heat_deposited_mw = Output(lambda s: s.power.p_blkt_heat_deposited_mw)
    p_shld_heat_deposited_mw = Output(lambda s: s.power.p_shld_heat_deposited_mw)
    p_div_heat_deposited_mw = Output(lambda s: s.power.p_div_heat_deposited_mw)
    p_fw_div_heat_deposited_mw = Output(lambda s: s.heat_transport.p_fw_div_heat_deposited_mw)
    eta_turbine = Output(lambda s: s.heat_transport.eta_turbine)
    etath_liq = Output(lambda s: s.heat_transport.etath_liq)
    temp_turbine_coolant_in = Output(lambda s: s.heat_transport.temp_turbine_coolant_in)
    p_plant_primary_heat_mw = Output(lambda s: s.heat_transport.p_plant_primary_heat_mw)
    p_div_secondary_heat_mw = Output(lambda s: s.heat_transport.p_div_secondary_heat_mw)
    i_div_primary_heat = Output(lambda s: s.power.i_div_primary_heat)
    f_p_div_primary_heat = Output(lambda s: s.power.f_p_div_primary_heat)
    delta_eta = Output(lambda s: s.power.delta_eta)
    p_shld_secondary_heat_mw = Output(lambda s: s.heat_transport.p_shld_secondary_heat_mw)
    p_hcd_secondary_heat_mw = Output(lambda s: s.heat_transport.p_hcd_secondary_heat_mw)
    n_primary_heat_exchangers = Output(lambda s: s.heat_transport.n_primary_heat_exchangers)

    def __call__(
        self,
        p_fw_coolant_pump_mw=Input(lambda s: s.heat_transport.p_fw_coolant_pump_mw),
        p_blkt_coolant_pump_mw=Input(lambda s: s.heat_transport.p_blkt_coolant_pump_mw),
        p_fw_blkt_coolant_pump_mw=Input(
            lambda s: s.primary_pumping.p_fw_blkt_coolant_pump_mw
        ),
        eta_coolant_pump_electric=Input(lambda s: s.fwbs.eta_coolant_pump_electric),
        p_shld_coolant_pump_mw=Input(lambda s: s.heat_transport.p_shld_coolant_pump_mw),
        p_div_coolant_pump_mw=Input(lambda s: s.heat_transport.p_div_coolant_pump_mw),
        p_blkt_breeder_pump_mw=Input(lambda s: s.heat_transport.p_blkt_breeder_pump_mw),
        p_hcd_electric_total_mw=Input(lambda s: s.heat_transport.p_hcd_electric_total_mw),
        p_hcd_injected_total_mw=Input(lambda s: s.current_drive.p_hcd_injected_total_mw),
        p_blkt_nuclear_heat_total_mw=Input(lambda s: s.fwbs.p_blkt_nuclear_heat_total_mw),
        f_nuc_pow_bz_liq=Input(lambda s: s.fwbs.f_nuc_pow_bz_liq),
        p_fw_nuclear_heat_total_mw=Input(lambda s: s.fwbs.p_fw_nuclear_heat_total_mw),
        p_fw_rad_total_mw=Input(lambda s: s.fwbs.p_fw_rad_total_mw),
        p_beam_orbit_loss_mw=Input(lambda s: s.current_drive.p_beam_orbit_loss_mw),
        p_fw_alpha_mw=Input(lambda s: s.physics.p_fw_alpha_mw),
        p_beam_shine_through_mw=Input(lambda s: s.current_drive.p_beam_shine_through_mw),
        p_cp_shield_nuclear_heat_mw=Input(lambda s: s.fwbs.p_cp_shield_nuclear_heat_mw),
        p_shld_nuclear_heat_mw=Input(lambda s: s.fwbs.p_shld_nuclear_heat_mw),
        p_plasma_separatrix_mw=Input(lambda s: s.physics.p_plasma_separatrix_mw),
        p_div_nuclear_heat_total_mw=Input(lambda s: s.fwbs.p_div_nuclear_heat_total_mw),
        p_div_rad_total_mw=Input(lambda s: s.fwbs.p_div_rad_total_mw),
        p_fw_div_heat_deposited_mw=Input(
            lambda s: s.heat_transport.p_fw_div_heat_deposited_mw
        ),
        p_fw_hcd_nuclear_heat_mw=Input(lambda s: s.fwbs.p_fw_hcd_nuclear_heat_mw),
        p_fw_hcd_rad_total_mw=Input(lambda s: s.fwbs.p_fw_hcd_rad_total_mw),
        i_shld_primary_heat=Input(lambda s: s.heat_transport.i_shld_primary_heat),
        eta_turbine=Input(lambda s: s.heat_transport.eta_turbine),
        etath_liq=Input(lambda s: s.heat_transport.etath_liq),
        delta_eta=Input(lambda s: s.power.delta_eta),
        temp_blkt_coolant_out=Input(lambda s: s.fwbs.temp_blkt_coolant_out),
        outlet_temp_liq=Input(lambda s: s.fwbs.outlet_temp_liq),
        temp_turbine_coolant_in=Input(
            lambda s: s.heat_transport.temp_turbine_coolant_in
        ),
    ):
        return calculate_component_thermal_powers(
            self.i_p_coolant_pumping,
            p_fw_coolant_pump_mw,
            p_blkt_coolant_pump_mw,
            p_fw_blkt_coolant_pump_mw,
            eta_coolant_pump_electric,
            p_shld_coolant_pump_mw,
            p_div_coolant_pump_mw,
            p_blkt_breeder_pump_mw,
            p_hcd_electric_total_mw,
            p_hcd_injected_total_mw,
            self.i_blkt_dual_coolant,
            p_blkt_nuclear_heat_total_mw,
            f_nuc_pow_bz_liq,
            p_fw_nuclear_heat_total_mw,
            p_fw_rad_total_mw,
            p_beam_orbit_loss_mw,
            p_fw_alpha_mw,
            p_beam_shine_through_mw,
            p_cp_shield_nuclear_heat_mw,
            p_shld_nuclear_heat_mw,
            p_plasma_separatrix_mw,
            p_div_nuclear_heat_total_mw,
            p_div_rad_total_mw,
            p_fw_div_heat_deposited_mw,
            p_fw_hcd_nuclear_heat_mw,
            p_fw_hcd_rad_total_mw,
            i_shld_primary_heat,
            self.i_thermal_electric_conversion,
            self.i_blanket_type,
            eta_turbine,
            etath_liq,
            delta_eta,
            temp_blkt_coolant_out,
            outlet_temp_liq,
            temp_turbine_coolant_in,
            self.secondary_cycle_liq,
        )


class Cryo(ExplicitFunction):
    """cottax node: `calculate_cryo`."""

    i_tf_sup: int = eqx.field(static=True)
    inuclear: int = eqx.field(static=True)

    helpow = Output(lambda s: s.heat_transport.helpow)
    qss = Output(lambda s: s.power.qss)
    qac = Output(lambda s: s.power.qac)
    qcl = Output(lambda s: s.power.qcl)
    qmisc = Output(lambda s: s.power.qmisc)
    qnuc = Output(lambda s: s.fwbs.qnuc)

    def __call__(
        self,
        tfcryoarea=Input(lambda s: s.tfcoil.tfcryoarea),
        coldmass=Input(lambda s: s.structure.coldmass),
        p_tf_nuclear_heat_mw=Input(lambda s: s.fwbs.p_tf_nuclear_heat_mw),
        ensxpfm=Input(lambda s: s.pf_power.ensxpfm),
        t_plant_pulse_plasma_present=Input(
            lambda s: s.times.t_plant_pulse_plasma_present
        ),
        c_tf_turn=Input(lambda s: s.tfcoil.c_tf_turn),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        qnuc=Input(lambda s: s.fwbs.qnuc),
    ):
        return calculate_cryo(
            self.i_tf_sup,
            self.inuclear,
            tfcryoarea,
            coldmass,
            p_tf_nuclear_heat_mw,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )


class CryoLoads(ExplicitFunction):
    """cottax node: `calculate_cryo_loads`."""

    i_tf_sup: int = eqx.field(static=True)
    i_pf_conductor: int = eqx.field(static=True)
    inuclear: int = eqx.field(static=True)

    helpow = Output(lambda s: s.heat_transport.helpow)
    p_cryo_plant_electric_mw = Output(lambda s: s.heat_transport.p_cryo_plant_electric_mw)
    helpow_cryal = Output(lambda s: s.heat_transport.helpow_cryal)
    cryo_cool_req = Output(lambda s: s.tfcoil.cryo_cool_req)
    qss = Output(lambda s: s.power.qss)
    qac = Output(lambda s: s.power.qac)
    qcl = Output(lambda s: s.power.qcl)
    qmisc = Output(lambda s: s.power.qmisc)
    qnuc = Output(lambda s: s.fwbs.qnuc)

    def __call__(
        self,
        tfcryoarea=Input(lambda s: s.tfcoil.tfcryoarea),
        coldmass=Input(lambda s: s.structure.coldmass),
        p_tf_nuclear_heat_mw=Input(lambda s: s.fwbs.p_tf_nuclear_heat_mw),
        ensxpfm=Input(lambda s: s.pf_power.ensxpfm),
        t_plant_pulse_plasma_present=Input(
            lambda s: s.times.t_plant_pulse_plasma_present
        ),
        c_tf_turn=Input(lambda s: s.tfcoil.c_tf_turn),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
        qnuc=Input(lambda s: s.fwbs.qnuc),
        eff_tf_cryo=Input(lambda s: s.tfcoil.eff_tf_cryo),
        temp_tf_cryo=Input(lambda s: s.tfcoil.temp_tf_cryo),
        p_cp_resistive=Input(lambda s: s.tfcoil.p_cp_resistive),
        p_tf_leg_resistive=Input(lambda s: s.tfcoil.p_tf_leg_resistive),
        p_tf_joints_resistive=Input(lambda s: s.tfcoil.p_tf_joints_resistive),
        pnuc_cp_tf=Input(lambda s: s.fwbs.pnuc_cp_tf),
        temp_cp_coolant_inlet=Input(lambda s: s.tfcoil.temp_cp_coolant_inlet),
        qss=Input(lambda s: s.power.qss),
        qac=Input(lambda s: s.power.qac),
        qcl=Input(lambda s: s.power.qcl),
        qmisc=Input(lambda s: s.power.qmisc),
    ):
        return calculate_cryo_loads(
            self.i_tf_sup,
            self.i_pf_conductor,
            self.inuclear,
            tfcryoarea,
            coldmass,
            p_tf_nuclear_heat_mw,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
            eff_tf_cryo,
            temp_tf_cryo,
            p_cp_resistive,
            p_tf_leg_resistive,
            p_tf_joints_resistive,
            pnuc_cp_tf,
            temp_cp_coolant_inlet,
            qss,
            qac,
            qcl,
            qmisc,
        )
