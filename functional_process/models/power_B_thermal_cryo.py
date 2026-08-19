"""Pure-functional port of the thermal-power-balance and cryogenics sub-unit of
`process/models/power.py` (registry unit #14, chunk B).

Audit record: `functional_process/models/power_B_thermal_cryo.md`. Covers
`Power.component_thermal_powers` (814-1036), `Power.plant_thermal_efficiency`
(1935-2071), `Power.plant_thermal_efficiency_2` (2073-2116),
`Power.calculate_cryo_loads` (1037-1118) and `Power.cryo` (1773-1852) -- see the
audit record's data-footprint table for the full trace and, in particular, the six
self-loop findings ("The `delta_eta` self-loop" and "The `eta_turbine`/`etath_liq`/
`temp_turbine_coolant_in`/`p_fw_div_heat_deposited_mw`/`p_fw_blkt_coolant_pump_mw`
self-loops" sections), which is why `component_thermal_powers`'s node-level split
below exists: `calculate_component_thermal_powers` (the pure function) is unchanged,
but at the node level each of the six self-references is cut into its own tiny
`FixedPointFunction` (`DeltaEtaStep`, `EtaTurbineStep`, `EtathLiqStep`,
`TempTurbineCoolantInStep`, `PFwDivHeatDepositedMwStep`, `PFwBlktCoolantPumpMwStep`
-- per `cottax.interfaces.pytree_namespace_module`), separate from
`ComponentThermalPowers` (an ordinary `ExplicitFunction` for every other output,
reading all six current values as plain `Input`s). See each class's own docstring.

All five source methods are tier-1 once ported: no internal iteration anywhere in
this chunk (the `cryo`/`calculate_cryo_loads` pair looks like it could be tier-2 from
its name, but it is a single straight-line evaluation, not a solve).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    Input,
    Output,
)

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


def calculate_p_fw_heat_deposited_mw(
    p_fw_nuclear_heat_total_mw,
    p_fw_rad_total_mw,
    p_fw_coolant_pump_mw,
    p_beam_orbit_loss_mw,
    p_fw_alpha_mw,
    p_beam_shine_through_mw,
):
    """Extracted verbatim from `calculate_component_thermal_powers`
    (`power.py:923-930`, the inline `p_fw_heat_deposited_mw` sum), reused by
    `PFwDivHeatDepositedMwStep` below -- see `power_B_thermal_cryo.md`'s "The
    `p_fw_div_heat_deposited_mw` self-loop" section."""
    return (
        p_fw_nuclear_heat_total_mw
        + p_fw_rad_total_mw
        + p_fw_coolant_pump_mw
        + p_beam_orbit_loss_mw
        + p_fw_alpha_mw
        + p_beam_shine_through_mw
    )


def calculate_p_fw_div_heat_deposited_mw(
    i_p_coolant_pumping,
    p_fw_heat_deposited_mw,
    p_div_heat_deposited_mw,
    p_fw_div_heat_deposited_mw,
):
    """The conditional-ownership pass-through for
    `.heat_transport.p_fw_div_heat_deposited_mw`, extracted verbatim from
    `calculate_component_thermal_powers` (`power.py:955-961`) so both it and
    `PFwDivHeatDepositedMwStep` below share one body. See
    `calculate_component_thermal_powers`'s own docstring for the conditional-ownership
    finding this implements (a *different* partition of `i_p_coolant_pumping` than
    `calculate_p_fw_blkt_coolant_pump_mw`'s)."""
    if i_p_coolant_pumping != PumpingPowerModelTypes.MECHANICAL_WITH_PRESSURE_DROP:
        return p_fw_heat_deposited_mw + p_div_heat_deposited_mw
    return p_fw_div_heat_deposited_mw


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

    This is the block that *produces* `.power.delta_eta`; `component_thermal_powers`
    also *consumes* the entering value of the same field earlier in the same call
    (via `calculate_plant_thermal_efficiency`) -- see `power_B_thermal_cryo.md`'s "The
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

    p_fw_heat_deposited_mw = calculate_p_fw_heat_deposited_mw(
        p_fw_nuclear_heat_total_mw,
        p_fw_rad_total_mw,
        p_fw_coolant_pump_mw,
        p_beam_orbit_loss_mw,
        p_fw_alpha_mw,
        p_beam_shine_through_mw,
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

    p_fw_div_heat_deposited_mw = calculate_p_fw_div_heat_deposited_mw(
        i_p_coolant_pumping,
        p_fw_heat_deposited_mw,
        p_div_heat_deposited_mw,
        p_fw_div_heat_deposited_mw,
    )

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
    qnuc = calculate_cryo_qnuc(i_tf_sup, inuclear, p_tf_nuclear_heat_mw, qnuc)
    qss, qac, qcl, qmisc = calculate_cryo_q_loads(
        i_tf_sup,
        tfcryoarea,
        coldmass,
        ensxpfm,
        t_plant_pulse_plasma_present,
        c_tf_turn,
        n_tf_coils,
        qnuc,
    )
    helpow = calculate_helpow(qss, qnuc, qac, qcl, qmisc)

    return helpow, qss, qac, qcl, qmisc, qnuc


def calculate_cryo_qnuc(i_tf_sup, inuclear, p_tf_nuclear_heat_mw, qnuc):
    """`Power.cryo`'s nuclear-heating line, `process/models/power.py:1823-1825`.

    Split out of `calculate_cryo` because `.fwbs.qnuc` is the one field of the five
    `cryo` writes whose ownership is gated by a **different** condition
    (`inuclear == 0 and i_tf_sup == 1`) from the other four
    (`calculate_cryo_loads`'s outer superconducting guard). Keeping the two
    conditions in separate functions -- and therefore in separate
    `FixedPointFunction` nodes, `CryoQNucStep` and `CryoQLoadsStep` -- is what makes
    each resulting `FixedPoint` problem *uniformly* degenerate or non-degenerate.
    A single node owning all five would be an identity on the `qnuc` row and a
    constant on the other four whenever `inuclear == 1`, which is a rank-deficient
    SAND equality block `sand.degenerate_fixed_points` could not drop (it drops only
    problems whose residual vanishes entirely).

    `i_tf_sup == 1` implies `calculate_cryo_loads`'s outer guard, so this condition
    alone is exactly PROCESS's: `cryo()` runs, and inside it the `inuclear` test
    passes.

    Parameters
    ----------
    i_tf_sup, inuclear :
        Static switches -- see `calculate_cryo`.
    p_tf_nuclear_heat_mw :
        Nuclear heating in TF coils (MW). `.fwbs.p_tf_nuclear_heat_mw`.
    qnuc :
        Entering value of `.fwbs.qnuc` (W), returned unchanged when this function
        does not own it ("Issue #511: if inuclear = 1: qnuc is input").

    Returns
    -------
    :
        `qnuc` (W).
    """
    if inuclear == 0 and i_tf_sup == 1:
        qnuc = 1.0e6 * p_tf_nuclear_heat_mw
    return qnuc


def calculate_cryo_q_loads(
    i_tf_sup,
    tfcryoarea,
    coldmass,
    ensxpfm,
    t_plant_pulse_plasma_present,
    c_tf_turn,
    n_tf_coils,
    qnuc,
):
    """`Power.cryo`'s four unconditional load terms, `power.py:1820-1841`.

    `qss` (conduction/radiation), `qac` (AC losses), `qcl` (current leads) and
    `qmisc` (the 45% miscellaneous allowance) -- everything `cryo()` writes except
    `.fwbs.qnuc`, which `calculate_cryo_qnuc` above owns. `qmisc` is a function of
    the other three **and** of the post-`calculate_cryo_qnuc` `qnuc`, which is why
    `qnuc` is a parameter here and why `CryoQLoadsStep` reads `.fwbs.qnuc` (the field
    `CryoQNucStep` owns) rather than recomputing it.

    Unlike `calculate_cryo_qnuc` this carries no guard of its own: within `cryo()`
    all four are written unconditionally. Their conditional ownership is one level
    up -- `calculate_cryo_loads` only calls `cryo()` at all when
    `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` -- and that guard lives in
    `CryoQLoadsStep.step`, not here, so this function stays a plain formula.

    Parameters
    ----------
    i_tf_sup :
        Static switch -- see `calculate_cryo`.
    tfcryoarea, coldmass, ensxpfm, t_plant_pulse_plasma_present, c_tf_turn,
    n_tf_coils :
        See `calculate_cryo`.
    qnuc :
        `.fwbs.qnuc` (W) **after** `calculate_cryo_qnuc`, i.e. the value `qmisc` is
        actually built from in PROCESS's own statement order.

    Returns
    -------
    :
        `(qss, qac, qcl, qmisc)`, all in W.
    """
    qss = 4.3e-4 * coldmass
    if i_tf_sup == 1:
        qss = qss + 2.0e0 * tfcryoarea

    qac = 1.0e3 * ensxpfm / t_plant_pulse_plasma_present

    if i_tf_sup == 1:
        qcl = 13.6e-3 * n_tf_coils * c_tf_turn
    else:
        qcl = 0.0e0

    qmisc = 0.45e0 * (qss + qnuc + qac + qcl)

    return qss, qac, qcl, qmisc


def calculate_helpow(qss, qnuc, qac, qcl, qmisc):
    """`Power.cryo`'s return value, `power.py:1843-1852` -- total helium heat removal
    at cryogenic temperature (W), clipped at zero.

    Its own function because it has two call sites: `calculate_cryo` (the un-split
    port of `Power.cryo`, kept for the Tier-1 contract against PROCESS) and
    `calculate_cryo_plant_loads` (the node-level split, which gets the five `q*`
    terms from the graph rather than recomputing them).

    The `jnp.maximum(0, ...)` is PROCESS's own `max(0.0e0, ...)`, reproduced. It is a
    clip on a **sum**, not on the argument of a square root, so it carries none of
    the `jnp.sqrt(jnp.maximum(0, x))` derivative hazard recorded elsewhere in this
    port -- the subgradient at the kink is JAX's usual choice and the value is
    correct on both sides.

    Returns
    -------
    :
        `helpow` (W).
    """
    return jnp.maximum(0.0e0, qmisc + qss + qnuc + qac + qcl)


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

    **This is the un-split port**, kept as the function the Tier-1 contract compares
    against PROCESS. Its body is now assembled from the same four pieces the
    registered nodes use -- `calculate_cryo_qnuc` (`CryoQNucStep`),
    `calculate_cryo_q_loads` (`CryoQLoadsStep`), `cryo_is_active` and
    `calculate_cryo_plant_loads` (`CryoLoads`) -- so the node-level split cannot
    drift from the function this contract validates. One rearrangement is worth
    stating: `calculate_cryo_qnuc` is called *outside* the superconducting guard,
    where PROCESS's `qnuc` write sits inside it. That is an identity, not a change --
    the inner condition is `inuclear == 0 and i_tf_sup == 1`, and `i_tf_sup == 1`
    already implies the outer guard.

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
    qnuc = calculate_cryo_qnuc(i_tf_sup, inuclear, p_tf_nuclear_heat_mw, qnuc)
    if cryo_is_active(i_tf_sup, i_pf_conductor):
        qss, qac, qcl, qmisc = calculate_cryo_q_loads(
            i_tf_sup,
            tfcryoarea,
            coldmass,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )
    helpow, p_cryo_plant_electric_mw, helpow_cryal, cryo_cool_req = (
        calculate_cryo_plant_loads(
            i_tf_sup,
            i_pf_conductor,
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
            qnuc,
        )
    )

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


def cryo_is_active(i_tf_sup, i_pf_conductor):
    """PROCESS's guard on whether `Power.cryo` is called at all
    (`process/models/power.py:1054-1057`).

    Named rather than repeated because three places need exactly this predicate:
    `calculate_cryo_loads` (the un-split port, kept for the Tier-1 contract),
    `calculate_cryo_plant_loads`, and `CryoQLoadsStep.step`. Both arguments are
    static configuration switches, so this is an ordinary Python `bool`, never a
    traced value.
    """
    return i_tf_sup == 1 or i_pf_conductor == PFConductorModel.SUPERCONDUCTING


def calculate_cryo_plant_loads(
    i_tf_sup,
    i_pf_conductor,
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
    qnuc,
):
    """The four fields `Power.calculate_cryo_loads` owns **unconditionally**.

    `.heat_transport.helpow`, `.heat_transport.p_cryo_plant_electric_mw`,
    `.heat_transport.helpow_cryal` and `.tfcoil.cryo_cool_req` are all written on
    every path through `power.py:1049-1118` (the first two are re-initialised to
    `0.0` at `:1049-1050` before the guard, the third is safe to compute
    unconditionally for the reason `calculate_cryo_loads`'s docstring gives, and the
    fourth is outside every branch) -- so unlike the five `q*` fields they are not
    conditionally owned and need no fixed point. This function is the part of
    `calculate_cryo_loads` that `CryoLoads` (an ordinary `ExplicitFunction`) owns,
    with the five `q*` terms arriving as plain inputs from `CryoQNucStep`/
    `CryoQLoadsStep` instead of being recomputed here.

    `helpow` is `calculate_helpow` of the five `q*` terms -- the same expression
    `Power.cryo` returns -- under the guard, and exactly `0.0` outside it.

    Parameters
    ----------
    i_tf_sup, i_pf_conductor :
        Static switches -- see `calculate_cryo_loads`.
    eff_tf_cryo, temp_tf_cryo, p_cp_resistive, p_tf_leg_resistive,
    p_tf_joints_resistive, pnuc_cp_tf, temp_cp_coolant_inlet :
        See `calculate_cryo_loads`.
    qss, qac, qcl, qmisc, qnuc :
        `.power.qss`/`qac`/`qcl`/`qmisc` and `.fwbs.qnuc` (W), as they stand after
        `Power.cryo` would have run -- i.e. the values `CryoQLoadsStep` and
        `CryoQNucStep` own.

    Returns
    -------
    :
        `(helpow, p_cryo_plant_electric_mw, helpow_cryal, cryo_cool_req)`.
    """
    if cryo_is_active(i_tf_sup, i_pf_conductor):
        helpow = calculate_helpow(qss, qnuc, qac, qcl, qmisc)
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

    return helpow, p_cryo_plant_electric_mw, helpow_cryal, cryo_cool_req


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
    """cottax node: `calculate_component_thermal_powers`'s outputs **other than**
    the six self-referencing fields split into their own `FixedPointFunction`s below
    (`DeltaEtaStep`, `EtaTurbineStep`, `EtathLiqStep`, `TempTurbineCoolantInStep`,
    `PFwDivHeatDepositedMwStep`, `PFwBlktCoolantPumpMwStep`).

    `calculate_component_thermal_powers` itself reads the entering value of each of
    those six `VarPath`s and also produces a freshly-computed value on the same
    `VarPath` later in the same call -- six genuine single-node self-loops (`Output`'s
    `where` and one `Input`'s `where` naming the identical `VarPath`), which `cottax`
    refuses to build as one node (`ValueError: reads [...], which it also owns`,
    confirmed directly against this class before these splits existed). Each
    `FixedPointFunction` below isolates one self-reference; this class keeps every
    *other* output of `calculate_component_thermal_powers` and takes all six fields as
    ordinary plain-`Input`s (the current/entering values), same as every other
    parameter here. Ownership of the six real `VarPath`s belongs to the six
    `FixedPoint` problem nodes, not to this class -- see `power_B_thermal_cryo.md`.

    Registering `ComponentThermalPowers` together with the six `FixedPointFunction`s
    in one graph is **not** attempted here (out of scope, deferred to
    `total_process.py`'s later consolidation pass). `to_graph(ComponentThermalPowers(...))`
    no longer raises on any field (confirmed directly, see the individual
    `FixedPointFunction` docstrings below for each split's own `to_graph` check).
    """

    i_p_coolant_pumping: int = eqx.field(static=True)
    i_blkt_dual_coolant: int = eqx.field(static=True)
    i_thermal_electric_conversion: int = eqx.field(static=True)
    i_blanket_type: int = eqx.field(static=True)
    secondary_cycle_liq: int = eqx.field(static=True)

    # .primary_pumping.p_fw_blkt_coolant_pump_mw is NOT declared here -- owned by
    # PFwBlktCoolantPumpMwStep's FixedPoint problem node (see below). Still read
    # below, as a plain Input.
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
    # .heat_transport.p_fw_div_heat_deposited_mw is NOT declared here -- owned by
    # PFwDivHeatDepositedMwStep's FixedPoint problem node (see below). Still read
    # below, as a plain Input.
    # .heat_transport.eta_turbine is NOT declared here -- owned by EtaTurbineStep's
    # FixedPoint problem node (see below). Still read below, as a plain Input.
    # .heat_transport.etath_liq is NOT declared here -- owned by EtathLiqStep's
    # FixedPoint problem node (see below). Still read below, as a plain Input.
    # .heat_transport.temp_turbine_coolant_in is NOT declared here -- owned by
    # TempTurbineCoolantInStep's FixedPoint problem node (see below). Still read
    # below, as a plain Input.
    p_plant_primary_heat_mw = Output(lambda s: s.heat_transport.p_plant_primary_heat_mw)
    p_div_secondary_heat_mw = Output(lambda s: s.heat_transport.p_div_secondary_heat_mw)
    i_div_primary_heat = Output(lambda s: s.power.i_div_primary_heat)
    f_p_div_primary_heat = Output(lambda s: s.power.f_p_div_primary_heat)
    # .power.delta_eta is NOT declared here -- DeltaEtaStep's FixedPoint problem node
    # owns it (see the class docstring above and "The delta_eta self-loop" in
    # power_B_thermal_cryo.md). delta_eta is still read below, as a plain Input.
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
        result = calculate_component_thermal_powers(
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
        # `calculate_component_thermal_powers` is unchanged -- it still returns all
        # 27 elements (see its own docstring's Returns section). Six are dropped
        # here, not reimplemented, since a FixedPointFunction's FixedPoint problem
        # node owns each of the six real VarPaths, not this node:
        # p_fw_blkt_coolant_pump_mw (index 0, PFwBlktCoolantPumpMwStep),
        # p_fw_div_heat_deposited_mw (15, PFwDivHeatDepositedMwStep),
        # eta_turbine (16, EtaTurbineStep), etath_liq (17, EtathLiqStep),
        # temp_turbine_coolant_in (18, TempTurbineCoolantInStep) and
        # delta_eta (23, DeltaEtaStep). Named unpacking, not index slicing, since the
        # six dropped indices are not contiguous.
        (
            _p_fw_blkt_coolant_pump_mw,
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
            _p_fw_div_heat_deposited_mw,
            _eta_turbine,
            _etath_liq,
            _temp_turbine_coolant_in,
            p_plant_primary_heat_mw,
            p_div_secondary_heat_mw,
            i_div_primary_heat,
            f_p_div_primary_heat,
            _delta_eta,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            n_primary_heat_exchangers,
        ) = result
        del (
            _p_fw_blkt_coolant_pump_mw,
            _p_fw_div_heat_deposited_mw,
            _eta_turbine,
            _etath_liq,
            _temp_turbine_coolant_in,
            _delta_eta,
        )
        return (
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
            p_plant_primary_heat_mw,
            p_div_secondary_heat_mw,
            i_div_primary_heat,
            f_p_div_primary_heat,
            p_shld_secondary_heat_mw,
            p_hcd_secondary_heat_mw,
            n_primary_heat_exchangers,
        )


class DeltaEtaStep(FixedPointFunction):
    """The `.power.delta_eta` self-loop, cut.

    `calculate_component_thermal_powers` reads the entering `.power.delta_eta` (it
    feeds `calculate_plant_thermal_efficiency`'s `CCFE_HCPB_VALUE_WITH_DIVERTOR`/
    `STEAM_RANKINE_CYCLE` branches) and produces a freshly-computed
    `.power.delta_eta` later in the same call -- see `power_B_thermal_cryo.md`'s "The
    `delta_eta` self-loop" section, the same shape `test_harness.md` documents for
    `power_at_ignition_point`/`st_phys`. `cottax` refuses to build a single node that
    both reads and owns one `VarPath`
    (`cottax.interfaces.pytree_namespace_module.NodalDeclaration`'s docstring: *"a
    node may not read what it owns"*) -- confirmed directly:
    `to_graph(ComponentThermalPowers(...))` raises `ValueError: reads
    ['.power.delta_eta', ...], which it also owns` before this class existed.
    `FixedPointFunction` is the structural admission this shape needs: `step` reads
    the real `.power.delta_eta` and writes a minted `^cond.power.delta_eta` copy (an
    ordinary node, no self-reference); the `FixedPoint` problem this class also
    declares (via `node_definitions`) reads that minted copy and owns the real
    `.power.delta_eta`. Neither piece reads and owns the same `VarPath`.

    **Deliberately self-contained, not wired to `ComponentThermalPowers`'s own
    outputs.** `step` recomputes `p_fw_blkt_coolant_pump_mw`/
    `p_fw_blkt_heat_deposited_mw`/`p_shld_heat_deposited_mw`/`p_div_heat_deposited_mw`
    from the same raw, externally-owned inputs `calculate_component_thermal_powers`
    itself reads (via the shared `calculate_p_fw_blkt_coolant_pump_mw`/
    `calculate_p_fw_blkt_heat_deposited_mw`/`calculate_p_shld_heat_deposited_mw`/
    `calculate_p_div_heat_deposited_mw` helpers -- extraction, not reimplementation),
    rather than reading `ComponentThermalPowers`'s `Output`s for the same
    quantities. Reading those `Output`s instead would recreate a cycle at the
    two-node level (`ComponentThermalPowers` -> `DeltaEtaStep` -> `ComponentThermalPowers`,
    via `.power.delta_eta`) -- exactly the risk `_audit/next_steps.md` § 5 flags:
    *"splitting `component_thermal_powers` would very plausibly just turn one
    self-referencing node into two mutually-referencing ones representing the same
    local loop, not reveal a new one."* Reading the same raw external inputs twice
    (fan-out, not a cycle) keeps this node's own graph acyclic and independently
    buildable; only the deliberate, minted `delta_eta` cut is a loop at all.

    **`delta_eta` plays no part in `step`'s own computation** -- confirmed by
    inspection (see `calculate_delta_eta`'s docstring) and pinned by
    `test_delta_eta_step_gradient_is_exactly_zero_wrt_delta_eta` in
    `test_power_B_thermal_cryo.py`: `d(delta_eta_next)/d(delta_eta) == 0` exactly, for
    every switch combination this chunk supports, not just the two branches of
    `calculate_plant_thermal_efficiency` that read it. It is declared as a `step`
    parameter anyway (per this shape's own convention: `step` reads the value it is
    about to replace) purely for structural symmetry with the self-loop it is cutting,
    not because the computation needs it. Were a driver ever assigned to the
    resulting `FixedPoint` problem node (not done here -- see `_audit/next_steps.md`
    § 5, "What stays deferred"), this zero derivative means it converges in exactly
    one iteration regardless of algorithm, from any starting value.
    """

    i_p_coolant_pumping: int = eqx.field(static=True)
    i_blkt_dual_coolant: int = eqx.field(static=True)
    i_thermal_electric_conversion: int = eqx.field(static=True)

    delta_eta = Output(lambda s: s.power.delta_eta)

    def step(
        self,
        p_fw_coolant_pump_mw=Input(lambda s: s.heat_transport.p_fw_coolant_pump_mw),
        p_blkt_coolant_pump_mw=Input(lambda s: s.heat_transport.p_blkt_coolant_pump_mw),
        p_fw_blkt_coolant_pump_mw=Input(
            lambda s: s.primary_pumping.p_fw_blkt_coolant_pump_mw
        ),
        p_fw_nuclear_heat_total_mw=Input(lambda s: s.fwbs.p_fw_nuclear_heat_total_mw),
        p_fw_rad_total_mw=Input(lambda s: s.fwbs.p_fw_rad_total_mw),
        p_blkt_nuclear_heat_total_mw=Input(
            lambda s: s.fwbs.p_blkt_nuclear_heat_total_mw
        ),
        p_blkt_breeder_pump_mw=Input(lambda s: s.heat_transport.p_blkt_breeder_pump_mw),
        p_beam_orbit_loss_mw=Input(lambda s: s.current_drive.p_beam_orbit_loss_mw),
        p_fw_alpha_mw=Input(lambda s: s.physics.p_fw_alpha_mw),
        p_beam_shine_through_mw=Input(lambda s: s.current_drive.p_beam_shine_through_mw),
        p_cp_shield_nuclear_heat_mw=Input(lambda s: s.fwbs.p_cp_shield_nuclear_heat_mw),
        p_shld_nuclear_heat_mw=Input(lambda s: s.fwbs.p_shld_nuclear_heat_mw),
        p_shld_coolant_pump_mw=Input(lambda s: s.heat_transport.p_shld_coolant_pump_mw),
        p_plasma_separatrix_mw=Input(lambda s: s.physics.p_plasma_separatrix_mw),
        p_div_nuclear_heat_total_mw=Input(lambda s: s.fwbs.p_div_nuclear_heat_total_mw),
        p_div_rad_total_mw=Input(lambda s: s.fwbs.p_div_rad_total_mw),
        p_div_coolant_pump_mw=Input(lambda s: s.heat_transport.p_div_coolant_pump_mw),
        i_shld_primary_heat=Input(lambda s: s.heat_transport.i_shld_primary_heat),
        delta_eta=Input(lambda s: s.power.delta_eta),
    ):
        del delta_eta  # see class docstring -- verified numerically inert here

        p_fw_blkt_coolant_pump_mw = calculate_p_fw_blkt_coolant_pump_mw(
            self.i_p_coolant_pumping,
            p_fw_coolant_pump_mw,
            p_blkt_coolant_pump_mw,
            p_fw_blkt_coolant_pump_mw,
        )
        p_fw_blkt_heat_deposited_mw = calculate_p_fw_blkt_heat_deposited_mw(
            self.i_blkt_dual_coolant,
            p_fw_nuclear_heat_total_mw,
            p_fw_rad_total_mw,
            p_blkt_nuclear_heat_total_mw,
            p_blkt_breeder_pump_mw,
            p_fw_blkt_coolant_pump_mw,
            p_beam_orbit_loss_mw,
            p_fw_alpha_mw,
            p_beam_shine_through_mw,
        )
        p_shld_heat_deposited_mw = calculate_p_shld_heat_deposited_mw(
            p_cp_shield_nuclear_heat_mw, p_shld_nuclear_heat_mw, p_shld_coolant_pump_mw
        )
        p_div_heat_deposited_mw = calculate_p_div_heat_deposited_mw(
            p_plasma_separatrix_mw,
            p_div_nuclear_heat_total_mw,
            p_div_rad_total_mw,
            p_div_coolant_pump_mw,
        )
        _, _, _, _, delta_eta_next = calculate_delta_eta(
            p_fw_blkt_heat_deposited_mw,
            i_shld_primary_heat,
            p_shld_heat_deposited_mw,
            p_div_heat_deposited_mw,
            self.i_thermal_electric_conversion,
        )
        return delta_eta_next


class EtaTurbineStep(FixedPointFunction):
    """The `.heat_transport.eta_turbine` self-loop, cut.

    `calculate_component_thermal_powers` reads the entering `.heat_transport.eta_turbine`
    (it is `calculate_plant_thermal_efficiency`'s first parameter) and produces a
    freshly-computed `.heat_transport.eta_turbine` in the same call -- the same
    read-before-(re)write shape `delta_eta` had, see `DeltaEtaStep` above and
    `power_B_thermal_cryo.md`'s "The `eta_turbine`/`etath_liq`/`temp_turbine_coolant_in`
    self-loops" section.

    **`step` needs no placeholder for `temp_turbine_coolant_in`.** Inspecting every
    branch of `calculate_plant_thermal_efficiency` confirms `eta_turbine`'s value never
    depends on the *entering* `temp_turbine_coolant_in`: the two branches that use a
    turbine-inlet temperature at all (`STEAM_RANKINE_CYCLE`,
    `SUPERCRITICAL_CO2_BRAYTON_CYCLE`) overwrite it locally from
    `temp_blkt_coolant_out` before using it; every other branch never reads it at all
    to compute `eta_turbine`. So `step` calls the shared, unmodified
    `calculate_plant_thermal_efficiency` with an unused `0.0` placeholder for
    `temp_turbine_coolant_in` and keeps only its first return element -- reuse, not
    reimplementation of the branch logic (`TempTurbineCoolantInStep` below is the
    node responsible for `temp_turbine_coolant_in` itself, with its own `eta_turbine`
    placeholder for the same reason, symmetrically).

    **Not numerically degenerate in one uniform direction** -- unlike `delta_eta`
    (identically zero gradient everywhere), `eta_turbine`'s dependence on its own
    entering value is branch-dependent: identity (gradient exactly `1`) on the
    pass-through sub-branches (`USER_INPUT`; `CCFE_HCPB_VALUE`/
    `CCFE_HCPB_VALUE_WITH_DIVERTOR` with `i_blanket_type != CCFE_HCPB`; the
    `i_thermal_electric_conversion` default arm), and exactly zero on the
    sub-branches that overwrite `eta_turbine` from other inputs (`CCFE_HCPB_VALUE`/
    `CCFE_HCPB_VALUE_WITH_DIVERTOR` with `i_blanket_type == CCFE_HCPB`,
    `STEAM_RANKINE_CYCLE` with a matching blanket, `SUPERCRITICAL_CO2_BRAYTON_CYCLE`).
    Confirmed by `jax.grad` per switch combination in
    `test_eta_turbine_step_gradient_wrt_eta_turbine` -- see that test for which
    combinations land on which value.
    """

    i_thermal_electric_conversion: int = eqx.field(static=True)
    i_blanket_type: int = eqx.field(static=True)

    eta_turbine = Output(lambda s: s.heat_transport.eta_turbine)

    def step(
        self,
        eta_turbine=Input(lambda s: s.heat_transport.eta_turbine),
        delta_eta=Input(lambda s: s.power.delta_eta),
        temp_blkt_coolant_out=Input(lambda s: s.fwbs.temp_blkt_coolant_out),
    ):
        eta_turbine_next, _ = calculate_plant_thermal_efficiency(
            eta_turbine,
            delta_eta,
            temp_blkt_coolant_out,
            0.0,  # temp_turbine_coolant_in placeholder -- see class docstring
            self.i_thermal_electric_conversion,
            self.i_blanket_type,
        )
        return eta_turbine_next


class EtathLiqStep(FixedPointFunction):
    """The `.heat_transport.etath_liq` self-loop, cut.

    `calculate_component_thermal_powers` reads the entering `.heat_transport.etath_liq`
    (it is `calculate_plant_thermal_efficiency_2`'s first parameter) and produces a
    freshly-computed `.heat_transport.etath_liq` in the same call -- same shape as
    `eta_turbine`/`delta_eta` above.

    **`step` needs no placeholder for `temp_turbine_coolant_in`** for the same reason
    `EtaTurbineStep` doesn't: `calculate_plant_thermal_efficiency_2`'s
    `secondary_cycle_liq == 4` branch overwrites its local `temp_turbine_coolant_in`
    from `outlet_temp_liq` before using it, and the `== 2` branch never reads it to
    compute `etath_liq` at all. `step` reuses the shared, unmodified
    `calculate_plant_thermal_efficiency_2` with an unused `0.0` placeholder and keeps
    only its first return element.

    **Not numerically degenerate in one uniform direction**, same shape as
    `eta_turbine`: identity (gradient `1`) for `secondary_cycle_liq == 2` (plain
    pass-through), exactly zero for `== 4` (`etath_liq` recomputed from
    `outlet_temp_liq` alone). Confirmed by `jax.grad` in
    `test_etath_liq_step_gradient_wrt_etath_liq`.
    """

    secondary_cycle_liq: int = eqx.field(static=True)

    etath_liq = Output(lambda s: s.heat_transport.etath_liq)

    def step(
        self,
        etath_liq=Input(lambda s: s.heat_transport.etath_liq),
        outlet_temp_liq=Input(lambda s: s.fwbs.outlet_temp_liq),
    ):
        etath_liq_next, _ = calculate_plant_thermal_efficiency_2(
            etath_liq,
            outlet_temp_liq,
            0.0,  # temp_turbine_coolant_in placeholder -- see class docstring
            self.secondary_cycle_liq,
        )
        return etath_liq_next


class TempTurbineCoolantInStep(FixedPointFunction):
    """The `.heat_transport.temp_turbine_coolant_in` self-loop, cut.

    `calculate_component_thermal_powers` threads `temp_turbine_coolant_in` through
    *both* `calculate_plant_thermal_efficiency` and `calculate_plant_thermal_efficiency_2`,
    in that order (see `power_B_thermal_cryo.md`'s "Write-ordering on
    `temp_turbine_coolant_in`") -- the entering value feeds the first call, whose
    output value feeds the second call, whose output is the value actually written
    back. A genuine read-before-(re)write self-loop on the same `VarPath`, same shape
    as the other five in this file.

    **`step` needs placeholders for `eta_turbine`/`etath_liq`/`delta_eta`.** Neither
    `calculate_plant_thermal_efficiency`'s nor `calculate_plant_thermal_efficiency_2`'s
    `temp_turbine_coolant_in` output depends on `eta_turbine`/`etath_liq` (those
    parameters are only ever used to compute the *other* return element) or on
    `delta_eta` (only used to compute `eta_turbine`) -- confirmed by inspection of
    every branch of both functions. `step` therefore calls both shared, unmodified
    functions in the same order `calculate_component_thermal_powers` does, with `0.0`
    placeholders for the three unused parameters, and keeps only the
    `temp_turbine_coolant_in` element of each.

    **Not numerically degenerate in one uniform direction.** The total derivative
    w.r.t. the entering `temp_turbine_coolant_in` is exactly `1` only when *both*
    stages pass it through unchanged (`i_thermal_electric_conversion` selects a
    branch that doesn't touch it, i.e. `CCFE_HCPB_VALUE`/`CCFE_HCPB_VALUE_WITH_DIVERTOR`/
    `USER_INPUT`/the default arm, **and** `secondary_cycle_liq == 2`); it is exactly
    `0` for every other combination (either stage overwrites it from
    `temp_blkt_coolant_out`/`outlet_temp_liq`, severing the dependency on the entering
    value regardless of what the other stage does). Confirmed by `jax.grad` in
    `test_temp_turbine_coolant_in_step_gradient_wrt_temp_turbine_coolant_in`.
    """

    i_thermal_electric_conversion: int = eqx.field(static=True)
    i_blanket_type: int = eqx.field(static=True)
    secondary_cycle_liq: int = eqx.field(static=True)

    temp_turbine_coolant_in = Output(lambda s: s.heat_transport.temp_turbine_coolant_in)

    def step(
        self,
        temp_turbine_coolant_in=Input(
            lambda s: s.heat_transport.temp_turbine_coolant_in
        ),
        temp_blkt_coolant_out=Input(lambda s: s.fwbs.temp_blkt_coolant_out),
        outlet_temp_liq=Input(lambda s: s.fwbs.outlet_temp_liq),
    ):
        _, temp_turbine_coolant_in_mid = calculate_plant_thermal_efficiency(
            0.0,  # eta_turbine placeholder -- see class docstring
            0.0,  # delta_eta placeholder -- see class docstring
            temp_blkt_coolant_out,
            temp_turbine_coolant_in,
            self.i_thermal_electric_conversion,
            self.i_blanket_type,
        )
        _, temp_turbine_coolant_in_next = calculate_plant_thermal_efficiency_2(
            0.0,  # etath_liq placeholder -- see class docstring
            outlet_temp_liq,
            temp_turbine_coolant_in_mid,
            self.secondary_cycle_liq,
        )
        return temp_turbine_coolant_in_next


class PFwDivHeatDepositedMwStep(FixedPointFunction):
    """The `.heat_transport.p_fw_div_heat_deposited_mw` self-loop, cut.

    `calculate_component_thermal_powers` takes `p_fw_div_heat_deposited_mw` as the
    entering value of a **conditional-ownership pass-through** (see that function's
    own docstring and `power_B_thermal_cryo.md`): owned here (recomputed from
    `p_fw_heat_deposited_mw + p_div_heat_deposited_mw`) whenever `i_p_coolant_pumping
    != MECHANICAL_WITH_PRESSURE_DROP`; passed through unchanged (its only other
    producer anywhere in `process/` is `models/ife.py`, out of scope) when it *is*
    `MECHANICAL_WITH_PRESSURE_DROP`. Same read-before-(re)write shape as the other
    five self-loops in this file.

    `step` rebuilds `p_fw_heat_deposited_mw`/`p_div_heat_deposited_mw` from the same
    raw, externally-owned inputs `calculate_component_thermal_powers` itself reads
    (via the shared `calculate_p_fw_heat_deposited_mw`/`calculate_p_div_heat_deposited_mw`
    helpers), then calls the shared `calculate_p_fw_div_heat_deposited_mw` helper --
    the same "read the same raw inputs twice, not another node's `Output`s" pattern
    `DeltaEtaStep` established, to avoid recreating a two-node cycle with
    `ComponentThermalPowers`.

    **Piecewise degenerate, not uniform**: gradient w.r.t. the entering
    `p_fw_div_heat_deposited_mw` is exactly `1` (identity) for
    `i_p_coolant_pumping == MECHANICAL_WITH_PRESSURE_DROP`, exactly `0` otherwise
    (recomputed from raw inputs alone). Confirmed by `jax.grad` in
    `test_p_fw_div_heat_deposited_mw_step_gradient`.
    """

    i_p_coolant_pumping: int = eqx.field(static=True)

    p_fw_div_heat_deposited_mw = Output(
        lambda s: s.heat_transport.p_fw_div_heat_deposited_mw
    )

    def step(
        self,
        p_fw_div_heat_deposited_mw=Input(
            lambda s: s.heat_transport.p_fw_div_heat_deposited_mw
        ),
        p_fw_nuclear_heat_total_mw=Input(lambda s: s.fwbs.p_fw_nuclear_heat_total_mw),
        p_fw_rad_total_mw=Input(lambda s: s.fwbs.p_fw_rad_total_mw),
        p_fw_coolant_pump_mw=Input(lambda s: s.heat_transport.p_fw_coolant_pump_mw),
        p_beam_orbit_loss_mw=Input(lambda s: s.current_drive.p_beam_orbit_loss_mw),
        p_fw_alpha_mw=Input(lambda s: s.physics.p_fw_alpha_mw),
        p_beam_shine_through_mw=Input(lambda s: s.current_drive.p_beam_shine_through_mw),
        p_plasma_separatrix_mw=Input(lambda s: s.physics.p_plasma_separatrix_mw),
        p_div_nuclear_heat_total_mw=Input(lambda s: s.fwbs.p_div_nuclear_heat_total_mw),
        p_div_rad_total_mw=Input(lambda s: s.fwbs.p_div_rad_total_mw),
        p_div_coolant_pump_mw=Input(lambda s: s.heat_transport.p_div_coolant_pump_mw),
    ):
        p_fw_heat_deposited_mw = calculate_p_fw_heat_deposited_mw(
            p_fw_nuclear_heat_total_mw,
            p_fw_rad_total_mw,
            p_fw_coolant_pump_mw,
            p_beam_orbit_loss_mw,
            p_fw_alpha_mw,
            p_beam_shine_through_mw,
        )
        p_div_heat_deposited_mw = calculate_p_div_heat_deposited_mw(
            p_plasma_separatrix_mw,
            p_div_nuclear_heat_total_mw,
            p_div_rad_total_mw,
            p_div_coolant_pump_mw,
        )
        p_fw_div_heat_deposited_mw_next = calculate_p_fw_div_heat_deposited_mw(
            self.i_p_coolant_pumping,
            p_fw_heat_deposited_mw,
            p_div_heat_deposited_mw,
            p_fw_div_heat_deposited_mw,
        )
        return p_fw_div_heat_deposited_mw_next


class PFwBlktCoolantPumpMwStep(FixedPointFunction):
    """The `.primary_pumping.p_fw_blkt_coolant_pump_mw` self-loop, cut.

    `calculate_component_thermal_powers` takes `p_fw_blkt_coolant_pump_mw` as the
    entering value of a **conditional-ownership pass-through** (see that function's
    own docstring and `power_B_thermal_cryo.md`): owned here (recomputed from
    `p_fw_coolant_pump_mw + p_blkt_coolant_pump_mw`) whenever `i_p_coolant_pumping`
    is `USER_INPUT`/`FRACTION_OF_HEAT`; passed through unchanged (produced elsewhere,
    `process/models/blankets/hcpb.py`/`blanket_library.py`, unit #13, not yet ported)
    for `MECHANICAL`/`MECHANICAL_WITH_PRESSURE_DROP`. Same read-before-(re)write shape
    as the other five self-loops in this file -- and the simplest of the six, since
    the shared helper `calculate_p_fw_blkt_coolant_pump_mw` (already extracted for
    `DeltaEtaStep`'s use) is a direct, self-contained function of exactly this node's
    `step` inputs, needing no further decomposition.

    **Piecewise degenerate, not uniform**: gradient w.r.t. the entering
    `p_fw_blkt_coolant_pump_mw` is exactly `1` (identity) for `i_p_coolant_pumping in
    {MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`, exactly `0` otherwise (recomputed
    from `p_fw_coolant_pump_mw + p_blkt_coolant_pump_mw` alone). Confirmed by
    `jax.grad` in `test_p_fw_blkt_coolant_pump_mw_step_gradient`.
    """

    i_p_coolant_pumping: int = eqx.field(static=True)

    p_fw_blkt_coolant_pump_mw = Output(
        lambda s: s.primary_pumping.p_fw_blkt_coolant_pump_mw
    )

    def step(
        self,
        p_fw_blkt_coolant_pump_mw=Input(
            lambda s: s.primary_pumping.p_fw_blkt_coolant_pump_mw
        ),
        p_fw_coolant_pump_mw=Input(lambda s: s.heat_transport.p_fw_coolant_pump_mw),
        p_blkt_coolant_pump_mw=Input(lambda s: s.heat_transport.p_blkt_coolant_pump_mw),
    ):
        p_fw_blkt_coolant_pump_mw_next = calculate_p_fw_blkt_coolant_pump_mw(
            self.i_p_coolant_pumping,
            p_fw_coolant_pump_mw,
            p_blkt_coolant_pump_mw,
            p_fw_blkt_coolant_pump_mw,
        )
        return p_fw_blkt_coolant_pump_mw_next


class Cryo(ExplicitFunction):
    """cottax node: `calculate_cryo`. **Not registered** -- superseded by the split.

    Same position as `PlantThermalEfficiency`/`PlantThermalEfficiency2` in this file:
    the raw, un-split node for a PROCESS function that both reads and owns fields, so
    `to_graph(Cryo(...))` raises `ValueError: reads ['.fwbs.qnuc'], which it also
    owns` and no graph can contain it. It is kept because `calculate_cryo` is still
    the function the Tier-1 contract compares against `Power.cryo` directly, and the
    class documents the node that function *would* be.

    What is registered instead is the three-way split of
    `Power.calculate_cryo_loads` (which is the only caller of `Power.cryo`):
    `CryoQNucStep` owns `.fwbs.qnuc`, `CryoQLoadsStep` owns
    `.power.qss`/`qac`/`qcl`/`qmisc`, and `CryoLoads` owns the four unconditionally
    written `.heat_transport`/`.tfcoil` outputs. See `CryoQLoadsStep`'s docstring for
    why the five `q*` fields are two nodes rather than one.
    """

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


class CryoQNucStep(FixedPointFunction):
    """The `.fwbs.qnuc` self-loop, cut.

    `Power.cryo` writes `.fwbs.qnuc` only when `inuclear == 0 and i_tf_sup == 1` and
    otherwise leaves the entering value in place -- PROCESS's own comment at
    `process/models/power.py:1825` says so outright: *"Issue #511: if inuclear = 1:
    qnuc is input"*. The ported `calculate_cryo_qnuc` is faithful to that, taking the
    incumbent as an argument, which makes the field a read **and** a write of one
    node and `to_graph` refuses it (`ValueError: reads ['.fwbs.qnuc'], which it also
    owns`). This is `_audit/next_steps.md` §5's Shape B, and it gets §5's answer: the
    node owns the field and reads the minted `^cond` copy, so the "keep the
    incumbent" arm is a **fixed point** (`u = g(u)`), not a self-loop.

    The shape carries the switch correctly in both directions, exactly as
    `plasma_profiles.py`'s `IonVolAvgTemperature` does for
    `.physics.temp_plasma_ion_vol_avg_kev`:

    - `inuclear == 0 and i_tf_sup == 1` -- `g` is `1e6 * p_tf_nuclear_heat_mw`, with
      no dependence on the unknown at all, so the residual `g(u) - u` has derivative
      `-1`: well-posed, one Picard step from anywhere, and SAND-solvable.
    - otherwise -- `g` is the exact identity, the residual is structurally zero, and
      `functional_process.sand.degenerate_fixed_points` detects that by
      differentiation and drops the problem, reverting `.fwbs.qnuc` to an ordinary
      boundary input. Which *is* PROCESS's "qnuc is input" semantics, recovered from
      structure rather than from a comment.

    **Why this is its own node and not merged into `CryoQLoadsStep`.** The other four
    `q*` fields are gated by a different condition (`calculate_cryo_loads`'s outer
    superconducting guard). One node owning all five would, at `inuclear == 1` with
    `i_tf_sup == 1`, be an identity on the `qnuc` row and a constant on the other
    four: a residual block with one structurally-zero row that
    `degenerate_fixed_points` could **not** drop, because it drops only problems whose
    residual vanishes entirely. Splitting keeps each problem uniformly degenerate or
    uniformly well-posed.

    Registered unconditionally rather than behind a `Switch`: `.fwbs.inuclear` and
    `.tfcoil.i_tf_sup` select which arm of one function body runs, not which of two
    node definitions exists, and the "keep the incumbent" arm is representable here --
    that is the whole point of the fixed-point shape.
    """

    i_tf_sup: int = eqx.field(static=True)
    inuclear: int = eqx.field(static=True)

    qnuc = Output(lambda s: s.fwbs.qnuc)

    def step(
        self,
        qnuc=Input(lambda s: s.fwbs.qnuc),
        p_tf_nuclear_heat_mw=Input(lambda s: s.fwbs.p_tf_nuclear_heat_mw),
    ):
        return calculate_cryo_qnuc(
            self.i_tf_sup, self.inuclear, p_tf_nuclear_heat_mw, qnuc
        )


class CryoQLoadsStep(FixedPointFunction):
    """The `.power.qss`/`qac`/`qcl`/`qmisc` self-loop, cut -- one node, four unknowns.

    `Power.calculate_cryo_loads` calls `Power.cryo` only when
    `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` (`power.py:1054-1057`);
    outside that guard these four fields are left exactly as they entered, so the
    ported `calculate_cryo_loads` threads them through as pass-through parameters and
    the corresponding node both reads and owns them. Shape B again, same treatment as
    the six self-loops above and as `CryoQNucStep`.

    All four share **one** guard and are written together by one PROCESS statement
    block, so they are one `FixedPointFunction` with four `Output`s rather than four
    nodes: `g` is either constant in all four unknowns (guard true) or the identity in
    all four (guard false), never a mixture. That homogeneity is what
    `sand.degenerate_fixed_points` needs -- see `CryoQNucStep`'s docstring for why
    `.fwbs.qnuc` is *not* folded in here despite being written by the same PROCESS
    function.

    **`qnuc` is read, not recomputed.** `qmisc = 0.45 * (qss + qnuc + qac + qcl)` uses
    the value `Power.cryo` has just written, so this node reads `.fwbs.qnuc` -- the
    real `VarPath` `CryoQNucStep`'s paired `FixedPoint` owns. That is an ordinary
    edge, not a cycle: `CryoQNucStep` reads only `.fwbs.p_tf_nuclear_heat_mw` and its
    own minted copy, neither of which this node owns. (The "rebuild from raw inputs
    rather than read another node's `Output`" rule `DeltaEtaStep` follows exists to
    avoid recreating a two-node cycle; there is no cycle to avoid here.)

    **Under the reference configuration the fixed point is well-posed, not
    degenerate**: `i_tf_sup = 1` makes the guard true, so `g` does not depend on any
    of the four unknowns and the residual Jacobian is exactly `-I`.
    """

    i_tf_sup: int = eqx.field(static=True)
    i_pf_conductor: int = eqx.field(static=True)

    qss = Output(lambda s: s.power.qss)
    qac = Output(lambda s: s.power.qac)
    qcl = Output(lambda s: s.power.qcl)
    qmisc = Output(lambda s: s.power.qmisc)

    def step(
        self,
        qss=Input(lambda s: s.power.qss),
        qac=Input(lambda s: s.power.qac),
        qcl=Input(lambda s: s.power.qcl),
        qmisc=Input(lambda s: s.power.qmisc),
        qnuc=Input(lambda s: s.fwbs.qnuc),
        tfcryoarea=Input(lambda s: s.tfcoil.tfcryoarea),
        coldmass=Input(lambda s: s.structure.coldmass),
        ensxpfm=Input(lambda s: s.pf_power.ensxpfm),
        t_plant_pulse_plasma_present=Input(
            lambda s: s.times.t_plant_pulse_plasma_present
        ),
        c_tf_turn=Input(lambda s: s.tfcoil.c_tf_turn),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
    ):
        if not cryo_is_active(self.i_tf_sup, self.i_pf_conductor):
            return qss, qac, qcl, qmisc
        return calculate_cryo_q_loads(
            self.i_tf_sup,
            tfcryoarea,
            coldmass,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )


class CryoLoads(ExplicitFunction):
    """cottax node: `calculate_cryo_plant_loads` -- the unconditionally-owned part of
    `Power.calculate_cryo_loads`.

    Owns the four fields that method writes on every path through it:
    `.heat_transport.helpow`, `.heat_transport.p_cryo_plant_electric_mw`,
    `.heat_transport.helpow_cryal` and `.tfcoil.cryo_cool_req`. The five conditionally
    owned `q*` fields moved to `CryoQNucStep`/`CryoQLoadsStep` and are read here as
    plain `Input`s -- the same division `ComponentThermalPowers` has with the six
    `*Step` nodes above it, and for the same reason: two nodes owning one `VarPath`
    is an ownership collision `Graph` would reject, and would defeat the split.

    `.fwbs.inuclear` is no longer a static field of this node: `qnuc` arrives already
    resolved from `CryoQNucStep`, which is the only place that switch is read.

    **What registering it closes.** `.heat_transport.helpow` (read by `Bldgs` and
    `CryogenicSystemCost`) and `.heat_transport.p_cryo_plant_electric_mw` (read by
    `Acpow`, `PlantElectricProductionReactor` and `AuxiliaryComponentCoolingCost`)
    were boundary inputs -- five registered readers consuming seeded values for a
    quantity PROCESS computes on this run's path. See
    `_audit/boundary_inputs_audit.md` §4c (b9)/(b10) and §7 item 7.
    """

    i_tf_sup: int = eqx.field(static=True)
    i_pf_conductor: int = eqx.field(static=True)

    helpow = Output(lambda s: s.heat_transport.helpow)
    p_cryo_plant_electric_mw = Output(
        lambda s: s.heat_transport.p_cryo_plant_electric_mw
    )
    helpow_cryal = Output(lambda s: s.heat_transport.helpow_cryal)
    cryo_cool_req = Output(lambda s: s.tfcoil.cryo_cool_req)

    def __call__(
        self,
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
        qnuc=Input(lambda s: s.fwbs.qnuc),
    ):
        return calculate_cryo_plant_loads(
            self.i_tf_sup,
            self.i_pf_conductor,
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
            qnuc,
        )
