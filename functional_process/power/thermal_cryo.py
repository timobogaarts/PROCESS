"""Pure-functional port of the thermal-power-balance and cryogenics sub-unit of
`process/models/power.py` (registry unit #14, chunk B).

Audit record: `functional_process/_audit/units/models/power/thermal_cryo.md`. Covers
`Power.component_thermal_powers` (814-1036), `Power.plant_thermal_efficiency`
(1935-2071), `Power.plant_thermal_efficiency_2` (2073-2116),
`Power.calculate_cryo_loads` (1037-1118) and `Power.cryo` (1773-1852) -- see the
audit record's data-footprint table for the full trace and, in particular, the six
self-loop findings ("The `delta_eta` self-loop" and "The `eta_turbine`/`etath_liq`/
`temp_turbine_coolant_in`/`p_fw_div_heat_deposited_mw`/`p_fw_blkt_coolant_pump_mw`
self-loops" sections), which is why `component_thermal_powers`'s node-level split in
`functional_process.models.power.thermal_cryo` exists: `calculate_component_thermal_powers`
(the pure function) is unchanged, but at the node level each of the six self-references
is owned by its own node, separate from `ComponentThermalPowers`.

All five source methods are tier-1 once ported: no internal iteration anywhere in
this chunk (the `cryo`/`calculate_cryo_loads` pair looks like it could be tier-2 from
its name, but it is a single straight-line evaluation, not a solve).
"""

import jax.numpy as jnp

from functional_process.vocabulary import (
    BlktModelTypes,
    ElectricConversionModelTypes,
    PFConductorModel,
    ProcessValueError,
    PumpingPowerModelTypes,
    constants,
)


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
    `_audit/naming_convention.md`), not `FromExactly`s -- ordinary Python `if`/`elif`, not
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
    reference, because PROCESS crashes there. See `thermal_cryo.md`.

    Parameters
    ----------
    eta_turbine :
        Thermal-to-electric conversion efficiency, in/out.
        `.heat_transport.eta_turbine`.
    delta_eta :
        Loss in efficiency from low-grade divertor heat collection.
        `.power.delta_eta`. Read here as the value **already in `data`** -- see
        `thermal_cryo.md`'s self-loop finding; this function does not compute
        it (`calculate_component_thermal_powers` does, later in the same PROCESS
        call, from *this* call's `eta_turbine`/`f_p_div_primary_heat` outputs).
    temp_blkt_coolant_out :
        Blanket coolant outlet temperature (K). `.fwbs.temp_blkt_coolant_out`.
    temp_turbine_coolant_in :
        Turbine coolant inlet temperature (K), in/out.
        `.heat_transport.temp_turbine_coolant_in`. Two of the five branches write it;
        the other three pass it through unchanged (see `thermal_cryo.md`'s
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
            eta_turbine = eta_turbine_ccfe_hcpb_value()
        return eta_turbine, temp_turbine_coolant_in

    if (
        i_thermal_electric_conversion
        == ElectricConversionModelTypes.CCFE_HCPB_VALUE_WITH_DIVERTOR
    ):
        if i_blanket_type == BlktModelTypes.CCFE_HCPB:
            eta_turbine = eta_turbine_ccfe_hcpb_value_with_divertor(delta_eta)
        return eta_turbine, temp_turbine_coolant_in

    if i_thermal_electric_conversion == ElectricConversionModelTypes.USER_INPUT:
        return eta_turbine, temp_turbine_coolant_in

    if i_thermal_electric_conversion == ElectricConversionModelTypes.STEAM_RANKINE_CYCLE:
        if i_blanket_type == BlktModelTypes.CCFE_HCPB:
            temp_turbine_coolant_in = temp_turbine_coolant_in_from_blanket_coolant(
                temp_blkt_coolant_out
            )
            eta_turbine = eta_turbine_steam_rankine_cycle(
                temp_blkt_coolant_out, delta_eta
            )
        return eta_turbine, temp_turbine_coolant_in

    if (
        i_thermal_electric_conversion
        == ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
    ):
        temp_turbine_coolant_in = temp_turbine_coolant_in_from_blanket_coolant(
            temp_blkt_coolant_out
        )
        eta_turbine = eta_turbine_supercritical_co2(temp_blkt_coolant_out)
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
        final value (see `thermal_cryo.md`).
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
        return (
            etath_liq_supercritical_co2(outlet_temp_liq),
            temp_turbine_coolant_in_from_liquid_breeder(outlet_temp_liq),
        )

    raise ProcessValueError(
        f"secondary_cycle_liq ={secondary_cycle_liq} is an invalid option."
    )


def eta_turbine_ccfe_hcpb_value():
    """`i_thermal_electric_conversion == CCFE_HCPB_VALUE` (0) with
    `i_blanket_type == CCFE_HCPB`: the CCFE HCPB reference efficiency, a literal
    (`power.py:1988`).

    **Reads nothing at all.** That is the point of splitting
    `calculate_plant_thermal_efficiency`'s five-way branch into occupants: this arm's
    node has no ports on its input side, where one node carrying the switch declared
    `.heat_transport.eta_turbine`, `.power.delta_eta` and `.fwbs.temp_blkt_coolant_out`
    regardless of which arm was live (`_audit/next_steps.md` §14.2).
    """
    return 0.411e0


def eta_turbine_ccfe_hcpb_value_with_divertor(delta_eta):
    """`i_thermal_electric_conversion == CCFE_HCPB_VALUE_WITH_DIVERTOR` (1) with
    `i_blanket_type == CCFE_HCPB`: the same literal, less the divertor-heat efficiency
    penalty (`power.py:1994`). Reads `.power.delta_eta` and nothing else.
    """
    return 0.411e0 - delta_eta


def eta_turbine_steam_rankine_cycle(temp_blkt_coolant_out, delta_eta):
    """`i_thermal_electric_conversion == STEAM_RANKINE_CYCLE` (3) with
    `i_blanket_type == CCFE_HCPB`: a log fit in the turbine inlet temperature
    (`power.py:2030-2035`). Reads `.fwbs.temp_blkt_coolant_out` and `.power.delta_eta`.
    """
    return (
        0.1802e0
        * jnp.log(temp_turbine_coolant_in_from_blanket_coolant(temp_blkt_coolant_out))
        - 0.7823e0
        - delta_eta
    )


def eta_turbine_supercritical_co2(temp_blkt_coolant_out):
    """`i_thermal_electric_conversion == SUPERCRITICAL_CO2_BRAYTON_CYCLE` (4): a
    different log fit, and no `i_blanket_type` sub-branch (`power.py:2043-2046`).
    Reads `.fwbs.temp_blkt_coolant_out` and nothing else.
    """
    return (
        0.4347e0
        * jnp.log(temp_turbine_coolant_in_from_blanket_coolant(temp_blkt_coolant_out))
        - 2.5043e0
    )


def temp_turbine_coolant_in_from_blanket_coolant(temp_blkt_coolant_out):
    """The turbine inlet temperature the two computing arms of
    `calculate_plant_thermal_efficiency` set: twenty kelvin below the blanket coolant
    outlet (`power.py:2031`/`:2044`).
    """
    return temp_blkt_coolant_out - 20.0e0


def temp_turbine_coolant_in_from_liquid_breeder(outlet_temp_liq):
    """The turbine inlet temperature `calculate_plant_thermal_efficiency_2`'s
    `secondary_cycle_liq == 4` arm sets (`power.py:2107`), **overwriting** whatever
    stage one wrote -- which is why this arm's occupant reads
    `.fwbs.outlet_temp_liq` and not `.fwbs.temp_blkt_coolant_out`.
    """
    return outlet_temp_liq - 20.0e0


def etath_liq_supercritical_co2(outlet_temp_liq):
    """`secondary_cycle_liq == 4`: the liquid breeder's own supercritical-CO2
    efficiency (`power.py:2108`). Reads `.fwbs.outlet_temp_liq` and nothing else.
    """
    return (
        0.4347e0 * jnp.log(temp_turbine_coolant_in_from_liquid_breeder(outlet_temp_liq))
        - 2.5043e0
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
    below.
    """
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
    (`power.py:895-897`), reused by `DeltaEtaStep` below.
    """
    return p_cp_shield_nuclear_heat_mw + p_shld_nuclear_heat_mw + p_shld_coolant_pump_mw


def calculate_p_div_heat_deposited_mw(
    p_plasma_separatrix_mw,
    p_div_nuclear_heat_total_mw,
    p_div_rad_total_mw,
    p_div_coolant_pump_mw,
):
    """Extracted verbatim from `calculate_component_thermal_powers`
    (`power.py:898-902`), reused by `DeltaEtaStep` below.
    """
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
    `PFwDivHeatDepositedMwStep` below -- see `thermal_cryo.md`'s "The
    `p_fw_div_heat_deposited_mw` self-loop" section.
    """
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
    `calculate_p_fw_blkt_coolant_pump_mw`'s).
    """
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
    (via `calculate_plant_thermal_efficiency`) -- see `thermal_cryo.md`'s "The
    `delta_eta` self-loop" section. Isolated here so `DeltaEtaStep` (the
    `FixedPointFunction` node, below) can compute the same next-iterate without
    duplicating this logic by hand.

    **Note the entering `delta_eta` value plays no part in this computation at all** --
    confirmed by inspection (no parameter here is derived from `eta_turbine`/
    `temp_turbine_coolant_in`, the only two quantities `delta_eta` can influence via
    `calculate_plant_thermal_efficiency`) and pinned by
    `test_delta_eta_step_gradient_is_exactly_zero_wrt_delta_eta` in
    `test_thermal_cryo.py`.

    Returns
    -------
    :
        `(p_plant_primary_heat_mw, p_div_secondary_heat_mw, i_div_primary_heat,
        f_p_div_primary_heat, delta_eta)`.
    """
    if i_thermal_electric_conversion == ElectricConversionModelTypes.CCFE_HCPB_VALUE:
        p_plant_primary_heat_mw = (
            p_fw_blkt_heat_deposited_mw + i_shld_primary_heat * p_shld_heat_deposited_mw
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
    5-way use of the same field -- see `thermal_cryo.md`).

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

    (
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
    ) = calculate_component_thermal_powers_owned(
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
        p_fw_hcd_nuclear_heat_mw,
        p_fw_hcd_rad_total_mw,
        i_shld_primary_heat,
        i_thermal_electric_conversion,
    )

    # The six values this port's other occupant families own. Computed here and nowhere
    # else in the node graph: `ComponentThermalPowers` recomputed and then discarded all
    # six (see `calculate_component_thermal_powers_owned`), which is why they left it.
    p_fw_blkt_coolant_pump_mw = calculate_p_fw_blkt_coolant_pump_mw(
        i_p_coolant_pumping,
        p_fw_coolant_pump_mw,
        p_blkt_coolant_pump_mw,
        p_fw_blkt_coolant_pump_mw,
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
    delta_eta = 0.339e0 * f_p_div_primary_heat

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


def calculate_component_thermal_powers_owned(
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
    p_fw_hcd_nuclear_heat_mw,
    p_fw_hcd_rad_total_mw,
    i_shld_primary_heat,
    i_thermal_electric_conversion,
):
    """The twenty-one outputs `ComponentThermalPowers` actually owns.

    **Six of `calculate_component_thermal_powers`'s twenty-seven return values are not
    computed here at all**, and that is the point. They are
    `.primary_pumping.p_fw_blkt_coolant_pump_mw`,
    `.heat_transport.p_fw_div_heat_deposited_mw`, `.eta_turbine`, `.etath_liq`,
    `.temp_turbine_coolant_in` and `.power.delta_eta` -- each owned by its own occupant
    family in this file, and each **discarded** by `ComponentThermalPowers` after being
    recomputed. The node read all six, recomputed them, threw the results away, and so
    declared itself a consumer of four fixed points it does not consume
    (`_audit/switch_kwarg_survey.md` §6, the residue of reads dead at *every* value of
    the node's own switches).

    Dropping the recomputation drops **seven reads** -- `.heat_transport.eta_turbine`,
    `.etath_liq`, `.temp_turbine_coolant_in`, `.p_fw_div_heat_deposited_mw`,
    `.power.delta_eta`, `.fwbs.temp_blkt_coolant_out` and `.fwbs.outlet_temp_liq` -- and
    **two static switches**, `i_blanket_type` and `secondary_cycle_liq`, which fed
    nothing else (`_audit/next_steps.md` §14.2).
    `.primary_pumping.p_fw_blkt_coolant_pump_mw` stays a read: the pump totals genuinely
    use it. No computed number moves -- every one of the twenty-one is the same
    expression it was, and the composite above still returns all twenty-seven.

    Parameters and returns are `calculate_component_thermal_powers`'s, less those.
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
    p_coolant_pump_loss_total_mw = p_coolant_pump_elec_total_mw - p_coolant_pump_total_mw

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
    if i_tf_sup == 1:
        return calculate_cryo_q_loads_superconducting_tf(
            tfcryoarea,
            coldmass,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )
    return calculate_cryo_q_loads_resistive_tf(
        coldmass, ensxpfm, t_plant_pulse_plasma_present, qnuc
    )


def calculate_cryo_q_loads_superconducting_tf(
    tfcryoarea,
    coldmass,
    ensxpfm,
    t_plant_pulse_plasma_present,
    c_tf_turn,
    n_tf_coils,
    qnuc,
):
    """`calculate_cryo_q_loads` at `i_tf_sup == SUPERCONDUCTING` (1) -- the reference
    run's.

    The conduction/radiation term carries the TF cryogenic surface and the current
    leads carry the TF turn current, so this arm reads `.tfcoil.tfcryoarea`,
    `.tfcoil.c_tf_turn` and `.tfcoil.n_tf_coils`; its sibling reads none of the three
    (`_audit/next_steps.md` §14.2).
    """
    qss = 4.3e-4 * coldmass + 2.0e0 * tfcryoarea
    qac = 1.0e3 * ensxpfm / t_plant_pulse_plasma_present
    qcl = 13.6e-3 * n_tf_coils * c_tf_turn
    return qss, qac, qcl, 0.45e0 * (qss + qnuc + qac + qcl)


def calculate_cryo_q_loads_resistive_tf(
    coldmass, ensxpfm, t_plant_pulse_plasma_present, qnuc
):
    """`calculate_cryo_q_loads` at `i_tf_sup != SUPERCONDUCTING`, reached only when the
    PF coils are superconducting (otherwise `Power.cryo` is not called at all).

    `qss` is the cold mass alone and `qcl` is exactly zero, so **three reads leave with
    this arm**: `.tfcoil.tfcryoarea`, `.tfcoil.c_tf_turn`, `.tfcoil.n_tf_coils`.
    """
    qss = 4.3e-4 * coldmass
    qac = 1.0e3 * ensxpfm / t_plant_pulse_plasma_present
    qcl = 0.0e0
    return qss, qac, qcl, 0.45e0 * (qss + qnuc + qac + qcl)


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
        `tf_coil_power.md`).
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
    if i_tf_sup == 2:
        # The aluminium arm, kept whole here and **not** given an occupant:
        # `('i_tf_sup', 2)` is `UNPORTED` at the `power.tf_power` slot, so no machine
        # this port assembles reaches it. It stays in the composite because the
        # composite is what the Tier-1 contract diffs against PROCESS.
        helpow = (
            calculate_helpow(qss, qnuc, qac, qcl, qmisc)
            if cryo_is_active(i_tf_sup, i_pf_conductor)
            else 0.0e0
        )
        p_cryo_plant_electric_mw = (
            (
                1.0e-6
                * (constants.TEMP_ROOM - temp_tf_cryo)
                / (eff_tf_cryo * temp_tf_cryo)
                * helpow
            )
            if cryo_is_active(i_tf_sup, i_pf_conductor)
            else 0.0e0
        )
        helpow_cryal = (
            p_cp_resistive
            + p_tf_leg_resistive
            + p_tf_joints_resistive
            + pnuc_cp_tf * 1.0e6
        )
        p_cryo_plant_electric_mw = p_cryo_plant_electric_mw + (
            1.0e-6
            * (constants.TEMP_ROOM - temp_cp_coolant_inlet)
            / (eff_tf_cryo * temp_cp_coolant_inlet)
            * helpow_cryal
        )
        cryo_cool_req = (
            helpow * ((293 / temp_tf_cryo) - 1) / ((293 / 4.5) - 1)
            + helpow_cryal * ((293 / temp_cp_coolant_inlet) - 1) / ((293 / 4.5) - 1)
        ) / 1.0e3
        return helpow, p_cryo_plant_electric_mw, helpow_cryal, cryo_cool_req
    if cryo_is_active(i_tf_sup, i_pf_conductor):
        return calculate_cryo_plant_loads_active(
            eff_tf_cryo,
            temp_tf_cryo,
            temp_cp_coolant_inlet,
            qss,
            qac,
            qcl,
            qmisc,
            qnuc,
        )
    return calculate_cryo_plant_loads_inactive(temp_tf_cryo, temp_cp_coolant_inlet)


def calculate_cryo_plant_loads_active(
    eff_tf_cryo,
    temp_tf_cryo,
    temp_cp_coolant_inlet,
    qss,
    qac,
    qcl,
    qmisc,
    qnuc,
):
    """`calculate_cryo_plant_loads` where PROCESS calls `Power.cryo` at all --
    `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` (`power.py:1054-1057`) -- and
    the TF coil is not aluminium.

    **Four reads leave with the aluminium arm's removal**: `.tfcoil.p_cp_resistive`,
    `.tfcoil.p_tf_leg_resistive`, `.tfcoil.p_tf_joints_resistive` and
    `.fwbs.pnuc_cp_tf` are read only at `i_tf_sup == 2`, which this port refuses at the
    `power.tf_power` slot. `helpow_cryal` is therefore identically zero here, and its
    contribution to `cryo_cool_req` is written as such rather than as a live term.
    """
    helpow = calculate_helpow(qss, qnuc, qac, qcl, qmisc)
    p_cryo_plant_electric_mw = (
        1.0e-6
        * (constants.TEMP_ROOM - temp_tf_cryo)
        / (eff_tf_cryo * temp_tf_cryo)
        * helpow
    )
    return (
        helpow,
        p_cryo_plant_electric_mw,
        *_cryo_cool_req_no_aluminium(helpow, temp_tf_cryo, temp_cp_coolant_inlet),
    )


def calculate_cryo_plant_loads_inactive(temp_tf_cryo, temp_cp_coolant_inlet):
    """`calculate_cryo_plant_loads` where PROCESS does **not** call `Power.cryo`:
    `i_tf_sup != 1` with resistive PF coils.

    `helpow` and `p_cryo_plant_electric_mw` are literal zeros (`power.py:1049-1050`
    re-initialises them before the guard and nothing writes them again), so this arm
    reads **none** of `.tfcoil.eff_tf_cryo` or the five `q*` fields -- five edges into
    the cryogenics that no such machine makes.
    """
    return (
        0.0e0,
        0.0e0,
        *_cryo_cool_req_no_aluminium(0.0e0, temp_tf_cryo, temp_cp_coolant_inlet),
    )


def _cryo_cool_req_no_aluminium(helpow, temp_tf_cryo, temp_cp_coolant_inlet):
    """`(helpow_cryal, cryo_cool_req)` on every arm where the TF coil is not aluminium
    -- i.e. every arm this port can assemble.

    `helpow_cryal` is `0.0` there (`power.py:1108`), so `cryo_cool_req`'s second term
    is written as the literal zero it is rather than as a product of four reads.
    """
    helpow_cryal = 0.0e0
    cryo_cool_req = (
        helpow * ((293 / temp_tf_cryo) - 1) / ((293 / 4.5) - 1)
        + helpow_cryal * ((293 / temp_cp_coolant_inlet) - 1) / ((293 / 4.5) - 1)
    ) / 1.0e3
    return helpow_cryal, cryo_cool_req


def calculate_p_fw_div_heat_deposited_mw_summed(
    p_fw_nuclear_heat_total_mw,
    p_fw_rad_total_mw,
    p_fw_coolant_pump_mw,
    p_beam_orbit_loss_mw,
    p_fw_alpha_mw,
    p_beam_shine_through_mw,
    p_plasma_separatrix_mw,
    p_div_nuclear_heat_total_mw,
    p_div_rad_total_mw,
    p_div_coolant_pump_mw,
):
    """The first wall's and the divertor's deposited heat, summed --
    `PFwDivHeatDepositedMwSummed`'s own composition of `calculate_p_fw_heat_deposited_mw`
    and `calculate_p_div_heat_deposited_mw`, moved out of the declaration and into a
    named function (`_audit/formulas_split.md` step 1).
    """
    return calculate_p_fw_heat_deposited_mw(
        p_fw_nuclear_heat_total_mw,
        p_fw_rad_total_mw,
        p_fw_coolant_pump_mw,
        p_beam_orbit_loss_mw,
        p_fw_alpha_mw,
        p_beam_shine_through_mw,
    ) + calculate_p_div_heat_deposited_mw(
        p_plasma_separatrix_mw,
        p_div_nuclear_heat_total_mw,
        p_div_rad_total_mw,
        p_div_coolant_pump_mw,
    )


def calculate_p_fw_blkt_coolant_pump_mw_summed(
    p_fw_coolant_pump_mw, p_blkt_coolant_pump_mw
):
    """`PFwBlktCoolantPumpMw`'s own sum of the first wall's and blanket's pump powers
    (the `USER_INPUT`/`FRACTION_OF_HEAT` arm, where `power` owns the field outright --
    not `calculate_p_fw_blkt_coolant_pump_mw`'s conditional-ownership pass-through,
    which is a different function for a different arm), moved out of the declaration
    and into a named function (`_audit/formulas_split.md` step 1).
    """
    return p_fw_coolant_pump_mw + p_blkt_coolant_pump_mw


def calculate_cryo_qnuc_when_computed(p_tf_nuclear_heat_mw):
    """`inuclear == 0 and i_tf_sup == 1`: `.fwbs.qnuc` from `p_tf_nuclear_heat_mw`
    alone, matching `calculate_cryo_qnuc`'s computed arm without that function's
    incumbent-`qnuc` read (see `CryoQNuc`'s docstring for why the two arms are split
    across separate nodes rather than one function). Moved out of the declaration and
    into a named function (`_audit/formulas_split.md` step 1).
    """
    return 1.0e6 * p_tf_nuclear_heat_mw
