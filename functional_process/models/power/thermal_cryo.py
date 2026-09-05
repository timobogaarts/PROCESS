"""Pure-functional port of the thermal-power-balance and cryogenics sub-unit of
`process/models/power.py` (registry unit #14, chunk B).

Audit record: `functional_process/_audit/units/models/power/thermal_cryo.md`. Covers
`Power.component_thermal_powers` (814-1036), `Power.plant_thermal_efficiency`
(1935-2071), `Power.plant_thermal_efficiency_2` (2073-2116),
`Power.calculate_cryo_loads` (1037-1118) and `Power.cryo` (1773-1852) -- see the
audit record's data-footprint table for the full trace and, in particular, the six
self-loop findings ("The `delta_eta` self-loop" and "The `eta_turbine`/`etath_liq`/
`temp_turbine_coolant_in`/`p_fw_div_heat_deposited_mw`/`p_fw_blkt_coolant_pump_mw`
self-loops" sections), which is why `component_thermal_powers`'s node-level split
below exists: `calculate_component_thermal_powers` (the pure function) is unchanged,
but at the node level each of the six self-references is owned by its own node, separate
from `ComponentThermalPowers`.

**Five of the six are not `FixedPointFunction`s any anymore** (`_audit/next_steps.md`
§14.2/§14.11). Splitting the switches each carried showed the self-read existed only on
the arm where PROCESS's own body is `return x` -- which is not a fixed point but the
statement that the field is an **input**, spelled in the tree as an empty slot. So
`EtaTurbineStep`/`EtathLiqStep`/`TempTurbineCoolantInStep`/`PFwDivHeatDepositedMwStep`/
`PFwBlktCoolantPumpMwStep` are gone, replaced by the `EtaTurbine`/`EtathLiq`/
`TempTurbineCoolantIn`/`PFwDivHeatDepositedMw`/`PFwBlktCoolantPumpMw` families below;
only `DeltaEtaStep` is still one, and only because its own three switches are not split
yet. `ComponentThermalPowers` no longer reads five of the six at all: it recomputed and
**discarded** them, so it was wired into four fixed points it does not consume. See each
class's own docstring.

All five source methods are tier-1 once ported: no internal iteration anywhere in
this chunk (the `cryo`/`calculate_cryo_loads` pair looks like it could be tier-2 from
its name, but it is a single straight-line evaluation, not a solve).
"""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    From,
    OutputInto,
)

from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    CoilNuclearHeatingModel,
)
from functional_process.paths import (
    current_drive,
    fwbs,
    heat_transport,
    pf_power,
    physics,
    power,
    primary_pumping,
    structure,
    tfcoil,
    times,
)
from functional_process.vocabulary import constants
from functional_process.vocabulary import ProcessValueError
from functional_process.vocabulary import BlktModelTypes
from functional_process.vocabulary import PFConductorModel
from functional_process.vocabulary import (
    ElectricConversionModelTypes,
    PumpingPowerModelTypes,
)
from functional_process.vocabulary import TFConductorModel


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


class PlantThermalEfficiency(ExplicitFunction):
    """cottax node: `calculate_plant_thermal_efficiency`."""

    i_thermal_electric_conversion: ElectricConversionModelTypes = eqx.field(static=True)
    i_blanket_type: BlktModelTypes = eqx.field(static=True)

    eta_turbine = OutputInto(heat_transport)
    temp_turbine_coolant_in = OutputInto(heat_transport)

    def __call__(
        self,
        eta_turbine=From(heat_transport),
        delta_eta=From(power),
        temp_blkt_coolant_out=From(fwbs),
        temp_turbine_coolant_in=From(heat_transport),
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

    secondary_cycle_liq: ElectricConversionModelTypes = eqx.field(static=True)

    etath_liq = OutputInto(heat_transport)
    temp_turbine_coolant_in = OutputInto(heat_transport)

    def __call__(
        self,
        etath_liq=From(heat_transport),
        outlet_temp_liq=From(fwbs),
        temp_turbine_coolant_in=From(heat_transport),
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
    `where` and one `FromExactly`'s `where` naming the identical `VarPath`), which `cottax`
    refuses to build as one node (`ValueError: reads [...], which it also owns`,
    confirmed directly against this class before these splits existed). Each
    `FixedPointFunction` below isolates one self-reference; this class keeps every
    *other* output of `calculate_component_thermal_powers` and takes all six fields as
    ordinary plain-`FromExactly`s (the current/entering values), same as every other
    parameter here. Ownership of the six real `VarPath`s belongs to their own nodes,
    not to this class -- see `thermal_cryo.md`.

    **And five of the six are not read here either, since `_audit/next_steps.md` §14.2.**
    This node recomputed `p_fw_div_heat_deposited_mw`, `eta_turbine`, `etath_liq`,
    `temp_turbine_coolant_in` and `delta_eta` internally and **discarded** every one of
    them -- `switch_kwarg_survey.md` §6's "reads dead at every value of the node's own
    switches" residue -- so it declared itself a consumer of four fixed points it does
    not consume. Only `.primary_pumping.p_fw_blkt_coolant_pump_mw` is genuinely used (the
    pump totals sum it) and only it is still read. Two static switches went with the
    recomputation: `i_blanket_type` and `secondary_cycle_liq` fed nothing else.
    """

    i_p_coolant_pumping: PumpingPowerModelTypes = eqx.field(static=True)
    i_blkt_dual_coolant: BlanketDualCoolantModel = eqx.field(static=True)
    i_thermal_electric_conversion: ElectricConversionModelTypes = eqx.field(static=True)

    # .primary_pumping.p_fw_blkt_coolant_pump_mw is NOT declared here --
    # `PFwBlktCoolantPumpMw` owns it. Still read below, as a plain FromExactly: the
    # pump totals sum it.
    p_fw_blkt_coolant_pump_elec_mw = OutputInto(power)
    p_shld_coolant_pump_elec_mw = OutputInto(power)
    p_div_coolant_pump_elec_mw = OutputInto(power)
    p_blkt_breeder_pump_elec_mw = OutputInto(power)
    p_coolant_pump_total_mw = OutputInto(power)
    p_coolant_pump_elec_total_mw = OutputInto(heat_transport)
    p_coolant_pump_loss_total_mw = OutputInto(heat_transport)
    p_hcd_electric_loss_mw = OutputInto(heat_transport)
    p_blkt_liquid_breeder_heat_deposited_mw = OutputInto(power)
    p_fw_blkt_heat_deposited_mw = OutputInto(power)
    p_fw_heat_deposited_mw = OutputInto(power)
    p_blkt_heat_deposited_mw = OutputInto(power)
    p_shld_heat_deposited_mw = OutputInto(power)
    p_div_heat_deposited_mw = OutputInto(power)
    # .heat_transport.p_fw_div_heat_deposited_mw / .eta_turbine / .etath_liq /
    # .temp_turbine_coolant_in are NOT declared here -- each is owned by its own family
    # below -- and, since `_audit/next_steps.md` §14.2, **not read here either**: this
    # node recomputed all four and discarded the results.
    p_plant_primary_heat_mw = OutputInto(heat_transport)
    p_div_secondary_heat_mw = OutputInto(heat_transport)
    i_div_primary_heat = OutputInto(power)
    f_p_div_primary_heat = OutputInto(power)
    # .power.delta_eta is NOT declared here -- DeltaEtaStep's FixedPoint problem node
    # owns it (see "The delta_eta self-loop" in thermal_cryo.md) -- and is no longer
    # read here either, for the same reason as the four above.
    p_shld_secondary_heat_mw = OutputInto(heat_transport)
    p_hcd_secondary_heat_mw = OutputInto(heat_transport)
    n_primary_heat_exchangers = OutputInto(heat_transport)

    def __call__(
        self,
        p_fw_coolant_pump_mw=From(heat_transport),
        p_blkt_coolant_pump_mw=From(heat_transport),
        p_fw_blkt_coolant_pump_mw=From(primary_pumping),
        eta_coolant_pump_electric=From(fwbs),
        p_shld_coolant_pump_mw=From(heat_transport),
        p_div_coolant_pump_mw=From(heat_transport),
        p_blkt_breeder_pump_mw=From(heat_transport),
        p_hcd_electric_total_mw=From(heat_transport),
        p_hcd_injected_total_mw=From(current_drive),
        p_blkt_nuclear_heat_total_mw=From(fwbs),
        f_nuc_pow_bz_liq=From(fwbs),
        p_fw_nuclear_heat_total_mw=From(fwbs),
        p_fw_rad_total_mw=From(fwbs),
        p_beam_orbit_loss_mw=From(current_drive),
        p_fw_alpha_mw=From(physics),
        p_beam_shine_through_mw=From(current_drive),
        p_cp_shield_nuclear_heat_mw=From(fwbs),
        p_shld_nuclear_heat_mw=From(fwbs),
        p_plasma_separatrix_mw=From(physics),
        p_div_nuclear_heat_total_mw=From(fwbs),
        p_div_rad_total_mw=From(fwbs),
        p_fw_hcd_nuclear_heat_mw=From(fwbs),
        p_fw_hcd_rad_total_mw=From(fwbs),
        i_shld_primary_heat=From(heat_transport),
    ):
        return calculate_component_thermal_powers_owned(
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
            p_fw_hcd_nuclear_heat_mw,
            p_fw_hcd_rad_total_mw,
            i_shld_primary_heat,
            self.i_thermal_electric_conversion,
        )


class DeltaEtaStep(FixedPointFunction):
    """The `.power.delta_eta` self-loop, cut.

    `calculate_component_thermal_powers` reads the entering `.power.delta_eta` (it
    feeds `calculate_plant_thermal_efficiency`'s `CCFE_HCPB_VALUE_WITH_DIVERTOR`/
    `STEAM_RANKINE_CYCLE` branches) and produces a freshly-computed
    `.power.delta_eta` later in the same call -- see `thermal_cryo.md`'s "The
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
    `test_thermal_cryo.py`: `d(delta_eta_next)/d(delta_eta) == 0` exactly, for
    every switch combination this chunk supports, not just the two branches of
    `calculate_plant_thermal_efficiency` that read it. It is declared as a `step`
    parameter anyway (per this shape's own convention: `step` reads the value it is
    about to replace) purely for structural symmetry with the self-loop it is cutting,
    not because the computation needs it. Were a driver ever assigned to the
    resulting `FixedPoint` problem node (not done here -- see `_audit/next_steps.md`
    § 5, "What stays deferred"), this zero derivative means it converges in exactly
    one iteration regardless of algorithm, from any starting value.
    """

    i_p_coolant_pumping: PumpingPowerModelTypes = eqx.field(static=True)
    i_blkt_dual_coolant: BlanketDualCoolantModel = eqx.field(static=True)
    i_thermal_electric_conversion: ElectricConversionModelTypes = eqx.field(static=True)

    delta_eta = OutputInto(power)

    def step(
        self,
        p_fw_coolant_pump_mw=From(heat_transport),
        p_blkt_coolant_pump_mw=From(heat_transport),
        p_fw_blkt_coolant_pump_mw=From(primary_pumping),
        p_fw_nuclear_heat_total_mw=From(fwbs),
        p_fw_rad_total_mw=From(fwbs),
        p_blkt_nuclear_heat_total_mw=From(fwbs),
        p_blkt_breeder_pump_mw=From(heat_transport),
        p_beam_orbit_loss_mw=From(current_drive),
        p_fw_alpha_mw=From(physics),
        p_beam_shine_through_mw=From(current_drive),
        p_cp_shield_nuclear_heat_mw=From(fwbs),
        p_shld_nuclear_heat_mw=From(fwbs),
        p_shld_coolant_pump_mw=From(heat_transport),
        p_plasma_separatrix_mw=From(physics),
        p_div_nuclear_heat_total_mw=From(fwbs),
        p_div_rad_total_mw=From(fwbs),
        p_div_coolant_pump_mw=From(heat_transport),
        i_shld_primary_heat=From(heat_transport),
        delta_eta=From(power),
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


class EtaTurbine(ExplicitFunction):
    """The `.heat_transport.eta_turbine` family -- one occupant per arm of
    `.fwbs.i_thermal_electric_conversion` x `.fwbs.i_blanket_type`.

    **This was `EtaTurbineStep`, a `FixedPointFunction`, and splitting the switches
    deleted the fixed point.** The self-loop existed because
    `calculate_plant_thermal_efficiency` takes the entering `eta_turbine` and returns
    it unchanged on four of its eight arms -- `USER_INPUT`, and `CCFE_HCPB_VALUE` /
    `CCFE_HCPB_VALUE_WITH_DIVERTOR` / `STEAM_RANKINE_CYCLE` with a non-HCPB blanket.
    A body whose whole content is `return eta_turbine` is not a fixed point; it is the
    statement that **the field is an input**, and the tree spells that as an **empty
    slot** (`indat.ETA_TURBINE[4] is None`), exactly as it spells `inuclear`'s
    (`_audit/next_steps.md` §14.4) and `cplife`'s.

    That matters beyond tidiness. `switch_kwarg_survey.md` §4.7 measured this node as
    one of two `FixedPoint`s that are the **identity map on the reference machine**,
    and singled it out: *"this one has a live downstream --
    `availability.electric_production` reads `.heat_transport.eta_turbine` to form
    `p_plant_electric_gross_mw`, which reaches `.costs.coe`, this run's objective. So a
    quantity the objective depends on is nominally driven while being, on this
    configuration, an unowned boundary input in disguise."* It is no longer in disguise.

    The four computing occupants read genuinely different fields -- nothing,
    `.power.delta_eta`, `.fwbs.temp_blkt_coolant_out`, or both -- where the one node
    declared all three unconditionally.
    """

    eta_turbine = OutputInto(heat_transport)


class EtaTurbineCcfeHcpbValue(EtaTurbine):
    """`i_thermal_electric_conversion == CCFE_HCPB_VALUE` (0) with
    `i_blanket_type == CCFE_HCPB`.

    **A node with no inputs at all**: the efficiency is a literal.
    """

    def __call__(self):
        return eta_turbine_ccfe_hcpb_value()


class EtaTurbineCcfeHcpbValueWithDivertor(EtaTurbine):
    """`i_thermal_electric_conversion == CCFE_HCPB_VALUE_WITH_DIVERTOR` (1) with
    `i_blanket_type == CCFE_HCPB`. Reads `.power.delta_eta` alone.
    """

    def __call__(self, delta_eta=From(power)):
        return eta_turbine_ccfe_hcpb_value_with_divertor(delta_eta)


class EtaTurbineSteamRankineCycle(EtaTurbine):
    """`i_thermal_electric_conversion == STEAM_RANKINE_CYCLE` (3) with
    `i_blanket_type == CCFE_HCPB`. Reads `.fwbs.temp_blkt_coolant_out` and
    `.power.delta_eta`.
    """

    def __call__(
        self,
        temp_blkt_coolant_out=From(fwbs),
        delta_eta=From(power),
    ):
        return eta_turbine_steam_rankine_cycle(temp_blkt_coolant_out, delta_eta)


class EtaTurbineSupercriticalCo2(EtaTurbine):
    """`i_thermal_electric_conversion == SUPERCRITICAL_CO2_BRAYTON_CYCLE` (4), at any
    blanket type. Reads `.fwbs.temp_blkt_coolant_out` alone.
    """

    def __call__(self, temp_blkt_coolant_out=From(fwbs)):
        return eta_turbine_supercritical_co2(temp_blkt_coolant_out)


class EtathLiq(ExplicitFunction):
    """The `.heat_transport.etath_liq` family -- one occupant per
    `.fwbs.secondary_cycle_liq` value.

    **This was `EtathLiqStep`, a `FixedPointFunction`, and splitting the switch deleted
    the fixed point** -- the same shape as `EtaTurbine` above.
    `calculate_plant_thermal_efficiency_2`'s `secondary_cycle_liq == 2` arm is
    `return etath_liq`, i.e. "the liquid-breeder efficiency is an input", which is an
    **empty slot** (`indat.ETATH_LIQ[1] is None`); its `== 4` arm computes from
    `.fwbs.outlet_temp_liq` alone and never reads the entering value.
    """

    etath_liq = OutputInto(heat_transport)


class EtathLiqSupercriticalCo2(EtathLiq):
    """`secondary_cycle_liq == 4` -- the reference run's. Reads `.fwbs.outlet_temp_liq`
    alone.
    """

    def __call__(self, outlet_temp_liq=From(fwbs)):
        return etath_liq_supercritical_co2(outlet_temp_liq)


class TempTurbineCoolantIn(ExplicitFunction):
    """The `.heat_transport.temp_turbine_coolant_in` family -- one occupant per arm of
    `.fwbs.i_thermal_electric_conversion` x `.fwbs.i_blanket_type` x
    `.fwbs.secondary_cycle_liq`.

    **This was `TempTurbineCoolantInStep`, a `FixedPointFunction`, and splitting the
    three switches deleted the fixed point.** The field is written by two stages in
    order (`calculate_plant_thermal_efficiency` then `_2`, see `thermal_cryo.md`'s
    "Write-ordering on `temp_turbine_coolant_in`"), and the self-loop existed only
    because *both* stages can pass the entering value through. Once the switches select
    occupants there are exactly three cases, and none of them reads what it owns:

    * stage two writes it (`secondary_cycle_liq == 4`) -- from `.fwbs.outlet_temp_liq`,
      **overwriting** stage one, so stage one's inputs are not read at all;
    * stage two passes through and stage one writes it -- from
      `.fwbs.temp_blkt_coolant_out`;
    * both pass through -- the field is an **input**, spelled as an empty slot
      (`indat.TEMP_TURBINE_COOLANT_IN[2] is None`).

    The one node carrying all three switches declared both source fields on every
    configuration, and read the field it owned on top.
    """

    temp_turbine_coolant_in = OutputInto(heat_transport)


class TempTurbineCoolantInFromLiquidBreeder(TempTurbineCoolantIn):
    """`secondary_cycle_liq == 4` -- the reference run's. Stage two overwrites whatever
    stage one wrote, so this reads `.fwbs.outlet_temp_liq` and **not**
    `.fwbs.temp_blkt_coolant_out`, whichever `i_thermal_electric_conversion` is set.
    """

    def __call__(self, outlet_temp_liq=From(fwbs)):
        return temp_turbine_coolant_in_from_liquid_breeder(outlet_temp_liq)


class TempTurbineCoolantInFromBlanketCoolant(TempTurbineCoolantIn):
    """`secondary_cycle_liq == 2` with an `i_thermal_electric_conversion` arm that
    writes the turbine inlet temperature -- `STEAM_RANKINE_CYCLE` with a CCFE HCPB
    blanket, or `SUPERCRITICAL_CO2_BRAYTON_CYCLE` at any blanket. Reads
    `.fwbs.temp_blkt_coolant_out` alone.
    """

    def __call__(self, temp_blkt_coolant_out=From(fwbs)):
        return temp_turbine_coolant_in_from_blanket_coolant(temp_blkt_coolant_out)


class PFwDivHeatDepositedMw(ExplicitFunction):
    """The `.heat_transport.p_fw_div_heat_deposited_mw` family -- one occupant per arm
    of `.fwbs.i_p_coolant_pumping`.

    **This was `PFwDivHeatDepositedMwStep`, a `FixedPointFunction`, and splitting the
    switch deleted the fixed point.** `calculate_p_fw_div_heat_deposited_mw` is a
    conditional-ownership pass-through (`power.py:955-961`): on
    `MECHANICAL_WITH_PRESSURE_DROP` it returns the entering value unchanged -- the
    field's only other producer anywhere in `process/` is `models/ife.py`, out of scope
    -- and on every other value it recomputes from raw inputs and never reads it. So
    the self-read existed only on the arm where the field is an **input**, which is an
    empty slot (`indat.P_FW_DIV_HEAT_DEPOSITED[1] is None`).
    """

    p_fw_div_heat_deposited_mw = OutputInto(heat_transport)


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


class PFwDivHeatDepositedMwSummed(PFwDivHeatDepositedMw):
    """`i_p_coolant_pumping != MECHANICAL_WITH_PRESSURE_DROP` -- the reference run's.

    The first wall's and the divertor's deposited heat, summed. Rebuilds both from the
    same raw, externally-owned inputs `calculate_component_thermal_powers` itself reads
    rather than from that node's `Output`s -- the "read the same raw inputs twice, not
    another node's outputs" pattern this file established, which keeps the two nodes
    fanned out rather than mutually referencing.
    """

    def __call__(
        self,
        p_fw_nuclear_heat_total_mw=From(fwbs),
        p_fw_rad_total_mw=From(fwbs),
        p_fw_coolant_pump_mw=From(heat_transport),
        p_beam_orbit_loss_mw=From(current_drive),
        p_fw_alpha_mw=From(physics),
        p_beam_shine_through_mw=From(current_drive),
        p_plasma_separatrix_mw=From(physics),
        p_div_nuclear_heat_total_mw=From(fwbs),
        p_div_rad_total_mw=From(fwbs),
        p_div_coolant_pump_mw=From(heat_transport),
    ):
        return calculate_p_fw_div_heat_deposited_mw_summed(
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


class PFwBlktCoolantPumpMw(ExplicitFunction):
    """`.primary_pumping.p_fw_blkt_coolant_pump_mw` at the `.fwbs.i_p_coolant_pumping`
    values where `power` owns it: `USER_INPUT` and `FRACTION_OF_HEAT`.

    **This was `PFwBlktCoolantPumpMwStep`, a `FixedPointFunction`, and it is not one
    now.** The slot already spelled the other two values as `None` (arm `1` of
    `indat.P_FW_BLKT_COOLANT_PUMP` -- `hcpb.py` owns the field on `MECHANICAL` /
    `MECHANICAL_WITH_PRESSURE_DROP`), so the occupant's own `i_p_coolant_pumping` kwarg
    could only ever hold a value on which the body's `if` takes the *computing* branch.
    Carrying the switch made the node read the field it owns for a branch it can never
    take; removing it makes the node an ordinary sum of two pump powers.
    """

    p_fw_blkt_coolant_pump_mw = OutputInto(primary_pumping)

    def __call__(
        self,
        p_fw_coolant_pump_mw=From(heat_transport),
        p_blkt_coolant_pump_mw=From(heat_transport),
    ):
        return calculate_p_fw_blkt_coolant_pump_mw_summed(
            p_fw_coolant_pump_mw, p_blkt_coolant_pump_mw
        )


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

    i_tf_sup: TFConductorModel = eqx.field(static=True)
    inuclear: CoilNuclearHeatingModel = eqx.field(static=True)

    helpow = OutputInto(heat_transport)
    qss = OutputInto(power)
    qac = OutputInto(power)
    qcl = OutputInto(power)
    qmisc = OutputInto(power)
    qnuc = OutputInto(fwbs)

    def __call__(
        self,
        tfcryoarea=From(tfcoil),
        coldmass=From(structure),
        p_tf_nuclear_heat_mw=From(fwbs),
        ensxpfm=From(pf_power),
        t_plant_pulse_plasma_present=From(times),
        c_tf_turn=From(tfcoil),
        n_tf_coils=From(tfcoil),
        qnuc=From(fwbs),
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


def calculate_cryo_qnuc_when_computed(p_tf_nuclear_heat_mw):
    """`inuclear == 0 and i_tf_sup == 1`: `.fwbs.qnuc` from `p_tf_nuclear_heat_mw`
    alone, matching `calculate_cryo_qnuc`'s computed arm without that function's
    incumbent-`qnuc` read (see `CryoQNuc`'s docstring for why the two arms are split
    across separate nodes rather than one function). Moved out of the declaration and
    into a named function (`_audit/formulas_split.md` step 1).
    """
    return 1.0e6 * p_tf_nuclear_heat_mw


class CryoQNuc(ExplicitFunction):
    """`.fwbs.qnuc` when PROCESS computes it: `inuclear == 0` and `i_tf_sup == 1`.

    **This replaces a `FixedPointFunction`, and the fixed point was an artefact of the
    switch.** `calculate_cryo_qnuc` is `if inuclear == 0 and i_tf_sup == 1: qnuc = 1e6 *
    p_tf_nuclear_heat_mw; return qnuc` -- so one arm computes the field from another
    field and the other returns the incumbent untouched, which is PROCESS's own *"Issue
    #511: if inuclear = 1: qnuc is input"*. Written as one node carrying `inuclear` as a
    static kwarg, that body both reads and owns `.fwbs.qnuc`, which `to_graph` refuses
    outright -- Shape B -- so it had to be declared as a fixed point over a minted
    `^cond` copy, and the "qnuc is input" arm then showed up as a *structurally
    degenerate* problem that `sand.degenerate_fixed_points` detected at runtime by
    differentiating the residual and dropped.

    Split into occupants, none of that exists. The computed arm reads
    `p_tf_nuclear_heat_mw` and **not** the incumbent, so it is an ordinary
    `ExplicitFunction`; the input arm is not a node at all, it is an empty slot, and
    `.fwbs.qnuc` is an ordinary boundary input. What was recovered by differentiating a
    residual is now stated by the tree, and one driven block goes with it.
    """

    qnuc = OutputInto(fwbs)

    def __call__(self, p_tf_nuclear_heat_mw=From(fwbs)):
        return calculate_cryo_qnuc_when_computed(p_tf_nuclear_heat_mw)


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

    i_tf_sup: TFConductorModel = eqx.field(static=True)
    inuclear: CoilNuclearHeatingModel = eqx.field(static=True)

    qnuc = OutputInto(fwbs)

    def step(
        self,
        qnuc=From(fwbs),
        p_tf_nuclear_heat_mw=From(fwbs),
    ):
        return calculate_cryo_qnuc(
            self.i_tf_sup, self.inuclear, p_tf_nuclear_heat_mw, qnuc
        )


class CryoQLoads(ExplicitFunction):
    """The `.power.qss`/`qac`/`qcl`/`qmisc` family -- one occupant per arm of
    `.tfcoil.i_tf_sup` x `.pf_coil.i_pf_conductor`.

    **This was `CryoQLoadsStep`, a `FixedPointFunction`, and splitting the two switches
    deleted the fixed point.** `Power.calculate_cryo_loads` calls `Power.cryo` only
    when `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` (`power.py:1054-1057`);
    outside that guard the four fields are left exactly as they entered, and the node
    read what it owned purely to pass it through. That arm is an **empty slot**
    (`indat.CRYO_Q_LOADS[2] is None`): the four fields are inputs there. Inside the
    guard neither occupant reads any of the four, so neither is a fixed point.

    The two computing arms differ by three reads: `qss` carries the TF cryogenic
    surface and `qcl` the TF turn current only when the TF coil is superconducting, so
    the resistive-TF occupant declares neither `.tfcoil.tfcryoarea` nor
    `.tfcoil.c_tf_turn` nor `.tfcoil.n_tf_coils`.

    **`qnuc` is read, not recomputed.** `qmisc = 0.45 * (qss + qnuc + qac + qcl)` uses
    the value `Power.cryo` has just written, so both occupants read `.fwbs.qnuc` -- the
    real `VarPath` `CryoQNuc` owns. An ordinary edge, not a cycle.
    """

    qss = OutputInto(power)
    qac = OutputInto(power)
    qcl = OutputInto(power)
    qmisc = OutputInto(power)


class CryoQLoadsSuperconductingTf(CryoQLoads):
    """`i_tf_sup == SUPERCONDUCTING` (1) -- the reference run's."""

    def __call__(
        self,
        qnuc=From(fwbs),
        tfcryoarea=From(tfcoil),
        coldmass=From(structure),
        ensxpfm=From(pf_power),
        t_plant_pulse_plasma_present=From(times),
        c_tf_turn=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return calculate_cryo_q_loads_superconducting_tf(
            tfcryoarea,
            coldmass,
            ensxpfm,
            t_plant_pulse_plasma_present,
            c_tf_turn,
            n_tf_coils,
            qnuc,
        )


class CryoQLoadsResistiveTf(CryoQLoads):
    """`i_tf_sup != SUPERCONDUCTING` with `i_pf_conductor == SUPERCONDUCTING` -- the
    PF coils are what needs cooling.

    **Three reads leave with this occupant**: `.tfcoil.tfcryoarea`,
    `.tfcoil.c_tf_turn`, `.tfcoil.n_tf_coils`.
    """

    def __call__(
        self,
        qnuc=From(fwbs),
        coldmass=From(structure),
        ensxpfm=From(pf_power),
        t_plant_pulse_plasma_present=From(times),
    ):
        return calculate_cryo_q_loads_resistive_tf(
            coldmass, ensxpfm, t_plant_pulse_plasma_present, qnuc
        )


class CryoLoads(ExplicitFunction):
    """The unconditionally-owned part of `Power.calculate_cryo_loads` -- one occupant
    per arm of `.tfcoil.i_tf_sup` x `.pf_coil.i_pf_conductor`.

    Owns the four fields that method writes on every path through it:
    `.heat_transport.helpow`, `.heat_transport.p_cryo_plant_electric_mw`,
    `.heat_transport.helpow_cryal` and `.tfcoil.cryo_cool_req`. The five conditionally
    owned `q*` fields belong to `CryoQNuc`/`CryoQLoads` and are read here as plain
    `FromExactly`s.

    **Both switches were `eqx.field(static=True)`s here and neither is now**
    (`_audit/next_steps.md` §14.2). They gate two things: whether the cryoplant runs at
    all, and whether the TF coil is aluminium. The second has no occupant --
    `('i_tf_sup', 2)` is `UNPORTED` at the `power.tf_power` slot -- so **four reads
    leave the family entirely**: `.tfcoil.p_cp_resistive`,
    `.tfcoil.p_tf_leg_resistive`, `.tfcoil.p_tf_joints_resistive` and
    `.fwbs.pnuc_cp_tf` are the aluminium arm's alone. The first splits the remaining
    two occupants, and the inactive one reads neither `.tfcoil.eff_tf_cryo` nor any of
    the five `q*` fields.

    **What registering it closes.** `.heat_transport.helpow` (read by `Bldgs` and
    `CryogenicSystemCost`) and `.heat_transport.p_cryo_plant_electric_mw` (read by
    `Acpow`, the electric-production occupants and `AuxiliaryComponentCoolingCost`)
    were boundary inputs -- see `_audit/boundary_inputs_audit.md` §4c (b9)/(b10).
    """

    helpow = OutputInto(heat_transport)
    p_cryo_plant_electric_mw = OutputInto(heat_transport)
    helpow_cryal = OutputInto(heat_transport)
    cryo_cool_req = OutputInto(tfcoil)


class CryoLoadsActive(CryoLoads):
    """`i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING` -- the reference run's.
    The cryoplant runs, so the five `q*` terms and `.tfcoil.eff_tf_cryo` are live.
    """

    def __call__(
        self,
        eff_tf_cryo=From(tfcoil),
        temp_tf_cryo=From(tfcoil),
        temp_cp_coolant_inlet=From(tfcoil),
        qss=From(power),
        qac=From(power),
        qcl=From(power),
        qmisc=From(power),
        qnuc=From(fwbs),
    ):
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


class CryoLoadsInactive(CryoLoads):
    """`i_tf_sup != 1` with resistive PF coils: PROCESS never calls `Power.cryo`, so
    `helpow` and `p_cryo_plant_electric_mw` are literal zeros.

    **Six reads leave with this occupant**: `.tfcoil.eff_tf_cryo` and the five `q*`
    fields. `temp_tf_cryo`/`temp_cp_coolant_inlet` stay because `cryo_cool_req`'s
    formula divides by both unconditionally, exactly as PROCESS's does.
    """

    def __call__(
        self,
        temp_tf_cryo=From(tfcoil),
        temp_cp_coolant_inlet=From(tfcoil),
    ):
        return calculate_cryo_plant_loads_inactive(temp_tf_cryo, temp_cp_coolant_inlet)
