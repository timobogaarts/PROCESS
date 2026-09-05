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
from functional_process.models.power.thermal_cryo import (
    calculate_component_thermal_powers,
    calculate_component_thermal_powers_owned,
    calculate_cryo,
    calculate_cryo_loads,
    calculate_cryo_plant_loads,
    calculate_cryo_plant_loads_active,
    calculate_cryo_plant_loads_inactive,
    calculate_cryo_q_loads,
    calculate_cryo_q_loads_resistive_tf,
    calculate_cryo_q_loads_superconducting_tf,
    calculate_cryo_qnuc,
    calculate_cryo_qnuc_when_computed,
    calculate_delta_eta,
    calculate_helpow,
    calculate_p_div_heat_deposited_mw,
    calculate_p_fw_blkt_coolant_pump_mw,
    calculate_p_fw_blkt_coolant_pump_mw_summed,
    calculate_p_fw_blkt_heat_deposited_mw,
    calculate_p_fw_div_heat_deposited_mw,
    calculate_p_fw_div_heat_deposited_mw_summed,
    calculate_p_fw_heat_deposited_mw,
    calculate_p_shld_heat_deposited_mw,
    calculate_plant_thermal_efficiency,
    calculate_plant_thermal_efficiency_2,
    cryo_is_active,
    eta_turbine_ccfe_hcpb_value,
    eta_turbine_ccfe_hcpb_value_with_divertor,
    eta_turbine_steam_rankine_cycle,
    eta_turbine_supercritical_co2,
    etath_liq_supercritical_co2,
    temp_turbine_coolant_in_from_blanket_coolant,
    temp_turbine_coolant_in_from_liquid_breeder,
)
from functional_process.vocabulary import (
    BlktModelTypes,
    ElectricConversionModelTypes,
    PFConductorModel,
    ProcessValueError,
    PumpingPowerModelTypes,
    TFConductorModel,
    constants,
)

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just the few that turn
# out to be unused here (see `power/electric_production.py`'s commit for why a partial
# list is the wrong move: it silently drops the D101/D102 checks on the rest).
__all__ = [
    "BlanketDualCoolantModel",
    "BlktModelTypes",
    "CoilNuclearHeatingModel",
    "ComponentThermalPowers",
    "Cryo",
    "CryoLoads",
    "CryoLoadsActive",
    "CryoLoadsInactive",
    "CryoQLoads",
    "CryoQLoadsResistiveTf",
    "CryoQLoadsSuperconductingTf",
    "CryoQNuc",
    "CryoQNucStep",
    "DeltaEtaStep",
    "ElectricConversionModelTypes",
    "EtaTurbine",
    "EtaTurbineCcfeHcpbValue",
    "EtaTurbineCcfeHcpbValueWithDivertor",
    "EtaTurbineSteamRankineCycle",
    "EtaTurbineSupercriticalCo2",
    "EtathLiq",
    "EtathLiqSupercriticalCo2",
    "ExplicitFunction",
    "FixedPointFunction",
    "From",
    "OutputInto",
    "PFConductorModel",
    "PFwBlktCoolantPumpMw",
    "PFwDivHeatDepositedMw",
    "PFwDivHeatDepositedMwSummed",
    "ProcessValueError",
    "PumpingPowerModelTypes",
    "TFConductorModel",
    "TempTurbineCoolantIn",
    "TempTurbineCoolantInFromBlanketCoolant",
    "TempTurbineCoolantInFromLiquidBreeder",
    "calculate_component_thermal_powers",
    "calculate_component_thermal_powers_owned",
    "calculate_cryo",
    "calculate_cryo_loads",
    "calculate_cryo_plant_loads",
    "calculate_cryo_plant_loads_active",
    "calculate_cryo_plant_loads_inactive",
    "calculate_cryo_q_loads",
    "calculate_cryo_q_loads_resistive_tf",
    "calculate_cryo_q_loads_superconducting_tf",
    "calculate_cryo_qnuc",
    "calculate_cryo_qnuc_when_computed",
    "calculate_delta_eta",
    "calculate_helpow",
    "calculate_p_div_heat_deposited_mw",
    "calculate_p_fw_blkt_coolant_pump_mw",
    "calculate_p_fw_blkt_coolant_pump_mw_summed",
    "calculate_p_fw_blkt_heat_deposited_mw",
    "calculate_p_fw_div_heat_deposited_mw",
    "calculate_p_fw_div_heat_deposited_mw_summed",
    "calculate_p_fw_heat_deposited_mw",
    "calculate_p_shld_heat_deposited_mw",
    "calculate_plant_thermal_efficiency",
    "calculate_plant_thermal_efficiency_2",
    "constants",
    "cryo_is_active",
    "current_drive",
    "eqx",
    "eta_turbine_ccfe_hcpb_value",
    "eta_turbine_ccfe_hcpb_value_with_divertor",
    "eta_turbine_steam_rankine_cycle",
    "eta_turbine_supercritical_co2",
    "etath_liq_supercritical_co2",
    "fwbs",
    "heat_transport",
    "jnp",
    "pf_power",
    "physics",
    "power",
    "primary_pumping",
    "structure",
    "temp_turbine_coolant_in_from_blanket_coolant",
    "temp_turbine_coolant_in_from_liquid_breeder",
    "tfcoil",
    "times",
]


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
