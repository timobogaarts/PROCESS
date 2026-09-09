"""The power subsystem's namespace -- thermal and electric flows, and cryogenics."""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.models.power.electric_production import (
    Acpow,
)
from functional_process.cottax.models.power.pf_coil_power import PfCoilPowerSupplies
from functional_process.cottax.models.power.tf_coil_power import (
    TfPowerResistive,
    TfPowerSuperconducting,
)
from functional_process.cottax.models.power.thermal_cryo import (
    ComponentThermalPowers,
    CryoLoads,
    CryoQLoads,
    CryoQNuc,
    DeltaEtaStep,
    EtathLiq,
    EtaTurbine,
    PFwBlktCoolantPumpMw,
    PFwDivHeatDepositedMw,
    TempTurbineCoolantIn,
)


class Power(ModelNamespace):
    """Thermal and electric power flows, cryogenics, and the plant's own consumption."""

    pf_coil_power: PfCoilPowerSupplies | None = dataclasses.field(kw_only=True)
    """`Power.pfpwr` -- **the one subsystem of `power.py` a tokamak has and a
    stellarator does not**, so the one slot in this namespace that can be honestly
    absent.
    """

    tf_power: TfPowerResistive | TfPowerSuperconducting = dataclasses.field(kw_only=True)
    """TF-coil power supplies (`.tfcoil.i_tf_sup`, default 1 = superconducting)."""

    # `tf_coil_power.py` (unit #14 chunk A). `TfPowerResistive`/
    # `TfPowerSuperconducting` are registered under `TOPOLOGY_SWITCHES`'s new
    # `.tfcoil.i_tf_sup` switch instead of here -- see that switch's own comment.
    # `thermal_cryo.py` (unit #14 chunk B). Six of `calculate_
    # component_thermal_powers`'s outputs are genuine single-node self-loops (each
    # field's *entering* value is read, then a freshly-computed value is written back to
    # the same `VarPath` later in the same PROCESS call) -- already split into their own
    # occupant families/`FixedPointFunction`s, same "Shape B" treatment as
    # `plasma_composition`'s `first_call`/`Avail`'s `cplife` above.
    #
    # **`component_thermal_powers`/`delta_eta_step` used to be the two switches
    # `_audit/switch_kwarg_survey.md` exempted as "too costly to split" -- withdrawn
    # 2026-09-09, see `naming_convention.md` § "Switches are not ports".** Both slots
    # are now occupant families the same as everything else here: `indat.py` resolves
    # `i_p_coolant_pumping`/`i_blkt_dual_coolant`/`i_thermal_electric_conversion` once
    # and selects one of `ComponentThermalPowers`'s twelve arms (the full three-way
    # product) or `DeltaEtaStep`'s eight (a coarser, binary role for
    # `i_blkt_dual_coolant` here -- only `calculate_p_fw_blkt_heat_deposited_mw`'s `in
    # (1, 2)` guard reads it) from a registry, instead of the switch ever being carried
    # as a static kwarg into a node body.
    component_thermal_powers: ComponentThermalPowers = dataclasses.field(kw_only=True)
    delta_eta_step: DeltaEtaStep = dataclasses.field(kw_only=True)
    eta_turbine: EtaTurbine | None = dataclasses.field(kw_only=True)
    """`.heat_transport.eta_turbine` -- **or nothing, which is what the reference run
    gets.** This was `eta_turbine_step`, a `FixedPointFunction` carrying
    `i_thermal_electric_conversion` and `i_blanket_type`.
    """
    etath_liq: EtathLiq | None = dataclasses.field(kw_only=True)
    """`.heat_transport.etath_liq` -- the same shape as `eta_turbine`, on
    `.fwbs.secondary_cycle_liq`.
    """
    temp_turbine_coolant_in: TempTurbineCoolantIn | None = dataclasses.field(
        kw_only=True
    )
    """`.heat_transport.temp_turbine_coolant_in` -- three arms of
    `i_thermal_electric_conversion` x `i_blanket_type` x `secondary_cycle_liq`, one of
    them absent.
    """
    p_fw_div_heat_deposited_mw: PFwDivHeatDepositedMw | None = dataclasses.field(
        kw_only=True
    )
    """`.heat_transport.p_fw_div_heat_deposited_mw` -- owned on every
    `i_p_coolant_pumping` value except `MECHANICAL_WITH_PRESSURE_DROP`, where PROCESS
    passes the entering value through and the field's only other producer is
    `models/ife.py`, out of scope.
    """
    p_fw_blkt_coolant_pump_mw: PFwBlktCoolantPumpMw | None = dataclasses.field(
        kw_only=True
    )
    """`.primary_pumping.p_fw_blkt_coolant_pump_mw` -- **owned here on two of
    `i_p_coolant_pumping`'s four values and by the blanket on the other two.** This
    node's own docstring already called the field *"a conditional-ownership
    pass-through"* and named the other producer -- `process/models/blankets/hcpb.py`,
    *"not yet ported, registry unit #13"*.
    """
    # `PlantThermalEfficiency`/`PlantThermalEfficiency2` (the raw, un-split
    # `ExplicitFunction`s `EtaTurbineStep`/`EtathLiqStep`/`TempTurbineCoolantInStep` are
    # extracted from) are NOT registered: each is *itself* a genuine, still-unresolved
    # self-loop on its own -- `to_graph(PlantThermalEfficiency(...))` raises
    # `ValueError: reads [...], which it also owns` directly (confirmed this pass, not
    # merely asserted), since both own and read `eta_turbine`/`temp_turbine_coolant_in`
    # (`etath_liq`/`temp_turbine_coolant_in` for the second). They are superseded by the
    # three `*Step` `FixedPointFunction`s above for graph purposes, not usable
    # standalone.
    #
    # `Power.calculate_cryo_loads` (`_audit/boundary_inputs_audit.md` §7 item 7) is the
    # second wave of exactly that Shape-B gap, and it is now split the same way. Its
    # raw node `Cryo` stays NOT registered for the same reason
    # `PlantThermalEfficiency` does -- `to_graph(Cryo())` raises `ValueError: reads
    # ['.fwbs.qnuc'], which it also owns` on every one of its three arms, deliberately
    # (`Cryo`'s own docstring) -- and the three families below replace it for real use:
    #   * `CryoQNuc` owns `.fwbs.qnuc`, conditionally written by PROCESS under
    #     `inuclear == 0 and i_tf_sup == 1` ("Issue #511: if inuclear = 1: qnuc is
    #     input", `power.py:1825`) -- an occupant or `None`, not a `FixedPointFunction`,
    #     since which arm applies is known at `indat.py` assembly time;
    #   * `CryoQLoads` owns `.power.qss`/`qac`/`qcl`/`qmisc`, conditionally written
    #     under the *other* guard, `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING`
    #     (`power.py:1054-1057`), which is why the five fields are two families and not
    #     one -- see `CryoQNuc`'s docstring for the degeneracy argument;
    #   * `CryoLoads` owns the four fields written on every path
    #     (`.heat_transport.helpow`, `.p_cryo_plant_electric_mw`, `.helpow_cryal`,
    #     `.tfcoil.cryo_cool_req`) and reads the five `q*` as plain `FromExactly`s.
    # This closes `.heat_transport.helpow` (read by `Bldgs`, `CryogenicSystemCost`) and
    # `.heat_transport.p_cryo_plant_electric_mw` (read by `Acpow`,
    # `PlantElectricProductionReactor`, `AuxiliaryComponentCoolingCost`) as boundary
    # inputs. `inuclear=0`/`i_pf_conductor=0` are `fwbs_variables.py:81`/
    # `pfcoil_variables.py:230`'s defaults, neither set by `REFERENCE_INPUT_FILE`;
    # `i_tf_sup=1` is `tfcoil_variables.py:261`'s, likewise unset -- the same value the
    # rest of this file's TF-coil registrations already carry.
    cryo_q_nuc: CryoQNuc | None = dataclasses.field(kw_only=True)
    """`.fwbs.qnuc` -- **and the slot is empty when PROCESS takes it as an input.**
    `inuclear` was a static kwarg on a `FixedPointFunction` here.
    """

    cryo_q_loads: CryoQLoads | None = dataclasses.field(kw_only=True)
    """`.power.qss`/`qac`/`qcl`/`qmisc` -- **or nothing**, when PROCESS never calls
    `Power.cryo`.
    """

    cryo_loads: CryoLoads = dataclasses.field(kw_only=True)
    """The unconditionally-owned cryogenic loads -- one occupant per arm of the same two
    switches.
    """
    # `electric_production.py` (unit #14 chunk C). `i_pf_energy_storage_source=2`
    # matches `pf_power_variables.py:18`'s default.
    acpow: Acpow = dataclasses.field(kw_only=True)
    """Plant AC power requirement -- one occupant per
    `.pf_power.i_pf_energy_storage_source` value.
    """
