"""The cost model's namespace -- which cost nodes this machine has, and their
settings.

The nodes themselves are in `costs.py`; this module is the naming scope that groups
them, and it sits beside them for that reason (`model_tree_design.md` §11). It
carries no switch: which occupant fills a switched slot is `indat.py`'s answer,
never a subsystem's.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.costs.costs import (
    AtmosphericRecoveryCost,
    AuxiliaryComponentCoolingCost,
    AuxiliaryFacilityPowerCost,
    BlanketCost,
    ConstructedCost,
    ConvertFpyToCalendar,
    CostOfElectricity,
    CryogenicSystemCost,
    DieselGeneratorsCost,
    DivertorCost,
    ElectricPlantEquipmentCost,
    EnergyStorageCost,
    FirstWallCost,
    FuelHandlingCost,
    FuellingSystemCost,
    FuelProcessingCost,
    FusionPowerIslandCost,
    HeatRejectionCost,
    HeatTransportSystemCost,
    IndirectCosts,
    InstrumentationAndControlCost,
    LowVoltageCost,
    MagnetsCost,
    MaintenanceEquipmentCost,
    MiscPlantEquipmentCost,
    NuclearBuildingVentilationCost,
    PowerConditioningCost,
    PowerInjectionCost,
    ReactorCoolingSystemCost,
    ReactorCost,
    ShieldCost,
    StructuresCost,
    SwitchyardCost,
    TfCoilPowerConditioningCost,
    TfMagnetCostSuperconducting,
    TotalPlantDirectCost,
    TransformersCost,
    TurbinePlantEquipmentCost,
    VacuumSystemCost,
    VacuumVesselAssemblyCost,
)


class Costs(ModelNamespace):
    """`costs.py`'s 1990 cost model: the `.costs.i_cost_model == 0` arm, 43 nodes."""

    # Registry unit #18, `models/costs/costs.py` -- the whole `.costs.coe` chain, 42 of
    # `Costs`'s 43 computational methods (`acc2211`..`coelc` plus the two accumulations
    # `Costs.run()` performs inline). This is the `.costs.i_cost_model == 0`
    # (`CostModels.PROCESS_1990`) arm of the switch below; see there for why it is an
    # arm rather than an unconditional entry, and `costs.md`'s coverage map for
    # the per-method derivation of this list.
    #
    # Every static kwarg below is checked against the modelled run on every harness run
    # by `mda_harness.switch_audit`, so none of them can silently drift the way
    # `i_confinement_time`/`i_thermal_electric_conversion`/`i_p_coolant_pumping` did
    # (`next_steps.md` §8.2). Their values here are PROCESS's own bare defaults *and*
    # the reference run's, except `iohcl` -- flagged individually below.
    convert_fpy_to_calendar: ConvertFpyToCalendar = ConvertFpyToCalendar()
    structures_cost: StructuresCost = StructuresCost()  # Account 21
    # **`ife` was an `eqx.field(static=True)` on each of these six nodes and is gone**
    # (`_audit/next_steps.md` §14.2). Each has a real PROCESS `ife == 1` arm reading an
    # entirely different set of `.ife.*` fields (2-D material-mass arrays for the three
    # Account-221 nodes, driver-cost tables for Account 223, extra cooling loads for
    # 2262, a target-mass model for 2272); none of `.ife.*` is ported. The refusal moved
    # to `machine_from_indat`, which answers `ife` **once**, at assembly, for all six at
    # once -- rather than seven times, at trace time, inside seven node bodies. The
    # classes here are the magnetic-confinement occupants and no longer see the integer.
    first_wall_cost: FirstWallCost = FirstWallCost()  # Account 221.1
    blanket_cost: BlanketCost = BlanketCost()  # Account 221.2
    shield_cost: ShieldCost = ShieldCost()  # Account 221.3
    # Account 221.4 (reactor structure) has **no slot here, deliberately**: a
    # stellarator has no reactor-structure account to compute. `st_strc`
    # (`stellarator.py:334-337`) sets `.structure.fncmass` and `.structure.gsmass` to a
    # literal `0.0` with its own reason -- "many of the masses are simply set to zero to
    # avoid double-counting of structural components that are specified differently for
    # tokamaks" -- so `ReactorStructureCost` computed an exact zero, and landed on the
    # right number by luck rather than by modelling anything this device has. See the
    # note next to `pf_magnet_cost`'s former slot below for the full argument and what a
    # tokamak has to restore.
    divertor_cost: DivertorCost = DivertorCost()  # Account 221.5
    reactor_cost: ReactorCost = ReactorCost()  # Account 221 total
    # **`supercond_cost_model` was an `eqx.field(static=True)` here and is a slot now**
    # (`_audit/next_steps.md` §14.2). Its two arms read disjoint fields --
    # `.costs.ucsc` + `.tfcoil.m_tf_coil_superconductor` against `.costs.sc_mat_cost_0`
    # + `.tfcoil.j_crit_str_0` + `.tfcoil.j_crit_str_tf` -- so the single node declared
    # three edges the reference run does not make. `cost_variables.py:552`'s default is
    # `0`, `PER_KG`. **Only the superconducting arm
    # of `acc2221` is registered.** `TfMagnetCostResistive` is ported (same file) but
    # not registered: the two arms share no body and read disjoint fields, so they are
    # two nodes rather than one node with a static `i_tf_sup` kwarg, and pairing them as
    # a real `Switch` would need that switch to nest inside this one -- nested switches
    # are a still-open gap (`next_steps.md` §1). `.tfcoil.i_tf_sup == 1` is both
    # PROCESS's own default (`tfcoil_variables.py:261`) and the reference run's value,
    # so the registered arm is the right one under either; the `.tfcoil.i_tf_sup`
    # `Switch` already in this tuple would additionally have to gain a `costs.py` arm
    # before a resistive run could assemble, which is the honest statement of what is
    # missing.
    # Account 222.1
    tf_magnet_cost_superconducting: TfMagnetCostSuperconducting = dataclasses.field(
        kw_only=True
    )
    # **Accounts 222.2 (PF magnets) and 225.2 (PF coil power conditioning) have no slot
    # here, and 221.4 (reactor structure) above has none, deliberately. A stellarator
    # has no PF coil system and no separately-accounted reactor structure, so this tree
    # has no node for their costs.**
    #
    # The evidence is `_audit/cost_boundary_inputs.md` §§4-6, which measured all three:
    # every output of all three is exactly `0.0` in PROCESS's own converged reference
    # run, and their agreement in the MDA harness was vacuous (`trivial_agreements`).
    # They were not merely zero. `caller.py:272-275` returns before `pfcoil.run()` and
    # `Power.pfpwr` on a stellarator, so all twelve `.pf_coil.*` and all seven
    # `.pf_power.*` reads kept their dataclass defaults for the whole run; with
    # `n_cs_pf_coils = 0` both of `acc2222`'s loops unrolled to zero iterations, leaving
    # **21 of `PfMagnetCost`'s 27 declared reads dead**, not merely multiplied by zero
    # (§5.1). A node whose ports assert a dependence on a subsystem the device does not
    # have is the `EcrhDensityLimit` bug class this tree already names twice, and the
    # reason `WardTaylorAvailability` is deliberately unregistered: it computes a value
    # the configuration never computes, and lands on the right number by luck.
    #
    # Deleted outright rather than made a switched slot: there is no tokamak occupant to
    # switch *against*, and inventing a variant mechanism for a family with one member
    # and no alternative is exactly the paradigm-for-nothing this project declines
    # elsewhere. **When the tokamak arrives, `Costs` splits then** --
    # `_audit/cost_boundary_inputs.md`'s category (d) rows already record every producer
    # file:line the three nodes need (`pfcoil.py`'s `PFCoil.pfcoil`/`CSCoil.ohcalc`,
    # `power.py`'s `Power.pfpwr`, `structure.py:47-52`), so bringing them back is
    # mechanical, and §12 of that file says what to restore.
    #
    # What this costs the graph: `.costs.c2214`, `.costs.c2222` and `.costs.c2252`
    # become unowned boundary inputs, seeded from their `cost_variables.py` defaults of
    # `0.0` -- the same value the nodes produced -- so `.costs.c221`/`.c222`/`.c225`
    # and everything above them (`c22`, `c2`, `cdirt`, `concost`, `coe`) are unmoved,
    # measured rather than argued. Their fourteen sub-accounts (`c22221`-`c22224`,
    # `c22521`-`c22527`) had no reader in this graph at all and simply leave it.
    vacuum_vessel_assembly_cost: VacuumVesselAssemblyCost = (
        VacuumVesselAssemblyCost()
    )  # Account 222.3
    magnets_cost: MagnetsCost = MagnetsCost()  # Account 222 total
    power_injection_cost: PowerInjectionCost = PowerInjectionCost()  # Account 223
    vacuum_system_cost: VacuumSystemCost = VacuumSystemCost()  # Account 224
    tf_coil_power_conditioning_cost: TfCoilPowerConditioningCost = (
        TfCoilPowerConditioningCost()
    )  # Account 225.1
    # Account 225.2 (PF coil power conditioning) has no slot -- see the note above
    # `vacuum_vessel_assembly_cost`. **`energy_storage_cost` below is deliberately
    # *kept*** even though its outputs are zero too: it is zero because of
    # `i_pulsed_plant`, a switch, not because of a subsystem this device lacks. A pulsed
    # stellarator would want it, so it belongs to the switch-conversion work
    # (`model_tree_design.md` §8 step 6), not here.
    # `i_pulsed_plant`/`istore` were static kwargs here (`pulse_variables.py:30`/`:16`)
    # until the slot became a family. `istore == 3` stays unported (a third reads-set:
    # `.heat_transport.p_plant_primary_heat_mw`, `.times.t_plant_pulse_no_burn`,
    # `.pulse.dtstor`).
    energy_storage_cost: EnergyStorageCost = dataclasses.field(kw_only=True)
    """Account 225.3, and **the reads follow the arm.** At `i_pulsed_plant == 0` this
    account is identically zero and reads nothing; at `== 1` it scales an itemised
    literal by net electric power. One node carrying the switch declared both reads
    unconditionally, so the graph claimed a `.heat_transport -> .costs` edge that the
    reference run does not make. `large_tokamak_eval.IN.DAT` sets `i_pulsed_plant = 1`,
    which is why this is one of the four values `_audit/tokamak_scope.md` found the tree
    contradicting."""
    power_conditioning_cost: PowerConditioningCost = (
        PowerConditioningCost()
    )  # Account 225 total
    reactor_cooling_system_cost: ReactorCoolingSystemCost = (
        ReactorCoolingSystemCost()
    )  # Account 2261
    auxiliary_component_cooling_cost: AuxiliaryComponentCoolingCost = (
        AuxiliaryComponentCoolingCost()
    )  # Account 2262
    cryogenic_system_cost: CryogenicSystemCost = CryogenicSystemCost()  # Account 2263
    heat_transport_system_cost: HeatTransportSystemCost = (
        HeatTransportSystemCost()
    )  # Account 226 total
    fuelling_system_cost: FuellingSystemCost = FuellingSystemCost()  # Account 2271
    # Account 2272 -- also the sole producer of
    # `.physics.wtgpd`, the one field `costs.py` writes outside `.costs.*`.
    fuel_processing_cost: FuelProcessingCost = FuelProcessingCost()
    atmospheric_recovery_cost: AtmosphericRecoveryCost = (
        AtmosphericRecoveryCost()
    )  # Account 2273
    nuclear_building_ventilation_cost: NuclearBuildingVentilationCost = (
        NuclearBuildingVentilationCost()
    )  # Account 2274
    fuel_handling_cost: FuelHandlingCost = FuelHandlingCost()  # Account 227 total
    instrumentation_and_control_cost: InstrumentationAndControlCost = (
        InstrumentationAndControlCost()
    )  # Account 228
    maintenance_equipment_cost: MaintenanceEquipmentCost = (
        MaintenanceEquipmentCost()
    )  # Account 229
    fusion_power_island_cost: FusionPowerIslandCost = (
        FusionPowerIslandCost()
    )  # Account 22 total
    turbine_plant_equipment_cost: TurbinePlantEquipmentCost = (
        TurbinePlantEquipmentCost()
    )  # Account 23
    switchyard_cost: SwitchyardCost = SwitchyardCost()  # Account 241
    transformers_cost: TransformersCost = TransformersCost()  # Account 242
    low_voltage_cost: LowVoltageCost = LowVoltageCost()  # Account 243
    diesel_generators_cost: DieselGeneratorsCost = DieselGeneratorsCost()  # Account 244
    auxiliary_facility_power_cost: AuxiliaryFacilityPowerCost = (
        AuxiliaryFacilityPowerCost()
    )  # Account 245
    electric_plant_equipment_cost: ElectricPlantEquipmentCost = (
        ElectricPlantEquipmentCost()
    )  # Account 24 total
    misc_plant_equipment_cost: MiscPlantEquipmentCost = (
        MiscPlantEquipmentCost()
    )  # Account 25
    heat_rejection_cost: HeatRejectionCost = HeatRejectionCost()  # Account 26
    total_plant_direct_cost: TotalPlantDirectCost = (
        TotalPlantDirectCost()
    )  # `.costs.cdirt`, inline in `Costs.run()`
    indirect_costs: IndirectCosts = IndirectCosts()  # Account 9
    constructed_cost: ConstructedCost = (
        ConstructedCost()
    )  # `.costs.concost`, inline in `Costs.run()`
    cost_of_electricity: CostOfElectricity | None = dataclasses.field(kw_only=True)
    """Whether this run has a cost of electricity at all, and if so which centrepost
    treatment -- `.costs.ireactor`, `.costs.ipnet` and `.physics.itart` jointly
    (`cost_variables.py:521`/`:515`, `physics_variables.py:994`; defaults `1`, `0`, `0`).
    `.costs.coe`, `coecap`, `coeoam`, `coefuelt`, `moneyint`, `capcost`.

    **A node-existence condition, not a branch, and now spelled as one.**
    `Costs.run()` calls `coelc()` only when `ireactor == 1 and ipnet == 0`
    (`costs.py:82-83`); on any other pair PROCESS leaves all six fields at whatever they
    already held, and `CostOfElectricity.__check_init__` says so outright -- *"this node
    must not exist"*. It existed anyway: `ireactor` drove the
    `availability.electric_production` slot while this constructor kwarg said
    `CALCULATED` regardless, so `ireactor = 0` assembled `PowerProfilesOverTime` (which
    computes no `.heat_transport.p_plant_electric_net_mw`) *and* a `CostOfElectricity`
    reading it. One switch, two answers.

    **`None` is the occupant of the other arm, and that is what `None` is for.** Step 4b
    removed all four `| None`s, two as unreachable and two as configurations this port
    cannot honestly assemble -- none of the four was a case where *PROCESS itself*
    computes nothing. This one is: at `ireactor == 0` the six fields simply keep their
    entering values, which is exactly what an absent occupant means (cottax:
    *"an unproduced slot: it assembles nothing, and whatever read its outputs surfaces
    as a boundary input. Absence, spelled as absence."*). Refusing the value instead
    would have made a ported, registered occupant -- `PowerProfilesOverTime` --
    unreachable, which is the defect step 4c had just finished removing from
    `BlanketShieldPowerExponential`.

    **`ireactor`, `ipnet`, `ife` and `itart` were all static kwargs on the occupant and
    none is now** (`_audit/next_steps.md` §14.2). The first two are what select this
    slot's arm, so restating them on the occupant was a second answer to a question the
    slot had already answered; `ife` is refused once at assembly for all seven
    Account-22x nodes. The fourth, `itart`, was the one that cost the graph something:
    `costs.py:2769-2783`'s centrepost replacement cost exists only at `itart == 1`, so a
    single node carrying the switch read `.costs.cplife_cal`, `.costs.cpstcst` and
    `.costs.cplife` on a machine that reads none of them -- and `.costs.cplife` is owned
    by a `FixedPoint` that is the identity map here. It is a third arm of this slot now,
    not a kwarg, and the previous note's "both `itart` arms are implemented in one
    function ... a deliberate size-aware deviation" is withdrawn with the policy that
    allowed it.
    """
