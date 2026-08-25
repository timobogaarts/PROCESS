"""The running graph assembly of every ported stellarator unit.

Imports each ported unit's `cottax` node declaration and assembles them into one
`Graph` via `to_graph`. Run directly for a smoke check (builds the graph, prints its
node/variable count); `render_xdsm.py` imports `GRAPH` from here to draw it.

This is the whole graph **as it currently exists**, not a claim that the stellarator MDA
is assembled: most nodes here are still islands with unowned (external) reads, since
their producers haven't been ported yet. It exists so there is always one place the next
ported unit joins, and one place to point a visual inspection at. See
`_audit/unit_registry.md`'s "Ported so far" for what is and isn't in it.

**There is no single graph, and the tree is what says so.** A topology-changing switch
selects which nodes exist, so what this module exports is a `StellaratorProcess` -- a
tree of typed slots, each holding the model that fills it -- and `to_graph(machine)`.
`GRAPH` is `REFERENCE_MACHINE`'s, kept as a module-level name because `render_xdsm.py`
and the smoke check want a default to point at. `machine_from_indat` is the only place
an `i_*` integer is read, and its docstring explains why assembly time is the only
correct place to resolve one (short version: no switch in PROCESS is ever an iteration
variable or a scan variable, so no switch can change between two evaluations of one
assembled graph).

**The tree itself carries no configuration.** Every slot the factory fills has *no
default*, so `StellaratorProcess()` raises `TypeError` rather than quietly producing one
particular machine; a default is admissible only where there is nothing to decide. That
is the property that would have caught the `i_confinement_time` 34-vs-38 and
`i_plasma_ignited` 0-vs-1 registration bugs, both of which lived in a slot default where
nothing was looking.

`EcrhDensityLimit(i_plasma_pedestal=PlasmaProfileShapeType.PARABOLIC_PROFILE)` keeps its
setting as a static kwarg rather than becoming a slot of its own. It is `naming_convention.md`'s other category -- a
formula-changing switch kept as a static kwarg on one node's `fn` -- because
`i_plasma_pedestal != PARABOLIC_PROFILE` has no formula at all in `density_limits.py`
and no node's existence depends on it.

**Every such static kwarg is enum-typed**, per `_audit/model_tree_design.md` §4
("Settings stay on the occupant, enum-typed"): the upstream `IntEnum` where PROCESS
declares one, and `functional_process/models/switch_enums.py`'s minimal local
definition where it does not. `IntEnum` members compare and hash equal to their `int`
values, so nothing numeric moves -- what moves is that `PROCESS_1990` cannot typo into
`KOVARI_2014` the way `0` typos into `1`, the defect class with five recorded instances
(`_audit/switch_elimination_design.md` §5(A)). The bare integers below are exactly the
two categories that are *not* switches: shape/resolution counts
(`n_plasma_profile_elements`, `n_cs_pf_coils`) and set membership (`imp_indices`),
§3(b)/(c) of that same document.

**Update, later pass**: `coils/coils.py`'s `intersect` is now registered, as `Intersect`
(an `ImplicitFunction`/`RootFind` pair) -- see below, near `WindingPackIntersectInputs`/
`WindingPackTotalSizePost`. The old `WindingPackJTfWp`/monolithic-
`winding_pack_total_size` registration this paragraph used to describe is gone; see that
registration's own comment
for the replacement and why the whole `j_tf_wp` self-loop turned out not to need a
`FixedPointFunction` at all.

Still not included despite being ported: `coils/calculate.py`'s `st_coil` itself -- a
real tier-3 orchestrator, not self-contained, still audit-only.
`physics/superconductors.py` (unit #22) and `physics/impurity_radiation.py`'s two leaf
functions (unit #23) -- every real call site's arguments are locals inside a
not-yet-wired unit (`plasma_composition`, itself registered below, but its
impurity-array locals stay per-index minted paths, not a whole-array edge into
`impurity_radiation.py`'s own leaves). `coils/coils.py`'s `jcrit_from_material` (unit
#10, an 8-way switch on `i_tf_sc_mat`, one `ExplicitFunction` node per branch) --
**investigated and still not registered, for a structural reason, not an oversight**:
its `FromExactly`s (`.tfcoil.t_helium`/`b_max`) are per-sample locals of
`winding_pack_curves`'s 200-point sampling loop (`b_max = b_max_k[k]`, an array, not a
scalar), and PROCESS has exactly one real call site for the whole dispatch, inside that
same sampling loop (confirmed: `grep`ing `process/models/stellarator/coils/calculate.py`
for `jcrit_from_material` finds only its own `jcrit_vector[k] = jcrit_from_material(...)`
per-sample assignment) -- there is no single-point scalar call site for these 8 nodes to
bind to as written, so registering them would assert a wiring that does not exist in
PROCESS, not merely one this pass hasn't gotten to yet. `winding_pack_curves` keeps its
own internal `_critical_current_density_by_material` dispatcher rather than calling
these 8 nodes' underlying functions directly, for the same reason `calculate.md`
documents: it deliberately diverges from `coils.py`'s own `jcrit_from_material` on the
REBCO branch (`coils.py` reproduces a real PROCESS call-site bug there; `calculate.py`'s
local copy sidesteps it so this port has *a* working REBCO branch) -- collapsing the two
would either regress REBCO or stop reproducing the bug faithfully, so they stay two
independent implementations, documented as such, not one deduplicated further.

`blankets/hcpb.py` (unit #13) is ported (3/3 in-scope functions, 3 `ExplicitFunction`
nodes) but **deliberately still not registered here**: all three are only ever called
from `stellarator.py`'s `blanket_neutronics()`, itself only reached under
`.fwbs.blktmodel == 1` (S2 of the `st_fwbs` synthesis, `stellarator_E_fwbs_synthesis.md`,
`next_steps.md` §3) -- the `blktmodel == 1` arm's own live call-site bug
(`blanket_neutronics()` calls two `hcpb.py` `@staticmethod`s with zero arguments) blocks
it, not a registration decision. **The other two of S2's three arms are now wired**, as
a joint `Switch` below (`ExponentialAttenuationBlanketShieldPower`/
`DetailedPowerflowBlanketShieldPower`) -- see that `Switch`'s own docstring for why a
single joint switch over `.fwbs.blktmodel`/`.heat_transport.ipowerflow` together, not a
plain `.fwbs.blktmodel` switch, is what the real 3-arm dispatch needs, and for the
`ScTfCoilNuclearHeating` registration bug this wiring found and fixed along the way
(unconditional registration was wrong for PROCESS's own default configuration, same
of bug already fixed once for `EcrhDensityLimit`).
"""

import dataclasses
import functools
import re
from pathlib import Path

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ModelNamespace, to_graph

from functional_process.models.availability import Avail, CplifeAvail
from functional_process.models.buildings import Bldgs, BldgsSizes, TfCoilEnvelope
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
from functional_process.models.physics.composition import (
    CalculateEffectiveChargeIonisationProfiles,
    PlasmaComposition,
)
from functional_process.models.physics.confinement_time import (
    ConfinementTime,
    DoubleAndTripleProduct,
    StellaratorConfinementTime,
)
from functional_process.models.physics.dimensionless_parameters import (
    DimensionlessPlasmaParameters,
)
from functional_process.models.physics.exhaust import RadiationFraction
from functional_process.models.physics.fusion_reactions import (
    FusionRates,
    SetFusionPowers,
)
from functional_process.models.physics.plasma_profiles import (
    IonVolAvgTemperature,
    LModeProfileReset,
    ParabolicGradientLengths,
    ParabolicProfileValues,
    ProfileFactors,
)
from functional_process.models.physics.profiles import (
    DensityProfile,
    NeProfileIntegral,
    ParabolicOnAxisDensities,
    ParabolicOnAxisTemperatures,
    ParabolicTemperatureProfile,
    PedestalOnAxisDensities,
    PedestalOnAxisTemperatures,
    PedestalTemperatureProfile,
    ProfileGrid,
    TeProfileIntegral,
)
from functional_process.models.physics.pure_formulas import (
    AuxiliaryPhysicsQuantities,
    ElectronThermalEnergy,
    FastAlphaBeta,
    IonElectronEquilibration,
    IonThermalEnergy,
    TotalPlasmaHeatingPower,
)
from functional_process.models.physics.radiation_power import (
    ImpurityRadiationTotals,
    PlasmaRadiationPowers,
    SynchrotronRadiationPower,
)
from functional_process.models.power.electric_production import (
    Acpow,
    PlantElectricProductionReactor,
    PowerProfilesOverTime,
)
from functional_process.models.power.tf_coil_power import (
    TfPowerResistive,
    TfPowerSuperconducting,
)
from functional_process.models.power.thermal_cryo import (
    ComponentThermalPowers,
    CryoLoads,
    CryoQLoadsStep,
    CryoQNucStep,
    DeltaEtaStep,
    EtathLiqStep,
    EtaTurbineStep,
    PFwBlktCoolantPumpMwStep,
    PFwDivHeatDepositedMwStep,
    TempTurbineCoolantInStep,
)
from functional_process.models.stellarator.build import (
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
    # `BlktmodelBlanketThickness` deliberately NOT imported/registered here any more --
    # see the comment next to `Build` below, and `unit_registry.md` row 2: PROCESS's
    # own default is `blktmodel = 0`, under which this node's own docstring says it must
    # not be instantiated at all (`conditional-ownership-by-run-config`, same shape as
    # `.physics.aspect`). Unconditional registration was a real bug, the same class
    # already fixed once for `EcrhDensityLimit` -- found and fixed this pass.
    Build,
)
from functional_process.models.stellarator.coils.calculate import (
    CoilCasing,
    CoilCoilToroidalGap,
    CoilCrossSectionalArea,
    CoilCurrent,
    CoilHalfWidths,
    CoilRadialThickness,
    CoilsSummaryVariables,
    CoilToroidalThickness,
    HorizontalPorts,
    LenTfCoil,
    PlasmaFacingCoilArea,
    StoredMagneticEnergy,
    TfCryoArea,
    VerticalPorts,
    WindingPackGeometry,
    WindingPackIntersectInputs,
    WindingPackTotalSizePost,
    ZTfInsideHalf,
)
from functional_process.models.stellarator.coils.coils import Intersect
from functional_process.models.stellarator.coils.forces import (
    MaxForceDensity,
    MaximumStress,
)
from functional_process.models.stellarator.coils.mass import CoilsMass
from functional_process.models.stellarator.coils.quench import QuenchProtection
from functional_process.models.stellarator.density_limits import (
    EcrhDensityLimit,
    SudoDensityLimit,
)
from functional_process.models.stellarator.divertor import Divertor
from functional_process.models.stellarator.geometry import (
    DefaultAspectRatio,
    StellaratorPlasmaGeometry,
    StellaratorScalingFactors,
)
from functional_process.models.stellarator.heating import (
    BeamCurrent,
    EcrhHeating,
    FusionGain,
    InjectedPowerTotal,
    LowhybHeating,
)
from functional_process.models.stellarator.initialization import PulseDurations
from functional_process.models.stellarator.neoclassics import (
    EffectiveThermalDiffusivity,
    ProfileValues,
)
from functional_process.models.stellarator.plasma_physics import (
    ClippedRadiationPowers,
    FusionPowerTotalsMw,
    FusionTotalsNoBeam,
    HeatingAndRadiationPower,
    NeutronWallLoad,
    PoloidalFieldFromRotationalTransform,
    RadiatedWallLoadAndFraction,
    StellaratorBetaAndStoredEnergy,
    ThermalEnergyTotals,
    TotalField,
)
from functional_process.models.stellarator.preset_config import (
    StellaratorMachineConfig,
    read_stellarator_config_file,
)
from functional_process.models.stellarator.stellarator_fwbs_s1_s5 import (
    CryostatAndVvGeometry,
    FwBlanketShieldGeometry,
)
from functional_process.models.stellarator.stellarator_fwbs_s2 import (
    DetailedPowerflowBlanketShieldPower,
    ExponentialAttenuationBlanketShieldPower,
)
from functional_process.models.stellarator.stellarator_fwbs_s3 import DivertorPlateMass
from functional_process.models.stellarator.stellarator_fwbs_s4 import (
    BlanketComponentMasses,
    ShieldMass,
)
from functional_process.models.stellarator.structure import (
    StructureMasses,
)
from functional_process.models.stellarator.tf_nuclear_heating import (
    ScTfCoilNuclearHeating,
)
from functional_process.models.switch_enums import (
    BlanketDualCoolantModel,
    BlanketLifetimeModel,
    CoilNuclearHeatingModel,
    CostOfElectricityModel,
    FastAlphaPressureModel,
    IFEModel,
    NetElectricPowerModel,
    NeutronWallLoadModel,
    PFEnergyStorageSource,
    PlantOperationModel,
    PowerFlowModel,
    SphericalTokamakModel,
    SuperconductorCostModel,
    ThermalStorageModel,
)
from functional_process.models.vacuum import DuctDiameterRootFind, VacuumOld
from process.data_structure.blanket_variables import BlktModelTypes
from process.data_structure.pfcoil_variables import PFConductorModel
from process.data_structure.physics_variables import (
    ConfinementRadiationLossModel,
    ConfinementTimeModel,
    PlasmaIgnitionModel,
)
from process.models.physics.current_drive import CurrentDriveModel
from process.models.physics.profiles import PlasmaProfileShapeType
from process.models.power import ElectricConversionModelTypes, PumpingPowerModelTypes
from process.models.superconductors import SuperconductorModel
from process.models.tfcoil.base import TFConductorModel


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
    # `ife=0` (`ife_variables.py:253`). Each of these five nodes has a real PROCESS
    # `ife == 1` arm that reads an entirely different set of `.ife.*` fields (2-D
    # material-mass arrays for the three Account-221 nodes, driver-cost tables for
    # Account 223, extra cooling loads for 2262, a target-mass model for 2272); none of
    # `.ife.*` is ported, so the ported functions refuse that value rather than
    # returning a magnetic-confinement number for an IFE device.
    first_wall_cost: FirstWallCost = FirstWallCost(
        ife=IFEModel.MAGNETIC_CONFINEMENT
    )  # Account 221.1
    blanket_cost: BlanketCost = BlanketCost(
        ife=IFEModel.MAGNETIC_CONFINEMENT
    )  # Account 221.2
    shield_cost: ShieldCost = ShieldCost(
        ife=IFEModel.MAGNETIC_CONFINEMENT
    )  # Account 221.3
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
    # `supercond_cost_model=0` (`cost_variables.py:552`). **Only the superconducting arm
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
    tf_magnet_cost_superconducting: TfMagnetCostSuperconducting = (
        TfMagnetCostSuperconducting(supercond_cost_model=SuperconductorCostModel.PER_KG)
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
    power_injection_cost: PowerInjectionCost = PowerInjectionCost(
        ife=IFEModel.MAGNETIC_CONFINEMENT
    )  # Account 223
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
    # `i_pulsed_plant=0`/`istore=1` (`pulse_variables.py:30`/`:16`). `istore == 3` is
    # unported (a third reads-set: `.heat_transport.p_plant_primary_heat_mw`,
    # `.times.t_plant_pulse_no_burn`, `.pulse.dtstor`) and unreachable here, since a
    # steady-state plant never enters the `istore` dispatch at all.
    energy_storage_cost: EnergyStorageCost = EnergyStorageCost(
        i_pulsed_plant=PlantOperationModel.CONTINUOUS,
        istore=ThermalStorageModel.ELECTROWATT_OPTION_1,
    )  # Account 225.3
    power_conditioning_cost: PowerConditioningCost = (
        PowerConditioningCost()
    )  # Account 225 total
    reactor_cooling_system_cost: ReactorCoolingSystemCost = (
        ReactorCoolingSystemCost()
    )  # Account 2261
    auxiliary_component_cooling_cost: AuxiliaryComponentCoolingCost = (
        AuxiliaryComponentCoolingCost(ife=IFEModel.MAGNETIC_CONFINEMENT)
    )  # Account 2262
    cryogenic_system_cost: CryogenicSystemCost = CryogenicSystemCost()  # Account 2263
    heat_transport_system_cost: HeatTransportSystemCost = (
        HeatTransportSystemCost()
    )  # Account 226 total
    fuelling_system_cost: FuellingSystemCost = FuellingSystemCost()  # Account 2271
    # Account 2272 -- also the sole producer of
    fuel_processing_cost: FuelProcessingCost = FuelProcessingCost(
        ife=IFEModel.MAGNETIC_CONFINEMENT
    )
    # `.physics.wtgpd`, the one field `costs.py` writes outside `.costs.*`.
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
    # `ireactor=1`/`ipnet=0` (`cost_variables.py:521`/`:515`) are preconditions, not
    # ports: `Costs.run()` calls `coelc()` only when both hold (`costs.py:82-83`), and
    # the node refuses any other pair rather than producing a `.costs.coe` PROCESS would
    # have left at its previous value. `itart=0` (`physics_variables.py:994`) selects
    # the no-centrepost arm; unlike the `ife`/`i_tf_sup` splits above, *both* `itart`
    # arms are implemented in one function, so this node reads
    # `.costs.cpstcst`/`cplife_cal`/`cplife` either way -- a deliberate size-aware
    # deviation from `traceability_policy.md`'s split default, argued in the function's
    # own docstring.
    cost_of_electricity: CostOfElectricity = CostOfElectricity(
        ife=IFEModel.MAGNETIC_CONFINEMENT,
        itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
        ireactor=CostOfElectricityModel.CALCULATED,
        ipnet=NetElectricPowerModel.SCALED_POSITIVE,
    )  # `.costs.coe`


REFERENCE_STELLA_CONF = (
    Path(__file__).resolve().parent.parent
    / "tests/regression/input_files/stellarator_helias.stella_conf.json"
)
"""`REFERENCE_INPUT_FILE`'s `istell == 6` machine-config companion.

`Stellarator.st_new_config()` opens `f"{data.globals.output_prefix}stella_conf.json"`
before anything else runs, so for the reference run this file *is* the machine being
designed. Read here, at assembly time, and handed to `StellaratorMachineConfig` as static
data -- the whole point of unit #8's shape decision (`preset_config.md`): `istell == 6`'s
file I/O is a `non-traceable-external-call` that never has to enter a traced body,
because which machine is being designed cannot change during a solve.

Named next to `TOPOLOGY_SWITCHES` rather than beside `REFERENCE_INPUT_FILE` below only
because the `.stellarator.istell` switch needs it; the two must stay companions (same
stem, same directory), which is what PROCESS's own `output_prefix` convention enforces
for a real run."""


class ProfileParameterisationParabolic(ModelNamespace):
    """Parabolic profiles: the `.physics.i_plasma_pedestal == 0` occupant, 7 nodes."""

    # Not just a formula switch: `density_limits.py:146-150` shows
    # PROCESS itself has no formula for `st_d_limit_ecrh` outside
    # `i_plasma_pedestal == 0` -- the `else` arm only logs an error and
    # produces no defined `dlimit_ecrh`/`bt_max_ecrh`. So the value-0
    # requirement `EcrhDensityLimit` already enforces internally
    # (`density_limits.py`'s `calculate_ecrh_density_limit` raises
    # otherwise) is not a stricter precondition than this arm's own --
    # co-locating the static kwarg next to the switch value that selects
    # it is what next_steps.md's "one source of truth" proposal reduces
    # to for this single instance, with no extra machinery needed: there
    # is only one place `i_plasma_pedestal=0` is written now.
    ecrh_density_limit: EcrhDensityLimit = EcrhDensityLimit(
        i_plasma_pedestal=PlasmaProfileShapeType.PARABOLIC_PROFILE
    )
    parabolic_temperature_profile: ParabolicTemperatureProfile = (
        ParabolicTemperatureProfile()
    )
    parabolic_on_axis_densities: ParabolicOnAxisDensities = ParabolicOnAxisDensities()
    parabolic_on_axis_temperatures: ParabolicOnAxisTemperatures = (
        ParabolicOnAxisTemperatures()
    )
    parabolic_gradient_lengths: ParabolicGradientLengths = ParabolicGradientLengths()
    # `ParabolicProfileValues` -- the line-average/density-weighted tail
    # of `parabolic_parameterisation`. `plasma_profiles.md`'s "cottax
    # node" section deferred it pending that record's open question 1
    # ("`i_plasma_pedestal` holds two different switch roles"); the
    # comment above on `EcrhDensityLimit(i_plasma_pedestal=0)` is that
    # question's answer for this switch's only instance, so the blocker
    # is gone. Without it `.physics.temp_plasma_electron_density_
    # weighted_kev`/`temp_plasma_ion_density_weighted_kev` were boundary
    # inputs and iteration variable 4 had no path into fusion
    # reactivity or beta -- see that class's docstring.
    parabolic_profile_values: ParabolicProfileValues = ParabolicProfileValues()
    # The seven-field L-mode reset (`plasma_profiles.py:92-117`), the
    # producer these fields never had. Belongs on this arm and only this
    # arm: PROCESS applies it in `parabolic_parameterisation`, i.e.
    # exactly when `i_plasma_pedestal == 0`. Without it a cold run
    # carries the input file's `nd_plasma_pedestal_electron`/
    # `nd_plasma_separatrix_electron` into `DensityProfile`, whose one
    # formula is the pedestal one and only degenerates to the parabolic
    # profile once they are zero -- so a parabolic run silently got a
    # pedestal density profile. See `LModeProfileReset`'s docstring for
    # the measured effect on the SAND cold solve.
    l_mode_profile_reset: LModeProfileReset = LModeProfileReset()


class ProfileParameterisationPedestal(ModelNamespace):
    """Pedestal profiles: the `.physics.i_plasma_pedestal == 1` occupant, 3 nodes."""

    pedestal_temperature_profile: PedestalTemperatureProfile = (
        PedestalTemperatureProfile()
    )
    pedestal_on_axis_densities: PedestalOnAxisDensities = PedestalOnAxisDensities()
    pedestal_on_axis_temperatures: PedestalOnAxisTemperatures = (
        PedestalOnAxisTemperatures()
    )
    # No pedestal-arm counterpart to `EcrhDensityLimit`: PROCESS's own
    # default configuration (`i_plasma_pedestal == 1`) never computes a
    # real ECRH density limit at all -- see the note on the value-0 arm.
    # `dlimit_ecrh`/`bt_max_ecrh` are therefore genuinely unproduced in
    # this arm, not merely unported.


class BlanketShieldPowerExponential(ModelNamespace):
    """Exponential-attenuation blanket/shield power: `blktmodel == 0 & ipowerflow == 0`.

    That is arm **1** of `_blanket_shield_power_arm`, `st_fwbs`'s
    `stellarator.py:683-729`. This docstring used to say "the `blktmodel == 1`
    occupant", which was simply wrong -- `blktmodel == 1` is `blanket_neutronics()`,
    the arm that has no occupant at all (arm 0, `UNPORTED`). The mislabelling and the
    inverted key derivation it belonged to were fixed together; see
    `_blanket_shield_power_arm`.
    """

    # Over the line length and left that way -- see `Physics`'s own note: the slot
    # name and the occupant class are both this long and `ruff format` strips
    # parentheses from around an annotation.
    exponential_attenuation_blanket_shield_power: ExponentialAttenuationBlanketShieldPower = ExponentialAttenuationBlanketShieldPower()  # noqa: E501
    # `ScTfCoilNuclearHeating` moved here from the unswitched part -- this arm is
    # its one genuine caller that keeps its `p_tf_nuclear_heat_mw`
    # output (`stellarator.py:727-728`); the `blktmodel == 1` arm calls
    # it too but discards that particular output (`stellarator.py:465-476`
    # unpacks nine `_`s and keeps only `flu_tf_neutron_fast_peak`), and arm 2
    # computes its own, different `p_tf_nuclear_heat_mw` formula.
    # Leaving it unconditional was a real bug, the same
    # class already fixed once for `EcrhDensityLimit`: PROCESS's actual
    # default configuration lands in arm 2, not this one, so the
    # old unconditional placement was computing SC-coil TF nuclear
    # heating via the wrong formula for the default `GRAPH`.
    sc_tf_coil_nuclear_heating: ScTfCoilNuclearHeating = ScTfCoilNuclearHeating()


class StellaratorCoils(ModelNamespace):
    """The modular-coil set: geometry, current, casing, ports, structure, cryogenics.

    A third level because a real SCC lives here (`model_tree_design.md` §4's criterion
    for a sub-namespace), not because `coils/calculate.py` is one file.
    """

    # unit #9, coils/calculate.py
    coil_toroidal_thickness: CoilToroidalThickness = CoilToroidalThickness()
    coil_radial_thickness: CoilRadialThickness = CoilRadialThickness()
    coil_cross_sectional_area: CoilCrossSectionalArea = CoilCrossSectionalArea()
    coil_half_widths: CoilHalfWidths = CoilHalfWidths()
    plasma_facing_coil_area: PlasmaFacingCoilArea = PlasmaFacingCoilArea()
    coil_coil_toroidal_gap: CoilCoilToroidalGap = CoilCoilToroidalGap()
    coils_summary_variables: CoilsSummaryVariables = CoilsSummaryVariables()
    stored_magnetic_energy: StoredMagneticEnergy = StoredMagneticEnergy()
    winding_pack_geometry: WindingPackGeometry = WindingPackGeometry()
    coil_current: CoilCurrent = CoilCurrent()
    coil_casing: CoilCasing = CoilCasing()
    vertical_ports: VerticalPorts = VerticalPorts()
    horizontal_ports: HorizontalPorts = HorizontalPorts()
    # `st_coil`'s formula for `.build.z_tf_inside_half` -- see `Build`'s own comment
    # above (unit #2, build.py) for why this one, not `Build`'s, owns the field.
    z_tf_inside_half: ZTfInsideHalf = ZTfInsideHalf()
    # `.tfcoil.tfcryoarea`, carved out of the same inline `st_coil` geometry block as
    # `ZTfInsideHalf` and for the same reason (the eager `st_coil` orchestrator is not
    # registered, so anything only it computes has no owner). Prerequisite for
    # `CryoQLoadsStep` below: without it, registering the cryo nodes would have traded
    # two boundary inputs for one new one (`_audit/boundary_inputs_audit.md` §4c (c1)'s
    # sibling gap, §7 items 4 and 7). Of its two neighbours in that block,
    # `min_bending_radius` still stays unported for want of any reader.
    tf_cryo_area: TfCryoArea = TfCryoArea()
    # `.tfcoil.len_tf_coil`, the third formula from that same block, and the one held
    # back longest: four registered nodes read it (`StructureMasses`,
    # `PlasmaFacingCoilArea`, `CoilsMass`, `TfMagnetCostSuperconducting`) while it had
    # no producer at all. **The stale-vs-fresh decision it was waiting on is resolved,
    # in favour of binding fresh** -- there is no feedback path back into it (measured:
    # its producer's owned input comes from `StellaratorScalingFactors`, unreachable
    # from all four readers), so a `FixedPointFunction` self-loop modelling PROCESS's
    # read-before-write would be a degenerate fixed point that
    # `degenerate_fixed_points` deletes on sight. See `LenTfCoil`'s own docstring for
    # the full argument and the one honest caveat. This also closes the cold-start
    # `nan` in `.costs.c22211`/`.c2221` (`costs.md`).
    len_tf_coil: LenTfCoil = LenTfCoil()
    # unit #12, coils/mass.py
    coils_mass: CoilsMass = CoilsMass()
    # unit #11, coils/forces.py
    max_force_density: MaxForceDensity = MaxForceDensity()
    maximum_stress: MaximumStress = MaximumStress()
    # unit #14, coils/quench.py
    quench_protection: QuenchProtection = QuenchProtection()
    # `coils/calculate.py`'s `winding_pack_total_size` (unit #9's remaining tier-2 gap),
    # now the full three-piece split: `WindingPackIntersectInputs` (pre-`intersect`,
    # mints `.stellarator.wp_width_r`/`.lhs`/`.rhs`), `coils.py`'s `Intersect`
    # (`ImplicitFunction`/`RootFind`, owns `.stellarator.wp_width_r_min`),
    # `WindingPackTotalSizePost` (post-`intersect`, owns `.tfcoil.j_tf_wp` along with
    # everything else `winding_pack_post_intersect` computes). `i_tf_sc_mat=1` matches
    # `tfcoil_variables.py:246`'s default (ITER Nb3Sn).
    #
    # **An earlier pass registered `WindingPackJTfWp` here instead** -- a
    # `FixedPointFunction` that isolated just `.tfcoil.j_tf_wp` by re-running the whole
    # (unchanged) `winding_pack_total_size` pure function internally, duplicating the
    # 200-point sampling and eager `intersect` call the three-piece split next to it
    # already did, unregistered. That duplication is why it is gone: once
    # `WindingPackTotalSizePost` owns `.tfcoil.j_tf_wp` (it used to discard the fresh
    # value, deferring ownership to `WindingPackJTfWp`) and `WindingPackIntersectInputs`
    # reads it (already did), the self-reference closes through the three real nodes
    # plus `Intersect`'s own `RootFind` problem -- one merged 4-node SCC, the same
    # "Shape A" cross-node-cycle shape as `Divertor`/`AFwTotalWithPowerflow` below, not a
    # self-loop on any single node. `Blocking`/`to_graph()` finds it with no
    # `FixedPointFunction`/`Cut` wrapper at all (confirmed:
    # `test_winding_pack_intersect_split_forms_one_combined_cycle` in
    # `test_calculate.py`).
    # `Intersect` (like every other undriven declared node in this graph) needs no
    # production driver to be registered here -- structural admission only, driving
    # deferred, per `_audit/next_steps.md` §5.
    winding_pack_intersect_inputs: WindingPackIntersectInputs = (
        WindingPackIntersectInputs(i_tf_sc_mat=SuperconductorModel.ITER_NB3SN)
    )
    intersect: Intersect = Intersect()
    winding_pack_total_size_post: WindingPackTotalSizePost = WindingPackTotalSizePost()


class StellaratorFwbs(ModelNamespace):
    """First wall, blanket and shield -- the `st_fwbs` chunk's registered nodes."""

    blanket_shield_power: (
        BlanketShieldPowerExponential | DetailedPowerflowBlanketShieldPower
    ) = dataclasses.field(kw_only=True)
    """Blanket/shield power deposition, on `.fwbs.blktmodel` x `.heat_transport.
    ipowerflow` jointly -- one slot, two integers, resolved by
    `_blanket_shield_power_arm` in `machine_from_indat`.

    A **ragged** family, which is allowed and deliberate: the arm-1 occupant
    (`blktmodel == 0 & ipowerflow == 0`) is a two-node namespace (it also owns the
    TF-coil nuclear heating), the arm-2 one (`blktmodel == 0 & ipowerflow == 1`,
    PROCESS's own default and the reference run) a single node. Occupants of one slot
    need not have equal shape or equal output sets; what checks the consequences is the
    boundary postcondition, not a shape rule.

    Arm 0 -- `blktmodel == 1`, at either `ipowerflow` -- is refused: it is
    `blanket_neutronics()`, which calls `hcpb.nuclear_heating_*`, unported. That is also
    why the `| None` this annotation used to carry was **dead**: `0` is the only arm
    outside the registry, it is in `UNPORTED`, and it raises -- absence was never
    reachable.
    """

    blanket_masses: BlanketComponentMasses = dataclasses.field(kw_only=True)
    """Blanket component masses, on `.fwbs.blktmodel` x `.fwbs.blkttype` jointly,
    resolved by `_blanket_mass_arm`.

    Only arm 2 -- `blktmodel == 0` with a solid breeder, `blkttype not in {1, 2}`,
    which is PROCESS's own default and the reference run -- has an occupant; the
    liquid-breeder sub-arm (1) and the `blktmodel != 0` mass arm (0) are refused with
    their reasons in `UNPORTED`. Same dead `| None` as the slot above, for the same
    reason: arms `0` and `1` are the only others `_blanket_mass_arm` can return and
    both raise.
    """

    # `st_fwbs` S1/S5 (`stellarator_E_fwbs_synthesis.md`), portable now, no blocker.
    fw_blanket_shield_geometry: FwBlanketShieldGeometry = FwBlanketShieldGeometry()
    cryostat_and_vv_geometry: CryostatAndVvGeometry = CryostatAndVvGeometry()
    # `st_fwbs` S3 (`stellarator_fwbs_s3.md`). Reads `.divertor.a_div_surface_total`,
    # which `Divertor` owns -- an ordinary acyclic edge, not a cycle: `Divertor`'s own
    # inputs have no dependency back on anything `st_fwbs`/`DivertorPlateMass` produces
    # (verified directly against `divertor.py`'s `FromExactly`s), so PROCESS's own staleness
    # here (`st_fwbs` runs before `st_div`, so it reads the *previous* `run()`'s value)
    # is a call-order artifact of its imperative code, not a genuine two-way dependency.
    # Registering this the ordinary way (`Divertor` before `DivertorPlateMass` in
    # topological order) is strictly more self-consistent than PROCESS's own lagged
    # read -- confirmed by the build below staying at the same SCC count.
    divertor_plate_mass: DivertorPlateMass = DivertorPlateMass()
    # `st_fwbs` S4's shield-mass block (`stellarator_fwbs_s4.md`). Unswitched, not
    # behind a `Switch` because `stellarator.py:1195-1206` is outside every branch in
    # `st_fwbs` -- no `blktmodel`, `blkttype` or `ipowerflow` guard -- so both outputs
    # exist in every configuration. Its sibling `BlanketComponentMasses` *is* switched,
    # see `TOPOLOGY_SWITCHES`'s `.fwbs.blktmodel,.fwbs.blkttype` entry. Closes
    # `_audit/boundary_inputs_audit.md` § 4c (b5)/(b6): `Bldgs` and `ShieldCost` were
    # reading `.fwbs.whtshld`, and `ShieldCost` `.fwbs.wpenshld`, as boundary inputs.
    shield_mass: ShieldMass = ShieldMass()


class Stellarator(ModelNamespace):
    """Everything device-specific: the machine's own geometry, coils, and FWBS.

    `.stellarator.*` is not just the `stellarator.py` module -- several nodes here own
    `.build.*`/`.tfcoil.*` fields, because the *model* that computes them is the
    stellarator's, whatever area PROCESS files the field under.
    """

    machine_config: StellaratorMachineConfig = dataclasses.field(kw_only=True)
    """The 34 `.stellarator_config.stella_config_*` fields, from a `stella_conf.json`.

    Filled at `.stellarator.istell == 6` (machine config read from file) and at no other
    value, because there is no other value: `istell == 0` is a **tokamak**, which this
    tree has no counterpart namespace for, and `istell in 1..5` selects one of five
    hardcoded presets whose tables are not transcribed. All six are in `UNPORTED` and
    raise, which is why this slot needs no `| None` -- it used to hold one for the
    tokamak, and the tokamak is gone.

    A node with **no inputs**: the machine config is strictly upstream of every design
    variable, so it adds a source to the DAG and no cycle. **This is what makes the graph
    runnable from a cold `DataStructure`** -- before it, these 34 fields were unowned
    boundary inputs seeded from a converged run, and stepped cold they were all `0.0`,
    making `.tfcoil.n_tf_coils` zero and the first division by it emit non-finite values
    in 16 blocks.
    """

    heating: EcrhHeating | LowhybHeating = dataclasses.field(kw_only=True)
    """Which auxiliary heating model runs (`.stellarator.isthtr`, default 1 = ECRH).

    The NBI arm (`isthtr == 3`) is refused: `st_heat`'s NBI branch calls
    `current_drive.culnbi()`, a model this port has not audited.
    """

    fw_area: AFwTotalNoPowerflow | AFwTotalWithPowerflow = dataclasses.field(
        kw_only=True
    )
    """First-wall area (`.heat_transport.ipowerflow`, default 1).

    **The switch that decides whether the graph has a cycle**, which is why it is a slot
    and could never have been one node branching internally:
    `AFwTotalWithPowerflow` reads `.fwbs.f_ster_div_single`, which `divertor` owns, while
    `divertor` reads `.first_wall.a_fw_total`, which both occupants own -- so
    `ipowerflow != 0` has a genuine two-node SCC and `ipowerflow == 0` is acyclic.
    `test_machine.py` asserts both halves.
    """

    coils: StellaratorCoils = StellaratorCoils()

    fwbs: StellaratorFwbs = dataclasses.field(kw_only=True)

    # unit #1 chunks
    sudo_density_limit: SudoDensityLimit = SudoDensityLimit()
    # EcrhDensityLimit moved to TOPOLOGY_SWITCHES's `i_plasma_pedestal` switch -- its
    # static kwarg is no longer independent of that switch's value, see there.
    structure_masses: StructureMasses = StructureMasses()
    # ScTfCoilNuclearHeating moved to TOPOLOGY_SWITCHES's new joint `blktmodel`/
    # `ipowerflow` switch (value=1 arm) -- see that switch's own comment. Unconditional
    # placement here was a real bug: PROCESS's own default configuration lands in the
    # switch's value=2 arm, which computes `.fwbs.p_tf_nuclear_heat_mw` via a different
    # formula (`DetailedPowerflowBlanketShieldPower`), not this one.
    # unit #2, build.py -- `BlktmodelBlanketThickness` deliberately NOT here (see the
    # import comment above): PROCESS's own default `blktmodel = 0` means this node must
    # not be instantiated at all (`conditional-ownership-by-run-config`), a bug fixed
    # this pass, same class as `ScTfCoilNuclearHeating`/`EcrhDensityLimit` above.
    # `Build` no longer owns `.build.z_tf_inside_half` -- moved to `coils/calculate.py`'s
    # `ZTfInsideHalf` (registered below, near the rest of unit #9/#10's coil nodes), a
    # real ordering-artifact bug found via the block-by-block MDA-vs-PROCESS comparison
    # harness: two independent PROCESS writers of this field, `Build`'s formula was the
    # transient one, not the one that survives a real run's final report pass. See
    # `build.py`'s `calculate_build` docstring and `ZTfInsideHalf`'s own for the full
    # account.
    build: Build = Build()
    # unit #4, divertor.py
    divertor: Divertor = Divertor()
    # unit #5, heating.py
    injected_power_total: InjectedPowerTotal = InjectedPowerTotal()
    beam_current: BeamCurrent = BeamCurrent()
    fusion_gain: FusionGain = FusionGain()
    # unit #6, initialization.py
    pulse_durations: PulseDurations = PulseDurations()
    # unit #7, neoclassics.py (scalar-argument functions only, see module docstring)
    profile_values: ProfileValues = ProfileValues()
    effective_thermal_diffusivity: EffectiveThermalDiffusivity = (
        EffectiveThermalDiffusivity()
    )
    # `plasma_physics.py` (chunk 1B of unit #1). `StellaratorBetaAndRhoStar` is
    # still NOT registered: its `.physics.rho_star` output is algebraically identical to
    # `DimensionlessPlasmaParameters`'s own `rho_star` formula above (same inputs, same
    # expression, confirmed by direct comparison) -- a genuine redundant-duplicate-write
    # in PROCESS itself (`st_phys` and `outplas` both compute it), not a porting choice.
    # Registering both would be a duplicate-ownership conflict, the same shape as
    # `IterPhysicsBasisElongation`/`ConfinementTime`'s `kappa_ipb` above.
    #
    # **What is new: dropping that node used to cost `.physics.beta_total_vol_avg` and
    # `.physics.e_plasma_beta` their only producer as collateral.** They are not in
    # conflict with anything -- only `rho_star` was. `StellaratorBetaAndStoredEnergy`
    # (same pure function, same 13 inputs, `rho_star`'s return value discarded) owns
    # exactly those two and is registered instead. `.physics.beta_total_vol_avg` is
    # constraint 24's only argument, one of the 14 active constraints of the reference
    # run, so this is what makes that constraint assemblable at all -- see that class's
    # own docstring.
    stellarator_beta_and_stored_energy: StellaratorBetaAndStoredEnergy = (
        StellaratorBetaAndStoredEnergy()
    )
    poloidal_field_from_rotational_transform: PoloidalFieldFromRotationalTransform = (
        PoloidalFieldFromRotationalTransform()
    )
    total_field: TotalField = TotalField()
    # `stellarator.py:2152-2166`: `st_phys`'s two zero-clips on the radiation power
    # densities and the two total powers formed from them. Owns the real
    # `.physics.pden_plasma_*_rad_mw` fields, which `PlasmaRadiationPowers` now mints
    # as `*_unclipped` -- the clip has two disagreeing call sites in PROCESS, so it
    # belongs to this caller, not to `calculate_radiation_powers`. Also gives
    # `.physics.p_plasma_inner_rad_mw` (read by `StellaratorConfinementTime`) its first
    # producer -- `_audit/boundary_inputs_audit.md` §7 item 6.
    clipped_radiation_powers: ClippedRadiationPowers = ClippedRadiationPowers()
    # `i_pflux_fw_neutron`/`ipowerflow` static, per `physics_variables.py:1006`/
    # `heat_transport_variables.py:94`'s defaults (`1`). With `i_pflux_fw_neutron == 1`
    # both functions take their first branch unconditionally -- `ipowerflow`'s value is
    # inert for the actual computed result at this default, but still required as a
    # field; kept matching `.heat_transport.ipowerflow`'s own registered default above
    # for consistency, not because it changes anything here.
    neutron_wall_load: NeutronWallLoad = NeutronWallLoad(
        i_pflux_fw_neutron=NeutronWallLoadModel.SCALED_PLASMA_SURFACE_AREA,
        ipowerflow=PowerFlowModel.COMPREHENSIVE_2014,
    )
    # `i_plasma_ignited=1` (IGNITED, `stellarator_helias.IN.DAT:126`) -- **not**
    # `physics_variables.py:881`'s bare default `0`, which this registration used to
    # carry. Third site of the same mismatch (`PlasmaComposition`/`ConfinementTime` are
    # the other two), all three found together by `mda_harness.py`'s `switch_audit`.
    # Checked before flipping: `plasma_physics.py:273-274` adds
    # `p_hcd_injected_total_mw` into `powht` only under NON_IGNITED, so the IGNITED arm
    # reads a strict subset of the inputs -- nothing new to wire.
    heating_and_radiation_power: HeatingAndRadiationPower = HeatingAndRadiationPower(
        i_plasma_ignited=PlasmaIgnitionModel.IGNITED
    )
    radiated_wall_load_and_fraction: RadiatedWallLoadAndFraction = (
        RadiatedWallLoadAndFraction(
            i_pflux_fw_neutron=NeutronWallLoadModel.SCALED_PLASMA_SURFACE_AREA,
            ipowerflow=PowerFlowModel.COMPREHENSIVE_2014,
        )
    )
    thermal_energy_totals: ThermalEnergyTotals = ThermalEnergyTotals()
    # `geometry.py` (chunk 1C of unit #1). `DefaultAspectRatio` is the
    # `1 not in data.numerics.ixc` conditional-ownership case (module docstring): the
    # bare `NumericsData` dataclass default (`ixc = [0, 0, ...]`, no real iteration-
    # variable ID ever present) makes `1 not in ixc` true, so this node is instantiated
    # unconditionally here, matching PROCESS's own bare-default configuration -- the
    # same convention every topology switch's own `default` already follows.
    # `StellaratorScalingFactors` takes `aspect` as a plain `FromExactly` regardless of source
    # (this node's own output, when active, or an external iteration-variable value
    # otherwise), so no further wiring decision is needed here.
    default_aspect_ratio: DefaultAspectRatio = DefaultAspectRatio()
    stellarator_scaling_factors: StellaratorScalingFactors = StellaratorScalingFactors()
    stellarator_plasma_geometry: StellaratorPlasmaGeometry = StellaratorPlasmaGeometry()


class PhysicsProfiles(ModelNamespace):
    """Plasma profile shapes and the volume averages taken over them.

    A third level for the same reason `StellaratorCoils` is one: the density/temperature
    profile nodes and the fusion rates that read them form an SCC.
    """

    parameterisation: (
        ProfileParameterisationParabolic | ProfileParameterisationPedestal
    ) = dataclasses.field(kw_only=True)
    """How the plasma profiles are shaped (`.physics.i_plasma_pedestal`, default 1).

    Not a formula switch: the parabolic occupant carries `EcrhDensityLimit` as well, a
    node the pedestal occupant has no counterpart for at all -- under
    `i_plasma_pedestal == 1` PROCESS never computes a real ECRH density limit, so
    `.stellarator.dlimit_ecrh`/`bt_max_ecrh` are genuinely unproduced there. Ragged arms
    again, and the reason this could not be a static kwarg on one node.
    """

    # unit #12, physics/plasma_profiles.py
    profile_factors: ProfileFactors = ProfileFactors()
    # unit #21, physics/profiles.py -- arms not gated by `i_plasma_pedestal` only
    profile_grid: ProfileGrid = ProfileGrid(
        n_plasma_profile_elements=201
    )  # `physics_variables.py:1054` default
    ne_profile_integral: NeProfileIntegral = NeProfileIntegral()
    te_profile_integral: TeProfileIntegral = TeProfileIntegral()
    density_profile: DensityProfile = DensityProfile()
    # `plasma_profiles.py`. Unswitched, not under `.physics.i_plasma_pedestal`:
    # PROCESS writes `.physics.temp_plasma_ion_vol_avg_kev` in `parameterise_plasma`
    # *before* the branch, so it runs in both arms. A `FixedPointFunction` rather than an
    # `ExplicitFunction` because the field is conditionally owned by *data*
    # (`f_temp_plasma_ion_electron > 0`) -- see the class's own docstring for why that is
    # the honest shape and not a workaround.
    ion_vol_avg_temperature: IonVolAvgTemperature = IonVolAvgTemperature()


class PhysicsConfinementTime(ModelNamespace):
    """Energy confinement, and what is derived from it."""

    model: ConfinementTime = dataclasses.field(kw_only=True)
    """Which confinement-time scaling runs (`.stellarator.istell == 6`, Helias).

    **The one slot whose family already existed**: `StellaratorConfinementTime`
    subclasses
    `ConfinementTime`, so the annotation is the base and the occupant satisfies it --
    `model_tree_design.md` §4 case 1, the shared-body case. Every other switched slot in
    this tree is annotated with a union of its occupants, because no base was ever
    written for them; see this module's own note above `machine_from_indat`. The bare
    `ConfinementTime` was this slot's `istell == 0` occupant until that arm was deleted;
    a one-member family is what remains, not a spurious one.

    **The three settings live in the factory, and this slot has no default, because of
    them.** They are the reference run's, not `physics_variables.py`'s bare defaults, and
    each was a registration bug once: `i_confinement_time` was `34` (IPB98(y,2), a
    tokamak H-mode law) against ISS04's `38`, and `i_plasma_ignited` was `0` against the
    input file's `1`, which alone was the residual ~1.2 % `t_energy_confinement`
    disagreement. Both found by the MDA harness, never by a check -- and a slot default
    is exactly where a check could not see them, since the defaults test compared
    occupant *classes* only.
    """

    # unit #10, physics/confinement_time.py. `IterPhysicsBasisElongation` (the standalone
    # wrap of `calculate_iter_physics_basis_elongation`) is deliberately NOT registered:
    # `ConfinementTime` already computes and owns `.physics.kappa_ipb` itself (it calls
    # the same underlying function internally, then returns the value as its own 8th
    # output) -- registering both would be a duplicate-ownership conflict on one VarPath,
    # not two independent nodes. `i_confinement_time`/`i_rad_loss`/`i_plasma_ignited` are
    # kept as static kwargs (`ConfinementTime.__call__`'s dispatch needs concrete Python
    # ints, not traced values -- see confinement_time.py's own docstring, fixed during
    # this consolidation pass, for why all three needed `eqx.field(static=True)`).
    # `i_confinement_time=38` (ISS04, stellarator) -- **not** `physics_variables.py`'s own
    # bare default (`34`, IPB98(y,2), a tokamak H-mode scaling law): that default is
    # PROCESS's own tokamak-first choice, not one this project's stellarator scope should
    # inherit uncritically. Found and corrected via the block-by-block MDA-vs-PROCESS
    # comparison harness (`mda_harness.py`): registering `34` fed a tokamak confinement
    # formula stellarator inputs, producing a degenerate `t_energy_confinement` that
    # cascaded into `DoubleAndTripleProduct.ntau == 0.0` and several `inf` values
    # downstream (reciprocals of ~0). The ISS04 branch was already fully ported
    # (`confinement_time.py:1743`) -- this was a wrong static default at registration,
    # not a missing-physics gap. `i_confinement_time` genuinely has ~40 possible values
    # (tokamak and stellarator scaling laws both); a real `Switch`/`Alternative` covering
    # more of them is a separate, larger follow-up, not done here -- `38` is the
    # pragmatic, scope-appropriate single default for now, same discipline as
    # `i_rad_loss=1` alongside it.
    #
    # `i_plasma_ignited=1` (IGNITED) -- **not** `physics_variables.py:881`'s bare
    # default `0`, which is what this registration originally carried. Found by
    # `mda_harness.py`'s `switch_audit`, the systemic check added for exactly this
    # defect class; `stellarator_helias.IN.DAT:126` sets `i_plasma_ignited = 1`.
    # This was the sole cause of the residual ~1.2% `t_energy_confinement`/`ntau`
    # disagreement `next_steps.md` §8 previously listed as open and undiagnosed:
    # `confinement_time.py:1333-1334` adds `p_hcd_injected_total_mw` into
    # `p_plasma_loss_mw` only under NON_IGNITED, so registering `0` inflated the loss
    # power PROCESS's real ignited run never adds, and `t_energy_confinement` (and
    # everything scaled by it) came out correspondingly off.
    # Checked before flipping, same discipline as `i_thermal_electric_conversion`:
    # the IGNITED arm simply *omits* that one addition -- it reads a strict subset of
    # the NON_IGNITED arm's inputs, so it needs nothing this port does not wire.
    # **Not registered here**: `ConfinementTime`/`StellaratorConfinementTime` are
    # arms of `TOPOLOGY_SWITCHES`'s `.stellarator.istell` switch -- see that switch
    # for why the device mode decides which node produces this block's 20th read.
    double_and_triple_product: DoubleAndTripleProduct = DoubleAndTripleProduct()


class Physics(ModelNamespace):
    """Plasma physics: composition, profiles, fusion rates, beta, exhaust."""

    # **Filed here on physical grounds, against PROCESS's own filing.** Both live in
    # `stellarator.py`'s `st_phys` and every earlier grouping put them under
    # `stellarator` because of it -- but they own **only** `.physics.*` fields, exactly
    # as `FusionRates`/`SetFusionPowers` beside them do, and nothing about them is
    # stellarator-specific. Leaving them under `stellarator` was measured to be the
    # sole reason the density/fusion/pedestal cycle crossed a subsystem boundary, which
    # `switch_elimination_design.md` §11.1 had recorded as never happening. §11.3 says
    # the stronger claim is a grouping "declared on physical grounds" rather than
    # mirrored from PROCESS's file layout, and this is the first place the two
    # disagree -- so the declaration wins, and §11.1's containment result survives.
    fusion_power_totals_mw: FusionPowerTotalsMw = FusionPowerTotalsMw()
    # The `else` arm of `stellarator.py:2002-2054`, three identities -- and the only
    # producer of `.physics.fusden_total`/`.fusden_alpha_total`/`.p_dt_total_mw`,
    # which were boundary inputs until it landed. Unconditional because the arm is
    # selected by `i_plasma_ignited == IGNITED` on this run, not merely by the absence
    # of a beam, and because the beam arm calls the unportable `reactions.beam_fusion`
    # (unit #19) -- there is no second arm to switch between. See
    # `_audit/boundary_inputs_audit.md` §4c (b7)/(b8) and the class's own docstring.
    fusion_totals_no_beam: FusionTotalsNoBeam = FusionTotalsNoBeam()
    profiles: PhysicsProfiles = dataclasses.field(kw_only=True)

    confinement_time: PhysicsConfinementTime = dataclasses.field(kw_only=True)

    # unit #19, physics/fusion_reactions.py
    fusion_rates: FusionRates = FusionRates()
    set_fusion_powers: SetFusionPowers = SetFusionPowers()
    # unit #20, physics/radiation_power.py
    synchrotron_radiation_power: SynchrotronRadiationPower = SynchrotronRadiationPower()
    # `imp_indices` is a graph-assembly-time fact (which impurity species this machine
    # has), not a per-evaluation switch -- see `ImpurityRadiationTotals`'s docstring.
    # All 14 species: H/He are always recomputed by `plasma_composition()`, and species
    # 2-13 are held non-zero by iteration variables 125-136's lower bound (1e-8, 22
    # orders above the 1e-30 selection threshold) in the reference configuration this
    # scope targets. A run without those iteration variables active could legitimately
    # need a narrower tuple; nothing here checks that yet (radiation_power.md § open
    # questions 2).
    impurity_radiation_totals: ImpurityRadiationTotals = ImpurityRadiationTotals(
        imp_indices=tuple(range(14))
    )
    plasma_radiation_powers: PlasmaRadiationPowers = PlasmaRadiationPowers()
    # unit #9 chunk A, physics/pure_formulas.py -- five already-pure formulas
    # lifted verbatim, no entanglement, no switch-driven topology split.
    ion_electron_equilibration: IonElectronEquilibration = IonElectronEquilibration()
    auxiliary_physics_quantities: AuxiliaryPhysicsQuantities = (
        AuxiliaryPhysicsQuantities()
    )
    total_plasma_heating_power: TotalPlasmaHeatingPower = TotalPlasmaHeatingPower()
    electron_thermal_energy: ElectronThermalEnergy = ElectronThermalEnergy()
    ion_thermal_energy: IonThermalEnergy = IonThermalEnergy()
    # `i_beta_fast_alpha` kept as a static kwarg, not a Switch -- both branches read the
    # same six variables (pure_formulas.md's "switches touched"), same
    # shape as `EcrhDensityLimit`'s `i_plasma_pedestal`. Default `1` (WARD),
    # `physics_variables.py:875`.
    fast_alpha_beta: FastAlphaBeta = FastAlphaBeta(
        i_beta_fast_alpha=FastAlphaPressureModel.WARD
    )
    # unit #9 chunk B, physics/composition.py. `plasma_composition`'s
    # `.physics.first_call` turned out not to be a genuine cycle at all -- its real
    # referent (`f_temp_plasma_electron_density_vol_avg`, from `plasma_profiles.py`) has
    # no dependency back on this node, so `first_call` is an ordering artifact of
    # PROCESS's imperative call sequence (`next_steps.md` §5), not ported. An earlier
    # draft represented it as a `NextFirstCall`/`FixedPointFunction` self-loop; removed.
    # The second Shape-B self-loop this chunk's own audit found this session
    # (`.impurity_radiation.f_nd_impurity_electron_array`, read at indices 2-13, written
    # at 0/1) is resolved without any `Cut`/`FixedPoint` machinery at all -- per-index
    # `VarPath`s (`s.impurity_radiation.f_nd_impurity_electron_array[i]`) make the read
    # and write ranges genuinely disjoint `VarPath`s, not one whole-array self-reference.
    # `i_plasma_ignited=IGNITED` -- **not** `physics_variables.py:881`'s bare
    # default (`i_plasma_ignited = 0`, NON_IGNITED). `stellarator_helias.IN.DAT:126`
    # sets `i_plasma_ignited = 1` (IGNITED), and the converged run confirms it: the
    # `switch_audit` check `mda_harness.py` now runs over every registered static kwarg
    # reported `registered=False but .physics.i_plasma_ignited == True`. Same defect
    # class as `i_confinement_time`/`i_thermal_electric_conversion` below -- a bare
    # `*_variables.py` default copied uncritically into a registration.
    # This node used to spell the same fact as a `bool` named `is_ignited`, which
    # `switch_elimination_design.md` §3 classifies as kind (d) alias/noise and which
    # cost `mda_harness.STATIC_KWARG_ALIASES` a hand-written `bool(v == 1)` entry
    # just so `switch_audit` could resolve it. It is now spelled and typed as
    # PROCESS spells it, `PlasmaIgnitionModel` under PROCESS's own field name, and
    # resolves by name like every other static kwarg.
    # Checked before flipping, same discipline as `i_thermal_electric_conversion`
    # below: the IGNITED arm needs no input this port does not already wire --
    # `composition.py:219-222` is `nd_beam_ions = 0` under IGNITED
    # versus `nd_plasma_electrons_vol_avg * f_nd_beam_electron` otherwise, so the
    # ignited arm reads a strict *subset* of the non-ignited arm's inputs.
    plasma_composition: PlasmaComposition = PlasmaComposition(
        i_plasma_ignited=PlasmaIgnitionModel.IGNITED
    )
    # Over the line length and left that way: slot name and occupant class are both
    # this long, and the annotation cannot be wrapped -- `ruff format` strips
    # parentheses from around an annotation. An import alias would hide the class name.
    calculate_effective_charge_ionisation_profiles: CalculateEffectiveChargeIonisationProfiles = CalculateEffectiveChargeIonisationProfiles()  # noqa: E501
    # unit #9 chunk C, physics/dimensionless_parameters.py -- the one real
    # computation inside the 1095-line `outplas` reporting method.
    dimensionless_plasma_parameters: DimensionlessPlasmaParameters = (
        DimensionlessPlasmaParameters()
    )
    # unit #11, physics/exhaust.py
    radiation_fraction: RadiationFraction = RadiationFraction()


class Power(ModelNamespace):
    """Thermal and electric power flows, cryogenics, and the plant's own consumption."""

    tf_power: TfPowerResistive | TfPowerSuperconducting = dataclasses.field(
        kw_only=True
    )
    """TF-coil power supplies (`.tfcoil.i_tf_sup`, default 1 = superconducting).

    The aluminium arm (`i_tf_sup == 2`) is refused rather than aliased onto the resistive
    occupant: it runs the identical branch in PROCESS, but saying so by registry absence
    keeps the claim visible instead of burying it in a shared occupant.
    """

    # `tf_coil_power.py` (unit #14 chunk A). `TfPowerResistive`/
    # `TfPowerSuperconducting` are registered under `TOPOLOGY_SWITCHES`'s new
    # `.tfcoil.i_tf_sup` switch instead of here -- see that switch's own comment.
    # `thermal_cryo.py` (unit #14 chunk B). Six of `calculate_
    # component_thermal_powers`'s outputs are genuine single-node self-loops (each
    # field's *entering* value is read, then a freshly-computed value is written back to
    # the same `VarPath` later in the same PROCESS call) -- already split this session
    # into their own `FixedPointFunction`s, same "Shape B" treatment as
    # `plasma_composition`'s `first_call`/`Avail`'s `cplife` above.
    # `i_blkt_dual_coolant=0`/`i_blanket_type=1`/`secondary_cycle_liq=4` match
    # `fwbs_variables.py`'s own defaults (lines 526, 70, 273) and agree with this run,
    # confirmed by `mda_harness.py`'s `switch_audit`.
    #
    # `i_p_coolant_pumping=1` (`PumpingPowerModelTypes.FRACTION_OF_HEAT`,
    # `power.py:23`) -- **not** `fwbs_variables.py:249`'s bare default `2`
    # (`MECHANICAL`), which all four registrations below used to carry.
    # `stellarator_helias.IN.DAT:198` sets `1`. Flagged but not fixed by the pass that
    # corrected `i_thermal_electric_conversion`; fixed here, and now checked
    # automatically rather than by luck (`switch_audit`). Checked before flipping,
    # same discipline: the two switch-dependent bodies are conditional-ownership
    # pass-throughs, and value `1` selects the *recompute* side of both, out of
    # arguments these nodes already take --
    # `calculate_p_fw_blkt_coolant_pump_mw` (`thermal_cryo.py:206-211`)
    # returns `p_fw_coolant_pump_mw + p_blkt_coolant_pump_mw` for `1 not in
    # {MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`, and
    # `calculate_p_fw_div_heat_deposited_mw` (`thermal_cryo.py:308-310`)
    # returns `p_fw_heat_deposited_mw + p_div_heat_deposited_mw` for
    # `1 != MECHANICAL_WITH_PRESSURE_DROP`. Both operands are already `FromExactly`s (or
    # rebuilt from `FromExactly`s) on every node below, so no arm has a hole in it.
    #
    # `i_thermal_electric_conversion=2` (`ElectricConversionModelTypes.USER_INPUT`) --
    # **not** `0` (`CCFE_HCPB_VALUE`, `fwbs_variables.py:264`'s bare default). Found and
    # corrected via the block-by-block MDA-vs-PROCESS comparison harness
    # (`mda_harness.py`): `stellarator_helias.IN.DAT` sets this explicitly (line 203),
    # and the wrong hardcoded `0` fed a completely different branch of
    # `calculate_plant_thermal_efficiency`/`calculate_component_thermal_powers`/
    # `calculate_delta_eta` than PROCESS's own real run took -- confirmed as the exact,
    # sole cause of `ComponentThermalPowers`/`EtaTurbineStep`/`DeltaEtaStep`'s
    # disagreements (bit-for-bit match once corrected, `PicardDriver` itself was never
    # at fault). Every branch these four nodes' `step`/`__call__` bodies already read
    # this switch through -- `USER_INPUT` needs no input this port doesn't already wire
    # (it is a pure identity pass-through for `eta_turbine`, confirmed against
    # `power.py:1992-1994`), so this is a like-for-like default correction, not a new
    # port. Duplicated identically across these four registrations rather than one
    # shared source of truth -- same caution `i_confinement_time` (just above) already
    # flags for itself: a real `Switch`/`Alternative` covering
    # `ElectricConversionModelTypes`'s 5 values is a separate, larger follow-up, not
    # done here.
    component_thermal_powers: ComponentThermalPowers = ComponentThermalPowers(
        i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT,
        i_blkt_dual_coolant=BlanketDualCoolantModel.SINGLE_COOLANT_SOLID_BREEDER,
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
        secondary_cycle_liq=(
            ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
        ),
    )
    delta_eta_step: DeltaEtaStep = DeltaEtaStep(
        i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT,
        i_blkt_dual_coolant=BlanketDualCoolantModel.SINGLE_COOLANT_SOLID_BREEDER,
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
    )
    eta_turbine_step: EtaTurbineStep = EtaTurbineStep(
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
    )
    etath_liq_step: EtathLiqStep = EtathLiqStep(
        secondary_cycle_liq=(
            ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
        )
    )
    temp_turbine_coolant_in_step: TempTurbineCoolantInStep = TempTurbineCoolantInStep(
        i_thermal_electric_conversion=ElectricConversionModelTypes.USER_INPUT,
        i_blanket_type=BlktModelTypes.CCFE_HCPB,
        secondary_cycle_liq=(
            ElectricConversionModelTypes.SUPERCRITICAL_CO2_BRAYTON_CYCLE
        ),
    )
    p_fw_div_heat_deposited_mw_step: PFwDivHeatDepositedMwStep = (
        PFwDivHeatDepositedMwStep(
            i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT
        )
    )
    p_fw_blkt_coolant_pump_mw_step: PFwBlktCoolantPumpMwStep = PFwBlktCoolantPumpMwStep(
        i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT
    )
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
    # `PlantThermalEfficiency` does -- `to_graph(Cryo(...))` raises `ValueError: reads
    # ['.fwbs.qnuc'], which it also owns` -- and the three nodes below replace it:
    #   * `CryoQNucStep` owns `.fwbs.qnuc`, conditionally written by PROCESS under
    #     `inuclear == 0 and i_tf_sup == 1` ("Issue #511: if inuclear = 1: qnuc is
    #     input", `power.py:1825`);
    #   * `CryoQLoadsStep` owns `.power.qss`/`qac`/`qcl`/`qmisc`, conditionally written
    #     under the *other* guard, `i_tf_sup == 1 or i_pf_conductor == SUPERCONDUCTING`
    #     (`power.py:1054-1057`), which is why the five fields are two nodes and not
    #     one -- see `CryoQNucStep`'s docstring for the degeneracy argument;
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
    cryo_q_nuc_step: CryoQNucStep = CryoQNucStep(
        i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        inuclear=CoilNuclearHeatingModel.FRANCES_FOX,
    )
    cryo_q_loads_step: CryoQLoadsStep = CryoQLoadsStep(
        i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        i_pf_conductor=PFConductorModel.SUPERCONDUCTING,
    )
    cryo_loads: CryoLoads = CryoLoads(
        i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        i_pf_conductor=PFConductorModel.SUPERCONDUCTING,
    )
    # `electric_production.py` (unit #14 chunk C). `i_pf_energy_storage_source=2`
    # matches `pf_power_variables.py:18`'s default.
    acpow: Acpow = Acpow(i_pf_energy_storage_source=PFEnergyStorageSource.LINE)


class Buildings(ModelNamespace):
    """Plant buildings."""

    sizing: Bldgs | BldgsSizes = dataclasses.field(kw_only=True)
    """Which building-size model runs (`.buildings.i_bldgs_size`, default 0 = ITER 1992).

    The two occupants share `a_plant_floor_effective`/`volnucb`, which is what proved
    them mutually exclusive back when exclusivity had to be *detected* from colliding
    output ownership. It is now by construction: one slot, one occupant.
    """

    # unit #15, buildings.py -- unconditional preamble, feeds both `i_bldgs_size` arms
    tf_coil_envelope: TfCoilEnvelope = TfCoilEnvelope()


class Vacuum(ModelNamespace):
    """Vacuum pumping and the duct sizing problem."""

    # unit #16, vacuum.py -- `"old"` branch only, matching PROCESS's own default
    # (`.vacuum.i_vacuum_pumping = "old"`, `vacuum_variables.py:18`). Not gated by a
    # `Switch`: the `"simple"` alternative (`VacuumPumpingSimple`) owns a disjoint
    # output, so this switch fails `check_arms_are_exclusive` -- see
    # `TOPOLOGY_SWITCHES`'s docstring above. `VacuumPumpingSimple` stays
    # ported-but-unregistered.
    vacuum_old: VacuumOld = VacuumOld()
    # `DuctDiameterRootFind` -- registered as a deliberate island: every `VarPath` it
    # reads/owns is minted and unique to it (`.vacuum.d_duct`/`l1`/`l2`/`l3`/`xmult_i`/
    # `ceff_i`), so it has no producer/consumer edge to any other node registered here
    # today, the same shape `coils.py`'s unregistered `Jcrit*` nodes are flagged with
    # (see this module's own docstring). Registered anyway, on explicit instruction, as
    # a perfectly valid undriven `RootFind` problem sitting in the graph -- see that
    # class's own docstring. `vacuum.py`'s own `DuctFeasibility` (a bare `Feasibility`
    # `DeclaredNode`, not a `NodalDeclaration` -- see its docstring for why it cannot be
    # passed to `to_graph()`/listed here the same way) is *not* registered: joining it
    # with this node into one combined block is demonstrated in `test_vacuum.py`, not
    # asserted by this graph.
    duct_diameter_root_find: DuctDiameterRootFind = DuctDiameterRootFind()


class Availability(ModelNamespace):
    """Plant availability and component lifetimes."""

    electric_production: PowerProfilesOverTime | PlantElectricProductionReactor = (
        dataclasses.field(kw_only=True)
    )
    """Net electric power over the pulse cycle (`.costs.ireactor`, default 1).

    `ireactor == 1` is the reactor arm, which owns `.heat_transport.
    p_plant_electric_net_mw` -- the field constraint 16 reads. `ireactor == 0` computes
    the power *profiles* only.
    """

    # `PowerProfilesOverTime`/`PlantElectricProductionReactor` are the two arms of the
    # `.costs.ireactor` slot below, not unswitched members -- see that slot.
    # `availability.py` (unit #17). `Stellarator.run()`'s solve-time branch calls
    # `self.availability.avail()` directly (`stellarator.py:175`), bypassing
    # `.costs.i_plant_availability`'s dispatch entirely -- so `Avail` (not `Avail2`/
    # `AvailSt`) is the node actually exercised at solve time regardless of that
    # switch's value, and belongs in the unswitched part, not behind a slot. Its
    # `.costs.cplife` self-loop is resolved the same way as `plasma_composition`'s
    # `first_call`/`thermal_cryo.py`'s six fields above: `CplifeAvail`
    # (`FixedPointFunction`) owns `.costs.cplife` alone; `Avail` (`ExplicitFunction`)
    # owns every other output, reading `cplife` as a plain `FromExactly`.
    # `CpLifetimeSuperconducting`/`CpLifetimeResistive` are deliberately NOT registered:
    # `CplifeAvail.step` duplicates their `i_tf_sup` dispatch inline instead of calling
    # them (see `CplifeAvail`'s own docstring) precisely so only one node ever owns
    # `.costs.cplife` -- registering both pairs together would conflict.
    # `WardTaylorAvailability` is NOT registered either: PROCESS's own default
    # `.costs.i_plant_availability = 2` (MORRIS, `cost_variables.py:408`) means `avail()`
    # 's internal `WARD_TAYLOR` branch (`i_plant_availability == 1`) never fires, so
    # `.costs.f_t_plant_available` has no producer under the default configuration --
    # unconditional registration would reproduce the `EcrhDensityLimit` bug class
    # (computing a value the default configuration never computes), and it cannot be a
    # `Switch` either (no counterpart node exists for any other value, so
    # `check_arms_are_exclusive` would reject a one-real-arm pairing, same as
    # `i_vacuum_pumping`/`i_cost_model`). `ibkt_life=0`/`itart=0` match
    # `cost_variables.py:416`/`physics_variables.py:994`'s defaults.
    avail: Avail = Avail(
        ibkt_life=BlanketLifetimeModel.NEUTRON_FLUENCE,
        itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
    )
    cplife_avail: CplifeAvail = CplifeAvail(
        i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
    )


class StellaratorProcess(ModelNamespace):
    """One device, configured whole: every slot in this port, and what fills it.

    **Named for the device, because that is what it is.** There is no tokamak arm and
    `Stellarator` below has no counterpart namespace, so a general "machine" is not what
    this class describes. *Machine* stays the noun for an **instance** of it --
    `machine_from_indat`, `REFERENCE_MACHINE`, `graph_for(machine=...)` -- and the mixed
    vocabulary is deliberate: the class is a device, an instance of it is a machine.

    **A slot the factory fills has no default.** Every slot `machine_from_indat` passes
    is `dataclasses.field(kw_only=True)`, here and in the sub-namespaces; every other
    slot keeps its default. A default is admissible only where there is nothing to
    decide. `StellaratorProcess()` therefore raises `TypeError` instead of silently
    producing one particular machine -- which is what it did until this pass, with the
    reference run's switch values transcribed into slot constructor kwargs
    (`i_confinement_time = 38`, `i_plasma_ignited = 1`, against PROCESS's own `34` and
    `0`) where the test that claimed to police defaults could not see them, because it
    compared occupant classes only. `kw_only` is what lets a defaulted and an
    undefaulted slot sit in any order, so no sub-namespace had to be reordered.

    **The tree is the configuration.** A node is named by the path that reaches it, so
    `.stellarator.coils.coil_current` says where a model belongs, and every line below
    reads as *slot = occupant*: the field name is the place in the machine, the
    annotation is what may fill it, and the right-hand side is what does.

    **A slot is a place, so the class name is not in the node's name.** Swapping an
    occupant renames nothing (`model_tree_design.md` §3.2), and a `NodePath` here is a
    working address: `eqx.tree_at(lambda m: m.stellarator.coils.coil_current, ...)`
    reaches exactly what `.stellarator.coils.coil_current` spells. The cost is real and
    accepted -- reading a drawing no longer tells you which class computes a node -- and
    the mitigation (a renderer label `slot: OccupantClass`) is deferred until that
    actually hurts.

    **Slot names are the snake_case of their occupant's class**, mechanically, including
    where a shorter noun would read better. The rule is worth more than the wording: it
    is checkable, and it means no slot name is a judgement call to be relitigated.

    **Grain: the subsystem, with a third level only where the sub-area is a real thing**
    -- an SCC lives inside it (`stellarator.coils`, `physics.profiles`), or it is a slot
    something could be swapped into (`physics.confinement_time`) -- and never merely a
    filename. `switch_elimination_design.md` §11.1 measured why: every genuine cycle is
    contained within one subsystem and spans several files inside it, so the subsystem is
    the right grain for a model group and the file is not. That is also why the audit
    chunk letters do not appear here -- and, since `model_tree_design.md` §10, not in
    the filenames either: `physics/pure_formulas.py`, `power/thermal_cryo.py` and
    `stellarator/plasma_physics.py` are named for what is in them, not for the letter the
    port was chunked under for auditing. `stellarator_fwbs_s1_s5` is the one place the
    chunking is still legible, held back from that rename because `st_fwbs`'s S1-S6
    re-chunking is still live (`next_steps.md` §3), so a name carrying it would move
    again.

    Binding order is the order written here (`vars()`, not the MRO -- a namespace is
        written, not inherited), and it is only a tiebreak: the run order is derived.
    """

    costs: Costs = dataclasses.field(kw_only=True)
    """The cost model (`.costs.i_cost_model`), and a slot with exactly one occupant.

    The 1990 model is it. Both other values are in `UNPORTED` and raise, for different
    reasons that are recorded there: `== 1` (KOVARI_2014) is PROCESS's own default and
    would compute no cost of electricity at all; `== 2` injects a user-supplied `Model`
    at runtime and has no PROCESS-side subgraph to port.

    The reference run sets `i_cost_model = 0` explicitly -- the input file's own comment
    is *"the 2015 does not work yet for stellarators"* -- so `Costs()` is the occupant
    for every run in this project's scope.
    """

    stellarator: Stellarator = dataclasses.field(kw_only=True)

    physics: Physics = dataclasses.field(kw_only=True)

    power: Power = dataclasses.field(kw_only=True)

    buildings: Buildings = dataclasses.field(kw_only=True)

    vacuum: Vacuum = Vacuum()
    """The one sub-namespace that keeps a default: nothing inside it is switched, so
    there is nothing for `machine_from_indat` to decide and no configuration for a
    default to smuggle in. The other five hold a switched slot somewhere beneath them,
    which is why they cannot be default-constructed either."""

    availability: Availability = dataclasses.field(kw_only=True)


REFERENCE_INPUT_FILE = "tests/regression/input_files/stellarator_helias.IN.DAT"
"""The run this whole port is validated against -- `mda_harness.py`, `mda_constraint_
harness.py` and every number in `_audit/next_steps.md` \u00a7 8 use it. Named here so
`REFERENCE_CONFIGURATION` can be checked against it mechanically instead of by eye."""

_ISTELL_PRESET_REASON = (
    "`istell` in 1..5 selects one of five hardcoded machine presets (Helias 5/4/3, "
    "W7-X 30/50) copied onto `StellaratorConfigData` by `preset_config.py`'s reflective "
    "`hasattr`/`setattr` loop; only `istell == 6` (config read from file) is in scope. "
    "See `core/solver/switches.md` § `data.stellarator.istell` -- second role, whose "
    "disposition is still open in `_audit/next_steps.md` § 2. The confinement-time "
    "binding itself would be identical to the `istell == 6` occupant; it is the "
    "surrounding preset data that is unported"
)
"""Shared by all five refused `istell` presets -- one reason, five values."""

REFERENCE_MACHINE_SWITCHES = {
    "istell": 6,  # `stellarator_helias.IN.DAT:137`
    "isthtr": 1,  # `:139` -- equals PROCESS's own default, listed anyway
    "i_plasma_pedestal": 0,  # `:118`
    "i_cost_model": 0,  # `:248`
    "ireactor": 1,  # `:245` -- equals PROCESS's own default, listed anyway
}
"""The switch values `REFERENCE_INPUT_FILE` actually sets, as a faithful transcription.

**Every switch the file sets explicitly is listed, including ones whose value happens to
equal PROCESS's own default** (`isthtr`, `ireactor`). Listing them regardless makes this
a transcription of the file rather than a diff against PROCESS's defaults, and means a
future change to a default cannot silently move the reference run.
`test_machine.py` parses the file and checks this dict against it, both ways, so
the two cannot drift.

This exists as data, not as behaviour: `machine_from_indat` reads the file itself. It is
here so the check has something to compare against.
"""

UNPORTED = {
    ("istell", 0): (
        "istell == 0 is a tokamak, and this tree has no tokamak: `Stellarator` has no "
        "counterpart namespace, so assembling it would give stellarator geometry, "
        "stellarator coils and stellarator FWBS driven by a tokamak confinement "
        "scaling -- a device nobody has built and this port has never tested. Refused "
        "rather than absent, because it is the kind where assembling anyway hands you a "
        "graph that looks complete and is wrong. Consequence, stated rather than "
        "papered over: `istell` has no usable default here, so an IN.DAT that does not "
        "set `istell = 6` is refused"
    ),
    ("istell", 1): _ISTELL_PRESET_REASON,
    ("istell", 2): _ISTELL_PRESET_REASON,
    ("istell", 3): _ISTELL_PRESET_REASON,
    ("istell", 4): _ISTELL_PRESET_REASON,
    ("istell", 5): _ISTELL_PRESET_REASON,
    ("isthtr", 3): (
        "the NBI branch of `st_heat` calls `current_drive.culnbi()`, a model that is "
        "not audited yet (registry unit #5)"
    ),
    ("blktmodel_ipowerflow", 0): (
        "S2's blktmodel == 1 arm is `blanket_neutronics()`, which calls "
        "`self.hcpb.nuclear_heating_blanket()`/`nuclear_heating_shield()` with zero "
        "arguments against 2-/7-keyword-argument @staticmethods -- a live PROCESS bug "
        "that would TypeError the moment this arm actually executes (unit_registry.md "
        "row 13, next_steps.md §3). hcpb.py's own 3 ported nodes "
        "(NuclearHeatingBlanket/Shield/Magnets) exist but are not usable here until "
        "that call site has a resolution."
    ),
    ("blktmodel_blkttype", 0): (
        "the blktmodel != 0 blanket-mass arm (stellarator.py:1093-1181) computes "
        "m_blkt_steel_total/m_blkt_beryllium from six .build.bl{u,m,p}{i,o}th "
        "sub-assembly thicknesses, additionally writes .fwbs.whtblbreed and "
        ".fwbs.f_a_blkt_cooling_channels, and writes neither .fwbs.m_blkt_li2o nor "
        ".fwbs.m_blkt_vanadium at all -- a different node with a different port set, "
        "not written yet. Refused rather than assembled empty: BlanketCost reads all "
        "four masses unconditionally, so an empty arm would silently hand it boundary "
        "values for fields PROCESS does compute on that arm."
    ),
    ("blktmodel_blkttype", 1): (
        "the liquid-breeder sub-arm (blkttype in {1, 2}, WCLL/HCLL, "
        "stellarator.py:1058-1066) writes .fwbs.wtbllipb and .fwbs.m_blkt_lithium in "
        "place of .fwbs.m_blkt_li2o/.m_blkt_beryllium -- different fields, not a "
        "different formula for the same ones. Not ported: neither replacement field "
        "has a reader in this graph, and stellarator_helias.IN.DAT leaves blkttype at "
        "its default of 3. Values 1 and 2 select the identical formula, so this one "
        "entry covers both; there is no separate value=2 entry because there is no "
        "separate behaviour to name."
    ),
    ("i_tf_sup", 2): (
        "aluminium TF (i_tf_sup == 2) runs the identical calculate_tf_power_resistive "
        "branch as i_tf_sup == 0 -- `Power.tfpwr` dispatches on `i_tf_sup != 1` only, "
        "one formula for both. Request `.tfcoil.i_tf_sup == 0` instead; it fills the "
        "slot with the same occupant. Kept as a refused value rather than a second "
        "registry entry pointing at TfPowerResistive so the claim stays visible."
    ),
    ("i_cost_model", 1): (
        "KOVARI_2014 (i_cost_model == 1) is PROCESS's own default cost model and is "
        "unported: costs_2015.py has no cottax nodes, so on that arm this port computes "
        "no cost of electricity at all and .costs.coe/.costs.concost would surface as "
        "unowned boundary inputs. Filling the slot with the 1990 model instead would "
        "compute *a different number for the same field* -- worse than the "
        "EcrhDensityLimit bug class, which merely computed a value the configuration "
        "never asks for. This used to be spelled as a slot holding None; it is a "
        "refusal now, because a tree with no optional slots cannot say 'absent'"
    ),
    ("i_cost_model", 2): (
        "i_cost_model == 2 injects a user-supplied Model instance at runtime "
        "(process/main.py's `costs` setter, lines 766-768) -- there is no PROCESS-side "
        "subgraph to port at all, so no occupant can exist here. Refused rather than "
        "left absent: unlike KOVARI_2014, a caller asking for this has a model in mind "
        "that this graph has never seen."
    ),
}
"""Why a known PROCESS value has no occupant, verbatim from the `Alternative(unported=)`
declarations this replaced.

**Refusal, and nothing else.** A value in here raises `NotImplementedError` naming the
reason. It used to have a quieter sibling -- a slot holding `None`, meaning *"this
configuration's graph does not compute these values"* -- and that spelling is gone: all
four `| None`s left the tree, two because they were unreachable (every joint key outside
`BLANKET_MASSES`/`BLANKET_SHIELD_POWER` already raised) and two because the
configurations they stood for, `i_cost_model == 1` and `istell == 0`, are ones this port
cannot honestly assemble. Both kinds of value are refused now and the distinction
survives only in the reasons: `i_cost_model == 1` would hand you a graph that computes no
cost of electricity, `== 2` and `istell == 0` would hand you one that looks complete and
is wrong.

Keyed by `(field, value)`. For the two dispatches that read two integers at once the
`field` is the joint name `blktmodel_ipowerflow` / `blktmodel_blkttype` and the `value`
is an **arm index**, not a switch value -- see `_blanket_shield_power_arm` /
`_blanket_mass_arm`, whose docstrings are the mapping.

One of those arms, `("blktmodel_blkttype", 0)`, is unreachable through
`machine_from_indat` and kept anyway: `blktmodel == 1` selects arm 0 of *both*
dispatches, and the shield-power slot is resolved first, so the reason that surfaces is
the `blanket_neutronics()` one. The mass-arm reason is still the correct record of what
`stellarator.py:1093-1181` does, and it is what a future occupant of that arm has to
answer; it is not deleted merely because a sibling refusal masks it.
"""


def _slot_occupant(field, value, registry, *, build=None):
    """One registry lookup, with both failure modes spelled out.

    A miss on the registry *and* on `UNPORTED` means a value PROCESS has never had, or a
    typo -- reported with the values that do exist, which is the "a typo'd value fails
    loudly" property the old `Switch.choose` had and is worth keeping.

    Raises
    ------
    NotImplementedError
        The value is a real PROCESS branch this port has not written an occupant for;
        the recorded reason is in the message.
    ValueError
        The value is not one PROCESS has, or is a typo.
    """
    if value in registry:
        occupant = registry[value]
        return build(occupant) if build is not None else occupant()
    if (field, value) in UNPORTED:
        raise NotImplementedError(
            f"{field} == {value} is a real PROCESS branch but is not ported: "
            f"{UNPORTED[field, value]}"
        )
    raise ValueError(
        f"{field} == {value} is not a known value; this port has occupants for "
        f"{sorted(registry)} and records why it has none for "
        f"{sorted(v for f, v in UNPORTED if f == field)}"
    )


CONFINEMENT_TIME = {6: StellaratorConfinementTime}
"""`.stellarator.istell` -> the confinement-time occupant.

One entry, because this tree has one device. The tokamak arm (`istell == 0`, the bare
`ConfinementTime`) is in `UNPORTED`: it was never a real configuration here, only a
tokamak scaling law bolted onto stellarator geometry, coils and FWBS. `ConfinementTime`
remains the annotation on `PhysicsConfinementTime.model`, since
`StellaratorConfinementTime` subclasses it -- the family is real, and has one member.
"""

HEATING = {1: EcrhHeating, 2: LowhybHeating}
""".stellarator.isthtr` -> the auxiliary-heating occupant."""

FW_AREA = {0: AFwTotalNoPowerflow, 1: AFwTotalWithPowerflow}
"""`.heat_transport.ipowerflow` -> the first-wall-area occupant."""

PROFILE_PARAMETERISATION = {
    0: ProfileParameterisationParabolic,
    1: ProfileParameterisationPedestal,
}
"""`.physics.i_plasma_pedestal` -> the profile-shape occupant."""

BUILDING_SIZING = {
    0: Bldgs,
    1: functools.partial(BldgsSizes, i_hcd_primary=CurrentDriveModel.ITER_NEUTRAL_BEAM),
}
"""`.buildings.i_bldgs_size` -> the building-size occupant."""

TF_POWER = {0: TfPowerResistive, 1: TfPowerSuperconducting}
"""`.tfcoil.i_tf_sup` -> the TF-power occupant."""

ELECTRIC_PRODUCTION = {
    0: PowerProfilesOverTime,
    1: functools.partial(
        PlantElectricProductionReactor,
        itart=SphericalTokamakModel.CONVENTIONAL_ASPECT_RATIO,
        i_tf_sup=TFConductorModel.SUPERCONDUCTING,
        i_blkt_dual_coolant=BlanketDualCoolantModel.SINGLE_COOLANT_SOLID_BREEDER,
        i_p_coolant_pumping=PumpingPowerModelTypes.FRACTION_OF_HEAT,
    ),
}
"""`.costs.ireactor` -> the electric-production occupant."""

BLANKET_SHIELD_POWER = {
    1: BlanketShieldPowerExponential,
    2: DetailedPowerflowBlanketShieldPower,
}
"""`_blanket_shield_power_arm(blktmodel, ipowerflow)` -> the blanket/shield-power
occupant. Keyed by the **arm index** that function documents, never by a switch value."""

BLANKET_MASSES = {2: BlanketComponentMasses}
"""`_blanket_mass_arm(blktmodel, blkttype)` -> the blanket-mass occupant, same kind of
key."""


def _blanket_shield_power_arm(blktmodel: int, ipowerflow: int) -> int:
    """Which arm of `st_fwbs`'s blanket/shield-power dispatch a pair of switches selects.

    `stellarator.py:608-...`, transcribed:

    ```
    if blktmodel == 1:              -> arm 0   blanket_neutronics(); UNPORTED
    else:                           # blktmodel == 0
        if ipowerflow == 0:         -> arm 1   BlanketShieldPowerExponential
        else:                       -> arm 2   DetailedPowerflowBlanketShieldPower
    ```

    So `blktmodel` is the **outer** test and `ipowerflow` only distinguishes the two
    arms *inside* `blktmodel == 0`. Arm 2 is PROCESS's own default (`blktmodel = 0`,
    `ipowerflow = 1`) and the reference run.
    """
    if blktmodel == 1:
        return 0
    return 2 if ipowerflow == 1 else 1


def _blanket_mass_arm(blktmodel: int, blkttype: int) -> int:
    """Which arm of `st_fwbs`'s blanket-mass dispatch a pair of switches selects.

    `stellarator.py:1056-1091`, transcribed:

    ```
    if blktmodel == 0:
        if blkttype in {1, 2}:      -> arm 1   liquid breeder (WCLL/HCLL); UNPORTED
        else:                       -> arm 2   BlanketComponentMasses (solid breeder)
    else:                           # blktmodel == 1
                                    -> arm 0   sub-assembly thicknesses; UNPORTED
    ```

    Again `blktmodel` is the outer test; `blkttype` is consulted only inside
    `blktmodel == 0`. Arm 2 is PROCESS's own default (`blktmodel = 0`, `blkttype = 3`)
    and the reference run. `blkttype`'s values 1 and 2 select the identical formula, so
    they share arm 1.
    """
    if blktmodel != 0:
        return 0
    return 1 if blkttype in {1, 2} else 2


COST_MODEL = {0: Costs}
"""`.costs.i_cost_model` -> the cost-model occupant. `1` (KOVARI_2014, PROCESS's own
default) and `2` are both refused, with their reasons in `UNPORTED`; the slot used to
default to `None` for the first of them and no longer can."""

_INDAT_INTEGER = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*(\*.*)?$")


def switches_from_indat(input_file):
    """Every `name = <integer>` this input file sets, as a plain dict.

    Deliberately not a full IN.DAT parser: the only thing a machine is built from is
    integer switches, and PROCESS's own `SingleRun` is what reads everything else.
    A name the file never mentions is simply absent, which is what "falls through to the
    default" means.
    """
    text = Path(input_file).read_text()
    found = {}
    for line in text.splitlines():
        match = _INDAT_INTEGER.match(line)
        if match:
            found[match.group(1)] = int(match.group(2))
    return found


def machine_from_indat(input_file, stella_conf=None):
    r"""The `StellaratorProcess` an IN.DAT describes -- the only thing that builds one.

    Every slot below is passed explicitly because every slot below has no default: there
    is no `StellaratorProcess()` to apply deltas to any more. A switch the file does not
    mention still falls through to PROCESS's own default, but it falls through *here*,
    as the second argument to `switches.get`, where it is visible and cited.

    **The only place in this port an `i_*` integer is ever read.** Everything downstream
    sees a tree of model instances; nothing else has to know that `6` means Helias or
    that `blktmodel` and `ipowerflow` are consulted together.

    **Why assembly time is the only correct place for this, and not a preference.** Every
    switch PROCESS has is a constant for the whole solve:

        grep -n "\"i_\|'i_" process/core/solver/iteration_variables.py   -> no matches
        grep -n "\"i_\|'i_\|istell" process/core/scan.py                -> no matches

    No switch is an iteration variable and none is a scan variable, so no switch can
    change between two evaluations of one assembled graph, and `Scan` re-solves from
    scratch per point anyway. A switch therefore carries no derivative, participates in
    no edge, and has nothing to contribute to a `Graph` -- which is cottax's position
    that graph structure is decided by the caller once, not re-read per evaluation.

    The rejected alternative was one node owning the union of every variant's ports and
    branching internally. That would make a node read `eta_ecrh_injector_wall_plug` *and*
    `eta_lowhyb_injector_wall_plug` regardless of which is live, inventing graph edges
    that do not exist in the run being modelled, and would put a non-differentiable
    integer on a port. It also loses the result `Stellarator.fw_area` records: **a switch
    can decide whether the graph has a cycle**, which is not a fact any single fused node
    could express.

    Joint dispatch is ordinary code here rather than a mechanism: `blktmodel` is read
    together with `ipowerflow` for one slot and with `blkttype` for another, by
    `_blanket_shield_power_arm`/`_blanket_mass_arm`, each of which turns a pair of
    **legal switch values** into the **arm index** its registry is keyed on. No switch
    value is ever used as a key, and no switch has a default outside its own declared
    domain. So is cross-slot coherence -- `istell == 6` sets both the machine config and
    the confinement binding, because they are two consequences of one choice, which is
    why the two are resolved together, into named locals, before anything else.

    Raises
    ------
    NotImplementedError
        The file asks for a real PROCESS branch this port has no occupant for. **A file
        that sets nothing at all raises here**, on `istell`: PROCESS's own default is
        `0`, a tokamak, and this tree has none. `istell` has no usable default and that
        is the intent.
    """
    switches = switches_from_indat(input_file)

    def pick(field, registry, default, **kw):
        return _slot_occupant(field, switches.get(field, default), registry, **kw)

    # The device is resolved first, on its own, and both its consequences together:
    # PROCESS's own default is `istell = 0`, a tokamak, which is in `UNPORTED`, so a
    # file that never mentions `istell` is refused naming `istell` rather than whichever
    # slot the constructor happened to evaluate first.
    istell = switches.get("istell", 0)
    machine_config = _slot_occupant(
        "istell",
        istell,
        {6: StellaratorMachineConfig},
        build=lambda cls: cls(
            machine_config=read_stellarator_config_file(
                REFERENCE_STELLA_CONF if stella_conf is None else stella_conf
            )
        ),
    )
    confinement_time = _slot_occupant(
        "istell",
        istell,
        CONFINEMENT_TIME,
        build=lambda cls: cls(
            i_confinement_time=ConfinementTimeModel.ISS04_STELLARATOR,
            i_rad_loss=ConfinementRadiationLossModel.CORE_ONLY,
            i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
        ),
    )
    # The two joint dispatches, resolved into named locals before the constructor call
    # for the same reason `istell` is: so the *first* thing a refused combination
    # reports is the one the caller asked for, not whichever slot Python evaluated
    # first. Every default here is PROCESS's own -- `fwbs_variables.py:479` for
    # `blktmodel`, `:494` for `blkttype`, `heat_transport_variables.py:94` for
    # `ipowerflow` -- and the switch *values* are turned into **arm indices** by the two
    # named functions above, which is the only thing the registries are keyed on.
    #
    # This used to read `blktmodel = switches.get("blktmodel", 2)` and pass that value
    # through where an arm index was wanted. `2` is not a legal `blktmodel` at all: it
    # was a sentinel meaning "not set", picked so the reference run happened to land on
    # arm 2. The consequence was an inverted mapping -- stating PROCESS's own default
    # `blktmodel = 0` was refused, while `blktmodel = 1` (KIT HCPB neutronics) silently
    # assembled `BlanketShieldPowerExponential`, a node written for a different switch's
    # arm. That is the `ScTfCoilNuclearHeating` bug class, reintroduced by a key
    # derivation instead of by a registration.
    blktmodel = switches.get("blktmodel", 0)
    blanket_shield_power = _slot_occupant(
        "blktmodel_ipowerflow",
        _blanket_shield_power_arm(blktmodel, switches.get("ipowerflow", 1)),
        BLANKET_SHIELD_POWER,
    )
    blanket_masses = _slot_occupant(
        "blktmodel_blkttype",
        _blanket_mass_arm(blktmodel, switches.get("blkttype", 3)),
        BLANKET_MASSES,
    )
    return StellaratorProcess(
        costs=pick("i_cost_model", COST_MODEL, 1),
        stellarator=Stellarator(
            machine_config=machine_config,
            heating=pick("isthtr", HEATING, 1),
            fw_area=pick("ipowerflow", FW_AREA, 1),
            fwbs=StellaratorFwbs(
                blanket_shield_power=blanket_shield_power,
                blanket_masses=blanket_masses,
            ),
        ),
        physics=Physics(
            profiles=PhysicsProfiles(
                parameterisation=pick("i_plasma_pedestal", PROFILE_PARAMETERISATION, 1),
            ),
            confinement_time=PhysicsConfinementTime(model=confinement_time),
        ),
        power=Power(tf_power=pick("i_tf_sup", TF_POWER, 1)),
        buildings=Buildings(sizing=pick("i_bldgs_size", BUILDING_SIZING, 0)),
        availability=Availability(
            electric_production=pick("ireactor", ELECTRIC_PRODUCTION, 1)
        ),
    )


REFERENCE_MACHINE = machine_from_indat(REFERENCE_INPUT_FILE)
"""The machine `stellarator_helias.IN.DAT` describes -- the run this port is validated
against (`istell = 6`, `i_plasma_pedestal = 0`, `i_cost_model = 0`; every other switch at
PROCESS's own default)."""


def graph_for(machine=None):
    """The assembled graph for one machine; `REFERENCE_MACHINE` if unstated.

    **There is no bare form to fall back to any more, and that is the point.** Five
    registration bugs here shared one root cause: a value copied from PROCESS's bare
    `*_variables.py` defaults rather than from the run modelled (`i_confinement_time` 34
    vs 38, `i_thermal_electric_conversion` 0 vs 2, `i_p_coolant_pumping` 2 vs 1,
    `i_plasma_ignited` 0 vs 1, and `i_cost_model` 1 vs 0, which left `.costs.coe` with no
    producer and 43 nodes unregistered), each found by the MDA harness after the fact.
    The first fix was to stop *defaulting* to the silent-IN.DAT graph; this argument's
    default has been `REFERENCE_MACHINE` since. The second is that the silent-IN.DAT
    graph can no longer be built at all -- `StellaratorProcess()` raises, because every
    switched slot lost its default -- so a machine comes from an IN.DAT or from an
    explicit `eqx.tree_at` on one, and from nowhere else.
    """
    return to_graph(REFERENCE_MACHINE if machine is None else machine)


GRAPH = graph_for()
"""`REFERENCE_MACHINE`'s graph -- the `stellarator_helias.IN.DAT` run this port is
validated against."""

if __name__ == "__main__":
    n_vars = sum(
        len(node.inputs) + len(node.outputs) for node in GRAPH.definitions.values()
    )
    print(f"{len(GRAPH.definitions)} nodes, {n_vars} ports (inputs + outputs, unmerged)")
    for name, node in GRAPH.definitions.items():
        print(f"  {name.path_str()}: {len(node.inputs)} in, {len(node.outputs)} out")

    print("\ncycles, per machine:")
    for label, machine in (
        ("the reference machine", REFERENCE_MACHINE),
        (
            # Both slots `ipowerflow` decides, not just `fw_area`: it also picks arm 1
            # of the blanket/shield-power dispatch. Swapping one and not the other used
            # to be the only spelling available, because arm 1 was unreachable through
            # `machine_from_indat` at all -- the joint key was derived from an illegal
            # `blktmodel` sentinel. It is reachable now, and this what-if says
            # `ipowerflow = 0` coherently.
            "ipowerflow = 0",
            eqx.tree_at(
                lambda m: (
                    m.stellarator.fw_area,
                    m.stellarator.fwbs.blanket_shield_power,
                ),
                REFERENCE_MACHINE,
                (AFwTotalNoPowerflow(), BlanketShieldPowerExponential()),
            ),
        ),
    ):
        cycles = graph_for(machine).cycles
        print(f"  {label}: {[[n.path_str() for n in c] for c in cycles] or 'acyclic'}")
