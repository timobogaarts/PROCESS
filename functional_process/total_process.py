"""The running graph assembly of every ported stellarator unit.

Imports each ported unit's `cottax` node declaration and assembles them into one
`Graph` via `to_graph`. Run directly for a smoke check (builds the graph, prints its
node/variable count); `render_xdsm.py` imports `GRAPH` from here to draw it.

This is the whole graph **as it currently exists**, not a claim that the stellarator MDA
is assembled: most nodes here are still islands with unowned (external) reads, since
their producers haven't been ported yet. It exists so there is always one place the next
ported unit joins, and one place to point a visual inspection at. See
`_audit/unit_registry.md`'s "Ported so far" for what is and isn't in it.

**There is no single graph.** A topology-changing switch selects which nodes exist, so
what this module exports is `build_graph(configuration)`; `GRAPH` is the one PROCESS's
own switch defaults produce, kept as a module-level name because `render_xdsm.py` and the
smoke check want a default to point at. `TOPOLOGY_SWITCHES` below enumerates the arms,
and `configuration.py` explains why assembly time is the only correct place to resolve
them (short version: no switch in PROCESS is ever an iteration variable or a scan
variable, so no switch can change between two evaluations of one assembled graph).

`EcrhDensityLimit(i_plasma_pedestal=0)` is deliberately *not* a `Switch` here. It is
`naming_convention.md`'s other category -- a formula-changing switch kept as a static
kwarg on one node's `fn` -- because `i_plasma_pedestal != 0` has no formula at all in
`density_limits.py` and no node's existence depends on it.

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
its `Input`s (`.tfcoil.t_helium`/`b_max`) are per-sample locals of
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
(unconditional-in-`COMMON` was wrong for PROCESS's own default configuration, same class
of bug already fixed once for `EcrhDensityLimit`).
"""

from pathlib import Path

from functional_process.configuration import (
    Alternative,
    Configuration,
    Switch,
    build_graph,
)
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
    PfCoilPowerConditioningCost,
    PfMagnetCost,
    PowerConditioningCost,
    PowerInjectionCost,
    ReactorCoolingSystemCost,
    ReactorCost,
    ReactorStructureCost,
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
from functional_process.models.physics.confinement_time import (
    ConfinementTime,
    DoubleAndTripleProduct,
    StellaratorConfinementTime,
)
from functional_process.models.physics.exhaust import RadiationFraction
from functional_process.models.physics.fusion_reactions import (
    FusionRates,
    SetFusionPowers,
)
from functional_process.models.physics.physics_A_pure_formulas import (
    AuxiliaryPhysicsQuantities,
    ElectronThermalEnergy,
    FastAlphaBeta,
    IonElectronEquilibration,
    IonThermalEnergy,
    TotalPlasmaHeatingPower,
)
from functional_process.models.physics.physics_B_composition import (
    CalculateEffectiveChargeIonisationProfiles,
    PlasmaComposition,
)
from functional_process.models.physics.physics_C_outplas import (
    DimensionlessPlasmaParameters,
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
from functional_process.models.physics.radiation_power import (
    ImpurityRadiationTotals,
    PlasmaRadiationPowers,
    SynchrotronRadiationPower,
)
from functional_process.models.power_A_tf_coil_power import (
    TfPowerResistive,
    TfPowerSuperconducting,
)
from functional_process.models.power_B_thermal_cryo import (
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
from functional_process.models.power_C_electric_production import (
    Acpow,
    PlantElectricProductionReactor,
    PowerProfilesOverTime,
)
from functional_process.models.stellarator.build import (
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
    # `BlktmodelBlanketThickness` deliberately NOT imported/registered here any more --
    # see `COMMON`'s own comment next to `Build`, and `unit_registry.md` row 2: PROCESS's
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
from functional_process.models.stellarator.preset_config import (
    StellaratorMachineConfig,
    read_stellarator_config_file,
)
from functional_process.models.stellarator.stellarator_B_st_phys import (
    FusionPowerTotalsMw,
    ClippedRadiationPowers,
    FusionTotalsNoBeam,
    HeatingAndRadiationPower,
    NeutronWallLoad,
    PoloidalFieldFromRotationalTransform,
    RadiatedWallLoadAndFraction,
    StellaratorBetaAndStoredEnergy,
    ThermalEnergyTotals,
    TotalField,
)
from functional_process.models.stellarator.stellarator_C_geometry import (
    DefaultAspectRatio,
    StellaratorPlasmaGeometry,
    StellaratorScalingFactors,
)
from functional_process.models.stellarator.stellarator_D_structure import (
    StructureMasses,
)
from functional_process.models.stellarator.stellarator_F_tf_nuclear_heating import (
    ScTfCoilNuclearHeating,
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
from functional_process.models.vacuum import DuctDiameterRootFind, VacuumOld

COSTS_1990 = (
    # Registry unit #18, `models/costs/costs.py` -- the whole `.costs.coe` chain, 42 of
    # `Costs`'s 43 computational methods (`acc2211`..`coelc` plus the two accumulations
    # `Costs.run()` performs inline). This is the `.costs.i_cost_model == 0`
    # (`CostModels.PROCESS_1990`) arm of the switch below; see there for why it is an
    # arm rather than an unconditional `COMMON` entry, and `costs.md`'s coverage map for
    # the per-method derivation of this list.
    #
    # Every static kwarg below is checked against the modelled run on every harness run
    # by `mda_harness.switch_audit`, so none of them can silently drift the way
    # `i_confinement_time`/`i_thermal_electric_conversion`/`i_p_coolant_pumping` did
    # (`next_steps.md` §8.2). Their values here are PROCESS's own bare defaults *and*
    # the reference run's, except `iohcl` -- flagged individually below.
    ConvertFpyToCalendar,
    StructuresCost,  # Account 21
    # `ife=0` (`ife_variables.py:253`). Each of these five nodes has a real PROCESS
    # `ife == 1` arm that reads an entirely different set of `.ife.*` fields (2-D
    # material-mass arrays for the three Account-221 nodes, driver-cost tables for
    # Account 223, extra cooling loads for 2262, a target-mass model for 2272); none of
    # `.ife.*` is ported, so the ported functions refuse that value rather than
    # returning a magnetic-confinement number for an IFE device.
    FirstWallCost(ife=0),  # Account 221.1
    BlanketCost(ife=0),  # Account 221.2
    ShieldCost(ife=0),  # Account 221.3
    ReactorStructureCost,  # Account 221.4
    DivertorCost,  # Account 221.5
    ReactorCost,  # Account 221 total
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
    TfMagnetCostSuperconducting(supercond_cost_model=0),  # Account 222.1
    # `n_cs_pf_coils=0` (`pfcoil_variables.py:323`) and `iohcl=0` are the two loop
    # bounds of `acc2222`, which `costs.md` originally recorded as a structural JAX
    # blocker ("dynamic-length loop"). They are not dynamic: neither is an iteration
    # variable or a scan variable, so both are graph-assembly-time facts and the loops
    # unroll at trace time -- the same treatment `ImpurityRadiationTotals.imp_indices`
    # already gets. **`iohcl=0` is the one deliberate deviation from a PROCESS default
    # in this tuple** (`build_variables.py:177` says `1`): the reference stellarator run
    # has no central solenoid, and `switch_audit` confirms `.build.iohcl == 0` on it.
    # With `n_cs_pf_coils == 0` the default `iohcl = 1` would compute `npf = -1` and
    # then index `r_pf_coil_middle[-1]`, i.e. price a central solenoid out of the last
    # (unset) array slot -- reproduced faithfully by the port, but not what this run
    # does.
    PfMagnetCost(
        n_cs_pf_coils=0, iohcl=0, i_pf_conductor=0, supercond_cost_model=0
    ),  # Account 222.2
    VacuumVesselAssemblyCost,  # Account 222.3
    MagnetsCost,  # Account 222 total
    PowerInjectionCost(ife=0),  # Account 223
    VacuumSystemCost,  # Account 224
    TfCoilPowerConditioningCost,  # Account 225.1
    PfCoilPowerConditioningCost,  # Account 225.2
    # `i_pulsed_plant=0`/`istore=1` (`pulse_variables.py:30`/`:16`). `istore == 3` is
    # unported (a third reads-set: `.heat_transport.p_plant_primary_heat_mw`,
    # `.times.t_plant_pulse_no_burn`, `.pulse.dtstor`) and unreachable here, since a
    # steady-state plant never enters the `istore` dispatch at all.
    EnergyStorageCost(i_pulsed_plant=0, istore=1),  # Account 225.3
    PowerConditioningCost,  # Account 225 total
    ReactorCoolingSystemCost,  # Account 2261
    AuxiliaryComponentCoolingCost(ife=0),  # Account 2262
    CryogenicSystemCost,  # Account 2263
    HeatTransportSystemCost,  # Account 226 total
    FuellingSystemCost,  # Account 2271
    FuelProcessingCost(ife=0),  # Account 2272 -- also the sole producer of
    # `.physics.wtgpd`, the one field `costs.py` writes outside `.costs.*`.
    AtmosphericRecoveryCost,  # Account 2273
    NuclearBuildingVentilationCost,  # Account 2274
    FuelHandlingCost,  # Account 227 total
    InstrumentationAndControlCost,  # Account 228
    MaintenanceEquipmentCost,  # Account 229
    FusionPowerIslandCost,  # Account 22 total
    TurbinePlantEquipmentCost,  # Account 23
    SwitchyardCost,  # Account 241
    TransformersCost,  # Account 242
    LowVoltageCost,  # Account 243
    DieselGeneratorsCost,  # Account 244
    AuxiliaryFacilityPowerCost,  # Account 245
    ElectricPlantEquipmentCost,  # Account 24 total
    MiscPlantEquipmentCost,  # Account 25
    HeatRejectionCost,  # Account 26
    TotalPlantDirectCost,  # `.costs.cdirt`, inline in `Costs.run()`
    IndirectCosts,  # Account 9
    ConstructedCost,  # `.costs.concost`, inline in `Costs.run()`
    # `ireactor=1`/`ipnet=0` (`cost_variables.py:521`/`:515`) are preconditions, not
    # ports: `Costs.run()` calls `coelc()` only when both hold (`costs.py:82-83`), and
    # the node refuses any other pair rather than producing a `.costs.coe` PROCESS would
    # have left at its previous value. `itart=0` (`physics_variables.py:994`) selects
    # the no-centrepost arm; unlike the `ife`/`i_tf_sup` splits above, *both* `itart`
    # arms are implemented in one function, so this node reads
    # `.costs.cpstcst`/`cplife_cal`/`cplife` either way -- a deliberate size-aware
    # deviation from `traceability_policy.md`'s split default, argued in the function's
    # own docstring.
    CostOfElectricity(ife=0, itart=0, ireactor=1, ipnet=0),  # `.costs.coe`
)
"""`costs.py`'s 1990 cost model: the `.costs.i_cost_model == 0` arm, 43 nodes."""

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

TOPOLOGY_SWITCHES = (
    # `.stellarator.istell` -- PROCESS's master pipeline switch (tokamak / stellarator
    # / IFE), `core/solver/switches.md` § `data.stellarator.istell`. Only ONE block is
    # filed under it here: `ConfinementTime`'s 20th read. That is not the switch's full
    # fan-out (switches.md counts 44 sites) -- it is the part this graph currently has
    # two real arms for, and the switch is declared rather than the binding hardcoded
    # because changing which node produces a read changes an edge, which is this
    # module's own criterion for a topology switch (`configuration.py`'s docstring).
    #
    # PROCESS names that parameter `q95` and the tokamak caller passes `.physics.q95`,
    # but the stellarator caller passes `.stellarator.iotabar` into the same positional
    # slot (`process/models/stellarator/stellarator.py:2312` against the signature at
    # `process/models/physics/confinement_time.py:79`) -- and ISS04 consumes it as
    # `iotabar**0.41`. Registering the base class unconditionally therefore fed the
    # stellarator scaling law a safety factor. Found by `mda_harness.py` and confirmed
    # arithmetically: `q95 = 1.03`, `iotabar = 1.0`, `1.03**0.41 = 1.01219284...` against
    # the harness's reported `rel_diff` of `1.219e-02`.
    #
    # `default=0` is PROCESS's own (`stellarator_variables.py:46`), per the
    # `Switch.default` contract -- so the *bare* `GRAPH` keeps the tokamak binding.
    # **Every stellarator consumer must pass
    # `Configuration({".stellarator.istell": 6})`**; the MDA harness
    # does (`run_mda_harness.py`), matching `stellarator_helias.IN.DAT:137`. Note the
    # rest of this module's registrations are stellarator models regardless of this
    # switch -- the bare-default graph was already device-incoherent in that sense and
    # this switch does not fix that, it just stops one binding from being silently wrong.
    Switch(
        path=".stellarator.istell",
        default=0,  # `stellarator_variables.py:46`
        alternatives=(
            Alternative(
                value=0,
                declarations=(
                    ConfinementTime(
                        i_confinement_time=38, i_rad_loss=1, i_plasma_ignited=1
                    ),
                ),
            ),
            Alternative(
                value=6,
                declarations=(
                    StellaratorConfinementTime(
                        i_confinement_time=38, i_rad_loss=1, i_plasma_ignited=1
                    ),
                    # `preset_config.py` (unit #8). The 34 numeric
                    # `.stellarator_config.stella_config_*` fields, read from
                    # `REFERENCE_STELLA_CONF` at assembly time and owned by a node with
                    # **no inputs** -- the machine config is strictly upstream of every
                    # design variable, so this adds a source to the DAG and no cycle.
                    #
                    # Filed under `istell` rather than in `COMMON` because it is
                    # genuinely `istell`-gated: `istell == 0` is a tokamak and has no
                    # `stella_config_*` at all, so registering it unconditionally would
                    # make the tokamak graph produce stellarator machine data. Arms 1..5
                    # stay `unported` (their five preset tables are not transcribed);
                    # `select_stellarator_config_scalars` is already generic over any
                    # config mapping, so porting them is transcription, not design.
                    #
                    # **This is what makes the graph runnable from a cold
                    # `DataStructure`.** Before it, these 34 fields were unowned boundary
                    # inputs seeded from a converged run; stepped cold they were all
                    # `0.0`, `.tfcoil.n_tf_coils` (=`coilspermodule * symmetry`) was `0`,
                    # and the first division by it made 16 blocks emit non-finite values.
                    StellaratorMachineConfig(
                        machine_config=read_stellarator_config_file(
                            REFERENCE_STELLA_CONF
                        )
                    ),
                ),
            ),
            *(
                Alternative(
                    value=value,
                    unported=(
                        "`istell` in 1..5 selects one of five hardcoded machine presets "
                        "(Helias 5/4/3, W7-X 30/50) copied onto "
                        "`StellaratorConfigData` by "
                        "`preset_config.py`'s reflective `hasattr`/`setattr` loop; only "
                        "`istell == 6` (config read from file) is in scope. See "
                        "`core/solver/switches.md` § `data.stellarator.istell` -- "
                        "second role, whose disposition is still open in "
                        "`_audit/next_steps.md` § 2. "
                        "The confinement-time binding itself would be identical to the "
                        "`value=6` arm; it is the surrounding preset data that is "
                        "unported"
                    ),
                )
                for value in (1, 2, 3, 4, 5)
            ),
        ),
    ),
    Switch(
        path=".stellarator.isthtr",
        default=1,  # `stellarator_variables.py:87`
        alternatives=(
            Alternative(value=1, declarations=(EcrhHeating,)),
            Alternative(value=2, declarations=(LowhybHeating,)),
            Alternative(
                value=3,
                unported=(
                    "the NBI branch of `st_heat` calls `current_drive.culnbi()`, a "
                    "model that is not audited yet (registry unit #5)"
                ),
            ),
        ),
    ),
    Switch(
        path=".heat_transport.ipowerflow",
        default=1,  # `heat_transport_variables.py:94`
        alternatives=(
            Alternative(value=0, declarations=(AFwTotalNoPowerflow,)),
            Alternative(value=1, declarations=(AFwTotalWithPowerflow,)),
        ),
    ),
    Switch(
        path=".physics.i_plasma_pedestal",
        default=1,  # `physics_variables.py:889`
        alternatives=(
            Alternative(
                value=0,
                declarations=(
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
                    EcrhDensityLimit(i_plasma_pedestal=0),
                    ParabolicTemperatureProfile,
                    ParabolicOnAxisDensities,
                    ParabolicOnAxisTemperatures,
                    ParabolicGradientLengths,
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
                    ParabolicProfileValues,
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
                    LModeProfileReset,
                ),
            ),
            Alternative(
                value=1,
                declarations=(
                    PedestalTemperatureProfile,
                    PedestalOnAxisDensities,
                    PedestalOnAxisTemperatures,
                    # No pedestal-arm counterpart to `EcrhDensityLimit`: PROCESS's own
                    # default configuration (`i_plasma_pedestal == 1`) never computes a
                    # real ECRH density limit at all -- see the note on the value-0 arm.
                    # `dlimit_ecrh`/`bt_max_ecrh` are therefore genuinely unproduced in
                    # this arm, not merely unported.
                ),
            ),
        ),
    ),
    Switch(
        path=".buildings.i_bldgs_size",
        default=0,  # `buildings_variables.py:206`, `BuildingsModel.ITER_1992`
        alternatives=(
            Alternative(value=0, declarations=(Bldgs,)),  # ITER_1992
            Alternative(
                value=1,  # CHAPMAN_2024
                declarations=(
                    # `.current_drive.i_hcd_primary` resolved outside the traced
                    # function (an enum lookup) -- static kwarg, not an `Input`. Default
                    # `5` per `current_drive_variables.py:190`, same move as
                    # `ConfinementTime`'s static switches.
                    BldgsSizes(i_hcd_primary=5),
                ),
            ),
        ),
    ),
    Switch(
        # Synthetic, deliberately not a single real `.area.field` string (see
        # `configuration.py`'s own note that `path` is a lookup key, not a `VarPath`):
        # S2 (`blanket_shield_tf_nuclear_power`, `stellarator_E_fwbs_synthesis.md`) is a
        # genuine *joint* dispatch on two independent PROCESS switches at once --
        # `.fwbs.blktmodel == 1` selects one arm outright; `!= 1` (any other value)
        # selects a second dispatch on `.heat_transport.ipowerflow` -- not two nested
        # binary `Switch`es, because the two `!= 1` arms are exclusive alternatives of
        # each other (see below) in exactly the sense `check_arms_are_exclusive` checks,
        # while `blktmodel == 1` vs. `!= 1` is a different axis entirely. `value` below
        # is an arm index, not a literal `blktmodel`/`ipowerflow` value -- documented in
        # each `Alternative`'s own comment.
        path=".fwbs.blktmodel,.heat_transport.ipowerflow",
        default=2,  # blktmodel=0 (`fwbs_variables.py:479`) != 1, ipowerflow=1 (default
        # above) -- PROCESS's real default configuration is arm 3.
        alternatives=(
            Alternative(
                value=0,  # blktmodel == 1
                unported=(
                    "S2's blktmodel == 1 arm is `blanket_neutronics()`, which calls "
                    "`self.hcpb.nuclear_heating_blanket()`/`nuclear_heating_shield()` "
                    "with zero arguments against 2-/7-keyword-argument @staticmethods "
                    "-- a live PROCESS bug that would TypeError the moment this arm "
                    "actually executes (unit_registry.md row 13, next_steps.md §3). "
                    "hcpb.py's own 3 ported nodes (NuclearHeatingBlanket/Shield/"
                    "Magnets) exist but are not usable here until that call site has a "
                    "resolution."
                ),
            ),
            Alternative(
                value=1,  # blktmodel != 1, ipowerflow == 0 -- the "old model"
                declarations=(
                    ExponentialAttenuationBlanketShieldPower,
                    # `ScTfCoilNuclearHeating` moved here from `COMMON` -- this arm is
                    # its one genuine caller that keeps its `p_tf_nuclear_heat_mw`
                    # output (`stellarator.py:727-728`); the `blktmodel == 1` arm calls
                    # it too but discards that particular output, and arm 3 (value=2,
                    # below) computes its own, different `p_tf_nuclear_heat_mw` formula.
                    # Leaving it unconditionally in `COMMON` was a real bug, the same
                    # class already fixed once for `EcrhDensityLimit`: PROCESS's actual
                    # default configuration lands in arm 3 below, not this one, so the
                    # old unconditional placement was computing SC-coil TF nuclear
                    # heating via the wrong formula for the default `GRAPH`.
                    ScTfCoilNuclearHeating,
                ),
            ),
            Alternative(
                value=2,  # blktmodel != 1, ipowerflow == 1 -- the "new model" (default)
                declarations=(DetailedPowerflowBlanketShieldPower,),
            ),
        ),
    ),
    Switch(
        # Second synthetic joint key, same shape and for the same reason as S2's above:
        # `st_fwbs` S4's blanket-mass block (`stellarator.py:1056-1181`,
        # `stellarator_fwbs_s4.md`) dispatches on `.fwbs.blktmodel` *and*, nested inside
        # its `== 0` arm, on `.fwbs.blkttype`. It cannot join S2's switch (that one's
        # arms are indexed by `ipowerflow`, which this block never reads, so this node
        # would have to be duplicated across two of its arms), and it does not factor
        # into two independent real-field switches either: `.fwbs.m_blkt_total` is
        # accumulated *across* both axes (breeder part chosen by `blkttype`, then
        # steel + vanadium added, both inside `blktmodel == 0`), so a `blktmodel`-only
        # node plus a `blkttype`-only node would need a third node invented to own a
        # sum PROCESS writes as two statements in one straight line. `value` is an arm
        # index, not a literal field value. The full write-up, including the rejected
        # alternatives, is in `stellarator_fwbs_s4.md` § "registration".
        #
        # Known gap, recorded rather than hidden: `.fwbs.blktmodel` is now an axis of
        # two synthetic keys, and `configuration.py` cannot check that a caller keeps
        # them consistent about it -- arm indices on a comma-joined path are opaque to
        # it. Both switches default to the `blktmodel = 0` arm and S2's
        # `blktmodel == 1` arm is `unported` (so it raises first), so no assemblable
        # configuration is affected today. See `stellarator_fwbs_s4.md`'s open
        # question 1.
        path=".fwbs.blktmodel,.fwbs.blkttype",
        default=2,  # blktmodel = 0 (`fwbs_variables.py:479`) and blkttype = 3 (`:494`),
        # neither set by `stellarator_helias.IN.DAT` -- so PROCESS's own default *and*
        # the reference run both land on arm 2.
        alternatives=(
            Alternative(
                value=0,  # blktmodel != 0
                unported=(
                    "the blktmodel != 0 blanket-mass arm (stellarator.py:1093-1181) "
                    "computes m_blkt_steel_total/m_blkt_beryllium from six .build."
                    "bl{u,m,p}{i,o}th sub-assembly thicknesses, additionally writes "
                    ".fwbs.whtblbreed and .fwbs.f_a_blkt_cooling_channels, and writes "
                    "neither .fwbs.m_blkt_li2o nor .fwbs.m_blkt_vanadium at all -- a "
                    "different node with a different port set, not written yet. "
                    "Refused rather than assembled empty: BlanketCost reads all four "
                    "masses unconditionally, so an empty arm would silently hand it "
                    "boundary values for fields PROCESS does compute on that arm."
                ),
            ),
            Alternative(
                value=1,  # blktmodel == 0, blkttype in {1, 2} -- liquid breeder
                unported=(
                    "the liquid-breeder sub-arm (blkttype in {1, 2}, WCLL/HCLL, "
                    "stellarator.py:1058-1066) writes .fwbs.wtbllipb and .fwbs."
                    "m_blkt_lithium in place of .fwbs.m_blkt_li2o/.m_blkt_beryllium -- "
                    "different fields, not a different formula for the same ones. Not "
                    "ported: neither replacement field has a reader in this graph, and "
                    "stellarator_helias.IN.DAT leaves blkttype at its default of 3. "
                    "Values 1 and 2 select the identical formula (the same 'three "
                    "values, two arms' shape as .tfcoil.i_tf_sup's {0, 2}), so this "
                    "one arm covers both; there is no separate value=2 alternative "
                    "because there is no separate behaviour to name."
                ),
            ),
            Alternative(
                value=2,  # blktmodel == 0, blkttype not in {1, 2} -- solid breeder
                # (HCPB), PROCESS's default and the reference run's arm
                declarations=(BlanketComponentMasses,),
            ),
        ),
    ),
    Switch(
        path=".tfcoil.i_tf_sup",
        default=1,  # `tfcoil_variables.py:261`, `TFConductorModel.SUPERCONDUCTING`
        alternatives=(
            # `Power.tfpwr` dispatches on `i_tf_sup != 1` only -- values 0 (resistive
            # copper) and 2 (aluminium) both run the identical `calculate_tf_power_
            # resistive` node, not two separate formulas. Declaring both as ordinary
            # `Alternative`s (same `value=0` node set under two different literal
            # values) is exactly what `test_configuration.py::
            # test_arms_select_different_node_sets` exists to reject -- a switch whose
            # arms don't pick different node sets belongs in the other
            # `naming_convention.md` category, a static kwarg, not a `Switch` value.
            # Since `i_tf_sup` genuinely does need a real `Switch` for value 1 vs.
            # {0, 2}, value 2 is marked `unported` pointing at value 0's identical
            # result rather than duplicated -- `.tfcoil.i_tf_sup == 0` assembles the
            # correct (and only) resistive-arm graph for either literal value.
            Alternative(value=0, declarations=(TfPowerResistive,)),
            Alternative(value=1, declarations=(TfPowerSuperconducting,)),
            Alternative(
                value=2,
                unported=(
                    "aluminium TF (i_tf_sup == 2) runs the identical "
                    "calculate_tf_power_resistive branch as i_tf_sup == 0 -- "
                    "`Power.tfpwr` dispatches on `i_tf_sup != 1` only, one formula for "
                    "both. Request `.tfcoil.i_tf_sup == 0` instead; it assembles the "
                    "same node set. Kept as a separate, unported value rather than a "
                    "second Alternative pointing at TfPowerResistive so this switch "
                    "does not fail test_arms_select_different_node_sets."
                ),
            ),
        ),
    ),
    # `.costs.i_cost_model` -- which cost model runs at all. Resolved in exactly one
    # place in PROCESS, `process/main.py`'s `Models.costs` `@property` (lines 745-764),
    # which picks a whole `Model` instance *before* any model runs and injects it;
    # neither `costs.py` nor `costs_2015.py` reads the switch internally (grepped, zero
    # hits in either), and `stellarator.py` only ever calls `self.costs.run()` on
    # whatever was injected (`stellarator.py:176`, the solve-time call). So this is a
    # textbook topology switch -- `_audit/schema.md`'s own switch template names it as
    # the precedent -- and the two arms are genuinely disjoint subgraphs: `costs.py`
    # writes 114 distinct `.costs.*` fields, `costs_2015.py` writes 4 `.costs_2015.*`
    # scalars plus a 100-slot `s_*` array, and the *only* two `VarPath`s both write are
    # `.costs.coe` and `.costs.concost`.
    #
    # **What changed since `costs.md`/`unit_registry.md` last recorded this as "not
    # wireable yet".** That reasoning was right on its own terms and is now obsolete on
    # both counts. (a) It said the two ported *subsets* shared no output, so
    # `check_arms_are_exclusive` would reject the pairing -- true when only 23 leaf
    # sub-accounts were ported, no longer relevant now that this arm owns
    # `.costs.coe`/`.costs.concost` themselves and `costs_2015.py` has no arm to pair
    # against at all (a single-ported-arm switch has no pair to check). (b) It said
    # unconditional registration "would reproduce the `EcrhDensityLimit` bug class,
    # since `i_cost_model = 1` by default means `costs.py`'s `Costs` model never even
    # runs" -- correct about PROCESS's *bare* default, but that is not the run this
    # project validates against: `tests/regression/input_files/
    # stellarator_helias.IN.DAT:229` selects `i_figure_merit = 6`
    # (`FiguresOfMerit.COST_OF_ELECTRICITY`, i.e. minimise `.costs.coe`) and line 248
    # sets `i_cost_model = 0` explicitly, with the file's own comment: "0: 1990 cost
    # module, the 2015 does not work yet for stellarators". So `costs.py` is the right
    # arm for every stellarator run in scope, and it is the arm PROCESS itself insists
    # on for them.
    #
    # **Why the default arm is `unproduced` rather than `unported`.** `GRAPH` is built
    # at import time from the bare default configuration (bottom of this module), so an
    # `unported` default arm would raise `NotImplementedError` on `import
    # functional_process.total_process` and break the package. Three ways out were
    # weighed:
    #
    #   1. *Register the 43 nodes unconditionally in `COMMON`.* Rejected: the default
    #      `GRAPH` would then compute `.costs.coe` with the 1990 model on a
    #      configuration where PROCESS computes it with the 2015 one. That is worse than
    #      the `EcrhDensityLimit`/`WardTaylorAvailability` bug class the registry warns
    #      about twice -- those computed a value the default configuration never
    #      computes; this would compute a *different number for the same field*.
    #   2. *Change what `graph_for()`'s bare default means* (e.g. `GRAPH =
    #      graph_for(REFERENCE_CONFIGURATION)`, or making `GRAPH` lazy so the bare
    #      default is allowed to fail). Structurally the most honest answer -- asking
    #      for PROCESS's bare-default configuration *should* fail, because it is not
    #      fully ported -- but it changes a shared contract (`Switch.default`'s "a
    #      silent IN.DAT reproduces PROCESS's own defaults", pinned by
    #      `test_default_configuration_matches_process_defaults`) and every one of
    #      `test_configuration.py`'s per-switch parametrised assemblies, `test_mda.py`,
    #      `mda.py`'s default argument and `render_xdsm.py` would have to carry a
    #      configuration. Left on the table deliberately, not taken here: it is the
    #      user's call, not a change to make as a side effect of porting `coelc`.
    #   3. *An arm that assembles as empty* -- taken. It says exactly what is true: in
    #      the `KOVARI_2014` configuration this port computes no cost of electricity at
    #      all, so `.costs.coe` has no producer and any consumer of it shows up as an
    #      unowned (boundary) input rather than being silently handed the 1990 number.
    #      The default `GRAPH` is byte-for-byte the graph it was before this arm
    #      existed, so nothing that depended on it moves.
    #
    # See `configuration.py`'s `Alternative` docstring for the `unported`/`unproduced`
    # distinction, and `_audit/next_steps.md` §9 for the write-up.
    Switch(
        path=".costs.ireactor",
        default=1,  # `cost_variables.py:521`
        alternatives=(
            # `ireactor == 0`: PROCESS does *not* compute the five electric-production
            # fields at all (`process/models/power.py`'s `plant_electric_production`
            # guards them with `if ireactor == 1`), so on this arm they keep whatever
            # they entered the call with and only the *profiles* are produced. That is
            # exactly `PowerProfilesOverTime`, which reads the two carried-over values
            # (`p_plant_electric_gross_mw`, `p_plant_electric_net_mw`) as boundary
            # inputs -- the honest shape for an arm where nothing in the graph produces
            # them.
            Alternative(value=0, declarations=(PowerProfilesOverTime,)),
            # `ireactor == 1`: the five are computed before use, so the entering values
            # are dead and the node is ordinary and acyclic. See
            # `PlantElectricProductionReactor`'s own docstring for why
            # `PlantElectricProduction` itself is not registerable and why splitting on
            # this switch (rather than a `FixedPointFunction`, the treatment
            # `power_B_thermal_cryo.py`'s six `*Step` nodes get) is the correct answer
            # here: this is not a fixed point, it is a static switch selecting whether a
            # read exists.
            Alternative(
                value=1,
                declarations=(
                    PlantElectricProductionReactor(
                        itart=0,  # `.physics.itart`, `physics_variables.py:994`
                        i_tf_sup=1,  # `.tfcoil.i_tf_sup`
                        i_blkt_dual_coolant=0,  # `.fwbs.i_blkt_dual_coolant`
                        i_p_coolant_pumping=1,  # `.heat_transport.i_p_coolant_pumping`
                    ),
                ),
            ),
        ),
    ),
    Switch(
        path=".costs.i_cost_model",
        default=1,  # `cost_variables.py:327`, `CostModels.KOVARI_2014`
        alternatives=(
            Alternative(value=0, declarations=COSTS_1990),  # PROCESS_1990
            Alternative(
                value=1,  # KOVARI_2014
                unproduced=(
                    "costs_2015.py has no cottax nodes at all (2 of its 13 methods are "
                    "ported as plain functions; the 8 `calc_*` methods that fill its "
                    "100-slot `s_cost` array, and the `total_costs`/`coe` accumulation "
                    "on top of them, are not). Assembling this arm therefore leaves "
                    ".costs.coe and .costs.concost without a producer, which is the "
                    "true state of this port -- registering costs.py's 1990-model "
                    "nodes here instead would compute the same two fields by the wrong "
                    "cost model. stellarator_helias.IN.DAT:248 selects i_cost_model = "
                    "0 for exactly this reason ('the 2015 does not work yet for "
                    "stellarators'), so no run in this project's scope needs this arm."
                ),
            ),
            Alternative(
                value=2,  # USER_PROVIDED
                unported=(
                    "i_cost_model == 2 injects a user-supplied Model instance at "
                    "runtime (process/main.py's `costs` setter, lines 766-768) -- there "
                    "is no PROCESS-side subgraph to port at all, so no arm can exist "
                    "here. Refused rather than assembled empty: unlike KOVARI_2014, a "
                    "caller asking for this has a model in mind that this graph has "
                    "never seen."
                ),
            ),
        ),
    ),
)
"""Switches whose value decides which nodes exist. See `configuration.py`.

Grows as ported units bring more arms: `build.py`'s `blktmodel`, the blanket
CCFE-HCPB/DCLL split are known to belong here (`core/solver/switches.md`), but none of
their arms is ported yet, so declaring them now would be a switch with one arm and no
choice to make.

**`.vacuum.i_vacuum_pumping` and `.costs.i_cost_model` were investigated and are
deliberately NOT here**, unlike `i_bldgs_size` above -- both looked like the same shape
at first glance but fail this project's own exclusivity requirement
(`Switch.check_arms_are_exclusive`, exercised by `test_non_exclusive_arms_are_rejected`):
`VacuumPumpingSimple`/`VacuumOld` (`models/vacuum.py`) own completely disjoint output
sets (`n_iter_vacuum_pumps` vs. `n_vac_pumps_high`/`n_vv_vacuum_ducts`/`dlscal`/
`m_vv_vacuum_duct_shield`/`dia_vv_vacuum_ducts` -- no field in common at all, unlike
`Bldgs`/`BldgsSizes`'s shared `a_plant_floor_effective`/`volnucb`), and `costs.py`'s 23
ported leaf cost nodes share no output with `costs_2015.py` either (which in any case has
no `cottax` node written yet for either of its 2 ported functions) -- the only fields the
two cost files' *full* models share (`.costs.coe`/`.costs.concost`) belong to the
top-level accumulation neither file has ported. See `vacuum.md`'s and `costs.md`'s
registry rows for the resulting "ported but not registrable yet" treatment.
"""

COMMON = (
    # unit #1 chunks
    SudoDensityLimit,
    # EcrhDensityLimit moved to TOPOLOGY_SWITCHES's `i_plasma_pedestal` switch -- its
    # static kwarg is no longer independent of that switch's value, see there.
    StructureMasses,
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
    Build,
    # unit #4, divertor.py
    Divertor,
    # `st_fwbs` S1/S5 (`stellarator_E_fwbs_synthesis.md`), portable now, no blocker.
    FwBlanketShieldGeometry,
    CryostatAndVvGeometry,
    # `st_fwbs` S3 (`stellarator_fwbs_s3.md`). Reads `.divertor.a_div_surface_total`,
    # which `Divertor` owns -- an ordinary acyclic edge, not a cycle: `Divertor`'s own
    # inputs have no dependency back on anything `st_fwbs`/`DivertorPlateMass` produces
    # (verified directly against `divertor.py`'s `Input`s), so PROCESS's own staleness
    # here (`st_fwbs` runs before `st_div`, so it reads the *previous* `run()`'s value)
    # is a call-order artifact of its imperative code, not a genuine two-way dependency.
    # Registering this the ordinary way (`Divertor` before `DivertorPlateMass` in
    # topological order) is strictly more self-consistent than PROCESS's own lagged
    # read -- confirmed by the build below staying at the same SCC count.
    DivertorPlateMass,
    # `st_fwbs` S4's shield-mass block (`stellarator_fwbs_s4.md`). In `COMMON` and not
    # behind a `Switch` because `stellarator.py:1195-1206` is outside every branch in
    # `st_fwbs` -- no `blktmodel`, `blkttype` or `ipowerflow` guard -- so both outputs
    # exist in every configuration. Its sibling `BlanketComponentMasses` *is* switched,
    # see `TOPOLOGY_SWITCHES`'s `.fwbs.blktmodel,.fwbs.blkttype` entry. Closes
    # `_audit/boundary_inputs_audit.md` § 4c (b5)/(b6): `Bldgs` and `ShieldCost` were
    # reading `.fwbs.whtshld`, and `ShieldCost` `.fwbs.wpenshld`, as boundary inputs.
    ShieldMass,
    # unit #5, heating.py
    InjectedPowerTotal,
    BeamCurrent,
    FusionGain,
    # unit #6, initialization.py
    PulseDurations,
    # unit #12, physics/plasma_profiles.py
    ProfileFactors,
    # unit #21, physics/profiles.py -- arms not gated by `i_plasma_pedestal` only
    ProfileGrid(n_plasma_profile_elements=201),  # `physics_variables.py:1054` default
    NeProfileIntegral,
    TeProfileIntegral,
    DensityProfile,
    # unit #7, neoclassics.py (scalar-argument functions only, see module docstring)
    ProfileValues,
    EffectiveThermalDiffusivity,
    # unit #19, physics/fusion_reactions.py
    FusionRates,
    SetFusionPowers,
    # unit #20, physics/radiation_power.py
    SynchrotronRadiationPower,
    # `imp_indices` is a graph-assembly-time fact (which impurity species this machine
    # has), not a per-evaluation switch -- see `ImpurityRadiationTotals`'s docstring.
    # All 14 species: H/He are always recomputed by `plasma_composition()`, and species
    # 2-13 are held non-zero by iteration variables 125-136's lower bound (1e-8, 22
    # orders above the 1e-30 selection threshold) in the reference configuration this
    # scope targets. A run without those iteration variables active could legitimately
    # need a narrower tuple; nothing here checks that yet (radiation_power.md § open
    # questions 2).
    ImpurityRadiationTotals(imp_indices=tuple(range(14))),
    PlasmaRadiationPowers,
    # unit #9, coils/calculate.py
    CoilToroidalThickness,
    CoilRadialThickness,
    CoilCrossSectionalArea,
    CoilHalfWidths,
    PlasmaFacingCoilArea,
    CoilCoilToroidalGap,
    CoilsSummaryVariables,
    StoredMagneticEnergy,
    WindingPackGeometry,
    CoilCurrent,
    CoilCasing,
    VerticalPorts,
    HorizontalPorts,
    # `st_coil`'s formula for `.build.z_tf_inside_half` -- see `Build`'s own comment
    # above (unit #2, build.py) for why this one, not `Build`'s, owns the field.
    ZTfInsideHalf,
    # `.tfcoil.tfcryoarea`, carved out of the same inline `st_coil` geometry block as
    # `ZTfInsideHalf` and for the same reason (the eager `st_coil` orchestrator is not
    # registered, so anything only it computes has no owner). Prerequisite for
    # `CryoQLoadsStep` below: without it, registering the cryo nodes would have traded
    # two boundary inputs for one new one (`_audit/boundary_inputs_audit.md` §4c (c1)'s
    # sibling gap, §7 items 4 and 7). Of its two neighbours in that block,
    # `min_bending_radius` still stays unported for want of any reader.
    TfCryoArea,
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
    LenTfCoil,
    # unit #12, coils/mass.py
    CoilsMass,
    # unit #11, coils/forces.py
    MaxForceDensity,
    MaximumStress,
    # unit #14, coils/quench.py
    QuenchProtection,
    # unit #9 chunk A, physics/physics_A_pure_formulas.py -- five already-pure formulas
    # lifted verbatim, no entanglement, no switch-driven topology split.
    IonElectronEquilibration,
    AuxiliaryPhysicsQuantities,
    TotalPlasmaHeatingPower,
    ElectronThermalEnergy,
    IonThermalEnergy,
    # `i_beta_fast_alpha` kept as a static kwarg, not a Switch -- both branches read the
    # same six variables (physics_A_pure_formulas.md's "switches touched"), same shape as
    # `EcrhDensityLimit(i_plasma_pedestal=0)`. Default `1`, `physics_variables.py:875`.
    FastAlphaBeta(i_beta_fast_alpha=1),
    # unit #9 chunk B, physics/physics_B_composition.py. `plasma_composition`'s
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
    # `is_ignited=True` -- **not** `physics_variables.py:881`'s bare default
    # (`i_plasma_ignited = 0`, NON_IGNITED). `stellarator_helias.IN.DAT:126` sets
    # `i_plasma_ignited = 1` (IGNITED), and the converged run confirms it: the
    # `switch_audit` check `mda_harness.py` now runs over every registered static kwarg
    # reported `registered=False but .physics.i_plasma_ignited == True`. Same defect
    # class as `i_confinement_time`/`i_thermal_electric_conversion` below -- a bare
    # `*_variables.py` default copied uncritically into a registration.
    # `is_ignited` is `bool`, not the raw `int` switch, because
    # `physics_B_composition.py:134-136`'s port maps it to PROCESS's own
    # `PlasmaIgnitionModel(i_plasma_ignited) == NON_IGNITED` compare; `True` here means
    # IGNITED (`physics_variables.py:45-49`).
    # Checked before flipping, same discipline as `i_thermal_electric_conversion`
    # below: the IGNITED arm needs no input this port does not already wire --
    # `physics_B_composition.py:219-222` is `nd_beam_ions = 0` under `is_ignited`
    # versus `nd_plasma_electrons_vol_avg * f_nd_beam_electron` otherwise, so the
    # ignited arm reads a strict *subset* of the non-ignited arm's inputs.
    PlasmaComposition(is_ignited=True),
    CalculateEffectiveChargeIonisationProfiles,
    # unit #9 chunk C, physics/physics_C_outplas.py -- the one real computation inside
    # the 1095-line `outplas` reporting method.
    DimensionlessPlasmaParameters,
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
    DoubleAndTripleProduct,
    # unit #11, physics/exhaust.py
    RadiationFraction,
    # unit #15, buildings.py -- unconditional preamble, feeds both `i_bldgs_size` arms
    TfCoilEnvelope,
    # unit #16, vacuum.py -- `"old"` branch only, matching PROCESS's own default
    # (`.vacuum.i_vacuum_pumping = "old"`, `vacuum_variables.py:18`). Not gated by a
    # `Switch`: the `"simple"` alternative (`VacuumPumpingSimple`) owns a disjoint
    # output, so this switch fails `check_arms_are_exclusive` -- see
    # `TOPOLOGY_SWITCHES`'s docstring above. `VacuumPumpingSimple` stays
    # ported-but-unregistered.
    VacuumOld,
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
    DuctDiameterRootFind(),
    # `stellarator_B_st_phys.py` (chunk 1B of unit #1). `StellaratorBetaAndRhoStar` is
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
    StellaratorBetaAndStoredEnergy,
    PoloidalFieldFromRotationalTransform,
    TotalField,
    FusionPowerTotalsMw,
    # The `else` arm of `stellarator.py:2002-2054`, three identities -- and the only
    # producer of `.physics.fusden_total`/`.fusden_alpha_total`/`.p_dt_total_mw`,
    # which were boundary inputs until it landed. Unconditional because the arm is
    # selected by `i_plasma_ignited == IGNITED` on this run, not merely by the absence
    # of a beam, and because the beam arm calls the unportable `reactions.beam_fusion`
    # (unit #19) -- there is no second arm to switch between. See
    # `_audit/boundary_inputs_audit.md` §4c (b7)/(b8) and the class's own docstring.
    FusionTotalsNoBeam,
    # `stellarator.py:2152-2166`: `st_phys`'s two zero-clips on the radiation power
    # densities and the two total powers formed from them. Owns the real
    # `.physics.pden_plasma_*_rad_mw` fields, which `PlasmaRadiationPowers` now mints
    # as `*_unclipped` -- the clip has two disagreeing call sites in PROCESS, so it
    # belongs to this caller, not to `calculate_radiation_powers`. Also gives
    # `.physics.p_plasma_inner_rad_mw` (read by `StellaratorConfinementTime`) its first
    # producer -- `_audit/boundary_inputs_audit.md` §7 item 6.
    ClippedRadiationPowers,
    # `i_pflux_fw_neutron`/`ipowerflow` static, per `physics_variables.py:1006`/
    # `heat_transport_variables.py:94`'s defaults (`1`). With `i_pflux_fw_neutron == 1`
    # both functions take their first branch unconditionally -- `ipowerflow`'s value is
    # inert for the actual computed result at this default, but still required as a
    # field; kept matching `.heat_transport.ipowerflow`'s own registered default above
    # for consistency, not because it changes anything here.
    NeutronWallLoad(i_pflux_fw_neutron=1, ipowerflow=1),
    # `i_plasma_ignited=1` (IGNITED, `stellarator_helias.IN.DAT:126`) -- **not**
    # `physics_variables.py:881`'s bare default `0`, which this registration used to
    # carry. Third site of the same mismatch (`PlasmaComposition`/`ConfinementTime` are
    # the other two), all three found together by `mda_harness.py`'s `switch_audit`.
    # Checked before flipping: `stellarator_B_st_phys.py:273-274` adds
    # `p_hcd_injected_total_mw` into `powht` only under NON_IGNITED, so the IGNITED arm
    # reads a strict subset of the inputs -- nothing new to wire.
    HeatingAndRadiationPower(i_plasma_ignited=1),
    RadiatedWallLoadAndFraction(i_pflux_fw_neutron=1, ipowerflow=1),
    ThermalEnergyTotals,
    # `stellarator_C_geometry.py` (chunk 1C of unit #1). `DefaultAspectRatio` is the
    # `1 not in data.numerics.ixc` conditional-ownership case (module docstring): the
    # bare `NumericsData` dataclass default (`ixc = [0, 0, ...]`, no real iteration-
    # variable ID ever present) makes `1 not in ixc` true, so this node is instantiated
    # unconditionally here, matching PROCESS's own bare-default configuration -- the
    # same convention every topology switch's own `default` already follows.
    # `StellaratorScalingFactors` takes `aspect` as a plain `Input` regardless of source
    # (this node's own output, when active, or an external iteration-variable value
    # otherwise), so no further wiring decision is needed here.
    DefaultAspectRatio,
    StellaratorScalingFactors,
    StellaratorPlasmaGeometry,
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
    WindingPackIntersectInputs(i_tf_sc_mat=1),
    Intersect(),
    WindingPackTotalSizePost(),
    # `power_A_tf_coil_power.py` (unit #14 chunk A). `TfPowerResistive`/
    # `TfPowerSuperconducting` are registered under `TOPOLOGY_SWITCHES`'s new
    # `.tfcoil.i_tf_sup` switch instead of here -- see that switch's own comment.
    # `power_B_thermal_cryo.py` (unit #14 chunk B). Six of `calculate_
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
    # `calculate_p_fw_blkt_coolant_pump_mw` (`power_B_thermal_cryo.py:206-211`)
    # returns `p_fw_coolant_pump_mw + p_blkt_coolant_pump_mw` for `1 not in
    # {MECHANICAL, MECHANICAL_WITH_PRESSURE_DROP}`, and
    # `calculate_p_fw_div_heat_deposited_mw` (`power_B_thermal_cryo.py:308-310`)
    # returns `p_fw_heat_deposited_mw + p_div_heat_deposited_mw` for
    # `1 != MECHANICAL_WITH_PRESSURE_DROP`. Both operands are already `Input`s (or
    # rebuilt from `Input`s) on every node below, so no arm has a hole in it.
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
    ComponentThermalPowers(
        i_p_coolant_pumping=1,
        i_blkt_dual_coolant=0,
        i_thermal_electric_conversion=2,
        i_blanket_type=1,
        secondary_cycle_liq=4,
    ),
    DeltaEtaStep(
        i_p_coolant_pumping=1, i_blkt_dual_coolant=0, i_thermal_electric_conversion=2
    ),
    EtaTurbineStep(i_thermal_electric_conversion=2, i_blanket_type=1),
    EtathLiqStep(secondary_cycle_liq=4),
    TempTurbineCoolantInStep(
        i_thermal_electric_conversion=2, i_blanket_type=1, secondary_cycle_liq=4
    ),
    PFwDivHeatDepositedMwStep(i_p_coolant_pumping=1),
    PFwBlktCoolantPumpMwStep(i_p_coolant_pumping=1),
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
    #     `.tfcoil.cryo_cool_req`) and reads the five `q*` as plain `Input`s.
    # This closes `.heat_transport.helpow` (read by `Bldgs`, `CryogenicSystemCost`) and
    # `.heat_transport.p_cryo_plant_electric_mw` (read by `Acpow`,
    # `PlantElectricProductionReactor`, `AuxiliaryComponentCoolingCost`) as boundary
    # inputs. `inuclear=0`/`i_pf_conductor=0` are `fwbs_variables.py:81`/
    # `pfcoil_variables.py:230`'s defaults, neither set by `REFERENCE_INPUT_FILE`;
    # `i_tf_sup=1` is `tfcoil_variables.py:261`'s, likewise unset -- the same value the
    # rest of this file's TF-coil registrations already carry.
    CryoQNucStep(i_tf_sup=1, inuclear=0),
    CryoQLoadsStep(i_tf_sup=1, i_pf_conductor=0),
    CryoLoads(i_tf_sup=1, i_pf_conductor=0),
    # `power_C_electric_production.py` (unit #14 chunk C). `i_pf_energy_storage_source=2`
    # matches `pf_power_variables.py:18`'s default.
    Acpow(i_pf_energy_storage_source=2),
    # `PowerProfilesOverTime`/`PlantElectricProductionReactor` are the two arms of the
    # `.costs.ireactor` `Switch` below, not `COMMON` members -- see that switch.
    # `availability.py` (unit #17). `Stellarator.run()`'s solve-time branch calls
    # `self.availability.avail()` directly (`stellarator.py:175`), bypassing
    # `.costs.i_plant_availability`'s dispatch entirely -- so `Avail` (not `Avail2`/
    # `AvailSt`) is the node actually exercised at solve time regardless of that
    # switch's value, and belongs in `COMMON`, not behind a `Switch`. Its
    # `.costs.cplife` self-loop is resolved the same way as `plasma_composition`'s
    # `first_call`/`power_B_thermal_cryo.py`'s six fields above: `CplifeAvail`
    # (`FixedPointFunction`) owns `.costs.cplife` alone; `Avail` (`ExplicitFunction`)
    # owns every other output, reading `cplife` as a plain `Input`.
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
    Avail(ibkt_life=0, itart=0),
    CplifeAvail(i_tf_sup=1, itart=0),
    # `plasma_profiles.py`. In `COMMON` and not under `.physics.i_plasma_pedestal`:
    # PROCESS writes `.physics.temp_plasma_ion_vol_avg_kev` in `parameterise_plasma`
    # *before* the branch, so it runs in both arms. A `FixedPointFunction` rather than an
    # `ExplicitFunction` because the field is conditionally owned by *data*
    # (`f_temp_plasma_ion_electron > 0`) -- see the class's own docstring for why that is
    # the honest shape and not a workaround.
    IonVolAvgTemperature(),
)
"""Nodes present in every configuration -- everything no topology switch gates."""


REFERENCE_INPUT_FILE = "tests/regression/input_files/stellarator_helias.IN.DAT"
"""The run this whole port is validated against -- `mda_harness.py`, `mda_constraint_
harness.py` and every number in `_audit/next_steps.md` \u00a7 8 use it. Named here so
`REFERENCE_CONFIGURATION` can be checked against it mechanically instead of by eye."""

REFERENCE_CONFIGURATION = Configuration({
    ".stellarator.istell": 6,  # `stellarator_helias.IN.DAT:137`
    ".stellarator.isthtr": 1,  # `:139` -- equals `Switch.default`, listed anyway
    ".physics.i_plasma_pedestal": 0,  # `:118`
    ".costs.i_cost_model": 0,  # `:248`
    ".costs.ireactor": 1,  # `:245` -- equals `Switch.default`, listed anyway
})
"""The topology-switch choices `REFERENCE_INPUT_FILE` actually makes.

**Every switch the input file sets explicitly is listed, including ones whose value
happens to equal `Switch.default`** (`isthtr = 1`). Listing them regardless makes this a
faithful transcription of the file rather than a diff against PROCESS's defaults, and
means a future change to a `Switch.default` cannot silently move the reference run.
Switches the file never mentions are deliberately absent: they fall through to
`Switch.default`, which is exactly what a real run does for a variable a silent IN.DAT
never sets. `test_configuration.py::
test_reference_configuration_matches_the_input_file` parses the file and checks this
dict against it, so the two cannot drift.

**Why this exists, and why `graph_for()` defaults to it rather than to `Configuration()`.**
Five separate registration bugs in this project shared one root cause: a value copied
from PROCESS's bare `*_variables.py` default rather than from the run being modelled --
`i_confinement_time` (34 vs 38), `i_thermal_electric_conversion` (0 vs 2),
`i_p_coolant_pumping` (2 vs 1), `i_plasma_ignited` (0 vs 1), and `i_cost_model` (1 vs 0,
which left `.costs.coe` with no producer and 43 nodes unregistered). Each was found by
the MDA harness, individually, after the fact. Making the *bare* graph mean "PROCESS's
silent-IN.DAT defaults" put that trap directly in the path of anyone assembling a graph
without thinking about it -- while every consumer that mattered (the harness, the SAND
prototype) had to remember to pass a configuration, and the ones that forgot were the
bugs. Reversing the default puts the burden where the unusual case is.

`PROCESS_DEFAULT_CONFIGURATION` below is still available, still meaningful, and still
tested -- the silent-IN.DAT graph is a real thing to want, just not the thing this
project's own tooling should get by accident."""

PROCESS_DEFAULT_CONFIGURATION = Configuration()
"""Every switch at its own `Switch.default`, i.e. the graph a silent IN.DAT produces.
`Switch.default` remains PROCESS's own value read from `process/data_structure/`, and
`test_configuration.py::test_switch_defaults_match_process` checks each one against the
cited field -- changing `graph_for()`'s default does not weaken that contract."""


def graph_for(configuration=None):
    """The assembled graph for one configuration; `REFERENCE_CONFIGURATION` if unstated.

    **Not `Configuration()`** -- see `REFERENCE_CONFIGURATION`'s docstring for the five
    bugs that choice caused. Pass `PROCESS_DEFAULT_CONFIGURATION` explicitly for the
    silent-IN.DAT graph.
    """
    return build_graph(
        REFERENCE_CONFIGURATION if configuration is None else configuration,
        COMMON,
        TOPOLOGY_SWITCHES,
    )


GRAPH = graph_for()
"""`REFERENCE_CONFIGURATION`'s graph -- the `stellarator_helias.IN.DAT` run this port is
validated against (`istell = 6`, `i_plasma_pedestal = 0`, `i_cost_model = 0`; every other
switch at PROCESS's own default)."""

if __name__ == "__main__":
    n_vars = sum(
        len(node.inputs) + len(node.outputs) for node in GRAPH.definitions.values()
    )
    print(f"{len(GRAPH.definitions)} nodes, {n_vars} ports (inputs + outputs, unmerged)")
    for name, node in GRAPH.definitions.items():
        print(f"  {name.path_str()}: {len(node.inputs)} in, {len(node.outputs)} out")

    print("\ncycles, per configuration:")
    for path, value in ((None, None), (".heat_transport.ipowerflow", 0)):
        configuration = Configuration({path: value} if path else {})
        cycles = graph_for(configuration).cycles
        label = f"{path} = {value}" if path else "PROCESS defaults"
        print(f"  {label}: {[[n.path_str() for n in c] for c in cycles] or 'acyclic'}")
