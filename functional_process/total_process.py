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

from functional_process.configuration import (
    Alternative,
    Configuration,
    Switch,
    build_graph,
)
from functional_process.models.availability import Avail, CplifeAvail
from functional_process.models.buildings import Bldgs, BldgsSizes, TfCoilEnvelope
from functional_process.models.physics.confinement_time import (
    ConfinementTime,
    DoubleAndTripleProduct,
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
    ParabolicGradientLengths,
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
    DeltaEtaStep,
    EtathLiqStep,
    EtaTurbineStep,
    PFwBlktCoolantPumpMwStep,
    PFwDivHeatDepositedMwStep,
    TempTurbineCoolantInStep,
)
from functional_process.models.power_C_electric_production import (
    Acpow,
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
    PlasmaFacingCoilArea,
    StoredMagneticEnergy,
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
from functional_process.models.stellarator.stellarator_B_st_phys import (
    FusionPowerTotalsMw,
    HeatingAndRadiationPower,
    NeutronWallLoad,
    PoloidalFieldFromRotationalTransform,
    RadiatedWallLoadAndFraction,
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
from functional_process.models.vacuum import DuctDiameterRootFind, VacuumOld

TOPOLOGY_SWITCHES = (
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
    # `is_ignited=False` matches `physics_variables.py:881`'s default
    # (`i_plasma_ignited = 0`, `ConfinementTime`'s same default above).
    PlasmaComposition(is_ignited=False),
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
    # `i_rad_loss=1`/`i_plasma_ignited=0` below.
    ConfinementTime(i_confinement_time=38, i_rad_loss=1, i_plasma_ignited=0),
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
    # deliberately NOT registered: its `.physics.rho_star` output is algebraically
    # identical to `DimensionlessPlasmaParameters`'s own `rho_star` formula above (same
    # inputs, same expression, confirmed by direct comparison) -- a genuine
    # redundant-duplicate-write in PROCESS itself (`st_phys` and `outplas` both compute
    # it), not a porting choice. Registering both would be a duplicate-ownership
    # conflict, the same shape as `IterPhysicsBasisElongation`/`ConfinementTime`'s
    # `kappa_ipb` above -- `StellaratorBetaAndRhoStar` is the one left out, since
    # `DimensionlessPlasmaParameters` was already registered. This also leaves
    # `.physics.beta_total_vol_avg`/`.physics.e_plasma_beta` without a producer in this
    # graph for now -- flagged, not resolved (see the registry).
    PoloidalFieldFromRotationalTransform,
    TotalField,
    FusionPowerTotalsMw,
    # `i_pflux_fw_neutron`/`ipowerflow` static, per `physics_variables.py:1006`/
    # `heat_transport_variables.py:94`'s defaults (`1`). With `i_pflux_fw_neutron == 1`
    # both functions take their first branch unconditionally -- `ipowerflow`'s value is
    # inert for the actual computed result at this default, but still required as a
    # field; kept matching `.heat_transport.ipowerflow`'s own registered default above
    # for consistency, not because it changes anything here.
    NeutronWallLoad(i_pflux_fw_neutron=1, ipowerflow=1),
    HeatingAndRadiationPower(i_plasma_ignited=0),  # `physics_variables.py:881` default
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
    # `i_p_coolant_pumping=2`/`i_blkt_dual_coolant=0`/`i_blanket_type=1`/
    # `secondary_cycle_liq=4` all match `fwbs_variables.py`'s own defaults (lines 249,
    # 526, 70, 273) -- `i_p_coolant_pumping` is known to disagree with at least one real
    # run (`stellarator_helias.IN.DAT` sets `i_p_coolant_pumping = 1`, `FRACTION_OF_HEAT`,
    # not `2`/`MECHANICAL`), found by the same MDA-vs-PROCESS harness pass as the fix
    # just below but out of this pass's scope -- flagged, not fixed, here.
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
        i_p_coolant_pumping=2,
        i_blkt_dual_coolant=0,
        i_thermal_electric_conversion=2,
        i_blanket_type=1,
        secondary_cycle_liq=4,
    ),
    DeltaEtaStep(
        i_p_coolant_pumping=2, i_blkt_dual_coolant=0, i_thermal_electric_conversion=2
    ),
    EtaTurbineStep(i_thermal_electric_conversion=2, i_blanket_type=1),
    EtathLiqStep(secondary_cycle_liq=4),
    TempTurbineCoolantInStep(
        i_thermal_electric_conversion=2, i_blanket_type=1, secondary_cycle_liq=4
    ),
    PFwDivHeatDepositedMwStep(i_p_coolant_pumping=2),
    PFwBlktCoolantPumpMwStep(i_p_coolant_pumping=2),
    # `PlantThermalEfficiency`/`PlantThermalEfficiency2` (the raw, un-split
    # `ExplicitFunction`s `EtaTurbineStep`/`EtathLiqStep`/`TempTurbineCoolantInStep` are
    # extracted from) are NOT registered: each is *itself* a genuine, still-unresolved
    # self-loop on its own -- `to_graph(PlantThermalEfficiency(...))` raises
    # `ValueError: reads [...], which it also owns` directly (confirmed this pass, not
    # merely asserted), since both own and read `eta_turbine`/`temp_turbine_coolant_in`
    # (`etath_liq`/`temp_turbine_coolant_in` for the second). They are superseded by the
    # three `*Step` `FixedPointFunction`s above for graph purposes, not usable
    # standalone. `Cryo`/`CryoLoads` are also NOT registered for the same reason, found
    # this pass and not yet fixed: `Cryo` reads and owns `.fwbs.qnuc`
    # (`conditional-ownership-by-run-config`, its own docstring already names the
    # shape); `CryoLoads` reads and owns `.fwbs.qnuc` plus `.power.qss`/`qac`/`qcl`/
    # `qmisc` (it calls `calculate_cryo` internally under its own guard, inheriting the
    # same self-reference on four more fields). Both confirmed directly via `to_graph`,
    # not just asserted -- see `unit_registry.md`/`next_steps.md` for the write-up. No
    # `FixedPointFunction` split exists yet for either; left as ported-but-unregistered,
    # a second wave of the same Shape-B gap `next_steps.md` §5 already tracks.
    # `power_C_electric_production.py` (unit #14 chunk C). `i_pf_energy_storage_source=2`
    # matches `pf_power_variables.py:18`'s default.
    Acpow(i_pf_energy_storage_source=2),
    # `PowerProfilesOverTime`'s whole output set is a strict subset of
    # `PlantElectricProduction`'s (the real PROCESS caller, `plant_electric_production`,
    # calls `power_profiles_over_time` internally) -- registered here on its own since
    # `PlantElectricProduction` itself is NOT registerable: found this pass, it owns
    # *and* reads five more fields (`p_plant_electric_gross_mw`, `p_turbine_loss_mw`,
    # `p_plant_electric_recirc_mw`, `p_plant_electric_net_mw`,
    # `f_p_plant_electric_recirc`), a third genuine still-unresolved self-loop alongside
    # `Cryo`/`CryoLoads` above, confirmed directly via `to_graph`. Left
    # ported-but-unregistered.
    PowerProfilesOverTime,
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
)
"""Nodes present in every configuration -- everything no topology switch gates."""


def graph_for(configuration=None):
    """The assembled graph for one configuration; PROCESS's defaults if unstated."""
    return build_graph(configuration or Configuration(), COMMON, TOPOLOGY_SWITCHES)


GRAPH = graph_for()
"""The default configuration's graph: `isthtr = 1` (ECRH), `ipowerflow = 1`."""

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
