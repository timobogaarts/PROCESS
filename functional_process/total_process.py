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

Not included here despite being ported: `coils/coils.py` -- no node written; its own
audit found no basis yet for the wiring a node would assert. `coils/calculate.py`'s
`winding_pack_total_size` and `st_coil` itself -- real tier-2/tier-3 units, not
self-contained, still audit-only. `physics/superconductors.py` (unit #22) and
`physics/impurity_radiation.py`'s two leaf functions (unit #23) -- same reason as
`coils.py`: every real call site's arguments are locals inside a not-yet-wired unit
(`jcrit_from_material`/`plasma_composition` respectively), so no node is written for
either yet; see each unit's own audit record.

`blankets/hcpb.py` (unit #13) is ported (3/3 in-scope functions, 3 `ExplicitFunction`
nodes) but **deliberately not registered here**: all three are only ever called from
`stellarator.py`'s `blanket_neutronics()`, itself only reached under `.fwbs.blktmodel ==
1` (S2 of the `st_fwbs` synthesis, `stellarator_E_fwbs_synthesis.md`, `next_steps.md`
§3) -- a dispatch this module does not build yet. Registering them unconditionally
would also be a real graph error, not just premature: `NuclearHeatingMagnets` writes
`.fwbs.p_tf_nuclear_heat_mw`, which `ScTfCoilNuclearHeating` (chunk 1F, already in
`COMMON`) also writes -- confirmed against `process/models/stellarator/stellarator.py`
that these are genuinely alternative producers under different `blktmodel` arms, not a
redundant pair. Resolving this needs S2's own `blktmodel`/`i_tf_sup` dispatch design
(a `Switch`, most likely), not a hasty registration here -- see `unit_registry.md` row
13 and `next_steps.md` §3.
"""

from functional_process.configuration import (
    Alternative,
    Configuration,
    Switch,
    build_graph,
)
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
from functional_process.models.stellarator.build import (
    AFwTotalNoPowerflow,
    AFwTotalWithPowerflow,
    BlktmodelBlanketThickness,
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
)
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
from functional_process.models.stellarator.stellarator_fwbs_s3 import DivertorPlateMass
from functional_process.models.vacuum import VacuumOld

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
    ScTfCoilNuclearHeating,
    # unit #2, build.py
    BlktmodelBlanketThickness,
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
    # unit #9 chunk B, physics/physics_B_composition.py -- `plasma_composition` itself is
    # NOT registered: it reads-then-conditionally-writes `.physics.first_call` (a genuine
    # self-loop, one VarPath read and owned by the same function), which needs `Cut`
    # machinery this module doesn't have yet. See physics_B_composition.md and
    # next_steps.md §5 for the (unresolved) finding.
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
    # Defaults: `i_confinement_time=34` (IPB98(y,2)), `i_rad_loss=1`,
    # `i_plasma_ignited=0` (`physics_variables.py:962,954,881`).
    ConfinementTime(i_confinement_time=34, i_rad_loss=1, i_plasma_ignited=0),
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
