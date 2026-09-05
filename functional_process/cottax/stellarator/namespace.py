"""The device's own namespaces -- its coils, its FWBS, and `Stellarator` itself.

Beside the nodes they name (`model_tree_design.md` §11): twenty of this package's
modules used to be imported into `total_process.py` for no reason but to be named
in a slot here. `BlanketShieldPowerExponential` is an *occupant of a switched
slot*; which occupant a machine gets is `indat.py`'s answer, and nothing here reads
a switch.

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
not-yet-wired unit (`plasma_composition`, itself registered in
`models/physics/namespace.py`, but its impurity-array locals stay per-index minted
paths, not a whole-array edge into `impurity_radiation.py`'s own leaves).
`coils/coils.py`'s `jcrit_from_material` (unit #10, an 8-way switch on `i_tf_sc_mat`,
one `ExplicitFunction` node per branch) --
**investigated and still not registered, for a structural reason, not an oversight**:
its `FromExactly`s (`.tfcoil.t_helium`/`b_max`) are per-sample locals of
`winding_pack_curves`'s 200-point sampling loop (`b_max = b_max_k[k]`, an array, not a
scalar), and PROCESS has exactly one real call site for the whole dispatch, inside that
same sampling loop (confirmed: `grep`ing `process/models/stellarator/coils/calculate.py`
for `jcrit_from_material` finds only its own `jcrit_vector[k] = jcrit_from_material(...)`
per-sample assignment) -- there is no single-point scalar call site for these 8 nodes to
bind to as written, so registering them would assert a wiring that does not exist in
PROCESS, not merely one this pass hasn't gotten to yet. `calculate.py` keeps its own
eight `jcrit_*` functions rather than calling these 8 nodes' underlying functions
directly, for the same reason `calculate.md` documents: it deliberately diverges from
`coils.py`'s own `jcrit_from_material` on the REBCO branch (`coils.py` reproduces a real
PROCESS call-site bug there; `calculate.py`'s local copy sidesteps it so this port has
*a* working REBCO branch) -- collapsing the two would either regress REBCO or stop
reproducing the bug faithfully, so they stay two independent implementations, documented
as such, not one deduplicated further. **What `i_tf_sc_mat` does reach is the
`winding_pack_intersect_inputs` slot below**, whose eight occupants are `calculate.py`'s
own (`_audit/next_steps.md` §14.5) -- the switch selects a class there, at the one call
site the dispatch really has.

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

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.cottax.stellarator.build import (
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
from functional_process.cottax.stellarator.coils.calculate import (
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
from functional_process.cottax.stellarator.coils.coils import Intersect
from functional_process.cottax.stellarator.coils.forces import (
    MaxForceDensity,
    MaximumStress,
)
from functional_process.cottax.stellarator.coils.mass import CoilsMass
from functional_process.cottax.stellarator.coils.quench import QuenchProtection
from functional_process.cottax.stellarator.density_limits import (
    SudoDensityLimit,
)
from functional_process.cottax.stellarator.divertor import Divertor
from functional_process.cottax.stellarator.geometry import (
    DefaultAspectRatio,
    StellaratorPlasmaGeometry,
    StellaratorScalingFactors,
)
from functional_process.cottax.stellarator.heating import (
    BeamCurrent,
    EcrhHeating,
    FusionGain,
    InjectedPowerTotal,
    LowhybHeating,
)
from functional_process.cottax.stellarator.initialization import PulseDurations
from functional_process.cottax.stellarator.neoclassics import (
    EffectiveThermalDiffusivity,
    ProfileValues,
)
from functional_process.cottax.stellarator.plasma_physics import (
    ClippedRadiationPowers,
    HeatingAndRadiationPower,
    NeutronWallLoad,
    PoloidalFieldFromRotationalTransform,
    RadiatedWallLoadAndFraction,
    StellaratorBetaAndStoredEnergy,
    ThermalEnergyTotals,
    TotalField,
)
from functional_process.cottax.stellarator.preset_config import (
    StellaratorMachineConfig,
)
from functional_process.cottax.stellarator.stellarator_fwbs_s1_s5 import (
    CryostatAndVvGeometry,
    FwBlanketShieldGeometry,
)
from functional_process.cottax.stellarator.stellarator_fwbs_s2 import (
    DetailedPowerflowBlanketShieldPower,
    DetailedPowerflowBlanketShieldPowerUserInputPumping,
    ExponentialAttenuationBlanketShieldPower,
)
from functional_process.cottax.stellarator.stellarator_fwbs_s3 import DivertorPlateMass
from functional_process.cottax.stellarator.stellarator_fwbs_s4 import (
    BlanketComponentMasses,
    ShieldMass,
)
from functional_process.cottax.stellarator.structure import (
    StructureMasses,
)
from functional_process.cottax.stellarator.tf_nuclear_heating import (
    ScTfCoilNuclearHeating,
)


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
    coils_mass: CoilsMass = dataclasses.field(kw_only=True)
    """The TF coil masses -- one occupant per `.tfcoil.i_tf_sc_mat` value, keyed by
    `indat.COILS_MASS_MATERIAL`.

    **This slot's switch was answered by a module constant until
    `_audit/next_steps.md` §14.2**, so no instrument in the port could see it: a
    material other than ITER Nb3Sn assembled the right winding-pack occupant next door
    and a coil-mass node still reading `.tfcoil.dcond[0]`."""
    # unit #11, coils/forces.py
    max_force_density: MaxForceDensity = MaxForceDensity()
    maximum_stress: MaximumStress = MaximumStress()
    # unit #14, coils/quench.py
    quench_protection: QuenchProtection = QuenchProtection()
    # `coils/calculate.py`'s `winding_pack_total_size` (unit #9's remaining tier-2 gap),
    # now the full three-piece split: `winding_pack_intersect_inputs` (pre-`intersect`,
    # a **slot**, mints `.stellarator.wp_width_r`/`.lhs`/`.rhs` and
    # `.wp_width_r_min_guess`), `coils.py`'s `Intersect`
    # (`ImplicitFunction`/`RootFind`, owns `.stellarator.wp_width_r_min`),
    # `WindingPackTotalSizePost` (post-`intersect`, owns `.tfcoil.j_tf_wp` along with
    # everything else `winding_pack_post_intersect` computes).
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
    winding_pack_intersect_inputs: WindingPackIntersectInputs = dataclasses.field(
        kw_only=True
    )
    """The pre-`intersect` curves, on `.tfcoil.i_tf_sc_mat` -- one occupant per
    superconductor, `WINDING_PACK_MATERIAL` in `indat.py`.

    A slot since `_audit/next_steps.md` §14.5, where it was an
    `i_tf_sc_mat=SuperconductorModel.ITER_NB3SN` constructor kwarg on a single node that
    branched internally. The eight branches read different `.tfcoil.*` fields, so that
    node declared six reads dead at this run's value -- and one of them,
    `.tfcoil.j_tf_wp`, was the **sole back-edge closing the four-node coils SCC**
    (`_audit/switch_kwarg_survey.md` §4.6). Only the Bi-2212 occupant reads it; with
    every other material the driven block is `Intersect` and its own `^problem`, which
    is the cycle the model genuinely has. This is the sub-namespace's reason for
    existing (`model_tree_design.md` §4) getting smaller, and it is measured, not
    claimed."""
    intersect: Intersect = Intersect()
    winding_pack_total_size_post: WindingPackTotalSizePost = WindingPackTotalSizePost()


class StellaratorFwbs(ModelNamespace):
    """First wall, blanket and shield -- the `st_fwbs` chunk's registered nodes."""

    blanket_shield_power: (
        BlanketShieldPowerExponential
        | DetailedPowerflowBlanketShieldPower
        | DetailedPowerflowBlanketShieldPowerUserInputPumping
    ) = dataclasses.field(kw_only=True)
    """Blanket/shield power deposition, on `.fwbs.blktmodel` x `.heat_transport.
    ipowerflow` x `.fwbs.i_p_coolant_pumping` jointly -- one slot, three integers,
    resolved by `_blanket_shield_power_arm` in `machine_from_indat`.

    A **ragged** family, which is allowed and deliberate: the arm-1 occupant
    (`blktmodel == 0 & ipowerflow == 0`) is a two-node namespace (it also owns the
    TF-coil nuclear heating), the arm-2 one (`blktmodel == 0 & ipowerflow == 1 &
    i_p_coolant_pumping == 1`, PROCESS's own default and the reference run) a single
    node, and the arm-3 one (the same at `i_p_coolant_pumping == 0`, `helias_5b`) a
    single node owning **four fields fewer** -- the coolant pumping powers, which that
    value of the switch makes run inputs rather than computed values. Occupants of one
    slot need not have equal shape or equal output sets; what checks the consequences is
    the boundary postcondition, not a shape rule.

    The third integer joined on 2026-08-31 and it was a live defect, not a refinement:
    the arm-2 occupant was assembled for `helias_5b` too, and answered `16.8 MW` of
    FW+blanket pumping power where that file states `176.0`.

    Arm 0 -- `blktmodel == 1`, at either `ipowerflow` -- is refused: it is
    `blanket_neutronics()`, which calls `hcpb.nuclear_heating_*`, unported. Arm 4 --
    `i_p_coolant_pumping` mechanical, at `ipowerflow == 1` -- is refused because
    PROCESS itself raises there. That is also
    why the `| None` this annotation used to carry was **dead**: every arm outside the
    registry is in `UNPORTED`, and they raise -- absence was never reachable.
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
    """The 34 `.stellarator_config.stella_config_*` fields, for whichever machine.

    Filled at every `.stellarator.istell` a stellarator has -- `1`-`5` from
    `preset_config.py`'s hardcoded tables (Helias 5b/4/3, W7-X 30/50), `6` from a
    `stella_conf.json` -- and never at `istell == 0`, which is a **tokamak** and has no
    counterpart namespace in this tree. That is why this slot needs no `| None`: it used
    to hold one for the tokamak, and the tokamak is gone.

    **The occupant is the same node in all six cases**, and only its static payload
    differs, because which table or file was read changes no field's identity and no
    node's reads (`preset_config.md` § "switches touched"). So this slot is not keyed on
    `istell` at all; `indat.machine_config_for_istell` resolves the payload before the
    constructor runs.

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

    coils: StellaratorCoils = dataclasses.field(kw_only=True)
    """The coil sub-namespace. No default any more, because one of its members is a slot
    (`winding_pack_intersect_inputs`, on `.tfcoil.i_tf_sc_mat`) and a namespace holding a
    slot cannot be default-constructed -- the same reason `fwbs` below has none."""

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
    neutron_wall_load: NeutronWallLoad = dataclasses.field(kw_only=True)
    """Neutron wall load -- one occupant per arm of `.physics.i_pflux_fw_neutron` x
    `.heat_transport.ipowerflow` (`indat.py`'s `_wall_load_arm`).

    **Both switches were static kwargs here and neither is now**
    (`_audit/next_steps.md` §14.2). Threading them (step 4d) fixed the coherence half of
    the defect -- `ipowerflow` already decides `fw_area` and, jointly with `blktmodel`,
    `fwbs.blanket_shield_power`, and an `ipowerflow = 0` machine used to assemble
    `AFwTotalNoPowerflow` and `BlanketShieldPowerExponential` alongside two nodes still
    saying `COMPREHENSIVE_2014` -- but left the reads half: the node declared all three
    arms' fields, four of which are dead at this machine's values, and one of the four
    (`.first_wall.a_fw_total`) is `fw_area`'s own output.
    """
    heating_and_radiation_power: HeatingAndRadiationPower = dataclasses.field(
        kw_only=True
    )
    """Heating power, SOL radiation split and alpha power to the wall -- one occupant
    per `.physics.i_plasma_ignited` value.

    **The switch was a static kwarg here and is a slot now** (`_audit/next_steps.md`
    §14.2). The note it replaces recorded the check made before flipping its *value*
    from PROCESS's bare default to the file's -- "the IGNITED arm reads a strict subset
    of the inputs, nothing new to wire" -- and that subset is the defect: the node
    declared `.current_drive.p_hcd_injected_total_mw`, a cross-subsystem edge no ignited
    run makes."""
    radiated_wall_load_and_fraction: RadiatedWallLoadAndFraction = dataclasses.field(
        kw_only=True
    )
    """Radiated wall load and radiation fraction -- the same three arms as
    `neutron_wall_load` above, from the same `_wall_load_arm` dispatch."""
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
