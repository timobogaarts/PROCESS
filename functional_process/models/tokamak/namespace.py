"""The tokamak's own subsystems -- fourteen filled, eleven still empty.

Beside the nodes it names (`model_tree_design.md` §11), exactly as
`models/stellarator/namespace.py` sits beside the stellarator's. **It shipped with not
one slot occupied**; the first tokamak porting wave filled fourteen of the twenty-five,
and the eleven that are left keep the original spelling `... | None = None`, which cottax
reads as *"an unproduced slot: it assembles nothing, and whatever read its outputs
surfaces as a boundary input. Absence, spelled as absence."*

**Why an empty namespace was worth writing at all**, and why the same argument now
applies to the eleven that remain. It is the difference between an estimate and a
measurement. `next_steps.md` §13.9 asked for the *variable-level* work list a second
device costs; `tokamak_scope.md` and `tokamak_call_surface.md` could only count switches
and functions, because a boundary is a property of an assembled graph and there was no
tokamak graph to take it of. With `TokamakProcess` assembled out of the shared subsystems
and these slots left honestly empty, `boundary.boundary()` enumerates the missing
variables by name and `boundary.readers_of()` says who wants each one --
`_audit/tokamak_boundary.md` is that enumeration, and it is what the wave was aimed at.

**The slots are the traced call surface, not a guess.** Every name below is the
snake_case of a `Model` class that `tokamak_call_surface.md` §A recorded
`Caller._call_models_once` actually entering with `istell == 0` and `ife == 0`, under a
`sys.setprofile` hook. Nothing was read off `caller.py` and nothing was globbed: the
scope rule that produced §A exists because an earlier non-recursive glob silently missed
`models/stellarator/coils/` (6 files, 1950 LOC), and because `models/geometry/**` (11
files) lives under `process/models/` while being reached zero times by any solve.

**What is deliberately *not* here.** The five device-agnostic subsystems --
`costs` (40 nodes), `power` (20), `availability` (4), `vacuum` (3), `buildings` (2) --
are slots of `TokamakProcess` itself, beside `tokamak`, because they are not the
tokamak's: measured, they touch no `.stellarator*` data at all, and `costs/costs.py` is
entered by *the same 42 functions* on both devices (§C). `physics` likewise: 31 of its 33
nodes are shared, and the two that are not are a slot each in
`models/physics/namespace.py` (`confinement_time.scaling`, whose tokamak occupant
`IterIpb98y2ConfinementTime` already exists, and
`profiles.parameterisation.ecrh_density_limit`, which is stellarator-only and absent on a
tokamak by construction -- PROCESS computes no ECRH density limit at `i_plasma_pedestal
== 1`).

Sub-models reached only by injection or inheritance are slots in their own right where
§A named them, and are folded into their parent where it did not. `cs_coil`/`cs_fatigue`
are separate slots because `Models.__init__` injects them into `PFCoil` as distinct
models with their own switches (`i_cs_superconductor`, and the fatigue life `ncycle`
constraint); `blankets/blanket_library.py` and `tfcoil/base.py` are *not* slots, because
§A found them reached purely as base classes of `ccfe_hcpb` and
`cicc_superconducting_tf_coil` -- they are those occupants' bodies, not siblings.
`engineering/ivc_functions.py` is three plain module functions imported by three of the
slots below, not a model at all.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.blankets.namespace import CcfeHcpb
from functional_process.models.cryostat import Cryostat
from functional_process.models.fw import FirstWall
from functional_process.models.namespace import Build, Divertor
from functional_process.models.physics.physics import PlasmaEnergyFromBeta
from functional_process.models.physics.plasma_fields import PlasmaFields
from functional_process.models.physics.tokamak_namespace import (
    TokamakCurrentDrive,
    TokamakPhysics,
    TokamakPlasmaGeom,
    TokamakPulse,
)
from functional_process.models.structure import Structure
from functional_process.models.tfcoil.namespace import CiccSuperconductingTfCoil
from functional_process.models.vacuum.vacuum import VacuumVesselElliptical


class Tokamak(ModelNamespace):
    """Everything a conventional tokamak has and a stellarator does not.

    Twenty-five slots; **fourteen of them now have occupants and eleven are still
    `None`.** A namespace with `None` slots contributes no node for them and is
    explicitly allowed by cottax (`ModelNamespace`'s own refusal is for a namespace with
    no *slots*, which is a wrong argument rather than an empty one), so the eleven that
    are still empty behave exactly as all twenty-five did: whatever reads their outputs
    surfaces as a boundary input, enumerated by name in `_audit/tokamak_boundary.md`.

    **The annotation is the promise, and it is now kept slot by slot.** The class this
    file shipped with typed every slot `ModelNamespace | None` on the ground that *"there
    is no class to name yet"*, and said a slot would gain a real annotation the day it
    gained a real occupant, the way `physics.confinement_time.scaling` did. Fourteen
    have. Five of those fourteen are annotated with a **node** rather than a namespace
    (`plasma_beta`, `first_wall`, `structure`, `cryostat`, `vacuum_vessel`), because a
    slot may hold either -- `Physics.fusion_rates` always has -- and wrapping one node in
    a namespace to make the types uniform would put a meaningless key in front of its
    name.

    **A slot the factory fills has no default**, here as everywhere else in this tree.
    Twelve of the fourteen are `dataclasses.field(kw_only=True)`; the two that keep a
    default are the ones with nothing to decide -- no switch anywhere beneath them, so no
    configuration for a default to smuggle in. That distinction is not cosmetic: a
    defaulted slot is a slot `machine_from_indat` never asks a switch about, and an
    `UNPORTED` refusal that never fires is the `EcrhDensityLimit` bug class waiting to
    happen. `first_wall` and `structure` look unswitched -- each has exactly one occupant
    class -- and are factory-filled anyway, precisely so that a file asking for
    `i_pflux_fw_neutron == 0` or a resistive TF is *refused* rather than quietly given
    the arm that was written.
    """

    # ---- plasma geometry and the tokamak arm of the shared physics body -------------

    plasma_geom: TokamakPlasmaGeom = dataclasses.field(kw_only=True)
    """`physics/plasma_geometry.py::PlasmaGeom`, §A row 0 -- 7 entered functions, 549
    entered LOC, 1 of them shared with the stellarator
    (`calculate_iter_physics_basis_elongation`, already ported). Site of decision 2
    (`i_plasma_geometry`, 15 reads) and one read of decision 1."""

    physics: TokamakPhysics = dataclasses.field(kw_only=True)
    """`physics/physics.py::Physics`, §A row 2 -- the tokamak arm of the one 6931-line
    file, 11 entered functions of which 3 are shared and already in `.physics`.

    **The name collides with `TokamakProcess.physics` deliberately.** `.physics` is the
    shared plasma-physics subsystem, 31 nodes, entered by both devices;
    `.tokamak.physics` is the eight functions of `physics.py` only the tokamak path
    reaches. Naming it anything else would hide that they are the same PROCESS file. §E
    found six of the seventeen new topology decisions read inside it -- 1, 4, 5, 6, 7 and
    15 -- which makes it the single largest blocker in the tokamak scope by decision
    count."""

    plasma_inductance: ModelNamespace | None = None
    """`physics/physics.py::PlasmaInductance`, §A row 2.1 -- injected into `Physics` and
    run at `physics.py:356`. Site of decision 5 (`i_ind_plasma_internal_norm`, 4 reads,
    `get_ind_internal_norm_value` confirmed entered)."""

    plasma_beta: PlasmaEnergyFromBeta = PlasmaEnergyFromBeta()
    """`physics/physics.py::PlasmaBeta`, §A row 2.3 (`physics.py:429`). Site of decision
    7 (`i_beta_component`, 8 reads, `get_beta_norm_max_value` confirmed entered). Its
    pure `calculate_plasma_beta` is already what constraint 1 calls -- `CLAUDE.md`'s
    "`Compare`, already present in embryonic form"."""

    plasma_current: ModelNamespace | None = None
    """`physics/plasma_current.py::PlasmaDiamagneticCurrent`, §A row 2.4
    (`physics.py:527`) -- 4 entered functions, 296 entered LOC, none shared. Site of
    decision 1 (`i_plasma_current`) and one read each of 2 and 4, plus the file's one
    read of `.stellarator.iotabar` (`:196`), which the tokamak arm does not take."""

    bootstrap_current: ModelNamespace | None = None
    """`physics/bootstrap_current.py::PlasmaBootstrapCurrent`, §A row 2.5
    (`physics.py:543`) -- 14 entered functions, 1228 entered LOC, unported, no registry
    row. Site of decision 6 (`i_bootstrap_current`).

    §F's finding applies here and changes its shape: the trace enters all 14 functions at
    `i_bootstrap_current = 4`, because `get_bootstrap_current_fraction_value` evaluates
    the whole family and indexes it. So this switch selects a *value from a computed
    vector*, not a subgraph -- one node producing the family plus an index, not an
    occupant per arm."""

    l_h_transition: ModelNamespace | None = None
    """`physics/l_h_transition.py::PlasmaConfinementTransition`, §A row 2.6
    (`physics.py:788`) -- 23 entered functions, 1124 entered LOC, unported. Computes
    every L-H threshold and selects, the same shape as `bootstrap_current` above."""

    scrape_off_layer: ModelNamespace | None = None
    """`physics/scrape_off_layer.py::ScrapeOffLayer`, §A row 2.7 (`physics.py:832`) -- 5
    entered functions, 226 entered LOC, unported."""

    density_limit: ModelNamespace | None = None
    """`physics/density_limit.py::PlasmaDensityLimit`, §A row 2.8 (`physics.py:870`) --
    11 entered functions, 531 entered LOC, 0 shared. Site of decision 8
    (`i_density_limit = 7`), and the third computes-then-selects family.

    One of its eleven, `calculate_greenwald_density_limit`, is already ported (unit #21's
    note); the node is not registered anywhere, because the stellarator path never enters
    this file."""

    current_drive: TokamakCurrentDrive = dataclasses.field(kw_only=True)
    """`physics/current_drive.py::CurrentDrive` and its four injected sources
    (`NeutralBeam`, `ElectronCyclotron`, `LowerHybrid`, `ElectronBernstein`), run from
    `physics.py:593` when `i_hcd_calculations != 0` -- 5 entered functions, 737 entered
    LOC, unported. Site of decision 10 (`i_hcd_primary = 10`, 14 reads here and further
    reads inside three already-ported files)."""

    plasma_fields: PlasmaFields = dataclasses.field(kw_only=True)
    """`physics/plasma_fields.py::PlasmaFields`, injected into `Physics`
    (`physics.py:197`)
    -- 1 entered function, 67 entered LOC. The tokamak's counterpart to
    `.stellarator.poloidal_field_from_rotational_transform`."""

    # ---- the machine ---------------------------------------------------------------

    build: Build = dataclasses.field(kw_only=True)
    """`build.py::Build`, §A row 1 (`caller.py:288`) -- 6 entered functions but 2306 of
    the file's 2360 LOC entered, unported, nothing shared. Site of decision 3
    (`i_single_null`, 4 reads) and one read of decision 10.

    The radial and vertical build is the tokamak's structural spine and has no
    stellarator counterpart at all: `models/stellarator/build.py`'s `Build` is a
    different model in a different file."""

    cicc_superconducting_tf_coil: CiccSuperconductingTfCoil = dataclasses.field(
        kw_only=True
    )
    """`tfcoil/superconducting.py::CICCSuperconductingTFCoil`, §A row 3
    (`caller.py:306`) -- 19 entered functions, 2457 entered LOC, unported, nothing
    shared, and with `tfcoil/base.py::TFCoil` (8 functions, 753 LOC) reached through it
    by inheritance rather than by any call in `caller.py`. Site of decision 17
    (`n_tf_coils`, 17 reads here).

    **This is where the only CoolProp obstacle in the whole tokamak scope sits.**
    `quench_heat_protection_current_density` (`superconducting.py:1366`) reaches
    `tfcoil/quench.py`, 450 CoolProp calls per `_call_models_once` (§D). Everything else
    in the tokamak scope is merely unwritten; this one waits on `next_steps.md` §5's
    unresolved wrapping policy, and it is on the chain constraints 34/35/36/74/75 read.
    """

    pf_coil: ModelNamespace | None = None
    """`pfcoil.py::PFCoil`, §A row 4 (`caller.py:319`) -- with `cs_coil` and `cs_fatigue`
    below, 24 entered functions and 3525 entered LOC, **zero ported, zero shared**. With
    `cicc_superconducting_tf_coil` this is 40 % of the unported tokamak surface.

    Site of decisions 11 and 13 (`i_pf_superconductor`, `n_pf_coil_groups`), and the
    reason `costs` grows back: `model_tree_design.md` §8 step 4c deleted Accounts 222.2
    and 225.2 because a stellarator has no PF coil system, and
    `cost_boundary_inputs.md` category (d) carries the producer `file:line` for each."""

    cs_coil: ModelNamespace | None = None
    """`pfcoil.py::CSCoil`, injected at `main.py:652` -- 11 of `pfcoil.py`'s 24 entered
    functions. The central solenoid, and the site of decision 12
    (`i_cs_superconductor = 1`).

    A slot of its own rather than part of `pf_coil` because PROCESS injects it as a
    separate `Model` with its own switch; a stellarator has none at all
    (`st_init` sets `data.build.iohcl = 0` unconditionally)."""

    cs_fatigue: ModelNamespace | None = None
    """`cs_fatigue.py::CsFatigue`, injected at `main.py:652` and reached through
    `pfcoil.py:3492` -- 1 entered function (`ncycle`), 93 entered LOC."""

    pulse: TokamakPulse = dataclasses.field(kw_only=True)
    """`pulse.py::Pulse`, §A row 5 (`caller.py:322`) -- 12 entered functions, 236 entered
    LOC, 3 shared. Decision 15 (`pulsetimings`) is read at `physics.py:476` and **nowhere
    else in all of `process/models/**`**, which is the sharpest single-site result in §E.

    `large_tokamak_eval.IN.DAT` sets `i_pulsed_plant = 1`, which is why the *cost* side
    of a pulsed plant (Account 225.3) is already a slot -- `costs.energy_storage_cost` --
    and was one of the four pins `tokamak_scope.md` found the tree contradicting."""

    divertor: Divertor = dataclasses.field(kw_only=True)
    """`divertor.py::Divertor`, §A row 6 (`caller.py:324`) -- 5 entered functions, 262
    entered LOC, unported. Site of decision 9 (`i_div_heat_load = 2`, 5 reads) and one
    read of decision 3.

    Not to be confused with `.stellarator.divertor`, which is
    `models/stellarator/divertor.py` and is ported -- a different model of a different
    device's divertor, and one half of the one cycle the stellarator graph has."""

    first_wall: FirstWall = dataclasses.field(kw_only=True)
    """`fw.py::FirstWall`, §A row 7 (`caller.py:327`) -- 6 entered functions, 299 entered
    LOC, unported. Imports `FluidProperties`, but reaches CoolProp **zero** times on this
    reference run: every one of its CoolProp sites is behind
    `.fwbs.i_p_coolant_pumping == MECHANICAL` (2) and the file sets 3. Dormant, not
    absent -- a second tokamak IN.DAT can wake it (§D)."""

    shield: ModelNamespace | None = None
    """`shield.py::Shield`, §A row 8 (`caller.py:329`) -- 4 entered functions, 270
    entered LOC, unported. Decision 14 (`i_shld_primary_heat`) is read in `power.py`, not
    here."""

    vacuum_vessel: VacuumVesselElliptical = dataclasses.field(kw_only=True)
    """`vacuum.py::VacuumVessel`, §A row 9 (`caller.py:331`) -- 3 entered functions.

    **A confirmed registry prediction.** Unit #16 recorded `VacuumVessel` as *"confirmed
    unreachable on the stellarator pipeline, no action needed"*; the tokamak trace
    reaches it. Its file-mate `Vacuum` is ported and is a slot of `.vacuum`, shared."""

    ccfe_hcpb: CcfeHcpb = dataclasses.field(kw_only=True)
    """`blankets/hcpb.py::CCFE_HCPB`, §A row 10 (`caller.py:345`) -- 7 entered functions,
    956 entered LOC, 0 shared, and `blankets/blanket_library.py` (14 functions, 822 LOC)
    reached through it by inheritance (`hcpb.py:25`), never by a call in `caller.py`.

    Three of the seven are ported (unit #13) and none is registered: on the stellarator
    they are reachable only through `blanket_neutronics()`, whose live PROCESS call-site
    bug blocks that arm. The tokamak reaches them directly.

    Tokamak-only *on this run*: `.fwbs.i_blanket_type = 1` (CCFE_HCPB) is a default, not
    a file setting, and `= 5` routes to `blankets/dcll.py` instead."""

    cryostat: Cryostat = Cryostat()
    """`cryostat.py::Cryostat`, §A row 11 (`caller.py:351`) -- 2 entered functions, 69
    entered LOC). **Not** the stellarator's: that is `stellarator.py:1282-1330`, ported
    as part of unit #1 chunk S5 and already a slot of `.stellarator.fwbs`."""

    structure: Structure = dataclasses.field(kw_only=True)
    """`structure.py::Structure`, §A row 12 (`caller.py:354`) -- 2 entered functions, 200
    entered LOC. **Not** unit 1D, which is `models/stellarator/structure.py`.

    This is the model whose absence let `costs` drop Account 221.4: `st_strc` sets
    `.structure.fncmass`/`.gsmass` to a literal `0.0` on a stellarator, so
    `ReactorStructureCost` computed an exact zero and landed on the right number by luck.
    A tokamak restores both."""

    water_use: ModelNamespace | None = None
    """`water_use.py::WaterUse`, §A row 19 (`caller.py:385`) -- 7 entered functions, 265
    entered LOC, unported and not reached by the stellarator at all. The one balance-of-
    plant model that is *not* device-agnostic, because `caller.py` returns before it on
    the stellarator path."""
