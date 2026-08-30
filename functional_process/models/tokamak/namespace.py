"""The tokamak's own subsystems -- twenty-eight filled, one still empty.

Beside the nodes it names (`model_tree_design.md` §11), exactly as
`models/stellarator/namespace.py` sits beside the stellarator's. **It shipped with not
one slot occupied**; the first tokamak porting wave filled fourteen of the twenty-five,
the second and third waves' consolidation filled eleven more (and added three --
`diamagnetic_current`, `pfirsch_schluter_current`, `current_fractions` -- that
`bootstrap_current.md` found no home for), the cold-boundary wave (2026-08-27) added
`first_wall_geometry`, the missing-producer wave (2026-08-30) added `cs_fatigue`, and
the one that is left keeps the original
spelling `... | None = None`, which cottax reads as *"an unproduced slot: it assembles
nothing, and whatever read its outputs surfaces as a boundary input. Absence, spelled
as absence."* That one is `water_use`, and it is a scoping record rather than a pending
port: `water_use.md` measured that nothing in `process/` reads any `.water_use.*`
output. `cs_fatigue` was the other, on the same footing -- until its reader turned out
to be live (`cs_fatigue.py`'s module docstring, and constraint 90 on
`low_aspect_ratio_DEMO`), which is the difference between the two cases and why only
one of them moved.

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

**What is deliberately *not* here.** The five near-device-agnostic subsystems --
`costs` (40 nodes), `power` (21), `availability` (4), `vacuum` (3), `buildings` (2) --
are slots of `TokamakProcess` itself, beside `tokamak`, because they are not the
tokamak's: measured, they touch no `.stellarator*` data at all, and `costs/costs.py` is
entered by *the same 42 functions* on both devices (§C). **"Agnostic" is one node weaker
than it was as of 2026-08-30**: `power` gained `pf_coil_power` (`Power.pfpwr`, the
PF-coil power supply), which a stellarator has no PF coils to need and which is `None`
in its `Power`. That is still not a reason to move the namespace here -- twenty of its
twenty-one nodes are shared, and §C measured that `pfpwr` is the *only* subsystem of
`power.py` a tokamak adds -- but the slot is a genuine device difference inside an
otherwise shared namespace, so the count above is now "one slot that can be absent"
rather than "no device dependence at all". `physics` likewise: 31 of its 33
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
from functional_process.models.cs_fatigue import CsFatigue
from functional_process.models.fw import FirstWall, FirstWallGeometry
from functional_process.models.namespace import Build, Divertor
from functional_process.models.pfcoil.namespace import CSCoil, PFCoil
from functional_process.models.physics.bootstrap_current import (
    BootstrapCurrentFractionScaling,
    PlasmaCurrentFractions,
    PlasmaDiamagneticCurrentFraction,
    PlasmaPfirschSchluterCurrentFraction,
)
from functional_process.models.physics.density_limit import TokamakDensityLimit
from functional_process.models.physics.l_h_transition import LHThresholdPower
from functional_process.models.physics.plasma_current import TokamakPlasmaCurrent
from functional_process.models.physics.plasma_fields import PlasmaFields
from functional_process.models.physics.plasma_inductance import TokamakPlasmaInductance
from functional_process.models.physics.scrape_off_layer import TokamakScrapeOffLayer
from functional_process.models.physics.tokamak_namespace import (
    TokamakCurrentDrive,
    TokamakPhysics,
    TokamakPlasmaBeta,
    TokamakPlasmaGeom,
    TokamakPulse,
)
from functional_process.models.shield import TokamakShield
from functional_process.models.structure import Structure
from functional_process.models.tfcoil.namespace import CiccSuperconductingTfCoil
from functional_process.models.vacuum.vacuum import VacuumVesselElliptical


class Tokamak(ModelNamespace):
    """Everything a conventional tokamak has and a stellarator does not.

    Twenty-nine slots; **twenty-eight of them now have occupants and one is still
    `None`.** A namespace with `None` slots contributes no node for them and is
    explicitly allowed by cottax (`ModelNamespace`'s own refusal is for a namespace with
    no *slots*, which is a wrong argument rather than an empty one), so the one that
    is still empty behaves exactly as all twenty-five originals did: whatever reads
    its outputs surfaces as a boundary input, enumerated by name in
    `_audit/tokamak_boundary.md`.

    **The annotation is the promise, and it is now kept slot by slot.** The class this
    file shipped with typed every slot `ModelNamespace | None` on the ground that *"there
    is no class to name yet"*, and said a slot would gain a real annotation the day it
    gained a real occupant, the way `physics.confinement_time.scaling` did.
    Twenty-eight have. Twelve of those are annotated with a **node** rather than a
    namespace (`plasma_beta`, `first_wall`, `first_wall_geometry`, `structure`,
    `cryostat`, `cs_fatigue`, `vacuum_vessel`, `bootstrap_current`,
    `diamagnetic_current`, `pfirsch_schluter_current`, `current_fractions`,
    `l_h_transition`), because a
    slot may hold either -- `Physics.fusion_rates` always has -- and wrapping one node in
    a namespace to make the types uniform would put a meaningless key in front of its
    name.

    **A slot the factory fills has no default**, here as everywhere else in this tree.
    Twenty-three of the twenty-eight are `dataclasses.field(kw_only=True)`; the five
    that keep a default (`plasma_beta`, `cryostat`, `cs_fatigue`, `current_fractions`,
    `first_wall_geometry`) are the ones
    with nothing to decide -- no switch anywhere beneath them, so no configuration for
    a default to smuggle in. That distinction is not cosmetic: a
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

    plasma_inductance: TokamakPlasmaInductance = dataclasses.field(kw_only=True)
    """`physics/physics.py::PlasmaInductance`, §A row 2.1 -- injected into `Physics` and
    run at `physics.py:356`. Site of decision 5 (`i_ind_plasma_internal_norm`, 4 reads,
    `get_ind_internal_norm_value` confirmed entered). Three slots; its Wesson occupant
    supersedes `plasma_current.py`'s `WessonInternalInductance` -- see
    `TokamakPlasmaCurrent`'s docstring for the ownership argument."""

    plasma_beta: TokamakPlasmaBeta = dataclasses.field(kw_only=True)
    """`physics/physics.py::PlasmaBeta`, §A row 2.3 (`physics.py:429`). Site of decision
    7 (`i_beta_component`, 8 reads, `get_beta_norm_max_value` confirmed entered). Its
    pure `calculate_plasma_beta` is already what constraint 1 calls -- `CLAUDE.md`'s
    "`Compare`, already present in embryonic form".

    **A node until 2026-08-27, a namespace and a factory-filled slot since.** It gained
    the beta-limit block that closes constraint 24's three §11.5 missing producers, and
    with it a real switch (`i_beta_norm_max`) -- so it moves out of the four
    "nothing to decide, keep the default" slots the docstring above enumerates and into
    the twenty-three the factory fills. The list above is left as it was written
    because it is an argument about a distinction, not a census; this slot is the
    worked example of a slot crossing it, and of *why* the distinction is not cosmetic:
    defaulted, a file asking for the Menard scaling would have been handed Wesson."""

    plasma_current: TokamakPlasmaCurrent = dataclasses.field(kw_only=True)
    """`physics/plasma_current.py::PlasmaCurrent`, §A row 2.4
    (`physics.py:527`) -- 4 entered functions, 296 entered LOC, none shared. Site of
    decision 1 (`i_plasma_current`) and one read each of 2 and 4, plus the file's one
    read of `.stellarator.iotabar` (`:196`), which the tokamak arm does not take.
    Three slots (the record's fourth moved to `plasma_inductance` -- its own open
    question 1's rule, applied)."""

    bootstrap_current: BootstrapCurrentFractionScaling | None = dataclasses.field(
        kw_only=True
    )
    """`physics/bootstrap_current.py::PlasmaBootstrapCurrent`, §A row 2.5
    (`physics.py:543`) -- 14 entered functions, 1228 entered LOC. Site of decision 6
    (`i_bootstrap_current`): `4` (Sauter) is written, `0` (USER_INPUT) is an **empty
    slot** (`.current_drive.f_c_plasma_bootstrap` becomes a boundary input, and it is
    an `IN.DAT` variable), the other twelve are UNPORTED.

    §F's "one node producing the family plus an index" reading is **overridden** by the
    settled computes-then-selects policy (`bootstrap_current.md` open question 2, wave
    coordinator 2026-08-26): one occupant class per switch value, each declaring only
    its own arm's reads. The unselected thirteen family members are dead work and are
    not computed."""

    diamagnetic_current: PlasmaDiamagneticCurrentFraction = dataclasses.field(
        kw_only=True
    )
    """`physics/plasma_current.py::PlasmaDiamagneticCurrent` (`physics.py:527`) --
    **a new slot, not one of the traced twenty-five**: `tokamak_boundary.md` folded it
    into `plasma_current`, whose record then declared it out of scope, so
    `bootstrap_current.md`'s registration instructions gave it a home beside its only
    consumer (`current_fractions` below). `i_diamagnetic_current == 0` (none) is
    written; `1`/`2` (Hender/SCENE fits) are UNPORTED."""

    pfirsch_schluter_current: PlasmaPfirschSchluterCurrentFraction = dataclasses.field(
        kw_only=True
    )
    """`physics.py:534-541`'s Pfirsch-Schluter fraction -- a new slot on the same
    grounds as `diamagnetic_current`. `i_pfirsch_schluter_current == 0` (none) is
    written; `1` (SCENE fit) is UNPORTED."""

    current_fractions: PlasmaCurrentFractions = PlasmaCurrentFractions()
    """`Physics.calculate_plasma_current_fractions` (`physics.py:558-591`) -- sums the
    bootstrap, diamagnetic and Pfirsch-Schluter fractions into the inductive and
    auxiliary ones. Unswitched; the slot that closes `HcdPrimaryInjectedPower`'s
    `.physics.f_c_plasma_auxiliary` boundary read."""

    l_h_transition: LHThresholdPower = dataclasses.field(kw_only=True)
    """`physics/l_h_transition.py::PlasmaConfinementTransition`, §A row 2.6
    (`physics.py:788`) -- 23 entered functions, 1124 entered LOC. A single node, not a
    namespace: one switch (`i_l_h_threshold`, 21 values), one owned `VarPath`
    (`.physics.p_l_h_threshold_mw`). The Martin-2008 family (`6`-`8`, `19`-`21`) has
    written occupants, `19` (aspect-corrected nominal) live; the other fifteen values'
    formulas are ported and Tier-1-tested but have no occupant yet
    (`l_h_transition.md` "## ported"). The computes-then-selects reading is overridden
    the same way `bootstrap_current`'s is."""

    scrape_off_layer: TokamakScrapeOffLayer = dataclasses.field(kw_only=True)
    """`physics/scrape_off_layer.py::ScrapeOffLayer`, §A row 2.7 (`physics.py:832`) -- 5
    entered functions, 226 entered LOC. Eight flat slots, one switched
    (`i_len_sol_outboard_power_decay`)."""

    density_limit: TokamakDensityLimit = dataclasses.field(kw_only=True)
    """`physics/density_limit.py::PlasmaDensityLimit`, §A row 2.8 (`physics.py:870`) --
    11 entered functions, 531 entered LOC, 0 shared. Site of decision 8
    (`i_density_limit = 7`). Three slots: the Greenwald limit and fraction are
    unconditional (PROCESS fills the whole family regardless of the switch), and only
    the *enforced* limit answers `i_density_limit`."""

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

    pf_coil: PFCoil = dataclasses.field(kw_only=True)
    """`pfcoil.py::PFCoil`, §A row 4 (`caller.py:319`) -- twelve slots including the
    inductance matrix. Site of decisions 11 and 13 (`i_pf_superconductor`,
    `n_pf_coil_groups`), resolved with the rest of the package's joint predicate by
    `indat._pf_coil_system_arm` -- one predicate, fifteen slots, resolved once
    (`models/pfcoil/namespace.py`'s own docstring).

    The registration closes `Structure`'s and `Cryostat`'s three PF-coil boundary
    reads, and it is why `costs` grows back: `model_tree_design.md` §8 step 4c deleted
    Accounts 222.2 and 225.2 because a stellarator has no PF coil system, and
    `cost_boundary_inputs.md` category (d) carries the producer `file:line` for each.
    The package's four-node cycle is cut and driven in `mda.py`, not here."""

    cs_coil: CSCoil | None = dataclasses.field(kw_only=True)
    """`pfcoil.py::CSCoil`, injected at `main.py:652` -- seven slots. The central
    solenoid, and the site of decision 12 (`i_cs_superconductor = 1`).

    A slot of its own rather than part of `pf_coil` because PROCESS injects it as a
    separate `Model` with its own switch; a stellarator has none at all
    (`st_init` sets `data.build.iohcl = 0` unconditionally).

    **`| None` since 2026-08-30, and a tokamak can now take that arm too.** The
    spherical tokamaks set `iohcl = 0` in their own IN.DATs
    (`spherical_tokamak_eval.IN.DAT:69`, `st_regression.IN.DAT:1485`), and `pfcoil()`
    then never calls `ohcalc` (`pfcoil.py:1048-1050`) -- so this is the same absence the
    stellarator has, arrived at from an input rather than from `st_init`. `indat.CS_COIL`
    maps arm 2 to `None` and every `.tokamak.pf_coil` occupant on that arm drops the CS
    reads instead of reading zeros; that is the difference between a machine without a
    solenoid and a machine whose solenoid has no size."""

    cs_fatigue: CsFatigue = CsFatigue()
    """`cs_fatigue.py::CsFatigue`, injected at `main.py:652` and reached through
    `pfcoil.py:3492` -- 1 entered function (`ncycle`), 93 entered LOC.

    **Filled 2026-08-30, and the empty slot was not free.** `.cs_fatigue.n_cycle` is
    constraint 90's operand, and `low_aspect_ratio_DEMO.IN.DAT` activates that
    constraint: with no owner the field sat at its `0.0` default, so `1 - 0 /
    n_cycle_min` evaluated to exactly `+1.000000` with an identically zero gradient row
    and both of that machine's SAND cells stopped at zero iterations. A node, not a
    namespace, for the same reason `l_h_transition` is one: one function, one owned
    `VarPath`. A default rather than a factory-filled slot because there is no switch
    anywhere beneath it -- the `plasma_beta`/`cryostat` rule, and `ncycle` has no
    branch of any kind.

    The `f_c_plasma_inductive` guard PROCESS applies at the *call* site
    (`pfcoil.py:3488`) lives on the occupant, as a `jnp.where`; `cs_fatigue.py`'s node
    docstring says why it cannot be resolved at assembly time the way an `i_*` switch
    is."""

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
    absent -- a second tokamak IN.DAT can wake it (§D).

    A family of two since 2026-08-27: `.divertor.n_divertors` picks the single-null or
    the double-null occupant, and they differ in reads."""

    first_wall_geometry: FirstWallGeometry = FirstWallGeometry()
    """`fw.py::FirstWall.set_fw_geometry` (`fw.py:347-352`) -- **a new slot, not one of
    the traced twenty-five**, on the same grounds as `diamagnetic_current`:
    `tokamak_boundary.md` folded it into `first_wall`, whose record then declared it out
    of that slot's boundary, and `cold_boundary.md` (2026-08-27) measured it as producer
    1 of the cold MDA's missing four -- `.build.dr_fw_inboard`/`.build.dr_fw_outboard`,
    the boundary zeros behind 7 of the 11 non-finite roots. A slot of its own because
    `first_wall` reads both fields it would otherwise own; a default because there is no
    switch anywhere beneath it (the `plasma_beta`/`cryostat` rule)."""

    shield: TokamakShield = dataclasses.field(kw_only=True)
    """`shield.py::Shield`, §A row 8 (`caller.py:329`) -- 4 entered functions, 270
    entered LOC. Decision 14 (`i_shld_primary_heat`) is read in `power.py`, not here.
    Two slots; the volumes slot's D-shaped arm joins the existing
    `_fw_blkt_vv_shape_arm` joint key rather than minting a second."""

    vacuum_vessel: VacuumVesselElliptical = dataclasses.field(kw_only=True)
    """`vacuum.py::VacuumVessel`, §A row 9 (`caller.py:331`) -- 3 entered functions.

    **A confirmed registry prediction.** Unit #16 recorded `VacuumVessel` as *"confirmed
    unreachable on the stellarator pipeline, no action needed"*; the tokamak trace
    reaches it. Its file-mate `Vacuum` is ported and is a slot of `.vacuum`, shared.

    A family of two since 2026-08-27: `.divertor.n_divertors` picks the single-null or
    the double-null occupant, the latter reading seven fields fewer."""

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
