"""The devices: `StellaratorProcess` and `TokamakProcess`, one slot per subsystem.

Each subsystem's own namespace -- the typed slots that name its ported nodes -- lives
beside its models, in `models/<subsystem>/namespace.py`; this module only says which
subsystems each device has. **Not one `i_*` integer appears here.** Turning an IN.DAT
into a machine, and that machine into a `Graph`, is `indat.py`'s job, and
`model_tree_design.md` §11 records why the two are separate files.

**Two devices, as siblings, and neither inside the other.** `TokamakProcess` is not a
`StellaratorProcess` with a slot swapped and it does not wrap one: the two share five
device-agnostic subsystems and a plasma-physics core, and differ in the one slot that
holds everything device-specific (`stellarator` / `tokamak`). Written as siblings, no
existing node name moves -- `.costs.blanket_cost` is spelled the same on both machines,
and the reference stellarator graph is bit-identical before and after this class was
added. The measurement that says the sharing is real rather than hoped-for is
`tokamak_call_surface.md` §C: `costs` (40 nodes), `power` (20), `availability` (4),
`vacuum` (3) and `buildings` (2) read no `.stellarator*` data at all, `costs/costs.py`
is entered by *the same 42 functions* on both devices, and `physics` is 31 of 33 shared.

This tree is the whole graph **as it currently exists**, not a claim that the stellarator
MDA is assembled: most nodes in it are still islands with unowned (external) reads, since
their producers haven't been ported yet. It exists so there is always one place the next
ported unit joins, and one place to point a visual inspection at. See
`_audit/unit_registry.md`'s "Ported so far" for what is and isn't in it.

**There is no single graph, and the tree is what says so.** A topology-changing switch
selects which nodes exist, so what this module exports is a `StellaratorProcess` -- a
tree of typed slots, each holding the model that fills it -- and `to_graph(machine)`.
`indat.GRAPH` is `REFERENCE_MACHINE`'s, kept as a module-level name there because
`render_xdsm.py` and the smoke check want a default to point at.
`indat.machine_from_indat` is the only place an `i_*` integer is read, and its docstring
explains why assembly time is the only correct place to resolve one (short version: no
switch in PROCESS is ever an iteration variable or a scan variable, so no switch can
change between two evaluations of one assembled graph).

**The tree itself carries no configuration.** Every slot the factory fills has *no
default*, so `StellaratorProcess()` raises `TypeError` rather than quietly producing one
particular machine; a default is admissible only where there is nothing to decide. That
is the property that would have caught the `i_confinement_time` 34-vs-38 and
`i_plasma_ignited` 0-vs-1 registration bugs, both of which lived in a slot default where
nothing was looking.

`EcrhDensityLimit(i_plasma_pedestal=PlasmaProfileShapeType.PARABOLIC_PROFILE)` keeps its
setting as a static kwarg rather than becoming a slot of its own. It is other category --
`naming_convention.md`'s a formula-changing switch kept as a static kwarg on one node's
`fn` -- because `i_plasma_pedestal != PARABOLIC_PROFILE` has no formula at all in
`density_limits.py` and no node's existence depends on it.

**Every such static kwarg is enum-typed**, per `_audit/model_tree_design.md` §4
("Settings stay on the occupant, enum-typed"): the upstream `IntEnum` where PROCESS
declares one, and `functional_process/models/switch_enums.py`'s minimal local
definition where it does not. `IntEnum` members compare and hash equal to their `int`
values, so nothing numeric moves -- what moves is that `PROCESS_1990` cannot typo into
`KOVARI_2014` the way `0` typos into `1`, the defect class with five recorded instances
(`_audit/switch_elimination_design.md` §5(A)). The bare integers left in the subsystem
namespaces are exactly the two categories that are *not* switches: shape/resolution
counts (`n_plasma_profile_elements`, `n_cs_pf_coils`) and set membership
(`imp_indices`), §3(b)/(c) of that same document.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.availability.namespace import Availability
from functional_process.models.buildings.namespace import Buildings
from functional_process.models.costs.namespace import Costs
from functional_process.models.physics.namespace import Physics
from functional_process.models.power.namespace import Power
from functional_process.models.stellarator.namespace import Stellarator
from functional_process.models.tokamak.namespace import Tokamak
from functional_process.models.vacuum.namespace import Vacuum


class StellaratorProcess(ModelNamespace):
    """One device, configured whole: every slot in this port, and what fills it.

    **Named for the device, because that is what it is.** `Stellarator` below now *does*
    have a counterpart namespace -- `Tokamak`, every slot of it empty -- and
    `TokamakProcess` below is its sibling class, so a general "machine" is still not what
    this class describes: it describes one of two devices. *Machine* stays the noun for
    an
    **instance** of either -- `machine_from_indat`, `REFERENCE_MACHINE`,
    `graph_for(machine=...)` -- and the mixed vocabulary is deliberate: the class is a
    device, an instance of it is a machine.

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


class TokamakProcess(ModelNamespace):
    """A conventional tokamak: the shared subsystems, and one empty device slot.

    **A sibling of `StellaratorProcess`, not a variant of it and not a container for
    one.** The device is the top-level class because the device is what a machine *is*;
    making "which device" a slot inside one class would have put a `.device.` key in
    front of `.stellarator.*`'s 60-odd node names for nothing, and this port's own rule
    is that a slot is a place and swapping an occupant renames nothing -- a rename of
    every existing node is the opposite of that. So no existing name moves: the reference
    stellarator machine is 159 nodes and 316 declared boundary reads before and after,
    with an identical per-node `(name, type, inputs, outputs)` digest.

    **The duplication between the two classes is five lines and is deliberate.** Every
    shared subsystem is annotated identically in both, and factoring them into a base
    class was declined: a `ModelNamespace` names its nodes by its *public fields* and
    binding order is `vars()` and not the MRO (see `StellaratorProcess`'s last
    paragraph), so a shared base would name nodes through a mechanism the rest of this
    tree deliberately does not use. Five repeated annotations is the cheaper honesty.

    **What this class is for.** It is the smallest thing that can be *assembled*, and
    assembly is what turns `tokamak_scope.md`'s and `tokamak_call_surface.md`'s counts of
    missing switches and missing functions into a count of missing **variables**.
    `tokamak_scope.md` §"Not built, and why" declined to write this class on the ground
    that it *"would refuse at its first slot"* -- `physics.confinement_time` was keyed on
    `istell` and had no tokamak entry. That registry is keyed on `i_confinement_time` now
    and `IterIpb98y2ConfinementTime` (IPB98(y,2), value 34) is a written, harness-tested
    occupant, so the refusal is gone and the same file's §"The order this implies" step 3
    is what this class is.

    Nothing in it is a claim that a tokamak *runs*: `tokamak` below holds twenty-five
    empty slots and the graph this assembles computes no tokamak build, no plasma current
    and no PF coils. `_audit/tokamak_boundary.md` is the honest statement of what that
    costs, variable by variable, and it is a measurement taken *of this class* rather
    than an estimate written beside it.
    """

    costs: Costs = dataclasses.field(kw_only=True)
    """The cost model (`.costs.i_cost_model`), and the same one slot the stellarator has.

    `large_tokamak_eval.IN.DAT:112` sets `i_cost_model = 0` as well, so the 1990 model is
    the occupant here too, and §C measured that this is not a coincidence of two input
    files: **both devices enter the same 42 functions of `costs/costs.py`**, with the
    tok-only and stell-only sets both empty.

    **It will grow, and the growth is already scheduled.** `model_tree_design.md` §8 step
    4c deleted Accounts 221.4, 222.2 and 225.2 because a stellarator has no reactor
    structure, no PF coils and no PF coil power conditioning -- a port-side pruning, not
    a PROCESS-side device difference. `cost_boundary_inputs.md` category (d) carries the
    producer `file:line` for every one a tokamak restores, and the boundary record shows
    `.costs.c2214`/`.c2222`/`.c2252` sitting on this machine's boundary as the evidence.
    """

    tokamak: Tokamak = Tokamak()
    """Everything device-specific -- and it is empty.

    Defaulted, uniquely among the device slots, because there is nothing to decide: not
    one of its twenty-five slots has an occupant to choose between, so `Tokamak()` is the
    only value it can take and a factory argument would be ceremony. The moment one slot
    gains a second occupant this becomes `dataclasses.field(kw_only=True)` like every
    other switched slot, by the rule `StellaratorProcess` states: **a slot the factory
    fills has no default.**

    `.stellarator` has no counterpart default for exactly that reason -- five of its
    slots are switched today."""

    physics: Physics = dataclasses.field(kw_only=True)
    """The shared plasma-physics core -- 31 of its 33 nodes are device-agnostic.

    The two that are not are slots inside it, and both are answered differently here:

    * `.physics.confinement_time.scaling` -- `IterIpb98y2ConfinementTime`, IPB98(y,2),
      `i_confinement_time = 34` (`large_tokamak_eval.IN.DAT:300`), against the Helias
      run's ISS04 (38). Already written and harness-tested; `tokamak_scope.md`'s finding
      that *"the refusal is about the absent device, not about absent physics"* is this.
    * `.physics.profiles.parameterisation.ecrh_density_limit` -- stellarator-only, and
      absent here **by construction rather than by omission**: the file sets
      `i_plasma_pedestal = 1` (`:291`), which selects `ProfileParameterisationPedestal`,
      which has no such slot at all, because PROCESS itself computes no ECRH density
      limit outside `i_plasma_pedestal == 0` (its `else` arm only logs an error).
    """

    power: Power = dataclasses.field(kw_only=True)
    """Thermal and electric power flows -- 20 nodes, no `.stellarator*` read among them.

    §C sharpens the sharing claim in one place: `power.py` gains exactly one subsystem on
    a tokamak, not twenty. Shared, 11 functions / 1522 lines; tokamak-new, `Power.pfpwr`
    and its four `_pf_loss_*` helpers -- the PF-coil power supply, which a stellarator
    has no PF coils to need. It hangs off `.tokamak.pf_coil`, and nothing else in
    `power.py` is device-tied."""

    buildings: Buildings = dataclasses.field(kw_only=True)

    vacuum: Vacuum = Vacuum()
    """Default for the same reason it is on a stellarator: nothing inside it is switched.

    Its file-mate `VacuumVessel` is *not* here -- it is `.tokamak.vacuum_vessel`, empty.
    Unit #16 recorded it as "confirmed unreachable on the stellarator pipeline"; the
    tokamak trace reaches it (`caller.py:331`), which is one of the two registry
    predictions §C confirmed live.
    """

    availability: Availability = dataclasses.field(kw_only=True)
    """Plant availability -- 4 nodes, and the one place the two devices differ by a
    single stack frame.

    The stellarator bypasses `.costs.i_plant_availability`'s dispatch entirely
    (`stellarator.py:175` calls `avail()` directly); the tokamak enters
    `Availability.run`, and at `i_plant_availability = 0`
    (`large_tokamak_eval.IN.DAT:113`) that dispatch lands in the `else` arm at
    `availability.py:116` -- **`avail()`, the same arm the bypass reaches, and the arm
    already ported and registered.** So the difference costs nothing here.
    """
