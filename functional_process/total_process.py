"""The device: `StellaratorProcess`, one slot per subsystem, and nothing else.

Each subsystem's own namespace -- the typed slots that name its ported nodes -- lives
beside its models, in `models/<subsystem>/namespace.py`; this module only says which
subsystems a stellarator has. **Not one `i_*` integer appears here.** Turning an IN.DAT
into a `StellaratorProcess`, and that machine into a `Graph`, is `indat.py`'s job, and
`model_tree_design.md` §11 records why the two are separate files.

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
from functional_process.models.vacuum.namespace import Vacuum


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
