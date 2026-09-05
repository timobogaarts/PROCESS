# The model tree — one typed pytree of models, superseding switches, the settings tree, and `Configuration`

**Status: settled design, fully implemented.** This is the current architecture of
`functional_process`'s model/switch layer. It superseded `switch_elimination_design.md`
and `path_refactor.md` Part B; both of those are now stubs pointing here. The migration
(six steps, landed 2026-08-20 through 2026-08-27) and every gate measured along the way
are in git history — none of that is needed to use or extend the tree today, only the
design itself.

## 1. The design, in one paragraph

There is **one tree**: `eqx.Module` namespaces whose public fields are **typed slots**,
each slot occupied by a model instance that carries its own settings as
`eqx.field(static=True)` fields (every declaration already did this — this design acts
on that fact instead of splitting it into a parallel settings tree). `Machine()` — all
field defaults — **is** PROCESS's bare-default configuration. A run-specific machine is
built by applying IN.DAT deltas to that tree in one factory, `machine_from_indat`, which
is the **only** place an `i_*` switch integer is ever read. `to_graph(machine)` assembles
the graph; node names are slot paths, stable across occupant swaps.
`Switch`/`Alternative`/`Configuration`/`TOPOLOGY_SWITCHES`/`declarations_for`/
`build_graph`/the `{ROOT: subtree}` idiom are all deleted — there is no separate
registry mechanism any more.

```python
class Physics(ModelNamespace):
    profiles: Profiles = Profiles()
    confinement_time: ConfinementTime = ConfinementTimeISS04(rad_loss=RadLoss.CORE_ONLY)


class Machine(ModelNamespace):
    physics: Physics = Physics()
    costs: CostModel | None = (
        None  # PROCESS's default cost model is unported: honestly absent
    )


HELIAS = machine_from_indat(REFERENCE_INPUT_FILE)  # Machine() + IN.DAT deltas
GRAPH = to_graph(HELIAS)
```

Swapping a model is a functional update at a named, typed address, the same idiom as the
variable side and as all of JAX: `eqx.tree_at(lambda m: m.physics.confinement_time,
HELIAS, ConfinementTimePetty08())`.

## 2. What this bought, compressed to the shape of the gap it closed

Every open gap the previous integer-keyed-registry mechanism accumulated was a symptom of
keying arms by integer in a flat registry instead of placing occupants in a tree. Placing
them made each gap stop being expressible, not merely patched: one arm setting several
literal values collapses to two registry entries mapping to one occupant class; nested
switches (`i_tf_sup` under `i_cost_model`) are just a nested slot; exclusivity is
structural (one field, one occupant) instead of proved by output collision; an arm that
sets two unrelated places is just the factory setting two slots (a `Switch`'s `place`
conflated *selection* with *placement*; a factory is selection, the tree is placement);
"unported"/"absent"/"present" are a registry miss / `None` in an `X | None` slot / an
occupant — absence spelled as absence, not a three-state flag.

## 3. The tree surface

`ModelNamespace` (a marker base class in `cottax.interfaces.pytree_namespace_module`)
lets `to_graph` walk an **instance** the same way it already walked a namespace class:
public dataclass fields, in declaration order, each one key deeper as a `GetAttrKey` —
"a node's name is the access path into the tree that declared it, kind follows the
container", the same rule `VarPath`'s recorder already applies to variables. A field
holding `None` contributes nothing (`unproduced`, §4); a field holding anything else
unexpected raises, deliberately stricter than the old class-walker's silent skip, because
a `ModelNamespace`'s fields are all deliberate.

**Node identity is the slot path, not the occupant class** — swapping a model no longer
renames the node, and XDSM/DSM diffs between two machines align row for row. The known
cost is that the occupant's class name no longer appears in the node's name; a
`slot: OccupantClass` renderer label is deferred until a drawing actually needs it, not
built pre-emptively. Slot naming is snake_case throughout, including never-switched
nodes, so the whole tree reads uniformly.

## 4. Slots, families, occupants

A slot's type annotation names its family, chosen by what the arms share: **arms sharing
a body** subclass the shared concrete class (the derived-signature pattern —
`_rebound_signature` — copies the base's signature and replaces one default, so an arm
can't drift when the base gains a parameter); **arms sharing nothing** get an empty
abstract family base purely as documentation; **an arm that is several nodes** (e.g. the
43-node 1990 cost model) is a `ModelNamespace` subtree, and ragged arms (unequal shape,
unequal output sets between two occupants of one slot) are fine — the boundary
postcondition (§5) is what checks the consequences, not the type system.

**The hard floor, absolute in one direction**: a value that changes the reads-set is a
different occupant class; unioning reads to keep a family inside one node is forbidden —
it invents edges only one variant uses, the exact defect the switch architecture existed
to remove. For reads-*identical* values: **closed option sets** (a numerical scheme, an
accounting convention — `ipnet`, `istore`, `iohcl`) stay `eqx.field(static=True)`,
enum-typed (never a bare `int` — an enum can't typo into another enum member the way `0`
typos into `1`, the defect class with five confirmed instances); **formula families that
plausibly grow** (e.g. `i_beta_fast_alpha`, two published formulas today) default to
occupants *even while today's members share reads*, because an enum family's only escape
when a third member needs a new read is the forbidden union. Promoting a dial to a family
later is a local edit — the slot already exists, node identity is the slot path (stable
across occupants), and consumers bind `VarPath`s, never occupant classes — so nothing
forces the choice to be made once and for all up front.

## 5. `machine_from_indat` — the only place integers live

One factory: read the IN.DAT, apply deltas to `Machine()`, return the tree. Per switched
family, a registry keyed by the upstream `IntEnum` maps a member to a constructing
callable; a member with no registry entry is `UNPORTED` and raises `NotImplementedError`
with a recorded reason (the old `Alternative(unported=...)` reason strings move here
verbatim); a value in neither raises naming the enum and its known members. Joint
dispatch (e.g. `blktmodel` × `ipowerflow` × `blkttype`) and cross-slot coherence (one
switch setting two unrelated slots) are both just ordinary code in the factory — no
special mechanism needed. `Machine()`'s field defaults are pinned against PROCESS's bare
`DataStructure()` defaults by test, the same contract the old `Switch.default` pin
enforced.

## 6. The boundary postcondition

One check, after assembly: **every read has an owner or is on the declared boundary**
(`allowed`, pinned from the audited boundary set). This single function is simultaneously
the partial-output-overlap hazard check (occupant B replaces A, which also owned `y`;
`y`'s consumers orphan — caught here, on the consumer side, not the producer side), the
`unowned_inputs`-must-not-grow postcondition for what a design selection means, and the
swap contract ("swap the occupant, rebuild, assert no read lost its owner"). It lives in
`functional_process`, not cottax: cottax's `Graph` already exposes unowned inputs, this
is policy about which ones a given machine is allowed to have. `switch_audit` survives
unchanged in mechanism — it still introspects `eqx.field(static=True)` attributes on
graph-held instances, which occupants still are.

## 7. Serialisation and settings — derived, not a second tree

No parallel settings tree. Walking the model tree yields `{slot_path: (occupant class
name, {static field: value})}`, and that single derived view serves as the JSON form, the
diff between two machines, and a schema check (field names and enum types validated by
construction, since loading goes through real constructors) — all without a second tree
or a `materialise` step. A settings sweep still costs recompiles, not `vmap`, because
settings live in the treedef; that's correct and unchanged from before this design.

## 8. What this does not solve

- **`free` and what a design vector means** — untouched; this design gives a *selection*
  a spelling, not a design vector a meaning.
- **Whether PROCESS's remaining coupling is cyclic** — unaffected; the tree names models,
  `Blocking` still derives structure from reads/writes as before.
- **Fused-node splitting** (`ComponentThermalPowers`, ~5 models fused into one node) —
  still real, still gated on a per-branch reads/writes analysis nobody has done. The tree
  gives each fragment a slot to land in; it doesn't do the split.

## 9. Repository layout convention

Filenames follow slot names, not the audit's original chunking (`physics_A_pure_
formulas.py` → `physics/pure_formulas.py`, etc. — done 2026-08-25, `git log --follow` on
any current file recovers the rename). The rule, so it doesn't need re-litigating per
file: **the stem names what is in the file; the directory names the subsystem.** Drop the
audit chunk letter and the subsystem prefix (the directory already carries the
subsystem); where no directory carries it, create one; where the residue would be
PROCESS's abbreviated source-routine name rather than a description of content, use the
content name instead.

Each subsystem's `ModelNamespace` classes live in that subsystem's own
`models/<subsystem>/namespace.py` (not the package `__init__.py` — an `__init__.py` that
imports every model in a package drags in foreign subsystems on any single-unit import,
and namespaces genuinely import across subsystems on physical grounds, e.g. `physics`
naming two `stellarator` nodes). Everything that knows PROCESS's *input encoding*
(switches, registries, `UNPORTED`, the factory) lives in `indat.py`, separately from the
tree — **the tree knows no switches and the factory knows them all**. `total_process.py`
holds only `StellaratorProcess`, the top-level slot-per-subsystem assembly.
