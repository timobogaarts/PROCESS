# The model tree — one typed pytree of models, superseding switches, the settings tree, and `Configuration`

**Status: settled design; §8 steps 1–4d are implemented.** Step 4d landed 2026-08-25
and closes the last four of `switch_kwarg_survey.md` band (a)'s live incoherences: a
switch that decides a slot is now read in exactly one place and threaded, one switch the
factory had no business reading at all (`i_plasma_pedestal`, which `st_init` overwrites)
is read from PROCESS's forcing instead, and `costs.cost_of_electricity` becomes a slot
whose other arm is **absent** rather than a node its own constructor says must not exist.
**The reference machine does not move by one number**; the reference tree is still 156
nodes. Full account in step 4d below. Step 5 is next and unblocked.

*(Earlier status.)* Step 4c landed 2026-08-25 and
is two corrections rather than a re-spelling: the joint blanket keys are derived from
legal switch values by two named functions instead of from an illegal `blktmodel = 2`
sentinel, and three cost nodes for a subsystem a stellarator does not have are deleted.
**The tree is 156 nodes, not 159.** Full account in step 4c below.

*(Earlier status.)* Step 4 landed
2026-08-25: the ten switches are ten typed slots, `machine_from_indat` is the only place
an `i_*` integer is read, and `configuration.py` — `Switch`, `Alternative`,
`Configuration`, `build_graph`, `declarations_for` — is deleted along with
`TOPOLOGY_SWITCHES` (505 lines). All 159 nodes are now in the tree; the 59 that were bare
class-named tuples at the root are slot-named, 43 of them under `.costs.*`. Gates, all
strict and all met: node count, block count, driven count, unowned inputs and the per-node
(inputs, outputs, type) signature identical; MDA harness identical; SAND C2 and C3
**byte-identical** to the pre-step-4 run (31 it / `objf 1.217757347`, 99 it /
`1.217757378` — measured against `HEAD`, not against this file's stale 42/100). Step 5 is
next and unblocked.

**One deviation from §4, stated rather than hidden:** a switched slot whose occupants
share no base is annotated with a **union of its occupant classes**, not with an empty
abstract family base. Only `ConfinementTime` had a real family
(`StellaratorConfinementTime` subclasses it); creating eight new abstract bases across
eight model modules is churn this step did not need, and §4's own stated purpose for a
base — "the slot's type and the family's documentation" — is served by the union plus the
slot docstring. Step 6 is where families acquire bases, and promoting a union to a base is
the *local* edit §4 already promises it is.

*(Earlier status.)* Step 3
landed 2026-08-25: `total_process.COMMON`'s nested-class tree is eleven `ModelNamespace`
classes and `COMMON` is now a `Machine()` **instance**, so every node is named by its
snake_case slot path (`.stellarator.coils.coil_current`) and `configuration.ROOT` /
the `{ROOT: subtree}` idiom are deleted. Gates all identical: 159 nodes, 139 blocks /
14 driven, 348 unowned inputs, a byte-identical per-node input/output/type signature,
MDA harness 499/34/3/0 · 557/0 · 61/0/3/0, suite 3730. Two name literals needed
re-pinning and both are recorded (`path_refactor.md` §B.5's correction). Steps 4–6 are
next and unblocked.

*(Earlier status, kept for the step 1/2 gate numbers.)* §8 steps 1 and 2 (2026-08-20) —
step 1 with every gate identical (3724 passed; harness 499/34/3/0, 61/0/3/0,
`not_enum_typed` empty; the 16 local enums live in
`functional_process/models/switch_enums.py`), step 2 green in `~/jaxgraph` (501 passed,
3 skipped) with slots minting `GetAttrKey`s per §3.1 and no rendering change per §3.3.
Steps 3–6 wait on `path_refactor.md` Part A.** This document supersedes
`switch_elimination_design.md` §§5–6, §10, §12–13 (design content; its measurements stay
live and are cited below) and `path_refactor.md` Part B §§B.3–B.4b/B.6 (the nested-class
tree and the switch-arm placement question). `next_steps.md` §1's four open
`Switch`/`Alternative` gaps and §12.2's alternatives mechanism resolve here; §12.1
(`free`) and §12.3 (design-variable meaning) are **orthogonal and unchanged** — this
document is how a *selection* is spelled, `free` is what a selection *means*, and the two
share exactly one artifact (the boundary postcondition, §6).

**Sequencing constraint carried over unchanged:** `path_refactor.md` Part A (the
`From`/`OutputInto` conversion, 2157 sites) lands first and whole. This design replaces
Part B; it does not touch Part A.

## 1. The one-paragraph design

There is **one tree**: `eqx.Module` namespaces whose public fields are **typed slots**,
each slot occupied by a model instance that carries its own settings as
`eqx.field(static=True)` fields — which every declaration already does
(`switch_elimination_design.md` §10.2: "model and settings are already one pytree — per
node"; this design acts on that observation instead of splitting it back into parallel
trees). `Machine()` — all field defaults — **is** PROCESS's bare-default configuration.
A run-specific machine is built by applying IN.DAT deltas to that full default tree in
one factory, `machine_from_indat`, which is the only place an `i_*` integer is ever
read. `to_graph(machine)` assembles the graph; node names are slot paths, stable across
occupant swaps. `Switch`, `Alternative`, `Configuration`, `TOPOLOGY_SWITCHES`,
`declarations_for`, `build_graph` and the `{ROOT: subtree}` idiom are all deleted.

```python
class Profiles(ModelNamespace):
    grid: ProfileGrid = ProfileGrid(n_plasma_profile_elements=201)
    parameterisation: ProfileParameterisation = ProfileParameterisationParabolic()

class Physics(ModelNamespace):
    profiles: Profiles = Profiles()
    confinement_time: ConfinementTime = ConfinementTimeISS04(rad_loss=RadLoss.CORE_ONLY)
    fusion_rates: FusionRates = FusionRates()

class Machine(ModelNamespace):
    physics: Physics = Physics()
    costs: CostModel | None = None          # PROCESS's default cost model is unported: honestly absent
    heating: HeatingModel = HeatingECRH()

HELIAS = machine_from_indat(REFERENCE_INPUT_FILE)   # Machine() + IN.DAT deltas
GRAPH  = to_graph(HELIAS)
```

Swapping a model is a functional update at a named, typed address — the same idiom as
the variable side and as all of JAX:

```python
variant = eqx.tree_at(lambda m: m.physics.confinement_time, HELIAS,
                      ConfinementTimePetty08())
```

## 2. Why the previous designs are superseded, in one table

Every open gap the previous mechanism accumulated is a symptom of keying arms by integer
in a flat registry instead of placing occupants in a tree. The resolutions are not
patches; each gap stops being expressible:

| open gap (where recorded) | resolution here |
|---|---|
| One arm, several literal values — `blkttype in {1,2}` vs `3` (`next_steps.md` §1) | Two registry entries mapping to one occupant class. The integer was never the identity. §5. |
| Nested switches — `i_tf_sup` under `i_cost_model`, the resistive-TF cost graph unassemblable (`next_steps.md` §1, §9) | Trees nest. `CostModel1990(tf_magnet=TfMagnetCostSC())` is a slot inside an occupant. §4. |
| Exclusivity proved only by output collision (`next_steps.md` §1, §12.2) | One field, one occupant: exclusive by construction. The *real* hazard — partial output overlap — moves to the consumer-side boundary postcondition, which is where §12.2 already said it belongs. §6. |
| Two values selecting the identical node set — `i_tf_sup` 0 and 2 (`next_steps.md` §1) | Two registry entries, one class. §5. |
| Synthetic comma-joined switch path `.fwbs.blktmodel,.fwbs.blkttype` | One slot; the factory reads both integers to choose its occupant. No synthetic path exists. §5. |
| One arm, two places — `istell == 6` sets `StellaratorMachineConfig` *and* the confinement binding, which killed per-`Switch` `place` (`path_refactor.md` §B.4) | The factory sets two slots. `Switch` conflated *selection* with *placement*; a factory is selection, the tree is placement. §5. |
| Three-state `Alternative` (`declarations`/`unported`/`unproduced`) (`next_steps.md` §9) | Present occupant / `Unported(reason)` registry entry that refuses at parse / `None` in an `X \| None` slot that assembles nothing. Absence is spelled as absence. §4, §5. |
| Settings tree + `materialise` + the `is_leaf` zero-leaf wrinkle (`switch_elimination_design.md` §12.7, §13) | No second tree. Settings are static fields on the occupants — where they already live. The three uses of a settings tree (diff, serialise, audit) are served by a *derived* view. §7. |
| Swap has an address but no contract (`path_refactor.md` §B.4b) | The boundary postcondition: rebuild, assert no read lost its owner. §6. |
| "Dials that should be slots" — 28 registrations mixing genuine settings with mis-filed model choices (`path_refactor.md` §B.4b item 2) | §4's refined criterion decides per field: reads-set differs → occupant class, always; reads-identical → enum only for a *closed* option set, occupants for a growing formula family. The mis-filed ones are scheduled work: §8 step 6. |

## 3. The tree surface — what cottax gains

### 3.1 `ModelNamespace`

A marker base class, exported by `cottax.interfaces.pytree_namespace_module`:

```python
class ModelNamespace(eqx.Module):
    '''A namespace of models, as an instance: public fields are slots.'''
```

`node_and_names` gains one branch, tested **after** `isinstance(x, NodalDeclaration)`
(a declaration is both a Module and a leaf, and the leaf reading is the one every caller
means — the same ordering rule the class-namespace branch already documents):

- a `ModelNamespace` instance is a namespace; its **public dataclass fields, in
  declaration order** (declaration order is binding order, the same invariant the class
  walker keeps), are its entries;
- each entry sits **one key deeper, under the field name, as a `GetAttrKey`** — the key
  *replaces* the declaration's name (the existing mapping rule), but its kind records the
  truth: a slot is reached by attribute access, so its name is spelled the way it is
  reached. **The naming rule, stated once: a node's name is the access path into the tree
  that declared it — kind follows the container**, exactly as `VarPath`'s recorder works.
  Mapping keys therefore stay `DictKey` (that *is* how a dict is accessed), and a
  `NodePath` into a `Machine` is a working address — `get_at`/`eqx.tree_at` can take it
  to fetch or swap the occupant, the same one-address-system property `VarPath`s have
  into `DataStructure`. (Decided in review, reversing this document's first draft, which
  had slots minting `DictKey`s for uniformity with the legacy surfaces — rejected because
  a `DictKey` path merely *resembles* the location, and because the namespace distinction
  is carried by the `VarPath`/`NodePath` types, not by key kinds);
- a field holding `None` contributes nothing (this is `unproduced`, §4);
- a field holding anything that is not a namespace entry (`NodalDeclaration`,
  `NodeDefinition` via mapping, `Mapping`, namespace class, sequence, `ModelNamespace`,
  `None`) **raises**. Stricter than the class walker's silent skip, deliberately: a class
  body inevitably carries methods and docstrings, a `ModelNamespace`'s fields are all
  deliberate, so an unexpected value is a mistake, not clutter. Settings do **not** live
  on namespaces — they live on occupants — and this rule is what enforces that.

**An unplaced `ModelNamespace` instance sits at the root.** A namespace *class* is a
definition and keeps its class name when nothing placed it; an *instance* is a
composition — many machines are instances of one `Machine` class, so the class name is
incidental and naming by it would be wrong. This asymmetry is what retires the
`{ROOT: subtree}` idiom: `to_graph(HELIAS)` names its nodes `physics.profiles.grid`,
with no container key to strip.

### 3.2 Node identity is the slot path

`physics.confinement_time` is the node **whichever occupant fills it**. Swapping a model
no longer renames the node; minted names follow via the existing `named(at)` mechanism
(`^problem.physics.confinement_time` is stable across occupants); XDSM/DSM diffs between
two machines align row for row. This answers `switch_elimination_design.md` §12.4's "is
node identity stable" question the right way round: identity is the *place*; only
regrouping renames, and regrouping is rare and deliberate.

**Slot naming convention: snake_case field names throughout**, including never-switched
nodes (`fusion_rates: FusionRates = FusionRates()` — the class correspondence lives in
the annotation, one token away). This renames every node once (§8 step 3's gate handles
it); `path_refactor.md` §B.5 measured the string-reference surface at effectively one
entry, matched with `in`.

The known cost — the class name no longer appears in the node name — is accepted, and
the mitigation (renderers labelling a node `slot: OccupantClass`) is **deferred**: build
it when reading a drawing actually hurts, not before.

### 3.3 Rendering needs no change — the name is the access path

Because slots mint `GetAttrKey`s (§3.1), jax's own `keystr` already renders a
Machine-tree node name as `.physics.confinement_time` and a minted name as
`^cond.physics.beta` (`MintKey.__str__` is `^name`; verified in the env). So
`Path.path_str` **stays `keystr`**, unchanged, and brackets appear exactly where a dict
genuinely is the container (`['physics']` for a mapping-declared name) — which is
truthful, not noise. `spell_flat`/`xDSMFormatterFlat` remain as the opt-in flat spelling
for the legacy `DictKey`-named surfaces. (An earlier draft of this section made the flat
spelling `path_str`'s default; withdrawn in review — it was only needed while slots
minted `DictKey`s, and it erased the dict-vs-attribute distinction in display for any
caller whose data really does contain a dict.) `path_refactor.md` §B.2 item 1 is mooted
for the Machine tree by the key kind itself; it remains open only for the legacy
surfaces it was written about.

### 3.4 What cottax deliberately does *not* gain — considered and rejected in review

- **A closed roster of `MintKey`s.** Rejected: cottax had a forced roster before and
  removed it on purpose. The defaults (`^cond`, `^problem`, `^hat`) are clear at a
  glance; a user or an interface overriding them is a feature, not a leak.
- **Renaming `^hat`.** Rejected: `hat` is the conventional XDSM spelling for a coupling
  copy — the convention is the reason it is named that.
- **Splitting `^cond` into `^res`/`^next`** (RootFind residual vs FixedPoint next
  iterate). Rejected for now: the problem node's type already disambiguates, and the
  rename would touch every driven block for a distinction no reader has yet needed.
- **Any switch/variant concept in cottax.** The whole point: variant selection is the
  *caller's* factory plus the caller's tree. cottax sees only the materialised tree.

The minted namespaces already satisfy the requirement that provoked reviewing them:
`MintKey` is a distinct `KeyEntry` *kind*, so `^cond.physics.beta` cannot collide with
any caller pytree, `is_minted` answers by kind, and the `^` sigil marks the namespace at
a glance. `^problem.*` names the only minted *nodes*; every other `^` name is a
variable. That invariant holds today and is worth a sentence in cottax's docs, nothing
more.

## 4. Slots, families, occupants

**A slot's annotation names its family.** Three cases, decided by what the arms share:

1. **Arms share a body** → occupants subclass the shared concrete class, rebinding reads
   or overriding methods via the derived-signature pattern `next_steps.md` §8.3
   established (`_rebound_signature`: copy the base signature, replace one default, so
   an arm cannot drift when the base gains a parameter). The slot is annotated with the
   base. Existing instance: `StellaratorConfinementTime(ConfinementTime)`.
2. **Arms share nothing** (the `i_tf_sc_mat` shape: 8 genuinely different reads-sets, no
   shared body) → an empty abstract family base exists purely as the slot's type and the
   family's documentation; each occupant subclasses it directly.
3. **An arm is several nodes** (the parabolic-profile arm: 7 nodes; the `istell == 6`
   machine-config payload; the 43-node 1990 cost model) → the occupant is a
   `ModelNamespace` subtree, and the family base is a namespace base class
   (`ProfileParameterisation`, with `ProfileParameterisationParabolic` /
   `ProfileParameterisationPedestal` as subtree subclasses). Ragged arms are fine — two
   occupants of one slot need not have equal shape or equal output sets
   (`next_steps.md` §12.2's "do not require arms to have equal output sets" carries
   over); the boundary postcondition (§6) is what checks the consequences.

**Occupant naming convention:** `<Family><Variant>` — `ConfinementTimeISS04`,
`CostModel1990`, `HeatingECRH`. The family prefix groups alphabetically and reads as
"which slot does this fill"; the variant suffix is the enum member's name where one
exists upstream.

**Typing is documentation plus static analysis, not a runtime gate.** Dataclasses do not
enforce annotations and neither does equinox. A `Machine.__check_init__` walking
annotations with `isinstance` is cheap and allowed, but the check that actually
matters is structural (§6) — a well-typed tree can still orphan a read, and the
postcondition catches both.

**`unproduced` becomes `None`.** A slot typed `X | None` holding `None` assembles
nothing; consumers of the arm's outputs surface as unowned boundary inputs, exactly the
semantics `next_steps.md` §9 argued into the third `Alternative` state — now spelled as
absence instead of as a state flag. Its soundness condition (only valid when no other
producer exists) transfers unchanged and is checked by the same postcondition. The first
instance transfers directly: `Machine()`'s default is `costs=None`, because PROCESS's
default cost model (`KOVARI_2014`) is unported.

**Settings stay on the occupant, enum-typed — and the model/setting boundary is refined
beyond the reads-set test.** The hard floor is `traceability_policy.md`'s split default,
unchanged and absolute in one direction: a value that changes the reads-set is a
different occupant class, and **unioning reads to keep a family inside one node is
forbidden** — it invents edges only one variant uses, the exact sin the switch
architecture was rejected for. For reads-*identical* values the criterion (decided in
review, on extensibility grounds) is *"closed option or growing family?"*:

- **Closed option sets** — a numerical scheme, an accounting convention, a flag whose
  value set is fixed by what PROCESS is (`ipnet`, `istore`, `iohcl`, ...) — stay
  `eqx.field(static=True)`, typed with the upstream `IntEnum` (never a bare `int`:
  `PROCESS_1990` cannot typo into `KOVARI_2014` the way `0` typos into `1` — the defect
  class with five instances, §8.2; `IntEnum == int`, so nothing numeric moves). This is
  their correct *final* form, not a way station.
- **Formula families** — alternative physical models that plausibly acquire members
  (`i_beta_fast_alpha` is two published formulas today) — default to **occupants even
  while today's members share reads**. An enum family is closed: the day a new member
  needs a read the family doesn't express, the enum's only cheap escape is the
  forbidden union. A slot family is open — a new occupant subclass with its own reads
  drops in.

**What keeps this choice low-stakes either way, worth stating so nobody avoids a
conversion by widening reads instead: promoting a dial to a family later is a *local*
edit.** The slot already exists, node identity is the slot path (stable across
occupants, §3.2), and consumers bind `VarPath`s, never occupant classes — so the blast
radius of a promotion is the owning module plus one `machine_from_indat` registry line.
Nothing else in the tree, the graph, or any consumer moves.

The six recorded split-default deviations ("reads-set differs, kept static anyway",
`traceability_policy.md`) sit *below* even the hard floor and are scheduled work, not a
posture: §8 step 6 converts them, family by family.

## 5. `machine_from_indat` — the only place integers live

One factory: read the IN.DAT (and the `*_variables.py` defaults behind it), apply deltas
to `Machine()` — constructor arguments for the literal parts, `eqx.tree_at` for deep
ones — and return the tree. Per switched family, a registry keyed by the upstream enum:

```python
CONFINEMENT_TIME: dict[ConfinementTimeModel, Callable[..., ConfinementTime]] = {
    ConfinementTimeModel.ISS04: ConfinementTimeISS04,
    # absent members are unported; see UNPORTED below
}
UNPORTED = {
    (ConfinementTimeModel, member): "reason, verbatim from the old Alternative(unported=...)"
    for ...
}
```

- **A registry hit** constructs the occupant (a callable, because some occupants need
  payload — `StellaratorMachineConfig(machine_config=read_stellarator_config_file(...))`).
- **An `UNPORTED` hit raises `NotImplementedError` with the recorded reason.** The
  reason strings in today's `Alternative(unported=...)` declarations are real audit
  content and move here verbatim — 52 `Alternative` declarations for `istell` collapse
  to registry absence plus one error message that still names why.
- **A miss on both** raises naming the enum and the known members — the "typo'd value
  fails loudly" property of `Switch.choose`, kept.

Joint dispatch (`blktmodel` × `ipowerflow` × `blkttype`) is ordinary code in the
factory: read the integers, choose one occupant for one slot. Cross-slot coherence
(`istell == 6` sets both the machine config and the confinement binding) is likewise
just the factory setting two slots — with a comment, and if wanted a `Machine`-level
validator, but no mechanism.

**The default contract transfers exactly.** Today:
`test_default_configuration_matches_process_defaults` pins `Switch.default` against the
`data_structure` dataclass defaults. New spelling: **`Machine()`'s field defaults are
PROCESS's bare defaults**, pinned by the same test re-targeted — walk `Machine()`,
compare each slot's implied switch value (via the registries, inverted) and each
occupant's enum-typed settings against a bare `DataStructure()`. A silent IN.DAT still
reproduces PROCESS's own defaults, and `machine_from_indat` on the reference file
reproduces today's `REFERENCE_CONFIGURATION` graph.

## 6. The boundary postcondition — one check, three former owners

After assembly: **every read has an owner or is on the declared boundary.**

```python
def check_boundary(graph, allowed: frozenset[VarPath]) -> None:
    orphans = unowned_inputs(graph) - allowed
    if orphans: raise ...   # naming each orphan and, where known, the slot whose occupant lost it
```

`allowed` for the reference machine is pinned from `boundary_inputs_audit.md`'s audited
set (349 today; regenerate, don't retype). A test asserts the reference machine's
boundary equals the pin; any tree edit that grows it fails naming the new orphans.

This one function is simultaneously:

- the **partial-overlap hazard** check (`next_steps.md` §12.2: occupant B replaces A,
  which also owned `y`; `y`'s consumers orphan; caught here — "the check belongs on
  consumers, not producers", implemented as stated);
- **`free`'s postcondition** (`next_steps.md` §12.1: "`unowned_inputs` must not grow" —
  the missing-producer class, eight instances, never once found by a check);
- the **swap contract** (`path_refactor.md` §B.4b item 1, verbatim: "swap the occupant,
  rebuild, assert no read lost its owner").

It lives in `functional_process`, not cottax — cottax's `Graph` already exposes
unowned inputs; this is policy about which ones a given machine is allowed to have.

`switch_audit` **survives unchanged in mechanism** — it introspects
`eqx.field(static=True)` attributes on graph-held instances, which is exactly what
occupants still are. §8 step 1 makes it enum-aware *first*, per
`switch_elimination_design.md` §8's standing warning: the net that caught five bugs
lands before anything it guards is moved.

## 7. Serialisation and the settings view — derived, later, optional

Walking the tree yields `{slot_path: (occupant class name, {static field: value})}`.
That single derived view is:

- the **JSON form** — `{"physics": {"confinement_time": {"$model":
  "ConfinementTimeISS04", "rad_loss": "CORE_ONLY"}}}` — loaded back through a
  name → class registry;
- the **diff** between two machines (`jax.tree_util` on treedefs, or diff the views);
- the **schema check** `switch_elimination_design.md` §12.7 wanted — field names and
  enum types validated by construction, since loading goes through real constructors.

Nothing here is load-bearing for §8's migration; it is listed so nobody rebuilds a
settings tree to get it. (`switch_elimination_design.md` §13 measured why the schema's
value is modest: 58 of 60 static slots are `int`-typed today — enum-typing them in §8
step 1 is what makes the schema worth anything.)

**A settings sweep still costs recompiles, not `vmap`** — settings live in the treedef;
that is correct and unchanged (`switch_elimination_design.md` §12.7's caveat, carried
forward so it keeps surprising nobody).

## 8. Migration plan — steps, owners, gates

Baseline to record before anything moves: `pytest functional_process -q` (3724 passed —
the 3704 this document first recorded was stale; re-measured at `aca0fb6d`),
`cd ~/jaxgraph && pytest` (482 passed), `run_mda_harness.py` (499 / 34 / 3 / 0
ungrounded; 557 walked, 0 unaccounted; 61 switch kwargs, 0 mismatched), `GRAPH` 159
nodes / 138 blocks / 14 driven. The disk-cached harness (17 s) runs per step.

**Step 1 — enum-typed settings + enum-aware `switch_audit`.** *(functional_process;
independent, safe now)* Convert the ~60 static slots from `int` to their upstream
`IntEnum`s (delete the `is_ignited` bool alias, `switch_elimination_design.md` §3(d),
and reclassify §3's (b)/(c) shape/set kwargs as explicitly-not-switches while there).
Extend `switch_audit` to resolve and report member names.
*Gates:* all baseline numbers **identical** (`IntEnum == int`); the audit's
61 / 0 / 3 / 0 line unchanged; run it once against a bare `DataStructure()` and confirm
it still reads as a diff against PROCESS defaults.

**Step 2 — cottax: `ModelNamespace`.** *(~/jaxgraph)*
§3.1's walker branch, minting `GetAttrKey` slot names, with its own tests (order
preservation, `None` skip, non-entry refusal, unplaced-instance-at-root, `named(at)`
moving minted names under a slot, `^problem.<GetAttrKey path>` intact). No rendering
change (§3.3).
*Gates:* `~/jaxgraph` suite green including new tests; `pytest functional_process`
byte-identical in outcome (the change is purely additive upstream).

**Step 3 — convert the node tree: nested classes → `Machine` Module tree. [DONE
2026-08-25.]**
*(functional_process; after Part A lands, per the sequencing rule)* Rewrite
`path_refactor.md` §B.3's `COMMON` class tree as `ModelNamespace`s with snake_case
slots; `to_graph(machine)` directly, `{ROOT: ...}` deleted. **Node names change once,
here** — re-pin every test and record that holds a node name, in this step and not
dribbled across steps.
*Gates:* 159 nodes, 138 blocks / 14 driven, MDA harness 499 / 34 and 557 / 0
**identical** (values do not move when names do); `EXCLUDED_NODE_NAMES` still matches;
grep for stale spelled node names in `_audit/*.md` and fix the citations.

*A second silent breakage, found only by looking at the picture.* The gates above are
about the graph, and they all held while the **coloured DSM was destroyed**:
`visualization/grouping.py`'s `_tree_keys` read a group off the leading `DictKey`s of a
node name, on the rule that a `GetAttrKey` means a variable place. Step 3 inverted that
premise by design (§3.1: slots mint `GetAttrKey`s so a name is a working address), so
every machine node lost its prefix, `grouped` reported **`1 group(s) at depth 1`** where
there are six, exited 0, and wrote both files. Nothing in the suite covers what the
picture looks like. Fixed by counting both key kinds -- in a `NodePath` both *are*
namespace positions -- and asking the question the kind was standing in for directly:
`group_of(..., among=graph.nodes)`, since a name minted over a node unmints to a node
and one minted over a variable does not. Restored to 7 groups (six subsystems plus
`UNGROUPED`), 57 cross-group edges, 1/7 contiguous in run order. **The lesson generalises
past this step: a gate on numbers cannot see a renderer, and every proxy-by-key-kind in
the port is now suspect for the same reason.**

*As run:* all identical, **measured before and after with one script rather than against
the recorded figures** — 159 nodes, **139** blocks / 14 driven, 348 unowned inputs, and a
sha of every node's (inputs, outputs, type) unchanged. The 139/348 differ from the 138/349
recorded above and in `next_steps.md`'s table; the conversion did not move them (before ==
after), so the recorded pair was stale and is corrected there. `EXCLUDED_NODE_NAMES` did
**not** still match and had to be re-pinned — `path_refactor.md` §B.5 predicted it would
survive, and why that prediction failed is recorded there. `test_configuration.py`'s
`divertor_cycle` was the one test holding a node name; the two were the whole surface, as
that section's *count* correctly said.

**Step 4 — switches → slots + `machine_from_indat`; delete the old mechanism. [DONE
2026-08-25.]**
*(functional_process; the flag-day step, one sitting)* Move each of the 10 switches'
arms into slot occupants (§4), write the factory and registries (§5) with the
`unported` reasons carried verbatim, define `Machine()` defaults = PROCESS defaults,
delete `configuration.py`'s `Switch`/`Alternative`/`Configuration` and
`TOPOLOGY_SWITCHES`. Re-target `test_configuration.py`: exclusivity tests become
by-construction (delete) plus boundary-postcondition tests (step 5 can land its check
early here if convenient); the defaults test re-targets per §5; per-arm assembly tests
become per-occupant `to_graph(eqx.tree_at(...))` tests.
*Gates:* `to_graph(machine_from_indat(REFERENCE_INPUT_FILE))` has the same node
*definitions* set as the step-3 reference graph; MDA harness identical; SAND C2
(42 it, `objf 1.2177574`) and C3 land on the same point as baseline; `switch_audit`'s
line unchanged. Every one of these must be **identical**, not merely green — this step
is a re-spelling, and any numeric movement is a bug in the conversion.

*As run:* all met, and **compared against `HEAD` rather than against the numbers recorded
here**, which is what the gate should have said: the recorded SAND C2 (42 it) and C3 (100
it, stalling) were stale by several intervening changes, and measuring against them would
have reported a failure that was not one. Run side by side, before and after are
byte-identical — C2 31 it / `objf 1.217757347`, C3 99 it / `1.217757378`. Also identical:
159 nodes, 139 blocks / 14 driven, 348 unowned inputs, the per-node (inputs, outputs,
type) sha, and the MDA harness's 499/34/3/0 · 557/0 · 61/0/3/0.

*What did move, correctly:* the **grouped DSM**. `costs` is a real subsystem of 43 nodes
now rather than 43 root-level singletons, so the picture goes from 7 groups / 57
cross-group edges / 1-of-7 contiguous to **8 / 144 / 0-of-8**. That is §11.4's
provenance-against-structure measurement finally including the cost model, not a
regression — and it says the run order interleaves every subsystem, none contiguous.

**Step 4b — the tree stops carrying configuration; `Machine` → `StellaratorProcess`.
[DONE 2026-08-25.]**
*(functional_process; one sitting, no numeric movement expected and none found)* Step 4
left the configuration in two places: the factory *and* the slot defaults. The defaults
were the worse of the two, because nothing checked them —
`test_machine_defaults_are_process_defaults` compared occupant *classes* only, so the
reference run's `i_confinement_time = 38` / `i_plasma_ignited = 1` sat in the occupant's
constructor kwargs where no test could see that PROCESS's own defaults are `34` / `0`.
**The contract §2 and §5 above spell as "`Machine()`'s field defaults *are* PROCESS's
bare defaults" was therefore never true, and is retired rather than repaired**: a slot
`machine_from_indat` fills gets `dataclasses.field(kw_only=True)` and no default, every
other slot keeps its default, and a default is admissible only where there is nothing to
decide. `StellaratorProcess()` raises `TypeError`. `kw_only` works on `eqx.Module`, so
no sub-namespace had to be reordered. Eleven switched slots and the five spine
namespaces above them lost their defaults; `vacuum` is the one sub-namespace that keeps
one, because nothing inside it is switched.

*Four things went with it.* **(a) The class is `StellaratorProcess`** — the tree is this
project's configuration for one device, not a general machine abstraction. The
sub-namespace `Stellarator` keeps its name, and *machine* stays the noun for an
**instance** (`machine_from_indat`, `REFERENCE_MACHINE`, `graph_for(machine=)`), which
the class docstring says out loud so the mixed vocabulary reads as chosen rather than
drifted. **(b) The `istell == 0` arm is deleted**: `CONFINEMENT_TIME` is
`{6: StellaratorConfinementTime}` and `("istell", 0)` is an `UNPORTED` entry. There is no
`Tokamak` namespace mirroring `Stellarator`, so that arm assembled stellarator geometry,
stellarator coils and stellarator FWBS under a *tokamak* confinement scaling — a device
nobody has built and this port has never tested. It is the "assembling anyway hands you a
graph that looks complete and is wrong" kind, hence refused and not merely absent.
*Consequence, stated rather than papered over:* `istell` has no usable default, so
`machine_from_indat` on a file that does not set `istell = 6` raises. The factory now
resolves `istell`'s two consequences (machine config, confinement binding) into named
locals **before** the constructor call, so a silent IN.DAT is refused for `istell` rather
than for whichever slot Python happened to evaluate first. **(c) All four `| None`s are
gone.** Two stood for live configurations and are `UNPORTED` entries now (`istell == 0`;
`i_cost_model == 1`, KOVARI_2014, PROCESS's own default — `costs_2015.py` has no cottax
nodes, so that arm computes no cost of electricity at all and `.costs.coe` /
`.costs.concost` would surface as unowned boundary inputs). The other two were **dead,
and were checked to be dead before removal**: `BLANKET_MASSES` is `{2: …}` and
`BLANKET_SHIELD_POWER` is `{1: …, 2: …}`, and every other joint key `machine_from_indat`
can derive is in `UNPORTED` (`0`, plus `1` for the mass slot) and raises — `None` was
unreachable on both. `UNPORTED`'s docstring loses its "refusal, not absence" distinction
accordingly: absence has no spelling left in the tree. **(Corrected by step 4d, which
puts one `| None` back**: `costs.cost_of_electricity` at `ireactor != 1 or ipnet != 0`,
the one configuration in the survey where PROCESS itself computes nothing. The
distinction is back in use; what stays true is the reason each of *these four* went.) **(d) `COMMON` is deleted** —
`COMMON = Machine()` was vestigial, reached by nothing but its own docstring and prose.

*Gates, all met.* `GRAPH` **159 nodes, unchanged**; `python -m
functional_process.render_xdsm` exits 0; `pytest functional_process` **3732 → 3725
passed**, 3347 skipped on both sides, and the `--collect-only` id diff is exactly the
seven `test_machine.py` ids this step removes or adds — no other file's collection moved
by one test. `ruff check` on the two files goes 11 → 7 findings, all seven pre-existing
`E501`s on lines this change did not touch.

*Tests re-targeted.* `test_machine_defaults_are_process_defaults` (8 params) is deleted
and **replaced** by `test_a_silent_indat_is_refused_naming_istell` — the same question
put to the only thing that still answers it. `test_an_absent_occupant_assembles_as_
nothing` becomes `test_the_default_cost_model_is_refused_with_its_reason`.
`test_the_1990_cost_model_is_the_only_producer_of_coe` is kept and reworded: dropping the
occupant is a structural what-if reachable only through `eqx.tree_at`, not "PROCESS's
default", which is not a configuration this tree admits any more. Because two of
PROCESS's own defaults are now refused, every temp IN.DAT in the refusal tests is written
over a `BASELINE_INDAT` of `{istell: 6, i_cost_model: 0}` — without it each case would
fail on whichever of those two the constructor reached first instead of on the value
under test. `SLOTS` loses its "PROCESS's own default" column, which had no reader left.

*Follow-ups, deliberately not done here.* **(i)** Seven comments under `models/**` still
say `COMMON` (`plasma_profiles.py` ×4, `profiles.py`, `radiation_power.py`,
`stellarator_fwbs_s4.py`). The nine inside `total_process.py` were reworded because this
change was already editing those lines; a sweeping reword of the rest would be
unreviewable next to this diff, which is the same call §8's vocabulary note already made
for `Switch`/`Alternative`/`Configuration`. **(ii) 30 slots still pass a switch as a
static constructor kwarg** — measured, not estimated: 32 slots carry constructor kwargs
at all, less `profile_grid` and `impurity_radiation_totals`, whose kwargs are the
shape/membership counts §3(b)/(c) exempts. That is step 6's job, family by family; this
step moved only the switches that were already slots.

**Step 4c — the joint blanket keys stop lying, and three cost nodes for a subsystem
this device does not have are deleted. [DONE 2026-08-25.]**
*(functional_process; one sitting, `total_process.py` and `test_switch_coverage.py` only)*
Two corrections, neither of them a re-spelling. Both were found by audits this design
commissioned — `switch_kwarg_survey.md` §4.3 and `cost_boundary_inputs.md` §§4–6 — and
neither could have been found by any gate this file names, which is the general lesson
worth keeping: **every gate in §8 is a same-configuration identity check, and both of
these defects were invisible on the reference configuration.**

*(a) The joint keys.* `machine_from_indat` derived the two joint registry keys as

```python
blktmodel = switches.get("blktmodel", 2)                 # PROCESS's default is 0
... blktmodel if switches.get("ipowerflow", 1) else 0    # blktmodel_ipowerflow
... blktmodel if switches.get("blkttype", 3) >= 3 else 1 # blktmodel_blkttype
```

i.e. it passed a switch *value* where an **arm index** was wanted, and made that fit by
defaulting `blktmodel` to `2` — **not a legal value of `blktmodel` at all**
(`fwbs_variables.py:479`: `{0, 1}`, default `0`). That is a sentinel meaning "the file
did not say", chosen so the reference run landed on the right arm, and it inverted the
mapping: stating PROCESS's own default `blktmodel = 0` explicitly was **refused**, while
leaving it unstated worked — the exact opposite of `switches_from_indat`'s documented
contract that a name the file never mentions falls through to the default.

The true branch structure, read off `st_fwbs` rather than inferred:

| | `stellarator.py` | arm | occupant |
|---|---|---:|---|
| `blktmodel == 1` | `:608`, `blanket_neutronics()` | 0 | none — `UNPORTED` |
| `blktmodel == 0 & ipowerflow == 0` | `:683-729`, exponential attenuation + ITER-90 SC heating | 1 | `BlanketShieldPowerExponential` |
| `blktmodel == 0 & ipowerflow == 1` | `:730-…`, detailed powerflow | 2 | `DetailedPowerflowBlanketShieldPower` |
| `blktmodel == 1` | `:1093-1181`, sub-assembly thicknesses | 0 | none — `UNPORTED` |
| `blktmodel == 0 & blkttype ∈ {1,2}` | `:1058-1066`, liquid breeder | 1 | none — `UNPORTED` |
| `blktmodel == 0 & blkttype ∉ {1,2}` | `:1067-1075`, solid breeder | 2 | `BlanketComponentMasses` |

`blktmodel` is the **outer** test in both dispatches and the second switch only separates
the two arms *inside* `blktmodel == 0`. **So the mapping was inverted in exactly the
sense that `blktmodel`'s two legal values reached each other's arms**, and one further
arm (`ipowerflow == 0`, a *ported* occupant) was unreachable at every input. Measured,
before and after:

| override | before | after |
|---|---|---|
| reference (nothing set) | arm 2 + arm 2 | **unchanged** |
| `blktmodel = 0` (PROCESS's own default) | `NotImplementedError` | arm 2 + arm 2 — the reference machine |
| `blktmodel = 1` | assembles `BlanketShieldPowerExponential`, then refuses at the mass slot citing the *liquid-breeder* reason | refuses citing `blanket_neutronics()`, which is `blktmodel == 1`'s actual reason |
| `ipowerflow = 0` | `NotImplementedError` | `BlanketShieldPowerExponential` — reachable at last |
| `blkttype ∈ {1,2}` | refuses, liquid-breeder reason | unchanged |

The `blktmodel = 1` row is the serious one and is the `ScTfCoilNuclearHeating` bug class
this document already records as found-and-fixed once: a configuration silently
assembling a node written for a *different* switch's arm. It was reintroduced by a key
derivation rather than by a registration, which is why no registration review caught it.

The fix keeps the arm-index keys — `unit_registry.md`'s rows and every `UNPORTED` reason
already describe arms, and re-keying to pairs would have meant enumerating `blkttype`'s
whole domain — but derives them from **legal switch values only**, in two named
functions, `_blanket_shield_power_arm(blktmodel, ipowerflow)` and
`_blanket_mass_arm(blktmodel, blkttype)`, whose docstrings are the transcription above.
`blktmodel` defaults to PROCESS's `0`. Both are resolved into named locals before the
`StellaratorProcess(...)` call, for the same reason `istell` is: so a refused
configuration reports the arm the caller asked for and not whichever slot Python reached
first. `BlanketShieldPowerExponential`'s own docstring said "the `blktmodel == 1`
occupant" and was simply wrong; it is corrected in place.

`("blktmodel_blkttype", 0)` is now unreachable through the factory — `blktmodel == 1`
selects arm 0 of both dispatches and the shield-power slot resolves first — and is
**kept**, because it remains the correct record of what `stellarator.py:1093-1181` does
and is what an occupant of that arm has to answer. That is said at `UNPORTED`.

*(b) Three cost slots deleted.* `cost_boundary_inputs.md` measured four cost nodes whose
every output is exactly `0.0` in PROCESS's own converged reference run. Three of them —
`.costs.pf_magnet_cost` (222.2), `.costs.pf_coil_power_conditioning_cost` (225.2),
`.costs.reactor_structure_cost` (221.4) — are zero because a **stellarator has no such
subsystem**, not because a switch is off: `caller.py:272-275` returns before
`pfcoil.run()` and `Power.pfpwr`, and `st_strc` (`stellarator.py:334-337`) sets both
structure masses to a literal `0.0` with its own reason ("to avoid double-counting …
specified differently for tokamaks"). With `n_cs_pf_coils = 0` both of `acc2222`'s loops
unrolled to zero iterations, leaving **21 of `PfMagnetCost`'s 27 declared reads dead**,
not merely multiplied by zero. A node whose ports assert a dependence on a subsystem the
configuration does not have is the `EcrhDensityLimit` bug class, and the reason
`WardTaylorAvailability` is deliberately unregistered; these three landed on the right
number by luck, which is why nothing caught them. **`.costs.energy_storage_cost` is
deliberately kept**: it is zero because of `i_pulsed_plant`, a switch, and a pulsed
stellarator would want it — that one belongs to step 6.

Deleted outright, not made a switched slot: there is no tokamak occupant to switch
*against*, and a variant mechanism for a family with one member and no alternative is
the paradigm-for-nothing this project declines elsewhere. **`Costs` splits when the
tokamak arrives**; `cost_boundary_inputs.md` §12 records exactly what has to come back
and from which producer.

*Gates, as run.* Both changes measured against a `git archive HEAD` copy of the tree with
`cottax` pinned to its own `HEAD` (a concurrent refactor in `~/jaxgraph` was mid-flight
and briefly unimportable; pinning both sides is what makes the comparison a comparison).

- **(a) alone moved nothing**: 159 nodes, 348 unowned inputs, 14 cycles and the per-node
  (inputs, outputs, type) sha all identical to `HEAD` — which is the correct result for a
  key derivation that was already landing on the right arm for *this* file.
- **(b)**: **159 → 156 nodes**, exactly the three removed. Blocks 139 → 136, SAND graph
  172 → 169 nodes / 43 → 40 blocks, grouped DSM **144 cross-group edges, unchanged** —
  the three were leaves in their own subsystem. (The grouped DSM reports **7** groups on
  both sides, not the 8 step 4 recorded; measured before and after with one binary, so
  the 8 is stale, not something this step moved.)
- **MDA harness, the safety argument, quoted rather than summarised**: `499 / 34 / 3 / 0
  · 557 walked, 0 unaccounted · 61 kwargs, 0 mismatched` becomes `485 / 34 / 3 / 0 · 543
  walked, 0 unaccounted · 57 kwargs, 0 mismatched`. Every one of the 14 lost agreements
  came out of `both-sides-exactly-zero` (73 → 59) — i.e. the port lost 14 *vacuous*
  agreements and no real one — and **the `all disagreements:` block is byte-identical**,
  `.costs.c22`/`.c2`/`.cdirt`/`.concost`/`.coe` included. The 4 kwargs are
  `PfMagnetCost`'s four statics. Nothing else moved.
- **Boundary**: `.costs.c2214`, `.costs.c2222` and `.costs.c2252` become unowned inputs
  and are seeded from their `cost_variables.py` defaults of `0.0`, which is the value the
  nodes produced. Their fourteen sub-accounts (`c22221`-`c22224`, `c22521`-`c22527`) had
  no reader in the graph and simply leave it. **The boundary does not grow, it shrinks,
  348 → 320**: the three added are outnumbered by 31 removed, and *what* is removed is
  the point — every `.pf_coil.*` and `.pf_power.*` read, plus `.structure.fncmass` and
  `.structure.gsmass`. Those were precisely `boundary_inputs_audit.md`'s category (d),
  "correct for this configuration and wrong for any other". Deleting the nodes deletes
  the category. §6's `check_boundary` is step 5's and does not exist yet, so there was no
  pin to update; **step 5's pin should be generated after this step, not before.**
- `pytest tests/functional_process -q` → **3752 passed, 3347 skipped**, unchanged; the
  three render entry points exit 0.

*Tests re-targeted.* `test_switch_coverage.py`'s `_CAUSES_A_REFUSAL` had three rows
asserting the pre-fix behaviour, and two of them were wrong about it. `("ipowerflow", 0,
…)` **moves to `_CHANGES_A_SLOT`** — its arm has a ported occupant, so a refusal was
never the right assertion — probing the joint slot rather than `fw_area`, which would
have passed either way. `("blktmodel", 1, …)` stays a refusal but its key changes from
`("blktmodel_blkttype", 1)` to `("blktmodel_ipowerflow", 0)`: the old row passed because
the *mass* slot refused, which happened only after the shield-power slot had already
chosen the wrong node. `("blkttype", 1, …)` is unchanged and was always right. Net
collection is unchanged, five and five becoming six and four.

*A note on what this step says about §8's gates.* Every step above gates on "identical",
which is right for a re-spelling and blind to a defect that is correct on the reference
configuration and wrong everywhere else. Both of 4c's defects are of that shape, and both
were found by reading PROCESS's own source against the tree — `switch_kwarg_survey.md`'s
method (4) and `cost_boundary_inputs.md`'s check 2 — not by any number.
`switch_kwarg_survey.md` §7's proposed "no declared read may be dead at the value the
slot holds" test is the one gate that would have caught (b) mechanically; there is still
nothing that would catch (a) but reading the branch.

**Step 4d — a switch is answered once. [DONE 2026-08-25.]**
*(functional_process; one sitting, `total_process.py`, `test_machine.py` and
`test_switch_coverage.py` only)* `switch_kwarg_survey.md` §5 band (a) is five **live
incoherences** — cases where the tree holds two different answers to one question, as
opposed to the merely untidy 32-slot kwarg surface the rest of that survey measures. Step
4c closed the first. This closes the other four, and they share one shape: *a switch that
decides which node exists was also written into a constructor kwarg somewhere else, and
the two could disagree.* Measured, before, by building the machine and reading the tree:

| override | assembled | and also held |
|---|---|---|
| `i_tf_sup = 0` | `TfPowerResistive` at `power.tf_power` | `SUPERCONDUCTING` at **five** other slots |
| `ipowerflow = 0` | `AFwTotalNoPowerflow` + `BlanketShieldPowerExponential` | `COMPREHENSIVE_2014` at the two `stellarator.*_wall_load*` slots |
| `ireactor = 0` | `PowerProfilesOverTime` (which computes no `.heat_transport.p_plant_electric_net_mw`) | a `CostOfElectricity` reading it, carrying `ireactor = 1` |
| `i_plasma_pedestal = 1` | `ProfileParameterisationPedestal` | — a machine PROCESS cannot produce at all |

*(a) `i_tf_sup` and `ipowerflow` are threaded, not split.* Both are resolved into a named
local in `machine_from_indat` and passed to the occupants that branch on them; the five
`i_tf_sup` kwargs (`power.cryo_q_nuc_step`, `cryo_q_loads_step`, `cryo_loads`,
`availability.cplife_avail`, and the one inside `ELECTRIC_PRODUCTION`'s `functools.
partial`, which becomes a pair of named builder functions taking the switch) and the two
`ipowerflow` kwargs are gone. **Threading is the right size of fix here and splitting is
not**: a split gives each arm its own reads, which is worth having and is
`switch_kwarg_survey.md` band (b)'s whole subject — seven of these eight (slot, switch)
pairs are on band (b)'s list and stay there. Band (a)'s defect is narrower and cheaper:
two places answering one question. `availability.cplife_avail` is the case where
splitting would have been actively wrong today, and the survey's §4.8 says why —
`CpLifetime{Superconducting,Resistive}` return the *fresh* centrepost lifetime while
`CplifeAvail.step` returns the availability-adjusted one, so dropping either in as an
occupant silently loses the adjustment; and the switch that actually matters at that slot
is `itart`, not `i_tf_sup`. The slot is resolved **before** the local is threaded, so
`i_tf_sup == 2`'s `UNPORTED` refusal still fires before any occupant can be handed an
unported value.

*(b) `costs.cost_of_electricity` becomes a slot, and its other arm is `None`.* PROCESS
calls `coelc()` only when `ireactor == 1 and ipnet == 0` (`costs.py:82-83`); on any other
pair it leaves `.costs.coe` and its five companions at whatever they held, and
`CostOfElectricity.__check_init__` already said so — *"this node must not exist"*. It
existed anyway. It is keyed now on the arm index `_cost_of_electricity_arm(ireactor,
ipnet)` returns, the same discipline `_blanket_shield_power_arm`/`_blanket_mass_arm`
follow, and taking `ipnet` in was not optional: it is the other half of one precondition,
so an `ireactor`-only fix would have left the identical defect reachable by one line of
IN.DAT.

**This puts a `| None` back, and step 4b's own reasoning is the argument for it.**
`switch_kwarg_survey.md` §4.2 called this conversion "blocked on the tree regaining a way
to say 'no occupant', which step 4b deliberately closed off". That is wrong, and it is
worth saying why, because the mistake is easy to repeat: cottax has always spelled
absence — `ModelNamespace`'s docstring shows it, `node_and_names` has the branch
(*"an unproduced slot: it assembles nothing, and whatever read its outputs surfaces as a
boundary input. Absence, spelled as absence."*), and
`test_the_1990_cost_model_is_the_only_producer_of_coe` was already building a
`costs = None` tree. What step 4b removed was **four particular `| None`s**, and its own
account of why is the test this one has to pass: two were unreachable, two stood for
configurations *this port* cannot honestly assemble, and **none of the four was a case
where PROCESS itself computes nothing**. This is that case, and it is the only one in the
survey. `UNPORTED`'s docstring loses nothing: refusal and absence are different, they
always were, and the distinction is back in use rather than merely described.

The alternative the survey recommended — refuse `ireactor = 0` outright — was rejected
because it would have made `PowerProfilesOverTime`, a ported and registered occupant,
unreachable through the factory. That is exactly the defect step 4c had just finished
removing from `BlanketShieldPowerExponential`, and trading one for the other is not a fix.

*(c) `i_plasma_pedestal` stops being read from the IN.DAT.* `st_init`
(`process/models/stellarator/initialization.py:31`) assigns
`data.physics.i_plasma_pedestal = 0` unconditionally on every `istell != 0` run, in the
same block that zeroes the central solenoid. So the file's value is **dead on this
device**, and the factory was assembling the pedestal arm for runs PROCESS executes with
parabolic profiles. `PROFILE_PARAMETERISATION` is resolved from a named constant,
`ST_INIT_I_PLASMA_PEDESTAL = 0`, whose docstring is the citation.

The other option was to refuse a file that sets it to something PROCESS will overwrite.
Rejected on fidelity: PROCESS runs such a file happily, and a port that declines an input
PROCESS accepts is modelling something other than the run. **Silently ignoring a
user's value is the cost, and it is paid visibly** — in that constant's docstring, in a
new `ForcedByProcess` category in `test_switch_coverage.SWITCH_INVENTORY`, and in
`test_a_process_forced_switch_cannot_move_the_machine`, which asserts that overriding
`i_plasma_pedestal` leaves the *whole* assembled machine identical.
`ProfileParameterisationPedestal` stays registered and unreachable through the factory,
for the reason `("blktmodel_blkttype", 0)` stays in `UNPORTED`: it is the correct record
of a real PROCESS branch, and `eqx.tree_at` reaches it the way every structural what-if
does. `switch_kwarg_survey.md` §7 records `iohcl` as the same shape, forced by the same
`st_init` block, and it has no slot at all — so nothing about it changes here.

*Measured, after.* `i_tf_sup = 0` → all five sites read `0`. `ipowerflow = 0` → both
wall-load nodes read `0`, 157 nodes. `ireactor = 0` (or `ipnet = 1`) → **155 nodes**, the
six `.costs.coe`-chain fields leaving the graph. `i_plasma_pedestal = 1` → the reference
machine, unchanged, `repr`-identical.

*Gates, all met, and every one of them is a same-configuration identity check by design —
this step removes duplicate answers that all currently agree, so nothing about the
reference run may move.*

- `GRAPH` **156 nodes, 136 blocks / 14 driven, 320 unowned inputs**, all unchanged, and
  the per-node (name, type, sorted inputs, sorted outputs) sha256 is **identical**
  (`ae3c93c7…`), measured on both sides with one binary.
- **MDA harness identical**: `485 / 34 / 3 / 0 · 543 walked, 0 unaccounted · 57 kwargs,
  0 mismatched`, and the `all disagreements:` block **byte-identical** by `diff`.
- `pytest tests/functional_process` **3755 → 3770 passed**, 3347 skipped on both sides.
  The 15 are all this step's: 12 in `test_switch_coverage.py` (10 coherence cases, its
  completeness check, the forced-switch test; net of one `_CHANGES_A_SLOT` row leaving
  and one joining) and 3 in `test_machine.py` (the new `ireactor_ipnet` slot, through
  `test_every_registered_occupant_assembles` ×2 and `test_occupants_of_one_slot_differ`).
- `render_xdsm` and `render_xdsm grouped` exit 0; grouped still reports 7 groups, 136
  blocks, **144 cross-group edges**.
- `ruff check` on the three files: **7 findings before, 7 after**, the same seven
  pre-existing `E501`s; `ruff format --check`'s one pre-existing diff is unchanged.

*A measurement caveat, recorded because it cost an hour.* `~/jaxgraph` was being edited
by a concurrent session mid-step, and a mid-flight `cottax` makes `run_mda_harness` raise
`TypeError: condition map of ['.physics.profiles.ion_vol_avg_temperature'] takes 1
unknown(s) …, got 0` — **on a clean `PROCESS` tree as well as on this one**, which is how
it was identified. Every number above is measured with `cottax` pinned to its own `HEAD`
(`git archive` into a scratch dir, `PYTHONPATH` ahead of the editable install), both
sides, the same discipline step 4c used and for the same reason.

*Tests re-targeted, and one added that is not a re-target.*
`test_switch_coverage.py`'s `_CHANGES_A_SLOT` loses `i_plasma_pedestal` — it is not a
`FACTORY_READ_SWITCHES` member any more — and gains `ipnet`, whose probe is the one place
in that list where the expected occupant type is `NoneType`. `SWITCH_INVENTORY` gains a
fourth category, `ForcedByProcess`, for the switch the file sets and the factory
deliberately does not read. `test_machine.py`'s `SLOTS` gains the `ireactor_ipnet` row
and two build helpers, `_electric_production` (the registry's entries are builders taking
`.tfcoil.i_tf_sup` now) and `_costs` (`Costs` has a slot of its own and can no longer be
default-constructed).

**The addition is `test_no_slot_contradicts_a_factory_switch`**, and it is the gate this
step's whole class of defect was missing. For every switch that decides a slot *and*
appears as a static field on some occupant, it assembles a machine at every value that
assembles and asserts that **every** field of that name in the whole tree holds the value
the factory resolved — one walk per case, so a site added later is covered the day it is
added rather than the day someone remembers to list it. Its companion,
`test_every_factory_switch_with_a_static_field_is_covered`, intersects
`FACTORY_READ_SWITCHES` with the field names actually present in `REFERENCE_MACHINE` and
requires the case table to match, so a newly-hardcoded switch cannot escape by not being
listed; it doubles as the positive control for the walker. All four of this step's
defects fail it, and — the point — **none of them could be seen by any gate in §8**,
every one of which is an identity check on the reference configuration. That is the same
lesson step 4c recorded, now with a mechanical answer for one of its two halves.

**Step 5 — the boundary postcondition.** *(functional_process; small)* §6's
`check_boundary` + the pinned reference boundary set, generated from
`boundary_inputs_audit.md`'s accounting rather than typed. Wire it into graph assembly
tests and into the swap tests from step 4.
*Gates:* reference machine passes; a deliberate orphan-inducing swap in a test fails
naming the orphan.

**Step 6 — convert the dial-riding model families to occupant families.**
*(functional_process; after step 4, and incremental — one family per pass, no flag
day)* The six recorded split-default deviations (`i_confinement_time`, `i_rad_loss`,
`i_plasma_ignited`, `supercond_cost_model`, `i_pf_conductor`, `itart`) plus any
formula family §4's refined criterion catches (`i_beta_fast_alpha`, ...), each
converted to case-1/case-2 occupants (§4): shared bodies via the subclass pattern,
in-scope members as classes, out-of-scope members as registry absence with the reason.
*Gates, per family:* every baseline number identical; `check_boundary` passes;
`switch_audit`'s counted totals change by exactly the converted entries (a static
kwarg that becomes an occupant leaves the kwarg audit and enters the tree/registry
comparison — record the per-family delta, never absorb it silently).

**Step 7 (optional, unscheduled) — the derived serialisation view** (§7).

Steps 1 and 2 are independent of each other and of Part A, and can run in parallel.
Steps 3–5 are sequential and sit after Part A; step 6 follows step 4 and can proceed
family by family whenever convenient — later is a scheduling fact, not a design
posture. Per the standing dispatch conventions
(`next_steps.md` §4b): an agent touches only its own step's files and runs its own
tests plus the gates named for its step; registry/`next_steps` bookkeeping is the
consolidation pass's job.

**A vocabulary note, since the old names outlived their code.** `Switch`, `Alternative`,
`Configuration`, `check_arms_are_exclusive` and `TOPOLOGY_SWITCHES` are gone, but they are
still named in per-unit `.md` records and a few model docstrings, where they describe
reasoning that remains true. The translation: an *arm* is an **occupant**, a *switch* is a
**slot**, `Alternative(unported=)` is an entry in **`UNPORTED`**, `Alternative(unproduced=)`
is a slot holding **`None`**, and `Configuration` is a **`Machine`**. Those records were
deliberately not swept: the concepts moved, the prose did not, and rewriting twenty audit
files to change a noun would cost more than it clarifies.

## 9. What this does not solve, stated so nobody reads it as solved

- **`free` and design-variable meaning** (`next_steps.md` §12.1/§12.3) — untouched.
  This design gives a selection a spelling; `free` gives a design vector a meaning.
- **Whether PROCESS's remaining coupling is cyclic** — unaffected; the tree names
  models, `Blocking` still derives structure.
- **The `slot: OccupantClass` renderer label** — deferred until a drawing hurts (§3.2).
- **Fused-node splitting** (`ComponentThermalPowers`, ~5 models in one node,
  `switch_elimination_design.md` §10.3) — still real, still gated on the per-branch
  reads/writes analysis that section says is missing. The tree gives each fragment a
  slot to land in; it does not do the split.

## 10. Filenames follow the slot names — the audit chunk letters are gone

**Done 2026-08-25.** Ten modules still carried the audit's chunking in their filenames
(`physics_A_pure_formulas.py`, `power_B_thermal_cryo.py`,
`stellarator_F_tf_nuclear_heating.py`, ...). §3.2 already ruled that a node's identity is
the *place* it occupies, not the class that fills it, and `total_process.py`'s
`StellaratorProcess` docstring had already applied that to slot names, saying in as many
words that the chunk letters "are how the port was chunked for auditing, not what the
machine is made of". The filesystem was the last place still showing the scaffolding. It
now shows the machine.

**The rule, stated so it is checkable and not relitigated per file** — the same standard
§3.2 sets for slot names:

> **The stem names what is in the file; the directory names the subsystem.** Drop the
> audit chunk letter and the subsystem prefix, because the directory already carries the
> subsystem. Where no directory carries it, create one rather than folding the subsystem
> into the stem. Where the residue is PROCESS's abbreviated source-routine name rather
> than a description of the content, use the content name — which is what this
> directory's existing modules already do (`st_build` → `build.py`, `st_div` →
> `divertor.py`, `st_init` → `initialization.py`).

| old | new |
|---|---|
| `models/physics/physics_A_pure_formulas.py` | `models/physics/pure_formulas.py` |
| `models/physics/physics_B_composition.py` | `models/physics/composition.py` |
| `models/physics/physics_C_outplas.py` | `models/physics/dimensionless_parameters.py` |
| `models/power_A_tf_coil_power.py` | `models/power/tf_coil_power.py` |
| `models/power_B_thermal_cryo.py` | `models/power/thermal_cryo.py` |
| `models/power_C_electric_production.py` | `models/power/electric_production.py` |
| `models/stellarator/stellarator_B_st_phys.py` | `models/stellarator/plasma_physics.py` |
| `models/stellarator/stellarator_C_geometry.py` | `models/stellarator/geometry.py` |
| `models/stellarator/stellarator_D_structure.py` | `models/stellarator/structure.py` |
| `models/stellarator/stellarator_F_tf_nuclear_heating.py` | `models/stellarator/tf_nuclear_heating.py` |

Each rename is three files — the module, its record under `_audit/units/**`, and its case
under `tests/functional_process/**` — moved together with `git mv` so the three mirrors
stay in step and history follows.

**The three `power_*` were the only real decision.** They sat at `models/` top level, so
stripping alone would have lost the subsystem: `models/thermal_cryo.py` says less than
`power_B_thermal_cryo.py` did. `models/` already namespaces by subsystem (`physics/`,
`stellarator/`, `costs/`, `blankets/`), so the rule's second clause applies and
`models/power/` was created — one package per PROCESS source model, which is what the
other four are. `tf_coil_power.py` keeps `power` in the stem even inside `power/`,
because that is its subject (TF coil *power conversion*, not a TF coil model); the same
mild repetition already stands in `costs/costs.py`.

**Two names are content names, not the mechanical strip, per the rule's third clause.**
`physics_C_outplas` → `dimensionless_parameters.py`: `outplas` is PROCESS's *reporting*
routine, and the record's own finding is that its 1095 lines contain one computation and
no output at all, so the stripped residue would have named the file after the one thing
that is not in it. `stellarator_B_st_phys` → `plasma_physics.py`: `st_phys` is an
abbreviation, and this directory already expands them; `physics.py` would have been the
literal precedent but makes a bare "physics.py" reference ambiguous against
`process/models/physics/physics.py`, which the records cite constantly.

**One name is weaker than the rest, and it is worth saying which.** `pure_formulas.py`
describes provenance (five functions that were already pure `@staticmethod`s in PROCESS),
not a subject — and "pure" is true of every module in this package. It survives because
the chunk genuinely has no single subject: ion/electron equilibration, burnup and
fuelling rate, heating power, stored thermal energy, and fast-alpha beta. A subject name
here would be a worse lie than a provenance one.

**The four `stellarator_fwbs_s*.py` were deliberately held back.** `next_steps.md` §3 is
still open on `st_fwbs`: S2 (`blanket_shield_tf_nuclear_power`) and S3
(`divertor_mass_and_first_call_seed`) are unported, and the S1–S6 re-chunking that would
decide what the surviving files are called is live. Renaming them now would rename them
twice; renaming late is cheaper than that. `stellarator_fwbs_s1_s5.py` is therefore the
one place the audit's chunking is still legible in the tree, which
`total_process.py`'s docstring and `path_refactor.md` both now say explicitly.

**Gates, measured before and after:** `pytest tests/functional_process -q` → 3755 passed,
3347 skipped, unchanged; `len(GRAPH.nodes)` → 156, unchanged (node names are slot paths,
so a filename cannot move them, and this confirms it); `--collect-only` test ids →
set-identical after substituting the ten renamed stems, 2717 ids remapped, none added or
lost. `ruff format --check` delta is zero. `ruff check`'s delta is the six `N999 Invalid
module name` findings the capitals were raising, now gone — which is also why
`pyproject.toml`'s `N999` entry in `per-file-ignores` for `tests/functional_process/**`
was removed: no filename in the port or its cases has a capital letter any more.

**Two stale references were left, both deliberately:** `next_steps.md` and
`mda_harness.py` were being edited concurrently in another session and are not this
change's to touch. Between them they still name `physics_A_pure_formulas`,
`physics_B_composition`, `power_B_thermal_cryo`, `power_C_electric_production` and
`stellarator_B_st_phys`. Everything else in the repo was swept, including three
references that were line-wrapped mid-stem and so invisible to a line-based grep
(`path_refactor.md`, `stellarator_fwbs_s2.md`, `test_tf_coil_power.py`) — worth
remembering next time a stem is renamed, because a `grep` that finds nothing is not
proof.
