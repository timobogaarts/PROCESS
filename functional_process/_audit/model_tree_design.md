# The model tree — one typed pytree of models, superseding switches, the settings tree, and `Configuration`

**Status: settled design; §8 steps 1, 2 and 3 are implemented and committed.** Step 3
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

*As run:* all identical, **measured before and after with one script rather than against
the recorded figures** — 159 nodes, **139** blocks / 14 driven, 348 unowned inputs, and a
sha of every node's (inputs, outputs, type) unchanged. The 139/348 differ from the 138/349
recorded above and in `next_steps.md`'s table; the conversion did not move them (before ==
after), so the recorded pair was stale and is corrected there. `EXCLUDED_NODE_NAMES` did
**not** still match and had to be re-pinned — `path_refactor.md` §B.5 predicted it would
survive, and why that prediction failed is recorded there. `test_configuration.py`'s
`divertor_cycle` was the one test holding a node name; the two were the whole surface, as
that section's *count* correctly said.

**Step 4 — switches → slots + `machine_from_indat`; delete the old mechanism.**
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
