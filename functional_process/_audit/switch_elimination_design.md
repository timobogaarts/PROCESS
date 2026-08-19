# Replacing integer switches with model selection — feasibility and blast radius

**Question asked**: get rid of the `i_*` integer switch mechanism entirely — "models should
be swapped, not arbitrary switches that mean nothing" — and make the selection
JSON-serialisable: which models are included, plus specific selections for particular
switches. Does that work, and what does it cost?

**Short answer**: yes in principle, and the proof already exists in this repo. But
"switch" currently denotes four different things, only one of which should become a model
choice; and the change splits cleanly into a cheap half worth doing now and an expensive
half that is already this project's declared policy and should stay incremental.

## 1. It works in principle — and that is already established, not a new claim

`configuration.py`'s module docstring proves the load-bearing property:

    grep -n "\"i_\|'i_" process/core/solver/iteration_variables.py   -> no matches
    grep -n "\"i_\|'i_\|istell" process/core/scan.py                 -> no matches

No switch is an iteration variable and no switch is a scan variable, so **no switch can
change between two evaluations of one assembled graph**. `Scan` re-solves from scratch per
point anyway. A graph is therefore a pure function of a static selection made once — which
is exactly the condition under which a declarative, serialisable document can fully
determine it. JSON works.

## 2. It is also already the declared policy — practice has drifted from it

`traceability_policy.md` § "Switches — the split default" already says what the question
asks for:

> **Default: split.** A switch whose branches read different `VarPath` sets ... becomes
> separate ported functions/nodes, one per value, chosen at graph-build time.
> **Exception: static kwarg**, only when the per-branch reads-sets are **provably
> identical**.

So this is not a new architecture. It is finishing the one already chosen. The gap is
enforcement: the split-default deviation count reached **six** by the costs port, each
individually justified and recorded, none reversed.

## 3. The trap: "switch" is four different things

Measured on the assembled graph: **143 nodes, of which 31 carry static kwargs, spanning 29
distinct kwarg names** (51 kwarg sites, per `mda_harness.switch_audit`). They are not one
category:

| kind | examples | should it become a model choice? |
|---|---|---|
| **(a) model selection** | `i_confinement_time` (52 values), `i_cost_model`, `i_thermal_electric_conversion`, `i_p_coolant_pumping`, `i_blanket_type`, `i_tf_sup` | **Yes.** This is the question's target. |
| **(b) shape / resolution** | `n_plasma_profile_elements=201`, `n_cs_pf_coils=0` | **No.** Static because array shapes and loop trip counts must be concrete under `jit`. They are numbers and should stay numbers; calling them "models" would be a category error. |
| **(c) set membership** | `imp_indices=(0..13)` — which impurity species exist | **No, but not a switch either.** A set, not a choice. Already flagged by `switch_audit` as having no backing `DataStructure` field. |
| **(d) alias / noise** | `is_ignited=True` (a bool restatement of `i_plasma_ignited == 1`) | **Delete, don't rename.** Pure redundancy; `switch_audit` already needs a special case for it. |

Conflating (a) with (b)/(c) is the main way this refactor could go wrong. A JSON document
that lists `n_plasma_profile_elements` as a "model" would be worse than what exists now.

## 4. The names already exist upstream — this is the big enabler

**51 `IntEnum` classes exist across `process/`**, covering most of the switches in question
(`CostModels`, `ConfinementTimeModel` (52 members), `PlasmaIgnitionModel`,
`ElectricConversionModelTypes`, `PumpingPowerModelTypes`, `DensityLimitModel`, ...). We
would not be inventing a vocabulary; we would be promoting one.

Two things about *where* they live are diagnostic:

- Many live in **`process/core/io/plot/summary.py`** — PROCESS invented these names for
  **reports**, not for model selection. The names exist precisely because the integers
  "mean nothing" to a human reading output. That is the question's own complaint,
  independently confirmed by PROCESS's own codebase.
- This port **already converts int → name at node entry** in several places
  (`PlasmaIgnitionModel(i_plasma_ignited)` ×8, `ConfinementRadiationLossModel(int(i_rad_loss))`,
  `ConfinementTimeModel(...)`). The integer is already noise in the middle of the pipeline:
  a name is chosen upstream, flattened to an int to cross the interface, and immediately
  re-expanded to a name. Removing the flattening removes a real, existing round trip.

## 5. Recommendation: two separable changes, not one

### (A) Rename the selection layer — cheap, do this first

Replace integer values with named identities in `configuration.py`'s vocabulary only:

```python
Configuration({".costs.i_cost_model": 0})          # today
Configuration({"costs": "PROCESS_1990"})           # after (A)
```

No node is split; no graph shape changes. **This alone kills the bug class that has cost
this project five separate defects** (`i_confinement_time` 34/38, `i_thermal_electric_conversion`
0/2, `i_p_coolant_pumping` 2/1, `i_plasma_ignited` 0/1, `i_cost_model` 1/0) — because a
name cannot be silently copied off the wrong default the way an integer can. `PROCESS_1990`
versus `KOVARI_2014` does not typo into itself; `0` versus `1` does.

### (B) Split nodes per branch — expensive, keep incremental

Eliminating static kwargs means one node class per branch. Correct where reads-sets differ
(the policy's own criterion), and genuinely hard where they do not:

- **`ComponentThermalPowers` carries five static kwargs simultaneously**
  (`i_p_coolant_pumping`, `i_blkt_dual_coolant`, `i_thermal_electric_conversion`,
  `i_blanket_type`, `secondary_cycle_liq`). A full cartesian split is a combinatorial
  explosion of node classes. This is *why* the six deviations exist, not an oversight.
- **`i_confinement_time` has 52 enum members.** Only one is in scope; `Alternative(unported=...)`
  already handles the rest, so the cost is bounded — but it is 52 declarations to write out.
- **Selection does not always factor per-switch.** `.fwbs.blktmodel,.heat_transport.ipowerflow`
  is already declared under a *synthetic* comma-joined path whose `value` is an arm index,
  not a field value, because the dispatch is genuinely joint. Any "one JSON key per switch"
  scheme must accommodate this; it is existing evidence, not a hypothetical.

**So the honest end state is not "zero static kwargs".** It is "static kwargs only where
the reads-set is provably identical" — i.e. the policy as written, actually enforced, with
(b)/(c)-class parameters explicitly reclassified so they stop being counted as switches at
all.

## 6. The JSON shape

Two candidates, and they are not exclusive:

```jsonc
// node-centric — works today, needs no splitting
{"nodes": {"ConfinementTime": {"model": "ISS04_STELLARATOR", "radiation_loss": "CORE_ONLY"}}}

// include-list — closest to "models should be swapped", requires (B) per unit
{"include": ["Iss04StellaratorConfinementTime", "Costs1990", "EcrhHeating"]}
```

Recommend **node-centric first** (it is a faithful serialisation of what
`declarations_for` already computes), migrating entries into the include-list form as (B)
lands unit by unit. That way the document is useful immediately and converges on the
target shape without a flag day.

## 7. Blast radius, measured

| surface | size | nature of change |
|---|---|---|
| `configuration.py` | 249 lines | **Core.** `Alternative.value: int`, `Switch.default: int`, `Configuration.choices: dict[str, int]` all become name-typed. |
| `total_process.py` | 1236 lines | 8 `Switch` declarations, `REFERENCE_CONFIGURATION`, and 31 node registrations carrying static kwargs. |
| `test_configuration.py` | 290 lines, 46 tests | Several parameterised directly on integer arm values. |
| `mda_harness.py` | 683 lines | **`switch_audit` is the risk — see below.** |
| audit records | **22 `.md` files, 133 cited `i_x = N` lines** | Mechanical but voluminous; the repo's standard requires citations stay accurate. |
| `unit_registry.md` switch table | 17 rows | Each records a split/keep-static decision that this work would revisit. |
| per-unit tests | many | `Tier1Contract` samples pass switch integers as ordinary arguments. |

Plus `run_mda_harness.py`, `render_xdsm.py`, `mda.py`, which consume `graph_for()` and are
largely unaffected.

## 8. The one thing not to break

**`switch_audit` must survive the refactor.** It is the check that caught five of these
bugs, and it works by comparing each registered integer kwarg against the same field's
value on a converged `DataStructure`. If selection becomes names, that comparison needs an
enum-aware form — name → int → field — or the safety net that motivated this whole line of
work disappears in the act of acting on it. Any implementation plan should land the
enum-aware `switch_audit` *before* changing the registrations, not after.

## 9. Verdict

- **Feasible**: yes, and the enabling property is already proven and already relied on.
- **Already policy**: yes; (B) is enforcement, not redesign.
- **Do (A) first**: bounded, mechanical, and it removes by construction the single most
  expensive recurring defect class in this project's history.
- **Do (B) incrementally**, per unit, on the existing reads-set criterion — and first
  reclassify the (b)/(c)/(d) kwargs so the target is stated honestly instead of as a
  count that can never reach zero.


## 10. Revision after discussion — composition, not enumeration

The framing in §3/§5 above is superseded on two points. Recorded rather than rewritten,
because the correction is the useful part.

### 10.1 It is sugar over a model list, and the list already exists

Measured: `total_process.COMMON` is a **tuple of 79 declarations** (21 instances + 58 bare
classes). A `Switch` is nothing more than a mapping `int -> tuple[declaration]` — the
`.costs.i_cost_model == 0` arm simply contributes 43 more entries to that same list.
`declarations_for` concatenates. There is no other mechanism. So "replace switches with a
model list" is not a redesign of the assembly layer; it is **deleting an indirection** and
naming the list directly.

### 10.2 Model and settings are already one pytree — per node

Every declaration is an `eqx.Module`, so `ProfileGrid(n_plasma_profile_elements=201)`
already *is* "the model together with its settings, as one pytree". Two properties of that,
both worth stating explicitly since neither is obvious:

- **Traced leaves are empty.** Every setting is `eqx.field(static=True)`, so a
  configuration pytree has no differentiable content. `jax.tree.map` over it does nothing.
  The pytree-ness is structural bookkeeping, not autodiff plumbing.
- **That is a feature, not a limitation**: static fields live in the *treedef*, so changing
  a setting changes the jit cache key and forces recompilation — which is exactly correct,
  and comes for free.

What does **not** exist yet is a *top-level* configuration pytree grouping models by
provenance (costing / buildings / magnets / power balance). `COMMON` is flat. Adding that
grouping is additive — `declarations_for` already flattens — and composes naturally with
the above.

### 10.3 §5(B)'s "combinatorial explosion" objection is withdrawn — and the replacement
### framing (composition) was also wrong

§5(B) argued that splitting `ComponentThermalPowers`'s five simultaneous static kwargs
means a cartesian product of node classes. That is true only of *enumeration*, so the
objection is withdrawn. But the first replacement — one node holding five **sub-model
fields** — is also wrong, and the simpler reading is right: **it is not one composed
model, it is about five models fused into one node.**

Measured: `ComponentThermalPowers` has **21 outputs and 31 inputs** under five switches,
and the outputs cluster visibly along them — `p_*_coolant_pump_elec_mw` /
`p_coolant_pump_total_mw` with `i_p_coolant_pumping`; `p_blkt_liquid_breeder_*` /
`p_blkt_breeder_pump_elec_mw` with `i_blkt_dual_coolant`; and so on. It is a faithful port
of `calculate_component_thermal_powers`, one large PROCESS procedure — the fusion comes
from PROCESS, not from a modelling choice here.

So the right decomposition is **split into roughly five nodes, each carrying one model
choice**, not one node parameterised five ways. That is strictly better for the stated
goal: each becomes independently swappable, and the graph then shows the real structure
instead of hiding it inside a 21-output node. There is precedent *within this same class*
— six `FixedPointFunction`s were already split out of it to isolate self-loops.

**Caveat, stated rather than glossed**: the output-to-switch clustering above is inferred
from names and switch semantics, not yet from a per-branch reads/writes analysis. The
honest test is the split-default's own criterion — do the branches read different
`VarPath` sets — and that analysis has not been done for this node. It should be, before
the split is designed.

### 10.4 [RETRACTED] "The real blocker: ports are class-derived, not instance-derived"

**This section claimed instance fields cannot change a node's reads-set, and that fixing
it needed an upstream `~/jaxgraph` change. That is false.** Tested directly:
`ExplicitFunction._params` is an ordinary property, so a declaration can override it to
compute its ports from its own static fields, and `to_graph` accepts the resulting
*instance* and builds correctly:

```python
class Composed(ExplicitFunction):
    source: str = eqx.field(static=True)
    ...
    @property
    def _params(self):                      # ports derived from the instance
        ...                                 # -> Composed(source="b1") reads .q.b1
                                            #    Composed(source="b2") reads .q.b2
```

Both assemble; the base class is unaffected. `COMMON` already mixes 21 instances with 58
bare classes, so instance-passing was never in question — what was wrong was the claim
that an instance's *fields* cannot drive its ports.

Consequences of the retraction:

- **No upstream prerequisite.** This whole direction is unblocked today.
- **`StellaratorConfinementTime` need not have been a subclass.**
  `ConfinementTime(q95_source=...)` with an instance-derived `_params` would have worked
  and is simpler. The subclass is not wrong — it also gives the arm its own node name,
  since `NodalDeclaration.name` is `type(self).__name__` — but it was not forced, and
  `next_steps.md` § 8.3's reasoning ("the declaring class *is* the unit of rebinding")
  overstates the constraint.
- **One real caveat remains**: `_params` is underscore-private, so overriding it reaches
  into cottax's internals rather than a documented extension point. Promoting it to a
  supported hook is a small, worthwhile upstream change — but a convenience, not a
  blocker.

### 10.5 Revised order

1. **Enum-aware `switch_audit`** — § 8's warning stands and comes first: it is the check
   that caught five bugs, and it must not be lost in the act of acting on this.
2. **Name the list directly**, deleting `Switch`/`Alternative`/`Configuration`'s integer
   indirection; group by provenance while touching it.
3. **Split fused nodes** (`ComponentThermalPowers` first) on the reads-set criterion, one
   model choice per node — preceded by the per-branch reads/writes analysis § 10.3 says
   is missing.
4. **JSON**: a name -> class registry over the result. Cheap once 2 lands.
5. *Optional, upstream*: promote `_params` (or an equivalent) to a documented hook.

## 11. Provenance grouping vs. derived structure — measured

The proposal: group models by **provenance** (costing / buildings / magnets / power
balance) on human, physical grounds, let **cottax derive the run order** from actual
reads/writes, and then *compare the two*. Measured on the current 143-node graph, using
the port's own module layout as the provenance grouping and `Blocking.scc(driven_graph(
GRAPH))` as the derived one.

### 11.1 The result is scale-dependent, and that is the finding

| grouping grain | SCCs with >1 real node | how many cross a boundary |
|---|---|---|
| top-level subsystem (`stellarator`, `physics`, `costs`, ...) | 3 | **0** |
| source file (`stellarator.build`, `stellarator.divertor`, ...) | 3 | **3** |

Every genuine cycle is **contained within one subsystem** and **spans several files inside
it**:

- `DensityProfile` / `FusionRates` / `PlasmaComposition` / `ParabolicOnAxisDensities` —
  three files, all `physics`.
- `WindingPackIntersectInputs` / `Intersect` / `WindingPackTotalSizePost` — two files, both
  `stellarator.coils`.
- `Divertor` / `AFwTotalWithPowerflow` — `stellarator.divertor` + `stellarator.build`.

Meanwhile there are **346 cross-subsystem edges** (`stellarator -> power_B_thermal_cryo`
35, `physics -> stellarator` 30, `stellarator -> costs` 19, ...). So subsystems are heavily
coupled — just **acyclically**. That is precisely why the MDA needs only 11 driven blocks
out of 128.

### 11.2 What the comparison actually bought

A concrete answer to a question we could not otherwise settle: **the subsystem is the right
grain for a model group; the file is not.** File boundaries never contain a cycle, and
PROCESS's file layout is therefore not the natural computational unit. Subsystem boundaries
contain every one.

That is the argument for doing this comparison as a standing instrument rather than a
one-off. The two groupings answer different questions — provenance is about ownership,
naming and configuration; structure is about scheduling — and neither should be derived
from the other. Where they agree, the human name is trustworthy. Where they disagree, the
disagreement is the signal:

- a provenance group scattered across many blocks is a *label*, not a module;
- an SCC spanning provenance groups is a **genuine cross-subsystem feedback loop**, which
  is the single most interesting structural fact a design like this can surface.

Half the tooling already exists: `render_xdsm.py` calls `render_dsm_html`. A DSM ordered by
provenance next to one ordered by SCC is the standard form of exactly this comparison.

### 11.3 Two caveats, both material

- **Provenance here is PROCESS's file layout**, mirrored by the port's module tree — not an
  independently declared physical grouping. So § 11.1 partly measures PROCESS's filing
  habits, not physics. Declaring the grouping ourselves on physical grounds and *then*
  finding the same containment would be a far stronger statement; it is the version worth
  building.
- **"No cross-subsystem SCC" may be an artifact of scope.** `next_steps.md` records that
  more SCCs are expected once the orchestration layer (currently unported) lands, and
  `CLAUDE.md` asserts PROCESS has feedback loops among plasma physics, TF coil and build.
  None of those appear here. The honest reading is "not in the ported subset", not "they do
  not exist".

### 11.4 On a pytree-aware node interface

Asked whether one is needed. Precisely: **not for any of the above.**

- instance-derived ports already work (§ 10.4);
- provenance grouping is metadata over the declaration list and needs nothing new;
- switch removal needs nothing new.

What it *would* buy is the **JSON goal**: today a node's ports are declared as `__call__`
parameter *defaults*, so a declaration is only fully described by a Python function
signature. Expressing ports as **data** — a pytree of `VarPath`s carried by the instance —
makes a node declaration serialisable end to end, and makes instance-derived ports a
first-class feature instead of an override of the underscore-private `_params`.

So: worth building **for serialisation**, not as a prerequisite for the clustering or the
switch work. Sequencing it after § 10.5's step 2 keeps it honest — by then the model list
is named and grouped, and the interface has a concrete thing to serialise.

## 12. Brainstorm — what cottax should provide for hierarchical node names

Prompted by: `NodePath`s as pytree locations make provenance free, make subtree swap
natural, visualise well, and let settings be addressed off the node's own path
(`settings.plasma.profiles`). § 11.4 concluded a pytree-aware node interface was optional;
that conclusion was made about *serialisation* alone and understates the case.

**Correction to § 10/§ 11's framing on where provenance belongs.** An earlier draft argued
provenance should be assigned at registration, "a property of the assembly, not the model".
That is wrong: a model *is* a `buildings.cost` model — its physical origin is intrinsic, and
no real case in this codebase has one declaration class belonging under two provenances.
What was actually being conflated is two different problems:

- **provenance** — intrinsic, belongs to the model;
- **instance disambiguation** — two instances of one class needing distinct node names;
  genuinely an assembly concern, and today impossible, since node name *is*
  `type(self).__name__`.

A good design solves both, and should not solve the second by compromising the first.

### 12.1 The principle actually at stake

`NodalDeclaration.name` (`cottax/interfaces/pytree_namespace_module.py:190-194`):

    # Nodes are named by class, exactly as on the flat surface -- only *variables*
    # live in the caller's structure; a discipline's own identity does not.
    return NodePath((DictKey(type(self).__name__),))

This is a deliberate asymmetry, not an oversight. But cottax *already* derives variable
names from a position in the caller's pytree (`Input(lambda s: s.physics.rmajor)` ->
`.physics.rmajor`). Deriving node names from a position in the caller's **model** pytree is
the symmetric completion of that same idea. **`Graph` already accepts hierarchical
`NodePath`s** — verified directly: `Graph(path_map({...}))` with three-component paths
builds, `topological_order` and `Blocking.scc` work, ragged depth is fine. Only the
*declaration surface* cannot express one.

### 12.2 Options

**A. The model tree is a pytree; node paths are its key paths.** *(recommended)*

```python
# functional_process/models/physics/profiles.py  -- provenance declared by the model's own module
MODELS = {"grid": ProfileGrid(n_plasma_profile_elements=201), "density": DensityProfile()}

# total_process.py -- composition only, no naming decisions
MODELS = {"physics": {"profiles": profiles.MODELS, "fusion": fusion.MODELS}, "costs": ...}
to_graph(MODELS)     # names are key paths: physics.profiles.density
```

- Symmetric with the variable side; nothing new conceptually.
- **Provenance stays intrinsic** — each module declares its own subtree; `total_process`
  only concatenates. This is what answers the objection above.
- **Solves instance disambiguation for free**: the *key* names the node, so two instances
  of one class coexist (`{"coarse": ProfileGrid(n=51), "fine": ProfileGrid(n=201)}`).
- Subtree swap/delete become dict operations; settings mirror the same shape.
- Cost: node name decouples from class name, so the code<->node correspondence weakens.
  Mitigate by defaulting an omitted key to the class name.

**B. An intrinsic `namespace` declared on the class.**

```python
class DensityProfile(ExplicitFunction):
    namespace: ClassVar = ("physics", "profiles")
```

- Most literally "provenance is intrinsic"; needs no restructuring of `COMMON`.
- Repetitive across a module (mitigable with per-subsystem base classes).
- Does **not** solve instance disambiguation.

**C. Derive from `__module__`.** Zero declaration burden, and the source tree really is the
physical organisation. **But § 11.1 measured that file granularity is the wrong grain** —
every SCC crosses file boundaries while none crosses subsystem boundaries — so the full
module path groups too finely, and truncating it is a fudge. Also couples node identity to
file layout.

**D. Provenance as metadata beside identity, not in it.** Preserves cottax's current
principle and still gives grouping and visualisation, but loses subtree-swap-by-path and
the settings-address symmetry, which are the main prizes.

### 12.3 What cottax would need for (A)

1. **`NodalDeclaration.name` must stop being hardcoded** to `type(self).__name__` — it
   needs to be derivable from the declaration's position, not its class.
2. **`to_graph`/`node_and_names` must accept a nested mapping of declarations.** Today the
   mapping form (added by the fix recorded in `next_steps.md` § 8) accepts only a raw
   `NodeDefinition` — it requires `.owns`, so passing a `NodalDeclaration` raises
   `AttributeError`. Verified directly. This is the single concrete blocker, and it is a
   small extension of a change already made.
3. **A readable renderer for hierarchical names.** `path_str()` currently gives
   `['physics']['profiles']['DensityProfile']`; dotted output is wanted for XDSM/DSM
   labels, reports and error messages.
4. **Prefix/subtree queries on `Graph`** — e.g. `graph.under(("physics",))`. This repo
   currently does substring matching on `path_str()` (`mda_harness.EXCLUDED_NODE_NAMES`),
   which is exactly the fragile idiom a real subtree query would replace.
5. **Prefix-aware grouping in `Blocking`/visualisation**, so the provenance-vs-structure
   comparison of § 11 can be rendered natively (a DSM ordered by provenance beside one
   ordered by SCC) rather than scripted ad hoc.

Items 1-2 are the minimum for (A); 3-5 are what make it pay off.

### 12.4 Open questions worth deciding before building

- **Should the key default to the class name when omitted?** Keeps the common case terse
  and preserves code<->node correspondence, at the cost of two ways to spell one thing.
- **Is node identity stable under regrouping?** Putting provenance in the path means moving
  a model between groups renames it, invalidating any stored reference (audit records,
  `EXCLUDED_NODE_NAMES`-style lists, saved schedules). Acceptable if groups are stable;
  worth an explicit decision rather than a discovery.
- **Does a hierarchical name interact with minted paths?** `^cond.*`/`^hat.*`/`^problem.*`
  prefixes already occupy a namespace of their own; how a minted node under
  `physics.profiles` should be spelled is undecided.

### 12.5 Refinement — namespaces are classes/modules, not dicts

§ 12.2(A) sketched the model tree as nested `dict`s. **Classes (or modules) are the better
surface, and for a reason that strengthens § 12.1's symmetry argument rather than merely
being nicer to type**: the variable side is *already* attribute access on a namespace —
`DataStructure` is a dataclass of dataclasses and `Input(lambda s: s.physics.rmajor)`
traverses it by attribute. A class-of-classes for models is the exact mirror; dicts would
introduce an asymmetry precisely where the design is arguing for symmetry.

```python
class PhysicsProfiles:
    coarse = ProfileGrid(n_plasma_profile_elements=51)
    fine   = ProfileGrid(n_plasma_profile_elements=201)   # impossible today: node name is the class name

class Physics:
    profiles = PhysicsProfiles
    fusion   = PhysicsFusion
```

Secondary gains: attribute typos fail at import instead of producing a silently misnamed
node; editors can complete the tree; no string keys to keep in sync with anything.

**Modules as leaves, composed explicitly into the tree.** A bare module gives provenance
for free, but only at *file* grain — which § 11.1 measured to be the wrong grain (every SCC
crosses file boundaries; none crosses subsystem boundaries). Composing modules explicitly
keeps provenance originating in the model's own module while letting the composition choose
the grain.

**Implementation wrinkle for the walker**: 58 of `COMMON`'s 79 entries are *bare classes*.
A `NodalDeclaration` subclass held as a class attribute is both "a class" (namespace-shaped)
and "a declaration" (a leaf), so `issubclass(x, NodalDeclaration)` must be tested before the
namespace branch or bare-class registrations get walked into.

### 12.6 Minted paths — spelling is settled; subtree membership is not

§ 12.4 listed "how minted paths interact with hierarchical names" as undecided. Overstated:
`^cond.physics.profiles.*` is the natural extension of the existing `^cond.constraints.c32`
/ `^problem.sand`, and for *variables* the question does not arise at all — a `VarPath`
carries its own `DataStructure`-mirroring namespace, independent of node hierarchy.

The real question is narrower and is about **subtree queries**: with the mint prefix
outermost, `^problem.physics.profiles.density` does **not** live under `physics`. Swapping
out the `physics.profiles` subtree would therefore leave its minted problem nodes behind —
almost certainly wrong. Two ways out:

- **provenance outermost** (`physics.profiles.^problem.density`) — subtree-correct, uglier,
  and changes how every existing minted name reads;
- **prefix outermost, and `graph.under()` is mint-aware** — keeps current spelling, puts the
  burden in one query function.

The second looks right, but it should be decided deliberately: `Residualise`/`Combine` mint
nodes constantly and the SAND layer depends on exactly this.

### 12.7 Settings as a parallel tree, zipped by `materialise`

The settled shape. **Three trees, not two**, each with one job:

| tree | contents | static/traced | who owns it |
|---|---|---|---|
| **model tree** | which models exist (`NodalDeclaration` classes), as a class/module namespace | structure | the models' own modules (provenance is intrinsic, § 12.5) |
| **settings tree** | their static parameters, *same shape* | static -> lives in the treedef | the run's configuration |
| **env** | boundary inputs | traced, differentiable | already exists (`mda_harness` builds it from a converged run) |

`materialise(models, settings)` zips the first two into the concrete instantiated tree
cottax consumes. This replaces the integer `Configuration` outright: the settings tree *is*
the configuration.

**The settings schema is derivable, not declared.** Every model already declares its
parameters as `eqx.field(static=True)` with types (`ProfileGrid.n_plasma_profile_elements: int`).
So `materialise` can validate the settings tree against the model tree's own static fields
and reject a misspelled or ill-typed key at materialise time. A checkable schema for free —
something baking values into constructor calls never provides.

**The model/settings boundary already has a criterion**: `traceability_policy.md`'s split
default. A branch that changes the *reads-set* is a different model and belongs in the model
tree; one that changes only a value is settings. The same rule governs the switch work
(§ 5(B), § 10.3) and this boundary — a good sign the decomposition is real rather than
aesthetic.

**Absence is `None`**, so the two trees keep exactly equal shape rather than settings being
a sparse overlay (an earlier draft claimed otherwise; withdrawn). Verified: `tree_map` is
driven by the first tree's structure, so with the model tree leading, `None` in the settings
position is accepted as that subtree with no `is_leaf` argument needed. The only place
`None`-as-empty-node would bite is flattening the settings tree *alone*
(`tree_leaves(None) == []`), which `materialise` never needs to do.

**Caveat that will surprise someone: a settings sweep costs recompiles, not `vmap`.**
Settings are static and live in the treedef — which is exactly why changing one correctly
invalidates the jit cache (§ 10.2). But it means § 11's measured batching win (46x at B=64)
applies only to *traced* values, i.e. `env`. Sweeping `n_plasma_profile_elements` over 64
values is 64 compilations. Unavoidable and correct for anything shape-changing, but it is
not the same operation as a scan over `rmajor`.

**`materialise` belongs to the caller, not cottax.** cottax receives an already-materialised
tree and stays thin.
