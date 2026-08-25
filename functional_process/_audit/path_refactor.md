# The two path refactors — variable ports, then node names

**What this file is.** The plan for the two naming conversions in this port, both of
them client work against cottax features that are already landed. **Part A is DONE**
(2026-08-20): all 36 files converted by the §A.3 codemod, every file proven by the §A.4
side-by-side checker (**2222 ports total across the wave commits, all identical and in
order**), all 36 §A.5 renames applied with pure functions and sample keys untouched, all
43 §A.6 escapes de-lambda'd, and the out-of-census `path_of`/`FromExactly` lambda sites
converted through cottax's `resolve(area.field, VarPath)` — so the §A.6 postcondition
holds flat: `grep -rn "lambda s:" functional_process --include='*.py'` → **0**. Gates
after the whole conversion: `pytest functional_process -q` **3724 passed** and the MDA
harness line **byte-identical** (499/34/3/0 · 557/0 · 61/0/3/0). The one-shot `_codemod/`
tooling lived and died inside the Part A commits, per its own docstring. Part B is
superseded by `model_tree_design.md` (see its banner below).

**Every node name this file spells is now historical.** `model_tree_design.md` §8 step 3
landed the machine tree, and a node is named by the **snake_case slot path** that reaches
it: `physics.profiles.DensityProfile` below is `.physics.profiles.density_profile` today,
`stellarator.Divertor` is `.stellarator.divertor`, and the class name is gone from the
name entirely (identity is the *place*, that design's §3.2). The reasoning in these
sections stands; the spellings are what they were when it was written.

They are **two refactors, not one**, and the distinction is the reason they can be
sequenced rather than negotiated: the declaration surface carries **variable** paths into
the caller's data structure (`.area.field`, a `VarPath`), the model tree carries **node**
names (`physics.DensityProfile`, a `NodePath`). Different namespaces, no overlap
(`switch_elimination_design.md` §14 established this and it still holds — cottax's
`node_and_names` now tells the two apart by `KeyEntry` *kind*, `GetAttrKey` for a variable
place against `DictKey` for a model-tree position).

Their designs live elsewhere and are not restated here: Part A's in `test_harness.md`
§"Declaration surface — `From` / `OutputInto`", Part B's in
`switch_elimination_design.md` §§12–14. This file is what to *do*, in what order, and
which checks make each step provable rather than merely tested.

## 0. Verified state, measured against the live trees

| check | value |
|---|---|
| `pytest functional_process -q` | **3704 passed**, 3344 skipped, 0 failed (matches `next_steps.md`) |
| `cd ~/jaxgraph && pytest -q` | **482 passed**, 3 skipped — both cottax-side prerequisites below are changes to this tree |
| `run_mda_harness.py` | **499 agreements** (23 array-valued), **34 disagreements** (0 driven), 3 unverifiable, **0 ungrounded**; 557 walked / 0 unaccounted; 61 switch kwargs / 0 mismatched |
| declaration census, AST pass | **2315 declarations**: 2078 mechanical, 36 body-rename, 43 escape-hatch, 158 already converted |
| … the 36 renames | all in **19 `__call__` methods**; **0** pass the parameter as a keyword argument in the body |
| … spread | **36 files** hold the 2157 unconverted ones; `costs/costs.py` alone has 453 |
| `Output(..., static=)` / `tags=` in the port | **0** — every conversion is single-argument |
| `.md` records citing the old spelling | **125 occurrences across 23 files** |
| assembled graph | **88 `COMMON` entries** (63 bare classes), **10 topology switches**, **147 declarations** in `REFERENCE_CONFIGURATION`, **159 nodes** |
| hardcoded node-name strings in code | **1** (`mda_harness.EXCLUDED_NODE_NAMES`) |
| cottax `From`/`OutputInto`/`Area` | landed, `4eea7ab` |
| cottax hierarchical node paths (`named(at)`, `node_and_names(xs, at=None)`) | landed, `789df8b` |

The census reproduces the figures the pilot commit (`e0c4c134`) recorded, digit for digit,
so that record is still current and the conversion has not drifted under it.

### Two defects found while measuring, both prerequisites

- **`paths.py`'s module docstring documents a design cottax withdrew.** It describes
  "three forms, one rule each", of which the first — a **bare area as a parameter
  default**, `qnuc=data.fwbs` with no wrapper — is refused by cottax today:

      TypeError: A.__call__: parameter 'rmajor' has no read to bind ...
      Write `rmajor=From(physics)` ...

  That was dropped in cottax `fd3211c` ("Drop the docs and the adapter left over from the
  two-spelling design"). The docstring's closing claim about the callable form — *"the one
  form that cannot be built against the wrong data structure"* — is also no longer true:
  the recorder form (`impurity_radiation.f_nd_impurity_electron_array[2]`) is built off
  `Root(AREAS)`, which refuses a misspelled area at declaration time, so it is at least as
  safe. Verified directly, both halves.

- **`Input` still silently accepts a bare `Area`.** (`Input` is `FromExactly` since
  §A.7; it is named as it then was, because that is when this was found.)
  `Input(physics)` on a parameter named
  `rmajor` produces `.physics.rmajor` — identical to `From(physics)`. `From`'s own
  docstring says this two-spelling ambiguity is exactly what the surface exists to remove
  (*"this is not `Input(physics.rmajor)` with `Input(physics)` also allowed"*), and
  `From`/`OutputInto` type-check their argument, but nothing guards the other direction.
  Verified directly.

  This matters here beyond tidiness: a codemod that emitted `Input(area)` instead of
  `From(area)` would produce **identical ports and pass every test in the repo**, across
  2078 sites, while restoring the ambiguity the conversion is being done to remove. The
  guard is what makes the codemod's output checkable by construction rather than by
  reading it.

## Part A — `VarPath`: the declaration surface

### A.1 What changes, and what provably does not

    nd_plasma_electron_profile_integral = Output(
        lambda s: s.physics.nd_plasma_electron_profile_integral
    )                                    ->  ... = OutputInto(physics)

    def __call__(self,
        profile_x=FromExactly(lambda s: s.physics.radius_plasma_profile_norm),
    )                                    ->  radius_plasma_profile_norm=From(physics),

**Only node classes are touched.** The conversion reaches `__call__`/`step` signatures and
`Output` class attributes and nothing else — never a `calculate_*` pure function. That is
not an aspiration, it is checkable and checked: every `Tier1Contract`/`Tier2Contract` sets
`ported` to the **pure function** (`ported = calculate_profile_grid`) and keys `samples` on
*its* parameter names; **no test in the repo sets `ported` to a `__call__`**. So the whole
tier-1/2 harness — the value and gradient agreement that is the point of this project — is
structurally out of the blast radius, including for the 36 renames.

### A.2 Step 0 — land the two corrections in §0 first

**Both are done.**

- cottax `FromExactly`/`Output` now refuse a bare `Area` (`_refuse_bare_area`). The distinction
  cannot be by type — `From`/`OutputInto` *build* one of these holding an `Area`, which is
  how `resolve`'s area branch is reached at all — so it is a private `_sugar` flag that
  only the two sugar constructors set. One existing test,
  `test_an_area_completes_a_declaration_with_its_own_name`, asserted the behaviour now
  forbidden: a leftover of the withdrawn two-spelling design that outlived it, replaced by
  `test_the_escape_hatch_refuses_an_area`.
- `paths.py`'s docstring rewritten to the surface that exists: two calls, told apart by
  which you call; `From`/`OutputInto` refuse a path and `FromExactly`/`Output` refuse an area;
  and the escape hatch does not need a lambda.

Verified inert: `~/jaxgraph` **482 passed** (same count — one test replaced by one),
`pytest functional_process` **3704 passed**, unchanged.

### A.3 The codemod

AST-driven, per file, in this order:

1. Locate every convertible site by `ast` — a parameter default `FromExactly(lambda s: s.A.f)`
   where `f` equals the parameter name, and a class attribute `x = Output(lambda s: s.A.f)`
   where `f` equals `x`. Replace the call's exact source span with `From(A)` / `OutputInto(A)`.
2. Rewrite the imports: add `From`/`OutputInto` to the `cottax.interfaces.pytree_namespace_module`
   import, add the area names to the `functional_process.paths` import, drop
   `FromExactly`/`Output`
   where no escape hatch remains in the file.
3. `ruff format` then `ruff check`. The diff is large in both directions — a three-line
   wrapped `Output(...)` collapses to one line — so formatting is part of the step, not a
   follow-up.

`libcst` is **not** installed in `process_port`; `ast` plus source-span replacement is
sufficient here because every target is a single expression with no interior comments, and
`ruff format` normalises the result. Do not add a dependency for this.

### A.4 The inertness proof — per file, not per suite

The pilot's method, made reusable: load the pre-conversion file (from `git show`) and the
post-conversion file **side by side as two modules**, and for every `NodalDeclaration`
subclass present in both, assert the `In`/`Out` `VarPath` tuples are identical **and in the
same order**.

This is the step that makes the conversion provable rather than tested, and it is not
redundant with `pytest`. The suite would not catch two same-typed reads swapped inside one
node — the values still flow, the graph still assembles, the numbers still land — and 2078
sites is far past the point where reading the diff catches that. Run it per file, report
the port count per file, and treat a file as unconverted until its count matches.

### A.5 The 36 renames — the only judgement in Part A

The parameter is spelled differently from the field, so the sugar cannot apply without
renaming the parameter to the field name and renaming in the body. All 36 sit in **19
`__call__` methods** and **not one of them** passes the parameter as a keyword argument
anywhere in its body — every call into the pure function is positional (checked
mechanically across all 36, not sampled). So each is a two-line edit and none can silently
rebind an argument. Two are more than a rename and get individual attention, both flagged
by the pilot commit:

- `pure_formulas.py` binds `nd_plasma_vol_avg` to `.physics.nd_plasma_electrons_vol_avg`
  in `ElectronThermalEnergy` and to `.physics.nd_plasma_ions_total_vol_avg` in
  `IonThermalEnergy` — **electrons and ions under one local name**. This is precisely the
  confusion the naming rule removes, and the conversion is what surfaces it.
- `confinement_time.py` binds `temp_plasma_electrons_vol_avg_kev` to
  `.physics.temp_plasma_electron_vol_avg_kev` (electron**s** vs electron).

The rest are ordinary abbreviations (`te`, `zeff`, `rho`, `profile_x`, `pfr`/`pfm`/`tfh`
in `buildings.py`). Rename them; `standards.md`'s `<type>_<system>_<description>_<units>`
scheme is what the field names already carry, and a local abbreviation is exactly the
second spelling this surface exists to delete.

### A.6 The 43 escapes — and why `lambda s:` still goes to **zero**

All 43 are array-element addressing that no parameter name can spell — 42 of them elements
of `.impurity_radiation.f_nd_impurity_electron_array`, plus `coils/mass.py`'s
`.tfcoil.dcond[I_TF_SC_MAT_ITER_NB3SN - 1]`. They keep the escape hatch, but they do
**not** need to keep the lambda: cottax's `Area` supports subscripting, so

    FromExactly(lambda s: s.impurity_radiation.f_nd_impurity_electron_array[2])
    ->  FromExactly(impurity_radiation.f_nd_impurity_electron_array[2])

resolves to the identical `.impurity_radiation.f_nd_impurity_electron_array[2]` (verified
directly, both `FromExactly` and `Output`). So the whole port converges on one namespace object
and the postcondition is not "2078 of 2157 sites converted" but the flat, greppable

    grep -rn "lambda s:" functional_process --include='*.py'   ->  0

Worth a test, for the same reason `From` type-checks its argument: it is the one check that
cannot pass by accident.

### A.7 The escape hatch is `FromExactly`, not `Input`. Done.

The two calls differ on **two** axes at once — direction, and whether the declaring name
completes the place — and the names only tracked one of them. The sharp version of the
problem: **the write side was already a stem-pair and the read side was not.**

| | area, completed by the declaring name | whole place, verbatim |
|---|---|---|
| **write** | `OutputInto(area)` | `Output(place)` |
| **read**, before | `From(area)` | `Input(place)` |
| **read**, now | `From(area)` | `FromExactly(place)` |

`OutputInto`/`Output` share a stem, so a reader sees at a glance that they are two
spellings of one idea and the suffix says which rule applies. `From`/`Input` shared
nothing and did not read as a pair at all. `Input` named the *syntax* — and cottax
already requires every parameter to carry a read, so "input" was guaranteed by position
and the word was free to carry the thing that actually distinguishes the two calls, which
it did not.

Frequency comes out right too: the common case (2078 sites) keeps the short name and the
rare one (43) takes the long one.

**Not an alias — a rename of the type.** `FromExactly = Input` would put two working
spellings back, exactly what §0's guard was added to prevent. `From` returns a
`FromExactly` the same way `OutputInto` returns an `Output`, so the two sides now have
identical shape: a class for the escape hatch, a function for the sugar.

Landed: cottax (the class, `From`'s return, the docstrings, `test_interfaces_pytree_
namespace.py`) and 1534 sites across 47 port files, a pure textual rename.
`flat_namespace_module` has no `Input`, so that surface is untouched. Verified inert —
`~/jaxgraph` **482 passed**, `pytest functional_process` **3704 passed**.

**One cost, stated:** `FromExactly` is eight characters longer than `Input`, so `E501`
across the port goes **224 → 351** (unsorted imports go the other way, 27 → 6). Part A deletes essentially all of those lines (they
become `From(area)`), so the overhang is temporary by construction — but it is real until
then, and the port is not `ruff format`-clean at HEAD (44 files would be reformatted), so
reformatting to absorb it would bury the rename in unrelated churn. Left alone
deliberately.

### A.8 Order

Largest first, so the biggest diff lands against the quietest tree: `costs/costs.py` (453),
`buildings.py` (193), `electric_production.py` (149),
`stellarator/coils/calculate.py` (136), `availability.py` (115),
`plasma_physics.py` (101), then the tail of 30 files. One commit per file or per
small group, each with its inertness count in the message — a 2157-site single commit is
unreviewable and unbisectable.

### A.9 The records

125 occurrences across 23 `.md` files quote the old spelling. Update a record when its
module is converted, in the same commit, so the two never disagree. `_audit/schema.md`,
`_audit/cottax_review.md` and `models/blankets/hcpb.md` (27) carry the most.

## Part B — `NodePath`: hierarchical node names

> **SUPERSEDED in design by `_audit/model_tree_design.md`.** The tree is now `eqx.Module`
> namespaces with typed snake_case slots (instances, not nested classes — a class
> attribute has no `tree_at`, so B.3's surface cannot express a swap), node identity is
> the slot path, and the switch arms B.4 could not place become slot occupants chosen by
> a `machine_from_indat` factory. Still live below: B.1 (what landed in cottax), B.5 (the
> measured reference surface), and B.4's exclusivity-by-construction observation, which
> the new design adopts. B.2's formatter-wiring item is mooted for the Machine tree —
> slots mint `GetAttrKey`s, which `keystr` already renders readably
> (`model_tree_design.md` §3.1/§3.3) — and remains open only for the legacy `DictKey`
> surfaces it was written about; B.3, B.4, B.4b and B.6 are replaced by that document's
> §§3–6. The Part-A-first sequencing rule stands unchanged.

### B.1 Already landed in cottax — verified, not assumed

`to_graph({'physics': {'profiles': [DensityProfile]}, 'costs': [Costs1990]})` builds, and
names its nodes by the key path that reaches them — three `DictKey`s, verified directly.
(It *renders* them `['physics']['profiles']['DensityProfile']`; that is §B.2, not a
naming problem.) `Graph`,
`topological_order` and `Blocking.scc` accept ragged hierarchical paths, and two 3-key
minted names already exist in the driven graph today
(`^problem.physics.proton_rate_density`, `^problem.fwbs.f_ster_div_single`). Classes work
as namespaces as well as mappings, with `issubclass(x, NodalDeclaration)` tested first so
a bare-class registration is read as a leaf and not walked into
(`switch_elimination_design.md` §12.5's wrinkle, handled upstream).

**The declaration surface was the only thing that could not express a hierarchical name,
and it no longer is.** Part B is client work only.

### B.2 Missing in cottax — two items, both small

**The dotted renderer is written but not wired up.** `switch_elimination_design.md` §13
was right: `spell_flat` and `xDSMFormatterFlat` exist, in
`cottax/interfaces/spelling.py`, re-exported by both namespace modules. What is missing
is the last two steps — `cottax.visualization` does not export the formatter (it exports
`Formatter`, `NoFormat`, `RaGraphFormat`), and `functional_process/render_xdsm.py` passes
no formatter at all, so nothing in this port uses it today.

That is the difference between `['physics']['profiles']['DensityProfile']` and
`physics.profiles.DensityProfile` in every XDSM label, DSM label, error message and
report. Cosmetic while names are one key long; hierarchical names are markedly worse to
read in bracket spelling, so this lands **with** Part B and not after it.

`graph.under(prefix)` (§12.3 item 4) is the second item and the one with a caller
waiting: `mda_harness.py` currently does substring matching on `path_str()`, which is the
fragile idiom a real subtree query replaces.

### B.3 The tree — landed for `COMMON`

    class COMMON:
        class stellarator:
            class coils:
                CoilCurrent = CoilCurrent()
            class fwbs:
                ShieldMass = ShieldMass()
            Build = Build()
        class physics:
            class profiles:
                DensityProfile = DensityProfile()
            class confinement_time:
                DoubleAndTripleProduct = DoubleAndTripleProduct()
            FusionRates = FusionRates()

Nested **classes**, not dicts — `switch_elimination_design.md` §12.5's settled position, and
the reason is symmetry rather than taste: the variable side is already attribute access on
the caller's pytree (`data.physics.rmajor`), so the model side should be attribute access on
a model pytree. Verified working end to end; ragged depth is fine.

**Every line reads *slot = occupant*.** `Build = Build()` — the attribute is the place in the
machine, the right-hand side is the model filling it. That is what makes a swap a one-line
edit at a named address, and it is uniform with the 28 registrations that already carried
configuration (`ComponentThermalPowers(i_p_coolant_pumping=1, ...)`).

**Grain: subsystem, with a third level only where the sub-area is a real thing** — an SCC
lives in it (`stellarator.coils`, `physics.profiles`), or it is a slot something could be
swapped into (`physics.confinement_time`) — never merely a filename. §11.1 measured why:
every genuine cycle is contained within one subsystem and spans several files inside it. So
no audit-chunk letters survive into identity — nor, since `model_tree_design.md` §10,
into filenames: `physics/pure_formulas.py` and `power/thermal_cryo.py` are named for what
is in them. `stellarator_fwbs_s1_s5` still carries how the port was *chunked for
auditing*, because `st_fwbs`'s S1–S6 re-chunking is still live (§3) and a name carrying
it would move again.

11 groups, 88 nodes placed. `to_graph` receives `{ROOT: COMMON}` rather than `COMMON` — a
namespace class contributes its own class name as the outermost key when nothing placed it,
so the container would otherwise name every node `COMMON.stellarator...`. Keying at the root
strips it and lets several subtrees merge as siblings.

**Verified inert.** 159 nodes, 139 blocks, 14 cycles — *identical* to the flat `COMMON`
measured immediately before, with the same code. `pytest functional_process` **3704 passed**;
MDA harness 499 / 34 / 3 / 0, 557 walked, 0 unaccounted, 61 switch kwargs 0 mismatched. Minted
names inherited the tree for free (`^problem.physics.profiles.IonVolAvgTemperature`), which is
§12.6's mint-prefix-outermost rule working as designed.

One test broke and deserved to: `test_configuration.py` hardcoded `"['Divertor']"`, jax's
bracket spelling of a flat name. Now `spell_flat(n) == "stellarator.Divertor"` — the fragile
idiom §12.3 item 3 exists to replace.

### B.4 Switch arms — still flat, and *why* the obvious design does not work

**59 of 147 nodes are still bare class names.** They are the switch arms, and placing them is
the remaining half of Part B.

The plan said: put the arm at the switch's own position in the tree, one `place` per `Switch`.
**That was tried and it is wrong** — `.stellarator.istell`'s `value=6` arm declares
`StellaratorMachineConfig` (→ `stellarator`) *and* `StellaratorConfinementTime` (→
`physics.confinement_time`). One arm, two places. A per-`Switch` or even per-`Alternative`
`place` cannot express it, so the field was added and then removed again.

**The shape that does work: an arm is a subtree, exactly like `COMMON`.**

    class _HeliasArm:
        class stellarator:
            StellaratorMachineConfig = StellaratorMachineConfig(machine_config=...)
        class physics:
            class confinement_time:
                StellaratorConfinementTime = StellaratorConfinementTime(...)

`build_graph` already merges subtrees this way (`{ROOT: subtree}` per arm), so the machinery
is in place and only the ten switches' 470 lines of `Alternative(declarations=(...))` need
rewriting. Nothing blocks it; it is a separate, mechanical pass.

**One observation it hands to §12.2, not yet acted on.** Once an arm is a subtree, two arms
filling the same slot are exclusive *by construction* — there is one attribute, and the
second assignment would overwrite the first. That is a proof of exclusivity that needs no
output collision, which is exactly the gap §1 records for `.vacuum.i_vacuum_pumping` and
`.costs.i_cost_model` (exclusive by PROCESS's own `if`/`elif` while owning no field in
common). `check_arms_are_exclusive` is deliberately **not** relaxed on the strength of it
here — §12.2's real hazard is *partial* output overlap, and the check for that belongs on
consumers.

### B.4b What the tree does not yet buy — swapping

The tree gives a swap its **address**; two things are still missing before a swap means what
it should.

1. **No contract check.** A slot's contract is the set of variables downstream reads from it,
   so the check is structural, not nominal: swap the occupant, rebuild, assert no read lost
   its owner. That is §12.1's `free` postcondition ("`unowned_inputs` must not grow") and
   §12.2's "the check belongs on consumers, not producers" arriving at the same function from
   two directions. Roughly one function; not written.

2. **Several things one would want to swap are dials, not slots.** `physics.confinement_time`
   holds *one* occupant carrying `i_confinement_time=38`, where PROCESS has ~30 scalings. So
   "swap the confinement model" is editing a number *inside* an occupant — the thing a model
   tree exists to make impossible. §1 already lists this as six instances of "reads-set
   genuinely differs, kept static anyway", and §12.7 states the criterion they violate: *a
   branch that changes the reads-set is a different model and belongs in the model tree; one
   that changes only a value is settings.*

   28 of 147 registrations carry configuration as constructor arguments, and those 28 are a
   **mixture** of genuine dials and mis-filed model choices. Separating them is the
   model/settings boundary work, and it is the same question §12 approaches from the `free`
   side.

### B.5 Blast radius, measured

88 `COMMON` entries (63 bare classes, 25 instances), 10 switches, 147 declarations, 159
nodes. Node names are referenced almost entirely **as class objects**, not as strings:
there is exactly **one** hardcoded node-name string in the whole port,
`mda_harness.EXCLUDED_NODE_NAMES = ("DuctDiameterRootFind",)`, and it is matched with `in`
against `path_str()`, so it survives the rename to `vacuum.DuctDiameterRootFind`
unchanged. Every other `path_str()` call site compares names to each other, not to a
literal.

**[The survival half of that was wrong, and step 3 is what showed it.]** The *count* was
right — one hardcoded node-name string, plus one more in a test (`test_configuration.py`'s
`divertor_cycle`), and nothing else in the port compares a name to a literal. But the
string did **not** survive: the machine tree drops the class name from a node's name
altogether, so `"DuctDiameterRootFind"` matched nothing and the entry had to become
`"duct_diameter_root_find"`. Matching with `in` buys tolerance to a *prefix* changing, not
to the matched substring itself being replaced — and a rename that renames the whole name
is exactly the case a substring match cannot absorb. The failure would have been silent
(one extra node entering the comparison, no exception), which is why the step's gate is
the MDA harness's numbers being *identical* rather than the suite being green.

That is a small enough surface that Part B is a one-sitting change once the tree is written
— the work is in writing **35** modules' namespace classes (the number that contribute a
registered declaration today), not in chasing references.

### B.6 Explicitly not in scope

- **The settings tree and `materialise(models, settings)`** (§12.7). Part B's switch
  resolution is `materialise`-shaped on purpose, so the settings work later extends it
  rather than replacing it, but the settings tree itself waits.
- **The alternatives redesign** (§12.2) — exclusivity keyed on consumers rather than
  producers, and the partial-output-overlap hazard. Part B supplies an observation
  (§B.4), not a change.
- **`free(graph, paths)`** (§12.1) and everything downstream of it.

## Sequencing and the checks that gate each step

Part A first, whole, then Part B. They do not conflict — different namespaces — but Part A
touches 36 files and 2157 sites, so nothing else should be in flight while it lands.

Gates, in order:

1. cottax `FromExactly`/`Output` refuse a bare `Area`; the escape hatch is renamed from
   `Input`; `paths.py`'s docstring matches the surface. **All done — §A.2, §A.7.**
2. Per converted file: the side-by-side port comparison reports **N ports, all identical**.
3. After Part A: `grep -rn "lambda s:" functional_process --include='*.py'` → **0**;
   `pytest functional_process -q` → **3704 passed** unchanged; `run_mda_harness.py` → **499
   agreements / 34 disagreements / 0 ungrounded** unchanged. Part A is a renaming; every
   one of these numbers must be *identical*, not merely green. Any movement is a bug in the
   codemod, and the disk-cached MDA harness (17 s) is cheap enough to run per file.
4. After Part B: the same three, plus `len(GRAPH.definitions) == 159` and the same block
   count (**138 blocks, 14 driven**) — a node tree changes names, not structure.

`cd ~/jaxgraph && pytest` (**482 passed**, 3 skipped) gates both cottax-side changes.
