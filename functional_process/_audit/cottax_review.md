# cottax design review, from the PROCESS port

**What this is.** A usability and design review of `cottax` (`~/jaxgraph`, `src/cottax/`)
as exercised by its largest real client, `functional_process/`. Not a bug hunt. Nothing in
either repo was modified; no test was changed. Every claim below is anchored to a
`file:line` or to a measurement recorded in § Measurements.

**Standing caveat, applied throughout.** cottax is pre-Phase-4 and still settling
(`~/jaxgraph/CLAUDE.md` § "Rules that must not be quietly undone": *"Validate before
porting … cottax is a second opinion that must agree with jax-sn and jaxmdo"*). Every
recommendation below is therefore priced against that: a change is only worth proposing
if its payoff survives the possibility that the surface it touches moves again. § 7 ranks
them on exactly that basis, and § 7.4 lists the things I deliberately do **not**
recommend.

---

## 0. The three verdicts, stated

**Ease of use: good at the seams, bad at the surface.** The parts of cottax a client
touches *once per architecture* — `Graph`, `Blocking`, `Plan`/`rewrites`, `Schedule`,
`AbstractDriver` — are excellent, and the port uses them almost exactly as the design
documents imagine. The part a client touches *once per node* — the declaration surface in
`interfaces/pytree_namespace_module.py` — is the weakest thing in the package. 30 % of the
port's non-test model code (8 396 of 27 814 lines) sits inside node-declaration class
bodies, 87 % of those bodies are a single forwarding call to an already-pure function, and
97.8 % of the 1 347 `Input` declarations restate a name the parameter already carries.
That is not "some ceremony"; that is the dominant cost of using cottax at scale, and it is
fixable without touching the core.

**Restrictiveness: right almost everywhere, and the one place it hurts is the one place
the design already knows about.** I found exactly one refusal that cost the client real
modelling fidelity — the no-projection rule biting array-element ownership
(`graph.py:41-88`; recorded in `_audit/optimise_design.md:179-230`). Every other refusal I
traced either caught a genuine error or forced a decision that *should* be forced. The
`Blocking` nesting invariant, `Graph.problem_type`'s two-problem refusal, `Drive`'s
`issubclass(drives)` check and the `reads ∩ owns` refusal all earned their keep with
citable instances. The framework is **not** too rigid.

**Looseness: this is where cottax is actually weakest, and it is narrower than the
"missing producer" framing suggests.** cottax cannot and should not try to catch
"a variable no node owns" as a class — it has no way to know a boundary input is
unintended. But it *is* too loose in four specific, checkable places where it has all the
information it needs and declines to use it: single-output result binding
(`evaluate.py:52-69`), `GraphOp.apply` being public and skipping `check` (`plan.py:73`),
the `ConditionMap` seam dropping the problem's structure (`evaluate.py:135-178`), and the
absence of any hook to validate declared `VarPath`s against the caller's own data
structure at assembly time — even though `tools/pytree.py:167-179` already implements
exactly that check. Three of the four are one-field or one-argument fixes.

---

## 1. Where cottax earned its keep

These are not politeness; each is a place where the abstraction did work the client would
otherwise have had to do, and I can name the instance.

**1.1 `Graph.__check_init__`'s containment rules turned a whole bug class into a build
error.** `graph.py:41-88` refuses four things: duplicate ownership, an owner inside an
owner, a read enclosing an output, and an output inside a read. The last two are the ones
that matter, and the client hit them for real:
`_audit/optimise_design.md:198-204` records the exact `ValueError` raised when an
`Optimise` tried to own `.impurity_radiation.f_nd_impurity_electron_array[2]` while a
model read the enclosing array. Without that check the edge would have vanished silently
and the optimiser would have been differentiating a constant. `total_process.py`'s
`DefaultAspectRatio` policy (drop the producer when the iteration variable is active)
exists *because* this check refuses the alternative loudly.

**1.2 Structural implicitness is the design's best idea, and it works.** A node may not
read what it owns (`spec.py:70-76`), so a fixed point must be written as a minted copy and
a cycle is always at least two nodes. `ImplicitFunction`
(`interfaces/pytree_namespace_module.py:278-324`) makes that free: the client writes one
class with `Output(...)` and a `residual(...)` reading the same path, and gets a
`CallableNode` owning `^cond.<var>` plus a `RootFind` owning `<var>` — two nodes, in a
cycle, with the minting invisible. `models/vacuum.py:320-376` is 12 lines of declaration
for a genuine Newton problem. This is the single cleanest thing in the framework.

**1.3 The blocking/driver split is exactly the thesis, and the client cashed it.**
`Blocking` says what is solved together; the driver is supplied at `schedule_for` and
never in the graph (`evaluate.py:384-415`). `functional_process/mda.py:220-262` assigns
one driver per driven block *mechanically, by problem type* — nine lines of dispatch over
`blocking.problem_types` for 14 driven blocks. Compare what it replaces: PROCESS's
`Caller.call_models` re-running the entire pipeline up to ten times per evaluation and
checking idempotence. The measured payoff is recorded in
`_audit/next_steps.md` § 11.9: one MDF condition evaluation (a whole converged MDA) is
**0.53 ms against PROCESS's 43.6 ms**, and its Jacobian **0.71 ms against
`fcnvmc2`'s 741 ms — 82× and 1038×**. That is the framework's central claim, measured on
a real code.

**1.4 `AbstractDriver.drives` as an `AbstractClassVar` is the right shape.**
`evaluate.py:181-198` plus `Drive.__check_init__`'s `issubclass` (`evaluate.py:222`).
The client wrote four drivers (`core/solver/drivers.py:151,286,377,437`) and the file's
own docstring argues, correctly, that a driver belongs in the client and not in cottax —
`pyvmcon` is a PROCESS dependency, not a cottax one. cottax shipping **49 lines** of
drivers total (`drivers/optimistix.py`, 44 lines) and still having its driver seam used
unmodified by an SQP, a Picard, a seeded Newton and an SLSQP is strong evidence the seam
is at the right altitude.

**1.5 `ConditionMap` as a callable `eqx.Module` is the whole derivative story, and it
held.** `evaluate.py:135-178`. `jax.jacfwd(cm)` differentiates in the unknowns with the
context closed over as a `PathMap`. `core/solver/drivers.py:78-98` does exactly that and
nothing else; MDF then differentiates *through thirteen driven blocks* (12 `lax.while_loop`
Picards and one `optimistix` root find) with the same object, validated against a central
difference at worst `3.3e-06` (`_audit/next_steps.md` § 11.9). No autodiff opinion was
added to the seam and none was needed.

**1.6 The feedback loop between the two repos is working.** Three upstream changes landed
from this client and are in the tree today: `Feasibility` (`problem.py:143-236`, commit
`d1b54ec`), the `to_graph`/`node_and_names` mapping fix, and hierarchical node paths
(`interfaces/pytree_namespace_module.py:208-238,442-531`, commit `789df8b`, prototyped in
`_audit/switch_elimination_design.md` § 13 at +108/−19 lines in one file). `CutClosure`
now takes several cuts with an explicit `place` (`rewrites.py:284-285`), which is precisely
what `mda.py:122-142` needs to fold two cuts of one cycle into one `FixedPoint`. This is a
healthy relationship and it should keep working the same way.

**1.7 Names as places, not strings, paid off in an unexpected corner.** The client's
biggest single win from `VarPath` was not readability: it was that
`.impurity_radiation.f_nd_impurity_electron_array[i]` as a `SequenceKey` component turned
an apparent self-loop into an ordinary node, because indices 0-1 are written and 2-13 are
read (`_audit/unit_registry.md` row B; `models/physics/physics_B_composition.py:386`). No
`Cut`, no `FixedPoint`, no machinery — the naming scheme was fine-grained enough to say
what was true.

---

## 2. Where cottax is too rigid — and it is a short list

**2.1 No projection, and it costs real fidelity. (The one that matters.)**
`graph.py:41-88` forbids an owner/reader overlap rather than resolving it; `~/jaxgraph/
CLAUDE.md` § "Names" is explicit that projection "is what would make them legal" and is
out of scope. The consequence, measured in `_audit/optimise_design.md:179-230`: **an
`Optimise` cannot own an array element as long as any node reads the enclosing array, and
every real consumer of impurity fractions reads the array.** The client's workaround is
visible at `models/physics/radiation_power.py:620-650` — **fourteen** separate `Input`
ports for one 14-element PROCESS array, reassembled inside the body. 47 indexed ports
exist in the port today.

This is not a wrong decision; the refusal is honest and the alternative (owning a
14-vector of which 13 entries are not unknowns) is worse. But it is the one restriction
that made the client write something it did not want to write, and it will recur for any
CDS-shaped client — which is cottax's stated target.

**2.2 `Blocking`'s nesting invariant is right, and enforced two layers before its
consumer exists.** `blocking.py:90-143`: an interior must be the block minus *exactly one*
node, which must be declared and in a cycle. The invariant is genuinely elegant — it makes
`Blocking.problems[i]` derivable (`blocking.py:205-229`) rather than stored, so "which
problem is outer" has one representation and cannot disagree with itself. I would keep it.

But: `Schedule.steps` (`evaluate.py:342-351`) builds `Call`/`Drive` from
`blocking.subgraphs` and **never reads `blocking.inner`**. So today the invariant guards a
field nothing consumes, and the only thing it can do is refuse. `mdf.py:19-45` documents
the consequence precisely, and `test_mdf.py:108,130` pins both halves: `Blocking.nest`
records MDF correctly and `schedule_for` then refuses it. The gap is exactly one field's
type — `Drive.body` is a `Graph` (`evaluate.py:251-257`) where MDF needs a `Step` — and
the client closed it in **40 lines** (`mdf.py:406-445`, `MdfConditionMap`), with
`VmconDriver` handed the object unchanged and unable to tell the difference. That is the
measure of how narrow the gap is, and it is the strongest argument for closing it upstream.

**2.3 `Blocking.merge` is documented and does not exist.** `~/jaxgraph/CLAUDE.md`
lines 659-660 describe `merge(names)` at length as "what makes coarsening usable". There
is no `def merge` anywhere in `src/`. So coarsening beyond `Blocking.fused` requires
hand-writing the partition. Nothing in the client wants it today (it uses `Blocking.scc`
in 23 places and `Blocking.fused` in 3, only for rendering), so this is a documentation
defect rather than a felt restriction — but it is the shape of restriction a reader would
expect to be able to escape.

**2.4 `Optimise` has no bounds, and I think that is wrong.** `problem.py:71-98`. PROCESS
carries a box bound per iteration variable and hands VMCON `lbs`/`ubs` as distinct
arguments; re-expressing them as inequality constraints changes the QP subproblem and
therefore the iterates (`core/solver/drivers.py:504-512` argues this at length). The
client parked them on the driver as `VmconDriver.bounds`. That is defensible — a bound
*is* algorithm-adjacent — but it means two `Optimise` problems that are the same problem
can only be told apart by the driver assigned to them, which is exactly the inversion
`drives` exists to prevent. This is an open design question, not a defect; it should be
answered deliberately rather than by default.

**2.5 Two declaration surfaces, one of which has fallen behind.**
`interfaces/flat_namespace_module.py` (474 lines) and
`interfaces/pytree_namespace_module.py` (550 lines) duplicate `Output`,
`NodalDeclaration`, `ExplicitFunction`, `ImplicitFunction`, `FixedPointFunction` and
`node_and_names`. The pytree one gained hierarchical placement; the flat one's
`node_and_names` (`flat_namespace_module.py:281-323`) takes no `at` and cannot express a
nested namespace. A client picking the wrong surface silently gets the older feature set.

**2.6 What is *not* too rigid, stated so it does not get "fixed".** `Graph.problem_type`
refusing two problems in a block (`graph.py:432-460`) forced the client to make a real
modelling decision it had been avoiding, and `mda.py:122-142`'s comment records it
explicitly ("*the two cut variables of the density/fusion cycle are two unknowns of one
Picard iteration, not two nested loops*"). `Graph.driven`'s three-way refusal
(`graph.py:462-499`) produced the message that told the client its density/fusion cycle
needed a *second* cut — "*still cyclic with its problem(s) removed*" — which is how
`mda.CUTS` gained `.physics.fusden_alpha_total`. Neither should be relaxed.

---

## 3. Where cottax is too loose

The framing I was given — "should cottax be able to catch the missing-producer class?" —
overstates what is available to it. cottax has no schema for the caller's world; a
variable nothing owns is a legitimate boundary input by design (`graph.py:166-168`,
`~/jaxgraph/CLAUDE.md`: *"There is no `InputNode`"*), and that is right. **The port's
graph has 352 unowned inputs, every one of them read** (measured, § M4). No amount of
graph theory distinguishes the 345 intended ones from the 7 the client found by hand
(`_audit/boundary_inputs_audit.md`). That part is inherently the client's business.

But four narrower things *are* cottax's, and they are where it is genuinely too permissive.

**3.1 Single-output binding cannot distinguish a 1-tuple value from a 1-tuple mistake.**
`evaluate.py:52-69`: with one output, the whole result is bound. So a body that returns
`(x,)` for a single `Output` puts a tuple in the env, silently. I reproduced it (§ M6):
`env['.a.y']` came back as `(Array(6.),)` and the downstream `jnp` consumer swallowed it
without complaint. This is not hypothetical — `_audit/optimise_design.md:889-899` records
**six** live instances in `models/power_B_thermal_cryo.py` (lines 1208, 1267, 1308, 1371,
1443, 1488), invisible under `PicardDriver` (`ravel_pytree` flattens a 1-tuple happily)
and fatal the moment `Residualise` mints a `Compare` that subtracts. A seventh
(`ZTfInsideHalf`) is recorded separately in `_audit/next_steps.md` § 8.

The convention is correct — an output value may be any pytree, so a single output *must*
take the whole result. What is missing is that cottax knows more than it uses at the
declaration surface: `ImplicitFunction`/`FixedPointFunction` mint a `Compare` whose body
is `operator.sub`, so the residual's shape is knowable, and `NodalDeclaration` could
refuse a `step`/`residual` that returns a tuple whose length equals the declared output
count when that count is 1. Even a `strict_arity=True` flag on the declaration surface
would have caught seven bugs.

**3.2 `GraphOp.apply` is public and skips `check`.** `plan.py:73-78`: `check` is the
precondition, `apply` is the effect, `__call__` is check-then-apply. `apply` is the
natural-looking verb and the client used it, twice, on the hot path:
`functional_process/mda.py:142` (`FixedPointCut(tuple(cuts), place=place).apply(graph)`)
and `functional_process/sand.py:653` (`Delete(degenerate).apply(driven)`). Both bypass
every precondition the op declares. Nothing went wrong here, because `Blocking` catches
the downstream consequence — but the split exists precisely so that a precondition is
checked against the graph it will meet, and the public name that does not do that is the
one clients reach for. Rename it (`_apply`), or make `apply` check.

**3.3 The `ConditionMap` seam throws away the problem's structure.**
`evaluate.py:135-178` carries `body`, `unknowns`, `conditions`, `context` — a flat tuple
of condition names. `Drive` knows the problem (`evaluate.py:230-234`) and passes only the
map (`evaluate.py:288`). So a driver whose `drives = Optimise` **cannot ask which
condition is the objective**, even though cottax's own `AbstractDriver` docstring promises
exactly that pairing (`evaluate.py:186-190`). The client's workaround is two integer
fields on the driver (`core/solver/drivers.py:534-535`, `n_equality`/`n_inequality`; also `:185-186` on `SlsqpDriver`) set by
whoever assembles the `Drive` (`mda.py:246-256` reads them off `Optimise.equalities`/
`inequalities`). That is a positional contract on a seam whose entire purpose is to be
type-checked, and it can silently mislabel a constraint. **Minimal fix: one extra field on
`ConditionMap`, set in `Drive.condition_map` (`evaluate.py:265-278`).** Already scoped in
`_audit/optimise_design.md` § 8 item 1.

**3.4 Nothing validates a declared `VarPath` against the caller's data structure, though
cottax already implements the check.** A typo'd `Input(lambda s: s.physics.rmajorr)`
builds a perfectly legal graph with one more boundary input, one more sink, and no
complaint (reproduced, § M7). `tools/pytree.py:167-179` (`check_pytree_readable`) catches
it in one call given the CDS (§ M7). The port never calls it — `read_values`/`collapse`
have **zero uses** in `functional_process/` — and cottax offers no place to call it at
assembly. Run against the real graph, it flags 33 of 886 non-minted variables as not
resolving against `DataStructure()` (§ M5); all 33 look like the port's documented
*invented* names (e.g. `.stellarator.dlimit_ecrh`, called out as invented at
`models/stellarator/density_limits.py:12-15`), so the useful form is
"check, minus an explicit allowlist" — which turns the invented names into a declared list
and every future typo into an immediate failure.

**3.5 Two smaller ones, worth knowing about.** `read_values` drops any name whose value is
`None` (`tools/pytree.py:238`) — a real place holding `None` silently vanishes from the
returned dict, and `Schedule.run` then reports it as *"which nothing here has produced"*
(`evaluate.py:44-48`). And `Out.static` is a claim nothing verifies (`spec.py:48-51`,
acknowledged in cottax's own docs); the port never uses it, so this is latent rather than
live.

---

## 4. Ease of use: the declaration surface is the problem

This is the finding I would act on first, and it is entirely below the core.

**4.1 The numbers.** 185 node classes in `functional_process/models/**` (173
`ExplicitFunction`, 11 `FixedPointFunction`, 2 `ImplicitFunction`). Their class bodies
account for **8 396 of 27 814** non-test model lines — 30 %. **161 of 185 (87 %)** have a
body that is a *single forwarding call* to an already-pure function. The ports: 1 347
single-line `Input(lambda s: ...)` declarations, of which **1 317 (97.8 %)** have the
parameter name equal to the path's leaf; 540 `Output(lambda s: ...)`, of which **540
(100 %)** have the attribute name equal to the leaf. (§ M1, § M2.)

**4.2 What that means.** In the overwhelmingly dominant case the `lambda` carries exactly
one bit of information the surrounding code does not already have: *which area*. Everything
else is restated three times — once as the parameter name, once as the lambda's leaf, once
in the forwarding call's argument list. `models/buildings.py:883-1010` is the reductio:
**~130** consecutive lines of `name=Input(lambda s: s.buildings.name)`. Compare
`models/stellarator/density_limits.py:153-176`, the small end, where the wrapper is still
23 lines for a 6-argument function.

**4.3 Why the flat surface is not the answer.** `flat_namespace_module` already does
"parameter name *is* the variable name" — but it is a flat namespace, and PROCESS's names
are two-level (`.physics.rmajor`). The port correctly picked the pytree surface. What is
missing is the middle: a way to say the area once.

**4.4 The fix is small and additive.** Anything that lets a declaration supply a default
root — a class-level `AREA = 'buildings'` consumed by a bare `Input()`, or
`Input('buildings')` meaning "`.buildings.<parameter name>`" — collapses ~1 300 of 1 347
input declarations and all 540 output declarations to one token each. It changes no
semantics: `path_of` (`interfaces/pytree_namespace_module.py:65-79`) still produces the
same exact `VarPath`, and an explicit `lambda` still wins for the 2.2 % that differ. This
is sugar over sugar, in a module that is already explicitly *not* core
(`~/jaxgraph/CLAUDE.md`: *"Not re-exported … the core must never depend on it"*), so the
blast radius is one file and the pre-Phase-4 caveat barely applies.

**4.5 Three cottax features the largest client does not use at all.** Measured by grep
over `functional_process/`: `tags`/`Tagged` (0), `Out.static` (0 declarations),
instance-derived ports via overriding `_params` (0 — and
`_audit/switch_elimination_design.md` § 10.4 records that it *works* and is the answer to
a problem the client solved by subclassing instead, held back only because `_params` is
underscore-private), `Graph.from_pytree` (0), `Graph.reach` (0), `Graph.ancestors` (0),
`read_values`/`collapse` (0), `Plan.recipe` (0), `Plan.trunk` (0).

That last pair deserves its own sentence. **"The derivation is the value" has not earned
its keep in this client.** `Plan` appears four times, always as `(Plan(g) + op).graph`
discarded immediately (`sand.py:417`, `mdf.py:219`, `sand.py:497-505`), and twice the
client skips it entirely via `op.apply(g)` (§ 3.2). No plan is ever kept, queried or
replayed. I would not delete `Plan` on this evidence — the client is one data point and
`ComposedOp` genuinely needs the check-as-you-go loop (`plan.py:96-106`) — but the claim
that recording the derivation is what makes provenance a query is, so far, unpaid for.

---

## 5. Where the client is using it wrong

Kept separate deliberately; this is the half that conflates most easily with the above.

**5.1 `Switch`/`Alternative` is the client's, not cottax's, and the recorded "gaps" are
gaps in the client's own layer.** `functional_process/configuration.py` (250 lines) is
entirely local; `grep -rn "Switch\|Alternative" ~/jaxgraph/src/` returns nothing. So
`_audit/next_steps.md:57-104`'s four open items — one arm holding several literal values,
nested switches, "reads-set differs but kept static anyway", exclusivity proved only by
colliding output ownership, and two values selecting the identical node set — are the
client's design debt, not cottax's.

**Is the abstraction wrong, or incomplete?** Neither: it is in the right place. The module
docstring's argument (`configuration.py:8-27`) is correct and checkable — no PROCESS switch
is an iteration variable or a scan variable, so no switch can change between two
evaluations of one assembled graph, so graph structure is decided once by the caller. That
*is* cottax's position. Two of the four gaps are trivial (`value: int` → `values: tuple`;
an explicit `exclusive_by=` assertion beside the ownership-collision proof). The nested-switch
gap is real but small — it is a fold over a decision tree, not a new concept. None of them
argues for cottax growing a `Switch`.

**The one thing worth handing upstream** is the *check*, not the mechanism:
`Switch.check_arms_are_exclusive` (`configuration.py:178-208`) proves exclusivity by
showing the arms collide on an owned output — i.e. that `Graph.__check_init__` *would*
have raised. That is a genuinely reusable idea ("these two node sets cannot coexist in one
graph"), and it is one query on two `PathMap`s.

**5.2 The client discards provenance cottax offers for free.** § 3.2's `op.apply` calls
mean `Plan.ops` never records the port's two `FixedPointCut`s. `mda.CUTS`'s docstring
(`mda.py:48-95`) is a beautifully-argued *comment* where `Plan.ops` would have been a
value. Switching `mda.driven_graph` to accumulate a `Plan` and return `.graph` is a
three-line change that restores the check-then-apply discipline and makes the derivation
inspectable.

**5.3 `total_process.COMMON` is still a flat tuple though the upstream fix landed.**
`_audit/switch_elimination_design.md` § 13 recommends landing hierarchical node paths and
then converting one small subsystem. The cottax half is **in the tree**
(`interfaces/pytree_namespace_module.py:208-238,442-531`, commit `789df8b`); the client
half has not been done. `total_process.py:899` is still a flat 80-entry tuple and
`mda_harness.EXCLUDED_NODE_NAMES` still does substring matching on `path_str()`, which is
exactly the fragile idiom the change exists to replace. This is now purely client work.

**5.4 The port validates names only at harness time, not assembly time.** See § 3.4. The
client *does* close the loop — `mda_harness` grounds every unowned input from a real
`DataStructure` and reports 0 ungrounded / 21 errors — but only via a full harness run
against a converged reference. A `check_pytree_readable` call in `build_graph`
(`configuration.py:245-249`) would catch the same class in milliseconds at import.

---

## 6. Two structural observations about cottax itself

**6.1 Half of cottax is renderers.** Line counts of `src/cottax/`: visualization **4 121
(50 %)**, core + tools 2 806 (34 %), interfaces 1 080 (13 %), drivers **49 (0.6 %)**
(§ M3). For a package whose thesis is "explicit blocking plus autodiff-visible drivers",
shipping 49 lines of drivers and 4 121 lines of pictures is a striking allocation. I do not
think it is wrong — `xdsm_html.py` and `ragraph_dsm.py` are the artefacts that make the
structure *arguable* to a human, and the port uses both (`render_xdsm.py`) — but it should
be a conscious position rather than an accident, and it does mean a new client's first
impression of "what cottax is" is skewed.

**6.2 `~/jaxgraph/CLAUDE.md` has drifted from the tree in at least three places.**
`Blocking.merge` is documented at length (lines 659-660) and absent. The test count is
given as 307/318/363 in three places; the suite is now **456 passed, 2 skipped** (§ M8).
`interfaces.py` is described as dead code shadowed by the package — true, but the surface
the largest client actually uses (`pytree_namespace_module`) gets no § of its own. Since
this file is the design document, drift in it is more expensive than drift in code
comments.

---

## 7. Recommendations, ranked by value / cost

Cost is my estimate of the change in cottax; value is what the port's own record says it
buys. The pre-Phase-4 caveat means I recommend **nothing** that moves `spec.py`,
`graph.py` or `problem.py`'s finals.

### 7.1 Do these — high value, low cost, no core surface touched

| # | change | where | value | cost |
|---|---|---|---|---|
| 1 | **Default-area sugar on `Input`/`Output`** — a class-level root, or `Input('buildings')` meaning `.buildings.<param>` | `interfaces/pytree_namespace_module.py` | collapses **1 317/1 347** inputs and **540/540** outputs; addresses the dominant cost of using cottax (§ 4) | one file, additive, no semantics change |
| 2 | **Give `ConditionMap` the problem definition** — one field, set in `Drive.condition_map` | `evaluate.py:265-278` | removes the only positional contract on the driver seam; deletes `n_equality`/`n_inequality` from the client (§ 3.3) | ~5 lines |
| 3 | **Make `apply` check, or rename it `_apply`** | `plan.py:73-78` | closes a precondition bypass the client already walked into twice (§ 3.2) | ~3 lines + client fixups |
| 4 | **Promote `_params` to a documented hook** | `interfaces/pytree_namespace_module.py:262` | unblocks instance-derived ports, which `_audit/switch_elimination_design.md` § 10.4 measured as working and needed | rename + docstring |
| 5 | **Reconcile `CLAUDE.md` with the tree** (`merge`, test counts, the pytree surface) | `~/jaxgraph/CLAUDE.md` | the design doc is the product here (§ 6.2) | prose |

### 7.2 Do these next — high value, moderate cost

| # | change | where | value | cost |
|---|---|---|---|---|
| 6 | **`Drive.body` becomes a `Step`; `Schedule.steps` consumes `Blocking.inner`** | `evaluate.py:251-257,342-351` | this *is* MDF. The client proved the gap is 40 lines (`mdf.py:406-445`) and that an unmodified driver cannot tell the difference. Payoff already measured: 82×/1038× vs PROCESS | real change to `evaluate.py`; needs a nested-drivers story for `schedule_for` |
| 7 | **An assembly-time `check_against(cds)` on `to_graph`/`Graph`** with an explicit allowlist | new, over `tools/pytree.py:167-179` | catches typo'd ports at import; the check already exists and is called by nobody (§ 3.4, § M5/M7) | thin wrapper; the allowlist design is the only real question |
| 8 | **Arity strictness at the declaration surface** (opt-in) | `interfaces/pytree_namespace_module.py` | seven recorded 1-tuple bugs, six of them still latent (§ 3.1) | must not change `_bind`'s pytree-valued-output rule |
| 9 | **Implement `Blocking.merge`, or delete it from the docs** | `blocking.py` | resolves § 2.3 either way | small |

### 7.3 Decide, don't build

- **Bounds on `Optimise`** (§ 2.4). This is a "are bounds structure or algorithm?" question.
  The client's driver-field workaround is legitimate and there is no urgency; but leaving
  it undecided means every optimiser client re-derives the same answer.
- **The two declaration surfaces** (§ 2.5). Either keep `flat_namespace_module` in feature
  parity or say in writing that it is frozen and `pytree_namespace_module` is the surface.
  The current state — silent divergence — is the worst of the three.

### 7.4 Do **not** do these

- **Do not build projection** (§ 2.1). It is the one restriction that genuinely costs the
  client, and it is also the single largest change conceivable to `graph.py` — a trie,
  prefix-substitution rewires, assembly. cottax is pre-Phase-4; the payoff (one array in
  one client, and only when the optimiser owns an element of it) does not come close.
  Revisit only when a second client hits it, or when IMAS-shaped CDS ownership becomes
  routine. The "give the sub-part its own output port" answer, ugly as
  `radiation_power.py:620-650` is, is the right one for now.
- **Do not add a `Switch`/`Alternative` to cottax** (§ 5.1). Graph structure decided once
  by the caller is the right position and the client's local layer is in the right place.
- **Do not try to catch "missing producer" structurally** (§ 3, opening). cottax cannot
  know. Recommendation 7 is the checkable half; the rest is the client's audit.
- **Do not weaken `Graph.problem_type` or `Graph.driven`** (§ 2.6). Both refusals produced
  the messages that told the client what its model actually was.

---

## Measurements

All run in `~/miniconda/envs/process_port/bin/python`, with
`jax.config.update("jax_enable_x64", True)` before any array. No repo file was modified.

- **M1** — 1 347 single-line `Input(lambda s: s.<area>.<leaf>)` ports across
  `functional_process/models/**` (non-test); **1 317 (97.8 %)** have parameter name ==
  leaf. 540 `Output(...)`; **540 (100 %)** have attribute name == leaf. 22 distinct top-level
  areas.
- **M2** — 185 node classes (173 `ExplicitFunction`, 11 `FixedPointFunction`, 2
  `ImplicitFunction`); **8 396 of 27 814** non-test model lines lie inside their class
  bodies (30 %); **161/185 (87 %)** have a body that is one forwarding `return f(...)`.
  Largest: `BldgsSizes` (`models/buildings.py:883`), 273 lines.
- **M3** — `src/cottax/` = 8 187 lines. visualization 4 121 (50 %), core (`spec`, `graph`,
  `plan`, `problem`, `rewrites`, `blocking`, `evaluate`) + `tools` 2 806 (34 %), interfaces
  1 080 (13 %), drivers 49 (0.6 %), `__init__` 131.
- **M4** — `total_process.GRAPH`: 158 nodes, 901 variables, **352 unowned inputs**, 240
  sinks. `mda.driven_graph(GRAPH)`: 160 nodes, 14 cycles; `Blocking.scc` → 138 blocks, 14
  driven. All 352 unowned inputs have at least one reader (by construction), so all 352 are
  potential frozen-derivative boundaries.
- **M5** — of 886 non-minted variables in `GRAPH`, **33 do not resolve against
  `DataStructure()`** under `cottax.tools.pytree._stops_at`. All 33 are documented invented
  names (e.g. `.stellarator.dlimit_ecrh`, `.vacuum.d_duct`, `.physics.*_unclipped`).
- **M6** — a single-`Output` node returning `(x,)` binds a 1-tuple into the env with no
  error; a downstream `jnp` consumer swallows it. Reproduced on a 2-node toy graph.
- **M7** — a typo'd `Input(lambda s: s.a.yy)` builds a legal graph (one extra unowned
  input, one extra sink, `is_acyclic` true, no warning).
  `check_pytree_readable(cds, [v.keys for v in g.variables])` raises
  `KeyError: '.a.yy names no place in this data structure'`.
- **M8** — `cd ~/jaxgraph && pytest -q` → **456 passed, 2 skipped**, 19.3 s.
  (`~/jaxgraph/CLAUDE.md` states 307 / 318 / 363 in three places.)

---

## What I did not reach

Stated so the gaps are not mistaken for clean bills of health.

- **I did not run `pytest functional_process`.** Its state is taken from
  `_audit/next_steps.md`'s verified-state table (3 690 passed). Another agent was running
  JAX-heavy work concurrently and the constraint was to prefer small experiments.
- **I did not run any harness** (`run_mda_harness.py`, `run_sand_harness.py`,
  `run_mdf_harness.py`) or reproduce any convergence, timing or Jacobian number. Every
  performance figure quoted (82×, 1038×, 0.53 ms, iteration counts) is the client's own
  measurement, cited as such.
- **`visualization/` got a structural read only** — line counts, the module split, and
  what `render_xdsm.py` calls. I did not review `xdsm.py`, `xdsm_html.py`,
  `ragraph_dsm.py` or the new untracked `sequencing.py` for design quality, and I did not
  render anything. Since that is 50 % of the package, this is the largest hole in the
  review.
- **`primitive.py` (the parked tracing half)** was read only through
  `~/jaxgraph/CLAUDE.md`'s summary; I did not recover it from git. Its four recorded
  findings (glue collapse, `GlueMerge`, `Untraced`, `eval_shape` normalising statics) are
  reported, not verified.
- **`tools/path.py`, `tools/cache.py`, `tools/minting.py`, `tools/tags.py`** were read but
  not stress-tested; the `cached_query`-off-the-instance argument (equinox counting
  `__dict__` entries at flatten) is taken on the docs' word and `tests/test_cache.py`'s
  existence.
- **I did not review the harness layer** (`_harness/contracts.py`, `mda_harness.py`,
  `sand_harness.py`, ~2 000 lines) except where it touches a cottax API. Its Tier-1/Tier-2
  contract design is out of scope for a *framework* review but is where most of the
  evidence about cottax's value actually got generated.
- **I did not evaluate cottax against its two other stated clients** (`jax_sn`,
  `jaxmdo`). Every judgement here is from one client's vantage, and "cottax must agree
  with jax-sn and jaxmdo, not replace them" is a constraint I could not test.
- **`_audit/unit_registry.md` (92 KB) and `_audit/next_steps.md` (85 KB) were read
  selectively**, by section and by grep, not end to end.
