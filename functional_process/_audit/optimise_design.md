# The `Optimise` layer — design

---
kind: design
status: **built**. §1–§4 and §6 held up and are now implemented
        (`functional_process/sand.py`, `core/solver/drivers.py`,
        `sand_harness.py`, `run_sand_harness.py`, `test_sand.py`);
        §5's ladder has been run end to end. **§10 is the record of what was
        actually measured and supersedes every number in §3.3, §5 and §6.3.**
confidence: high throughout §1–§4, §6 and §10 (all measured on the real graph);
            §5's *reasoning* stands, its *numbers* are superseded by §10
---

> **Read §10 first if you want numbers.** §1–§9 are the design as argued before it was
> built, kept because the reasoning is still what the code does. Where a number appears
> in both, §10's is the measured one: three of §5's and §6.3's have moved (the graph has
> grown by 47 nodes since, and two defects §5 could only guess at have been found and
> fixed).

**Scope.** How the ~82 ported constraints (`functional_process/core/solver/constraints.py`),
the 16 ported objective branches (`functional_process/core/solver/objectives.py`) and
PROCESS's iteration variables become a real, solvable `cottax.problem.Optimise` over the
graph `functional_process/mda.py` already drives — and what drives *that*. This is
`_audit/next_steps.md` §8's final recommended step ("wiring constraints/objective into an
actual `Optimise` `DeclaredNode` in `total_process.py` at all (still not done)"), and
`CLAUDE.md`'s stated thesis for the whole rewrite.

**This is built.** §1–§4 and §6 are implemented (`functional_process/sand.py`,
`core/solver/drivers.py`, `sand_harness.py`, `run_sand_harness.py`, `test_sand.py`) and
§5's ladder has been run end to end; §7's stages 0–8 are done. What §1–§9 preserve is the
*reasoning*, because it is still what the code does — where a number appears in both,
§10's is the measured one.

**Reference run used throughout.** `SingleRun(stellarator_helias.IN.DAT, "vmcon")`, run to
convergence in-process, 46 VMCON iterations, "PROCESS found a feasible solution":

| | |
|---|---|
| `i_figure_merit` | `+6` (`COST_OF_ELECTRICITY`, positive ⇒ minimise) |
| `n_iteration_variables` | 8; `ixc = [2, 3, 4, 6, 10, 56, 59, 109]` |
| `n_equality_constraints` | 2 (set explicitly at `stellarator_helias.IN.DAT:12`) |
| `n_inequality_constraints` | 12 |
| `icc` | `[2, 16, 24, 8, 17, 18, 67, 82, 83, 62, 32, 34, 35, 65]` |
| `epsfcn` | `0.01` — PROCESS's finite-difference step is **1 %** |

Graph used throughout: `total_process.graph_for(Configuration({".physics.i_plasma_pedestal":
0}))` — the configuration `mda_harness.py` already establishes matches this input file — 99
nodes, 101 after `mda.driven_graph()`'s two `FixedPointCut`s.

---

## 0. The recommended shape, in one paragraph

Register the constraints and the objective as **ordinary `CallableNode`s owning minted
`^cond.*` variables**, and one **`Optimise` `DeclaredNode` owning the active iteration
variables' `VarPath`s** — all of which are already boundary inputs of the graph, so nothing
collides (§1, verified). Then **do not nest**: `Residualise` every `FixedPoint` in the graph
and `Combine` every remaining problem with the `Optimise` into one `^problem.sand`
(§6, verified end to end on the real graph). Answer that single block with a
**`VmconDriver` written in `functional_process/core/solver/drivers.py` beside `PicardDriver`,
backed by `pyvmcon`** — PROCESS's own SQP, already installed — fed `jax.jacfwd` of the
block's `ConditionMap` instead of PROCESS's 1 %-step finite differences (§4). Validate it in
three stages, of which the decisive one is a **per-cell comparison of the port's reduced
(Schur-complemented) Jacobian against PROCESS's own `Evaluators.fcnvmc2` Jacobian at
PROCESS's converged point** (§5, prototyped: agreement to `8e-14` in the best cells, with
every disagreement traceable to a named, already-tracked gap).

The single riskiest unknown — "does an `Optimise` block containing other driven blocks nest
cleanly?" — is settled and the answer is **no, and it cannot be made to today**; §6 gives
the three independent mechanisms that refuse it and why SAND is the correct answer rather
than a workaround.

---

## 1. Unknowns: `ITERATION_VARIABLES` / `numerics.ixc` → `Optimise.design`

### 1.1 The claim, checked

`CLAUDE.md` says: *"The integer ID and its `(module, name, array_index)` triple is exactly a
`VarPath`; the ID itself is throwaway indirection once names are structural."* Checked
against the real table (`process/core/solver/iteration_variables.py:20-40` for the
`IterationVariable` dataclass, `:43` onwards for the 83 entries) — **true, with two
qualifications**.

The mechanical mapping is:

```
VarPath((GetAttrKey(iv.module), GetAttrKey(iv.target_name or iv.name)))
       + (SequenceKey(iv.array_index),)   if iv.array_index is not None
```

`iv.module` is already a `DataStructure` attribute name — `load_iteration_variables`
(`iteration_variables.py:277`) does exactly `getattr(data, iteration_variable.module)`, and
`set_scaled_iteration_variable` (`:354`) does `getattr(module, target_name or name)`, so the
two-key path *is* PROCESS's own accessor spelled structurally. `target_name` needs no
special treatment beyond `target_name or name`: it is the storage location, `name` is the
human label, and cottax already separates those (`Path.path_str()` is the spelling, the
`Path` is the identity — `~/jaxgraph` `CLAUDE.md` §Names).

**Qualification 1 — bounds have no home on `Optimise`.** `cottax.problem.Optimise`
(`~/jaxgraph/src/cottax/problem.py:71-140`) has `objective`, `design`, `equalities`,
`inequalities` and nothing else. PROCESS carries per-variable bounds two ways: the
`IterationVariable`'s own `lower_bound`/`upper_bound` defaults, and `numerics.boundl`/`boundu`
as overridden per input file (`load_scaled_bounds`, `iteration_variables.py:406`), then hands
them to VMCON as separate `lbs`/`ubs` arguments (`solver.py:248-250`). Two options, and the
recommendation is the first:

- **(a) bounds as a driver field** — `VmconDriver(lower=..., upper=...)`, ordered to match
  `Drive.unknowns`. Keeps the problem cottax already has, and reproduces PROCESS exactly:
  VMCON treats bounds as bounds, not as constraints, and a bound-as-constraint changes the
  QP subproblem and therefore the iterates. Cost: the ordering is a positional contract
  between the driver instance and the block it is later assigned to, checkable only at
  assignment time (`mda.default_drivers` is where that check belongs — it already walks
  `blocking.problems`, `mda.py:96-112`).
- **(b) bounds as extra inequality conditions** — two more `^cond.*`-owning nodes per design
  variable (`l − x`, `x − u`), 16 extra conditions for this run. Structurally honest, needs
  no upstream change, but it is a *different optimisation problem* from the one PROCESS
  solves and will not reproduce its iterates. Worth keeping as the fallback if `Optimise`
  never grows bounds.

**Qualification 2 — scaling is not cosmetic.** `load_iteration_variables` sets
`scale[i] = 1/x_i` and `xcm[i] = x_i * scale[i]` (`iteration_variables.py:348-352`), so VMCON
sees every iteration variable at exactly `1.0` initially, and the bounds scaled to match. The
measured scale vector for this run is `[0.1818, 0.05, 0.1429, 5e-21, 1.0, 0.02857, 1.4286,
10.0]` — a spread of 20 orders of magnitude, entirely from `nd_plasma_electrons_vol_avg` ≈
1.7e20. **A driver that ignores this will not converge on the same path PROCESS does, and may
not converge at all.** The scaling belongs on the driver (it is an algorithm choice, not
structure) and must be derived from the *starting* values, exactly as PROCESS does — not from
a fixed table.

### 1.2 What this run activates, and whether the graph has it

Verified. All 8 active iteration variables resolve to `VarPath`s that are **in the graph, and
all 8 are boundary inputs** (`graph.unowned_inputs`) — not owned by any node:

| ixc | `VarPath` | in graph |
|---|---|---|
| 2 | `.physics.b_plasma_toroidal_on_axis` | boundary input |
| 3 | `.physics.rmajor` | boundary input |
| 4 | `.physics.temp_plasma_electron_vol_avg_kev` | boundary input |
| 6 | `.physics.nd_plasma_electrons_vol_avg` | boundary input |
| 10 | `.physics.hfact` | boundary input |
| 56 | `.tfcoil.t_tf_superconductor_quench` | boundary input |
| 59 | `.tfcoil.f_a_tf_turn_cable_copper` | boundary input |
| 109 | `.physics.f_nd_alpha_thermal_electron` | boundary input |

This is the best possible answer: an `Optimise` node owning them introduces **no
dual-ownership conflict**, because a boundary input has no producer for it to fight with.
Registering the `Optimise` turns eight free inputs into eight owned variables and nothing
else changes.

It also **closes `objectives.md`'s open question 1** ("`.physics.rmajor` and
`.physics.b_plasma_toroidal_on_axis` have no `Output` producer — genuine iteration variable,
or a porting gap?"). Both are ixc entries of this run. Genuine free design variables; not a
hole.

Across the whole 83-entry table against this graph: **7 owned, 42 boundary inputs, 34 absent.**
The 7 owned ones are the ones that would collide if activated:

| ixc | `VarPath` | owner |
|---|---|---|
| 1 | `.physics.aspect` | `DefaultAspectRatio` |
| 12 | `.tfcoil.j_tf_coil_full_area` | `CoilsSummaryVariables` |
| 13 | `.build.dr_tf_inboard` | `CoilRadialThickness` |
| 29 | `.build.dr_bore` | `Build` |
| 60 | `.tfcoil.c_tf_turn` | `WindingPackTotalSizePost` |
| 140 | `.tfcoil.dr_tf_wp_with_insulation` | `WindingPackTotalSizePost` |
| 172 | `.tfcoil.dx_tf_side_case_min` | `CoilCasing` |

ixc 1 is not an accident: `total_process.py`'s own `DefaultAspectRatio` comment already says
that node exists *only* for the `1 not in data.numerics.ixc` case. That is the general
policy, made explicit: **a node that produces a value PROCESS would otherwise let the solver
pick must be dropped from the graph when the corresponding ixc is active** — the same
assembly-time selection `configuration.py` already performs for topology switches, keyed on
`ixc` instead of on a switch value. `Graph.__check_init__` will refuse the alternative loudly,
which is the right failure mode. For the other six this is untested (none is active in this
run); it is the one part of §1 that is *inferred*, not verified.

### 1.3 The `target_name`/`array_index` case (IDs 125–136) — verified hard, not papered over

`CLAUDE.md`'s Difficulties section flags these; here is exactly what breaks.

IDs 125–136 (`iteration_variables.py:112-201`) address one element of
`.impurity_radiation.f_nd_impurity_electron_array` — a **numpy array of shape (14,)**, verified
against `DataStructure()`. Three separate findings, all measured:

1. **Reading works.** `cottax.tools.pytree.get_at(data, (GetAttrKey('impurity_radiation'),
   GetAttrKey('f_nd_impurity_electron_array'), SequenceKey(2)))` returns the element. (With
   `FlattenedIndexKey` it raises `KeyError` — `SequenceKey` is the right key kind here.)
2. **Writing back into a `DataStructure` does not.** `check_pytree_writeable` refuses with
   *"resolves but is not a place this pytree flattens to, so nothing could be written back
   there"* — a numpy array is a jax pytree **leaf**, so `[2]` is inside a leaf, not a slot the
   cut reaches. This does **not** block a run (cottax's `Env` is a plain `dict[VarPath, Any]`,
   `evaluate.py:30-37`, and `env[var] = value` needs no pytree at all) but it does block
   `cottax.tools.pytree.collapse` — i.e. handing a solved answer back into a `DataStructure`.
3. **The graph refuses the overlap outright.** Verified directly:

   ```
   ValueError: output VarPath(.impurity_radiation.f_nd_impurity_electron_array[2]) of
   NodePath(.Opt) lies inside VarPath(.impurity_radiation.f_nd_impurity_electron_array),
   read by NodePath(.Reader) -- that read would silently miss this output
   ```

   This is `Graph.__check_init__`'s containment rule (`~/jaxgraph` `CLAUDE.md` §The graph),
   and it is *correct*: any model that reads the whole impurity array would silently miss the
   element the optimiser owns.

**Consequence, stated plainly: an `Optimise` cannot own an array element as long as any node
reads the enclosing array, and every real consumer of impurity fractions reads the array.**
The three honest ways out, none free:

- **Split the array into per-element `VarPath`s at the port level** — each impurity fraction
  its own variable, assembled into an array by an explicit node. Correct, mechanical,
  and touches every impurity-reading model. This is the one that actually fits cottax
  ("Want a sub-part addressable? Give it its own output port", `~/jaxgraph` `CLAUDE.md`
  §Rules).
- **Own the whole array and write the element inside a node** — an `Optimise` owning
  `f_nd_impurity_electron_array` with the other 13 elements pinned. Expressible today, but it
  hands the driver a 14-vector of which 13 entries are not unknowns, which is a lie about the
  problem's dimension and will confuse any bounds/scaling.
- **Projection in cottax** — the feature `~/jaxgraph` `CLAUDE.md` names as what "would make
  them legal". Not implemented, explicitly out of scope there.

None of IDs 125–136 is active in `stellarator_helias.IN.DAT`, so **this is not on the critical
path for the first `Optimise`** — but it must not be quietly assumed away, and the first
recommendation above is the one to plan for.

---

## 2. Constraints → the `Optimise`'s conditions

### 2.1 §6's conclusion still holds, and gets sharper

`next_steps.md` §6 argued constraints need no node — "an ordinary `Compare(...)` node over
outputs that are already ordinary ported model nodes, or … not even a node, a bare residual
read". `constraints.md`'s module docstring refines that: the overwhelming majority are the
bare-residual shape; constraint 1 is the one confirmed `Compare` (`constraints.md:18-66`).

With a real `Optimise` in hand the conclusion **holds but needs one correction**: a constraint
*does* need a node, just not an interesting one.

`Optimise.equalities` and `Optimise.inequalities` are `tuple[In, ...]` (`problem.py:79-80`) —
`In` is a read of one **`VarPath`**. A problem reads conditions; it never computes them
(`~/jaxgraph` `CLAUDE.md`: *"a bodyless `DeclaredNode` like `Feasibility` reads pre-computed
residual values"*, echoed in `vacuum.py:490-491`). So every active constraint must own a
variable. Since a constraint residual has no `DataStructure` field
(`constraints.md`'s module docstring says so explicitly — `ConstraintManager.evaluate_constraint`
returns it straight to the solver), that variable is **minted**: `^cond.constraints.c<N>`,
using `MintKey('cond')` — the same namespace `Compare`/`Residualise` already open, and the
namespace `~/jaxgraph` `CLAUDE.md` says is deliberately the general word because "what a
`Problem` *reads* is a condition whatever shape it has, which with `Optimise` in the picture
covers an objective as well".

So the wiring is still thin, but it is: **one `CallableNode` per active constraint**, whose
`fn` is the already-ported `constraint_<N>` with its static switch arguments bound at
assembly time (`functools.partial`), whose `inputs` are the remaining parameters resolved to
`VarPath`s, and whose single output is `^cond.constraints.c<N>`.

**Which of the four returned numbers is the condition:** the **normalised residual**, index 1
of `(residual, normalised_residual, constraint_value, constraint_bound)`. Verified against
PROCESS's own assembly: `constraint_eqns` (`process/core/solver/constraints.py:1958-2013`)
takes `result.normalised_residual` and appends **`-tmp_cc`** (`:2007`, comment: *"Reverse the
sign so it works as an inequality constraint (cc(i) > 0)"*), and VMCON's convention is
`i(x) >= 0` (`pyvmcon.problem.AbstractProblem` docstring). Therefore:

> PROCESS's `-normalised_residual >= 0` **is exactly** cottax's `g(x) <= 0`
> (`problem.py:80`) with `g = normalised_residual`, **unflipped**.

Checked numerically at the converged point: every one of the 12 inequality entries of
`constraint_eqns`'s `cc` is `>= 0`, so every `normalised_residual` is `<= 0`. For `leq(value,
bound)` (`constraints.py`'s ported helper) `normalised_residual = value/bound − 1 <= 0 ⟺
value <= bound`; for `geq`, `1 − value/bound <= 0 ⟺ value >= bound`. The convention lines up
with no adapter. **Do not use index 0 (`residual`)** — it is unnormalised and its scale spans
1e-10 to 9e7 in this run (measured), which is exactly the conditioning VMCON's normalisation
exists to remove.

### 2.2 The positional equality/inequality split

PROCESS's split is positional in `numerics.icc`, and this run proves it is not merely a
convention — it is *load-bearing and user-chosen*. `stellarator_helias.IN.DAT:12` sets
`n_equality_constraints = 2`; `set_active_constraints` (`process/core/init.py:1277-1294`)
then derives `n_inequality_constraints = 14 − 2`. The **first two entries of `icc`** are the
equalities: constraint 2 (global power balance, genuinely an equality) and **constraint 16
(“Net electric power lower limit”, whose ported body is a `geq`)**. Verified at the converged
point: constraint 16's `cc` is `−3.78e-09` — driven to *equality*, not merely satisfied.

**What that becomes here:** the split is a property of the *assembly*, not of the constraint
function, and cottax already models it that way — `Optimise` has two separate tuples and the
constraint node is identical in both cases. So:

- the ported `constraint_<N>` stays exactly as it is (`leq`/`geq`/`eq` unchanged);
- assembly reads `icc[:meq]` into `equalities` and `icc[meq:]` into `inequalities`;
- **the ported body's own `leq`/`geq`/`eq` choice is not authority for which tuple it lands
  in.** Constraint 16 is the counterexample; it must be documented at the assembly site so
  nobody "fixes" it later.

This is strictly better than PROCESS: an ordering convention becomes a typed field, and
`Optimise.__repr__` (`problem.py:133-140`) prints which is which.

### 2.3 When several constraint IDs collapse into one `Compare`

`CLAUDE.md`'s Difficulties section asks for a policy. Recommendation, with the reasoning:

**Default: one node per active constraint ID. Never merge.** cottax's own rule is that
"Several `Compare`s is how a caller says the residuals *are* separate — separately prunable
sinks" (`~/jaxgraph` `CLAUDE.md` §Rewrites). PROCESS's IDs are separately prunable by
construction: `numerics.icc` selects an arbitrary subset per run, so two IDs that happen to
share operands today are still independently activatable tomorrow. Merging would bake one
run's `icc` into the graph, which is the same mistake as an `OutputNode`.

**The one case where `Compare` is right** is the shape it was built for: a constraint whose
body *re-derives* a quantity and compares it to a stored field. Constraint 1 is the only
confirmed instance (`constraints.md:24-28`), and even there the merge is not across IDs — it
is one `Compare` with one pair, which is just the node above wearing cottax's own body. Use
`Compare` when and only when the re-derived side is *already a node's output*, because then
`Compare` removes a duplicated computation; otherwise the re-derivation lives in the
constraint's `fn` and a plain `CallableNode` is correct. `constraints.md:1616-1617` already
flags one future candidate ("if ported as its own node elsewhere, this branch becomes
`Compare`-shaped").

**Registration is unconditional; activation is not.** §6's *"Likely register unconditionally,
not behind a `Switch`"* is right for the *function*, but a node that owns `^cond.constraints.cN`
and is in no `Optimise` is a **prunable sink**, not free: it runs on every condition evaluation
inside the `Drive` block for nothing. Since `icc` is fixed for a whole solve (same argument
`configuration.py:8-19` makes for switches — no constraint ID is an iteration or scan
variable), **build constraint nodes only for the active `icc`**, at assembly time. The ported
function set stays complete and unconditional; only the *nodes* are per-run.

### 2.4 What the graph can actually support today — measured

**Superseded — see §10.7.** This section classified each of the 14 active constraints as
LIVE (at least one argument owned by a node reachable from the design variables) or INERT,
and found **12 of 14 live**, with **c16** and **c24** inert for want of a producer. Both
gaps are now closed (`PlantElectricProductionReactor`, `StellaratorBetaAndStoredEnergy`),
all 14 are live, and `sand.constraint_nodes` **raises** on any active `icc` entry it cannot
assemble rather than silently dropping it — an `Optimise` over 12 of PROCESS's 14
constraints is a *different problem*, so comparing its answer to PROCESS's would be
meaningless.

What survives from the classification, and is still the right test to apply to a new
constraint: a bound that shows as "not in graph" (`pflux_fw_neutron_max_mw`,
`sig_tf_wp_max`, `f_t_alpha_energy_confinement_min`, …) is **not** a gap — it is a user
input, and becomes an ordinary boundary input the moment a constraint node reads it. An
argument that is *constant with respect to the design* is the real gap.

---

## 3. Objective

### 3.1 `CLAUDE.md`'s "query, not a node" — confirmed in spirit, refuted in letter

`CLAUDE.md` and `next_steps.md` §6 both say the objective is *"a per-run selection of which
existing output is 'wanted' … a `Graph.prune`-style query"*, and `objectives.py`'s module
docstring is built on that. Checked against `problem.py`'s actual API:

`Optimise.objective` is a single **`In`**, i.e. one `VarPath` (`problem.py:77`), and
`Optimise.minimised` returns `self.objective.var` (`:91-92`). So the selection *is* a query —
which variable — exactly as claimed. **But the sixteen `objective_metric_<id>` functions are
not selections; they are arithmetic** (`0.2 * rmajor`, `coe / 100`, `0.95*(rmajor/9) −
0.05*(t_burn/7200)`), and arithmetic needs a body. Only `objective_metric_18`
(`NULL_FIGURE_OF_MERIT`, returns `1.0`) is bodyless, and even it needs an owner.

So the accurate statement, which the doc should replace §6's phrasing with:

> **Which** `i_figure_merit` is a per-run query; **the metric itself is one ordinary
> `CallableNode`**, minted per run from `OBJECTIVE_METRICS[figure_of_merit]`, owning one
> minted variable (`^cond.numerics.objf`) that the `Optimise` reads as its `objective` `In`.

This is not a retreat from "no `OutputNode`": the node is *not in the graph until an
`Optimise` is assembled*, and a different `i_figure_merit` mints a different node. The
structure is still per-query; it is just that the query's answer is a node rather than a name.
`Graph.prune(wanted)` (`graph.py:510-524`) is then genuinely useful downstream — pruning to
`{objective, every active condition}` is what drops the model nodes no active constraint
needs.

### 3.2 Sign

`objective_function` computes `objective_sign = np.sign(i_figure_merit)` (`objectives.py:54`)
and returns `objective_sign * objective_metric` (`:105`); the ported functions deliberately
return the unsigned metric (`functional_process/core/solver/objectives.py:26-33`).

**Where the sign is applied: in the objective node's `fn`, at assembly time, as a constant
multiplier.** `fn = lambda *a: sign * OBJECTIVE_METRICS[merit](*a)`. Not on the driver (a
driver that silently maximises is a driver whose `drives` claim is a lie), not on the
`Optimise` (it has no field for it, and adding one would put a per-run scalar in the
structure). `Optimise`'s contract is minimise — `problem.py:133-140` prints "minimises" — so
the graph should present something to minimise.

### 3.3 What is computable today — measured, and it is a problem

**Superseded — closed.** This section recorded that this run's own `i_figure_merit = 6`
(`COST_OF_ELECTRICITY`) was **not computable at all**, because `.costs.coe` had no producer
anywhere in the ported graph, and §5's ladder was written around that fact. `costs.py` is
now ported and registered and `CostOfElectricity` owns `.costs.coe`; see
`next_steps.md` §9. Objectives 4 and 7 (`.tfcoil.tfcmw`, `cdirt`/`concost`) were the other
two in that state; `cdirt`/`concost` now have producers too.

The distinction the section drew is still worth keeping when a new objective is wired: an
objective whose arguments are all **boundary inputs** (1 `MAJOR_RADIUS` is literally
`0.2 × design`, and so are 8, 10, 11, 14, 15, 16, 17) is a function of the design variables
with no model in between — legitimate, but it exercises none of the graph.

---

## 4. The driver

### 4.1 It belongs here, not upstream — same reasoning as `PicardDriver`, and it survives scrutiny

`drivers.py:1-16` states why `PicardDriver` was written in this repo: unlike the
`Feasibility`/`to_graph` gaps (real core-library holes, fixed upstream), *"a Picard driver is
exactly the kind of generic, swappable solver choice `AbstractDriver` exists to make
pluggable, not core graph machinery"*. `AbstractDriver.drives`'s own docstring
(`~/jaxgraph/src/cottax/evaluate.py:186-190`) names all three pairings — *"a Newton drives
`RootFind`, a Picard `FixedPoint`, an optimiser `Optimise`"* — and cottax ships only the
first (`~/jaxgraph/src/cottax/drivers/optimistix.py:14`).

The same reasoning applies, and more strongly: an SQP is a much larger algorithm choice than a
Picard, and the sensible backing solver (`pyvmcon`) is a **PROCESS** dependency, not a cottax
one. **Write `VmconDriver` in `functional_process/core/solver/drivers.py`, beside
`PicardDriver`.** Upstream stays free of a numpy SQP and of PROCESS's solver.

Two things this driver needs that `NewtonDriver`/`PicardDriver` do not, and **only one of them
can be solved locally**:

- **It must split its conditions into objective / equalities / inequalities, and
  `ConditionMap` cannot tell it.** `ConditionMap` carries exactly `body`, `unknowns`,
  `conditions`, `context` (`evaluate.py:135-178`) — a flat tuple of condition names, with no
  type information and no reference to the problem. `Drive` knows (`Drive.problem`,
  `Drive.subgraph[problem]`) but passes only the `ConditionMap` (`evaluate.py:288`). The
  ordering *is* reliable — `Drive.conditions` is `subgraph[problem].reads`
  (`evaluate.py:246-249`) and `Optimise.inputs` is `(objective, *equalities, *inequalities)`
  (`problem.py:82-84`) — so the split is recoverable **from counts alone**. Local answer:
  `VmconDriver(n_equality=…, n_inequality=…)`, constructed by `mda.default_drivers`, which
  already has the problem node in hand (`mda.py:96-112`). Upstream answer, better: see §8.
- **Bounds and scaling**, per §1.1 — driver fields, derived from the starting guess.

### 4.2 Backing solver: what is actually installed

Checked in `process_port`:

| package | version | constrained optimisation? |
|---|---|---|
| `pyvmcon` | 2.4.2 | **yes** — SQP, `eq` + `ineq`, box bounds, caller-supplied gradients |
| `scipy` | 1.18.0 | **yes** — `SLSQP`, `trust-constr`, `Bounds`/`NonlinearConstraint` |
| `cvxpy` / `clarabel` / `osqp` | 1.7.5 / 0.11.1 / 1.1.3 | present (pyvmcon's QP subproblem backend) |
| `optimistix` | 0.1.0 | **no** — `minimise`/`least_squares`/`root_find`/`fixed_point` only; no `constraint=` anywhere in its public API |
| `jaxopt`, `nlopt`, `cyipopt` | — | not installed |

**Recommendation: `pyvmcon`.** Reasons, in order of weight:

1. **It is the same algorithm PROCESS uses**, wired identically (`process/core/solver/solver.py:192`
   builds a `VmconProblem`; `:246-260` calls `pyvmcon.solve` with bounds, `max_iter`,
   `epsilon`, the CLARABEL QP backend). With the algorithm held fixed, any difference between
   the port's answer and PROCESS's isolates to the *model and the gradients* — which is the
   only comparison worth making. A different SQP would confound the two.
2. **Its interface is exactly the seam autodiff plugs into.** `AbstractProblem.__call__(x) ->
   Result(f, df, eq, deq, ie, die)` demands the caller supply `df`, `deq`, `die`. PROCESS fills
   them from `Evaluators.fcnvmc2`'s finite differences (`evaluators.py:85-148`); the port fills
   them from one `jax.jacfwd` of the `ConditionMap`. That substitution, at that seam, *is* the
   thesis of the rewrite, expressed as a 20-line class.
3. Already installed, already a PROCESS dependency, no new pin.

**What it costs**, stated honestly:

- **It is NumPy, so the `Drive` step stops being traceable.** `PicardDriver`/`NewtonDriver` are
  `jax.lax.while_loop`/`optimistix` and stay inside one traced program; `mda_harness.py`'s own
  docstring already leans on that (*"`Schedule` runs one JIT-traced program, not node-by-node"*).
  A `VmconDriver` runs a host-side loop that calls back into jitted condition/Jacobian
  evaluations. At the *outermost* block nothing differentiates through it, so this is
  acceptable — but it must be written down, because it means the whole schedule is no longer
  one trace, and a `VmconDriver` nested inside anything would be wrong.
- **Sign flip at the boundary**: VMCON wants `ie >= 0`, cottax's `Optimise` says `g <= 0`, so
  the driver passes `-g`. One line, but exactly the kind of line that is wrong for a year if
  it is not tested; §5's stage 1 catches it.
- Convergence failures arrive as exceptions (`VMCONConvergenceException` and friends,
  `solver.py:262-272`), not as a status flag. `PicardDriver` returns whatever it has after
  `max_iter`; `VmconDriver` should follow `solver.py`'s own pattern and return `e.x` rather
  than propagating, with the failure reported out of band.

**Second opinion: `scipy.optimize.minimize(method="SLSQP")`**, same `jacfwd`-supplied
gradients, as a swappable alternative — cheap, and the point of `AbstractDriver` being a
per-block choice. Worth having precisely because "PROCESS's answer" and "the stated problem's
answer" are different claims (§5.4) and a second solver separates them.

**A traceable option, for later, not now:** an augmented-Lagrangian or `relu`-penalty merit
function driven by `optimistix.BFGS`. Fully jittable and differentiable-through, but it is a
different algorithm with weaker constraint guarantees, so it cannot be the reference
implementation. `problem.py:163-170` records the same reduction as the standard move for
`Feasibility`.

### 4.3 `Feasibility` (i.e. `DuctFeasibility`)

`Feasibility` (`problem.py:143-236`) is `Optimise` minus the objective. Two ways to answer it,
both cheap once `VmconDriver` exists:

- **fold it in** — `Feasibility + Optimise → Optimise` (`problem.py:111-117`), which is what
  `vacuum.py:484-491` already anticipates and what `Feasibility`'s own docstring calls "the
  main reason to reach for it". Nothing new needed at all: `Combine` in §6 already does this.
- **drive it standalone** — a `FeasibilityDriver` that is `VmconDriver` with a constant
  objective, or the least-squares merit reduction `problem.py:163-170` describes.

Either way `DuctFeasibility`'s "unvalidated by design" status (`next_steps.md` §8) is lifted by
the same work, but note what that validation *can* say: `solve_duct_geometry` is "one specific
heuristic (shrink by 10 % until it fits), not 'any feasible point'", so the check is
"the driver's answer is feasible", never "the driver's answer is PROCESS's answer".

### 4.4 Batching the whole MDA — measured, and it works

Raised as a side question and worth recording, because the answer is not obvious from the
source. **Nothing in SAND form is conditional per evaluation**: every switch is resolved
at graph-assembly time (`configuration.py`'s whole argument), so one assembled graph is a
fixed dataflow and `jax.vmap` applies to it. Confirmed on the real graph, both forms:

| | single, jitted | B=8 | B=64 |
|---|---|---|---|
| SAND `ConditionMap` (acyclic, 62-node body) | 0.401 ms | 0.153 ms/pt | **0.070 ms/pt** |
| SAND batched `jacfwd` (22 × 17) | 0.539 ms | 0.283 ms/Jac | **0.280 ms/Jac** |
| **driven MDA `Schedule`** (9 driven blocks, Newton/Picard) | 0.527 ms | 0.070 ms/pt | **0.0115 ms/pt** |

The third row is the important one: it is the *driven* schedule, with all 9
`NewtonDriver`/`PicardDriver` blocks and their `lax.while_loop`s inside, not the
residualised acyclic form — **~46× throughput at B=64**. Compile time barely moves with
batch size (2.22 s → 2.80 s), confirming one trace regardless of `B`.

**Guarded against dead-code elimination**, which would have made these numbers
meaningless: the timed function stacks *all 573* produced variables, and 169 of them
genuinely vary across the batch. The 404 that do not are simply not downstream of the
swept variable, so XLA hoists them out of the batch — a real saving for a single-variable
sweep, not a measurement artifact.

**Where this is worth using: `Scan`, not the optimiser.** `process/core/scan.py`
re-solves the entire system per scan point with nothing reused between points
(`CLAUDE.md`'s architecture section says so explicitly). That is embarrassingly parallel
and is exactly what `vmap` collapses into one trace. An SQP is inherently sequential and
gains nothing directly — though batched multi-start, or batched finite-difference
validation of the AD Jacobian, both fit.

**Caveats, none of them measured away:**

- **`vmap` over `while_loop` runs until *every* batch element's predicate is false**, so a
  batch costs its worst-case element, not its average. Every point timed here starts near
  the converged solution and needs a similar iteration count; a scan spanning genuinely
  different convergence difficulty has not been tried and would not scale this cleanly.
- **`lax.cond` under `vmap` becomes `select`, executing both branches.**
  `_solve_vacuum_pumping_old` (`models/vacuum.py:587`) has one per gas species, so
  `VacuumOld` pays ~2x under batching. Correct, not free.
- Measured on **CPU** (this env has no CUDA jaxlib, see `CLAUDE.md`). A GPU would change
  the shape of these numbers substantially, in the batched form's favour.
- 24 of the 573 outputs were non-finite when this was measured, from the placeholder-seeded
  island inputs §5.2's `c32` note describes — batching neither introduced nor fixed them,
  and closing that exclusion removed them.


---

## 5. Validation

The MDA harness's model is `mda_harness.py`: run PROCESS in-process, seed the port from
PROCESS's own converged values, diff. The `Optimise` layer's analogue is **not** one harness
but a ladder of three, because they establish three genuinely different things. It is built
as `sand_harness.py`/`run_sand_harness.py`, beside the existing two harnesses.

### 5.1 Stage A — conditions at the reference point (cheapest, strictly stronger than what exists)

Seed the design variables with PROCESS's converged values, run the MDA schedule, then evaluate
every `^cond.*` the assembled `Optimise` reads and diff against PROCESS's own
`constraint_eqns` output at the same point.

**Prototyped and run.** The `Drive`'s `ConditionMap` evaluated in **0.04 s**. Against PROCESS's
`-cc` (its normalised residuals) for the 12 live constraints:

```
        port                PROCESS
c8    -4.867613e-01      -4.867613e-01     exact
c17   -5.101639e-01      -5.101639e-01     exact
c18   -6.818613e-01      -6.818613e-01     exact
c67   -5.260039e-01      -5.260039e-01     exact
c82   -5.662977e-01      -5.662977e-01     exact
c83   -2.075736e-09      -2.075736e-09     exact
c34   -8.602314e-01      -8.602314e-01     exact
c35   -1.936917e-07      -1.936917e-07     exact
c65   -9.758968e-01      -9.758968e-01     exact
c2    +9.035892e-03      -7.375414e-10     off  (see below)
c62   -1.056535e-01      -1.191346e-01     off  (~11 %)
c32   -6.884017671e-01      -6.884017671e-01     exact
```

Ten of twelve agree to printed precision. `c2` and `c62` are the **already-tracked**
`ConfinementTime`/`fusrat` disagreements (`next_steps.md` §8 "Still open") surfacing in a new
place — c2 is a power balance and c62 a confinement-time ratio, so this is consistent, not
new. **`c32` is now exact too — the `inf` recorded here originally, and the diagnosis
attached to it, were both wrong.** That diagnosis blamed "the prototype's crude argument
resolution reading `sig_tf_wp_max = 0.0`"; `sig_tf_wp_max` in fact resolves correctly to
`4.0e8`. The real cause was the coil-island exclusion feeding a `0.0` placeholder into
`MaxForceDensity`'s `/ a_tf_wp_no_insulation` (`models/stellarator/coils/forces.py:34`).
Closed — see §5.2's note on the non-finite `c32` row.

**What Stage A proves:** the constraint layer's *values* are right when fed a real,
self-consistent state through the graph rather than off a `DataStructure`. That is one step
beyond `mda_constraint_harness.py`, which reads arguments straight from `data` and therefore
cannot catch a mis-wired `In`. **What it does not prove:** anything about the optimiser.

**Design note found the hard way:** seed the `Drive`'s `context` from the **MDA schedule's own
output env**, not from the `DataStructure`. Two of the 154 context variables
(`.physics.radius_plasma_profile_norm`, `.tfcoil.a_tf_wp_no_insulation`) have no
`DataStructure` field, and a scalar `0.0` placeholder for an array-valued one crashes
downstream (`plasma_profiles._simpson`, `y.shape[0]`) — the same array-shaped-ungrounded-input
limitation `mda_harness.py` documents. Running the plain MDA first produced **574** variables
and left **0** ungrounded.

### 5.2 Stage B — the Jacobian (the decisive check, and the one that measures the prize)

Compare the port's `jax.jacfwd` Jacobian against PROCESS's own `Evaluators.fcnvmc2`
finite-difference Jacobian, **at PROCESS's converged point, cell by cell**.

There is one subtlety that has to be got right or the comparison is meaningless:

> **A SAND Jacobian and PROCESS's finite-difference Jacobian are not the same derivative.**
> The SAND `ConditionMap` holds the coupling variables fixed as unknowns; PROCESS's
> `call_models` re-converges its Gauss–Seidel loop at every perturbed point, so `cnorm` is a
> **total** derivative. They agree only where the coupling does not respond.

They are related by one linear solve. Partition the SAND Jacobian into design columns `D`,
coupling columns `Y`, residual rows `R` and condition rows `C`:

```
dC/dD |total  =  J_CD  −  J_CY · (J_RY)⁻¹ · J_RD
```

and then, to reach PROCESS's spelling, `cnorm[i, j] = −(1/scale_i) · (dC/dD)[j, i]` — the
minus from `constraint_eqns:2007`, the `1/scale` because `xv = x · scale`
(`iteration_variables.py:348-352`).

**Prototyped and run.** `jacfwd` produced the full **22 × 17** Jacobian in **5.9 s** (one
trace). The reduction was applied and compared against `fcnvmc2` (8 × 14, `epsfcn = 0.01`).
Relative agreement per cell:

```
c65   x2: 7.9e-14   x3: 3.4e-05   x56: 1.0e-04
c8    x3: 1.5e-14   x6: 1.9e-04   x109: 1.9e-04
c67   x3: 1.6e-08   x2: 1.6e-05   x6: 1.7e-04   x109: 1.9e-04
c18   x3: 1.9e-06   x2: 1.6e-05   x6: 3.6e-04   x109: 1.7e-04
c17   x2: 1.6e-05   x3: 6.9e-05   x6: 7.7e-03   x109: 1.6e-04
c2    x2: 1.3e-02   x3: 1.2e-02   x10: 1.2e-02  x109: 1.2e-03
c62   x2: 1.2e-02   x3: 1.2e-02   x10: 1.2e-02
```

and three structured disagreements, each of which **names a specific defect**:

- **column x4 (`temp_plasma_electron_vol_avg_kev`) is wrong for every constraint** — the port
  gives `+1.3e-04` where PROCESS gives `−1.65e+00` for c8, and similarly for c17/c18/c67. The
  port's electron-temperature sensitivity is essentially absent. This is a real, findable,
  previously-invisible bug: every value the MDA harness checks is *correct at the reference
  point*, so no value comparison could have found it. It is the single most valuable thing
  this stage produced.
- **columns x2 and x59 were zero for c82/c83/c32/c35** where PROCESS has real sensitivity,
  because the prototype excluded the coil island
  (`Intersect`/`WindingPackIntersectInputs`/`WindingPackTotalSizePost`,
  `mda_harness.EXCLUDED_NODE_NAMES`). Not a port bug; a demonstration that the exclusion
  *mattered* for the coil constraints and could not survive into the `Optimise` layer.
  **The exclusion is closed** (see the note below on constraint 32), so those columns have
  a live path; §10.5 is the current per-cell record.
- **c2/c62 at ~1.2e-2** — the same `ConfinementTime` residual §8 records, now with a number
  attached in derivative space too.

**Is PROCESS's finite difference trustworthy as a reference?** Measured: re-running `fcnvmc2`
at `epsfcn = 1e-3` and `1e-4` changes the Jacobian by a **median 3.4e-13** relative — i.e. for
almost every cell PROCESS's 1 %-step central difference is exact, because the dependence is
locally linear. **One** cell changes by **23 %**, so it is not uniformly trustworthy, and the
harness must carry a per-cell error bar. Use exactly the existing convention: Richardson
extrapolation, `(4/3)|D(h) − D(h/2)|` plus a round-off floor, and assert agreement within the
reference's own error bar times a safety factor (`_audit/test_harness.md:157-161`). That
machinery already exists in `functional_process/_harness/finite_difference.py` and should be
reused rather than re-derived.

**Cost.** `fcnvmc2` for `n = 8`, `m = 14` — 16 full PROCESS pipeline evaluations — took
**0.9 s**. The port's `jacfwd`, measured properly:

| | |
|---|---|
| cold, unjitted, one call | 5.843 s |
| `eqx.filter_jit`, first call (trace + compile) | 1.620 s |
| **jitted steady state, median of 20** | **0.539 ms** |
| `ConditionMap` alone, jitted median | 0.401 ms |

**Time an unjitted cold call and you will conclude there is no speed win** — that 5.9 s is
tracing plus compilation plus one execution, not the cost of a Jacobian, and this section
originally drew exactly the wrong conclusion from it. The real comparison is **0.54 ms
against PROCESS's 0.9 s**, on the *larger* Jacobian, with compilation paid once per shape.
Correctness of the jitted result was checked, not assumed: jitted and unjitted agree to
`6.1e-14` across every finite cell with identical non-finite masks (a naive `np.allclose`
reports `False` purely because `NaN != NaN`).

**§4.2's "the schedule stops being one trace" caveat is real but nearly free.**
`pyvmcon.AbstractProblem.__call__(x) -> Result` hands back `(f, df, eq, deq, ie, die)` for
one iterate — VMCON asks for the **whole Jacobian at a point**, then does its QP subproblem
on the host. So the trace boundary is *per SQP iteration*, and each iteration is one jitted
`jacfwd` call with compilation paid once for the whole solve. The host-side QP is over a
handful of variables, negligible beside a 0.9 s PROCESS pipeline sweep. There is no need to
keep the solve inside a single trace to get the win, and no need for a jittable optimiser
on performance grounds.

**The non-finite `c32` row — traced and closed.** This measurement originally found 17
non-finite cells, all of them one row (every entry of `^cond.constraints.c32`, 2 `inf` and
15 `nan`), with the other rows fully finite. The confirmed cause was **`MaxForceDensity`'s
trailing `/ a_tf_wp_no_insulation` (`models/stellarator/coils/forces.py`) dividing by the
`0.0` placeholder the coil-island exclusion left behind** — not a port defect, an artefact
of `mda_harness.EXCLUDED_NODE_NAMES`, and not (as first guessed) `dr_tf_wp_with_insulation`,
which was correct throughout. It was fixed at its cause rather than patched: the coil island
is no longer excluded, grounded by three `KNOWN_MINT_VALUES` reconstructions off PROCESS's
own stored fields —

```
wp_width_r_min          = .tfcoil.dr_tf_wp_with_insulation      (a starting guess only;
                                                                 `Intersect` re-solves it)
a_tf_wp_no_insulation   = .tfcoil.dx_tf_wp_primary_toroidal * .tfcoil.dr_tf_wp_with_insulation
a_tf_wp_with_insulation = (dr_tf_wp_with_insulation + 2*dx_tf_wp_insulation)
                          * (dx_tf_wp_primary_toroidal + 2*dx_tf_wp_insulation)
```

— each checked independently (feeding the `a_tf_wp_no_insulation` reconstruction into
PROCESS's own `j_tf_wp = coilcurrent * 1e6 / a_tf_wp_no_insulation` reproduces the stored
`j_tf_wp` to the last printed digit). Result: **17 → 0 non-finite cells**, every row finite,
and `Intersect` gains the first PROCESS-comparable value check it has ever had — its
`Tier2Contract` has none by construction, and its solved answer now agrees with
`.tfcoil.dr_tf_wp_with_insulation` within `compare`'s `rtol=1e-6`. `Intersect`'s `RootFind`
also joins the SAND problem as an extra design variable and equality, which is the
structurally honest shape. **One item is still open from that investigation**: the x2/x59
columns of c82/c83/c32/c35, measured as spuriously zero while the island was cut out, now
have a live path but have not been re-checked against `fcnvmc2` cell by cell.

The general point stands and was load-bearing: an excluded island does not merely zero some
columns, it can make a whole constraint row non-differentiable, which no SQP accepts.

**What the corrected numbers let this section claim.** The win is *both* exactness and
speed, where the original draft conceded speed. Exactness: the same derivative with no
step-size choice, from a single trace. Speed: 0.54 ms against 0.9 s at `n = 8` on a
larger Jacobian — and that gap widens, since `jacfwd` costs one trace where `fcnvmc2`
costs `2n` full pipeline evaluations, so scaling to the 83-variable ceiling multiplies
PROCESS's cost by ~10 and the port's by much less. The earlier "PROCESS's finite
differences are faster" conclusion was an artifact of timing a cold, unjitted call, not a
property of the methods.

**Still true and still worth keeping**: PROCESS's finite differences are, per the
step-independence measurement, essentially exact for almost every cell here, so this is
not a case where AD rescues a broken reference. It replaces a good reference with an
exact one, faster.


### 5.3 Stage C — the solve

Run the `VmconDriver` and check where it lands. Three rungs, of which C1 and C2 are **built
and run** (see §10.6) and C3 is **open**:

- **C1:** solve from a perturbed start with a live objective and check the driver reaches a
  point where the equalities vanish and the inequalities hold. **No PROCESS number to compare
  against** — this is exactly the epistemic position `next_steps.md`'s "validation-chain
  question" describes for `Intersect`/`DuctDiameterRootFind`: *"reproduces PROCESS's own
  formula, solved more tightly than PROCESS's own loose iteration"*, not *"matches PROCESS's
  own reported number"*. What C1 establishes is that the machinery solves the problem it
  states.
- **C2:** start the driver *at* PROCESS's converged point. A correct driver over an identical
  model would take a null step; any movement is a modelling gap or a driver bug, and Stage B's
  per-cell Jacobian says which. Measured in §10.6 — it converges and does *not* stay put, for
  reasons that are now named.
- **C3, still blocked:** `i_figure_merit = 6`, all 14 constraints, solve from the input
  file's own starting values, compare all 8 converged iteration variables against PROCESS's.
  **This is the only check that would justify the sentence "the port reproduces PROCESS's
  optimisation", and it is not available yet** — the blocker is no longer porting but
  graph completeness at a cold point (§10.6). Say so wherever it is cited.

### 5.4 What "matches PROCESS" can and cannot mean here

Following `next_steps.md`'s own model for this kind of precision:

- PROCESS's converged `x` is a point where **PROCESS's finite-difference-approximated** KKT
  conditions hold to **its own** tolerance (`epsilon = tolerance`, convergence parameter
  `2.4e-07` after 46 iterations), using a **1 % perturbation** on a pipeline that is itself
  only iterated to `check_agreement`'s `rtol = 1e-6` over at most 10 Gauss–Seidel passes
  (`process/core/caller.py:96-126`). It is not the exact optimum of the stated problem.
- Therefore **"the port's optimiser lands on PROCESS's `x`" and "the port's optimiser solves
  the stated problem" are different claims**, and they can differ by more than either
  tolerance. The measured size of the first gap is available: the FD-vs-AD Jacobian difference
  in the cells where the model agrees (median ~1e-4 here, `2.3e-1` in the worst cell) *is* the
  error PROCESS's own optimiser was steering with.
- The claim Stage B licenses, precisely: *"at PROCESS's own answer, the port's model and its
  exact derivatives agree with PROCESS's model and its finite-difference derivatives, cell by
  cell, to within the finite difference's own Richardson error bar — except in these named
  cells, for these named reasons."* That is a stronger and more diagnostic statement than any
  value comparison, and it is what the `Optimise` layer should be judged on until C3 exists.

---

## 6. Interaction with the driven MDA — the highest-risk unknown, settled

**Question:** the graph already has 11 driven SCCs. An `Optimise` sits outside all of them.
Does `Blocking`/`schedule_for` nest cleanly?

**Answer: no. `Blocking` records the nesting; `evaluate.py` does not consume it. This cannot
be worked around from this repo.** Verified three ways, on a toy graph and on the real one.

### 6.1 What happens structurally

Adding constraint nodes, an objective node and an `Optimise` to the 101-node driven graph gives
115 nodes, and — as expected — **the `Optimise` fuses the driven SCCs into one giant SCC**
(largest component: 54 nodes, containing `^problem['Intersect']`,
`^problem.physics.proton_rate_density`, `^problem.fwbs.f_ster_div_single` and `.Opt`). It has
to: the `Optimise` owns variables everything reads and reads variables everything produces.

`Blocking.problem_types` then raises, correctly:

```
ValueError: block [...] declares 2 problems ([...]) -- one driver answers one problem, so
`Combine` them into a single problem over every unknown, or nest one inside the other.
```

### 6.2 Nesting is refused, at three independent points

`Blocking.nest(N('Opt'))` **works** and produces exactly the right structure — on the real
graph its interior holds the three inner problems. But:

1. **`Schedule.steps` never reads `Blocking.inner`.** `evaluate.py:342-351` builds one
   `Call`/`Drive` per entry of `blocking.subgraphs`; `inner` appears nowhere in `evaluate.py`.
2. **`schedule_for` resolves drivers against the flat blocks.** `evaluate.py:403-412` walks
   `blocking.blocks`, so the outer `Optimise`'s driver and the interior problems' drivers all
   claim the *same* block. Verified on the real graph:
   `ValueError: block [...] is claimed by more than one driver (['.Opt',
   '^problem.fwbs.f_ster_div_single', '^problem.physics.proton_rate_density',
   "^problem['Intersect']"])`. There is no slot in `schedule_for`'s `drivers` mapping for an
   interior's drivers at all.
3. **`Drive.__check_init__` asks `Graph.problem_type`** (`evaluate.py:221`), which raises on a
   block with two declared problems (`graph.py:452-459`).

And cottax's own `CLAUDE.md` states it, in §Evaluation: *"`Schedule` is itself a `Step`, but
**nothing builds a nested one yet** … Until then a driven block's body must be acyclic once its
problems are dropped, so **`schedule_for` still refuses MDF**."* The three refusals above are
that sentence, measured.

One further trap worth recording, because it is a *wrong answer* rather than an error: on this
graph the giant block's `runnable` (`graph.py:501-508`, which drops **every** `DeclaredNode`)
*is* acyclic, so `Graph.driven` returns `True` — every inner problem is a `DeclaredNode` and
gets dropped along with the outer one. If `Graph.problem_type`'s two-problem refusal were ever
relaxed, `Drive.body` would silently evaluate the objective and constraints **with the inner
MDA unconverged**, its coupling variables read from `context` as frozen constants. That would
be a plausible-looking wrong number. The refusal is load-bearing.

### 6.3 SAND works, today, on the real graph — verified end to end

The alternative is the one `problem.py` was built for and cottax's own docs name:
`Optimise + RootFind → Optimise` is **SAND** (*"the unknowns join `design`, the residuals join
`equalities`"*, `~/jaxgraph` `CLAUDE.md` §The problem types; `problem.py:104-110`). So:

```python
p = Plan(graph_with_optimise)
for n in [x for x in p.graph.declared if isinstance(p.graph[x], FixedPoint)]:
    p = p + Residualise(n)  # FixedPoint -> RootFind
p = p + Combine(
    NodePath(
        ...(
            "sand",
        )
    ),
    tuple(p.graph.declared),
)
schedule = schedule_for(Blocking.scc(p.graph), {p.graph.declared[0]: VmconDriver(...)})
```

Measured on the real graph (with `DuctDiameterRootFind` dropped, 99 nodes — the coil island is no
longer excluded, see §5.2's note on the non-finite `c32` row):

| | |
|---|---|
| `FixedPoint`s residualised | 9 |
| problems combined | 12 → one `^problem.sand` |
| resulting `Optimise` | **18 design, 11 equalities, 11 inequalities** |
| `Blocking.scc` → `problem_types` | one `Optimise`, everything else `None` |
| `schedule_for` | **builds** |
| the `Drive` | 69 nodes, 18 unknowns, 23 conditions, 160 context vars |
| `ConditionMap(...)` | evaluates, 0.04 s |
| `jax.jacfwd` | 23 × 18; **0.54 ms** jitted steady state (5.9 s cold+untraced — see §4's corrected Cost note) |

Design = the iteration variables + the coupling unknowns; equalities = the real equality
plus the fixed-point residuals; inequalities = the real ones. **Only 69 of the 99 nodes end
up inside the driven block** — the rest run before and after it, so this is not "the whole
graph as one opaque block"; the acyclic remainder is still ordinary `Call` steps. That is
precisely the structural win `CLAUDE.md` argues for, obtained without any upstream change.

**Ordering is reliable.** `Drive.conditions` came back as
`(objective, …equalities…, …inequalities…)`, matching `Optimise.inputs` (`problem.py:82-84`) —
which is what makes §4.1's count-based split sound.

### 6.4 Two real defects SAND exposed that the MDA never could

Both found while getting §6.3 to run; both are in files another agent may be editing, so they
are reported, not fixed.

**(a) Six 1-tuple returns.** `Residualise` mints a `Compare` whose body is
`operator.sub(condition, unknown)`, and it raised
`TypeError: unsupported operand type(s) for -: 'tuple' and 'ArrayImpl'`. Cause:
`functional_process/models/power/thermal_cryo.py` lines **1208, 1267, 1308, 1371, 1443,
1488** each `return (x,)` from a `FixedPointFunction.step` with a single `Output`. cottax's
`_bind` (`evaluate.py:52-69`, the `len(owns) == 1` branch at `:54-56`) puts the whole return
under a single output, so the env holds
`(value,)` instead of `value`. `PicardDriver` never notices — `ravel_pytree` flattens a 1-tuple
happily and the comparison happens on the flat vector — so this is invisible today and fatal
the moment anything subtracts. **Exactly the bug class `next_steps.md` §8 already records once**
(`ZTfInsideHalf`: *"a 1-tuple return where cottax's single-`Output` binding convention wants
the bare value"*). Worth a `grep -n 'return (.*,)$' functional_process/models/` sweep.

**(b) Two structurally degenerate fixed points.** With (a) worked around, the residual block
`J_RY` came out **singular**: the rows and columns for `^cond^cond.heat_transport.eta_turbine`
and `^cond^cond.costs.cplife` are **identically zero**. Cause: with
`i_thermal_electric_conversion = 2` (`USER_INPUT`) `EtaTurbineStep`'s `step` is an exact
identity — its own docstring says so (*"identity (gradient exactly 1) on the pass-through
sub-branches (`USER_INPUT`; …)"*) — so `r = g(u) − u ≡ 0` and the fixed-point equation is
`u = u`, determining nothing. `CplifeAvail` is the same in this configuration.

This is a genuine architectural consequence, not a bug in the nodes: **a `FixedPointFunction`
that is an identity in the active configuration is a well-posed Picard problem (converges in
one step from anywhere) and a rank-deficient SAND equality.** Any SQP will fail on it. The fix
is assembly-time and belongs with `configuration.py`'s existing per-configuration selection:
**when residualising for SAND, drop any fixed point whose residual is structurally zero in
this configuration** — its unknown reverts to an ordinary boundary input. Detecting it is one
`jacfwd` of the residual block, which the harness computes anyway.

### 6.5 If MDF is wanted later

It is a genuine upstream feature, scoped in cottax's own docs: *"Nesting lands on `Drive.body`,
which becomes a `Step` rather than a `Graph`"*, plus `schedule_for` needing somewhere to put an
interior's drivers. That is a real change to `evaluate.py`, not a one-liner, and §8 lists it as
such. **SAND is not a workaround for it** — it is a different, standard, well-understood MDO
architecture with its own advantages (one solver, exact coupled derivatives, no nested
convergence tolerance) and its own costs (a much larger design vector, and every coupling
variable needs a starting guess and a scale). The recommendation is to build SAND now and treat
MDF as a later second opinion, not to wait.

---

## 7. Staged implementation plan

**Stages 0–8 are done** — the 1-tuple sweep, the `constraint_node`/`objective_node`
factories, `optimise_for`, `sand()`, the Stage A and Stage B harnesses, `VmconDriver` and
`mda.default_drivers`'s `Optimise` arm, Stage C1/C2, and `Feasibility` folding. The record
of what each produced is §10.

| # | stage | check | notes |
|---|---|---|---|
| 9 | *(blocked)* **Stage C3**: `i_figure_merit = 6`, all 14 constraints, cold solve vs PROCESS's 8 converged values | all 8 within tolerance | no longer blocked on porting — blocked on graph completeness at a cold point (§10.6): the SAND block's context variables that no `Call` step produces and that have no real input value |

---

## 8. Items that need an upstream change in `~/jaxgraph`

Separated deliberately, per the precedent in §8 of `next_steps.md` (the `to_graph()`/
`node_and_names` fix). **None of these blocks stages 0–8 above.**

1. **`ConditionMap` cannot describe an `Optimise`** (§4.1). It carries `body`, `unknowns`,
   `conditions`, `context` (`evaluate.py:135-178`) — a flat condition tuple. A driver whose
   `drives = Optimise` therefore cannot tell an equality from an inequality from the
   objective, even though `Drive` knows (`Drive.problem`, `evaluate.py:230-234`). This is the
   same *class* of gap as `to_graph()`'s: cottax's own `AbstractDriver` docstring
   (`evaluate.py:186-190`) promises the `Optimise` pairing, and the seam handed to the driver
   cannot express it. **Minimal fix: give `ConditionMap` the problem definition** (one extra
   field, set in `Drive.condition_map`, `evaluate.py:265-278`) — then a driver asks
   `cm.problem.equalities` instead of being told a count. Locally workaroundable, so not
   urgent; but the workaround is a positional contract that can silently go wrong, which is
   exactly what `drives` exists to prevent.
2. **`Optimise` has no bounds** (§1.1). Box bounds on design variables are not a solver detail
   — PROCESS carries them per iteration variable and VMCON takes them as a distinct argument.
   Worth raising upstream as a question (are bounds structure or algorithm?), not as an
   assumed defect; the driver-field workaround is legitimate.
3. **`Schedule` does not consume `Blocking.inner`** (§6.2/§6.5) — i.e. MDF. Already a known,
   documented cottax limitation, not a surprise. Only worth raising if MDF is actually wanted.

---

## 9. Open questions, and the experiment that settles each

1. ~~**The x4 (`temp_plasma_electron_vol_avg_kev`) column.**~~ **Closed — §10.5a.** The
   experiment proposed here (`jacfwd` the MDA schedule, not the SAND block, and walk the
   graph in topological order for the first node whose output derivative is zero where it
   should not be) is the one that found it, and it has since found a second defect of the
   same class (§10.5c). Keep the method.
2. **Do the other six owned-and-also-an-ixc variables (§1.2) really need the
   `DefaultAspectRatio` treatment?** Only ixc 1 is confirmed. *Experiment:* for each, construct
   a graph with that variable active and check `Graph.__check_init__` refuses, then check
   PROCESS's own source for whether the producer is conditional on `ixc` the way
   `stellarator.py` is for `aspect`.
3. **Is the degenerate-identity fixed point (§6.4b) configuration-specific or a modelling
   error?** `EtaTurbineStep` with `i_thermal_electric_conversion = 2` is an identity by
   construction; whether it should then be registered at all is a `configuration.py` question
   this pass did not answer.
4. **`VmconDriver` inside `jit`.** §4.2 asserts the schedule stops being one trace. Not
   measured — *experiment:* build the schedule, `eqx.filter_jit` it, and see what breaks and
   how much the surrounding `Call` steps still gain from being traced individually.
5. **Whether SAND's 17-dimensional design vector converges from a cold start.** §6.3 proves it
   assembles and differentiates; nobody has run an SQP on it. The 9 coupling unknowns need
   starting guesses and scales that PROCESS's `xcm`/`scale` machinery says nothing about,
   because PROCESS never exposes them as unknowns. This is the largest genuinely unknown cost
   of choosing SAND over MDF, and stage 7 is where it is found out.
6. **Whether `constraints.md`'s "Compare-shaped in future" candidates (e.g. `:1616-1617`)
   change §2.3's policy.** Not investigated.

---

## 10. As built — the Stage C session, measured

Everything below was run in `process_port` on the current tree against
`tests/regression/input_files/stellarator_helias.IN.DAT`. **It supersedes every number in
§3.3, §5 and §6.3**, which were measured on a 99-node graph before the costs subsystem,
the coil island and three producers this session added. Reproduce with:

```bash
$PY -m pytest functional_process -q          # see `next_steps.md`'s "Verified state"
$PY functional_process/run_mda_harness.py    # the MDA harness, unchanged in scope
$PY functional_process/run_sand_harness.py   # the ladder below, one PROCESS run, ~4 min
```

### 10.1 What was built

| file | what |
|---|---|
| `core/solver/drivers.py` | **`VmconDriver`** beside `PicardDriver` — `pyvmcon` fed one `jax.jacfwd` of the block's `ConditionMap`. Bounds by `VarPath`, PROCESS's `1/x_start` design scaling, per-condition scaling, count-based eq/ineq split, `VMCONConvergenceException -> e.x`. |
| `sand.py` | the assembly: `constraint_nodes`, `objective_node`, `optimise_graph`, `degenerate_fixed_points`, `sand_graph`, `sand_schedule`, `residual_condition_scales`, `iteration_variable_path`, `design_bounds`. |
| `sand_harness.py` | the ladder: `reference_run`, `mda_env`, `assemble`, `stage_a`, `port_jacobian`, `process_jacobian_with_error`, `reduce_jacobian`, `to_process_spelling`. |
| `run_sand_harness.py` | one entry point running A → B → C off a single PROCESS run. |
| `test_sand.py` | 16 assembly/driver tests, no PROCESS run needed. |
| `mda.py` | `default_drivers` grew its `Optimise` arm — and **reads the eq/ineq counts off the `Optimise` node**, so §4.1's "positional contract" concern never arises. |

### 10.2 §4.1's design decisions, revisited after building them

- **Sign convention: confirmed, and it is load-bearing.** cottax's `g <= 0` against
  `pyvmcon`'s `i >= 0` needs exactly one negation, matching
  `constraints.py:2007`'s own `-tmp_cc`. `test_sand.py::
  test_vmcon_driver_reaches_a_known_constrained_optimum` is built so the *unflipped*
  driver stops at the unconstrained minimum, so the line cannot rot silently.
  Equalities are passed **unnegated**: `h = 0` and `-h = 0` are the same set, `pyvmcon`'s
  convergence test takes `abs(lambda_eq @ eq)`, and only the multiplier's sign moves.
- **Bounds on the driver, keyed by `VarPath` and not positional.** §1.1 recommended
  driver fields but left the ordering as "a positional contract checkable only at
  assignment time". Keying by name removes the contract entirely, and it is what cottax's
  own rule ("every query takes exact names") asks for.
- **The count-based split needed no counting.** §4.1 proposed
  `VmconDriver(n_equality=…, n_inequality=…)` constructed by `mda.default_drivers`. That
  is what happens — but the counts come from `len(problem.equalities)` /
  `len(problem.inequalities)` on the `Optimise` node itself, which `default_drivers` has
  in hand. `__call__` still re-checks the total against `len(conditions.conditions)`.
  The upstream fix (§8.1) remains the right one; the local workaround turned out not to
  be a positional contract at all.
- **One thing §4 did not anticipate, and it decided whether the solve worked at all:
  the *conditions* need scaling too.** PROCESS's fourteen constraints arrive already
  normalised (`value/bound − 1`, O(1) by construction — §2.1 says exactly why). SAND's
  residual equalities do **not**: `g(u) − u` carries `u`'s units, and on this run that is
  a spread from `1e-3` to `1e5` inside one equality block. VMCON weights every constraint
  equally, so it takes steps that satisfy the small residuals and destroy the large ones.
  **Measured**: without scaling, C2 from PROCESS's own converged point runs to
  `max_iter = 100` with `max|eq|` stuck at `2.3e5` and the convergence parameter stalled
  at `4e-6`; with each residual scaled by `1/|u|` (a *relative* residual — exactly the
  shape PROCESS's own normalisation has), the same solve converges in **47 iterations** to
  `7.4e-9`. The scaling is per-condition and explicit (`VmconDriver.condition_scale`,
  supplied by `sand.residual_condition_scales`) precisely so PROCESS's own fourteen keep
  a factor of `1.0` and the iterates stay comparable.

### 10.3 The SAND shape, re-derived

Measured when the SAND layer was built, on a **146-node** `graph_for()`. The graph has
grown since (`next_steps.md`'s "Verified state"), and the SAND problem with it — it is now
25 conditions × 18 design — so read the *shape* below, not the individual counts.

| | §6.3 (stale) | as built |
|---|---|---|
| nodes in the assembled SAND graph | — | 160 |
| `FixedPoint`s residualised | 9 | **8** (two are dropped as degenerate first) |
| degenerate identity fixed points dropped | 2 (named) | **2, detected** (`EtaTurbineStep`, `CplifeAvail`) |
| design | 18 | **17** (8 `ixc` + 9 coupling) |
| equalities | 11 | **11** (2 real + 9 residuals) |
| inequalities | 11 | **12** |
| the `Drive` | 69 nodes | **115 nodes**, 17 unknowns, 24 conditions, 330 context |
| schedule steps | — | **45** — so 45 of the 160 nodes still run as ordinary `Call`s |

§6.4b's "filter for degenerate fixed points" is implemented as
`sand.degenerate_fixed_points`, which **differentiates** each candidate's residual rather
than listing names: a `FixedPointFunction` that is an identity in the active
configuration is a well-posed Picard problem and a rank-deficient SAND equality, and
which ones those are is a per-configuration fact.

**§6.4a's six 1-tuple returns are fixed** (`thermal_cryo.py:1208,1267,1308,1371,
1443,1488`), and the sweep found a seventh (`vacuum.py:180`, `VacuumPumpingSimple`).
`coils.py:619` and the two in test files are `AbstractDriver.__call__` returns, where a
tuple is the contract — not instances.

### 10.4 Stage A — 13 of 14 constraints exact

Conditions evaluated through the graph at PROCESS's converged point, against
`constraint_eqns`' own `-cc`:

| | |
|---|---|
| bit-exact or to `~1e-16` relative | **c24, c8, c17, c18, c67, c82, c83, c62, c32, c34, c65** — 11 of 14 |
| exact to `~1e-16` **absolute**, i.e. limited only by the residual itself being `~0` | **c2** (`-7.375409e-10` vs `-7.375413e-10`, `rel 6.0e-7`) and **c35** (`rel 1.2e-9`) |
| genuinely off | **c16** alone (`+1.760e-02` against `+3.78e-09`), plus the objective (`1.235974` against `1.214917`, `rel = 1.73e-2`) |
| all 9 SAND residuals | `0.0` (largest `1.7e-16`) — the MDA really is converged there |

So: **13 of 14 constraints agree, 11 of them bit-for-bit**; the two at `1e-7`/`1e-9`
*relative* are constraints whose own value is `1e-9`/`2e-7`, so their absolute agreement
is `~1e-16` and the relative figure is float noise, not disagreement. Only c16 differs
materially, and §10.4's cause below covers it and the objective together.
`run_sand_harness.py`'s own summary line counts the strict `rel < 1e-9` test and
therefore prints `11 of 15` (the 15th comparison is the objective) — read the table, not
the count.

§5.1's `c2` and `c62` disagreements are **gone**: both were the `ConfinementTime`
`q95`/`iotabar` binding bug, closed in `next_steps.md` §8.3.

**c16 and the objective are one cause, and it is PROCESS's, not the port's.** Measured by
instrumenting a real solve rather than inferred:

```
Buildings.run(output=False)   -> a_plant_floor_effective = 563075.16   (solve pass)
Buildings.run(output=True)    -> a_plant_floor_effective = 680433.44   (report pass)
   the only differing input:  .build.z_tf_inside_half  4.1556 -> 7.3592
```

`Stellarator.run(output=True)` reruns `st_build`/`st_coil` in the **opposite order** to
the solve pass (`stellarator.py:143-144` against `:157-158`), so the reported
`z_tf_inside_half` is `st_coil`'s and the solve-time one is `st_build`'s — the dual-write
`next_steps.md` §8 records as `ZTfInsideHalf`. `Buildings.run` reads it
(`process/models/buildings.py:52-54`) and recomputes `a_plant_floor_effective` from it,
but the report pass calls `power.output_plant_electric_powers()` instead of
`plant_electric_production()` (`stellarator.py:148-152`), so
`.heat_transport.p_plant_electric_base_total_mw` is never recomputed and keeps `89.461`
while `a_plant_floor_effective` moves. **PROCESS's converged `DataStructure` is therefore
internally inconsistent, and the port is the self-consistent side**: from PROCESS's own
formula and PROCESS's own stored `a_plant_floor_effective` it gets `107.065`. The
resulting `+17.604 MW` offset propagates linearly and *exactly* — every one of the
eighteen affected fields differs by `17.604` (or `-17.604` for the net power), through
`Acpow` and the cost accumulation to `.costs.coe` at `rel = 1.73e-2`. c16's residual is
`1 − p_net/1000 = 17.604/1000`. Recorded in `mda_harness.EXPLAINED_DISAGREEMENTS`,
deliberately not suppressed.

**What Stage A proves:** the constraint layer's values are right when fed a real,
self-consistent state *through the graph*, which `mda_constraint_harness.py` cannot
check because it reads arguments straight off a `DataStructure`. **What it does not
prove:** anything about the optimiser.

### 10.5 Stage B — the Jacobian, and the two defects it found

Port `jacfwd`, Schur-reduced to the design columns, against PROCESS's own central
differences at `epsfcn = 0.01` with a per-cell Richardson error bar from
`_harness/finite_difference.fd_gradient_with_error`.

| | |
|---|---|
| port Jacobian | **24 × 17**, `0` non-finite cells |
| compile (`eqx.filter_jit`, first call) | 5.1 s |
| **jitted steady state, median of 10** | **0.69 ms** |
| PROCESS `fcnvmc2` (`2n+1 = 17` pipeline sweeps) | **1.00 s** |
| PROCESS with Richardson bars (`5n = 40` sweeps) | 2.83 s |

So the port's Jacobian is **~1450× faster than PROCESS's**, on a larger matrix, and is
exact rather than a 1 %-step difference. Compilation is paid once per shape and amortises
over a whole solve — a 47-iteration C2 solve costs 7.6 s *including* it.

**The reduction has to be equilibrated, and that is a real finding.** `J_RY`'s raw
singular values span `4.9e14` to `2.1e-15` (condition number `2.4e29`) — entirely units,
not rank deficiency: scaling each coupling column by its own value and each row by its
largest entry gives a condition number of **12.1**. `sand_harness.reduce_jacobian` does
that, and the identity it uses (`J_CY J_RY⁻¹ J_RD = (J_CY C) A⁻¹ (R⁻¹ J_RD)` for
`A = R⁻¹ J_RY C`) is exact algebra, not an approximation.

Per-cell agreement, `|port − PROCESS| / |PROCESS|`:

```
              x2          x3          x4          x6         x10         x56         x59        x109
objf   1.90e-01*   3.43e-01*   1.79e-01*   1.79e-01*   0.00e+00    1.97e-02*   3.38e-01*   2.41e-01*
c2     8.93e-05    1.06e-04    1.07e-03    6.35e-04*   1.00e-04    0.00e+00    0.00e+00    2.26e-04*
c16    4.78e-01*   4.15e-02*   4.94e-02*   4.92e-02*   0.00e+00    7.41e-05    4.81e-01*   4.93e-02*
c24    2.00e-04   1.35e+282    2.30e-01    9.69e-16    0.00e+00    0.00e+00    0.00e+00    4.18e-13
c8     0.00e+00    1.59e-14    1.55e-04*   1.87e-04*   0.00e+00    0.00e+00    0.00e+00    1.87e-04*
c17    1.59e-05    6.86e-05    4.01e-06    7.70e-03*   0.00e+00    0.00e+00    0.00e+00    1.61e-04*
c18    1.59e-05    1.88e-06    1.30e-04    3.57e-04*   0.00e+00    0.00e+00    0.00e+00    1.73e-04*
c67    1.59e-05    1.63e-08    2.05e-04*   1.72e-04*   0.00e+00    0.00e+00    0.00e+00    1.89e-04*
c82    3.55e-03    3.03e-05    0.00e+00    0.00e+00    0.00e+00    0.00e+00    2.04e-04    0.00e+00
c83    3.51e-03    1.77e-06    0.00e+00    0.00e+00    0.00e+00    0.00e+00    3.03e-04    0.00e+00
c62    8.49e-05    1.07e-04    5.10e+00*   6.66e+00*   1.00e-04    0.00e+00    0.00e+00    1.41e-01*
c32    1.65e-02    1.44e-05    0.00e+00    0.00e+00    0.00e+00    0.00e+00    1.15e-04    0.00e+00
c34    2.21e-03    7.57e-05    0.00e+00    0.00e+00    0.00e+00    1.00e-04    4.17e-04    0.00e+00
c35    8.85e-03    7.24e-05    0.00e+00    0.00e+00    0.00e+00    1.18e-05    9.26e-05    0.00e+00
c65    7.88e-14    3.36e-05    0.00e+00    0.00e+00    0.00e+00    1.00e-04    0.00e+00    0.00e+00
```

(`*` = outside the finite difference's own Richardson bar × 4. `c24 x3`'s `1.35e+282` is
a division by PROCESS's own `0` in that cell, not a large disagreement: the port gives
`~1e-266` where PROCESS gives exactly `0`.)

**§5.2's `x2`/`x59` bullet, re-measured with the coil island back in as instructed:
the columns are alive and they agree.** c82/c83/c32/c34/c35/c65 all have real `x2` and
`x59` sensitivity now, matching PROCESS to between `1.8e-6` and `1.65e-2`. The stale
bullet's "these columns are zero" is dead and the exclusion's removal is vindicated. The
17 non-finite cells §5.2 recorded are also gone — but see below, they came back once and
for a *different* reason.

#### 10.5a The x4 column: a real port defect, found, fixed, and confirmed fixed

§9's open question 1 was right. **First measurement**, before the fix:

```
       port          PROCESS
c8    +1.27e-04     -1.65e+00
c17   +1.12e-02     +1.58e+00
c18   -6.90e-03     -2.01e+00
c67   +9.80e-04     -1.40e+00
```

i.e. the port's sensitivity to `temp_plasma_electron_vol_avg_kev` was essentially absent.
Traced by differentiating the SAND body with respect to that one unknown and walking the
topological order: `ParabolicOnAxisTemperatures` gave `temp_plasma_electron_on_axis_kev` a
relative sensitivity of `1.00` and `temp_plasma_ion_on_axis_kev` one of `2e-16`, and
`FusionRates`' `sigmav_dt_average` one of `2.4e-16` — the DT reactivity, which goes as
roughly `T²`, was a constant.

**Cause: three fields had no producer at all.** `.physics.temp_plasma_ion_vol_avg_kev`,
`.physics.temp_plasma_electron_density_weighted_kev` and
`.physics.temp_plasma_ion_density_weighted_kev` were boundary inputs, so ion temperature
was structurally disconnected from the electron temperature the optimiser varies. The two
pure functions were already ported (`plasma_profiles.py`'s
`calculate_ion_vol_avg_temperature` and `calculate_parabolic_profile_values`); only the
**nodes** were missing, deferred by that record's own open question 1 (which
`total_process.py`'s `EcrhDensityLimit(i_plasma_pedestal=0)` had already settled in
practice).

Fixed by registering `IonVolAvgTemperature` (a `FixedPointFunction`, because the field is
conditionally owned *by data* — `f_temp_plasma_ion_electron > 0` — so the "keep the
incumbent" arm is an identity fixed point that `degenerate_fixed_points` would drop, which
is exactly PROCESS's semantics recovered from structure) and `ParabolicProfileValues`
(under the `i_plasma_pedestal == 0` arm). **After the fix**:

```
       port          PROCESS      relative
c2    -3.129026e-01  -3.132385e-01  1.07e-03
c8    -1.648998e+00  -1.649253e+00  1.55e-04
c17   +1.584691e+00  +1.584697e+00  4.01e-06
c18   -2.010091e+00  -2.009830e+00  1.30e-04
c67   -1.400490e+00  -1.400778e+00  2.05e-04
c24   -1.233812e+00  -1.602395e+00  2.30e-01   (within the FD's own error bar, 4.91e-01)
```

The MDA harness gained **+7 agreements and 0 new disagreements** from those two nodes, so
they reproduce PROCESS's converged values exactly as well.

**This is the single most valuable thing the `Optimise` layer has produced.** Every value
the MDA harness checks was correct at the reference point throughout; no value comparison
could have found it. That is the case for building Stage B, made concrete.

#### 10.5b A second defect, exposed by closing the first

Registering `StellaratorBetaAndStoredEnergy` gave constraint 24 a live argument, which
pulled `FastAlphaBeta` into the differentiated set for the first time — and its whole row
came back non-finite. Cause: `jnp.sqrt(jnp.maximum(0.0, temp_sum_20 - 0.65))`
(`pure_formulas.py`), the exact JAX trap `next_steps.md` §9 already records for
`costs.py:2874-2888` — `sqrt` has an infinite derivative at zero and `inf * 0` is `nan`.
**The clamp is *active* on this run** (`temp_sum_20 = 0.6449` against the `0.65`
threshold), so this was not hypothetical. Fixed with the standard double `jnp.where`;
value-identical, and the Jacobian went from 17 non-finite cells to 0.

Worth generalising: *a defect only becomes visible when something downstream of it reads
into a condition*. Both of this session's gradient defects were latent for exactly that
reason, and both surfaced the moment a producer gap was closed.

#### 10.5c Rows that still disagree, and why

- **`objf` (every column) and `c16` (every column)** — the `z_tf_inside_half` staleness of
  §10.4. The port's `p_plant_electric_base_total_mw` is a live function of
  `a_plant_floor_effective`, PROCESS's is a frozen solve-pass number, so both the value
  and its derivative differ. Not a port defect; not fixable without choosing which of
  PROCESS's two `z_tf_inside_half` values to model, which `next_steps.md` §8 already
  decided the other way.
- **`c62`, columns x4 / x6 / x109 — DIAGNOSED AND CLOSED, and the cause was not local to
  `c62`.** `f_t_alpha_energy_confinement`'s *value* was exact (Stage A) while its
  derivative was wrong by a factor with a sign flip on x4. The cause was three fields with
  no producer at all — `.physics.fusden_total`, `.fusden_alpha_total` and
  `.p_dt_total_mw` — so `t_alpha_confinement = nd_alphas / fusden_alpha_total` had a frozen
  denominator and its temperature derivative was **structurally absent**. A new node
  `FusionTotalsNoBeam` (`plasma_physics.py`, the `else` arm of
  `stellarator.py:2002-2054`, three identities) gives them producers. Measured on the
  `c62` row: x4 `5.10e+00*` → `5.42e-04`, x6 `6.66e+00*` → `3.99e-05`, x109 `1.41e-01*` →
  `3.25e-07`, and c2/c8/c17/c18/c67 improved alongside, several losing their error-bar
  stars. **The only starred cells left in the whole Jacobian are `objf` and `c16` (the
  report-pass/solve-pass inconsistency above) and `c24 x3` (a division by PROCESS's own
  exact zero).** This is the **third** instance of the same defect class as §10.5a's
  iteration variable 4: *a missing producer that every value test passes and only a
  gradient sees.* It also enlarged the density/fusion cycle from 4 nodes to 6 and forced a
  second `mda.CUTS` entry (`next_steps.md` §5).

### 10.6 Stage C — the solve

`VmconDriver`, `max_iter = 100`, `tolerance = 1e-8`, PROCESS's own per-run bounds, design
variables scaled by `1/x_start`, residual equalities scaled by `1/|u|`.

**C2 — start at PROCESS's own converged `x`. Converges; does not stay put.**

| | |
|---|---|
| SQP iterations | **34** (PROCESS: 46) |
| wall clock | **7.1 s**, most of it one-off trace + compilation (PROCESS: **98.9 s**) |

The iteration count came down from 47 once `FusionTotalsNoBeam` closed the `c62` derivative
row (§10.5c) — a sharper Jacobian, fewer steps.

The per-variable landing below was measured on the earlier, 47-iteration run and has **not**
been re-measured since; treat the individual numbers as indicative and the framing beneath
them as the durable part. At that point the driver reached a feasible KKT point with
convergence parameter `7.4e-09` (PROCESS's own: `2.4e-07`), `max abs(equality) = 3.2e-05`,
worst inequality `−7.4e-10`, and objective `1.235974` → `1.217184`.

Where it landed, against PROCESS:

```
ixc   name                                PROCESS            port              rel
  2   b_plasma_toroidal_on_axis            4.703724864       4.703887028      3.45e-05
  3   rmajor                              26.69437015       26.64768541       1.75e-03
  4   temp_plasma_electron_vol_avg_kev     5.673208206       5.723894383      8.93e-03
  6   nd_plasma_electrons_vol_avg          1.745959785e+20   1.726999479e+20  1.09e-02
 10   hfact                                1.0555869         1.050262424      5.04e-03
 56   t_tf_superconductor_quench          35.3199295        32.47607803       8.05e-02
 59   f_a_tf_turn_cable_copper             0.7380005121      0.7223124459     2.13e-02
109   f_nd_alpha_thermal_electron          0.03359040614     0.02935880415    1.26e-01
```

**This is a result, and it should be read precisely.** The driver did not take a null
step, and it should not have been expected to: PROCESS's answer is a KKT point of
PROCESS's problem, and §10.4/§10.5c show the port's objective differs from PROCESS's by
`1.73 %` at that very point, with a correspondingly different gradient. The port
minimised **its own** objective by `1.52 %` and stopped at a feasible KKT point of **its
own** problem to `7.4e-09` — an order of magnitude tighter than PROCESS's own `2.4e-07`.
Five of the eight variables agree to better than `1.1e-02`; the three that move most
(`f_nd_alpha_thermal_electron`, `t_tf_superconductor_quench`,
`f_a_tf_turn_cable_copper`) are the ones whose constraints are least active. The honest
statement is:

> **the port's optimiser solves the problem the port's graph states, quickly and
> tightly; it does not reproduce PROCESS's `x`, and the largest identified reason it
> should not is a documented inconsistency in PROCESS's own converged state.**

Whether the residual difference is *entirely* that, or partly the undiagnosed `c62`
derivative row, is not established. Do not claim the port reproduces PROCESS's
optimisation.

**C3 — cold start from the input file's own values. Does not run, and the blocker is not
the optimiser.** 0 SQP iterations: `pyvmcon`'s QP fails at the first iterate. Diagnosed
by evaluating the `ConditionMap` at the cold point directly — **20 of the 24 conditions
are `nan`, and 340 of the 408 Jacobian cells with them**, before any solver step is taken.

The cause is the SAND block's **context**, not its unknowns. `SingleRun.__init__` runs
`init_process` and nothing else, so every model-computed field in the cold
`DataStructure` is still its dataclass default: `.physics.vol_plasma = 0.0`,
`.physics.b_plasma_total = 0.0`, `.tfcoil.dr_tf_wp_with_insulation = 0.0`,
`.tfcoil.dx_tf_wp_primary_toroidal = 0.0`, and so on. The `Drive` reads **330** context
variables; the schedule's `Call` steps produce many of them, but not all, and the rest are
seeded from those zeros, which divide. The obvious alternative — run the plain MDA at the
cold design first and seed from *its* output — fails earlier still: `Intersect`'s
`NewtonDriver` gets a non-finite input, because its `KNOWN_MINT_VALUES` seed is
`.tfcoil.dr_tf_wp_with_insulation`, which is `0.0` before any model has run.

So the accurate statement is: **the port cannot yet run its own pipeline from a cold
input file at all**, with or without an optimiser. `mda_harness.py` never had to face
this because it always seeds from a converged run. Closing it is a graph-completeness
question (every SAND context variable produced by a `Call` step from real inputs), not an
`Optimise`-layer one, and it is the prerequisite for the only check that would license
"the port reproduces PROCESS's optimisation from scratch".

### 10.7 Two constraints that had to be un-blocked first, and how

§2.4 recorded c16 and c24 as **INERT**, and §5.3's plan quietly assumed a 12-constraint
problem. That is not a comparison worth making: an `Optimise` over 12 of PROCESS's 14
active constraints is a different problem, so `sand.constraint_nodes` now **raises** on
any active `icc` entry it cannot assemble, and a caller who wants the reduced problem must
say `omit={…}` and gets it back in the report. Both gaps were closed instead:

- **c16** needed `.heat_transport.p_plant_electric_net_mw`. `PlantElectricProduction` was
  ported but refused by `to_graph` — it declares five fields as both `Output` and `Input`.
  That is **not** a modelling cycle: it is PROCESS's conditional-ownership pass-through,
  live **only on the `ireactor == 0` arm**, and `.costs.ireactor` is a static switch. New
  node `PlantElectricProductionReactor` is the `ireactor == 1` arm with the five dead
  reads simply not declared; `.costs.ireactor` becomes a two-armed `Switch` whose other
  arm is the existing `PowerProfilesOverTime` (a strict subset of its outputs, reading the
  two carried-over values as boundary inputs, which is what `ireactor == 0` genuinely
  does). This matters beyond c16: **`CostOfElectricity` reads
  `p_plant_electric_net_mw`**, so before this the run's own objective was a function of a
  boundary input along that whole path.
- **c24** needed `.physics.beta_total_vol_avg`. `StellaratorBetaAndRhoStar` was left
  unregistered because its `.physics.rho_star` collides with
  `DimensionlessPlasmaParameters` — a redundant duplicate write in PROCESS itself. Only
  `rho_star` was in conflict; dropping the whole node cost `beta_total_vol_avg` and
  `e_plasma_beta` their only producer as collateral. New node
  `StellaratorBetaAndStoredEnergy` is the same pure function with the third return value
  discarded.

### 10.8 Harness effect of this session, controlled

*Superseded.* A point-in-time before/after table. Current harness numbers live in
`_audit/next_steps.md`'s "Verified state" block. The two durable readings from it: the x4
fix added agreements and no disagreements, and **all 18 disagreements this session added
are the one `+17.604 MW` offset of §10.4**, checked arithmetically rather than asserted and
written up in `mda_harness.EXPLAINED_DISAGREEMENTS`.

### 10.9 Open, with what would settle each

1. ~~**`c62`'s derivative row.**~~ **Closed** — the cause was three fields with no producer,
   one node upstream; see §10.5c. The topological walk that found x4 found this too, which
   is now twice that method has worked.
2. **A genuine cold start (C3), i.e. §5.3's C3.** Blocked on graph completeness, §10.6.
   *Experiment:* list the SAND `Drive`'s 330 context variables that no `Call` step
   produces and that have no real input value, and close them.
3. **Whether C2's remaining `1e-2`–`1e-1` moves are entirely the `z_tf_inside_half`
   staleness.** *Experiment:* pin `.heat_transport.p_plant_electric_base_total_mw` to
   PROCESS's own `89.461` as a boundary input, re-solve C2, and see how much of the move
   survives. That is not a graph the port should ship, but it is a clean controlled test.
4. **Array-indexed iteration variables (IDs 125–136)** — unchanged from §1.3.
   `sand.iteration_variable_path` raises rather than approximating, and
   `test_sand.py` pins that.
5. **The upstream items of §8** are unchanged and none blocked this work. §8.1's
   `ConditionMap`-carries-the-problem fix turned out to be *less* urgent than argued
   (§10.2), and a fourth item joins them: **`Optimise` has nowhere to put per-condition
   scaling either**, and for SAND that is not optional (§10.2's measurement).

## 11. The tokamak (2026-08-27) — SAND stood up for `large_tokamak_eval.IN.DAT`

Measured in `process_port` against `cottax` pinned at `db4f025` (`git archive HEAD`),
on the tree at "The cold boundary, measured" (`06ba9d8d`) plus this session's changes:
tokamak graph **203 nodes**. Reproduce with

```bash
$PY functional_process/run_sand_harness.py --machine
```

### 11.1 The study, resolved from the file alone

`ixc = [4, 6]` (`temp_plasma_electron_vol_avg_kev`, `nd_plasma_electrons_vol_avg`),
25 `icc` with `n_equality_constraints = 2` (c1 beta consistency, c2 power balance),
`i_figure_merit = 7` (capital cost — `NumericsData`'s **default**; the file never sets
it), `epsfcn = 0.001`, and — the fact that frames every number below —
`i_process_run_mode = -2`: an **evaluation** file. PROCESS answers it with `fsolve`
(HYBRD) over the two equalities and merely *reports* the 23 inequalities
(`process/main.py:449-457`, `solver.py:365-409`). At its solution two of them are
violated: `c16` (net electric below the 400 MW requirement, normalised `+3.67e-02`)
and `c68` (`p_div_bt_q_aspect_rmajor` 10.495 > 10, `+4.95e-02`). PROCESS's own run:
1.9 s; `n_solver_iterations` stays 0 (fsolve does not count into it).

### 11.2 What the assembly had to say no to, and how it says it

Three reductions, all detected mechanically, all in the report rather than a log line:

- **The PF-coil ring is an array-unknown fixed point** —
  `^problem.pf_coil.ind_pf_cs_plasma_mutual.cycle` owns a circuit-by-circuit mutual-
  inductance *matrix* and the per-coil `n_pf_coil_turns` *vector* — and every
  per-condition seam of the SAND layer is scalar (`sand.array_valued_problems` names
  them: `scaled_problem`'s `jnp.stack`, the count-based eq/ineq split, per-unknown
  bounds, `residual_condition_scales`' `1/|u|`). `sand_harness.assemble` now drops such
  a problem exactly as it drops a degenerate one and reports it
  (`report["array_valued"]`); its loop-carried unknowns freeze at the stage seed
  (converged-MDA values for C2). Making the driver element-wise is a recorded later
  decision, same standing as §1.3's array-element `ixc` refusal.
- **Nine constraints read nothing the SAND block produces** — c24, c26, c27, c31, c32,
  c36, c60, c65, c72 — because their producers (TF stress/superconductor blocks in
  `process/models/tfcoil/superconducting.py`, the CS criticals in
  `process/models/pfcoil.py`) are unported, so every value read is a boundary constant
  (§11.5). Structurally that puts the constraint node *outside* the combined problem's
  SCC, where its `^cond.*` reaches neither the drive's body nor its context and
  `ConditionMap.__call__` dies on a `KeyError` (`sand.constraints_outside_block`'s
  docstring walks the seam). The assembly's second pass omits them **loudly** via the
  existing `omit=`/`report["omitted"]` machinery; an *equality* in that state is
  refused outright, never reduced away.
- Degenerate fixed points: **none** on this graph.

Final shape: 144-node drive, **10 unknowns** (2 design + 8 coupling), **25 conditions**
(objective + 2 equalities + 14 inequalities + 8 residuals), 367 context, 79 schedule
steps; 6 problems residualised.

### 11.3 Stage A — 14 of 17 exact, and the three that are not are known

Both real equalities sit at their absolute floor (`c1` −2.38e-14 vs −2.60e-14, `c2`
+1.25e-12 vs +1.25e-12 — residuals of a converged fsolve, relative comparison
meaningless). Every one of the 14 assembled inequalities is bit-exact or ~1e-13,
**including** the two PROCESS leaves violated (`c16` +3.674513054e-02, `c68`
+4.949055142e-02 — identical to PROCESS's own). The objective agrees to `1.33e-06`
relative. One residual is not zero at the seed: `^cond.physics.proton_rate_density`
= −0.125 — the MDA's Picard iterate of the density/fusion cycle parked within its own
`rtol`, not a wiring defect (the other seven residuals are 0 or −2e-16).

### 11.4 Stage B — the Schur-reduced Jacobian, and what the 1.00-rows are

Port `(25, 10)` full Jacobian, reduced onto the two design columns and compared with
PROCESS's own FD (per-cell Richardson bars, 10 sweeps): **9 of 17 rows agree to
machine/noise level** (c1, c5, c9, c30, c25, c33, c34, c35, c81); the objective and
c2, c15, c16, c62 agree to 1e-3–1e-1 (the missing-`pden_plasma_ohmic_mw` class); and
**three rows read `1.00e+00` relative — the port's total derivative is exactly zero
where PROCESS's is not**: `c8` (`pflux_fw_neutron_mw`), `c13` (`t_plant_pulse_burn`)
and `c68`. Each is a §11.5 missing producer *seen as a sensitivity*: the port's value
chain passes through a frozen boundary constant (`dr_fw_*` for c8's wall area,
`vs_cs_pf_total_burn`/`res_plasma` for c13, `p_div_bt_q_aspect_rmajor_mw` itself for
c68), so the design cannot move them. The same defect class Stage B was built to find
(§10.5), found on a second machine by the same method.

### 11.5 The missing-producer audit (the §10 lesson, run first)

For every read of every active constraint and the objective: resolved `VarPath`,
producer in the 203-node graph or boundary, and — for boundary reads — whether
PROCESS's own solve moves them (cold vs converged `DataStructure`). **17 variables
across 14 constraints have a producer PROCESS runs and the graph lacks** (the two
`nd_plasma_electrons_vol_avg` rows are `ixc` 6 itself, owned by the `Optimise`, not
gaps):

| read | PROCESS producer |
|---|---|
| `.physics.pden_plasma_ohmic_mw` (c2) | `plasma_ohmic_heating`, `physics/physics.py:1690` |
| `.physics.beta_thermal_vol_avg`, `.beta_toroidal_vol_avg`, `.beta_vol_avg_max` (c24) | `physics/physics.py:3811-3831` |
| `.pf_coil.j_cs_critical_flat_top_end` (c26), `.j_cs_critical_pulse_start` (c27), `.temp_cs_superconductor_margin` (c60), `.stress_shear_cs_peak` (c72) | `pfcoil.py:3631/3675/3679/3508` |
| `.tfcoil.j_tf_wp_critical`, `.temp_tf_superconductor_margin` (c33/c36) | `tfcoil/superconducting.py:2725/2749` |
| `.tfcoil.j_tf_wp_quench_heat_max` (c35), `.superconducting_tfcoil.vv_stress_quench` (c65) | `superconducting.py` quench block (`:1424`) |
| `.tfcoil.sig_tf_case`, `.sig_tf_wp`, `.sig_tf_cs_bucked` (c31/c32/c72) | `superconducting.py` stress block (`:2208-2271` etc.; `sig_tf_cs_bucked` is `None` even converged — never written at `i_tf_bucking = 1`) |
| `.physics.nd_plasma_pedestal_electron` (c81) | pedestal Greenwald scaling, `physics/profiles.py:318` — the tokamak instance of the pair `_audit/cold_boundary.md` records |
| `.physics.p_div_bt_q_aspect_rmajor_mw` (c68) | `physics/physics.py:818` |

Everything else resolves to a graph producer or a genuine input (bound values,
`f_p_alpha_plasma_deposited`, `beta_total_vol_avg` — fixed here because `ixc` 5 is not
active). Consistent with, and extending from the constraint surface's side,
`_audit/cold_boundary.md`'s 29-name overwrite list.

### 11.6 Stage C

- **C2, the declared study**: pyvmcon's first QP raises `QSPSolverException` ("no
  feasible solution") — `c68` is violated by +4.9% at PROCESS's own answer and its port
  gradient row is identically zero (§11.4), so no linearised step can satisfy it: **the
  full 25-constraint problem is locally infeasible at the reference machine, and
  PROCESS never noticed because evaluation mode does not enforce inequalities.**
  `VmconDriver` swallows the exception and returns the start (the harness now says so
  in so many words), which lands — vacuously — on PROCESS's `x` exactly.
- **C2, the fsolve analogue** (all 23 inequalities omitted — the problem PROCESS
  actually solved): **1 SQP iteration**, conv `2.163e-13`, `objf 0.928607253`
  (PROCESS's `objective_function` at the same point: `0.928606022`), `max|eq|
  1.25e-12`, and distance to PROCESS's converged design **1.4e-16 / 0.0** on
  `temp_plasma_electron_vol_avg_kev` / `nd_plasma_electrons_vol_avg`. With 2 dof
  consumed by 2 equalities the objective is inert and the comparison is exact — the
  port's SAND stays where PROCESS's fsolve landed, to machine precision.
- **C3 (cold)**: **not solved, by design of the harness** — the pre-solve probe finds
  **7 of 25 conditions non-finite at the cold seed** (`objf`, the `delta_eta` and
  `t_plant_pulse_burn` residuals, `c13`, `c16` = `nan`; `c33`, `c35` = `inf`) and
  stops. All seven trace to `_audit/cold_boundary.md`'s six boundary zeros / four
  unported producers: `r_tf_inboard_in/out = 0` → `a_tf_inboard_total = 0` →
  `j_tf_wp = inf` (c33/c35); `res_plasma = vs_cs_pf_total_burn = 0` →
  `t_plant_pulse_burn = nan` (its residual and c13); `dr_fw_* = 0` → the ccfe_hcpb
  chain → power/costs → c16 and the objective. A cold-start seeding rule is a
  separate, recorded decision — the same position `run_sand_harness._seed` took for
  the stellarator — and the four producers `cold_boundary.md` names are the actual
  fix.

### 11.7 What changed in the code

`sand.py`: `switch_values_for` (+ `SWITCH_PARAMETER_NAMES`) — per-run static switch
values read off the initialised `DataStructure`; `array_valued_problems`;
`constraints_outside_block`. `sand_harness.py`: `mda_env` seeds `^guess.*` off the
*driven* graph (`Assign` mints the ports; the old `starts_for`-on-the-undriven-graph
shape seeded none and every `Drive.role_data` raised); `assemble` gains
`switch_values`, the array-unknown drop, the external-constraint second pass.
`run_sand_harness.py`: `--machine`/`--input` (spelled as `run_mda_harness`'s, whose
`input_file` it imports); `_inputs_only` at the solve call (cottax's owned-name guard;
`mdf._inputs_only`'s pattern); a pre-solve finiteness probe that stops a stage whose
seed the models cannot evaluate; Stage B keyed by assembled conditions rather than
`icc` position. `test_sand.py`: six tokamak analogues (file transcription, switch/
contract drift both ways, design-vars-are-boundary, full assembly order, PF-ring
detection). The stellarator ladder re-run after all of it is **numerically identical
line for line** (tmp paths aside) to the run before these changes; note its C2/C3 on
the *current* tree at `db4f025` run to `max_iter = 100` oscillating around
`objf ≈ 1.218` — the recorded 31/99-iteration numbers are from `ef093ba` and the
156-node graph, and nobody had been able to run SAND at `db4f025` before the
`mda_env` fix. **[That last sentence is right and its reading of "oscillating" is not
— §12 bisects both axes: the solve converges on the known optimum at 326/258
iterations, and `max_iter = 100` is all that stopped it.]**

## 12. The stellarator "oscillation" (2026-08-27) — a solve stopped two-thirds through

§11.7's closing note reported the stellarator C2/C3 running to `max_iter = 100`
"oscillating around `objf ≈ 1.218`", against a recorded **31 / 99** iterations. Both
axes that had moved since — the cottax version and the PROCESS tree — were bisected
independently, off one cached PROCESS run so every comparison shares a seed.

### 12.1 The cottax axis is inert

`run_sand_harness` (stellarator, current tree), cottax pinned by `git archive`:

| pin | shape | Stage A | C2 | C3 |
|---|---|---|---|---|
| `ef093ba` | — | — | — | — (does not import: `cottax.problem.Driven` does not exist yet, and `sand.py` imports it) |
| `b7c5572` | 14 unknowns × 21 conditions | 11 of 15 exact | 100, unconverged | 100, unconverged |
| `f0bf9bb` | identical | identical | identical | identical |
| `db4f025` | identical | identical | identical | identical |

The three reports that run are **byte-identical apart from wall-clock timings** — Stage
A to the last digit, Stage B's whole table, and every one of the 200 traced iterates. So
"slsqp + theory" and the owned-name guard moved nothing here, and the driver stack is not
the cause. **`ef093ba` cannot be measured at all**, which matters: the recorded 31/99 was
taken against it, and the port can no longer be built against it.

### 12.2 It is not oscillation — it is a cap

Raise the cap and the same solve finishes, on the same point:

| | SQP iterations | conv | `objf` | `x109` |
|---|---|---|---|---|
| C2, `max_iter = 100` | 100 | 4.5e-02 (last) | wandering | — |
| C2, `max_iter = 400` | **326** | **8.8e-11** | **1.217757338** | 0.0299518325 |
| C3, `max_iter = 400` | **258** | **8.0e-09** | **1.217757452** | 0.0299518175 |

§11.11's free optimum is `objf 1.217757336` at `x109 = 0.0299518`, and both stages land
there — C2 and C3 agree with each other to six digits on all eight `ixc`. What
`max_iter = 100` produced was a solve stopped two-thirds of the way through, and the
tail of an unconverged VMCON trace is indistinguishable from oscillation when read as
one.

### 12.3 Why the count grew, measured

The pre-round-2 tree (`62bc7048`, the commit before "A switch selects an occupant"),
with the one `mda_env` seeding fix backported so it can run at `b7c5572` at all, on the
**same cached PROCESS run and the same seeds**:

| tree | SAND block | C2 iterations | `objf` | `x2` |
|---|---|---|---|---|
| `62bc7048` (pre-round-2) | 22 unknowns, 16 equalities, 12 inequalities | **131** | 1.217757351 | 4.70928485 |
| current | 14 unknowns, 8 equalities, 12 inequalities | **326** | 1.217757338 | 4.70928438 |

Same point to seven digits; 2.5× the iterations. The cause is round 2 itself, and it is
**a correctness improvement, not a defect**: ten `FixedPoint`s dissolved when the
topology switches became slots, five of them to empty slots where PROCESS's own body is
*"x is an input"*. **Five of the ten were inside the SAND block**, and between them they
carried eight coupling unknowns: `etath_liq_step`, `temp_turbine_coolant_in_step`,
`p_fw_div_heat_deposited_mw_step`, `p_fw_blkt_coolant_pump_mw_step` (one each) and
`cryo_q_loads_step` (four — the cryo `q*` node's, `.power.qac` among them). Their
equalities were nearly-linear `u = g(u)` rows through the power system: cheap for the
SQP, and an easy subspace for its BFGS. Removing them leaves the same problem with its
nonlinearity concentrated into fewer conditions, and those quantities computed inside
the block's acyclic body instead — which is a step from SAND towards MDF for exactly
that part of the graph, and §11.10 already records that VMCON finds MDF harder.
The recorded 31 is from a third configuration again (a 156/159-node
graph, cottax `ef093ba`, and an `mda_env` that seeded `^guess` off the *undriven* graph
— a path that no longer mints those ports), so it is not reproducible even in principle.

### 12.4 Two things ruled out, so the next reader does not re-run them

- **Condition scaling is not the lever.** `^cond.stellarator.wp_width_r_min` — the
  `Intersect` residual `VmconDriver.condition_scale` already names as "the largest row
  left and the only one whose units are genuinely *not* its unknown's" — is the argmax
  of `|eq|` in **290 of 326** iterates (and 127 of 131 on the pre-round-2 tree, so it is
  not new). Sweeping its factor, everything else held:

  | factor | C2 iterations | outcome |
  |---|---|---|
  | 1.0 | 296 | **not converged** (best conv 2.9e-04) |
  | 1.3948 (`1/\|u\|`, the current rule) | 326 | converged, 8.8e-11 |
  | 0.1 | 219 | converged, 3.7e-09 |
  | 0.01 | 235 | converged, 5.4e-09 |
  | 0.001 | 246 | converged, 6.4e-09 |

  A 219–326 spread is inside this problem's own noise (§10.2 and
  `residual_condition_scales` record 33–73 across an earlier sweep). The rule is wrong
  in *kind* for a `RootFind` residual — `1/|u|` is derived for `g(u) − u` and this row's
  units are the interpolated curve's, not the abscissa's — and it is still not what
  costs the iterations. Left alone deliberately.
- **It is the problem, not the solver.** `SlsqpDriver` on the identical problem is also
  short of the optimum at 60 iterations (`x109` 0.03175 against 0.0299518) and reaches it
  given more. Two independently written SQPs, two QP solvers, two line searches, same
  answer about how hard this is — §11.10's own experiment, re-run.

### 12.5 What changed in the code

`run_sand_harness.SAND_MAX_ITER = 500`, threaded to the driver through
`sand.sand_schedule(max_iter=)` and `mda.default_drivers(max_iter=)` — the same shape
`bounds`/`callback`/`condition_scale` already travel, and `None` keeps `VmconDriver`'s
own default so MDF and every other `Optimise` are untouched. **The driver's default of
100 is left as it is**: it is PROCESS's `n_iteration_max` for PROCESS's own
eight-variable problem, which is the right default for anything reproducing PROCESS and
the wrong one for a block PROCESS never solves.
`test_sand.py::test_max_iter_reaches_the_driver_and_defaults_to_the_drivers_own` pins
both halves.
