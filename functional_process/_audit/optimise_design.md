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
  **[The conclusion holds, the evidence does not: §13.5 measures SLSQP stopping short at
  **87**, not 60, and shows it is SLSQP's own `ftol` — a single *absolute* threshold
  checked against the step size and the objective change, not a KKT test — firing at a
  feasible but non-optimal point. `x109 = 0.03175` does not reproduce in any of six
  configurations. §13.6 also finds the two solvers do *not* agree about the difficulty
  once the residual scaling is varied: unscaled, SLSQP converges in 142 and pyvmcon
  declares victory in 7 at the wrong point.]**

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

## 13. Two SQPs and PROCESS, timed (2026-08-27)

§12 left two numbers standing on one leg each: *"`SlsqpDriver` is also short of the
optimum at 60 iterations"*, and the implicit comparison between the port's **326**
SQP iterations and PROCESS's **46**. This section measures both properly -- the two
drivers on the *identical* assembled problem with the jit discipline stated, and
PROCESS's own solve of the same machine with the tolerance and the formulation
difference put side by side rather than left to the reader.

Measured on the tree at `f1e2ccfb`, cottax pinned at `db4f025` by `git archive` onto
`PYTHONPATH` (the live `~/jaxgraph` working tree is dirty; the archive is not), in
`process_port` with `jax_enable_x64` on and `JAX_PLATFORMS=cpu`. Note `jax 0.11.0`
here, where `CLAUDE.md` records 0.11.1 -- drift in the env, not in the tree, and §12.1
already showed this axis inert.

### 13.1 Method, so every number below can be read

- **One PROCESS run seeds everything.** `sand_harness.reference_run` has no cache of
  its own, so the run timed in §13.2 is a genuine uncached solve; it is pickled to the
  session scratchpad afterwards and every solver run below starts from that same
  `ReferenceRun`. The C2 path is `run_sand_harness.main`'s, function for function
  (`mda_env` -> `assemble` -> `sand_schedule` -> `_seed(design=every unknown)` ->
  `_inputs_only`), so the assembled problem is the one §12 measured: **124-node drive,
  14 unknowns, 21 conditions (objective + 8 equalities + 12 inequalities), 319 context,
  42 schedule steps**, nothing degenerate, nothing array-valued, nothing omitted.
- **Both drivers receive that same problem**, through `sand.sand_schedule(..., driver=)`:
  same graph, same C2 seed, same `bounds`, same `condition_scale`. `scaled_problem`
  exists for exactly this (`drivers.py`'s own docstring), so only the solver differs.
- **Jit discipline.** The two callables timed are the two each driver builds per solve --
  `jax.jit(flat_conditions)` and `jax.jit(jax.jacfwd(flat_conditions))` over a flat
  unknown vector. The **first** call (trace + compile + one execution) is reported
  separately and excluded from the steady state; every timed call afterwards is wrapped
  in `jax.block_until_ready`. Medians of 50 repeats (value) and 20 (Jacobian), with
  min/max shown because the max is a scheduler artefact, not a distribution.
- **Compile inside a solve is not excludable by warming.** Each driver builds its jits
  as fresh closures inside `__call__`, so JAX's cache misses and the whole 7.6 s is paid
  inside SQP iteration 1. Two numbers therefore travel together everywhere below: the
  **total** wall clock (compile included -- what a user waits) and the **steady state**,
  derived from the callback timestamps between iteration 2 and the last, which excludes
  compile by construction.
- **Evaluation counting is non-invasive**: a `pyvmcon.solve` wrapper for `VmconDriver`,
  a `drivers.scaled_problem` wrapper for `SlsqpDriver`. "eval" means the same thing for
  both -- **one condition value *and* one `jacfwd`, at one distinct point** (SLSQP's
  driver caches the pair per point, so its `nfev` and its distinct-point count coincide).
- **`max|eq|` is always reported in `VmconDriver.condition_scale`'s units**, including
  for the unscaled runs, so the column compares across the table. Unscaled it is
  meaningless: `^cond.physics.fusden_alpha_total`'s factor is `2.3e-18`.
- Wall clocks are `time.perf_counter`; the two headline configurations were run **three
  times** and are reported min/median/max, the tolerance sweep once each (iteration
  counts are deterministic and were identical across repeats; only timings moved).

### 13.2 PROCESS, re-measured

| | |
|---|---|
| VMCON iterations | **46** |
| wall clock (`SingleRun.run`) | **109.05 s** (`2.37 s` per iteration) |
| convergence parameter | `2.396e-07` |
| `epsvmc` (its stopping rule) | `1.0e-06` -- the `NumericsData` default; the file never sets it |
| `maxcal` | 100 |
| `epsfcn` | 0.01 (the file's own; the default is 1e-3) |
| `objf` | `1.166072708` (`i_figure_merit = 6`, cost of electricity) |
| problem | **8** `ixc`, **2** equalities + **12** inequalities |

The recorded 94 s is the same run on a quieter machine; 109 s is what it costs here, and
the ratios below use the number measured in the same session as the port's.

Two details of PROCESS's stopping rule that matter for §13.4. Its convergence parameter
first drops below `epsvmc` at **iteration 44** (`1.402e-07`) and it keeps going to 46,
because `force_vmcon_inequality_satisfication = 1` adds
`all(ie >= -1e-8)` to pyvmcon's test (`solver.py:207-243`) -- a second criterion
`VmconDriver` does not pass. PROCESS also pins the QP solver
(`qsp_options={"solver": cvxpy.CLARABEL}`) where `VmconDriver` takes pyvmcon's default.
Neither is a defect; both are reasons the two iteration counts are not the same
experiment even before the formulation differs.

Where the 2.37 s per iteration goes is the entire point of the rewrite: one VMCON
iteration costs `1 + 2n = 17` full pipeline evaluations (`fcnvmc1` plus `fcnvmc2`'s
forward/backward differences at `epsfcn`), and each of those re-runs the whole
Gauss-Seidel loop up to 10 times.

### 13.3 The jit probe -- what one evaluation of the port's block costs

| callable | first call (trace + compile + exec) | jitted min / **median** / max |
|---|---|---|
| condition map (21 conditions from 14 unknowns) | `1.757 s` | 0.211 / **0.259** / 2.479 ms (n=50) |
| `jax.jacfwd` of it (the `(21, 14)` Jacobian) | `5.834 s` | 0.442 / **0.484** / 1.623 ms (n=20) |

**Compile 7.59 s once; 0.74 ms per value-and-Jacobian pair thereafter**, 0 non-finite
cells. `jacfwd` costs 1.87x the value map for 14 columns -- forward-mode over a block
whose width is its unknown count, which is the shape cottax's `Blocking` was supposed to
buy. Against PROCESS's 2371 ms per iteration for the same job, that is **~3200x** on the
evaluation alone, and it is why the rest of this section is about the QP.

### 13.4 The two drivers, C2, `max_iter = 500`

`objf* = 1.217757336` is §11.11's free optimum. "dist" is the largest relative
difference from pyvmcon's own answer over the 14 unknowns.

| driver | its | outcome | `objf - objf*` | `max\|eq\|` | dist | wall (s) | steady ms/it | evals |
|---|---|---|---|---|---|---|---|---|
| **pyvmcon**, scaled | **326** | conv `8.80e-11` | `+2.46e-09` | `1.55e-08` | -- | 11.34 / **11.93** / 12.17 | 12.9 / **13.1** / 14.3 | 651 |
| SLSQP `ftol=1e-8` (default) | 87 | status 0 "success" | `+8.09e-05` | `1.19e-13` | `7.33e-02` | 8.26 | 5.9 | 558 |
| SLSQP `ftol=1e-10` | 87 | status 0 | `+8.09e-05` | `1.19e-13` | `7.33e-02` | 8.66 | 7.2 | 558 |
| SLSQP `ftol=1e-11` | 88 | status 0 | `+8.09e-05` | `1.19e-13` | `7.33e-02` | 7.73 | 5.5 | 558 |
| SLSQP `ftol=1e-12` | 106 | status 0 | `+1.15e-05` | `9.51e-13` | `2.65e-03` | 8.80 | 5.6 | 578 |
| **SLSQP `ftol=1e-13`** | **150** | **status 0** | **`-6.54e-10`** | **`7.93e-14`** | **`5.61e-06`** | 8.19 | 5.4 | 811 |
| SLSQP `ftol=1e-14` | 500 | status 9, **iteration limit** | `-6.55e-10` | `5.55e-13` | `9.88e-06` | 11.80 | 8.3 | 4431 |
| SLSQP `ftol=1e-16` | 500 | identical to `1e-14`, line for line | | | | 11.95 | 8.6 | 4431 |

Both solvers reach the same point. At the tolerance where SLSQP actually converges to it
(`ftol=1e-13`), the two answers agree to **`5.6e-06` in the worst coordinate and `1.0e-07`
in the median** -- two independently written SQPs, two QP solvers, two line searches, one
optimum, and §12.2's `objf 1.2177573` confirmed a third time.

**Is the port solver-bound or evaluation-bound? pyvmcon: solver-bound, decisively.**
Excluding compile, its 651 evaluations cost **0.69 s** and everything else costs
**3.83 s** -- per iteration, **~2.1 ms evaluating and ~11.8 ms in the QP subproblem,
line search and host bookkeeping (90 %)**. `pyvmcon` builds and solves its QP through
`cvxpy` in Python at every iteration; at 0.74 ms per exact Jacobian the model has stopped
being the cost.

**SLSQP is the mirror image.** Its QP is compiled Fortran LSQ, so "everything else" is
0.55 s over 142 iterations (3.9 ms), while its line search asks for **5.5 distinct
points per iteration** against pyvmcon's 2.0 -- 4.9 ms of evaluation per iteration.
Roughly balanced, tilting evaluation-bound. It converges in **less than half** the
iterations for **more total derivative work** (785 Jacobians against 651).

### 13.5 "SLSQP short at 60" -- its own test, firing early, on a loose absolute tolerance

Not an iteration cap and not a different optimum. At its default `ftol=1e-8` SLSQP stops
at **87** iterations (its own cap is 100) reporting `status 0, "Optimization terminated
successfully"`, at a point that is **feasible and not optimal**: `max|eq| 1.19e-13`,
inequality margin `-6.2e-14`, and `objf` **`8.09e-05` above the optimum** with the design
`7.3e-02` off in `t_tf_superconductor_quench`. Its trace tail says exactly what its test
saw -- the objective change per iteration crawling `-1.6e-08`, `-1.6e-09`, `-1.8e-09` for
seven iterations before a last step fixed the residual and the run stopped.

That is `ftol`'s documented semantics, and scipy's own docstring is the citation: *"This
value controls the final accuracy for checking various optimality conditions; gradient of
the lagrangian and absolute sum of the constraint violations should be lower than `ftol`.
Similarly, computed step size and the objective function changes are checked against this
value."* One **absolute** number against four different quantities -- so on an objective
of order 1 with a `1e-4`-scale optimality gap still open, a step-size/objective-change
test at `1e-8` fires long before the KKT conditions do. `pyvmcon`'s parameter is not
comparable: it is `|df . delta| + sum |lambda_j c_j|`, a genuine multiplier-weighted
optimality measure.

**And `ftol` is not only a stopping threshold.** `scipy/optimize/_slsqp_py.py:464` sets
the LSQ subproblem's `tol = 10 * acc`, so tightening it changes the *trajectory*, not
just when the trajectory stops. Measured: `1e-8`/`1e-10`/`1e-11` all halt at the same
wrong point; `1e-12` runs to 106 and a nearer one; `1e-13` runs to 150 and the optimum;
`1e-14` never satisfies the test at all. There is a cliff between `1e-12` and `1e-13`,
not a knob.

**The recorded pair "60 iterations, `x109 = 0.03175`" does not reproduce.** Six
configurations were tried (scaled and unscaled residuals x `ftol` from `1e-8` to `1e-16`
x caps 100 and 500) and every one lands with `x109` in `0.0299518 +/- 3e-8`; none takes 60
iterations. The observation's *shape* -- SLSQP stopping short of the optimum and calling
it success -- is real and is measured above at 87 iterations; the two specific numbers
are from a configuration this tree no longer has, the same standing as §12.3's
irreproducible 31.

**Two things not to misread from a capped SLSQP run.** Its trace row *N* is not its
answer at `maxiter=N`: capped at 142 the driver returns `objf 1.217723993`, while the
callback at iteration 142 held `1.217757299` -- scipy's iteration-limit exit returns an
internal iterate one line-search step out of step with the last callback. And running
past its stopping test is not free: at `ftol=1e-14` SLSQP reaches the optimum around
iteration 147 (`max|eq| 6.7e-13`) and then, with nothing left to fire, **degrades
monotonically** -- `max|eq|` creeping `1.2e-13 -> 1.8e-04` between iterations 160 and 480
-- before snapping back at 498.

### 13.6 The condition scaling is `VmconDriver`'s, and it is wrong for SLSQP

Dropping `sand.residual_condition_scales` entirely (not §12.4's one-row sweep -- **every**
residual factor set to 1) and rerunning both drivers from the same seed:

| driver, residuals unscaled | its | outcome | `objf - objf*` | `max\|eq\|` | dist to pyvmcon-scaled | wall (s) |
|---|---|---|---|---|---|---|
| **SLSQP `ftol=1e-8`** (default!) | **142** | status 0 | `-6.55e-10` | `7.06e-12` | `5.26e-06` | 8.57 / **8.83** / 9.38 |
| pyvmcon | **7** | conv `6.68e-09` | `+1.76e-03` | `2.38e-09` | -- (a different point) | 8.09 |

Two findings, opposite directions:

- **SLSQP does better without the scaling**: at its own default tolerance it converges in
  142 iterations onto pyvmcon's answer, where the scaled problem at the same tolerance
  stops short at 87. The factors were derived for VMCON's merit function and its QP
  weighting (`VmconDriver.condition_scale`'s docstring records why), and SLSQP's
  BFGS-with-safeguard plus its own row handling do not want them.
- **pyvmcon without the scaling declares convergence in 7 iterations at the wrong
  point** (`objf` `1.76e-03` high). That is not a slow solve, it is a **false positive**:
  its convergence parameter is `sum |lambda_j c_j|`, the unscaled residual rows are
  enormous, the QP drives their multipliers to nothing, and the sum reads converged.
  §12.4's sweep of the single `wp_width_r_min` factor found only "not converged at 296
  iterations"; dropping all six is worse than that, and the scaling is load-bearing in a
  way that section did not see.

The consequence for the harness is recorded, not acted on: `residual_condition_scales`
is currently threaded to *whatever* driver `sand_schedule` builds. It is a
`VmconDriver` tuning parameter and the fact that `SlsqpDriver` takes the same field is a
seam, not an equivalence.

### 13.7 The comparison, stated with the formulation difference in it

**The two problems are not the same problem**, and no iteration count is honest without
that on the same page:

| | PROCESS | the port's SAND block |
|---|---|---|
| unknowns | **8** (`ixc 2, 3, 4, 6, 10, 56, 59, 109`) | **14** -- the same 8, plus 6 coupling unknowns PROCESS never exposes (`temp_plasma_ion_vol_avg_kev`, `wp_width_r_min`, `delta_eta` and three residualised `^hat.*` fixed points) |
| equalities | 2 -- `c2` and `c16`, by **position**: the file sets `n_equality_constraints = 2` and `icc` opens `2, 16, ...`, so `c16` is solved as an equality although `ConstraintManager` registers it `">="` (`constraints.py:639`). §1's positional-split warning, live on the reference run; the port reproduces it faithfully | **8** -- the same 2, plus 6 residual rows for the coupling |
| inequalities | 12 | 12, identical |
| architecture | **MDF** -- every evaluation re-converges the Gauss-Seidel MDA (up to 10 passes) | **SAND** -- the MDA is opened up and its coupling handed to the SQP |
| derivatives | finite differences, `2n` pipeline sweeps per iteration at `epsfcn = 0.01` | one exact `jax.jacfwd`, 0.484 ms |
| stopping | `epsvmc = 1e-6`, **plus** `all(ie >= -1e-8)` | pyvmcon `epsilon = 1e-8`, no second criterion |
| `objf` at its answer | `1.166072708` | `1.217757336` -- **not comparable**, this is §5.1's 17.6 MW self-consistency offset, not a worse optimum |

**Iterations to equal tolerance.** Running the port's pyvmcon to PROCESS's own stopping
rule instead of its own (identical across all three repeats):

| convergence parameter | port, SQP iterations | at t (s), one run |
|---|---|---|
| `<= 1e-4` | 165 | 9.6 |
| `<= 1e-5` | 196 | 10.0 |
| **`<= 1e-6` (PROCESS's `epsvmc`)** | **299** | 11.3 (10.7 -- 11.5 over four runs) |
| `<= 1e-7` | 307 | 11.6 |
| `<= 1e-8` (`VmconDriver`'s own) | 326 | 11.9 |

So the answer to *"is PROCESS also ~300?"* is **no, and matching the tolerance barely
moves it**: 299 against 46. Tolerance explains 27 of the port's 326 iterations; the other
253 are the formulation -- 14 unknowns and 8 equalities against 8 and 2, with the MDA's
coupling promoted into the SQP where MDF hid it inside every evaluation. §11.10 already
recorded that VMCON finds MDF easier, and §12.3 measured the same effect from the other
side (dissolving 5 `FixedPoint`s took C2 from 131 to 326). This is that, quantified
against PROCESS.

**The wall clock, which is the headline.**

| | wall clock | iterations | ms per iteration |
|---|---|---|---|
| PROCESS (MDF, FD, `epsvmc = 1e-6`) | **109.05 s** | 46 | 2371 |
| port, pyvmcon to `conv 8.8e-11` | **11.93 s** (7.1 s of it one-off compile) | 326 | 37 total, **13.1 steady** |
| port, pyvmcon to PROCESS's `1e-6` | **11.3 s** | 299 | -- |
| port, SLSQP unscaled to the same point | **8.83 s** (7.6 s of it compile) | 142 | 62 total, **5.7 steady** |

**9.1x end to end against PROCESS, and 12.4x with the better driver -- while taking 7.1x
more SQP iterations.** Per iteration the gap is **181x** (2371 ms -> 13.1 ms), and the
port throws most of that away again inside `cvxpy`: at PROCESS's own iteration count of
46 the port's evaluation cost would be 0.1 s.

Two caveats a reader of that table must carry. The port's number is the SAND solve only;
from a cold interpreter it is preceded by ~4.6 s of imports and ~8.0 s for the MDA seed
and assembly (both dominated by their own one-off jit compiles), so a cold end-to-end
port run is ~24.5 s against PROCESS's 109 s -- **4.5x**, still. And C2 *starts at
PROCESS's answer*: this is a comparison of solver cost on one machine, not of a cold
design study, which is C3's question and is measured at 258 iterations in §12.2.

### 13.8 What this changes, and what it does not

- The rewrite's efficiency case is **not** "fewer iterations" -- it is 7x more of them.
  It is that an iteration costs 13 ms instead of 2.4 s, and that the 13 ms is now **90 %
  QP solver**, i.e. the model has been optimised out of the critical path entirely. The
  next order of magnitude is in `pyvmcon`'s `cvxpy` QP, not in the port.
- **`VmconDriver` is not obviously the right default any more.** SLSQP, unscaled, at its
  own default tolerance, reaches the same optimum in 142 iterations and 8.8 s. What keeps
  `VmconDriver` the reference is unchanged and is not about speed: it is *the same
  solver PROCESS calls*, which is what makes any disagreement attributable to the model
  and the derivatives (its own docstring's argument). A recorded decision, not a defect.
- §12.4's *"it is the problem, not the solver"* survives in the form that matters -- both
  SQPs need >>46 iterations on the SAND formulation and land on the same optimum -- but
  its evidence, "SLSQP is also short at 60", was a **stopping-rule artefact**, and its
  conclusion that the two solvers agree about the difficulty is only half right: 142
  against 326 is a factor of 2.3 between them, and the condition scaling flips which one
  wins.
- Nothing here was changed in the port. The measurement scripts stayed in the session
  scratchpad.

## 14. MDF's stellarator C2 (2026-08-29) — a second cap, and an `objf` row arbitrated

§16.1b of `next_steps.md` reported `run_mdf_harness` at **C2 200 iterations, not
converged** and **C3 60, not converged**, against the Verified-state table's *129,
converged* / *200, not converged*, and attributed the move to the 2026-08-27 wave day.
Both halves of that are wrong, and the first one is wrong in the way §12 was already
wrong once: **C2 was not failing, it was stopped at its cap two-thirds of the way
through a 523-iteration solve.** C3 was never capped at all.

### 14.1 Method — so a bisect step costs 40 s and not 3 minutes

`run_mdf_harness.main` spends most of its wall clock on things Stage C does not need:
one uncached PROCESS solve (99 s), PROCESS's 40-sweep finite-difference Jacobian, a
15 s central difference and the `call_models` cost probe. The bisect therefore ran a
probe that is `_measure`'s path function for function — `mdf.assemble` -> `mdf.seed` ->
`mdf.prime` -> `mdf.solve(bounds=, callback=, tolerance=1e-8, max_iter=MAX_ITER)` — with
the PROCESS side pickled once.

- **Pickled: `data`, `cold`, `ixc`, `icc`, `n_equality`, `i_figure_merit`, `converged`.**
  Not the `ReferenceRun` dataclass, whose fields move across the range and whose
  `bounds` carry `VarPath`s; `bounds` is rebuilt per commit from `cold.numerics.boundl/u`
  through that commit's own `sand.iteration_variable_path`. The cache is sound because
  the diff of `process/` between `3cb0843b` and `a95ee891` is **empty** — PROCESS itself
  did not change anywhere in the probed range.
- **Two axes, pinned independently.** The repo axis by a detached checkout in this
  worktree with `PYTHONPATH=<worktree>`; the cottax axis by `git archive` from
  `~/jaxgraph` into the scratchpad, ahead of the editable install on `PYTHONPATH`. Both
  were verified per run by printing `mdf.__file__` and `cottax.__file__` — the first
  attempt silently resolved `functional_process` off the *main* checkout through the
  editable install and produced four meaningless "byte-identical" results before the
  print caught it.
- The probe reproduces the harness **to the digit** at HEAD (C2 200/`conv 1.132e-02`/
  `objf 1.227934766`, C3 60/`conv 7.916e-02`/`objf 1.222852961`), which is what licenses
  using it in place of the harness.

### 14.2 The repo axis is inert from `1db889f6` to HEAD, and so is cottax

Every cell is C2/C3 at `MAX_ITER = 200`, `tolerance = 1e-8`, from the one cached PROCESS
run:

| tree | cottax | shape: nodes / inner driven / inner unknowns | C2 | C3 |
|---|---|---|---|---|
| `a95ee891` (HEAD) | live (`~/jaxgraph` working tree) | 165 / 5 / 6 | **200**, not conv, `conv 1.132e-02`, `objf 1.227934766` | **60**, not conv |
| `a95ee891` | `f0bf9bb` | 165 / 5 / 6 | identical, every digit | identical |
| `1db889f6` (round 2) | `f0bf9bb` | 165 / 5 / 6 | identical, every digit | identical |
| `62bc7048` (its parent) | `f0bf9bb` | **174 / 12 / 16** | **144**, not conv, `conv 4.133e-02`, `objf 1.225749709` | **200**, not conv |
| `62bc7048` | `b7c5572` | 174 / 12 / 16 | identical to `f0bf9bb` | identical |
| `682f9266` | `b7c5572` | 174 / 12 / 16 (340 inputs) | **113**, not conv | **58**, not conv |
| `62bc7048` | live | — | **does not run**: `db4f025`'s owned-name guard, the thing `ce5725c8` fixed | — |
| `3cb0843b` | `ef093ba` | 171 / 13 / 17 | **does not run**: `TypeError` inside `PicardDriver`, `ConditionMap` got 0 unknowns | — |

So: **the wave day is not the cause and neither is anything after it.** The whole day,
`ce5725c8`, `2e50246c`, `fa9804ad`, `8baec007` and every cottax version from `b7c5572`
through today are *byte-identical* on these numbers. The one commit that moves them is
**`1db889f6`, "A switch selects an occupant, and ten fixed points were the switch"**
(2026-08-26 22:09) — the same commit §12.3 already identified as the cause of SAND's
131 -> 326.

Two of the three commits the brief flagged as suspects are in that inert set
(`ce5725c8`, `2e50246c`); `728af014` is not in this repo's history at all.

### 14.3 What `1db889f6` did to MDF's problem

The shape column above is the whole mechanism. Round 2 turned ten topology switches into
slots, and seven of the driven blocks that dissolved were inside the MDA that MDF
re-converges at every trial point:

| | `62bc7048` | `1db889f6` and after |
|---|---|---|
| graph nodes | 174 | 165 |
| inner blocks | 156 | 154 |
| **inner driven blocks** | **12** | **5** |
| **inner unknowns** | **16** | **6** |
| schedule inputs | 334 | 310 |

MDF's *outer* problem is untouched — 8 design, 15 conditions, 2 equalities, 12
inequalities, before and after. What changed is the function those conditions are:
quantities that used to be converged by a `PicardDriver` inside the MDA are now computed
in the acyclic body, and PROCESS's own body for several of them is *"x is an input"*.
This is §12.3's finding seen from MDF's side, and it lands harder here because MDF's
whole formulation is the MDA.

### 14.4 It is a cap. C2 converges at 523, on SAND's own answer

`MAX_ITER` raised to 3000, everything else held (HEAD, live cottax; run twice, identical
to every digit, 16.5 s and 26.7 s — the second overlapped the SAND harness):

| | SQP iterations | outcome | `objf` | `max\|eq\|` |
|---|---|---|---|---|
| C2, `max_iter = 200` | 200 | **not converged**, `conv 1.132e-02` | 1.227934766 | 3.714e-03 |
| C2, `max_iter = 3000` | **523** | **converged**, `conv 7.400e-09` | **1.217758052** | 3.813e-13 |

`objf* = 1.217757336` is §11.11's free optimum and §12.2's SAND answer. The convergence
parameter is **not monotone** — it first touches `1e-4` at iteration 8 and then wanders
for four hundred more — so a "first below tolerance" reading of the trace is
meaningless; the endgame is abrupt (`1e-6` at 512, `1e-8` at 522). That non-monotonicity
is exactly why the capped tail read as oscillation, the same misreading §12.2 corrected
for SAND.

**And the two architectures agree.** MDF's 523-iteration C2 answer against the SAND
harness's own C2 (326 iterations, re-run this session and byte-identical to its pin):

| ixc | SAND | MDF | rel |
|---|---|---|---|
| 2 | 4.709284379 | 4.709291744 | 1.56e-06 |
| 3 | 26.63924217 | 26.63924531 | 1.18e-07 |
| 4 | 5.723905428 | 5.723838522 | 1.17e-05 |
| 6 | 1.73146923e+20 | 1.731494805e+20 | 1.48e-05 |
| 10 | 1.049850069 | 1.049847113 | 2.82e-06 |
| 56 | 31.76081658 | 31.76071645 | 3.15e-06 |
| 59 | 0.7177501169 | 0.7177490444 | 1.49e-06 |
| 109 | 0.0299518325 | 0.0299517401 | 3.08e-06 |
| `objf` | 1.217757338 | 1.217758052 | 5.86e-07 |

Two formulations of the same graph — one exposing the coupling to the SQP, one hiding it
inside every evaluation — reaching the same point to `1.2e-07 .. 1.5e-05`. That is a
stronger statement about the port than either number alone.

**Round 2 made MDF's C2 better, not worse.** Uncapped on the parent commit
(`62bc7048` x `f0bf9bb`, `max_iter = 3000`): C2 still stops at **144** — the cap was
never what held it — at `objf 1.225749709`, with `x4` **pinned at its lower bound of 3**
and `x109 = 0.0249` against the optimum's `0.0299518`. So the pre-round-2 configuration
does not reach the optimum at all, and the post-round-2 one does. The correct summary is
not "C2 stopped converging"; it is *"C2 started converging, and took 523 iterations to do
it, and the harness's cap was 200."*

### 14.5 C3 is not a cap, and never was: `pyvmcon`'s QP goes infeasible at 60

C3 at `max_iter = 3000` stops at **60**, at the same point, with the same trace.
Wrapping `pyvmcon.solve` says why:

```
STAGE C3: 60 callbacks, max_iter 3000
pyvmcon raised: QSPSolverException -- QSP failed to solve, indicating no feasible
solution could be found.
```

`VmconDriver.__call__` catches `VMCONConvergenceException` (of which this is one) by
design, keeps `e.x`, and reports the failure "out of band through `callback`" — but the
callback only carries a per-iteration convergence number, which cannot distinguish
*gave up* from *ran out*. So "converged: False" was two entirely different events
printed identically, and the natural response to the C2 one (raise the cap) is inert for
the C3 one. This is a real seam in `VmconDriver`'s reporting and it is **not** fixed here
— `drivers.py` is shared with SAND and changing its callback protocol moves SAND's
harness too. What is fixed is that the MDF report now says which happened
(`run_mdf_harness._why_it_stopped`).

The cold solve has never converged in any configuration measured: `62bc7048` x `f0bf9bb`
uncapped runs to **274** and also stops on the QP, at a badly wrong point
(`x56 = 1.043` against 35.32, `x109 = 0.0900` against 0.02995). Post-round-2's 60 is a
worse *count* and a much better *point*. Open, and now stated precisely enough to work
on: the cold MDF problem defeats `pyvmcon`'s QP subproblem, and the next experiment is
`SlsqpDriver` on the identical problem (§13.6's shape) plus a look at whether a bound is
active where the QP dies.

### 14.6 The recorded "129, converged" cannot be rebuilt, and is not a target

`129 / 200` entered the Verified-state table at `a33b4afa` (2026-08-19) and was
re-asserted "byte-identical" at `682f9266` (2026-08-26) against cottax `ef093ba`. Neither
configuration runs today: `ef093ba` predates the `Equality`/`Inequality`/`Objective`
roles `mdf.py` has imported since `682f9266`, and the closest tree that *does* import
them (`3cb0843b`) dies inside `PicardDriver` when run against it. The nearest
reconstructible ancestors give **113** (`682f9266` x `b7c5572`) and **144**
(`62bc7048` x `b7c5572`/`f0bf9bb`), both **not converged**, both stopping on the QP
rather than on a cap.

That is the same standing §12.3 gave SAND's recorded 31 and §13.5 gave its "SLSQP short
at 60": **a number from a configuration this tree no longer has.** The honest record is
not "129 regressed to 200" but "129 was never re-measured after round 2, and the quantity
it named stopped existing".

### 14.7 Stage B's `objf` row — the port's AD is right, and the gap is §10.4 in derivative form

Stage B stars the `objf` row in 7 of 8 columns at 1.26e-02 .. 1.51e-01 relative, while
Stage B0 reports the port's Jacobian self-consistent to 3.31e-06. Those are compatible,
and the first thing to notice is that **Stage B0's `alive` mask barely covers this row**:
it keeps cells above `1e-8 x |finite|.max()`, and the maximum over the whole `(15, 8)`
matrix is 34.03, so the objf row contributes 6 of the 60 live cells and its `x6` column
(`1.58e-20`, because `x6 ~ 1.7e+20`) is excluded by construction. The row therefore
needed §11.11's own method run on it directly.

The port's AD against **the port's own central difference** of the same condition, with
Richardson bars, at four step sizes (unscaled, i.e. `d(objf)/dx` in the port's own
coordinates):

| ixc | AD | CD `eps=1e-2` | `1e-3` | `1e-4` | `1e-5` | rel at `1e-5` |
|---|---|---|---|---|---|---|
| x2 | 4.363959e-02 | 4.359522e-02 | 4.363953e-02 | 4.363959e-02 | 4.363959e-02 +- 4.0e-10 | **2e-10** |
| x3 | -8.624002e-02 | -8.514369e-02 | -8.497998e-02 | -8.623974e-02 | -8.623973e-02 +- 9.7e-11 | **3e-06** |
| x4 | -6.306003e-01 | -6.262961e-01 | -6.246770e-01 | -6.306004e-01 | -6.306003e-01 +- 7.0e-10 | **7e-10** |
| x6 | -1.576264e-20 | -1.565315e-20 | -1.576270e-20 | -1.576264e-20 | -1.576264e-20 +- 1.7e-29 | **4e-10** |
| x10 | 0 | 0 | 0 | 0 | 0 | — |
| x56 | -1.269556e-04 | -1.269653e-04 | -1.269557e-04 | -1.269556e-04 | -1.269556e-04 +- 5.0e-11 | **2e-09** |
| x59 | -2.323351e-01 | -2.323217e-01 | -2.323350e-01 | -2.323351e-01 | -2.323351e-01 +- 2.4e-09 | **2e-11** |
| x109 | 5.819851e+00 | 5.520309e+00 | 5.819851e+00 | 5.819851e+00 | 5.819851e+00 +- 5.8e-08 | **2e-10** |

**`jacfwd` is the `h -> 0` limit of the port's own map**, to 2e-10 in seven columns and
3e-06 in `x3`. The port's derivative is not in question.

The same row against PROCESS, in PROCESS's spelling (columns divided by
`reference.scale`), with PROCESS's own Richardson error bar from
`process_jacobian_with_error`:

| ixc | port AD | port CD `1e-5` | PROCESS FD | PROCESS err | rel | **in units of PROCESS's own error bar** |
|---|---|---|---|---|---|---|
| x2 | 2.400177e-01 | 2.400177e-01 | 2.370392e-01 | 2.63e-04 | 1.26e-02 | **11x** |
| x3 | -1.724800e+00 | -1.724795e+00 | -1.652839e+00 | 2.66e-03 | 4.35e-02 | **27x** |
| x4 | -4.414202e+00 | -4.414202e+00 | -4.032380e+00 | 2.59e-03 | 9.47e-02 | **147x** |
| x6 | -3.152528e+00 | -3.152528e+00 | -2.879600e+00 | 1.12e-03 | 9.48e-02 | **243x** |
| x10 | 0 | 0 | 0 | 1.57e-12 | 0 | 0 |
| x56 | -4.443446e-03 | -4.443446e-03 | -4.357456e-03 | 3.33e-07 | 1.97e-02 | **259x** |
| x59 | -1.626346e-01 | -1.626346e-01 | -1.551626e-01 | 9.26e-06 | 4.82e-02 | **807x** |
| x109 | 5.819851e-01 | 5.819851e-01 | 5.055766e-01 | 1.30e-02 | 1.51e-01 | **5.9x** |

**So this is not resolution, on either side.** It is 11 to 807 times PROCESS's *own* FD
error bar, and the port's side is exact to 1e-9. The two are differentiating **different
functions**, and which function differs is already recorded: §10.4's `+17.604 MW`
self-consistency offset. `Buildings.run` recomputes `a_plant_floor_effective` from
`z_tf_inside_half`, `Stellarator.run(output=True)` leaves the report-pass value of one
and the solve-pass value of the other in the same `DataStructure`, and the port — reading
that structure and being internally consistent about it — computes a net electric power
982.4 MW where PROCESS's stored one is 1000.0. That is why `objf` itself differs by
1.73e-02 at the reference point (Stage A) and why `c16`, an *equality* here by PROCESS's
positional split, sits at `1.760e-02` where PROCESS has `3.8e-09`. A function that
differs has a derivative that differs; the per-column ratios (1.013, 1.044, 1.095, 1.095,
1.020, 1.048, 1.151) are not one constant, so the offset is not a pure rescaling of the
objective — which is what one would expect of an offset entering through a denominator
alongside a cost numerator that also moves.

**Is this what made C2 wander? No.** C2 converges, in 523 iterations, on the same point
SAND reaches from a completely different formulation. A wrong objective gradient would
not do that. The `objf` row is a *separate, already-explained* finding and it belongs to
§10.4's ledger, not to the solve.

### 14.8 Should MDF's C2 reproduce PROCESS's `x`? No — and it reproduces SAND's

C2's converged design is 1.2e-03 to 1.08e-01 from PROCESS's per iteration variable,
worst at `x109` (1.08e-01) and `x56` (1.01e-01). Every part of that is accounted for and
none of it is a solver defect:

- **PROCESS's answer is a converged point of PROCESS's own problem**, which is not the
  stated one: its objective is 17.604 MW inconsistent with its own stored state (§10.4),
  and c16 is off by `1.76e-02` there. A point where an equality is violated by 1.8 % is
  not a feasible point of the port's problem, so no correct optimiser could stay at it.
  `sand_harness.py`'s module docstring and §11.6 already say this; §14.7 measures it in
  the gradient.
- **`x56` and `x109` are flat directions** (§11.11), so even the residual difference is
  concentrated exactly where the objective barely cares.
- **The cross-check that does apply is SAND**, and it passes to 1.2e-07 .. 1.5e-05
  (§14.4). Two architectures, two condition sets, two blockings, one answer.

The right expectation for MDF C2 is therefore "converges, agrees with SAND, and differs
from PROCESS by §10.4 plus the flat directions" — which is what it now does. It is not
"reproduces PROCESS's `x`", and the Verified-state table should never have implied it.

### 14.9 What changed in the code

- **`run_mdf_harness.MAX_ITER` 200 -> 800.** Measured, not rounded: C2 needs 523, and 800
  gives it the same ~1.5x margin `SAND_MAX_ITER = 500` gives SAND's 326 (§12.5). The
  driver's own default of 100 is still left alone, for §12.5's reason.
- **`run_mdf_harness._why_it_stopped`**, printed on the `converged:` line: the convergence
  test passed / it reached the cap / the driver stopped short at N of MAX_ITER because
  `pyvmcon` raised. Three days of reading one boolean as one event is what this exists to
  prevent. Pinned by
  `test_mdf.py::test_the_harness_distinguishes_a_cap_from_a_driver_that_gave_up`, which
  also pins `MAX_ITER > 523`.
- **`mdf.py`'s module docstring**, whose structural counts (174 nodes, 131-node SCC,
  twelve problems, 111 inner blocks, 11 driven) were `62bc7048`'s and had been stale
  since round 2. Now 165 / 123 / five / 112 / 4, with the old numbers and their cause
  named so the change is not silently lost.
- **Nothing in `drivers.py`, `sand.py` or `sand_harness.py`.** The stellarator SAND
  harness was re-run this session as the control: **C2 326, C3 258**, byte-identical to
  §11.11's pins.

Before and after, `run_mdf_harness` (stellarator), same machine, same session:

| | before | after |
|---|---|---|
| C2 | 200 iterations, **not converged**, `conv 1.132e-02`, `objf 1.227934766` | **523 iterations, converged**, `conv 7.400e-09`, `objf 1.217758052` |
| C3 | 60 iterations, not converged | 60 iterations, not converged — **and the report now says `pyvmcon` raised, not that it ran out** |

## 15. Why the port took ten times PROCESS's iterations (2026-08-30) — it was the QP solver

The open question `next_steps.md` §16.10 ranked second, taken deliberately: PROCESS
converges the Helias stellarator in **46** VMCON iterations; SAND took **326** and MDF
**523**. The recorded suspicion listed three candidates — `epsfcn = 0.01` smoothing,
the QP solver, and §10.4's self-consistency defect. It was the second, and it was not
a subtle version of it.

### 15.1 The four arguments that differ

Both sides call the **same** `pyvmcon.solve`. That was the whole point of choosing
`pyvmcon` (this file's §4, and `VmconDriver`'s class docstring): a different SQP would
confound "the port's model differs" with "the port's optimiser differs". So the search
space is exactly the arguments, and there are only four:

| `pyvmcon.solve` argument | PROCESS | the port, before today |
|---|---|---|
| `qsp_options` | `{"solver": cvxpy.CLARABEL}` (`solver.py:253`) | **not passed** |
| `epsilon` | `data.numerics.epsvmc` = `1e-6` | `1e-8` |
| the Jacobian | `Evaluators.fcnvmc2`, forward/backward FD at `epsfcn` | `jax.jacfwd` |
| `initial_B` | `I`, and `2I` on a retry that only fires if VMCON never iterated | `I` |

`pyvmcon.solve_qsp` ends `qsp.solve(**{"solver": cp.OSQP, **options})`. **Not passing
`qsp_options` is not "taking PROCESS's default" — it is taking `pyvmcon`'s, which is a
different solver.** Every SQP subproblem in every port solve ever recorded was answered
by OSQP, a first-order ADMM method whose `cvxpy` defaults are `eps_abs = eps_rel = 1e-5`,
where PROCESS used CLARABEL, an interior-point method returning ~1e-9. VMCON's search
direction *is* the QP solution and `calculate_new_B`'s quasi-Newton update is driven by
the multipliers the QP returns, so this does not show up as a wrong answer. It shows up
as many more iterations to the same one — which is exactly the symptom that was recorded.

### 15.2 The measurement

Stellarator, `stellarator_helias.IN.DAT`, one `reference_run()` per matrix, both
formulations, both starts. PROCESS: 46 iterations, `conv 2.396e-07`, `epsvmc 1e-6`,
`epsfcn 0.01`.

| formulation | start | QP solver | `epsilon` | iterations | conv | outcome |
|---|---|---|---|---|---|---|
| MDF | C2 warm | OSQP | 1e-8 | 523 | 7.40e-09 | converged |
| MDF | C2 warm | OSQP | 1e-6 | 513 | 3.22e-07 | converged |
| MDF | C2 warm | CLARABEL | 1e-8 | 45 | 7.23e-02 | **QP `infeasible`** |
| MDF | C3 cold | OSQP | 1e-8 | 60 | 7.92e-02 | QP raised |
| MDF | C3 cold | CLARABEL | 1e-8 | 67 | 2.54e-09 | converged |
| MDF | C3 cold | CLARABEL | 1e-6 | **58** | 8.91e-07 | converged |
| SAND | C2 warm | OSQP | 1e-8 | 326 | 8.80e-11 | converged |
| SAND | C2 warm | OSQP | 1e-6 | 299 | 8.45e-07 | converged |
| SAND | C2 warm | CLARABEL | either | 207 | 5.95e-02 | **QP `unbounded`** |
| SAND | C3 cold | OSQP | 1e-8 | 258 | 8.03e-09 | converged |
| SAND | C3 cold | CLARABEL | 1e-8 | 83 | 8.08e-10 | converged |
| SAND | C3 cold | CLARABEL | 1e-6 | **58** | 7.43e-08 | converged |

`C2/OSQP/1e-8 = 326` and `C3/OSQP/1e-8 = 258` reproduce the previously recorded SAND
baselines exactly, so the harness is measuring what it always measured. Nothing hit a
cap in any cell.

**Cold, at PROCESS's own tolerance, both formulations take 58 iterations against
PROCESS's 46.** Two independent formulations of the same machine — SAND with 14
unknowns and 21 conditions, MDF with 8 and 15 — landing on the identical count is worth
more than either number: what remains is 1.26x, and it is no longer a solver artefact.

**`epsilon` is not the lever.** `1e-8 -> 1e-6` buys 27 of SAND's 326 and 10 of MDF's
523. The tolerance explanation recorded in §16.10 accounted for the part of the gap it
was measured on and no more; it was never going to reach 10x.

### 15.3 The gradient hypothesis, killed

`VmconDriver.epsfcn` was added to settle the remaining candidate — that PROCESS's 1 %
perturbation smooths a non-smooth function and that its "bad" gradients are therefore
an advantage. It reproduces `Evaluators.fcnvmc2`'s quotient exactly, in the driver's
own scaled coordinates. MDF, cold, CLARABEL, `epsilon = 1e-6`:

| Jacobian | iterations | objf |
|---|---|---|
| `jax.jacfwd`, exact | **58** | 1.217757951 |
| PROCESS's FD, `epsfcn = 0.01` | 193 | 1.218210202 |
| PROCESS's FD, `epsfcn = 0.001` | 314 | 1.217712816 |

The exact Jacobian is **3.3x better than PROCESS's own**, and the trend is monotone in
the wrong direction for the smoothing story. Whatever PROCESS's 46 buys, it is not
bought by its finite differences. This is the second time the harness has been asked
whether PROCESS's derivative is secretly load-bearing and answered no.

**The warm start inverts it, and that is the interesting half.** Same configuration,
starting from PROCESS's converged x: exact `jacfwd` fails at 45 on an `infeasible` QP,
while `epsfcn = 0.01` converges in **39** and `epsfcn = 0.001` in 66. So the smoothing
is not helping PROCESS solve faster; it is helping it survive a start where the exact
linearisation is inconsistent. That is a robustness property, not a speed one, and it
is the only thing found today that argues *for* PROCESS's derivative.

### 15.4 CLARABEL is not uniformly better, and the failures are informative

From PROCESS's own converged point, CLARABEL stops in both formulations and OSQP does
not. The statuses are different, and should not be merged:

- **MDF C2: `infeasible`** — the linearised constraints have no common point inside
  the bounds.
- **SAND C2: `unbounded`** — the QP objective descends without limit along a feasible
  ray, which requires `B` to have lost positive definiteness. VMCON's equation 9 exists
  precisely to prevent that, and `pyvmcon.calculate_new_B` logs `"The B matrix has at
  least one negative eigenvalue ... This will cause a crash next iteration!"` when it
  happens. SAND's coupling unknowns carry no bounds (`VmconDriver.bounds`: an unknown
  with no entry is unbounded on both sides), so there is a ray for it to find.

OSQP reports neither in any *stellarator* cell. Its three `user_limit`s across the warm
SAND cells are ADMM hitting its own iteration cap and returning the iterate regardless
— the mechanism, made visible: **at the subproblems where CLARABEL refuses, OSQP
quietly returns a direction it has not verified, and the solve survives on it.** That
the port's numbers were ever obtained is partly an accident of that leniency.

**OSQP is not blind to infeasibility in general, and the tokamak proves it.** The full
`large_tokamak_eval` matrix — all eight cells, both starts, both solvers, both
tolerances, ~90–102 s each — is *identical*: **0 iterations**, first QP `infeasible`,
`QSPSolverException`, in every one. So what OSQP is lenient about is narrower than the
paragraph above on its own suggests: **ill-conditioned but feasible subproblems**, where
it returns an uncertified iterate instead of a status. Where the linearisation is
genuinely inconsistent it says so, cleanly, the same as CLARABEL.

**The tokamak therefore carries no evidence either way on the QP-solver question, and
now demonstrably so rather than by omission.** Its SAND block never takes a first step
from *either* start. That is the full-constraint infeasibility §16.9 in `next_steps.md`
already recorded and `_why_no_step` already localises — structural, upstream of any
solver choice — and it is consistent with the previous session's result that only the
reduced fsolve-analogue (2 equalities, the inequalities omitted) solves cold. PROCESS
gives 0 VMCON iterations here regardless, because the tokamak run is `fsolve`, so there
was never a baseline to compare against.

One incidental finding worth keeping: the returned x is `1.43e-16` from PROCESS's
converged answer in the C2 cells and `1.68e-02` in the C3 cells — i.e. **each start
handed straight back, unmoved**. That is the signature of `VmconDriver` swallowing
`QSPSolverException` and returning `e.x`, and at the C2 start (which *is* PROCESS's x)
it reads as a perfect agreement with PROCESS unless something checks the iteration
count. `_why_no_step` exists for exactly this ambiguity; a harness reporting `worst_rel`
without it would have reported a triumph.

Why the warm start is hard at all is not mysterious once looked at: `initial_B = I` in
scaled coordinates makes the first step essentially the negative gradient, and PROCESS's
converged x is *near* but not *at* the port's optimum, so the first steps are large. The
MDF C2 trace blows up to `objf 2.81, conv 5.96` by iteration 5 before recovering. Cold,
where nothing is nearly stationary, the identity Hessian costs less.

### 15.5 What did not change, and one thing to look at next

`worst relative deviation of the port's x from PROCESS's` is **1.08e-01 in every
converged cell** — both formulations, both QP solvers, both starts, both tolerances —
while every one of them agrees on `objf` to six digits (1.21775...). The 11 % sits on
`f_nd_alpha_thermal_electron` (ID 109) and 9.9 % on `t_tf_superconductor_quench` (56);
`rmajor` agrees to 2.0e-03 and `b_plasma_toroidal_on_axis` to 1.1e-03. That is a flat
objective in two variables, not a disagreement the solver introduced, and it is
untouched by everything in this section. It wants its own look.

### 15.6 Landed

- `VmconDriver.qsp_solver`, defaulting to `"CLARABEL"` — PROCESS's choice, and the one
  under which the cold solve (the only start PROCESS ever takes) works in both
  formulations. **The warm regression is real and is not fixed**: C2 in both harnesses
  now stops short where it used to converge. That is a correct report of a degenerate
  QP replacing a concealed one, but it is a visible behaviour change, and choosing what
  to do about it — a fallback ladder, bounds on SAND's coupling unknowns, a better
  `initial_B` — is a design decision, not a measurement. See `next_steps.md` §17.
- `VmconDriver.epsfcn`, defaulting to `None`.
- `condition_scale`'s "a fifth to a third of QP subproblems solving inaccurately"
  withdrawn: `optimal_inaccurate` appears **zero times in 1854 subproblems** across the
  whole SAND matrix. It was an OSQP artefact — ADMM reports `user_limit`.

## 16. The cold-start matrix, and twenty-two producers that were never there (2026-08-30)

§17.4 of `next_steps.md` carried two items: the cold start across the full matrix, and
"c72's infeasibility", **promoted to blocking** because every cell of the tokamak SAND
ablation returned zero iterations on an `infeasible` first QP. Both are answered here.
The answer is not about the solver, the formulation, the seed or the QP: **the port's
graph does not produce everything PROCESS computes**, and c72 is what that looks like
from the far end of a long chain.

This section records three wrong answers before the right one, at length, because each
was persuasive, each was argued from real measurements, and the thing that killed each
was cheap and could have been run first.

### 16.1 The matrix, cold, at PROCESS's own `epsvmc`

Four reference configurations (IFE excluded; the two ST files blocked per §18), both
formulations, three drivers, `tolerance = 1e-6` throughout so counts are comparable with
PROCESS's own.

| configuration | form | driver | it | conv | feasible at the answer |
|---|---|---|---|---|---|
| `stellarator_helias` | MDF | CLARABEL | 58 | 8.909e-07 | **yes** |
| `stellarator_helias` | MDF | OSQP | 60 | 7.916e-02 | no (worst ie +2.5e+00) |
| `stellarator_helias` | MDF | SLSQP | 26 | -- | **yes** |
| `stellarator_helias` | SAND | CLARABEL | 58 | 7.434e-08 | converged |
| `stellarator_helias` | SAND | OSQP | 214 | 2.481e-07 | converged |
| `stellarator_helias` | SAND | SLSQP | 112 | -- | converged |
| `large_tokamak_nof` | MDF | CLARABEL | **0** | -- | first QP empty |
| `large_tokamak_nof` | MDF | OSQP | **0** | -- | first QP empty |
| `large_tokamak_nof` | MDF | SLSQP | 6 | -- | no (ie **+3.9e+02**) |
| `large_tokamak_nof` | SAND | CLARABEL | 2 | 1.352e+01 | no (ie +1.3e+02) |
| `large_tokamak_nof` | SAND | OSQP | 2 | 1.352e+01 | no (ie +1.3e+02) |
| `large_tokamak_nof` | SAND | SLSQP | 35 | -- | no (ie +9.0e+01) |
| `low_aspect_ratio_DEMO` | MDF | CLARABEL | **0** | -- | first QP empty |
| `low_aspect_ratio_DEMO` | MDF | OSQP | **0** | -- | first QP empty |
| `low_aspect_ratio_DEMO` | MDF | SLSQP | 15 | -- | no (ie +7.1e+02) |
| `low_aspect_ratio_DEMO` | SAND | CLARABEL | **0** | -- | c90, §16.7 |
| `low_aspect_ratio_DEMO` | SAND | OSQP | **0** | -- | c90, §16.7 |
| `low_aspect_ratio_DEMO` | SAND | SLSQP | 23 | -- | no (ie +2.5e+02) |
| `large_tokamak_eval`* | MDF | CLARABEL | **0** | -- | first QP empty |
| `large_tokamak_eval`* | MDF | OSQP | **0** | -- | first QP empty |
| `large_tokamak_eval`* | MDF | SLSQP | 500 | -- | worst-dx **3.39e-12**, ie +2.8e+02 |
| `large_tokamak_eval`* | SAND | CLARABEL | **0** | -- | c72 +2.73e+02, constant |
| `large_tokamak_eval`* | SAND | OSQP | **0** | -- | c72 +2.73e+02, constant |
| `large_tokamak_eval`* | SAND | SLSQP | 500 | -- | worst-dx 3.38e-12, ie +2.7e+02 |

\* evaluation-mode file: PROCESS reports `0 iterations, conv 0` and solves it with
`fsolve` over the equalities alone, so these six cells are **not** a comparison against
PROCESS. §16.9 of the previous section measured its fsolve analogue (MDF 3 iterations
under both QP solvers, SLSQP 4, worst-dx 3.33e-07).

The stellarator rows reproduce §15.2's numbers **exactly** (58/60, 58/214) from a harness
rebuilt from scratch after the machine lost power and cleared `/tmp`. That is what makes
the tokamak rows worth reading.

**The two `large_tokamak_eval` SLSQP rows are the single most informative cells here.**
They reach `worst-dx 3.4e-12` -- PROCESS's design to twelve digits -- with equalities at
`1.3e-15`, and still report `worst ie +2.8e+02`. The port lands exactly on PROCESS's
answer and correctly reports that PROCESS's answer violates three of its own
inequalities, which `fsolve` never examines. Everything the port computes is right
there; what differs is that the port looks.

**Caveat on this table's own feasibility column.** It applies an absolute `1e-6` to
conditions in whatever units they carry. That is right for MDF, whose conditions are all
PROCESS's normalised residuals, and **wrong for SAND**, whose residual equalities are in
physical units -- on a variable of order `1e20` a relative error of `1e-6` *is* `1e14`.
The `max|eq|` figures on SAND rows are therefore not evidence of anything and are omitted
above; read `conv` and `worst ie`. `condition_scale` exists for exactly this and the
report should apply it.

### 16.2 A recorded result, retracted: SLSQP does not solve `large_tokamak_nof`

`next_steps.md` recorded "SLSQP converges `large_tokamak_nof` in 36 iterations" and
reasoned from it that the port's problem was solvable and the blockage `pyvmcon`-specific.
**It is not converged.** The 35-iteration run stops with c72 violated by `+9.0e+01`. The
old harness had no feasibility check, so a run that merely *returned* was recorded as a
run that *solved*.

The inference built on it was the reason c72 was framed as a solver question at all, and
it is backwards. `pyvmcon.solve_qsp` (`vmcon.py:322-337`) builds the linearised
constraints as hard `cvxpy` constraints with bounds in `cp.Variable(bounds=...)` and
raises when `delta.value is None`: no elastic mode, no relaxation. SLSQP solves a relaxed
subproblem, never refuses, and returns an infeasible point. `pyvmcon` refusing was the
more honest report every time.

### 16.3 Three wrong answers, and what killed each

Recorded in full because the pattern is the lesson: in each case an algebraic argument
looked conclusive, was stated as established, and was overturned by a measurement that
cost minutes.

**(a) "The burn-time loop is degenerate."**
`vs_plasma_burn_required = v_loop * csawth * (t_ramp + t_burn)` (`physics.py:4886`),
`t_burn = |vs_cs_pf_total_burn| / v_loop - t_ramp` (`pulse.py:302`), `csawth` defaults to
`1.0` and neither file overrides it, and at PROCESS's converged point
`vs_plasma_burn_required = 273.6610190288024` against `vs_cs_pf_total_burn =
-273.6610190288024` -- **identical to sixteen digits**. Substituting gives `t_burn =
t_burn`. PROCESS appears to confess it, above the formula: *"N.B. t_plant_pulse_burn on
first iteration will not be correct ... but the value will be correct on subsequent
calls."*

Killed by measuring `d(t_plant_pulse_burn)/dx` with PROCESS's own finite differences at
its converged point: **strongly nonzero on 13 of 20 design variables**, largest
`2.035026e+04` on `dr_bore` (relative 2.83). `d(c13)/dx` is `d(t_burn)/dx / 7200` to
every digit -- c13 *is* the burn time against its floor -- which is how the optimiser
parks it at `t_plant_pulse_burn = 7200.0595` with c13 active at `+8.27e-06`.

**The sixteen-digit agreement is a bookkeeping tautology**: PROCESS computes the one
quantity from the other for reporting. An exact algebraic identity between two *reported*
quantities is not evidence about the rank of the system that produced them. The
distinguishing measurement is the derivative row, and it costs 40 pipeline evaluations.

**(b) "The cold design is genuinely infeasible; the port is honest and PROCESS is not."**
At PROCESS's *converged* design the port reproduces its whole state -- **983 of 1039
variables to 1e-9** on `large_tokamak_nof`, **989 of 1045** on `low_aspect_ratio_DEMO`,
worst real disagreement `4.4e-04`, and the burn-time/volt-second/CS-stress chain to six
digits. Cold, they diverge 55x. Since `Caller.call_models` runs at most ten Gauss-Seidel
passes and `check_agreement` only tests the objective and constraints, PROCESS's cold
state is plainly not converged -- so the port, which drives its loops, must be the
consistent one and the cold design must really be infeasible.

Killed by §16.4. The port's cold state is consistent with a **smaller model set**. The
983/1039 agreement is not evidence of correctness: at PROCESS's converged design
`mdf.seed` fills boundary inputs from PROCESS's own `DataStructure`, handing every
missing producer exactly the right value. **The measurement that looked like vindication
was taken at the one point where the defect is structurally invisible.**

**(c) "Seeding is the fix; SAND should tolerate an infeasible start."**
`_seed` fills coupling unknowns and `^hat.*` cuts from a completed MDA at the cold design
-- burn time `142424` where PROCESS's iteration 0 sees `2568`. SAND promotes coupling
into the optimiser, so it should escape. Reseeding all 1034 coupling values from
PROCESS's own cold state moves c72 from `+386` to `+1.8` under SLSQP: two orders of
magnitude, so the seed is genuinely doing damage.

But seeded from PROCESS's **converged answer** -- where c72 is active and satisfied --
SAND takes 2 steps and ends at c72 `+5.4e-01`. It is handed a feasible point and walks
away from it. If starting at the answer does not work, the start is not the problem.
(That row is §17.1's unmeasured C2 regression, and it is more serious than filed: not a
cosmetic diagnostic-stage wobble but SAND failing to hold a feasible point. §17.1's
cheapest hypothesis -- SAND's coupling unknowns have no `bounds` entries at all and are
unbounded both ways -- is still unmeasured and is the next thing to try.)

### 16.4 The actual finding: twenty-two producers are missing

The question none of the above asked: **does the port's graph produce everything PROCESS
computes?**

Method, appealing to no audit record: snapshot every numeric field of every area of a
`SingleRun`'s `DataStructure` before any model runs; evaluate one `Evaluators.fcnvmc1` at
the cold `x`; snapshot again. Everything that moved is something PROCESS's pipeline
computes. Intersect with the port's boundary `input` entries -- variables nothing in the
graph owns -- minus the run's iteration variables, which are boundary inputs by intent.

**PROCESS writes 789 of 2270 fields in one pass. Twenty-two of them are boundary inputs
of the port's graph.** Twenty are frozen at exactly `0.0`:

```
.physics.beta_poloidal_vol_avg          0  vs  1.0874279
.physics.dlamie                         0  vs  17.810652
.pf_coil.p_pf_electric_supplies_mw      0  vs  4.8813983
.pf_coil.temp_cs_superconductor_margin  0  vs  3.4208032
.pf_power.ensxpfm                       0  vs  17038.228
.pf_power.srcktpm                       0  vs  1113.0075
.tfcoil.sig_tf_case                     0  vs  5.9391981e+08
.tfcoil.sig_tf_wp                       0  vs  4.9699609e+08
.build.dr_tf_inner_bore                 0  vs  11.794021
.fwbs.dewmkg                            0  vs  14404818
... 22 total, every one disagreeing
```

`.tfcoil.sig_tf_case` and `.tfcoil.sig_tf_wp` appear **nowhere** in
`functional_process/models/`. Six of the twenty-two have an owning node somewhere and are
still unowned on the tokamak -- a wiring problem, not a porting one.

This is the defect class `boundary.py`'s docstring names: *"A slot whose new occupant does
not write what the old one wrote does not fail: its consumers silently fall back to
whatever value sits in the `DataStructure` ... That defect class has eight recorded
instances and not one of them was found by a check."* Twenty-two more, and again not
found by a check.

### 16.5 Why no existing stage could see it

`boundary.py` already counts the boundary and pins it -- 297 inputs on the stellarator,
361 on the tokamak -- and already says the number *"should only ever go down, because
everything else on it is a read whose producer is not ported yet"*. What it could not do
is tell the ~109 genuine inputs from the unported producers. A field PROCESS writes every
pass is definitionally the second kind.

`tokamak_boundary.md` did this analysis by hand for `large_tokamak_eval`'s traced surface
and got it right -- but by grepping `process/` for a writer and reasoning about whether
that writer was dormant. That is correct where it was applied, must be redone per
configuration, and cannot see a writer that fires for a reason the reader did not
anticipate. Running the pipeline and diffing is mechanical and complete.

And **Stage A / C2 cannot see it by construction**, per §16.3(b): they seed boundary
inputs from PROCESS's converged `DataStructure`. Only a cold start exposes a missing
producer.

### 16.6 What landed, and the honest result

`calculate_poloidal_beta` + `PoloidalBeta`, registered `.tokamak.plasma_beta.poloidal`.
The slot list ran `physics.py:3818-3822` (`toroidal`) then `3831-3835` (`thermal`) and
skipped `3825` between them. `constraint_48`'s docstring has recorded the hole since
`batch5.md` -- *"not yet ported anywhere in `functional_process`"* -- and ported over it,
because its own read was harmless. The read that was not harmless is in
`models/pfcoil/currents.py::calculate_equilibrium_currents`, inside
`log(8*aspect) + beta_poloidal_vol_avg + l_i/2 - 1.5`.

`boundary.computed_by_process` and `boundary.unproduced_but_computed` landed with it,
pinned at eighteen rows on the MDA graph (`missing_producers_tokamak.txt`; three more
appear once the constraint surface is added, because constraints declare reads the MDA
graph never makes). Tokamak boundary 361 -> 360.

**The port did not move the tokamak.** Re-measured after it landed: MDF `0/0/3`
iterations, SAND `2/2/35`, c72 still `+3.9e+02` / `+1.3e+02` / `+9.1e+01`. `conv` shifts
from `1.352e+01` to `1.364e+01` and MDF's SLSQP arm from 6 to 3 iterations, so the value
*is* consumed -- it is simply not decisive. The prediction that this was "the root of the
c72 chain" was an O(1)-term-in-an-O(1)-bracket argument and it is **not supported**: it
was stated before the switch arm (`i_pf_current`) was checked, and before the other
twenty-one were ruled out as dominating. Landing it was right; the causal claim was
another instance of §16.3's pattern.

### 16.7 Two more defects, independent of all of the above

**c90 blocks `low_aspect_ratio_DEMO` outright.** Both SAND cells stop at zero iterations,
`_why_no_step` naming `^cond.constraints.c90 +1.00e+00` -- the `1 - value/bound`
signature of `value = 0`. `cs_fatigue.ncycle` is unported and the port says so in three
places (`models/pfcoil/namespace.py:97`, `models/pfcoil/stresses.py:35-37`,
`models/tokamak/namespace.py:12`). `stresses.py` reads *"Neither is read by any active
constraint"* -- true when written, for the stellarator and `large_tokamak_nof`; DEMO
activates c90. Same shape as the array-element `ixc` refusal that cost three
configurations (`0dca2227`) and as `constraint_48`'s hole: a documented omission whose
justification quietly expired.

**`degenerate_fixed_points` cannot inspect any multi-node cycle.** It builds the body it
differentiates as `graph.subgraph(tuple(set(producers.values())))` -- the direct producers
only, one level deep. For a single-node cycle that is the whole body, and the two it was
written for (`EtaTurbineStep`, `CplifeAvail`) both were. For any longer cycle a kept node
reads something no kept node produces, `_run_acyclic` raises `KeyError`, and the bare
`except Exception: continue` -- commented *"undetectable is not degenerate"* -- files the
block as healthy. **Five of six driven blocks on `large_tokamak_nof` fail this way**,
including `^problem.times.t_plant_pulse_burn.cycle`, whose cycle spans eight nodes. So the
port has had no working degeneracy check on any non-trivial loop, in either formulation,
for as long as those loops have existed -- the second time this function's `except` has
silently reported every fixed point healthy (its docstring records the first). The body
wants `graph.ancestors` with the declared problem nodes excluded; including them
re-creates the cycles, which is its own `ValueError`. Not fixed here: it is a change on
SAND's assembly path, it wants its own test, and nothing above depends on it.

> **Fixed, and this paragraph's two numbers are wrong; §17 is the measurement.** The
> count is **one of six**, not five, and the exception is a `ValueError` from `jnp.stack`
> on an array unknown, not a `KeyError` from a short body. The conclusion above
> nevertheless *understates* the damage: the one-level body does not usually raise, it
> quietly freezes the missing inputs at `env`'s values and returns a number for a
> different function. On `large_tokamak_nof` that number is `J = -I` **exactly** for the
> burn-time cycle -- a perfectly conditioned, unambiguously healthy fixed point, where
> the real residual has `cond 9.1e+12`. Both defects had to be repaired before that block
> could be seen at all.

**`mdf.inner_residuals` crashes on the tokamak.** `float(np.asarray(env[unknown]))` at
`mdf.py:750` raises on any array-valued inner unknown, and the burn-time cycle owns
`n_pf_coil_turns`, an array. The one instrument for *"did the MDA converge"* -- whose
docstring says *"`PicardDriver` stops at `max_iter` silently"* -- has never been able to
run on the configurations that need it most. Same defect class as the `reference_run`
array bug (`d2890d90`).

> **Fixed. §17 is the measurement.** It reduces over the array now, keeping the
> element with the worst *relative* gap, and it has been run on all four assembling
> configurations at both the cold design and PROCESS's converged one.

### 16.8 What this changes

- **c72 is not a solver blocker and not a PF-coil porting bug.** It is the far end of a
  chain that starts at missing producers. §17.4's promotion of it to blocking was correct
  about severity and wrong about cause.
- **The cold start is the only place the port is tested against a graph it has to produce
  itself.** Everything else hands it PROCESS's answers. That makes cold starts the
  harness's most valuable stage, not its most fragile one.
- **The guard exists now** and is pinned; the number may only go down.
- Twenty-one producers remain, plus `ncycle`.

### 16.9 The three waves that followed, and the shapes of the same hole

Pinned at eighteen after the first landing (thirteen now). The next five were taken
together, and each is a different reason a producer goes missing:

| variable | PROCESS writes it at | producer landed | what the port used instead |
|---|---|---|---|
| `.physics.dlamie` | `physics.py:279-283`, inline in `Physics.run` | `.tokamak.physics.coulomb_logarithm` | `0.0` |
| `.physics.pflux_plasma_surface_neutron_avg_mw` | `physics.py:835-837`, inline | `.tokamak.physics.plasma_surface_neutron_flux` | `0.0` |
| `.fwbs.p_div_rad_total_mw` | `divertor.py:52-56` | `.tokamak.divertor.heat_flux_split` (extended) | `0.0` |
| `.blanket.deg_blkt_inboard_poloidal_plasma` | `hcpb.py:64-69` | `.tokamak.ccfe_hcpb.inboard_poloidal_angle` | `0.0` |
| `.buildings.dz_tf_cryostat` | `cryostat.py:58-60` | `.tokamak.cryostat` (extended) | **`2.5`** |

- **Inline arithmetic with no staticmethod around it** (the two `physics.py` rows). The
  wave that ported `physics.py` walked its `calculate_*` staticmethods; these two are
  three lines each written straight into `run()`, and nothing pointed at them.
- **A slot's boundary row is not the machine's** (`p_div_rad_total_mw`). `divertor.md`
  dropped `incident_radiation_power` *deliberately and with the reason recorded* —
  `tokamak_boundary.md`'s `.tokamak.divertor` row listed two reads and this was not one.
  Four nodes in the assembled machine read it. Same story, same wording, for the blanket
  angle: `blanket_library.py`'s module docstring called the two poloidal-angle helpers
  out of scope because none of their writes reaches `.tokamak.ccfe_hcpb`'s own boundary
  variables — while `.tokamak.divertor` was reading one of them. **The scope test wave 1
  applied was per-slot, and this defect class is per-machine.**
- **A device-conditional writer** (`dlamie`). Every grep hit outside `physics.py` is a
  *read*, two of them in `stellarator.py`, and the only write is in `Physics.run`, which
  `caller.py:272-275` returns before ever reaching on the stellarator arm. So the same
  `VarPath` is a genuine boundary input on one machine and a missing producer on the
  other, and it is correct for it to appear in `reference_boundary.txt` and not in
  `reference_boundary_tokamak.txt`.
- **An input PROCESS overwrites** (`dz_tf_cryostat`). The hard one. It is a real
  `InputVariable` (`core/input.py:377`, default `2.5`), so the port was not reading a
  suspicious `0.0` — it was reading a plausible number, `2.5` against PROCESS's
  `5.5730055`. Twenty of the original twenty-two rows could have been found by grepping
  the cold state for zeros; this one could not, which is the argument for measuring
  PROCESS's write set instead. It is nonetheless a producer and not an input: the
  overwrite is unconditional, `caller.py` runs the cryostat (`:351`) before the
  buildings (`:370`), and every other read of the field in `process/` is inside an
  `if output:` reporting block in `build.py`. No live read ever sees the input value.

**Two further waves landed the same day**, taking the pin from twenty-two to **five**
and the tokamak boundary from 361 to **353**:

- **build** (`.tokamak.build`'s `tf_top_height`, `blkt_upper_thickness`,
  `tf_inner_bore`): `.build.z_tf_top`, `.build.dz_tf_upper_lower_midplane`,
  `.build.dz_blkt_upper`, `.build.dr_tf_inner_bore`. `z_tf_top` is read by
  `TfCoilShapeDShapeSingleNull` (coil arc placement) and `pfcoil/geometry.py` (divertor
  PF coil placement), so at the cold `0.0` **the graph drew a TF coil whose top sat on
  the midplane**. Four producers landed with *zero* new declared reads.
- **PF power** (`.power.pf_coil_power`, the port of `Power.pfpwr`, wholly unported
  until now): `.pf_power.srcktpm`, `.pf_power.ensxpfm`, `.heat_transport.peakmva`,
  `.pf_coil.p_pf_electric_supplies_mw`, plus `.pf_coil.temp_cs_superconductor_margin`
  from `.tokamak.cs_coil.temperature_margin`. This wave moved the input count *up*
  (356 -> 357) because the node declares five genuine `IN.DAT` reads of its own --
  which is why the boundary total alone is not the measure and
  `computed_by_process` is.

**What is still missing: one row.** ~~`.costs.c2214`, `.costs.c2222`, `.costs.c2252`,
`.fwbs.dewmkg`,~~ `.tfcoil.str_wp`, plus `.tfcoil.sig_tf_case` and `.tfcoil.sig_tf_wp` on
the constraint surface only.

**`.costs.c2222` was the last of the four and the only one whose blocker was not a
missing model.** Its account had been ported and tested for two waves; what kept it off
the graph was the *node's shape*. `PfMagnetCost` carried `.costs.supercond_cost_model`
as a static kwarg across two arms whose strand-cost reads are disjoint, so registering it
would have declared four edges the reference run does not make -- and the fourth,
`.pf_coil.j_crit_str_pf`, was itself a field PROCESS computes and nothing owned, so the
registration would have *moved* this row rather than closed it. Splitting the node into
`PfMagnetCostPerKg`/`PerKam` and porting `superconpf`'s PF call
(`.tokamak.pf_coil.strand_critical_current`) closed both
(`cost_boundary_inputs.md` §13.4). The generalisation is worth carrying into the
remaining rows: **a node whose ports over-declare is as effective at holding a row on
this list as a node that does not exist**, and harder to see, because the row names the
account while the defect is in the edges. The pin went 2 -> 1, and the tokamak's input
half went 369 -> 375 on seven genuine `IN.DAT` reads -- the same "the total alone is not
the measure" point the PF-power bullet above makes.

The last three are all outputs of
`process/models/tfcoil/base.py::stresscl` -- 1053 lines, 65 parameters, a 224-line
`plane_stress` solver and an `argmax` reduction -- which is a unit with its own registry
row, not a slot to fill. **The cost of that absence is measured and it is not small:
constraints 31 and 32 are active on `large_tokamak_nof` and the port evaluates both as
`0 <= max`** -- dropped constraints, not merely wrong values, which no residual reports
-- and `str_wp = 0` is the *peak* of the Nb3Sn strain fit feeding constraints 33 and 36,
so the absence is optimistic. `tfcoil/base.md`'s standing note that `stresscl` "feeds
only stresses, which no boundary read depends on" was true of the ten boundary reads and
is false of the constraint surface.

---

## 17. Both instruments repaired, and what they measure (2026-08-30)

§16.7's last two defects are fixed (`functional_process/sand.py::fixed_point_residuals`,
`functional_process/mdf.py::inner_residuals`) and run on all four assembling
configurations at **both** the cold design and PROCESS's converged one. This section is
what they say. Nothing here is inferred; every number is a print from one run per
configuration, cross-checked between two independently-built instruments (the graph-side
`fixed_point_residuals` and a `Drive.condition_map`-side Jacobian taken off the MDF
schedule) that agree on every block they both see.

### 17.1 What was actually broken, corrected

Both §16.7 paragraphs got the mechanism right and the count wrong.

| claim in §16.7 | measured |
|---|---|
| "five of six driven blocks on `large_tokamak_nof` fail this way" | **one of six** -- `^problem.times.t_plant_pulse_burn.cycle`, on all three tokamaks |
| the failure is `_run_acyclic` raising `KeyError` | `ValueError: Cannot stack arrays with different numbers of dimensions: got (), (22, 22), (22,)` -- the **array** defect, in `jnp.stack(...).reshape(len(reads))` |
| "cannot inspect any multi-node cycle" | it inspects them and gets the **wrong answer**, which is worse |

The last row is the finding. A one-level body usually *runs*: the inputs its missing
nodes would have produced are already in `env`, so `_run_acyclic` reads them as frozen
constants and returns a Jacobian for a function that is not the block's residual.
Measured directly, old body against the transitive closure, same `env`, at PROCESS's
converged design on `large_tokamak_nof`:

| block | old body | this body | max cell difference |
|---|---|---|---|
| `^problem.times.t_plant_pulse_burn.cycle` | 3 nodes, `J = -I` exactly | 56 nodes | **1.31e+06** |
| `^problem.physics.proton_rate_density.cycle` | 3 nodes | 18 nodes | **2.15e-01** |
| `^problem.tokamak.cicc_...dr_tf_plasma_case` | 1 node | 3 nodes | 0 warm; **1.9e-02** cold |
| `^problem.power.delta_eta_step` | 1 node | 83 nodes | 0, exactly |
| `^problem.physics.profiles.ion_vol_avg_temperature` | 1 node | 1 node | 0, exactly |
| `^problem.tfcoil.dx_tf_wp_primary_toroidal` | 1 node | 12 nodes | 0, exactly |

`J = -I` for the burn-time cycle is the whole argument for the closure: the one-level
body reports `g` as **not depending on its own unknowns at all**, i.e. a perfectly
conditioned, full-rank, unambiguously healthy fixed point, where the real residual has
`cond 9.1e+12`. And it reports it as a *number*, so nothing distinguishes it from a
measurement. `delta_eta_step`'s 1-vs-83 row is the control: a genuine self-loop whose
83-node closure is all upstream of the unknown contributes exactly zero, so the closure
costs accuracy nowhere.

**The `except` is narrowed rather than removed.** `fixed_point_residuals` returns a
`FixedPointResidual` per block carrying either a Jacobian or the exception that stopped
it, and `degenerate_fixed_points` -- which `reference_problem` acts on -- now **raises**
if any block could not be measured, naming each. "Checked, healthy" and "could not check"
are different answers and a caller that cannot tell them apart will act on the wrong one.
After the fix no block on any of the four configurations is undetectable, so nothing
raises today; the guard is for the next one.

### 17.2 Every driven block, all four configurations

Rank of the residual Jacobian `d(g(u) - u)/du`, flattened over array unknowns, at both
seeds. `n` is the flattened unknown count.

| configuration | block | n | rank warm | rank cold | note |
|---|---|---|---|---|---|
| `stellarator_helias` | `physics.profiles.ion_vol_avg_temperature` | 1 | 1 | 1 | |
| | `power.delta_eta_step` | 1 | 1 | 1 | |
| | `physics.proton_rate_density.cycle` | 2 | 2 | 2 | cond 1.08 / 1.28 |
| | `fwbs.f_ster_div_single` | 1 | 1 | 1 | |
| | `stellarator.coils.intersect` (`RootFind`) | 1 | 1 | -- | |
| `large_tokamak_nof` | `tokamak.cicc_...dr_tf_plasma_case` | 1 | **0** | 1 | **degenerate warm** |
| | `physics.profiles.ion_vol_avg_temperature` | 1 | 1 | 1 | |
| | `power.delta_eta_step` | 1 | 1 | 1 | |
| | `physics.proton_rate_density.cycle` | 3 | 3 | 3 | cond 1.24 / 1.29 |
| | `tfcoil.dx_tf_wp_primary_toroidal` | 1 | 1 | 1 | |
| | `times.t_plant_pulse_burn.cycle` | 507 | 506 | 507 | cond **9.1e+12** / 7.6e+12 |
| `low_aspect_ratio_DEMO` | `tokamak.cicc_...dr_tf_plasma_case` | 1 | **0** | 1 | **degenerate warm** |
| | the three scalar self-loops | 1 each | 1 | 1 | |
| | `physics.proton_rate_density.cycle` | 3 | 3 | 3 | cond 1.15 both |
| | `times.t_plant_pulse_burn.cycle` | 507 | 506 | 506 | cond **1.6e+13** / 3.1e+13 |
| `large_tokamak_eval` | `tokamak.cicc_...dr_tf_plasma_case` | 1 | 1 | 1 | |
| | the three scalar self-loops | 1 each | 1 | 1 | |
| | `physics.proton_rate_density.cycle` | 3 | 3 | 3 | cond 1.15 both |
| | `times.t_plant_pulse_burn.cycle` | 507 | 506 | 506 | cond **9.8e+12** / 9.4e+12 |

**The burn-time cycle's `506` is not a finding and must not be read as one.** Its
smallest singular value is `2.3e-07` to `3.4e-07` in every cell above while the largest
is `2.6e+06` to `5.2e+06`, so `numpy.linalg.matrix_rank`'s default threshold
(`max(M, N) * eps * s_max`, i.e. `2.9e-07` to `5.8e-07` here) lands *on top of* `s_min`
and the verdict flips on rounding. The honest statement is the conditioning, and it is
the same at every seed on every tokamak: **a 507x507 residual block with
`cond ~1e13`**, four to five orders worse than any other block in the port. §16.3(a)
killed "the burn-time loop is degenerate" with a derivative measurement and was right to;
this is the weaker and better-supported successor claim, and it is the first time the
block has been looked at at all.

**`^problem.tokamak.cicc_superconducting_tf_coil.dr_tf_plasma_case` is genuinely
degenerate at PROCESS's converged design on two of the three tokamaks**, and this is not
a defect -- it is the port correctly detecting that PROCESS is treating the field as an
input there. `dr_tf_plasma_case_from_input` is `max(u, m)` with `m` independent of `u`
(`models/tfcoil/base.py:256-271`), so `dg/du` is `0` where the clamp binds and `1` where
it does not; where it does not, `r = g(u) - u` is constant and the SAND equality
determines nothing. `reference_problem` therefore deletes the problem warm and keeps it
cold on `large_tokamak_nof` and `low_aspect_ratio_DEMO`. **The consequence is worth
stating plainly: the SAND problem assembled at PROCESS's answer and the SAND problem
assembled cold are not the same problem on those two files** -- one unknown and one
equality apart -- which is a fact §16.3(c)'s "seeded at the converged answer, SAND walks
away" was measured across without anyone knowing. It is unchanged by this repair (the
old one-node body found this one too); it is recorded because nothing had recorded it.

### 17.3 Did the inner solves converge?

`mdf.inner_residuals` at `PicardDriver.max_iter = 20` (the default) and again at `200`,
seeded from PROCESS's converged `DataStructure` and from the cold one. Relative gap
`|g(u) - u| / |u|`, worst element of each unknown.

| configuration | seed | worst inner residual | converged? | any value move at cap 200? |
|---|---|---|---|---|
| `stellarator_helias` | warm | `0.00e+00` | yes | no value moved |
| | cold | `1.27e-08` (`^hat.fwbs.f_ster_div_single`) | yes | no value moved |
| `large_tokamak_nof` | warm | `2.79e-14` | yes | no value moved |
| | cold | **`3.10e-05`** (`^hat.pf_coil.n_pf_coil_turns`) | **no** | **yes, up to `4.9e-05` relative** |
| `low_aspect_ratio_DEMO` | warm | `1.06e-14` | yes | no value moved |
| | cold | **`2.64e-06`** (`^hat.pf_coil.ind_pf_cs_plasma_mutual`) | **no** | **yes, up to `5.6e-06` relative** |
| `large_tokamak_eval` | warm | `1.06e-14` | yes | no value moved |
| | cold | `9.88e-07` (`^hat.pf_coil.ind_pf_cs_plasma_mutual`) | borderline | **yes, up to `9.9e-07` relative** |

Every unconverged row is the burn-time cycle and nothing else, and it is unconverged
**only cold**. The cap raise is the confirmation rather than the detection: a block whose
answer moves when `max_iter` goes from 20 to 200 was not at its fixed point at 20, and
all three tokamaks move cold while nothing anywhere moves warm.

**What this costs.** MDF's whole correctness claim is that its conditions are PROCESS's
constraints *evaluated at a converged MDA*; a cold `large_tokamak_nof` evaluation is
therefore evaluating them `3e-05` relative off the fixed point of the largest and
worst-conditioned block in the graph, and `jacfwd` through a `lax.while_loop`
differentiates the iteration that ran rather than the fixed point it did not reach. That
is a plausible contributor to §16.1's `large_tokamak_nof` MDF rows and it is **not
established as the cause of anything** -- saying more would be §16.3's pattern again. The
cheap next measurement is §16.1's matrix re-run with `PicardDriver(max_iter=200)`; it was
not run here, and this section does not claim a result for it.

### 17.4 `mdf.assemble` does not call `degenerate_fixed_points`, and should not

Recorded because the question was asked. `mdf.py`'s module docstring already argues it:
an identity `FixedPoint` is a perfectly well-posed Picard problem (it converges in one
step from anywhere) and only becomes a rank-deficient *equality* because SAND residualises
it, so there is nothing for MDF to drop. Two things reinforce that and one qualifies it.

- Dropping the problem would **change the graph MDF optimises**, and the two formulations
  have to optimise the same one or §16.1's table compares nothing. `mdf.assemble` already
  keeps `DuctDiameterRootFind` for exactly this reason, "for comparability, not because
  MDF needs it".
- The degeneracy is **seed-dependent** (§17.2), so calling it in `assemble` would make
  MDF's graph depend on where it was seeded, which is the property `prime()`'s frozen
  guess exists to avoid.
- **The qualification.** A degenerate block's unknown keeps whatever `prime()` seeded it
  with, and `inner_residuals` reports it at relative gap `0.00e+00` -- indistinguishable
  from a well-determined block that converged. So the honest reading of every
  `0.00e+00` row above requires the rank column beside it, which is why both instruments
  were repaired together and why this section prints them side by side. Making
  `inner_residuals` say so itself would mean handing it the ranks, i.e. an MDA-sized
  Jacobian per call; **not done**, and the reason is cost, not principle.
## 17. The cold start becomes a stage (2026-08-30)

§16 ended with a rule and no instrument: **"a check performed where the seed supplies the
answer is not a check"**, and *"the cold start is the only place the port is tested against
a graph it has to produce itself."* Twenty-two missing producers had been found by a
throwaway script; nothing in the harness could have found the twenty-third. This section
records the stage that closes that -- `functional_process/cold_start.py`,
`tests/functional_process/test_cold_start.py`, `functional_process/reference_cold_start.txt`
-- and the three things it found on its first run.

### 17.1 The measurement, and the one line that makes it different

`mda_harness.compare(graph, data)` gained a `seed` argument, defaulting to `data`. That is
the whole mechanism. Every existing stage seeds boundary inputs from the same converged
`DataStructure` it diffs against, so a variable nothing owns is handed PROCESS's answer
through the boundary and passes on a number the port never computed. `cold_start` passes
**two different structures**:

- `ColdState.seed` -- the `DataStructure` as `init_process` left it, before any model ran
  (a `deepcopy` of `SingleRun(...).data`; `set_scaled_iteration_variable` destroys it on
  the first model call, which is why `sand_harness.ReferenceRun.cold` exists too).
- `ColdState.process` -- the same structure after `load_iteration_variables` and one
  `Evaluators.fcnvmc1` at the cold `x`. `fcnvmc1` rather than a bare
  `_call_models_once` because it is what PROCESS's own solver calls on iteration zero,
  Gauss-Seidel loop and all.

Every genuine `IN.DAT` input is identical in the two -- PROCESS's pipeline does not write
them, which is what makes them inputs -- so the *only* thing the split changes is that a
computed field is seeded with its uninitialised default. `boundary.computed_by_process`
now delegates to `cold_state` rather than running its own `SingleRun`, so the
declaration-side question (*does anything own it?*) and the value-side question (*does the
port compute it?*) are answered from one evaluation at one point.

### 17.2 The discriminator, measured rather than argued

§16.3 records three persuasive, wrong answers, each killed by a cheap measurement. The
cheap measurement for *"is a cold disagreement the port's fault, or has PROCESS simply not
settled?"* is to run PROCESS's own map further from where `check_agreement` stopped it and
see what moves. `ColdState.unsettled`/`drift` do exactly that, `EXTRA_PASSES = 3`.

| configuration | GS passes | fields still moving | worst drift | smallest disagreement | largest |
|---|---|---|---|---|---|
| `stellarator_helias` | 5 | 88 | `7.00e-07` | `1.21e-04` | `5.27e-01` |
| `large_tokamak_nof` | 6 | 15 | `2.74e-08` | `1.17e-06` | `1.00e+00` |
| `low_aspect_ratio_DEMO` | 5 | 0 | `0` | `1.04e-06` | `1.00e+00` |
| `large_tokamak_eval` | 6 | 0 | `0` | `1.10e-06` | `1.00e+00` |

**PROCESS's cold state is an exact fixed point of PROCESS's own pipeline on two of the
four**, and creeps at `7e-07` / `2.7e-08` on the others -- in every case at least 40x
below that configuration's *smallest* reported disagreement. The independent check on
`large_tokamak_nof`: thirty further passes move the burn time, the TF temperature margin,
`dlscal` and `.costs.coe` by **exactly zero to eight significant figures**. So
"the port drives its loops and PROCESS does not" -- true in general, and the reason
§16.3(b) was tempting -- explains nothing in this pin. The two sides are two fixed points
of two maps, which is a difference in the maps.

`.numerics.n_model_calls` is excluded (`BOOKKEEPING`): the probe increments it, and it was
the single reported motion on the two configurations that are exact fixed points -- a
textbook case of a measurement artefact hiding a null result.

### 17.3 A third outcome that only a cold measurement has: nothing to compare against

`Physics.outplas` is called from `Physics.output()` and never from `run()`
(`physics.py:219-223`), so `.physics.nu_star`, `.rho_star` and `.beta_mcdonald` are
computed by PROCESS's **report** pass only. Warm, the converged `DataStructure` holds the
report pass's values and `mda_harness.compare` sees them agree; cold, PROCESS holds `0.0`
and the port holds its own correct answer. Neither side is wrong and the comparison is
empty.

These are counted in `ColdReport.output_pass_only` and scored as neither agreement nor
defect -- and **the category is derived from PROCESS's measured write set, never from a
hand list**, the same discipline `boundary.computed_by_process` applies to the converse
question. A new one appears as a count moving, not as a silent pass.

### 17.4 What this stage found

**Result at the cold point, first run:**

| configuration | agree | disagree | output-pass-only | errors |
|---|---|---|---|---|
| `stellarator_helias` | 453 | 49 | 2 | 27 |
| `large_tokamak_nof` | 631 | 79 | 3 | 22 |
| `low_aspect_ratio_DEMO` | 672 | 44 | 3 | 22 |
| `large_tokamak_eval` | 688 | 24 | 3 | 22 |

`large_tokamak_nof` warm, for comparison, is **682 / 33**. The 46-row gap is the stage's
whole reason for existing, and it is two causes.

**(a) `.tfcoil.str_wp` -- the last missing producer, priced.** It is the one row of
`boundary.missing_producers_tokamak.txt`, an output of the unported
`process/models/tfcoil/base.py::stresscl`. Seeded at `DataStructure()`'s `0.0`, which is
the *peak* of the Nb3Sn critical-current fit, so every critical current comes out high and
the TF temperature margin with it: `1.58` against PROCESS's `1.24` on `large_tokamak_nof`
(+27 %), `2.53` against `2.29` on DEMO, `4.77` against `4.48` on eval. Seven variables on
each of the three tokamaks, and the error is **optimistic** on exactly the quantity
constraints 33 and 36 read. Confirmed by substitution on `large_tokamak_eval`: replacing
the seeded `0.0` with PROCESS's own cold `0.0018442328` removes exactly those seven rows
and adds none (688/27 -> 695/20 on the raw `compare`). §16.9 predicted the *shape* of this
from the declaration side; this is the number.

**(b) `inductance.NOH = 30` is wrong on two of the three tokamaks -- a new defect.**
`PFCoil.induct` splits the CS into
`noh = ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] - r_pf_coil_inner[CS]))`
pancake segments (`pfcoil.py:1758-1765`) and every inductance depends on that integer.
The port pins `NOH = 30`, measured on `large_tokamak_eval`. Measured from PROCESS's own
`DataStructure`s:

| configuration | ratio cold | `noh` cold | ratio converged | `noh` converged |
|---|---|---|---|---|
| `large_tokamak_eval` | 29.028 | **30** | 29.028 | **30** |
| `large_tokamak_nof` | 31.746 | 32 | 26.867 | 27 |
| `low_aspect_ratio_DEMO` | 27.010 | 28 | 26.407 | 27 |

The constant is right on exactly the file it was measured on. Confirmed by substitution:
with `NOH = 32` on `large_tokamak_nof` the cold result goes **631/82 to 662/51**, 31 of
the 65 chained rows vanish outright and the other 34 fall by one to two orders
(`.times.t_plant_pulse_burn` `3.76e-04 -> 8.89e-06`; `.pf_coil.ind_pf_cs_plasma_mutual`
`5.10e-04 -> 6.55e-05`) to below `PicardDriver`'s own `1e-4` tolerance.

**The fix is not a different constant.** `noh` is 32 at this file's cold design and 27 at
its converged one, so no assembly-time integer is right at both --
`_audit/units/models/pfcoil/inductance.md`'s "a structural integer that the solve moves"
is not a stylistic worry, it is a live wrong answer whose size is now measured. Pinned
with that reason; `test_the_pinned_noh_is_right_on_one_configuration_and_wrong_on_two` is
the standing measurement and will fail the day the convention question is answered.

**Why the warm harness saw neither.** (a) is invisible warm by construction -- the seed
supplies `str_wp`. (b) *is* visible warm, at `1.68e-04` on
`.pf_coil.ind_pf_cs_plasma_mutual`, and had been sitting in `run_mda_harness`'s "all
disagreements" list unexplained; what the cold stage added was the *cause*, because `noh`
cold and `noh` converged differ, which is the observation that identifies the mechanism.

**(c) The stellarator's 44-row chain is PROCESS disagreeing with itself, and is not a
defect.** `Stellarator.run(output=False)` runs `st_coil` then `st_build`;
`Stellarator.run(output=True)` runs `st_build` then `st_coil` (`stellarator.py:141-146`
against `:159-165`). Measured at the cold design: PROCESS's solve pass leaves
`.build.z_tf_inside_half = 3.611990999471611`, **PROCESS's own output-pass order leaves
`5.513665371874896`, and the port computes `5.513665371874896`** -- the report-pass arm to
sixteen digits, with `.buildings.a_plant_floor_effective` following at `424256.91005`
against PROCESS's own `424256.91004`. Everything below is that one geometry through the
buildings volumes, the site accounts, and the AC-power and cost chain, up to `.costs.coe`
at `3.4e-02`.

`mda_harness.EXPLAINED_DISAGREEMENTS`'s `.heat_transport.p_plant_electric_base_total_mw`
entry is the same finding from the other side: warm, PROCESS *stores* the report-pass
geometry, so the port agrees on `z_tf_inside_half` and disagrees only on the one field the
report pass never recomputes; cold, PROCESS stores the solve-pass geometry, so the
disagreement is the whole chain. The port is self-consistent with one of PROCESS's two
arms and PROCESS is not self-consistent between them, so neither point can be made to
agree everywhere.

### 17.5 The pin, and the rule that every row carries a reason

`reference_cold_start.txt` holds, per configuration, the agreement count, the *error*
count and every disagreeing and output-pass-only path. The error count is pinned because
that is the one bucket where a regression is silent -- an entry there is a variable
neither passed nor failed. The cold tokamaks sit at 22 against the warm run's 20, and the
two extra are the same `Physics.output()` cause as §17.3 seen as a shape rather than a
value: `calculate_effective_charge_ionisation_profiles` is also output-pass-only, so
`.physics.n_charge_plasma_effective_profile` is `(0,)` in the cold structure against the
port's `(201,)`. **`cold_start.ACCEPTED` must have an entry for every pinned disagreement
or the stage refuses**, and it is keyed on
`(configuration, path)` rather than on the path, because the cause is per-machine:
`.costs.coe` is off by `3.4e-02` on the stellarator through the report-pass geometry and
by `2.2e-04` on `large_tokamak_nof` through `noh`.

That rule is the whole lesson of §16 written into the harness. In a bare list of paths a
row somebody chased and a row nobody looked at read identically, and the twenty-two
missing producers survived weeks inside exactly that ambiguity. Six causes cover all 196
pinned disagreements: the two defects above, the stellarator pass-order chain, the
already-recorded `n_pf_coil_turns` dead tail and `VacuumOld` duct-solve tolerance (both
deferring to `mda_harness.EXPLAINED_DISAGREEMENTS` as the authority rather than restating
it), and `large_tokamak_eval`'s ten residual rows at `1.1e-06`, which are `PicardDriver`'s
own `rtol = atol = 1e-4` showing through a comparison run at `1e-6` -- **two orders better
than the driver promises**, and a standing caveat for reading any row of this pin below
`1e-4` inside or downstream of a `Drive`.

`test_the_reason_table_has_no_rows_that_no_longer_disagree` closes the other half:
`ACCEPTED` may not outlive what it explains. Landing `stresscl` should make seven of its
entries fail, which is the point -- the failure mode this whole file keeps recording
(`constraint_48`, `0dca2227`, `stresses.py`'s "neither is read by any active constraint")
is a documented omission whose justification quietly expired.

### 17.6 What this does not do

- **It does not fix either defect.** Both are pinned with their evidence. `stresscl` is a
  1053-line unit with its own registry row; `noh` is an open convention question whose
  answer is not a number.
- **It does not run the constraint surface.** `unproduced_but_computed` already records
  that the MDF-assembled graph shows more missing producers than the MDA graph, because
  constraints declare reads the MDA never makes. The same is true here: this stage
  measures the MDA graph's owned variables, so a constraint-only read like
  `.tfcoil.sig_tf_case` cannot appear.
- **It says nothing about derivatives.** Stage B remains the only check that a right value
  has the right sensitivity.

## 18. Where a cold solve's 100 seconds go (2026-08-31) — it is compile, and mostly not the solve's

`run_cold_matrix` takes ~1690 s for 14 solves, ~120 s per solve. §13.4 measured a full
326-iteration stellarator SAND solve at **11.93 s wall plus 7.59 s compile** and §13.3 put
one value-and-Jacobian pair at **0.74 ms**. Both of those numbers are still right. They
are numbers about a *driver*, handed a problem that was already assembled and an env that
was already primed, and the matrix pays for the assembling and the priming.

### 18.1 Method

Two configurations (`stellarator_helias`, the subject of §13.4, and `large_tokamak_nof`,
the largest that solves), two formulations, and **two processes each**: a `probe` process
that builds and never solves, and a `solve` process that solves and never probes. Nothing
is measured second in a process where the same thing was measured first, because JAX's
in-process compile cache has produced three wrong conclusions in this investigation
already. `jax_enable_x64` on, `provider` boundary mode, PROCESS reference disk-cached.

- **TRACE / LOWER / XLA compile are separated** with `jax.make_jaxpr(fn)(x)`, then
  `jax.jit(fn).lower(x)` (which retraces — so LOWER is an upper bound on the lowering
  alone), then `.compile()`. Equation counts are off the same jaxpr, top-level and
  counted recursively into sub-jaxprs.
- **Evaluation against everything else** is a wrapper on `pyvmcon.solve` that times each
  `problem(x)` call, exactly §13.1's instrument. The **first** call is reported separately
  because it carries both jits' whole trace-lower-compile; "OTHER" is
  `pyvmcon.solve` total minus every `problem(x)`, i.e. the QP, the line search and the
  host bookkeeping.
- Scratch scripts only; **no library code changed in this pass.**

### 18.2 The phase table, cold, one process per column

Seconds. `-` means the phase does not exist for that formulation.

| phase | helias MDF | helias SAND | tokamak MDF | tokamak SAND |
|---|---|---|---|---|
| `machine_from_indat` + `graph_for` | 0.09 | 0.06 | 0.34 | 0.37 |
| `reference_run` (disk-cached) | 1.43 | 1.52 | 1.45 | 1.50 |
| `mda_env` (warm) | - | **3.75** | - | **9.54** |
| `assemble` | 0.13 | **9.87** | 0.18 | **34.36** |
| `seed` | 0.13 | 0.02 | 0.12 | 0.03 |
| `prime` (MDF) / schedule + scales + 2nd `mda_env` (SAND) | **8.33** | 0.08 | **30.64** | 0.12 |
| solve wall (compile included) | **11.55** | **14.89** | **42.04** | **63.82** |
| **whole process** | **21.7** | **30.5** | **75.0** | **110.0** |

`stellarator_helias` reproduces §13.4 (21.7 s and 30.5 s against its ~20 s), so the tree
has not drifted; **`large_tokamak_nof` is the 100 s row**, and a matrix row is both
formulations — 185 s per tokamak configuration, which is the 1690 s.

### 18.3 The condition map and its `jacfwd`, cold, each in its own process

| | value eqns (top/rec) | TRACE | LOWER | XLA | steady | `jacfwd` eqns | TRACE | LOWER | XLA | steady |
|---|---|---|---|---|---|---|---|---|---|---|
| helias MDF (8 unknowns) | 3974 / 6905 | 0.52 | 0.29 | 1.15 | 0.5 ms | 7899 / 13791 | 2.02 | 0.52 | **4.35** | 0.6 ms |
| helias SAND (14) | 2994 / 4437 | 0.25 | 0.21 | 0.95 | 0.3 ms | 6269 / 8688 | 1.18 | 0.36 | **4.14** | 0.7 ms |
| tokamak MDF (20) | 14401 / 19437 | 1.61 | 0.69 | 4.36 | 0.8 ms | 32560 / 43846 | 6.56 | 1.43 | **22.26** | 3.3 ms |
| tokamak SAND (26) | 11675 / 13438 | 1.15 | 0.55 | 3.80 | 0.8 ms | 27993 / 31525 | 3.63 | 1.49 | **22.75** | 12.7 ms |

**The `jacfwd` hypothesis is half right.** It is *not* an equation blow-up: forward-mode
Jacobian programs here are **2.2x–2.4x** the forward program's equations, never
`n_unknowns` times it — JVP roughly doubles the equation count and the 8–26 tangent
columns ride in the *shapes*, not in new equations. But `jacfwd`'s XLA pass is
**5.1x/6.0x** the value map's on the tokamaks for 2.3x the equations, and at 22.3 s /
22.8 s it is **~65 % of the whole solve's compile**. So it is the lever, just not for the
reason proposed. And because its program tracks the forward program's size, §24.9's
39,127 -> 13,283 vectorisation of the MDA jaxpr propagated into it: that pass bought the
tokamak solves roughly a 3x compile reduction as a side effect nobody measured.

### 18.4 Steady state per iteration: evaluation against the QP

Excluding the first `problem(x)` call, which is the two jits.

| | its | evals | first eval (= both compiles) | evals after | ms/it evaluating | ms/it OTHER (QP + line search + host) |
|---|---|---|---|---|---|---|
| helias MDF | 67 | 133 | **8.83 s** | 0.155 s | 2.3 | **8.9** |
| helias SAND | 90 | 179 | **7.43 s** | 0.232 s | 2.6 | **10.3** |
| tokamak MDF | 7 | 13 | **34.78 s** | 0.054 s | 7.7 | **11.9** |
| tokamak SAND | 10 | 19 | **32.20 s** | 0.085 s | 8.5 | **12.7** |

§13.4's 2.1 ms evaluating against 11.8 ms in the QP is reproduced to within a tenth of a
millisecond on the stellarator, four months and one QP-solver change later. **The solve
loop is not where any of the missing time is**: on the tokamak the entire steady state is
**0.14 s of a 42–64 s solve**, and the tokamaks converge in 7 and 10 iterations.

### 18.5 Three alternative explanations, closed with numbers

- **Recompiles per iteration** (the `weak_type` signature split found in `mda_env`
  today): **ruled out.** 132 further evaluations cost 0.155 s and 18 cost 0.085 s — one
  recompile would be seconds, and every table above would show it.
- **`cvxpy` QP scaling with condition count**: **ruled out.** OTHER is 8.9 / 10.3 / 11.9 /
  12.7 ms per iteration at 15, 21, 27 and 33 conditions. Flat, and the same order §13.4
  measured.
- **The matrix doing per-row work a bare solve does not**: **ruled out.** Every phase the
  matrix runs is in §18.2's table; the two formulations sum to the row, and the row sums
  to the matrix.

### 18.6 What it actually is: ~1000 eager per-primitive XLA compiles per phase

The compile the solve pays (§18.3) is real but is the *smaller* half on the tokamak.
`cProfile`, cumulative, on the two biggest phases:

| phase | wall | `backend_compile_and_load` | number of compiles | per compile |
|---|---|---|---|---|
| `mdf.prime`, tokamak | 32.6 s | **22.5 s** | **988** | 23 ms |
| `sand_harness.assemble`, tokamak | 37.8 s | **26.6 s** | **1050** | 25 ms |
| tokamak SAND solve, `_run_acyclic` steps | ~31 s of 63.8 | (of 51.8 s over 769) | ~767 tiny | 66 ms |

All three are **the same defect**: a cottax graph run through `evaluate._run_acyclic`
**un-jitted**, so every `jnp` primitive is dispatched eagerly and XLA compiles each one
on its own. §24.4 of `next_steps.md` already fixed this for `sand_harness.mda_env`
(805 eager compiles -> one `filter_jit`); three paths were not in that pass's scope:

- **`mdf.prime`** — one eager MDA run to fill the env.
- **`sand.degenerate_fixed_points` -> `fixed_point_residuals`**, which is 37.5 s of
  `assemble`'s 37.8 s: six `jax.jacfwd` calls over inner fixed-point residuals, evaluated
  op-by-op under `jvp`+`vmap`.
- **the SAND schedule's 92 non-`Drive` `Call` steps at solve time** — 95 `_run_acyclic`
  calls inside the one `schedule(...)`, which is why the tokamak's solve wall is 63.8 s
  while `pyvmcon.solve` inside it is 32.4 s.

### 18.7 The verdict, and the lever (not applied in this pass)

**The 5x is not in the solve.** §13.3 and §13.4 measured a driver on a problem someone
else had built; a cold matrix row pays for building it. Per cold tokamak solve, roughly:
**~33 s of jitted compile inside the driver (two-thirds of it `jacfwd`'s XLA pass), ~30 s
of eager per-primitive compile in the un-jitted build/prime phase, ~31 s more of the same
in the schedule's `Call` steps, and 0.14 s of actual solving.**

Two levers, in order of size and both cheap to state:

1. **Jit the three remaining eager paths**, exactly as §24.4 did for `mda_env`:
   `mdf.prime`, `fixed_point_residuals`, and the schedule's `Call` steps. ~60 s of the
   tokamak's 185 s, and the pattern is already proven in this tree. The cost is §24.4's
   own: `check_antichain` refuses a path named both whole and by element, so each path
   needs its boundary checked first.
2. **`jacfwd`'s XLA pass**, 22 s per tokamak solve. It scales with the forward program, so
   §24.9's next concentration (`bootstrap_current.py`, 2037 equations) pays twice. There
   is no evidence here for a cheaper Jacobian formulation — the equation count is already
   only 2.3x the forward program, which is what forward mode should cost.

Neither is applied here. This section is the measurement.

## 19. §18.7's lever cost a converged solve, and the reason is one ulp (2026-08-31)

`stellarator_helias`'s cold SAND row regressed from **90 iterations converged at
`objf 1.21775735`** to **108 `stopped`** at `max|eq| 2.85e-02`, `objf 1.22217408`,
between the two commits of 2026-08-31. Bisected by A/B in one process, not inferred.

### 19.1 The cause: `run_schedule`'s undriven-group jit, upstream of the driver

Four changes landed that day and only one moves this row. With everything else held,
swapping the solve's `run_schedule(solve_schedule, ..., whole=False)` for a plain
`solve_schedule(env)` gives **90 converged, `1.21775735`** — the pre-regression number
exactly. `mda_env`'s jit (§24.4 of `next_steps.md`), `_strongly_typed`, and the
`bootstrap_current`/`safe_math` rewrites are all *in* the reproducing run and none of
them is implicated.

**The seed did not move; the trajectory did.** The cold and warm `mda_env` envs and the
`_seed`ed start were compared key by key between the two runs: **0 of 317 seeded values
differ**, bitwise (831 MDA keys likewise, the one "difference" being a `nan` that is not
equal to itself). §24.4's ~1e-12 seed drift is therefore *not* this.

What differs is one step later. `stellarator_helias`'s SAND solve schedule is
`25 x Call`, then the `Drive`, then `20 x Call`. Running those first 25 as one
`eqx.filter_jit`ed group instead of step by step changes **two** of 384 values:

| path | relative | absolute |
|---|---|---|
| `.heat_transport.etath_liq` | `4.85e-16` | `1.67e-16` |
| `.tfcoil.a_tf_turn_steel` | `4.43e-16` | `2.17e-19` |

One ulp each, XLA reassociation — the same effect §24.4 measured on `mda_env`. And
`.tfcoil.a_tf_turn_steel` is **inside the `Drive`'s own context**, so VMCON is handed a
different problem. The two solves are then bit-identical for **14 SQP iterations**
(convergence, `f`, `max|eq|`, `min ie` agree to all 17 digits at 0..13) and diverge at
15. An SQP trajectory is not a continuous function of its data; a converged solve and an
infeasible stop are both reachable from points 1 ulp apart.

### 19.2 `max|eq| 2.85e-02` is a real violation, not a scale artefact

The trap §24.11 hit twice does not apply here. The worst equality at the stopped point is
`^cond.stellarator.wp_width_r_min` at scaled `+2.851e-02` with
`residual_condition_scales` factor **`1.3948`** — an O(1) factor, i.e. `|u| = 0.717` —
so the unscaled residual is `+2.044e-02` and the relative one really is 2.9 %. The next
worst is `^cond.constraints.c16` at `9.5e-04`. The solve stopped genuinely infeasible.
(The two `1e-18`-scaled rows, `fusden_alpha_total` and `proton_rate_density`, *are* the
divide-by-nothing shape and sit at `7.7e-04`/`6.8e-04`; they are not the reported
maximum either way.)

### 19.3 The fix, and why it costs nothing

`_schedule_runners` now fuses only the group **after the last `Drive`** and leaves every
group a driver is downstream of eager (`_eager_group`). The `Drive`'s **body** stays
jitted: it runs after the driver has converged and cannot move it.

Nothing is given up. §24.11's own split says where the eager cost is — on
`large_tokamak_nof`'s SAND solve the undriven steps are **2.8 s of 108** against the
`Drive`'s **105.4 s** — so the body jit is the whole saving and the group jit was buying
2.6 % at the price of the answer. Measured after the fix: `stellarator_helias` cold SAND
is **90, converged, `1.2177573520529628`, `max|eq| 1.84e-06`**, identical to the fully
eager solve. `test_sand.py` 147 passed, `test_mdf.py` + `test_cold_matrix.py` 52 passed.

**The rule this establishes:** a jit boundary may be moved freely *downstream* of a
host-side driver and never *across* its inputs. Fusing reassociates, reassociation moves
the last bit, and every iterative host-side algorithm in this harness — VMCON here, and
`SeededNewtonDriver` for the same reason — turns a last-bit change in its data into a
different sequence of steps. `mda_env`'s jit is safe under this rule only because its
drives are Newton solves converged to a tolerance; the SAND `Optimise` is not, and
§24.4's "a row moving by one iteration is a consequence" understated it by 18 iterations
and a status.

**Still open, and this fix does not touch it:** the row's `d objf 2.34e-03` against
PROCESS and `worst dx 1.08e-01` at ixc 6/109 are unchanged — the port converges to its own
answer, not PROCESS's, and that gap is §17/§24.10's business.

### 19.1 The rule in §19 is a workaround, not a principle — reopened 2026-08-31

§19 concludes "a jit boundary may move freely downstream of a host-side driver, never
across its inputs". **That is a description of what preserved today's answer, not a
design anyone should keep.** It says the port may only jit the parts of its graph that
nothing downstream is sensitive to, which on this graph is a minority of it.

The situation it papers over: XLA reassociates float arithmetic when it fuses, so
fusing *anything* can move last bits; §19 measured `.tfcoil.a_tf_turn_steel` moving
`4.43e-16` relative and the SQP turning that into 18 extra iterations and a status flip.

Two real answers, neither taken:

1. **Constrain XLA's float reassociation** and jit the whole graph. Costs fusion, buys
   determinism across our own refactors -- which is the property that actually matters
   here, distinct from bit-matching PROCESS (the user's standing ruling is that matching
   PROCESS bit-for-bit is *not* a goal; reproducibility across port changes is another
   question and this is the evidence for it).
2. **Make the solve tolerate `1e-16`.** If a last-bit perturbation flips convergence the
   solve is on a knife edge and **both** answers are luck -- including the 90-converged
   one §19 restored. Scaling, tolerances or a restart policy, not arithmetic freezing.

Do not extend §19's rule to new sites without revisiting this.

## 20. The fused/eager split is one ulp of one variable, and the diagnosis is still open (2026-09-01)

§19.1 offered two answers to §19's fusion problem and said neither had been taken. This
section takes the **second** one -- *"if a last-bit perturbation flips convergence the
solve is on a knife edge and **both** answers are luck"* -- and tests its **first half**,
which comes back true. The second half, *why*, is **not settled here**: §20.12 records
three candidate causes and rules out or weakens each, and none of them survives as an
explanation. Every number below is measured in this working tree, one process per table,
and marked as such; the causal reading is flagged wherever it is a reading.

**What was measured, in one line:** on `stellarator_helias`'s cold SAND solve a **hand
perturbation of `.tfcoil.a_tf_turn_steel` by -1 ulp, with every group eager**, reproduces
the fused run's 108-`stopped` outcome **bit for bit** -- same `objf`, same `max|eq|`,
`max rel |dx| = 0.000e+00` across all eight iteration variables. So the fused/eager
difference **is** that one bit, and nothing else the fusion did contributes.

**What is not established:** that "the problem is intrinsically a knife edge". §20.12
measures the constraint Jacobian's condition number (SAND `1.7e4`, carried entirely by
**one** row; MDF `42` on the same physical problem), the active set at the fork and at
the solution, and what the assembly dropped or froze (nothing) -- and no single one of
those explains the flip. **This project's record is that "it is the conditioning" has
been the wrong answer every previous time it was reached** (§17's OSQP-for-CLARABEL,
§16.1b's `MAX_ITER`, §12.2's cap, `SlsqpDriver`'s own docstring on five overturned
diagnoses), so the diagnosis is held open rather than asserted.

**A wording caution that will otherwise be misread:** "eager" here and in §15 does **not**
mean "not compiled". `VmconDriver` jits the condition map it iterates and `jax.jacfwd` of
it (`drivers.py`, `evaluate = jax.jit(flat_conditions)`), and `SlsqpDriver` does the same
through `scaled_problem`, so the arithmetic the SQP actually differentiates has always
been one XLA program. What "eager" names is the **schedule walk around the driver** --
whether the `Call` steps outside the `Drive` are run step by step or fused into one
program. The fused/eager axis is a schedule-shape choice, never a compiled/interpreted
one.

### 20.1 What was run, and the flag it was run behind

`run_schedule`/`_schedule_runners` take a new `fuse_upstream` parameter. `False` -- still
the default -- is §19.3's protected grouping. `True` is the whole-program policy: every
undriven group fused, upstream ones included, leaving eager only the host-side driver
itself, which cannot trace at all (`cvxpy`, `pyvmcon`, a Python callback). Nothing else
about the schedules changed; the two policies are the same steps in the same order
through the same nodes.

`SlsqpDriver` gained an `outcome` field, written the way `mdf.MdfNewtonDriver`'s already
was, because SLSQP's status is not recoverable from the returned point and *converged* /
*iteration limit* / *positive directional derivative* are three different outcomes this
comparison has to keep apart. It takes a new `drivers.Outcome` -- a `dict` hashed by
**identity** -- rather than a bare `dict`, which cannot be a driver field at all:
`Schedule.__hash__` reaches every driver through `Graph.__hash__`, and `run_schedule`
hashes the schedule on every call, so a `dict` field raises `TypeError: unhashable type`
at the schedule, several frames from the driver carrying it.

Everything else is `run_cold_matrix.cold_sand`'s own path: the warm MDA env for assembly
and `residual_condition_scales`, the cold MDA env for `_seed`, `max_iter = 500`,
`VmconDriver`'s default `1e-6` convergence test, `max|eq|` always reported in
`condition_scale`'s units so the column compares across the table (§13.1's rule).

### 20.2 What fusing the 25 upstream `Call` steps actually moves, with the sign

§19 gave relative magnitudes; the ulp counts and directions are the part that matters and
they were not recorded. Measured by running the group both ways on the same seeded env
and diffing every value the `Drive` is handed:

| path | eager | fused | relative | ulp | in the `Drive`'s context |
|---|---|---|---|---|---|
| `.tfcoil.a_tf_turn_steel` | `0.0004898559999999999` | `0.0004898559999999997` | `4.427e-16` | **-2** | **yes** |
| `.heat_transport.etath_liq` | `0.3434546216433674` | `0.34345462164336754` | `4.849e-16` | +3 | no |

2 of 384 values, exactly as §19 found. **The fusion moves the one value the driver reads
downward by two ulp**, which is the whole of the difference between the two solves.

### 20.3 The hand perturbation, eager throughout -- `VmconDriver`

Each row is the reference cold SAND solve with the named value moved by hand in the env
the `Drive` is handed, and **no** fusion anywhere:

| run | its | status | `objf` | `max\|eq\|` | `min ie` | `max rel \|dx\|` vs eager |
|---|---|---|---|---|---|---|
| eager (§19.3's default) | **90** | converged | `1.2177573520529628` | `1.84e-06` | `-8.83e-10` | -- |
| fused | **108** | stopped | `1.2221740786343283` | `2.85e-02` | `1.12e-11` | `6.12e-01` |
| eager, `a_tf_turn_steel` **-1 ulp** | **108** | stopped | `1.2221740786343283` | `2.85e-02` | `1.12e-11` | `6.12e-01` |
| eager, `a_tf_turn_steel` **-2 ulp** | **108** | stopped | `1.2221740786343283` | `2.85e-02` | `1.12e-11` | `6.12e-01` |
| eager, `a_tf_turn_steel` **+1 ulp** | 90 | converged | `1.2177573520529628` | `1.84e-06` | `-8.83e-10` | `0.00e+00` |
| eager, `a_tf_turn_steel` **+2 ulp** | 90 | converged | `1.2177573520529628` | `1.84e-06` | `-8.83e-10` | `0.00e+00` |
| eager, `etath_liq` +-1, +2 ulp | 90 | converged | `1.2177573520529628` | `1.84e-06` | `-8.83e-10` | `0.00e+00` |

Three things this says that §19 could not:

1. **`max rel |dx|` between the fused run and the eager `-1 ulp` run is `0.000e+00`.**
   Not "similar" -- the same eight floats. One bit on one variable is a complete
   explanation of the 18 extra iterations and the status flip; nothing else the fusion
   did contributes.
2. **The edge has a side.** Down flips it, up does not, and `-2` behaves like `-1`. So
   the 90-converged answer is not robust to the perturbation, it merely happens to sit on
   the tolerant side of *this* boundary. A refactor that moved the arithmetic the other
   way would have handed §19 the opposite conclusion.
3. **`.heat_transport.etath_liq` is inert**, confirming §19's context argument by
   measurement rather than by inspection: a value the `Drive` does not read cannot move
   it, however many ulp it slips.

### 20.4 `SlsqpDriver` on the same problem -- the same edge, sharper

The main experiment §19.1 asked for. A second SQP with its own QP solver, its own line
search and its own Hessian update, on the identical problem through `scaled_problem`:

| run | its | scipy verdict | `objf` | `max\|eq\|` | `min ie` |
|---|---|---|---|---|---|
| **scaled** (`VmconDriver`'s `condition_scale`), eager | **153** | success | `1.2175536499979944` | `2.44e-11` | `-2.22e-15` |
| scaled, fused | **500** | iteration limit | `1.2176017027671482` | `2.33e-03` | `-7.77e-16` |
| scaled, eager, `a_tf_turn_steel` **-2 ulp** | **500** | iteration limit | `1.2176017027671482` | `2.33e-03` | `-7.77e-16` |
| scaled, eager, `a_tf_turn_steel` **-1 ulp** | **500** | iteration limit | `1.2176474064` | `6.42e-05` | `-0.00e+00` |
| scaled, eager, `a_tf_turn_steel` **+1 ulp** | **500** | iteration limit | `1.2175875902` | `4.89e-06` | `-3.33e-16` |
| **unscaled** (§13.6's setting -- the one SLSQP wants), eager | **237** | success | `1.2176260901487665` | `5.95e-13` | `+6.66e-16` |
| unscaled, fused | **500** | iteration limit | `1.2176903287194838` | `9.94e-06` | `-2.22e-16` |
| unscaled, eager, `a_tf_turn_steel` **-2 ulp** | **500** | iteration limit | `1.2176903287194838` | `9.94e-06` | `-2.22e-16` |

`ftol = 1e-8` (the class default), `maxiter = 500`, same `bounds`, same `jax.jacfwd`
Jacobian, same sign translation. §16.6's note that the residual scaling is load-bearing
for pyvmcon and harmful to SLSQP is reproduced here cold -- unscaled SLSQP reaches
`max|eq| 5.9e-13` where scaled reaches `2.4e-11` -- and **it does not change the
verdict**, which is the answer to *"if scaling changes the verdict, that is itself the
finding"*: it does not.

- **`max rel |dx|` between the fused run and the eager `-2 ulp` run is `0.000e+00`** in
  both scalings. Same result as §20.3 by a completely different solver: the fusion is
  exactly a two-ulp perturbation and nothing else.
- **SLSQP is more fragile, not less.** `+1 ulp` -- which `VmconDriver` shrugs off --
  takes it from 153 iterations and *"Optimization terminated successfully"* to the 500
  cap. Every one of the four perturbations tried breaks it. So the eager 153-converged
  row is luck by the same argument, and by a wider margin.
- **Under whole-program fusion SLSQP does not give a stable answer either.** That is the
  question this section was opened to answer, and the answer is no.

### 20.5 `large_tokamak_nof`: the same fusion is inert, for a structural reason

Same experiment on the other configuration §24.11 profiled. Fusing its 24 upstream `Call`
steps moves **1 of 440** values -- `.heat_transport.etath_liq`, +3 ulp again -- and that
value is **not in this `Drive`'s context**. So the prediction is that fusion changes
nothing, and it is what happens:

| run | its | status | `objf` | `max\|eq\|` | `min ie` | eager vs fused |
|---|---|---|---|---|---|---|
| `VmconDriver` | 10 | converged | `1.6000000000354158` | `4.51e-06` | `-9.58e-05` | `dx 0.00e+00` |
| `SlsqpDriver`, scaled | 12 | success | `1.6000000000000654` | `1.98e-14` | `-6.00e-11` | `dx 0.00e+00` |
| `SlsqpDriver`, unscaled | 19 | *positive directional derivative* | `1.6000000000000000` | `1.87e-14` | `-1.73e-14` | `dx 0.00e+00` |

Bitwise agreement between the two policies, for all three drivers. **The fusion
sensitivity is not a property of the jit policy; it is a property of whether the one or
two values fusion happens to move are read by a host-side driver.** That is a fact about
one graph and one arithmetic reassociation, and it is not something a policy can be
written against -- which is the argument against §19's rule stated as a measurement
rather than as a preference.

### 20.6 The optimum is flat: three "converged" runs, three points

Not asked for, and worth recording because it does not depend on any ulp. Three runs on
the stellarator that all report success land on three different points:

| run | `objf` | `max\|eq\|` | `x109` | `max rel \|dx\|` vs `VmconDriver` |
|---|---|---|---|---|
| `VmconDriver`, 90 its | `1.2177573520529628` | `1.84e-06` | `0.0299518` | -- |
| `SlsqpDriver` scaled, 153 its | `1.2175536499979944` | `2.44e-11` | `0.0326286` | `8.94e-02` |
| `SlsqpDriver` unscaled, 237 its | `1.2176260901487665` | `5.95e-13` | `0.0323901` | `8.14e-02` |

(All three in one process, so the `dx` column is a like-for-like diff and not two runs
compared through a printed table. The two SLSQP answers are `7.31e-03` apart from each
other.) SLSQP's answers are **more feasible and lower in objective** than the one VMCON
declares converged, at design points ~8-9 % away. On `large_tokamak_nof` it is starker: VMCON
and SLSQP differ by `1.50e-01` relative in `x` while both report `objf = 1.6` to twelve
digits. An objective that flat over that much of the design space means the *answer* is weakly
determined even where the solve succeeds. Read as a fact about the objective, which is
what it is; it is **not** evidence that the constraint Jacobian is ill-conditioned, and
§20.12 measures that separately and finds it is not (SAND `1.7e4` from one row, MDF
`42`).

### 20.7 What the default should be

**`fuse_upstream=True`, i.e. one jit over everything the host-side driver does not sit
in.** Recommended here, and **applied** -- see §20.9, written after the user accepted the
finding.

The case, from the numbers above rather than from taste:

- §19's rule was adopted because it restored a converged row. §20.3 measures that the row
  it restored is one bit from the row it replaced, and that the sign of that bit -- not
  anything the rule states -- decides which. Whatever the underlying cause turns out to
  be (§20.12 leaves it open), a rule that selects an outcome by which way one ulp fell is
  not buying determinism, and calling it one costs the reader the ability to see that
  this solve is fragile.
- It is not buying speed either, in either direction. §24.11's split (2.8 s of 108 on
  `large_tokamak_nof`) says the upstream groups are a rounding error against the `Drive`,
  and the solve-only wall times here -- 10-12 s per stellarator run, 34-40 s per tokamak
  run, first row in each process paying compile -- separate the two policies by less than
  the run-to-run spread. So this is a design choice, not a performance one, and the
  design the user has ruled for is the single jit.
- It cannot be extended safely anyway. §20.5 shows the rule's protection is contingent on
  which values a given fusion happens to reassociate and whether a given `Drive` reads
  them -- a per-configuration accident, re-rolled by every model edit, that no policy
  statement can cover. §19.1's *"do not extend §19's rule to new sites"* is right, and
  the reason is now measured.

**The honest consequence, stated up front:** flipping the default puts
`stellarator_helias`'s cold SAND row back to **108, stopped, `objf 1.22217408`,
`max|eq| 2.85e-02`** -- a genuine 2.9 % violation, per §19.2, which that section
established is not a scale artefact. That is a worse-looking table and a truer one. The
row should carry §20.3's sentence: this solve is 1 ulp from converging and 1 ulp from
stopping, and which it does is not evidence about the model.

**What the flip costs elsewhere.** It changes `run_cold_matrix.py`'s published
stellarator row, and that file and `reference_cold_matrix.txt` were being edited in
parallel with this measurement, so the regeneration and its provenance are handled
separately from this section -- the flip itself is §20.9.

### 20.8 What would actually fix it, and what would not

- **Constraining XLA's reassociation** (§19.1 item 1) is ruled out by the user and is in
  any case now beside the point: it would freeze *one* source of last-bit motion in a
  solve that §20.3 shows is sensitive to last-bit motion from **any** source -- a model
  edit, a `jnp` version, a different CPU. It would buy reproducibility across our own
  refactors of the traced arithmetic only, and hide the fragility rather than remove it.
- **Neither SQP is the problem.** Two independently written solvers, two QP backends, two
  Hessian strategies, two scalings, all flip on the same bit of the same variable
  (§20.4). The next thing to try is therefore not a third optimiser.
- **The open work is to find the mechanism, and §20.12 says which three explanations it
  is not.** The reproducer is cheap and deterministic: perturb `.tfcoil.a_tf_turn_steel`
  by `-1 ulp` in the env the `Drive` is handed, eager throughout, and the failure appears
  with no jit anywhere in the picture. Any candidate fix has to survive **both** signs of
  `+-1` and `+-2 ulp`, has to do so for SLSQP too (§20.4: it survives none of them), and
  -- from §20.11 -- has to do so for **MDF**, which flips on 1 ulp of a design start
  despite a constraint Jacobian conditioned at `42`.

### 20.9 The flip, applied (2026-09-01)

`run_schedule`, `_schedule_runners` and `_driven_runner` now default to
`fuse_upstream=True`. **Every undriven group is one jit, upstream of a `Drive` included;
the only thing left eager is the host-side driver itself**, which cannot trace at all.
`fuse_upstream=False` still restores §19.3's grouping and is kept for one purpose:
diffing the two policies is how §20.2-§20.5 were measured and how any successor to them
will be.

**What this does to the published numbers.** `stellarator_helias`'s cold SAND row goes
back to **108, `stopped`, `objf 1.22217408`, `max|eq| 2.85e-02`** -- a genuine 2.9 %
violation, per §19.2, which established that it is not a scale artefact. The other
configuration measured here, `large_tokamak_nof`, does not move at all (§20.5: bitwise
agreement between the policies for all three drivers).

**That is a real number, and §20.3 is why it is not a regression the flip caused.** The
90-converged row it replaces is reproduced, bit for bit, by *any* eager run whose
`.tfcoil.a_tf_turn_steel` sits on the `+` side of one ulp, and the 108-stopped row by any
run on the `-` side. Both are the same solve, and which one a given build prints is
decided by a bit that no part of this codebase chooses, states, or tests. Defaulting to
the policy that happened to print the better number would report that selection as a
property of the model.

The row should therefore be read with §20.3's sentence attached: **this solve is 1 ulp
from converging and 1 ulp from stopping, and which it does is not, on its own, evidence
about the port's physics.** *Why* it is that sensitive is open (§20.12) and is the thing
worth fixing; the default is not what fixes it either way. The 2.34e-03 `d objf` and 1.08e-01 `worst dx` against PROCESS that §19
left open are unchanged and remain §17/§24.10's business.

`run_cold_matrix.py` and `reference_cold_matrix.txt` are untouched by this change; their
regeneration and its provenance note are handled separately.

### 20.10 §15's CLARABEL table, re-measured eager and fused (2026-09-01)

§15's counts are the authority for every iteration number this project quotes, §17's
*"the gap is 1.26x"* included, and **every row of it predates `run_schedule`**. With the
default now `fuse_upstream=True` that table describes a path that is no longer the
default one, so here is the counterpart. `stellarator_helias`, CLARABEL throughout, one
process, `SAND_MAX_ITER = 500` / MDF `MAX_ITER = 800`, `dx` against the same cell's eager
run:

| cell | its | `conv` | outcome | `objf` | `max\|eq\|` | `dx` vs eager |
|---|---|---|---|---|---|---|
| MDF C3 `1e-8` eager | 67 | `2.54e-09` | converged | `1.21775735097` | `4.60e-11` | -- |
| MDF C3 `1e-8` **fused** | **67** | `2.54e-09` | converged | `1.21775735097` | `4.60e-11` | **`0.00e+00`** |
| MDF C3 `1e-6` eager | **58** | `8.91e-07` | converged | `1.21775795134` | `6.72e-08` | -- |
| MDF C3 `1e-6` **fused** | **58** | `8.91e-07` | converged | `1.21775795134` | `6.72e-08` | **`0.00e+00`** |
| MDF C2 `1e-8` eager | 45 | `7.23e-02` | `QSPSolver` | `1.22542131693` | `3.23e-03` | -- |
| MDF C2 `1e-8` **fused** | 45 | `7.23e-02` | `QSPSolver` | `1.22542131693` | `3.23e-03` | **`0.00e+00`** |
| MDF C2 `1e-6` eager | 45 | `7.23e-02` | `QSPSolver` | `1.22542131693` | `3.23e-03` | -- |
| MDF C2 `1e-6` **fused** | 45 | `7.23e-02` | `QSPSolver` | `1.22542131693` | `3.23e-03` | **`0.00e+00`** |
| SAND C3 `1e-8` eager | **90** | `8.09e-10` | converged | `1.21775735205` | `1.84e-06` | -- |
| SAND C3 `1e-8` **fused** | **108** | `7.96e-02` | `QSPSolver` | `1.22217407863` | `2.85e-02` | `6.12e-01` |
| SAND C3 `1e-6` eager | **80** | `1.73e-08` | converged | `1.21775769566` | `1.14e-04` | -- |
| SAND C3 `1e-6` **fused** | **108** | `7.96e-02` | `QSPSolver` | `1.22217407863` | `2.85e-02` | `6.12e-01` |
| SAND C2 `1e-8` eager | **231** | `2.70e-09` | converged | `1.21775734211` | `9.85e-07` | -- |
| SAND C2 `1e-8` **fused** | **500** | `9.88e-04` | cap(500) | `1.21774371415` | `1.07e-02` | `1.88e-03` |
| SAND C2 `1e-6` eager | **178** | `3.65e-07` | converged | `1.21775930434` | `1.93e-03` | -- |
| SAND C2 `1e-6` **fused** | **422** | `6.53e-07` | converged | `1.21776186746` | `1.96e-05` | `8.38e-04` |

**The control §15 asked for, and it splits.** MDF's four eager cells reproduce §15
*exactly*, `conv` included -- C3 **67** at `1e-8`, C3 **58** at `1e-6` with
`conv 8.91e-07` to three digits, C2 **45** giving up in the QP solver (§15's
"infeasible"). **SAND's do not**: §15 records C3 83 / **58** and C2 207 `unbounded`;
this tree gives C3 **90** / **80** and C2 **231 converged**. Nothing in this session
touched SAND's formulation, so those three rows moved between 2026-08-30 and now for
reasons that are not the fusion work and are not identified here -- they are simply
stale. **This is stated as a measurement of today's tree, not as a correction to §15's
own measurement of its tree.**

**MDF is bitwise invariant to fusion, in all four cells.** That is structural rather than
lucky, and §20.11 says why: MDF's outer driver closes over 294 context names and **not
one of them is a value the schedule computes** -- the whole MDA is re-run inside every
evaluation, so there is nothing a schedule-level fusion can move. Measured directly:
fusing `mdf.eager` against a fully eager walk moves **1 of 859** values
(`.physics.nu_star`, and that "difference" is a `nan` failing to equal itself), and the
**primed env the solve actually closes over is identical in 308 of 308 values**.

**SAND moves in all four.** C3 both ways is the §19 flip. C2 is new and is not a flip to
a worse answer in one case and is in the other: at `1e-8` fused runs to the 500 cap where
eager converged at 231, and at `1e-6` fused converges in 422 where eager took 178, both
landing within `1e-3` relative of the eager answer.

### 20.11 MDF under the §20.3 reproducer -- and the control that overturns the obvious reading

**The perturbation §20.3 uses cannot be applied to MDF at all**, and that is the first
result. `.tfcoil.a_tf_turn_steel` and `.heat_transport.etath_liq` are **not in MDF's
outer context and not in its primed env** (measured: `False` for both, on a context of
294 names). In SAND they are outputs of the 25 `Call` steps that run *once*, outside the
`Drive`; in MDF the equivalent computation happens *inside* every evaluation. So MDF has
no analogue to perturb, which is exactly why fusion cannot reach it.

The two things MDF *does* hand its driver once and freeze are the five `^guess.*` start
ports (filled by `prime`) and the eight design start values. Both were perturbed, eager
throughout, `1e-8`:

| run | its | `conv` | outcome | `objf` | `max\|eq\|` |
|---|---|---|---|---|---|
| MDF C3, unperturbed | **67** | `2.54e-09` | converged | `1.21775735097` | `4.60e-11` |
| `^guess.power.delta_eta` `+-1`, `+-2 ulp` | **67** | `2.54e-09` | converged | `1.21775735097` | `4.60e-11` |
| `^guess.physics.fusden_alpha_total` `-1 ulp` | **67** | `2.54e-09` | converged | `1.21775735097` | `4.60e-11` |
| `^guess.fwbs.f_ster_div_single` `-2 ulp` | **67** | `2.54e-09` | converged | `1.21775735097` | `4.60e-11` |
| **`.physics.rmajor` `-1 ulp`** | **114** | `4.22e-02` | **`QSPSolver`** | `1.21799442644` | `6.86e-04` |
| **`.physics.rmajor` `+1 ulp`** | **85** | `5.49e-02` | **`QSPSolver`** | `1.22624525641` | `4.50e-03` |
| `.physics.hfact` `-1 ulp` | **86** | `6.08e-09` | converged | `1.21775738697` | `3.59e-09` |

And the same two design starts under SAND, for symmetry:

| run | its | `conv` | outcome | `objf` | `max\|eq\|` |
|---|---|---|---|---|---|
| SAND C3, unperturbed | **90** | `8.09e-10` | converged | `1.21775735205` | `1.84e-06` |
| `.physics.rmajor` `-1 ulp` | **71** | `8.54e-10` | converged | `1.21775739006` | `1.00e-08` |
| **`.physics.hfact` `-1 ulp`** | **275** | `3.78e+01` | **`QSPSolver`** | `4.89365312691` | `9.99e+01` |

**Two findings, and the second is the one that matters.**

1. **A `^guess.*` start is inert, bit for bit.** Four perturbations, four runs identical
   to all 17 digits in `objf` and identical in `x`. That is not luck: a `^guess.*` value
   starts an inner solve that is then driven to its own tolerance inside every
   evaluation, so a last-bit change in where it starts is absorbed before the outer
   driver ever sees a condition. **This is a real robustness property of MDF's shape**,
   and it is why fusion cannot move MDF.

2. **MDF is *not* robust to the perturbation itself.** One ulp on `.physics.rmajor` --
   the design start, which nothing re-converges -- takes MDF from 67 converged to **114
   and a QP-solver failure** at `-1`, and to **85 and a QP-solver failure** at `+1`.
   Both directions, and both worse than the unperturbed run. SAND survives the same
   perturbation on `rmajor` and fails catastrophically on `hfact` (`objf 4.89`,
   `max|eq| 99.9`).

**So the reading "SAND is fragile, MDF is sound" is wrong, and it was the reading the
first draft of this section was heading for.** Both formulations flip on one ulp of what
their SQP starts from. What differs is only **which inputs are exposed**: SAND freezes a
once-computed context that a jit boundary can reassociate, MDF freezes only a design
start and a set of guesses that get re-converged. MDF's fusion-invariance is a
consequence of that exposure difference, not of the underlying solve being better
behaved.

### 20.12 What the fork is not: four diagnostics, three explanations weakened

The obvious inference from §20.3 -- *"the problem is intrinsically ill-conditioned"* --
is an inference, and this project's record is that it has been the wrong one every
previous time (§17 OSQP-for-CLARABEL, §16.1b `MAX_ITER`, §12.2 a cap, §24.10 an
evaluation-mode file, §24.9 unrolled loops; `SlsqpDriver`'s docstring on five overturned
diagnoses in one session). So it was tested. Instrumented run of both SAND branches --
unperturbed, and `.tfcoil.a_tf_turn_steel -1 ulp` -- recording the full condition vector
and the Jacobian VMCON is handed at every SQP iteration.

**(a) The active set does not change at the fork, and `c24` is not in it there.** The two
branches are bit-identical through iteration 13. At **14** the *only* recorded quantity
that differs is `conv` (`0.007118612933389929` against `...93017`, `4e-13` relative);
every equality, every inequality and `x` itself are still bit-identical. At **15** all of
them differ. So the divergence enters through the convergence parameter one iteration
before it reaches any constraint -- there is no constraint entering or leaving the active
set at the fork. Across iterations 11-17 the active set (`|ie| < 1e-6`) is
`^cond.constraints.c83` alone, with `c35` joining once at 12. `c24` is at `-3.9e-02` at
13 and `+1.0e-03` at 14 -- nowhere near active. **The `c24` kink is not the trigger.**

**(b) The kink is present in both formulations, and only one of them is fusion-stable.**
At the *solution* the active set is the same four constraints in both -- `c24`, `c83`,
`c62`, `c35` -- matching §17.2's "one of exactly four binding inequalities". MDF carries
`fast_alpha_beta`'s clamped square root in its active set exactly as SAND does, converges
at `42` condition number, and still flips on 1 ulp of `rmajor` (§20.11). **A kink both
formulations share cannot explain a fragility both formulations have and a fusion
sensitivity only one of them has.** Smoothing the clamp was therefore not run; it is not
ruled out as a *fix*, only as *the* explanation.

**(c) The conditioning is a number, and it is modest -- but it is also wildly uneven.**
Condition number of the constraint Jacobian VMCON is handed, over the whole trajectory:

| | min | median | max |
|---|---|---|---|
| SAND branch A (90 its) | `1.244e+04` | `1.660e+04` | `3.502e+04` |
| SAND branch B (108 its) | `1.241e+04` | `1.660e+04` | `3.503e+04` |
| **MDF (67 its)** | **`3.783e+01`** | **`4.241e+01`** | **`1.078e+03`** |

`1.7e4` in double precision leaves ~11 digits; it does not by itself turn a `4e-16` input
into a different answer, and **MDF at `42` flips anyway**. So the condition number is not
the mechanism. What it *is* is a clean measurement of a formulation gap nobody had taken:
**SAND's entire conditioning excess is one row.** Dropping each row in turn at iteration
13 (`cond 1.758e+04`):

| row dropped | resulting `cond(J)` | its value | its row norm |
|---|---|---|---|
| **`^cond.stellarator.wp_width_r_min`** | **`6.75e+01`** | `+1.67e-02` | **`2.892e+03`** |
| `^cond.constraints.c24` | `1.760e+04` | `-3.94e-02` | `3.01e+00` |
| `^cond.constraints.c83` | `1.843e+04` | `+6.36e-12` | `7.63e-01` |
| ... every other row | `1.76e+04`-`1.15e+08` | | `0.09`-`5.4` |

The largest singular value of the whole 20x14 Jacobian is `2.892e+03` and the next is
`8.589e+00`; that top value **is** this row. It is the same row §19.2 found holding the
worst equality at the 108-stopped point, and the same one `VmconDriver.condition_scale`'s
docstring already describes as *"the coil-island (`Intersect`) residual, the largest row
left and the only one whose units are genuinely not its unknown's"* -- where equilibrating
it by its row norm took `2.1e4` to `85` and C2 from 62 iterations to `max_iter`. That
measurement stands; what is new is the **MDF comparison**, which says the well-conditioned
version of this problem exists (`42`) and is the one that does not expose this residual as
a constraint at all. `residual_condition_scales` gives this row a factor of `1.3948`,
which normalises its *value* and leaves its *derivative* at `2892`.

**(d) Nothing was dropped or frozen.** The named suspicion -- a missing producer or a
value held at its seed -- is measurable from the assembly's own report and is clean on
this configuration: `degenerate: 0`, `array_valued: 0`, `omitted: 0`, `external: 0`, with
4 `residualised` problems, 8 design, 2 PROCESS equalities and 12 PROCESS inequalities. An
early reading that `c83` was an identity (it sits at `~6e-12` for most of the trajectory)
is **withdrawn on measurement**: it starts at `-2.361e-01` and converges early, in *both*
formulations, so it is an ordinary binding constraint and not a degenerate row.

**Where that leaves the diagnosis.** Measured: the fused/eager difference is one ulp of
one variable; both formulations flip on one ulp of a design start; the constraint
Jacobian is modestly conditioned in SAND and well conditioned in MDF; the active set does
not change at the fork; nothing is frozen or dropped. **Not established:** what actually
amplifies `4e-16` into a different outcome over 14 iterations. It is not the `c24` kink
alone, not the condition number alone, and not a frozen value. The next thing to look at
is the QP solve itself -- every failing run in §20.10 and §20.11 ends in `QSPSolver`,
i.e. `cvxpy` refusing a subproblem, which is the one component nobody has instrumented
and which is exactly where §17's last confident diagnosis turned out to live.

### 20.13 What §17's "1.26x" headline should now say

`next_steps.md` is not this file's to edit; this is the wording, with the measurements
behind it.

§17 rests on §15's *"58 iterations in both formulations against PROCESS's 46"*, and the
strength of the claim is that **two independent formulations agree**. After §20.10:

- **MDF's 58 stands, and is stronger than it was.** It reproduces exactly (`conv
  8.91e-07`), and it is now also measured to be **bitwise invariant to the whole-program
  jit** -- the same 58, the same `objf` to 12 digits, `dx 0.00e+00`. Fusion does not
  touch it, and §20.11 says why structurally rather than by luck.
- **SAND's 58 does not reproduce in this tree, eager or fused.** Cold at `1e-6` it is
  **80** eager and **108 stopped** fused. Its `1e-8` row is 90, not §15's 83, and its C2
  row is 231 converged, not 207 `unbounded`. Those moved for reasons predating this
  session and not identified here.
- **Neither 58 survives the ulp reproducer.** MDF's goes to 114/85 with a QP failure on
  `+-1 ulp` of `.physics.rmajor` (§20.11).

Suggested correction: *"the gap is 1.26x"* should be attributed to **MDF alone** -- 58
against PROCESS's 46, reproduced and jit-invariant -- and the *"both formulations agree
on 58"* clause withdrawn until SAND's rows are re-measured and the move from 83/58 to
90/80 is explained. The 1.26x ratio itself is unchanged; what is withdrawn is the
independent second witness.

## 21. The fork is one cell of the Jacobian, and a jit-only `nan` in the SVD (2026-09-01)

§20.12 closed with *"the next thing to look at is the QP solve itself -- every failing
run ends in `QSPSolver`, i.e. `cvxpy` refusing a subproblem, which is the one component
nobody has instrumented"*. It is instrumented here. The QP is **not** where the
divergence starts, and `cvxpy` is not doing anything unexpected: the fork enters through
**one cell of the constraint Jacobian**, one iteration earlier than §20.12 could see, and
`cvxpy` is deterministic across 14 consecutive bit-identical subproblems.

Two further things came out of the same instrument, both of them independent of the fork:
a **jitted and an unjitted `jax.jacfwd` of the same map at the same point do not agree**
(§21.2), and `st_regression`'s MDF arm was blocked by a `nan` that exists only under
`jit`, whose mechanism is a repeated singular value inside `PFCoil.solv` (§21.3, fixed).

**Provenance, because the env moved under this measurement.** Part way through, `cottax`
was reinstalled editable in `process_port` -- until then it resolved to a snapshot copy,
not to `~/jaxgraph/src`, and the tree behind it gained a `DriverOut` mechanism. Every
comparison in this section is **between two runs in one process**, so none of them
straddles the change; and the two that most needed it were re-taken afterwards and
reproduce **digit for digit**: the fork control (90 against 108, first differing QP 14,
the same `conv` values to all 17 digits) and §21.2's jit/eager Jacobian table (same 105
cells, same `5.154e-13`). The cottax change is therefore measured to be inert here, not
assumed to be. The one number taken before the swap and not re-taken is flagged where it
appears.

### 21.1 (A), not (B): the QP's *inputs* differ, in one derivative cell

Both SAND branches -- unperturbed, and `.tfcoil.a_tf_turn_steel -1 ulp` in the env the
`Drive` is handed, §20.3's reproducer, eager throughout -- run with `pyvmcon`'s problem
object wrapped in a recorder (so **every** evaluation is captured, line-search trials
included, not only the ones the callback sees) and `pyvmcon.vmcon.solve_qsp` replaced by
a verbatim copy of itself that also keeps `cvxpy`'s `Problem.status`, `solver_stats`,
`delta` and both multiplier vectors.

**Every VMCON problem evaluation, in order.** Branch A makes 179, branch B 217. The
first one whose result differs is **#27**, and at #27 the design point `x` is
bit-identical:

| quantity | agreement at evaluation #27 |
|---|---|
| `f`, `eq`, `ie` (every condition **value**) | bit-identical |
| `df`, `die` | bit-identical |
| **`deq`** | **1 of 112 cells differs, by 1 ulp** |

The cell is `^cond.constraints.c16` / `.stellarator.wp_width_r_min`:
`-0.16700190547850377` against `-0.1670019054785038`, `1.66e-16` relative. Evaluations
0-26 agree in all six quantities, and `x` at #27 is the same 14 floats, so this is the
**direct** effect of the perturbed context at that point and not an accumulation: a
deterministic map, the same `x`, one different context value, one different output bit.

**What that says about the perturbation itself, and it is worth stating plainly:** one
ulp of `.tfcoil.a_tf_turn_steel` is invisible in *every condition value at every point
the solve visits* and visible in *exactly one derivative cell*. The conditions can agree
bitwise while their derivatives do not, because `jax.jacfwd` closes over the context --
which is why §20.12's iteration-by-iteration diff of the condition vector saw nothing
until `conv` moved.

**The QP, subproblem by subproblem.**

| | branch A | branch B |
|---|---|---|
| first QP whose **inputs** (`x`, `B`, `f`, `df`, `eq`, `deq`, `ie`, `die`) differ | \-- | **14** |
| first QP whose **outputs** (`delta`, `lamda_eq`, `lamda_ie`) differ | \-- | **14** |
| QPs 0-13 | bit-identical inputs | **bit-identical outputs** |

**So it is (A), and (B) is refuted by measurement.** Fourteen consecutive QP subproblems
with bit-identical data returned bit-identical search directions and multipliers;
CLARABEL is deterministic on this problem. The QP's outputs differ at 14 because its
*inputs* do.

At QP 14, with `x`, `f`, `df`, `eq`, `ie` and `die` all still bit-identical:

| QP 14 | cells differing | max `|ulp|` | max relative |
|---|---|---|---|
| `deq` | **1 of 112** (the same `c16`/`wp_width_r_min` cell) | 1 | `1.66e-16` |
| `B` (VMCON's Hessian approximation) | 135 of 196 | 160 | `1.91e-14` |
| `delta` (the search direction) | 14 of 14 | `9.6e+06` | `1.79e-09` |
| `lamda_equality` | 8 of 8 | `9.3e+05` | `1.68e-10` |
| `lamda_inequality` | 12 of 12 | `2.7e+10` | `4.06e-06` |

`B` differs before `deq` reaches the QP because `calculate_new_B` is fed the *line
search's* evaluation at the previous iteration -- #27 and #28 -- which is exactly where
the one cell first moved. So the amplification chain, end to end and every step of it
measured, is:

**1 ulp in one `deq` cell -> 160 ulp in `B` -> `1.8e-09` in the search direction ->
`4e-06` in a multiplier that sits at `1e-11` -> `conv` differs by `4e-13` at iteration 14
(§20.12's observation) -> `x` differs at iteration 15 -> 18 more iterations and an
infeasible QP at 108.**

### 21.2 A jitted and an unjitted `jax.jacfwd` do not agree on `stellarator_helias`

Asked as a correctness question in its own right, and the answer is that **they differ**,
by about a thousand times more than the fusion perturbation that flips the solve.

`VmconDriver` builds `jacobian = jax.jit(jax.jacfwd(flat_conditions))`. Compared against
a bare `jax.jacfwd(flat_conditions)` at the same points of the same cold solve (control
first: **repeated calls are bit-identical both ways**, so neither side is itself
nondeterministic):

| at SQP iterate 27 | jit vs eager |
|---|---|
| Jacobian, cells differing | **105 of 294** |
| Jacobian, max relative | **`5.15e-13`** (`^cond.constraints.c67` / `.physics.temp_plasma_electron_vol_avg_kev`) |
| objective-gradient row, max relative | `3.69e-13` (`d objf / d .stellarator.wp_width_r_min`) |
| values, `^cond.constraints.c16` | `1.18e-13` relative |
| values, `^cond.stellarator.wp_width_r_min` | `8.84e-12` relative |
| values, `^cond.fwbs.f_ster_div_single` | `6.63e-11` relative |
| values, `^cond.constraints.c83` | `3.71e-05` relative -- on a residual of `6e-12` |

(The one entry that looks alarming, a residual at `-1.3e-15` against `-8.9e-16`, is a
cancellation to nothing and is reported as `4.5e-01` relative by arithmetic rather than
by meaning.) The same picture holds at iterates 13, 40 and 178: 102-107 of 294 cells,
the same eight rows touched every time.

**Put beside §20.2 this is the headline of this section.** Fusing the 25 upstream `Call`
steps moves `.tfcoil.a_tf_turn_steel` by `4.4e-16` and that flips a converged solve into
an infeasible stop. The jit boundary *inside the driver* -- which has been there the
whole time, on every run, in both fusion policies (§20's own wording caution) -- moves
the Jacobian the SQP is handed by up to `5e-13`. **The schedule-fusion axis is not the
port's largest source of last-bit motion in the SQP's data; the driver's own jit is,
by three orders of magnitude.** That does not make the fused/eager finding wrong -- it
makes it a small instance of a general fact, and it retires "fuse or do not fuse the
upstream groups" as anything that could be called a determinism policy.

Nothing here says which of the two Jacobians is *right*. Both are correct to the
precision floating point allows; XLA reassociates when it fuses and that is all this is.
What it does say is that the port's answer on this configuration is a function of a
compilation decision, and no `fuse_upstream` setting changes that.

**And the difference is load-bearing for the answer, not merely present in it.** The
whole cold SAND solve was re-run with `VmconDriver`'s `jax.jit(jax.jacfwd(...))` replaced
by a bare `jax.jacfwd(...)` -- the *value* path keeps its jit, so the Jacobian's
compilation is the only thing that changes, and the schedule is §19.3's eager one with no
perturbation anywhere:

| `stellarator_helias`, cold SAND, C3 `1e-8`, eager schedule | its | outcome | `objf` | `max\|eq\|` | `conv` | s |
|---|---|---|---|---|---|---|
| Jacobian **jitted** (the default, and every number in §19-§20) | **90** | converged | `1.2177573520529628` | `1.84e-06` | `8.09e-10` | 12 |
| Jacobian **unjitted** | **70** | `QSPSolver` | `1.2226348012227457` | `2.19e-03` | `8.49e-02` | 218 |

A **third** outcome from the same problem, worse than either of §20's two, reached by
changing nothing but how the derivative is compiled. (Run twice, either side of the
cottax reinstall, byte-identical `objf` and `x` both times.) The 18x wall-clock cost is
also why the jit is there and is not going away; the point is not that the unjitted
Jacobian is better -- it is that **three different compilation choices give three
different answers**, and §20.3's ulp perturbation is the smallest of the three levers,
not a special one.

### 21.3 `st_regression`'s non-finite jitted row: a repeated singular value in `PFCoil.solv`

`run_cold_matrix`'s `st_regression` MDF arm fails at the **first** SQP evaluation with
`ValueError: the SQP was handed a non-finite problem`, `non-finite derivative rows:
['^cond.constraints.c16']`, every condition *value* finite. HEAD `9335f784` recorded that
the unjitted `jax.jacfwd` is fully finite at the same point. Reproduced here directly:
the whole `c16` row is `nan` under `jax.jit(jax.jacfwd(f))` and finite in all 14 columns
under `jax.jacfwd(f)`, with the value finite either way.

**`safe_math.py` is not involved, and was the first suspect.** The obvious reading -- a
`select`/`where` guard whose dead branch leaks an `inf` into the tangent -- is wrong here:
`safe_pow`/`safe_sqrt`'s double select keeps the untaken branch's *argument* away from
zero, so neither branch ever evaluates `0.0 ** (p-1)`, and §24.12's `lax.select` rewrite
did not change that. The file was not edited. The `safe_sqrt` in `_pf_coil_sizes` sits
*downstream* of the real fault and merely carries it.

**Localised, not guessed.** One `jax.jvp` of the entire MDA with every intermediate value
as an output -- 17,246 scalar slots -- jitted and eager. 405 slots have a non-finite
tangent under jit and **none** does eager. Ordered by the step that produces them, the
earliest is

    step 116  call[.tokamak.pf_coil.equilibrium_currents]   .pf_coil.ccls[1..3]

and the other 402 are its downstream cone: `c_pf_cs_coil_*_ma`, `c_pf_cs_coils_peak_ma`,
`n_pf_coil_turns`, `r_pf_coil_outer_max`, `b_pf_coil_peak`, the cryostat volumes, the
`t_plant_pulse_burn` fixed-point block, the PF masses, the costs, and finally `c16`
(net electric power), which is the row the SQP refuses.

**The mechanism.** `_solv` (`models/pfcoil/currents.py`, porting `PFCoil.solv`) takes
`jnp.linalg.svd` of the damped least-squares matrix. On `st_regression`'s equilibrium
solve that matrix has **a repeated singular value**:

    sigma = [1.25498145e-07, 1.00000000e-09, 1.00000000e-09]

and the repetition is **structural, not incidental**. There is one field point, so the
two field rows have rank 1; the smoothing block is `alfa * I` over `n_groups` columns; so
with `n_groups >= 3` at least two directions carry nothing but `alfa`, and both come back
as `sigma = alfa = 1e-9`. JAX's SVD JVP computes `dU`/`dV` by dividing by
`sigma_i^2 - sigma_j^2`. Whether that divisor is *exactly* zero is a rounding accident,
and it is decided by the compilation: bit patterns of the same `sigma`, printed from
inside the same program with `jax.debug.print` of
`lax.bitcast_convert_type(sigma, int64)`:

| | sigma[1] | sigma[2] |
|---|---|---|
| eager (op by op) | `4472406533629990550` | `4472406533629990549` |
| **one `jax.jit`** | `4472406533629990549` | `4472406533629990549` |

One ulp apart eagerly, **bit-identical fused** -- so `1/(sigma_i^2 - sigma_j^2)` is
`1/0` in the jitted program only. Reproduced away from PROCESS entirely, in five lines:
`jvp` of `V diag(1/s) U^T b` at `diag(1, 2, 3)` gives `[-1.83, -0.92, -0.61]`; at
`diag(1, 2, 2)` gives `[nan, nan, nan]`; at `diag(1, 2, nextafter(2))` gives
`[-2, -1, -1]`.

**Was the finite eager derivative right?** Yes, and this is worth knowing before deciding
how much it polluted: at the same point the SVD JVP, the analytic least-squares
sensitivity and a central finite difference of `_solv` itself agree to **8 significant
figures** in three independent perturbation directions (the tangents are genuinely of
order `1e+16` there -- `sigma_min` is `1e-9` -- and the finite difference confirms that
too). So the defect was `nan` or nothing; it was not silently wrong numbers. It was
`nan` only because the tie landed exactly, which is the part a fusion decision chooses.

**The fix** (`models/pfcoil/currents.py`): a `jax.custom_jvp` on `_solv`. The **value** is
the existing SVD path, untouched, so no configuration's numbers move. The **tangent**
comes from the normal equations the solution already satisfies,

    A^T A dx = dA^T (b - A x) + A^T (db - dA x),

an `n_groups x n_groups` linear solve that is defined whenever `A` has full column rank
and does not care whether two singular values coincide. `U` and `V` are individually
undefined at a repeated singular value; `x` is not, and `x` is all anyone reads.

**After, measured:**

| | before | after |
|---|---|---|
| `st_regression`, `c16` Jacobian row, jitted | `nan` in all 14 columns | finite |
| `st_regression`, `c16` Jacobian row, eager | finite | finite |
| `st_regression` MDF C3 `1e-6`, cold | `ValueError` at evaluation 0 | **4 its, converged**, `conv 4.95e-09`, `max|eq| 4.76e-06` |
| the same row, eager vs fused | \-- | `dx 0.00e+00` |
| `stellarator_helias` SAND C3 `1e-8` eager | 90, converged, `1.2177573520529628` | identical, `dx 0.00e+00` |
| `large_tokamak_nof` SAND C3 `1e-8` fused | 10, converged, `1.6000000000354158` | identical count and `objf`, `dx 1.58e-10` |
| `large_tokamak_nof` MDF C3 `1e-8` fused | 7, converged | identical count and `objf`, `dx 2.11e-12` |

(The two tokamak `dx` columns are not zero because the derivative is now computed by a
different arithmetic path; the iteration count, the status and `objf` to twelve digits
are unchanged. `stellarator_helias` is bitwise unmoved because the stellarator graph has
no PF-coil equilibrium solve to reach.) `tests/functional_process/models/pfcoil`:
**508 passed with `--fp-gradients`**, which includes
`TestCalculateEfcCurrents::test_gradient_agreement` against **PROCESS's own finite
difference** on both the legacy and the fuzz sample.

**The limit of the rule, stated rather than buried.** With a singular value at or below
`_SIGMA_FLOOR` (`1e-10`) `_solv` reproduces PROCESS's carried-`zvec` behaviour and is no
longer the least-squares solution, so this rule is not its derivative there. It is not a
regression: the function is discontinuous in `sigma` at the floor and has no derivative
there in any spelling, and the SVD JVP was `nan`-prone there for the same tie reason. On
every configuration measured `sigma_min` is `alfa = 1e-9`, an order of magnitude above the
floor, so the floor is not active on any live path.

### 21.4 What `cvxpy` actually reports, and where the failing run really stops

§17's lesson was that a solver default can hide in plain sight, so this was checked at the
call site rather than inferred from `VmconDriver.qsp_solver`'s default.

- **CLARABEL is genuinely the solver.** `qsp.solver_stats.solver_name` is `'CLARABEL'` on
  **all 109** subproblems of the failing branch. (`cvxpy.CLARABEL` is the string
  `"CLARABEL"`, so the driver passing the name and PROCESS passing `cvxpy.CLARABEL`
  are the same argument.)
- **Status counts: 108 `optimal`, 1 `infeasible`, 0 `optimal_inaccurate`.** That is a
  second, independent confirmation of §15's withdrawal of the "a fifth to a third of QP
  subproblems solve inaccurately" claim.
- **The failing subproblem is infeasible, not inaccurate.** CLARABEL exits after **3**
  interior-point iterations with `qsp.value = inf` and `delta.value is None`; the
  preceding 108 took 10-13 iterations each and returned `optimal`.

**And one thing nobody had noticed, which the recorder makes obvious.** The point at
which the QP is infeasible is *far worse* than the point before it:

| | iteration 107 (last one the callback sees) | iteration 108 (where the QP fails) |
|---|---|---|
| `max\|eq\|` | `2.85e-02` | **`6.44e-01`** |
| `min ie` | `+1.12e-11` | **`-2.01e+00`** |
| `cond(J)` | `1.65e+04` | **`2.56e+06`** |
| `\|df\|` | `5.59e+00` | `1.90e+01` |

So the published row *"108, stopped, `max|eq| 2.85e-02`"* is **iteration 107's** state --
the last one a callback fires for -- and VMCON's actual last point is one line-search
step beyond it, at a place where the linearised problem has no feasible point at all. The
open question §20.12 handed on should therefore be re-aimed: not *"why does the QP
fail"* (it fails because it is handed an infeasible linearisation) but **"why does the
line search accept a step from `max|eq| 2.9e-02` to `6.4e-01`"** -- i.e. `perform_linesearch`
and the `mu` penalty update, which are still uninstrumented.

### 21.5 What is not ruled out

- **Why the same 1-ulp context change is invisible in every value and visible in one
  derivative cell** is measured but not explained at the level of the arithmetic: which
  expression in `c16`'s path amplifies `a_tf_turn_steel` into
  `d/d wp_width_r_min` and not into the value has not been traced.
- **Which of the three outcomes in §21.2's table is the right answer**, if any. The
  unjitted-Jacobian run is not a control that says the jitted one is wrong; it is a third
  data point saying the answer depends on the compilation. Nothing here identifies a
  formulation change that would make all three agree, and `SlsqpDriver` was not put
  through this axis (§20.4 put it through the ulp one).
- **The line search.** §21.4 names it as the next thing to instrument and does not
  instrument it. Nothing here says the accepted step is *wrong* -- only that it is large
  and lands somewhere the QP cannot linearise.
- **Whether `st_regression`'s newly-converging MDF row is a good answer.** It reports
  `min ie -1.51e-03`, i.e. a violated inequality at the point it calls converged. That is
  the cold matrix's business, not this section's; what is established here is only that
  the row is no longer blocked by a `nan`.
- **Other configurations' PF-coil derivatives.** The `custom_jvp` changes the derivative
  path wherever `_solv` runs, which is every tokamak. Two were measured (`st_regression`,
  `large_tokamak_nof`); `large_tokamak_eval`, `low_aspect_ratio_DEMO` and
  `spherical_tokamak_eval` were not, and belong to the next cold-matrix regeneration.

## 22. The SQP drivers go through `jax.pure_callback`, and the SAND schedule becomes one program (2026-09-01)

`VmconDriver` and `SlsqpDriver` now run their host-side solve inside one
`jax.pure_callback`, exactly the way `cottax.drivers.SLSQPDriver`
(`~/jaxgraph/src/cottax/drivers/scipy_slsqp.py`) does, and report `(Steps, Converged,
Status)` through `DriverOut` ports instead of through a mutable `Outcome` dict.
`drivers.Outcome` is deleted and `Status` moved from `mdf` to `drivers`.

### 22.1 The verdict, before and after

`sand_harness.run_schedule` probes each schedule with a whole-schedule jit once and
caches the verdict; `schedule_verdict` reads it back. On `stellarator_helias`'s cold SAND
solve schedule, verbatim:

**Before**

    (False, "TracerArrayConversionError: The numpy.ndarray conversion method
    __array__() was called on traced array with shape float64[14]
    The error occurred while tracing the function run at
    /home/tbogaarts/miniconda/envs/process_port/lib/python3.12/site-packages/equinox/_jit.py:43
    for jit. This concrete value was not available in Python because it depends on the
    values of the arguments dynamic_nodonate['first'][283], ... [296].
    See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerArrayConversionError")

`float64[14]` is the block's 14 unknowns; the 14 named arguments are their `^guess.*`
ports. The `__array__` call was `design_scale(np.asarray(flat_start))` -- PROCESS's own
`1/x_start` conditioning, which needs *values*.

**After**

    (True, None)

### 22.2 Compiles and wall time -- `stellarator_helias` SAND, one process each

Counted by patching `jax._src.compiler.backend_compile_and_load`. The solve schedule is
run twice in the same process.

| | verdict | first call | second call |
|---|---|---|---|
| before, probe on (probe fails, walks) | `(False, TracerArrayConversionError)` | 9 compiles, 11.4 s | 2 compiles, 9.30 s |
| before, `whole=False` (the cold matrix's path) | `(False, None)` | 9 compiles, 12.0 s | 2 compiles, 8.59 s |
| **after, probe on** | **`(True, None)`** | **4 compiles, 10.8 s** | **2 compiles, 7.63 s** |
| after, `whole=False` | `(False, None)` | 11 compiles, 10.8 s | 3 compiles, 7.96 s |

**"One compile" here means the optimiser is *outside* the compiler, not compiled by
it.** The reference number this was measured against -- the in-graph root find's *0
compiles / 0.02 s* second call (§21's tree, `next_steps.md` §24.11) -- is not reachable
here and it would be wrong to report a number as if it were. That solve is
`optimistix.Newton` inside a `lax.while_loop`: the whole iteration is *in* the XLA
program, so a second call is a cache hit and a dispatch. This one is `pyvmcon` +
`cvxpy` + a Python line search behind a host callback: the second call re-runs all 108
SQP iterations on the host, at full price. What the wrap buys is that the block stops
being a hole in the schedule -- everything around it fuses -- not that VMCON got faster.

**The 2 compiles on every call, including the second, are the driver's own inner jits.**
`jax.jit(flat_conditions)` and `jax.jit(jacfwd(flat_conditions))` are built inside
`host`, so they close over that call's recombined `ConditionMap` and are a fresh function
object each solve: jax's cache misses and retraces. This is unchanged behaviour (the
before-rows show the same 2), and it is the obvious next saving -- hoisting `live` out of
the closure into an argument would make them cacheable across solves. Not done here: it
changes what the jitted function sees, and this change's whole acceptance criterion was
that nothing the solver sees moves.

**The inner jits were kept, deliberately, unlike the `cottax` driver being copied.**
`SLSQPDriver` leaves its model eager because a cottax test model is a handful of ops. One
evaluation here converges most of a PROCESS graph; `next_steps.md` §24.11's split (the
SAND `Drive` costing 105.4 s of 108 on `large_tokamak_nof`, at 729 compiles, when run op
by op) is what that would cost. The `pure_callback` boundary is per *solve*, not per
iteration, so there is nothing about the wrap that makes an inner jit wrong.

**First-call wall time is unchanged within noise** (11.4 s / 12.0 s before, 10.8 s
after, two runs). That is the expected result and worth stating as such: one host round
trip per solve is not a cost anything can measure against a 108-iteration SQP.

### 22.3 The cold matrix: every SAND row reproduces bitwise

Full `run_cold_matrix.py --provider` pass, 7 configurations, 481 s, diffed against
`functional_process/reference_cold_matrix.txt`.

**All five SAND rows are byte-for-byte identical**, `st_regression`'s `FAILED` included:

    stellarator_helias    SAND  108 SQP it  stopped    1.22217408  max|eq| 2.85e-02  min ie 1.12e-11
    helias_5b             SAND    7 SQP it  converged 0.764215517  max|eq| 1.10e-13  min ie 7.02e-02
    large_tokamak_nof     SAND   10 SQP it  converged         1.6  max|eq| 4.51e-06  min ie -9.58e-05
    low_aspect_ratio_DEMO SAND  500 SQP it  cap(500)  -0.401520642 max|eq| 1.09e-11  min ie -3.32e-03
    st_regression         SAND       FAILED  KeyError: VarPath(^cond.numerics.objf)

The `stellarator_helias` row is the sensitive one (§21.2: a `5.15e-13` Jacobian
difference moves it between 70 its/`QSPSolver` and 90 converged), and it reproduces to
the last printed digit -- `1.2221740786343283`, `0.02851456050213086`,
`1.1216805262392882e-11`, and all eight `ixc` values -- on both the walk path and the
whole-jit path. So the `pure_callback` boundary moves nothing here. The explicit
`float64` conversion at the boundary is why: the reference driver converts for exactly
this reason and the same conversion was written here.

**Four MDF rows moved, and none of the movement is this change's.** They are the
regeneration §21.5 asked for:

| row | reference | now | cause |
|---|---|---|---|
| `st_regression` MDF | `FAILED`, non-finite derivative row `^cond.constraints.c16` | converged in 4, `min ie -1.51e-03` | §21.3's `custom_jvp` on `_solv` |
| `large_tokamak_eval` MDF | `max\|eq\|` 3.53e-14, blks 247, drvn 6 | 3.47e-14, blks 25, drvn 2 | ditto, plus the in-graph root find's `interior_*` shape columns |
| `spherical_tokamak_eval` MDF | blks 245, drvn 4 | blks 29, drvn 2 | in-graph root find shape columns |
| both eval rows' notes | no "Stated IN the graph" clause | present | ditto |

The argument that this is not the callback wrap, stated so it can be checked rather than
believed: (a) both `*_eval` rows are `RootFind`s answered by `MdfNewtonDriver`, which
this change does not touch at all; (b) `st_regression` MDF failed on a `nan` *derivative*
row, and nothing in a `pure_callback` around a solve can make a `nan` finite, whereas
§21.3's `custom_jvp` is documented as doing exactly that for exactly that row; (c) the
reference table lacks the "Stated IN the graph" note that HEAD's own `cold_mdf` emits
unconditionally on a root-find row, so it predates that code and is stale for those rows
independently of anything here. What is **not** established is a controlled before/after
for those four rows: that would need the matrix re-run with this change alone reverted,
and it was not done.

### 22.4 The latent mis-binding in `sand_harness._driven_runner`

`_driven_runner` re-implements `cottax.evaluate.Drive.__call__` so the driver can stay
eager while the body is jitted. Its binding read

    if len(converged) != len(step.unknowns): raise ValueError(...)
    env.update(zip(step.unknowns, converged, strict=True))

-- the unknowns alone, where `Drive.__call__` binds `self.unknowns + self.reports`
(`evaluate.py:373-388`) and `AbstractDriver.__call__`'s contract is *"the problem's
unknowns, positionally, ... followed by one value per kind in `reports`, in that order"*
(`problem.py:684`). Nothing had caught it because every driver that had ever reached this
function reported nothing. It is fixed to `step.unknowns + step.reports`, and
`test_driven_runner_binds_reports_not_unknowns` in `tests/functional_process/test_sand.py`
is the case that would have.

**The expected failure mode is not the available one, and that is a correction worth
recording.** The natural worry -- and the one this fix was briefed as addressing -- is
that the old code "either raises spuriously or silently binds report values to unknown
paths". Only the first is reachable. A driver returns its unknowns *first* and its
reports after (`problem.py:684`), so `zip(step.unknowns, answered)` pairs every unknown
with its own value whether or not `strict=True` is set; the truncation would have dropped
the verdict, never mis-assigned it. So the bug's entire reachable surface was a loud
`ValueError` at the first driver that reported anything, which is what happened. Reported
here rather than quietly fixed: a tidy "found and fixed a silent mis-binding" would have
been the more flattering sentence and the false one.

### 22.5 What the wrap costs, stated

- **No JVP.** `jax.pure_callback` defines no derivative rule. Measured, not assumed:
  `jax.jvp` and `jax.grad` through one both raise
  `ValueError: Pure callbacks do not support JVP. Please use jax.custom_jvp to use
  callbacks while taking gradients.` So a `VmconDriver` nested inside another
  differentiated block fails loudly and names the fix -- **not** a silent zero, which is
  what was expected before it was measured. It is still a loss: the refusal used to
  arrive at trace time and now arrives only when something asks for a derivative.
- **What writing it would take.** The derivative of a converged constrained solve is the
  implicit KKT system: differentiate the stationarity and active-constraint conditions at
  the optimum and solve the resulting linear system for `dx/dp`. That needs the Lagrange
  multipliers, and `pyvmcon.solve` already returns them (`lamda_equality`,
  `lamda_inequality`, currently discarded as `_lambda_eq`/`_lambda_ie`), plus a decision
  about which inequalities are active and what to do at a degenerate one. `scipy` returns
  the same for `SlsqpDriver`. So the road is open and nothing here nests one; the same
  note cottax's own SLSQP driver makes.
- **Exceptions cross the boundary.** `_refuse_non_finite`'s `ValueError` is raised inside
  `host`. Eagerly it propagates unchanged, which is what `run_cold_matrix`'s per-arm
  `except Exception` catches and what the `st_regression` MDF row's reason used to be.
  Under a trace it surfaces through jax's callback machinery instead, and no claim is
  made here about what type it arrives as -- that was not measured.
- **The per-iteration `callback` now runs inside a `pure_callback`.** It still works
  (`run_cold_matrix._recorder`'s trace is what the 108/objf/max|eq| numbers above come
  from, taken after the wrap). But jax is free to elide a `pure_callback` whose outputs
  are unused, so a trace is no longer evidence that a solve ran -- the reported ports
  are.
- **`Steps` is counted by the callback, and that is provably inert.** `pyvmcon.solve`
  substitutes its own `lambda _i, _result, _x, _con: None` when handed `None`
  (`vmcon.py`), so installing a counting callback calls one where the library would have
  called one anyway. `pyvmcon` returns no iteration count and this is the same number
  `len(trace)` gives the two ladder harnesses.
- **Status codes are per driver, and that is why `Status` is port-local.**
  `VMCON_CONVERGED = 0`; `VMCON_STATUS` gives `1` for a bare `VMCONConvergenceException`
  (the `max_iter` exhaustion), `2` for `QSPSolverException` (first QP infeasible), `3`
  for `LineSearchConvergenceException`. `SlsqpDriver` reports `scipy`'s own
  `OptimizeResult.status` instead. `Status` moved from `mdf` to `drivers` because three
  drivers now write one; `mdf` re-exports the name.
- **`SlsqpDriver.outcome`'s `message`, `nfev` and `fun` are gone** with `Outcome`. A
  message is a string and cannot survive a trace, which is why `Status` is an integer;
  nothing in this tree read `nfev` or `fun`, and either could become a further
  `DriverOut` kind the day something does.

### 22.6 What was not done

- No controlled re-run of the four moved MDF rows with this change alone reverted
  (§22.3).
- The inner jits are still rebuilt per solve (§22.2); hoisting `live` into an argument to
  make them cacheable is untried.
- `SlsqpDriver` was wrapped and its reports tested on the toy problem, but no SAND or MDF
  row was run through it after the change -- the controlled VMCON/SLSQP comparison §15
  and §20.4 rest on has not been re-taken.
- The `pure_callback` is `vmap_method='sequential'`; no batched solve was attempted.
- **`run_cold_matrix.cold_sand` still passes `whole=False`**, so the published matrix is
  measured on the walk path and does not take the whole-schedule jit the wrap now makes
  available. That flag was there to skip a probe known to fail; it is now skipping a
  probe known to succeed, and flipping it is a one-word change that would move every SAND
  row onto the fused path. Deliberately not done in the same change as the wrap: §19/§20
  are two full sections about a one-ulp fusion difference flipping this exact solve
  between 90-converged and 108-stopped, and the whole acceptance criterion here was that
  the rows do not move. The probe run in §22.2 says the `stellarator_helias` row does not
  move; the other four were not checked that way.

## 23. Neither the row's weight nor the lift: the fork survives both (2026-09-01)

**Where this was measured, and it is not the working tree.** Every number in this
section comes from a scratch rsync of the live tree (uncommitted changes included) at

    /tmp/claude-1000/-home-tbogaarts-PROCESS/df0c22b1-02c2-4e73-99ce-b061606f318d/
      scratchpad/PROCESS_sand

with `cottax` shared, read-only, from `~/jaxgraph/src`. `sand.py`/`sand_harness.py`
there gained three parameters (§23.9); nothing was applied to `/home/tbogaarts/PROCESS`,
and the diff is `<copy>/section23.patch` (it applies cleanly to the live tree; checked
with `patch --dry-run`, not applied). The runner is `<scratchpad>/exp23/exp23.py`, with
`struct23.py` (the Arm 2 shape probe), `jac23.py` (§23.6's control) and `summary.py`
beside it -- **outside** the copy, invoked with `PYTHONPATH=<copy>`, which is why the
import trap below does not reach them.

**The import trap, checked rather than assumed.** `$PY <tree>/functional_process/x.py`
puts the *script's* directory on `sys.path[0]`, where the `functional_process` package
is not found, so the import falls through to the editable install at
`/home/tbogaarts/PROCESS` -- and the usual verification, `$PY -c "import
functional_process"` run in the tree, **passes anyway** because `-c` puts the cwd on the
path. Two independent facts say this section did not fall into it. (a) The runner prints
`functional_process.__file__` as its own first line and asserts it names the copy, per
run. (b) Every arm calls `sand_harness.assemble(..., keep=...)`, a parameter the live
tree's `assemble` does not have -- a run that had resolved to the live tree would have
died on `TypeError`, not returned a number. Nothing here needed re-taking.

**The question.** §20-§22 leave `stellarator_helias` SAND forking on 1 ulp of
`.tfcoil.a_tf_turn_steel` between *90 converged* and *108 stopped*, with two candidate
mechanisms: the conditioning excess that §20.12 traced entirely to one row
(`^cond.stellarator.wp_width_r_min`, row norm `2892`, `cond(J)` `1.7e4` against MDF's
`42`), and the structural *lift* of that row -- `Intersect`'s `RootFind` is combined into
the SQP by SAND and left in the graph by MDF. **Measured, both are refuted as *the*
mechanism**, and the sharpest single number is §23.5's: the arm that removes the lift
gets the best conditioning of any arm and the worst answer of any arm.

### 23.1 Arm 0 reproduces, and the 2892 row is re-derived rather than inherited

Cold SAND, `--provider` boundary, `max_iter = 500`, `whole=False`, both `fuse_upstream`
policies. §22.3's published row and §20.3's eager row both come back to the last printed
digit:

| | its | status | `objf` | `max\|eq\|` | `min ie` |
|---|---|---|---|---|---|
| fused (the published row) | **108** | stopped | `1.2221740786343283` | `0.02851456050213086` | `1.1216805262392882e-11` |
| eager | **90** | converged | `1.2177573520529628` | `1.839436700669305e-06` | `-8.82673122093447e-10` |

`^driver_out.status^problem.sand` is `2` on the 108 row -- `QSPSolverException`, i.e.
`cvxpy` refusing a subproblem -- and `0` on the 90 row. The block is 124 nodes, 14
design, 8 equalities, 12 inequalities, 46 schedule steps.

`cond(J)` over the **20x14 constraint block** VMCON is handed (rows by
`condition_scale`, columns by `design_scale`), and every equality row's norm in the same
units, at the seeded start:

| | `cond(J)` |
|---|---|
| at the seeded start | **`2.527e+04`** |
| at the last iterate (108 branch) | `1.648e+04` |
| at the last iterate (90 branch) | `1.655e+04` |

| equality row | norm at start |
|---|---|
| **`^cond.stellarator.wp_width_r_min`** | **`3.029e+03`** |
| `^cond.physics.fusden_alpha_total` | `5.214e+00` |
| `^cond.physics.proton_rate_density` | `4.954e+00` |
| `^cond.constraints.c16` | `3.412e+00` |
| `^cond.constraints.c2` | `1.932e+00` |
| `^cond^cond.physics.temp_plasma_ion_vol_avg_kev` | `1.745e+00` |
| `^cond^cond.power.delta_eta` | `1.694e+00` |
| `^cond.fwbs.f_ster_div_single` | `1.650e+00` |

(inequality rows span `5.2e-02` to `8.0e+00`.) The drop-one sweep reproduces §20.12(c):
at the last iterate, dropping `^cond.stellarator.wp_width_r_min` takes `cond(J)`
`1.648e+04 -> 6.588e+01`; the next-best drop leaves it at `1.648e+04`. §20.12 measured
`2892` and `6.75e+01` at iteration 13 of a different branch; `3029`/`2811` and `65.9`
here. **The figure is now this section's own.**

The row's *bare* norm -- before `residual_condition_scales` multiplies it by `1.39475`
-- is **`2171.62`**, and its diagonal cell `d(^cond.wp)/d(.wp)` is `2160.78`, so the row
is essentially its own diagonal. That is what makes it a length: `1/|u|` is `1/0.6286`,
and equilibration wants `1/2172 = 4.605e-04`, a factor **3030** smaller. §20.12's
sentence -- *"`residual_condition_scales` normalises its value and leaves its derivative
at 2892"* -- is exactly right, and Arm 1 is the test of what fixing that buys.

### 23.2 Arm 1a: one row equilibrated -- converged, and bitwise fusion-invariant

`^cond.stellarator.wp_width_r_min`'s factor replaced by `1/2171.62 = 4.6049e-04`
(its row norm then measures `1.0000` exactly, as it must); every other factor left on the
`1/|u|` units rule.

| | `cond(J)` start | `cond(J)` last | its | status | `objf` | `max\|eq\|` in Arm 0's units |
|---|---|---|---|---|---|---|
| fused | `9.915e+01` | `5.709e+01` | **121** | converged | `1.2177573462463236` | `1.14e-06` |
| eager | `9.915e+01` | `5.709e+01` | **121** | converged | `1.2177573462463236` | `1.14e-06` |

`cond(J)` `2.5e4 -> 99`, and **the two fusion policies agree bit for bit**: same
iteration count, same `objf` to all 17 digits, `0 of 8` design variables moved,
`max rel |dx| = 0.000e+00`. That is MDF's property, on SAND, from one number.

Read on its own this is the answer the brief asked for. §23.4 is why it is not.

**A units note that matters for every row below.** `max|eq|` as the harnesses print it is
in `condition_scale`'s own units (§13.1), which is precisely what these arms change, so
the column is re-measured for every arm at its own final point on **`residual_condition_
scales`' ruler** -- the one every published row uses. That is the `max|eq|base` column.
Arm 0's fused row reads `2.851e-02` on both rulers; Arm 1a's `1.35e-09` own-units figure
is `1.14e-06` on Arm 0's.

### 23.3 Arm 1b: equilibrating *every* equality row is worse than equilibrating one

All eight equality rows divided by their own norm at the start point -- PROCESS's own
`c2` and `c16` included, which departs from `VmconDriver.condition_scale`'s standing rule
that PROCESS's constraints keep `1.0`, and is what "full row equilibration" means.

| | `cond(J)` start | its | status | `objf` | `max\|eq\|base` |
|---|---|---|---|---|---|
| fused | `9.550e+01` | 451 | converged | `1.217757347386511` | `8.97e-07` |
| eager | `9.550e+01` | **500** | **cap(500)** | `1.2219175164590899` | `5.70e-01` |

Better conditioning than Arm 1a (`95.5` against `99.2`), **four times** the iterations,
one arm of the pair failing outright, and a `6.6e-02` fork between the two policies. So
"more equilibration" is not "better", and `cond(J)` does not order these outcomes.

**Arm 1c**, a generalisable form -- equilibrate only the *outlier* equality rows, those
whose bare norm exceeds `10x` the median (on this file: `wp_width_r_min` at `2172`,
`proton_rate_density` at `5.5e+15`, `fusden_alpha_total` at `2.2e+18`) -- converges both
ways, 161/154 iterations, `objf` agreeing to `1.7e-07`, and is **not** bitwise
invariant. It is the rule that would apply to any configuration; it is not Arm 1a.

### 23.4 The factor sweep, which refutes §23.2's reading

§20.12 warns that this problem's iteration count swings 33-73 across a factor sweep of
this same row, so one lucky factor proves nothing. The sweep was therefore taken. Only
`^cond.stellarator.wp_width_r_min`'s factor varies; everything else is Arm 0.

| factor | `cond(J)` start | fused | eager | fusion-invariant? |
|---|---|---|---|---|
| `1.39475` (**Arm 0**, the `1/\|u\|` rule) | `2.53e+04` | 108 **stopped** `1.2221741` | 90 conv `1.2177574` | no, `4.4e-01` |
| `1e-1` | `1.81e+03` | 152 conv | 136 conv | no, `1.1e-06` |
| `1e-2` | `1.87e+02` | 151 conv | 272 conv | no, `3.8e-04` |
| `5e-3` | `1.05e+02` | 49 **stopped** `1.3237234` | 49 **stopped** `1.3237234` | **yes -- on a wrong answer** |
| `1e-3` | `8.65e+01` | 107 conv | 107 conv | **yes**, `0.0e+00` |
| `4.605e-4` (**Arm 1a**, `1/`row norm) | `9.92e+01` | 121 conv | 121 conv | **yes**, `0.0e+00` |
| `3e-4` | `1.13e+02` | 112 conv | 97 conv | no, `1.1e-04` |
| `1e-4` | `1.45e+02` | 96 conv | 228 conv | no, `2.7e-04` |
| `3e-5` | `1.54e+02` | **500 cap** `1.2327832` | 271 conv | no, `2.1e-02` |
| `1e-5` | `1.55e+02` | 103 conv | 103 conv | `5.7e-15` (same path, last-bit) |

Three things this kills, and the first is my own §23.2 reading:

1. **Bitwise fusion invariance is not a property of equilibration.** It appears at
   `1e-3` and at `4.6e-4`, is absent at `3e-4` and `1e-4`, and reappears at `1e-5`. It
   also appears at `5e-3` **on a solve that stopped at 49 iterations with `objf 1.3237`
   and `min ie -0.41`** -- a badly infeasible point both policies happen to reach
   identically. Invariance here means "the two policies took the same discrete path this
   time", not "the formulation is insensitive". MDF's invariance (§20.13) may still be
   structural; SAND's, at any factor, is not shown to be.
2. **`cond(J)` does not order the outcomes.** Every factor from `1e-1` down sits between
   `86` and `1812`, a 20x band, inside which the results run from *121 converged
   bitwise* to *49 stopped* to *500 cap*. The two failures (`5e-3`, `3e-5`) are not at
   the ends of the conditioning range; `5e-3`'s `105` is better-conditioned than
   `1e-4`'s `145`, which converged twice.
3. **What *is* robust is convergence, not the answer's stability.** 15 of the 18
   down-weighted runs converge, all to `objf 1.2177574 +- 4e-07`, against the baseline's
   1 of 2 and its two branches `0.36 %` apart in `objf` and `44 %` apart in `x`. So
   down-weighting this row converts a *categorical* fork into a *numerical* one, of
   order `1e-4` in `x`, most of the time. That is a real improvement and it is not a fix.

### 23.5 Arm 2: the lift removed -- the best conditioning of any arm, and the worst answer

`sand_graph(keep={^problem.stellarator.coils.intersect})` leaves that `RootFind` in the
graph, and `sand_schedule(nest=True)` answers the `Optimise` at the outer level with
`Blocking.scc(...).nest(...)` -- `mdf.in_graph_root_find`'s own step 4 on a different
outer problem. The block shape moves exactly as intended:

| | Arm 0 | Arm 2 |
|---|---|---|
| drive nodes | 124 | 125 |
| design / equalities / inequalities | 14 / 8 / 12 | **13 / 7 / 12** |
| `.stellarator.wp_width_r_min` | outer unknown | **inner unknown** |
| `^cond.stellarator.wp_width_r_min` | outer equality | **inner condition** |
| inner `Drive`s in the body | 0 | **1**, over a 123-step body |
| `cond(J)` at start / last | `2.53e+04` / `1.65e+04` | **`8.10e+01` / `5.30e+01`** |

`81` at the start and `53` at the last iterate is the closest any arm gets to MDF's `42`,
and the drop-one sweep now finds nothing to drop: the best single removal leaves
`80.3` of `81.0`. The conditioning pathology is **gone**, structurally, not by weighting.

The solve:

| | its | status | `objf` | `d objf` vs PROCESS | `max\|eq\|base` | `min ie` |
|---|---|---|---|---|---|---|
| fused | 37 | **stopped** (`QSPSolverException`) | `1.2395613934868683` | `+2.03e-02` | `2.04e-02` | `-2.04e-03` |
| eager | 37 | **stopped** | `1.2395614052556239` | `+2.03e-02` | `2.04e-02` | `-2.04e-03` |

Worse than every other arm on this table. It walks to `.physics.hfact = 1.19999999` --
pinned at its upper bound of `1.2` -- and `cvxpy` then refuses a subproblem at 37. It is
also **not** fusion-invariant, though its fork is `8.4e-07` in `x` rather than Arm 0's
`4.4e-01`, so the two policies are on the same trajectory rather than on different ones.

**`low_aspect_ratio_DEMO` gets no Arm 2**, on its own terms rather than by budget: the
brief conditions that run on Arm 2 helping, and it does not -- and that file's graph has
no `^problem.stellarator.coils.intersect` to keep, so the arm is undefined there.

### 23.6 The control that makes §23.5 mean something: Arm 2's derivative is Arm 0's, reduced

An in-graph root find could easily have been differentiating a different function, in
which case §23.5 would be a measurement of a bug. It is not. At the same seeded design
point, with `J0` Arm 0's `21x14` Jacobian and `J2` Arm 2's `20x13`, `r` the `wp`
residual row and `y` the `wp` column:

    J2  ==  J0[C, D] - J0[C, y] J0[r, D] / J0[r, y]

-- `sand_harness.reduce_jacobian`'s Schur complement with a `1x1` `J_RY`. Measured:

| | |
|---|---|
| condition **values** (20 shared rows) | **bit-identical**, max relative `0.000e+00` |
| condition **derivatives** (260 cells) | max relative **`2.242e-15`** |
| cells above `1e-8` relative | **0** |
| `J0[r, y]` | `2.160777e+03` |
| `wp` residual at the seed | `-8.5e-14` (the MDA has converged it) |

So the nested formulation evaluates the same conditions and differentiates the same
function, to the last bit and to round-off respectively. **Arm 2 fails with the correct
gradients.**

**Arm 2t** is a second control on the same arm: the inner `SeededNewtonDriver`'s
`rtol`/`atol` moved from `1e-4` to `1e-10` through `sand_schedule(inner_drivers=...)`
(verified to have taken -- the driver in the built schedule reports `1e-10`). The result
is **bit-identical to Arm 2** in every column. `mdf.py`'s standing warning that MDF
*"needs the inner solve to converge at every trial point"* is therefore not what limits
this arm; a scalar Newton on this block lands on the same float either way.

### 23.7 `low_aspect_ratio_DEMO`

The baseline reproduces §22.3's row (`cap(500)`, `objf -0.401520642`) to
`-0.4015206418715735`, and -- worth recording -- **that row is already bitwise
fusion-invariant**: `0 of 19` design variables move, identical `objf`. A row that caps
can be perfectly fusion-stable, which is one more reason not to read invariance as
health.

Arm 1c (outlier equalities, the only Arm-1 rule that is defined off this file's own
numbers) equilibrates two rows and is **inert**: `cond(J)` `871.5 -> 871.6`, still
`cap(500)`, `objf -0.4014727` against `-0.4015206`. Its one visible effect is on
feasibility, `min ie` `-3.32e-03 -> -3.73e-05`. This file has no `2892` row -- its
equality norms span `0.70` to `12.7` -- so there is nothing for the rule to catch, and
its cap is not a conditioning problem. The rule being inert where there is no outlier is
the behaviour wanted; it is not evidence for the rule.

### 23.8 The table

`stellarator_helias` unless marked. `d objf` is relative to PROCESS's `1.2149167845`
(`-0.40629623` on LAR); the `+2.34e-03` shared by every converged row is
`EXPLAINED_OBJECTIVE_READS`' `+17.604 MW` chain, not a disagreement.

| arm | `cond(J)` start | fuse | its | status | `objf` | `d objf` | `max\|eq\|base` | `min ie` | fusion-invariant |
|---|---|---|---|---|---|---|---|---|---|
| **0** baseline | `2.53e+04` | on | 108 | stopped | `1.2221740786` | `+5.97e-03` | `2.85e-02` | `1.12e-11` | **no**, `dx 4.4e-01` |
| | | off | 90 | converged | `1.2177573521` | `+2.34e-03` | `1.84e-06` | `-8.8e-10` | |
| **1a** wp row `1/`norm | `9.92e+01` | on | 121 | converged | `1.2177573462` | `+2.34e-03` | `1.14e-06` | `-6.8e-09` | **yes**, `dx 0.0e+00` |
| | | off | 121 | converged | `1.2177573462` | `+2.34e-03` | `1.14e-06` | `-6.8e-09` | |
| **1b** all eq rows | `9.55e+01` | on | 451 | converged | `1.2177573474` | `+2.34e-03` | `8.97e-07` | `-1.0e-08` | **no**, `dx 6.6e-02` |
| | | off | 500 | cap(500) | `1.2219175165` | `+5.76e-03` | `5.70e-01` | `-7.2e-06` | |
| **1c** outlier eq rows | `8.98e+01` | on | 161 | converged | `1.2177575531` | `+2.34e-03` | `1.78e-06` | `-2.0e-08` | **no**, `dx 5.9e-05` |
| | | off | 154 | converged | `1.2177577654` | `+2.34e-03` | `5.27e-05` | `1.3e-11` | |
| **2** `Intersect` in graph | `8.10e+01` | on | 37 | stopped | `1.2395613935` | `+2.03e-02` | `2.04e-02` | `-2.0e-03` | **no**, `dx 8.4e-07` |
| | | off | 37 | stopped | `1.2395614053` | `+2.03e-02` | `2.04e-02` | `-2.0e-03` | |
| **2t** + inner tol `1e-10` | `8.10e+01` | on/off | 37 | stopped | *bit-identical to Arm 2* | | | | |
| LAR **0** baseline | `8.72e+02` | on | 500 | cap(500) | `-0.4015206419` | `+1.18e-02` | `1.37e-07` | `-3.3e-03` | **yes**, `dx 0.0e+00` |
| | | off | 500 | cap(500) | `-0.4015206419` | `+1.18e-02` | `1.37e-07` | `-3.3e-03` | |
| LAR **1c** outlier rows | `8.72e+02` | on | 500 | cap(500) | `-0.4014726541` | `+1.19e-02` | `1.37e-07` | `-3.7e-05` | **yes**, `dx 0.0e+00` |
| | | off | 500 | cap(500) | `-0.4014726541` | `+1.19e-02` | `1.37e-07` | `-3.7e-05` | |

### 23.9 The verdict, and what it is not

**Conditioning is not the mechanism, and the structural lift is not the mechanism.**
Stated as the two experiments that would have shown otherwise and did not:

- If the fork were *the row's weight in the QP*, the factor sweep would be monotone in
  the factor and every sufficiently-equilibrated point would behave like Arm 1a. It is
  not and they do not (§23.4): `5e-3` stops at a bad point, `3e-5` caps on one policy,
  and `1e-4` -- a factor **between** two bitwise-invariant ones -- forks.
- If the fork were *the lift*, removing it would reproduce MDF's behaviour. It removes
  MDF's conditioning (`81`, from `2.5e4`, against MDF's `42`) and reproduces none of
  MDF's robustness: 37 iterations, stopped, `2.03 %` from PROCESS, still not
  fusion-invariant (§23.5) -- and with **provably correct derivatives** (§23.6) and a
  **provably irrelevant inner tolerance** (§23.6, Arm 2t).

What the arms *do* establish, and it is worth having:

1. **`residual_condition_scales`' factor for this one row is a bad one, and that is now
   measured rather than argued.** `1.39475` leaves the row's derivative at `2172`; at
   any factor from `1e-1` down, 15 of 18 runs converge to one optimum against the
   baseline's 1 of 2 with its branches `44 %` apart in `x`. The units rule is right about
   *units* and this row's units are not its unknown's -- `VmconDriver.condition_scale`'s
   own docstring already says so -- so a row-norm term for exactly the rows whose units
   are not their unknown's is a defensible change. It buys convergence, not stability.
2. **The in-graph move is a design preference here, not a necessity, and on this
   configuration it is currently a regression.** It is structurally clean, it builds with
   no new mechanism, its derivative is exactly right, and its answer is the worst on the
   table. Nothing here argues against MDF-style nesting in general; it argues that
   *this* block's lift is not what breaks *this* solve.
3. **The fragility is in the trajectory, not in the formulation.** Every lever tried on
   this configuration moves it, and none of them removes the sensitivity: 1 ulp of
   `a_tf_turn_steel` (§20.3), the schedule's fusion policy (§19), one row's weight
   anywhere in a decade-wide band (§23.4), removing that row from the problem entirely
   (§23.5) -- and, independently and after this section's runs, a module-level compiled-
   call cache in another agent's tree, which takes the same solve from *108 stopped* to
   *257 converged* at `objf 1.21775737` and MDF from 67 to 108 iterations. That patch is
   **not** in the copy measured here, so Arm 0 is still the published baseline; it is
   reported because it is a sixth independent lever moving the same fork, which is
   evidence about the configuration and not about any of these arms. A solve whose
   outcome six unrelated levers can flip is not being held back by one Jacobian row.

**What is still not established** -- and this is the same sentence §20.12 and §21.5 end
on, now with two more explanations removed rather than one: what actually amplifies a
`4e-16` input into a different outcome. It is not the `c24` kink (§20.12b), not the
condition number (§20.12c, §23.4), not a frozen or dropped value (§20.12d), not the QP
solver (§21.1), not this row's weight (§23.4), not this row's presence in the problem
(§23.5), not the inner solve's tolerance (§23.6). The one thing every failing branch
still has in common is `QSPSolverException` on a QP whose *inputs* are the ones that
differ -- and §21.1 has already shown `cvxpy` to be deterministic given those inputs.
The next thing with any evidence behind it is the merit function and the line search
between the QP and the next iterate, which no section has instrumented.

### 23.10 What landed in the copy, and what was not done

Three parameters, all defaulting to today's behaviour, all with the arm they exist to
run named in their docstrings:

- **`sand.sand_graph(keep=())`** -- declared problems neither residualised nor combined.
  A knob rather than a hard-coded exception, so the question is re-runnable on any file.
- **`sand.sand_schedule(nest=False, inner_drivers=None)`** -- `Blocking.scc(...).nest(
  <the Optimise>)` instead of the flat blocking, and per-problem driver overrides for a
  kept problem's algorithm. `sand_schedule` and `sand.constraints_outside_block` now
  find the `Optimise` **by type off `graph.definitions`** rather than through
  `blocking.problems`, which raises on the two-problem block `keep` deliberately creates.
- **`sand_harness.assemble(keep=())`** -- forwarded; a kept problem is never dropped as
  degenerate or array-valued.

`tests/functional_process/test_sand.py` + `test_mdf.py`: **193 passed** in the copy.
`ruff check`/`format` clean on the two edited files (the pre-existing `I001`/`F401` on
`sand.py`'s import block is untouched and predates this).

Not done:

- No test pins the new parameters. The arms are a scratch runner outside the copy, not a
  case; `keep`/`nest` reaching `main` should arrive with one.
- Arm 2 was run at one nesting only (the `Optimise` nested, `Intersect` inner). The
  converse -- `Intersect` outer, as `mdf.in_graph_root_find` does for its own root find
  -- was not tried and is a different formulation, not a variant of this one.
- The `1c` outlier rule's threshold (`10x` the median) was chosen once and not swept. On
  `stellarator_helias` it catches three rows and on `low_aspect_ratio_DEMO` two; whether
  it is the right rule is not settled by two files.
- No arm was run on the other five configurations, and no SLSQP arm was run at all
  (§22.6's gap is still open).

## 24. The inner jit leaves the driver, and one configuration of seven moves (2026-09-01)

**Measured in a scratch copy of the working tree, not in `~/PROCESS`:**
`/tmp/claude-1000/-home-tbogaarts-PROCESS/df0c22b1-02c2-4e73-99ce-b061606f318d/scratchpad/PROCESS_mdf`.
Two other agents were running against `~/PROCESS` and `~/jaxgraph` at the time, so every
number below comes from a tree only this work could touch. `cottax` was shared read-only
from `~/jaxgraph/src`. The copy's `functional_process/` and `process/` were verified
identical to the live tree at the start except for the two files this section changes
(`core/solver/drivers.py`, and the new `core/solver/host_cache.py`) and four
visualization artefacts not on any solve path.

§22.6's second bullet -- *"the inner jits are still rebuilt per solve; hoisting `live`
into an argument to make them cacheable is untried"* -- is now tried, and the section
splits into what it bought (§24.1), what it cost (§24.2), and a measurement trap that
invalidated the first two attempts to find out (§24.3).

### 24.1 The jit moves to `core/solver/host_cache.py`, and both cache facts are measured

`VmconDriver` and `SlsqpDriver` no longer call `jax.jit` at all. Two module-level
`eqx.filter_jit` functions -- `flat_conditions` and `flat_condition_jacobian` -- take the
`ConditionMap` as an **argument** and are shared by both drivers.

**Why a new module and not `ConditionMap`, and not the driver.** The only reason any of
this exists is that `pyvmcon` and `scipy.optimize` iterate in Python, on the host, so
there is no enclosing jit for the per-iteration model call to be hoisted into. That is a
forced inelegance and it belongs next to the drivers that force it. Putting the
compilation on `ConditionMap` would put a notion of "compiled" into the graph's own
vocabulary that only a host-loop driver needs -- a jax-native constrained optimiser would
call `jax.jacfwd(conditions)` inside its own trace and meet no compilation boundary at
all. Putting it inside the driver is what was wrong before: `jax.jit` caches on the
identity of the function it wraps, and a closure built inside `host` is a fresh object
every solve.

**Two functions, not one.** `jax.jacfwd(f)` returns an ordinary Python function that
traces `f` under JVP; it is *not* compiled, and calling one eagerly runs the block op by
op. So the `jacfwd` goes inside a second jitted function rather than being wrapped around
the first.

**The cache-key facts, both measured rather than assumed.** They are a *correctness*
gate, not only a performance one: a module-level cache that collided would hand one
problem another problem's compiled program, silently.

| direction | probe | result |
|---|---|---|
| same map, two `eqx.combine`s -> same key | `hashable_partition(cmap, eqx.is_array)[1]` | hash equal, `==` True, distinct objects |
| same map, context **values** changed -> same key | tree-map `x * 1.01` over the array half | hash equal (**correct**: the values are traced arguments, so one program serves both -- and the second call returns a *different answer* from the same program, which is the proof that they are not baked in) |
| `spherical_tokamak_eval` vs `st_regression` -> different key | two `MdfConditionMap`s | hash **unequal**, `==` False |
| same file, optimise vs root find -> different key | two problem shapes of one file | hash **unequal**, `==` False |
| `unravel` as a static argument | `jax.flatten_util.ravel_pytree` | returns `jax._src.util.HashablePartial`; two `unravel`s of the same pytree structure hash equal and compare equal |

End to end on `spherical_tokamak_eval`'s MDF condition map: first call 1 compile for
values and 1 for the Jacobian; a recombined map and a third combine each cost **0**.

**A note on `Static`, because the obvious probe fails.** `eqx.Module.__hash__` on the
raw static half raises `TypeError: unhashable type: 'list'`. That is not a hole in the
scheme -- `eqx.filter_jit` does not use it. It goes through
`equinox._compile_utils.hashable_partition`, which is what the table above measures.
Reported because the first probe written for this looked like a refutation and was not.

**The invariant that must not be undone, and it is preserved.** `eqx.combine` stays
**inside** the `jax.pure_callback`, in `_sqp_callback.wrapped`, at runtime, on concrete
arrays. `_sqp_callback` is unchanged by this section. Nothing dynamic is captured in a
closure built at trace time -- which is exactly why the first design considered here
(hoisting the jitted callables into `VmconDriver.__call__`) was abandoned: `__call__`
runs at *trace* time, so a closure over `dyn` built there and executed at callback time
would capture a tracer outside its trace and raise `UnexpectedTracerError` the first time
a solve was nested. The module-level design gets this for free, because `cmap` is an
argument and never a capture. What `__call__` still reads off `conditions` at trace time
is `conditions.conditions` and `conditions.unknowns` -- tuples of `VarPath`, i.e. names,
never arrays -- for `condition_scale` and the bounds, exactly as before.
`vmap_method="sequential"` on the `pure_callback` is untouched.

**Compiles, `stellarator_helias` SAND, one process each, the schedule run twice.**
Same instrument as §22.2 (patching `jax._src.compiler.backend_compile_and_load`).

| | first call | second call |
|---|---|---|
| before, probe on (`whole=None`) | 4 compiles | 2 compiles |
| **after, probe on** | 4 compiles | **0 compiles** |
| before, `whole=False` (the cold matrix's path) | 11 compiles | 3 compiles |
| **after, `whole=False`** | 11 compiles | **1 compile** |

The target was 2 -> 0 and it is met. The walk path gains the same 2, which the
`__call__`-hoist design would **not** have delivered: there the driver is invoked eagerly
once per `run_schedule` call, so anything built in `__call__` is rebuilt too. A
module-level function has one identity for the life of the process, so the cache survives
both.

Wall times are deliberately not tabulated here: the change moves this solve from 108
iterations to 257 (§24.2), so before-and-after seconds are not measuring the same work.
Per-iteration cost is §24.4.

### 24.2 The bitwise gate: six configurations of seven reproduce, `stellarator_helias` does not

Full `run_cold_matrix.py --provider` pass before and after, both correctly resolved
(§24.3), diffed row by row.

**Ten of twelve rows are byte-for-byte identical**, `st_regression SAND`'s `FAILED` and
`low_aspect_ratio_DEMO SAND`'s `cap(500)` included. The two that move are both
`stellarator_helias`:

| row | before | after |
|---|---|---|
| `stellarator_helias` MDF | 67 it, converged, `objf 1.21775735`, `max\|eq\| 4.60e-11`, `min ie -2.59e-10` | 108 it, converged, `objf 1.21775747`, `max\|eq\| 4.20e-11`, `min ie -1.95e-10` |
| `stellarator_helias` SAND | 108 it, **stopped**, `objf 1.22217408`, `max\|eq\| 2.85e-02`, `d objf 5.97e-03`, `worst dx 5.98e-01` | 257 it, **converged**, `objf 1.21775737`, `max\|eq\| 3.57e-05`, `d objf 2.34e-03`, `worst dx 1.08e-01` |

**The gate is failed, and it is failed in the good direction -- which does not make it
passed.** The SAND row moves from a stopped solve that was two and a half percent off in
its equalities to a converged one that agrees with the MDF row's objective to seven
digits and halves the gap to PROCESS. Nothing was widened to get that; it is what the
solver did. But the acceptance criterion was *bitwise*, and this is not bitwise, so the
change is reported as landed-and-moving rather than landed-clean, and whether to keep it
is not this measurement's call.

**Constant folding is the demonstrated cause, not the suspected one.** Two independent
lines of evidence, both taken at `stellarator_helias`'s cold SAND start, comparing
`jax.jit(lambda flat_x: stack(cmap(*unravel(flat_x))))` (A, arrays closed over) against
`host_cache.flat_conditions(cmap, flat_x, unravel)` (B, arrays as arguments) on **the same
`cmap` object**:

1. **The optimised HLO.** A folds; B cannot.

   | | parameters | constants | lines |
   |---|---|---|---|
   | values, A closed-over | 736 | 1279 | 10627 |
   | values, B argument | 2105 | 1109 | 14075 |
   | jacobian, A closed-over | 2943 | 4270 | 34109 |
   | jacobian, B argument | 6089 | 3046 | 38451 |

2. **A three-way value comparison against an eager call.** 17 of 21 conditions and 100 of
   294 Jacobian cells differ between A and B. Almost all of it is 1-2 ulp. Of the 17
   moved conditions, **B agrees with the eager call on 6 and A on 2** -- i.e. the
   argument form stays closer to unfused arithmetic, which is what a folded constant
   subgraph would predict. `plain jax.jit` reproduces `eqx.filter_jit`'s B bitwise, so
   equinox contributes nothing to the difference.

**Why 1-2 ulp becomes a different solve is already in this file.** Two of the moved
conditions are SAND coupling residuals that are ratios of near-zero quantities
(`^cond.physics.proton_rate_density` reads exactly `-0.25` under A and `+0.25` under B;
`^cond.stellarator.wp_width_r_min` flips sign at `2.27e-13`), so a last-bit move in the
denominator is an order-one move in the residual. §19 and §20 are two full sections about
this exact solve flipping between 90-converged and 108-stopped on a `-2` ulp change to one
variable, and §20's conclusion was that the eager answer had been on the tolerant side of
one bit by accident. This is the third landing of the same coin.

**What that implies for any version of this fix.** The hazard is not specific to the
design chosen: the whole mechanism by which the cache becomes reusable is that the solve's
arrays stop being trace-time constants, so XLA loses the freedom to fold them. Any
formulation that makes the compiled program reusable across solves gives up the folding.
"Keep the 2 compiles per solve" and "keep this row bitwise" are the same choice.

### 24.3 A measurement trap: `python functional_process/run_cold_matrix.py` does not measure this tree

The first two matrix passes taken for §24.2 were void, and the reason is worth recording
because nothing announced it and both looked exactly like valid runs.

Invoking `$PY functional_process/run_cold_matrix.py` from the copy's root puts
**`<copy>/functional_process`** on `sys.path[0]` -- the *script's* directory, not the
current one, and `''` is not added for a script invocation. `<copy>/functional_process`
contains no `functional_process` package, so `from functional_process import mdf` falls
through to the editable install and resolves to **`/home/tbogaarts/PROCESS`**. Measured
directly: `drivers.__file__` came back as `/home/tbogaarts/PROCESS/functional_process/
core/solver/drivers.py` and `hasattr(drivers, "flat_conditions")` was `False`.

`CLAUDE.md`'s check (`$PY -c "import functional_process; print(...)"`) passes in that same
directory, because `python -c` *does* put `''` first. So the check and the run disagree,
and the run is the one that matters. The fix is `PYTHONPATH=<copy>`, and the tell that
something was wrong was a contradiction rather than an error: a standalone script and the
matrix reported different answers for the same configuration on the same tree.

**What was salvaged.** The pre-change baseline pass was itself run against the live tree,
and the live tree's `drivers.py` was verified byte-identical to the copy's pre-change
version at the time (and every other file on the solve path identical), so it stands as a
valid "before". The "after" pass was re-taken with `PYTHONPATH` set. It ran out of memory
part way (15 GB box, two other agents, `LLVM ERROR: Unable to allocate section memory`),
so the last three configurations were re-run one process at a time; all three reproduce
the baseline bitwise.

### 24.4 Where a `stellarator_helias` SAND iteration's time actually goes

Warm second call, `whole=False`, 257 SQP iterations, 7.62 s wall of which
`pyvmcon.solve` is 7.497 s. Timed with `perf_counter` around each section and
`jax.block_until_ready` inside the jitted wrappers so device time is not misattributed;
CLARABEL's own time is `cvxpy`'s `solver_stats.solve_time`.

| section | total s | calls | ms/iteration | % of solve |
|---|---|---|---|---|
| 1 jitted values (`flat_conditions`) | 2.390 | 513 | 9.30 | 31.9 |
| 2 jitted Jacobian (`flat_condition_jacobian`, 14 columns) | 2.453 | 513 | 9.54 | 32.7 |
| 3 `cvxpy` problem construction | 0.182 | 257 | 0.71 | 2.4 |
| 4 `cvxpy` canonicalisation/compilation | 2.159 | 257 | 8.40 | 28.8 |
| 5 CLARABEL's own solve | 0.069 | 257 | 0.27 | 0.9 |
| 6 `pyvmcon` line search and the rest | 0.244 | | 0.95 | 3.3 |

**Neither half is the whole answer, and that is the useful result.** The model calls are
64 % and the QP is 31 %, so no amount of jit hygiene alone gets near the ~0.5 ms per
Jacobian target -- and neither does fixing `cvxpy` alone.

**The QP's time is canonicalisation, not solving, by a factor of 31.** `pyvmcon.solve_qsp`
(`vmcon.py:312-329`) builds a fresh `cp.Variable`, `cp.Minimize` and `cp.Problem` on
**every** call; there is no `cp.Parameter` anywhere and therefore no DPP-parametrised
problem to re-solve. So `cvxpy` re-canonicalises and re-compiles a 14-variable QP 257
times, at 8.40 ms each, to hand CLARABEL 0.27 ms of actual work. That is the textbook
failure mode and it is confirmed both by reading the source and by the split above. Fixing
it is a `pyvmcon` change (a parametrised problem built once) or a native QP, not anything
this port can do from outside.

**Forward-mode AD is nearly free here, and that was not expected.** Per *call*, values
cost 4.66 ms and the 14-column Jacobian 4.78 ms -- a 2.5 % premium for fourteen tangents.
So the 9.5 ms/iteration attributed to the Jacobian is not fourteen forward passes' worth
of arithmetic; it is one dispatch's worth. At this size the model calls are dominated by
per-call overhead rather than FLOPs, which is the same conclusion §18 reached about
compiles and is why the remaining lever is *fewer, larger* host round trips rather than a
cheaper Jacobian.

(513 calls to 257 iterations is 2 per iteration: VMCON evaluates at the trial point and
again in the line search, and `_Problem.__call__` computes values and Jacobian together.)

### 24.5 `low_aspect_ratio_DEMO` MDF: the stop is a refused QP, and the answer is feasible

The row reads `stopped` at 10 iterations with `max|eq| 5.93e-12` and `min ie -1.41e-06`,
and `MAX_ITER` is 800, so the cap is not what ended it. Reading the driver's own reported
`Status` says what did:

    Steps=10  Converged=False  Status=2      # 2 = QSPSolverException

VMCON's 11th quadratic subproblem could not be solved -- `cvxpy` raised, `pyvmcon` turned
it into `QSPSolverException`, and `VmconDriver` kept the last point. It is not a line
search failure (`Status=3`) and not exhaustion (`Status=1`).

**The point it kept is feasible.** Evaluating every condition at the returned answer:

- all four equalities at `|.| <= 2.2e-14`;
- **all 25 inequalities satisfied**, the tightest being `^cond.constraints.c5` at
  `+7.74e-12` in VMCON's sign;
- the last convergence value is `1.41e-07`, against a `1e-8` tolerance -- one order of
  magnitude short of the KKT test, not two.

The `-1.41e-06` on the table is the *trace's* last callback value, recorded at the
iterate VMCON was standing on when the QP was refused; the answer `VmconDriver` returns
and `mdf.solve` re-runs the MDA at is a step further on and is clean. Both numbers are
correct; they are measured at different points.

**So the KKT-versus-feasibility hypothesis does not explain this row.** The proposed
mechanism -- a violation with a near-zero multiplier being invisible to
`|∇f·δ| + |Σλ_eq·eq| + |Σλ_ie·ie|` -- requires a violation, and there is none. The row is
a feasible, nearly-KKT point that the QP solver declined to improve on. PROCESS takes 16
iterations on the same file and gets `objf -0.40629623` against the port's
`-0.40631157`, a `3.78e-05` gap, with `worst dx 1.15e-04`.

**And PROCESS's own answer is the less feasible of the two, in the port's arithmetic.**
Evaluating the port's conditions at PROCESS's converged `x` leaves two inequalities
violated -- `^cond.constraints.c31` at `-4.35e-05` and `^cond.constraints.c90` at
`-2.14e-06` in VMCON's sign -- where the port's own answer violates none. That is the
scale against which "essentially feasible" should be read on this file.

### 24.6 `st_regression` MDF converges in 4 because its objective is a constant

The `custom_jvp` on `_solv` (§21.3) does fix the row: measured here at **4 iterations,
`Status=0`, `Converged=True`, convergence value `4.95e-09`** against a `1e-8` tolerance.
That reproduces, and it is a real fix to a real `nan`.

**It is also not a solved optimisation, and the table already said so.** The row's
`d objf 1.00e+00` and `worst dx 9.90e+01 at ixc 152` are not near-misses; the objective
printed `-0` at *every* iteration of the trace, from iteration 0. The reason is
structural:

- `st_regression.IN.DAT` states `i_figure_merit = -5`, i.e. **maximise `FUSION_GAIN_Q`**,
  and `sand.objective_node` mints a node reading exactly one path,
  `.current_drive.big_q_plasma`;
- that path is **not owned by the tokamak graph**. Measured: `VarPath(.current_drive.
  big_q_plasma) in set(graph.owned)` is `False` for `st_regression` and for
  `large_tokamak_nof`, and `True` only for the stellarator graph, whose
  `models/stellarator/heating.py` is the port's sole producer of it;
- so it is a boundary input, seeded from the cold `DataStructure` at `0.0`, which the
  provider does not answer. The objective is therefore `-1 * 0.0 = -0.0` everywhere, with
  an identically zero gradient.

**What VMCON actually solved is a feasibility problem**, and it found a feasible-ish point
in 4 iterations. Nothing pins the design vector along the objective's (empty) descent
direction, which is precisely why `ixc 152` can sit 99x away from PROCESS's value without
the solve noticing. PROCESS's own `big_q_plasma` is `16.5886` and its objective
`-16.5885765`.

**And this row *is* the KKT hypothesis' best case.** With `∇f ≡ 0` the convergence test
`|∇f·δ| + |Σλ_eq·eq| + |Σλ_ie·ie|` loses its first term exactly, so it is testing the
multiplier products alone -- and a violated inequality carrying a small multiplier is
invisible to it. That is consistent with converging at `4.95e-09` while
`^cond.constraints.c16` is violated by `1.51e-03`.

**Is `-1.51e-03` a bad answer?** On that constraint alone, no -- and the comparison is
worth stating because it is the opposite of the expected one. `c16` is the *same*
constraint whose Jacobian row was `nan` before §21.3. Evaluating the port's conditions at
**PROCESS's own converged `x`** puts `c16` at `-1.06e-01` in VMCON's sign: PROCESS's
answer violates it by seventy times more than the port's does. So `min ie -1.51e-03` is
inside what PROCESS itself lands on for this constraint, and it is not the thing wrong
with this row. The constant objective is.

The `st_regression` **SAND** row still fails at assembly with
`KeyError: VarPath(^cond.numerics.objf)`, unchanged and not investigated here.

### 24.7 Converging and agreeing are different axes, and here is each separately

Read off the (identical) before and after passes; `stellarator_helias` is quoted after
§24.2's move, with the before value in brackets.

**Did the solve converge?**

| configuration | MDF | SAND |
|---|---|---|
| `stellarator_helias` | converged, 108 it [67] | converged, 257 it [**stopped**, 108] |
| `helias_5b` | converged, 4 it | converged, 7 it |
| `large_tokamak_nof` | converged, 7 it | converged, 10 it |
| `large_tokamak_eval` | converged, 3 it (root find) | n/a -- the file states a root find |
| `low_aspect_ratio_DEMO` | **stopped**, 10 it (`Status=2`, §24.5) | **cap(500)** |
| `spherical_tokamak_eval` | converged, 2 it (root find) | n/a |
| `st_regression` | converged, 4 it -- of a constant objective (§24.6) | **FAILED** to assemble |

**Does it agree with PROCESS?** A different question, and the ranking is different.

| configuration | `d objf` | `worst dx` | verdict |
|---|---|---|---|
| `large_tokamak_eval` MDF | -- (no objective) | `3.29e-12` | **agrees** -- PROCESS's own `fsolve` answer, to round-off |
| `spherical_tokamak_eval` MDF | -- | `3.64e-09` | **agrees** |
| `large_tokamak_nof` MDF | `1.16e-11` | `2.01e-02` | objective agrees to 11 digits; design differs 2 % at ixc 57 |
| `large_tokamak_nof` SAND | `4.78e-12` | `6.37e-01` | objective agrees; design differs **64 %** at ixc 135 |
| `low_aspect_ratio_DEMO` MDF | `3.78e-05` | `1.15e-04` | **agrees** to four digits, though the solve stopped |
| `helias_5b` MDF / SAND | `9.13e-04` | `2.01e-03` / `6.18e-03` | agrees to three digits |
| `stellarator_helias` MDF | `2.34e-03` | `1.08e-01` | partial; see below |
| `stellarator_helias` SAND | `2.34e-03` [`5.97e-03`] | `1.08e-01` [`5.98e-01`] | after §24.2, identical to the MDF row's disagreement |
| `low_aspect_ratio_DEMO` SAND | `1.18e-02` | `7.09e-02` | **does not agree**, and did not converge |
| `st_regression` MDF | `1.00e+00` | `9.90e+01` | **does not agree** -- constant objective (§24.6) |

Two rows make the point that the axes are independent in *both* directions:
`low_aspect_ratio_DEMO` MDF **stopped** and agrees to four digits; `st_regression` MDF
**converged** and does not agree at all.

`stellarator_helias`'s `2.34e-03` is characterised, not chased, as briefed: MDF and SAND
now disagree with PROCESS by the *same* amount at the *same* design distance
(`1.08e-01` at ixc 109), which is new information -- before this change the two
formulations disagreed with PROCESS differently, and it was open whether that was the
formulations or the solver trajectories. It was the trajectory. The remaining gap is a
single number shared by both formulations and is the one the matrix's own note points at
`EXPLAINED_DISAGREEMENTS['.heat_transport.p_plant_electric_base_total_mw']` for;
separating "the port's physics differs" from "the two objectives were evaluated at two
different points" still needs the port's objective at PROCESS's own `x`, which is a Stage
A measurement and was not taken here.

### 24.8 The MDF compile census, and the 5 are not what was expected

Every configuration, MDF only, two calls each, in one process per configuration. Compiles
counted by patching `jax._src.compiler.backend_compile_and_load`, and each compile's MLIR
`sym_name` and line count recorded, so a count can be resolved into *which* programs.
Taken after §24.1; the two rows §24.1 changes are `solve`'s, and they are called out.

| configuration | `prime` 1st | `prime` 2nd | solve 1st | solve 2nd | verdict |
|---|---|---|---|---|---|
| `spherical_tokamak_eval` (root find) | 1, 8.67 s | 0, 0.02 s | **5, 17.11 s** | **0, 0.02 s** | `(True, None)` |
| `large_tokamak_eval` (root find) | 1, 10.81 s | 0, 0.02 s | 5, 19.24 s | 0, 0.04 s | `(True, None)` |
| `st_regression` | 1, 8.29 s | 0, 0.01 s | 854, 57.44 s | 9, 2.58 s | `(True, None)` |
| `helias_5b` | 7, 3.51 s | 0, 0.01 s | 260, 20.14 s | 7, 1.88 s | `(True, None)` |
| `low_aspect_ratio_DEMO` | 1, 12.52 s | 0, 0.02 s | 971, 84.37 s | 14, 4.62 s | `(True, None)` |
| `large_tokamak_nof` | 1, 9.95 s | 0, 0.02 s | 970, 84.65 s | 14, 4.32 s | `(True, None)` |
| `stellarator_helias` | 7, 3.49 s | 0, 0.01 s | 268, 27.74 s | 7, 7.96 s | `(True, None)` |

**Every schedule jits whole, on all seven, both `prime` and the two in-graph root finds:
`schedule_verdict` is `(True, None)` everywhere and there is no refusal reason to quote.**
`mdf.prime` reproduces the reference point exactly (1 compile on the tokamaks;
`spherical_tokamak_eval` 1 compile / 0 on the second call), and the stellarators' 7 is the
same program plus six 5-line scalar casts.

**What the in-graph root find's 5 compiles are.** The brief's hypothesis -- XLA emitting
the `lax.while_loop` body and condition as separate modules, plus `optimistix` staging --
is **refuted**. Named, on `spherical_tokamak_eval`:

    jit_run                     33935 MLIR lines     <- the entire schedule, one program
    jit_stage                       5 MLIR lines
    jit_convert_element_type        5 MLIR lines
    jit_convert_element_type        5 MLIR lines
    jit_convert_element_type        5 MLIR lines

`large_tokamak_eval` is the same five with `jit_run` at 37 506 lines. So it is **one real
program and four trivial scalar ones**: the `while_loop` is *inside* `jit_run`, exactly as
an accepted whole-schedule jit implies, and the other four are eager `jnp` ops at the
boundary -- three dtype canonicalisations and jax's own internal `stage_p` primitive
(`jax/_src/core.py:1298`), not `optimistix`'s. Reported as a correction: the "5 = loop
body + condition + staging" story was plausible, is the sort of thing that gets repeated,
and is wrong.

**`mdf.solve` costs ~950 XLA compiles that nothing needs, and they are not the driver's.**
The 971 on `low_aspect_ratio_DEMO` resolve as:

    jit_flat_condition_jacobian  50102 lines   x1     <- the driver's Jacobian (§24.1)
    jit_flat_conditions          19086 lines   x1     <- the driver's values   (§24.1)
    jit_multiply  x136, jit_add x86, jit_subtract x70, jit_true_divide x64,
    jit_dynamic_slice x49, jit_integer_pow x48, jit_broadcast_in_dim x48,
    jit__where x42, jit__reduce_sum x27, jit_scatter x24, ... , a few jit_while

-- i.e. **two** programs for the whole solve, and ~969 one-primitive programs. That
per-primitive pattern is §18.6's signature of an *eager* `Schedule.__call__`, and there is
exactly one in this path: `mdf.solve`'s last act is

    out = mdf.eager(_inputs_only(mdf, at))

-- a direct schedule call, **not** through `sand_harness.run_schedule`. `mdf.prime` a few
lines away does go through `run_schedule` and costs 1 compile for the same schedule. So
this is a one-line asymmetry costing roughly 950 compiles and, at §18's ~25 ms each, most
of the ~60 s gap between the two `solve` columns. It is not the SQP: the SQP is the two
big programs, and after §24.1 it is two on the first call and **zero** thereafter.

The stellarators' 260/268 rather than ~970 is the smaller graph (154 nodes against 243),
same mechanism.

**What the second call still costs (7-14 compiles) is the same eager re-run** meeting
shapes its first pass did not, not the driver: the driver's two programs are cache hits by
then, which is what §24.1 bought.

### 24.9 What was not done, and what is now open

- **The bitwise gate is failed** (§24.2). Whether to keep §24.1 given that
  `stellarator_helias` moves is not decided here. No tolerance was widened and no row was
  re-baselined.
- **`mdf.solve`'s eager final MDA re-run (§24.8) was measured, not fixed.** Routing it
  through `run_schedule` is a one-line change with an obvious prediction (~950 compiles ->
  ~1, as `prime` already shows) and it was left alone because it would move a second thing
  in the same pass as §24.1, and this file has three sections about what that costs.
- **`SlsqpDriver` shares the two helpers** (`scaled_problem` builds no `jax.jit` of its
  own any more) but, as in §22.6, no SAND or MDF row was run through it, so the shared
  path is exercised only by the unit tests.
- **`pyvmcon`'s per-iteration `cvxpy` rebuild (§24.4)** is diagnosed and untouched; it is
  upstream code.
- **`st_regression`'s objective (§24.6) is unported, not broken.** Porting a producer for
  `.current_drive.big_q_plasma` on the tokamak graph would turn that row from a feasibility
  solve into the optimisation the file states. Until then the row's `d objf` and `worst dx`
  columns are not measuring what they appear to measure, and the table should probably say
  so in its own notes rather than only here.
- **`st_regression` SAND** still fails at assembly with
  `KeyError: VarPath(^cond.numerics.objf)`; not investigated.
- **`low_aspect_ratio_DEMO`'s refused QP (§24.5)** was identified (`Status=2`) but not
  diagnosed: what makes the 11th subproblem infeasible for `cvxpy`/CLARABEL, and whether a
  different `qsp_solver` takes the step, is untried. PROCESS's own 16 iterations on the
  same file say a step exists.
- Scoped tests after §24.1: `tests/functional_process/test_mdf.py test_sand.py
  test_cold_matrix.py` -- **223 passed**, unchanged.

---

## 26. The missing-producer census, all seven configurations (2026-09-01)

> **Measured in a scratch copy, not in the live tree**, at
> `/tmp/claude-1000/-home-tbogaarts-PROCESS/df0c22b1-02c2-4e73-99ce-b061606f318d/scratchpad/PROCESS_producers`,
> against `HEAD 6bb65494` plus that copy's own changes to `boundary.py`,
> `core/solver/drivers.py` and their two test files. Source changes are supplied as
> `.patch` files in the copy and were **not** applied to `~/PROCESS`. Every number
> below was taken with `PYTHONPATH` set to the copy and with
> `functional_process.__file__` printed as the first line of each run, per §27.5 of
> `next_steps.md`.

`next_steps.md` §27.4 asks the question this section answers: *how much of the
instability currently blamed on the solver is actually a missing producer?* The answer,
across the seven in-scope configurations, is **four paths, on three files, and none of
them on `stellarator_helias`**.

### 26.1 The measurement, and why the existing pins could not have made it

Two independent discriminators were run over all seven files and they agree row for row.

**Discriminator A -- structural, no PROCESS at all.** A path this configuration's graph
*reads* and does not *own*, which some **other** configuration's graph does own. That is
`big_q_plasma`'s shape exactly. It costs seven graph assemblies and answers **24 to 31
rows per configuration** -- far too many to be a verdict, and useful only as a ranking
(`boundary.owned_elsewhere`). `.physics.aspect` is on every tokamak's list because a
stellarator's graph computes it from the config file, and it is a perfectly genuine
tokamak input.

**Discriminator B -- the value side.** `provider.answers_for`'s `computed` reason:
PROCESS's own pipeline writes this path from cold, so the port freezes a seed where
PROCESS has a live value. Independently cross-checked against a third quantity -- the
frozen seed differenced against PROCESS's **converged** `DataStructure`
(`sand_harness.reference_run`) -- and the two agree exactly, `0/0/0/0/1/3/4` per file in
`CONFIGURATIONS` order.

**Neither had ever been run on the graph that has the conditions in it, and that is the
whole finding.** `reference_boundary*.txt`, `missing_producers_tokamak.txt`,
`reference_provider_*.txt` and `boundary.unproduced_but_computed` are all measured on
`driven_graph(graph_for(machine_from_indat(...)))` -- the **models**. The objective node
and the constraint nodes are inserted later, by `mdf.mdf_graph`/`sand.optimise_graph`. A
path read *only* by a condition is therefore invisible to every existing pin, and
`.current_drive.big_q_plasma` is exactly that: nothing among the 241 model nodes of
`st_regression` reads it, so it never entered that file's boundary at all.
`reference_provider_st_regression.txt` reports **one** `computed` row; the same
measurement over `mdf_graph`'s graph reports **four**.

### 26.2 [measured] The census -- 4 paths, 8 rows, 3 files

Frozen seed against PROCESS's own converged value, over the problem graph, iteration
variables excluded (the optimiser owns those on purpose):

| configuration | path | seed | PROCESS converged | read by |
|---|---|---|---|---|
| `stellarator_helias` | -- none -- | | | |
| `helias_5b` | -- none -- | | | |
| `large_tokamak_nof` | -- none -- | | | |
| `large_tokamak_eval` | -- none -- | | | |
| `low_aspect_ratio_DEMO` | `.cs_fatigue.n_cycle_min` | `20000.0` | `29382.181` | `.Constraint90` |
| `spherical_tokamak_eval` | `.build.r_cp_top` | `0.0` | `1.2089` | `.tokamak.cicc_superconducting_tf_coil.tf_coil_shape` |
| | `.constraints.pflux_fw_rad_max_mw` | `0.0` | `0.49896` | `.Constraint67` |
| | `.physics.p_plasma_separatrix_rmajor_mw` | `0.0` | `40.2816` | `.Constraint56` |
| `st_regression` | `.current_drive.big_q_plasma` | `0.0` | `16.58858` | `.Objective` |
| | `.build.r_cp_top` | `0.0` | `1.3405` | `.tokamak.cicc_superconducting_tf_coil.tf_coil_shape` |
| | `.constraints.pflux_fw_rad_max_mw` | `0.0` | `0.36324` | `.Constraint67` |
| | `.physics.p_plasma_separatrix_rmajor_mw` | `0.0` | `40.0000` | `.Constraint56` |

**One of the eight is a false positive, and it was found by reading the code the
discriminator flagged.** `.cs_fatigue.n_cycle_min` on `low_aspect_ratio_DEMO` is a
**dead read**. PROCESS's `constraint_equation_90` has a *side effect* --
`if data.costs.ibkt_life == 1 and data.cs_fatigue.bkt_life_csf == 1:
data.cs_fatigue.n_cycle_min = data.costs.bktcycles` (`constraints.py:1895`) -- and that
file sets both switches (`:626`, `:634`). The port's `constraint_90` already reproduces
the override *in the value it compares*, and `ibkt_life`/`bkt_life_csf` are bound
statically, so the node's body is `partial(constraint_90, ibkt_life=1, bkt_life_csf=1)`
and the `n_cycle_min` argument is discarded. `.costs.bktcycles` **is** owned on that
configuration. The frozen `20000.0` is never used. The wart is that the node still
declares the read, which puts a path on the boundary that nothing consumes.

That leaves **seven live rows on two files**, and both files are spherical tokamaks.

### 26.3 [measured] Blast radius, ranked

| rank | path | configuration(s) | objective? | constraint? | cone | why it ranks here |
|---|---|---|---|---|---|---|
| 1 | `.current_drive.big_q_plasma` | `st_regression` | **yes, it *is* the objective** | -- | 1 | `objective_metric_5` is the identity on it. The objective and its whole gradient row are identically zero, so VMCON's convergence test loses its first term exactly and the run solves a **feasibility problem** and reports `converged` |
| 2 | `.physics.p_plasma_separatrix_rmajor_mw` | `st_regression` | no | c56 (`leq`, driven) | 1 | frozen `0.0` against a bound of `40`, so c56 reads as satisfied with margin and has a zero row. **PROCESS's own converged answer is `39.99999999988` -- c56 is *active*, sitting exactly on its bound.** The port is solving a strictly relaxed problem whose single most binding constraint is absent |
| 3 | `.physics.p_plasma_separatrix_rmajor_mw` | `spherical_tokamak_eval` | no | c56 (reported, not driven) | 1 | same freeze; PROCESS reads `40.2816` against the same bound of `40`, i.e. **violated at PROCESS's own answer**, where the port prints it comfortably satisfied. Evaluation mode, so it does not change the solve -- it changes what the row says |
| 4 | `.build.r_cp_top` | both STs | no | 1 inequality | 43 nodes | the centrepost top radius, frozen at `0.0` where PROCESS has `1.21`/`1.34` m. Read by `.tokamak.cicc_superconducting_tf_coil.tf_coil_shape`, so it is *live* in the model chain rather than inert -- a wrong value propagating, not a dead one. Already on `reference_provider_*.txt` as this port's only pre-existing `computed` row |
| 5 | `.constraints.pflux_fw_rad_max_mw` | both STs | no | c67 (`leq`) | 1 | frozen `0.0` against a bound of `1.2`; PROCESS reads `0.36`/`0.50`, i.e. **not** binding at its own answer. A zero gradient row and an inert constraint, but not one that changes where the optimum is |
| -- | `.cs_fatigue.n_cycle_min` | `low_aspect_ratio_DEMO` | no | c90, discarded | 1 | see §26.2 -- a declared read the body throws away. Cosmetic |

### 26.4 [measured] `stellarator_helias` SAND: **no.** Nothing in that problem is frozen

`next_steps.md` §27.4's headline question, asked of the configuration under most
suspicion (§21, §23, §24 -- six independent levers flip its outcome), and the answer is
a clean negative:

- **0** boundary inputs of that configuration's problem graph are classified `computed`.
- **0** of its 294 boundary inputs differ from PROCESS's converged `DataStructure`.
- **0** of its 15 driven conditions -- the objective, 2 equalities, 12 inequalities -- is
  unreachable from its 8 design variables. Every row of the Jacobian is live.

The 27 `owned-elsewhere` rows it does have are all either declared `IN.DAT`
inputs/defaults (`.tfcoil.dx_tf_turn_general`, `.physics.kappa`) or the `unwritten`
shape §22 already documents -- a path PROCESS's own stellarator pipeline never writes
either, so PROCESS computes with the same untouched default the port does
(`.physics.plasma_current`, `.physics.p_plasma_ohmic_mw`, `.pf_power.ensxpfm`,
`.build.r_tf_inboard_mid`, ... all at `0.0` on both sides). `.physics.dlamie`'s case
from §22 generalises to fifteen more paths.

**So the `stellarator_helias` fork is not a missing producer, and §27.6's remaining
candidate -- instrument the merit function and the line search -- keeps the whole of the
evidence.** This is the fourth structural explanation ruled out for that configuration
and the first one ruled out *cheaply*: it cost no solve.

### 26.5 The guard, implemented -- and it is structural, not numeric

Two checks, deliberately a matched pair. Both follow `drivers._refuse_non_finite`'s
convention: refuse loudly, name the offending rows, and stay a *measurement* --
`run_cold_matrix.run_one` catches any raise per phase and records it as a row with a
reason.

**1. `boundary.inert_conditions` / `refuse_inert_conditions` -- before anything runs.**
A condition of the stated problem that is not in `Graph.reach(design)` has an
identically zero Jacobian row by construction: no design variable reaches it, so its
value is a constant over the whole design space. It is a walk over declared reads and
owns -- **no PROCESS run, no seed, no solve, no Jacobian**; the whole seven-file census
is seven graph assemblies:

```
$ $PY -m functional_process.boundary --inert
stellarator_helias.IN.DAT          8 design,  15 driven condition(s) -> 0 inert
helias_5b.IN.DAT                   3 design,   6 driven condition(s) -> 1 inert
    .Constraint11    1/2 operand(s) frozen, 25 in cone: .physics.rmajor
large_tokamak_nof.IN.DAT          20 design,  27 driven condition(s) -> 0 inert
large_tokamak_eval.IN.DAT          2 design,   2 driven condition(s) -> 0 inert  (+8/23 reported-only)
low_aspect_ratio_DEMO.IN.DAT      19 design,  26 driven condition(s) -> 0 inert
spherical_tokamak_eval.IN.DAT      3 design,   3 driven condition(s) -> 0 inert  (+2/15 reported-only)
    reported-only .Constraint56    2/2 operand(s) frozen, 2 in cone: ...
    reported-only .Constraint67    2/2 operand(s) frozen, 2 in cone: ...
st_regression.IN.DAT              14 design,  19 driven condition(s) -> 3 inert
    .Objective       1/1 operand(s) frozen, 1 in cone: .current_drive.big_q_plasma
    .Constraint56    2/2 operand(s) frozen, 2 in cone: .constraints.p_plasma_separatrix_rmajor_max_mw, .physics.p_plasma_separatrix_rmajor_mw
    .Constraint67    2/2 operand(s) frozen, 2 in cone: .constraints.pflux_fw_rad_max, .constraints.pflux_fw_rad_max_mw
```

**`n of n operands frozen` is the discriminator, and it separates the two kinds
cleanly.** Every one of the four defects reads `1/1` or `2/2` -- a condition comparing
two constants. Every benign row reads `0/2`, `1/2` or `1/3`.

An **evaluation-mode file's inequalities are `reported`, not `driven`**, and are
excluded from the refusal: PROCESS root-finds the equalities alone on
`i_process_run_mode = -2` and never examines the inequalities, so eight of
`large_tokamak_eval`'s 23 are inert *by design*. They are still printed, because
`spherical_tokamak_eval`'s `.Constraint56` is one of them and it is a real defect.

**2. `drivers._refuse_inert_objective` -- at the first Jacobian the SQP forms.** Row 0
is the objective (`_Problem.__call__`'s own `f=values[0]`) and the columns are the
design variables; if the whole row is zero the driver refuses. Checked **once**, at the
first evaluation, not per iteration -- a zero objective gradient at an interior iterate
is a legitimate thing for a well-posed problem to reach; one at the starting point is a
statement about the problem. Other zero rows are *named and not refused on*, because
`helias_5b`'s c11 is one and that file converges.

Run end to end, the `st_regression` MDF row goes from `converged`/`objf -0` to:

```
st_regression MDF: ValueError: the objective ^cond.numerics.objf has an identically zero
gradient with respect to all 14 design variable(s), so this is not an optimisation: the
SQP will solve the feasibility problem that remains and report it as converged.
  design variables: ['.physics.temp_plasma_electron_vol_avg_kev', ... ]
  other conditions with an all-zero row: ['^cond.constraints.c56', '^cond.constraints.c67']
```

-- one `FAILED` row with the cause in the notes, and the numeric check independently
rediscovering c56 and c67, which the static one had already named without running
anything.

### 26.6 Two predictions this section made and measurement refuted

- **"An inert condition caused by the file's own problem statement will have an empty
  frozen-read cone."** False, and it would have made the check useless. Every chain
  ends at the boundary: `helias_5b`'s `.Constraint11` has **25** perfectly ordinary
  inputs in its cone (`.build.dr_blkt_inboard`, `.tfcoil.tftmp`, ...) and looked exactly
  like a defect. The field was rewritten to hold the node's **own** operands, where the
  ratio `len(frozen)/operands` does separate the two, and `Inert.frozen`'s docstring
  records the refutation rather than the first design.
- **"`.cs_fatigue.n_cycle_min` on `low_aspect_ratio_DEMO` is a missing producer."** Both
  discriminators said so and both were right about the *declaration*; the read is dead
  (§26.2). Whatever is wrong with that configuration's SAND cap at 500 is not this.

And one correction to a claim already in the tree: `models/physics/exhaust.py`'s module
docstring says `calculate_psep_over_r_metric` is left unported because *"no active
constraint and no ported node reads `.physics.p_plasma_separatrix_rmajor_mw`, so an
occupant for it would be a producer with no consumer."* That was true when it was
written and is **not true now** -- constraint 56 is active on both spherical tokamaks
and reads exactly that path. It is the same shape as the `.physics.p_div_bt_q_aspect_
rmajor_mw`/constraint 68 finding §11.5 records two bullets above it, and it is
"the same three lines and one `physics.py` line" by that docstring's own account.

### 26.7 Not resolved

- **Whether porting the four producers fixes anything.** Out of scope by instruction --
  this section finds and ranks them; it does not fix them. What is *known* is that
  `st_regression` cannot be measured at all until `big_q_plasma` has a producer, because
  the file's objective is the only thing that makes it an optimisation.
- **`spherical_tokamak_eval` MDF's `min ie -1.38e+01`** is not explained by anything
  here: all three of its frozen paths make constraints read as *satisfied*, so none of
  them can produce a violation of that size. Unchanged as an open item.
- **`st_regression` SAND's `KeyError: VarPath(^cond.numerics.objf)`** shares the root
  (`sand` has no objective condition to prune to when the metric is a bare boundary
  read) but was not investigated further; the guard does not reach it, because assembly
  fails before any driver is built.
- **The `unwritten` bucket** -- 65 to 68 rows on every configuration, paths PROCESS's
  own pipeline never writes either -- is faithful by construction *on the machines
  measured*, but §22's `.physics.dlamie` case shows the same path can be `unwritten` on
  one machine and `computed` on another. Nothing checks that a path is `unwritten`
  for the same *reason* PROCESS leaves it alone.
- **Only `mdf.mdf_graph`'s assembly is checked.** SAND's `sand.optimise_graph` inserts
  the same conditions, so the census transfers; the guard is not wired into the SAND
  path, and `run_sand_harness.py` does not call `refuse_inert_conditions` either.
- Scoped tests: `tests/functional_process/test_boundary.py` (**39 passed**),
  `tests/functional_process/core/solver/test_drivers.py` (**15 passed**), plus the two
  `run_cold_matrix` rows quoted above.

### 26.8 The `from the seed` paths, enumerated -- and none of them is downstream of PROCESS's solve

Every configuration's `--provider` boundary line ends *"N from the seed"* (18/18/21/22/21/20/19
in `CONFIGURATIONS` order). Those are the paths `provider.answers_for` classifies but
cannot *answer* -- `a.independent` is false -- so they keep `reference.cold`'s value.
They are the exact set the port still needs a live PROCESS for. **Reproduced to the row**:
the same counts fall out of a fresh `provide(driven_graph(graph_for(...)), ...)` over the
model graph, and the problem graph adds 0--3 more (the conditions' own reads, §26.1).

#### 26.8.1 [measured] Every one is an init-time constant. The concern is dependence, not contamination

For each of the 30 distinct rows, three values were compared: `cold_state.seed` (after
`init_process`, before any model), `cold_state.process` (after one `Evaluators.fcnvmc1`
pass) and `reference_run(...).data` (PROCESS's **converged** answer).

**Every non-`computed` row is bit-identical across all three.** No model pass moves them
and no solve moves them. So the coordinator's caution is confirmed by measurement rather
than by argument: these are a *starting state*, `install` runs with `disagreeing=False`
so every value it does write is bit-identical to the one it replaces, and nothing here is
the port being handed a piece of PROCESS's answer.

**The exception is exactly the `computed` reason, and it is §26's list.** Five rows move
(`.build.r_cp_top`, `.constraints.pflux_fw_rad_max_mw`,
`.physics.p_plasma_separatrix_rmajor_mw`, `.current_drive.big_q_plasma`,
`.cs_fatigue.n_cycle_min`) -- the missing producers, already ranked in §26.3, and the
only from-the-seed rows that are downstream of anything.

#### 26.8.2 [measured] The enumeration, by class

| rows | path(s) | reason | on | class |
|---|---|---|---|---|
| 12 | `.impurity_radiation.f_nd_impurity_electron_array[2..13]` | `derived` | 7/7 (11/7 where element 12 is `ixc` 135) | **1** -- `init.py:382-385` copies the declared input array `f_nd_impurity_electrons` element-wise |
| 4 | `.impurity_radiation.{temp_impurity_keV_array, pden_impurity_lz_nd_temp_array, impurity_arr_zav, m_impurity_amu_array}` | `derived` | 7/7 | **3** -- atomic-physics **data tables**, read by `init_imp_element` from 28 shipped `.dat` files in `process/data/lz_non_corona_14_elements/`. Neither an `IN.DAT` input nor a computation |
| 1 | `.divertor.n_divertors` | `derived` | 5/7 | **1** -- `init.py:607-617`, `2` for a double null and `1` otherwise, off `.physics.i_single_null` |
| 4 | `.pf_coil.{zref, rref, c_pf_coil_turn_peak_input, j_pf_coil_wp_peak}` | `input` | 5/2/3/3 of 7 | **2** -- declared `InputVariable`s the file *does* name, whose text is an **array**; `provider._scalar` returns `None` for it and `answer` falls back to the seed |
| 1--2 | `.tfcoil.dcond[0/2/4/8]` | `default` | 4/3/1/2 of 7 | **2** -- a declared input the file does not name, addressed per element; `answer` never marks an element independent |
| 1--4 | the five `computed` paths | `computed` | see §26.2 | **missing producer** |

#### 26.8.3 [measured] The gap is `provider.py`'s, not the port's -- `native.py` already derives almost all of it

Asked from the other side, statically (`native_state(input_file).values` against each
problem graph's boundary, no solve), the answer is far smaller than the 18--22:

| configuration | boundary places | native cannot answer |
|---|---|---|
| `stellarator_helias` | 308 | **5** |
| `helias_5b` | 302 | **5** |
| `large_tokamak_nof` | 412 | **5** |
| `large_tokamak_eval` | 410 | **5** |
| `low_aspect_ratio_DEMO` | 411 | **5** |
| `spherical_tokamak_eval` | 389 | **5** |
| `st_regression` | 391 | **6** |

`native.DERIVATIONS` already carries `_initialise_imprad` (which fills the four tables
from the port's own **vendored** `impurity_tables()`/`M_IMPURITY_AMU_ARRAY`, so the
`.dat` files are a build-time dependency and not a runtime one), `_alias_impurity_fractions`
(the twelve element rows) and `_single_or_double_null` (`n_divertors`), and
`DATACLASS_DEFAULTS` + `read_indat` cover the `pf_coil` arrays and `dcond`. **So class 1
and class 3 are already derived somewhere in this tree; what is missing is a rule in
`provider.py`, whose answer ladder has no `derived` arm and falls through to the seed.**
That is a smaller and much more specific work item than "18 paths need deriving".

#### 26.8.4 [measured] The `none: 5` column is closed, and it is benign

The five that appear on *every* configuration are
`.vacuum.{l1, l2, l3, ceff_i, xmult_i}`, and the two instruments agree exactly: they are
the `nothing` rows `provider.install` counts (`None` in both columns) **and** the whole
of the static native gap. They are **not `DataStructure` fields at all** --
`VacuumData` has no such attributes -- so `provider._get` returns `None` because the
attribute is absent. They are *minted* names for locals of PROCESS's
`_solve_vacuum_pumping_old` per-species loop, read by one node,
`.vacuum.duct_diameter_root_find`, which `models/vacuum/namespace.py:22` registers as a
**deliberate island**: no producer edge, no consumer edge, undriven, cone of 2, reaching
no condition on any configuration. Nothing evaluates them in a solve.

One correction to a docstring in the tree: `provider.install` says *"`.vacuum.l1` and
four siblings **default to `None`**"*. They have no default because they have no field;
the behaviour is the same and the reason is not.

`st_regression`'s sixth is `.current_drive.big_q_plasma` -- the native side finding the
same missing producer from the other direction, with no graph reachability argument
needed.

#### 26.8.5 What this does not settle

- **`provider.py` gaining a `derived` arm is untried.** Whether wiring `native.DERIVATIONS`
  into the provider's ladder leaves the matrix bitwise is a one-line experiment and a
  measurement, and it was not run here.
- **`--native` was not run.** The comparison above is static, per the instruction; a
  native *solve* pass would say whether answering these paths independently changes an
  answer, and §22.7's result (13 `off` paths cost four of seven configurations their
  solve) says not to assume it does not.
- **The four impurity tables are vendored, not validated here.** That
  `impurity_tables()` reproduces `init_imp_element`'s reads of the 28 `.dat` files is
  `native.py`'s claim and this section relies on it; it was not re-measured.
- **One pre-existing test failure, met in passing and not caused by anything here.**
  `tests/functional_process/test_provider.py::test_each_configuration_s_answer_is_its_pin[st_regression]`
  fails at `HEAD 6bb65494`. Confirmed not to be this section's: the same test fails in
  the same copy with `boundary.py` and `drivers.py` restored from `~/PROCESS`
  (1 failed, 6 passed, 3 s), and neither change touches `provider.provide`'s path. The
  diff is a **new guess port** -- `^guess.tfcoil.dr_tf_plasma_case` where the pin has
  `^guess.times.t_plant_pulse_burn` -- plus one extra `unwritten defaults
  .vacuum.xmult_i` row. By `boundary.py`'s own rule a `guess` moves only when a `Drive`
  does, so `reference_provider_st_regression.txt` is stale against a structural change,
  and the fix is `$PY -m functional_process.provider --write`. Not done here, because
  regenerating a published pin belongs to whoever made the structure move.
---

## 28. The carried values become arguments, and the one-line fix does not work (2026-09-01)

> **Measured in a scratch copy, not in the live tree**, at
> `/tmp/claude-1000/-home-tbogaarts-PROCESS/df0c22b1-02c2-4e73-99ce-b061606f318d/scratchpad/PROCESS_arrays`.
> The copy's baseline is **`b14da8c1`** ("Refuse a solve whose objective cannot move,
> and find the four that could not") -- verified file by file: every path under
> `functional_process/` and `process/` is byte-identical to that commit. `~/PROCESS`
> moved on to `3107bf49` while this section was being measured, which is why the
> baseline is named by commit and not by "HEAD". Two other agents were running against
> `~/PROCESS` and `~/jaxgraph` throughout, so nothing here was applied to either; the
> source changes are supplied as `.patch` files in the copy. `cottax` was shared
> read-only from `~/jaxgraph/src`. Every run below set `PYTHONPATH` to the copy and
> printed `functional_process.__file__` as its first line (§24.3's trap).

§25 censused the compiled XLA modules and found nine values on `stellarator_helias` and
twelve on `large_tokamak_nof` baked in as compile-time constants, every one of them a
*node output* rather than a boundary input, and named one line per node as the cause:
`models/initialisation.py`'s input-less `ExplicitFunction`s return Python floats. This
section fixes that. It splits into a census that found more of them than §25 could see
(§28.1), a **refutation of the proposed repair** (§28.2), what actually works and what it
costs (§28.3), the compile-time constants before and after (§28.4), the Scan question --
which the change does **not** answer, and the three separate causes that is now known to
have (§28.5) -- the bitwise gate (§28.6), the design question the owner asked (§28.7), and
what is left open (§28.8).

### 28.1 Fourteen nodes, not nine values — the census taken from the graph

§25 counted *constants in a compiled module*, which undercounts twice: a module holds one
`f64[] constant(0)` however many nodes supply a zero, and a module only contains the nodes
that block actually reaches. Counted from the graph instead — every `CallableNode` with no
inputs, called, its output leaves type-checked — the seven tracked configurations hold
**fourteen** such declarations between them, **twenty-six output paths**:

| declaration | outputs | on |
|---|---|---|
| `.initialisation.tf_cryoplant_efficiency` | `.tfcoil.eff_tf_cryo` | all 7 |
| `.initialisation.tf_insulation_youngs_modulus` | `.tfcoil.eyoung_ins` | 5 tokamaks |
| `.initialisation.tf_conductor_youngs_modulus` | `.tfcoil.eyoung_cond_axial`, `…_trans` | 5 tokamaks |
| `.initialisation.pf_coil_resistivity` | `.pf_coil.rho_pf_coil` | 5 tokamaks |
| `.initialisation.beam_electron_density_fraction` | `.physics.f_nd_beam_electron` | 5 tokamaks |
| `.initialisation.energy_storage_building_volume` | `.buildings.esbldgm3` | 4 |
| `.initialisation.stellarator_solenoid_absent` | `.build.dr_cs`, `.build.dr_cs_tf_gap` | 2 stellarators |
| `.initialisation.stellarator_pulse_times` | the four `.times.t_plant_pulse_*` | 2 stellarators |
| `.costs.energy_storage_cost` | `.costs.c2253` | 4 |
| `.tokamak.diamagnetic_current` | `.current_drive.f_c_plasma_diamagnetic` | 3 |
| `.tokamak.pfirsch_schluter_current` | `.current_drive.f_c_plasma_pfirsch_schluter` | 3 |
| `.tokamak.current_drive.secondary_heating` | `.current_drive.eta_cd_hcd_secondary`, `…p_hcd_secondary_extra_heat_mw`, `.heat_transport.p_hcd_secondary_electric_mw` | 5 tokamaks |
| `.tokamak.ccfe_hcpb.centrepost_neutronics` | `.fwbs.pnuc_cp_tf`, `…p_cp_shield_nuclear_heat_mw`, `…pnuc_cp`, `…neut_flux_cp` | 3 |
| `.tokamak.cicc_superconducting_tf_coil.croco_turn_cable_space_extra_void` | `.tfcoil.f_a_tf_turn_cable_space_extra_void` | 2 |

Five nodes / nine paths on `stellarator_helias` — **exactly §25's nine** — and nine nodes
/ fifteen paths on `large_tokamak_nof` against §25's twelve, the three extra being
`CentrepostNeutronicsAbsent` outputs that block never reaches. So §25's count is right for
what it measured and this one is the superset.

`DoubleNullUpperBuild` is deliberately **not** in the table: it reads
`.build.dz_shld_lower`/`.build.dz_vv_lower` from the graph, so its outputs are already
traced values and there is nothing to fix. It is the only member of `initialisation.py`'s
family that computes rather than carries, and §28.7 turns on that.

An AST scan over `functional_process/` for a node body returning a Python literal (or a
`self.<field>`) finds twelve classes and no thirteenth, which agrees with the dynamic
census once the two nodes whose bodies call a ported *function* returning literals
(`croco_turn_cable_space_extra_void`, `calculate_centrepost_neutronics_absent`) are added.
Nodes *with* inputs were checked too and none returns a constant limb.

### 28.2 `return jnp.asarray(...)` buys nothing, and this was measured before changing anything

The repair named in §25 -- one line per node, `return jnp.asarray(...)` so the value
becomes an `eqx.is_array` leaf -- **does not work**, and the reason is two facts that
compose.

**Fact 1: an array built inside the traced body is a jaxpr constant, exactly as the
Python float was.** Minimal probe, `eqx.filter_jit` over `(x * v + 1.0) * 3.0`, optimised
HLO:

| where the value lives | parameters | constants |
|---|---|---|
| `v` a Python-float field of the node | 2 | 4 |
| **`v` an array field of the node** | **4** | **3** |
| `0.13` a literal in the body | 2 | 4 |
| **`jnp.asarray(0.13)` in the body** | **2** | **4** |

The last row is the proposed fix and it is bit-for-bit the row above it.

**Fact 2: the node's fields are not reachable from the compiled function's arguments at
all.** `cottax`'s `ExplicitFunction.node_definition` builds
`CallableNode(fn=self.__call__)`, and a plain bound method is not a pytree.
`jax.tree_util.tree_leaves` of a real graph node definition whose declaration holds a
`jnp.asarray(0.13)` returns **zero** array leaves. So even moving the conversion to the
field would not have helped on its own: `eqx.filter_jit` never sees it, the value is
consumed at trace time, and it lands in the module as a constant and in the cache key as
configuration.

Both were measured before a line of the port was edited. **Flagging it as the prediction
that failed**: the task's mechanism ("`eqx.filter_jit` traces `eqx.is_array` leaves and
bakes everything else") is right about `filter_jit` and wrong about what `filter_jit` can
see here.

### 28.3 What works: the declaration reaches the trace through `fn`

`functional_process/models/carried.py` is the new home. Three pieces:

- **`carried(default=...)`** — `eqx.field(kw_only=True, converter=jnp.asarray, …)`. The
  conversion happens when the declaration is *built*, at assembly. `jnp.asarray` of a
  Python float keeps `weak_type=True`, so promotion is unchanged.
- **`carried_all(…)`** — the same for the one node whose ported function hands back four
  values together.
- **`CarriesValues`** — an `ExplicitFunction` subclass overriding exactly one query:
  `node_definition` builds `CallableNode(fn=jtu.Partial(_apply, self))`. `jax.tree_util.Partial`
  registers its `args` as pytree children and its `func` as static aux, so the
  declaration's **array** leaves become parameters of the compiled program and its
  non-array leaves stay in the cache key, which is where they belong.

Measured on the minimal probe with a module-level `_apply`: 4 parameters / 3 constants,
`hash()` works, `==` works, and two declarations differing only in the carried value
**share one compiled program** (second call, 0 compiles).

**`fn=self` was tried and rejected.** Using the declaration directly as the callable is a
pytree with no wrapper and has the same effect on the trace, but `Graph.__hash__` hashes
`tuple(self.definitions.items())` and an `eqx.Module` holding a `jax.Array` is
unhashable — which would take out `mdf._hashable`, `sand_harness._SCHEDULE_WHOLE` and
`_SCHEDULE_RUNNERS` at once. A `Partial` hashes by identity and keeps all of them.
`hash(GRAPH)` was re-checked after the change and still answers.

**What it costs, stated rather than hidden.** `CallableNode.__check_init__` binds
`len(inputs)` positional arguments against `inspect.signature(fn)`; a `Partial` reports
`(*args)`, so the arity check is inert for these fourteen nodes. Every one of them
declares no inputs, so there is no arity to get wrong — but it is one more reason this
belongs in `cottax`, where the signature is still in hand. The general statement is *a
node's own array fields are data the graph should trace, not configuration it should
specialise on*, and that is a property of `CallableNode`, not of these fourteen classes.

**One collateral repair.** `mda_harness._declaration_modules` walks
`node.fn.__self__` to find the declaration behind a node definition. A `Partial` has no
`__self__`; without a `functools.partial` limb the walk stops at the `fn` leaf and every
`CarriesValues` declaration goes **unaudited in silence** — a missing switch registration
would read as "no switches to check" rather than as a mismatch. The limb is added and
`test_switch_coverage.py` passes.

### 28.4 The compile-time constants: measured on `stellarator_helias`'s SAND Jacobian

§25's instrument, re-taken. The `(cmap, flat_x, unravel)` the cold SAND solve hands
`host_cache.flat_condition_jacobian` on its **first** call are captured by a spy that then
aborts the solve; the two `host_cache` functions are lowered and compiled from those
arguments and their optimised HLO counted. The pre-change numbers reproduce §25 exactly,
which is the check that the instrument is the same one.

| | parameters | constants | constant bytes | lines |
|---|---|---|---|---|
| values, before | 2105 | 1109 | 9269 | 14084 |
| **values, after** | **2116** | **1108** | **9261** | **14104** |
| jacobian, before | 6089 | 3046 | 24636 | 38626 |
| **jacobian, after** | **6105** | **3045** | **24628** | **38657** |

**The constant *count* barely moves and that is not a failure -- it is what a shared
constant pool looks like.** Seven of the nine values on this configuration are `0.0`, and
XLA holds one `f64[] constant(0)` however many places want a zero, so removing seven zeros
removes no constants at all. The count that answers the question is which *literals* are
in the module:

| literal | before, values | after, values | before, jacobian | after, jacobian |
|---|---|---|---|---|
| `constant(0.13)` (`.tfcoil.eff_tf_cryo`) | 1 | **0** | 1 | **0** |
| `constant(31557600)` (`.times.t_plant_pulse_burn`) | 1 | **0** | 2 | **0** |
| distinct scalar literals in the module | 235 | **233** | 268 | **266** |

Both carried values that are not zero are **gone from the compiled program**, replaced by
parameters (+11 and +16 parameter references), and the modules grow by 20 and 31 lines --
arithmetic XLA can no longer do at compile time. The remaining `3045` constants are the
port's own coefficients, not node outputs.

**One §25 claim this section could not reproduce.** §25's Arm D found `1 / eyoung_ins`,
`4 * eyoung_ins` and a series stiffness folded on the host at `tfcoil/stress.py:188`.
Searching this configuration's pre-change modules for the analogous products of its own
carried values -- `1/0.13`, `4*0.13`, `1/3.15576e7`, `2*3.15576e7`, `3.15576e7/2` -- finds
**none**. Not a contradiction: Arm D was measured on a tokamak, and the TF stress model
is not in the stellarator's SAND block at all. It does mean the folding half of the case
is unverified *on this configuration*; what is verified here is the presence of the
constants and their removal.

### 28.5 The Scan win was **not** bought, and the three causes are now separately measured

This is the part the task asked to measure rather than assert, and the measurement says
the change does not deliver it.

**The probe.** `stellarator_helias`, two machines identical but for
`.tfcoil.eff_tf_cryo` -- `0.13` as `indat.resolve_eff_tf_cryo` gives it, and `0.14`, which
is what an IN.DAT naming `eff_tf_cryo = 0.14` produces (`indat.graph_for`'s docstring: a
machine comes from an IN.DAT or from an explicit `eqx.tree_at` on one). The *whole
occupant* is replaced rather than its `value` leaf, because `eqx.tree_at` is tree surgery
and does not run a field's converter -- writing `0.14` into a `carried()` field would put a
Python float back and measure the very thing being removed. Each arm builds the cold SAND
condition map and compiles its values and its Jacobian; every XLA compilation in the
process is counted by patching `jax._src.compiler.backend_compile_and_load` (§22.2's
instrument). Arm 1 pays for the first compile of anything; **arm 2 is the scan point.**

| | arm 1 | arm 2 | condition-map static halves equal |
|---|---|---|---|
| before (`HEAD 6bb65494`) | 2 compiles | **2 compiles** | False |
| after (`CarriesValues`) | 2 compiles | **2 compiles** | False |
| after **+ every node's `fn` a `jtu.Partial`** (probe, see below) | 2 compiles | **2 compiles** | False |

The condition value moves as it should in every arm (`1.3351754993456164` ->
`1.3194361957424183`), so the perturbation is real and reaches the answer; and arm 1's
value is **bitwise identical before and after the change**, which is the first evidence
for §28.6's gate.

**Why, in three parts, each localised.** `eqx.filter_jit` keys its cache on
`hashable_partition(x, eqx.is_array)[1]` -- the argument with its array leaves removed --
so anything non-array that differs between two machines forces a recompile.

1. **The carried values.** A Python-float payload is a non-array leaf and lands in the
   key. This is what §25 named and what this section fixes. Measured directly: with the
   old spelling the graph's static halves differ; the fourteen declarations no longer
   contribute.
2. **Every node's `fn`, which is a bound method** (`cottax`'s
   `ExplicitFunction.node_definition`). Two *equal* declarations produce bound methods
   that compare and hash **unequal** -- measured: `ConvertFpyToCalendar() ==
   ConvertFpyToCalendar()` is `True`, `hash` equal, and `a.__call__ == b.__call__` is
   `False` with unequal hashes. So on `stellarator_helias` **141 static leaves of the
   graph differ** between two machines that differ in one number, one per ordinary node,
   and they would differ between *any* two assemblies of the same file. This is the
   dominant cause and it is one line of `cottax`, not of this port: with
   `ExplicitFunction.node_definition` monkeypatched to build `fn=jtu.Partial(_apply,
   self)` for every node -- a **probe**, nothing in `functional_process/` was changed by
   it and `~/jaxgraph` was not touched -- the graph's static halves become **equal**.
3. **The constraint nodes' `functools.partial`.** With (1) and (2) out of the way the
   SAND condition map's static half still differs at exactly **three** leaves, and all
   three are `functools.partial(constraint_N, <switch>=...)` from
   `sand.py:496`/`sand.py:618` (`bound = functools.partial(fn, **static) if static else
   fn`) -- the three active constraints on this file that bind a switch. A plain
   `functools.partial` is not a pytree and compares by identity, so a rebuilt constraint
   never matches the cached one. `jax.tree_util.Partial` in those two places would
   decompose it and leave the bound switch values in the key *by value*, where they
   belong. **Not changed here** -- it is a third defect of the same family and it deserves
   its own bitwise gate.

So the honest statement is: **this change removes one of three independent causes of the
per-scan-point recompile, and the other two are now named, localised and measured.** The
Scan win arrives when all three are gone; none of the three is more than a few lines.

### 28.6 The bitwise gate: `--provider` reproduces every row, `--native` moves two residual cells

**`--provider`, against `functional_process/reference_cold_matrix.txt`.** All twelve rows,
every column -- iteration count, status, `objf` to nine figures, `PRO objf`, `d objf`,
`worst dx`, `max|eq|`, `min ie` -- plus the whole boundary-values block, reproduce
**byte for byte**, with one apparent exception that is measured *not* to be this change's:

| row | the pin says | this pass says |
|---|---|---|
| `st_regression` MDF | `4` SQP it, `converged`, `objf -0`, `d objf 1.00e+00`, `worst dx 9.90e+01` | `FAILED` -- `_refuse_inert_objective`: *"the objective `^cond.numerics.objf` has an identically zero gradient with respect to all 14 design variable(s)"* |

**That row is `b14da8c1`'s, not this section's, and it was checked rather than assumed.**
`drivers._refuse_inert_objective` was added by `b14da8c1` ("Refuse a solve whose objective
cannot move, and find the four that could not", 21:26) and
`reference_cold_matrix.txt` was last regenerated by `2e949920` (20:42), **the commit
before it**. So the pin predates the guard and is stale for exactly the configurations the
guard's own message says it found. The proof is direct: with the seven changed files
restored from the live tree and `models/carried.py` removed -- i.e. this copy at
`b14da8c1` exactly -- `run_cold_matrix.run_one("st_regression.IN.DAT", PROVIDER)` gives
the *same* `FAILED`, and the objective's gradient row is `[0.]*14` with
`max|.| = 0` and `0/14` non-zero entries. It is `[0.]*14` after the change too, bit for
bit. **Nothing about the objective row moved.**

That also disposes of an explanation this section drafted and then had to withdraw: that
the zeros ceasing to be foldable constants had turned a not-quite-zero objective row into
an exactly-zero one. It was never not-quite-zero. **Flagged as a prediction that failed.**

So the honest reading of the `--provider` gate is **twelve of twelve**, including the two
rows §24 moved (`stellarator_helias` MDF 108/`1.21775747`, SAND 257/`1.21775737`),
`low_aspect_ratio_DEMO SAND`'s `cap(500)` and `st_regression SAND`'s `FAILED`. Given §24's
measurement that a `5.15e-13` Jacobian difference flips this stellarator between outcomes,
that is the result the change had to produce.

**`--native`, against a baseline taken in this copy before the change** (there was no
`native_baseline.txt`; the pre-change pass is now written to `<copy>/native_baseline.txt`,
577 s, seven configurations). **Eleven of twelve rows byte for byte**; one moves, and only
in its two residual columns:

| row | before | after |
|---|---|---|
| `low_aspect_ratio_DEMO` SAND | 107 it, `converged`, `objf -0.399154819`, `max\|eq\| 4.96e-13`, `min ie 1.44e-12` | 107 it, `converged`, `objf -0.399154819`, **`max\|eq\| 3.34e-13`**, **`min ie 8.45e-12`** |

Same iteration count, same status, same objective to nine figures; the converged
equality residual and the least inequality margin move at the `1e-13`/`1e-12` level. That
is the rounding shift the change was expected to cause -- arithmetic XLA used to fold is
now done at run time -- landing in the one place where it is visible without changing
what the solve did. **It is a real bitwise move and it is reported as one; nothing was
widened.** Every other native row, `st_regression`'s two `FAILED`s included, is identical.

### 28.7 `value: float = dataclasses.field(kw_only=True)` — the objection is right, and the fix is not "make it an input"

The objection, in the owner's words: *"that `value: float` is not a kw_only thing;
that's bad design. It should then not be a model but an input no?"* Three parts.

**1. He is right that these are not models.** A body of `return self.value` has no
function in it. The arrow from the node's inputs to its outputs is the identity on a
datum the graph did not compute, which is the definition of a *source*, not of a model.
Calling it an `ExplicitFunction` does not change what it is, and the `kw_only` field is
the tell: it is the one datum the class exists to carry, and it is spelled as a
constructor detail because there is no vocabulary for "this node's output is stated, not
derived".

Of the fifteen declarations in this family (the fourteen of §28.1 plus
`DoubleNullUpperBuild`), **one derives from graph values and fourteen do not**:

| kind | declarations | what the body does |
|---|---|---|
| **derives** | `DoubleNullUpperBuild` | reads `.build.dz_shld_lower`/`.build.dz_vv_lower`, writes the upper pair. A real node. |
| **carries a resolution** | `TfCryoplantEfficiency`, `TfInsulationYoungsModulus`, `TfConductorYoungsModulus`, `PfCoilResistivity`, `BeamElectronDensityFraction`, `EnergyStorageBuildingVolume` | `return self.value`. The *derivation* is real -- a sentinel test, a switch-keyed literature table, a physical consistency rule -- and it lives in `indat.resolve_*`, at assembly. The node carries the answer. |
| **carries a literal** | `StellaratorSolenoidAbsent`, `StellaratorPulseTimes`, `EnergyStorageCostUnpulsed`, `NoDiamagneticCurrent`, `NoPfirschSchluterCurrent`, `HcdSecondaryHeatingNone`, `CrocoTurnCableSpaceExtraVoid`, `CentrepostNeutronicsAbsent` | returns a constant. PROCESS's own source on that arm is a literal assignment (or, for four of them, nothing at all -- the `DataStructure` default stands). |

So the honest count is: **the port has one model and fourteen constant sources in this
family**, and the fourteen are wearing a model's clothes.

**2. But "make it an input" would give the wrong number, measurably.** A boundary input
is answered by whatever the provider can find, and for these paths what it finds is
wrong: `.tfcoil.eff_tf_cryo` comes back `-1.0`, which is a *sentinel and not a value*, and
`.power.thermal_cryo` divides by it; `.buildings.esbldgm3` comes back `1.0e3` m^3 on a
plant that stores nothing; `.build.dr_cs` comes back `0.811 m` on a machine with no
solenoid. Those are the `off` rows of every `reference_provider_*.txt` pin, and §22.7
measured the cost of believing the provider at thirteen of them: **four of seven
configurations lose their solve**. Whatever these things are, deleting them is not the
answer.

**3. The real gap is that cottax has two kinds of name and needs three.** Today a
variable is either *owned by a node* or *a boundary input*, and the port's entire
argument for `initialisation.py` is "owned, because owned is enforced and boundary is
not". What is missing is a third: **a name the graph owns whose value is stated at
assembly** -- a `Source`/`Constant` declaration with declared outputs, an array-valued
payload, and no body. `ExplicitFunction` with `return self.value` is that declaration
written in the vocabulary that exists.

**Recommendation, and it is a recommendation -- nothing below is done here.**

1. **Add the declaration kind, in `cottax` (`Source`) or in `functional_process` until
   `cottax` has one.** `CarriesValues` should be read as its 80 % stand-in: it already
   holds the payload as data and already makes that data visible to the trace. What it
   does not do is *say* that the node has no body, which is the part the owner is
   objecting to.
2. **Move the eight literal carriers to it.** They lose nothing: there is no body worth
   calling, and the audit sentence ("on this arm PROCESS's own source is four literal
   assignments") is exactly a `Source`'s docstring.
3. **Move the six resolutions to it as well, and leave `indat.resolve_*` where it is.**
   The resolution is a function of *switches* and of the input file's raw text, and
   neither is a graph value in this port -- `machine_from_indat`'s docstring gives the
   rule for switches, and `initialisation.py`'s gives the rule for raw values
   (`mdf.seed` grounds an unrecognised `VarPath` at `0.0` silently, so a `.raw.*` read
   would resolve every sentinel against zero without a word). A node body cannot hold
   this computation; a `Source` payload is the honest place for its answer.
4. **Leave `DoubleNullUpperBuild` an `ExplicitFunction`.** It is the one member of the
   family that reads the graph and it should not move with the rest.
5. **Do not fold them into a defaults table.** `next_steps.md` §24.1's argument survives
   intact: a table has to be *consulted* by the provider, which is a second copy of
   `init.py` in a file the graph cannot depend on. Ownership is the point; the objection
   is about the *shape* of the owner, not about whether there should be one.

**One thing the redesign does not fix, and must not be allowed to hide.** A `Source`
whose payload is a Python `float` has exactly the defect this section is about: §28.2
measured that a scalar payload is a compile-time constant whatever the declaration is
called. The design fix and the compiler fix are independent, and `carried()` (or its
equivalent) has to survive the redesign.

### 28.8 What this section did not settle

- **The Scan win is not delivered by this change**, and §28.5 says so with numbers rather
  than hedging. Two of the three causes remain, both outside the scope of a change to
  `initialisation.py`: one line of `cottax` (`ExplicitFunction.node_definition`, the
  bound method) and two lines of `sand.py` (`functools.partial` -> `jax.tree_util.Partial`
  at `:496` and `:618`). Each deserves its own bitwise gate; neither was attempted here.
- **§25's Arm D folding could not be reproduced on `stellarator_helias`.** Arm D measured
  it on a tokamak's TF stress model, which this configuration's SAND block does not
  contain, and no analogous folded product of `0.13` or `3.15576e7` appears in either
  pre-change module. The constants were there and are now gone; that they were *folded*
  is, on this configuration, unverified. The `--native` row that did move
  (`low_aspect_ratio_DEMO SAND`, §28.6) is a tokamak and is the one place the folding is
  visible in an answer.
- **`reference_cold_matrix.txt` is stale at `st_regression MDF`** and this section did not
  regenerate it. The pin is `2e949920`'s and `b14da8c1` added the guard that changes that
  row; regenerating a published pin belongs to whoever moved the structure
  (§26's own closing rule).
- **The arity check is inert for the fourteen** (§28.3). Every one of them declares no
  inputs, so nothing is lost today, but a `CarriesValues` node that later grew a read
  would not get `CallableNode.__check_init__`'s "n inputs declared, but ..." error.
- **`tests/functional_process/models/test_initialisation.py`'s `SEED_FIELDS` docstring is
  now stale**: it says `StellaratorSolenoidAbsent` and `StellaratorPulseTimes` "carry
  theirs as literals in the body", and since this section they carry them as fields like
  everybody else. The test itself still passes (it checks the *seed*, not the node), and
  the line was left alone to keep the diff inside the port.
- **Graph hashing is unchanged in practice and worth stating anyway.** With `fn` a
  `Partial`, `hash(CallableNode)` becomes identity-based for the fourteen. That is not a
  regression: `hash(a.__call__) != hash(b.__call__)` already held for two *equal*
  declarations, so two separately assembled graphs never hashed equal before this section
  either. `hash(GRAPH)` was re-checked and still answers, which is what `mdf._hashable`
  and `sand_harness._SCHEDULE_WHOLE` need.
- **Nodes with inputs were checked and none carries a constant limb**, by AST scan; a body
  that returned `(x * 2, 0.0)` would be the same defect and there is none. The scan is a
  one-off, not a test -- there is no guard that stops the next one being written.


## 29. The three producers §26 found are ported, and one of them was binding (2026-09-01)

> **Measured in a scratch copy, not in the live tree**, at
> `/tmp/claude-1000/-home-tbogaarts-PROCESS/df0c22b1-02c2-4e73-99ce-b061606f318d/scratchpad/PROCESS_producers2`,
> against `HEAD 6bb65494` plus this copy's own changes. Source changes are supplied as
> `.patch` files in the copy and were **not** applied to `~/PROCESS`. Every run below
> set `PYTHONPATH` to the copy and printed `functional_process.__file__` as its first
> line, per `next_steps.md` §27.5 — and one measurement in this section was taken
> twice because that check caught it: `python -c` puts the *cwd* on `sys.path` ahead
> of `PYTHONPATH`, so a `cd`-less invocation measures `~/PROCESS` no matter what
> `PYTHONPATH` says. See §29.3.

§26 found four frozen boundary paths on the two spherical tokamaks and ranked them; it
explicitly left "whether porting the four producers fixes anything" out of scope
(§26.7). Three of the four are ported here — the fourth,
`.current_drive.big_q_plasma`, is stellarator heating and out of this scope too. The
short answer is that **two of the three change what the port reports and one changes
what problem it is solving**, and that the graph-shape cost is three nodes and one
declared read.

### 29.1 What was ported, and where each one went

| path | PROCESS source | ported as | slot | registry unit |
|---|---|---|---|---|
| `.physics.p_plasma_separatrix_rmajor_mw` | `exhaust.py:127-147` + `physics.py:811-816` | `calculate_psep_over_r_metric` / `PsepOverRMetric` | `.tokamak.physics.psep_over_r_metric` | #11 `physics/exhaust.py` |
| `.constraints.pflux_fw_rad_max_mw` (and `.physics.pflux_fw_rad_mw`) | `fw.py:130-144` | `calculate_radiated_wall_load_scaled_plasma_surface` / `RadiatedWallLoad` | `.tokamak.radiated_wall_load` | #32 `fw.py` |
| `.build.r_cp_top` | `build.py:1750-1813` | `calculate_r_cp_top_from_tf_inboard_out` / `RCpTopFromTfInboardOut` | `.tokamak.build.r_cp_top` | #26 `build.py` |

All three are unswitched *in the graph they land in*, and each for a different reason,
which is the part worth writing down:

- **`PsepOverRMetric`** — `Physics.run` computes it outside every `if`. Nothing to
  decide.
- **`RadiatedWallLoad`** — `i_pflux_fw_neutron != 1` is a real second arm
  (`a_fw_total`-normalised), and **no second refusal was minted for it**.
  `indat._first_wall_arm` already refuses that value at `('first_wall_arm', -3)`, and a
  graph holding this node holds `.tokamak.first_wall` too, so no assembled machine can
  reach the unwritten arm here without having been refused there. A duplicate refusal
  would assert the two arms can be chosen apart; they cannot.
- **`RCpTopFromTfInboardOut`** — a genuine new slot key,
  `(.physics.itart, .tfcoil.i_tf_sup)`, with the resistive-ST arm refused
  (`('r_cp_top_arm', -1)`).

### 29.2 [measured] The switch reading that the numbers, not the code, settled

`build.py`'s `r_cp_top` block reads like an `i_r_cp_top` dispatch and is not one. The
guard is `if itart == 1 and i_tf_sup != 1:` (`:1750`), i.e. a *resistive* spherical
tokamak; `i_r_cp_top` is only consulted inside it. **Both tracked spherical tokamaks set
`i_r_cp_top = 2`** (`spherical_tokamak_eval.IN.DAT:78`, `st_regression.IN.DAT:2029`)
**and both set `i_tf_sup = 1`, so on both files that input is inert.** Arm 2's formula
is `f_r_cp * r_tf_inboard_out` and both files set `f_r_cp = 1.4`, so an `i_r_cp_top`-first
dispatch would have produced a centrepost 40 % too wide on exactly the two
configurations this wave exists to fix — and it would have looked plausible.

The discriminator was PROCESS's own converged `DataStructure`, read for both files:

| file | `.build.r_cp_top` | `.build.r_tf_inboard_out` | `f_r_cp * r_tf_inboard_out` |
|---|---|---|---|
| `st_regression` | `1.3405301988363134` | `1.3405301988363134` | `1.8767422783708387` |
| `spherical_tokamak_eval` | `1.208855401921066` | `1.208855401921066` | `1.6923975626894923` |

Equal to the last bit, on both. `TestRCpTopSuperconductingSphericalTokamak` **executes**
that rather than asserting it: its adapter sets all four switches to the ST files' own
values, so a dispatch keyed the other way makes the value test fail rather than quietly
agree.

The same reading, in the other direction, is what makes `RadiatedWallLoad` correct:
`i_pflux_fw_neutron` defaults to `1` (`physics_variables.py:1006`), `st_regression`
comments its `= 2` out (`:2452`) and `spherical_tokamak_eval` never names it, so the
`ffwal`-scaled arm is live on both — and `0.92 * 439.5706 / 810.4940 = 0.4989610`
reproduces PROCESS's converged `pflux_fw_rad_max_mw` to every digit it prints.

### 29.3 A measurement trap met and avoided, twice

`next_steps.md` §27.5 asks every run to print `functional_process.__file__`. It earned
its keep here in a way §24.3's version did not cover: **`python -c` and `python -m` put
the *current working directory* on `sys.path` ahead of `PYTHONPATH`.** A command written
as `PYTHONPATH=$COPY python -c ...` with no `cd` measures whatever tree the shell
happens to be standing in — and in this session that was `~/PROCESS`. One SCC census in
§29.6 was taken twice for exactly this reason: the first pair of runs printed
`/home/tbogaarts/PROCESS/functional_process/__init__.py` for *both* the "before" and the
"after" arm and reported them identical, which is true and says nothing. The rule that
catches it is not "set `PYTHONPATH`" but "read the first line of the output".

**The second trap is `pytest`'s own diff wording, and it has already been misread once
in this file.** §26.8.5 reports the stale `st_regression` provider pin as *"a new guess
port ... plus one extra `unwritten defaults .vacuum.xmult_i` row"*. There is no extra
`xmult_i` row. `pytest`'s list comparison prints `Left contains one more item: <X>` where
`X` is the element at index `len(right)` of a **sorted** list — i.e. the trailing
element, not the semantic difference. `unwritten defaults .vacuum.xmult_i` sorts last in
every one of these pins, so it is named whenever the actual list is one longer for any
reason. Measured: `diff` of the regenerated pin against the checked-in one is exactly
one added line per configuration (§29.4), and `.vacuum.xmult_i` is not it.

### 29.4 [measured] The boundary, all seven configurations

`$PY -m functional_process.provider --write`, then `diff` against the checked-in pins:

| configuration | pin delta |
|---|---|
| `stellarator_helias` | **byte-identical** |
| `helias_5b` | **byte-identical** |
| `large_tokamak_nof` | `+ default defaults .constraints.f_fw_rad_max` |
| `large_tokamak_eval` | `+ default defaults .constraints.f_fw_rad_max` |
| `low_aspect_ratio_DEMO` | `+ default defaults .constraints.f_fw_rad_max` |
| `spherical_tokamak_eval` | `- computed process .build.r_cp_top`, `+ input indat .constraints.f_fw_rad_max` |
| `st_regression` | `- computed process .build.r_cp_top`, `+ input indat .constraints.f_fw_rad_max`, `+ guess process ^guess.tfcoil.dr_tf_plasma_case` |

**One declared read for three producers**, and it is a plain IN.DAT constant. Everything
else the three nodes read was already owned or already on the boundary. Contrast the
2026-08-27 TF wave, where four landed producers cost nine new reads.

The `st_regression` guess row is **not this wave's** — it is the stale pin §26.8.5
already documented, and regenerating the pin is the fix that section names and declines
to make. It is made here, because this wave had to regenerate that file anyway.

**`computed` reaches zero on every configuration**, which is what
`test_provider.py`'s "may only go down" was written for. That test's assertion
collapses to `{stem: [] for stem in CONFIGURATIONS}`; its name and its docstring's
2 -> 1 -> 0 history are kept deliberately.

**A pre-existing failure closes as a side effect.**
`test_provider.py::test_each_configuration_s_answer_is_its_pin[st_regression]`, which
§26.8.5 records as failing at HEAD and *not* caused by that section, passes now — the
pin regeneration is exactly the fix it named. `tests/functional_process/test_provider.py`
+ `test_registry_coverage.py`: **50 passed**, from 44 passed / 1 failed.

### 29.5 [measured] The values, at the port's own answer

`spherical_tokamak_eval` states a root find (`i_process_run_mode = -2`), so the port
solves PROCESS's own square system and every path below is read out of the converged
env, not out of a seed:

| path | port before | port after | PROCESS converged | relative |
|---|---|---|---|---|
| `.physics.p_plasma_separatrix_rmajor_mw` | `0.0` (frozen) | `40.281620625054124` | `40.28162046798473` | **3.9e-09** |
| `.constraints.pflux_fw_rad_max_mw` | `0.0` (frozen) | `0.4989610344677403` | `0.4989610384942579` | **8.1e-09** |
| `.physics.pflux_fw_rad_mw` | *not in the graph* | `0.4989610344677403` | `0.4989610384942579` | **8.1e-09** |
| `.build.r_cp_top` | `0.0` (frozen) | `1.2088554019210653` | `1.208855401921066` | **5.5e-16** |

The two `~1e-8` rows are **not** porting error: this row's `worst dx` against PROCESS's
own converged design vector is `3.64e-09`, so the port and PROCESS are evaluating at
slightly different points and these agree to the size of that gap. `r_cp_top` agrees to
half an ulp because it is an identity on a field the two already agreed on to half an
ulp.

Unit-level agreement is `Tier1Contract` and is separate from the above: five new
contracts, all green plain and under `--fp-gradients` — value agreement at machine
precision *and* gradient agreement against PROCESS's own finite difference with the
per-point Richardson bar. `test_exhaust.py` 82 passed, `test_fw.py` 90 passed,
`test_build.py` 202 passed. No tolerance was widened anywhere.

### 29.6 [measured] What moved in the cold matrix, and what did not

Both arms of the A/B were run in this session. The **baseline is a fresh run of the live
tree at `HEAD b14da8c1`**, not the checked-in `reference_cold_matrix.txt`: that pin was
measured at `6bb65494`, two commits back, and predates §26.5's own guard commit
("Refuse a solve whose objective cannot move"). Using it as the baseline would have
charged this wave for that commit's row.

**The five unaffected configurations, nine rows: bitwise on every numeric column.**
`stellarator_helias` MDF/SAND, `helias_5b` MDF/SAND, `large_tokamak_nof` MDF/SAND,
`large_tokamak_eval` MDF, `low_aspect_ratio_DEMO` MDF/SAND all reproduce
`reference_cold_matrix.txt` exactly — same `objf`, `d objf`, `worst dx`, `max|eq|`,
`min ie`, same SQP iteration count, same status. What moves is the graph size, by
exactly `+3` nodes everywhere (`graph` 154 -> 154 on the stellarators, which have none
of the three; 243 -> 246 and 245 -> 248 on the tokamaks) and the block count by the
same 3.

**The two spherical tokamaks:**

| row | before (`b14da8c1`) | after | reading |
|---|---|---|---|
| `spherical_tokamak_eval` MDF | converged, `worst dx 3.64e-09`, `max`\|`eq`\|` 1.16e-09`, 2 SQP | **identical**, graph 241 -> 244 | the solve cannot move: this file root-finds the *equalities*, and both new conditions are reported-only |
| `st_regression` MDF | `FAILED` — `_refuse_inert_objective` | `FAILED` — same guard | **not this wave's failure**, and the message is what changed |
| `st_regression` SAND | `FAILED` — `KeyError: ^cond.numerics.objf` | identical | §26.7's open item, untouched |

**The `st_regression` row is where this wave is actually visible, and it is visible in
the guard's own message rather than in a number:**

```
before:  other conditions with an all-zero row: ['^cond.constraints.c56', '^cond.constraints.c67']
after:   other conditions with an all-zero row: none
```

Both runs stop on the same refusal, because `st_regression`'s objective is
`.current_drive.big_q_plasma` and that producer is out of this wave's scope. What this
wave removed is the *rest* of the defect: two of the three identically zero Jacobian
rows are gone, and `boundary --inert` says the same thing statically, without running
anything:

```
before:  st_regression   14 design, 19 driven condition(s) -> 3 inert
             .Objective, .Constraint56, .Constraint67
after:   st_regression   14 design, 19 driven condition(s) -> 1 inert
             .Objective       1/1 operand(s) frozen, 1 in cone: .current_drive.big_q_plasma

before:  spherical_tokamak_eval  3 design, 3 driven -> 0 inert  (+2/15 reported-only)
             reported-only .Constraint56, .Constraint67
after:   spherical_tokamak_eval  3 design, 3 driven -> 0 inert  (+0/15 reported-only)
```

`spherical_tokamak_eval`'s reported-only inert list is **empty**, which is the first time
any configuration's has been.

### 29.7 [measured] §26.7's `min ie -1.38e+01` is explained — and it is a mis-signed column, not a defect this wave fixed

§26.7 left it open, noting that *"all three of its frozen paths make constraints read as
satisfied, so none of them can produce a violation of that size"*. That reasoning was
right and the conclusion it hinted at was wrong: **`-1.38e+01` is not a violation at
all.**

Every reported inequality of `spherical_tokamak_eval` MDF at the answer, before and
after (`^cond.constraints.cNN`, the value the port's own condition nodes carry):

| condition | before | after | note |
|---|---|---|---|
| `c81` | `-1.375904e+01` | `-1.375904e+01` | **this is `min ie`**, both times |
| `c16` | `-4.431010e+00` | `-4.432085e+00` | the only *other* row that moves — `r_cp_top` reaches the electric-power chain |
| `c15` | `-1.930108e+00` | `-1.930108e+00` | |
| `c56` | `-1.000000e+00` | **`+7.040516e-03`** | the frozen `0/40 - 1` becomes the real value |
| `c67` | `-1.000000e+00` | **`-5.841991e-01`** | the frozen `0/1.2 - 1` becomes `0.49896/1.2 - 1` |
| the other ten | | unchanged | |

**A `^cond.*` port carries PROCESS's own normalised residual, unnegated**, and
`drivers.py`'s own "Sign convention — checked, not assumed" block says so: `leq` returns
`value/bound - 1` and `geq` returns `1 - value/bound`, so **positive means violated for
both**, and it is the *driver* that passes `-g` to pyvmcon. The SQP arm of
`run_cold_matrix.cold_mdf` takes `min ie` off the driver's callback trace, which is
VMCON-signed; the **root-find arm takes it off the raw `^cond` values, which are not**.
So on the two `i_process_run_mode = -2` files the `min ie` column is
`min(PROCESS-normalised residuals)` — the *most comfortably satisfied* constraint —
while the table's own footer says "NEGATIVE IS VIOLATED".

`c81` is the "Ne" density-ratio inequality sitting at ~14.8x its bound; nothing is wrong
with it and nothing ever was. The genuinely violated inequality on that file at the
port's own answer is **`c56`, at `+7.04e-03`** — PROCESS violates the same constraint by
the same +0.70 % at its own answer (`40.2816` against `40`), which is §26.3's rank-3 row
and precisely what this wave gave a producer. `large_tokamak_eval`'s `-1.99e+00` is the
same artefact: §11.1 measured *its* violated constraint as c68 at `+4.949e-02`, positive.

**So this wave does not explain `-1.38e+01`; it makes the column's problem visible.**
Not fixed here: `run_cold_matrix.cold_mdf`'s root-find arm should negate, or the footer
should say the arms differ. Left as a one-line change for whoever owns that file, and
recorded rather than quietly patched because `reference_cold_matrix.txt` is a published
pin and every root-find row in it would move.

### 29.8 Predictions this section made and measurement refuted

- **"Porting `p_plasma_separatrix_rmajor_mw` will change `st_regression`'s answer,
  because c56 is its most binding constraint."** Refuted, and for a reason that was
  in front of me the whole time: that row does not *reach* an answer. It stops at
  `_refuse_inert_objective` before the first QP, both before and after, because the
  objective is `big_q_plasma` and that producer is out of scope. The constraint is live
  now and nothing has yet solved with it. **The claim "the port has been solving a
  relaxed problem" is true of the problem statement and cannot be demonstrated on a
  number until `big_q_plasma` lands.**
- **"The new nodes will be pruned out of the configurations that do not read them, so
  those graphs will be unchanged."** Refuted: all three are unconditional slots, so
  every tokamak graph gains all three (`+3` nodes on all five). Nothing numeric moved,
  but the graph-size columns did, and a claim of "bitwise" has to say which columns it
  means.
- **"Adding a producer for a path inside an existing SCC's neighbourhood risks growing
  the SCC."** Not refuted but not confirmed either — measured and flat.
  `Blocking.scc` over both spherical tokamaks is identical before and after, including
  the pre-existing three-node `dr_tf_inboard_winding_pack` / `tf_inboard_radii` /
  `dr_tf_plasma_case` block that `st_regression` alone has. `r_tf_inboard_out` is
  upstream of the new node and `r_cp_top` is downstream of everything that reads it.
- **"`i_r_cp_top = 2` is the arm both spherical tokamaks take, because both files set
  it."** Refuted by PROCESS's own converged `DataStructure` before a line was written
  (§29.2). This is the one that would have shipped a wrong number.

### 29.10 [measured] The pins and the guard's own tests, after

Four checked-in artefacts move, and every one of them moves in the direction its own
docstring calls good:

- **`reference_boundary_tokamak.txt`**: `377 -> 378` inputs, `11 -> 11` guesses. One
  line added, `input .constraints.f_fw_rad_max`, and nothing removed --
  `large_tokamak_eval` does not read any of the three producers' outputs (it takes the
  D-shape TF arm, not the picture frame, and states neither c56 nor c67), so this file
  pays the read and collects none of the benefit. `reference_boundary.txt` (the
  stellarator) is **byte-identical**, `289 + 6`.
- **The seven `reference_provider_*.txt`**: §29.4.
- **`test_boundary.py::test_st_regression_s_objective_is_inert_and_the_other_six_files_
  are_clean`**: `{".Objective", ".Constraint56", ".Constraint67"}` becomes
  `{".Objective"}`. This assertion is now doing its job in both directions -- it
  *caught* the defect four days ago and it *pins the fix* today, which is the strongest
  thing a regression test can be asked to do and the reason it was edited rather than
  deleted.
- **`test_boundary.py::test_an_evaluation_file_s_inequalities_are_reported_and_not_
  driven`**: `spherical_tokamak_eval`'s reported-only inert set becomes **empty**. The
  test's actual claim -- that an evaluation file's inequalities are `reported` and not
  `driven` -- was resting entirely on that file having two inert ones, so a second
  assertion was added on `large_tokamak_eval`'s eight, which are inert *by design* and
  are the case the separation exists for. Deleting the spherical half instead would have
  thrown away the regression test for the defect this wave closed.

`test_boundary.py` 28 passed; `test_registry_coverage.py` + `test_cold_matrix.py` +
`test_provider.py` + `test_boundary.py` **108 passed**, from 104 passed / 4 failed
before the pins and assertions were brought up to date.

Scoped test totals for the wave, all in this copy:

| what | result |
|---|---|
| `tests/functional_process/models` (whole subtree, plain) | **4803 passed, 4478 skipped**, 199 s |
| `test_build.py` + `test_fw.py` + `test_exhaust.py`, `--fp-gradients` | **374 passed**, 19 s |
| `test_registry_coverage` + `test_cold_matrix` + `test_provider` + `test_boundary` | **108 passed** |

The whole `models` subtree was **not** run under `--fp-gradients`: at 12 % after fifteen
minutes it is an hour-plus job and it is not this wave's scope. The three touched files
were, which is where the five new contracts live. No tolerance was widened anywhere, and
`gradient_safety`/`gradient_floor` are untouched.

### 29.9 Not resolved

- **`.current_drive.big_q_plasma`.** Out of scope by instruction and the reason
  `st_regression` still cannot be measured end to end. It is the last of §26.2's four
  and the only one left.
- **`run_cold_matrix`'s `min ie` sign on the root-find arm** (§29.7). Diagnosed, not
  fixed.
- **`reference_cold_matrix.txt` is stale against `HEAD b14da8c1`** — it is a
  `6bb65494` measurement and the guard commit moved the `st_regression` MDF row from
  `converged`/`objf -0` to `FAILED`. This wave deliberately does **not** regenerate it,
  because doing so would fold that commit's delta into this one's patch. Nine of its
  twelve rows are reproduced bitwise here; the three that are not are `st_regression`
  MDF (the guard) and the two `graph`/`nodes` columns on every tokamak (`+3`).
- **`st_regression` SAND's `KeyError: VarPath(^cond.numerics.objf)`.** Unchanged, same
  root as the objective's, not investigated.
- **The `i_pflux_fw_neutron != 1` and resistive-ST `r_cp_top` arms stay UNPORTED.** Both
  are refused with a recorded reason and neither is reachable from any input file in
  this repository.

## 30. Every active constraint, classified by what it owns (2026-09-01)

> **Measured in a scratch copy, not in the live tree**, at
> `/tmp/claude-1000/-home-tbogaarts-PROCESS/df0c22b1-02c2-4e73-99ce-b061606f318d/scratchpad/PROCESS_cshape`,
> against `HEAD 6bb65494` with **no source change of any kind made by this section** —
> the only files added are `audit_cshape.py` at the copy's root, its saved output
> `constraint_shape_census.txt`, and `section30.patch` (this section's own diff against
> `~/PROCESS`). The copy's `functional_process/run_cold_matrix.py` also differs from the
> live tree; that is a *live-tree* change made by another agent after this copy was
> taken, not a change here, and nothing measured below reads that module. Nothing was
> applied to `~/PROCESS`. Every
> number here is **static**: graph assembly, `sand._Resolver`, declared `reads`/`owns`,
> `Graph.reach`, `Graph.strongly_connected_components`, and one `Insert` probe.
> **No solve was run, no PROCESS pipeline was run, no seed was read.**
> `functional_process.__file__` was printed as the first line of every run.

The question this section answers: the repo owner's observation that *an equality
constraint is often a cycle in the graph wearing different clothes*, and that handing
such a relation to the SQP lifts graph structure into the optimiser (the move §23 tested
on a SAND residual). Nobody had looked at PROCESS's own constraints that way.

### 30.1 The axis, and why the obvious one is wrong

The first classification tried here — *"cycle-shaped if every argument is node-produced,
specification if any argument is exogenous"* — **is not the right axis, and it gives the
wrong answer on the two constraints it was supposed to settle.** Under it, `c1` (beta
consistency), `c2` (global power balance) and `c11` (`rbld == rmajor`) all come back
**specification** and `c83` comes back **cycle-shaped** — the exact reverse of the
scouting hypothesis and, as it turns out, the exact reverse of the truth.

The reason is that "exogenous" conflates three different things that the resolver reports
identically as *not owned by any node*:

| kind | what it is | example |
|---|---|---|
| **design** | an active `numerics.ixc` — a free unknown the SQP moves | `.physics.rmajor` (ID 3) on `large_tokamak_nof` |
| **input** | a prescribed value the file or a default supplies | `.constraints.pflux_fw_neutron_max_mw` |
| **frozen** | a path with no producer that PROCESS *does* compute — a porting gap | `.physics.p_plasma_separatrix_rmajor_mw` on the STs (§26) |

The axis that actually discriminates is the owner's: **does the constraint own an
unknown, or is every argument already computed?**

- **DETERMINING** — at least one argument is an active `ixc`. One equation, one named
  unknown; it *could* become a producer node plus a driver.
- **PURE CHECK** — every argument is owned by a node. It adds an equation and no
  unknown. It cannot become a producer: cottax refuses to mint one (see §30.4).
- **SPEC** — the free arguments are all prescribed inputs. A computed quantity against a
  given.
- **INERT** — no computed and no `ixc` argument at all. A constant compared to a
  constant; §26.5's refusal already covers these.

### 30.2 [measured] The classification grid, all seven configurations

`role/verdict`; `eq`/`ineq` is PROCESS's **positional** split (`icc[:n_equality]`), not
the body's `eq`/`leq`/`geq`. Full per-argument detail in `constraint_shape_census.txt`
(`##### final #####`).

| id | ste_hel | hel_5b | lt_nof | lt_eval | laDEMO | spt_eval | st_reg |
|---|---|---|---|---|---|---|---|
| c1 | -- | -- | eq/**DET** | eq/**DET** | eq/**DET** | eq/**DET** | eq/**DET** |
| c2 | eq/SPEC | eq/SPEC | eq/SPEC | eq/SPEC | eq/SPEC | eq/SPEC | eq/SPEC |
| c5 | -- | -- | ineq/**DET** | ineq/**DET** | ineq/**DET** | ineq/**DET** | ineq/**DET** |
| c8 | ineq/SPEC | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | -- | -- |
| c9 | -- | -- | ineq/SPEC | ineq/SPEC | -- | ineq/SPEC | ineq/SPEC |
| **c11** | -- | **eq/SPEC** | **eq/DET** | -- | **eq/DET** | **eq/SPEC** | **eq/SPEC** |
| c13 | -- | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | -- | -- |
| c15 | -- | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC |
| c16 | **eq**/SPEC | **eq**/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC |
| c17 | ineq/SPEC | -- | -- | -- | -- | ineq/SPEC | ineq/SPEC |
| c18 | ineq/SPEC | -- | -- | -- | -- | -- | -- |
| c24 | ineq/SPEC | ineq/SPEC | ineq/**DET** | ineq/SPEC | ineq/**DET** | ineq/SPEC | ineq/**DET** |
| c25 | -- | -- | ineq/SPEC | ineq/SPEC | -- | -- | -- |
| c26 | -- | -- | ineq/**DET** | ineq/SPEC | ineq/**DET** | -- | -- |
| c27 | -- | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | -- | -- |
| c30 | -- | -- | ineq/SPEC | ineq/SPEC | **eq**/SPEC | ineq/SPEC | ineq/SPEC |
| c31 | -- | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC |
| c32 | ineq/SPEC | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC |
| c33 | -- | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC |
| c34 | ineq/SPEC | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | -- | -- |
| c35 | ineq/**CHK** | -- | ineq/**CHK** | ineq/**CHK** | ineq/**CHK** | -- | -- |
| c36 | -- | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | -- | -- |
| c46 | -- | -- | -- | -- | -- | ineq/**CHK** | ineq/**CHK** |
| c56 | -- | -- | -- | -- | -- | ineq/*INERT* | ineq/*INERT* |
| c60 | -- | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | -- | -- |
| c62 | ineq/SPEC | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC | ineq/SPEC |
| c65 | ineq/SPEC | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | -- | -- |
| c67 | ineq/SPEC | -- | -- | -- | -- | ineq/*INERT* | ineq/*INERT* |
| c68 | -- | -- | ineq/**DET** | ineq/SPEC | ineq/**DET** | -- | -- |
| c72 | -- | -- | ineq/SPEC | ineq/SPEC | ineq/SPEC | -- | -- |
| c81 | -- | -- | ineq/**CHK** | ineq/**CHK** | ineq/**CHK** | ineq/**CHK** | ineq/**CHK** |
| c82 | ineq/**CHK** | -- | -- | -- | -- | -- | -- |
| c83 | ineq/**CHK** | -- | -- | -- | -- | -- | -- |
| c84 | -- | ineq/SPEC | -- | -- | -- | -- | -- |
| c90 | -- | -- | -- | -- | ineq/SPEC | -- | -- |

The headline count: of 35 distinct constraint ids across the seven files, **6 are pure
checks** (`c35`, `c46`, `c81`, `c82`, `c83`, and — see §30.3 — `c2` in every useful
sense), **6 are determining on at least one configuration** (`c1`, `c5`, `c11`, `c24`,
`c26`, `c68`), **2 are inert**, and the remaining ~21 are ordinary specifications
comparing one computed quantity against one prescribed bound. **The overwhelming
majority of PROCESS's constraint set is exactly what it looks like** — a limit against a
number in the input file. The interesting structure is confined to a handful.

### 30.3 [measured] Three things the scouting got wrong

Flagged as required, and all three were *my* predictions or the brief's, not artefacts.

**(a) `c1` is not a cycle. It is an assignment.** Beta consistency was the one
constraint whose own docstring calls it `Compare`-shaped. It is `DETERMINING`, and on
the three files where `.physics.beta_total_vol_avg` is `ixc 5`, the invertibility probe
(§30.4) says making the constraint its **producer adds no cycle at all — the graph stays
acyclic**. So `c1` is not an implicit relation needing a `RootFind`; it is a plain
`CallableNode` writing beta, and PROCESS *knows this*: `Stellarator.run()` refuses
`ixc 5` when `istell > 0` and directly overwrites `.physics.beta_total_vol_avg` with this
constraint's own right-hand side, commenting *"This replaces constraint equation 1 as it
is just an equality"* (already recorded in `constraints.py`'s `constraint_1` docstring).
The port's graph factory reproduces the split independently: `.physics.beta_total_vol_avg`
**is** node-produced on both stellarators (`.stellarator.stellarator_beta_and_stored_energy`)
and is **not** produced on any tokamak. The same physical statement is a node on one
device and an SQP equality with a free unknown on the other, and the choice is PROCESS's,
not the port's.

**(b) `c2` is not a determining constraint and it is not a specification either.** Global
power balance takes nine node-produced arguments plus `f_p_alpha_plasma_deposited` (a
physical constant, `0.95`, not a bound) and — on the two stellarators —
`.physics.pden_plasma_ohmic_mw`, an `unwritten` path PROCESS never writes either (§22).
It names **no** unknown. It is an equation with no local variable to own, on every one of
the seven files, and it is an equality on every one of the seven. It is the purest
instance of the owner's point: *it adds an equation without adding an unknown.* The
unknown it actually determines is `.physics.hfact` (`ixc 10`), five nodes upstream
through the confinement scaling — see §30.6 for why "the unknown" is not well defined.

**(c) `c83` is not "the sole active inequality at the solution", and this repo already
measured it.** §23(a) records the active set (`|ie| < 1e-6`) over SAND iterations 11–17
as `^cond.constraints.c83` **alone**, with `c35` joining once at 12 — which is where the
brief's claim comes from — but §23(b) records that **at the solution the active set is
four constraints in both formulations: `c24`, `c83`, `c62`, `c35`**, matching §17.2's
"one of exactly four binding inequalities". §23(d) additionally **withdraws** the reading
that `c83` is an identity: *"it starts at `-2.361e-01` and converges early, in both
formulations, so it is an ordinary binding constraint and not a degenerate row"*. So:
`c83` is binding at the solution, is sole-active for a stretch of the trajectory, and is
**not** an equality in disguise. Confirmed from the record, not re-measured — the tracked
`stellarator_helias.MFILE.DAT`/`OUT.DAT` in this tree are crash stubs (1.4 kB, no
constraint rows) and cannot answer it, and re-measuring costs a solve.

### 30.4 [measured] What a constraint would look like as graph structure

For every constraint, for every free argument, a probe node owning that argument and
reading the others was `Insert`ed and the resulting SCC measured. **This is the decisive
test and it needs no solve**: a constraint that *could* be graph structure must be able
to become a producer, and cottax's own `Graph.__check_init__` says whether it can.

**Every pure check is refused, by cottax, for the stated reason.**

```
c82  toroidalgap -> REFUSED: output(s) already produced here
     dx_tf_inboard_out_toroidal -> REFUSED
c83  available_radial_space  -> REFUSED: output(s) already produced here:
                                ['.build.available_radial_space']
     required_radial_space   -> REFUSED
c35  j_tf_wp -> REFUSED;  j_tf_wp_quench_heat_max -> REFUSED
c81  nd_plasma_electron_on_axis -> REFUSED;  nd_plasma_pedestal_electron -> REFUSED
c46  eps -> REFUSED;  plasma_current -> REFUSED;  c_tf_total -> REFUSED
```

That is the owner's claim demonstrated rather than argued: **a constraint over
only-computed quantities cannot become a producer, because every variable already has
one.** It is not a design preference; it is a type error.

**The determining constraints, and the cycle each would close:**

| constraint | owned variable | SCC formed | members |
|---|---|---|---|
| `c11` (tokamaks) | `.physics.rmajor` | **3 nodes** | `Close_c11`, `.tokamak.build.radial_build_to_plasma_centre`, `.tokamak.plasma_geom.minor_radius` |
| `c11` (`helias_5b`) | `.physics.rmajor` | **10 nodes** | `.stellarator.build`, `.stellarator.coils.{coil_current, coil_radial_thickness, intersect, winding_pack_intersect_inputs, winding_pack_total_size_post}`, `.stellarator.stellarator_{plasma_geometry, scaling_factors}`, `^problem.stellarator.coils.intersect` |
| `c1` | `.physics.beta_total_vol_avg` | **none — acyclic** | it is a producer, not a problem |
| `c1` | `.physics.nd_plasma_electrons_vol_avg` | 11 nodes | the whole density-profile/fusion-rate loop |
| `c5` | `.physics.nd_plasma_electrons_vol_avg` | 9 nodes | same loop, minus `set_fusion_powers` |
| `c24` | `.physics.beta_total_vol_avg` | 3 nodes | `.tokamak.plasma_beta.{thermal, toroidal}` |
| `c26` | `.pf_coil.j_cs_flat_top_end` | 16 nodes | the CS/PF sizing chain |
| `c68` | `.physics.rmajor` | 28 nodes | `rmajor`'s whole tokamak cone |

**Read the SCC size as bookkeeping, not as a category.** An earlier draft of this
section made much of `c11` on a tokamak closing a *three-node* cycle in a 243-node graph,
as though a short closure were a different kind of object from a long one. **It is not,
and that framing is withdrawn.** `c2`'s global power balance is the same object as
`c11`'s short chain: an equality closing a loop whose unknown sits at the nearest
producerless variable upstream. The only thing that changes with more than one equation
is that you must **choose which variable each equality determines** — a pivot. A pivot
changes the bookkeeping and the block sizes; it does not change the answer, and it does
not make one relation structural and another not. What the table above reports is *which
block you would get for a given pivot choice*, and §30.6 is where the choice itself is
measured.

**One honesty note on the probe.** It is mechanical: it reports what is *structurally*
possible, not what is *meaningful*. It cheerfully says that making `c2` the producer of
`f_p_alpha_plasma_deposited` closes a 5-to-7-node SCC — i.e. "solve the power balance for
the alpha deposition fraction". That is a physical absurdity and the probe cannot know
it. Read the SCC column as *"if you chose this unknown, this is the block you would
get"*, never as *"this is the unknown"*.

### 30.5 [measured] The `c11`/`c83` verdict: it is two relations, not one

The anomaly as briefed: `stellarator_helias` enforces `c83` and not `c11`; `helias_5b`
enforces `c11` and not `c83`; the same physical question appears as an equality on one
machine and a slack inequality on the other. The owner's sharpening: **a graph cycle
cannot be conditionally active.**

The owner is right, and the resolution is that **`c11` is not one relation. Its direction
of causation is set by the input file, and it points four different ways across seven
files.** The discriminator is one bit that is *not* in `icc` at all — whether
`ixc = 3` (`.physics.rmajor`) is active:

| configuration | `c11`? | `ixc 3`? | `c83`? | what `c11` *is* there |
|---|---|---|---|---|
| `stellarator_helias` | no | **yes** | **yes** | absent; `rmajor` free, pinned by `c83` |
| `helias_5b` | **yes** | no | no | **inert** — see below |
| `large_tokamak_nof` | **yes** | **yes** | no | **determining**: one equation, one unknown, 3-node cycle |
| `large_tokamak_eval` | no | no | no | absent; `rmajor` fixed, build unconstrained |
| `low_aspect_ratio_DEMO` | **yes** | **yes** | no | **determining**, 3-node cycle |
| `spherical_tokamak_eval` | **yes** | no | no | **specification**: build must sum to a prescribed `rmajor` |
| `st_regression` | **yes** | no | no | **specification** |

`.physics.rmajor` has **no producer in any of the seven graphs**, and is read by 14
nodes on a stellarator and 45 on a tokamak. `.build.rbld` is produced everywhere
(`.stellarator.build` / `.tokamak.build.radial_build_to_plasma_centre`) and — measured —
**read by nobody, on all seven**. So are `.build.available_radial_space` and
`.build.required_radial_space`. The dataflow is strictly one-directional, `rmajor →
everything → build → rbld`, source to sink. **There is no cycle in the graph today**;
`c11` creates one only by making the constraint the producer of `rmajor`, which is
legal exactly when `rmajor` is not otherwise determined — i.e. when `ixc 3` is active.

`c83` is a different statement with the same physics: both operands are minted by the
same `.stellarator.build` node, no producer can be minted for either (refused above), so
it owns nothing and closes nothing. It is a **pure check** — and on `stellarator_helias`
it is the only thing standing between the free `rmajor` and a build that does not close,
which it does through the SQP by pushing `ixc 2`, `3` and `59` (the three that reach it)
until two computed quantities line up. That is the owner's "cannot be satisfied locally"
exactly.

**And `helias_5b`'s `c11` is dead.** `Graph.reach` says **zero** of its three iteration
variables (`4` Te, `6` ne, `10` hfact) reach either operand. It is a constant compared to
a constant: a permanently-fixed row with an all-zero gradient in the SQP. This reproduces
§26.5's inert census independently (`.Constraint11  1/2 operand(s) frozen, 25 in cone`)
and §26.5's own note that *"`helias_5b`'s c11 is one [zero row] and that file converges"*.
So `helias_5b` — a stellarator — has **no working radial-build closure at all**: `c11`
is inert and `c83` is not in its `icc`, while its `Build` node computes `rbld`,
`available_radial_space` and `required_radial_space` and nobody reads any of them. That
is PROCESS's own problem statement, faithfully ported; it is not a port defect.

**Verdict.** Build↔major-radius is **not** a structural identity. It is a *choice of
which variable the radial build determines*, and PROCESS spells that choice in `ixc`,
not in `icc`. `c11` is graph structure on exactly two of the seven files. On three it is
an ordinary specification pinning the build to a fixed machine size. On one it is dead.
On one it is absent and its job is done by a slack inequality. The relation is real; the
*equation* is conditional, and legitimately so.

### 30.6 [measured] The `eq` / `free` split, and the structural rank of each equality block

**An earlier draft of this section asked whether each configuration was "square" and
called `helias_5b`'s zero degrees of freedom an accident. That framing was wrong and is
withdrawn.** Squareness is expected only in evaluation mode. The correct decomposition,
per configuration, is:

- the `eq` equalities **determine `eq` of the `ixc`** — the implicit system, i.e. the
  cycle(s) the graph would close;
- `ixc − eq` is the genuine **design freedom**;
- the inequalities plus the objective are the **actual optimisation**.

| configuration | mode | `ixc` | `eq` (implicit system) | `free` (design) | `ineq` | objective live w.r.t. |
|---|---|---|---|---|---|---|
| `stellarator_helias` | OPT `fom=6` | 8 | 2 `[2, 16]` | **6** | 12 | 8/8 `ixc` |
| `helias_5b` | OPT `fom=7` | 3 | 3 `[2, 11, 16]` | **0** declared | 2 | 3/3 `ixc` |
| `large_tokamak_nof` | OPT `fom=1` | 20 | 3 `[1, 2, 11]` | **17** | 23 | 1/20 (`ixc 3` directly) |
| `large_tokamak_eval` | EVAL `fsolve` | 2 | 2 `[1, 2]` | **0** | 23 | — no objective |
| `low_aspect_ratio_DEMO` | OPT `fom=-14` | 19 | 4 `[1, 2, 11, 30]` | **15** | 21 | 15/19 `ixc` |
| `spherical_tokamak_eval` | EVAL `fsolve` | 3 | 3 `[1, 2, 11]` | **0** | 15 | — no objective |
| `st_regression` | OPT `fom=-5` | 14 | 3 `[1, 2, 11]` | **11** | 15 | **0/14 — inert** (§26.3 rank 1) |

Both evaluation files are exactly square (2/2, 3/3), which is what `scipy.optimize.fsolve`
over the equalities alone requires. **That is a confirmation of the cycle reading, not an
anomaly**: in evaluation mode PROCESS is doing nothing but closing the implicit system,
and the count says so.

*(The `large_tokamak_nof` objective row reads `1/20` because `objective_metric_1` is
`0.2 * rmajor` and `rmajor` **is** `ixc 3` — the objective reads a design variable
directly, so its gradient is the constant `0.2` and it is perfectly live. An earlier
count here reported it as reached by zero `ixc`, which was a defect in the measurement,
not in the file: the liveness rule has to count "is an active `ixc`" as well as "is
downstream of one". `st_regression`'s `0/14` survives that correction and is the genuine
inert objective §26 already refuses on.)*

#### Can the equalities actually be closed? A structural rank, by bipartite matching

The question item 3 should have asked: **for each configuration, is there a choice of
`eq` of its `ixc` such that the equality block is nonsingular** — i.e. a well-posed
`RootFind` the graph could close, leaving `ixc − eq` to the optimiser? The structural
half of that is answerable statically and exactly: build the bipartite graph *equality ×
iteration variable*, with an edge wherever that `ixc` can reach that equality
(`Graph.reach`, plus the direct-read case), and take a **maximum matching**. Its size is
the **structural rank** — an upper bound on the true rank, and a *proof of singularity*
when it falls short (a zero-size matching side cannot be rescued by any numbers).

Measured (`audit_cshape.py match`, via
`networkx.algorithms.bipartite.maximum_matching`). **Only the rank is stable**: the
matching is not canonical and the specific pivot varies between runs — `helias_5b`'s two
live equalities came back as `c2 → ixc 6, c16 → ixc 4` on one run and
`c2 → ixc 4, c16 → ixc 6` on the next. That non-uniqueness is not noise to be suppressed;
it *is* the pivot freedom, and §30.6's closing paragraph is about it.

| configuration | structural rank | verdict |
|---|---|---|
| `stellarator_helias` | **2 / 2** | full — `c2 → ixc 3`, `c16 → ixc 2` |
| **`helias_5b`** | **2 / 3** | **STRUCTURALLY SINGULAR** — `c2 → ixc 6`, `c16 → ixc 4`, **`c11` unmatched** |
| `large_tokamak_nof` | **3 / 3** | full — `c1 → ixc 2`, `c2 → ixc 4`, `c11 → ixc 3` |
| `large_tokamak_eval` | **2 / 2** | full — `c1 → ixc 4`, `c2 → ixc 6` |
| `low_aspect_ratio_DEMO` | **4 / 4** | full — `c1 → 4`, `c2 → 5`, `c11 → 3`, `c30 → 2` |
| `spherical_tokamak_eval` | **3 / 3** | full — `c1 → 4`, `c2 → 6`, `c11 → 29` |
| `st_regression` | **3 / 3** | full — `c1 → 4`, `c2 → 5`, `c11 → 16` |

**So the answer to item 3 is yes on six of seven, and no on `helias_5b`.** Six
configurations have a pivot under which their equalities are a well-posed implicit system
the graph could close as a `RootFind`, handing `ixc − eq` design variables to the
optimiser. That is the structural licence for the whole restructuring; it does not
establish numerical nonsingularity, which needs a Jacobian and therefore a run.

**The pivot is genuinely free, and that is the point.** Every equality but `helias_5b`'s
`c11` has *many* candidate unknowns — `large_tokamak_nof`'s `c2` can be determined by any
of 10 of its 20 `ixc`, `c1` by any of 8, `c11` by any of 5 — and the matching picks one
arbitrarily. PROCESS makes no such choice at all: it hands the whole block to VMCON,
which is a perfectly good way of declining to pivot. A graph rewrite *must* pivot, and
nothing in the input file says how.

#### [measured] `helias_5b`: the coordinator's count holds, and the reading of it does not

Verified independently from the raw file rather than from the assembled problem
(`tests/regression/input_files/helias_5b.IN.DAT`):

```
:12   n_equality_constraints = 3
:16   icc = 2      *Global power balance (consistency equation)
:17   icc = 11     *Radial build (consistency equation)
:18   icc = 16     *Net electric power lower limit
:22   icc = 84     :23  icc = 24
:29   ixc = 4      :33  ixc = 6      :38  ixc = 10
:153  i_process_run_mode = 1     *Code operation switch (1: Optimisation, VMCON only)
:155  i_figure_merit     = 7     *Switch for figure-of-merit (7: Min Capital Cost)
```

**The count is right: an optimisation run with 3 iteration variables and 3 declared
equalities. The conclusion drawn from it is not.** The declared zero degrees of freedom
is not what the file actually poses, because **the equality block's structural rank is 2,
not 3**: `c11` is reachable from *none* of `ixc 4, 6, 10`, so no pivot exists that lets
the three equalities determine three unknowns. The effective problem is **3 unknowns, 2
live equations, 1 degree of design freedom, 2 inequalities, and a live objective** — a
genuine, if very small, optimisation.

**And on the guard question, the answer is the second of the two the coordinator named,
sharpened.** `helias_5b`'s objective is **not** inert in §26's sense. `objective_metric_7`
reads `.costs.cdirt` and `.costs.concost`; both are node-produced
(`.costs.total_plant_direct_cost`, `.costs.constructed_cost`) and both are reached by
**all three** design variables. Its gradient row is formally non-zero in every column.
So `_refuse_inert_objective` correctly does not fire, **and it must not be made to** —
the objective is fine; the defect is one row below it.

The defect is that **`c11` is an equality with an identically zero gradient row**, which
is precisely what §26.5 already records (`.Constraint11  1/2 operand(s) frozen, 25 in
cone`) and deliberately declines to refuse on, with the note *"`helias_5b`'s c11 is one
and that file converges"*. The two facts now join up: that file converges **because** the
dead equality restores the degree of freedom the declared count says it does not have.
Whether VMCON is being handed a satisfiable row or a permanently violated one depends on
the constant's value and is a run, not an assembly (§30.11).

**What no check would catch today.** `refuse_inert_conditions` catches a dead *condition*
one row at a time. It does not compute the equality block's structural rank, so it cannot
say "these three equalities can only ever determine two unknowns" — which is the same
fact stated where it is actionable, and is one bipartite matching over data the walk
already has.

### 30.7 [measured] The `ixc`-with-a-producer question — no defect, but the docstring is stale

`optimise_graph`'s docstring says an `ixc` whose variable *is* produced by a node is a
duplicate-ownership conflict `Graph.__check_init__` refuses, that the "drop the producer
when the `ixc` is active" policy generalises, and that it "is not applied here because no
such ID is active in this run". Checked on all seven:

**Conclusion sound, premise stale, and the policy is already applied — in `indat.py`.**

- **0 of 7 configurations has an active `ixc` that is also node-produced.** No conflict
  exists today, on any file. Nothing is broken.
- But `ixc 3` (`.physics.rmajor`) **is** active on three configurations
  (`stellarator_helias`, `large_tokamak_nof`, `low_aspect_ratio_DEMO`), so the
  docstring's "no such ID is active in this run" no longer describes the seven-file
  world. It is benign because `rmajor` has no producer in *any* of the seven graphs.
- The latent surface is not small: **5 to 13 of the 83 iteration variables are
  node-produced per configuration.** On the stellarators that includes `ID 1`
  (`.physics.aspect` ← `.stellarator.default_aspect_ratio`), `ID 5`
  (`.physics.beta_total_vol_avg`), `ID 13`, `ID 16`, `ID 29`, `ID 60`, `ID 140`; on the
  tokamaks `ID 7`, `ID 12`, `ID 65`, `ID 142` and, per file, `ID 13`/`ID 140`.
  Activating any of them in an `IN.DAT` fails assembly loudly rather than silently, which
  is the right failure — but it is a failure, and no test asserts the seven files avoid
  it.
- **The policy is not unapplied. It is applied once, and keyed on `ixc`** — the
  `dr_tf_inboard_winding_pack` slot in `indat.machine_from_indat` (`indat.py:5055`
  at the measured HEAD, `:5102` after §29 landed — cite the symbol, not the line):

  ```python
  dr_tf_inboard_winding_pack=_slot_occupant(
      "dr_tf_inboard_winding_pack",
      0 if 140 in ixc else 1,
      DR_TF_INBOARD_WINDING_PACK,
  ),
  ```

  This is the reason `ID 13` is produced and `ID 140` free on `large_tokamak_nof` and
  `st_regression`, and `ID 140` produced and `ID 13` free on `low_aspect_ratio_DEMO`,
  `large_tokamak_eval` and `spherical_tokamak_eval` — the pair swaps roles per file. **The
  problem statement already reshapes the model graph**, in exactly one place, silently,
  and `optimise_graph`'s docstring says the opposite. That is a documentation defect worth
  a one-line fix; it is not a behaviour defect.

### 30.8 Cross-configuration inconsistencies — the same statement, differently spelled

Beyond `c11`/`c83`, measured:

1. **`c16` (net electric power) is an equality on both stellarators and an inequality on
   all five tokamaks.** Same body (`geq`), same arguments, same meaning — `"produce at
   least P_net"` on a tokamak, `"produce exactly P_net"` on a stellarator. This is the
   `sand.py` module docstring's standing counterexample generalised: the split is
   positional and user-chosen, and the two device families chose differently.
2. **`c30` (injected-power upper limit) is an equality on `low_aspect_ratio_DEMO` and an
   inequality on the four other tokamaks that carry it.** A `leq` body driven to
   equality. Same shape as `c16`, different constraint, different direction of oddity.
3. **`c1` is a node on the stellarators and an SQP equality on the tokamaks** — §30.3a.
   PROCESS's own code makes this choice, with a comment saying why.
4. **`c24` (beta upper limit) changes class with `ixc 5`.** `DETERMINING` on
   `large_tokamak_nof`, `low_aspect_ratio_DEMO`, `st_regression` (where beta is free);
   `SPEC` on `large_tokamak_eval`, `spherical_tokamak_eval` (beta fixed in the file) and
   on both stellarators (beta node-produced). Three different structural roles for one
   inequality id, and again the discriminator is `ixc`, not `icc`.
5. **`c26`/`c68` do the same thing with `ixc 37` and `ixc 2`/`18`/`3`.**
6. **`c17` (radiation fraction) is a stellarator-and-ST constraint** appearing on
   `stellarator_helias`, `spherical_tokamak_eval` and `st_regression` and nowhere else —
   and on the two STs it reads `.physics.psolradmw`, which has **no producer**, where on
   the stellarator every operand is live. Same id, live on one family, half-frozen on
   another.
7. **`c82`/`c83` are structurally stellarator-only.** `.build.available_radial_space`,
   `.build.required_radial_space` and `.tfcoil.toroidalgap` have no producer in any
   tokamak graph, so those constraints could not be assembled there even if an `IN.DAT`
   asked for them. The stellarator's `Build` node offers *two* spellings of "does the
   build close" — `rbld` and the available/required pair — and the two stellarator files
   pick different ones.
8. **`c56`/`c67` are inert on both STs** and live on `stellarator_helias` (`c67`), for the
   missing-producer reasons §26 already itemises. Nothing new, but it shows up on this
   axis too.

### 30.9 The conditional-activation question, answered per determining constraint

*What does the graph do on a configuration where the relation is not active?*

- **`c11` absent, `rmajor` free (`stellarator_helias`).** The system is **not**
  under-determined: `rmajor` is one of 8 unknowns against 2 equalities and 12
  inequalities, and it is pinned by the objective (`fom = 6`) subject to `c83` and the
  three other inequalities §23(b) finds binding at the solution. The solve answers a
  *different question* — "the cheapest machine whose build fits", rather than "the
  machine whose build sums to `rmajor`". `rbld` is computed and discarded.
- **`c11` absent, `rmajor` fixed (`large_tokamak_eval`).** `rbld` is computed and nothing
  compares it to anything. The radial build is genuinely unchecked on that file. This is
  an evaluation run, so it is defensible — but it means the tracked reference output for
  `large_tokamak_eval` contains an `rbld` no constraint ever looked at.
- **`c1` absent (both stellarators).** Enforced by another route, and the *right* one:
  `.physics.beta_total_vol_avg` is node-produced, so the relation holds by construction
  at every iterate rather than only at convergence.
- **`c24` `SPEC` rather than `DETERMINING`.** Still enforced, just against a fixed beta
  instead of a free one.
- **`c2` is active on all seven**, so the question does not arise — which is itself worth
  noting: the one constraint that owns no unknown anywhere is also the one nobody omits.

### 30.10 Recommendation — diagnostic only, nothing to apply

Reproducing PROCESS comes first and MDF is PROCESS's own formulation; nothing here should
be acted on before the regression suite says the port matches. What this section buys is
a ranked list of what *would* be available later, with the cost of each already measured:

1. **`c1` → a producer node for `.physics.beta_total_vol_avg`.** Cheapest and cleanest:
   the probe says **no SCC forms**, PROCESS's own stellarator path already does exactly
   this, and it removes one equality row and one design variable from the SQP on the
   three files where `ixc 5` is active. Only faithful where `ixc 5` *is* active — on
   `large_tokamak_eval`/`spherical_tokamak_eval` beta is a prescribed input and the
   causation runs the other way.
2. **`c11` → a `RootFind` over `.physics.rmajor`, on the two files where `ixc 3` is
   active.** A 3-node SCC on a 243-node graph. Removes one equality and one design
   variable. **Not** applicable to `helias_5b`/`spherical_tokamak_eval`/`st_regression`,
   where `c11` is a genuine specification against a fixed `rmajor` — and this is the
   point: the rewrite must be per-configuration, because the relation genuinely is.
3. **Leave every pure check where it is.** `c35`, `c46`, `c81`, `c82`, `c83` cannot become
   graph structure; cottax refuses to mint the producer. They belong to the optimiser.
4. **Leave every specification where it is.** ~21 of 35 ids are a computed quantity
   against an input bound. There is nothing to lift.
5. **Add the structural-rank check nobody has.** `boundary.refuse_inert_conditions`
   catches a dead condition one row at a time; it cannot say *"these three equalities can
   only ever determine two unknowns"*. That is one maximum bipartite matching over
   exactly the reachability data that walk already computes (§30.6), it costs no solve,
   and it would have flagged `helias_5b` on the day it was added — as a **rank
   deficiency**, which is where the defect actually is, rather than as a zero-gradient
   row, which is only its symptom. It is also the precondition for every item above: a
   `RootFind` may only be closed over an equality block whose structural rank is full.
   A guard, not a restructuring, and the one thing here that could reasonably land soon.

### 30.11 [live-tree drift] §29 landed three producers while this section was measured, and it closes both INERT rows

Recorded because it would otherwise read as current. Everything above was measured
against `HEAD 6bb65494` as taken in the scratch copy. While it was being measured, the
live tree advanced to `3107bf49` ("Port the three producers a frozen boundary was
standing in for", §29), which lands producers for exactly two of the paths this section
reports as having none:

| path | was, here | is, in `3107bf49` | what moves in §30.2's grid |
|---|---|---|---|
| `.physics.p_plasma_separatrix_rmajor_mw` | frozen | `.tokamak.physics` `PsepOverRMetric` (`physics/exhaust.py`) | **`c56` on both STs: INERT → SPEC** |
| `.constraints.pflux_fw_rad_max_mw` | frozen | `.tokamak.radiated_wall_load` `RadiatedWallLoad` (`fw.py`) | **`c67` on both STs: INERT → SPEC** |
| `.build.r_cp_top` | frozen | `RCpTopFromTfInboardOut` (`build.py`) | nothing — no constraint reads it |

**Predicted, not re-measured**: with a producer, each of those becomes one node-produced
value against one `IN.DAT` bound, which is the `SPEC` shape, and both STs' inert-condition
count should fall to zero (`st_regression` keeping only its inert *objective*,
`.current_drive.big_q_plasma`, which §29 did not touch). The structural ranks in §30.6
should be unaffected — both STs are already full rank with slack — but the
`reached-by-ixc` cones on the two STs will grow slightly, since these producers sit inside
the model chain. Re-running `audit_cshape.py` against the newer tree is a five-minute job
and nobody should trust this paragraph over it.

`.physics.psolradmw` (read by `c17` on both STs) and `.tfcoil.sig_tf_cs_bucked` (read by
`c72` on the three conventional tokamaks) are **still unproduced at `3107bf49`** —
checked. The only `psolradmw` producer in the tree is
`models/stellarator/plasma_physics.py`, which is not on a tokamak's graph.

**A concurrent-edit loss, reported not fixed.** This section was applied to the live tree
as commit `882d3104` and then **removed by `3107bf49`**: both commits appended a new
`##` section at line 5500 of the same file, and the second overwrote the first.
`git show 882d3104:functional_process/_audit/optimise_design.md | grep -c "^## 30\."`
gives `1`; the same at `3107bf49` and at `HEAD` gives `0`. Nothing else from either
commit was lost — `3107bf49`'s own §29 is intact and this section's only footprint was
this file. The corrected text is re-appliable as `section30_on_head.patch` in the scratch
copy; it is deliberately **not** applied here, since editing the live tree is not this
section's to do and two more agents are working in it.

### 30.12 What could not be resolved

- **Whether `c83` sits at its bound at the port's own solution** was answered from §23's
  record, not re-measured. The tracked `stellarator_helias` MFILE/OUT in this tree are
  crash stubs and carry no constraint rows, and a fresh measurement costs a solve.
- **Whether `helias_5b`'s inert `c11` is satisfied or violated at its constant value** is
  a value question and needs a run. Structurally it is a constant; which constant is not
  visible from assembly, and it decides whether VMCON is carrying a harmless row or an
  infeasible one.
- **Numerical, as opposed to structural, nonsingularity of the six full-rank equality
  blocks.** A maximum matching is an upper bound on rank: it proves `helias_5b` singular
  and it *cannot* prove the other six nonsingular. Confirming those needs the equality
  Jacobian at a point, i.e. a run.
- **Which unknown each multi-argument equality should be pivoted onto.** The matching
  shows the choice exists and is wildly non-unique — `large_tokamak_nof`'s `c2` admits
  any of 10 of its 20 `ixc`. The pivot changes bookkeeping and not the answer, so nothing
  static prefers one; a rewrite must choose, and today PROCESS declines to by handing the
  whole block to VMCON.
- **Whether the `unwritten` paths (`beta_beam`, `pden_plasma_ohmic_mw`,
  `beta_thermal_vol_avg`, `beta_toroidal_vol_avg`, `sig_tf_cs_bucked`, `psolradmw`) are
  genuinely inert in PROCESS too** is taken from §22 and §26's census rather than
  re-measured; those measurements were made over the *model* graph and the `mdf` graph
  respectively, and this section's argument does not turn on them.
