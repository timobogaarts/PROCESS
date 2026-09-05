# Replacing integer switches with model selection — feasibility and blast radius

**Superseded in design by `model_tree_design.md`**: one typed `eqx.Module` tree of models
carrying their own settings, a single `machine_from_indat` factory as the only place
integers live, no `Switch`/`Alternative`/`Configuration`/settings-tree/`materialise`.
Kept here only for measurements not restated elsewhere, cited by `total_process.py`,
`render_xdsm.py`, and other audit records.

## §1 — It works in principle

`grep` over `iteration_variables.py` and `scan.py` for `i_*`/`istell` finds nothing: no
switch is ever an iteration variable or a scan variable, so no switch can change between
two evaluations of one assembled graph. A graph is a pure function of a static selection
made once — exactly the condition under which a declarative, JSON-serialisable selection
document can fully determine it.

## §3 — "Switch" is four different things (cited directly by code, keep this table)

| kind | example | becomes a model choice? |
|---|---|---|
| (a) model selection | `i_confinement_time`, `i_cost_model`, `i_tf_sup` | **Yes** — the actual target |
| (b) shape/resolution | `n_plasma_profile_elements=201` | No — must stay a concrete int under `jit` |
| (c) set membership | `imp_indices` (which impurity species exist) | No — a set, not a choice |
| (d) alias/noise | `is_ignited` (bool restatement of `i_plasma_ignited==1`) | Delete, don't rename (done) |

A count of "switches remaining" that includes (b)/(c)/(d) can never reach zero; only (a) is
in scope.

## §4 — Vocabulary already exists upstream

51 `IntEnum` classes exist across `process/` (`process/data_structure/*_variables.py` and
`process/models/**`, not `summary.py` as an earlier draft wrongly guessed), covering most
switches. 12 of the port's converted switch families reused an upstream enum directly; the
other 16 got a local one in `functional_process/models/switch_enums.py`. Promote, don't
invent.

## §8 — The one thing not to break: `switch_audit`

It compares each registered integer kwarg against the same field's value on a converged
`DataStructure`, and it has caught **five** wrong-default bugs this way
(`i_confinement_time` 34/38, `i_thermal_electric_conversion` 0/2, `i_p_coolant_pumping`
2/1, `i_plasma_ignited` 0/1, `i_cost_model` 1/0). Any move to name-typed selection needs an
enum-aware `switch_audit` landed **before** the registrations change, not after — losing
this check in the act of acting on it is the one way this refactor could make things worse.

## §11 — Provenance grouping vs. derived structure, measured

Grouped the (then) 143-node graph two ways and compared against `Blocking.scc`:

| grouping grain | SCCs with >1 node | how many cross a boundary |
|---|---|---|
| subsystem (`physics`, `stellarator`, `costs`, ...) | 3 | **0** |
| source file | 3 | **3** |

Every genuine cycle is contained within one subsystem and spans several files inside it
(e.g. `DensityProfile`/`FusionRates`/`PlasmaComposition` — three physics files;
`Divertor`/`AFwTotalWithPowerflow` — `stellarator.divertor` + `stellarator.build`).
**Subsystem is the right grain for a model group; file is not** — this is what fixed the
model tree's grain at "subsystem, two levels" in `model_tree_design.md`. Two caveats:
provenance here is PROCESS's own file layout, not an independently-declared physical
grouping; and the zero cross-subsystem SCCs may be an artifact of scope — more are expected
once the (still unported) orchestration layer lands, per `CLAUDE.md`'s note on plasma/TF/
build feedback.

## §13 — Feasibility investigation, key numbers

- The two blocking cottax changes turned out to be **one edit in one file**
  (`pytree_namespace_module.py`: `named(at)` + `node_and_names(xs, at=None)`), prototyped at
  **+108/−19 lines**, both suites unchanged, graph rebuilding to the same nodes and binding
  order. Landed since (`~/jaxgraph` `789df8b`).
- Hierarchical `NodePath`s already worked end to end before any declaration-surface change:
  two 3-key minted names (`^problem.physics.proton_rate_density`,
  `^problem.fwbs.f_ster_div_single`) existed in the driven graph already.
- `ComponentThermalPowers` is ~5 fused models in one node, not one model with 5 settings:
  measured 21 outputs / 31 inputs under five switches, with outputs clustering visibly by
  switch (e.g. `p_*_coolant_pump_elec_mw` with `i_p_coolant_pumping`). Splitting it into
  roughly five nodes (one model choice each) is still open — tracked in
  `model_tree_design.md`'s fused-node-splitting item, not done here.
- Blast radius at investigation time: 80 `COMMON` entries (58 bare classes, 22 instances),
  147 declarations, 9 topology switches, 60 static-field slots across 33 declarations (58
  of them `int`-typed). Re-measured after `model_tree_design.md` §8 step 1 landed: 64 slots
  / 36 declarations / 32 distinct names — 56 enum-typed, 6 genuinely shape/count (kind b), 2
  payloads, **0 bare-integer model switches left**, pinned by
  `switch_audit.not_enum_typed`. With almost every slot an `int` originally, a derived
  settings schema mainly buys catching *misspelled keys*, not wrong types.

## Where this stands now

The cottax half landed; `total_process.COMMON` conversion is client work tracked by
`model_tree_design.md`, not by this file. The design content this file originally worked
through (§§2, 5–7, 9–10, 12, 14 in the pre-cut version) is superseded there; the one design
conclusion from it that is not restated elsewhere — a switch arm cannot be placed as one
`place` per `Switch`, because a single arm can populate two unrelated subtrees at once — is
recorded in `path_refactor.md` instead of here.
