# Tokamak scope — what a second device costs, measured

**What this file is.** The size of the conventional-tokamak step, taken from the tree and
from `large_tokamak_eval.IN.DAT` rather than estimated from `unit_registry.md`. Every
number here is regenerable:

```bash
$PY -m functional_process.machine_survey            # defaults to large_tokamak_eval
$PY -m functional_process.boundary                  # the boundary, categorised
```

`next_steps.md` §13.9 sized this by hand at *"~16 genuinely new topology decisions for a
conventional large tokamak"*. Measured independently: **17**. The estimate was right, and
that agreement is the reason to trust the rest of §13.9 rather than re-derive it.

## The 33 integers, classified

| class | n | what it means |
|---|---|---|
| not topology | 6 | `icc`/`ixc`/`n_equality_constraints` are array-parse artefacts or belong to the *study*; `i_process_run_mode`, `output_costs` are run control; `p_fusion_total_max_mw` is a limit value that happens to be integral |
| the factory dispatches on it | 3 | `i_cost_model`, `i_plasma_pedestal`, `ipowerflow` — already slots |
| pinned in the tree | 7 | hardcoded as a static kwarg; **4 of them this file contradicts** |
| new | 17 | the port has never read it; 2 are counts rather than model choices |

**[UPDATED 2026-08-26 — three of the four are closed.]** `i_confinement_time`, `inuclear`
and `i_pulsed_plant` are now factory-dispatched slot families; only `i_p_coolant_pumping`
(5 slots) remains. The survey reads **6 not topology, 6 factory, 4 pinned (1 contradicted),
17 new** — and `unknown` did not move, which is the invariant worth watching: converting a
switch changes the factory/pinned split and never the tokamak's actual model debt.

Two findings from doing it, both structural rather than tidying:

- **`inuclear` removed a driven block.** Its arms read disjoint variables, and the
  "otherwise" arm is PROCESS's own *"if inuclear = 1: qnuc is input"* — so the computed
  arm is an ordinary node reading `p_tf_nuclear_heat_mw`, and the input arm is an **empty
  slot**. The `FixedPointFunction` that existed only because one body both read and owned
  `.fwbs.qnuc` is gone, along with its minted `^cond` copy and its `^guess` start port.
  Driven blocks 14 → 13. What `sand.degenerate_fixed_points` used to recover at runtime by
  differentiating a residual, the tree now states.
- **`i_pulsed_plant` was two dead reads.** At `i_pulsed_plant == 0` Account 225.3 is
  identically zero, so the node declared a `.heat_transport -> .costs` edge and a
  `.costs.fkind` edge that no run makes. The unpulsed occupant reads nothing at all.

And one correction to the paradigm as stated: `istore`'s two ported values read the *same*
variables and differ only in a literal, so they are **one** occupant carrying a static
kwarg — band (c). Splitting them was tried and `test_occupants_of_one_slot_differ` refused
it, correctly: a value that does not change which nodes exist is not a topology choice. The
rule is *no static kwarg where the branches differ in I/O*, not *no static kwarg*.

*Original text follows.*

**The four contradictions are the first deliverable.** `i_confinement_time` (file 34, tree
38), `i_p_coolant_pumping` (3 / 1), `i_pulsed_plant` (1 / 0), `inuclear` (1 / 0). These are
`switch_kwarg_survey.md` band (b) — a node branching internally on a static kwarg while
declaring the union of both arms' reads — and §13.9's point stands with a sharper number:
converting them *is* the first tokamak deliverable, not a prerequisite to it.

## The finding that changes the order

**`i_confinement_time = 34` is already ported.** `ConfinementTimeModel.ITER_IPB98Y2` (34)
is a written, harness-tested scaling in `models/physics/confinement_time.py:809`, and
`ConfinementTime` already takes `i_confinement_time` as an `eqx.field(static=True)`. What
blocks a tokamak here is **not a missing model**: `CONFINEMENT_TIME = {6:
StellaratorConfinementTime}` is keyed on `istell`, and the tokamak arm sits in `UNPORTED`
for a *combination* reason — *"a tokamak scaling law bolted onto stellarator geometry,
coils and FWBS"*. That reason dissolves the moment there is a tokamak device class to put
it in. The refusal is about the absent device, not about absent physics.

Same shape for the other three: each is a pinned answer to a question the tokamak asks
differently, not a model nobody has written.

## What is genuinely absent

The 17 new decisions are the tokamak's own core — plasma current and geometry
(`i_plasma_current`, `i_plasma_geometry`, `i_single_null`, `i_alphaj`,
`i_ind_plasma_internal_norm`), bootstrap and beta (`i_bootstrap_current`,
`i_beta_component`), density and divertor limits (`i_density_limit`, `i_div_heat_load`),
heating (`i_hcd_primary`), PF/CS conductors (`i_pf_superconductor`,
`i_cs_superconductor`, `n_pf_coil_groups`), shield heat (`i_shld_primary_heat`), pulse
timing (`pulsetimings`), availability (`i_plant_availability`), and `n_tf_coils`.

**The balance of plant is already there and is device-agnostic**: `costs` 40, `power` 20,
`availability` 4, `vacuum` 3, `buildings` 2 — 69 of the reference machine's 156 nodes,
about 44% of the graph, inherited. The counterweight is that `costs` will *grow* again:
`model_tree_design.md` §8 step 4c deleted the PF/CS/structure cost nodes because a
stellarator has none, and `cost_boundary_inputs.md`'s category (d) rows carry the
producer `file:line` for every one a tokamak restores.

## Traceability — flagged per switch, not deferred

`next_steps.md` §5 still records the CoolProp policy as flagged and unresolved. The survey
therefore carries a `coolprop` column, and it is deliberately a **weak** signal: it says
*some module reading this switch also reaches CoolProp*, i.e. the neighbourhood is
untraceable, never that this switch's own branch is. On `large_tokamak_eval` exactly one
new decision trips it. The six CoolProp-bound modules are `models/fw.py`,
`engineering/pumping.py`, `stellarator/stellarator.py`, `tfcoil/quench.py`,
`blankets/blanket_library.py`, `blankets/hcpb.py` — measured, not curated.

The policy this waits on is unchanged and is a real fork: a custom JAX primitive wrapping
CoolProp, or these nodes staying outside any differentiated block. Wrapping is the stated
intent; until it exists, a slot blocked this way is *scheduled* differently from one that
is merely unwritten, which is the whole reason the column is here.

## Not built, and why

No `Tokamak` namespace and no `TokamakProcess` class were created in this pass. An empty
device class cannot assemble — `physics.confinement_time`'s registry is keyed on `istell`
and has no tokamak entry — so it would refuse at its first slot and tell you less than
this file does. The scaffold is worth exactly one line of work once the first slot has an
occupant, and not before: this port's own rule is that a graph which looks complete and is
wrong is the failure worth refusing.

## The order this implies

1. Give `physics.confinement_time` a registry keyed on `i_confinement_time` rather than on
   `istell`, with `ITER_IPB98Y2` (34) as the tokamak occupant — band (d)'s rule: an
   occupant per value *this port supports*, everything else `UNPORTED`.
2. The other three contradicted pins, likewise (band (b)).
3. `Tokamak` namespace + `TokamakProcess`, now able to assemble something.
4. `check_boundary` on that machine then enumerates the missing *variables* the same way
   this file enumerates the missing decisions — which is what §13.9 asked for and what
   `boundary.py` now makes possible.
