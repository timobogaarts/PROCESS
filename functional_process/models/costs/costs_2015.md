---
kind: model-unit
status: draft
confidence: high
---

**See `costs.md` (sibling file, same directory) for the `i_cost_model` topology-switch
finding** — this dispatch's main deliverable, established once for both cost models and
not repeated here. Short version: `.costs.i_cost_model` is resolved in
`process/main.py`'s `Models.costs` `@property`, never read inside either `costs.py` or
`costs_2015.py`, and the two files' write-sets are disjoint except for `.costs.coe`/
`.costs.concost` — a genuine `TOPOLOGY_SWITCHES` candidate, the third real instance after
`isthtr`/`ipowerflow`/`i_plasma_pedestal`.

**Ported: 2 of 13 methods** — `calc_building_costs`, `calc_land_costs`
(`costs_2015.py`/`test_costs_2015.py`, this directory), chosen as the cleanest,
fully self-contained representatives of the file's dominant pattern (11 more `calc_*`
methods share the same shape — a fixed-length loop writing into a slice of a shared
100-element cost-item array — and are audit-only this pass purely for time budget, no
structural blocker beyond the one open design question below, which applies equally to
all 13).

## source

`process/models/costs/costs_2015.py`, 1227 lines, registry unit #18 (shared with
`costs.py`, see `costs.md`). `Costs2015.run()` calls, in order: `calc_building_costs`,
`calc_land_costs`, `calc_tf_coil_costs`, `calc_fwbs_costs`, `calc_remote_handling_costs`,
`calc_n_plant_and_vv_costs`, `calc_energy_conversion_system`,
`calc_remaining_subsystems`, then a final combination (`total_costs`, `concost`, `coe`,
`maintenance`, NaN-diagnostic branch) directly in `run()` itself. `output()` is pure
reporting. `value_function`/`ocost` are small helpers (`value_function` used by
`calc_fwbs_costs`/`calc_remaining_subsystems`; not read this pass).

**Transitive-closure check**: no method calls any other `Model`, `scipy`, or CoolProp
(same grep pattern as `costs.md`'s check, applied to this file — clean). **Every loop in
this file has a compile-time-constant range** (`for i in range(9)`, `range(9, 13)`,
`range(13, 20)`, `range(21, 27)`, `range(27, 31)`, `range(31, 34)`, `range(35, 60)`,
`range(100)` in the NaN-diagnostic branch only) — a structural contrast with `costs.py`'s
`acc2222`, which has the one genuinely dynamic-length loop found across both files.

### method inventory (13 methods)

| method | array indices | ported? | note |
|---|---|---|---|
| `calc_building_costs` | 0-8 | **yes** | 9 items, branch-free |
| `calc_land_costs` | 9-12 | **yes** | 4 items, branch-free |
| `calc_tf_coil_costs` | 13-19 | no | not read this pass — time budget |
| `calc_fwbs_costs` | 21-26 | no | uses `value_function` (separative-work-unit enrichment cost formula); not read this pass |
| `calc_remote_handling_costs` | 27-30 | no | not read this pass |
| `calc_n_plant_and_vv_costs` | 31-33 | no | not read this pass |
| `calc_energy_conversion_system` | — | no | not read this pass |
| `calc_remaining_subsystems` | 35-59 | no | largest remaining method (~370 lines); not read this pass |
| `value_function` | — | no | helper, static/pure by its own signature (`@staticmethod`-shaped, one line docstring seen: "Function for separative work unit calculation") — not read this pass |
| `ocost`/similar helpers | — | no | not read this pass |
| `run` | — | no (orchestration) | the call-order dispatcher + final `total_costs`/`coe`/`concost`/`maintenance` combination — a real tier-3 composition once all 13 `calc_*` are ported (needs every `s_cost` slice), not attempted here |
| `output` | — | no | pure reporting |

## data footprint

Full per-argument tables are in each ported function's own docstring in
`costs_2015.py`. Both functions are entirely `explicit-arg`: every read is a plain
parameter, used once, with no branching and no mid-function re-read. No
`implicit-io`/`redundant-duplicate-write`/`implicit-io-via-callee` found in either.

**Both functions write into *slices* of five shared 100-element arrays**
(`.costs_2015.s_cost`/`s_cref`/`s_k`/`s_kref`/`s_cost_factor`) — `calculate_building_costs`
owns indices `[0:9]`, `calculate_land_costs` owns `[9:13]`; the other 11 (unported)
sibling methods in the source own the remaining indices up to `[35:60]`. This is not an
overlap (each index has exactly one producing method in PROCESS, matching cottax's "one
producer per variable" rule), but it *is* a shape cottax's plain `VarPath`-per-output
model has no established idiom for — see "cottax node" below.

`s_label` (a parallel string array, one label per item, reporting-only — e.g. `"Admin
Buildings"`, `"Land purchasing"`) is **not ported**: not differentiable, not read by any
other computation in either function or (as far as this audit traced) by `run()`'s own
final combination. Same "reporting isn't quite inert but there's nothing to extract"
treatment `hcpb.md`/`coils/output.md` already give string-valued PROCESS fields.

## proposed signature(s)

See `costs_2015.py` — `calculate_building_costs`, `calculate_land_costs`, each returning
`(s_cost_factor, s_cref, s_k, s_kref, s_cost)` as length-9/length-4 `jnp` arrays over
their own index range (not the full 100-slot array — see "cottax node" for why a
node isn't written to bind these into `.costs_2015.s_cost` directly).

## cottax node

**None written for either function.** This is a real open design question, not a
deferred mechanical step (`schema.md`'s own guidance: "skip this section while open
questions about the signature itself are unresolved"). The obstacle: 13 sibling methods
in the source co-own disjoint *slices* of one shared 100-element array field per
quantity (`s_cost`, etc.) — the storage is one PROCESS variable, but the producer is
genuinely one-of-13 depending on which index range is asked about. Two ways to represent
this in cottax, neither clearly right yet:

1. **One `Output` per array element actually touched** (a `SequenceKey`-indexed
   `VarPath`, e.g. `.costs_2015.s_cost[0]` through `[8]`), so `calculate_building_costs`
   declares 45 outputs (5 arrays × 9 items) and `calculate_land_costs` declares 20 (5 × 4).
   Mechanical and matches `naming_convention.md`'s existing array-element convention for
   iteration variables (`f_nd_impurity_electron_array[2]`), but very verbose for what is
   conceptually "this function's contribution to one shared table", and doesn't obviously
   generalise well once all 13 methods are ported (up to ~5×25 = 125 `Output`s on the
   largest, `calc_remaining_subsystems`).
2. **A convention this project doesn't have yet**: something like `Compare`'s
   `place`-based one-node-many-pairs idea (`~/jaxgraph/CLAUDE.md`'s "Rewrites" section),
   but for *ownership* rather than *comparison* — several nodes each claiming a disjoint
   named slice of one logical array-valued variable. Nothing in `~/jaxgraph` today
   models this; would need a real design decision, not a per-unit workaround.

Flagged here as the first concrete instance of this gap (distinct from
`ImpurityRadiationTotals`'s `imp_indices` static kwarg, per unit #20's own docstring,
which solves *which indices exist at all* for one single node — not *several nodes
sharing ownership* of one array). Both functions are fully ported and harness-tested
regardless (see "tier signal") — a node is not a precondition for a Tier 1 contract,
per the existing `superconductors.py`/`impurity_radiation.py` precedent (functions
ported, zero nodes, for an unrelated reason — every real call site is a local inside an
unwired unit there, whereas here the blocker is the array-ownership question itself).

## tier signal

Both: **tier 1**. No internal iteration, no calls into other models, no `scipy`. Ported
as part of finishing this pass's bounded scope.

## switches touched

None found in either function — no `i_*` field read by either `calculate_building_costs`
or `calculate_land_costs`. `.costs.cost_factor_buildings`/`cost_factor_land` are plain
scaling multipliers (default `1.0`), not switches. `.costs.costexp` is a scaling
exponent (default `0.8`), same.

## calls into other models

None. Confirmed by grep and by direct read of both ported bodies.

## JAX-difficulty flags

- **`s_label`'s string values** — `minor`, `workaround-known`: dropped from the port
  entirely (see data footprint). A traced function cannot return a Python string as a
  differentiable leaf, and nothing downstream needs it as anything but a display label.
- **The array-slice-ownership question** (see "cottax node") — not a JAX-tracing
  difficulty in the usual sense (both functions trace and differentiate cleanly under
  `jax.jacfwd`, verified), but a real open modelling-convention gap this project doesn't
  have a name for yet. Flagged with `blocker` severity **for node-writing only** — it
  does not block porting or testing the underlying functions, which is why both are
  fully ported despite it.
- **Fixed-length loop unrolling** (`for i in range(9)`/`range(9, 13)`) — `minor`, no
  workaround needed: both loops have compile-time-constant length, so they are unrolled
  directly into `.at[i].set(...)` calls rather than any `lax.fori_loop`/`lax.scan`
  machinery. Contrast with `costs.py`'s `acc2222` (`costs.md`'s JAX-difficulty flags),
  which has a genuinely dynamic-length loop this file has no equivalent of.
- No CoolProp, no `scipy`, no data-dependent branching anywhere in either ported body.

## open questions

1. **The array-slice-ownership design question** (see "cottax node") — the main open
   item, applies to all 13 sibling methods equally, not just the 2 ported here. Worth
   resolving once, in the consolidation pass or a dedicated design pass, rather than
   re-deriving per method as the remaining 11 are ported.
2. **`run()`'s final combination** (`total_costs`, `concost`, `coe`, `maintenance`, the
   NaN-diagnostic branch) is a real tier-3 composition once all 13 `calc_*` methods are
   ported — it reads specific indices (`s_cost[8, 12, 20, 26, 30, 33, 34, 60]` for
   `total_costs`; a different, larger subset for `maintenance`) from the same shared
   array this record's "cottax node" section already flags as unresolved. Not attempted
   here; noted so whoever finishes the remaining 11 methods sees the end state this is
   building toward.
3. **11 audit-only methods, all time-budget, no blocker found** — `calc_tf_coil_costs`
   through `calc_remaining_subsystems` were not read in this pass. `value_function`
   (used by `calc_fwbs_costs`/`calc_remaining_subsystems`) looks like an ordinary pure
   helper from its docstring alone but was not verified by a direct read.
