# The non-finite `^cond.constraints.c32` Jacobian row — traced, classified, closed

---
kind: investigation
status: closed; the defect is fixed at its cause and the fix is measured
confidence: high — every claim below was run in `process_port`, not inferred
---

**The finding.** `_audit/optimise_design.md` §4's "Cost" note records that the SAND
`Optimise` block's `jax.jacfwd` Jacobian (22 conditions × 17 design variables) has 17
non-finite cells, all of them in one row: `^cond.constraints.c32`, 2 `inf` and 15 `nan`,
the whole row. The other 21 rows are fully finite. That section traced it "to the
`EXCLUDED_NODE_NAMES` boundary, not past it" and marked it **diagnosed-but-unconfirmed**.

**The answer, in one line.** The first non-finite value is
`.tfcoil.max_force_density = inf`, produced by `MaxForceDensity`
(`models/stellarator/coils/forces.py:17-36`), because its `a_tf_wp_no_insulation`
input was the harness's `0.0` placeholder. **Classification (b)** — an artifact of the
harness's island exclusion, not a port defect. It is now fixed at its cause:
`mda_harness.EXCLUDED_NODE_NAMES` no longer excludes the coil island, and the c32 row is
fully finite.

---

## 1. Reproduction — the finding is real

Reproduced before touching anything, from the parent session's own script
(`scratchpad/jac_nan.py`, built on `sand_run5.py`), on the graph
`total_process.graph_for(Configuration({".physics.i_plasma_pedestal": 0}))` with the
four `mda_harness.EXCLUDED_NODE_NAMES` nodes dropped (95 nodes), `Residualise`d and
`Combine`d into one `^problem.sand` exactly as `optimise_design.md` §6.3 prescribes:

```
SAND problem: 17 design, 10 eq, 11 ineq
Drive block: 63 nodes, 17 unknowns, 22 conditions, context 154
=== non-finite cells (row=condition, col=design) ===
  ^cond.constraints.c32 x .physics.b_plasma_toroidal_on_axis          = inf
  ^cond.constraints.c32 x .physics.rmajor                             = inf
  ^cond.constraints.c32 x .physics.temp_plasma_electron_vol_avg_kev   = nan
  ... 14 more, all on the same row ...
rows fully finite: 21 / 22
```

Matches `optimise_design.md`'s numbers exactly (17 cells, 2 `inf` + 15 `nan`, one row).

## 2. The first point of non-finiteness — measured node by node, not reasoned about

`scratchpad/probe32.py` walks the SAND block's `ConditionMap` body in topological order,
binding each node's outputs into a running env, and separately `jacfwd`s the composition
of the first `k` nodes for every `k`. Both walks agree on the same node:

```
MDA inputs seeded with the 0.0 placeholder: ['.tfcoil.a_tf_wp_with_insulation',
                                             '.tfcoil.a_tf_wp_no_insulation']

FIRST NON-FINITE VALUE: node ['MaxForceDensity']  .tfcoil.max_force_density = inf
                        node ['MaximumStress']    .tfcoil.sig_tf_wp         = inf
                        node .Constraint32        ^cond.constraints.c32     = inf

--- key winding-pack values ---
  .tfcoil.dr_tf_wp_with_insulation  = 0.7169721730005107
  .tfcoil.a_tf_wp_no_insulation     = 0.0            <-- the placeholder
  .tfcoil.sig_tf_wp_max             = 400000000.0
  .tfcoil.max_force_density         = inf
  .tfcoil.sig_tf_wp                 = inf

FIRST NON-FINITE DERIVATIVE at node #17 ['MaxForceDensity']
  output .tfcoil.max_force_density value=inf
    d/d .physics.b_plasma_toroidal_on_axis = inf
    d/d .physics.rmajor                    = inf
    d/d ... 15 more, matching the c32 row cell for cell ...
```

**The operation.** `functional_process/models/stellarator/coils/forces.py:17-36`,
`calculate_max_force_density`, whose last term is `/ a_tf_wp_no_insulation` (`:34`):

```python
    return (
        stella_config_max_force_density
        * f_st_i_total / f_st_n_coils
        * b_tf_inboard_peak_symmetric / stella_config_wp_bmax
        * stella_config_wp_area
        / a_tf_wp_no_insulation      # <-- forces.py:34, a_tf_wp_no_insulation == 0.0
    )
```

The `inf`/`nan` split is exactly what forward-mode AD produces from a `x/0`: the two
design variables that `max_force_density`'s *numerator* depends on linearly
(`b_plasma_toroidal_on_axis`, `rmajor`, through `b_tf_inboard_peak_symmetric`) give
`finite/0 = inf`; the other fifteen give `0/0`-shaped `nan`. The 17 cells of the c32 row
are the 17 columns of this one node's output derivative, propagated unchanged through
`MaximumStress` (a multiply) and `Constraint32` (a divide by a constant bound).

**Spot-checks of the parent session's two premises, both confirmed:**

- `constraint_32` is a bare `leq(sig_tf_wp, sig_tf_wp_max)`
  (`functional_process/core/solver/constraints.py:1089-1109`) — trivially differentiable.
- `calculate_maximum_stress` is `max_force_density * dr_tf_wp_with_insulation * 1.0e6`
  (`models/stellarator/coils/forces.py:55-61`) — likewise. `dr_tf_wp_with_insulation`
  itself was **fine** (`0.7170`, PROCESS's own value); the parent hypothesis that
  *that* input was the placeholder is wrong. The bad input is the *sibling* winding-pack
  quantity `a_tf_wp_no_insulation`, one node further up.

## 3. Classification: **(b)**, an exclusion artifact — with the evidence that rules out (a)

The task asked specifically to separate "a genuine NaN-gradient trap in a formula that
should be differentiable" from "the input placeholder is garbage". Three independent
pieces of evidence say it is the second:

1. **The value is `inf`, not finite.** The (a) signature is a *finite value with a
   non-finite gradient* — `jnp.sqrt(0)`, a `jnp.where` whose untaken branch is `nan`,
   `abs` at zero. Here `.tfcoil.max_force_density` is itself `inf` before any
   differentiation happens (§2's value walk, which does no `jacfwd` at all).
   `Tier1Contract.test_outputs_finite` (`_harness/contracts.py:236`) exists precisely to
   catch the value case and `test_gradient_finite` (`:254`) the gradient case; this is
   the former, and it would not have fired because the *unit* is fine — the *input* was
   invented by the harness.
2. **There is no guard to add.** `calculate_max_force_density` is a faithful port of
   `process/models/stellarator/coils/forces.py:24`, which divides by the identically
   named explicit argument at exactly the same place (as do `:68` and `:90`, the lateral
   and radial siblings). PROCESS has no zero-guard there either, and none is wanted:
   `a_tf_wp_no_insulation` is a winding-pack cross-sectional area, physically nonzero
   (`0.4284 m²` in this run). Per this repo's standing rule, a defect in PROCESS is
   ported and recorded, not silently fixed — and here there is not even a defect to
   record.
3. **Grounding the input removes the non-finiteness completely**, and does so without
   touching a line of model code (§5). If it were (a), the row would still be non-finite
   at a real operating point.

**Where the `0.0` came from.** `.tfcoil.a_tf_wp_no_insulation` is a genuine mint (a
Python local in PROCESS, `process/models/stellarator/coils/calculate.py:498`, the
source's own comment on `:493` says "not global"), classified **(a)** in
`_audit/next_steps.md` §8.1 row 2. Its producer in this port is
`WindingPackTotalSizePost` (`models/stellarator/coils/calculate.py:1135`) — which
`mda_harness.EXCLUDED_NODE_NAMES` deleted, turning the mint into an unowned boundary
input with no `DataStructure` field, which `compare`/the SAND prototype both seed with
their `0.0` placeholder. So the parent hypothesis ("the `EXCLUDED_NODE_NAMES` coil-island
exclusion feeding placeholder values into the winding-pack chain") is **confirmed** —
just one node further along the chain than guessed.

## 4. Can `EXCLUDED_NODE_NAMES` be closed? **Yes, and it is now closed.**

`next_steps.md` §8 gave the exclusion two justifications. One is false and one is real:

| justification | verdict |
|---|---|
| "every `VarPath` the island touches is minted, PROCESS never stores them, so there is nothing to compare against" | **false for this island.** `WindingPackTotalSizePost` declares 13 `Output`s (`coils/calculate.py:1122-1136`); **11 are real `DataStructure` fields** (`b_tf_inboard_peak_symmetric`, `dx_tf_wp_primary_toroidal`, `dx_tf_wp_secondary_toroidal`, `dr_tf_wp_with_insulation`, `j_tf_wp`, `n_tf_coil_turns`, `c_tf_turn`, `a_tf_wp_conductor`, `a_tf_wp_extra_void`, `a_tf_coil_wp_turn_insulation`, `a_tf_wp_steel`) and only 2 are mints. That sentence is true of `DuctDiameterRootFind` (whose exclusion stands) and was carried over to this island by analogy. |
| "`.stellarator.wp_width_r_min` seeded at `0.0` makes `NewtonDriver` fail and abort the whole schedule" | **true, and fixed at its cause.** |

### 4.1 The three groundings

All three are reconstructions from real, stored fields — the same mechanism
`KNOWN_MINT_VALUES` already uses for `.stellarator.coilcurrent`.

**`.stellarator.wp_width_r_min` ← `.tfcoil.dr_tf_wp_with_insulation`.** This one is new
here; §8.1 did not have it. PROCESS's `winding_pack_total_size` computes the crossing
point (`process/models/stellarator/coils/calculate.py:462`), clamps it
(`:465`, `wp_width_r_min = max(dx_tf_turn_general**2, wp_width_r_min)`), and then writes
`awp_rad = wp_width_r_min` **straight into the real field**
`data.tfcoil.dr_tf_wp_with_insulation` (`:481`, `:489`). So the stored field *is* this
unknown, after a clamp that is inactive here by a factor of 228:

```
dx_tf_turn_general**2 = 3.136e-03      dr_tf_wp_with_insulation = 7.170e-01
```

It is in any case only a `RootFind`'s **starting guess** — `Intersect` re-solves its own
residual from it. With the guess in place `NewtonDriver`'s `optx.root_find` converges and
the schedule runs; the solved answer is then compared back against the same field like
any other output and **agrees within `compare`'s `rtol=1e-6`** (verified by direct set
membership, not by absence from the disagreement list). That is the first
PROCESS-comparable number `Intersect` has ever had — its `Tier2Contract`
(`coils/test_coils.py:511`) has no value-agreement test by construction.

**`.tfcoil.a_tf_wp_no_insulation` and `.tfcoil.a_tf_wp_with_insulation`** — the two
reconstructions `next_steps.md` §8.1's "two harness-side follow-ups" section already
derived but flagged as "the one that most needs a second opinion first". They are read
straight off `calculate.py:494`/`:498`, whose right-hand sides are the three real fields
assigned immediately above at `:483-491`:

```
a_tf_wp_no_insulation   = .tfcoil.dx_tf_wp_primary_toroidal * .tfcoil.dr_tf_wp_with_insulation
a_tf_wp_with_insulation = (dr_tf_wp_with_insulation + 2*dx_tf_wp_insulation)
                          * (dx_tf_wp_primary_toroidal + 2*dx_tf_wp_insulation)
```

**Second opinion supplied, numerically.** PROCESS computes
`data.tfcoil.j_tf_wp = coilcurrent * 1e6 / a_tf_wp_no_insulation` (`calculate.py:499`)
from the local. Feeding the *reconstruction* into that same identity, with `coilcurrent`
taken from its own established inverse (`c_tf_total / (n_tf_coils * 1e6)`), reproduces
PROCESS's stored `j_tf_wp` to the last printed digit:

```
reconstruction -> 30620392.270788945      data.tfcoil.j_tf_wp = 30620392.270788945
```

and the geometric consistency check `dr / dx_tor == stella_config_wp_ratio` returns
exactly `1.2`. Neither reconstruction is circular *as a comparison*: the right-hand sides
are PROCESS's own stored numbers, so scoring the port's `a_tf_wp_*` against them tests
whether the port's resolved `wp_width_r_min` matches PROCESS's — which is exactly what
`Intersect` is on the hook for.

### 4.2 What changed

`functional_process/mda_harness.py`, and nothing else:

- `EXCLUDED_NODE_NAMES` drops from four names to one (`DuctDiameterRootFind`), with the
  old rationale corrected rather than deleted.
- `KNOWN_MINT_VALUES` gains the three entries above, each with its `file:line` derivation.

No model code was touched. No change is needed in `total_process.py`,
`next_steps.md`, `unit_registry.md` or `models/costs/**`. Audit records updated:
`models/stellarator/coils/coils.md` (the `wp_width_r_min` grounding and what it buys
`Intersect`), `models/stellarator/coils/calculate.md` (the two `a_tf_wp_*`
reconstructions are now live), `models/stellarator/coils/forces.md` (the diagnostic note
on the `/ a_tf_wp_no_insulation` division, explicitly recorded as *not* a defect).

## 5. Before / after — every number measured

### 5.1 The Jacobian (the question the investigation was opened on)

Same SAND assembly, same reference point, only the exclusion set and the three mint
seeds differ (`scratchpad/jac_nan.py` vs `jac_nan2.py`):

| | before | after |
|---|---|---|
| nodes after exclusions | 95 | 99 |
| SAND `Optimise` | 17 design, 10 eq, 11 ineq | **18 design, 11 eq, 11 ineq** |
| `Drive` block | 63 nodes, 17 unknowns, 22 conditions, 154 context | 69 nodes, 18 unknowns, 23 conditions, 160 context |
| **non-finite Jacobian cells** | **17 (all of the c32 row)** | **0** |
| rows fully finite | 21 / 22 | **23 / 23** |

The extra design variable and equality are `.stellarator.wp_width_r_min` and
`^cond.stellarator.wp_width_r_min` — `Intersect`'s `RootFind`, now inside the SAND
problem where it belongs.

### 5.2 Stage A (`optimise_design.md` §5.1) improves too, unasked

The same run prints every condition against PROCESS's own
`-normalised_residual`. **c32 goes from `+inf` to exact:**

```
        port                 PROCESS
c32   -6.884017671e-01    -6.884017671e-01     exact   (was +inf)
c8    -4.867612733e-01    -4.867612733e-01     exact
c17   -5.101638572e-01    -5.101638572e-01     exact
c18   -6.818612747e-01    -6.818612747e-01     exact
c67   -5.260038620e-01    -5.260038620e-01     exact
c82   -5.662976676e-01    -5.662976676e-01     exact
c83   -2.075735805e-09    -2.075735805e-09     exact
c34   -8.602314432e-01    -8.602314432e-01     exact
c35   -1.936916515e-07    -1.936916517e-07     exact
c65   -9.758967829e-01    -9.758967829e-01     exact
c2    +9.035891604e-03    -7.375413613e-10     off  (tracked: ConfinementTime)
c62   -1.056535051e-01    -1.191345646e-01     off  (tracked: fusrat, ~11 %)
```

**Stage A is now 10 / 12 exact, not 9 / 12**, and the underlying quantity agrees to
round-off: `.tfcoil.sig_tf_wp = 124639293.14001687` against PROCESS's
`con_value = 124639293.14001684`.

`optimise_design.md` §5.1's own footnote blamed c32's `inf` on "the prototype's crude
argument resolution reading `sig_tf_wp_max = 0.0`". That is **not** the cause:
`sig_tf_wp_max` resolves correctly to `4.0e8` here (§2's value dump). The cause is
`a_tf_wp_no_insulation`, i.e. the same exclusion §5.2 blamed for the zeroed x2/x59
columns. One cause, not two.

### 5.3 The MDA harness

Measured two ways. **(i) A controlled A/B in one process**
(`scratchpad/close_island_sets.py`), toggling only `EXCLUDED_NODE_NAMES` /
`KNOWN_MINT_VALUES` against an identical tree — necessary because another agent is
concurrently editing `total_process.py` and `models/costs/**`, so a plain wall-clock
before/after would attribute their changes to this one:

| | before | after |
|---|---|---|
| agreements | 237 | **296** |
| disagreements | 2 | **2** (the same two `VacuumOld` ones, already explained in `EXPLAINED_DISAGREEMENTS`) |
| unverifiable | 65 | **3** |
| ungrounded inputs | 2 | **0** |
| errors | 11 | 21 |
| switch audit | 34 checked, 0 mismatched | unchanged |

**(ii) The real entry point**, `$PY functional_process/run_mda_harness.py`, reproduces
the same five columns exactly: `296 / 2 / 3 / 0 / 21`. (Its "before" was independently
reproduced at `237 / 2 / 65 / 2 / 11`, matching the stated baseline.)

**70 `VarPath`s are newly compared and 0 stop being compared.** 59 of them are scalars
and land in `agreements`; the remaining 11 are the array-valued `.power.*_profile_mw`
family, which `compare` skips by design (its `float(np.asarray(...))` guard). The newly
scored set is the whole coil/TF/buildings/structure tail that the island fed:
`.tfcoil.a_tf_wp_no_insulation`, `.tfcoil.j_tf_wp`, `.tfcoil.sig_tf_wp`,
`.tfcoil.max_force_density`, `.tfcoil.m_tf_coils_total` and 12 further `.tfcoil.m_*`/
`a_*` masses and areas, `.stellarator.wp_width_r_min`, `.superconducting_tfcoil.vv_stress_quench`,
`.structure.coldmass`/`aintmass`/`clgsmass`, 13 `.buildings.*` volumes,
`.heat_transport.pacpmw`/`tlvpmw`/`p_tf_electric_supplies_mw`, `.rebco.coppera_m2`.
**All 59 agree first time** — no new disagreement anywhere.

**`unverifiable` collapses from 65 to 3.** The residue is exactly the three entries in
`KNOWN_UNVERIFIABLE_OUTPUTS` (`.physics.fusrat`, `.fwbs.f_a_fw_coolant_inboard`,
`.fwbs.f_a_fw_coolant_outboard`), all already documented. §8.1's prediction that
grounding rows 1/2 "would additionally unblock `CoilsMass`/`MaxForceDensity` and their
descendants out of the `unverifiable` column, which is where most of the 65 sit" is
confirmed in full.

### 5.4 The ten new `errors`, triaged individually against `process/`

`errors` is the *opposite side of the same edge* from `ungrounded_inputs`: an owned
output whose `VarPath` has no `DataStructure` field. These ten were previously hidden
inside `unverifiable`; making the island run exposes them. Each was checked against
`process/` separately, using §8.1's classification key. **All ten are (a) — correct
mints, structurally uncheckable.** None is a regression.

| `VarPath` | class | evidence |
|---|---|---|
| `.stellarator.wp_width_r`, `.stellarator.lhs`, `.stellarator.rhs` | (a) | the two sampled curves and their abscissa, locals in `winding_pack_total_size` (`process/models/stellarator/coils/calculate.py:389-450`); `Intersect`'s own docstring (`coils.py:521-545`) and `coils.md` already record all three as mints |
| `^cond.stellarator.wp_width_r_min` | (a) | `Intersect`'s **residual**, minted by `ImplicitFunction`. See the structural note below |
| `.buildings.tfro`, `.tfri`, `.tf_radial_dim`, `.tf_vertical_dim`, `.tfmtn` | (a) | five plain locals in `Buildings.run` (`process/models/buildings.py:41,45,48,52,57`), passed on to `bldgs_sizes`/the report and never stored; grepping `process/data_structure/` for each name returns nothing |
| `.superconducting_tfcoil.f_vv_actual` | (a) | a local returned bare from `calculate_quench_protection` (`process/models/stellarator/coils/quench.py:48,91`) into a local at `calculate.py:118`, forwarded as a kwarg at `:150`; no field of that name in `process/data_structure/` |

**One structural note worth recording, found by this triage.** `_ground_truth`'s
`unminted` fallback (`mda_harness.py`, rule 2) maps `^cond.X` back to `X`. That is
correct for a `FixedPointFunction`, whose `^cond.X` is `g(X)` and therefore equals `X` at
the fixed point — which is how the eight structural self-loops are scored today. It is
**not** correct for a `RootFind`/`ImplicitFunction`, whose `^cond.X` is a *residual* and
should be ≈ 0. It causes no false result today only because both registered `RootFind`s
(`Intersect`, `DuctDiameterRootFind`) have minted unknowns, so the lookup errors out
instead of comparing. **A future `RootFind` whose unknown is a real `DataStructure` field
would be reported as a large spurious disagreement.** Not fixed here (it needs a
problem-type-aware branch in `_ground_truth` and there is nothing to test it against
yet); recorded so it is not rediscovered as a mystery.

### 5.5 Suite

`$PY -m pytest functional_process -q` → **3567 passed, 2909 skipped**. The baseline for
this session was 3420 passed / 2783 skipped; the increase is the **concurrent costs
port**, not this change — nothing under test imports either symbol this change touches
(`grep -rn mda_harness functional_process`: the only importers are
`mda_constraint_harness.py`, of `converged_data` alone, and `run_mda_harness.py`, which
is a script, not a test). No test regressed.

`ruff check functional_process/mda_harness.py` reports 13 findings and `ruff format` is
clean. All 13 are pre-existing: 10 are present at `HEAD` (checked by running `ruff` on
`git show HEAD:functional_process/mda_harness.py`) and the other 3 (`ISC004` ×1,
`PERF401` ×2) come from this session's earlier, non-mine edits to the same file — none
of the three rules can be triggered by comments or by a dict of lambdas, which is all
this change adds. The one violation this change *did* introduce (an `E501`) was fixed
before landing.

## 6. What this unblocks, and what is still open

`optimise_design.md` §5.2 concluded that the exclusion "does not merely zero some
columns, it makes one whole constraint row non-differentiable, which no SQP can accept.
Closing the exclusion is a prerequisite for the `Optimise` layer, not an optimisation of
it." **That prerequisite is met.** Concretely:

- the c32 row is finite, so `VmconDriver` (§4) has a Jacobian it can use;
- the x2 / x59 columns of c82/c83/c32/c35, which §5.2 measured as spuriously zero
  because the island was cut out, now have a live path — **their values against
  `fcnvmc2` are not yet checked** (that is Stage B, §5.2/stage 5, not this task);
- `Intersect`'s `RootFind` joins the SAND problem as an 18th design variable and an
  11th equality, which is the structurally honest shape.

Still open, deliberately not attempted here:

1. **The x4 (`temp_plasma_electron_vol_avg_kev`) column** — `optimise_design.md` §9
   open question 1, untouched by this work and unrelated to it.
2. **Stage B per-cell Jacobian comparison** with the island in. Now worth re-running:
   §5.2's three "structured disagreements" were measured with the island *out*, so its
   second bullet (zeroed x2/x59) is expected to change and should be re-measured before
   it is cited again.
3. **`^cond.*` for a `RootFind` in `_ground_truth`** — §5.4's structural note.
4. **Six 1-tuple returns in `power_B_thermal_cryo.py`** (`optimise_design.md` §6.4a) are
   still worked around, not fixed, in every SAND script here; that file is outside this
   task's ownership.

### Precise edits `_audit/optimise_design.md` needs (not made — file not owned)

- **§5.1, the Stage A table (`:521-535`)**: `c32` reads `+inf … prototype artefact`.
  Replace with `c32   -6.884017671e-01   -6.884017671e-01   exact`, and change
  "Nine of twelve agree to printed precision" (`:537`) to "Ten of twelve". Delete the
  sentence beginning "`c32`'s `inf` is the prototype's own crude argument resolution
  reading `sig_tf_wp_max = 0.0`" (`:541-544`) — that diagnosis is wrong; `sig_tf_wp_max`
  resolves correctly to `4.0e8`. Replace it with a pointer to this file.
- **§4 "Cost", the non-finite-row paragraph (`:650-661`)**: the row is no longer
  non-finite. Keep the paragraph as history but mark it **closed**, replace
  "diagnosed-but-unconfirmed" with the confirmed cause
  (`MaxForceDensity`'s `/ a_tf_wp_no_insulation`, `coils/forces.py:35`, fed the `0.0`
  placeholder) and record `17 → 0` non-finite cells.
- **§6.3, the SAND table (`:794-803`)**: with the exclusion closed the numbers become
  99 nodes / 18 design / 11 eq / 11 ineq / a 69-node `Drive` with 18 unknowns, 23
  conditions and 160 context vars. The "(with the four island nodes … dropped, 95 nodes)"
  parenthetical (`:791`) should become "(with `DuctDiameterRootFind` dropped, 99 nodes)".
- **§5.2, second bullet (`:603-607`)**: "columns x2 and x59 are zero for c82/c83/c32/c35
  … because the prototype excluded the coil island" — the cause is removed; the bullet
  should be re-measured rather than restated.
