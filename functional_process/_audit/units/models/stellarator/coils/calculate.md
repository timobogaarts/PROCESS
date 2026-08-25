---
kind: model-unit
status: reviewed
confidence: high
---

**Ported — 12/12.** `functional_process/models/stellarator/coils/calculate.py` /
`test_calculate.py`. The first pass ported 10 of the 12 functions in this file (tier-1,
`ExplicitFunction` nodes, fuzz-only tests). This pass ports the remaining two:
**`winding_pack_total_size`** (tier-2, `Tier2Contract`, self-contained now that
`intersect`/`bmax_from_awp` are real ported functions in `coils/coils.py`) and
**`st_coil`** (tier-3, a plain composed function — no `cottax` node of its own, see
below — now that units #11/#12/#14 (`forces.py`/`mass.py`/`quench.py`) are also
ported and #13 (`output.py`) is confirmed pure reporting). Both were previously blocked
on registry units #10-14; that block is now resolved except for one narrow remainder
(`jcrit_from_material`'s dispatch itself, still unported in `coils.py` — see "switches
touched" below), which this unit works around with its own local, scoped restatement
rather than waiting on it.

Two real PROCESS bugs, and one real port bug, were found while porting this pass — see
"real PROCESS bugs found" below.

**Later pass: `winding_pack_total_size` split around `intersect`.** The `WindingPackTotalSize`
node this section originally described (below, now removed) called `coils.py`'s
`intersect` eagerly, in the middle of its own `__call__` body. It is replaced by three
nodes — `WindingPackIntersectInputs` (pre-`intersect`), `coils.py`'s `Intersect` (the
`intersect` root-find itself, now a genuine `ImplicitFunction`/`RootFind` pair), and
`WindingPackTotalSizePost` (post-`intersect`) — composing around `coils.py`'s `Intersect`
node instead of calling the plain function directly. The pure function
`winding_pack_total_size` itself is unchanged in behaviour (same signature, same return
values, still calls `intersect` eagerly internally) — only reorganised into
`winding_pack_pre_intersect`/`winding_pack_post_intersect` halves so both the plain
function and the node-graph split share one definition of "what happens before/after the
root-find" rather than two. See "cottax node" below for the full design, and
`_audit/next_steps.md` §7 / `coils.md`'s `Intersect` docstring for why this split is
worth doing (a first-class, swappable solver choice) even though nothing else in the
graph needs `intersect`'s internal unknowns visible (§7's own test for that, unchanged).
`WindingPackJTfWp`'s own `.tfcoil.j_tf_wp` `FixedPoint` split (this section's own
original subject) is untouched by this — it still calls the whole
`winding_pack_total_size` pure function directly, not these node classes.

## source
`process/models/stellarator/coils/calculate.py` (593 lines, full file in scope).
Registry unit #9 — contains `st_coil`, called directly from `Stellarator.run()`
(`stellarator.py:144,160`). One orchestrator (`st_coil`) plus 12 module-level helper
functions, all already taking a `DataStructure` and (mostly) some explicit args, in the
same partially-pure shape as `density_limits.py`.

## data footprint

`st_coil` itself does not compute much directly — it is almost entirely a sequence of
calls to the helpers below, plus ~15 lines of inline geometry (`z_tf_inside_half`,
`len_tf_coil`, `tfcryoarea`, `min_bending_radius`) that read further `data.*` fields not
listed per-helper below. Per-helper footprints (all `explicit-arg` once the `data`
parameter is dropped, since every helper already takes its real inputs positionally or
reads a small, fixed set of `data.*` fields with no branching):

| Function | reads | writes |
|---|---|---|
| `calculate_coil_toroidal_thickness` | `.tfcoil.dx_tf_wp_primary_toroidal`, `.dx_tf_side_case_min`, `.dx_tf_wp_insulation` | `.tfcoil.dx_tf_inboard_out_toroidal` |
| `calculate_coil_radial_thickness` | `.tfcoil.dr_tf_nose_case`, `.dr_tf_wp_with_insulation`, `.dr_tf_plasma_case`, `.dx_tf_wp_insulation` | `.build.dr_tf_inboard` **and** `.build.dr_tf_outboard` (same value — see note) |
| `calculate_coil_cross_sectional_area` | `a_tf_wp_with_insulation` (explicit arg), `.build.dr_tf_inboard`, `.tfcoil.dx_tf_inboard_out_toroidal` | `.tfcoil.a_tf_leg_outboard`, `.a_tf_coil_inboard_case` |
| `calculate_coil_half_widths` | `.tfcoil.dx_tf_inboard_out_toroidal` | `.tfcoil.tfocrn`, `.tficrn` (same value — see note) |
| `calculate_plasma_facing_coil_area` | `.tfcoil.n_tf_coils`, `.dx_tf_inboard_out_toroidal`, `.len_tf_coil` | `.tfcoil.tfsai`, `.tfsao` (same value — see note) |
| `calculate_coil_coil_toroidal_gap` | `r_coil_major`, `r_coil_minor` (explicit args), `.stellarator_config.stella_config_dmin`, `.stella_config_coil_rmajor`, `.stella_config_coil_rminor`, `.tfcoil.dx_tf_inboard_out_toroidal` | `.tfcoil.toroidalgap`; `coilcoilgap` is return-only, never stored (see note) |
| `calculate_coils_summary_variables` | `coilcurrent`, `r_coil_major`, `r_coil_minor`, `awp_rad` (explicit args), `.tfcoil.n_tf_coils`, `.a_tf_leg_outboard` | `.tfcoil.a_tf_inboard_total`, `.c_tf_total`, `.j_tf_coil_full_area`, `.r_b_tf_inboard_peak_symmetric` |
| `calculate_inductance` | `r_coil_minor` (explicit arg), `.stellarator_config.stella_config_inductance`, `.stellarator.f_st_rmajor`, `.stellarator_config.stella_config_coil_rminor`, `.stellarator.f_st_n_coils` | none — reporting-only, see note |
| `calculate_stored_magnetic_energy` | `r_coil_minor` (explicit arg), same 4 fields as `calculate_inductance`, plus `.tfcoil.c_tf_total`, `.n_tf_coils` | `.tfcoil.e_tf_magnetic_stored_total_gj` |
| `calculate_winding_pack_geometry` | `.tfcoil.dx_tf_turn_general`, `.dx_tf_turn_steel`, `.dx_tf_turn_insulation` | `.tfcoil.a_tf_turn_cable_space_no_void`, `.a_tf_turn_steel` |
| `calculate_current` | `.stellarator.f_st_b`, `.stellarator_config.stella_config_i0`, `.stellarator.f_st_rmajor`, `.f_st_n_coils` | `.stellarator.f_st_i_total` (write); `coilcurrent` is return-only, never stored (see note) |
| `calculate_casing` | `.tfcoil.dr_tf_nose_case` | `.tfcoil.dr_tf_plasma_case`, `.dx_tf_side_case_min` (same value — see note) |
| `calculate_vertical_ports` | `.stellarator_config.stella_config_max_portsize_width`, `.stellarator.f_st_rmajor`, `.f_st_n_coils` | `.stellarator.vporttmax`, `.vportpmax`, `.vportamax` |
| `calculate_horizontal_ports` | same 3 reads as `calculate_vertical_ports` | `.stellarator.hporttmax`, `.hportpmax`, `.hportamax` |

**Note — four functions write the identical value to two distinct fields**
(`calculate_coil_radial_thickness`: `dr_tf_inboard`==`dr_tf_outboard`;
`calculate_coil_half_widths`: `tfocrn`==`tficrn`; `calculate_plasma_facing_coil_area`:
`tfsai`==`tfsao`; `calculate_casing`: `dr_tf_plasma_case`==`dx_tf_side_case_min`). Not a
`redundant-duplicate-write` in this audit's sense (that label is for the *same* field
written twice) — these are two *different* fields that happen to share one formula, per
each docstring's own "same as inboard" / "assumed constant" comment. Ported as two
return values each, not collapsed to one, since downstream readers address them by
their separate names.

**`coilcurrent` (from `calculate_current`) and `coilcoilgap` (from
`calculate_coil_coil_toroidal_gap`) are locals in `st_coil`, never written to `data`.**
`coilcoilgap` only ever reaches `write()` (reporting) — dropped from the node, same
treatment as `calculate_inductance`'s `msupstr`-style reporting-only output.
`coilcurrent` is different: it also feeds `calculate_coils_summary_variables` (in scope,
ported) and `winding_pack_total_size` (now also ported, see below) — a real graph edge,
not reporting. Minted `.stellarator.coilcurrent` for it (see the port's `CoilCurrent`
node docstring) — **open question 1 below (from the first pass) is now resolved**:
`WindingPackIntersectInputs`'s and `WindingPackTotalSizePost`'s own `coilcurrent`
`From`s both read this exact `VarPath` (the pre-/post-`intersect` split, a later pass —
see the top-of-file note — kept both halves reading it, since `winding_pack_curves` and
`winding_pack_post_intersect` each need it independently), so four nodes (`CoilCurrent`,
`CoilsSummaryVariables`, `WindingPackIntersectInputs`, `WindingPackTotalSizePost`) now
genuinely share it, confirming it was the right call.

Re-verified in the MDA triage (`_audit/next_steps.md` §8.1) that the mint is still
correct: `coilcurrent` is assigned as a local at `calculate.py:46` from
`calculate_current`'s bare `return coilcurrent` (`calculate.py:378`), and nothing in
`process/` stores it (grepped: every hit is a parameter name or a local). **It is,
however, exactly recoverable from a real field** — `calculate.py:276` writes
`data.tfcoil.c_tf_total = data.tfcoil.n_tf_coils * coilcurrent * 1.0e6`, and this port's
own `quench.py:201` already inverts precisely that (`coilcurrent = c_tf_total /
(n_tf_coils * 1e6)`). That inverse is what a `mda_harness.KNOWN_MINT_VALUES` entry would
use to make `CoilCurrent`'s output comparable; recorded here rather than acted on, since
`mda_harness.py` is outside this unit. *(That entry has since been added, along with the
two `a_tf_wp_*` ones below and `.stellarator.wp_width_r_min` — see
`_audit/closed/constraint_32_investigation.md`.)*

### `winding_pack_total_size` and `st_coil` (this pass)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.n_tf_coils`, `.tfcoil.i_tf_sc_mat` (switch, static), `.stellarator_config.stella_config_a1`/`_a2`/`_wp_ratio`, `.tfcoil.tftmp`/`tmargmin`/`b_crit_upper_nbti`/`bcritsc`/`f_a_tf_turn_cable_copper`/`fhts`/`t_crit_nbti`/`tcritsc`/`f_a_tf_turn_cable_space_extra_void`, `.constraints.f_j_tf_wp_critical_max`, `.tfcoil.a_tf_turn_cable_space_no_void` (from `WindingPackGeometry`), `.tfcoil.dx_tf_turn_general`, `.tfcoil.dx_tf_wp_insulation`, `.tfcoil.a_tf_turn_steel` (from `WindingPackGeometry`) | read | explicit-arg | `winding_pack_total_size` |
| `.tfcoil.j_tf_wp` | read, then written to the same field later in the same call | **implicit-io, cross-call** | see "real PROCESS bugs found" — the read is genuinely a *previous call's* output, not this call's; kept as two independent things (an `From` and an `Output` on the same `VarPath`) rather than collapsed, since collapsing would hide the bug |
| `.tfcoil.b_tf_inboard_peak_symmetric`, `.dx_tf_wp_primary_toroidal`, `.dx_tf_wp_secondary_toroidal` (same value as primary — see the four-functions note above, same treatment), `.dr_tf_wp_with_insulation`, `.j_tf_wp` (fresh value), `.n_tf_coil_turns`, `.c_tf_turn`, `.a_tf_wp_conductor`, `.a_tf_wp_extra_void`, `.a_tf_coil_wp_turn_insulation`, `.a_tf_wp_steel` | write | explicit-arg | `winding_pack_total_size` |
| `.tfcoil.a_tf_wp_no_insulation`, `.tfcoil.a_tf_wp_with_insulation` | write | **minted** | see "cottax node" below — matches `mass.py`'s/`forces.py`'s already-shipped `From`s at these exact paths, not a fresh invention. Re-verified in the MDA triage (`_audit/next_steps.md` §8.1): both are plain Python locals in `winding_pack_total_size` (`process/models/stellarator/coils/calculate.py:496-501`), and the source itself says so in the comment on line 496 ("not global"). **Trap:** fields with exactly these two names *do* exist, at `.superconducting_tfcoil.*` (`process/data_structure/superconducting_tf_coil_variables.py:35,40`) — but they are written only by the tokamak resistive TF model (`process/models/tfcoil/resistive.py:310,334`), which never runs for a stellarator, so rebinding the mints there would compare against `DataStructure()`'s bare `0.0`. Both are nonetheless *reconstructible* from real fields at any converged state: `a_tf_wp_no_insulation == .tfcoil.dx_tf_wp_primary_toroidal * .tfcoil.dr_tf_wp_with_insulation` and `a_tf_wp_with_insulation == (.tfcoil.dr_tf_wp_with_insulation + 2*.tfcoil.dx_tf_wp_insulation) * (.tfcoil.dx_tf_wp_primary_toroidal + 2*.tfcoil.dx_tf_wp_insulation)` — both read straight off `calculate.py:483-501`, which writes the three real fields and then forms the two locals from them. **Both reconstructions are now live** in `mda_harness.KNOWN_MINT_VALUES` (constraint-32 investigation, `_audit/closed/constraint_32_investigation.md`), which is part of what took this node's whole SCC out of `mda_harness.EXCLUDED_NODE_NAMES`; the `a_tf_wp_no_insulation` identity was cross-checked against PROCESS's own `data.tfcoil.j_tf_wp = coilcurrent * 1e6 / a_tf_wp_no_insulation` (`calculate.py:499`), which it reproduces to the last printed digit. |
| — (return-only) | — | reporting-only | `fraction_area_superconductor_of_wp` (`f_a_scu_of_wp` in `st_coil`) only ever reaches `write()`, same treatment as `coilcoilgap` |
| `.stellarator.r_coil_major`/`r_coil_minor`, `.tfcoil.dx_tf_turn_steel`/`dx_tf_turn_insulation`, `.stellarator.f_st_b`, `.stellarator_config.stella_config_i0`, `.stellarator.f_st_rmajor`/`f_st_n_coils`, `.tfcoil.dr_tf_nose_case`, `.stellarator_config.stella_config_max_portsize_width`/`_dmin`/`_coil_rmajor`/`_coil_rminor`/`_inductance`/`_maximal_coil_height`/`_coillength`/`_coilsurface`/`_min_bend_radius`, `.tfcoil.den_tf_coil_case`/`den_tf_wp_turn_insulation`/`a_tf_wp_coolant_channels`, `.physics.rmajor`/`rminor`/`b_plasma_toroidal_on_axis`, `.build.dr_fw_plasma_gap_inboard`/`dr_fw_inboard`/`dr_blkt_inboard`/`dr_shld_blkt_gap`/`dr_shld_inboard`/`dr_fw_plasma_gap_outboard`/`dr_fw_outboard`/`dr_blkt_outboard`/`dr_shld_outboard`/`dr_vv_inboard`/`dr_vv_outboard`, `.tfcoil.t_tf_superconductor_quench`/`t_tf_quench_detection`, `.stellarator_config.stella_config_max_force_density`/`_max_force_density_mnm`/`_max_lateral_force_density`/`_max_radial_force_density`/`_wp_bmax`/`_wp_area`/`_centering_force_max_mn`/`_min_mn`/`_avg_mn` | read | explicit-arg | `st_coil`'s own further reads, beyond what `winding_pack_total_size` needs (feeding `mass.py`/`quench.py`/`forces.py`) |
| `.fwbs.den_steel`, `.tfcoil.dcond[i_tf_sc_mat - 1]` | read | explicit-arg (`den_steel`, `den_tf_sc_material`) | `st_coil` → `calculate_coils_mass` (`process/models/stellarator/coils/mass.py:88`). The *pure function* still takes the already-indexed scalar; **the node no longer mints a `VarPath` for it** — `CoilsMass` reads the real `.tfcoil.dcond[0]` since the MDA triage (`_audit/next_steps.md` §8.1), see `mass.md`'s cottax-node section for the full argument |
| `.tfcoil.len_tf_coil` | **read before this same call's own write** | **implicit-io, cross-call — a second, independent instance of the same bug class as `j_tf_wp` above** | see "real PROCESS bugs found"; kept as `len_tf_coil_stale` (explicit input, feeds only `calculate_plasma_facing_coil_area`) vs. `len_tf_coil` (fresh local, feeds `calculate_coils_mass`/`forces.calculate_centering_force_*`) — never collapsed into one |
| `.build.z_tf_inside_half`, `.tfcoil.len_tf_coil` (fresh write), `.tfcoil.tfcryoarea` | write | explicit-arg | `st_coil`'s inline geometry block — open question 2 from the first pass. Resolved twice over, in the same direction each time: **two of the three are now their own function + node** (`calculate_z_tf_inside_half`/`ZTfInsideHalf`, then `calculate_tfcryoarea`/`TfCryoArea`), because "one call site, keep it inline" stops holding the moment the graph needs an owner for the field and the eager `st_coil` orchestrator is not registered. `len_tf_coil` stays inline: it is the stale-read bug row above, and giving it a producer would silently switch `PlasmaFacingCoilArea` from the stale value to the fresh one (`_audit/boundary_inputs_audit.md` §4c (c1)) — a decision, not a port. `min_bending_radius` stays inline because nothing reads it. |
| — (return-only) | — | reporting-only | `min_bending_radius`, `inductance` (`st_coil` locals only reaching `write()`) |
| every `data.tfcoil.*`/`.build.*` field written by `calculate_coils_mass`/`calculate_quench_protection`/`forces.calculate_*` | write | explicit-arg | via the already-ported functions from units #11/#12/#14, called unchanged |

## proposed signature(s)
See the port file — each function above is ported with its original name and the
`data.*` reads promoted to explicit parameters, one-for-one with the table above. No
composition into fewer functions was done; each is already a natural, independent unit,
**except** `winding_pack_total_size`, which factors its curve-sampling half out into
`winding_pack_curves` (an internal seam, not independently audited/tested, that lets the
harness's residual function rebuild the same `(wp_width_r, lhs, rhs)` curves the solve
itself uses — see `TestWindingPackTotalSize`'s own `_winding_pack_total_size_residual`),
and, **a later pass**, further into `winding_pack_pre_intersect`/
`winding_pack_post_intersect` (the same internal-seam status, not independently
audited/tested; `winding_pack_total_size` itself still composes them, unchanged in
behaviour — see the top-of-file note and "cottax node" below).
`_critical_current_density_by_material` is a **local, scoped restatement** of
`jcrit_from_material`'s 8-way dispatch (`coils/coils.py`, still unported — out of this
unit's file boundary), calling the real ported material models in
`functional_process/models/physics/superconductors.py` directly; it is not the audited
port of `jcrit_from_material` itself, which stays unit #10's to do (see "switches
touched").

## cottax node
**Actually written**, three more classes on top of the first pass's ten, in this pass —
**superseded by a later pass**, described further below, which splits the middle one
(`WindingPackTotalSize`) into three around `coils.py`'s `Intersect`. The original
description is kept (below) as the accurate record of what the `j_tf_wp`/`FixedPoint`
split resolved; the split-around-`intersect` section that follows describes only what
changed on top of it.

- **`WindingPackTotalSize`** (this pass's `ExplicitFunction`, **removed by the later
  pass below** — kept here as history) had `i_tf_sc_mat` a static field, same
  treatment as `EcrhDensityLimit.i_plasma_pedestal`. Its `.tfcoil.a_tf_wp_with_insulation`/
  `a_tf_wp_no_insulation` `Output`s were **minted, cross-checked against two already-shipped
  consumers**: `coils/mass.py`'s `CoilsMass` and `coils/forces.py`'s `MaxForceDensity`
  (etc.) already declared `From`s at exactly these two paths (`mass.md`'s own "cottax
  node" section: "should mint its output under this exact name") — this node was that
  producer, not a fresh, independent choice. Discovering this also surfaced a **real bug
  in the first pass's own `CoilCrossSectionalArea`**, fixed in this pass: its
  `a_tf_wp_with_insulation` `From` read `s.tfcoil.dr_tf_wp_with_insulation` (a
  dimensionally different field — winding-pack *radial thickness*, not *area*) because
  no producer existed yet for the correct path when it was written; it now reads
  `s.tfcoil.a_tf_wp_with_insulation`, matching `CoilsMass`/`MaxForceDensity` — unaffected
  by the later split, since `WindingPackTotalSizePost` (below) is now that producer
  instead, at the identical `VarPath`.

  **`j_tf_wp` self-loop — resolved (later pass, not the one that first wrote this
  node)**: `WindingPackTotalSize` originally declared `j_tf_wp` as **both** an `From`
  and an `Output` on the same `VarPath` — faithful to the source's genuine
  self-referential read (see data footprint), but `spec.py` forbids a node reading what
  it owns, so `to_graph(WindingPackTotalSize(...))` raised `ValueError: reads
  ['.tfcoil.j_tf_wp'], which it also owns` — confirmed directly before making any
  change, the same failure class `Avail`'s `.costs.cplife` self-loop hits
  (`_audit/next_steps.md` §5, "Shape B"). Resolved by the same mechanism that section
  names: `.tfcoil.j_tf_wp`'s ownership is now split out into a new node,
  **`WindingPackJTfWp`** (`FixedPointFunction`, `i_tf_sc_mat` a static field like
  `WindingPackTotalSize`'s own) — `step` reads the real `.tfcoil.j_tf_wp` plus every
  other input `winding_pack_total_size` needs and returns the fresh value as the next
  iterate; `FixedPointFunction` mints the cut internally (body writes
  `^cond.tfcoil.j_tf_wp`, a separate bodyless `FixedPoint` problem node owns the real
  `.tfcoil.j_tf_wp` and reads that minted copy). `WindingPackTotalSize` no longer
  declares a `j_tf_wp` `Output` at all — it keeps `j_tf_wp` as a plain, non-owning
  `From` (the current committed value) and its `__call__` simply discards the
  `j_tf_wp_new` element of `winding_pack_total_size`'s return tuple, since
  `WindingPackJTfWp.step` is now where that value is kept and minted. **Untouched by the
  later split-around-`intersect` pass below**: `WindingPackJTfWp` still calls the whole
  `winding_pack_total_size` pure function directly (which still composes the same two
  halves internally, unchanged in behaviour), not the new node classes.

  `WindingPackJTfWp.step` necessarily re-runs the *entire* `winding_pack_total_size`
  computation (`intersect`'s 200-point sampled crossing included), not some smaller
  slice: the resolved winding-pack width `dr_tf_wp_with_insulation` itself depends on
  `j_tf_wp` whenever `i_tf_sc_mat == 2` (Bi-2212), via the `lhs` curve `intersect`
  crosses, so there is no self-contained sub-computation smaller than the whole function
  to isolate. This duplicates that sampling between the two nodes, deliberately, rather
  than factoring a third `wp_width_r_min`-producing node out from under both (that would
  only relocate the duplication, not remove it, and is an unrequested design change).

  **`i_tf_sc_mat`-conditioning of the self-loop**: no explicit pass-through/identity
  branch was written for `i_tf_sc_mat != 2` (unlike `plasma_composition`'s `first_call`
  or `Avail`'s `cplife`, which do special-case their non-cycling branch). It falls out
  of `_critical_current_density_by_material`'s existing 8-way dispatch instead: only the
  `i_tf_sc_mat == 2` branch reads `j_wp` at all, and `i_tf_sc_mat` is a static field
  (resolved at Python/trace time, "switches are not ports" per `naming_convention.md`),
  so for every other material `step`'s traced body never reads its `j_tf_wp` parameter —
  `d(step)/d(j_tf_wp) == 0` identically, a degenerate but entirely valid fixed point any
  correct driver converges to in one iteration, confirmed directly by `jax.grad` at
  `i_tf_sc_mat` in `{1, 5, 7}` (`test_calculate.py::test_winding_pack_j_tf_wp_step_is_a_degenerate_fixed_point_off_bi2212`).
  `i_tf_sc_mat == 2` is the one genuine, non-trivial self-loop — not independently
  gradient-tested here, for the same `bi2212` validity-domain fragility already flagged
  under "JAX-difficulty flags" and in `TestWindingPackTotalSize`'s own sample-selection
  docstring (narrow domain, not exercised by this unit's tests either way).

**Update, later consolidation pass: `WindingPackJTfWp` (described just above) is
deleted.** It duplicated the entire `winding_pack_total_size` computation a second time
just to isolate `j_tf_wp`, sitting unregistered right next to the three-node split below
which already recomputed the same thing independently. Once `WindingPackTotalSizePost`
(below) took ownership of `.tfcoil.j_tf_wp` instead of discarding it, and
`WindingPackIntersectInputs` (below) kept reading the real value as it already did, the
self-reference closed through the three real nodes plus `Intersect`'s own `RootFind`
problem as one ordinary 4-node cross-node cycle ("Shape A", `_audit/next_steps.md` §5) --
no `FixedPointFunction`/`Cut` needed at all, confirmed via
`to_graph()`/`.cycles` (`test_calculate.py::
test_winding_pack_intersect_split_forms_one_combined_cycle`). This is now registered in
`total_process.py` in place of `WindingPackJTfWp`.

### `winding_pack_total_size` split around `intersect` (later pass)

`WindingPackTotalSize` (above) called `coils.py`'s `intersect` eagerly, in the middle of
its own `__call__`. Replaced by three nodes, composing around `coils.py`'s `Intersect` —
now a genuine `ImplicitFunction`/`RootFind` pair (see `coils.md`'s own "cottax node"
section for `Intersect`'s full design) — instead of calling the plain `intersect`
function directly:

- **`WindingPackIntersectInputs`** (`ExplicitFunction`, `i_tf_sc_mat` a static field,
  same treatment as the removed `WindingPackTotalSize`'s own) — the *pre*-`intersect`
  half. Calls `winding_pack_pre_intersect` (a later-pass split of
  `winding_pack_total_size`'s own body, sharing `winding_pack_curves` unchanged) and
  mints `.stellarator.wp_width_r`/`.lhs`/`.rhs` — exactly the `VarPath`s `coils.py`'s
  `Intersect` reads as its own `From`s, per `coils.md`'s own earlier sketch of this
  split, not a fresh invention here. `intersect`'s own starting guess
  (`wp_width_r_min_guess`) is computed but not wired through as a declared `Output` —
  `xin` has no port in the `ImplicitFunction`/`RootFind` shape at all (see `Intersect`'s
  docstring).
- **`coils.py`'s `Intersect`**, reused directly, unmodified — its minted `VarPath`s
  happen to match this call site exactly (they were designed to, from `coils.md`'s own
  draft), so no call-site-specific wrapper class was needed here. `Intersect`'s own
  `RootFind` problem owns `.stellarator.wp_width_r_min`.
- **`WindingPackTotalSizePost`** (`ExplicitFunction`, **no `i_tf_sc_mat`** — nothing in
  `winding_pack_post_intersect` depends on the material dispatch, which is entirely
  upstream) — the *post*-`intersect` half. Reads `.stellarator.wp_width_r_min` as a
  plain, ordinary `From` (owned by `Intersect`'s `RootFind` problem, not by this node —
  an ordinary cross-node edge, not a second self-loop), calls
  `winding_pack_post_intersect`, and mints `.tfcoil.a_tf_wp_with_insulation`/
  `a_tf_wp_no_insulation` at the same `VarPath`s the removed `WindingPackTotalSize`
  already minted them at — this node is now their producer, `CoilsMass`/`MaxForceDensity`
  unaffected.

Verified directly, not just asserted: `to_graph(WindingPackIntersectInputs(...),
Intersect())` assembles into one coupled block with a single `RootFind` problem
(`test_calculate.py::test_winding_pack_intersect_pair_assembles_around_the_root_find`);
**updated** -- the full three-node split now merges with `Intersect`'s own `RootFind`
into a single combined 4-node SCC once `WindingPackTotalSizePost` owns `.tfcoil.j_tf_wp`
(`test_calculate.py::test_winding_pack_intersect_split_forms_one_combined_cycle`), not
"assembles alongside a separate `WindingPackJTfWp` split" as an earlier draft of this
section said (that class is deleted, see above). Driving the merged 4-node block needs a
*generic* `RootFind` driver, not `coils.py`'s `IntersectBisectionNewtonPolish` --
that driver reaches into `conditions.context` directly for `wp_width_r`/`lhs`/`rhs`,
which stops working once `WindingPackIntersectInputs` (their producer) sits *inside* the
driven block rather than outside it. `test_calculate.py`'s own test-only
`_GenericBisectionRootFind` (calling `conditions(x)` generically instead) reproduces
exactly the same `dr_tf_wp_with_insulation`/`a_tf_wp_with_insulation` the plain
`winding_pack_total_size` function computes by calling `intersect` eagerly
(`test_calculate.py::test_winding_pack_intersect_driven_matches_the_pure_function`) --
same numbers, reached through the new driven path instead of the old eager one, per
`_audit/next_steps.md` §7's own framing of what this conversion is for.

- No node for `st_coil` itself. It is the union of the individual nodes above (this
  file's 15 -- 13 tier-1 plus `WindingPackIntersectInputs`/`WindingPackTotalSizePost` --
  plus `coils.py`'s `Intersect` and units #11/#12/#14's own already-registered ones)
  rather than a computation of its own — writing one more giant `ExplicitFunction`
  wrapping all of them would duplicate exactly the decomposition `cottax`'s graph exists
  to replace, and (per `spec.py`) a node cannot both read and own
  `j_tf_wp` or `len_tf_coil` the way the pre-split `WindingPackTotalSize` and `st_coil`'s
  own inline geometry each do — the same same-`VarPath` conflict either way. `st_coil`'s
  job is the *composition order/wiring*,
  which belongs in `total_process.py`'s `Graph` assembly (out of this unit's scope, see
  `schema.md`), not in a new node. The pure function is still written and tested (see
  `test_calculate.py::test_st_coil_matches_process_end_to_end`) as the artifact that
  wiring pass will read off.

Not yet registered in `total_process.py` (out of this fork's scope — the parent
conversation is consolidating that centrally to avoid write conflicts with other units
being audited in parallel).

## tier signal
**Tier 1** for the first pass's 10 functions — no internal solve, no calls to another
file, no data-dependent Python control flow beyond the source's own already-static
branches (none of these 10 have any). `calculate_winding_pack_geometry` has one `if
dx_tf_turn_cable_space_average < 0: logger.warning(...)` — dropped in the port (see
JAX-difficulty flags), not a computational branch.

**`winding_pack_total_size`: Tier 2, ported this pass.** Samples 200 points into a
`(lhs, rhs)` curve pair, then calls `intersect()` (`coils/coils.py`, unit #10, now
ported, tier-2 itself) — the internal Newton–Raphson root-find this function's own
tier-2 classification is really about. Genuinely self-contained now: `intersect`/
`bmax_from_awp` are real ported functions, and the one remaining dependency
(`jcrit_from_material`'s dispatch) is worked around with a local, scoped restatement
(see "proposed signature(s)") rather than waited on — the same "calls into another
unported unit" shape `power_at_ignition_point` was blocked on in `density_limits.md`,
except here the actual blocker (the material models) already exists as real code, so
the block is resolvable without landing unit #10 first. `Tier2Contract`, same pattern as
`coils.md`'s `TestIntersect`: no value-agreement test by construction, residual-based
pass criterion instead (see `test_calculate.py`).

**`st_coil`: Tier 3, ported this pass.** Composition of ten tier-1 functions (this
file) with `winding_pack_total_size` (tier-2, above), `calculate_coils_mass`
(`coils/mass.py`, unit #12, ported), `calculate_quench_protection`
(`coils/quench.py`, unit #14, ported), and `forces.calculate_*` (`coils/forces.py`,
unit #11, ported) — `write` (`coils/output.py`, unit #13, confirmed pure reporting) is
dropped, not ported, matching the source's `output: bool` flag having no effect on any
returned value. No new internal solver is introduced at the composition level (per
`test_harness.md`'s tier-3 definition) — every genuine solve already lives inside
`winding_pack_total_size`. Validated end-to-end against real PROCESS at one realistic
point (`test_st_coil_matches_process_end_to_end`), **not** via a `Tier3Contract` — none
exists yet (`test_harness.md`'s "Not built" section) — so this is a single hand-written
comparison, not fuzzed or gradient-checked; see the test's own docstring for why that is
an acknowledged, deliberate gap rather than an oversight.

## switches touched
- `.tfcoil.i_tf_sc_mat` — read inside `winding_pack_total_size` (sampling-bounds choice,
  `== 6`) and inside `_critical_current_density_by_material` (the 8-way material
  dispatch, mirroring `jcrit_from_material`'s own branches — see `coils.md`/
  `superconductors.md`'s switches tables for the full per-branch reads-set). Kept as a
  **static field** on `WindingPackIntersectInputs` (the pre-`intersect` node, since the
  material dispatch is entirely upstream of `intersect` — `WindingPackTotalSizePost`
  needs no `i_tf_sc_mat` at all) and on `WindingPackJTfWp`, unchanged by the later split
  (`naming_convention.md`'s "switches are not ports"), same treatment as
  `EcrhDensityLimit.i_plasma_pedestal`. Not read anywhere in the 10 first-pass functions.
- `_critical_current_density_by_material` is this unit's own local dispatcher, not the
  audited port of `jcrit_from_material` (`coils/coils.py`, unit #10) — it exists
  because `winding_pack_total_size` genuinely needs *a* working dispatch and `coils.py`
  is out of this unit's file boundary. Whoever eventually ports `jcrit_from_material`
  itself should very likely **replace**, not merely parallel, this dispatcher (split
  one node per `i_tf_sc_mat` value, per `switches.md`'s guidance and
  `superconductors.md`'s own sketch) rather than leave two independent restatements of
  the same 8-way switch alive in the codebase.

## calls into other models
- **First-pass 10 functions**: none — each is self-contained arithmetic on its own
  inputs.
- **`winding_pack_total_size`**: `intersect`, `bmax_from_awp` (`coils/coils.py`, unit
  #10, ported); `itersc`, `bi2212`, `jcrit_nbti`, `western_superconducting_nb3sn`,
  `jcrit_rebco`, `gl_nbti`, `gl_rebco` (`physics/superconductors.py`, unit #22, ported)
  via `_critical_current_density_by_material`.
- **`st_coil`**: everything `winding_pack_total_size` calls, plus `calculate_coils_mass`
  (unit #12), `calculate_quench_protection` (unit #14), `forces.calculate_*` (unit #11).

## JAX-difficulty flags
- **`logger.warning`/`logger.info` on a data-dependent condition**
  (`calculate_winding_pack_geometry`, `if dx_tf_turn_cable_space_average < 0`) —
  `workaround-known` (same class as `naming_convention.md`'s existing guidance for
  data-dependent branches): a Python-level `if` on a value that would be traced cannot
  execute under `jit`/`jacfwd`. Dropped in the port rather than converted to
  `jax.lax.cond` + `jax.debug.print`, since it is diagnostic-only (does not change the
  computed value, only logs) — flagging the removal rather than silently doing it.
- **`b_max > bc20m` branches (`i_tf_sc_mat` 1/3)** — `minor`, `needs-lax-cond-or-where`,
  resolved with `jnp.where` in `_critical_current_density_by_material`, matching
  `coils.md`'s own flag for this exact branch shape.
- **`bi2212` (`i_tf_sc_mat == 2`)** — `minor`, domain fragility not JAX-traceability:
  `bi2212`'s validity domain (`temp <= 20`, `6 <= b <= 104`) is narrow enough that the
  200-point sweep runs well outside it for much of its range, and this branch alone also
  depends on the stale-`j_tf_wp` input (see data footprint). Not exercised by this
  unit's own tests (see `test_calculate.py`'s sample-selection docstring) — a real
  fragility of this specific material branch, not a JAX-traceability problem, and not
  fixed here.
- Everything else in `winding_pack_total_size`/`st_coil` is plain array/scalar
  arithmetic plus calls to already-JAX-traceable ported functions — no new dynamic
  shapes, no new external calls.

## real PROCESS bugs found
1. **`coils.py:136`'s `jcrit_rebco(t_helium, b_max, 0)` call is genuinely broken** —
   confirmed by directly running PROCESS's own `winding_pack_total_size` with
   `i_tf_sc_mat = 6` at a realistic HELIAS5B-like operating point while writing this
   port: `TypeError: jcrit_rebco() takes 2 positional arguments but 3 were given`. This
   was already flagged (not reproduced) by `superconductors.md`/`coils.md`'s audits from
   a static read; this pass confirms it is **live and reachable**, not theoretical — any
   stellarator run selecting REBCO (`i_tf_sc_mat = 6`) crashes PROCESS outright the
   moment `winding_pack_total_size` is called. Not fixed (the call site is in
   `coils.py`, out of this unit's boundary); the port's own REBCO branch
   (`_critical_current_density_by_material`, `i_tf_sc_mat == 6`) calls the real,
   2-argument `jcrit_rebco` and works correctly on its own terms.
2. **`st_coil`'s `calculate_plasma_facing_coil_area(data)` call reads `len_tf_coil`
   before `st_coil` itself (re)computes it** — the source calls
   `calculate_plasma_facing_coil_area` several lines *before* the "Coil dimensions"
   block that assigns `data.tfcoil.len_tf_coil`. Confirmed by running PROCESS's own
   `st_coil` on a fresh `DataStructure` (`len_tf_coil` defaults to `0.0`):
   `data.tfcoil.tfsai`/`tfsao` (the plasma-facing coil area) come out **exactly `0.0`**.
   This is the same class of bug as finding 3 below (`j_tf_wp`) — an undeclared,
   cross-call fixed point PROCESS's outer `Caller.call_models` Gauss-Seidel loop
   (`../../../CLAUDE.md` "Implicit cycles are hidden, not declared") happens to wash out
   after enough rounds once `len_tf_coil` stabilises, but which is genuinely wrong on
   any round where it hasn't (e.g. the very first call, or any round following a change
   to `r_coil_minor`/`n_tf_coils`). Reproduced faithfully, not fixed: the port's
   `st_coil` takes `len_tf_coil_stale` as its own explicit input, separate from the
   freshly-computed `len_tf_coil` used everywhere else in the same function.
3. **`winding_pack_total_size` reads `data.tfcoil.j_tf_wp` before overwriting it** —
   only the `i_tf_sc_mat == 2` (Bi-2212) branch actually uses this read
   (`jstrand = j_wp / (1 - f_he)`), but the read happens inside the 200-point sampling
   loop, *before* this same call's own `data.tfcoil.j_tf_wp = coilcurrent * 1e6 /
   a_tf_wp_no_insulation` write near the end. A second, independent instance of finding
   2's bug class, in the same file. Reproduced faithfully: the port's `j_tf_wp` is an
   explicit input (the stale prior value) *and* a separate output (this call's fresh
   value) — see the `WindingPackTotalSize` node docstring for why this also means the
   node cannot currently join a `Graph` as declared.
4. **Port bug (not PROCESS's), found and fixed this pass**: `calculate.py`'s own
   `CoilCrossSectionalArea` node (first pass) had its `a_tf_wp_with_insulation` `From`
   wired to the wrong `VarPath` (`.tfcoil.dr_tf_wp_with_insulation`, a *length*, instead
   of an *area*) — see "cottax node" above.

## open questions
1. ~~Is `.stellarator.coilcurrent` the right home...~~ **Resolved** — see the
   data-footprint note above; `WindingPackIntersectInputs`/`WindingPackTotalSizePost`
   (the pre-split `WindingPackTotalSize`'s replacement) reading it confirms the minting.
2. ~~`st_coil`'s inline geometry...~~ **Resolved** — ported as ordinary locals inside
   `st_coil` itself, see the data-footprint table's last rows.
3. ~~`WindingPackTotalSize`'s `j_tf_wp` self-loop...~~ **Resolved** (later pass, see
   "cottax node" above): split into `WindingPackJTfWp` (`FixedPointFunction`, owns
   `.tfcoil.j_tf_wp`) and a `WindingPackTotalSize` (subsequently itself split further,
   into `WindingPackIntersectInputs`/`WindingPackTotalSizePost` — see item 6 below) that
   only reads it. `to_graph` confirmed to succeed on both individually and together
   (`test_calculate.py::test_winding_pack_j_tf_wp_assembles_as_a_fixed_point_node`,
   `::test_winding_pack_intersect_inputs_node_assembles_and_does_not_own_j_tf_wp`,
   `::test_winding_pack_total_size_post_node_assembles_and_does_not_own_j_tf_wp`,
   `::test_winding_pack_intersect_split_assembles_end_to_end`). This is a
   *representability* fix only, per `next_steps.md` §5's own scoping — no `Drive`/solver
   algorithm is assigned to the resulting `FixedPoint` problem here, and `st_coil`'s
   separate, still-open `len_tf_coil` self-loop (finding 2) is untouched by this
   resolution — same bug class, different `VarPath`, not addressed by this pass since
   `st_coil` has no node of its own to split in the first place (see "cottax node"
   above, "No node for `st_coil` itself").
4. **`_critical_current_density_by_material`'s relationship to unit #10's own eventual
   `jcrit_from_material` port** — flagged under "switches touched": whoever ports
   `coils.py`'s `jcrit_from_material` for real should almost certainly replace this
   unit's local dispatcher (import from wherever that lands, or re-derive
   `winding_pack_total_size` to call the real per-branch nodes) rather than leave both
   alive — not attempted here since that unit is out of this fork's boundary.
5. **`den_tf_sc_material` (feeding `st_coil` → `calculate_coils_mass`) is still the
   already-indexed-scalar placeholder `mass.md` flagged** (`.tfcoil.dcond[i_tf_sc_mat -
   1]`, not resolved by this unit either) — `st_coil`'s own port takes it as a plain
   explicit argument, same as `mass.py`'s `CoilsMass` already does; the real lookup node
   is still unit #10/whoever's design question, per `mass.md`'s own open question 1.
6. **[RESOLVED, later pass]** `winding_pack_total_size`'s own internal `intersect` call
   is now also represented structurally, not just called eagerly — see "cottax node"
   above, "`winding_pack_total_size` split around `intersect`", and `coils.md`'s
   `Intersect`/`IntersectBisectionNewtonPolish`. Unlike item 3's `j_tf_wp` split, this
   one *is* accompanied by a concrete driver (`coils.py`'s
   `IntersectBisectionNewtonPolish`) — but only for test purposes, constructing a `Drive`
   used in `test_calculate.py`/`test_coils.py`, not registered anywhere as the graph's
   real answer. The structural declaration (`Intersect` itself, undriven) stays as
   swappable as any other undriven problem node in `_audit/next_steps.md` §5's own
   accounting.


## Update: `TfCryoArea` — the second node carved out of the inline geometry block

`.tfcoil.tfcryoarea` (`process/models/stellarator/coils/calculate.py:92-101`) had **no
producer anywhere in the graph**: the formula was ported, but only inside the eager
`st_coil` orchestrator, which `total_process.py` deliberately does not register. It is
now `calculate_tfcryoarea` + `TfCryoArea`, shaped exactly like
`calculate_z_tf_inside_half`/`ZTfInsideHalf` — a shared function with two call sites
(`st_coil` itself and the node), so the formula is not duplicated.

Why now: `.tfcoil.tfcryoarea` is an input of `power_B_thermal_cryo.py`'s
`CryoQLoadsStep` (it feeds `Power.cryo`'s `qss` term). Registering the cryogenic-load
nodes without it would have closed two boundary inputs and opened one
(`_audit/boundary_inputs_audit.md` §4c (c1)'s "sibling gap in the same three lines",
and §7 items 4 and 7).

One faithfulness note worth recording, because it looks like a simplification and is
not: PROCESS's `tfcryoarea` line reads `data.stellarator.r_coil_minor` where the two
lines above it use `st_coil`'s local `r_coil_minor`. They are the same value — the
local is bound from the field at `calculate.py:41` and nothing between writes it — so
the single parameter is faithful.

Cycle risk: none, measured. Its inputs are `.stellarator_config.stella_config_coilsurface`
and `_coil_rminor` (both boundary) and `.stellarator.f_st_rmajor`/`r_coil_minor` (both
owned by `StellaratorScalingFactors`, which is upstream of every reader).

Tested as `TestTfCryoArea` (`Tier1Contract`, `fuzz_bounds`-only, same provenance
argument as `TestZTfInsideHalf` — there is no standalone PROCESS entry point for one
inline line, and `st_coil`'s own end-to-end test already checks `checks["tfcryoarea"]`
against real PROCESS) plus a node-level assembly/ownership test.
