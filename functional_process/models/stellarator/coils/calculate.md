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
`WindingPackTotalSize`'s own `coilcurrent` `Input` reads this exact `VarPath`, so three
nodes (`CoilCurrent`, `CoilsSummaryVariables`, `WindingPackTotalSize`) now genuinely
share it, confirming it was the right call.

### `winding_pack_total_size` and `st_coil` (this pass)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.n_tf_coils`, `.tfcoil.i_tf_sc_mat` (switch, static), `.stellarator_config.stella_config_a1`/`_a2`/`_wp_ratio`, `.tfcoil.tftmp`/`tmargmin`/`b_crit_upper_nbti`/`bcritsc`/`f_a_tf_turn_cable_copper`/`fhts`/`t_crit_nbti`/`tcritsc`/`f_a_tf_turn_cable_space_extra_void`, `.constraints.f_j_tf_wp_critical_max`, `.tfcoil.a_tf_turn_cable_space_no_void` (from `WindingPackGeometry`), `.tfcoil.dx_tf_turn_general`, `.tfcoil.dx_tf_wp_insulation`, `.tfcoil.a_tf_turn_steel` (from `WindingPackGeometry`) | read | explicit-arg | `winding_pack_total_size` |
| `.tfcoil.j_tf_wp` | read, then written to the same field later in the same call | **implicit-io, cross-call** | see "real PROCESS bugs found" — the read is genuinely a *previous call's* output, not this call's; kept as two independent things (an `Input` and an `Output` on the same `VarPath`) rather than collapsed, since collapsing would hide the bug |
| `.tfcoil.b_tf_inboard_peak_symmetric`, `.dx_tf_wp_primary_toroidal`, `.dx_tf_wp_secondary_toroidal` (same value as primary — see the four-functions note above, same treatment), `.dr_tf_wp_with_insulation`, `.j_tf_wp` (fresh value), `.n_tf_coil_turns`, `.c_tf_turn`, `.a_tf_wp_conductor`, `.a_tf_wp_extra_void`, `.a_tf_coil_wp_turn_insulation`, `.a_tf_wp_steel` | write | explicit-arg | `winding_pack_total_size` |
| `.tfcoil.a_tf_wp_no_insulation`, `.tfcoil.a_tf_wp_with_insulation` | write | **minted** | see "cottax node" below — matches `mass.py`'s/`forces.py`'s already-shipped `Input`s at these exact paths, not a fresh invention |
| — (return-only) | — | reporting-only | `fraction_area_superconductor_of_wp` (`f_a_scu_of_wp` in `st_coil`) only ever reaches `write()`, same treatment as `coilcoilgap` |
| `.stellarator.r_coil_major`/`r_coil_minor`, `.tfcoil.dx_tf_turn_steel`/`dx_tf_turn_insulation`, `.stellarator.f_st_b`, `.stellarator_config.stella_config_i0`, `.stellarator.f_st_rmajor`/`f_st_n_coils`, `.tfcoil.dr_tf_nose_case`, `.stellarator_config.stella_config_max_portsize_width`/`_dmin`/`_coil_rmajor`/`_coil_rminor`/`_inductance`/`_maximal_coil_height`/`_coillength`/`_coilsurface`/`_min_bend_radius`, `.tfcoil.den_tf_coil_case`/`den_tf_wp_turn_insulation`/`a_tf_wp_coolant_channels`, `.physics.rmajor`/`rminor`/`b_plasma_toroidal_on_axis`, `.build.dr_fw_plasma_gap_inboard`/`dr_fw_inboard`/`dr_blkt_inboard`/`dr_shld_blkt_gap`/`dr_shld_inboard`/`dr_fw_plasma_gap_outboard`/`dr_fw_outboard`/`dr_blkt_outboard`/`dr_shld_outboard`/`dr_vv_inboard`/`dr_vv_outboard`, `.tfcoil.t_tf_superconductor_quench`/`t_tf_quench_detection`, `.stellarator_config.stella_config_max_force_density`/`_max_force_density_mnm`/`_max_lateral_force_density`/`_max_radial_force_density`/`_wp_bmax`/`_wp_area`/`_centering_force_max_mn`/`_min_mn`/`_avg_mn` | read | explicit-arg | `st_coil`'s own further reads, beyond what `winding_pack_total_size` needs (feeding `mass.py`/`quench.py`/`forces.py`) |
| `.fwbs.den_steel`, `.tfcoil.dcond[i_tf_sc_mat - 1]` | read | explicit-arg (`den_steel`, `den_tf_sc_material`) | `st_coil` → `calculate_coils_mass`; `den_tf_sc_material` is the same already-indexed-scalar treatment `mass.md` gives it — this port doesn't decide the `i_tf_sc_mat` lookup either |
| `.tfcoil.len_tf_coil` | **read before this same call's own write** | **implicit-io, cross-call — a second, independent instance of the same bug class as `j_tf_wp` above** | see "real PROCESS bugs found"; kept as `len_tf_coil_stale` (explicit input, feeds only `calculate_plasma_facing_coil_area`) vs. `len_tf_coil` (fresh local, feeds `calculate_coils_mass`/`forces.calculate_centering_force_*`) — never collapsed into one |
| `.build.z_tf_inside_half`, `.tfcoil.len_tf_coil` (fresh write), `.tfcoil.tfcryoarea` | write | explicit-arg | `st_coil`'s inline geometry block — **open question 2 from the first pass, now resolved**: kept as ordinary locals inside `st_coil` itself (one call site each), not extracted into a separate function/node |
| — (return-only) | — | reporting-only | `min_bending_radius`, `inductance` (`st_coil` locals only reaching `write()`) |
| every `data.tfcoil.*`/`.build.*` field written by `calculate_coils_mass`/`calculate_quench_protection`/`forces.calculate_*` | write | explicit-arg | via the already-ported functions from units #11/#12/#14, called unchanged |

## proposed signature(s)
See the port file — each function above is ported with its original name and the
`data.*` reads promoted to explicit parameters, one-for-one with the table above. No
composition into fewer functions was done; each is already a natural, independent unit,
**except** `winding_pack_total_size`, which factors its curve-sampling half out into
`winding_pack_curves` (an internal seam, not independently audited/tested, that lets the
harness's residual function rebuild the same `(wp_width_r, lhs, rhs)` curves the solve
itself uses — see `TestWindingPackTotalSize`'s own `_winding_pack_total_size_residual`).
`_critical_current_density_by_material` is a **local, scoped restatement** of
`jcrit_from_material`'s 8-way dispatch (`coils/coils.py`, still unported — out of this
unit's file boundary), calling the real ported material models in
`functional_process/models/physics/superconductors.py` directly; it is not the audited
port of `jcrit_from_material` itself, which stays unit #10's to do (see "switches
touched").

## cottax node
**Actually written**, three more classes on top of the first pass's ten:

- **`WindingPackTotalSize`** (`ExplicitFunction`, `i_tf_sc_mat` a static field, same
  treatment as `EcrhDensityLimit.i_plasma_pedestal`). Its `.tfcoil.a_tf_wp_with_insulation`/
  `a_tf_wp_no_insulation` `Output`s are **minted, cross-checked against two already-shipped
  consumers**: `coils/mass.py`'s `CoilsMass` and `coils/forces.py`'s `MaxForceDensity`
  (etc.) already declared `Input`s at exactly these two paths (`mass.md`'s own "cottax
  node" section: "should mint its output under this exact name") — this node is that
  producer, not a fresh, independent choice. Discovering this also surfaced a **real bug
  in the first pass's own `CoilCrossSectionalArea`**, fixed in this pass: its
  `a_tf_wp_with_insulation` `Input` read `s.tfcoil.dr_tf_wp_with_insulation` (a
  dimensionally different field — winding-pack *radial thickness*, not *area*) because
  no producer existed yet for the correct path when it was written; it now reads
  `s.tfcoil.a_tf_wp_with_insulation`, matching `CoilsMass`/`MaxForceDensity` and this
  node's own `Output`. `WindingPackTotalSize` also declares `j_tf_wp` as **both** an
  `Input` and an `Output` on the same `VarPath` — faithful to the source's genuine
  self-referential read (see data footprint), but `spec.py` forbids a node reading what
  it owns, so this node **cannot join a `Graph` as declared**; a minted "previous value"
  copy is needed first — flagged in the class docstring, not resolved here.
- No node for `st_coil` itself. It is the union of the individual nodes above (this
  file's 13 plus units #11/#12/#14's own already-registered ones) rather than a
  computation of its own — writing one more giant `ExplicitFunction` wrapping all of
  them would duplicate exactly the decomposition `cottax`'s graph exists to replace,
  and (per `spec.py`) a node cannot both read and own `j_tf_wp` or `len_tf_coil` the way
  `WindingPackTotalSize` and `st_coil`'s own inline geometry each do — the same
  same-`VarPath` conflict either way. `st_coil`'s job is the *composition order/wiring*,
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
  **static field** on `WindingPackTotalSize` (`naming_convention.md`'s "switches are not
  ports"), same treatment as `EcrhDensityLimit.i_plasma_pedestal`. Not read anywhere in
  the 10 first-pass functions.
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
   `CoilCrossSectionalArea` node (first pass) had its `a_tf_wp_with_insulation` `Input`
   wired to the wrong `VarPath` (`.tfcoil.dr_tf_wp_with_insulation`, a *length*, instead
   of an *area*) — see "cottax node" above.

## open questions
1. ~~Is `.stellarator.coilcurrent` the right home...~~ **Resolved** — see the
   data-footprint note above; `WindingPackTotalSize` reading it confirms the minting.
2. ~~`st_coil`'s inline geometry...~~ **Resolved** — ported as ordinary locals inside
   `st_coil` itself, see the data-footprint table's last rows.
3. **`WindingPackTotalSize`'s `j_tf_wp` self-loop** (finding 3 above) needs a real
   design decision before this node can join any `Graph`: mint a `^prev`-style copy for
   the read side (the "previous value" cottax's fixed-point handling already has a
   shape for, per `~/jaxgraph/CLAUDE.md`'s "The graph" — "a fixed point is always
   written with a minted copy") is the most likely answer, but is a genuine open
   modelling question about what "previous" means once this sits inside a real solver
   loop rather than PROCESS's own ad hoc re-evaluation — not decided here.
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
