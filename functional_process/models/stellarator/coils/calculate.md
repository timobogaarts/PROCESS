---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `functional_process/models/stellarator/coils/calculate.py` /
`test_calculate.py` — 10 of the 12 functions in this file, all tier-1, all
`ExplicitFunction` nodes with passing tests (fuzz-only, see the test file's docstring
for why there's no `legacy_sample`). `st_coil` itself and `winding_pack_total_size` are
**not** ported — see below — and stay audit-only.

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
ported) and `winding_pack_total_size` (out of scope, see below) — a real graph edge, not
reporting. Minted `.stellarator.coilcurrent` for it (see the port's `CoilCurrent` node
docstring) rather than leaving it un-sourceable.

## proposed signature(s)
See the port file — each function above is ported with its original name and the
`data.*` reads promoted to explicit parameters, one-for-one with the table above. No
composition into fewer functions was done; each is already a natural, independent unit.

## cottax node
**Actually written.** Ten `ExplicitFunction` classes in `calculate.py`, one per ported
function (`CoilToroidalThickness`, `CoilRadialThickness`, `CoilCrossSectionalArea`,
`CoilHalfWidths`, `PlasmaFacingCoilArea`, `CoilCoilToroidalGap`,
`CoilsSummaryVariables`, `StoredMagneticEnergy`, `WindingPackGeometry`, `CoilCurrent`,
`CoilCasing`, `VerticalPorts`, `HorizontalPorts` — 13 classes, since two of the 12
source functions with no downstream reader in this file (`calculate_inductance`) get no
node, matching the `calculate_intercoil_mass_scaling_reference` precedent in
`stellarator_D_structure.py`). Not yet registered in `total_process.py` (out of this
fork's scope — the parent conversation is consolidating that centrally to avoid write
conflicts with other units being audited in parallel).

## tier signal
**Tier 1** for all 10 ported functions — no internal solve, no calls to another file, no
data-dependent Python control flow beyond the source's own already-static branches (none
of these 10 have any). `calculate_winding_pack_geometry` has one `if
dx_tf_turn_cable_space_average < 0: logger.warning(...)` — dropped in the port (see
JAX-difficulty flags), not a computational branch.

**Not ported, not tier-1:**
- **`winding_pack_total_size`** — samples 200 points into a `(lhs, rhs)` curve pair,
  then calls `intersect()` (`coils/coils.py`, registry unit #10, not yet audited/ported)
  — a genuine Newton–Raphson root-find over piecewise-linear-interpolated data (up to
  100 iterations, `epsy`-based convergence check, logs and clamps on failure rather than
  raising). This is a real tier-2 shape (an `ImplicitFunction`/`RootFind` pair would be
  the natural port target), but it is **not self-contained**: it also calls
  `bmax_from_awp` and `jcrit_from_material`, both in `coils/coils.py`. Per the standing
  practice (`_audit/schema.md`'s cottax-node section), a tier-2 unit only gets ported
  once confirmed self-contained — this one is blocked on unit #10 landing first, exactly
  the same shape as `density_limits.md`'s `power_at_ignition_point` being blocked on
  `st_phys`.
- **`st_coil`** — the orchestrator. Calls all 10 ported functions above plus
  `winding_pack_total_size`, `calculate_coils_mass` (`coils/mass.py`, unit #12),
  `calculate_quench_protection` (`coils/quench.py`, unit #14),
  `forces.calculate_*` (`coils/forces.py`, unit #11), and `write` (`coils/output.py`,
  unit #13, already reviewed as pure reporting). Tier-3 composition once units #10-14
  land; not attempted here.

## switches touched
- `.tfcoil.i_tf_sc_mat` — read twice inside `winding_pack_total_size` (both times
  `== 6`, selecting REBCO-specific sampling bounds; not read anywhere in the 10 ported
  functions). Out of this fork's scope since it only appears in the not-yet-ported
  function; flagging for whoever audits unit #10/completes `winding_pack_total_size`.
- No switches in any of the 10 ported functions.

## calls into other models
None among the 10 ported functions — each is self-contained arithmetic on its own
inputs. `st_coil`/`winding_pack_total_size` call into units #10-14, see tier signal.

## JAX-difficulty flags
- **`logger.warning`/`logger.info` on a data-dependent condition**
  (`calculate_winding_pack_geometry`, `if dx_tf_turn_cable_space_average < 0`) —
  `workaround-known` (same class as `naming_convention.md`'s existing guidance for
  data-dependent branches): a Python-level `if` on a value that would be traced cannot
  execute under `jit`/`jacfwd`. Dropped in the port rather than converted to
  `jax.lax.cond` + `jax.debug.print`, since it is diagnostic-only (does not change the
  computed value, only logs) — flagging the removal rather than silently doing it.
- `winding_pack_total_size`'s `intersect()` (once ported) will need real
  `RootFind`/fixed-iteration-count handling under JAX — `np.interp` isn't directly
  `jnp`-portable without care (piecewise-linear interpolation over a *traced* table),
  worth flagging early for whoever picks up unit #10/`winding_pack_total_size` even
  though it's out of this record's scope.
- Nothing else found — the 10 ported functions are plain arithmetic, no external calls,
  no dynamic shapes.

## open questions
1. Is `.stellarator.coilcurrent` (invented, see data-footprint note) the right home for
   this value once `winding_pack_total_size` is ported and also needs it as an input? At
   that point three nodes share one minted `VarPath` that PROCESS itself never
   allocated — worth a name check against whatever convention (if any) emerges from
   other similarly-invented names in this audit (`EcrhDensityLimit`'s
   `.stellarator.dlimit_ecrh`/`.bt_max_ecrh` in `density_limits.py` is the closest
   precedent so far).
2. `st_coil`'s ~15 lines of inline geometry (`z_tf_inside_half`, `len_tf_coil`,
   `tfcryoarea`, `min_bending_radius`) were not extracted into their own function/node
   here, since they belong to the orchestrator (tier-3, out of scope) rather than to any
   of the 10 self-contained helpers — flagging so whoever eventually ports `st_coil`
   itself doesn't miss them as "already covered."
