---
kind: model-unit
status: reviewed
confidence: high
---

## source
`process/models/stellarator/coils/output.py` (546 lines, full file in scope). One
module-level function, `write(...)`, called from `coils/calculate.py` (`st_coil`'s
output path). Confirmed by full read, not assumed from the filename.

## data footprint

Every value printed is either an explicit argument already computed by the caller, or a
direct `data.<area>.<field>` read at the print call site — no branching, no loop, no
internal solve, no call into any other model. `data.tfcoil.*` accounts for the large
majority of reads (~30 fields: `n_tf_coils`, `a_tf_inboard_total`, `dr_tf_inboard`,
`dr_tf_outboard`, `tficrn`, `tfocrn`, `dx_tf_inboard_out_toroidal`, `len_tf_coil`,
`c_tf_total`, `j_tf_wp`, `j_tf_wp_quench_heat_max`, `j_tf_coil_full_area`,
`b_tf_inboard_peak_symmetric`, `e_tf_magnetic_stored_total_gj`, `m_tf_coils_total`,
`m_tf_coil_superconductor`, `m_tf_coil_copper`, `m_tf_wp_steel_conduit`,
`m_tf_coil_conductor`, `a_tf_turn_cable_space_no_void`,
`f_a_tf_turn_cable_space_extra_void`, `dx_tf_turn_steel`, `dx_tf_turn_insulation`,
`a_tf_wp_conductor`, `f_a_tf_turn_cable_copper`, `a_tf_wp_steel`,
`a_tf_coil_wp_turn_insulation`, `a_tf_wp_extra_void`, `dr_tf_wp_with_insulation`,
`dx_tf_wp_primary_toroidal`, `dx_tf_wp_insulation`, `n_tf_coil_turns`, `c_tf_turn`,
`dr_tf_plasma_case`, `dr_tf_nose_case`, `dx_tf_side_case_min`, `a_tf_coil_inboard_case`,
`m_tf_coil_case`); `data.build.*` (`dr_tf_inboard`, `dr_tf_outboard`,
`r_tf_outboard_mid`, `z_tf_inside_half`); `data.stellarator.*` (`vporttmax`,
`vportpmax`, `vportamax`, `hporttmax`, `hportpmax`, `hportamax`). Not tabulated
per-row (all `read`, all `explicit-arg`-shaped `data.*` access, no writes, no
`implicit-io`) — consistent with this audit's established convention of not
enumerating a pure-reporting unit's reads one row at a time (see
`stellarator_G_output.md`).

23 further values (`a_tf_wp_no_insulation`, `centering_force_avg_mn`,
`centering_force_max_mn`, `centering_force_min_mn`, `coilcoilgap`, `coppera_m2`,
`coppera_m2_max`, `f_a_scu_of_wp`, `f_vv_actual`, `f_j_tf_wp_critical_max`,
`inductance`, `max_force_density`, `max_force_density_mnm`,
`max_lateral_force_density`, `max_radial_force_density`, `min_bending_radius`,
`r_coil_major`, `r_coil_minor`, `sig_tf_wp`, `dx_tf_turn_general`,
`t_tf_superconductor_quench`, `toroidalgap`, `allowed_quench_voltage`,
`quench_voltage`) arrive as plain function arguments, already computed by the caller
(`coils/calculate.py` — out of this unit's scope). No writes to `self.data`/`data`
anywhere in this file.

## proposed signature(s)
None. No separable computation exists to give a signature to (see "not purely inert"
below for the one nuance).

## cottax node
None — nothing to wrap. `write` has no output ports (writes nothing to `data`) and no
place in a `Graph` beyond a would-be sink with no computation, which is out of this
audit's scope by the same convention as `stellarator_G_output.md`.

## tier signal
Not applicable / out of port scope — pure reporting shell, no computation to assign a
tier to (same category as `stellarator_G_output.md`'s `st_phys_output`).

## switches touched
None. Zero `if`/`elif`/branching of any kind in the entire 547-line file — the whole
body is one straight-line sequence of `po.oheadr`/`po.osubhd`/`po.ovarre` calls.

## calls into other models
None — only `process.core.process_output` (`po.*`) formatting helpers.

## JAX-difficulty flags
- **Not purely inert, same pattern as `density_limits.py`'s `output()` and
  `stellarator_E3`'s output block**: several values are computed inline, for the
  printout only, and never stored — `r_coil_major / r_coil_minor` (line 111),
  `data.tfcoil.a_tf_inboard_total / data.tfcoil.n_tf_coils` (118),
  `1.0e-6 * data.tfcoil.c_tf_total` (175) and five more `c_tf_total`-derived ratios
  (181, 366-370, 376), five `.../ap` winding-pack fraction ratios (295, 307, 313, 319,
  and `ap = a_tf_wp_no_insulation` itself at line 288), `sig_tf_wp * 1.0e-6` (402),
  `coppera_m2 * 1.0e-6` (467), `coppera_m2 / coppera_m2_max` (473). `minor` severity in
  every case — plain division/scaling, no domain guards, no control flow — but flagged
  per this audit's now-recurring finding that "reporting" in this codebase routinely
  means "reporting plus a handful of uncommitted arithmetic," not zero computation.
  None of these were judged worth extracting as standalone functions: each is a
  single inline expression consumed immediately by one `po.ovarre` call, not a
  multi-step or reused calculation (contrast with `stellarator_E3`'s 49-line
  cryostat/VV core, which *was* worth extracting).
- No CoolProp or other external calls, no dynamic shapes.

## open questions
None.
