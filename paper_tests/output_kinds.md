# Output kinds in the PROCESS stellarator reference run

Companion to `decision_kinds.md`, which classifies the *boundary inputs* of
`indat.GRAPH -> mda.driven_graph` (and lists a first pass of rule-closed build outputs in its
§3.6). This file goes the other way: every **owned** (computed) place of the same driven graph
— 544 of them across 156 nodes, listed programmatically —

```python
from functional_process.cottax import indat, mda
g = mda.driven_graph(indat.GRAPH)
for node in g.nodes:
    print(node.spelling, [v.spelling for v in g[node].owns], [v.spelling for v in g[node].reads])
```

— is checked against its producing formula (`functional_process/cottax/models/**` and the
`process/models/**`/`process/core/**` sources they mirror) and flagged if the formula is really a
**build decision** (a thickness, coil dimension/current, coil count, aspect ratio, material
fraction, case size — something fixed before the machine exists) or an **operating decision** (a
temperature, density, fuelling, heating power, coolant setting, availability — something an
operator sets with the machine in hand) closed by a rule rather than left free. Everything else
— a state variable, or a quantity that follows necessarily from already-decided geometry/physics
with no discretion of its own — is not flagged.

Two more things are asked of every flag:

- **definition vs. sizing choice.** A *definition* is an equality that must hold physically with
  no slack (Ampere's law giving coil current from field/radius/turns, a geometric sum of
  already-fixed sub-thicknesses, a conservation relation). A *sizing choice* is an equality that
  is really an inequality PROCESS has pinned at zero slack ("current density equals the critical
  fraction," "thickness equals the floor," "count equals the ceiling of a required capacity").
  Only sizing choices are candidates for being lifted into design variables under a chance
  constraint; definitions stay derived however carefully audited.
- **`icc`/`ixc` — does PROCESS already catalogue the inequality this rule stands in for, and
  does the reference run use that catalogue entry or bypass it?** `icc` is PROCESS's constraint
  equation number (`process/core/solver/constraints.py`, named in `lablcc`,
  `process/data_structure/numerics.py`); `ixc` is its iteration-variable number (`lablxc`, same
  file). The `stellarator_helias.IN.DAT` reference run activates
  `icc = 2, 16, 24, 8, 17, 18, 67, 82, 83, 62, 32, 34, 35, 65` (14; 2 and 16 stated as
  equalities) and `ixc = 2, 3, 4, 6, 10, 109, 59, 56` (8). Anything outside those two lists is
  inactive here even if PROCESS has catalogued it.

Enumeration was exhaustive (every owned place reviewed); flagging was conservative (a flag needs
an actual sizing/closure formula, read at the source, not a suggestively-named variable).

## 1. Build decisions closed by a rule

### 1a. TF coil / magnet (`stellarator.coils.*`, `stellarator.stellarator_scaling_factors`, `stellarator.default_aspect_ratio`, `tfcoil.*`)

| place | node | rule | reads | reason | kind | inequality (sizing choice) | icc | ixc | this run |
|---|---|---|---|---|---|---|---|---|---|
| `.physics.aspect` | `default_aspect_ratio` | `aspect = stella_config_aspect_ref` (only exists because `ixc 1` is off) | `stella_config_aspect_ref` | aspect ratio is the designer's; here copied from the reference config | definition | -- | -- | -- | `ixc 1` (`aspect`) commented out in the file |
| `.tfcoil.n_tf_coils` | `stellarator_scaling_factors` | `n_tf_coils = coilspermodule × symmetry` ("overwrites n_tf_coils in input file") | `stella_config_coilspermodule`, `stella_config_symmetry` | coil count fixed by two config constants | definition | -- | -- | -- | -- |
| `.stellarator.r_coil_major`, `r_coil_minor` | `stellarator_scaling_factors` | similarity scaling of the reference coil set by `rmajor/rmajor_ref` (and `f_st_coil_aspect` for the minor radius) | `rmajor`, `f_st_coil_aspect`, `stella_config_coil_rmajor/rminor/rmajor_ref` | coil envelope / plasma-coil clearance fixed by rigid scaling, no independent knob | definition | -- | -- | -- | -- |
| `.stellarator.coilcurrent`, `f_st_i_total` | `coils.coil_current` | `I = i0·(B/B_ref)·(R/R_ref)/f_st_n_coils` — Ampere's law scaled off the reference (`f_st_n_coils ≡ 1`) | `f_st_b`, `stella_config_i0`, `f_st_rmajor`, `f_st_n_coils` | given B, R (decided elsewhere) and N (above), current follows necessarily | definition | -- | -- | -- | -- |
| `.stellarator.wp_width_r_min` | `^problem.stellarator.coils.intersect` (root-find over `.stellarator.coils.intersect`) | `w_r` solved so `j_tf_wp = f_j_tf_wp_critical_max·j_c(B_peak(w_r), tftmp+tmargmin)` | `r_coil_major/minor`, `coilcurrent`, `n_tf_coils`, `a1,a2,wp_ratio`, `tftmp`, `tmargmin`, `f_a_tf_turn_cable_copper`, `f_a_tf_turn_cable_space_extra_void`, `f_j_tf_wp_critical_max`, `a_tf_turn_cable_space_no_void`, `dx_tf_turn_general` | winding-pack cross-section — `ixc 140` in tokamak files — pinned to the critical-current fraction instead of left free | **sizing choice** | `j_tf_wp <= f_j_tf_wp_critical_max·j_c(B_peak, tftmp+tmargmin)`; safe = wider WP | **icc 33** (`I_op/I_critical (TF coil) upper limit`; verified verbatim in `constraint_equation_33`) | `ixc 140 dr_tf_wp_with_insulation` (exact-name match; the catalogue's own `itv 50` for icc 33 is dead/`NOT USED`) | **neither active** — rule-closed |
| `.tfcoil.dr_tf_wp_with_insulation` | `coils.winding_pack_total_size_post` | `dr = max(dx_tf_turn_general², wp_width_r_min)` | `wp_width_r_min`, `dx_tf_turn_general` | same decision, the WP radial thickness face | **sizing choice** | same, plus floor `dr >= dx_tf_turn_general²` | **icc 33** | `ixc 140` (exact match) | **neither active** |
| `.tfcoil.j_tf_wp` | `coils.winding_pack_total_size_post` | `j_tf_wp = I·1e6/(dr·dr/wp_ratio)` | `coilcurrent`, `wp_ratio`, (`dx_tf_turn_general` via `dr`) | the engineering current density the rule targets — the closure variable itself | **sizing choice** | **two distinct bounds on the same variable**: `icc 33` (rule-pinned, inactive) *and* `icc 35` TF quench hotspot J limit (`j_tf_wp <= j_tf_wp_quench_heat_max`), which **is** active here | icc 33 + icc 35 | none of its own (follows from `ixc 140` via the rule) | **icc 33 inactive/rule-closed; icc 35 active** (checked via `ixc 56`, `ixc 59`) — `j_tf_wp` is rule-pinned against one limit while genuinely optimiser-checked against a second, distinct one |
| `.tfcoil.dx_tf_wp_primary_toroidal`, `dx_tf_wp_secondary_toroidal` | `coils.winding_pack_total_size_post` | `dx = wp_width_r_min / stella_config_wp_ratio` | `wp_width_r_min`, `stella_config_wp_ratio` | WP toroidal width; the WP aspect ratio itself is held at the reference configuration's value | definition | -- | -- | -- | -- |
| `.tfcoil.dr_tf_plasma_case` | `coils.coil_casing` | `= dr_tf_nose_case` ("assumed constant until something better comes up") | `dr_tf_nose_case` | plasma-side case thickness, a structural choice, equated rather than sized | **sizing choice** | stands for `case thickness >= what case stress needs`, never actually asked | none — `icc 31` (TF case stress) reads `sig_tf_case`, computed only by the generic tokamak stress module, which is **not wired into this stellarator graph at all** | none catalogued for this exact place | nothing checked in this graph — the equality is the only thing that ever sets it |
| `.tfcoil.dx_tf_side_case_min` | `coils.coil_casing` | `= dr_tf_nose_case` (same placeholder) | `dr_tf_nose_case` | sidewall case thickness, same treatment | **sizing choice** | same | none (same reasoning) | `ixc 172 dx_tf_side_case_min` (exact-name match) | not active, and moot — the rule bypasses it by equality regardless |

*Corrections to `decision_kinds.md` §3.6 informed by this pass*: `.build.dr_tf_inboard` is a
different node's **sum** (`dr_tf_nose_case + dr_tf_wp_with_insulation + dr_tf_plasma_case +
2·dx_tf_wp_insulation`) of quantities already accounted for above — a consequence, not an
independent decision — and is dropped from the flagged list. `.tfcoil.e_tf_magnetic_stored_total_gj`,
`.build.z_tf_inside_half`, `len_tf_coil`, `tfcryoarea` are pure reference-scaled geometry that
"follows the coil set" (confirmed at the formula) with no sizing rule of their own, and are
likewise dropped from the flagged list (§3.6's "-- (follow the coil set)" already said as much).

### 1b. Radial/vertical build, first wall/blanket/shield/VV, divertor, vacuum (`stellarator.build`, `stellarator.fwbs.*`, `stellarator.divertor`, `vacuum.*`, `buildings.tf_coil_envelope`)

| place | node | rule | reads | reason | kind | inequality (sizing choice) | icc | ixc | this run |
|---|---|---|---|---|---|---|---|---|---|
| `.build.dr_fw_inboard`, `dr_fw_outboard` | `stellarator.build` | `2·radius_fw_channel + 2·dr_fw_wall` | `.fwbs.radius_fw_channel`, `.fwbs.dr_fw_wall` | first-wall thickness — a plain sum of two already-fixed sub-thicknesses (both already counted as build inputs in `decision_kinds.md` §3.2) | definition | -- | -- | -- | -- |
| `.build.dr_tf_outboard` | `stellarator.build` | `= dr_tf_inboard` | `.build.dr_tf_inboard` | outboard TF leg thickness collapsed to a copy of the inboard value rather than sized independently | **sizing choice** | stands for `dr_tf_outboard >= required_thickness_outboard`; safe = thicker; pinned to the (presumably more demanding) inboard value as a conservative stand-in | none | **ixc 75 `f_dr_tf_outboard_inboard`** — confirmed exact match: the general tokamak build computes `dr_tf_outboard = f_dr_tf_outboard_inboard·dr_tf_inboard` (`process/models/build.py:1895`); the stellarator branch hardcodes the ratio to `1.0` (`process/models/stellarator/initialization.py:27`) and the port inlines that, never reading the field | **not active** (75 not in the active-ixc-8) — the reference run leaves the ratio at the stellarator-forced 1.0 rather than optimising it |
| `.build.dr_shld_vv_gap_outboard` | `stellarator.build` | `= gapomin` | `.build.gapomin` | outboard shield-VV gap pinned to the minimum allowed gap instead of solved/optimised | **sizing choice** | `dr_shld_vv_gap_outboard >= gapomin`; safe = larger gap; pinned at the floor, zero slack | none — no `icc` checks this assignment; it's baked into the build formula, not a constraint residual | **ixc 31 `gapomin`** (exact match) | **not active** (31 not in the active-ixc-8); confirmed the *tokamak* build does the identical `= gapomin` in its own default (non-ripple-limited) branch too — not stellarator-specific |
| `.fwbs.life_fw_fpy` | `fwbs.fw_blanket_shield_geometry` | `min(abktflnc/pflux_fw_neutron_mw, life_plant)` | `.costs.abktflnc`, `.physics.pflux_fw_neutron_mw`, `.costs.life_plant` | blanket/FW replacement schedule — a belief (fluence limit) × an output (neutron flux), capped by plant life | **sizing choice** | `life_fw_fpy·pflux_fw_neutron_mw <= abktflnc` and `life_fw_fpy <= life_plant`; safe = shorter life (more fluence margin), pinned at whichever ceiling binds | **none at all** — `icc 53` is confirmed TF-coil-only (`flu_tf_neutron_fast_peak <= flu_tf_neutron_fast_max`, unrelated fields, checked directly in `constraint_equation_53`); `icc 85` (CP lifetime) is centrepost/ST-specific, irrelevant here, and inactive anyway | none — `abktflnc` is a plain, un-iterable input | **no catalogued analogue exists anywhere in PROCESS**; the belief-to-lifetime rule is the only place this limit lives |
| `.vacuum.n_vac_pumps_high` | `vacuum.vacuum_old` | max-over-4-gas-species of required vs. per-pump pumping speed, ×ducts, ×2 if cryopump, `floor(pumpn+0.5)` | `.physics.p_fusion_total_mw`, `.divertor.n_divertors`, `.tfcoil.n_tf_coils`, `.vacuum.pres_*`, `.vacuum.i_vac_pump_dwell`, `.vacuum.i_vacuum_pump_type` | number of vacuum pumps sized against a required pumping-speed target — a genuine equipment-count closure | **sizing choice** | `pumping_speed_installed(n) >= pumping_speed_required`; safe = more pumps | **none — icc 63 is a false match**, ruled out by reading `constraint_equation_63` directly: it bounds `n_iter_vacuum_pumps` of the *other* ("simple") vacuum dispatch arm, not this ("old") arm's `n_vac_pumps_high` | none | **no catalogued analogue** — purely internal closed-form sizing, invisible to the numerics catalogue |
| `.vacuum.dia_vv_vacuum_ducts` | `vacuum.vacuum_old` | Newton root-find for duct diameter matching conductance to the governing species' target, shrinking the target 10% at a time until it fits between TF coils | `.vacuum.l1/l2/l3`, `xmult_i`, `a1max`, required speeds | duct diameter sized by an internal root-find against a pumping-speed/fit target | **sizing choice** | `duct_conductance(d) >= required` and `0.25π·d² <= a1max`; safe = larger `d` | none | none | **no catalogued analogue** |
| `.vacuum.d_duct` (owned by `^problem.vacuum.duct_diameter_root_find`) | `^problem.vacuum.duct_diameter_root_find` | driven root-find `duct_conductance(d,l1,l2,l3,xmult_i) = ceff_i` | `^cond.vacuum.d_duct`, `^guess.vacuum.d_duct` | the declared/drivable parallel representation of the same duct-diameter decision | **sizing choice** | `duct_conductance(d) >= ceff_i`; safe = larger `d` | none | none | **no catalogued analogue** (same closure as above, cottax's driven form of it) |

Reviewed and confirmed **not** a decision, contrary to first appearance: `.divertor.a_div_surface_total`
(wetted-area consequence of the island-divertor heat-flux model, not a chosen dimension);
`.build.required_radial_space`/`available_radial_space` (pure geometric sums, the constraint-9
comparison); `.buildings.tf_coil_envelope`'s 5 outputs (pure derived geometry/mass); `.build.rspo`
(pinned to `rmajor`, a geometric simplification, not an independent pick); `.vacuum.n_vv_vacuum_ducts`
(`= n_tf_coils × n_divertors`, a count following from two decisions counted elsewhere).

### 1c. Plant equipment count (`power.component_thermal_powers`)

| place | node | rule | reads | reason | kind | inequality | icc | ixc | this run |
|---|---|---|---|---|---|---|---|---|---|
| `.heat_transport.n_primary_heat_exchangers` | `power.component_thermal_powers` | `ceil(p_plant_primary_heat_mw / 1000.0)` | `.heat_transport.p_plant_primary_heat_mw` | number of primary heat-exchanger units to cover the primary heat load; the ceiling pins the count at zero slack on the last unit | **sizing choice** | `n_primary_heat_exchangers × 1000 MW >= p_plant_primary_heat_mw`; safe = more units | **none** | **none** | **no catalogued analogue at all**, on either the tokamak or stellarator side. `n_primary_heat_exchangers` has exactly one other consumer in the whole codebase — the heat-exchanger capital-cost formula (`cost ∝ n·(load/n)^exphts`) — so this "sizing" feeds only a cost estimate, never a policed constraint or a free variable |

### 1d. Initialisation / pinning (`initialisation.*`)

| place | node | rule | reads | reason | kind | inequality | icc | ixc | this run |
|---|---|---|---|---|---|---|---|---|---|
| `.build.dr_cs`, `dr_cs_tf_gap` | `initialisation.stellarator_solenoid_absent` | pinned to 0 (both) | `^stated.build.dr_cs`, `^stated.build.dr_cs_tf_gap` | stellarator has no central solenoid to size — structural absence, not a value pinned at an optimality boundary | definition | -- | -- | -- | -- |
| `.buildings.esbldgm3` | `initialisation.energy_storage_building_volume` | pinned to 0 | `^stated.buildings.esbldgm3` | forced to 0 whenever `i_pulsed_plant = 0` (`process/core/init.py:827`) — no energy-storage building because the plant is steady-state, not pulsed (new finding, same pin pattern as `dr_cs = 0`) | definition | -- | -- | -- | -- |

## 2. Operating decisions closed by a rule

| place | node | rule | reads | reason | kind | inequality | icc | ixc | this run |
|---|---|---|---|---|---|---|---|---|---|
| `.times.t_plant_pulse_burn` (`^stated`) | `initialisation.stellarator_pulse_times` | pinned to `3.15576e7` s (1 year) | `^stated.times.t_plant_pulse_burn` | continuous operation encoded as one full-year "pulse"; the other three phase durations (precharge/ramp-up/ramp-down) are forced to 0 alongside it | definition | -- | -- | -- | -- |

That is the **only** operating decision found among all 544 owned places. See §4 for why the two
plausible extra candidates — blanket/divertor lifetime and vacuum-pump-count — are classed as
build (component design properties, materials/fluence-driven) rather than operating in this
report, against one contributing agent's initial call.

Nothing else in plasma physics, power flow, buildings, or costs sets a temperature, density,
fuelling rate, heating power, coolant setting, or availability by a closure formula — the
operating-type quantities that appear in formulas throughout the graph (`tdiv`, `tftmp`,
`f_p_*_coolant_pump_total_heat`, `pres_vv_chamber_base`, etc.) are all **read**, never **owned**,
by any node. `.physics.molflow_plasma_fuelling_required` (`rndfuel/burnup`) was checked
specifically — it is a pure physics consumption-rate consequence, no duty-cycle or margin factor
is added, so it is a definition of no decision at all (unflagged).

## 3. Counts

| | places |
|---|---|
| Total owned places (all 156 nodes) | **544** |
| Flagged **build** | **27** |
| Flagged **operating** | **1** |
| Total flagged | **28** |
| Unflagged (state / consequence / money / geometry with no independent discretion) | **516** |

By subsystem (owned places reviewed / flagged):

| subsystem | nodes | places reviewed | flagged |
|---|---|---|---|
| TF coil / magnet | 28 | 114 (34 are `machine_config`'s frozen HELIAS constants, not a rule output — see below) | 13 |
| Radial/vertical build, FW/blanket/shield/VV, divertor, vacuum | 13 | 74 | 8 |
| Plasma physics, stellarator misc. geometry/heating, power flow | 66 | 206 | 1 |
| Costs, buildings, availability, initialisation | 47 | 147 | 6 (5 build after the reclassification in §4, 1 operating) |
| Internal fixed-point cut copies (`^problem.physics.proton_rate_density.cycle`, `^problem.fwbs.f_ster_div_single`) | 2 | 3 | 0 — mechanical `^hat` plumbing for a self-referential physics cycle, not decisions |
| **Total** | **156** | **544** | **28** |

`.stellarator.machine_config` (34 places) copies HELIAS-5B reference-configuration constants
through with no reads at all — closer to a frozen belief input (like `kappa` in
`decision_kinds.md` §3.2) than a decision closed by a rule, since no formula chooses them in this
run. Not counted as flagged or as reviewed-and-cleared; noted separately because every
"definition (similarity)" row in §1a inherits its shape from these constants.

### Definition vs. sizing choice

| | count |
|---|---|
| Flagged **definitions** (no slack; not a chance-constraint candidate) | **14** |
| Flagged **sizing choices** (an inequality pinned at zero slack) | **14** |

The 14 sizing choices: `wp_width_r_min`, `dr_tf_wp_with_insulation`, `j_tf_wp`, `dr_tf_plasma_case`,
`dx_tf_side_case_min` (TF coil, 5); `dr_tf_outboard`, `dr_shld_vv_gap_outboard`, `life_fw_fpy`,
`n_vac_pumps_high`, `dia_vv_vacuum_ducts`, `d_duct` (build/vacuum, 6); `n_primary_heat_exchangers`
(power, 1); `life_blkt_fpy`, `life_div_fpy` (availability, 2 — see §4).

### The `icc`/`ixc` audit of the 14 sizing choices

| | sizing choices |
|---|---|
| With a catalogued **icc** (PROCESS constraint equation expresses the same inequality) | **3** — `wp_width_r_min`, `dr_tf_wp_with_insulation` (both icc 33), `j_tf_wp` (icc 33 **and** icc 35) |
| With a catalogued **ixc** (PROCESS iteration variable for the quantity) | **5** — the 3 above's `dr_tf_wp_with_insulation`/`wp_width_r_min` share ixc 140; plus `dx_tf_side_case_min` (ixc 172), `dr_tf_outboard` (ixc 75), `dr_shld_vv_gap_outboard` (ixc 31) |
| With **either** a catalogued icc or ixc | **6** (union of the two rows above; `j_tf_wp` has no ixc of its own, `dx_tf_side_case_min`/`dr_tf_outboard`/`dr_shld_vv_gap_outboard` have no icc) |
| With **neither** — no PROCESS catalogue analogue exists at all | **8** — `dr_tf_plasma_case`, `life_fw_fpy`, `life_blkt_fpy`, `life_div_fpy`, `n_vac_pumps_high`, `dia_vv_vacuum_ducts`, `d_duct`, `n_primary_heat_exchangers` |
| **Actually activated** by the reference run (checked by the optimiser rather than closed by the rule) | **1 of 14**, and only partially — `j_tf_wp` is checked against icc 35 (active, via ixc 56/59), while its own defining rule (icc 33 / ixc 140) is still rule-closed. Every other sizing choice is left **entirely** to the rule. |

This is the concrete evidence for the claim under test: of the machine's genuine sizing
decisions, PROCESS's own catalogue already has the vocabulary for barely more than a third
(6 of 14, and only via `icc`/`ixc` numbers this run switches off) — the rest (8 of 14, including
every replacement-lifetime rule and the entire vacuum-pumping subsystem) have **no catalogued
constraint or iteration variable at all**; the belief/output arithmetic in the model is the only
place the limit lives, active or not.

## 4. Borderline cases

- **`j_tf_wp`'s two bounds.** The winding-pack current density is simultaneously rule-pinned
  against the critical-current fraction (icc 33, inactive here) and independently checked against
  a distinct quench-hotspot limit (icc 35, active here via `t_tf_superconductor_quench` and
  `f_a_tf_turn_cable_copper`). Not a double-count: these are two different physical limits on one
  variable, and the run only ever asks the optimiser about the second.
- **`dr_tf_outboard = dr_tf_inboard`.** The "conservative stand-in" reading is this report's
  inference, not something the source code states — no independent outboard structural
  requirement is computed anywhere in the port. Confirmed, though, that PROCESS's tokamak build
  has a real free ratio (`ixc 75 f_dr_tf_outboard_inboard`) for exactly this quantity, and the
  stellarator branch hardcodes it to 1.0 rather than exposing it — so "sizing choice, currently
  rule-closed" is the right call even though the inequality itself is not literally stated in the
  formula.
- **`dr_tf_plasma_case` / `dx_tf_side_case_min`.** The weakest sizing-choice flags: the formula is
  a placeholder equality (`= dr_tf_nose_case`), not a computed sizing rule, and no `icc` is even
  wired to check case stress in this stellarator graph at all. Kept flagged because the quantity
  is unmistakably a build decision and this equality is the only thing that ever sets it — but
  there is no inequality in the source to point to, only the one the placeholder logically stands
  for.
- **Blanket/divertor lifetime — reclassified from operating to build.** One contributing pass
  classified `.fwbs.life_blkt_fpy` and `.costs.life_div_fpy` as *operating* decisions (reading
  "availability" broadly). This report reclassifies both as **build**, for consistency with the
  already-established treatment of the structurally identical `.fwbs.life_fw_fpy` (already build
  in `decision_kinds.md` §3.6) and of `.costs.life_plant` (build, §3.2: "a design/financing
  choice"). A component's fluence-limited replacement lifetime is a design property computed from
  a materials belief, not a knob an operator turns with the machine in hand; `.costs.cpfact` (the
  actual operating-style *capacity factor*) was correctly left unflagged by every pass as a pure
  propagation of `f_t_plant_available`. Verified (independently, by the agent that found them):
  no `icc`/`ixc` polices either quantity, for the identical reason `life_fw_fpy` has none — `abktflnc`
  and `adivflnc` are plain, un-iterable belief inputs (`process/core/input.py`), and `icc 8`/`icc 18`
  (both active here) bound the instantaneous flux/heat-load *rate*, not the accumulated fluence or
  the lifetime it produces.
- **`.buildings.esbldgm3` pinned to 0.** A second instance of the `dr_cs = 0` pattern
  (`decision_kinds.md` §3.2/§3.6 already have the solenoid case) — not previously listed. Kept as
  a definition (a structural absence: no pulsed-plant energy-storage building for a steady-state
  machine), not a sizing choice, since there is no inequality being pinned, only a switch
  (`i_pulsed_plant`) being read as false.
- **`n_primary_heat_exchangers`.** A genuine integer equipment-count decision (a `ceil()`) with
  zero downstream consequence beyond the cost model — it is never fed back into a design
  constraint or exposed as a free variable anywhere in PROCESS. Flagged because the definition
  ("a count sized by a rule") is met exactly, even though the design impact is limited to cost
  accounting in this run.
- **`.stellarator.coilcurrent`.** Classed as a *definition* (Ampere's law, `I ∝ B·R/N`, scaled off
  the reference configuration) rather than a build decision needing a rule at all — kept in the
  flagged table only because `decision_kinds.md` §3.6 already listed it and this report extends
  rather than silently drops established entries; under the strict "does the formula set something
  a person would otherwise freely choose" test it arguably shouldn't be flagged, since B, R and N
  are each decided (or already flagged) elsewhere and the current follows necessarily once they
  are fixed.
- **`.build.dr_tf_inboard`, `e_tf_magnetic_stored_total_gj`, `z_tf_inside_half`, `len_tf_coil`,
  `tfcryoarea`.** Dropped from the flagged set relative to `decision_kinds.md` §3.6's original
  (coarser) grouping: each is confirmed, at the formula, to be a pure consequence — a sum of
  already-counted thicknesses, or reference-configuration geometry scaled by `r_coil_minor` alone
  — with no sizing discretion of its own beyond what is already flagged in §1a.
- **Vacuum pump/duct count vs. `icc 63`.** Worth stating plainly: `icc 63` ("ITER-like vacuum pump
  number upper limit") sounds like exactly the catalogued analogue for `n_vac_pumps_high`, and was
  checked specifically for that reason — it turns out to bound a variable from PROCESS's *other*
  ("simple") vacuum-pumping dispatch arm, an unrelated model this stellarator graph does not use
  at all. The port's actual pump-sizing rule (the "old" arm) has no catalogued analogue,
  confirmed by reading the constraint body directly rather than trusting the name.

## Five most consequential flags for a robust design

1. **TF winding-pack sizing** (`wp_width_r_min` / `dr_tf_wp_with_insulation` / `j_tf_wp`,
   §1a) — the coil cross-section, current density and quench margin are entirely closed by the
   critical-current-fraction rule (icc 33/ixc 140, both inactive); only a *second*, distinct
   quench-hotspot bound (icc 35) is ever checked by the optimiser. The single biggest hidden
   sizing decision in the machine.
2. **Component replacement lifetimes** (`life_fw_fpy`, `life_blkt_fpy`, `life_div_fpy`, §1b/§4) —
   sized to fluence/heat-load beliefs (`abktflnc`, `adivflnc`) with **no PROCESS catalogue
   machinery at all**; if the belief is wrong, plant economics (COE, availability, `bktcycles`)
   move with no chance constraint to catch it.
3. **Vacuum pumping equipment** (`n_vac_pumps_high`, `dia_vv_vacuum_ducts`, `d_duct`, §1b) — an
   entire pump-count and duct-diameter sizing subsystem, invisible to PROCESS's own numerics
   catalogue (`icc 63` turned out to be a different model entirely).
4. **Radial-build thicknesses pinned to a copy or a floor** (`dr_tf_outboard = dr_tf_inboard`,
   `dr_shld_vv_gap_outboard = gapomin`, §1b) — both correspond to real PROCESS iteration variables
   (`ixc 75`, `ixc 31`) that exist and are switched off in this run; the tightest structural
   margins in the build are set to a copy or a bare floor by a hardcoded rule rather than checked.
5. **Primary heat-exchanger count** (`n_primary_heat_exchangers`, §1c) — a real integer equipment
   decision made silently by a `ceil()`, feeding only the cost model with no policing anywhere in
   PROCESS.
