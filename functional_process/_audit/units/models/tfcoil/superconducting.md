---
kind: model-unit
status: draft
confidence: high
---

**Not registered** — same note as `base.md`. The registry rows and `total_process.py`
bindings this record needs are named in the porting agent's report.

## source

`process/models/tfcoil/superconducting.py`, **partial**: `CICCSuperconductingTFCoil`
and the `SuperconductingTFCoil` layer above it, restricted to the minimal closure of
`.tokamak.cicc_superconducting_tf_coil`'s ten boundary reads
(`_audit/tokamak_boundary.md`). The file is 5153 lines; the closure is eight functions.

| in scope | lines | shape |
|---|---|---|
| `superconducting_tf_wp_geometry` | 1558–1793 | `@staticmethod`, pure; `i_tf_wp_geom` (3 arms) |
| `superconducting_tf_case_geometry` | 1795–1956 | `@staticmethod`, pure; two independent switches over disjoint outputs |
| `tf_wp_currents` | 1958–1970 | `@staticmethod`, takes `DataStructure`, 3 reads / 1 write |
| `peak_b_tf_inboard_with_ripple` | 1454–1556 | instance method; writes 3 fields as side effects; branches on `round(n_tf_coils)` |
| `tf_cable_in_conduit_averaged_turn_geometry` | 3238–3420 | `@staticmethod`, pure; two boolean switches |
| `tf_cicc_inboard_areas_and_fractions` | 3599–3670 | `@staticmethod`, pure, no switch |
| `superconducting_tf_coil_areas_and_masses` | 1972–2093 | instance method, `data` in / `data` out; `itart` |
| `run`'s inline `a_tf_turn` | 2700–2704 | one division written in `run` |

Everything excluded, and why, is in the port module's docstring. The two exclusions
worth repeating here:

- **The superconductor critical-surface physics is out of scope because it is already
  ported.** `tf_cable_in_conduit_superconductor_properties` (`:2806`) and
  `calculate_superconductor_temperature_margin` (`:1174`) dispatch on `i_tf_sc_mat` into
  fits that `functional_process/models/physics/superconductors.py` already carries, one
  class per material. Nothing on this slot's boundary reads their outputs, so they were
  not needed; whoever needs them next reuses that module rather than re-porting a fit.
- **`CROCOSuperconductingTFCoil` is a different occupant of the same slot**, selected by
  `.superconducting_tfcoil.i_tf_turn_type == 2` at `process/core/caller.py:307-313`.
  The default and the reference run take `1`.

## data footprint

Reference run: `large_tokamak_eval.IN.DAT`. Live switch values:
`i_tf_turns_integer = 0` (default, `tfcoil_variables.py:240`), hence
`i_tf_wp_geom = 1` — resolved from the `-1` `UNSET` default by
`process/core/init.py:977-989`; `i_tf_case_geom = 0`; `i_tf_sc_mat = 1`
(`IN.DAT:374`); `n_tf_coils = 16` (`IN.DAT:377`); `itart = 0`;
`i_dx_tf_turn_general_input = False` and `i_dx_tf_turn_cable_space_general_input = False`
(defaults, `:108,127`); `i_str_wp` is **not reached** — see §switches.

### `superconducting_tf_wp_geometry` (`:1558-1793`)

Reads (all `explicit-arg`, all live): `.build.r_tf_inboard_in`,
`.tfcoil.dr_tf_nose_case`, `.tfcoil.dr_tf_wp_with_insulation`,
`.superconducting_tfcoil.tan_theta_coil`, `.tfcoil.dx_tf_side_case_min`,
`.tfcoil.dx_tf_wp_insulation`, `.tfcoil.dx_tf_wp_insertion_gap`. **Identical on all
three arms.**

Writes (all `explicit-arg`): `.superconducting_tfcoil.{r_tf_wp_inboard_inner,
r_tf_wp_inboard_outer, r_tf_wp_inboard_centre, dx_tf_wp_toroidal_min,
dr_tf_wp_no_insulation, dx_tf_wp_toroidal_average, a_tf_wp_with_insulation,
a_tf_wp_no_insulation, a_tf_wp_ground_insulation}` and
`.tfcoil.{dx_tf_wp_primary_toroidal, dx_tf_wp_secondary_toroidal}`.

The `RECTANGULAR` arm sets `dx_tf_wp_secondary_toroidal` to the same value as the
primary with the source's own comment *"No secondary WP here but will set for
consistency"* (`:1632`) — carried forward, not normalised away.

### `superconducting_tf_case_geometry` (`:1795-1956`) — split in two

Two switches, and their output groups do not overlap, so the port is two node families
rather than one node carrying both:

| output | switch | reads |
|---|---|---|
| `.tfcoil.a_tf_coil_inboard_case`, `.tfcoil.a_tf_coil_outboard_case`, `.superconducting_tfcoil.a_tf_plasma_case`, `.superconducting_tfcoil.a_tf_coil_nose_case` | `i_tf_case_geom` | CIRCULAR reads `.build.r_tf_inboard_out`; STRAIGHT reads `.tfcoil.dr_tf_plasma_case` instead. Shared: `a_tf_inboard_total`, `n_tf_coils`, `a_tf_wp_with_insulation`, `a_tf_leg_outboard`, `rad_tf_coil_inboard_toroidal_half`, `tan_theta_coil`, `r_tf_wp_inboard_outer`, `r_tf_wp_inboard_inner`, `r_tf_inboard_in` |
| `.superconducting_tfcoil.dx_tf_side_case_average`, `.tfcoil.dx_tf_side_case_peak` | `i_tf_wp_geom` | RECTANGULAR and DOUBLE_RECTANGULAR read `dx_tf_side_case_min`, `tan_theta_coil`, `dr_tf_wp_with_insulation`; **TRAPEZOIDAL reads only `dx_tf_side_case_min`** |

Two measured invented edges removed: the composite node would have declared
`.tfcoil.dr_tf_plasma_case` on the circular arm (which never reads it) and
`tan_theta_coil`/`dr_tf_wp_with_insulation` on the trapezoidal sidewall arm (which
never reads them).

### `tf_wp_currents` (`:1958-1970`)

Reads `.tfcoil.c_tf_total`, `.tfcoil.n_tf_coils`,
`.superconducting_tfcoil.a_tf_wp_no_insulation`; writes `.tfcoil.j_tf_wp`. All
`explicit-arg`.

**No self-loop on `.tfcoil.j_tf_wp` on the tokamak path.** The stellarator port spends a
long comment on whether that field needs a `FixedPointFunction`
(`models/stellarator/namespace.py:227-259`); here the question is settled by reading the
body — the entering `j_tf_wp` is never consulted. PROCESS's own two test cases
(`test_tf_wp_currents`, identical but for the entering value, identical in expectation)
are independent confirmation.

### `peak_b_tf_inboard_with_ripple` (`:1454-1556`)

| VarPath | read/write | classification |
|---|---|---|
| `.tfcoil.n_tf_coils` | read | explicit-arg — *(live)* used **both** as the branch selector (`round`, `:1500`) and continuously (`tan(pi/n)`, `:1526`) |
| `.tfcoil.dx_tf_wp_primary_toroidal` | read | explicit-arg — fitted arms only |
| `.superconducting_tfcoil.dr_tf_wp_no_insulation` | read | explicit-arg — fitted arms only |
| `.superconducting_tfcoil.r_tf_wp_inboard_centre` | read | explicit-arg — fitted arms only |
| `.tfcoil.b_tf_inboard_peak_symmetric` | read | explicit-arg — every arm |
| `.superconducting_tfcoil.tf_fit_t` | write | explicit-arg — fitted arms only |
| `.superconducting_tfcoil.tf_fit_z` | write | explicit-arg — fitted arms only |
| `.superconducting_tfcoil.f_b_tf_inboard_peak_ripple_symmetric` | write | local-intermediate → returned one line later (`:1556`) |
| `.tfcoil.b_tf_inboard_peak_with_ripple` | write | explicit-arg — every arm |

The `else` arm returns at `:1519`, **before** the three `superconducting_tfcoil` fields
are assigned, so it owns one output where its siblings own four. Conditional ownership
that a static kwarg could not have expressed at all.

`b_tf_inboard_peak_with_ripple` is **not** one of the ten boundary reads. It is ported
anyway because it completes `run_base_superconducting_tf`'s field chain and is what
`quench_heat_protection_current_density`'s CoolProp half will need.

### `tf_cable_in_conduit_averaged_turn_geometry` (`:3238-3420`)

Reads on the live (both-flags-`False`) arm: `.tfcoil.j_tf_wp`, `.tfcoil.c_tf_turn`,
`.tfcoil.dx_tf_turn_steel`, `.tfcoil.dx_tf_turn_insulation`, `.tfcoil.layer_ins`,
`.superconducting_tfcoil.a_tf_wp_no_insulation`,
`.tfcoil.dia_tf_turn_coolant_channel`,
`.tfcoil.f_a_tf_turn_cable_space_extra_void`.

Writes: `.tfcoil.{a_tf_turn_cable_space_no_void, a_tf_turn_steel, a_tf_turn_insulation,
n_tf_coil_turns, dx_tf_turn_general, dx_tf_turn_conduit_full_average}` and
`.superconducting_tfcoil.{dr_tf_turn, dx_tf_turn, radius_tf_turn_cable_space_corners,
dx_tf_turn_cable_space_average, a_tf_turn_cable_space_effective,
f_a_tf_turn_cable_space_cooling}` — `.tfcoil.n_tf_coil_turns` is **boundary read #8**.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.c_tf_turn` | **read** | conditional-ownership-by-run-config | *(live)* on this arm the parameter is returned unchanged (`:3411`) and `run` writes it back (`:2372`) — an identity |
| `.tfcoil.dx_tf_turn_general` | read on two arms, write on all three | conditional-ownership-by-run-config | on the live arm it is `sqrt(a_tf_turn)`, computed, not read |

**Finding 1, contradicting `tokamak_boundary.md`. `.tfcoil.c_tf_turn` is a run input on
this configuration, not an output of this slot.** The boundary table lists it as one of
the ten variables the slot should produce. It is not produced: it is set explicitly at
`tests/regression/input_files/large_tokamak_eval.IN.DAT:371` (`c_tf_turn =
85462.67500253802`) and is iteration variable 60
(`process/core/solver/iteration_variables.py:78`), i.e. an optimiser unknown. Only the
`i_dx_tf_turn_general_input` and `i_dx_tf_turn_cable_space_general_input` arms compute it
(`:3310,3323`), and both are `False` here. Registering it as a boundary `input` is the
correct outcome; `CiccAveragedTurnGeometryFromCurrentPerTurn` reads it and does not own
it. **This reduces the slot's producible set from ten to nine.**

### `tf_cicc_inboard_areas_and_fractions` (`:3599-3670`)

Ten reads, nine writes, no switch, all `explicit-arg`. Writes
`.tfcoil.{a_tf_wp_coolant_channels, a_tf_wp_conductor, a_tf_wp_extra_void,
a_tf_coil_wp_turn_insulation, a_tf_wp_steel}` and
`.superconducting_tfcoil.{a_tf_coil_inboard_steel, f_a_tf_coil_inboard_steel,
a_tf_coil_inboard_insulation, f_a_tf_coil_inboard_insulation}`.

### `superconducting_tf_coil_areas_and_masses` (`:1972-2093`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.len_tf_coil` | read | explicit-arg | *(live)* |
| `.superconducting_tfcoil.a_tf_wp_with_insulation`, `.a_tf_wp_no_insulation` | read | explicit-arg | *(live)* |
| `.tfcoil.den_tf_wp_turn_insulation`, `.den_tf_coil_case` | read | explicit-arg | *(live)* |
| `.build.z_tf_inside_half`, `.build.dr_tf_inboard` | read | explicit-arg | *(live)* |
| `.tfcoil.a_tf_coil_inboard_case`, `.a_tf_coil_outboard_case` | read | explicit-arg | *(live)* |
| `.tfcoil.n_tf_coil_turns`, `.a_tf_turn_cable_space_no_void`, `.f_a_tf_turn_cable_space_extra_void`, `.f_a_tf_turn_cable_copper`, `.a_tf_wp_coolant_channels`, `.a_tf_turn_steel`, `.a_tf_coil_wp_turn_insulation`, `.n_tf_coils` | read | explicit-arg | *(live)* |
| `.tfcoil.dcond[i_tf_sc_mat - 1]` | read | explicit-arg | *(live)* array element; see below |
| `.fwbs.den_steel` | read | explicit-arg | *(live)* |
| `.tfcoil.cplen` | write, then **read** | local-intermediate | written `:1989`, read back `:2002,2012-2013` in the same straight-line body |
| `.tfcoil.m_tf_coil_wp_insulation`, `.m_tf_coil_case`, `.m_tf_coil_superconductor`, `.m_tf_coil_copper`, `.m_tf_wp_steel_conduit`, `.m_tf_coil_wp_turn_insulation`, `.m_tf_coil_conductor`, `.m_tf_coil`, `.m_tf_coils_total` | write | explicit-arg | four of these are boundary reads #4, #5, #6, #7 |
| `.tfcoil.whtcp`, `.whttflgs` | write | conditional-ownership-by-run-config | `itart == 1` only (`:2086-2093`) |

`.tfcoil.dcond[i_tf_sc_mat - 1]` is an **array-element `VarPath`** per
`naming_convention.md` § "Array elements", the same treatment
`models/stellarator/coils/mass.py::CoilsMass` already gives it: `dcond` is a real
nine-element field (`tfcoil_variables.py:157-170`), the index is a topology switch
resolved at graph-build time, so no lookup node is minted. `SuperconductingTfCoilAreas
AndMassesConventional` binds `dcond[0]` (`i_tf_sc_mat == 1`, ITER Nb3Sn); another
material needs a sibling overriding that one `FromExactly`.

### `run`'s inline `.tfcoil.a_tf_turn` (`:2700-2704`)

Reads `.tfcoil.c_tf_total`, `.tfcoil.j_tf_wp`, `.tfcoil.n_tf_coils`,
`.tfcoil.n_tf_coil_turns`; writes `.tfcoil.a_tf_turn`. All `explicit-arg`.

## proposed signature(s)

Shipped in `functional_process/models/tfcoil/superconducting.py`. Fifteen pure
functions; see that module.

## cottax node

Twenty-one classes in seven families:
`SuperconductingTfWpGeometry{Rectangular,DoubleRectangular,Trapezoidal}`,
`TfCaseAreas{CircularFront,StraightFront}`,
`DxTfSideCase{Rectangular,DoubleRectangular,Trapezoidal}`, `TfWpCurrents`,
`PeakBTfInboardWithRipple{16Coils,18Coils,20Coils,FlatAllowance}`,
`CiccAveragedTurnGeometryFromCurrentPerTurn`, `CiccIntegerTurnGeometry`
(2026-08-27), `CiccInboardAreasAndFractions`, `TfTurnArea`,
`SuperconductingTfCoilAreasAndMassesConventional`. The turn-geometry family base is
`CiccTurnGeometry` (`i_tf_turns_integer` first), with `CiccAveragedTurnGeometry` as
the averaged sub-family under it.

## tier signal

**Tier 1** throughout. No internal solve, no call to another `Model`.

## switches touched

| switch | values | ported | note |
|---|---|---|---|
| `i_tf_wp_geom` | 0, **1** *(live)*, 2 | all three | `-1` `UNSET` never reaches a model; `init.py:977-989` resolves it from `i_tf_turns_integer` |
| `i_tf_case_geom` | **0** *(live)*, 1 | both | |
| `round(n_tf_coils)` | **16** *(live)*, 18, 20, other | all four | treated as a switch: the arms select different fit coefficients **and** different reads |
| `i_dx_tf_turn_general_input` × `i_dx_tf_turn_cable_space_general_input` | **(F, F)** *(live)*, (T, ·), (F, T) | (F, F) only | the other two own `.tfcoil.c_tf_turn`; see finding 1 |
| `itart` | **0** *(live)*, 1 | `0` only | `1` additionally owns `whtcp`/`whttflgs` |
| `i_tf_sc_mat` | **1** *(live)*, 2–9 | `1` only | one occupant per material, differing in one `FromExactly` |
| `i_tf_turns_integer` | **0** *(live)*, 1 | both (`1` since 2026-08-27) | `1` selects `tf_cable_in_conduit_integer_turn_geometry` (`:3422-3598`), a different function; `low_aspect_ratio_DEMO`'s arm — see the dated section below |
| `.superconducting_tfcoil.i_tf_turn_type` | **1** *(live)*, 2 | `1` (CICC) | `2` is the whole CROCO class; resolved in `caller.py`, above every model |
| `i_tf_sup` | **1** *(live)*, 0, 2 | `1` | resolved in `caller.py:295-316`, above every model — `schema.md`'s "resolved above this file" |

**`i_str_wp` is not reached by anything in this port.** Its only site in the closure's
neighbourhood is `run` at `:2744-2747`, choosing between `.tfcoil.str_tf_con_res` and
`.tfcoil.str_wp` as the strain fed to
`calculate_superconductor_temperature_margin` — a function this scope excludes because
no boundary read depends on it. So `i_str_wp` is an open switch for whoever ports the
critical-current chain, not one this unit answers. Named explicitly because the
dispatch brief asked about it.

## calls into other models

None within scope. Out of scope, `run` calls `stresscl` (inherited from `TFCoil`) and
the superconductor property functions, and reads `.pf_coil.*` for the former only.

## JAX-difficulty flags

- `needs-lax-cond-or-where` — **minor**. `tf_wp_currents`' `max(1.0, ...)` (`:1963`) and
  the mass chain's `max(0.0, m_tf_coil_copper)` (`:2043`) become `jnp.maximum`. The
  degenerate-cable fallback (`:3384-3399`) becomes `jnp.where`; see D2.
- `x ** p` at zero — **workaround-known**. `sqrt(a_tf_turn)` (`:3334`) and
  `sqrt(layer_ins**2 + 4*a_tf_turn)` (`:3344`) use `safe_math.safe_sqrt`.
- **No `non-traceable-external-call` in this file.** Everything CoolProp-adjacent is in
  `quench.md`.

## defects found

- **D2.** `tf_cable_in_conduit_averaged_turn_geometry`'s degenerate-cable fallback
  (`:3384-3399`) fires *after* `a_tf_turn_cable_space_effective` and
  `f_a_tf_turn_cable_space_cooling` have been computed from the pre-fallback
  `a_tf_turn_cable_space_no_void` (`:3371-3382`), but *before* `a_tf_turn_steel`
  (`:3402`). So when the cable space comes out non-positive, two of the returned
  quantities keep the uncorrected value and one uses the corrected one. Almost certainly
  unintended; ported exactly as written, with a `jnp.where`, because the harness's job is
  agreement and not repair.
- **D3.** Two different copper densities in one coil model:
  `process/core/constants.py:289` has `DEN_COPPER = 8900.0` (used by the mass chain,
  `:2042`) while `process/models/tfcoil/quench.py:18` has `COPPER_DENSITY = 8960.0`
  (used by the quench integrand). See `quench.md`.
- `superconducting.py:2371` and `:2373` assign `self.data.tfcoil.dx_tf_turn_general` from
  the same source twice — a `redundant-duplicate-write`, cosmetic, one write in the port.

## open questions

1. **Does the orchestrator want `.tfcoil.c_tf_turn` produced anyway?** Finding 1 says
   nothing produces it on this configuration and it is iteration variable 60. If the
   assembled tokamak graph is expected to *close* over it rather than take it as a
   boundary input, that is an `Optimise` unknown, not a missing node — but it changes
   `tokamak_boundary.md`'s count for this slot from ten produced to nine.
2. **`b_tf_inboard_peak_with_ripple`'s `round(n_tf_coils)` split scales badly if
   `n_tf_coils` ever becomes an optimiser unknown.** It is not one today
   (`iteration_variables.py` does not list it), which is what makes a build-time branch
   legitimate. Worth a note in `core/solver/switches.md` when that file next gets a pass.

## 2026-08-27: the integer-turn arm, after `low_aspect_ratio_DEMO`'s silent mis-assembly

**The defect.** `low_aspect_ratio_DEMO.IN.DAT:954` sets `i_tf_turns_integer = 1`
(`:958` `n_tf_wp_pancakes = 20`, `:962` `n_tf_wp_layers = 10`). PROCESS's `run`
dispatches on it (`process/models/tfcoil/superconducting.py:2343-2439`: `NON_INTEGER`
-> `tf_cable_in_conduit_averaged_turn_geometry`, else ->
`tf_cable_in_conduit_integer_turn_geometry`, `:3423`). The factory read the switch
**only** for the `i_tf_wp_geom` `UNSET` resolution (`_tf_wp_geom`), so `machine_survey`
reported *"the factory dispatches on it"* while `_cicc_turn_geometry_arm` consulted only
the two `i_dx_*_input` booleans and silently kept
`CiccAveragedTurnGeometryFromCurrentPerTurn` -- `next_steps.md` §14.11's failure mode in
a survey-blind variant: a switch deciding two slots was wired to one. The warm MDA
harness caught it as a 25-variable cluster: the port computed a **square** turn
(`dr = dx = 0.05685`) where PROCESS's converged turn is **rectangular**
(`dr = 0.05467 x dx = 0.05910`, rel 4e-2), cascading at 1e-3..1e-5 into the WP areas,
the coil masses, `.costs.c222x`, `.power.qss`, the TF nuclear heating renormalisation
and the structure masses.

**`m_tf_coil_superconductor`'s 42 % is the same root cause through a near-cancellation,
not a second defect.** The mass formula (`:2026-2033`) is
`(len*n_turns*a_cable_no_void*(1-f_void)*(1-f_cu) - len*a_wp_coolant_channels)*dcond`.
With this file's `f_a_tf_turn_cable_copper = 0.90481` (`IN.DAT:265`) and
`dcond[4] = 6080` (`i_tf_sc_mat = 5`), the converged terms are 0.95362 m3 against
0.95106 m3 -- a 99.7 % cancellation leaving 2.56e-3 m3 (15.57 kg beside 72.2 t of
copper). The port's 1.3e-3 relative error on `a_tf_turn_cable_space_no_void` shifted
the first term by ~1.2e-3 m3, which against that residual is the observed 42 %. After
the fix the variable agrees; nothing was tuned.

**The port.** `cicc_integer_turn_geometry` (pure) + `CiccIntegerTurnGeometry`
(occupant), beside the averaged sibling; family base renamed `CiccTurnGeometry`
(`i_tf_turns_integer` answered first), slot renamed
`.tokamak.cicc_superconducting_tf_coil.cicc_turn_geometry` to stop saying "averaged"
about a slot that no longer is. Reads: `.tfcoil.{dr_tf_wp_with_insulation,
dx_tf_wp_insulation, dx_tf_wp_insertion_gap, n_tf_wp_layers, n_tf_wp_pancakes,
dx_tf_turn_steel, dx_tf_turn_insulation, dia_tf_turn_coolant_channel,
f_a_tf_turn_cable_space_extra_void}` and
`.superconducting_tfcoil.{dx_tf_wp_toroidal_min, c_tf_coil}` -- all `explicit-arg`
except the last two `data`-mediated pairs (`:3536-3546`). Owns seventeen outputs in
`run`'s write order, four of which the averaged arm never writes
(`dr_tf_turn_conduit_full`, `dx_tf_turn_conduit_full_toroidal`,
`dr_tf_turn_cable_space`, `dx_tf_turn_cable_space`) and one of which flips finding 1:
**`.tfcoil.c_tf_turn` is owned here** (`c_tf_coil` over the fixed turn count, `:3505`),
so it is not a boundary input of this machine. `n_tf_wp_layers`/`n_tf_wp_pancakes` are
plain continuous reads, not switches. The degenerate-cable fallback (`:3550-3567`) has
the same compute-then-correct ordering as the averaged arm's D2 and is ported the same
way (`jnp.where`, uncorrected `a_tf_turn_cable_space_effective`/
`f_a_tf_turn_cable_space_cooling`, corrected `a_tf_turn_steel`); the two negative-
dimension `logger.error`s are logging only, dropped.

**Validation.** `TestCiccIntegerTurnGeometry` (sample: PROCESS's own
`test_tf_cable_in_conduit_integer_turn_geometry`, the retired baseline-2018 point);
tfcoil case file 74 passed plain, 148 with `--fp-gradients`, 356 with `--fp-fuzz 5
--fp-gradients`. `test_machine.py::test_a_switch_that_decides_two_slots_decides_both`
is the regression for the dispatch gap itself, asserting both consequences per arm plus
the DEMO file. Warm MDA harness on the DEMO: 561 agreed / 65 off / 20 errors before ->
**601 / 30 / 20** after; the diff of disagreement sets is exactly the cicc cluster and
its downstream (35 variables), nothing added. The remaining 30 are the documented PF
dead-tail class and its 1e-6..6e-4 cascade; the 20 errors are the pre-existing
no-`DataStructure`-field bookkeeping, untouched.
