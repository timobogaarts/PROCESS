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
| `.tfcoil.whtcp`, `.whttflgs` | write | conditional-ownership-by-run-config | `itart == 1` only (`:2086-2093`); owned by `SuperconductingTfCoilAreasAndMassesSphericalTokamak` since 2026-08-27 |

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

Seven families. Six of them are twenty occupants:
`SuperconductingTfWpGeometry{Rectangular,DoubleRectangular,Trapezoidal}`,
`TfCaseAreas{CircularFront,StraightFront}`,
`DxTfSideCase{Rectangular,DoubleRectangular,Trapezoidal}`, `TfWpCurrents`,
`PeakBTfInboardWithRipple{16Coils,18Coils,20Coils,FlatAllowance}`,
`CiccAveragedTurnGeometryFromCurrentPerTurn`, `CiccIntegerTurnGeometry`
(2026-08-27), `CiccInboardAreasAndFractions`, `TfTurnArea`. The turn-geometry family
base is `CiccTurnGeometry` (`i_tf_turns_integer` first), with
`CiccAveragedTurnGeometry` as the averaged sub-family under it.

The seventh is the mass family, and it is **eighteen** occupants since 2026-08-27:
`<Material>SuperconductingTfCoilAreasAndMasses{Conventional,SphericalTokamak}` for all
nine `i_tf_sc_mat` materials, each leaf pairing one of two abstract `itart` arms
(`SuperconductingTfCoilAreasAndMasses{Conventional,SphericalTokamak}`, which declare the
outputs and the arm body) with one of nine abstract material classes
(`IterNb3snTfCoilMass` … `HazeltonZhaiRebcoTfCoilMass`, which declare `__call__` and its
one `FromExactly`). See the dated section below for why it is two axes rather than
eighteen flat classes.

## tier signal

**Tier 1** throughout. No internal solve, no call to another `Model`.

## switches touched

| switch | values | ported | note |
|---|---|---|---|
| `i_tf_wp_geom` | 0, **1** *(live)*, 2 | all three | `-1` `UNSET` never reaches a model; `init.py:977-989` resolves it from `i_tf_turns_integer` |
| `i_tf_case_geom` | **0** *(live)*, 1 | both | |
| `round(n_tf_coils)` | **16** *(live)*, 18, 20, other | all four | treated as a switch: the arms select different fit coefficients **and** different reads |
| `i_dx_tf_turn_general_input` × `i_dx_tf_turn_cable_space_general_input` | **(F, F)** *(live)*, (T, ·), (F, T) | (F, F) only | the other two own `.tfcoil.c_tf_turn`; see finding 1 |
| `itart` | **0** *(live)*, 1 | **both** (2026-08-27) | `1` additionally owns `whtcp`/`whttflgs` — conditional ownership, hence two occupants |
| `i_tf_sc_mat` | **1** *(live)*, 2–9 | **all nine** (2026-08-27) | nine occupants differing in one `FromExactly`, `.tfcoil.dcond[k]`; crossed with `itart` they are the mass slot's eighteen. Value 9 is portable here though `WINDING_PACK_MATERIAL` refuses it — see the dated section |
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
## 2026-08-27 — the `itart == 1` TF mass arm ported (ST frontier wave 5)

Both tracked spherical-tokamak files were refused at `itart_sc_tf_masses == 1`, whose
`UNPORTED` reason read *"the spherical-tokamak TF mass arm additionally owns
`.tfcoil.whtcp` and `.tfcoil.whttflgs` (`superconducting.py:2086-2093`), which the
conventional arm never writes — conditional ownership again."* The reason was accurate
and the fix is the established occupant-pair shape. `SuperconductingTfCoilAreasAndMasses
SphericalTokamak` now sits beside its conventional sibling and
`SC_TF_MASSES[SPHERICAL_TOKAMAK]` names it.

**The two arms read exactly the same twenty fields — no more, no fewer.** That is worth
stating because it is unusual: every other `itart` pair in this port (`base.md`'s picture
frame, `hcpb.py`'s shield heating) differs in its *reads* as well as its writes. Here the
whole difference is:

| | conventional (`itart == 0`) | spherical (`itart == 1`) |
|---|---|---|
| outboard length in the 2.2-factor case mass (`:1995-2006`) | `len_tf_coil - cplen` | `len_tf_coil` |
| outputs | ten | the same ten, **plus** `whtcp`, `whttflgs` (`:2085-2093`) |

The length difference is not two formulas but one fact stated twice: PROCESS's own
comment at `:1996-1997` says `.tfcoil.len_tf_coil` *excludes* the inboard leg at
`itart == 1`. `cplen` itself is formed identically above the branch (`:1988-1991`) and is
now `calculate_cplen`, called by both arms. The apportioning at `:2086-2093` divides by
`tfleng_sph = cplen + len_tf_coil` and not by `len_tf_coil`, which is consistent with the
same fact: at `itart == 1` the two lengths are disjoint, so their sum is the whole coil.
Ported as written.

Because the delta is that small, the shared sixty lines of mass algebra are a private
`_superconducting_tf_coil_masses` helper taking `cplen` and `len_tf_coil_case_outboard`
formed by the calling arm — the same treatment `hcpb.py::_nuclear_heating_shield` gives
its `itart` pair, and for the same reason: two copies of the algebra would be two places
for the coefficients to drift. The conventional arm's public signature, outputs and
Tier-1 case are unchanged by that refactor.

### `whtcp` is not a resistive centrepost mass on these runs

The field name suggests PROCESS's resistive-TART centrepost chain, and that chain does
write the same two fields — but at `i_tf_sup = 0`. Both ST files set `i_tf_sup = 1`
(`spherical_tokamak_eval.IN.DAT:356`, `st_regression.IN.DAT:820`), so on these runs
`:2086-2093` is the **sole producer** of `.tfcoil.whtcp` and `.tfcoil.whttflgs`, and it is
a pure re-apportioning of `m_tf_coils_total` by length. It reads nothing a conventional
run does not already produce, so **this arm introduces no new boundary input** — the
question the dispatch brief asked is answered in the negative, and there is nothing here
of the shape `base.md`'s `.build.r_cp_top` had.

Downstream, the two fields have real readers already ported: `costs.py`'s `c22211`/
`c22212` (`models/costs/costs.py:1616-1653`, the `itart == 1` cost arm) and `hcpb.py`'s TF
nuclear heating (`models/blankets/hcpb.py:491-554`, which uses the **outboard leg mass**
rather than the whole coil set). Both were reading a field with no producer until now.

### The samples, and what the second one is for

`sc-masses-st-baseline2018` is the conventional case's own point with `itart` flipped and
nothing else moved, so any disagreement between the two arms is the branch and nothing
else. `sc-masses-st-shortleg` puts the geometry where the arm lives — `len_tf_coil = 18.5`
against `cplen = 11.0`, twelve coils, REBCO density — so agreement is not an artefact of
conventional proportions. The reference adapter poisons `.tfcoil.whtcp`/`.whttflgs` with
NaN before the call, which makes *"PROCESS actually took the `itart == 1` branch"* an
executed check rather than an assumption: an untaken branch would leave NaN, not the
zero default that would silently agree with a port that also computed nothing.

### `i_tf_sc_mat = 9` — the slot is now reached at a value it does not answer

Recorded as the immediate follow-up, because it is a live wrong number waiting to happen.
`den_tf_sc_material` is bound on **both** arms to `FromExactly(tfcoil.dcond[0])` via the
module constant `I_TF_SC_MAT_ITER_NB3SN`. Both ST files set `i_tf_sc_mat = 9`
(`spherical_tokamak_eval.IN.DAT:355`, `st_regression.IN.DAT:827`), whose density is
`dcond[8] == 8500.0` and not `dcond[0] == 6080.0` (`tfcoil_variables.py:157-170`) — a
40 % error in the superconductor mass.

This is the pre-existing bake this record's § "switches touched" already carried as
"`i_tf_sc_mat`: `1` only", and it is the identical failure mode `next_steps.md` §14.5
diagnosed on `stellarator.coils.coils_mass`: *a module constant is not an
`eqx.field(static=True)`, so `switch_audit` cannot see it.* The conventional arm was safe
only because the one file that reaches it (`large_tokamak_eval`) sets `i_tf_sc_mat = 1`.
The spherical arm has no such luck.

**Not fixed here, deliberately.** The fix is to make the slot an `i_tf_sc_mat` family the
way `COILS_MASS_MATERIAL`/`WINDING_PACK_MATERIAL` already are — nine occupants differing
in one `FromExactly`, or an `itart × i_tf_sc_mat` arm index — and that is a change to
*both* arms and to the slot's key, not a change to the mass arm. Nothing is reachable in
the meantime: after this wave both ST files refuse at `n_divertors == 2` (below), so no
assembled machine reads `dcond[0]` on a `i_tf_sc_mat = 9` run. It must be closed before
one can.

> **Closed the same day** — see "the `i_tf_sc_mat` family" below. Both descriptions the
> paragraph offers turned out to be the same one: nine occupants differing in one
> `FromExactly`, crossed with the `itart` arm, keyed on the pair.

### Frontier probe

Both files move to the same next refusal, and it is not in this unit:

- `spherical_tokamak_eval.IN.DAT` — **refused**, `n_divertors == 2`
- `st_regression.IN.DAT` — **refused**, `n_divertors == 2`

*"the double-null arm, refused at **five** slots at once … `blanket_library.py:169-232`
… `hcpb.py:360-361` … `fw.py:194-197` … `vacuum.py:845-851` … `divertor.py:377-382` …
**Note that this is refused on `.physics.i_single_null`'s behalf**: `n_divertors` is
derived from it by `init.py:606-617`."* Both files do set `i_single_null = 0`
(`spherical_tokamak_eval.IN.DAT:292`, `st_regression.IN.DAT:638`), so the refusal is
theirs and not a mis-derivation. Neither file assembles, so no `machine_survey` run
applies to this wave.

Switch values these two files set that this unit's slots read, with line numbers:

| switch | `spherical_tokamak_eval.IN.DAT` | `st_regression.IN.DAT` |
|---|---|---|
| `itart = 1` | `:283` | `:66` |
| `i_tf_sup = 1` | `:356` | `:820` |
| `i_tf_sc_mat = 9` | `:355` | `:827` |
| `i_tf_shape = 2` | `:357` | `:803` |
| `i_single_null = 0` | `:292` | `:638` |

## 2026-08-27 — the `i_tf_sc_mat` family, and the defect above closed

The section before this one flagged its own defect and declined to fix it: both
`SuperconductingTfCoilAreasAndMasses{Conventional,SphericalTokamak}` bound
`den_tf_sc_material = FromExactly(tfcoil.dcond[0])` through a module constant
`I_TF_SC_MAT_ITER_NB3SN = 1`, which is `_audit/next_steps.md` §14.11's `CoilsMass` shape
exactly — *a switch answered outside `indat.py`, by a constant folded into a `FromExactly`
default, where `switch_audit` (which walks `eqx.field(static=True)` and nothing else)
cannot see it.* The constant is deleted. The slot is a family keyed on both switches, and
`indat.SC_TF_MASSES` has eighteen entries.

### Why the product is real, and why it is written as two axes

The brief asked whether this should be a 2 × 9 product of classes or the `dcond` element
threaded per material with `itart` kept as the class axis. **The threaded option does not
exist**, and saying why is the load-bearing part:

* `itart` **must** be a class axis. It changes the *owned* set — the spherical arm writes
  `.tfcoil.whtcp` and `.tfcoil.whttflgs` (`:2085-2093`) and the conventional arm writes
  neither. `OutputInto` is a class attribute; no kwarg makes one appear and vanish.
* `i_tf_sc_mat` **must** be a class axis too. It changes one read, and that read is a
  `FromExactly` **default**, evaluated when the class body executes. Nothing about an
  instance can move it: not an `eqx.field(static=True)`, not a constructor argument. This
  is precisely the fact §14.11 recorded when `CoilsMass` became a family — "a
  `FromExactly` default is fixed at class-definition time, so the index it selects is
  fixed with it."

Two forced class axes over independent switches is a product, so the occupant count is
2 × 9 = 18 and there is no arithmetic to argue about. What *was* a choice is how the
eighteen are written, and §14.11's own note on `ComponentThermalPowers` — *"the shape it
wants is **nesting** … rather than a flat product"*, a flat product refused there because
its cost exceeded what it bought — decides it. Here the nesting is free, so it is taken:

| | declares | count |
|---|---|---|
| `itart` arm (abstract) | the ten/twelve `OutputInto`s and `_masses`, the arm body | 2 |
| `i_tf_sc_mat` material (abstract) | `__call__`, whose only per-material entry is one `FromExactly(tfcoil.dcond[k])` | 9 |
| leaf (concrete, registered) | nothing — `class <Mat>...<Arm>(<Mat>TfCoilMass, ...<Arm>): pass` | 18 |

The nesting is **in the class hierarchy, not in the model tree**: the slot still holds one
node, `SC_TF_MASSES` still maps one configuration to one class, and no lookup node is
minted — `models/stellarator/coils/mass.py`'s reasoning ("the lookup's *input* is already
a real place and its index is static") is unchanged, and `.tfcoil.dcond[k]` is an
array-element `VarPath` per `_audit/naming_convention.md` § "Array elements".

**The reason to prefer it is the defect itself, not the line count.** What went wrong here
was one switch spelled twice, in two sibling classes, identically wrong. Eighteen flat
classes would put eighteen copies of that spelling in the file. With the material axis
factored out, each material's `dcond` element is written **once** and both `itart` arms
inherit it, so the two arms are now structurally incapable of naming different materials —
the same argument `_superconducting_tf_coil_masses` and `calculate_cplen` already make for
the shared *algebra*, applied to the shared *switch*. Both axes are abstract on their own
(an arm has no `__call__`, and cottax's `ExplicitFunction.__call__` is an `abstractmethod`;
a material has no `_masses`, which is one here too), so **the two classes that used to bake
the switch can no longer be instantiated at all.**

### `i_tf_sc_mat = 9` is portable here, and refused one file over — both are right

`indat.UNPORTED["i_tf_sc_mat", 9]` refuses `HAZELTON_ZHAI_REBCO` for the stellarator's
`winding_pack_intersect_inputs` and `coils_mass`, because `jcrit_from_material`
(`process/models/stellarator/coils/coils.py:52-160`) implements branches 1–8 and then
raises: there is no PROCESS arm to port, and a ninth occupant would have to invent the
model. **This slot does not call it.** `superconducting_tf_coil_areas_and_masses` uses the
material for exactly one thing — the density `dcond[i_tf_sc_mat - 1]` at
`process/models/tfcoil/superconducting.py:2024-2036`, which is the function's **only**
`dcond` read and is a bare table lookup with no dispatch of any kind. `dcond[8] == 8500.0`
is a real, populated element of a nine-long table (`tfcoil_variables.py:157-170`).

So: **the mass path needs the density and nothing else, and value 9 is fully portable
here.** The two facts do not contradict, because they are about different models. The
registries record the split rather than leaving it implicit — `WINDING_PACK_MATERIAL` and
`COILS_MASS_MATERIAL` key on the bare `"i_tf_sc_mat"`, so `_slot_occupant`'s `UNPORTED`
lookup finds the refusal; `SC_TF_MASSES` keys on `"itart_i_tf_sc_mat_sc_tf_masses"`, so it
does not. `UNPORTED`'s reason string was extended to say this outright, because an
unqualified "value 9 is not ported" three files away is exactly the kind of claim that
gets over-applied later.

It is checked rather than asserted:
`TestSuperconductingTfCoilAreasAndMassesStHazeltonZhaiRebco` drives PROCESS itself at
`i_tf_sc_mat = 9`. If the reading were wrong PROCESS would raise there instead of agreeing.

### The material axis is discriminated now, and it was not before

Both reference adapters used to write `den_tf_sc_material` into **every** slot of `dcond`
and pin `i_tf_sc_mat = 1`. That is why nothing caught the bake, and it is worse than it
sounds: four of the nine densities are `6080.0`, so even an adapter that left `dcond` at
its defaults could not tell `dcond[0]` from `dcond[4]`. `low_aspect_ratio_DEMO` is the live
instance — it sets `i_tf_sc_mat = 5` (`IN.DAT:910`), it was assembling the `dcond[0]`
occupant, and it got the right number anyway because `dcond[4] == dcond[0]`. **No value
test on any tracked tokamak could have failed on this defect.**

The adapters now take the occupant's `i_tf_sc_mat` as a leading argument, plant the
density in that element alone, and fill the other eight with `_DCOND_POISON = -1.0e9` —
`models/pfcoil/test_masses.py`'s technique, and safe here for a stronger reason than
there, since this function reads `dcond` exactly once. Measured: driving PROCESS at
`i_tf_sc_mat = 5` with the density planted in `dcond[0]` instead gives
`m_tf_coil_superconductor = -9.54e8` kg against the correct `5.80e3` kg — wrong in sign as
well as magnitude, at every sample. The ported side is unchanged throughout: the pure
functions still take the density already indexed, so the whole discrimination is on the
reference side, exactly as `TestPFCoilChainCsWstNb3Sn` does it.

Four of the product's eighteen corners are covered, which is enough because PROCESS's read
is a single unconditional `dcond[i_tf_sc_mat - 1]`: `(0, 1)` and `(1, 1)` (the existing
pair, now poisoned), plus `(0, 5)` — `low_aspect_ratio_DEMO`'s occupant, the case that
would have caught the original defect — and `(1, 9)` — both ST files' occupant, on the
`sc-masses-st-shortleg` point whose `den_tf_sc_material = 8500.0` is already `dcond[8]`'s
true value, so geometry and material are the ST's together rather than one grafted onto
the other. The NaN poisoning of `whtcp`/`whttflgs` is untouched, so the `itart == 1` branch
stays an executed check in both spherical cases.

**Validation.** tfcoil case file **91 passed** plain (81 before) and **182** with
`--fp-gradients` — the gradient half is exactly the half a plain run skips (91 + 91).
`test_machine.py`, `test_switch_coverage.py`, `test_boundary.py`,
`test_registry_coverage.py` → 280 passed / 43 skipped; `test_machine_survey.py`,
`test_mda.py`, `test_mdf.py`, `test_paths.py` → 38 passed. Ruff statistics on both edited
modules are at parity with their pre-change baselines, allowing for the new classes' own
`B008`/`D102`/`PLR6301` — the same three rules `models/stellarator/coils/mass.py` already
reports for the identical construct. (`indat.py` is at exact parity: 14 `E501`, 2
`PLR6201`, unchanged.)

### The three tokamak reference machines are unmoved, bit for bit

None of them sets `dcond`, so the table is PROCESS's default everywhere.

| file | `itart` | `i_tf_sc_mat` | occupant now | element | before |
|---|---|---|---|---|---|
| `large_tokamak_eval.IN.DAT` | 0 (default) | 1 (`:374`) | `IterNb3sn…Conventional` | `dcond[0]` = 6080.0 | `dcond[0]` — identical |
| `large_tokamak_nof.IN.DAT` | 0 (default) | 1 (`:583`) | `IterNb3sn…Conventional` | `dcond[0]` = 6080.0 | `dcond[0]` — identical |
| `low_aspect_ratio_DEMO.IN.DAT` | 0 (default) | 5 (`:910`) | `WstNb3sn…Conventional` | `dcond[4]` = 6080.0 | `dcond[0]` = 6080.0 — **the read moved, the number did not** |

`large_tokamak_eval`'s occupant class, its declared `dcond` read and every harness number
above are unchanged. `low_aspect_ratio_DEMO` is the one machine whose *declared read*
moves, and it moves onto the element its own switch names while landing on the same float
— which is the whole reason the poison exists rather than a value test.

Both ST files still refuse at `n_divertors == 2`, unchanged by this wave, so the wrong
density was never reachable in the interval between the two sections — but it is answered
now, before the double-null arm lands, which is what the earlier section asked for.

### Registration

`indat.py`, three hunks:

1. `SC_TF_MASSES` re-keyed from `SphericalTokamakModel` to
   `(SphericalTokamakModel, SuperconductorModel)`, built as a comprehension over a
   material → `(conventional, spherical)` table so the pairing cannot be mistyped, and the
   slot call is now
   `_slot_occupant("itart_i_tf_sc_mat_sc_tf_masses", (itart, i_tf_sc_mat), SC_TF_MASSES)`.
   It is this file's only two-switch key; every other composite arm reduces to a computed
   `_*_arm` integer, and this one cannot, because the two switches are independent rather
   than jointly selecting one behaviour.
2. **`i_tf_sc_mat` is resolved once, above the device branch**, beside `itart` — not
   re-read in `_tokamak_device`, which takes it as a threaded argument. It was resolved
   *below* the branch before, in the stellarator arm, under a comment noting that a
   tokamak's TF coils "will need their own answer to this same switch". They do; this is
   it, and it is the *same local*. That is the cross-slot coherence the brief asked about:
   `WINDING_PACK_MATERIAL`, `COILS_MASS_MATERIAL` and `SC_TF_MASSES` cannot name three
   different materials, because there is one `SuperconductorModel` value in the factory
   and all three are handed it. No slot is resolved at the read site, since the
   stellarator's two refuse value 9 and this one does not.
3. `UNPORTED["i_tf_sc_mat", 9]`'s reason extended to scope the refusal to the
   critical-surface slots (above).

## 2026-08-27 — the TF half of the missing-producer wave (§11.5)

`_audit/optimise_design.md` §11.5 measured **17 boundary variables across 14
constraints** whose producer PROCESS runs and the graph lacked. Re-measured on the
current tree before this wave — the same check, run from the SAND assembly's own
`constraint_nodes`/`objective_node` machinery on `large_tokamak_eval.IN.DAT` — the list
stands at **15**, the cold-producers wave having closed `.physics.pden_plasma_ohmic_mw`
(c2) and `sig_tf_cs_bucked` never having been a row (below). The TF side of it was seven
variables across six constraints:

| read | constraint | disposition after this wave |
|---|---|---|
| `.tfcoil.j_tf_wp_critical` | c33 | **produced** — `CiccSuperconductorProperties` |
| `.tfcoil.temp_tf_superconductor_margin` | c36 | **produced** — `TfSuperconductorTemperatureMargin` |
| `.tfcoil.j_tf_wp_quench_heat_max` | c35 | **produced** — `TfCoilQuenchHeatCurrentDensity` (`quench.md`) |
| `.superconducting_tfcoil.vv_stress_quench` | c65 | **produced** — `VvStressOnQuench` |
| `.tfcoil.sig_tf_case` | c31 | still a boundary input — `stresscl`, below |
| `.tfcoil.sig_tf_wp` | c32 | still a boundary input — `stresscl`, below |
| `.tfcoil.sig_tf_cs_bucked` | c72 | **not a missing producer at all** — below |

Four closed; the list is **15 -> 11**. The other nine rows belong to the concurrent
pfcoil/physics half.

### `sig_tf_cs_bucked` is a dead read, not a gap

§11.5 recorded that it is `None` "even converged — never written at `i_tf_bucking = 1`",
and that reading is right as far as it goes. What it does not say is that **the value is
never used on this machine either**. `constraint_72`'s body takes
`max(stress_shear_cs_peak, sig_tf_cs_bucked)` only when
`i_tf_bucking >= 2 and i_tf_inside_cs == TF_OUTSIDE_CS`; `large_tokamak_eval` has
`i_tf_bucking = 1`, so the other arm runs and reads `stress_shear_cs_peak` alone. It
appears in the missing-producer table only because `sand._bind` makes every non-switch
parameter of a constraint an `In` regardless of which static arm consumes it.

So c72's real gap on this machine is `.pf_coil.stress_shear_cs_peak` (`pfcoil.py:3508`),
which is the pfcoil half's row. Nothing is owed here. Worth stating because "a variable
that is `None` in PROCESS's own converged state" reads like a defect and is not one.

### `sig_tf_case` / `sig_tf_wp` — measured, and out of a wave's reach

Both come from one place: `stresscl`, called once by `run_and_output_stress`
(`superconducting.py:2095-2271`). The size of that is the answer:

| | |
|---|---|
| `TFCoil.stresscl` | `process/models/tfcoil/base.py:2222-3658`, **1439 lines**, `@numba.njit(cache=True)` |
| `extended_plane_strain` | `:3719-4234`, 517 lines, `njit` |
| `plane_stress` | `:4236-4458`, 224 lines, `njit` |
| `eyoung_parallel` / `eyoung_parallel_array` / `eyoung_t_nested_squares` / `eyoung_series` | `:3659-4653`, four more `njit` helpers |
| its return | **34 values**, of which nine are `(n_radial_array,)`-shaped arrays at `n_rad_per_layer = 500` |

There is no smaller seam: `sig_tf_case` and `sig_tf_wp` are two of the 34, produced by
the same layered elastic solve as the rest, and the four `data`-clamp lines at
`:2208-2231` (`x = x if x is None else computed`) are the only structure between it and
`data`. This is a unit of its own — a wave's worth on its own terms, with an `njit`
translation and array-valued outputs — and it is recorded here as such rather than
attempted. It is also the producer of `.tfcoil.str_wp`, which this wave has now made a
*declared* boundary input (below), so the dependency is at least visible.

### What was ported

Fifteen new pure functions and three node families in this module, plus one node in
`quench.py`. In dependency order:

**`vv_stress_on_quench` (c65).** `itoh_lambda_term`, `itoh_theta_factor_integral`,
`itoh_inductance_factor`, `vv_stress_on_quench` (the module-level Itoh surrogate,
`:4869-5153`) and `vv_stress_quench_from_build` (the method's own geometry prologue,
`:1381-1452`). One occupant, `VvStressOnQuench`, unswitched, **twenty-nine reads** —
seventeen from `.build`, and `.divertor.dz_divertor`, which is this slot's first read
outside the three TF areas. PROCESS's `np.clip(a_tf_coil_inboard_steel, 0, None)` and its
`# TODO: is this the correct current?` on `i_op` are carried across as written.

**`tf_cable_in_conduit_superconductor_properties` (c33).**
`cicc_superconductor_properties` (the shared tail), `clip_nb3sn_strain`, and four per-fit
entry points. Five occupants, `i_tf_sc_mat` in {1, 3, 4, 5, 7}; nine outputs each, in
`run`'s own write order.

**`calculate_superconductor_temperature_margin` (c36).**
`solve_current_sharing_temperature`, `_secant_step` and four per-fit margin functions.
Four occupants, `i_tf_sc_mat` in {1, 3, 4, 5}; two outputs each
(`.tfcoil.temp_tf_superconductor_margin` and `.tfcoil.temp_margin`, the same number, one
from `run` and one a side effect at `:1279`).

### The internal solve, and why scipy's iteration is replicated rather than improved on

`calculate_superconductor_temperature_margin` is a **root find**: PROCESS calls
`scipy.optimize.newton` with `fprime=None` and an explicit `x1 = 2 * tftmp`, i.e. the
**secant** method, `tol = rtol = 1e-6`, `maxiter = 50`, `disp=True`. This is the port's
second genuine internal solve after `models/vacuum/vacuum.py`'s duct diameter, and the
first whose answer a constraint reads.

`solve_duct_diameter` took the opposite decision — it *tightened* PROCESS's own tolerance
from `0.01` to `1e-10` — and that is the same rule applied to a different situation, not
an inconsistency. There, PROCESS's cutoff is a coarse heuristic and the unit is Tier 2,
so the endpoint is not the quantity under test. Here constraint 36 reads the endpoint and
the harness compares it against PROCESS's, so matching scipy's stopping rule is what
makes the comparison a value test rather than a tolerance negotiation. Measured agreement
at four materials: `5.6e-16`, `2.2e-16`, `0.0`, `0.0` relative.

**It is a `fori_loop`, not a `while_loop`, and the difference is the tangent.**
`solve_duct_diameter` can use `jax.lax.while_loop` because `Tier2Contract` never
differentiates `ported`. This one is Tier 1 and *is* differentiated, and `while_loop` has
no reverse rule — so the loop runs a fixed fifty trips with a converged flag. **Once
converged the carry collapses to the flat state `(answer, q, answer, q)` rather than
being frozen by masking**, and that is not a style choice: a masked carry still evaluates
`_secant_step` on the last real pair every remaining trip, a secant step whose
denominator has gone to zero is `inf`, and `jnp.where` discards it in value while
multiplying it by zero in the JVP — `nan`. Measured: `i_tf_sc_mat = 4` produced a `nan`
gradient under the masked form and the correct one under the flat collapse, with the
value test passing either way. The flat state has `q0 == q1` by construction, so
`_secant_step` takes its own bisection arm and returns `answer` exactly.

### D4 — one strain, clipped in one function and not the other

`tf_cable_in_conduit_superconductor_properties` clips the strain to `sign(s) * 0.5e-2`
outside the Nb3Sn fits' range (`:2910`, `:3017`, `:3052`) on a **local copy**. `run` then
re-reads `.tfcoil.str_wp` at `:2744-2747` and hands the **unclipped** value to
`calculate_superconductor_temperature_margin`. So when `|strain| > 0.5e-2` the two
functions evaluate the same critical surface at two different strains, and the
temperature margin is the one outside the fit's stated range. Ported exactly as written,
in both places; recorded here because it is not visible from either function alone.

### D5 — `i_tf_sc_mat = 2` cannot return

The Bi-2212 branch (`:2941-2984`) is the only arm that does not assign `bc20m`/`tc0m`,
and the function returns `TFSuperconductorLimits(..., bc20m=bc20m, tc0m=tc0m)` at
`:3160`. So `i_tf_sc_mat = 2` raises `UnboundLocalError` before any value exists. Not
ported, and refused with that as its reason — there is nothing to agree with.

### The switch axes, and what is refused

Two axes, and the second one is worth stating: **`i_str_wp` is a class axis of these two
slots.** It selects which field the strain is read from — `.tfcoil.str_tf_con_res` at
`0`, `.tfcoil.str_wp` at `1` (`:2897-2900` and `:2744-2747`) — and a `From` default is
fixed when the class body executes, exactly as a `FromExactly` default is. That is the
same argument that made `i_tf_sc_mat` a class axis for the mass slot, applied to a read
rather than to an index.

| `i_tf_sc_mat` | properties (c33) | margin (c36) |
|---|---|---|
| 1 ITER Nb3Sn *(live)* | ported | ported |
| 2 Bi-2212 | refused — D5 above | refused — **its own reason**: the margin function short-circuits it to `0.0` and never writes `.tfcoil.temp_margin` (`:1231-1233`), i.e. conditional ownership |
| 3 old Lubell NbTi | ported (**no strain read at all**) | ported (`c0 = 1.0e10` as a literal `run` passes) |
| 4 user-defined Nb3Sn | ported (reads `bcritsc`/`tcritsc`) | ported |
| 5 WST Nb3Sn | ported — `low_aspect_ratio_DEMO`'s | ported |
| 6, 8, 9 (TAPE) | refused — the function's own `SuperconductorShape.CABLE` guard (`:2882-2889`) refuses them before any arithmetic | refused, same |
| 7 Durham GL NbTi | **ported** | **refused** — D6 below |

`i_str_wp == 0` is refused across the board: PROCESS's default is `1`
(`tfcoil_variables.py:508`) and **no tracked input file sets the switch at all**, so the
arm is unreachable, and writing it is five more `__call__` bodies differing in one
parameter name. Registered rather than baked, so a file that does set it is refused
loudly instead of silently reading the other strain.

Both slots are total over the full 2 x 9 product — every pair has either an occupant or a
recorded reason, checked by `test_the_two_superconductor_slots_are_total`, and
`i_str_wp = 0` is refused end to end by `test_i_str_wp_zero_is_refused_end_to_end`. The
two keys are in `test_machine.py`'s `DERIVED_UNPORTED_KEYS` because their *value* is a
pair and no IN.DAT line selects one; those two cases are what that skip trades against.

### D6 — `i_tf_sc_mat = 7`'s temperature margin is complex, in PROCESS

The one asymmetry in the table above, and it is PROCESS's rather than the port's.
`superconductor_current_density_margin` branch 7 calls `gl_nbti`, which raises a negative
base to a fractional power while the secant search probes above `t_c0`; Python returns a
`complex`. Measured, on a point whose properties arm agrees exactly:

- at `b_tf_inboard_peak = 8.0`: `optimize.newton` **converges**, and PROCESS returns
  `temp_tf_superconductor_margin = 0.4561454861673191+1.2475645615451133e-12j` — a
  complex temperature margin, on its way into a float field.
- at `b_tf_inboard_peak = 12.5`: the same call dies with `TypeError: '<=' not supported
  between instances of 'complex' and 'float'` inside `gl_nbti` itself.

There is no real-valued PROCESS answer to agree with, so there is nothing to port: a JAX
float64 body returns `nan`, which would be *more* correct than the reference, and this
harness measures agreement rather than improving on PROCESS quietly. **The refusal is
scoped to the margin slot**: the same material's `CiccSuperconductorProperties` occupant
is written and agrees to `0.0e+00` on all nine outputs, because that function evaluates
the fit once at `tftmp` instead of searching upward.

### Two JAX-difficulty findings, both of the `next_steps.md` §9 class

**The Itoh `lambda` term is the fifth instance of the infinite-derivative-at-an-exact-
argument defect, and it was found by a gradient test.** `itoh_theta_factor_integral`
evaluates `itoh_lambda_term` at a `tau` matrix two of whose six entries are the literals
`1.0` and `-1.0` (`:4944-4947`). At `|tau| == 1` both arms sit exactly on a singularity of
their own derivative formula while the value is perfectly ordinary: `p * (1 - tau**2)` is
identically zero (`sqrt'(0) = inf`), and `(1 + omega*tau)/(tau + omega)` is identically
`+-1` (`arcsin'(+-1) = inf`). Before the guards, **every one of the eighteen
`d(vv_stress)/d(...)` was `nan` against a finite PROCESS finite difference while every
value test passed** — the exact signature this harness was built to catch. Fixed with
`safe_math.safe_sqrt` and a local `_safe_arcsin` written to the same double-`jnp.where`
idiom. `_safe_arcsin` lives in `superconducting.py` because it is the only site; it
belongs in `safe_math.py` the next time a second one appears.

**`.tfcoil.tfa` is read whole and indexed in the node body**, where every other
array-element read in this port is a `FromExactly(area.field[k])`. PROCESS reads
`self.data.tfcoil.tfa[0]` (`:1396`), so the literal transcription is
`FromExactly(tfcoil.tfa[0])` — and cottax refuses it, because `tf_coil_shape` owns the
whole vector and `_check_reads_match_owns` rejects a read lying *inside* an owned
variable (it would silently become a boundary input beside a produced array). The rule
that separates this from `.tfcoil.dcond[k]`, which stays a `FromExactly`, is **whether
the array has a producer in the same graph**: `dcond` is a constant table nothing owns,
`tfa` is an output. Worth adding to `naming_convention.md` § "Array elements" when that
file next gets a pass.

### Registration — `indat.py`, five hunks

1. `i_str_wp` resolved once in `_tokamak_device`, beside `i_tf_turns_integer`, because it
   decides two slots.
2. `CICC_SUPERCONDUCTOR_PROPERTIES` and `TF_SUPERCONDUCTOR_TEMPERATURE_MARGIN`, both
   keyed on `(i_str_wp, i_tf_sc_mat)` — the file's second and third two-switch keys after
   `SC_TF_MASSES`.
3. Five new `UNPORTED` reason constants (`_I_STR_WP_ZERO_REASON`,
   `_BI2212_UNBOUND_REASON`, `_BI2212_MARGIN_REASON`, `_SC_TAPE_REASON`,
   `_DURHAM_NBTI_COMPLEX_REASON`) and the 27 rows that use them; `UNPORTED["i_tf_sc_mat",
   9]`'s existing reason extended to say that the two new composite keys carry their own,
   differently-scoped refusals.
4. `_quench_helium_table` and `_QUENCH_GRID_FIELDS` — the CoolProp call and its guard; see
   `quench.md`.
5. The three new slots wired into the `CiccSuperconductingTfCoil` constructor.

`models/tfcoil/namespace.py` gains five slots (nineteen -> twenty-four).
`mda_harness.py` gains one static-kwarg kind (`FROZEN_INPUT`) and four table entries —
see `quench.md` for why.

### Acceptance, measured

**No new SCC.** `Blocking.scc` on both machines, before and after: the tokamak's seven
multi-node blocks are the same seven (three declared problems, the 4-node build/WP cycle,
the 8-node physics cycle, the 9-node PF ring, the `delta_eta` problem) and the
stellarator's six are unchanged. The guess half of the boundary is unmoved at 11, which
is the same claim from the other side — a landed producer that closed a loop would mint a
`Start`.

**Boundary pin: 347 -> 356 inputs, nine additions and zero removals.** Nothing left the
boundary, because all thirteen new outputs were read only by the constraint surface. The
nine are the four nodes' own declared reads: `.build.r_vv_inboard_out`,
`.build.dr_vv_shells`, `.tfcoil.theta1_coil`, `.tfcoil.theta1_vv` (the Itoh surrogate);
`.tfcoil.tftmp`, `.tfcoil.str_wp` (the critical surface and the margin, both);
`.tfcoil.rrr_tf_cu`, `.tfcoil.t_tf_quench_detection`,
`.constraints.flu_tf_neutron_fast_max` (the hotspot criterion). Eight are genuine PROCESS
inputs; **`.tfcoil.str_wp` is the ninth, and its producer is `stresscl`** — so the
boundary now states out loud that constraints 33 and 36 depend on the unported stress
chain, which the graph could not express while their producers were absent.

**§11.5 re-check: 15 rows -> 11.** The four TF rows above are gone; the eleven left are
`sig_tf_case`, `sig_tf_wp` and the nine pfcoil/physics rows.

**SAND, `large_tokamak_eval`:**

| | before | after |
|---|---|---|
| shape | 146 drive nodes, 9 unknowns, 24 conditions, 366 context, 14 inequalities | **155, 9, 26, 379, 16** |
| constraints omitted | **nine**: 24, 26, 27, **36**, 60, **65**, 72, 31, 32 | **seven** — c36 and c65 are inside the block now |
| Stage A | 14 of 17 exact | **16 of 19** exact |
| Stage B | 17 rows | 19; `c36` and `c65` both `0.00e+00` at both safety factors |
| C2 | 0 SQP iterations (first QP infeasible — `c68`, §11.6), lands on PROCESS's `x` at `1.4e-16 / 0.0` | unchanged |
| C3 (cold) | **not solved: 2 of 24 non-finite, `c33` `inf` and `c35` `inf`** | **not solved: 1 of 26, `c65` `nan`** |

**c33 and c35 were `inf` at the cold seed and are finite now**, which is exactly the rows'
point: each divided by a boundary constant PROCESS's own solve moves off zero and the port
had frozen at the `DataStructure` default.

**The one new cold non-finite is `c65`, and it is a boundary zero, not a defect.** Traced:
`.build.r_vv_inboard_out` is `0.0` in the cold `DataStructure`, and
`vv_stress_quench_from_build`'s `tf_vv_frac = r_tf_inboard_out / r_vv_inboard_out` is
`0/0`. That is `_audit/cold_boundary.md`'s bucket-2 class exactly — a field PROCESS's own
`Build.calculate_radial_build` writes before first use (`process/models/build.py:1836`
and `:1847`) and the port has no node for, so the bare default flows into a division. It
is a **seventh** entry for that record's six-boundary-zero table, reached because landing
c65's producer put the field on the boundary at all. The fix is that producer, and a
cold-start seeding rule remains the separate recorded decision it already was.

**Warm MDA harness, `large_tokamak_eval`** (before / after): declared graph 208 -> **212**
nodes, agreements 611 -> **624**, disagreements **16 -> 16**, errors **20 -> 20**,
ungrounded 0 -> 0, owned variables walked 647 -> **660**. The disagreement list is
**byte-identical** — all sixteen are the documented PF dead-tail
(`.pf_coil.n_pf_coil_turns`) and its `1e-6 .. 7.5e-4` cascade through costs and vacuum —
so the thirteen new owned variables all agree and nothing that was agreeing moved. The
static-kwarg line moves from *"checked: 8, not data-backed: 1, unresolved: 4"* with four
**MUST BE EMPTY** entries to *"checked: 10, not data-backed: 3, unresolved: 0"* with none.

**The stellarator is unmoved**: `run_mda_harness.py` with no argument, before and after,
is byte-identical from `declared graph:` to the end — 150 nodes, 472 agreements, 34
disagreements, 25 errors.

**Tests.** `tests/functional_process/models/tfcoil/` **275 passed** plain and **550** with
`--fp-gradients` (81 / 182 for `test_superconducting.py` alone before this wave).
`test_machine.py`, `test_boundary.py`, `test_switch_coverage.py`,
`test_registry_coverage.py`, `test_mda.py`, `test_mda_harness.py` -> 579 passed / 340
skipped. Ruff on the four edited modules is at parity with their pre-change baselines
(`E501` 24 -> 24 exactly; `D209` 1 -> 0) apart from the new occupants' own `B008` (+87)
and `D102` (+8) — the same two rules every `ExplicitFunction` family in this file already
reports.

## `i_tf_turn_type == 2` refused, and a silent mis-assembly closed (2026-08-29)

**The finding first: before this wave, a CroCo machine assembled silently as
cable-in-conduit.** `models/tfcoil/namespace.py`'s docstring already said the right thing
— "a machine whose `i_tf_turn_type` is `2` has a different occupant of
`.tokamak.cicc_superconducting_tf_coil` (`CROCOSuperconductingTFCoil`, unported), not a
different slot inside this one" — and **nothing checked it**: `i_tf_turn_type` appeared
nowhere in `indat.py`, so `machine_from_indat` built a `CiccSuperconductingTfCoil`
whatever the file said.

Measured rather than argued. A copy of `large_tokamak_eval.IN.DAT` with the single line
`i_tf_turn_type = 2` appended parses (`switches_from_indat` returns `2`;
`process/core/input.py:1071` declares the switch with `choices=[1, 2]`) and **assembles**,
producing `CiccSuperconductingTfCoil`. That is `low_aspect_ratio_DEMO`'s integer-turn
mis-assembly (`next_steps.md` §15) a second time, in a different slot, found the same way
— by asking what the factory does with a value nobody had put in front of it.

### the refusal, and why it is a refusal rather than an arm

`process/core/caller.py:298-313` resolves `i_tf_sup` and then `i_tf_turn_type` **above
every model** and runs `models.croco_sctfcoil` — a different `Model` class,
`CROCOSuperconductingTFCoil` (`superconducting.py:3773`) — instead of
`models.cicc_sctfcoil`. Its `run` computes a REBCO tape stack through
`tf_croco_averaged_turn_geometry`, `tf_turn_croco_cable_space_properties`,
`calculate_croco_cable_geometry` and `tf_croco_inboard_areas_and_fractions`, owning a
`.superconducting_tfcoil.*croco*`/`*hts_tape*` field set that does not exist in this port
at all, and it refuses integer turn geometry outright (`:3838`). **That is a whole unit,
not an occupant of any slot here.**

So the shape is `ife`'s, one level down: a switch that decides *which model runs* rather
than which arm of a model runs, answered once at assembly by
`_refuse_unported_switch("i_tf_turn_type", ...)`, and asked only on the superconducting
arm because that is the only branch of `caller.py` that reads it.

### this corrects the reason the ST files were being refused

Both `spherical_tokamak_eval.IN.DAT:72` and `st_regression.IN.DAT:800` set
`i_tf_turn_type = 2`. Until this wave they were caught instead by `_SC_TAPE_REASON` inside
`CICC_SUPERCONDUCTOR_PROPERTIES` — the `(i_str_wp, i_tf_sc_mat) == (1, 9)` cell — whose
text was *correct* ("a tape machine takes `CROCOSuperconductingTFCoil` instead ... so this
slot is never reached at those values") and whose position was wrong: it is a refusal
inside a slot that a CroCo machine never reaches, standing in for the absence of the model
that machine actually runs. Two consequences, both now fixed:

1. The ST closing wave's brief listed `i_tf_sc_mat = 9` as one of four remaining switch
   values and asked whether "the port refuses what PROCESS cannot do either" was the right
   answer for this site. It is not, and neither half of the hypothesis survives: PROCESS
   *can* do value 9 — the survey's quoted "no branch in `jcrit_from_material` at all" is
   `process/models/stellarator/coils/coils.py`'s dispatch, a **stellarator** function, and
   `models/pfcoil.py:4851` has a real `HAZELTON_ZHAI_REBCO` arm through `hijc_rebco` — and
   the tokamak's blocker was never the superconductor at all. It is the turn type.
2. The `(1, 9)`/`(0, 9)` tape cells stay exactly as they are. They are still true, still
   scoped to those two slots, and are now what they always should have read as: a
   statement about an unreachable cell, not the port's answer to a CroCo machine.

**No test bookkeeping was needed.** `_refuse_unported_switch` carries no registry, so
`test_machine.py::test_no_slot_registry_is_covered_by_nothing` has nothing new to place;
the four tracked files that assembled before still assemble
(`large_tokamak_eval`, `large_tokamak_nof`, `low_aspect_ratio_DEMO`,
`stellarator_helias`), which is the check a new refusal has to pass.
