---
kind: model-unit
status: draft
confidence: high
---

**Ported and registered.** `functional_process/models/tfcoil/croco.py`,
`tests/functional_process/models/tfcoil/test_croco.py`, registry row 56, one namespace
(`CrocoSuperconductingTfCoil`) filling the same `.tokamak.cicc_superconducting_tf_coil`
slot as its cable-in-conduit sibling.

This unit closes the **CroCo cluster** — items 1, 2 and 3 of `_audit/next_steps.md`
§18.2's eight model-level blockers on the two tracked spherical tokamaks. §18.2 argued
they were one package rather than three items; that turned out to be right for a reason
slightly stronger than the one given there, and §"what §18.2 got right and what it
missed" below records both.

## source

`process/models/tfcoil/superconducting.py`, **partial**, plus one function from
`process/models/superconductors.py`:

| in scope | lines | shape |
|---|---|---|
| `CROCOSuperconductingTFCoil.run` | 3776–4264 | the orchestration; the port keeps 7 of its statements and drops 5 as dead |
| `tf_croco_averaged_turn_geometry` | 4276–4389 | instance method, one hidden `self.data` read |
| `tf_croco_superconductor_properties` | 4391–4558 | instance method, `i_tf_sc_mat` dispatch, tape shapes only |
| `tf_turn_croco_cable_space_properties` | 4560–4607 | `@staticmethod`, pure |
| `tf_croco_inboard_areas_and_fractions` | 4609–4672 | `@staticmethod`, pure |
| `superconductors.calculate_croco_cable_geometry` | 1117–1196 | module function, pure |
| `superconductors.hijc_rebco` | 728–849 | module function, pure — the one new critical-surface fit |

Out of scope, each for its own measured reason:

- **`croco_voltage` (`:4677-4706`) — not ported at all.** Its return feeds only
  `.tfcoil.v_tf_coil_dump_quench_kv` (`:4020-4021`), which
  `quench_heat_protection_current_density`'s second return overwrites at `:4258-4259`
  before any reader; its two side-effect writes,
  `.superconducting_tfcoil.time2`/`tau2`, are read nowhere outside its own body, by grep
  over `process/`. So the string switch `.tfcoil.quench_model`
  (`core/input.py:1102`, choices `"linear"`/`"exponential"`) reaches nothing and needed
  no answer.
- **`output_croco_info` (`:4708-4865`)** and every other `output_*` — reporting.
- **`run`'s inline copper block (`:3930-3959`) except one line.**
  `a_tf_turn_croco_copper_bar`, `a_tf_turn_croco_cable_space_copper`,
  `a_tf_turn_copper_total`, `f_a_tf_turn_copper` and `a_tf_turn_croco_hastelloy` are read
  only by `output_croco_info` (`:4843-4858`), again by grep. Only
  `f_a_tf_turn_cable_space_cooling` (`:3948-3955`) survives into a computation
  (`quench_heat_protection_current_density`), and that line is
  `CrocoTurnCableSpaceCoolingFraction`.
- **`superconductors.current_sharing_rebco` — not needed, and not ported.** See
  finding 2.
- The integer-turn arm: **PROCESS raises on it** (`:3837-3839`).

Everything after the critical-current chain — `generic_tf_coil_area_and_masses`,
`superconducting_tf_coil_areas_and_masses`, `stresscl`, `vv_stress_on_quench`,
`quench_heat_protection_current_density` — is called by `CroCo`'s `run` with the same
arguments as `CICC`'s and is already ported. It is inherited, not re-ported: those slots
live on the shared `SuperconductingTfCoil` base in `models/tfcoil/namespace.py`.

## the five dead writes, and why they are the unit's main finding

A write is *dead* here in the strict sense: a later statement in the same `run`
overwrites it, and nothing reads it in between. All five were found by reading `run`
straight through and checking each field's next use; none is a judgement call.

| write | line | overwritten by | reader in between |
|---|---|---|---|
| `.tfcoil.a_tf_turn_cable_space_no_void` from the turn geometry | `:3813` | `tf_turn_croco_cable_space_properties`, `:3849` | none |
| `.tfcoil.a_tf_turn_steel` from the turn geometry | `:3816` | the same, `:3855` | none |
| `.superconducting_tfcoil.f_a_tf_turn_cable_space_cooling` from the cable space | `:3856` | the inline block, `:3948` | none |
| `.tfcoil.a_tf_wp_conductor` from the inboard areas | `:3912` | the inline block, `:3938`, **with the identical expression** | none |
| `.tfcoil.v_tf_coil_dump_quench_kv` from `croco_voltage()` | `:4020-4021` | the quench node's second return, `:4258-4259` | none |

**The first two are what makes this unit clean rather than awkward.**
`tf_croco_averaged_turn_geometry` does not take `a_tf_turn_cable_space_no_void` as an
argument: it reads `self.data.tfcoil.a_tf_turn_cable_space_no_void` off the data
structure at `:4375` — i.e. whatever the *previous pipeline pass* left there — computes
`a_tf_turn_steel` from it, and returns the entering value unchanged at `:4379`. That is a
genuine `implicit-io` read of a stale value, and in a graph it would be a self-reference
with no fixed point to drive. Because the cable-space node recomputes both fields three
statements later from scratch, **the port owns neither at the turn-geometry node and the
stale read disappears with them**. There is no ordering to reproduce, because there is no
live value: `CrocoAveragedTurnGeometryFromCurrentPerTurn` reads six fields and owns
seven, none of which it also reads.

The fourth is an ordinary `redundant-duplicate-write` — one expression, written twice,
kept once.

## the one true ordering hazard, and how dropping an output removes it

`d_sc_tf.f_a_tf_turn_copper = a_tf_turn_croco_cable_space_copper /
self.data.tfcoil.a_tf_turn` at `:3944-3946` reads `.tfcoil.a_tf_turn` **before** `run`
recomputes that field at `:3961-3965`. So PROCESS divides by the previous pass's turn
area, and a node reading `.tfcoil.a_tf_turn` would get the *current* one and disagree by
construction — the value would differ by however much the turn area moved between passes,
which on an unconverged design is not small.

The field is read only by `output_croco_info`, so the port simply does not own it and the
hazard has nowhere to appear. Recorded as **D1** rather than fixed: PROCESS's number is
what PROCESS reports, and the port is not in the business of quietly improving it.

## data footprint

Per node, in dependency order. `write` rows are the node's `OutputInto`s; every `read`
row is an `Input` in the node's `__call__`.

### `croco_turn_geometry` (`CrocoAveragedTurnGeometryFromCurrentPerTurn`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.j_tf_wp` | read | explicit-arg | |
| `.tfcoil.c_tf_turn` | read | conditional-ownership-by-run-config | owned by the two unported input-flag arms, read here; and **not** an iteration variable on either ST file, whose `ixc` are `[4, 6, 29]` and 14 entries not including 60 |
| `.tfcoil.dx_tf_turn_steel` | read | explicit-arg | |
| `.tfcoil.dx_tf_turn_insulation` | read | explicit-arg | |
| `.tfcoil.layer_ins` | read | explicit-arg | |
| `.superconducting_tfcoil.a_tf_wp_no_insulation` | read | explicit-arg | from `superconducting_tf_wp_geometry`, shared slot |
| `.tfcoil.a_tf_turn_insulation` | write | — | |
| `.tfcoil.n_tf_coil_turns` | write | — | |
| `.tfcoil.dx_tf_turn_general` | write | — | |
| `.superconducting_tfcoil.dr_tf_turn` | write | — | |
| `.superconducting_tfcoil.dx_tf_turn` | write | — | |
| `.tfcoil.dx_tf_turn_conduit_full_average` | write | — | |
| `.superconducting_tfcoil.dx_tf_turn_cable_space_average` | write | — | read by `stresscl`'s transverse smearing on `i_tf_turns_integer == 0` |
| ~~`.tfcoil.a_tf_turn_cable_space_no_void`~~ | (dropped) | implicit-io | read stale at `:4375`, written back unchanged at `:4379`, overwritten at `:3849` |
| ~~`.tfcoil.a_tf_turn_steel`~~ | (dropped) | implicit-io | computed from the stale read, overwritten at `:3855` |

### `croco_cable_space_properties` (`CrocoCableSpaceProperties`)

Reads `.tfcoil.dx_tf_turn_conduit_full_average` and `.tfcoil.dx_tf_turn_steel`, both
`explicit-arg`. Writes `.superconducting_tfcoil.dia_tf_turn_croco_cable`,
`.tfcoil.a_tf_turn_cable_space_no_void`,
`.superconducting_tfcoil.a_tf_turn_cable_space_effective`, `.tfcoil.a_tf_turn_steel`.
PROCESS's fifth return is dropped (table above).

### `croco_cable_geometry` (`CrocoCableGeometry`)

Five reads, all `explicit-arg` and all in `.superconducting_tfcoil`:
`dia_tf_turn_croco_cable` (from the slot above), `dx_tf_croco_strand_copper`,
`dx_tf_hts_tape_rebco`, `dx_tf_hts_tape_copper`, `dx_tf_hts_tape_hastelloy` — the last
four genuine inputs, set explicitly by both ST files
(`spherical_tokamak_eval.IN.DAT:73-76`). Ten writes, all in `.superconducting_tfcoil`:
`dia_tf_croco_strand_tape_region`, `n_tf_croco_strand_hts_tapes`,
`a_tf_croco_strand_copper_total`, `a_tf_croco_strand_hastelloy`,
`a_tf_croco_strand_solder`, `a_tf_croco_strand_rebco`, `a_tf_croco_strand`,
`dr_tf_hts_tape`, `dx_tf_hts_tape_total`, `dx_tf_croco_strand_tape_stack`.

### `croco_turn_cable_space_extra_void` (`CrocoTurnCableSpaceExtraVoid`)

No reads. Writes `.tfcoil.f_a_tf_turn_cable_space_extra_void = 0.0` (`:3895`).
**`conditional-ownership-by-run-config` across two `Model` classes**: the same `VarPath`
is a plain run input on the cable-in-conduit path and a computed constant here.

### `croco_inboard_areas_and_fractions` (`CrocoInboardAreasAndFractions`)

Ten reads, all `explicit-arg`: `.tfcoil.a_tf_turn_cable_space_no_void`,
`.tfcoil.n_tf_coil_turns`, `.tfcoil.f_a_tf_turn_cable_space_extra_void`,
`.tfcoil.a_tf_turn_insulation`, `.tfcoil.a_tf_turn_steel`,
`.tfcoil.a_tf_coil_inboard_case`, `.tfcoil.n_tf_coils`, `.tfcoil.a_tf_inboard_total`,
`.superconducting_tfcoil.a_tf_wp_ground_insulation`,
`.superconducting_tfcoil.a_tf_croco_strand`. Nine writes — the same nine the
cable-in-conduit twin owns.

### `croco_turn_cable_space_cooling_fraction` (`CrocoTurnCableSpaceCoolingFraction`)

Reads `.tfcoil.a_tf_turn_cable_space_no_void` and
`.superconducting_tfcoil.a_tf_croco_strand`; writes
`.superconducting_tfcoil.f_a_tf_turn_cable_space_cooling`, which
`quench_heat_protection_current_density` reads (`:4246`).

### `croco_superconductor_properties` (`HazeltonZhaiRebcoCrocoSuperconductorProperties`)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.a_tf_turn` | read | explicit-arg | from the shared `tf_turn_area` slot |
| `.tfcoil.b_tf_inboard_peak_with_ripple` | read | explicit-arg | |
| `.tfcoil.c_tf_turn` | read | explicit-arg | |
| `.tfcoil.tftmp` | read | explicit-arg | |
| `.superconducting_tfcoil.dr_tf_hts_tape` | read | explicit-arg | |
| `.superconducting_tfcoil.dx_tf_hts_tape_rebco` | read | explicit-arg | |
| `.superconducting_tfcoil.dx_tf_hts_tape_total` | read | explicit-arg | |
| `.superconducting_tfcoil.a_tf_croco_strand` | read | explicit-arg | |
| `.tfcoil.j_tf_wp_critical` | write | — | constraint 33's read |
| `.tfcoil.j_crit_str_tf` | write | — | written inside the function, `:4508` |
| `.superconducting_tfcoil.{f_c_tf_turn_operating_critical, j_tf_coil_turn, j_tf_superconductor, cur_tf_turn_croco_strand_critical, c_tf_turn_cables_critical, j_tf_superconductor_critical, b_..._zero_temp_strain, temp_..._zero_field_strain}` | write | — | the last two are the literals `(138, 92)` on this branch |
| ~~`.tfcoil.temp_margin`~~ | (dropped) | — | see finding 2 |

**No strain is read**, and that is not an omission — see finding 3.

### `tf_superconductor_temperature_margin` (`HazeltonZhaiRebcoCrocoTemperatureMargin`)

Reads `.superconducting_tfcoil.j_tf_superconductor`,
`.tfcoil.b_tf_inboard_peak_with_ripple`, the two `(bc20m, tc0m)` fields the slot above
owns, the three tape dimensions, and `.tfcoil.tftmp`. Writes
`.tfcoil.temp_tf_superconductor_margin` (constraint 36's read) and `.tfcoil.temp_margin`
— the same two fields, holding the same number, that the cable-in-conduit occupants own.

## findings

### 1. `.tfcoil.f_a_tf_turn_cable_space_extra_void` has a producer on a CroCo machine and none on a cable-in-conduit one

`run` sets it to a literal `0.0` at `:3895`, unconditionally, before the two nodes that
read it. Neither tracked ST file sets the input, and `tfcoil_variables.py`'s default is
`0.0` too — so a port that simply read it would agree numerically today and be reading a
coincidence. That is exactly the defect class `_audit/optimise_design.md` §16 spent a day
eliminating, and it is why a node with no reads at all (`CrocoTurnCableSpaceExtraVoid`)
is the right shape here rather than an over-engineering.

### 2. The CroCo properties function's temperature-margin tail is dead, so the port needs no second internal solve

`tf_croco_superconductor_properties` ends (`:4540-4547`) by calling
`superconductors.current_sharing_rebco` — a `scipy.optimize.newton` secant solve over
`jcrit_rebco` — and writing the result to `.tfcoil.temp_margin`. `run` then calls
`calculate_superconductor_temperature_margin`, which writes the *same field* at `:1278`
on every arm this namespace can reach, before anything reads it.

So the whole current-sharing block is dead on the CroCo path. The port does not have
`current_sharing_rebco` and does not need it; `.tfcoil.temp_margin` is owned by the
margin node alone, exactly as on the cable-in-conduit side. Worth stating plainly because
the alternative — porting a second replicated secant search — was the largest single
piece of work §18.5's estimate implied, and it turned out not to exist.

### 3. Both tape arms clip a strain that reaches nothing

`run` chooses a strain at `:4001-4004` (`.tfcoil.str_tf_con_res` at `i_str_wp == 0`,
`.tfcoil.str_wp` at `1`) and `tf_croco_superconductor_properties` chooses one again at
`:4443-4446`. On arm **9** the value is then clipped to `0.7e-2` with a `logger.error`
(`:4486-4492`) and handed to `superconductors.hijc_rebco`, **whose signature has no
strain argument** (`superconductors.py:728-736`). The clip is dead code guarding an
unused variable.

So `HazeltonZhaiRebcoCrocoSuperconductorProperties` reads no strain field, and
`.tfcoil.str_wp` is not a boundary input *of that node*. `i_str_wp` stays a key of both
CroCo registries anyway, because on arm **8** (`gl_rebco`) the strain is live.

### 4. What §18.2 got right and what it missed

§18.2's claim — "1-3 are one package; porting CroCo removes all three, but it does not
*port* those two slots; the CroCo namespace needs its own superconductor-properties and
temperature-margin occupants" — is exactly what happened, and the two new registries
(`CROCO_SUPERCONDUCTOR_PROPERTIES`, `CROCO_TEMPERATURE_MARGIN`) are those occupants.

What it did not say is the stronger structural fact the pair makes true: the two
properties functions guard on `SuperconductorShape` in their first four lines and take
**opposite** answers (`CABLE` at `:2882-2889`, `TAPE` at `:4435-4441`), so the two
registries *partition* the nine materials — no material has an occupant in both, and
every material has an occupant or a refusal naming the guard that excluded it in each.
`test_croco.py::test_croco_and_cicc_registries_partition_the_material_switch` checks it,
because two mirror-image refusal strings (`_SC_TAPE_REASON`, `_SC_CABLE_REASON`) are two
statements of one fact and the pair is where drift would show.

§18.5's size estimate was fair but two of its six functions did not need porting
(`croco_voltage`, `current_sharing_rebco`'s consumer) and the one it named as "the only
genuinely new material model", `hijc_rebco`, was indeed the only one.

### 5. `i_tf_stress_model` is the blocker behind the blocker

Landing this unit is what let `spherical_tokamak_eval.IN.DAT` reach the shared
`tf_stress` slot, which refuses `i_tf_stress_model == 0` (`extended_plane_strain`,
`base.py:3719-4234`, 517 lines). **That is a ninth model-level blocker `next_steps.md`
§18 does not list**, and its absence from §18 is not an error there: the stress slot
itself landed on 2026-08-30, after §18 was measured. Both ST files set it
(`spherical_tokamak_eval.IN.DAT:350`, `st_regression.IN.DAT:1223`). See the punch list.

## proposed signature(s)

Seven, all written and all in `models/tfcoil/croco.py`; the module's own docstring is
authoritative for each. In dependency order:
`croco_averaged_turn_geometry_from_current_per_turn`, `croco_cable_space_properties`,
`croco_cable_geometry`, `croco_turn_cable_space_extra_void`,
`croco_inboard_areas_and_fractions`, `croco_turn_cable_space_cooling_fraction`,
`croco_superconductor_properties_hijc_rebco` (with the shared tail
`_croco_superconductor_properties`), and `temperature_margin_hijc_rebco`.

`hijc_rebco` went to `models/physics/superconductors.py` beside the other eight fits, not
here — it is a material model, and unit #22 is where a material model lives.

## cottax node

Eight, in `croco.py`, each immediately after the function it wraps. Two family bases
(`CrocoTurnGeometry`, `CrocoSuperconductorProperties`); one occupant subclasses the
cable-in-conduit family base `TfSuperconductorTemperatureMargin`, because the *slot* is
the same and owns the same two fields whichever residual fills it.

The namespace is `models/tfcoil/namespace.py::CrocoSuperconductingTfCoil`, a sibling of
`CiccSuperconductingTfCoil` under a new shared base `SuperconductingTfCoil` that holds
the twenty-two slots both turn types have. That refactor is part of this unit: PROCESS's
two classes are related by inheritance and both `run`s open with the same
`run_base_superconducting_tf` call, so a base class is the port saying what PROCESS says
— and it is what let `indat.py` build the shared slots **once** and hand them to whichever
subclass `i_tf_turn_type` names, rather than transcribing thirteen `_slot_occupant` calls
a second time.

**The slot is still spelled `.tokamak.cicc_superconducting_tf_coil`.** See open
question 1.

## tier signal

**Tier 1** for seven of the eight; every function is pure, and the one internal solve
(`temperature_margin_hijc_rebco`) reuses `solve_current_sharing_temperature`, already
written and validated for the cable-in-conduit margins — the replicated
`scipy.optimize.newton` secant branch, not a different root finder.

## switches touched

| switch | values seen | note |
|---|---|---|
| `.superconducting_tfcoil.i_tf_turn_type` | 1, **2** *(this unit)* | resolved in `caller.py:298-313` above every model; `machine_from_indat` threads it to `_tokamak_device`, which picks the namespace |
| `.tfcoil.i_tf_turns_integer` | **0** *(live)*, 1 | `1` raises in PROCESS itself |
| `.tfcoil.i_dx_tf_turn_general_input` | **False** *(live)*, True | the two `True` arms own `c_tf_turn`; unported |
| `.tfcoil.i_dx_tf_turn_cable_space_general_input` | **False** *(live)*, True | as above |
| `.tfcoil.i_tf_sc_mat` | **9** *(live)*, 6, 8 | only the three `TAPE` shapes are reachable; `6` raises one call later, `8` is unreached by any tracked file |
| `.tfcoil.i_str_wp` | **1** *(default)*, 0 | decides nothing on arm 9 (finding 3); a registry key because arm 8 uses it |
| `.tfcoil.quench_model` | `""` *(unset)* | reaches only `croco_voltage`, which is dead — so this unit answers it by not needing to |

## calls into other models

`run` calls `run_base_superconducting_tf` (ported, `base.py` + `superconducting.py`),
`generic_tf_coil_area_and_masses`, `superconducting_tf_coil_areas_and_masses`,
`stresscl`, `vv_stress_on_quench` and `quench_heat_protection_current_density` — all
ported, all reached through the shared namespace base, none re-ported here.

## JAX-difficulty flags

- **`piecewise-constant-integer-count`** (`minor`, and the only one that changed a test).
  `n_croco_strand_hts_tapes` is `np.floor(...)` in PROCESS and `jnp.floor(...)` here, so
  the port's derivative is `0` almost everywhere — which is the derivative — while
  PROCESS's finite difference at `epsfcn = 1e-3` crosses whole tape steps and reports a
  slope. `TestCrocoCableGeometry` excuses the harness gradient checks with
  `static_argnames`, the same disposition `test_cs_fatigue.py::TestNCycle` takes for its
  trip count, and `test_croco_cable_geometry_gradient_within_one_step` measures the port
  against a central difference *inside* one step instead. That measurement:
  **1.0e-10 to 4.0e-9 relative** across five inputs and ten outputs.
- **`needs-lax-cond-or-where`** (`workaround-known`). `hijc_rebco`'s two `cur_critical`
  branches differ only in the sign that keeps `|1 - B/B_c|`'s base non-negative and
  *agree* at `B == B_c`, so the port writes one `jnp.where` on the base rather than two
  formulas. Value identity is by construction, not by tolerance, and the harness's fuzz
  range crosses the seam.
- **`fractional-power-at-zero`** (`workaround-known`). Three of `hijc_rebco`'s powers and
  two of `croco_cable_geometry`'s square roots go through `safe_pow`/`safe_sqrt`.
- **complex-vs-nan outside the domain** (`minor`, not reproduced). At
  `temp_conductor > t_c0`, `(1 - T/T_c0) ** 1.4` returns a Python **complex** in PROCESS
  and `nan` here — the same shape recorded for `gl_nbti` in
  `TfSuperconductorTemperatureMargin`. Neither is a value the other can be tested
  against; the fuzz bounds stay inside the domain (`t_c0 = 92 K` against a coolant
  temperature of a few K).
- **no CoolProp, no `numba`, no `scipy` beyond the replicated secant.** The unit adds no
  new non-traceable dependency.

## validation

`tests/functional_process/models/tfcoil/test_croco.py`: seven `Tier1Contract` cases and
seven explicit tests. `tests/functional_process/models/physics/test_superconductors.py`
gains `TestHijcRebco` (two legacy samples — PROCESS's own `test_hijc_rebco` case and the
ST tape).

Measured with `--fp-fuzz 40 --fp-gradients`: **every value and gradient case passes**,
with the one structural exclusion above. The reference adapters call PROCESS through
whatever surface it offers — a `@staticmethod` directly, a `DataStructure`-mediated
instance call for the two that read `self.data` — and set exactly the fields their
function reads, so "the port declares the right reads" is executed rather than asserted.

Three adapters slice PROCESS's return, and the slice **is** the finding: they drop
exactly the dead writes tabulated above.

## boundary effect

A CroCo tokamak assembles: measured on `large_tokamak_eval.IN.DAT` with
`i_tf_sc_mat = 9` and `i_tf_turn_type = 2` appended (the tracked ST files still refuse,
for the PF system and `i_tf_stress_model`). **244 nodes, of which the seven CroCo nodes
own 42 `VarPath`s**, including all eleven `*croco*`/`*hts_tape*` fields `run` computes.
`next_steps.md` §18.5 counted nineteen such fields with no producer anywhere in the port;
eleven of those now have one and the other eight are `output_croco_info`'s.

**`missing_producers_tokamak.txt` is unaffected and stays empty.** It is measured on
`large_tokamak_nof.IN.DAT`, a cable-in-conduit machine, whose namespace this unit does
not touch — the CroCo nodes exist only in a graph `i_tf_turn_type == 2` builds. The
tokamak and stellarator boundary pins are likewise unchanged.

## open questions

**OQ1. The slot is named for one of its two occupants.**
`.tokamak.cicc_superconducting_tf_coil` now holds either a `CiccSuperconductingTfCoil`
or a `CrocoSuperconductingTfCoil`, and the name says `cicc`. Renaming it to
`superconducting_tf_coil` would move every node under it — twenty-odd node paths — across
`reference_boundary_tokamak.txt`, the four DSM/XDSM exports and four test modules, for no
change in structure, so it was not done in this wave. It is a rename with a mechanical
diff and a real cost, and it should happen the next time something else touches those
pins.

**OQ2. Arm 8 (`DURHAM_REBCO`) is cheap and unwritten.** `gl_rebco` is already ported and
branch 8 of `superconductor_current_density_margin` is one line; the arm is a real,
complete PROCESS path and the only one of the three tape values that uses a strain. It is
refused rather than written because **no tracked input file selects it**, and this
harness measures agreement rather than asserting it. Two occupants and two registry rows
whenever something reaches it.

**OQ3. Is `.tfcoil.c_tf_turn` an input on an ST machine, or an unknown?** On the ported
arm the turn geometry reads it and does not own it, exactly as on the cable-in-conduit
side, where it is iteration variable 60. Neither ST file activates 60
(`spherical_tokamak_eval`'s `ixc` is `[4, 6, 29]`) and neither sets `c_tf_turn`, so it
enters at `tfcoil_variables.py`'s default — which is a boundary input whose value nobody
chose. Worth checking against PROCESS's own run once the PF package lands and the files
assemble; it is the kind of thing that is correct and still surprising.
