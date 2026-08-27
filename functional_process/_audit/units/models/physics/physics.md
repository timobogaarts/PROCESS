---
kind: model-unit
status: draft
confidence: medium-high
---

**Ported: the tokamak arm's minimal closure, not the file.**
`functional_process/models/physics/physics.py` declares eight pure functions and nine
cottax nodes (two of them family heads; the ramp-time family has two occupants since
2026-08-27, see §"2026-08-27 — pulse ramp-time arm 0"). Case:
`tests/functional_process/models/physics/test_physics.py`.

**No registry row yet.** `unit_registry.md` is the consolidation pass's file, and
`tests/functional_process/test_registry_coverage.py::test_every_record_file_is_in_the_registry`
will fail until a row naming this record is added. The row text is in
§"registration owed" below.

## source

`process/models/physics/physics.py` — 6931 lines, five `Model` classes (`Physics`,
`PlasmaBeta`, `PlasmaInductance`, `DetailedPhysics`, plus module-level helpers). This
record is **not** an audit of the file. It is an audit of the *closure* that produces
the eight variables `_audit/tokamak_boundary.md` § `.tokamak.physics` lists, which is
what this wave commissioned.

The eight, traced write by write at AST level (`grep` for the assignment target, then
read the enclosing block):

| # | output | write site | producer |
|---|---|---|---|
| 1 | `.physics.b_plasma_surface_poloidal_average` | `physics.py:313-324` | `plasma_fields.py:27-93` (`PlasmaFields.calculate_surface_averaged_poloidal_field`) |
| 2 | `.physics.pden_plasma_core_rad_mw` | `physics.py:751` | `radiation_power.py:29-141` (`calculate_radiation_powers`), copied out of `RadpwrData` **with no clip** |
| 3 | `.physics.p_plasma_inner_rad_mw` | `physics.py:758-760` | inline: `pden_plasma_core_rad_mw * vol_plasma` |
| 4 | `.physics.p_plasma_rad_mw` | `physics.py:764-766` | inline: `pden_plasma_rad_mw * vol_plasma` |
| 5 | `.physics.p_plasma_separatrix_mw` | `physics.py:800-809` **and again** `physics.py:843-845` | `exhaust.py:88-127` (`PlasmaExhaust.calculate_separatrix_power`), then an inline transform |
| 6 | `.times.t_plant_pulse_plasma_present` | `physics.py:516` | `pulse.py:71-79` (`PulseTimings.plasma_present`) |
| 7 | `.times.t_plant_pulse_total` | `physics.py:521` | `pulse.py:92-95` (`PulseTimings.total`) |
| 8 | `.physics.e_plasma_beta` | `physics.py:3912-3916` | `physics.py:4153-4176` (`PlasmaBeta.calculate_plasma_energy_from_beta`) |

**Three of the eight are not computed in `physics.py` at all**, and one is not computed
in `process/models/physics/` at all. `physics.py`'s role for 1, 5, 6 and 7 is the
`run()` shell: it selects a branch, calls out, and stores. That is worth stating plainly
because `tokamak_boundary.md` attributes all eight to `.tokamak.physics`, and for three
of them the *occupant* belongs somewhere else (§"divergence from `tokamak_boundary.md`").

### two contradictions with `tokamak_boundary.md`, both recorded rather than smoothed

1. **`.physics.e_plasma_beta` is not `Physics`'s.** It is written by `PlasmaBeta.run`
   (`physics.py:3912-3916`), which `Physics.run` invokes at `physics.py:429` as
   `self.beta.run()`. `tokamak_boundary.md` itself says as much in prose two lines under
   the table ("`PlasmaBeta` … is what owns it on a tokamak — decision 7's site") while
   the table row is under `.tokamak.physics`. The node written here
   (`PlasmaEnergyFromBeta`) should occupy **`.tokamak.plasma_beta`**, not
   `.tokamak.physics`.
2. **`.physics.b_plasma_surface_poloidal_average` is `PlasmaFields`'s.** The formula is
   `plasma_fields.py:83-93`; `physics.py:313-324` only stores its return.
   `models/tokamak/namespace.py` already has a `plasma_fields` slot ("1 entered
   function, 67 entered LOC"), and this is that one function. The occupant written here
   (`SurfaceAveragedPoloidalFieldAmperes`) should occupy **`.tokamak.plasma_fields`**.

Both are ported into `functional_process/models/physics/physics.py` regardless, because
this wave's file allocation gives this agent that module and not `plasma_fields.py`.
Moving either is a file move plus a registry row, not a re-port.

## the extraction seam

Uneven, and the unevenness is the point.

- **Already clean**: `calculate_plasma_energy_from_beta` (`physics.py:4153-4176`) and
  `calculate_separatrix_power` (`exhaust.py:88-127`) are `@staticmethod`s taking plain
  floats. Port verbatim.
- **Clean but switched**: `calculate_surface_averaged_poloidal_field`
  (`plasma_fields.py:27-93`) is an instance method whose only `self` use is
  `self.current.plascar_bpol` on the arm this port does not take. Its Ampere arm is one
  line and two of its eight arguments.
- **Inline in `run()`, no `calculate_*` to lift**: the radiation products
  (`physics.py:750-766`), the separatrix positivity transform (`:839-845`) and the
  ramp-time dispatch (`:463-498`). These are extracted by hand here and their references
  in the harness case are transcriptions of the source, not PROCESS callables — the same
  situation `plasma_physics.md` recorded for `st_phys`'s inline blocks.

## data footprint

Reference run: `tests/regression/input_files/large_tokamak_eval.IN.DAT` —
`i_plasma_current = 4` (`:288`), `i_pulsed_plant = 1` (`:330`), `pulsetimings = 0`
(`:392`), `i_beta_fast_alpha = 1` (`:290`), `i_confinement_time = 34` (`:300`);
`i_plasma_ignited` unset, so `0` = `NON_IGNITED` (`physics_variables.py:881`);
`i_rad_loss` unset, so `1` = `CORE_ONLY` (`physics_variables.py:954`). Rows marked
*(live)* are on that path.

### `b_plasma_surface_poloidal_average` — `physics.py:313-324`, `plasma_fields.py:27-93`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_current` | read | **switch** | *(live, `4`)* tested `!= 2` only (`plasma_fields.py:83`) |
| `.physics.plasma_current` | read | explicit-arg | *(live)* bound to the source's `cur_plasma` parameter — a call-site rename, `physics.py:316` |
| `.physics.len_plasma_poloidal` | read | explicit-arg | *(live)* produced by `plasma_geometry.py` (unit #24, unported) |
| `.physics.q95` | read | explicit-arg | **arm 2 only** — not read on the live arm |
| `.physics.aspect` | read | explicit-arg | **arm 2 only** |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | **arm 2 only** |
| `.physics.kappa` | read | explicit-arg | **arm 2 only** |
| `.physics.triang` | read | explicit-arg | **arm 2 only** |
| `.physics.b_plasma_surface_poloidal_average` | **write** | explicit-arg | *(live)* sole tokamak producer |

Five of the eight arguments are dead on the live arm. A union node would declare
**5 invented edges** here; the split removes all five.

### the radiation block — `physics.py:734-766`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.pden_plasma_core_rad_mw` | **write** | explicit-arg | *(live)* `= radpwrdata.pden_plasma_core_rad_mw`, `physics.py:751` — **no clip** |
| `.physics.pden_plasma_outer_rad_mw` | **write** | explicit-arg | *(live)* `physics.py:752` — no clip |
| `.physics.pden_plasma_sync_mw` | **write** | explicit-arg | `physics.py:750`; **not ported here** — `PlasmaRadiationPowers`/`SynchrotronRadiationPower` already own it |
| `.physics.pden_plasma_rad_mw` | **write** | explicit-arg | `physics.py:753`; already owned by `PlasmaRadiationPowers`, so a **read** for this unit |
| `.physics.p_plasma_sync_mw` | **write** | explicit-arg | `physics.py:755-757`; outside this slot's eight, **not ported** — see §open questions 3 |
| `.physics.p_plasma_inner_rad_mw` | **write** | explicit-arg | *(live)* `physics.py:758-760` |
| `.physics.p_plasma_outer_rad_mw` | **write** | explicit-arg | *(live)* `physics.py:761-763`; free from the same reads, so ported alongside |
| `.physics.p_plasma_rad_mw` | **write** | explicit-arg | *(live)* `physics.py:764-766` |
| `.physics.vol_plasma` | read | explicit-arg | *(live)* `plasma_geometry.py`'s, unported — a boundary input today |

In the port the two `pden_*` writes read
`.physics.pden_plasma_core_rad_mw_unclipped`/`pden_plasma_outer_rad_mw_unclipped`, the
mints `models/physics/radiation_power.py::PlasmaRadiationPowers` already owns.

### the separatrix block — `physics.py:790-845`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_ignited` | read | **switch** | *(live, `0` = `NON_IGNITED`)* `physics.py:795-796` |
| `.current_drive.p_hcd_injected_total_mw` | read | explicit-arg | *(live)* **`NON_IGNITED` arm only** — `physics.py:794`; the `IGNITED` arm passes `0.0` |
| `.physics.f_p_alpha_plasma_deposited` | read | explicit-arg | *(live)* |
| `.physics.p_alpha_total_mw` | read | explicit-arg | *(live)* `set_fusion_powers`'s, already ported |
| `.physics.p_non_alpha_charged_mw` | read | explicit-arg | *(live)* |
| `.physics.p_plasma_ohmic_mw` | read | explicit-arg | *(live)* `Physics.plasma_ohmic_heating`, `physics.py:1605`, unported |
| `.physics.p_plasma_rad_mw` | read | explicit-arg | *(live)* this unit's own `TotalRadiationPower` |
| `.physics.p_plasma_separatrix_mw` | **write** | **written twice in one pass** | *(live)* `physics.py:800-809`, then `physics.py:843-845` |

### the ramp-time / pulse block — `physics.py:463-521`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pulse.i_pulsed_plant` | read | **switch** | *(live, `1`)* `physics.py:464` |
| `.times.i_t_current_ramp_up` | read | **switch** | `physics.py:465`; **`i_pulsed_plant != 1` arm only** |
| `.times.pulsetimings` | read | **switch** | *(live, `0`)* `physics.py:476` — **its only read in all of `process/models/**`** |
| `.physics.plasma_current` | read | explicit-arg | *(live)* `physics.py:479` |
| `.times.t_plant_pulse_plasma_current_ramp_up` | **write** | explicit-arg | *(live)* `physics.py:478-480` |
| `.times.t_plant_pulse_plasma_current_ramp_down` | **write** | explicit-arg | *(live)* `physics.py:481-483` |
| `.times.t_plant_pulse_coil_precharge` | **read on the live arm, written on two others** | conditional-ownership-by-run-config | `physics.py:469-471` and `:489-492`; the source comment at `:477` says it is an input on the live arm |
| `.times.t_plant_pulse_fusion_ramp` | read | explicit-arg | *(live)* `physics.py:503` |
| `.times.t_plant_pulse_burn` | read | explicit-arg | *(live)* `physics.py:504` |
| `.times.t_plant_pulse_dwell` | read | explicit-arg | *(live)* `physics.py:506` |
| `.times.t_burn_0` | **write** | explicit-arg | `physics.py:513`, `= t_plant_pulse_burn`; **not ported** — see §open questions 3 |
| `.times.t_plant_pulse_plasma_present` | **write** | explicit-arg | *(live)* `physics.py:516` |
| `.times.t_plant_pulse_no_burn` | **write** | explicit-arg | *(live)* `physics.py:518` |
| `.times.t_plant_pulse_total` | **write** | explicit-arg | *(live)* `physics.py:521` |

### `e_plasma_beta` — `physics.py:3912-3916`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.beta_total_vol_avg` | read | explicit-arg | *(live)* not produced by anything ported on a tokamak — boundary input today |
| `.physics.b_plasma_total` | read | explicit-arg | *(live)* written at `physics.py:373-376` by `PlasmaFields.calculate_total_magnetic_field`, **unported** — boundary input today (§open questions 1) |
| `.physics.vol_plasma` | read | explicit-arg | *(live)* |
| `.physics.e_plasma_beta` | **write** | explicit-arg | *(live)* |
| `.physics.e_plasma_beta_thermal` | **write** | explicit-arg | `physics.py:3905-3909`, same static, `beta_thermal_vol_avg` binding; **not ported** — nothing but `outplas` (`physics.py:4666-4675`) reads it |

## the two carried flags

Both were handed to this unit by `next_steps.md` §2 and both resolve the same way.

**Flag A — the two `combine_radiation_powers` callers disagree on clipping at zero.**
Confirmed by reading both. `stellarator.py:2149-2158` assigns the four `RadpwrData`
fields and then clips `pden_plasma_core_rad_mw` and `pden_plasma_outer_rad_mw` with
`max(..., 0.0e0)`. `physics.py:750-753` assigns the same four and clips nothing;
`physics.py:755-766` forms the products straight off them. **This port follows
`physics.py`: no clip.** That is why `calculate_unclipped_radiation_powers` exists as a
node at all — it is `plasma_physics.py::calculate_clipped_radiation_powers` with the two
`max` calls deleted and nothing else changed, and it is the second half of the reason
`radiation_power.py::PlasmaRadiationPowers` mints `_unclipped` names rather than owning
the real fields. The clip is a property of one caller, not of the radiation model.

**Flag B — is `.physics.pden_plasma_core_rad_mw` clipped on the tokamak path?**
**No.** Measured by reading `physics.py:734-766` in full: between the
`calculate_radiation_powers` call (`:734-749`) and the last product (`:766`) there is no
`max`, no `min`, no comparison and no conditional of any kind. `physics.py:751` is a
bare assignment. The clip at `stellarator.py:2153-2158` has no counterpart here.

The consequence is a real behavioural divergence between the devices for the same
quantity, and it is **D1** below.

## proposed signature(s) — as written

```python
def calculate_surface_averaged_poloidal_field_amperes(cur_plasma, len_plasma_poloidal)
def calculate_unclipped_radiation_powers(
    pden_plasma_core_rad_mw_unclipped, pden_plasma_outer_rad_mw_unclipped, vol_plasma
)   # -> (pden_core, pden_outer, p_inner, p_outer)
def calculate_total_radiation_power(pden_plasma_rad_mw, vol_plasma)
def calculate_separatrix_power(
    f_p_alpha_plasma_deposited, p_alpha_total_mw, p_non_alpha_charged_mw,
    p_hcd_injected_total_mw, p_plasma_ohmic_mw, p_plasma_rad_mw
)
def force_positive_separatrix_power(p_plasma_separatrix_mw_raw)
def calculate_pulsed_plant_ramp_times(plasma_current)   # -> (ramp_up, ramp_down)
def calculate_continuous_plant_ramp_times(plasma_current)
                                        # -> (ramp_up, precharge, ramp_down)
def calculate_plasma_energy_from_beta(beta, b_field, vol_plasma)
```

Every one takes plain floats and returns plain floats/tuples; no `DataStructure`, no
`self`, no switch argument anywhere (the switches became occupants, below).

## cottax nodes

| node | slot it should occupy | owns | reads |
|---|---|---|---|
| `SurfaceAveragedPoloidalField` (head) | `.tokamak.plasma_fields` | — | — |
| ` └ SurfaceAveragedPoloidalFieldAmperes` | ″ | `.physics.b_plasma_surface_poloidal_average` | `.physics.plasma_current`, `.physics.len_plasma_poloidal` |
| `UnclippedRadiationPowers` | `.tokamak.physics` | `.physics.pden_plasma_core_rad_mw`, `.physics.pden_plasma_outer_rad_mw`, `.physics.p_plasma_inner_rad_mw`, `.physics.p_plasma_outer_rad_mw` | `.physics.pden_plasma_core_rad_mw_unclipped`, `.physics.pden_plasma_outer_rad_mw_unclipped`, `.physics.vol_plasma` |
| `TotalRadiationPower` | `.tokamak.physics` | `.physics.p_plasma_rad_mw` | `.physics.pden_plasma_rad_mw`, `.physics.vol_plasma` |
| `SeparatrixPower` (head) | `.tokamak.physics` | — | — |
| ` └ SeparatrixPowerNonIgnited` | ″ | `.physics.p_plasma_separatrix_mw_raw` *(mint)* | `.physics.f_p_alpha_plasma_deposited`, `.physics.p_alpha_total_mw`, `.physics.p_non_alpha_charged_mw`, `.current_drive.p_hcd_injected_total_mw`, `.physics.p_plasma_ohmic_mw`, `.physics.p_plasma_rad_mw` |
| `PositiveSeparatrixPower` | `.tokamak.physics` | `.physics.p_plasma_separatrix_mw` | `.physics.p_plasma_separatrix_mw_raw` |
| `PulseRampTimes` (head) | `.tokamak.physics` (or `.tokamak.pulse`, see OQ4) | — | — |
| ` └ PulseRampTimesPulsedDefault` | ″ | `.times.t_plant_pulse_plasma_current_ramp_up`, `.times.t_plant_pulse_plasma_current_ramp_down` | `.physics.plasma_current` |
| ` └ PulseRampTimesContinuousDefault` | ″ | `.times.t_plant_pulse_plasma_current_ramp_up`, `.times.t_plant_pulse_coil_precharge`, `.times.t_plant_pulse_plasma_current_ramp_down` | `.physics.plasma_current` |
| `PlasmaEnergyFromBeta` | `.tokamak.plasma_beta` | `.physics.e_plasma_beta` | `.physics.beta_total_vol_avg`, `.physics.b_plasma_total`, `.physics.vol_plasma` |

### the one minted name, and why

`.physics.p_plasma_separatrix_mw_raw` has no backing `DataStructure` field. It exists
because **PROCESS writes `.physics.p_plasma_separatrix_mw` twice in one pass** and three
call sites read the first value before the second is written:
`calculate_psep_over_r_metric` (`physics.py:811-816`),
`calculate_eu_demo_re_attachment_metric` (`:818-826`) and `ScrapeOffLayer.run` (`:832`).
Every consumer after `:845` — including all three of this slot's boundary readers in
`power` — sees the second. Two nodes cannot own one `VarPath`, and one node applying the
transform internally would hand the *post*-transform value to those three call sites,
which is a silent behaviour change.

This is the same shape and the same resolution as
`radiation_power.py::PlasmaRadiationPowers`'s `_unclipped` mints, and the same
"PROCESS's omission is not a reason to invent a namespace" argument
`confinement_time.py::ConfinementScalingInputs` made for
`.physics.nd_plasma_electron_line_19`. The consequence is bookkeeping and is stated
where it lands: `mda_harness` cannot compare a mint against PROCESS's converged state,
so it joins the not-data-backed category.

**Flagged for the orchestrator rather than assumed settled** — it is a dual-ownership
resolution and the brief says not to improvise one. If the answer is instead "fold the
transform into one node and let the three pre-transform readers bind to the
post-transform value", that is a two-line edit here and a recorded deviation there.

### already-ported sub-calls — do not re-port

- **`.times.t_plant_pulse_plasma_present` / `t_plant_pulse_no_burn` /
  `t_plant_pulse_total`**: `models/stellarator/initialization.py::PulseDurations`
  (`calculate_pulse_durations`) is **already** `PulseTimings.plasma_present`/`no_burn`/
  `total`, term for term — checked against `pulse.py:71-95`. It needs no tokamak
  counterpart, only a tokamak registration and (arguably) a move out of
  `models/stellarator/`. Two of this slot's eight outputs are therefore closed by a
  registration, not by a port.
- **`.physics.pden_plasma_rad_mw`,
  `.physics.pden_plasma_core_rad_mw_unclipped`/`pden_plasma_outer_rad_mw_unclipped`,
  `.physics.pden_plasma_sync_mw`**: `models/physics/radiation_power.py`'s three nodes.
  Already shared, already in `models/physics/namespace.py::Physics`.
- **`.physics.p_alpha_total_mw`, `.physics.p_non_alpha_charged_mw`**:
  `models/physics/fusion_reactions.py::SetFusionPowers`.

## tier signal

**Tier 1 for all seven functions.** No internal iteration, no `scipy.optimize`, no
`fsolve`, no `copy.deepcopy`, no CoolProp, no call into another `Model` from any ported
body. Every one is straight-line arithmetic over floats.

**Sample provenance is the weak point.** Two of the seven have real legacy points lifted
from `tests/unit/models/physics/test_physics.py` —
`test_calculate_surface_averaged_poloidal_field` (its `i_plasma_current = 3` and `= 4`
rows, the two of four that take the Ampere arm) and
`test_calculate_plasma_energy_from_beta`. The other five have **no PROCESS unit test at
all**, so their `legacy_sample`s are hand-built at `large_tokamak_eval`-scale values and
the real coverage is the fuzz draws. This is the same gap `build.md` and several
`coils/*.md` units record; converged sampling (`_harness/sampling.py`'s `converged`) is
still unimplemented, which is what would close it properly.

Two samples exist specifically to exercise arms no converged run reaches, and they are
the ones worth keeping if anything is ever trimmed:
`TestUnclippedRadiationPowers::negative-core-density` (the point where the tokamak and
stellarator references disagree — flag A/B) and
`TestSeparatrixPower::radiation-dominated-negative` together with
`TestForcePositiveSeparatrixPower::negative-raw-power` (the configuration
`physics.py`'s own "KLUDGE" comment exists for).

## switches touched

Four, all **split**, all with the reads-set evidence above. Per this wave's binding
policy no switch is a static kwarg, so every one is an occupant class or an `UNPORTED`
entry.

| switch | values | live | decision | evidence |
|---|---|---|---|---|
| `.physics.i_plasma_current` | 1–9 | `4` | **split**, 2 arms | `plasma_fields.py:83` tests `!= 2`. Arm A (`!= 2`) reads `plasma_current`, `len_plasma_poloidal`. Arm B (`== 2`) reads `q95`, `aspect`, `b_plasma_toroidal_on_axis`, `kappa`, `triang` and calls `PlasmaCurrent.plascar_bpol`. **Disjoint reads-sets, no shared body at all.** Two arms and not nine, because PROCESS's own test is binary — nine occupants would invent eight distinctions the source does not make |
| `.pulse.i_pulsed_plant` + `.times.pulsetimings` + `.times.i_t_current_ramp_up` | see below | `(1, 0, —)` | **split**, 4 arms, joint | `physics.py:463-498`. Neither switch decides it alone, so one family with a joint arm index — the shape `indat.py::_energy_storage_arm` and `_plasma_power_loss_arm` already use |
| `.physics.i_plasma_ignited` | `0`, `1` | `0` | **split**, 2 arms | `physics.py:793-798`: the `NON_IGNITED` arm reads `.current_drive.p_hcd_injected_total_mw`, the `IGNITED` arm passes the literal `0.0`. One read of difference, and it is a cross-area edge |
| `.divertor.n_divertors` | `0`, `1`, `2` | — | **out of scope** | `physics.py:850-861` splits `p_plasma_separatrix_mw` across upper/lower divertors into three fields none of which is in this slot's eight. Not ported, not decided |

### the joint pulse arm

| arm | condition | writes |
|---|---|---|
| 0 | `i_pulsed_plant != 1` and `i_t_current_ramp_up == 0` | ramp-up `= plasma_current / 5e5`, precharge `= ramp-up`, ramp-down `= ramp-up` (`physics.py:465-474`) |
| 1 | `i_pulsed_plant != 1` and `i_t_current_ramp_up != 0` | **nothing** — the three times are inputs |
| 2 *(live)* | `i_pulsed_plant == 1` and `pulsetimings == 0` | ramp-up `= plasma_current / 1e5`, ramp-down `= ramp-up` (`physics.py:476-483`) |
| 3 | `i_pulsed_plant == 1` and `pulsetimings != 0` | precharge `= max(precharge, ramp-up)`, ramp-down `= ramp-up` (`physics.py:485-498`) |

Arms 0 (since 2026-08-27) and 2 are written. Arms 1 and 3 are `UNPORTED`, and arm 3
for a reason stronger than
"not written yet": `physics.py:489-492` reads `.times.t_plant_pulse_coil_precharge` and
writes it back, so its occupant would read what it owns — cottax's hard error. It needs
either a `FixedPointFunction` or a producer split, and per the brief that is not a call
to improvise. See **OQ2**.

Arm 0 is also *not* arm 2 with a different literal, which is worth recording because it
is exactly the `istore` shape the policy debate turns on: arm 0 owns
`.times.t_plant_pulse_coil_precharge` as a **third output** that arm 2 does not write at
all. Even the literal-only reading of the exception does not reach it. That is why it
was ported (2026-08-27) as a second occupant, `PulseRampTimesContinuousDefault`, rather
than a kwarg on the first.

## calls into other models

From the ported bodies: **none.** Every one is self-contained arithmetic.

From the PROCESS code the ported bodies were extracted out of, and therefore relevant
to whoever wires the slot:

- `physics.py:314` → `PlasmaFields.calculate_surface_averaged_poloidal_field`
  (ported here, arm A only).
- `physics.py:801` → `PlasmaExhaust.calculate_separatrix_power` (ported here; out of
  `exhaust.md`'s stated scope, which deliberately left it — **if that scope widens, one
  of the two copies must go**).
- `physics.py:500-521` → `process/models/pulse.py::PulseTimings` (already ported as
  `models/stellarator/initialization.py::PulseDurations`; **reused, not re-ported**).
- `physics.py:373-392` → `PlasmaFields.calculate_total_magnetic_field`, producing
  `.physics.b_plasma_total`, which `PlasmaEnergyFromBeta` reads. **Unported**, and the
  one input of this unit's closure with no producer anywhere (OQ1).
- `physics.py:734-749` → `radiation_power.calculate_radiation_powers` (already ported
  and registered as three shared nodes).

## JAX-difficulty flags

- **F1 — `force_positive_separatrix_power` is `0/0` at exactly `x == 0`**, severity
  `documented`. `x / (1 - exp(-x))` has the removable singularity `→ 1` at the origin;
  PROCESS evaluates it as `0.0/0.0` and returns `nan` with a `RuntimeWarning`, not a
  raise. The port reproduces that rather than inventing the limit — `reference_domain_
  errors` does not apply (PROCESS does not signal), and a `jnp.where` here would be a
  behaviour change, not a guard. `test_gradient_finite_at_zero` skips the point on its
  own terms (value non-finite ⇒ no gradient claim), which is the correct outcome and not
  a suppression.
- **F2 — `exp(-x)` under/overflow**, severity `minor`. For `x ≳ 745` the exponential
  underflows to `0.0` and the transform is the exact identity; for `x ≲ -745` it
  overflows to `inf` and the result is `-0.0`. Both match `numpy` bit for bit, so the
  value test carries them; noted because a fuzz bound wider than the declared
  `(-20, 400)` would start exercising them.
- **F3 — no fractional powers, no `sqrt`, no `log` anywhere in this unit.** The
  `safe_pow`/`safe_sqrt` treatment (`next_steps.md` §9's trap) has nothing to apply to
  here — every body is `+`, `-`, `*`, `/` and one `exp`. `test_gradient_finite_at_zero`
  therefore passes structurally rather than by care.
- **F4 — division by a read that can be zero**: `len_plasma_poloidal` in
  `calculate_surface_averaged_poloidal_field_amperes`, `p_plasma_heating_mw`-style. Not
  guarded, because PROCESS does not guard it and a zero poloidal perimeter is not a
  reachable physical state (`plasma_geometry.py` is its sole producer and it is a
  perimeter). Recorded, not registered in `_harness/boundary.py`.
- **F5 — no in-place mutation, no arrays, no loops, no dynamic shapes.**
- **F6 — no CoolProp, no external library beyond `numpy`/`jax`.**
- **F7 — no `ProcessValueError` on a traced quantity.** The only raises in the source
  region are on switch values (`i_alphaj` at `physics.py:345`, `i_beta_norm_max` at
  `:3805`), which resolve at graph-assembly time and are outside this closure.

## suspected defects in PROCESS

Convention: **documented, not fixed.** Nothing in `process/` was touched.

**D1 — the tokamak does not clip the radiation densities the stellarator clips.
Confirmed by reading both call sites.** `stellarator.py:2153-2158` applies
`max(pden_plasma_core_rad_mw, 0.0)` and `max(pden_plasma_outer_rad_mw, 0.0)`;
`physics.py:750-766` applies neither. The same `calculate_radiation_powers` feeds both.
On a tokamak a negative core radiation density therefore propagates into
`.physics.p_plasma_inner_rad_mw` (`:758-760`) and, through
`confinement_time.py`'s `power_loss` arm, into `.physics.p_plasma_loss_mw` — where the
stellarator floors it. Whether the clip is a stellarator-specific physical correction or
a bug fix that was never applied to the other caller is not decidable from the source;
both readings are plausible, which is why it is documented. **Not measured to be live**
on `large_tokamak_eval` — this is a read of two call sites, not an instrumented run, and
`next_steps.md` §11.7's history is the reason not to promote it further.

**D2 — `.physics.p_plasma_separatrix_mw` holds two different values within one pass,
and three call sites read the first.** Confirmed by reading. `physics.py:800-809` writes
it; `:811-826` and `:832` read it; `:843-845` overwrites it with
`x / (1 - exp(-x))`; everything after reads the second. The source comments the
overwrite as a "KLUDGE" but nothing marks the three intervening reads as deliberately
pre-transform. If they are deliberate, this is a naming defect (one field, two
quantities); if they are not, `p_plasma_separatrix_rmajor_mw`,
`p_div_bt_q_aspect_rmajor_mw` and the whole scrape-off-layer model are computed from a
value the rest of the code considers unphysical. Not decidable from the source.

**D3 — arm 3 of the pulse dispatch reads and writes the same field.**
`physics.py:489-492`: `t_plant_pulse_coil_precharge = max(t_plant_pulse_coil_precharge,
t_plant_pulse_plasma_current_ramp_up)`. In PROCESS's re-run-until-idempotent loop this
is a monotone ratchet across passes, not a pure function of the inputs: the value
depends on how many times `Caller.call_models` has run. Unconfirmed as behaviourally
significant (arm 3 is not live on any tracked input checked here), but it is the reason
that arm cannot be ported as an `ExplicitFunction` at all.

**D4 — `.times.t_burn_0 = .times.t_plant_pulse_burn` (`physics.py:513`) is a
solver-protocol write inside a physics model.** The source comment says it exists "to
ensure that the burn time is used consistently; see convergence loop in fcnvmc1". It is
state for the evaluator, not a physics quantity, and it is written in the middle of
`Physics.run`. Not ported; recorded because a graph has nowhere natural to put it and
someone will have to decide where it goes.

## registration owed

The consolidation pass owns all of this; nothing below was edited by this unit.

1. **`unit_registry.md`** — a new row. Suggested text:
   `| 25 | physics/physics.py | tokamak arm: the closure producing tokamak_boundary.md's eight .tokamak.physics variables | functional_process/_audit/units/models/physics/physics.md | draft — 7 pure functions, 8 nodes (2 family heads). Ports the tokamak halves of three divergences: no radiation clip (vs. stellarator.py:2153-2158), the twice-written p_plasma_separatrix_mw (physics.py:800-809 then :843-845, resolved with one mint), and the pulsetimings dispatch whose only read in process/models/** is physics.py:476. Two of the eight outputs need no port at all — PulseDurations already is PulseTimings. |`
   Without it, `test_every_record_file_is_in_the_registry` fails.
2. **`models/tokamak/namespace.py`** — occupants for `physics`, `plasma_fields`,
   `plasma_beta` (and possibly `pulse`, OQ4).
3. **`indat.py`** — three arm functions and their dicts, plus `UNPORTED` entries.
   Spelled out in this agent's final report.

## open questions

1. **Nothing produces `.physics.b_plasma_total` on a tokamak.**
   `PlasmaEnergyFromBeta` reads it; `physics.py:373-376` writes it via
   `PlasmaFields.calculate_total_magnetic_field`, which is unported. The stellarator has
   the identical formula ported already
   (`models/stellarator/plasma_physics.py::calculate_total_field`, `sqrt(bt² + bp²)`)
   but bound to a stellarator node. Either that node is generalised, or
   `plasma_fields.py` gains a second occupant. One line either way; not done here
   because `plasma_fields.py` is not this agent's file and the target list did not name
   it.
2. **How should pulse arm 3 (`pulsetimings != 0`) be expressed?**
   `t_plant_pulse_coil_precharge = max(t_plant_pulse_coil_precharge, ...)` is a node
   reading what it owns (D3). `FixedPointFunction` is the brief's stated mechanism, but
   the brief also says to check the real producer first — and here the "real producer"
   is the input file, with PROCESS's outer loop turning it into a ratchet. Left
   `UNPORTED` pending a decision.
3. **Three writes inside this closure's blocks are deliberately not ported**, because
   they are outside the eight and nothing in the assembled graph reads them yet:
   `.physics.p_plasma_sync_mw` (`physics.py:755-757`), `.times.t_burn_0` (`:513`, D4)
   and `.physics.e_plasma_beta_thermal` (`:3905-3909`). Each is one line and one node
   whenever a consumer appears. Recorded so that "absent" is legible as a decision
   rather than an oversight.
4. **Does the pulse ramp-time family belong in `.tokamak.physics` or `.tokamak.pulse`?**
   The code is in `physics.py` and this record put it under `.tokamak.physics`, but
   `models/tokamak/namespace.py`'s `pulse` slot docstring already names decision 15
   (`pulsetimings`) as its own, and `pulse.py::Pulse` is where every other pulse-timing
   concern lives. A slot choice, not a re-port.
5. **`.physics.p_plasma_separatrix_mw_raw` — mint or fold?** See §"the one minted name".
   The mint is what is written; folding is a two-line change. Needs the orchestrator's
   call because it is a dual-ownership resolution.

## 2026-08-27 — pulse ramp-time arm 0 (`PulseRampTimesContinuousDefault`)

Ported the `i_pulsed_plant != 1 and i_t_current_ramp_up == 0` arm of the ramp-time
family, the single occupant that blocked **both**
`tests/regression/input_files/spherical_tokamak_eval.IN.DAT` and
`st_regression.IN.DAT` from assembling.

**Evidence for the combination ported.**

- `process/models/physics/physics.py:464` — `if self.data.pulse.i_pulsed_plant != 1:`;
  `:465` — `if self.data.times.i_t_current_ramp_up == 0:`; `:466-474` — the three
  writes, all `= plasma_current / 5.0e5` (ramp-up at `:466-468`, precharge at
  `:469-471`, ramp-down at `:472-474`), in that order.
- `spherical_tokamak_eval.IN.DAT:312` — `i_pulsed_plant = 0`; `st_regression.IN.DAT:2979`
  — `i_pulsed_plant = 0`. Neither file sets `i_t_current_ramp_up`, whose PROCESS default
  is `0` (`process/data_structure/times_variables.py:44`), and neither sets
  `t_plant_pulse_coil_precharge` (`st_regression.IN.DAT:3035` has it commented out), so
  the arm's ownership of precharge collides with no input leaf. Both files therefore
  select arm 0 exactly.
- No defect found in the arm: it is three straight-line assignments with no branch, no
  aliasing surprise, and no read of what it writes (unlike arm 3).

**What was written.** `calculate_continuous_plant_ramp_times` (pure, 1-in/3-out, next
to its sibling in `functional_process/models/physics/physics.py`) and
`PulseRampTimesContinuousDefault(PulseRampTimes)` with three `OutputInto(times)` ports
declared in PROCESS's write order. Case: `TestContinuousPlantRampTimes`
(`Tier1Contract`, reference transcribed from `physics.py:465-474` — no PROCESS
staticmethod exists; legacy point `plasma_current = 16528278.760008096` from
`tests/unit/models/physics/test_physics.py:392`, plus 5 fuzz draws over
`(1e6, 3e7)` A, seed 0). Test file green plain (116 passed, 116 skipped) and with
`--fp-gradients` (232 passed), cottax pinned at jaxgraph `db4f025`.

**Registration** (a deliberate, sanctioned deviation from the porting-agent convention —
the orchestrator merges this worktree): `indat.py` gained the import, the
`PULSE_RAMP_TIMES` entry `0: PulseRampTimesContinuousDefault` (with a docstring note on
which files select each arm), and lost the `("pulse_ramp_times_arm", 0)` `UNPORTED`
row. Arms 1 and 3 stay `UNPORTED` unchanged.

**Frontier probe** (`machine_from_indat` + `graph_for`, both files, with arm 0 wired):
neither assembles yet; both moved past this refusal to the same next one, verbatim:

    NotImplementedError: i_hcd_primary == 13 is a real PROCESS branch but is not
    ported: needs `ElectronCyclotron.electron_cyclotron_freethy` and the
    `i_ecrh_wave_mode` switch inside it; not written. **The next one worth writing**
    -- `spherical_tokamak_eval.IN.DAT:133` and `st_regression.IN.DAT:2522` both
    select it, and its wall-plug block is the one already ported

(identical for both files). Nothing beyond arm 0 was ported; the probe is measurement.


## 2026-08-27 — `plasma_ohmic_heating` ported (cold-boundary wave)

`cold_boundary.md` producer 3. `.physics.res_plasma` was one of the six cold boundary
zeros: `plasma_inductance.volt_seconds` computes `v_plasma_loop_burn = plasma_current
* res_plasma * f_c_plasma_inductive`, so the cold burn time was `abs(0)/0 - 10 = nan`
(jointly with producer 4's `vs_cs_pf_total_burn`). Ported as `plasma_ohmic_heating` +
`PlasmaOhmicHeating`, a new **fifth slot** `ohmic_heating` of `.tokamak.physics` —
unswitched (`Physics.run` computes it unconditionally, `:768-778`), so an instance
default per the `plasma_beta` rule.

The port reproduces a **live PROCESS defect**: the neo-classical enhancement guard is
the chained comparison `1.0 if 2.5 >= rmajor / rminor <= 4.0 else 4.3 - 0.6 * rmajor /
rminor` (`physics.py:1675`), which Python evaluates as `(2.5 >= A) and (A <= 4.0)`,
i.e. `A <= 2.5` — not the IPDG89 comment's "aspect ratios in the range 2.5 to 4.0".
Reproduced as `jnp.where(A <= 2.5, 1.0, 4.3 - 0.6*A)`; on the reference machine
(`A = 3.0` exactly) the enhancement arm is taken either way and lands on
`f_res_plasma_neo = 2.5`, `res_plasma = 4.0496e-9` — the converged values
`cold_boundary.md` records. Belongs beside this record's other suspected-defect
entries; ported faithfully, not fixed.

Two shells dropped, neither arithmetic: the `aspect` parameter (read only by the
negative-resistance `logger.error`, whose condition a traced function cannot raise on)
and that logger. The staticmethod's `zeff` parameter keeps its spelling in the pure
function; the node port is `.physics.n_charge_plasma_effective_vol_avg`, the actual
storage field `Physics.run` passes (`:782`) — the first draft of this wave declared
`zeff=From(physics)` and the cold probe caught it as an ungrounded boundary input
(`PhysicsData` has no `zeff`), which is the declaration-surface rule enforcing itself.

Data footprint: reads `.physics.f_c_plasma_inductive` (from `current_fractions`),
`.physics.kappa95` (plasma_geom), `.physics.plasma_current`, `.physics.rmajor`,
`.physics.rminor`, `.physics.temp_plasma_electron_density_weighted_kev` (profiles),
`.physics.vol_plasma`, `.physics.n_charge_plasma_effective_vol_avg` (composition),
`.physics.plasma_res_factor` (run input, `large_tokamak_eval.IN.DAT` sets `0.7`);
writes all four of the writeback's fields — `.physics.pden_plasma_ohmic_mw`,
`.physics.p_plasma_ohmic_mw` (closes the standing boundary read of
`separatrix_power` and `PlasmaPowerLoss`), `.physics.f_res_plasma_neo`,
`.physics.res_plasma`. No cycle created (`Blocking.scc`, both machines, measured this
wave: the node sits downstream of the density/fusion SCC and upstream of the merged
PF/volt-second SCC, on neither).

Tier 1; `test_physics.py::TestPlasmaOhmicHeating` diffs the real staticmethod (legacy
point = the converged operating point read off a live `SingleRun`; fuzz bounds
straddle the defect's kink at `A = 2.5` from both sides).

## 2026-08-27 — `PlasmaBeta.run`'s limit block ported (missing-producer wave, CS/physics half)

`optimise_design.md` §11.5's constraint-24 rows. All three of the fields
`constraint_24` compares — `.physics.beta_thermal_vol_avg`,
`.physics.beta_toroidal_vol_avg` and the bound `.physics.beta_vol_avg_max` — were
boundary constants at `0`, while PROCESS's own solve moves them to
`0.027869 / 0.033109 / 0.057040`. The constraint therefore compared a frozen zero
against a frozen zero, and no design variable could move either side of it.

**Four pure functions and four nodes, in a new namespace.**
`calculate_beta_norm_max_wesson` (`physics.py:3941-3974`),
`calculate_beta_limit_from_norm` (`:4180-4235`) — both PROCESS `@staticmethod`s taken
unchanged — plus `calculate_toroidal_beta` (`:3818-3822`) and
`calculate_thermal_beta` (`:3831-3835`), which are inline assignments in
`PlasmaBeta.run` with no separable PROCESS callable and are transcribed term for term.

**`.tokamak.plasma_beta` was a node slot and is a namespace now.** The rename
`.tokamak.plasma_beta` → `.tokamak.plasma_beta.energy_from_beta` is visible in every
pin. The alternative was a second, sibling `Tokamak` slot for the limits, which would
have split one PROCESS `Model` across two slots; `models/tokamak/namespace.py`'s own
docstring already allowed for a slot holding either kind, so this is the anticipated
case rather than a new licence.

**A chain, not a leaf, and the chain is what the gap actually was.**
`beta_vol_avg_max` reads `.physics.beta_norm_max`, which was *itself* a moving boundary
constant (`3.0` cold against PROCESS's converged `5.0273`) — so porting only
`calculate_beta_limit_from_norm` would have closed the §11.5 row and left the value
43% wrong and the derivative still dead. `get_beta_norm_max_value` (`:3723-3743`) is
therefore ported as well, as a switch slot on `.physics.i_beta_norm_max` with the
Wesson occupant written (`indat.BETA_NORM_MAX`). PROCESS computes all five scalings
unconditionally and then selects; the four unselected ones are dead work and are not
computed, per the wave's computes-then-selects policy. The `USER_INPUT` arm has **no
occupant and cannot have one** — `model_map` returns `physics_data.beta_norm_max`
itself, so its honest occupant is no node at all; `BETA_NORM_MAX`'s docstring records
that this makes `_slot_occupant` give a `ValueError` where a `NotImplementedError`
would read better, and why that is not fixed here.

**`.physics.i_beta_component` is not a switch of this block**, which is the useful
negative result: it looks like one (it is even named as decision 7 in §A) and it
selects only which *already computed* beta `constraint_24` compares against the one
limit PROCESS computes regardless. It stays a static kwarg of the ported constraint.

No cycle created (`Blocking.scc`, tokamak: the three raw cycles are unchanged in
membership — 4-node build/winding-pack, 8-node density, 9-node PF/volt-second). The
stellarator graph is untouched: `.tokamak.*` is not on it, and its harness numbers are
identical line for line.

Data footprint: reads `.physics.ind_plasma_internal_norm` (plasma_inductance),
`.physics.b_plasma_toroidal_on_axis`, `.plasma_current`, `.rminor`,
`.beta_total_vol_avg`, `.b_plasma_total`, `.beta_fast_alpha` (`fast_alpha_beta`) and
`.beta_beam` — the last a boundary input whose producer (`beam_fusion`) is unported and
is zero on this machine, declared rather than folded away. Writes
`.physics.beta_norm_max`, `.beta_vol_avg_max`, `.beta_toroidal_vol_avg`,
`.beta_thermal_vol_avg`.

Tier 1 throughout; four `Tier1Contract`s in `test_physics.py`, legacy points read off
the converged `large_tokamak_eval` run, green plain and under `--fp-gradients`.
