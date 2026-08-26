---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial — minimal closure only).** `pulse.py` / `test_pulse.py`:
`calculate_burn_time`, tier-1, the sole occupant of a new node `PulseBurnTime`.
Produces `.times.t_plant_pulse_burn`. `tohswg` (`.constraints.t_current_ramp_up_min`)
is UNPORTED — see "Not ported: `tohswg`" below. No `unit_registry.md` row and no
`next_steps.md` edit; registration is the consolidation pass's job (wave-1 brief) — see
"registration instructions" at the end.

## source

`process/models/pulse.py`, 316 lines (`tokamak_call_surface.md`'s own count: 236
entered LOC, 12 entered functions, 3 shared with other units — the `PulseTimings`
dataclass, whose properties are `stellarator/initialization.py::PulseDurations`'s
already-registered `calculate_pulse_durations`). Site of `caller.py:322`
(`models.pulse.run()`), `tokamak_call_surface.md` §A row 5.

`Pulse.run()` (`:142-162`) does exactly two things, both gated by
`self.data.pulse.i_pulsed_plant == 1` — a **topology switch**: `!= 1` means neither
call happens and this file contributes nothing to the graph at all.
`large_tokamak_eval.IN.DAT:330` sets `i_pulsed_plant = 1`, so this is live on the
reference run.

1. `self.tohswg(output=output)` (`:154`) — computes `.constraints.
   t_current_ramp_up_min`.
2. The burn-time calculation (`:158-162`) — `self.data.times.t_plant_pulse_burn =
   self.calculate_burn_time(...)`, `calculate_burn_time` itself at `:275-316`, already
   a `@staticmethod` with zero `self.data` access — the extraction seam is free, same
   shape as `density_limit.py`'s `calculate_density_limit`.

`PulseTimings` (`:15-129`) is a frozen dataclass of properties (`plasma_present`,
`no_burn`, `total`, the cumulative-time tuples) — already ported term-for-term as
`stellarator/initialization.py::calculate_pulse_durations`/`PulseDurations` (that
unit's own record checked it against `pulse.py:71-95`), and registered for **both**
devices via `.tokamak.pulse.durations` (`models/physics/tokamak_namespace.py:94-127`).
Not re-derived here.

`Pulse.output` (`:138-140`) just calls `run(output=True)` — no separate reporting
body, unlike most other units in this wave.

## prior work this record does not duplicate

Per the wave-1 dispatch brief, three producers touching this file's outputs are
already ported and registered elsewhere, and this record extends rather than
contradicts them:

- `.times.t_plant_pulse_plasma_present` / `.times.t_plant_pulse_total` (and
  `.times.t_plant_pulse_no_burn`) — `stellarator/initialization.py::PulseDurations`,
  registered as `.tokamak.pulse.durations` for both devices.
- `.times.t_plant_pulse_plasma_current_ramp_up` / `_down` — `functional_process/models/
  physics/physics.py::PulseRampTimesPulsedDefault`, registered as
  `.tokamak.pulse.ramp_times` (the `i_pulsed_plant == 1, pulsetimings == 0` arm; arm 3,
  `pulsetimings != 0`, is UNPORTED there for a documented `FixedPointFunction`/ratchet
  reason — `physics.md` OQ2).
- `.times.t_burn_0` (`physics.py:513`) — deliberately not ported anywhere
  (`physics.md` D4): a solver-protocol write (`# Reset second ... value`,
  `process/models/physics/physics.py:509`), not a physical quantity.

This record's only new closure is `.times.t_plant_pulse_burn`.

## data footprint

### `calculate_burn_time` (`:275-316`) — the ported closure

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.vs_cs_pf_total_burn` | read | explicit-arg | total volt-seconds in CS+PF coils available for burn; `pfcoil_variables.py:464`, default `0.0`. Producer is `pfcoil.py` (unported, another agent's wave-1 scope) — an ordinary boundary input here |
| `.physics.v_plasma_loop_burn` | read | explicit-arg | plasma loop voltage during burn; `physics_variables.py:1432`, default `0.0`. Producer is `physics.py`'s volt-second-requirement chain (not this pass's scope) |
| `.times.t_plant_pulse_fusion_ramp` | read | explicit-arg | `times_variables.py`, default `10.0`; `input.py:788` gives declared range `(0.0, 10000.0)`. A pure boundary input — no producer anywhere in `process/models/**` (grep confirms; it is set only from IN.DAT or its default) |
| `.times.t_plant_pulse_burn` | **write** | explicit-arg | sole producer on the tokamak path. `logger.error` on a negative result (`:306-314`) is a diagnostic side effect with no bearing on the returned value — dropped, same precedent as `structure.py`'s `aintmass` comment ("PROCESS logs and kludges ... dropped here as pure reporting"), except here there is not even a kludge: the value is returned unclamped either way |

No implicit reads, no switches, no calls into another model's method. This is as
clean an extraction seam as `plasma_geometry.md`'s functions 3–9 or `cryostat.md`'s
`calculate_r_cryostat_inboard` — a `@staticmethod` taking plain floats, returning a
plain float.

### `tohswg` (`:164-273`) — see "Not ported" below for the full reads list

## why this closure matters beyond `Pulse.run()` itself

`.times.t_plant_pulse_burn` is already a heavily-consumed **declared input** elsewhere
in the port, with no tokamak-path producer until now:

- `functional_process/core/solver/constraints.py::constraint_13`'s own docstring
  names `calculate_burn_time` as the "real *model* producer" it was written without,
  back when this codebase scoped only the stellarator ("out of this codebase's
  current scope"). **`icc = 13` ("Burn time lower limit") is active on
  `large_tokamak_eval.IN.DAT`** (`icc` list, line 16) — confirmed by grep, not
  inferred, so this is the reference run's own live constraint, not a hypothetical
  consumer.
- `core/solver/objectives.py`'s figures of merit 14, 16, 19 (`objective_metric_14/16/
  19`) all take `t_plant_pulse_burn` as a declared argument.
- `models/availability/availability.py`'s three capacity-factor sites (`cpfact =
  f_t_plant_available * (t_plant_pulse_burn / t_plant_pulse_total)`).
- `models/costs/costs.py`'s fuel-cost terms (several `From(times)` bindings).
- `models/stellarator/initialization.py::PulseDurations`, registered for both
  devices, reads it as a plain `From(times)` boundary input.

None of these needed to be touched or re-verified for this record — they already
declare the read correctly; they simply had no tokamak producer to bind to until this
node exists.

**Not itself flagged as a `.tokamak.pulse` boundary read in `tokamak_boundary.md`**
(that file attributes `pulse` zero boundary reads, line 151) — that method counts
model-to-model `VarPath` reads inside the currently-assembled graph's node bodies, not
solver-level reads (constraints, objectives). `t_plant_pulse_burn`'s consumers are all
of the latter kind, so the boundary doc's method does not see them; this is a limit of
that method, not evidence the output is unwanted. Flagging per the wave-1 hard rule
("if your reading of the source contradicts `tokamak_boundary.md`, say so").

## proposed signature(s)

```python
def calculate_burn_time(vs_cs_pf_total_burn, v_plasma_loop_burn, t_plant_pulse_fusion_ramp) -> float:
```

Matches PROCESS's own signature exactly, verbatim formula
(`(abs(vs_cs_pf_total_burn) / v_plasma_loop_burn) - t_plant_pulse_fusion_ramp`), minus
the dropped `logger.error`.

## cottax node

`PulseBurnTime(ExplicitFunction)`, in `functional_process/models/pulse.py`. Owns
`.times.t_plant_pulse_burn`; reads `.pf_coil.vs_cs_pf_total_burn`,
`.physics.v_plasma_loop_burn`, `.times.t_plant_pulse_fusion_ramp`. No switch of its
own — `i_pulsed_plant` decides whether the node exists in the graph at all (a topology
switch, per `naming_convention.md`'s first bullet), not a value the node's body reads.

## tier signal

**Tier 1.** No iteration, no `scipy.optimize`/`fsolve`, no calls into another model's
method, no CoolProp. Straight-line arithmetic.

**Sample provenance.** `tests/unit/models/test_pulse.py::test_calculate_burn_time_valid`
supplies three already-validated points (including a negative-`vs_cs_pf_total_burn`
case exercising `abs()`), lifted verbatim as legacy samples — per the wave-1 brief,
`Pulse.calculate_burn_time` is called in-process as the reference rather than the
hardcoded `expected` values in that test being copied; `calculate_burn_time` is a
bare `@staticmethod`, so no `DataStructure` adapter is needed (same shape as
`confinement_time.py`'s 48 scaling laws). Fuzz bounds are physically reasonable
domains, not PROCESS-declared iteration-variable bounds — none of the three arguments
is an `ITERATION_VARIABLES` entry.

## switches touched

`.pulse.i_pulsed_plant` — **topology**, not an occupant of this node. `== 1` on the
reference run; `!= 1` means `Pulse.run()` performs neither of its two computations, so
there is nothing for a `!= 1` occupant to compute. Consistent with
`tokamak_namespace.py`'s treatment of `i_hcd_calculations` ("topology, not an
occupant... answered by whether the slot is filled at all").

## calls into other models

None. `calculate_burn_time` is self-contained arithmetic.

## JAX-difficulty flags

None beyond the dropped `logger.error` (pure reporting, no value effect — see data
footprint table). `abs()` traces and differentiates cleanly (`jnp.abs`, one-sided
kink at exactly zero, standard behaviour, not a `safe_*` case — verified the sample
domain does not sit on it and the boundary-gradient test does not trip on it: zeroing
`vs_cs_pf_total_burn` gives a finite value and a finite, if kinked, gradient).

## Not ported: `tohswg` (`:164-273`)

Left unported this pass, for three independent reasons, any one of which would be
sufficient alone:

1. **Not live on the reference run.** `tohswg`'s sole output,
   `.constraints.t_current_ramp_up_min` (`:247-262`), is read only by constraint 41
   (`core/solver/constraints.py:1281-1299`, already ported: `constraint_41
   (t_plant_pulse_plasma_current_ramp_up, t_current_ramp_up_min)`), and
   `large_tokamak_eval.IN.DAT`'s `icc` list (lines 9–35) does **not** include `41` —
   confirmed by grep of the input file itself, not inferred from source. `icc = 13`
   (the burn-time constraint this pass's own output feeds) **is** present; `icc = 41`
   is not. Nothing on this reference arm reads `tohswg`'s output.
2. **Every read is PF-coil-owned, and PF coil is under concurrent edit.** `tohswg`
   reads `.pf_coil.c_pf_coil_turn` (`:182,186`), `.pf_coil.n_cs_pf_coils`
   (throughout), `.pf_coil.i_pf_conductor` (`:190`),
   `.pf_coil.p_cs_resistive_flat_top` (`:194`), `.pf_coil.c_pf_cs_coils_peak_ma`
   (`:197-199`), `.pf_coil.c_pf_coil_turn_peak_input` (`:210-212`),
   `.pf_coil.rhopfbus` (`:220`), `.pf_coil.ind_pf_cs_plasma_mutual` (`:228-231,
   235-237`), `.pf_coil.n_pf_coil_turns` (`:254-256`), plus `.pf_power.vpfskv`
   (`:224`) and `.physics.rmajor`/`.physics.plasma_current` (the only two non-PF
   reads). Every PF-coil field is a `functional_process/models/pfcoil/**` concern,
   fenced to a different agent this wave. Declaring these reads now risks binding
   against a producer that agent is mid-rewrite on.
3. **Dynamic array indexing the naming convention does not cover.** Every PF-coil
   array read above is indexed by `self.data.pf_coil.n_cs_pf_coils - 1` (and
   `ind_pf_cs_plasma_mutual` additionally by `self.data.pf_coil.
   n_pf_cs_plasma_circuits - 1`). `n_cs_pf_coils` is a run-topology count computed by
   `pfcoil.py` (`process/models/pfcoil.py:140-158`, accumulated from
   `n_pf_coils_in_group` per coil group plus one for the CS), **not** a literal the
   way `naming_convention.md`'s "Array elements" section describes
   (`f_nd_impurity_electron_array[2]`, a fixed `IterationVariable.array_index`).
   Whether `n_cs_pf_coils` can be treated as a build-time-known static count
   (plausible — it is set once from input-derived coil-group sizes and is never an
   iteration variable) or needs some other policy is not decided here. It also
   carries a genuine two-arm switch, `i_pf_conductor == SUPERCONDUCTING` vs.
   resistive (`:190-202`) — the resistive arm reads `p_cs_resistive_flat_top` and
   `c_pf_cs_coils_peak_ma`, which the superconducting arm (`r = 0.0`) does not; an
   ordinary split case per wave-1's binding policy, not itself a blocker, but one more
   reason this deserves its own pass once the PF-coil producers it depends on exist
   and are stable.

Per the wave-1 brief's hard rule ("stop on that item, port everything else, and
report it") this is reported, not improvised — see "open questions" below.

## open questions

1. **Should `tohswg` become its own node once `pfcoil.py` is ported?** Its reads are
   entirely PF-coil-owned (reason 2 above); whoever ports `pfcoil.py`'s `n_cs_pf_coils`
   accounting is the natural owner of the policy question in reason 3, and this node
   should probably be wired up in the same pass rather than by a `pulse.py`-focused one.
2. **Is `n_cs_pf_coils - 1` a build-time-static index in general, or only on this
   reference run?** If PF-coil count is fixed once a run's `IN.DAT` is parsed (never
   an iteration variable, never re-derived mid-solve), it is a legitimate static
   index and the naming convention should say so explicitly, the same way
   `naming_convention.md`'s `array_index` already does for iteration-variable
   elements. Not decided here — flagging so the eventual `pfcoil.py` port does not
   have to re-derive it from scratch.
3. **Registration:** should `PulseBurnTime` join `.tokamak.pulse` as a third slot
   (alongside `ramp_times` and `durations`), per `models/physics/tokamak_namespace.py`'s
   existing `TokamakPulse` namespace? That file is fenced this wave (shared
   namespace), so this is reported rather than done — see registration instructions
   below.

## registration instructions (for the consolidation pass)

- Add `PulseBurnTime` (from `functional_process/models/pulse.py`) as a third field on
  `functional_process/models/physics/tokamak_namespace.py::TokamakPulse`, alongside
  the existing `ramp_times: PulseRampTimes` and `durations: PulseDurations` —
  e.g. `burn_time: PulseBurnTime = PulseBurnTime()` (no switch, so no `kw_only`
  factory needed, same shape as `durations`).
- No `unit_registry.md` row exists for this unit yet; the consolidation pass should
  add one (suggested source-file grouping: alongside unit #25's `physics.py` row,
  since both are `.tokamak.pulse`-adjacent, or as its own row — `pulse.py` is a
  distinct source file from `physics.py`).
- No change needed to `functional_process/core/solver/constraints.py::constraint_13`
  or any other consumer — they already declare `t_plant_pulse_burn` as a plain
  `From(times)`/keyword read; they simply had nothing bound to it on the tokamak path
  until this node exists.
