---
kind: model-unit
status: draft
confidence: medium
---

**Ported and registered (cold-boundary wave, 2026-08-27).** `pfcoil/volt_seconds.py` /
`test_volt_seconds.py`: `calculate_pf_coil_turn_currents` and
`calculate_pf_cs_volt_seconds`, tier-1. Two cottax nodes, `PFCoilTurnCurrents`
(`.tokamak.pf_coil.turn_currents`) and `PFCoilVoltSeconds`
(`.tokamak.pf_coil.volt_seconds`).

This unit is `cold_boundary.md` producer 4: `.pf_coil.vs_cs_pf_total_burn` was one of
the six cold boundary zeros, and with `.physics.res_plasma` (producer 3) it made the
cold `pulse.burn_time` compute `abs(0)/0 - 10 = nan` -- the last of the 11 non-finite
roots. The package's `__init__.py` had scoped both blocks out of the sizing wave by
name ("everything reachable only through ... `vsec()`, `waveform`'s downstream
`c_pf_coil_turn` ... is UNPORTED"); this record is that debt paid.

## source

`process/models/pfcoil.py`:

| lines | what |
|---|---|
| `1082-1111` | tail of `pfcoil()`: per-turn circuit currents at the six waveform time points (`c_pf_coil_turn`) |
| `1615-1720` | `PFCoil.vsec` -- volt-second capability of the PF/CS system |

`vsec` is called third in `PFCoil.run` (`:79`), after `pfcoil()` and `induct()`; the
turn-current block is the last thing `pfcoil()` itself does, downstream of
`waveform()` (called at `:732`).

## data footprint

`PFCoilTurnCurrents` (`pfcoil.py:1082-1111`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.f_c_pf_cs_peak_time_array` | read | explicit-arg | `:1088`; owned by `PFCoilCurrentWaveform` |
| `.pf_coil.c_pf_coil_turn_peak_input` | read | explicit-arg | `:1091`; run input (`large_tokamak_eval.IN.DAT:241`) |
| `.pf_coil.c_pf_cs_coils_peak_ma` | read | explicit-arg | `:1092` -- read for its *sign* only (`math.copysign`); owned by `PFCoilCurrentWaveform` |
| `.physics.plasma_current` | read | explicit-arg | `:1107-1109`, the plasma circuit's flat-top rows |
| `.pf_coil.c_pf_coil_turn` | write | explicit-arg | whole `(NGC2, 6)`; rows 8-21 stay `0.0` exactly as PROCESS leaves them |

`PFCoilVoltSeconds` (`vsec`, `iohcl != 0` arm):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.iohcl` | read | switch | `1` on this arm; part of the package's joint predicate `_pf_coil_system_arm`, no parameter |
| `.pf_coil.n_pf_cs_plasma_circuits` / `.n_cs_pf_coils` | read | topology constants | `8`/`7`, graph-assembly data per the package `__init__.py`'s rule |
| `.pf_coil.ind_pf_cs_plasma_mutual` | read | explicit-arg | plasma row only (`[7, 0..6]`); owned by `PFCoilInductance` |
| `.pf_coil.c_pf_coil_turn` | read | explicit-arg | columns `1`/`2`/`4`, rows `0..6`; owned by `PFCoilTurnCurrents` |
| `.pf_coil.nef` | write | local-intermediate | loop bookkeeping, `n_pf_cs_plasma_circuits - 2 = 6` on this arm; not owned, same disposition as `inductance.md` gives it |
| `.pf_coil.vsdum` | write | local-intermediate | scratch; the loops' common subexpression, not materialised in the port |
| `.pf_coil.vs_pf_coils_total_ramp` | write | local-intermediate | reporting-only readers (`outvolt`) |
| `.pf_coil.vs_cs_ramp` / `.vs_cs_burn` | write | local-intermediate | reporting-only readers |
| `.pf_coil.vs_cs_pf_total_ramp` | write | local-intermediate | reporting-only readers (`outvolt`, plot summaries) |
| `.pf_coil.vs_pf_coils_total_burn` / `.vs_pf_coils_total_pulse` / `.vs_cs_total_pulse` | write | local-intermediate | reporting-only readers |
| `.pf_coil.vs_cs_pf_total_burn` | write | explicit-arg | **the cold-boundary zero**; read by `pulse.py:159` -> `calculate_burn_time` |
| `.pf_coil.vs_cs_pf_total_pulse` | write | explicit-arg | read by constraint 12 (`core/solver/constraints.py:582`, sign-flipped) |

Reader census for the local-intermediate rulings: grep over `process/` finds the seven
unowned stores only in `outvolt`/`outpf`, `core/io/mfile/comparison.py` and
`core/io/plot/summary.py` -- reporting, not computation. `c_pf_coil_turn` itself *is*
owned (not a local of `vsec`), because the CS stress-profile chain reads it
computationally (`pfcoil.py:4235`, UNPORTED -- the `cs_fatigue` chain) and the
producer-side whole-array argument of `inductance.md` applies unchanged.

## scope discipline

- **`vsdum` is not materialised.** It is written and read only inside `vsec`, column
  by column, as `M * c_turn` -- a common subexpression, folded into the sums.
- **`nef` is not owned**, per `inductance.md`'s precedent for the same field: loop
  bookkeeping fixed by the topology the package bakes.
- **The seven unowned `vs_*` sums stay locals** (reporting-only readers, measured --
  see the footprint table). Whoever gives `outvolt` a port owes them fields.

## proposed signature(s)

```python
def calculate_pf_coil_turn_currents(
    f_c_pf_cs_peak_time_array, c_pf_coil_turn_peak_input, c_pf_cs_coils_peak_ma,
    plasma_current,
) -> jnp.ndarray:  # c_pf_coil_turn, (NGC2, 6)

def calculate_pf_cs_volt_seconds(
    ind_pf_cs_plasma_mutual, c_pf_coil_turn,
) -> tuple[float, float]:  # vs_cs_pf_total_burn, vs_cs_pf_total_pulse; iohcl=1 baked
```

## cottax nodes

`PFCoilTurnCurrents` -> `.tokamak.pf_coil.turn_currents`; `PFCoilVoltSeconds` ->
`.tokamak.pf_coil.volt_seconds`. Both instance defaults, like every slot in this
package -- the whole package is one occupant set behind `indat._pf_coil_system_arm`'s
joint predicate, and neither block has a switch of its own inside that arm.

## tier signal

**Tier 1**, both functions. No iteration, no CoolProp, no calls into another model.

**Sample provenance.** No PROCESS unit test touches either block
(`tests/unit/models/test_pfcoil.py` covers neither `vsec` nor the `pfcoil()` tail).
`calculate_pf_cs_volt_seconds` diffs against the real `PFCoil.vsec` directly -- unlike
most of the package it *is* a separable callable -- with the legacy point read off a
converged in-process `SingleRun` and fuzz at +-20% around it.
`calculate_pf_coil_turn_currents` has no separable PROCESS callable (it is `pfcoil()`
inline), so its oracle is `test_masses.py`'s whole-`pfcoil()` chain contract, which
returns `c_pf_coil_turn` on both sides since this wave -- the same disposition
`currents.md` records for the other inline blocks.

## switches touched

| switch | reachable values | live | decision | evidence |
|---|---|---|---|---|
| `.build.iohcl` | `0`, `1` | `1` | **joint** (part of `_pf_coil_system_arm`, refused before assembly) | `vsec`'s `iohcl == 0` arm drops the CS terms and one circuit (`pfcoil.py:1622-1624`, `:1648`, `:1679`) -- a different reads-set and a different `nef` |

## deviations from PROCESS

- **`math.copysign` becomes `jnp.copysign`** (identical signbit semantics, including
  the `-0.0` corner).
- **Summation order**: PROCESS accumulates the six PF terms in a Python loop;
  the port uses `jnp.sum` over the same six elements. Agreement is within the tier-1
  default tolerance on every sample.

## the cycle merge

Registering these two nodes merged the five-node PF ring and the two-node
volt-second/burn-time ring into **one nine-node SCC** (`burn_time` reads
`vs_cs_pf_total_burn`; `vsec` reads the inductance matrix and the turn currents;
`flux_swing` reads `vs_plasma_ramp_required` from `plasma_inductance.volt_seconds`,
which reads `t_plant_pulse_burn` back from `burn_time`). Measured on the merged cycle
(27 owned variables, 18 with closing readers): no single cut suffices; exactly two
pairs do, both pairing `t_plant_pulse_burn` with a PF edge the pre-merge measurement
had already rejected as not PROCESS's carried state; the standing `mda.CUTS` trio
(`t_plant_pulse_burn` + PROCESS's `first_call` seed pair) is sufficient and each
member necessary. No `CUTS` change; `test_mda.py::
test_the_merged_pf_volt_second_burn_time_cycle_keeps_its_cuts` pins the table.

## open questions

None. The `iohcl == 0` arm and the CS stress-profile consumer of `c_pf_coil_turn`
stay UNPORTED with the rest of their chains.


## 2026-08-30 (evening) -- the spherical tokamaks' PF coil system, arm 2

`next_steps.md` §18.2 listed five of the eight blockers stopping
`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` as `pf_coil_system_arm`
deviations (`-1`, `-2`, `-3`, `-6`, `-7`). All five are closed. The package now carries
a `PFCoilTopology` (`models/pfcoil/__init__.py`) instead of five loose module
constants, and `indat._pf_coil_system_arm` has a third positive arm, `2`, for a machine
with **no central solenoid**: `iohcl = 0`, `n_pf_coil_groups = 4`,
`i_pf_location = (2, 3, 3, 4)`, `n_pf_coils_in_group = (2, 2, 2, 2)`,
`i_pf_superconductor = 9`, picture-frame TF. `.tokamak.cs_coil` is `None` on that arm.

**`-3` was a refusal that outlived its cause, and that is a correction to this
record's own frontier.** The predicate refused `itart == 1` *or* `itartpf != 0`.
Measured over `process/`: `itartpf` is read in exactly two places
(`pfcoil.py:1250`, `:411`) and both guard on `itart == 1 **and** itartpf == 0`, and
`core/init.py:640` overwrites `i_pf_location[:3]` under the same conjunction. Both
tracked ST files set `itartpf = 1`, so **neither ever reaches PROCESS's Peng and
Strickler ST arm** -- their PF coil system takes the conventional placement and the
conventional SVD current solve throughout. The predicate is now the conjunction, and
the ST arm stays UNPORTED with nothing reaching it.

**What changed here.** `calculate_pf_volt_seconds_no_central_solenoid` ports `vsec` at
`iohcl = 0` (`pfcoil.py:1622-1626`): `nef = n_pf_cs_plasma_circuits - 1` rather than
`- 2`, so the PF loop covers every circuit but the plasma, and `vs_cs_ramp`/
`vs_cs_burn` are **never assigned** (`:1647`, `:1677`, both guarded) so PROCESS's
totals are the PF sums plus those fields' storage default `0.0`. Reproduced by leaving
them out of the sums rather than adding a zero: the two are the same number and only
one of them is the same statement.

`PFCoilVoltSecondsNoCentralSolenoid` is a subclass -- same two reads, same two outputs,
different body -- because `vsec` reads the inductance matrix and the turn currents whole
on both arms. `PFCoilTurnCurrents` is one node for both, with the topology deciding
which row is the plasma's. Bit-exact against `vsec` in the scratch verification;
**an ST case in `test_volt_seconds.py` is owed** -- see `next_steps.md`.
