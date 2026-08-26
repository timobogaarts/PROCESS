---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial by switch, full by function).** `models/physics/plasma_inductance.py`
/ `tests/functional_process/models/physics/test_plasma_inductance.py`:
`calculate_volt_second_requirements`, `calculate_internal_inductance_wesson`,
`calculate_internal_inductance_menard`,
`calculate_normalised_internal_inductance_iter_3` — tier-1. Three cottax nodes:
`PlasmaInternalInductanceScalings`, `PlasmaInternalInductanceNormWesson`,
`PlasmaVoltSecondRequirements`, all for the `.tokamak.plasma_inductance` slot (currently
`None` in the tokamak namespace).

**Why it exists**: `.physics.vs_plasma_ramp_required` is one of the two boundary reads
`functional_process/models/pfcoil/currents.py::CSFluxSwing` declares
(`process/models/pfcoil.py:624`). This is its producer.

## source

`process/models/physics/physics.py`:

| lines | what |
|---|---|
| `4702-4711` | `class PlasmaInductance(Model)`, its `__init__` and its empty `output` |
| `4712-4750` | `run()` — three scalings, then the `i_ind_plasma_internal_norm` dispatch |
| `4752-4764` | `get_ind_internal_norm_value` — a three-entry dict, indexed by the switch |
| `4766-4900` | `calculate_volt_second_requirements`, `@staticmethod @nb.njit` |
| `4902-4945` | `calculate_normalised_internal_inductance_iter_3`, `@staticmethod @nb.njit` |
| `4947-4975` | `calculate_internal_inductance_menard`, `@staticmethod @nb.njit` |
| `4977-5005` | `calculate_internal_inductance_wesson`, `@staticmethod @nb.njit` |
| `5007-5150` | `output_volt_second_information` — pure reporting, UNPORTED |

**Deliberate departure from the mirror-path rule, and the first one in this port.**
`PlasmaInductance` is its own `Model`, injected into `Physics` (`physics.py:207`,
`self.inductance = plasma_inductance`) and given its own slot in the model tree, but it
is parked in the same 5000-line file as `Physics`. Mirroring the *file* would put this
port inside `functional_process/models/physics/physics.py`. The unit of the port is the
model, not the file someone parked it in, so it gets its own module, record and case.
Flagged rather than done quietly: `test_harness.md`'s three-file convention is about a
shared stem across three trees, and that still holds — it is the source-side mirror that
is one level finer here.

## data footprint

`PlasmaInternalInductanceScalings` (`run()`'s first three statements, `:4721-4736`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.alphaj` | read | explicit-arg | `:4722` |
| `.physics.kappa` | read | explicit-arg | `:4728` |
| `.physics.b_plasma_surface_poloidal_average` | read | explicit-arg | `:4732` — see § "a name that does not match its argument" |
| `.physics.plasma_current` | read | explicit-arg | `:4733` |
| `.physics.vol_plasma` | read | explicit-arg | `:4734` |
| `.physics.rmajor` | read | explicit-arg | `:4735` |
| `.physics.ind_plasma_internal_norm_wesson` | write | explicit-arg | `:4721` |
| `.physics.ind_plasma_internal_norm_menard` | write | explicit-arg | `:4727` |
| `.physics.ind_plasma_internal_norm_iter_3` | write | explicit-arg | `:4731` |

`PlasmaInternalInductanceNormWesson` (`:4738-4764`, the `WESSON` arm):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_ind_plasma_internal_norm` | read | switch | `:4741` — consumed at occupant selection, not a port |
| `.physics.ind_plasma_internal_norm_wesson` | read | explicit-arg | `:4761` |
| `.physics.ind_plasma_internal_norm_menard` | read | **not declared** | `:4762` — PROCESS builds the whole dict before indexing it, so it *reads* the arm it does not take. That is the union-of-arms invented edge the occupant split exists to remove |
| `.physics.ind_plasma_internal_norm` | write | explicit-arg | `:4743` |

`PlasmaVoltSecondRequirements` (`:4766-4900`, called from `Physics.physics()` at
`:929-950`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.csawth` | read | explicit-arg | `:939` |
| `.physics.eps` | read | explicit-arg | `:940` |
| `.physics.f_c_plasma_inductive` | read | explicit-arg | `:941` |
| `.physics.ejima_coeff` | read | explicit-arg | `:942` |
| `.physics.kappa` | read | explicit-arg | `:943` |
| `.physics.rmajor` | read | explicit-arg | `:944` |
| `.physics.res_plasma` | read | explicit-arg | `:945` |
| `.physics.plasma_current` | read | explicit-arg | `:946` |
| `.times.t_plant_pulse_fusion_ramp` | read | explicit-arg | `:947` |
| `.times.t_plant_pulse_burn` | read | explicit-arg | `:948` |
| `.physics.ind_plasma_internal_norm` | read | explicit-arg | `:949` |
| `.physics.vs_plasma_internal` | write | explicit-arg | `:930` |
| `.physics.ind_plasma` | write | explicit-arg | `:931` — read by `models/pfcoil/inductance.py::PFCoilInductance` |
| `.physics.vs_plasma_burn_required` | write | explicit-arg | `:932` |
| `.physics.vs_plasma_ramp_required` | write | explicit-arg | `:933` — **the boundary read** |
| `.physics.vs_plasma_ind_ramp` | write | explicit-arg | `:934` |
| `.physics.vs_plasma_res_ramp` | write | explicit-arg | `:935` |
| `.physics.vs_plasma_total_required` | write | explicit-arg | `:936` |
| `.physics.v_plasma_loop_burn` | write | explicit-arg | `:937` |

Not ported, immediately downstream:

| VarPath | read/write | note |
|---|---|---|
| `.physics.e_plasma_magnetic_stored` | write | `:952-954`, `0.5 * ind_plasma * plasma_current**2`. Written by `Physics.physics()`, not by `PlasmaInductance` — it belongs to `.tokamak.physics`, not to this slot |

## Statefulness

**None, and it was checked rather than assumed.** PROCESS's own comment at `:4882-4884`
warns that "`t_plant_pulse_burn` on first iteration will not be correct if the pulsed
reactor option is used, but the value will be correct on subsequent calls". That is a
property of the *pulse timing* loop, not of this function: `t_plant_pulse_burn` is read
from `.times`, which `process/models/pulse.py` owns, and nothing here reads a field it
writes. There is no `first_call` flag, no accumulator and no read-before-write in this
module — unlike `masses.py`'s `itr_sum`, where the same suspicion turned out to be
justified.

## a name that does not match its argument

`calculate_normalised_internal_inductance_iter_3`'s first parameter is
`b_plasma_poloidal_vol_avg` — a *volume*-averaged poloidal field, per its own docstring —
but the only call site passes `.physics.b_plasma_surface_poloidal_average`
(`physics.py:4732`), a *surface* average. One of the two names is wrong. The port reads
the field PROCESS reads and keeps PROCESS's parameter name, so the two disagree here
exactly as they disagree there. Recorded, not corrected: `l_i(3)` is not in this pass's
boundary and changing which field is read would be a physics change, not a port.

## proposed signature(s)

```python
def calculate_internal_inductance_wesson(alphaj) -> float
def calculate_internal_inductance_menard(kappa) -> float
def calculate_normalised_internal_inductance_iter_3(
    b_plasma_poloidal_vol_avg, c_plasma, vol_plasma, rmajor) -> float
def calculate_volt_second_requirements(
    csawth, eps, f_c_plasma_inductive, ejima_coeff, kappa, rmajor, res_plasma,
    plasma_current, t_plant_pulse_fusion_ramp, t_plant_pulse_burn,
    ind_plasma_internal_norm) -> tuple  # eight values
```

## cottax node

Three `ExplicitFunction`s in `functional_process/models/physics/plasma_inductance.py`;
ownership as in the tables. Registration: the `.tokamak.plasma_inductance` slot.

## tier signal

**Tier 1** for all four. No iteration, no `self.data` access inside any of them (all
four are `@staticmethod @nb.njit`), no CoolProp, no external library, and — in
`calculate_volt_second_requirements` — not a single branch.

**Sample provenance.** Legacy points read off a converged in-process `SingleRun` of
`large_tokamak_eval.IN.DAT`; there is no `tests/unit` case for any of the four.
Every reference is PROCESS's own function called directly, so there is no adapter and no
`DataStructure` back door to argue about.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.physics.i_ind_plasma_internal_norm` | `0` `USER_INPUT`, `1` `WESSON`, `2` `MENARD` | `1` (`large_tokamak_eval.IN.DAT:311`; PROCESS default is `0`) | **split** | `physics.py:4759-4764` — the three arms read three different fields |

**UNPORTED arms**, for `indat.py`'s `UNPORTED` table:

- `i_ind_plasma_internal_norm = 2` (`MENARD`) — an ordinary sibling occupant, one line
  to add: owns `.physics.ind_plasma_internal_norm`, reads
  `.physics.ind_plasma_internal_norm_menard`. Not written because it is not this run's
  value.
- `i_ind_plasma_internal_norm = 0` (`USER_INPUT`) — **not an occupant at all**, and this
  is the interesting one. That arm maps `.physics.ind_plasma_internal_norm` to *itself*
  (`:4760`), so a node for it would read the `VarPath` it owns. That is not a
  `FixedPointFunction` case: there is no fixed point to find, because the assignment is
  the identity. What it means structurally is that on that arm the field is a **run
  input with no producer** and the slot is simply empty. Worth stating because the
  wave brief's Shape-A/Shape-B rule has a third answer — "no node" — that the four
  earlier dissolved self-loops did not need.

## calls into other models

None. `PlasmaInductance.run()` is called from `Physics.run()` (`physics.py:356`) and
`calculate_volt_second_requirements` from `Physics.physics()` (`:938`); neither is a
call *out* of this unit.

## JAX-difficulty flags

- `jnp.sqrt(eps)` and `jnp.log(8.0 / eps)`: `eps` is an inverse aspect ratio, strictly
  positive on any physical machine, so no `safe_sqrt`/`safe_pow` is needed and none is
  used. `eps == 0` would be a zero-size plasma; the fuzz bounds stay in `[0.1, 0.8]`
  rather than hiding the fact.
- `jnp.log(1.65 + 0.89 * alphaj)` in the Wesson scaling needs `alphaj > -1.854`; a
  current profile index is positive by construction.
- The Hirshman-Neilson denominator `1 - eps + beps * kappa` is positive for
  `eps < 1` with `beps, kappa > 0`. No guard in PROCESS either.
- No CoolProp, no `scipy.special`, no `scipy.optimize`. Every function here is
  `@nb.njit` on the PROCESS side, i.e. already restricted to plain arithmetic.

## open questions

- **`calculate_volt_second_requirements`'s docstring says it returns six values; it
  returns eight** (`physics.py:4780`, `:4812-4821` list seven names for an eight-tuple).
  A source documentation bug, not a behaviour one. Not fixed here — `process/` is not
  this pass's to edit — but it is the kind of thing that misleads the next reader of the
  call site.
- **Should `PlasmaInternalInductanceScalings` be three nodes?** It is one, because
  `run()` evaluates all three scalings unconditionally on every pass regardless of the
  switch, and all three are stored and reported. Splitting them would invent a structure
  the source does not have. Flagged so the choice reads as deliberate.
