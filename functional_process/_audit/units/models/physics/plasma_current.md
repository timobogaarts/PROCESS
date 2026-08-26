---
kind: model-unit
status: draft
confidence: medium
---

**Partially ported (2026-08-26), see "## ported" below.** The subset live on
`tests/regression/input_files/large_tokamak_eval.IN.DAT`: the `i_plasma_current == 4`
(IPDG89) plasma current, the unconditional cylindrical safety factor, and the
`i_alphaj == 1` / `i_ind_plasma_internal_norm == 1` (Wesson) arms of the two
current-profile switches downstream of it. No `unit_registry.md` row and no
`next_steps.md` edit — registration is the consolidation pass's job
(`next_steps.md` §4b); see "## ported"'s registration instructions.

## source

Nominally `process/models/physics/plasma_current.py` (1176 lines). **In practice the
unit spans two files**, and that is the first finding of this audit rather than a
convenience.

### the chain is not one file

`PlasmaCurrent.run()` is **empty** (`plasma_current.py:75-76`, docstring
*"PlasmaCurrent model doesn't need to be run"*). The class is a bag of scaling formulas
plus a 114-line `output()`; the *caller* is `Physics.run()`, and the three quantities
immediately downstream of the plasma current are functions PROCESS files in
`process/models/physics/physics.py`:

| step | function | site | writes |
|---|---|---|---|
| 1 | `PlasmaCurrent.calculate_plasma_current` | called `physics.py:286-301` | `.physics.plasma_current` |
| 2 | `calculate_cylindrical_safety_factor` (module-level, `physics.py:54-99`) | called `physics.py:303-310` | `.physics.qstar` |
| 3 | `Physics.calculate_current_profile_index_wesson` (`physics.py:1136-1164`) + the `i_alphaj` branch at `physics.py:334-348` | inline in `Physics.run()` | `.physics.alphaj_wesson`, `.physics.alphaj` |
| 4 | `PlasmaInductance.run()` (`physics.py:4712-4750`) | called `physics.py:356` | `.physics.ind_plasma_internal_norm{,_wesson,_menard,_iter_3}` |

Each arrow is a straight data dependency: `qstar` reads `plasma_current`
(`physics.py:306`), `alphaj_wesson` reads `qstar` (`physics.py:331`),
`ind_plasma_internal_norm_wesson` reads `alphaj` (`physics.py:4722`). All three switches
involved — `i_plasma_current`, `i_alphaj`, `i_ind_plasma_internal_norm` — are on
`_audit/tokamak_scope.md`'s list of 17 new topology decisions, and they are the only
three of the seventeen that are links of a single chain.

**Filing decision.** Ported as one unit into
`functional_process/models/physics/plasma_current.py`, with `file:line` attribution into
`physics.py` on every function that came from there. The alternative — leaving steps 2-4
for whichever pass ports `physics.py`'s remainder — would have separated a switch
family's occupants from their sole consumer across a file boundary. Recorded as a
**deviation from the source layout**, not as a claim that PROCESS's filing is wrong:
`physics.py` is 7000 lines and holds at least three other units' worth of material.
`PlasmaInductance` in particular is a listed `.tokamak.*` slot of its own in
`_audit/tokamak_boundary.md`; if a later pass gives it a unit,
`calculate_internal_inductance_wesson` and `WessonInternalInductance` move there
wholesale and this record shrinks accordingly. Flagged in "## open questions", not
decided here.

### functions in `plasma_current.py`

**21 `def`s.** Eight are structure, not computation (`__new__` ×2, `full_name` ×2,
`__init__` ×2, `PlasmaCurrent.run` — empty, and `PlasmaDiamagneticCurrent.run` belongs
to a different chain). In audit scope:

| # | function | lines | shape |
|---|---|---|---|
| 1 | `PlasmaCurrent.output` | 78-197 | reporting only; writes nothing to `data` |
| 2 | `PlasmaCurrent.calculate_plasma_current` | 199-403 | 9-way `i_plasma_current` dispatch; **the unit's head** |
| 3 | `PlasmaCurrent.calculate_all_plasma_current_models` | 405-478 | loops #2 over all nine values, for reporting |
| 4 | `PlasmaCurrent.output_plasma_current_models` | 480-593 | writes the nine `.physics.c_plasma_*` reporting fields |
| 5 | `PlasmaCurrent.calculate_cyclindrical_plasma_current` | 595-623 | `@staticmethod`, pure — shared by 8 of 9 arms |
| 6 | `PlasmaCurrent.plascar_bpol` | 625-688 | `@staticmethod`, pure — value 2 only |
| 7 | `PlasmaCurrent.calculate_current_coefficient_peng` | 690-720 | `@staticmethod`, pure — value 1 |
| 8 | `PlasmaCurrent.calculate_plasma_current_peng` | 722-780 | value 2; calls #6 |
| 9 | `PlasmaCurrent.calculate_current_coefficient_ipdg89` | 782-815 | `@staticmethod`, pure — value 4 |
| 10 | `PlasmaCurrent.calculate_current_coefficient_todd` | 817-864 | `@staticmethod`, pure — values 5/6, via a `model` int |
| 11 | `PlasmaCurrent.calculate_current_coefficient_hastie` | 866-951 | `@staticmethod`, pure — value 7 |
| 12 | `PlasmaCurrent.calculate_current_coefficient_sauter` | 953-988 | `@staticmethod`, pure — value 8 |
| 13 | `PlasmaCurrent.calculate_current_coefficient_fiesta` | 990-1016 | `@staticmethod`, pure — value 9 |
| 14 | `PlasmaDiamagneticCurrent.run` | 1065-1094 | a **different chain** (`.current_drive.f_c_plasma_diamagnetic*`, `i_diamagnetic_current`); out of this pass's scope |
| 15 | `PlasmaDiamagneticCurrent.output` | 1096-1136 | reporting |
| 16 | `PlasmaDiamagneticCurrent.diamagnetic_fraction_hender` | 1138-1154 | `@nb.njit`, pure one-liner |
| 17 | `PlasmaDiamagneticCurrent.diamagnetic_fraction_scene` | 1156-1175 | `@nb.njit`, pure one-liner |

## the extraction seam

**Clean for the coefficients, and the head is a pure function already.**
`calculate_plasma_current` (#2) takes fourteen explicit keyword arguments and touches
`self.data` in exactly one place — the exception path at `plasma_current.py:388`, inside
`except ValueError`, reporting `self.data.physics.i_plasma_current` in a message. Every
coefficient function (#5, #7, #9-#13) is an already-pure `@staticmethod`, so porting one
arm is a transcription plus one multiplication.

The three `physics.py` members are equally clean: `calculate_cylindrical_safety_factor`
is module-level and `@nb.jit`ted (so provably `data`-free),
`calculate_current_profile_index_wesson` and `calculate_internal_inductance_wesson` are
`@staticmethod`s. The only stateful thing in the whole chain is `PlasmaInductance.run()`
(step 4), and it is a five-statement shell.

## data footprint

### step 1 — `.physics.plasma_current` (`physics.py:286-301`)

`Physics.run()` hands `calculate_plasma_current` fourteen values, all `.physics.*`:
`alphaj`, `alphap`, `b_plasma_toroidal_on_axis`, `eps`, `i_plasma_current`, `kappa`,
`kappa95`, `pres_plasma_thermal_on_axis`, `len_plasma_poloidal`, `q95`, `rmajor`,
`rminor`, `triang`, `triang95`. **That is the union of nine arms**; no single arm reads
more than seven. Per-arm reads are the "switches touched" table below.

### step 2 — `.physics.qstar` (`physics.py:303-310`)

Reads `.physics.rmajor`, `.rminor`, `.plasma_current`, `.b_plasma_toroidal_on_axis`,
`.kappa95`, `.triang95`. Unswitched.

### step 3 — `.physics.alphaj` (`physics.py:330-348`)

Writes `.physics.alphaj_wesson` unconditionally (reads `.qstar`, `.q0`), then a two-way
`i_alphaj` branch writing `.physics.alphaj`.

### step 4 — `.physics.ind_plasma_internal_norm` (`physics.py:4712-4750`)

Writes four fields unconditionally before the switch: `_wesson` (reads `.alphaj`),
`_menard` (reads `.kappa`), `_iter_3` (reads `.b_plasma_surface_poloidal_average`,
`.plasma_current`, `.vol_plasma`, `.rmajor`), then selects one of three into
`.ind_plasma_internal_norm` via `get_ind_internal_norm_value`
(`physics.py:4752-4764`).

### reporting-only writes, measured

`grep -rn` over `process/` for each field, excluding `output*` methods and
`data_structure/`:

| field | non-reporting readers |
|---|---|
| `.physics.alphaj_wesson` | **none** (only `plasma_current.py:137-138`, `output`) |
| `.physics.ind_plasma_internal_norm_wesson` | **none** |
| `.physics.ind_plasma_internal_norm_menard` | **none** |
| `.physics.ind_plasma_internal_norm_iter_3` | **none** (only `core/io/plot/summary.py:2754`) |
| `.physics.c_plasma_*` (nine fields, `plasma_current.py:501-527`) | **none** (only `summary.py:12315+`) |

Thirteen owned fields whose sole purpose is print. `_iter_3` is the expensive one: it
reads `.physics.b_plasma_surface_poloidal_average` and so would give this unit a read
edge to `plasma_fields.py`'s output purely for a printed number.

By contrast the four the port does carry all have real readers:

| field | readers outside this chain |
|---|---|
| `.physics.plasma_current` | ~40 sites across `pfcoil.py`, `tfcoil/*`, `physics/*`, `power/*`, `pulse.py` |
| `.physics.qstar` | `physics/density_limit.py:82`, `physics/confinement_time.py:232/251/340`, `stellarator/stellarator.py:2313` |
| `.physics.alphaj` | `physics/plasma_profiles.py:321`, `physics/bootstrap_current.py:126/183` |
| `.physics.ind_plasma_internal_norm` | `physics/plasma_geometry.py:354`, `physics/bootstrap_current.py:154/162`, `pfcoil.py:451/570` |

### `plasma_current_MA` does not exist as a field

`process/data_structure/physics_variables.py` declares `plasma_current` (line 1156) and
no MA sibling. `plasma_current_MA` appears only as an MFILE label, written inline as
`self.data.physics.plasma_current / 1.0e6` (`plasma_current.py:100-104`) and read back
by `core/io/mfile/comparison.py` and `core/io/plot/scans.py`. There is nothing to own —
answering the dispatch brief's "and its MA sibling if PROCESS stores both": it does not.

## the cycle that is not live here

**At `i_plasma_current == 7` (Connor-Hastie) the chain closes into a genuine three-node
SCC**, and at every other value it does not:

```
plasma_current --(physics.py:306)--> qstar --(physics.py:331)--> alphaj
      ^                                                            |
      +---------- (plasma_current.py:360, value 7 only) -----------+
```

`calculate_current_coefficient_hastie` takes `alphaj` (`plasma_current.py:359-368`),
`alphaj` comes from `alphaj_wesson`, and `alphaj_wesson` comes from `qstar`, which comes
from `plasma_current`. Under `i_alphaj == 0` (user input) the loop is broken at the top
instead, so the SCC needs **both** `i_plasma_current == 7` and `i_alphaj == 1`.

PROCESS does not declare this anywhere. It is absorbed by
`Caller.call_models`'s "run the whole pipeline up to 10 times until the objective and
constraints stop changing" (`CLAUDE.md`, "Implicit cycles are hidden, not declared") —
i.e. by a Gauss-Seidel sweep over *everything*, whose convergence at this switch value
nothing checks separately.

`large_tokamak_eval.IN.DAT` sets `i_plasma_current = 4`, so **this pass's graph is
acyclic** and no `FixedPointFunction` is involved. Worth recording as an instance of the
open empirical question `CLAUDE.md` tracks ("how much of PROCESS is genuinely cyclic"):
here the answer is *switch-dependent*, which is a shape the tree has not yet had to
represent — one occupant makes a block cyclic and its eight siblings do not.

## switches touched

### `i_plasma_current` (`physics_variables.py:843`, default `4`)

`PlasmaCurrentModel`, `plasma_current.py:24-64`. Nine values, and the reads genuinely
differ per arm — this is band (d) of `switch_kwarg_survey.md`, not a static kwarg.

| value | name | reads (beyond `rminor`/`rmajor`/`q95`/`b_plasma_toroidal_on_axis`) | ported |
|---|---|---|---|
| 1 | Peng analytic fit | `eps`, `len_plasma_poloidal` | no |
| 2 | Peng divertor (TART/STAR) | `aspect` (as `1/eps`), `kappa`, `triang`; **and does not use the cylindrical current at all** | no |
| 3 | Simple ITER (cylindrical) | none — `fq = 1.0` | no |
| 4 | IPDG89 | `eps`, `kappa95`, `triang95` | **yes** |
| 5 | Todd I | `eps`, `kappa95`, `triang95` | no |
| 6 | Todd II | `eps`, `kappa95`, `triang95` | no |
| 7 | Connor-Hastie | `alphaj`, `alphap`, `pres_plasma_thermal_on_axis`, `eps`, `kappa95`, `triang95`, `b_plasma_toroidal_on_axis` | no |
| 8 | Sauter | `eps`, `kappa`, `triang` | no |
| 9 | FIESTA ST | `eps`, `kappa`, `triang` | no |

Three notes the table cannot carry:

- **Value 2 is structurally different.** `plasma_current.py:322-330` assigns
  `plasma_current` directly and `:392` skips the `fq * cylindrical` product for it
  alone. Its occupant will not share this file's shape, and it is also the arm that
  makes `plasma_fields.py`'s `b_plasma_surface_poloidal_average` different (see below).
- **Values 5 and 6 differ only in a literal-valued `model` argument** to one function
  (`calculate_current_coefficient_todd(..., model=1|2)`, `plasma_current.py:344-355`),
  with identical reads. That is the `istore` shape (`tokamak_scope.md` band (c)), which
  would argue for *one* occupant carrying a static kwarg — **and it was ruled the other
  way: two classes**, per `next_steps.md` §14.2's one-class-per-value rule. See open
  question 4, which is where the decision and its date live. Consistent with values 8/9
  immediately below, which are the same situation without the tempting shared function.
- **Values 8 and 9 have identical reads** (`eps`, `kappa`, `triang`) but different
  formulas, so under `next_steps.md` §14.2 they are still two classes — the rule is one
  occupant per value, and identical reads do not merge them.

**The compound Sauter predicate is `plasma_geometry.py`'s, not this file's.**
`plasma_geometry.py:467-469` branches on `i_plasma_current == 8 or i_plasma_shape ==
SAUTER`; that unit's `PlasmaGeometryArm` family owns the disjunction and its record's
open question 2 asks that this file's split be wired consistently rather than
re-derived. This port honours that: its occupants are keyed on `i_plasma_current` alone
and no body here mentions `i_plasma_shape`. **The coupling becomes real the day a
`Sauter` (8) occupant is added here** — the factory must then fill both slots from the
one input value.

**`plasma_fields.py` is a third consumer of the same switch.**
`plasma_fields.py:83-86` selects `.physics.b_plasma_surface_poloidal_average` on
`i_plasma_current == 2` vs. everything else, and the `== 2` arm calls this file's
`plascar_bpol`. `SurfaceAveragedPoloidalFieldAmperes` (in
`functional_process/models/physics/physics.py`) already owns that `VarPath` for the
`!= 2` arm; this unit does not touch it, and the Peng arm stays UNPORTED with the rest
of value 2. So `i_plasma_current` is answered in **three** places on this port —
`PlasmaCurrentScaling`, `PlasmaGeometryArm`, and the poloidal-field family — which is
`model_tree_design.md` §8 step 4d's "a switch is answered once" being *used* three times
rather than answered three times, the same distinction `PhysicsConfinementTime`'s `tail`
slot already records for `i_rad_loss`.

### `i_alphaj` (`physics_variables.py:951`, default `0`; `large_tokamak_eval:275` sets `1`)

`CurrentProfileIndexModel`. Two values:

| value | name | body | ported |
|---|---|---|---|
| 0 | `USER_INPUT` | `physics.py:338` — literally `self.data.physics.alphaj = self.data.physics.alphaj` | **empty slot** |
| 1 | `WESSON` | `physics.py:343` — `alphaj = alphaj_wesson`, reads `qstar`, `q0` | **yes** |

Value 0 is an **empty slot**, not an unported model — the same shape
`tokamak_scope.md` records for `i_pulsed_plant == 0` and `inuclear`'s input arm. Under
it `.physics.alphaj` has no producer and is a boundary input. Nothing to write.

### `i_ind_plasma_internal_norm` (`physics_variables.py:948`, default `0`; `large_tokamak_eval:311` sets `1`)

`IndInternalNormModel`, `physics.py:4682-4699`. Three values, selected by a dict lookup
(`physics.py:4759-4764`) rather than an `if` chain:

| value | name | reads | ported |
|---|---|---|---|
| 0 | `USER_INPUT` | selects `.ind_plasma_internal_norm` from itself | **empty slot** |
| 1 | `WESSON` | `.alphaj` | **yes** |
| 2 | `MENARD` | `.kappa` | no — not live |

**`PlasmaInductance.run()` is band (b) in the source itself**: it computes all three
scalings unconditionally and only then picks one, so the PROCESS body reads six
variables where the live arm needs one. Splitting it into occupants is what removes the
other five, and four of those five reads exist *only* to fill reporting fields.

## calls into other models

None. Every function in the ported closure is self-contained arithmetic over `.physics`
floats; `calculate_plasma_current` calls only its own siblings, and the `physics.py`
three call nothing at all. No CoolProp anywhere in the chain (`tokamak_scope.md`'s
`coolprop` column is clean for all three switches).

## JAX-difficulty flags

- **One, and it survives into the port deliberately.** `qstar`'s
  divide-by-a-quotient (`physics.py:93-99`) differentiates to `nan` at
  `b_plasma_toroidal_on_axis == 0` while its value stays finite — the
  `_harness/boundary.py` unguarded-division class, caught by
  `test_gradient_finite_at_zero` and **registered there rather than repaired**, per the
  wave coordinator's ruling that faithfulness wins. See **D6**. Everything else in the
  ported arm is `jnp` substitution and nothing more: the IPDG89 coefficient, the
  cylindrical current, `qstar/q0 - 1` and `log(1.65 + 0.89α)` are smooth elementary
  expressions, no branch, no iteration, no lookup.
- The unported arms carry two: `plascar_bpol` (#6) has an `if aspect < c1` branch that
  changes the *formula* (`plasma_current.py:670`, `:683-686`) and would need a
  `jnp.where` over both branches with the usual NaN-through-the-untaken-branch care,
  since `np.log((1+y1)/(1-y1))` and `np.arctan(y1)` have different domains;
  `calculate_current_coefficient_fiesta` (#13) contains `triang**0.060`, which is NaN
  for `triang < 0` and has an infinite derivative at `triang = 0`.
- `@nb.njit` on three of the reference functions is irrelevant to the port and only
  affects the harness: the reference side is compiled, the port side traced.

## suspected defects in PROCESS

**D1 — `calculate_plasma_current`'s docstring describes a function that no longer
exists.** `plasma_current.py:259-263` promises
`(b_plasma_poloidal_average, qstar, plasma_current, betap, li)`; the body returns
`plasma_current` alone (`:403`). Four of the five have since moved to `plasma_fields.py`
and `physics.py`. Cosmetic, but it is the docstring an extractor would trust.

**D2 — `model` can be unbound.** `plasma_current.py:311-383` assigns `model` inside
`try:`, and `:392` reads it outside. `PlasmaCurrentModel(int(...))` raising `ValueError`
is caught and re-raised, so the live path is safe; any *other* exception inside the
`try` produces `UnboundLocalError` at `:392` rather than the intended error. Likewise
`fq` is unbound if a future enum member is added without a branch. Not reachable today.

**D3 — the `except` clause reads `self.data` in a function that otherwise does not.**
`plasma_current.py:388` uses `self.data.physics.i_plasma_current` in the error message
while the function has the value as a parameter (`i_plasma_current`). If a caller ever
passes a value different from `data`'s, the message names the wrong one. This is also
the single line that stops the function from being a `@staticmethod`.

**D4 — `calculate_all_plasma_current_models` (#3) evaluates all nine scalings on every
`output()` call**, including value 8's and 9's `triang**0.060` and value 2's
`arcsin`/`log`, at the machine's actual triangularity. It is guarded by nothing; on a
negative-triangularity machine `calculate_plasma_current` would raise from `:305-309`
for eight of the nine models, from inside a reporting routine. Not reachable on this
input (`triang = 0.5`).

**D5 — a stale legacy expectation.** `tests/unit/models/physics/test_physics.py::
test_calculate_plasma_current` records `expected_plasma_current = 18398455.678867526`;
PROCESS today returns `18398455.670608774` for that input (measured in `process_port`).
The test passes only because it uses `pytest.approx`'s default `rel=1e-6`. Harmless
here — this unit's harness computes "expected" by calling the reference, so the recorded
literal is used as a *sample point*, never as an answer — but it means the recorded
number is not a usable oracle for anything tighter than 1e-6.

**D6 — `qstar` is finite in value and `nan` in derivative at zero toroidal field.**
`physics.py:93-99` spells the middle factor as a division *by a quotient*,
`rminor**2 / (rmajor * plasma_current / b_plasma_toroidal_on_axis)`. At
`b_plasma_toroidal_on_axis == 0` the inner quotient is `+inf`, the outer division pulls
the value back to exactly `0.0` — the correct limit, which PROCESS itself returns without
complaint — and the tangent through `a / b` stays `nan`. Found by
`Tier1Contract.test_gradient_finite_at_zero`, i.e. by the check `_harness/boundary.py`
was built for, and it is that module's unguarded-division class rather than the
`x ** p` one.

**Registered, not repaired**, and that is a decision rather than an omission.
Reassociating to `rminor**2 * b / (rmajor * plasma_current)` removes the `nan`, gives
the same `0.0` at the boundary point, and is value-equal to about 2.6 ulp — measured
over 20 000 points in this unit's fuzz domain, 32% differ from PROCESS's form in the last
bits, worst relative difference `5.8e-16`, against `MACHINE_PRECISION`'s `rtol=1e-12`.
That repair was written, tested green, and then **reverted on the wave coordinator's
decision that faithfulness wins**: a ported body spells PROCESS's expression, and a
derivative defect PROCESS carries is recorded where the project already records them.
The resolution is one additive entry in `_harness/boundary.py`,
`("TestCalculateCylindricalSafetyFactor", "b_plasma_toroidal_on_axis")`, alongside the
`TestCalculateDivertorHeatLoadWade` entry for the same argument in the same class of
defect. Consequence to keep in view: a solver stepping through `b == 0` gets a poisoned
Jacobian row from this node, exactly as PROCESS would — which is what the register is
for, being "recorded, reported, and not subtracted from anything".

## tier signal

**Tier 1 throughout.** No function in the ported closure iterates, and none reads
`data`, so PROCESS's answer *is* ground truth and value agreement is checkable at
machine precision. Measured: all five direct diffs agree to the last bit (see
"## ported").

## proposed signature(s)

Matched exactly by the port:

```
calculate_cyclindrical_plasma_current(rminor, rmajor, q95, b_plasma_toroidal_on_axis)
calculate_current_coefficient_ipdg89(eps, kappa95, triang95)
calculate_plasma_current_ipdg89(eps, kappa95, triang95, rminor, rmajor, q95,
                                b_plasma_toroidal_on_axis)
calculate_cylindrical_safety_factor(rmajor, rminor, plasma_current,
                                    b_plasma_toroidal_on_axis, kappa95, triang95)
calculate_current_profile_index_wesson(qstar, q0)
calculate_internal_inductance_wesson(alphaj)
```

`calculate_plasma_current_ipdg89` is the one new function: the `i_plasma_current == 4`
path through #2 with the enum dispatch and the other eight arms removed. Its seven
arguments are exactly this arm's reads, against the head's fourteen.

## ported (2026-08-26)

Port: `functional_process/models/physics/plasma_current.py`. Tests:
`tests/functional_process/models/physics/test_plasma_current.py`. **`30 passed, 30
skipped`** on a plain run (gradient checks skip by default); **`60 passed`** with
`--fp-gradients`; **`156 passed`** with `--fp-gradients --fp-fuzz 5`.

### cottax nodes

| class | family | owns | reads |
|---|---|---|---|
| `Ipdg89PlasmaCurrent` | `PlasmaCurrentScaling` | `.physics.plasma_current` | `.physics.eps`, `.kappa95`, `.triang95`, `.rminor`, `.rmajor`, `.q95`, `.b_plasma_toroidal_on_axis` |
| `PlasmaCylindricalSafetyFactor` | (none — unconditional) | `.physics.qstar` | `.physics.rmajor`, `.rminor`, `.plasma_current`, `.b_plasma_toroidal_on_axis`, `.kappa95`, `.triang95` |
| `WessonCurrentProfileIndex` | `CurrentProfileIndexScaling` | `.physics.alphaj` | `.physics.qstar`, `.q0` |
| `WessonInternalInductance` | `NormalisedInternalInductanceScaling` | `.physics.ind_plasma_internal_norm` | `.physics.alphaj` |

Seven + six + two + one = sixteen declared reads, against the twenty PROCESS's four call
sites read for the same four outputs (fourteen for the current alone, six inside
`PlasmaInductance.run`). Two of the removed edges are structural rather than merely
redundant: `.physics.b_plasma_surface_poloidal_average` and `.physics.vol_plasma`
entered `PlasmaInductance` only to feed `ind_plasma_internal_norm_iter_3`, a printed
number, and the first of those is `plasma_fields.py`'s output — an inter-model edge that
no computed result depends on.

### registration instructions (for the consolidation pass)

Slot `.tokamak.plasma_current`, four occupants, in this evaluation order (acyclic —
`Ipdg89PlasmaCurrent` → `PlasmaCylindricalSafetyFactor` → `WessonCurrentProfileIndex` →
`WessonInternalInductance`; a real `Graph` derives it, this is only the witness):

```python
from functional_process.models.physics.plasma_current import (
    Ipdg89PlasmaCurrent,
    PlasmaCylindricalSafetyFactor,
    WessonCurrentProfileIndex,
    WessonInternalInductance,
)
```

- `plasma_current: PlasmaCurrentScaling` — factory-filled, no default (a switch answers
  it). `i_plasma_current == 4` → `Ipdg89PlasmaCurrent()`.
- `cylindrical_safety_factor: PlasmaCylindricalSafetyFactor = PlasmaCylindricalSafetyFactor()`
  — defaulted; nothing switches it.
- `current_profile_index: CurrentProfileIndexScaling | None` — factory-filled.
  `i_alphaj == 1` → `WessonCurrentProfileIndex()`; `i_alphaj == 0` → **empty**
  (`.physics.alphaj` becomes a boundary input).
- `internal_inductance: NormalisedInternalInductanceScaling | None` — factory-filled.
  `i_ind_plasma_internal_norm == 1` → `WessonInternalInductance()`;
  `== 0` → **empty** (`.physics.ind_plasma_internal_norm` becomes a boundary input).

`large_tokamak_eval.IN.DAT` fills all four (lines 288, 275, 311).

**Boundary inputs this slot then needs** (none of them produced here): `.physics.q95`
(iteration variable 18), `.q0`, `.b_plasma_toroidal_on_axis` (iteration variable 2).
`.eps`/`.rminor` come from `plasma_geometry.py`'s `PlasmaMinorRadius`,
`.kappa95`/`.triang95` from its `Ipdg89XPointPlasmaShape`, `.rmajor` is iteration
variable 3.

**`UNPORTED` entries for `indat.py`:**

| switch | value | reason |
|---|---|---|
| `i_plasma_current` | 1 | Peng analytic fit; not live on any tracked input. `calculate_current_coefficient_peng` is a 5-line pure staticmethod when needed. |
| `i_plasma_current` | 2 | Peng divertor (TART/STAR); not live, and structurally unlike every other arm — bypasses the cylindrical current and needs `plascar_bpol`'s two-branch `arctan`/`log`. Also the arm that changes `b_plasma_surface_poloidal_average` (`plasma_fields.py:83`). |
| `i_plasma_current` | 3 | Simple ITER cylindrical (`fq = 1`); not live. |
| `i_plasma_current` | 5, 6 | Todd I/II; not live. Identical reads, differing by one literal — **two** occupant classes when ported, per §14.2 and open question 4's ruling, not one with a static kwarg. |
| `i_plasma_current` | 7 | Connor-Hastie; not live, and the only arm that makes this chain a genuine SCC (see "## the cycle that is not live here") — needs a `FixedPointFunction` or a declared driven block, not just a transcription. |
| `i_plasma_current` | 8 | Sauter; not live. Must be wired together with `plasma_geometry.py`'s `PlasmaGeometryArm` Sauter occupant — one input value, two slots. |
| `i_plasma_current` | 9 | FIESTA ST; not live. `triang**0.060` is NaN for negative triangularity and has an infinite derivative at zero. |
| `i_alphaj` | 0 | **Empty slot**, not unported: PROCESS's arm is `alphaj = alphaj`. |
| `i_ind_plasma_internal_norm` | 0 | **Empty slot**: PROCESS selects the field from itself. |
| `i_ind_plasma_internal_norm` | 2 | Menard ST scaling (reads `.physics.kappa`); not live. |

### not ported in this pass

- The eight other `i_plasma_current` arms and their coefficient functions (#6-#8,
  #10-#13) — table above. Deliberately *not* ported-but-unwired (the call
  `plasma_geometry.py` made for `sauter_geometry`): each needs its own occupant and its
  own harness contract to be worth anything, and porting a formula with no oracle and no
  caller adds a maintained surface with no check on it.
- `.physics.alphaj_wesson`, `.ind_plasma_internal_norm_{wesson,menard,iter_3}` and the
  nine `.physics.c_plasma_*` — reporting-only, measured (table in "## data footprint").
  Same call `plasma_geometry.py`'s port made for `PlasmaGeom.output()`. The visible
  consequence: `WessonCurrentProfileIndex` owns `.physics.alphaj` directly instead of
  owning `alphaj_wesson` and copying it, so the family has one owned `VarPath` and no
  intermediate.
- `PlasmaCurrent.output`, `calculate_all_plasma_current_models`,
  `output_plasma_current_models` (#1, #3, #4) — reporting.
- `PlasmaDiamagneticCurrent` (#14-#17) — same PROCESS file, different chain
  (`.current_drive.f_c_plasma_diamagnetic*`, keyed on `i_diamagnetic_current`), not on
  the route to `.physics.plasma_current`. Two pure one-liners plus a two-way switch; a
  cheap unit for a later pass, and its own.
- `.physics.b_plasma_surface_poloidal_average` — already owned by
  `SurfaceAveragedPoloidalFieldAmperes` in
  `functional_process/models/physics/physics.py`. Not duplicated.

### deviations from PROCESS

1. **The chain is filed in one port file across two source files** — steps 2, 3 and 4
   come from `process/models/physics/physics.py`, not from `plasma_current.py`. See
   "## the chain is not one file". Every affected function's docstring carries its
   `physics.py:line` span.
2. **`WessonCurrentProfileIndex` writes `.physics.alphaj` directly**, where PROCESS
   writes `.physics.alphaj_wesson` (`physics.py:330`) and copies it (`physics.py:343`).
   Numerically identical; drops one owned reporting-only `VarPath`.
3. **`WessonInternalInductance` computes only the Wesson scaling**, where
   `PlasmaInductance.run()` computes all three and then selects (`physics.py:4721-4745`).
   Numerically identical for the live value; drops five reads and three reporting-only
   `VarPath`s. The harness diffs the port against the *whole* `run()` including the
   discarded work, so the equivalence is tested rather than asserted.
4. **The two guards in `calculate_plasma_current` are not carried**
   (`plasma_current.py:305-309`, negative triangularity outside value 8; `:385-389`,
   illegal switch value). Both are switch-domain checks, answered by which occupant
   exists — `naming_convention.md` § "switches are not ports". The negative-triangularity
   precondition remains the caller's, as in PROCESS; the test module keeps every sample
   and fuzz bound at `triang95 >= 0`.
**No expression is rewritten.** Every ported body spells PROCESS's arithmetic as PROCESS
spells it; the four deviations above are all about *which* fields are owned and which
branches exist, never about the algebra inside one. In particular
`calculate_cylindrical_safety_factor` keeps PROCESS's division-by-a-quotient verbatim —
see **D6**, which is registered as a boundary defect rather than repaired. No PROCESS
defect above is fixed.

## open questions

1. **Does `PlasmaInductance` become its own unit?** `_audit/tokamak_boundary.md` lists
   `.tokamak.plasma_inductance` as a slot distinct from `.tokamak.plasma_current` (both
   with zero boundary reads today). This pass ports the one piece of it that
   `large_tokamak_eval` reaches, into this unit, because its only input is this chain's
   `alphaj`. If the slot is built for real — it also owns
   `calculate_volt_second_requirements` (`physics.py:4766-4900`), a much larger function
   reading `res_plasma`, the pulse times and the Ejima coefficient — then
   `WessonInternalInductance` and `calculate_internal_inductance_wesson` should move
   there and this record should lose its step 4. **Needs a decision before the volt-second
   work starts, not before this lands.**
2. **Where is a switch-dependent SCC declared?** At `i_plasma_current == 7` this chain is
   cyclic and at every other value it is not. The tree currently has no way to say "this
   occupant makes its block driven" — `Blocking` is a property of the assembled graph,
   which is exactly right, but the *factory* has no signal that choosing occupant 7 costs
   a driven block. Not blocking anything today (value 7 is UNPORTED); flagged so that the
   pass which ports it does not discover it at assembly time.
3. **Three slots answer `i_plasma_current` and the factory must keep them consistent.**
   `PlasmaCurrentScaling` (here), `PlasmaGeometryArm` (`plasma_geometry.py`, via the
   compound `== 8 or i_plasma_shape == SAUTER`) and the
   `b_plasma_surface_poloidal_average` family (`physics.py`, via `== 2`). Today only one
   value is live in each and the three answers happen to agree trivially. There is no
   check that they agree; `model_tree_design.md` §8 step 4d's "a switch is answered once"
   is the place to record whether that wants enforcing, and `plasma_geometry.md`'s open
   question 2 is the same question from the other side.
4. **Todd I/II (`i_plasma_current` 5 and 6) — SETTLED: two occupant classes.** Decided
   by the wave coordinator (2026-08-26), so the next pass does not re-derive it. The two
   arms read the same three variables (`eps`, `kappa95`, `triang95`) and differ only in
   a literal `model=1|2` handed to `calculate_current_coefficient_todd`
   (`plasma_current.py:344-355`), which is the `istore` shape — but §14.2's binding
   policy binds: **one occupant class per switch value**, and the `istore` exception does
   not reach here. The "switches touched" table above records the older, undecided
   reading of this; **this item overrides it.** Neither value is live on
   `large_tokamak_eval`, so nothing changes in this pass — the decision is recorded for
   whichever pass ports them.
5. **Fuzz bounds here are hand-chosen, not taken from
   `bounds_from_iteration_variables`.** `q95`, `rmajor`, `b_plasma_toroidal_on_axis` and
   `aspect` *are* iteration variables (18, 3, 2, 1) with declared bounds; `kappa95`,
   `triang95` and `alphaj` are not, and mixing the two sources in one `fuzz_bounds` dict
   would have been more misleading than choosing all seven by hand. Same gap
   `plasma_geometry.md`'s question 3 records from the domain-guard side.
