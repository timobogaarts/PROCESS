---
kind: model-unit
status: draft
confidence: high
---

**Ported (2026-08-26).** `l_h_transition.py` declares all 21 `calculate_*` L-H/L-I
threshold scaling laws (all 21 `@staticmethod`s in
`process/models/physics/l_h_transition.py`, source order preserved) plus six cottax
occupant node classes for the Martin family (`i_l_h_threshold` values 6, 7, 8, 19, 20,
21), the value that is live on `large_tokamak_eval.IN.DAT` (19, PROCESS's own default)
being one of them. No `unit_registry.md` row, no `next_steps.md` edit -- registration is
the consolidation pass's job (`next_steps.md` §4b), matching `plasma_geometry.md`'s
precedent for the same reason. See "## ported" for registration instructions and the
UNPORTED table for the other 15 arms' pure formulas (ported, tested, not wired as
occupants).

## source

`process/models/physics/l_h_transition.py` (1477 lines, full file in scope).
`PlasmaConfinementTransition` is the tokamak L-H/L-I transition power threshold model,
and the sole site of `i_l_h_threshold`. Called from `main.py:701`
(`self.plasma_transition = PlasmaConfinementTransition()`); not found in
`process/core/caller.py`'s own `_call_models_once` grep, so its call site is elsewhere in
the tokamak call sequence assembled by `main.py`/`Models` -- out of this unit's scope to
trace further.

**Two `def`s carry structure, not computation**: `PlasmaConfinementTransitionModel.__new__`
(enum plumbing) and `PlasmaConfinementTransition.__init__` (sets `self.outfile`/`self.mfile`
only). **23 are in scope:**

| # | function | lines | shape |
|---|---|---|---|
| 1 | `run` | 66-88 | the stateful shell: one unconditional preamble call, one switch-indexed selection |
| 2 | `l_h_threshold_power` | 90-294 | **not a dispatch** -- computes all 21 arms unconditionally, returns them as a list; the switch never appears inside this function at all |
| 3 | `output` | 296-538 | reporting shell: prints all 21 values, one `p_l_h_threshold_mw`, several range-check warnings gated on `i_l_h_threshold ∈ {9,10,11}` / `{12,13,14}` |
| 4-24 | `calculate_iter1996_nominal` ... `calculate_martin08_aspect_lower` | 540-1477 | 21 `@staticmethod`s, pure, zero `self.data` access |

## the computes-then-selects shape, and why it is not a dispatcher

Unlike `plasma_geometry.py`'s 13-branch `if`/`if`/`if` (`plasma_geometry.md`'s load-bearing
table) or `confinement_time.py`'s single `if_confinement_time == N:` chain, `l_h_threshold_
power` (`l_h_transition.py:90-294`) contains **no branching on `i_l_h_threshold` at all**.
It calls all 21 static methods unconditionally and returns a 21-element list; `run()`
(`l_h_transition.py:86-88`) is where the switch appears, as a single list index:

```python
self.data.physics.p_l_h_threshold_mw = self.data.physics.l_h_threshold_powers[
    self.data.physics.i_l_h_threshold - 1
]
```

This is exactly the shape `next_steps.md` §14.2 and `bootstrap_current.md` name
"computes-then-selects" (an index into a precomputed table, not a branch), and the
settled policy for it (`bootstrap_current.md`, restated in the wave-1 brief) is the same
as for a branching dispatcher: **one occupant class per switch value that this port
supports**, each declaring only its own arm's reads -- not a family node computing all 21
plus an index. The difference from a true dispatcher is that here there is no shared
dispatch *logic* to avoid re-declaring (`l_h_threshold_power` itself has nothing but 21
independent calls), so the "split vs. static kwarg, weighed by shared-remainder size"
question `traceability_policy.md` leaves open does not arise: there is no shared body,
period, split is free.

## the extraction seam

**As clean as `plasma_geometry.md`'s functions 3-9.** All 21 `calculate_*` methods are
already `@staticmethod`s taking plain floats and returning a plain float -- `CallableNode.
fn` already, needing only `np.` -> `jnp.` (in fact the source uses no `np.` calls at all;
every formula is `**`/`/`/`*` on plain floats) and `safe_pow` for fractional exponents.
**Zero `self.data` access inside any of them.** The only computation outside the 21
statics is `dnla20 = 1e-20 * nd_plasma_electron_line` (`l_h_transition.py:131`), a single
line at the head of `l_h_threshold_power`, consumed by every arm.

The seam is at the `run()`/`l_h_threshold_power()` boundary: `run()` is a two-line shell
(call, then index), `l_h_threshold_power()` is a shell around 21 calls plus the `dnla20`
conversion, and every piece of arithmetic that is not that conversion is already inside a
`calculate_*` static.

## data footprint

Reference run: `tests/regression/input_files/large_tokamak_eval.IN.DAT` --
`i_l_h_threshold` unset, so PROCESS's own default `19` applies
(`physics_variables.py:1234`, `MARTIN08_ASPECT_NOMINAL`). Confirmed live: `icc = 15`
("LH power threshold limit") is in the file's active constraint list, so
`.physics.p_l_h_threshold_mw` is not merely computed but actually consumed by constraint
15 on this run.

### `run()` (lines 66-88)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.nd_plasma_electron_line` | read | explicit-arg | into `l_h_threshold_power` |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | |
| `.physics.rmajor` | read | explicit-arg | |
| `.physics.rminor` | read | explicit-arg | |
| `.physics.kappa` | read | explicit-arg | only `calculate_snipes1997_kappa` (arm 5) uses it |
| `.physics.a_plasma_surface` | read | explicit-arg | |
| `.physics.m_ions_total_amu` | read | explicit-arg | |
| `.physics.aspect` | read | explicit-arg | only the three `martin08_aspect_*` arms use it |
| `.physics.plasma_current` | read | explicit-arg | only the three `hubbard2012_*` arms use it |
| `.physics.i_l_h_threshold` | read | switch | selects which element of `l_h_threshold_powers` becomes `p_l_h_threshold_mw` -- graph-build-time, per `naming_convention.md` § "switches are not ports" |
| `.physics.l_h_threshold_powers` | **write** | reporting-only | the full 21-element list; consumed only by `output()` (lines 330-476) and `core/io/plot/summary.py` (mfile-based plotting) -- **not ported**, see "## ported" |
| `.physics.p_l_h_threshold_mw` | **write** | explicit-arg (per occupant) | the one element that matters to the compute graph -- consumed by constraints 15, 22, 73 |

### `output()` (lines 296-538)

Reporting-only: prints all 21 `l_h_threshold_powers` entries, `i_l_h_threshold`,
`p_l_h_threshold_mw` (with a different label depending on
`data.numerics.i_process_run_mode > 0 and data.numerics.active_constraints[14]`, i.e.
whether constraint 15 -- index 14, zero-based -- is active), and several
`logger.warning`/`po.ocmmnt` range checks against the Snipes 2000 fit's validity domain
when `i_l_h_threshold ∈ {9,10,11,12,13,14}`. No `data` writes. Out of scope, matching
`plasma_geometry.md`'s treatment of its own `output()`.

## coupling / SCC finding

**None found.** `l_h_threshold_power` calls no other `Model`'s method, and nothing found
in `process/` writes `.physics.p_l_h_threshold_mw` except this file. The consumers
(constraints 15, 22, 73) are read-only with respect to this unit -- a constraint is a
`Compare`/condition shape, not a node that writes back into `.physics.*`, so there is no
cycle through them. This is the case `_audit/tokamak_boundary.md` flags: "the boundary
does not see a producer nobody asks for" inverted -- here the producer *is* asked for, just
not by anything the boundary's plain-graph measurement counts (constraints are outside
that measurement, per the wave-1 brief).

## switches touched

One: `i_l_h_threshold`, range `(1, 21)` inclusive (`process/core/input.py:1006`), all 21
values have both an enum member and a live `calculate_*` implementation (no dead branches,
unlike `confinement_time.md`'s `PAZ_SOLDAN_NT`, and no missing docstring rows, unlike
`plasma_geometry.md`'s D7).

**Split** -- mandatory, stronger than "reads differ": the 21 arms take between 2 and 5
arguments each, from six distinct source fields (`dnla20` itself derived from one),
and no two arms outside the families below share an identical reads-set. A union node
would claim `dnla20` (⊂ `nd_plasma_electron_line`), `b_plasma_toroidal_on_axis`, `rmajor`,
`rminor`, `kappa`, `a_plasma_surface`, `m_ions_total_amu`, `aspect`, `plasma_current` --
9 fields, where the live arm (19) uses 5.

### the Martin family -- the six occupants wired in this pass

| `i_l_h_threshold` | enum member | reads (beyond `dnla20`) |
|---|---|---|
| 6 | `MARTIN08_NOMINAL` | `b_plasma_toroidal_on_axis`, `a_plasma_surface`, `m_ions_total_amu` |
| 7 | `MARTIN08_UPPER` | same as 6 |
| 8 | `MARTIN08_LOWER` | same as 6 |
| 19 *(live)* | `MARTIN08_ASPECT_NOMINAL` | 6's reads + `aspect` |
| 20 | `MARTIN08_ASPECT_UPPER` | 6's reads + `aspect` |
| 21 | `MARTIN08_ASPECT_LOWER` | 6's reads + `aspect` |

**Validated, not assumed**: arms 19/20/21 are arms 6/7/8's own formula (same coefficients,
same exponents on `dnla20`/`b_plasma_toroidal_on_axis`/`a_plasma_surface`/
`m_ions_total_amu`) multiplied by one extra factor,
`_martin08_aspect_correction(aspect)` (`l_h_transition.py:1328-1331` and its two literal
copies at 1396-1399, 1464-1467) -- confirmed by reading all six bodies side by side, not
inferred from naming. Arms 6/7/8's reads-set is therefore an exact subset of 19/20/21's
(missing only `aspect`), which is what makes wiring all six "trivially cheap" per the
wave-1 brief's sibling-arm allowance: no new reads-set analysis was needed beyond reading
the six bodies once.

### the other 15 arms -- pure functions ported, not wired

| `i_l_h_threshold` | enum member(s) | reads |
|---|---|---|
| 1-3 | `ITER1996_NOMINAL/UPPER/LOWER` | `dnla20`, `b_plasma_toroidal_on_axis`, `rmajor` |
| 4 | `SNIPES1997_ITER` | same three |
| 5 | `SNIPES1997_KAPPA` | 4's reads + `kappa` |
| 9-11 | `SNIPES2000_NOMINAL/UPPER/LOWER` | `dnla20`, `b_plasma_toroidal_on_axis`, `rmajor`, `rminor`, `m_ions_total_amu` |
| 12-14 | `SNIPES2000_CLOSED_DIVERTOR_*` | `dnla20`, `b_plasma_toroidal_on_axis`, `rmajor`, `m_ions_total_amu` (no `rminor`) |
| 15-17 | `HUBBARD2012_NOMINAL/LOWER/UPPER` | `plasma_current`, `dnla20` |
| 18 | `HUBBARD2017_I_MODE` | `dnla20`, `a_plasma_surface`, `b_plasma_toroidal_on_axis` |

Each family here has its own reads-set, independent of the Martin family and of each
other (e.g. the Snipes-2000 pair drops `m_ions_total_amu` for the closed-divertor variant
in one direction and adds `rminor` in the other -- not a strict subset relationship the
way Martin's is). Wiring any of these as occupants is future work: the pure formulas are
ported and Tier-1-tested (a free correctness check, since
`tests/unit/models/physics/test_l_h_transition.py` already validates every one against
PROCESS), but turning them into graph occupants needs the same one-body-read side by side
this record did for the Martin family, not done here because none of the 15 is live on
`large_tokamak_eval.IN.DAT` and the wave-1 brief scopes this pass to the reference arm plus
validated-cheap siblings.

## JAX-difficulty flags

- **F1 -- `if aspect <= 2.7: ... else: 1.0`** (`l_h_transition.py:1328-1331`, and its two
  copies), the three `martin08_aspect_*` statics -- `needs-lax-cond-or-where`, severity
  `workaround-known`. `aspect` is a plain differentiable argument, not a switch. Ported as
  `jnp.where`, which evaluates both branches; the "then" branch's own denominator,
  `1.0 - safe_pow(2.0 / (1.0 + aspect), 0.5)`, is singular only at `aspect == 1.0`, outside
  every physically meaningful aspect ratio and outside `ITERATION_VARIABLES[1]`'s declared
  bounds `(1.1, 10.0)` (per `plasma_geometry.md`'s own citation of that bound) -- documented,
  not guarded, same convention as that record's D1/F3.
- **F2 -- fractional powers**, severity `minor`, all `safe_pow` candidates: every exponent
  strictly between 0 and 1 across all 21 statics (`dnla20**0.75`, `a_plasma_surface**0.941`,
  `(plasma_current/1e6)**0.94`, `(2.0/(1.0+aspect))**0.5`, etc. -- the full list is in the
  port's inline `file:line` comments). Bases are all strictly non-negative on the physical
  domain (densities, fields, radii, masses, currents), so the `x == 0` derivative poison is
  a structural edge case rather than a live one, but `Tier1Contract.
  test_gradient_finite_at_zero` exercises every one regardless -- see "## boundary
  registrations" below for the one case that needed a register entry rather than a plain
  pass.
- **F3 -- unguarded division**, severity `workaround-known`: `2.0 / m_ions_total_amu`
  (every Martin-family arm) has no clamp. At `m_ions_total_amu == 0.0` the *value* itself
  goes non-finite (`inf`), so this is not the "value correct, gradient poisoned" class
  `safe_math.py` exists for -- `test_gradient_finite_at_zero` steps aside on it
  automatically (its own "a component whose value goes non-finite when zeroed" guard), and
  no `_harness/boundary.py` registration was needed (confirmed by running the check, not
  assumed -- see "## boundary registrations").
- **F4 -- non-traceable external calls**: **none.** No CoolProp, no external library; the
  entire file is `**`/`/`/`*` on plain floats plus one `if`.
- **F5 -- in-place mutation**: none. No arrays, no loops, no dynamic shapes.
- **F6 -- `l_h_threshold_powers` is a Python `list`, not an array**, built once per call by
  `l_h_threshold_power` and never mutated in place -- no `.at[i].set` idiom needed even if
  it were ported (it is not, see "## ported").

## boundary registrations

Ran `Tier1Contract.test_gradient_finite_at_zero` (via `--fp-gradients`) against every one
of the 21 ported statics. **Zero new entries needed in `_harness/boundary.py`.** The one
candidate (`m_ions_total_amu` at `0.0` in the six Martin-family formulas) is excused by the
check's own "value already non-finite" guard rather than needing a
`DIVISION_BY_ZERO_AT_BOUNDARY` register entry -- confirmed by running the suite, not
inferred; see "## test results" below. This is worth recording explicitly because the
wave-1 brief anticipated needing an addition here and the honest result is that this unit
did not earn one.

## suspected defects in PROCESS

**None found.** Unlike `plasma_geometry.md`'s eleven (D1-D11), nothing in this file
exhibits a sign flip, a stale docstring, a missing `else`, or a dead-vs-duplicate pair.
The one structural oddity -- `l_h_threshold_powers` computed in full every call regardless
of which single element `run()` actually uses -- is wasted work, not a correctness defect
(all 21 formulas are cheap, and PROCESS's own `output()` genuinely wants all 21 for
reporting), so it is not logged as one.

## tier signal

**Tier 1 for all 21 statics.** No `scipy.optimize`, no `fsolve`, no ad hoc fixed-iteration
loop, no call into another `Model`'s method, no CoolProp. `run()` is a two-line shell;
`l_h_threshold_power()` is a shell around 21 pure calls; `output()` is pure reporting.

**Sample provenance is strong, unlike `plasma_geometry.md`'s weak point.**
`tests/unit/models/physics/test_l_h_transition.py` already validates every one of the 21
statics against PROCESS with a legacy sample -- full coverage, not a subset -- so every
Tier1Contract below has a genuinely legacy point, not just a fuzz-only one.

## open questions

1. **Should the other 15 arms be wired as occupants in a future pass, and by whom?** Not
   answered here -- flagged in "## switches touched" above as future work, since none is
   live on any tracked regression input this pass checked (only `large_tokamak_eval`'s
   `.IN.DAT` was checked; other tracked files were not grepped for `i_l_h_threshold`).
2. **Where is `PlasmaConfinementTransition.run()` actually called from?** `main.py:701`
   constructs it, but the call sequence (`Models`/`Caller`) that invokes `.run()` was not
   traced -- out of scope for a source-level unit audit, flagged so the consolidation pass
   does not assume `caller.py`'s grep silence means unreachable.
3. **Is `.physics.kappa`'s read (arm 5 only) worth a `local-intermediate` note the way
   `plasma_geometry.md` tracked switch-dependent provenance?** Not relevant here: `kappa`
   is always the same field regardless of `i_l_h_threshold`, unlike `plasma_geometry.md`'s
   `kappa`/`triang` which swap between read and write sides depending on the switch. No
   further action needed; noted only because the parallel was worth ruling out explicitly.

## ported (2026-08-26)

Port: `functional_process/models/physics/l_h_transition.py`. Tests:
`tests/functional_process/models/physics/test_l_h_transition.py`.

**Scope: all 21 pure formulas (full closure, cheap and free-oracled), plus six cottax
occupants for the Martin family (`i_l_h_threshold ∈ {6,7,8,19,20,21}`)** -- the reference
arm (19) plus the five siblings whose reads-sets were validated (not merely assumed)
against it in this pass. `_audit/tokamak_boundary.md`'s `.tokamak.l_h_transition` slot has
zero declared boundary reads today because that measurement only counts the plain compute
graph; this pass's consumers are constraints 15/22/73, which the boundary measurement does
not walk -- consistent with, not contradicting, that zero.

| function | shape | i_l_h_threshold |
|---|---|---|
| `calculate_iter1996_nominal(dnla20, b_plasma_toroidal_on_axis, rmajor)` | verbatim port | 1 |
| `calculate_iter1996_upper(dnla20, b_plasma_toroidal_on_axis, rmajor)` | verbatim port | 2 |
| `calculate_iter1996_lower(dnla20, b_plasma_toroidal_on_axis, rmajor)` | verbatim port | 3 |
| `calculate_snipes1997_iter(dnla20, b_plasma_toroidal_on_axis, rmajor)` | verbatim port | 4 |
| `calculate_snipes1997_kappa(dnla20, b_plasma_toroidal_on_axis, rmajor, kappa)` | verbatim port | 5 |
| `calculate_martin08_nominal(dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu)` | verbatim port | 6 |
| `calculate_martin08_upper(...)` | verbatim port | 7 |
| `calculate_martin08_lower(...)` | verbatim port | 8 |
| `calculate_snipes2000_nominal(dnla20, b_plasma_toroidal_on_axis, rmajor, rminor, m_ions_total_amu)` | verbatim port | 9 |
| `calculate_snipes2000_upper(...)` | verbatim port | 10 |
| `calculate_snipes2000_lower(...)` | verbatim port | 11 |
| `calculate_snipes2000_closed_divertor_nominal(dnla20, b_plasma_toroidal_on_axis, rmajor, m_ions_total_amu)` | verbatim port | 12 |
| `calculate_snipes2000_closed_divertor_upper(...)` | verbatim port | 13 |
| `calculate_snipes2000_closed_divertor_lower(...)` | verbatim port | 14 |
| `calculate_hubbard2012_nominal(plasma_current, dnla20)` | verbatim port | 15 |
| `calculate_hubbard2012_upper(...)` | verbatim port | 16 |
| `calculate_hubbard2012_lower(...)` | verbatim port | 17 |
| `calculate_hubbard2017(dnla20, a_plasma_surface, b_plasma_toroidal_on_axis)` | verbatim port | 18 |
| `calculate_martin08_aspect_nominal(dnla20, b_plasma_toroidal_on_axis, a_plasma_surface, m_ions_total_amu, aspect)` | verbatim port | 19 *(live)* |
| `calculate_martin08_aspect_upper(...)` | verbatim port | 20 |
| `calculate_martin08_aspect_lower(...)` | verbatim port | 21 |

**Not ported in this pass:**

- `.physics.l_h_threshold_powers` (the full 21-element reporting array) -- consumed only
  by `output()` and `core/io/plot/summary.py`'s mfile-based plotting, both reporting paths
  outside this unit's scope. Same convention `plasma_geometry.md` applied to
  `PlasmaGeom.output()`.
- `PlasmaConfinementTransition.output()` -- pure reporting, no computation, out of scope.
- Occupant node classes for the other 15 `i_l_h_threshold` values -- see "## switches
  touched" and open question 1. Each has its pure formula ported and Tier-1-tested here,
  so wiring one later is a small, mechanical addition (declare the reads, write the
  `OutputInto`), not a re-derivation.

**Deviations from PROCESS: none.** Every ported formula is `np.`-free arithmetic
translated to `jnp` unchanged; `safe_pow` sites are value-identical for every input (see
`safe_math.py`'s own docstring) and the `jnp.where` for the aspect correction changes
nothing PROCESS's own `if`/`else` did not already compute on its taken branch.

**Cottax nodes:**

| class | family | owns | reads |
|---|---|---|---|
| `LHThresholdPower` | (abstract family, no `__call__`) | -- | -- |
| `Martin08NominalLHThresholdPower` | `LHThresholdPower` | `.physics.p_l_h_threshold_mw` | `.physics.nd_plasma_electron_line`, `.physics.b_plasma_toroidal_on_axis`, `.physics.a_plasma_surface`, `.physics.m_ions_total_amu` |
| `Martin08UpperLHThresholdPower` | `LHThresholdPower` | same | same |
| `Martin08LowerLHThresholdPower` | `LHThresholdPower` | same | same |
| `Martin08AspectNominalLHThresholdPower` | `LHThresholdPower` | `.physics.p_l_h_threshold_mw` | Martin08Nominal's reads + `.physics.aspect` |
| `Martin08AspectUpperLHThresholdPower` | `LHThresholdPower` | same | same |
| `Martin08AspectLowerLHThresholdPower` | `LHThresholdPower` | same | same |

## test results

Each of the 21 `Tier1Contract` classes carries `test_audit_record_exists`,
`test_value_agreement`, `test_outputs_finite` (all three run on a plain invocation) and
`test_gradient_finite`, `test_gradient_agreement`, `test_gradient_finite_at_zero` (gated
behind `--fp-gradients`), plus 4 plain structural/wrapping tests for the six Martin-family
occupants.

- Plain run (no gradients): `109 passed, 105 skipped`.
- `--fp-gradients` (1 sample per class -- the legacy point): `214 passed`.
- `--fp-gradients --fp-fuzz 5` (adds 5 fuzz samples per class): `550 passed`. Zero new
  `_harness/boundary.py` entries earned across any of it (see "## boundary
  registrations").
