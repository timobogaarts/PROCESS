---
kind: model-unit
status: draft
confidence: medium
---

**Partially ported (2026-08-26), one function; the file's main computation (`ncycle`) is
a stop item, reported below rather than ported.** Written as part of wave-1's
reachability-first dispatch (`next_steps.md`'s tokamak wave). No `unit_registry.md` row,
no `next_steps.md` edit; registration is the consolidation pass's job.

## reachability

**Reached, with a real (non-reporting) reader — constraint 90.**

`CsFatigue.ncycle` is called at `process/models/pfcoil.py:3492`
(`self.cs_fatigue.ncycle(...)`), inside `CSCoil` (constructed at `main.py:2992`, injected
into `PFCoil` per `main.py:651-652` — the constructor-injection evidence
`CLAUDE.md`'s "Models wiring" row names explicitly). `PFCoil.run()` is entered at
`process/core/caller.py:319`, tokamak-only. `cs_fatigue.py` is confirmed **entered** on
`large_tokamak_eval.IN.DAT`, not merely reachable in general: `tokamak_call_surface.md`
§B lists it (93 of 262 lines entered, 1 function), meaning the profiled trace on this
exact reference run's initial iteration-variable point calls `ncycle` at least once — the
call is gated by `if self.data.physics.f_c_plasma_inductive > 0.0e-4:`
(`pfcoil.py:3483-3484`, "only valid for pulsed reactor design"), so its being entered is
itself evidence that condition holds on this run (not re-derived from `f_c_plasma_
inductive`'s value directly in this pass).

`.cs_fatigue.n_cycle` (written by `ncycle`, `pfcoil.py:3489-3490`) is read at
`process/core/solver/constraints.py:1896-1908` (constraint 90, `"Lower limit for CS coil
stress load cycles"`, `n_cycle >= n_cycle_min`) — a real, general-purpose constraint, not
reporting. **Not active on `large_tokamak_eval.IN.DAT`**: `grep -n "^icc = " tests/
regression/input_files/large_tokamak_eval.IN.DAT` lists 25 active constraints and `90` is
not among them. `.cs_fatigue.n_cycle` is also printed at `pfcoil.py:3486-3492`'s output
block (reporting), gated by the same `f_c_plasma_inductive` condition.

**Verdict: outcome (a), reached with a real reader**, per the wave brief's own framing
("cs_fatigue likely feeds a CS stress-cycle constraint" — confirmed). The reader
(constraint 90) is a legitimate, generally-available PROCESS constraint even though this
specific tracked regression input does not activate it; the wave brief's outcome (a) does
not require the reader to be *live* on this run, only real, and this is a materially
different situation from `water_use.md`'s outcome (b) (zero readers anywhere in
`process/`, not merely an inactive constraint ID).

## source

`process/models/cs_fatigue.py` (263 lines, full file in scope). Three `def`s:

| # | function | lines | shape |
|---|---|---|---|
| 1 | `CsFatigue.ncycle` | 22-114 | stateful shell (one `self.data.cs_fatigue.*` read of coefficients) wrapping a `while` loop -- **the stop item**, see below |
| 2 | `CsFatigue.embedded_stress_intensity_factor` | 116-176 | `@staticmethod`, `@njit`, pure, **dead in `process/`** (`grep -rn "embedded_stress_intensity_factor" process/` finds only its own definition; its only caller anywhere is `tests/unit/models/test_cs_fatigue.py`) |
| 3 | `CsFatigue.surface_stress_intensity_factor` | 178-262 | `@staticmethod`, `@njit`, pure, called from inside `ncycle`'s loop (`:95-102`, twice per iteration, `phi = pi/2` and `phi = 0`) |

`CsFatigue.__init__`/`.output`/`.run` are structure/no-ops (`.output` and `.run` are both
empty bodies with only docstrings — "CsFatigue model doesn't need to be run"; the real
work happens through the injected `.ncycle(...)` method call from `PFCoil`, not through
`CsFatigue`'s own `Model.run()`).

## the extraction seam

`surface_stress_intensity_factor` is already a `@staticmethod` over plain floats, zero
`self.data` access -- the cleanest possible seam, needing only `np.` -> `jnp.` and the
`a <= c` branch's `jnp.where` treatment (see the port module's docstring). This is ported
in this pass.

`ncycle` itself has **no seam of this kind**: it is one function that reads five
coefficients off `self.data.cs_fatigue.*` (constants for the whole call, `explicit-arg`)
and then runs a `while` loop whose body calls `surface_stress_intensity_factor` and
accumulates `a`, `c`, `n_pulse`. There is no pure/impure split to make here — the loop
*is* the computation.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.cs_fatigue.paris_power_law`, `.walker_coefficient`, `.paris_coefficient`, `.sf_vertical_crack`, `.sf_radial_crack`, `.fracture_toughness`, `.sf_fast_fracture` | read | explicit-arg | material/safety-factor constants, read once at the top of `ncycle`, unconditional for the whole call |
| `.cs_fatigue.n_cycle` | **write** | explicit-arg | `pfcoil.py:3489`, the constraint-90 operand |
| `.cs_fatigue.t_crack_radial` | **write** | explicit-arg | `pfcoil.py:3490`; also read as `ncycle`'s own initial-condition input (`3 * t_crack_vertical` computed fresh inside `ncycle`, so not actually an implicit read of the stored field despite the name overlap -- `ncycle`'s `t_crack_radial` local shadows it) |
| loop-local `a`, `c`, `n_pulse`, `k_max`, `hoop_stress_MPa`, `cr`, `r` | read/write | **internal to the `while` loop** | not `data` fields; this is the stop item's actual substance |

## proposed signature(s)

```python
def surface_stress_intensity_factor(hoop_stress, t, w, a, c, phi) -> float
```

Ported this pass, verbatim except `jnp.where` for the `a <= c` branch.

**Not proposed for `ncycle`** — see "the stop item" below; a signature cannot be usefully
proposed before the tier/shape question is decided, since the answer changes what the
function returns (a scalar `(n_pulse, t_crack_radial)` pair from a `lax.while_loop`, vs.
something else entirely if the orchestrator picks a different resolution).

## cottax node

None written for `surface_stress_intensity_factor` — per `schema.md`, "skip this section
... while open questions about the signature itself are unresolved." This function does
not own a `VarPath` in PROCESS (its two call sites inside `ncycle`'s loop, `k_a`/`k_c`,
are locals feeding the loop body, not stored fields), and how it composes into `ncycle`'s
eventual port is exactly the open question below. Wrapping it in an `ExplicitFunction`
now would mean inventing `.cs_fatigue.hoop_stress`/`.a`/`.c`/... paths no real PROCESS run
ever writes — tried during this pass and reverted for exactly that reason.

## tier signal

**`surface_stress_intensity_factor`: Tier 1.** No `self.data`, no loop, no CoolProp.

**`ncycle`: neither Tier 1 nor Tier 2 cleanly — the stop item.** See below.

## the stop item: `ncycle`'s `while` loop

`process/models/cs_fatigue.py:85-111`:

```python
while (
    (a <= dz_cs_turn_conduit / self.data.cs_fatigue.sf_vertical_crack)
    and (c <= dr_cs_turn_conduit / self.data.cs_fatigue.sf_radial_crack)
    and (k_max <= self.data.cs_fatigue.fracture_toughness / self.data.cs_fatigue.sf_fast_fracture)
):
    k_a, k_c = self.surface_stress_intensity_factor(...)
    k_max = max(k_a, k_c)
    delta_n = delta / (cr * (k_max ** paris_power_law))
    a += delta * (k_a / k_max) ** paris_power_law
    c += delta * (k_c / k_max) ** paris_power_law
    n_pulse += delta_n
```

This is a hand-rolled Euler integration of crack growth (fixed step `delta = 1e-4` in
crack-area units) that runs until any of three physical thresholds is crossed, then
returns `n_pulse / 2` (the number of full fatigue cycles) and the final `t_crack_radial =
3 * t_crack_vertical` (a constant, computed once before the loop, not accumulated by it —
so only `n_cycle` is genuinely the loop's output).

**Why this does not fit `_audit/test_harness.md`'s existing two tiers:**

- **Not Tier 1** by the harness's own definition ("no internal iteration"). The trip
  count is data-dependent (how many `delta`-sized steps until a threshold is crossed),
  which is exactly what Tier 1 excludes.
- **Not a comfortable Tier 2 either.** `Tier2Contract`'s reasoning (`test_harness.md` §
  Tier 2) is built for a solver whose PROCESS answer *might not be converged ground
  truth* — the motivating case explicitly calls a model "exactly twice, hardcoded" with
  no convergence check. `ncycle` is a different animal: for a fixed `delta`, PROCESS's
  own answer is the *exact* result of that discretisation, not an approximation with an
  unclear stopping point. There is no "PROCESS's own convergence criteria" to tighten
  (`Tier2Contract`'s strategy 2) and no natural "defining equation" (`residual`) whose
  zero characterises the stopping point the way a root-find's residual does (`residual`)
  — the three-way `and` of physical thresholds is a stopping *condition*, not an equation
  a solution satisfies.
- **A `jax.lax.while_loop` port is plausible on its own terms** (JAX supports forward-mode
  AD, i.e. `jacfwd`, through `while_loop` — this harness's own gradient check uses
  `jacfwd`, never `grad`/reverse-mode, so that specific limitation of `lax.while_loop`
  wouldn't even bind here). But a faithful port using the *same* `delta` would reproduce
  PROCESS's value bit-for-bit (both are the same arithmetic, just JAX instead of numpy),
  which looks like a Tier-1 "value agreement at machine precision" story — except the
  *trip count itself* is a step function of the continuous inputs (each `delta`-sized
  increment either does or does not cross a threshold), so the true derivative of
  `n_pulse` with respect to any continuous input is generically **not smooth** — it has a
  sawtooth/staircase structure from the discretisation, on top of whatever smooth trend
  the underlying physics has. Whether PROCESS's own finite-difference gradient
  (`Evaluators.fcnvmc2`, perturbing one input by `epsfcn`) sees this staircase too (likely
  yes, since it also just reruns the same loop) is not measured in this pass.

**Not decided here — reported per the wave brief's escape valve** ("a switch shape the
conventions don't cover... stop on that item, port everything else, and report it. Do not
improvise a policy."). Candidates, not chosen among:

1. Port with `lax.while_loop`, classify Tier 1 anyway (the value-agreement story holds;
   accept that `test_gradient_agreement`'s tolerance may need per-unit tuning around the
   staircase, or that some fuzz points will fail near a step boundary through no fault of
   the port).
2. Port with `lax.while_loop`, classify Tier 2, and write a bespoke `residual` — e.g. "is
   `n_pulse` within one `delta`-step of the true crossing point" — that does not match
   `Tier2Contract`'s existing "plug the answer back into the defining equations" shape and
   would need either a new contract base class or a documented deviation.
3. Treat this as a genuinely new harness shape (a "Tier 2b": exact-but-discontinuous
   deterministic simulation) and propose the addition to `test_harness.md` before porting
   anything against it.

## calls into other models

None. `ncycle` calls only `self.surface_stress_intensity_factor` (same class).

## JAX-difficulty flags

- **`surface_stress_intensity_factor`'s `a <= c` branch** — `needs-lax-cond-or-where`,
  severity `workaround-known`. Resolved via `jnp.where`. **Measured, not assumed, that
  `jnp.where`'s JVP discards a genuinely-untaken branch's own non-finite arithmetic**
  (verified directly: a bare `1/c` in a discarded branch at `c == 0` does not leak) —
  but this does **not** cover the `x ** p` (`0 < p < 1`) zero-derivative trap
  (`_audit/next_steps.md` §9), which is a *local* infinite derivative independent of
  which `jnp.where` arm it sits in and leaks via the ordinary chain rule (`inf * 0 =
  nan`) whenever multiplied by a factor that is itself exactly zero at the same point.
  `h2_le`'s `a_c ** 0.75` (times `a_t_2`, both `0` at `a == 0`) is exactly this shape and
  is what `test_gradient_finite_at_zero` actually caught; every fractional power and bare
  `sqrt` in the function is now `safe_pow`/`safe_sqrt`-wrapped rather than reasoned about
  site by site. Worth carrying forward as a general note for the rest of this wave: the
  "double-`jnp.where` inside `safe_pow`/`safe_sqrt` protects a site regardless of which
  outer `jnp.where` branch it is in" claim from `plasma_geometry.md`'s D1-adjacent
  reasoning is correct for *division*-shaped non-finite values, but this file is a
  counterexample for *fractional-power*-shaped ones -- the outer `jnp.where` alone was
  not enough here.
- **A genuine PROCESS domain gap, confirmed by measurement**: `sqrt(a/t) * pi * c / (2 *
  w)` (the `cos(...)` argument) can exceed `pi/2` for physically unremarkable `t`, `w`,
  `a`, `c` combinations, making `cos` negative and `sqrt(1/cos(...))` `nan` on **both**
  the port and `CsFatigue.surface_stress_intensity_factor` itself (called directly,
  confirmed to return `nan` with no exception). Not flaggable via
  `reference_domain_errors` (PROCESS doesn't raise here); the test file's `fuzz_bounds`
  are chosen to stay clear of it instead, documented in a class comment there.
- **`ncycle`'s `while` loop** — the stop item; see above. Would be
  `needs-lax-while-loop`, no such tag exists yet in `traceability_policy.md` (only
  `needs-lax-cond-or-where` is enumerated there — this file is evidence the policy
  doc's list of idioms is not yet complete for a dynamic-trip-count loop, as distinct
  from `naming_convention.md`'s "in-place sequential mutation" idiom, which is a
  fixed-length `scan`-shaped recurrence, not this).
- No CoolProp, no fractional powers on a base that is zero anywhere on the physical
  domain of interest (`a`, `c`, `t`, `w` are all strictly positive crack/plate
  dimensions).

## suspected defects in PROCESS

**D1 — `surface_stress_intensity_factor`'s `cos(...)` argument has an unguarded domain
gap. Confirmed by measurement.** `sqrt(a_t) * pi * c / (2 * w)` can exceed `pi/2` for
`t`/`w`/`a`/`c` combinations with no special significance (e.g. `t=0.0194`, `w=0.00226`,
`a=0.00439`, `c=0.00492` — all within an order of magnitude of the file's own legacy
sample), making `cos(...)` negative and `sqrt(1/cos(...))` `nan`. Verified by calling
`CsFatigue.surface_stress_intensity_factor` directly at that point: it returns `nan`,
silently (`numpy.sqrt` of a negative float warns, does not raise). Whether this domain
is reachable through `ncycle`'s actual crack-growth trajectory on any tracked regression
input is not checked in this pass — `ncycle`'s loop only ever grows `a`/`c` from small
initial values, so it may or may not walk into this region in practice; flagged as a
JAX-difficulty item (see above) and as a real PROCESS gap worth someone's five-minute
check before assuming it is purely theoretical, per `next_steps.md` §11.7's own standing
caution against promoting an unmeasured reading to a finding — this one *is* measured at
one point, but not traced through an actual `ncycle` trajectory.

## open questions

1. **The stop item itself: which of the three candidates above (or another) should
   `ncycle`'s port follow?** This is the one decision this record asks the orchestrator
   to make; everything else in this file is ready to proceed once it is.
2. **Does PROCESS's own finite-difference gradient of `n_cycle` see the same staircase
   discontinuity a `lax.while_loop` port would?** Not measured. If yes, the existing
   Richardson-extrapolation error bar (`_harness/finite_difference.py`) may already
   widen enough near a step to absorb it; if no (e.g. `epsfcn`'s default step is small
   enough to almost always stay within one crossing), a real disagreement is likely at
   fuzz points near a threshold, and would need explaining rather than dismissing as a
   fuzz failure.
3. **Should `embedded_stress_intensity_factor` be ported for completeness**, the way
   `plasma_geometry.md` ported `sauter_geometry` despite it not being wired to any live
   occupant? Not done in this pass since (unlike `sauter_geometry`) it is not needed for
   `ncycle`'s eventual closure at all (only `surface_stress_intensity_factor` is called)
   — flagged as a cheap, optional addition rather than a blocker.

## ported (2026-08-26)

Port: `functional_process/models/cs_fatigue.py`. Tests:
`tests/functional_process/models/test_cs_fatigue.py`.

**One function**: `surface_stress_intensity_factor`, Tier 1, diffed directly against
`CsFatigue.surface_stress_intensity_factor` (exact `@staticmethod` match, no adapter).
Legacy sample from `tests/unit/models/test_cs_fatigue.py::test_surface_stress_
intensity_factor`. Fuzz bounds deliberately straddle `a == c` so both formula branches
and the `jnp.where` switch-over are exercised by `test_gradient_agreement`.

**Not ported**: `ncycle` (the stop item, see above) and `embedded_stress_intensity_factor`
(dead in `process/`, not needed for `ncycle`'s eventual closure either — see open
question 3).

**No cottax node** for the ported function — see "cottax node" above.

**Deviations from PROCESS**: the `a <= c` Python `if` becomes `jnp.where`; no arithmetic
change.
