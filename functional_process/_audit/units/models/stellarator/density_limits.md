---
kind: model-unit
status: draft
confidence: medium
---

**Cottax nodes added** (update, not part of the original pilot audit): `density_limits.py`
now also declares `SudoDensityLimit`/`EcrhDensityLimit` (`ExplicitFunction`s wrapping the
two tier-1 functions below unchanged), registered in `functional_process/total_process.py`.
`EcrhDensityLimit.dlimit_ecrh`/`.bt_max_ecrh` mint `.stellarator.dlimit_ecrh`/
`.bt_max_ecrh` — invented names, since `st_d_limit_ecrh`'s return values are never
stored to `data` anywhere in the source (confirmed by grep: they stay locals in
`st_density_limits`, clamped, then passed straight into `output()`).

**Re-verified in the MDA triage (`_audit/next_steps.md` §8.1), with line references.**
`st_d_limit_ecrh` returns `(dlimit_ecrh, bt_max)` at
`process/models/stellarator/density_limits.py:152`. Its only two callers both keep the
pair as locals: `st_density_limits` binds them to `ne0_max_ECRH`/`bt_ecrh`
(`density_limits.py:40-43`), `min`-clamps both (`:46-47`) and passes them to `output()`
(`:50`); `power_at_ignition_point` binds them to `ne0_max`/`bt_ecrh_max`
(`:191`) and uses them only to mutate a *deep-copied* proxy `DataStructure` (`:197-206`)
that is discarded. `process/data_structure/stellarator_variables.py` has neither name, nor
any `dlimit`/`bt_max` field. Class (a): genuine mints, correct as-is, unverifiable by the
harness.

Record stays `draft`
overall — the port only covers the tier-1 half, `power_at_ignition_point` is still
unported (see open question 3, now unblocked by 1B but not yet acted on).

## source
`process/models/stellarator/density_limits.py` (290 lines, full file in scope). 5 module-
level functions: `st_density_limits` (orchestrator), `st_sudo_density_limit`,
`st_d_limit_ecrh`, `power_at_ignition_point`, `output`.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | read 3x across the file (into `st_sudo_density_limit`, `st_d_limit_ecrh`, and inline `min()`), same immutable value each time within one call |
| `.physics.p_plasma_loss_mw` | read | explicit-arg | passed into `st_sudo_density_limit` as `powht` |
| `.physics.rmajor` | read | explicit-arg | |
| `.physics.rminor` | read | explicit-arg | |
| `.physics.nd_plasma_electrons_vol_avg` | read | explicit-arg | read *inside* `st_sudo_density_limit` via the passed `data` object even though most other inputs to that function are already explicit args — see open question 1 |
| `.physics.nd_plasma_electron_line` | read | explicit-arg | same as above |
| `.physics.nd_plasma_electrons_max` | write | — (redundant double-write) | written once *inside* `st_sudo_density_limit` (line 109, via the `data` back-door) and again by the caller assigning the same function's return value (`st_density_limits` line 30) to the identical field. Harmless (idempotent) but the internal write should be dropped when porting — the return value already carries it. |
| `.stellarator.max_gyrotron_frequency` | read | explicit-arg | also re-read in `output()` |
| `.physics.i_plasma_pedestal` | read | **not a real switch here — see open question 2** | only value `0` has a defined formula; other values hit a dead branch (see JAX-difficulty flags) |
| `.physics.nd_plasma_electron_on_axis` | read | explicit-arg | used to clamp `ne0_max_ECRH`; also re-read in `output()` |
| `.physics.temp_plasma_electron_on_axis_kev` | read | explicit-arg | `output()` only, reporting |
| `.stellarator.te0_ecrh_achievable` | read | explicit-arg | `output()` only, then passed into `power_at_ignition_point` |
| `.physics.alphat` | read | explicit-arg (via proxy) | `power_at_ignition_point`, read once before the internal solve |
| `.physics.alphan` | read | explicit-arg (via proxy) | same |
| `.physics.nd_plasma_electrons_vol_avg` | read then **write** (on proxy) | implicit-io | `power_at_ignition_point` overwrites this on the deep-copied proxy before calling `st_phys` — `st_phys` then implicitly depends on the just-written value rather than receiving it as an argument |
| `.physics.b_plasma_toroidal_on_axis` | read then **write** (on proxy) | implicit-io | same pattern |
| `.physics.temp_plasma_electron_vol_avg_kev` | write (on proxy) | implicit-io | written from `te0_available`, then implicitly read by `st_phys` |
| `.physics.p_plasma_loss_mw` | read (on proxy, **after** 2x `st_phys` calls) | implicit-io | value produced entirely inside `st_phys` — not resolvable from this file, `st_phys` is out of current audit scope (registry unit #1, pending) |
| `.physics.pscalingmw` | read (on proxy, after 2x `st_phys`) | implicit-io | same |

## proposed signature(s)

Tier-1, already effectively pure, straightforward port:
```python
def calculate_ecrh_density_limit(
    gyro_frequency_max: float, b_plasma_toroidal_on_axis: float, i_plasma_pedestal: int
) -> tuple[float, float]:  # (dlimit_ecrh, bt_max)
    ...
```

Tier-1 once the two implicit reads are made explicit (drop the `data` parameter and the
internal side-effect write entirely — the return value already carries everything the
caller needs):
```python
def calculate_sudo_density_limit(
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss_mw: float,
    rmajor: float,
    rminor: float,
    nd_plasma_electrons_vol_avg: float,
    nd_plasma_electron_line: float,
) -> float:  # nd_plasma_electrons_max
    ...
```

Tier-1 composition of the two above (drop the `f_output`/reporting branch from the
computational core — reporting is not in scope, see `output()` below):
```python
def calculate_density_limits(
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss_mw: float,
    rmajor: float,
    rminor: float,
    nd_plasma_electrons_vol_avg: float,
    nd_plasma_electron_line: float,
    max_gyrotron_frequency: float,
    i_plasma_pedestal: int,
    nd_plasma_electron_on_axis: float,
) -> tuple[float, float, float]:  # (nd_plasma_electrons_max, ne0_max_ECRH, bt_ecrh)
    ...
```

Tier-2, **blocked** — cannot write a real signature until registry unit #1 (`stellarator.py`,
specifically `st_phys`) is audited, since every field `st_phys` reads/writes internally is
currently invisible from this file:
```python
def solve_ignition_point(
    gyro_frequency_max: float,
    te0_available: float,
    alphat: float,
    alphan: float,
    nd_plasma_electrons_vol_avg: float,
    b_plasma_toroidal_on_axis: float,
    i_plasma_pedestal: int,
    # + whatever st_phys needs, TBD — this signature is provisional
) -> tuple[float, float]:  # (powerht_out, pscalingmw)
    ...
```

## tier signal

- `st_d_limit_ecrh`: **tier 1** — no data access, no internal solve, no calls out.
- `st_sudo_density_limit`: **tier 1** — pure once the `data` back-door is closed (see
  above); currently mixed explicit/implicit.
- `st_density_limits`: **tier 1** for the core path (composes two tier-1 functions); the
  `f_output=True` branch pulls in tier-2 material (see below) — a concrete instance of
  the "tier boundary depends on which branch is taken" issue flagged earlier for the
  audit as a whole, not just this file.
- `power_at_ignition_point`: **tier 2** — internal iterative solve, but not a real
  convergence-checked one: it calls `proxy_stellarator.st_phys(False)` exactly **twice**,
  hardcoded, with the comment *"The second call seems to be necessary for all values to
  'converge' (and is sufficient)"* (line 208-209 of the source). This is empirically
  tuned, not derived — flagged as a priority item for the later `Drive`-vs-original-
  behaviour comparison (per the "mimic original closure behaviour" goal), since a real
  convergence-checked driver may legitimately need a different number of iterations, or
  may reveal this 2-iteration approximation wasn't actually converged.
- `output`: not computational (reporting shell), but see JAX-difficulty flags — it isn't
  *purely* reporting.

## switches touched

- `i_plasma_pedestal` (`.physics.i_plasma_pedestal`) — see open question 2 below; not a
  clean split/keep-static case, needs its own decision.
- No other switches read in this file.

## calls into other models

- `power_at_ignition_point` calls `proxy_stellarator.st_phys(False)` (twice) — `st_phys`
  is a method on the `Stellarator` class itself (registry unit #1, `stellarator.py`,
  currently `pending`). This file cannot be fully audited independent of that unit.
- `output` calls `power_at_ignition_point` (same file) — see JAX-difficulty flags, this
  is a reporting function invoking real computation.

## JAX-difficulty flags

- **`copy.deepcopy(stellarator)`** in `power_at_ignition_point` — `blocker` (for the
  direct port) / not applicable to the pure rewrite target, since the whole point of the
  rewrite is to not need this: the deepcopy exists only to sandbox mutation of a stateful
  OOP model object, which a pure function naturally doesn't need. Recorded so the "why is
  this here" isn't lost — it's evidence of *why* `st_phys` needs isolating explicit
  inputs/outputs, not a pattern to replicate.
- **`ProcessValueError` raised on a data-dependent condition** in `st_sudo_density_limit`
  (`if arg <= 0.0e0: raise ...`) — `workaround-known` (`needs-lax-cond-or-where`): a
  domain-validity guard, standard `jnp.where(cond, jnp.nan, value)` or `checkify` pattern.
- **Likely latent bug, not a JAX-difficulty flag but worth surfacing**: in `st_d_limit_ecrh`,
  `dlimit_ecrh` is only assigned inside `if i_plasma_pedestal == 0:`; the `else` branch
  only logs an error and does not assign `dlimit_ecrh`, so a nonzero `i_plasma_pedestal`
  would raise `UnboundLocalError` in the original PROCESS source, not a handled error path.
  Not fixing this (not touching `process/`) — flagging so the pure port either preserves
  the same restriction explicitly (assert/precondition `i_plasma_pedestal == 0`) or is
  designed with the human reviewer's input on what should happen instead.
- **`min()`/`max()` clamps on physically-motivated bounds** (e.g.
  `max(proxy_stellarator.data.physics.p_plasma_loss_mw, 0.00001e0)`, with the comment "the
  radiation module sometimes returns negative heating power") — `minor`, trivially
  `jnp.maximum`/`jnp.minimum`, but flagged because it's evidence of upstream numerical
  instability elsewhere in the physics chain being papered over with a floor value rather
  than fixed at the source; worth keeping in mind when `st_phys`/radiation modules are
  audited.

## open questions

1. **Why does `st_sudo_density_limit` take both explicit args *and* the full `data`
   object?** It only uses `data` for 2 reads + 1 (redundant) write, all of which could be
   ordinary explicit args/return values. This looks like an incomplete past refactor
   rather than an intentional design — worth confirming there's no reason (e.g. some
   other call site relying on the side-effect write) before assuming it's safe to drop.
   Low-confidence: only checked this file, not all call sites of `st_sudo_density_limit`.
2. **`i_plasma_pedestal` isn't cleanly a split-or-keep-static switch.** Only value `0` has
   a real formula in `st_d_limit_ecrh`; other values are effectively unsupported (see
   latent-bug flag above). Recommend treating it as a **precondition** (this function
   requires `i_plasma_pedestal == 0`) rather than a formula switch needing either
   splitting or a static branch — but this is a judgment call for review, not applied
   unilaterally here.
3. **`power_at_ignition_point`'s pure signature is genuinely blocked on auditing
   `st_phys`** (registry unit #1). Recommend `st_phys` (or at least the portion of
   `stellarator.py` covering it) be prioritised early among the remaining ~18 units,
   since at least this one other unit already depends on it directly.
4. **Should the ignition-point solve be a first-class `run()`-time output rather than
   only computed inside `output()`?** `output()` calls `power_at_ignition_point` to get
   `powerht_local`/`pscalingmw_local` purely for reporting (lines 265-281) — this is a
   real computation, not formatting, living in a place the audit's general policy treats
   as out of scope (reporting shells). Flagging rather than deciding.

## naming-convention / schema notes for the process, not the content

- The nested-object-read open question in `naming_convention.md` (`.confinement.*` style)
  didn't come up in this file — no sub-namespace reads here. No evidence either way yet.
- One gap the schema doesn't cover: how to record a **redundant duplicate write** (the
  `.physics.nd_plasma_electrons_max` case) — used a free-text note in the classification
  column rather than a class label. Suggest adding `redundant-write` as a recognised
  classification value alongside `explicit-arg`/`implicit-io` if this pattern recurs.
- The "read then write on a deep-copied proxy, consumed by an opaque callee" pattern
  (`power_at_ignition_point`) doesn't map cleanly to the binary explicit-arg/implicit-io
  split — I used implicit-io for these rows since the *callee's* dependency is implicit,
  but the schema might eventually want a third label (e.g. `implicit-io-via-callee`) to
  distinguish "this function itself branches on stale state" from "this function's writes
  are consumed implicitly by something it calls." Left as-is for this pilot; flagging for
  the schema review.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

1 square root in this file has been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
