---
kind: model-unit
status: draft
confidence: high
---

## source
`process/models/physics/physics.py`, `Physics.outplas`, lines 1783-2877 (1095 lines —
by far the largest single method in the in-scope list). Registry unit #9, chunk C.

## scope note: 1095 lines, one computation

`outplas` is PROCESS's plasma-physics reporting routine — `self.outfile`/`self.mfile`
throughout, called only from `Physics.output()` (`physics.py:219-223`) and, in the
stellarator pipeline, `stellarator.py`'s `output=True` branch (step 5, per
`stellarator_A_orchestration.md`). Confirmed by direct search: `grep -n "self\.data\.[a-zA-Z_.]* = "`
over the full 1095-line body finds exactly **three** assignments, all in the first 40
lines (`physics.py:1790-1822`), and every one of the remaining ~1050 lines is either a
`process_output` call (`po.ovarre`/`po.oheadr`/`po.osubhd`/`po.oblnkl`/`po.ocmmnt`/
`po.oshead`) reading `self.data.*` for display, or a call into another injected
sub-model's own `.output()` method (`self.geometry.output()`, `self.current.output()`,
`self.fields.output()`, `self.beta.output_beta_information()`,
`self.output_temperature_density_profile_info()` — a same-class method **not** in this
unit's scope, confirmed separately to be a pure reporting shell (one write, to a
different-scope field, at `physics.py:3323`, well outside this method) —
`self.density_limit.output()`, `self.exhaust.output()`, `self.scrape_off_layer.output()`,
`self.plasma_transition.output()`, `self.plasma_bootstrap_current.output()`,
`self.dia_current.output()`). None of these are in scope for this unit (they belong to
other, already-scoped-elsewhere units, or are out of scope entirely) and none of them
compute anything this method could not do without — `outplas` never reads a value one
of them writes.

This is the same "reporting isn't quite inert" pattern already found in
`density_limits.py`'s `output()`, `coils/output.py`, and `stellarator_E3`'s
reporting tail — except here the inert part is unusually large (1095 lines) and the
non-inert part unusually small (3 lines), rather than the other way round.

**Four local Python variables computed purely for display/diagnostic logging**
(`p_plasma_imbalance_mw`, `p_reactor_imbalance_mw`, `p_electric_imbalance`,
`p_plasma_imbalance_mw` — power-balance in/out sums, each compared to `0.1` and logged
via `logger.error` if exceeded) are **not** ported: none is written to `self.data`, none
is read anywhere else in this function or consumed by any other unit, and each is a
straight-line sum/difference of already-computed `self.data` fields with no reuse
potential — the same judgement call `coils/output.py`'s record already made for its own
"several trivial inline arithmetic expressions for display only, none worth
extracting."

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.dlamie`, `.vol_plasma`, `.rmajor`, `.b_plasma_toroidal_on_axis`, `.eps`, `.nd_plasma_electron_line`, `.kappa`, `.e_plasma_beta`, `.plasma_current`, `.m_ions_total_amu` | read | explicit-arg | all ten read once each, straight-line, no branch |
| `.physics.nu_star`, `.rho_star`, `.beta_mcdonald` | write | explicit-arg | the only writes in the entire 1095-line method |

Everything else this method touches (several hundred `self.data.*` reads across the
remaining ~1050 lines) is read-only, for display purposes, and out of scope — not
tabulated individually; the audit's judgement is that none of it is a computation to
port, confirmed by the write-count grep above, not by inspection of a sample.

## proposed signature(s)
```python
def calculate_dimensionless_plasma_parameters(
    dlamie,
    vol_plasma,
    rmajor,
    b_plasma_toroidal_on_axis,
    eps,
    nd_plasma_electron_line,
    kappa,
    e_plasma_beta,
    plasma_current,
    m_ions_total_amu,
) -> tuple[float, float, float]: ...
```
Implemented in `dimensionless_parameters.py`.

## cottax node
`DimensionlessPlasmaParameters` (`ExplicitFunction`) in `dimensionless_parameters.py`. No other
node is proposed for this unit — the reporting body has nothing left to wrap.

## tier signal
1 — explicit pure function, three unrelated dimensionless-parameter formulas
(`nu_star`, `rho_star`, `beta_mcdonald`) sharing an argument list, no branches, no calls.

## switches touched
None inside the ported computation. `outplas`'s *reporting* body branches on
`.stellarator.istell` (whether to call `self.fields.output()`/`self.density_limit.output()`)
and `.divertor.n_divertors` (single vs. double null display) — both purely display-path
decisions, out of scope for this record since nothing downstream of them is ported.

## calls into other models
None, in the ported computation. The surrounding (unported) reporting body calls
`self.geometry.output()`, `self.current.output()`, `self.fields.output()`,
`self.beta.output_beta_information()`, `self.output_temperature_density_profile_info()`,
`self.density_limit.output()`, `self.exhaust.output()`, `self.scrape_off_layer.output()`,
`self.plasma_transition.output()`, `self.plasma_bootstrap_current.output()`,
`self.dia_current.output()` — all reporting-only calls on other units' own models, none
of which this unit's scope covers or needs.

## JAX-difficulty flags
None. `nu_star`'s `jnp.sqrt(eps)` and `rho_star`'s `jnp.sqrt(...)` both take physically
non-negative arguments in every real PROCESS configuration (`eps` is an inverse aspect
ratio; the `rho_star` radicand is a product of positive densities/temperatures/masses) —
no domain guard added, consistent with `rether`'s treatment in chunk A.

## real PROCESS defect found: `nu_star` is `nan` for **every** stellarator run

Found by the MDA cold-start catalogue (`DimensionlessPlasmaParameters` was one of two
blocks that ran and emitted a non-finite value from a cold `DataStructure`), then traced
and confirmed against PROCESS itself. **The port is faithful; PROCESS has the defect. No
code change made.**

The last factor of `nu_star` is a division by
`e_plasma_beta**2 * plasma_current` (`physics.py:1801`, ported verbatim at
`dimensionless_parameters.py:73`), and the numerator's second factor is
`15 * ELECTRON_CHARGE**4 * dlamie` (`physics.py:1792`).

`.physics.dlamie` and `.physics.plasma_current` are written in exactly one place each —
`Physics.physics()` at `physics.py:279` and `:286` — and **`Physics.physics()` is never
called in the stellarator pipeline**: `Caller._call_models_once` routes to
`Stellarator.run()`, which calls `self.physics.outplas()` directly
(`stellarator.py:130`) without ever running `Physics.physics()`. Both fields therefore
keep their `physics_variables.py` initial value of `0.0` for the whole run. So the final
division is a literal `0.0 / (e_plasma_beta**2 * 0.0)` = `0.0 / 0.0`.

Measured, not inferred, on `stellarator_helias.IN.DAT`:

| quantity | cold (after `st_new_config`) | PROCESS converged |
|---|---|---|
| `.physics.dlamie` | `0.0` | `0.0` |
| `.physics.plasma_current` | `0.0` | `0.0` |
| `.physics.nu_star` | — | **`nan`** |

The converged column is PROCESS's own `SingleRun(...).run()` result, read straight off
the live `DataStructure` — i.e. **PROCESS's finished stellarator solve stores `nan` in
`.physics.nu_star`**, not just the cold point. Calling the reference wrapper
(`test_dimensionless_parameters._reference_dimensionless_plasma_parameters`) with the cold
argument tuple reproduces it directly, with two `RuntimeWarning`s from
`physics.py:1791` (`divide by zero`, then `invalid value`):

```
PROCESS reference, cold args -> (nan, 0.0022367072720761564, 0.042798566786907036)
port,              cold args -> (nan, 0.0022367072720761564, 0.042798566786907036)
```

Which operand does what, isolated one at a time against the same reference:

- `dlamie = 17`, `plasma_current = 0` → `inf` (`plasma_current` alone is fatal);
- `dlamie = 0`, `plasma_current = 1e6` → `0.0`;
- both zero (the real stellarator state) → `nan`.

`rho_star` and `beta_mcdonald` are unaffected — neither reads either field, and both are
finite at the cold point.

**Why this is benign in PROCESS, and why it is benign here.** The whole
`beta_mcdonald`/`rho_star`/`nu_star` reporting block in `outplas` sits inside
`if self.data.stellarator.istell == 0:` (`physics.py:2784`), so the `nan` is computed
unconditionally at the top of the method and then never displayed for a stellarator; it
is a dead store. In the ported graph it is likewise a dead sink — **no node reads
`.physics.nu_star` or `.physics.rho_star`** (checked against the driven graph), so the
`nan` cannot propagate into any driver, residual or objective.

**Value *and* gradient, checked separately** (the `jnp.sqrt(jnp.maximum(0, x))` trap this
project has been bitten by twice is *not* what this is — the value is already `nan`, so
there is nothing for a gradient test to catch that a value test misses):

- `jax.jacfwd` at the cold point: `dnu_star` is `nan` w.r.t. all ten inputs; `drho_star`
  and `dbeta_mcdonald` are finite.
- `jax.jacfwd` at a tokamak-like point (`dlamie = 17.5`, `plasma_current = 1.8e7`, rest
  cold): the full 3x10 Jacobian is finite.

There is no derivative to lose here, because PROCESS's own value is undefined.

**Recommendation: leave it.** Guarding the denominator would make the port disagree with
PROCESS at the only point where they differ from "both undefined", and would report a
fabricated finite collisionality for a device whose net toroidal current is genuinely
zero — a modelling claim, not a port. The real defect ("PROCESS applies a
current-normalised tokamak collisionality formula to a currentless stellarator, then
hides the resulting `nan` behind an `istell` guard") is PROCESS's to fix, and belongs
upstream. Recorded here rather than acted on, per the standing
`radiation_power.md`/`acc223` precedent of reproducing PROCESS defects faithfully.

## open questions
None.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

2 square roots in this file have been rewritten from `x ** p` / `jnp.sqrt(x)` to
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
