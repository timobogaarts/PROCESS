---
kind: model-unit
status: draft
confidence: high
---

**Ported.** `exhaust.py` now declares `calculate_radiation_fraction`, unchanged from
source, plus one cottax node (`RadiationFraction`).

## source

`process/models/physics/exhaust.py` (221 lines). Registry unit #11, in-scope method
`calculate_radiation_fraction` (L194-220). The file's other three `@staticmethod`s
(`calculate_separatrix_power`, `calculate_psep_over_r_metric`,
`calculate_eu_demo_re_attachment_metric`, L88-192) are already pure, self-contained, no
`self.data` access — but not in the registry's stated scope for this unit and not called
by `calculate_radiation_fraction` itself, so not ported here (a mechanical follow-up if
this unit's scope is ever widened, not a blocker on anything). `PlasmaExhaust.run()` is
a no-op; `PlasmaExhaust.output()` is a pure reporting shell, out of scope by the schema's
own convention (formatted-output writers are never pure-port candidates).

## data footprint

`calculate_radiation_fraction` is already a clean `@staticmethod` in source — no
`self.data` access at all, both arguments explicit.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_plasma_rad_mw` | read | explicit-arg | |
| `.physics.p_plasma_heating_total_mw` | read | explicit-arg | bound to the port's `p_plasma_heating_mw` parameter — a call-site rename (source's own parameter is spelled `p_plasma_heating_mw`, matching neither PROCESS field exactly; kept as source spells it, same move `confinement_time.md` made for `zeff`/`ntau`) |
| `.physics.f_p_plasma_separatrix_rad` | write (by caller) | — | both call sites (`stellarator.py:2369-2373`, `physics.py:1080-1085`, unit #9) assign the return value directly onto this field; `calculate_radiation_fraction` itself performs no write |

## proposed signature(s)

```python
def calculate_radiation_fraction(p_plasma_rad_mw: float, p_plasma_heating_mw: float) -> float:
    ...
```
Unchanged from source's own signature and body shape (one domain guard, one division).

## cottax node

```python
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

class RadiationFraction(ExplicitFunction):
    f_p_plasma_separatrix_rad = Output(lambda s: s.physics.f_p_plasma_separatrix_rad)

    def __call__(
        self,
        p_plasma_rad_mw=Input(lambda s: s.physics.p_plasma_rad_mw),
        p_plasma_heating_mw=Input(lambda s: s.physics.p_plasma_heating_total_mw),
    ):
        return calculate_radiation_fraction(p_plasma_rad_mw, p_plasma_heating_mw)
```
Registered in `exhaust.py`, not yet wired into `total_process.py` — reserved for the
consolidation pass per this wave's boundary.

## tier signal

**Tier 1.** No internal iteration, no call into any other model, one domain guard
(`p_plasma_heating_mw == 0`) handled as an ordinary `jnp.where` rather than a raise (see
"JAX-difficulty flags").

## switches touched

None. `calculate_radiation_fraction` reads no `data.<area>.i_*` field.

## calls into other models

None directly. **Caller-side dependency, not a call from this unit**: both real call
sites (`stellarator.py:2369-2373`, `physics.py:1080-1085`) feed this function's
`p_plasma_heating_mw` argument from `.physics.p_plasma_heating_total_mw`, itself the
output of `physics.py`'s `calculate_total_plasma_heating_power` — one of unit #9's
in-scope methods, currently being audited by another agent in this same wave. This is a
data dependency at the *caller's* call site, not a call this unit's own body makes, so
it does not block porting `calculate_radiation_fraction` itself (its own signature takes
the value as a plain argument, same as every other unit's "close the `data` back door"
treatment) — flagged for whoever wires this node's `Input` up against unit #9's eventual
`total_process.py` registration, not resolved here.

## JAX-difficulty flags

- **`p_plasma_heating_mw == 0` domain guard** — `workaround-known`. Source returns a
  real, finite `0.0` (plus a logged warning) rather than raising, so this is *not* the
  `reference_domain_errors` case (`test_harness.md`'s tier-1 domain-guard convention is
  for a raise PROCESS signals invalid input with — this is PROCESS choosing a defined
  fallback value instead). Ported as `jnp.where` with a **safe denominator**
  (`jnp.where(zero_heating, 1.0, p_plasma_heating_mw)`) rather than a bare
  `p_plasma_rad_mw / p_plasma_heating_mw` inside the outer `where` — the latter would
  compute `x / 0.0` on the untaken branch, which back-propagates a NaN gradient into the
  *taken* branch as well (the exact `jnp.where`-leaks-NaN failure mode
  `test_harness.md`'s worked example and `next_steps.md`'s pinned regression test both
  exist to catch). Verified directly: `test_gradient_finite`/`test_gradient_agreement`
  pass at the `zero-heating-power` sample with this construction.
- No CoolProp calls, no `scipy.optimize`/`fsolve`, no `copy.deepcopy`, no switches.

## open questions

None outstanding for this unit.
