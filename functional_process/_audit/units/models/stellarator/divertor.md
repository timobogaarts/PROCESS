---
kind: model-unit
status: reviewed
confidence: high
---

**Ported.** `divertor.py` / `test_divertor.py`, tier-1 contract passing (fuzz only -- no
PROCESS unit test exercises `st_div`, see below). Whole file in scope, whole file ported
(no partial-scope reason needed): one computational function, one purely-reporting
`output()`.

## source
`process/models/stellarator/divertor.py` (234 lines, full file in scope). Two module-
level functions: `st_div` (the model) and `output` (reporting -- takes only
already-computed locals as arguments plus `data` fields it re-reads for display; no
further computation, confirmed by direct read).

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.flpitch` | read | explicit-arg | field line pitch (rad), `Theta` in source |
| `.physics.rmajor` | read | explicit-arg | |
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | `p_div` in source |
| `.divertor.anginc` | read | explicit-arg | angle of incidence (rad), `alpha` in source |
| `.divertor.xpertin` | read | explicit-arg | perp. heat transport coefficient, `xi_p` in source |
| `.divertor.tdiv` | read | explicit-arg | scrape-off temperature (eV), `T_scrape` in source |
| `.physics.m_fuel_amu` | read | explicit-arg | |
| `.stellarator.bmn` | read | explicit-arg | relative radial field perturbation |
| `.stellarator.shear` | read | explicit-arg | |
| `.stellarator.n_res` | read | explicit-arg | toroidal resonance number |
| `.stellarator.f_w` | read | explicit-arg | island size fraction factor |
| `.stellarator.m_res` | read | explicit-arg | poloidal resonance number |
| `.stellarator.fdivwet` | read | explicit-arg | wetted-area fraction of total plate area |
| `.stellarator.f_asym` | read | explicit-arg | heat load peaking factor |
| `.first_wall.a_fw_total` | read | explicit-arg | only for `f_ster_div_single`, at the very end |
| `.divertor.pflux_div_heat_load_mw` | write | explicit-arg | peak divertor heat load (MW/m2) |
| `.divertor.a_div_surface_total` | write | explicit-arg | total divertor plate area (m2) -- see cross-unit note below |
| `.fwbs.f_ster_div_single` | write | explicit-arg | a write into a *different* area (`.fwbs`, not `.divertor`) from within this file |

No `implicit-io`, `implicit-io-via-callee`, or `redundant-duplicate-write`. Every read is
a plain argument; every write happens exactly once, unconditionally, at the end of one
straight-line computation.

**Cross-unit note**: `.divertor.a_div_surface_total` is exactly the field chunk 1E2's
audit (`stellarator_E2_fwbs_neutronics.md`) found `st_fwbs` reading with a hardcoded
`50.0` fallback on its first call, because `st_fwbs` runs before `st_div` in
`Stellarator.run()`'s call order. Confirmed here: `st_div` is the unconditional, only
producer of this field -- there is nothing wrong with *this* file. The `50.0` fallback is
a call-order problem belonging entirely to `st_fwbs`/`run()` (unit #1), not to unit #4.

`output()`'s own reads (`.divertor.anginc`, `.divertor.xpertin`, `.divertor.tdiv`,
`.stellarator.f_rad`, `.stellarator.f_asym`, `.stellarator.m_res`, `.stellarator.n_res`,
`.stellarator.bmn`, `.stellarator.flpitch`, `.stellarator.f_w`, `.stellarator.shear`,
`.stellarator.fdivwet`, `.divertor.pflux_div_heat_load_mw`) are reporting-only, not
tabulated individually per this audit's established convention (see
`stellarator_G_output.md`). One read there, `.stellarator.f_rad`, is not read anywhere in
`st_div` itself -- display-only, worth noting since it means `output()`'s read set is not
a subset of `st_div`'s.

## proposed signature(s)

```python
def calculate_divertor(
    flpitch: float,
    rmajor: float,
    p_plasma_separatrix_mw: float,
    anginc: float,
    xpertin: float,
    tdiv: float,
    m_fuel_amu: float,
    bmn: float,
    shear: float,
    n_res: float,
    f_w: float,
    m_res: float,
    fdivwet: float,
    f_asym: float,
    a_fw_total: float,
) -> tuple[float, float, float]:
    # returns (pflux_div_heat_load_mw, a_div_surface_total, f_ster_div_single)
    ...
```
Drops the seven reporting-only intermediates (`a_eff`, `l_d`, `l_w`, `f_x`, `l_q`, `w_r`,
`Delta`) that `output()` prints but nothing downstream reads -- same convention as
`structure.md`/`tf_nuclear_heating.md` for locals that exist
only to be displayed.

## cottax node

**Actually written**, in `divertor.py` (`Divertor`, an `ExplicitFunction`), registered
in `functional_process/total_process.py`:

```python
class Divertor(ExplicitFunction):
    pflux_div_heat_load_mw = OutputInto(divertor)
    a_div_surface_total = OutputInto(divertor)
    f_ster_div_single = OutputInto(fwbs)

    def __call__(self, flpitch=From(stellarator), ..., a_fw_total=From(first_wall)):
        return calculate_divertor(flpitch, ..., a_fw_total)
```

## tier signal

**Tier 1.** No internal solve, no calls into other models, no switches, no
data-dependent Python control flow. Every `sqrt` argument is a product/ratio of
physically-positive quantities given a physically-plausible (all-positive) input domain,
so — unlike `density_limits.py`'s `st_sudo_density_limit` — the source never guards a
domain condition here; no `jnp.where`/domain-error handling was needed in the port.

## switches touched

None.

## calls into other models

None. Self-contained arithmetic on already-available `data` fields.

## JAX-difficulty flags

None found. Plain `numpy`/`jnp` arithmetic (`sqrt`, `pi`), no external calls, no dynamic
shapes, no control flow on a traced value.

## open questions

1. **No PROCESS unit test covers `st_div`.** `tests/unit/models/stellarator/` has no
   `test_divertor.py`, and `test_stellarator.py` never calls it directly. This port's
   test coverage (`test_divertor.py`) is fuzz-only against the real `st_div` reference —
   solid for catching a port/reference disagreement, but there is no independently
   human-validated operating point the way `density_limits.py`'s `helias_5b` samples
   provide. Worth noting if a future full-pipeline (tier-4) comparison run can supply a
   `converged_sample` here once that provenance exists (`_harness/sampling.py`).
2. Whether `output()`'s `.stellarator.f_rad` read (display-only, not read by `st_div`
   itself) indicates a stale/planned-but-unused connection to the radiated-power
   fraction, or is simply informational context unrelated to this model's own
   computation — not resolvable from this file alone.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

4 square roots in this file have been rewritten from `x ** p` / `jnp.sqrt(x)` to
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
