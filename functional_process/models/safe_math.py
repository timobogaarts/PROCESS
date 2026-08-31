"""Value-identical replacements for the operations whose derivative blows up at zero.

`_audit/next_steps.md` §9 records the defect class this module closes: a function that
is **value-correct everywhere and returns a non-finite derivative at one point**, so
every value test passes and only a gradient ever sees it.

The shape is `x ** p` with `0 < p < 1` (`jnp.sqrt` being `p = 0.5`) evaluated at exactly
`x == 0`. The value is `0`, which is right; the derivative is `p * x ** (p - 1) = +inf`,
and JAX's JVP then computes `inf * dx`, which is `+inf` in the tangent direction that
perturbs `x` and `nan` (`inf * 0`) in every other direction. The project has hit this
four times -- twice behind a `jnp.maximum` clamp (`costs.py`'s net-electric-power square
root, `pure_formulas.py`'s `fast_alpha_beta`) and twice as a bare power law --
and the last instance cost hours, because the SQP solver reported 46 non-finite Jacobian
cells as "the problem seems to be non-convex".

**These are not approximations.** Both functions dispatch on `x == 0` and evaluate the
*same* expression on the `x != 0` branch, so every bit of the result is what `x ** p` /
`jnp.sqrt(x)` would have produced -- including `nan` for a negative base, which PROCESS
reaches by raising instead (`_harness.contracts.Tier1Contract.reference_domain_errors`).
The zero branch returns `0.0 ** p` / `sqrt(0.0)`, which is again exactly what the
unguarded expression returns. Value identity is by construction, not by tolerance.

What changes is only the derivative *at* `x == 0`, where the unguarded expression has
none: it becomes `0`. That is the same convention JAX already applies at the kinks of
`jnp.maximum`/`jnp.abs` -- pick a finite element of the subdifferential rather than
poison the whole Jacobian row -- and it is what PROCESS's own one-sided finite
difference cannot distinguish from a very large slope anyway.

The idiom is the *double* select, and both halves are load-bearing. A select
evaluates both branches; a single `where(x > 0, x ** p, 0.0)` still computes
`0.0 ** (p - 1)` in the untaken branch and still leaks its `inf` into the tangent. The
inner select is what keeps the untaken branch's *argument* away from zero.

How the guard is spelled, and why not `jnp.where` (2026-08-31)
--------------------------------------------------------------
`_audit/next_steps.md` §24.11 measured this module at **1,524 jaxpr equations** in
`large_tokamak_nof`'s MDA program -- second only to `models/pfcoil/fields.py` -- across
122 `safe_pow` and 58 `safe_sqrt` invocations. Attributing them by primitive found that
**half of them were the spelling, not the guard**:

- `jnp.where` stages a `jit[name=_where]` wrapper around its `select_n` (386 `jit`
  equations), and promotes its Python-scalar branch inside that wrapper with a
  `convert_element_type` + `broadcast_in_dim` pair (214 + 42) that `lax.full_like` does
  once, statically, for free on a scalar.
- `jnp.zeros_like(x) ** p` was a **live `pow` equation computing a compile-time
  constant** (124 of the 246 `pow`s). `0.0 ** p` is exactly `0.0` for `p > 0`, `1.0` for
  `p == 0` and `inf` for `p < 0` -- exact in binary in every case -- so when `p` is not
  itself traced the constant is folded here rather than emitted.

Both are pure expression rewrites: `lax.select` *is* what `jnp.where` lowers to, and
`lax.full_like` preserves `x`'s shape, dtype and weak type, so no promotion changes.
**Neither branch, neither comparison and neither derivative is touched** -- the guard is
the same double select it always was. Verified bit-identical (value *and* `jax.grad`)
against the `jnp.where` spelling over `{0, 1e-300, 1e-8, 0.5, 1, 2, 1e12, -3, +-inf,
nan}` x `{p = 0.5, 0.25, 1, 0, -1.5, 2, 1.7}` x `{float64, float32, array, traced p,
int base, Python float}`: 0 mismatches.

| | `jnp.where` | `lax.select` |
|---|---|---|
| `safe_pow`, scalar | 7 | **4** |
| `safe_pow`, array | 10 | **6** |
| `safe_pow`, traced `p` | 7 | **5** |
| `safe_sqrt`, scalar | 6 | **4** |
| `safe_sqrt`, array | 10 | **6** |

**What was *not* done, deliberately.** No call site's guard was removed on the argument
that its input is provably positive. The guards exist because four real defects were
found without them, `_harness/boundary.py` registers the sites, and a proof of
positivity that holds today is not one that survives the next iteration variable moving.
The saving above is entirely in how the same guard is written.
"""

import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = ["safe_pow", "safe_sqrt"]


def _zero_to_the(like, p):
    """`0.0 ** p`, as an array shaped and typed like `like`.

    Folded to a constant when `p` is a concrete number -- `0.0 ** p` is `0.0`, `1.0` or
    `inf`, all exact -- and left as a staged `pow` when `p` is a tracer, where there is
    no constant to fold to. `numpy`'s `power` is asked rather than Python's `**` because
    Python raises `ZeroDivisionError` for `p < 0` where both `numpy` and XLA return
    `inf`, which is the value this branch must reproduce.
    """
    try:
        exponent = float(p)
    except (TypeError, ValueError):  # a tracer: no constant to fold to
        return jnp.zeros_like(like) ** p
    with np.errstate(divide="ignore", invalid="ignore"):
        return lax.full_like(like, np.power(np.float64(0.0), np.float64(exponent)))


def safe_pow(x, p):
    """`x ** p` with a finite derivative at `x == 0`.

    Parameters
    ----------
    x :
        The base.
    p :
        The exponent. Any value: the zero branch returns `0.0 ** p`, so `p > 0` gives
        `0.0`, `p == 0` gives `1.0` and `p < 0` gives `inf` -- in every case exactly
        what `x ** p` returns there. Only `0 < p < 1` actually needs this wrapper, but
        the wrapper is correct for all of them, so a site whose exponent is a variable
        needs no case analysis.

    Returns
    -------
    :
        `x ** p`, bit-identical for every `x != 0`, and `0.0 ** p` at `x == 0` with a
        derivative of `0` instead of `inf`/`nan`.
    """
    x = jnp.asarray(x)
    at_zero = x == 0.0
    powered = lax.select(at_zero, lax.full_like(x, 1.0), x) ** p
    return lax.select(at_zero, _zero_to_the(powered, p), powered)


def safe_sqrt(x):
    """`jnp.sqrt(x)` with a finite derivative at `x == 0`.

    Parameters
    ----------
    x :
        The radicand. Negative values still return `nan`, as `jnp.sqrt` does.

    Returns
    -------
    :
        `jnp.sqrt(x)`, bit-identical for every `x != 0`, and `0.0` at `x == 0` with a
        derivative of `0` instead of `inf`/`nan`.

    Notes
    -----
    Not `safe_pow(x, 0.5)`: `sqrt` is a primitive and `pow` is not, so the two are not
    guaranteed to round identically, and every call site here is a transcription of a
    PROCESS expression that used a square root.
    """
    x = jnp.asarray(x)
    at_zero = x == 0.0
    rooted = jnp.sqrt(lax.select(at_zero, lax.full_like(x, 1.0), x))
    return lax.select(at_zero, lax.full_like(rooted, 0.0), rooted)
