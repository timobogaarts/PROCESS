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

The idiom is the *double* `jnp.where`, and both halves are load-bearing. `jnp.where`
evaluates both branches; a single `jnp.where(x > 0, x ** p, 0.0)` still computes
`0.0 ** (p - 1)` in the untaken branch and still leaks its `inf` into the tangent. The
inner `where` is what keeps the untaken branch's *argument* away from zero.
"""

import jax.numpy as jnp

__all__ = ["safe_pow", "safe_sqrt"]


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
    at_zero = x == 0.0
    return jnp.where(at_zero, jnp.zeros_like(x) ** p, jnp.where(at_zero, 1.0, x) ** p)


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
    at_zero = x == 0.0
    return jnp.where(at_zero, 0.0, jnp.sqrt(jnp.where(at_zero, 1.0, x)))
