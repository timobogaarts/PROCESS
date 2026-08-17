"""PROCESS's own finite-difference scheme, reproduced, with an honest error bar.

The tier-1 gradient check compares `jax.jacfwd` of the port against the derivative
PROCESS itself would use. That reference is not exact, so the interesting question is
not "do they agree to 1e-12" (they will not) but "do they agree to within the reference's
own error". This module answers both halves: it reproduces the scheme exactly, and it
estimates the scheme's error at the point of evaluation by Richardson extrapolation.
"""

import numpy as np

PROCESS_EPSFCN = 1.0e-3
"""Default `data.numerics.epsfcn` (`process/data_structure/numerics.py`).

Note how large this is. A central difference at a *relative* step of 1e-3 has a
truncation error around `1e-6` relative — six orders of magnitude away from the float64
round-off that tier-1 *value* agreement is held to. Holding a gradient to a fixed
`rtol=1e-6` would therefore be a coin flip on the function's curvature; see
`fd_gradient_with_error`.
"""

_MACHINE_EPS = float(np.finfo(np.float64).eps)

REFERENCE_EVALUATION_ULPS = 64.0
"""How many ULPs of noise one *reference* evaluation carries, over a perfect rounding.

The round-off floor below asks how much of `(f_for - f_bac)` is noise rather than signal.
The natural-looking answer, `eps * |f|`, silently assumes the reference is computed to
within a single rounding. It is not: a PROCESS reference is a chain of tens to hundreds
of floating-point operations (and often a `Model.run()` that writes and re-reads `data`),
so its own evaluation error is some modest multiple of `eps * |f|`. Assuming one ULP is
not conservative, it is simply wrong, and it is wrong in the direction that fails correct
ports.

**Measured, not guessed.** `TestCollisionFrequency`'s `temperatures[0]` at
`x = 1.98e-15` needs 77 ULPs to explain the observed spread (the derivative there is
1.6e12 while `f` is 0.155, so the relative step `epsfcn * x = 2e-18` costs about six
digits to cancellation).

64 is deliberately *not* a bound — it is the typical scale, and `Tier1Contract.
gradient_safety` covers the remainder, which is the same division of labour the
truncation term already relies on. Do not read this as "no reference is worse than 64
ULPs"; read it as "one ULP is the wrong order of magnitude". The term only matters where
cancellation already dominates, so raising it does not blunt the check in the ordinary
regime — at `TestNormalizedCollisionFrequency`'s failing point it changes the error bar
by one part in 1e17.
"""


class ZeroPerturbationError(ValueError):
    """Raised when PROCESS's relative perturbation degenerates at `x == 0`.

    `Evaluators.fcnvmc2` perturbs by `x * (1 +/- epsfcn)`, so at `x == 0` both the
    forward and backward points collapse onto zero and the difference quotient is `0/0`.
    This is a real property of PROCESS's gradients, not a harness limitation: sample
    points sitting exactly on zero have no PROCESS reference derivative to compare
    against, and the sampler should avoid them.
    """


def central_difference(fn, x, epsfcn=PROCESS_EPSFCN):
    """Differentiate `fn` at `x` exactly as `Evaluators.fcnvmc2` does.

    Reproduces the scheme rather than using a textbook one: the step is *relative*
    (`x * (1 +/- epsfcn)`) and the denominator is the realised `xfor - xbac`, not the
    nominal `2 * epsfcn * x` — floating-point rounding makes those differ in the last
    bits, and PROCESS uses the realised one.

    Parameters
    ----------
    fn :
        Scalar-argument function returning a float or a 1-D array of floats.
    x :
        Point to differentiate at.
    epsfcn :
        Relative perturbation, PROCESS's `data.numerics.epsfcn`.

    Returns
    -------
    :
        Derivative of each output component with respect to `x`, as a 1-D array.

    Raises
    ------
    ZeroPerturbationError
        If `x == 0`, where the relative perturbation degenerates.
    """
    x = float(x)
    # Exact comparison is deliberate: it is exactly zero, not a neighbourhood of
    # zero, at which PROCESS's relative step collapses.
    if x == 0.0:  # noqa: RUF069
        raise ZeroPerturbationError(
            "PROCESS's relative perturbation x*(1+/-epsfcn) degenerates at x == 0"
        )

    x_for = x * (1.0 + epsfcn)
    x_bac = x * (1.0 - epsfcn)

    f_for = np.atleast_1d(np.asarray(fn(x_for), dtype=float))
    f_bac = np.atleast_1d(np.asarray(fn(x_bac), dtype=float))

    return (f_for - f_bac) / (x_for - x_bac)


def fd_gradient_with_error(fn, x, epsfcn=PROCESS_EPSFCN):
    """Return PROCESS's derivative at `x` alongside an estimate of its own error.

    The error bar comes from Richardson extrapolation rather than a hand-picked
    constant. For a central difference, `D(h) = f'(x) + C h^2 + O(h^4)`, so
    `D(h) - D(h/2) = (3/4) C h^2` and the truncation error of `D(h)` is
    `(4/3) |D(h) - D(h/2)|`. This self-calibrates: it is tight where the function is
    nearly linear and honestly loose where it is not, which is exactly the behaviour a
    fixed tolerance cannot have.

    A round-off floor is added because the extrapolation itself becomes meaningless once
    subtractive cancellation dominates, which for a relative step of `epsfcn` happens at
    roughly `REFERENCE_EVALUATION_ULPS * eps_machine * |f| / (epsfcn * |x|)`.

    **Neither term is a bound**, and that is the point of `Tier1Contract.gradient_safety`
    on top:

    - the truncation term is a *leading-order* extrapolation, so it under-reports
      wherever the neglected `O(h^4)` term is comparable to the `O(h^2)` one;
    - the round-off term models a typical evaluation noise, not a worst case (see
      `REFERENCE_EVALUATION_ULPS`).

    Both shortfalls were measured rather than assumed, on the two `neoclassics.py`
    gradient failures that motivated this docstring. In each the port was *correct* —
    refining the step showed `jacfwd` is the `h -> 0` limit, agreeing to 3e-11
    relative at `epsfcn = 1e-4` — and the bar was too tight, by a factor of about
    1.8 beyond the safety multiplier in both cases. That agreement between two
    unrelated failure modes is why the response was to fix the two terms and size the
    multiplier, rather than to special-case either test.

    Parameters
    ----------
    fn :
        Scalar-argument function returning a float or a 1-D array of floats.
    x :
        Point to differentiate at.
    epsfcn :
        Relative perturbation, PROCESS's `data.numerics.epsfcn`.

    Returns
    -------
    :
        `(derivative, error_estimate)`, both 1-D arrays of the same shape.
    """
    d_h = central_difference(fn, x, epsfcn)
    d_half = central_difference(fn, x, epsfcn / 2.0)

    truncation = (4.0 / 3.0) * np.abs(d_h - d_half)

    f_x = np.atleast_1d(np.asarray(fn(float(x)), dtype=float))
    roundoff = (
        REFERENCE_EVALUATION_ULPS
        * _MACHINE_EPS
        * np.abs(f_x)
        / (epsfcn * abs(float(x)))
    )

    return d_h, truncation + roundoff
