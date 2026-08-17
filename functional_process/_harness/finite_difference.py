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
    roughly `eps_machine * |f| / (epsfcn * |x|)`.

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
    roundoff = _MACHINE_EPS * np.abs(f_x) / (epsfcn * abs(float(x)))

    return d_h, truncation + roundoff
