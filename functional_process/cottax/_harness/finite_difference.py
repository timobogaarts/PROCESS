"""PROCESS's own finite-difference scheme, reproduced, with an honest error bar."""

import numpy as np

PROCESS_EPSFCN = 1.0e-3
"""Default `data.numerics.epsfcn` (`process/data_structure/numerics.py`)."""

_MACHINE_EPS = float(np.finfo(np.float64).eps)

REFERENCE_EVALUATION_ULPS = 64.0
"""How many ULPs of noise one *reference* evaluation carries, over a perfect rounding.
"""


class ZeroPerturbationError(ValueError):
    """Raised when PROCESS's relative perturbation degenerates at `x == 0`."""


def central_difference(fn, x, epsfcn=PROCESS_EPSFCN):
    """Differentiate `fn` at `x` exactly as `Evaluators.fcnvmc2` does."""
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
    """Return PROCESS's derivative at `x` alongside an estimate of its own error."""
    d_h = central_difference(fn, x, epsfcn)
    d_half = central_difference(fn, x, epsfcn / 2.0)

    truncation = (4.0 / 3.0) * np.abs(d_h - d_half)

    f_x = np.atleast_1d(np.asarray(fn(float(x)), dtype=float))
    roundoff = (
        REFERENCE_EVALUATION_ULPS * _MACHINE_EPS * np.abs(f_x) / (epsfcn * abs(float(x)))
    )

    return d_h, truncation + roundoff
