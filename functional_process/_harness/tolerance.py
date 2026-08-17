"""Tolerances as named objects carrying their justification.

A bare float in an assertion loses the only thing worth knowing about it: why that
number and not a looser one. `Tolerance` keeps the reason attached, so a tolerance that
was widened to make a test pass is visible as such in the diff rather than looking like
it was always that way.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    """A relative/absolute tolerance pair and the reason it is what it is."""

    rtol: float
    atol: float = 0.0
    reason: str = ""

    def mismatches(self, actual, expected):
        """Return the indices where `actual` and `expected` disagree.

        Parameters
        ----------
        actual :
            Values produced by the port.
        expected :
            Values produced by the PROCESS reference.

        Returns
        -------
        :
            Tuple of `(index, actual, expected, absolute_error, allowed)` for every
            component outside tolerance. Empty when everything agrees.
        """
        actual = np.atleast_1d(np.asarray(actual, dtype=float))
        expected = np.atleast_1d(np.asarray(expected, dtype=float))

        out = []
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            allowed = self.atol + self.rtol * abs(e)
            error = abs(a - e)
            # NaN never compares equal, so catch it explicitly rather than letting
            # `error <= allowed` silently report a mismatch with an unhelpful message.
            if np.isnan(a) != np.isnan(e) or (
                not np.isnan(a) and not (error <= allowed)
            ):
                out.append((i, a, e, error, allowed))
        return tuple(out)

    def describe(self):
        """One-line rendering for use in an assertion message."""
        return f"rtol={self.rtol:g} atol={self.atol:g} ({self.reason})"


MACHINE_PRECISION = Tolerance(
    rtol=1e-12,
    atol=0.0,
    reason=(
        "tier 1: no solver is involved on either side, so the only permitted "
        "difference is float64 round-off from a reassociated expression. A looser "
        "tolerance here is hiding a real discrepancy, not absorbing solver noise"
    ),
)
"""Default tier-1 value tolerance. See `_audit/test_harness.md` § Tier 1."""
