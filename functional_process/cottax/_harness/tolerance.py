"""Tolerances as named objects carrying their justification."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    """A relative/absolute tolerance pair and the reason it is what it is."""

    rtol: float
    atol: float = 0.0
    reason: str = ""

    def mismatches(self, actual, expected):
        """Return the indices where `actual` and `expected` disagree."""
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


@dataclass(frozen=True)
class DeclaredDeviation:
    """A unit that **deliberately does not compute PROCESS's expression**."""

    reason: str
    bound: Tolerance
    record: str

    def describe(self):
        """One-line rendering for use in an assertion message."""
        return (
            f"DECLARED DEVIATION ({self.reason}) bounded at {self.bound.describe()}, "
            f"measured in {self.record}"
        )


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
