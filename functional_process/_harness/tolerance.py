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


@dataclass(frozen=True)
class DeclaredDeviation:
    """A unit that **deliberately does not compute PROCESS's expression**.

    This is not a tolerance. `Tolerance` answers *"how much round-off is allowed"*;
    this answers *"the port computes something else on purpose, here is what and by how
    much"*. The two must not be confused, which is why loosening `value_tolerance` is
    the wrong way to accommodate a regularised port: it would make the instrument unable
    to see a genuine discrepancy in the same unit, and it would look identical in a diff
    to a test that was quietly widened.

    A contract carrying one is checked *more* strictly than a plain tier-1 unit, not
    less. `Tier1Contract.test_value_agreement` still runs, against `bound`; and
    `test_declared_deviation_is_real` additionally requires that **at least one sample
    actually exceeds the ordinary tier-1 tolerance**. A declared deviation that is not
    needed is therefore a *failure* -- so this cannot be used as a silent tolerance
    loosener, and a reader can tell the two apart at a glance because a loosened
    tolerance has no `reason`, no `record` and no obligation to be exercised.

    Attributes
    ----------
    reason :
        Why the port does not compute PROCESS's expression, in one sentence.
    bound :
        The disagreement this deviation permits. Should be the *measured* worst case
        with a little headroom, not a round number chosen to pass.
    record :
        The audit record (and section) that measures it. Read by
        `test_declared_deviation_is_documented`, which checks the file exists.
    """

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
