"""What the port's drivers answer, as cottax contracts: the kind of problem, the data
each takes and the reports it makes. The algorithms are in `drivers.py`, each a
`DriverImplementation` of one of these.

A contract's reports are what the graph holds a place for, so they are the contract's,
chosen where a problem is driven; an implementation computes what it can.
"""

from cottax.core.driven import DriverContract, PerProblem, PerUnknown, Role, Start, Steps
from cottax.assemblies.jax import (
    ConstrainedMinimiser,
    FixedPointIterator,
    RootFinder,
)
from cottax.relational import is_root_find

__all__ = [
    "Bracket",
    "BracketedRootFinding",
    "Converged",
    "CurveLhs",
    "CurveRhs",
    "CurveX",
    "IntersectRootFinding",
    "PortFixedPointIteration",
    "PortMinimisation",
    "PortRootFinding",
    "Start",
    "Status",
    "Steps",
    "is_single_unknown_root_find",
]


class Converged(PerProblem):
    """Reported: whether the driver's own convergence test passed."""


class Status(PerProblem):
    """Reported: the driver's own exit code (each driver's codes are its own)."""


class Bracket(PerUnknown):
    """A first guess of an interval holding the unknown's root, `(lower, upper)`: where
    a bracketing driver looks first, not a wall. Changes the path, not the answer."""


class CurveX(PerUnknown):
    """The x samples of the two curves whose crossing the unknown is."""


class CurveLhs(PerUnknown):
    """The left curve's values at `CurveX`."""


class CurveRhs(PerUnknown):
    """The right curve's values at `CurveX`."""


def is_single_unknown_root_find(node) -> bool:
    """`is_root_find`, and exactly one unknown -- what a bracketed root find answers.
    Whether that unknown is a scalar is a fact about its value, not the node, and is
    checked at the call.
    """
    return is_root_find(node) and len(node.unknowns) == 1


class PortRootFinding(RootFinder):
    """Equalities from a `Start`, reporting `reports`."""

    reports: tuple[type[Role], ...] = ()

    @property
    def takes(self) -> tuple[type[Role], ...]:
        return (Start,)


class PortFixedPointIteration(FixedPointIterator):
    """`u = g(u)` from a `Start`, reporting `reports`."""

    reports: tuple[type[Role], ...] = ()

    @property
    def takes(self) -> tuple[type[Role], ...]:
        return (Start,)


class PortMinimisation(ConstrainedMinimiser):
    """One objective, with or without relations, from a `Start`, reporting `reports`."""

    reports: tuple[type[Role], ...] = ()

    @property
    def takes(self) -> tuple[type[Role], ...]:
        return (Start,)


class BracketedRootFinding(DriverContract):
    """A single-unknown root find answered by bracketing, from a `Start` and, if
    `bracket`, a `Bracket`."""

    accepts = staticmethod(is_single_unknown_root_find)
    reports: tuple[type[Role], ...] = ()
    bracket: bool = False

    @property
    def takes(self) -> tuple[type[Role], ...]:
        return (Start, Bracket) if self.bracket else (Start,)


class IntersectRootFinding(DriverContract):
    """A root find that is the crossing of two sampled curves, taking the samples."""

    accepts = staticmethod(is_root_find)

    @property
    def takes(self) -> tuple[type[Role], ...]:
        return (Start, CurveX, CurveLhs, CurveRhs)

    @property
    def reports(self) -> tuple[type[Role], ...]:
        return ()
