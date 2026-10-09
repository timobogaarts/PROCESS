"""Where the port's own roles go: one `Mint` per role, spelled after it (`^converged`,
`^status`, ...), as cottax named a role's places before a role needed a rule."""

from cottax.assemblies.jax import Mint

__all__ = ["Bracket", "Converged", "CurveLhs", "CurveRhs", "CurveX", "Status"]


class Converged(Mint):
    """Where a driver reports whether its convergence test passed."""


class Status(Mint):
    """Where a driver reports its exit code."""


class Bracket(Mint):
    """Where a bracketing driver's first interval is read."""


class CurveX(Mint):
    """Where the x samples of two crossing curves are read."""


class CurveLhs(Mint):
    """Where the left curve's samples are read."""


class CurveRhs(Mint):
    """Where the right curve's samples are read."""
