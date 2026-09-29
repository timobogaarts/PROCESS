"""The driver answering the coil intersect's root find: `intersect`'s own algorithm,
over the two sampled curves it is handed as driver data."""

import jax.numpy as jnp

from functional_process.cottax.architectures.contracts import (
    CurveLhs,
    CurveRhs,
    CurveX,
    IntersectRootFinding,
    Start,
)
from functional_process.cottax.architectures.reading import Gaps, GapsDriver
from functional_process.models.stellarator.coils.coils import intersect


class IntersectBisectionNewtonPolish(GapsDriver[IntersectRootFinding]):
    """Answering `Intersect`'s declared root find -- exactly the
    algorithm `intersect` (above) already uses: `optx.Bisection` over the curves'
    full x-overlap, then a few exact Newton corrections
    (`_intersect_newton_polish`).

    The curves arrive as driver data (`CurveX` / `CurveLhs` / `CurveRhs`), one place per
    unknown per role; the start is optional, since `intersect`'s own domain clamping
    makes any point a safe `xin` and the median of the x samples is the principled
    default.

    It never evaluates a gap: the curves are the statement, sampled.
    """

    needs = frozenset({CurveX, CurveLhs, CurveRhs})
    provides = frozenset()
    computes = ()

    def solve(self, gaps: Gaps, data):
        start = data.get(Start)
        (wp_width_r,) = data[CurveX]
        (lhs,) = data[CurveLhs]
        (rhs,) = data[CurveRhs]
        xin = start[0] if start is not None else jnp.median(wp_width_r)
        return (intersect(wp_width_r, lhs, wp_width_r, rhs, xin),)
