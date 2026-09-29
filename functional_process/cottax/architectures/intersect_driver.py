"""The driver answering the coil intersect's root find, moved out of the model module
unchanged: it is ported with the other drivers (Phase 2)."""

import jax  # noqa: F401
import jax.numpy as jnp
import optimistix as optx  # noqa: F401
from cottax.execution.driver import GapDriver, Gaps
from cottax.execution.drivers.kinds import Start
from cottax.interfaces import is_root_find
from cottax.pytree.mint import Minted
from functional_process.models.physics.superconductors import (
    bi2212,  # noqa: F401
    gl_nbti,  # noqa: F401
    gl_rebco,  # noqa: F401
    itersc,  # noqa: F401
    jcrit_nbti,  # noqa: F401
    jcrit_rebco,  # noqa: F401
    western_superconducting_nb3sn,  # noqa: F401
)
from functional_process.models.stellarator.coils.coils import (
    bmax_from_awp,  # noqa: F401
    intersect,
    intersect_residual,
    j_crit_cable_from_fraction,  # noqa: F401
    jcrit_from_material_bi2212,
    jcrit_from_material_gl_nbti,
    jcrit_from_material_gl_rebco,
    jcrit_from_material_iter_nb3sn,
    jcrit_from_material_iter_nb3sn_user_defined,
    jcrit_from_material_nbti_lubell,
    jcrit_from_material_rebco,
    jcrit_from_material_wst_nb3sn,
)


CurveX = Minted("curve_x")
CurveLhs = Minted("curve_lhs")
CurveRhs = Minted("curve_rhs")
"""How `IntersectBisectionNewtonPolish` names the three curve samples it brackets over.

Driver **data**, not context: what an algorithm reads for itself is an ordinary read at
a name derived from the problem's unknown, so `Assign` mints `^curve_x.stellarator
.wp_width_r_min` and friends and a `Rename` points each at the node that computes it (or
a caller supplies it at the boundary). The block's own values are not reachable from a
driver -- a reading of the statement answers `gaps(*unknowns)` and nothing else.
"""


class IntersectBisectionNewtonPolish(GapDriver):
    """Concrete `GapDriver` answering `Intersect`'s declared root find -- exactly the
    algorithm `intersect` (above) already uses: `optx.Bisection` over the curves'
    full x-overlap, then a few exact Newton corrections
    (`_intersect_newton_polish`).

    The curves arrive as driver data (`CurveX` / `CurveLhs` / `CurveRhs`), one place per
    unknown per naming; the start is optional, since `intersect`'s own domain clamping
    makes any point a safe `xin` and the median of the x samples is the principled
    default.

    A `GapDriver` because a root find is what it answers, and it never calls the seam:
    the curves *are* the statement, sampled, so nothing here evaluates a gap and
    `square` would buy a refusal at the cost of one evaluation.
    """

    accepts = staticmethod(is_root_find)
    requires = (Start, CurveX, CurveLhs, CurveRhs)

    def solve(self, gaps: Gaps, data):
        start = data.get(Start)
        (wp_width_r,) = data[CurveX]
        (lhs,) = data[CurveLhs]
        (rhs,) = data[CurveRhs]
        xin = start[0] if start is not None else jnp.median(wp_width_r)
        return (intersect(wp_width_r, lhs, wp_width_r, rhs, xin),)
