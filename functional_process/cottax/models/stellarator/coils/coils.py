"""Pure-functional port of `process/models/stellarator/coils/coils.py` (registry #10)."""

import jax  # noqa: F401
import jax.numpy as jnp
import optimistix as optx  # noqa: F401
from cottax.execution.driver import GapDriver, Gaps
from cottax.execution.drivers.kinds import Start
from cottax.interfaces import is_root_find
from cottax.interfaces.pytree_namespace_module import (
    From,
    ImplicitFunction,
    OutputInto,
)
from cottax.pytree.mint import Minted

from functional_process.cottax.paths import (
    stellarator,
    tfcoil,
)
from functional_process.cottax.wraps import WrapsFunction
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


class JcritIterNb3sn(WrapsFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 1` (ITER Nb3Sn)."""

    fn = jcrit_from_material_iter_nb3sn

    t_helium = From(tfcoil)
    b_max = From(tfcoil)

    j_crit_sc = OutputInto(tfcoil)


class JcritBi2212(WrapsFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 2` (Bi-2212)."""

    fn = jcrit_from_material_bi2212

    t_helium = From(tfcoil)
    b_max = From(tfcoil)
    j_tf_wp = From(tfcoil)
    f_a_tf_turn_cable_space_extra_void = From(tfcoil)
    fhts = From(tfcoil)
    f_a_tf_turn_cable_copper = From(tfcoil)

    j_crit_sc = OutputInto(tfcoil)


class JcritNbtiLubell(WrapsFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 3` (NbTi, Lubell scaling)."""

    fn = jcrit_from_material_nbti_lubell

    t_helium = From(tfcoil)
    b_max = From(tfcoil)

    j_crit_sc = OutputInto(tfcoil)


class JcritIterNb3snUserDefined(WrapsFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 4` (ITER Nb3Sn,
    user-defined).
    """

    fn = jcrit_from_material_iter_nb3sn_user_defined

    t_helium = From(tfcoil)
    b_max = From(tfcoil)
    bcritsc = From(tfcoil)
    tcritsc = From(tfcoil)

    j_crit_sc = OutputInto(tfcoil)


class JcritWstNb3sn(WrapsFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 5` (WST Nb3Sn)."""

    fn = jcrit_from_material_wst_nb3sn

    t_helium = From(tfcoil)
    b_max = From(tfcoil)

    j_crit_sc = OutputInto(tfcoil)


class JcritRebco(WrapsFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 6` (REBCO, CroCo strand)."""

    fn = jcrit_from_material_rebco

    t_helium = From(tfcoil)
    b_max = From(tfcoil)

    j_crit_sc = OutputInto(tfcoil)


class JcritGlNbti(WrapsFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 7` (Durham GL Nb-Ti)."""

    fn = jcrit_from_material_gl_nbti

    t_helium = From(tfcoil)
    b_max = From(tfcoil)
    b_crit_upper_nbti = From(tfcoil)
    t_crit_nbti = From(tfcoil)

    j_crit_sc = OutputInto(tfcoil)


class JcritGlRebco(WrapsFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 8` (Durham GL REBCO)."""

    fn = jcrit_from_material_gl_rebco

    t_helium = From(tfcoil)
    b_max = From(tfcoil)

    j_crit_sc = OutputInto(tfcoil)


# Not assembled into a `Switch`/`Alternative` group here, and not registered in
# `total_process.py` -- out of this pass's boundary (see the module docstring above and
# `coils.md`'s "cottax node" section). All 8 classes above mint the identical output
# `VarPath` (`.tfcoil.j_crit_sc`), which is exactly what `Switch.check_arms_are_exclusive`
# wants to see to accept them as one mutually-exclusive group once assembled -- the
# consolidation step is: `Switch(path=".tfcoil.i_tf_sc_mat", alternatives=(Alternative(
# value=1, declarations=(JcritIterNb3sn,)), Alternative(value=2, declarations=(
# JcritBi2212,)), ...))`, one `Alternative` per class above, no further code.


class Intersect(ImplicitFunction):
    """cottax node: `intersect`, as a genuine `ImplicitFunction`/`RootFind` pair."""

    wp_width_r_min = OutputInto(stellarator)

    def residual(
        self,
        wp_width_r_min=From(stellarator),
        wp_width_r=From(stellarator),
        lhs=From(stellarator),
        rhs=From(stellarator),
    ):
        return intersect_residual(wp_width_r_min, wp_width_r, lhs, wp_width_r, rhs)


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
