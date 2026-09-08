"""Pure-functional port of `process/models/stellarator/coils/coils.py` (registry #10).
"""

import jax  # noqa: F401
import jax.numpy as jnp
import optimistix as optx  # noqa: F401
from cottax.evaluate import (
    AbstractDriver,
    ConditionMap,
)
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ImplicitFunction,
    OutputInto,
    resolve,
)
from cottax.problem import (
    RootFind,
    Start,
    is_root_find,
)
from cottax.spec import (
    VarPath,
)

from functional_process.models.physics.superconductors import (
    bi2212,  # noqa: F401
    gl_nbti,  # noqa: F401
    gl_rebco,  # noqa: F401
    itersc,  # noqa: F401
    jcrit_nbti,  # noqa: F401
    jcrit_rebco,  # noqa: F401
    western_superconducting_nb3sn,  # noqa: F401
)
from functional_process.cottax.paths import (
    stellarator,
    tfcoil,
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


class JcritIterNb3sn(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 1` (ITER Nb3Sn)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_iter_nb3sn(t_helium, b_max)


class JcritBi2212(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 2` (Bi-2212)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
        j_tf_wp=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        fhts=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
    ):
        return jcrit_from_material_bi2212(
            t_helium,
            b_max,
            j_tf_wp,
            f_a_tf_turn_cable_space_extra_void,
            fhts,
            f_a_tf_turn_cable_copper,
        )


class JcritNbtiLubell(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 3` (NbTi, Lubell scaling)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_nbti_lubell(t_helium, b_max)


class JcritIterNb3snUserDefined(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 4` (ITER Nb3Sn,
    user-defined).
    """

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
        bcritsc=From(tfcoil),
        tcritsc=From(tfcoil),
    ):
        return jcrit_from_material_iter_nb3sn_user_defined(
            t_helium, b_max, bcritsc, tcritsc
        )


class JcritWstNb3sn(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 5` (WST Nb3Sn)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_wst_nb3sn(t_helium, b_max)


class JcritRebco(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 6` (REBCO, CroCo strand)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_rebco(t_helium, b_max)


class JcritGlNbti(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 7` (Durham GL Nb-Ti)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
        b_crit_upper_nbti=From(tfcoil),
        t_crit_nbti=From(tfcoil),
    ):
        return jcrit_from_material_gl_nbti(
            t_helium, b_max, b_crit_upper_nbti, t_crit_nbti
        )


class JcritGlRebco(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 8` (Durham GL REBCO)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_gl_rebco(t_helium, b_max)


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


_WP_WIDTH_R_PATH = resolve(stellarator.wp_width_r, VarPath)
_LHS_PATH = resolve(stellarator.lhs, VarPath)
_RHS_PATH = resolve(stellarator.rhs, VarPath)


class IntersectBisectionNewtonPolish(AbstractDriver):
    """Concrete `AbstractDriver` answering `Intersect`'s declared `RootFind` -- exactly
    the algorithm `intersect` (above) already uses: `optx.Bisection` over the curves'
    full x-overlap, then a few exact Newton corrections (`_intersect_newton_polish`).
    """

    accepts = staticmethod(is_root_find)
    requires = (Start,)

    def __call__(self, conditions: ConditionMap, data):
        # `requires` stays empty: this driver does not *need* a start -- `intersect`'s
        # own domain clamping makes any point a safe `xin`, so it has a principled
        # default. `data.get` rather than `data[Start]` for exactly that reason.
        start = data.get(Start)
        wp_width_r = conditions.context[_WP_WIDTH_R_PATH]
        lhs = conditions.context[_LHS_PATH]
        rhs = conditions.context[_RHS_PATH]
        xin = start[0] if start is not None else jnp.median(wp_width_r)
        return (intersect(wp_width_r, lhs, wp_width_r, rhs, xin),)
