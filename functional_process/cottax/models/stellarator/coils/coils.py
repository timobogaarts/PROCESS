"""Pure-functional port of `process/models/stellarator/coils/coils.py` (registry #10)."""

import jax  # noqa: F401

from functional_process.cottax.paths import (
    stellarator,
    tfcoil,
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
from functional_process.models.stellarator.coils.coils import (
    bmax_from_awp,  # noqa: F401
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

# cottax node: `jcrit_from_material`, `i_tf_sc_mat == 1` (ITER Nb3Sn).
JcritIterNb3sn = tfcoil.function(jcrit_from_material_iter_nb3sn, "j_crit_sc")


# cottax node: `jcrit_from_material`, `i_tf_sc_mat == 2` (Bi-2212).
JcritBi2212 = tfcoil.function(jcrit_from_material_bi2212, "j_crit_sc")


# cottax node: `jcrit_from_material`, `i_tf_sc_mat == 3` (NbTi, Lubell scaling).
JcritNbtiLubell = tfcoil.function(jcrit_from_material_nbti_lubell, "j_crit_sc")


# cottax node: `jcrit_from_material`, `i_tf_sc_mat == 4` (ITER Nb3Sn,
# user-defined).
JcritIterNb3snUserDefined = tfcoil.function(
    jcrit_from_material_iter_nb3sn_user_defined, "j_crit_sc"
)


# cottax node: `jcrit_from_material`, `i_tf_sc_mat == 5` (WST Nb3Sn).
JcritWstNb3sn = tfcoil.function(jcrit_from_material_wst_nb3sn, "j_crit_sc")


# cottax node: `jcrit_from_material`, `i_tf_sc_mat == 6` (REBCO, CroCo strand).
JcritRebco = tfcoil.function(jcrit_from_material_rebco, "j_crit_sc")


# cottax node: `jcrit_from_material`, `i_tf_sc_mat == 7` (Durham GL Nb-Ti).
JcritGlNbti = tfcoil.function(jcrit_from_material_gl_nbti, "j_crit_sc")


# cottax node: `jcrit_from_material`, `i_tf_sc_mat == 8` (Durham GL REBCO).
JcritGlRebco = tfcoil.function(jcrit_from_material_gl_rebco, "j_crit_sc")


# Not assembled into a `Switch`/`Alternative` group here, and not registered in
# `total_process.py` -- out of this pass's boundary (see the module docstring above and
# `coils.md`'s "cottax node" section). All 8 classes above mint the identical output
# `VarPath` (`.tfcoil.j_crit_sc`), which is exactly what `Switch.check_arms_are_exclusive`
# wants to see to accept them as one mutually-exclusive group once assembled -- the
# consolidation step is: `Switch(path=".tfcoil.i_tf_sc_mat", alternatives=(Alternative(
# value=1, declarations=(JcritIterNb3sn,)), Alternative(value=2, declarations=(
# JcritBi2212,)), ...))`, one `Alternative` per class above, no further code.


# cottax node: `intersect`, as a genuine `ImplicitFunction`/`RootFind` pair.
def _intersect(wp_width_r_min, wp_width_r, lhs, rhs):
    return intersect_residual(wp_width_r_min, wp_width_r, lhs, wp_width_r, rhs)


Intersect = stellarator.residual(_intersect, "wp_width_r_min")
