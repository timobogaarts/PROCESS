"""Pure-functional port of `process/models/stellarator/coils/coils.py` (registry #10).

Audit record: `functional_process/models/stellarator/coils/coils.md`. Two of the four
source functions are ported here (`j_crit_cable_from_fraction`, `bmax_from_awp`), both
already tier-1. The other two are **not** ported -- see the record's open questions:

- `jcrit_from_material` is a genuine 8-way switch (`i_tf_sc_mat`) whose branches call
  into `process.models.superconductors` (`itersc`, `bi2212`, `jcrit_nbti`,
  `western_superconducting_nb3sn`, `jcrit_rebco`, `gl_nbti`, `gl_rebco`), each with a
  different reads-set (e.g. only branch 4 reads `b_crit_sc`/`t_crit_sc`, only branch 7
  reads `b_crit_upper_nbti`/`t_crit_nbti`). That module is not yet a registry unit --
  porting any branch here would be porting a formula this audit hasn't looked at yet.
- `intersect` is a generic tabulated-curve root-finder (Newton-Raphson-style, fixed
  100-iteration cap with an early-`break`) -- self-contained (no calls into other
  models) and a real tier-2 candidate, but its unknowns are whole arrays (`x1, y1, x2,
  y2`), which doesn't fit the harness's scalar-kwarg sample/fuzz machinery without more
  thought than this pass had time for. Flagged, not rushed.
"""

def j_crit_cable_from_fraction(j_crit_sc, f_tf_conductor_copper, f_he):
    """Critical current density of a cable, from its superconductor and void fractions.

    `j_crit_cable = j_crit_sc * (non-copper fraction of conductor) * (conductor
    fraction of cable)`. Already pure in the source -- no `data` access.
    """
    return j_crit_sc * (1.0 - f_tf_conductor_copper) * (1.0 - f_he)


def bmax_from_awp(wp_width_radial, current, n_tf_coils, r_coil_major, r_coil_minor,
                   stella_config_a1, stella_config_a2):
    """Fitted peak field on the TF coil winding pack, as a function of its width."""
    return (
        2e-1  # mu x 1e6, to use current in MA
        * current
        * n_tf_coils
        / (r_coil_major - r_coil_minor)
        * (stella_config_a1 + stella_config_a2 * r_coil_major / wp_width_radial)
    )


# No `cottax` node for either function yet. Both are called only from unit #9's
# (`coils/calculate.py`) internal winding-pack solve, and every one of their real call-
# site arguments (`coilcurrent`, `wp_width_r_min`, `r_coil_major`, `r_coil_minor`, and
# `j_crit_sc`/`f_tf_conductor_copper`/`f_he` for `j_crit_cable_from_fraction`, called
# from inside `jcrit_from_material`) is a *local* computed inside that solve, not an
# established `.area.field` this audit has independently verified -- wrapping either as
# an `ExplicitFunction` now would assert a wiring this pass has no basis for (see
# `schema.md`: "skip this section... while open questions about the signature itself are
# unresolved"). Correct home for both nodes is wherever unit #9 declares its own solve.
