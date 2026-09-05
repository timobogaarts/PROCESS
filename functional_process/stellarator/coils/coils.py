"""Pure physics functions extracted from
`functional_process.models.stellarator.coils.coils`, per `_audit/formulas_split.md`
step 2 phase A. The graph declarations that wrap these stay in that sibling
module.
"""

import jax
import jax.numpy as jnp
import optimistix as optx

from functional_process.models.physics.superconductors import (
    bi2212,
    gl_nbti,
    gl_rebco,
    itersc,
    jcrit_nbti,
    jcrit_rebco,
    western_superconducting_nb3sn,
)


def j_crit_cable_from_fraction(j_crit_sc, f_tf_conductor_copper, f_he):
    """Critical current density of a cable, from its superconductor and void fractions.

    `j_crit_cable = j_crit_sc * (non-copper fraction of conductor) * (conductor
    fraction of cable)`. Already pure in the source -- no `data` access.
    """
    return j_crit_sc * (1.0 - f_tf_conductor_copper) * (1.0 - f_he)


def bmax_from_awp(
    wp_width_radial,
    current,
    n_tf_coils,
    r_coil_major,
    r_coil_minor,
    stella_config_a1,
    stella_config_a2,
):
    """Fitted peak field on the TF coil winding pack, as a function of its width."""
    return (
        2e-1  # mu x 1e6, to use current in MA
        * current
        * n_tf_coils
        / (r_coil_major - r_coil_minor)
        * (stella_config_a1 + stella_config_a2 * r_coil_major / wp_width_radial)
    )


# No graph node for either function yet. Both are called only from unit #9's
# (`coils/calculate.py`) internal winding-pack solve, and every one of their real call-
# site arguments (`coilcurrent`, `wp_width_r_min`, `r_coil_major`, `r_coil_minor`, and
# `j_crit_sc`/`f_tf_conductor_copper`/`f_he` for `j_crit_cable_from_fraction`, called
# from inside `jcrit_from_material`) is a *local* computed inside that solve, not an
# established `.area.field` this audit has independently verified -- wrapping either as
# an `ExplicitFunction` now would assert a wiring this pass has no basis for (see
# `schema.md`: "skip this section... while open questions about the signature itself are
# unresolved"). Correct home for both nodes is wherever unit #9 declares its own solve.


# `jcrit_from_material`'s 8-way `i_tf_sc_mat` dispatch, split one pure function per
# branch (`coils.md`: reads-sets genuinely differ, no shared body to speak of -- the
# opposite shape from a formula-only switch, so split-by-default applies cleanly). Every
# branch computes `j_crit_sc` (A/m2, in the superconductor) and returns it scaled by
# `1e-6` (to MA/m2) -- the one thing every source branch shares, applied once at the very
# end of the source's `if`/`elif` chain, not duplicated per branch there either. `strain
# = -0.005` ("for now a small value", source comment) is likewise common to every branch
# that calls a strain-dependent material model (all but `bi2212`/`jcrit_rebco`, which take
# no strain argument at all).
#
# Each source branch also computes a `j_crit_cable = j_crit_cable_from_fraction(...)`
# local that is never read again -- `jcrit_from_material` returns only
# `j_crit_sc * 1e-6`, confirmed by reading the source function to its final `return`
# (`process/models/stellarator/coils/coils.py:162`). Dead in the original, so not ported
# here either -- same treatment `calculate.py`'s `_critical_current_density_by_material`
# (the local stand-in this replaces, see the module docstring above and `coils.md`) already
# gave it.


def jcrit_from_material_iter_nb3sn(t_helium, b_max):
    """Critical current density, ITER Nb3Sn critical-surface model (`i_tf_sc_mat == 1`).

    `bc20m = 32.97`, `tc0m = 16.06` are fixed literals in the source ("these are values
    taken from sctfcoil.f90"). The source's `if b_max > bc20m: j_crit_sc = 1e-9 else:
    itersc(...)` is a data-dependent branch on a continuous input, not a switch --
    `jnp.where`, per `coils.md`'s JAX-difficulty flags -- followed by an unconditional
    `max(1e-9, j_crit_sc)` floor either way, also kept.

    Returns
    -------
    :
        `j_crit_sc`, MA/m2 (the `jcrit_from_material` return value, `j_crit_sc * 1e-6`).
    """
    bc20m, tc0m = 32.97, 16.06
    strain = -0.005
    j_crit_sc, _b_critical, _temp_critical = itersc(t_helium, b_max, strain, bc20m, tc0m)
    j_crit_sc = jnp.where(b_max > bc20m, 1.0e-9, j_crit_sc)
    j_crit_sc = jnp.maximum(1.0e-9, j_crit_sc)
    return j_crit_sc * 1.0e-6


def jcrit_from_material_bi2212(
    t_helium,
    b_max,
    j_tf_wp,
    f_a_tf_turn_cable_space_extra_void,
    fhts,
    f_a_tf_turn_cable_copper,
):
    """Critical current density, Bi-2212 HTS model (`i_tf_sc_mat == 2`).

    The only branch that does not compute `j_crit_sc` directly from a material model:
    `bi2212` returns a *cable* (strand) current density, which the source backs out to
    `j_crit_sc` by dividing out the copper fraction. `jstrand` is built from `j_wp`
    (source parameter name; the real call site passes `data.tfcoil.j_tf_wp`, the *stale*
    prior-call value -- see `calculate.py`'s `WindingPackJTfWp` docstring for why that
    self-reference needed its own `FixedPointFunction`, out of this node's scope since it
    only *reads* `j_tf_wp`, never owns it). No `1e-9` floor in this branch (source has
    none here, unlike branches 1/3/6).

    Returns
    -------
    :
        `j_crit_sc`, MA/m2.
    """
    f_he = f_a_tf_turn_cable_space_extra_void
    jstrand = j_tf_wp / (1.0 - f_he)
    j_crit_cable, _t_margin = bi2212(b_max, jstrand, t_helium, fhts)
    j_crit_sc = j_crit_cable / (1.0 - f_a_tf_turn_cable_copper)
    return j_crit_sc * 1.0e-6


def jcrit_from_material_nbti_lubell(t_helium, b_max):
    """Critical current density, NbTi (Lubell scaling) model (`i_tf_sc_mat == 3`).

    `bc20m = 15.0`, `tc0m = 9.3`, `c0 = 1.0` fixed literals in the source. Same
    `b_max > bc20m` guard and `1e-9` floor as `jcrit_from_material_iter_nb3sn` (branch 1)
    -- the source repeats this pattern verbatim for both `itersc`-family branches that
    use literal `bc20m`.

    Returns
    -------
    :
        `j_crit_sc`, MA/m2.
    """
    bc20m, tc0m, c0 = 15.0, 9.3, 1.0
    j_crit_sc, _t_critical = jcrit_nbti(t_helium, b_max, c0, bc20m, tc0m)
    j_crit_sc = jnp.where(b_max > bc20m, 1.0e-9, j_crit_sc)
    j_crit_sc = jnp.maximum(1.0e-9, j_crit_sc)
    return j_crit_sc * 1.0e-6


def jcrit_from_material_iter_nb3sn_user_defined(t_helium, b_max, bcritsc, tcritsc):
    """Critical current density, ITER Nb3Sn with user-defined `bc20m`/`tc0m`
    (`i_tf_sc_mat == 4`).

    As branch 1, but `bc20m`/`tc0m` come from `.tfcoil.bcritsc`/`.tfcoil.tcritsc`
    (source parameter names `b_crit_sc`/`t_crit_sc`) instead of literals -- the one
    branch this audit's `1e-9` floor/`b_max > bc20m` guard does **not** apply to (the
    source has neither for this branch, confirmed by re-reading -- only branches 1, 3, 6
    clamp).

    Returns
    -------
    :
        `j_crit_sc`, MA/m2.
    """
    strain = -0.005
    j_crit_sc, _b_critical, _temp_critical = itersc(
        t_helium, b_max, strain, bcritsc, tcritsc
    )
    return j_crit_sc * 1.0e-6


def jcrit_from_material_wst_nb3sn(t_helium, b_max):
    """Critical current density, WST Nb3Sn model (`i_tf_sc_mat == 5`).

    `bc20m = 32.97`, `tc0m = 16.06` -- same literals as branch 1, different material
    model (`western_superconducting_nb3sn`, not `itersc`). No `b_max > bc20m` guard and
    no `1e-9` floor in this branch (source has neither here).

    Returns
    -------
    :
        `j_crit_sc`, MA/m2.
    """
    bc20m, tc0m = 32.97, 16.06
    strain = -0.005
    j_crit_sc, _b_critical, _t_critical = western_superconducting_nb3sn(
        t_helium, b_max, strain, bc20m, tc0m
    )
    return j_crit_sc * 1.0e-6


def jcrit_from_material_rebco(t_helium, b_max):
    """Critical current density, REBCO 2nd-generation HTS model (`i_tf_sc_mat == 6`).

    **The source's real call site looks broken**: `coils.py:136` calls
    `superconductors.jcrit_rebco(t_helium, b_max, 0)` -- three positional arguments --
    but `jcrit_rebco`'s signature (`process/models/superconductors.py:167`, unchanged in
    the port, `functional_process/models/physics/superconductors.py`) takes exactly two
    (`temp_conductor`, `b_conductor`). Executing this branch as written would raise
    `TypeError`, confirmed by `superconductors.md`'s open question 1 (found independently
    while auditing `superconductors.py`, not reproduced here). This function calls
    `jcrit_rebco`'s real 2-argument signature, matching the treatment
    `calculate.py`'s `_critical_current_density_by_material` already gave this same
    branch (see that function's own docstring) -- not a faithful reproduction of a
    call that cannot execute, since there is nothing faithful to reproduce.

    Returns
    -------
    :
        `j_crit_sc`, MA/m2.
    """
    j_crit_sc, _validity, _b_c20max, _temp_c0max = jcrit_rebco(t_helium, b_max)
    j_crit_sc = jnp.maximum(1.0e-9, j_crit_sc)
    return j_crit_sc * 1.0e-6


def jcrit_from_material_gl_nbti(t_helium, b_max, b_crit_upper_nbti, t_crit_nbti):
    """Critical current density, Durham Ginzburg-Landau Nb-Ti model
    (`i_tf_sc_mat == 7`).

    `bc20m`/`tc0m` come from `.tfcoil.b_crit_upper_nbti`/`.tfcoil.t_crit_nbti` (source
    parameter names match the port's field names exactly, no renaming) -- the other
    branch (with branch 4) whose bounds are a genuine `data` read. No `1e-9` floor.

    Returns
    -------
    :
        `j_crit_sc`, MA/m2.
    """
    strain = -0.005
    j_crit_sc, _b_critical, _t_critical = gl_nbti(
        t_helium, b_max, strain, b_crit_upper_nbti, t_crit_nbti
    )
    return j_crit_sc * 1.0e-6


def jcrit_from_material_gl_rebco(t_helium, b_max):
    """Critical current density, Durham Ginzburg-Landau REBCO model
    (`i_tf_sc_mat == 8`).

    `bc20m = 429`, `tc0m = 185` fixed literals ("A0 calculated for tape cross section
    already", source comment). No `1e-9` floor.

    Returns
    -------
    :
        `j_crit_sc`, MA/m2.
    """
    bc20m, tc0m = 429.0, 185.0
    strain = -0.005
    j_crit_sc, _b_critical, _temp_critical = gl_rebco(
        t_helium, b_max, strain, bc20m, tc0m
    )
    return j_crit_sc * 1.0e-6


def intersect_residual(x, x1, y1, x2, y2):
    """What vanishes at the intersection of two tabulated `(x, y)` curves.

    `residual(x) = y1_interp(x) - y2_interp(x)`, using the same piecewise-linear
    interpolation PROCESS's own `intersect` uses (`np.interp` there, `jnp.interp` here
    -- both linear, both defined identically off the same tabulated points). This *is*
    the defining equation `intersect` below solves, and what `TestIntersect.residual` in
    `test_coils.py` plugs both PROCESS's answer and the port's answer back into, per
    `_audit/test_harness.md`'s tier-2 pass criterion.
    """
    return jnp.interp(x, x1, y1) - jnp.interp(x, x2, y2)


def _intersect_newton_polish(x, x1, y1, x2, y2, steps=8):
    """A handful of exact Newton corrections once `x` is known to be near the root.

    Not part of the audited interface -- an internal refinement step folded into
    `intersect` below. `jnp.interp` is *exactly* piecewise-linear, so once `x` sits
    inside the correct linear segment a Newton step is not merely a local
    approximation, it lands on that segment's algebraic solution up to float64
    round-off -- the same reason PROCESS's own secant scheme sometimes reaches the
    machine-precision floor in a single step (`test_coils.py`'s `n=25`/`n=200` samples).
    A fixed, small step count (rather than a convergence check) is fine here: each step
    past the first is a no-op once round-off is reached, and `lax.scan` keeps this
    traceable.
    """

    def _step(x, _):
        f = intersect_residual(x, x1, y1, x2, y2)
        df = jax.grad(intersect_residual)(x, x1, y1, x2, y2)
        return x - jnp.where(df != 0.0, f / df, 0.0), None

    x, _ = jax.lax.scan(_step, x, None, length=steps)
    return x


def intersect(x1, y1, x2, y2, xin):
    """Find the x (abscissa) at which two tabulated `(x, y)` curves cross.

    Pure-functional replacement for PROCESS's `intersect`
    (`process/models/stellarator/coils/coils.py`): a fixed 100-iteration
    finite-difference Newton-Raphson loop, with an early `break` on
    `abs(residual) < epsy` and an ad hoc clamp-and-bail if `x` leaves `[xmin, xmax]` (see
    `coils.md`'s JAX-difficulty flags -- that control flow has no faithful
    `jax`-traceable translation, since it needs a data-dependent early exit). This port
    keeps PROCESS's defining equation (`intersect_residual`, same interpolation) but
    drives it with a real, convergence-checked root-find instead: bisection over
    `[xmin, xmax]` (the two curves' overlapping x-range -- always a valid bracket
    whenever the curves actually cross there, since the residual then changes sign
    across it) down to a coarse tolerance, followed by a few exact Newton corrections
    (`_intersect_newton_polish`) once that bracket has localised `x` to the correct
    linear segment. `xin` is clamped into `[xmin, xmax]` first, matching PROCESS's own
    guess-clamping, but only ever seeds the bisection's *starting point* -- the bracket
    itself is the full overlap, not a window around `xin`, so (unlike PROCESS's local
    Newton-Raphson) a bad guess cannot walk the solve towards the wrong crossing or off
    the domain entirely.

    Does not raise or log on non-convergence or a non-overlapping domain -- a traced
    function cannot -- `optimistix`'s own `result` field on the intermediate solution is
    the equivalent of what PROCESS reports through `logger.error`.
    """
    xmin = jnp.maximum(jnp.min(x1), jnp.min(x2))
    xmax = jnp.minimum(jnp.max(x1), jnp.max(x2))
    x0 = jnp.clip(xin, xmin, xmax)

    bracketed = optx.root_find(
        lambda x, _: intersect_residual(x, x1, y1, x2, y2),
        optx.Bisection(rtol=0.0, atol=1e-10, flip="detect"),
        x0,
        options={"lower": xmin, "upper": xmax},
        throw=False,
        max_steps=100,
    ).value

    return _intersect_newton_polish(bracketed, x1, y1, x2, y2)
