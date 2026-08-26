"""Pure-functional port of `process/models/stellarator/coils/coils.py` (registry #10).

Audit record: `functional_process/_audit/units/models/stellarator/coils/coils.md`. All
four source functions are now ported: `j_crit_cable_from_fraction`/`bmax_from_awp`
(tier-1), `intersect` (tier-2), and `jcrit_from_material` -- a genuine 8-way switch
(`i_tf_sc_mat`) whose branches each call one already-ported material model from
`functional_process.models.physics.superconductors` (`itersc`, `bi2212`, `jcrit_nbti`,
`western_superconducting_nb3sn`, `jcrit_rebco`, `gl_nbti`, `gl_rebco`), with genuinely
different reads-sets per branch (only branch 4 reads `bcritsc`/`tcritsc`, only branch 7
reads `b_crit_upper_nbti`/`t_crit_nbti`, branches 1/3/5/8 use fixed literals, branches
2/6 use neither) -- per `traceability_policy.md`'s split-by-default, ported as **8
separate pure functions and 8 separate `ExplicitFunction` nodes**, one per `i_tf_sc_mat`
value, all minting the same output `VarPath` (`.tfcoil.j_crit_sc`) so a later
consolidation pass can assemble them into one `Switch`/`Alternative` group in
`total_process.py` -- not done here, see `coils.md`."""

import jax
import jax.numpy as jnp
import optimistix as optx
from cottax.evaluate import AbstractDriver, ConditionMap
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ImplicitFunction,
    OutputInto,
    resolve,
)
from cottax.problem import RootFind, Start
from cottax.spec import VarPath

from functional_process.models.physics.superconductors import (
    bi2212,
    gl_nbti,
    gl_rebco,
    itersc,
    jcrit_nbti,
    jcrit_rebco,
    western_superconducting_nb3sn,
)
from functional_process.paths import stellarator, tfcoil


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


# No `cottax` node for either function yet. Both are called only from unit #9's
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


class JcritIterNb3sn(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 1` (ITER Nb3Sn).

    `.tfcoil.j_crit_sc` (this node's `Output`) and `.tfcoil.t_helium`/`.tfcoil.b_max`
    (its `From`s) are **minted**, not established PROCESS `data` fields -- `j_crit_sc`
    is a local of `jcrit_from_material` itself (never stored to `data` anywhere PROCESS
    calls it, confirmed at its one real call site, `winding_pack_total_size`), and
    `t_helium`/`b_max` are themselves locals of that same call site's 200-point sampling
    loop (`t_helium = data.tfcoil.tftmp + data.tfcoil.tmargmin`, `b_max = b_max_k[k]` from
    `bmax_from_awp`), not yet-wired `.area.field` reads this audit can verify -- same
    minting precedent as `calculate.py`'s `CoilCurrent.coilcurrent`. See `coils.md`'s
    "cottax node" section for the wiring this leaves for a later consolidation pass.
    """

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_iter_nb3sn(t_helium, b_max)


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


class JcritBi2212(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 2` (Bi-2212).

    `j_tf_wp`, `f_a_tf_turn_cable_space_extra_void`, `fhts`, `f_a_tf_turn_cable_copper`
    are real, established `.tfcoil.*` fields -- already read as `From`s by
    `calculate.py`'s `WindingPackTotalSize`/`WindingPackJTfWp` nodes under these exact
    names, so this node's `From`s line up with those, not a fresh minting. Only
    `t_helium`/`b_max` (and the shared `.tfcoil.j_crit_sc` output) are minted -- see
    `JcritIterNb3sn`'s docstring.
    """

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


class JcritNbtiLubell(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 3` (NbTi, Lubell scaling)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_nbti_lubell(t_helium, b_max)


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


class JcritIterNb3snUserDefined(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 4` (ITER Nb3Sn, user-defined).

    `bcritsc`/`tcritsc` are real, established `.tfcoil.*` fields -- the one branch (along
    with branch 7) whose material-model bounds are a genuine `data` read rather than a
    literal, per `coils.md`'s/`superconductors.md`'s reads-set table.
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


class JcritWstNb3sn(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 5` (WST Nb3Sn)."""

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_wst_nb3sn(t_helium, b_max)


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


class JcritRebco(ExplicitFunction):
    """cottax node: `jcrit_from_material`, `i_tf_sc_mat == 6` (REBCO, CroCo strand).

    See `jcrit_from_material_rebco`'s docstring for the source call site's own bug
    (extra positional argument) this node does not reproduce.
    """

    j_crit_sc = OutputInto(tfcoil)

    def __call__(
        self,
        t_helium=From(tfcoil),
        b_max=From(tfcoil),
    ):
        return jcrit_from_material_rebco(t_helium, b_max)


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


class Intersect(ImplicitFunction):
    """cottax node: `intersect`, as a genuine `ImplicitFunction`/`RootFind` pair.

    `coils.md`'s own earlier sketch of this shape is realised here almost unchanged
    (that record's "cottax node" section carried the draft this class finalizes) --
    it, not a fresh invention, decided the `VarPath`s below: `.stellarator.wp_width_r`
    (used as both `x1` and `x2`, matching the real call site's own `intersect(wp_width_r,
    lhs, wp_width_r, rhs, ...)`), `.stellarator.lhs`, `.stellarator.rhs` (the two curves
    `winding_pack_total_size`, `coils/calculate.py`, samples), and
    `.stellarator.wp_width_r_min` (the crossing point itself, this node's one declared
    `Output`/unknown). None of these four have a PROCESS storage location -- all are
    minted, per `coilcurrent`'s precedent (`calculate.py`'s `CoilCurrent`), exactly as
    `coils.md` already flagged: real call-site arguments, not yet-wired `data` fields.

    `residual` reads the unknown itself back (`wp_width_r_min`, the same `VarPath` its
    own `Output` declares) alongside the two curves -- this is not a self-loop: the
    `CallableNode` this method becomes only *reads* `.stellarator.wp_width_r_min` (the
    current guess) and *writes* `^cond.stellarator.wp_width_r_min` (the residual); a
    separate, bodyless `RootFind` problem node (minted at `^problem.Intersect`) is what
    actually owns the real `.stellarator.wp_width_r_min` -- same shape as
    `test_interfaces_pytree_namespace.py`'s `Disc1` example (`y1` read back as both a
    curve point and the unknown).

    **`xin` has no port here, confirmed, not merely asserted** (`coils.md`'s open
    question 2, `_audit/next_steps.md`'s §5 "Cut" discussion): a `RootFind`'s starting
    guess comes from whatever `Drive`s the block (`evaluate.py`'s `Drive.__call__`:
    `guess = env[unknowns] if started else None`, handed to the driver positionally),
    never from an `In` on `residual` itself -- PROCESS's `xin` argument simply does not
    survive into this node's declared interface. It still lives on in the plain
    `intersect(x1, y1, x2, y2, xin)` function above, which any concrete driver (see
    `IntersectBisectionNewtonPolish` below) is free to call directly, seeding its own
    `xin` however it likes.

    **Why declare this at all, given `intersect` already works eagerly** (see
    `_audit/next_steps.md` §7): nothing else in the graph needs `intersect`'s internal
    unknowns visible -- §7's own test for that says no, unchanged by this class. The
    reason to declare it anyway is different: as a plain function, `intersect`'s solver
    algorithm (bisection bracket + Newton polish) is baked into the leaf itself, with no
    way to ask the graph "how is this block solved" or swap the answer. As an
    `ImplicitFunction`/`RootFind` pair, the algorithm becomes a `Drive`'s `driver`
    argument -- a first-class, inspectable, replaceable choice, structurally separate
    from *what* must vanish (`residual`) -- without committing this declaration itself to
    any one choice: undriven, this pair is a perfectly valid, if unproducing, `RootFind`
    problem sitting in the graph (`Graph.declared`), same as any other Shape A/B problem
    node recorded in `_audit/next_steps.md` §5.
    """

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

    **Test-only.** This is what `_audit/next_steps.md` §7's reframing (see `Intersect`'s
    own docstring) is actually buying: a concrete driver like this one lets a test build
    a real `Drive` and get a real converged number, while the *structural* declaration
    (`Intersect` itself) commits to no particular algorithm and stays swappable -- this
    class is one legitimate answer among others, not registered anywhere as *the*
    answer. See `test_coils.py` for the `Drive`/`schedule_for` construction that uses it.

    `conditions: ConditionMap` exposes the block's closed-over external inputs as
    `conditions.context` (a `VarPath`-keyed mapping) -- this driver reads
    `.stellarator.wp_width_r`/`.lhs`/`.rhs` directly out of it rather than treating the
    residual as a fully opaque `f: unknown -> residual` the way a generic root-finder
    would have to. That is what lets it call `intersect` itself (which needs the whole
    tabulated curve arrays, not just a residual evaluator at a point) instead of
    reimplementing bisection generically -- a deliberate choice, not a limitation of
    `AbstractDriver`'s interface: nothing stops a *different* concrete driver from being
    fully generic and calling `conditions(x)` instead.

    The `Start` port seeds `intersect`'s own `xin` (`start[0]`, since `Intersect` has
    exactly one unknown). `requires` names it, so a graph driven by this must have been
    through `Initialise` -- seeding the unknown's own name no longer reaches a driver,
    and on these curves that is not cosmetic: several `_crossing_curve_case` samples
    have more than one genuine in-domain crossing, so which root bisection lands on
    depends on where it starts.

    The median fallback below survives for a **direct** call with no `Start` data (the
    driver is a plain callable and the tests use it that way): `intersect`'s own
    domain-clamping makes any point a safe starting `xin`.
    """

    drives = RootFind
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
