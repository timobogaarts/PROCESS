"""Pure-functional port of `process/models/stellarator/coils/coils.py` (registry #10).

Audit record: `functional_process/models/stellarator/coils/coils.md`. Three of the four
source functions are ported here (`j_crit_cable_from_fraction`, `bmax_from_awp`, tier-1;
`intersect`, tier-2). `jcrit_from_material` is **not** ported -- see the record:

- `jcrit_from_material` is a genuine 8-way switch (`i_tf_sc_mat`) whose branches call
  into `process.models.superconductors` (`itersc`, `bi2212`, `jcrit_nbti`,
  `western_superconducting_nb3sn`, `jcrit_rebco`, `gl_nbti`, `gl_rebco`), each with a
  different reads-set (e.g. only branch 4 reads `b_crit_sc`/`t_crit_sc`, only branch 7
  reads `b_crit_upper_nbti`/`t_crit_nbti`). That module is not yet a registry unit --
  porting any branch here would be porting a formula this audit hasn't looked at yet.
"""

import jax
import jax.numpy as jnp
import optimistix as optx


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


# No `cottax` node for `intersect` either, and for the identical reason as the two
# functions above: its only call site (`winding_pack_total_size`'s
# `intersect(wp_width_r, lhs, wp_width_r, rhs, wp_width_r_min)`, `coils/calculate.py`) is
# entirely PROCESS locals from that unported unit's own solve loop, not established
# `.area.field` paths -- there is nothing here to declare `Input`s against yet. Once
# unit #9 mints real `VarPath`s for `wp_width_r`/`lhs`/`rhs`/`wp_width_r_min` (the same
# open question `calculate.md` already raises for `coilcurrent`), the natural
# declaration is a pytree-namespace `ImplicitFunction`:
#
#   class Intersect(ImplicitFunction):
#       wp_width_r_min = Output(lambda s: s.stellarator.wp_width_r_min)
#
#       def residual(
#           self,
#           x1=Input(lambda s: s.stellarator.wp_width_r),
#           y1=Input(lambda s: s.stellarator.lhs),
#           x2=Input(lambda s: s.stellarator.wp_width_r),
#           y2=Input(lambda s: s.stellarator.rhs),
#       ):
#           return intersect_residual(self.owns[0], x1, y1, x2, y2)  # sketch only
#
# -- paired automatically with a `RootFind` problem over `^cond.wp_width_r_min`. `xin`
# has no place in that shape at all: a `RootFind`'s starting guess is supplied by
# whatever drives the block (`Drive.__call__`'s `guess`), not declared as an `In` on the
# residual itself, so PROCESS's `xin` argument would simply not survive the port as a
# port -- worth flagging explicitly since it is exactly the kind of "this argument
# quietly disappears" case `_audit/naming_convention.md` doesn't yet have a category for.
