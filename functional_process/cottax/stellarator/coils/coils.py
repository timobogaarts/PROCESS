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
from functional_process.paths import (
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
    `ImplementedFunction` this method becomes only *reads* `.stellarator.wp_width_r_min` (the
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
