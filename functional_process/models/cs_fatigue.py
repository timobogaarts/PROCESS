"""Pure-functional port of `process/models/cs_fatigue.py` (partial -- see "not ported").

Audit record: `functional_process/_audit/units/models/cs_fatigue.md`.

**Scope.** `surface_stress_intensity_factor` (2026-08-26) and `ncycle` (2026-08-30).
`embedded_stress_intensity_factor` is dead in `process/` (no caller outside its own
PROCESS unit test -- confirmed by `grep -rn "embedded_stress_intensity_factor"
process/`) and is not needed by `ncycle` either, so it is not ported (same "don't port
dead code" instruction the wave brief states for whole files, applied here to one
function).

**`ncycle` was the record's stop item for four days, and it is ported on the terms that
record's open question 1 settled** (DECIDED-DEFERRED, 2026-08-27): an eager
`lax.while_loop`, Tier-1 value agreement, and the gradient checks structurally excused
because `n_cycle` is a count. What changed is only the "deferred" half -- that decision
rested on *"no reader needs `n_cycle` yet (constraint 90 is not active on any tracked
input and the CS stress chain feeding it is UNPORTED)"*, and both clauses have since
stopped being true. `stresses.py` landed the CS stress chain on 2026-08-27, and
`low_aspect_ratio_DEMO.IN.DAT` activates constraint 90 -- which, reading a
`.cs_fatigue.n_cycle` no node owned, evaluated `1 - 0 / n_cycle_min` = exactly
`+1.000000` with an identically zero gradient row, and stopped both of that
configuration's SAND cells at zero iterations. A constraint that is violated by a
constant cannot be steered, so nothing else about that machine could be measured either.

**The loop terminates, and not by luck.** `k_max` is `max(k_a, k_c)`, so one of the two
ratios `(k_a / k_max)`, `(k_c / k_max)` is exactly `1` on every pass and the
corresponding crack dimension advances by the full `delta`. `a` or `c` therefore
strictly increases towards a fixed bound each iteration, which bounds the trip count at
`(bound - start) / delta` -- a few hundred passes on the tracked inputs. A non-finite
input exits immediately instead, because `nan <= x` is `False`. This matters more here
than it would in a script: `lax.while_loop` has no iteration cap, so a loop that could
stall would stall the whole graph evaluation, not just this node.

The stop item's other two candidate resolutions (a bespoke Tier-2 `residual`, or a new
tier) are not revisited -- see the record for why the tier question was decided the way
it was.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto
from jax import lax

from functional_process.models.safe_math import safe_pow, safe_sqrt
from functional_process.paths import cs_fatigue, pf_coil, physics

# ---------------------------------------------------------------------------
# `surface_stress_intensity_factor` -- ports `CsFatigue.surface_stress_intensity_factor`,
# `process/models/cs_fatigue.py:178-262`, already a `@staticmethod`, zero `self.data`
# access. The one change: the source's `if a <= c: ... else: ...` becomes `jnp.where`,
# since `a`/`c` (crack depth/length) are plain differentiable crack-size floats, not a
# switch -- `needs-lax-cond-or-where` per `traceability_policy.md`.
#
# Both branches are computed unconditionally and selected with `jnp.where`.
# **`jnp.where`'s JVP does discard the untaken branch's tangent when that branch's own
# arithmetic is what goes non-finite** (measured directly: a bare `1/c` in a discarded
# branch at `c == 0` does not leak into the selected branch's gradient). It does **not**
# protect against the *different* trap this function actually hit during development: a
# fractional power (`x ** p`, `0 < p < 1`) whose base is exactly zero has an infinite
# *local* derivative regardless of which `jnp.where` arm it sits in, and if that infinite
# derivative is multiplied by another factor that is *itself* exactly zero at the same
# point (the ordinary chain rule, not the `jnp.where` selection), the product is `inf * 0
# = nan` before `jnp.where` ever gets a chance to select anything away. `h2_le`'s
# `a_c ** 0.75` term (multiplied by `a_t_2`, both exactly `0` at `a == 0`) is exactly
# this shape, found by this unit's own `test_gradient_finite_at_zero`, not by
# inspection. Every fractional power and every bare `sqrt` in this function is
# therefore wrapped in
# `safe_pow`/`safe_sqrt` (`models/safe_math.py`) rather than case-by-case reasoning about
# which ones are actually reachable at zero -- cheap, and correct for every base per that
# module's own docstring.
# ---------------------------------------------------------------------------


def surface_stress_intensity_factor(hoop_stress, t, w, a, c, phi):
    """Surface (semi-elliptical) crack stress-intensity factor.

    Ports `CsFatigue.surface_stress_intensity_factor`,
    `process/models/cs_fatigue.py:178-262`, unchanged arithmetic, `a <= c` selected via
    `jnp.where` instead of a Python `if`.

    Every fractional power and bare square root uses `safe_pow`/`safe_sqrt`
    (`_audit/next_steps.md` §9's `x ** p`, `0 < p < 1`, trap; `models/safe_math.py`)
    rather than `jnp.sqrt`/`**` directly -- see the module comment above for which
    specific site (`h2_le`'s `a_c ** 0.75`) was the one that actually failed
    `test_gradient_finite_at_zero` before this treatment, and why the others are guarded
    too rather than argued safe case by case.

    **Also a genuine PROCESS domain gap, not a porting defect**: `sqrt(a/t) * pi * c /
    (2 * w)` (the argument of the `cos(...)` in the denominator) can exceed `pi/2` for
    `t`, `w`, `a`, `c` values that are not otherwise unreasonable, making `cos` negative
    and the enclosing `sqrt(1 / cos(...))` `nan` on **both** sides -- confirmed by
    calling `CsFatigue.surface_stress_intensity_factor` directly at such a point and
    seeing it return `nan` with no exception (`numpy.sqrt` of a negative number warns,
    it does not raise). Not something `Tier1Contract.reference_domain_errors` can flag
    (that mechanism is for PROCESS *raising*); the test file's `fuzz_bounds` instead keep
    well clear of it, the same way `plasma_geometry.md`'s D1 is worked around rather than
    fixed.
    """
    bending_stress = 0.0  # hardcoded in the source; kept for faithful parity

    a_t = a / t
    a_t_2 = a_t**2.0
    sin_phi = jnp.sin(phi)
    cos_phi_2 = jnp.cos(phi) ** 2.0

    # `a <= c` formula
    a_c = a / c
    q_le = 1.0 + 1.464 * safe_pow(a_c, 1.65)
    m1_le = 1.13 - 0.09 * a_c
    m2_le = -0.54 + 0.89 / (0.2 + a_c)
    m3_le = 0.5 - 1.0 / (0.65 + a_c) + 14.0 * safe_pow(1.0 - a_c, 24.0)
    g_le = 1.0 + (0.1 + 0.35 * a_t_2) * (1.0 - sin_phi) ** 2.0
    f_phi_le = safe_pow(a_c**2.0 * cos_phi_2 + sin_phi**2.0, 0.25)
    p_le = 0.2 + a_c + 0.6 * a_t
    h1_le = 1.0 - 0.34 * a_t - 0.11 * a * a / (c * t)
    h2_le = (
        1.0
        + (-1.22 - 0.12 * a_c) * a_t
        + (0.55 - 1.05 * safe_pow(a_c, 0.75) + 0.47 * safe_pow(a_c, 1.5)) * a_t_2
    )

    # `a > c` formula
    c_a = c / a
    c_a_4 = c_a**4.0
    q_gt = 1.0 + 1.464 * safe_pow(c_a, 1.65)
    m1_gt = safe_sqrt(c_a) * (1.0 + 0.04 * c_a)
    m2_gt = 0.2 * c_a_4
    m3_gt = -0.11 * c_a_4
    g_gt = 1.0 + (0.1 + 0.35 * c_a * a_t_2) * (1.0 - sin_phi) ** 2.0
    f_phi_gt = safe_pow(c_a**2.0 * sin_phi**2.0 + cos_phi_2, 0.25)
    p_gt = 0.2 + c_a + 0.6 * a_t
    h1_gt = (
        1.0
        + (-0.04 - 0.41 * c_a) * a_t
        + (0.55 - 1.93 * safe_pow(c_a, 0.75) + 1.38 * safe_pow(c_a, 1.5)) * a_t_2
    )
    h2_gt = (
        1.0
        + (-2.11 + 0.77 * c_a) * a_t
        + (0.55 - 0.72 * safe_pow(c_a, 0.75) + 0.14 * c_a * 1.5) * a_t_2
    )

    a_le_c = a <= c
    q = jnp.where(a_le_c, q_le, q_gt)
    m1 = jnp.where(a_le_c, m1_le, m1_gt)
    m2 = jnp.where(a_le_c, m2_le, m2_gt)
    m3 = jnp.where(a_le_c, m3_le, m3_gt)
    g = jnp.where(a_le_c, g_le, g_gt)
    f_phi = jnp.where(a_le_c, f_phi_le, f_phi_gt)
    p = jnp.where(a_le_c, p_le, p_gt)
    h1 = jnp.where(a_le_c, h1_le, h1_gt)
    h2 = jnp.where(a_le_c, h2_le, h2_gt)

    return (
        (hoop_stress + (h1 + (h2 - h1) * safe_pow(sin_phi, p)) * bending_stress)
        * (
            (m1 + m2 * a_t_2 + m3 * a_t**4.0)
            * g
            * f_phi
            * jnp.sqrt(1.0 / jnp.cos(safe_sqrt(a_t) * jnp.pi * c / (2.0 * w)))
        )
        * safe_sqrt(jnp.pi * a / q)
    )


# ---------------------------------------------------------------------------
# `surface_stress_intensity_factor` still has **no node of its own**, and that has not
# changed with `ncycle`'s arrival -- it is `ncycle`'s loop body, not a sibling. It owns
# no `VarPath` in PROCESS (`k_a`/`k_c`, `process/models/cs_fatigue.py:95-102`, are two
# evaluations at different `phi` feeding the loop, not stored fields), so a node for it
# would have to invent places no run ever writes. It is Tier-1 tested on its own
# signature instead, which is what the harness is for.
#
# `ncycle` -- ports `CsFatigue.ncycle`, `process/models/cs_fatigue.py:22-114`. The seven
# `self.data.cs_fatigue.*` coefficient reads become explicit arguments (all seven are
# `explicit-arg` in the record's data-footprint table: read once, unconditionally, for
# the whole call), and the Python `while` becomes `lax.while_loop` over the same four
# loop-carried values the source closes over.
# ---------------------------------------------------------------------------

_DELTA = 1.0e-4
"""The fixed crack-area increment `ncycle` integrates with (`cs_fatigue.py:75`).

A module constant rather than an argument because PROCESS hardcodes it and nothing
reads it: it is the discretisation, and a different `delta` is a different answer that
no input file can ask for. It is also the reason Tier-1 value agreement is the right
contract here -- for a fixed step the termination is deterministic, so PROCESS's number
is exact for this discretisation rather than an approximation to something else.
"""

_PHI = (jnp.pi / 2.0, 0.0)
"""The two crack-front angles `k_a` and `k_c` are evaluated at.

PROCESS builds these as one `np.array([np.pi/2, 0])` (`pi_2_arr`, `cs_fatigue.py:83`)
and unpacks the vectorised result. Kept as two scalar evaluations instead: the
arithmetic is identical, and `surface_stress_intensity_factor`'s `jnp.where` branches
on `a <= c`, which is the same for both angles, so nothing is shared by vectorising.
"""


def calculate_n_cycle(
    max_hoop_stress,
    residual_stress,
    t_crack_vertical,
    dz_cs_turn_conduit,
    dr_cs_turn_conduit,
    paris_coefficient,
    paris_power_law,
    walker_coefficient,
    sf_vertical_crack,
    sf_radial_crack,
    fracture_toughness,
    sf_fast_fracture,
):
    """Allowable CS load cycles, by Euler integration of crack growth.

    Ports `CsFatigue.ncycle` (`process/models/cs_fatigue.py:22-114`), arithmetic
    unchanged. Two structural changes, neither of them numerical:

    * the seven `self.data.cs_fatigue.*` coefficient reads are arguments;
    * the Python `while` is `lax.while_loop` over `(a, c, n_pulse, k_max)`, the exact
      four values the source's loop carries.

    The initial `k_max = 0.0` is PROCESS's own (`:78`), and it is what makes the first
    pass unconditional: the fracture-toughness clause of the guard cannot be false
    before any stress-intensity factor has been computed.

    Every `**` is `safe_pow`, following this module's standing policy rather than a
    site-by-site argument -- and here the policy is load-bearing rather than cheap
    insurance, because three of the four exponents (`paris_power_law` twice, and
    `-paris_power_law * (walker_coefficient - 1)`) are *data*, so nothing rules out a
    run whose exponent lands in `(0, 1)` where a zero base has an infinite derivative
    (`_audit/next_steps.md` §9). `safe_pow` is value-identical everywhere else.

    Returns `n_pulse / 2` -- "two pulses, ramp to Vsmax and ramp down per cycle"
    (`:113`). PROCESS's second return value, `t_crack_radial`, is **not returned here**:
    it is `3 * t_crack_vertical` computed before the loop and never touched by it, and
    `.cs_fatigue.t_crack_radial` is a genuine `IN.DAT` input
    (`process/core/input.py:782`) that PROCESS overwrites with a derived value for
    reporting only (`pfcoil.py:2255`, inside `output()`). A node owning it would clobber
    an input to produce something nothing reads.

    Parameters
    ----------
    max_hoop_stress :
        Peak hoop stress in the CS conduit (Pa). `.pf_coil.stress_hoop_cs_inner`.
    residual_stress :
        Residual hoop stress in the structural material (Pa).
        `.cs_fatigue.residual_sig_hoop`.
    t_crack_vertical :
        Initial vertical crack size (m). `.cs_fatigue.t_crack_vertical`.
    dz_cs_turn_conduit :
        Vertical thickness of the CS conductor conduit (m) -- the plate thickness `t`
        the stress-intensity factor is taken over. `.cs_fatigue.dz_cs_turn_conduit`.
    dr_cs_turn_conduit :
        Radial thickness of the CS conductor conduit (m) -- the plate width `w`.
        `.cs_fatigue.dr_cs_turn_conduit`.
    paris_coefficient, paris_power_law :
        Paris crack-growth-law material coefficient and exponent.
        `.cs_fatigue.paris_coefficient`, `.cs_fatigue.paris_power_law`.
    walker_coefficient :
        Walker mean-stress-correction exponent. `.cs_fatigue.walker_coefficient`.
    sf_vertical_crack, sf_radial_crack :
        Safety factors on the two crack dimensions, dividing the conduit thicknesses to
        give the two size limits. `.cs_fatigue.sf_vertical_crack`,
        `.cs_fatigue.sf_radial_crack`.
    fracture_toughness, sf_fast_fracture :
        Fracture toughness (MPa m^1/2) and its safety factor -- their ratio is the
        fast-fracture limit on `k_max`. `.cs_fatigue.fracture_toughness`,
        `.cs_fatigue.sf_fast_fracture`.

    Returns
    -------
    :
        Allowable number of cycles.
    """
    walker_exponent = -paris_power_law * (walker_coefficient - 1.0)

    # Set units to MPa.
    max_hoop_stress_mpa = max_hoop_stress / 1.0e6
    residual_stress_mpa = residual_stress / 1.0e6

    # Mean stress ratio, and the Walker-corrected Paris coefficient it gives.
    r = residual_stress_mpa / (max_hoop_stress_mpa + residual_stress_mpa)
    cr = paris_coefficient / safe_pow(1.0 - r, walker_exponent)

    a_limit = dz_cs_turn_conduit / sf_vertical_crack
    c_limit = dr_cs_turn_conduit / sf_radial_crack
    k_limit = fracture_toughness / sf_fast_fracture

    def growing(state):
        a, c, _, k_max = state
        return (a <= a_limit) & (c <= c_limit) & (k_max <= k_limit)

    def step(state):
        a, c, n_pulse, _ = state
        phi_a, phi_c = _PHI
        k_a = surface_stress_intensity_factor(
            max_hoop_stress_mpa, dz_cs_turn_conduit, dr_cs_turn_conduit, a, c, phi_a
        )
        k_c = surface_stress_intensity_factor(
            max_hoop_stress_mpa, dz_cs_turn_conduit, dr_cs_turn_conduit, a, c, phi_c
        )
        k_max = jnp.maximum(k_a, k_c)
        return (
            a + _DELTA * safe_pow(k_a / k_max, paris_power_law),
            c + _DELTA * safe_pow(k_c / k_max, paris_power_law),
            n_pulse + _DELTA / (cr * safe_pow(k_max, paris_power_law)),
            k_max,
        )

    # Initial crack size: `a` vertical, `c` radial at three times that (`:69-72`).
    start = (
        jnp.asarray(t_crack_vertical, float),
        jnp.asarray(3.0 * t_crack_vertical, float),
        jnp.zeros_like(jnp.asarray(t_crack_vertical, float)),
        jnp.zeros_like(jnp.asarray(t_crack_vertical, float)),
    )
    _, _, n_pulse, _ = lax.while_loop(growing, step, start)
    return n_pulse / 2.0


class CsFatigue(ExplicitFunction):
    """cottax node: `.tokamak.cs_fatigue`. Owns `.cs_fatigue.n_cycle`, constraint 90's
    operand.

    **The `f_c_plasma_inductive` guard is this node's, not `calculate_n_cycle`'s.**
    PROCESS calls `ncycle` only when `.physics.f_c_plasma_inductive > 0.0e-4`
    (`pfcoil.py:3488`, "this is only valid for pulsed reactor design"), and leaves
    `.cs_fatigue.n_cycle` at its entering value otherwise. That guard lives at the call
    site, in `pfcoil.py`, so it belongs to the binding rather than to the ported
    function -- and it is a *computed* condition, not a switch, so `machine_from_indat`
    cannot resolve it the way it resolves `conditional-ownership-by-run-config` (a node
    either owns this field or does not, and which it is cannot be known until the
    physics has run). It is therefore a `jnp.where`, with `0.0` as the ungated value:
    `.cs_fatigue.n_cycle` is an output field with dataclass default `0.0`
    (`cs_fatigue_variables.py:15`) and no `IN.DAT` entry, so `0.0` *is* what PROCESS
    leaves there, exactly, rather than a stand-in for it. `t_crack_radial` could not be
    treated this way -- see `calculate_n_cycle`'s docstring for why it is not owned at
    all.
    """

    n_cycle = OutputInto(cs_fatigue)

    def __call__(
        self,
        stress_hoop_cs_inner=From(pf_coil),
        residual_sig_hoop=From(cs_fatigue),
        t_crack_vertical=From(cs_fatigue),
        dz_cs_turn_conduit=From(cs_fatigue),
        dr_cs_turn_conduit=From(cs_fatigue),
        paris_coefficient=From(cs_fatigue),
        paris_power_law=From(cs_fatigue),
        walker_coefficient=From(cs_fatigue),
        sf_vertical_crack=From(cs_fatigue),
        sf_radial_crack=From(cs_fatigue),
        fracture_toughness=From(cs_fatigue),
        sf_fast_fracture=From(cs_fatigue),
        f_c_plasma_inductive=From(physics),
    ):
        return jnp.where(
            f_c_plasma_inductive > 0.0,
            calculate_n_cycle(
                stress_hoop_cs_inner,
                residual_sig_hoop,
                t_crack_vertical,
                dz_cs_turn_conduit,
                dr_cs_turn_conduit,
                paris_coefficient,
                paris_power_law,
                walker_coefficient,
                sf_vertical_crack,
                sf_radial_crack,
                fracture_toughness,
                sf_fast_fracture,
            ),
            0.0,
        )
