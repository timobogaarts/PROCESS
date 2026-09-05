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
because `n_cycle` is a count. (The loop is a **masked `lax.scan`** since 2026-09-02 --
see `_MAX_CRACK_STEPS`; the tier and the value contract are unchanged.) What changed is only the "deferred" half -- that decision
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
input exits immediately instead, because `nan <= x` is `False`.

**That argument is now load-bearing rather than reassuring**: it is what licenses the
static bound in `_MAX_CRACK_STEPS`, which turned this from a `lax.while_loop` into a
masked `lax.scan` so that the whole graph can be differentiated in reverse mode at all
(`_audit/optimise_design.md` §31.9, §31.12). It also removes the hazard the original
form carried -- `lax.while_loop` has no iteration cap, so a loop that could stall would
stall the whole graph evaluation and not just this node.

The stop item's other two candidate resolutions (a bespoke Tier-2 `residual`, or a new
tier) are not revisited -- see the record for why the tier question was decided the way
it was.
"""

import jax.numpy as jnp
from jax import lax

from functional_process.models.safe_math import safe_pow, safe_sqrt

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

_MAX_CRACK_STEPS = 512
"""Static trip bound for the crack-growth integration -- what makes it a `scan`.

**Why a `scan` at all.** `lax.while_loop` has no reverse-mode rule -- an unbounded trip
count cannot be taped -- and this loop sits in the differentiated path via the condition
map, so it made `jax.jacrev`, and even `jax.grad` of a single condition, fail outright
on the whole graph (`_audit/optimise_design.md` §31.9). A `scan` transposes, storing
`_MAX_CRACK_STEPS` copies of a four-scalar carry, which is nothing. The value is
unchanged for the same discretisation: the body advances iff the original `while`
guard holds and is a masked no-op afterwards, which is that loop's semantics exactly.

**Why a bound exists at all** is this module's docstring's own termination argument,
restated as a number: `k_max` is `max(k_a, k_c)`, so one of the two ratios is exactly
`1` on every pass and the corresponding crack dimension advances by the full `_DELTA`.
The trip count is at most `max(a_limit - a_0, c_limit - c_0) / _DELTA`, i.e.
`max(dz_cs_turn_conduit / sf_vertical_crack, dr_cs_turn_conduit / sf_radial_crack) /
1e-4`. **`512` therefore covers any conduit up to `0.051 m` at a safety factor of `1`,
or `0.102 m` at `2`** -- PROCESS's own defaults are `0.022`/`0.07` at `2.0`.

**Why exactly this bound, and what it costs.** Measured (2026-09-02, §31.12): **115**
trips on PROCESS's defaults, *invariant* across `max_hoop_stress` from 100 to 800 MPa --
`surface_stress_intensity_factor` is linear in stress, so the `k_a/k_max` ratio that
sets the step cancels it, even though `n_cycle` itself moves over 200x. Worst case over
108 combinations spanning `dz_cs_turn_conduit` 0.011--0.044, `dr_cs_turn_conduit`
0.035--0.14, `t_crack_vertical` 0.3--2.0 mm and `paris_power_law` 2.5--4.5: **264**. So
512 is a little under 2x the worst case seen, and the masked no-ops are the price:
0.67 ms per call against 2.67 ms at 2048 and ~0.15 ms for the `while`. At a few hundred
condition-map evaluations per solve that difference is seconds, which is why the bound
is not simply set enormous. `lax.scan` is a rolled loop, so the *program* does not grow
with it either way -- only the runtime does.

**Exhausting it is made loud, not silent.** A truncated integration would otherwise
return a plausible, too-small cycle count. `calculate_n_cycle` checks the guard once
more at the end and returns `nan` if the crack was still growing, so a geometry beyond
this bound fails visibly and this constant is the one thing to raise.
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
    # `lax.scan` over a static bound rather than `lax.while_loop`, masked by the same
    # guard: see `_MAX_CRACK_STEPS` for the bound's derivation, its measured headroom,
    # and why reverse-mode AD is what forces the change. `step` is evaluated on every
    # pass and its result discarded once `growing` is false, which is exactly the
    # `while`'s semantics and is why the value does not move.

    def masked_step(state, _):
        advanced = step(state)
        keep = growing(state)
        return tuple(
            jnp.where(keep, new, old) for old, new in zip(state, advanced, strict=True)
        ), None

    final, _ = lax.scan(masked_step, start, None, length=_MAX_CRACK_STEPS)
    # Still growing after `_MAX_CRACK_STEPS`? Then the bound truncated the integration
    # and `n_pulse` is an underestimate that would otherwise look entirely plausible.
    # `nan` instead, per `_MAX_CRACK_STEPS`' last paragraph -- this cannot fire on any
    # geometry in that constant's measured sweep, and if it ever does, that constant is
    # what to raise.
    _, _, n_pulse, _ = final
    return jnp.where(growing(final), jnp.nan, n_pulse / 2.0)
