"""Pure-functional port of `process/models/cs_fatigue.py` (partial -- see "not ported").

Audit record: `functional_process/_audit/units/models/cs_fatigue.md`. Read it first --
especially "the stop item" (`CsFatigue.ncycle`'s `while` loop, deliberately not ported in
this pass) before assuming this file's closure is complete.

**Scope of this pass.** Only `surface_stress_intensity_factor`: a pure, loop-free
`@staticmethod` with no `self.data` access, needed as a building block for `ncycle`
(the audit record's stop item) but portable on its own today.
`embedded_stress_intensity_factor` is dead in `process/` (no caller outside its own
PROCESS unit test -- confirmed by `grep -rn "embedded_stress_intensity_factor"
process/`) and is not needed by `ncycle` either, so it is not ported (same "don't port
dead code" instruction the wave brief states for whole files, applied here to one
function).

`ncycle` itself -- the file's only entry on `tokamak_call_surface.md`'s reached-function
count -- is **not ported**. It is a hand-rolled Euler integration of crack growth with a
Python `while` loop whose trip count is data-dependent (stops when any of three physical
thresholds is crossed), closing over state (`a`, `c`, `n_pulse`, `k_max`) local to the
call. This does not fit either existing tier cleanly: `_audit/test_harness.md`'s Tier 1
excludes "internal iteration" by definition, and Tier 2's residual-based pass criterion
(`Tier2Contract`) is built for a solver whose PROCESS answer may not be converged ground
truth -- here PROCESS's answer *is* exact for the discretisation it uses (`delta=1e-4`
fixed step), so there is no "PROCESS didn't converge" story, and no natural "defining
equation" whose residual characterises the stopping point the way a root-find's does.
Per the wave brief's escape valve ("a switch shape the conventions don't cover -- stop
on that item ... report it"), this is reported rather than force-fit into a tier the
harness doesn't intend for this shape. See the audit record for the full reasoning and
the question left for the orchestrator.
"""

import jax.numpy as jnp

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
# No cottax node yet -- per `schema.md`'s "skip this section ... while open questions
# about the signature itself are unresolved". `surface_stress_intensity_factor` does not
# own a `VarPath` of its own in PROCESS (`k_a`/`k_c` in `CsFatigue.ncycle`,
# `process/models/cs_fatigue.py:95-102`, are two calls to it with different `phi` --
# `np.pi/2`, `0`, via `pi_2_arr` -- feeding a `while` loop, not a stored field), and how
# it composes into `ncycle`'s eventual port is exactly the open question this file's
# audit record reports rather than resolves. Wrapping it in a node now would mean
# inventing `VarPath`s no real PROCESS run ever writes.
# ---------------------------------------------------------------------------
