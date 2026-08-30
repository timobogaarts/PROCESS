"""Pure-functional port of the tokamak TF coil's **stress chain** --
`process/models/tfcoil/base.py`'s `tf_field_and_force` (`:1623-1821`) and `stresscl`
(`:2222-3274`), plus the four elasticity-smearing helpers and the `plane_stress` layer
solver `stresscl` calls (`:3659-3717`, `:4236-4670`).

Audit record: `functional_process/_audit/units/models/tfcoil/stress.md`.

**Why this is a module of its own and not part of `models/tfcoil/base.py`.** That file's
scope is the minimal closure of `.tokamak.cicc_superconducting_tf_coil`'s ten *boundary*
reads, and its docstring excludes both of these functions for one stated reason -- "feeds
only stresses, which no boundary read depends on". That was true of the boundary and
false of the **constraint surface**, which was added later: `large_tokamak_nof.IN.DAT:
146-147` activates constraints **31** (`sig_tf_case <= sig_tf_case_max`) and **32**
(`sig_tf_wp <= sig_tf_wp_max`), and with no producer for either operand the port
evaluated both as `0 <= max` -- two *dropped* constraints, not two wrong numbers, which
is worse because no residual reports it. `.tfcoil.str_wp` is the third: at
`i_str_wp == 1` (PROCESS's default, `tfcoil_variables.py:508`) it is the strain the
Nb3Sn critical-current surface (constraint 33) and the temperature margin (constraint
36) read, and **zero strain is the peak of that fit**, so its absence was optimistic
rather than neutral.

## What is ported: one cell of a four-switch space

`stresscl` has 65 parameters and four internal switches. This module writes the single
cell the tracked tokamaks take, and nothing else:

| switch | value ported | why the others are not here |
|---|---|---|
| `i_tf_sup` | `1` (superconducting) | **resolved above this file.** `caller.py:295-316` picks `CICCSuperconductingTFCoil` at `i_tf_sup == 1`; a resistive machine has a different occupant of `.tokamak`'s TF slot entirely, not a different arm of this node |
| `i_tf_stress_model` | `1` (generalised plane stress) | `0`/`2` route to `extended_plane_strain` (`base.py:3719-4234`, **517 lines**), a different solver returning three strain arrays this one does not compute. Refused in `indat.py`, and it is the live arm of both tracked *spherical* tokamaks -- see the record |
| `i_tf_bucking` | `1` (nose casing bucks, no CS layer) | `>= 2` (bucked-and-wedged) prepends a **central-solenoid** layer whose properties are rebuilt from scratch out of nine `.pf_coil` fields (`base.py:2531-2650`); `3` adds a Kapton interlayer on top. Neither reads-set is written |
| `i_tf_tresca` | **not read at all** | measured, not assumed: its two branches (`base.py:3196`, `:3218`) are both gated on `ii >= i_tf_bucking + 1`, and the two layers this node reports are `n_tf_bucking` and `n_tf_bucking - 1`. Neither can reach the gate, so no CEA out-of-plane correction and no von Mises array is computed here |
| `i_tf_turns_integer` | **both arms written** | it picks *which field* the cable-space width for the transverse smearing comes from -- `.superconducting_tfcoil.dx_tf_turn_cable_space_average` at `0`, `.superconducting_tfcoil.dr_tf_turn_cable_space` at `1` (`base.py:2745-2749`). A different read, so a different occupant; `low_aspect_ratio_DEMO` is the integer one and it assembles today |

`tf_field_and_force` carries a fifth: `itart == 1 and i_cp_joints == 1` (a spherical
tokamak with *sliding* centrepost joints) splits the vertical tension between centrepost
and legs by a different closed form and **owns `.tfcoil.f_vforce_inboard`**, which the
clamped-joint arm reads and returns unchanged. That arm is
unreachable through `machine_from_indat` for any occupant of this namespace:
`init.py:752-756` resolves `i_cp_joints == -1` (the default,
`tfcoil_variables.py:589`) to `0` for every superconducting coil, so only an input file
that sets `i_cp_joints = 1` *and* `itart = 1` on a superconducting machine reaches it,
and none of the tracked files does. Refused in `indat.py` rather than left inferred.

## What is deliberately not returned

`stresscl` returns 34 values. This node owns **five** -- the three the constraint
surface reads (`sig_tf_wp`, `sig_tf_case`, `str_wp`) and the two strains PROCESS stores
beside them (`casestr`, `insstrain`, `base.py:2991-2998`, one line each off quantities
already computed). The other 29 are either

* **reporting arrays** consumed only by `out_stress` (`base.py:3275`) -- the nine
  `sig_tf_*`/`str_tf_*` radial distributions, `radial_array`, `deflect`,
  `s_shear_cea_tf_cond`, the smeared moduli. PROCESS stores none of them in
  `DataStructure` on this arm; they are local variables of
  `SuperconductingTFCoil.tf_stress` passed straight to the printer
  (`superconducting.py:2105-2138`), so there is no `VarPath` to own; or
* **`sig_tf_cs_bucked`**, which `stresscl` leaves as `None` unless `i_tf_bucking >= 2`
  (`base.py:3230-3231`) -- conditional ownership, and this node is the arm that does not
  own it.

`.tfcoil.n_rad_per_layer` is likewise **not** a port: it is an `InputVariable`
(`input.py:1074`, range 1-500) that `SuperconductingTFCoil.tf_stress` overwrites with
`500` unconditionally at `superconducting.py:2100` before every call, so on this path it
is a constant of the model and not of the run. `N_RADIAL_ARRAY` below.

## JAX notes

Nothing here is untraceable -- no CoolProp, no external call. Three transcriptions are
not literal and each is marked at its site:

1. `plane_stress`'s `np.linalg.solve` becomes `jnp.linalg.solve` on the same
   row-equilibrated matrix. Same LU-with-partial-pivoting; agreement is to rounding, not
   to the bit, and PROCESS's own comment at `base.py:4404-4412` says as much about its
   own portability.
2. The `argmax` reduction over the radial array (`base.py:3199-3226`) becomes
   `jnp.argmax` over a `(n_tf_layer, n_radial_array)` reshape, with the degenerate case
   PROCESS's `ii_max = 0` initialiser produces spelled out -- see `_layer_peak_indices`.
3. `eyoung_series`'s division-by-zero guard is the double-`jnp.where` idiom of
   `models/safe_math.py`, and it is **load-bearing on the reference run**:
   `.tfcoil.eyoung_cond_axial` and `.tfcoil.eyoung_cond_trans` are both `0.0` there, so
   the zero branch is the one taken and an unguarded `l / eyoung` would put `inf` into
   the primal and `nan` into every tangent.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.paths import build, physics, superconducting_tfcoil, tfcoil
from process.core import constants

N_RADIAL_ARRAY = 500
"""`n_radial_array`: test points per stress layer.

`SuperconductingTFCoil.tf_stress` writes `self.data.tfcoil.n_rad_per_layer = 500`
(`process/models/tfcoil/superconducting.py:2100`) immediately before calling `stresscl`,
overwriting whatever the input file set. A quadrature resolution, kind (b) in
`_audit/switch_elimination_design.md` §3, and on this path not even a user input -- so a
module constant rather than a port or a static kwarg.
"""

N_TF_BUCKING = 1
"""`i_tf_bucking` resolved for a superconducting coil.

`process/core/init.py:891-895` turns the `-1` default (`tfcoil_variables.py:341`) into
`1` whenever `i_tf_sup != 0`, and `1` means "the nose casing is the bucking cylinder";
there is no separate central-solenoid layer. Also the number of layers *before* the
winding pack, which is what `stresscl` uses it for (`base.py:2528`).
"""


# ---------------------------------------------------------------------------
# The elasticity smearing helpers -- `process/models/tfcoil/base.py:3659-3717`,
# `:4459-4670`
# ---------------------------------------------------------------------------


def eyoung_parallel(
    eyoung_j_1, a_1, poisson_j_perp_1, eyoung_j_2, a_2, poisson_j_perp_2
):
    """Two members carrying a force in parallel, smeared. Unchanged.

    Ports `eyoung_parallel`, `process/models/tfcoil/base.py:3660-3716`. The area-weighted
    average of both moduli and both Poisson's ratios; see PROCESS Issue #1205 for the
    derivation the original docstring cites.

    Parameters
    ----------
    eyoung_j_1, a_1, poisson_j_perp_1 :
        The first member's Young's modulus in the load direction (Pa), its
        cross-sectional area perpendicular to that direction, and its Poisson's ratio
        between the load direction and the transverse one.
    eyoung_j_2, a_2, poisson_j_perp_2 :
        The same triplet for the second member. PROCESS uses this slot as a running sum
        when compositing more than two members, and so does
        `eyoung_parallel_array` below.

    Returns
    -------
    :
        `(eyoung_j_3, a_3, poisson_j_perp_3)` -- the smeared triplet.
    """
    poisson_j_perp_3 = (poisson_j_perp_1 * a_1 + poisson_j_perp_2 * a_2) / (a_1 + a_2)
    eyoung_j_3 = (eyoung_j_1 * a_1 + eyoung_j_2 * a_2) / (a_1 + a_2)
    return eyoung_j_3, a_1 + a_2, poisson_j_perp_3


def eyoung_series(eyoung_j_1, l_1, poisson_j_perp_1, eyoung_j_2, l_2, poisson_j_perp_2):
    """Two members carrying a force in series, smeared.

    Ports `eyoung_series`, `process/models/tfcoil/base.py:4602-4670`: the harmonic mean
    of the two moduli weighted by length, with a zero-modulus member short-circuiting
    the composite to zero.

    **The guard is not cosmetic here.** PROCESS branches on
    `eyoung_j_1 * eyoung_j_2 == 0` and returns `eyoung_j_3 = 0` with the *other*
    member's Poisson's ratio. On the reference run both `.tfcoil.eyoung_cond_axial` and
    `.tfcoil.eyoung_cond_trans` are `0.0`, so this is the live branch -- and the
    unguarded expression evaluates `l_1 / 0.0` even when the branch is not taken, which
    is exactly the `inf`-into-every-tangent failure `models/safe_math.py` documents. The
    inner `jnp.where` is what keeps the untaken branch's divisor away from zero.

    Parameters
    ----------
    eyoung_j_1, l_1, poisson_j_perp_1 :
        The first member's Young's modulus in the load direction (Pa), its length
        *along* that direction (m), and its Poisson's ratio.
    eyoung_j_2, l_2, poisson_j_perp_2 :
        The same triplet for the second member.

    Returns
    -------
    :
        `(eyoung_j_3, l_3, poisson_j_perp_3)` -- the smeared triplet.
    """
    either_is_zero = (eyoung_j_1 * eyoung_j_2) == 0.0
    safe_1 = jnp.where(eyoung_j_1 == 0.0, 1.0, eyoung_j_1)
    safe_2 = jnp.where(eyoung_j_2 == 0.0, 1.0, eyoung_j_2)
    compliance = l_1 / safe_1 + l_2 / safe_2

    eyoung_j_3 = jnp.where(either_is_zero, 0.0, (l_1 + l_2) / compliance)
    poisson_j_perp_3 = jnp.where(
        either_is_zero,
        # `base.py:4658`: whichever member is *not* the zero one carries the ratio, and
        # if both are zero PROCESS takes member 2's.
        jnp.where(eyoung_j_1 == 0.0, poisson_j_perp_1, poisson_j_perp_2),
        (poisson_j_perp_1 * l_1 / safe_1 + poisson_j_perp_2 * l_2 / safe_2) / compliance,
    )
    return eyoung_j_3, l_1 + l_2, poisson_j_perp_3


def eyoung_parallel_array(eyoung_j_in, a_in, poisson_j_perp_in):
    """`n` members in parallel, composited by repeated `eyoung_parallel`.

    Ports `eyoung_parallel_array`, `process/models/tfcoil/base.py:4460-4522`. PROCESS
    takes `n` as its first argument and iterates `range(n)`; here the count is the length
    of the sequences, which is static at trace time and always equal to what PROCESS
    passes.

    Parameters
    ----------
    eyoung_j_in, a_in, poisson_j_perp_in :
        Sequences of per-member moduli (Pa), areas and Poisson's ratios, in PROCESS's
        own member order -- the running sum starts at `(0, 0, 0)`, so the order does not
        affect the result but the first member's area must be non-zero.

    Returns
    -------
    :
        `(eyoung_j_out, a_out, poisson_j_perp_out)` -- the composite triplet.
    """
    eyoung_j_out, a_out, poisson_j_perp_out = 0.0, 0.0, 0.0
    for eyoung_j, a, poisson_j_perp in zip(
        eyoung_j_in, a_in, poisson_j_perp_in, strict=True
    ):
        eyoung_j_out, a_out, poisson_j_perp_out = eyoung_parallel(
            eyoung_j, a, poisson_j_perp, eyoung_j_out, a_out, poisson_j_perp_out
        )
    return eyoung_j_out, a_out, poisson_j_perp_out


def eyoung_t_nested_squares(eyoung_j_in, l_in, poisson_j_perp_in):
    """`n` members whose cross-sections are nested squares, smeared transversely.

    Ports `eyoung_t_nested_squares`, `process/models/tfcoil/base.py:4524-4600`, with
    PROCESS's leading `n` argument dropped for the same reason as in
    `eyoung_parallel_array`. Each "leg" (a vertical slice of the square section, Figure
    10 of the TF coil documentation) is a series stack; the legs are then composited in
    parallel.

    Parameters
    ----------
    eyoung_j_in, l_in, poisson_j_perp_in :
        Sequences of per-member moduli (Pa), member thicknesses (m) and Poisson's
        ratios, innermost square first.

    Returns
    -------
    :
        `(eyoung_j_out, l_out, poisson_j_perp_out, eyoung_stiffest)` -- the composite
        triplet, plus the stiffest leg's modulus, which is the unsmearing factor
        `stresscl` divides by to recover the steel conduit's own stress.
    """
    n = len(eyoung_j_in)
    eyoung_j_working = [eyoung_j_in[0]]
    l_working = [l_in[0]]
    poisson_j_perp_working = [poisson_j_perp_in[0]]

    for ii in range(1, n):
        eyoung_j_working.append(eyoung_j_in[ii])
        l_working.append(l_working[ii - 1] + l_in[ii])
        poisson_j_perp_working.append(poisson_j_perp_in[ii])

        # Serial-composite the new layer of this member into every previous leg.
        for jj in range(ii):
            (
                eyoung_j_working[jj],
                l_working[jj],
                poisson_j_perp_working[jj],
            ) = eyoung_series(
                eyoung_j_working[ii],
                l_in[ii],
                poisson_j_perp_working[ii],
                eyoung_j_working[jj],
                l_working[jj],
                poisson_j_perp_working[jj],
            )

    # `base.py:4585`: `max()` over the legs, which is a reduction over a *static* list
    # of traced scalars, so `jnp.max` on the stack rather than Python's `max`.
    eyoung_stiffest = jnp.max(jnp.stack([jnp.asarray(e) for e in eyoung_j_working]))

    eyoung_j_out, l_out, poisson_j_perp_out = eyoung_parallel_array(
        eyoung_j_working, l_in, poisson_j_perp_working
    )
    return eyoung_j_out, l_out, poisson_j_perp_out, eyoung_stiffest


# ---------------------------------------------------------------------------
# The layer solver -- `process/models/tfcoil/base.py:4236-4458`
# ---------------------------------------------------------------------------


def plane_stress(*, nu, rad, ey, j, n_radial_array=N_RADIAL_ARRAY):
    """Generalised plane stress in a stack of current-carrying cylindrical layers.

    Ports `plane_stress`, `process/models/tfcoil/base.py:4236-4458` -- the CCFE model
    `i_tf_stress_model == 1` selects. Each layer's radial displacement solves
    `d/dr (1/r d(r u)/dr) = -(alpha r + beta / r)` with the Lorentz body force of a
    uniform current density; the two integration constants per layer are fixed by null
    radial stress at the inner and outer faces and by continuity of radial stress and
    displacement at every interface, which is the `2 * nlayers` linear system built
    below.

    PROCESS's `nlayers` argument is the length of `j` here, static at trace time.

    Parameters
    ----------
    nu :
        Per-layer Poisson's ratio between the two transverse directions, length
        `nlayers`.
    rad :
        Layer boundary radii (m), length `nlayers + 1`, inboard to outboard.
    ey :
        Per-layer Young's modulus in the transverse direction (Pa), length `nlayers`.
    j :
        Per-layer effective current density (A/m^2), length `nlayers`.
    n_radial_array :
        Test points per layer.

    Returns
    -------
    :
        `(sigr, sigt, r_deflect, rradius)` -- radial stress (Pa), toroidal stress (Pa),
        radial deflection (m) and the radius each was evaluated at (m), each of length
        `nlayers * n_radial_array` and laid out layer by layer.
    """
    nlayers = len(j)
    kk = ey / (1.0 - nu**2)

    # Lorentz force parameterisation. `alpha` is an array equation in PROCESS
    # (`base.py:4298`); `beta` needs the running inner-layer current, so it stays a
    # loop over a static count.
    alpha = 0.5 * constants.RMU0 * j**2 / kk
    beta_terms = []
    inner_layer_curr = 0.0
    for ii in range(nlayers):
        beta_terms.append(
            0.5
            * constants.RMU0
            * j[ii]
            * (inner_layer_curr - jnp.pi * j[ii] * rad[ii] ** 2)
            / (jnp.pi * kk[ii])
        )
        inner_layer_curr += jnp.pi * (rad[ii + 1] ** 2 - rad[ii] ** 2) * j[ii]
    beta = jnp.stack(beta_terms)

    aa = jnp.zeros((2 * nlayers, 2 * nlayers))
    bb = jnp.zeros((2 * nlayers,))

    # Null radial stress at rad[0] (`base.py:4318-4320`, `:4355-4358`).
    aa = aa.at[0, 0].set(kk[0] * (1.0 + nu[0]))
    aa = aa.at[0, 1].set(-kk[0] * (1.0 - nu[0]) / rad[0] ** 2)
    bb = bb.at[0].set(
        -kk[0]
        * (
            0.125 * alpha[0] * (3.0 + nu[0]) * rad[0] ** 2
            + 0.5 * beta[0] * (1.0 + (1.0 + nu[0]) * jnp.log(rad[0]))
        )
    )

    # Interface conditions: continuous radial stress and continuous displacement
    # (`base.py:4323-4340`, `:4361-4381`).
    for ii in range(nlayers - 1):
        aa = aa.at[2 * ii + 1, 2 * ii].set(kk[ii] * (1.0 + nu[ii]))
        aa = aa.at[2 * ii + 1, 2 * ii + 1].set(
            -kk[ii] * (1.0 - nu[ii]) / rad[ii + 1] ** 2
        )
        aa = aa.at[2 * ii + 1, 2 * ii + 2].set(-kk[ii + 1] * (1.0 + nu[ii + 1]))
        aa = aa.at[2 * ii + 1, 2 * ii + 3].set(
            kk[ii + 1] * (1.0 - nu[ii + 1]) / rad[ii + 1] ** 2
        )

        aa = aa.at[2 * ii + 2, 2 * ii].set(rad[ii + 1])
        aa = aa.at[2 * ii + 2, 2 * ii + 1].set(1.0 / rad[ii + 1])
        aa = aa.at[2 * ii + 2, 2 * ii + 2].set(-rad[ii + 1])
        aa = aa.at[2 * ii + 2, 2 * ii + 3].set(-1.0 / rad[ii + 1])

        bb = bb.at[2 * ii + 1].set(
            -kk[ii]
            * (
                0.125 * alpha[ii] * (3.0 + nu[ii]) * rad[ii + 1] ** 2
                + 0.5 * beta[ii] * (1.0 + (1.0 + nu[ii]) * jnp.log(rad[ii + 1]))
            )
            + kk[ii + 1]
            * (
                0.125 * alpha[ii + 1] * (3.0 + nu[ii + 1]) * rad[ii + 1] ** 2
                + 0.5 * beta[ii + 1] * (1.0 + (1.0 + nu[ii + 1]) * jnp.log(rad[ii + 1]))
            )
        )
        bb = bb.at[2 * ii + 2].set(
            -0.125 * alpha[ii] * rad[ii + 1] ** 3
            - 0.5 * beta[ii] * rad[ii + 1] * jnp.log(rad[ii + 1])
            + 0.125 * alpha[ii + 1] * rad[ii + 1] ** 3
            + 0.5 * beta[ii + 1] * rad[ii + 1] * jnp.log(rad[ii + 1])
        )

    # Null radial stress at rad[nlayers] (`base.py:4343-4350`, `:4384-4390`).
    last = nlayers - 1
    aa = aa.at[2 * last + 1, 2 * last].set(kk[last] * (1.0 + nu[last]))
    aa = aa.at[2 * last + 1, 2 * last + 1].set(
        -kk[last] * (1.0 - nu[last]) / rad[nlayers] ** 2
    )
    bb = bb.at[2 * last + 1].set(
        -kk[last]
        * (
            0.125 * alpha[last] * (3.0 + nu[last]) * rad[nlayers] ** 2
            + 0.5 * beta[last] * (1.0 + (1.0 + nu[last]) * jnp.log(rad[nlayers]))
        )
    )

    # Row equilibration before the solve, kept because it changes the answer: the
    # matrix is often very ill-conditioned and PROCESS scales each row's largest
    # element to 1 (`base.py:4404-4419`). Same scalar on the row and on `bb`, so the
    # solution is unchanged in exact arithmetic and stabler in floating point.
    row_scale = jnp.max(jnp.abs(aa), axis=1)
    cc = jnp.linalg.solve(aa / row_scale[:, None], bb / row_scale)
    c1 = cc[0::2]
    c2 = cc[1::2]

    # Radial distribution. PROCESS walks `jj` from `ii * n_radial_array` and evaluates
    # at `rad[ii] + dradius * (jj - n_radial_array * ii)`, i.e. `n_radial_array` evenly
    # spaced points starting *at* the layer's inner radius and stopping one step short
    # of its outer one (`base.py:4429-4433`).
    offsets = jnp.arange(n_radial_array)
    rradius = jnp.concatenate([
        rad[ii] + (rad[ii + 1] - rad[ii]) / n_radial_array * offsets
        for ii in range(nlayers)
    ])
    layer_of = jnp.repeat(jnp.arange(nlayers), n_radial_array)

    kk_r, nu_r = kk[layer_of], nu[layer_of]
    alpha_r, beta_r = alpha[layer_of], beta[layer_of]
    c1_r, c2_r = c1[layer_of], c2[layer_of]
    log_r = jnp.log(rradius)

    sigr = kk_r * (
        (1.0 + nu_r) * c1_r
        - ((1.0 - nu_r) * c2_r) / rradius**2
        + 0.125 * (3.0 + nu_r) * alpha_r * rradius**2
        + 0.5 * beta_r * (1.0 + (1.0 + nu_r) * log_r)
    )
    sigt = kk_r * (
        (1.0 + nu_r) * c1_r
        + (1.0 - nu_r) * c2_r / rradius**2
        + 0.125 * (1.0 + 3.0 * nu_r) * alpha_r * rradius**2
        + 0.5 * beta_r * (nu_r + (1.0 + nu_r) * log_r)
    )
    r_deflect = (
        c1_r * rradius
        + c2_r / rradius
        + 0.125 * alpha_r * rradius**3
        + 0.5 * beta_r * rradius * log_r
    )
    return sigr, sigt, r_deflect, rradius


def tresca_stress(stress_x, stress_y, stress_z):
    """The Tresca (maximum shear) criterion of three principal stresses.

    Ports `calculate_tresca_stress`,
    `process/models/engineering/materials.py:53-82`. Transcribed here rather than in a
    `models/engineering/materials.py` of its own because this is its only reader in the
    port so far; the file it comes from is otherwise untouched.

    Parameters
    ----------
    stress_x, stress_y, stress_z :
        The three principal stresses (Pa). Arrays broadcast elementwise.

    Returns
    -------
    :
        `max(|x - y|, |y - z|, |x - z|)` (Pa).
    """
    return jnp.maximum(
        jnp.maximum(jnp.abs(stress_x - stress_y), jnp.abs(stress_y - stress_z)),
        jnp.abs(stress_x - stress_z),
    )


def _layer_peak_indices(s_shear_tf, n_tf_layer, n_radial_array):
    """Where in each layer the shear stress peaks -- `base.py:3199-3213`, faithfully.

    PROCESS scans each layer's radial slice with `sig_max = 0.0` and `ii_max = 0`, taking
    a *strictly* greater value. Two consequences are reproduced here rather than
    normalised away:

    * ties go to the innermost point of the layer, which is what `jnp.argmax` does too;
    * a layer whose shear stress never exceeds zero keeps `ii_max = 0` -- the **global**
      index zero, i.e. layer 0's innermost point, not its own. That is a quirk of
      initialising the index outside the layer's own range, and it is unreachable on any
      physical run (a Tresca stress is a maximum of absolute differences, so it is
      zero only where all three principal stresses are equal at every one of the 500
      points), but a port that silently picked the layer's own first point would be a
      different function.

    Parameters
    ----------
    s_shear_tf :
        The Tresca stress at every radial station (Pa), length
        `n_tf_layer * n_radial_array`.
    n_tf_layer, n_radial_array :
        The layout of that array.

    Returns
    -------
    :
        One global index per layer, length `n_tf_layer`.
    """
    per_layer = s_shear_tf.reshape(n_tf_layer, n_radial_array)
    within = jnp.argmax(per_layer, axis=1)
    starts = jnp.arange(n_tf_layer) * n_radial_array
    return jnp.where(jnp.max(per_layer, axis=1) > 0.0, starts + within, 0)


# ---------------------------------------------------------------------------
# `tf_field_and_force` -- `process/models/tfcoil/base.py:1623-1821`
# ---------------------------------------------------------------------------


def tf_field_and_force_clamped_joints(
    *,
    r_tf_wp_inboard_outer,
    r_tf_wp_inboard_inner,
    r_tf_outboard_in,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
    b_tf_inboard_peak_symmetric,
    c_tf_total,
    n_tf_coils,
    dr_tf_plasma_case,
    rmajor,
    b_plasma_toroidal_on_axis,
    f_vforce_inboard,
):
    """In-plane centring force and vertical tension on a superconducting TF coil.

    Ports `TFCoil.tf_field_and_force`, `process/models/tfcoil/base.py:1623-1821`, on the
    `i_tf_sup == 1` conductor arm and the clamped-joint (`not (itart == 1 and
    i_cp_joints == 1)`) tension arm -- see this module's docstring for why the other
    two are absent.

    `f_vforce_inboard` is a **read and not an output here**: PROCESS returns it, but on
    this arm the returned value is the argument unchanged (`base.py:1817` never
    reassigns it), so declaring it an output would be an invented edge. The sliding-joint
    arm is the one that owns it.

    Parameters
    ----------
    r_tf_wp_inboard_outer, r_tf_wp_inboard_inner :
        Inboard winding pack outer/inner radius including ground insulation and the
        insertion gap (m).
    r_tf_outboard_in :
        Inner radius of the outboard leg (m).
    dx_tf_wp_insulation, dx_tf_wp_insertion_gap :
        Ground-insulation thickness and winding pack insertion gap (m); both are
        stripped off to get the conductor region.
    b_tf_inboard_peak_symmetric :
        Peak inboard field with no ripple allowance (T).
    c_tf_total :
        Total current in the whole TF coil set (A).
    n_tf_coils :
        Number of TF coils.
    dr_tf_plasma_case :
        Plasma-side case thickness (m).
    rmajor, b_plasma_toroidal_on_axis :
        Plasma major radius (m) and on-axis toroidal field (T) -- together the
        `R B` product the vertical tension integral is written in.
    f_vforce_inboard :
        The fraction of the total vertical tension carried inboard.

    Returns
    -------
    :
        `(cforce, vforce, vforce_outboard, vforce_inboard_tot)` -- centring force per
        coil per metre (N/m), inboard vertical tension per coil (N), outboard vertical
        tension per coil (N), and the inboard tension summed over the set (N).
    """
    # Conductor-region radii: ground insulation and the insertion gap come off both
    # faces (`base.py:1690-1696`, the `i_tf_sup == 1` arm).
    r_out_cond = r_tf_wp_inboard_outer - dx_tf_wp_insulation - dx_tf_wp_insertion_gap
    r_in_cond = r_tf_wp_inboard_inner + dx_tf_wp_insulation + dx_tf_wp_insertion_gap
    dr_cond = r_out_cond - r_in_cond

    cforce = 0.5 * b_tf_inboard_peak_symmetric * c_tf_total / n_tf_coils

    r_outboard_cond = (
        r_tf_outboard_in
        + dr_tf_plasma_case
        + dx_tf_wp_insulation
        + dx_tf_wp_insertion_gap
    )

    # `base.py:1728-1734` replaces a zero inner conductor radius with 1e-9 to keep the
    # logarithms finite in a machine with no bore. Kept: it is a guard PROCESS's own
    # answer depends on, not a numerical convenience of this port.
    r_in_cond = jnp.where(r_in_cond == 0.0, 1.0e-9, r_in_cond)

    vforce_tot = (
        0.5
        * (b_plasma_toroidal_on_axis * rmajor * c_tf_total)
        / (n_tf_coils * dr_cond**2)
        * (
            r_out_cond**2 * jnp.log(r_out_cond / r_in_cond)
            + r_outboard_cond**2 * jnp.log((r_outboard_cond + dr_cond) / r_outboard_cond)
            + dr_cond**2 * jnp.log((r_outboard_cond + dr_cond) / r_in_cond)
            - dr_cond * (r_out_cond + r_outboard_cond)
            + 2.0
            * dr_cond
            * (
                r_out_cond * jnp.log(r_in_cond / r_out_cond)
                + r_outboard_cond
                * jnp.log((r_outboard_cond + dr_cond) / r_outboard_cond)
            )
        )
    )

    vforce = f_vforce_inboard * vforce_tot
    vforce_outboard = vforce * ((1.0 / f_vforce_inboard) - 1.0)
    return cforce, vforce, vforce_outboard, vforce * n_tf_coils


# ---------------------------------------------------------------------------
# `stresscl` -- `process/models/tfcoil/base.py:2222-3274`
# ---------------------------------------------------------------------------


def _winding_pack_smeared_properties(
    *,
    r_tf_wp_inboard_inner,
    r_tf_wp_inboard_outer,
    tan_theta_coil,
    rad_tf_coil_inboard_toroidal_half,
    a_tf_coil_inboard_steel,
    a_tf_plasma_case,
    a_tf_coil_nose_case,
    eyoung_steel,
    poisson_steel,
    eyoung_cond_axial,
    poisson_cond_axial,
    eyoung_cond_trans,
    poisson_cond_trans,
    eyoung_ins,
    poisson_ins,
    eyoung_copper,
    poisson_copper,
    dx_tf_turn_insulation,
    dx_tf_wp_insertion_gap,
    dx_tf_wp_insulation,
    n_tf_coil_turns,
    dx_tf_turn_cable_space_eyoung,
    dia_tf_turn_coolant_channel,
    f_a_tf_turn_cable_copper,
    dx_tf_turn_steel,
    dx_tf_side_case_average,
    dx_tf_wp_toroidal_average,
    a_tf_coil_inboard_insulation,
    a_tf_wp_steel,
    a_tf_wp_conductor,
    a_tf_wp_with_insulation,
):
    """The winding pack's smeared elastic properties -- `base.py:2702-2848`.

    Split out of the ported `stresscl` below because it is the half of that function
    that has nothing to do with the layer solver: it turns a cable-in-conduit turn into
    the four moduli and ratios the solver needs, plus the stiffest-leg modulus the
    unsmearing corrections divide by. Every argument is described on the caller.

    Returns
    -------
    :
        `(r_wp_inner_eff, r_wp_outer_eff, eyoung_wp_trans_eff, poisson_wp_trans_eff,
        eyoung_wp_axial_eff, poisson_wp_axial_eff, eyoung_wp_stiffest_leg)`.
    """
    # Effective WP radii, chosen to preserve the true WP area rather than its true
    # radii -- PROCESS Issue #1048, `base.py:2706-2713`.
    shape = jnp.sqrt(tan_theta_coil / rad_tf_coil_inboard_toroidal_half)
    r_wp_inner_eff = r_tf_wp_inboard_inner * shape
    r_wp_outer_eff = r_tf_wp_inboard_outer * shape

    # Steel under the cylinder that represents the WP: the inboard steel less both
    # case pieces (`base.py:2722-2725`).
    a_wp_steel_eff = a_tf_coil_inboard_steel - a_tf_plasma_case - a_tf_coil_nose_case

    # Ground insulation and insertion gap, shared out over the turns (`base.py:2731`).
    t_ins_eff = (
        dx_tf_turn_insulation
        + (dx_tf_wp_insertion_gap + dx_tf_wp_insulation) / n_tf_coil_turns
    )

    # Transverse (radial/toroidal) modulus: four nested squares around the coolant
    # channel (`base.py:2751-2790`). The cable is a series stack of superconductor and
    # co-wound copper.
    cable_conductor, _, cable_poisson = eyoung_series(
        eyoung_cond_trans,
        (dx_tf_turn_cable_space_eyoung - dia_tf_turn_coolant_channel)
        * (1.0 - f_a_tf_turn_cable_copper),
        poisson_cond_trans,
        eyoung_copper,
        (dx_tf_turn_cable_space_eyoung - dia_tf_turn_coolant_channel)
        * f_a_tf_turn_cable_copper,
        poisson_copper,
    )
    (
        eyoung_wp_trans,
        _,
        poisson_wp_trans,
        eyoung_wp_stiffest_leg,
    ) = eyoung_t_nested_squares(
        # helium channel, cable, steel conduit, insulation
        [0.0, cable_conductor, eyoung_steel, eyoung_ins],
        [
            dia_tf_turn_coolant_channel,
            (dx_tf_turn_cable_space_eyoung - dia_tf_turn_coolant_channel),
            2.0 * dx_tf_turn_steel,
            2.0 * t_ins_eff,
        ],
        [poisson_steel, cable_poisson, poisson_steel, poisson_ins],
    )

    # Lateral casing correction: the WP and the two sidewalls carry the toroidal load
    # in series (`base.py:2792-2801`).
    eyoung_wp_trans_eff, _, poisson_wp_trans_eff = eyoung_series(
        eyoung_wp_trans,
        dx_tf_wp_toroidal_average,
        poisson_wp_trans,
        eyoung_steel,
        2.0 * dx_tf_side_case_average,
        poisson_steel,
    )

    # Vertical modulus: five members in parallel by area (`base.py:2806-2836`).
    eyoung_wp_axial, a_working, poisson_wp_axial = eyoung_parallel_array(
        [eyoung_steel, eyoung_ins, eyoung_copper, eyoung_cond_axial, 0.0],
        [
            a_tf_wp_steel,
            a_tf_coil_inboard_insulation,
            a_tf_wp_conductor * f_a_tf_turn_cable_copper,
            a_tf_wp_conductor * (1.0 - f_a_tf_turn_cable_copper),
            a_tf_wp_with_insulation
            - a_tf_wp_conductor
            - a_tf_coil_inboard_insulation
            - a_tf_wp_steel,
        ],
        [poisson_steel, poisson_ins, poisson_copper, poisson_cond_axial, poisson_steel],
    )

    # ... then the lateral case steel in parallel with all of it (`base.py:2840-2848`).
    eyoung_wp_axial_eff, _, poisson_wp_axial_eff = eyoung_parallel(
        eyoung_steel,
        a_wp_steel_eff - a_tf_wp_steel,
        poisson_steel,
        eyoung_wp_axial,
        a_working,
        poisson_wp_axial,
    )
    return (
        r_wp_inner_eff,
        r_wp_outer_eff,
        eyoung_wp_trans_eff,
        poisson_wp_trans_eff,
        eyoung_wp_axial_eff,
        poisson_wp_axial_eff,
        eyoung_wp_stiffest_leg,
    )


def tf_stress_plane_stress_bucked_case(
    *,
    r_tf_inboard_in,
    r_tf_wp_inboard_inner,
    r_tf_wp_inboard_outer,
    tan_theta_coil,
    rad_tf_coil_inboard_toroidal_half,
    dr_tf_plasma_case,
    a_tf_coil_inboard_steel,
    a_tf_plasma_case,
    a_tf_coil_nose_case,
    eyoung_steel,
    poisson_steel,
    eyoung_cond_axial,
    poisson_cond_axial,
    eyoung_cond_trans,
    poisson_cond_trans,
    eyoung_ins,
    poisson_ins,
    eyoung_copper,
    poisson_copper,
    dx_tf_turn_insulation,
    dx_tf_wp_insertion_gap,
    dx_tf_wp_insulation,
    n_tf_coil_turns,
    dx_tf_turn_cable_space_eyoung,
    dia_tf_turn_coolant_channel,
    f_a_tf_turn_cable_copper,
    dx_tf_turn_steel,
    dx_tf_side_case_average,
    dx_tf_wp_toroidal_average,
    a_tf_coil_inboard_insulation,
    a_tf_wp_steel,
    a_tf_wp_conductor,
    a_tf_wp_with_insulation,
    c_tf_total,
    vforce,
    a_tf_coil_inboard_case,
    a_tf_turn_steel,
    n_tf_graded_layers=1,
    n_radial_array=N_RADIAL_ARRAY,
):
    """Peak Tresca stresses and strains in a bucked superconducting TF coil inboard leg.

    Ports `TFCoil.stresscl`, `process/models/tfcoil/base.py:2222-3274`, on the
    `(i_tf_sup, i_tf_stress_model, i_tf_bucking) == (1, 1, 1)` cell -- the one the
    tracked tokamaks take. The module docstring says what each of those switches costs
    and why `i_tf_tresca` is absent altogether.

    The layer stack is `n_tf_bucking` + `n_tf_graded_layers` + 1 layers
    (`init.py:918-922`): the nose casing, which carries no current and buckles the whole
    leg; the winding pack, split into equal-thickness sub-layers with identical smeared
    properties; and the plasma-side case, again currentless. `plane_stress` solves them
    together; the answer is then unsmeared back onto the steel conduit and reduced to one
    peak shear stress per layer.

    Parameters
    ----------
    r_tf_inboard_in :
        Inner radius of the inboard TF leg (m) -- the innermost layer boundary.
    r_tf_wp_inboard_inner, r_tf_wp_inboard_outer :
        Winding pack inner/outer radius (m). Rescaled to preserve area, not radius.
    tan_theta_coil, rad_tf_coil_inboard_toroidal_half :
        Tangent of, and the value of, the coil's inboard toroidal half-angle (rad).
    dr_tf_plasma_case :
        Plasma-side case thickness (m) -- the outermost layer's thickness.
    a_tf_coil_inboard_steel, a_tf_plasma_case, a_tf_coil_nose_case :
        Inboard steel area and the two case areas (m^2), which together give the steel
        under the winding pack cylinder.
    eyoung_steel, poisson_steel :
        Structural steel's Young's modulus (Pa) and Poisson's ratio.
    eyoung_cond_axial, poisson_cond_axial, eyoung_cond_trans, poisson_cond_trans :
        The superconductor's axial and transverse moduli (Pa) and ratios. Both moduli are
        `0.0` on the reference run, which is the branch `eyoung_series` guards.
    eyoung_ins, poisson_ins, eyoung_copper, poisson_copper :
        Turn insulation's and copper's moduli (Pa) and ratios.
    dx_tf_turn_insulation, dx_tf_wp_insertion_gap, dx_tf_wp_insulation :
        Turn insulation thickness, winding pack insertion gap and ground insulation
        thickness (m).
    n_tf_coil_turns :
        Turns per coil.
    dx_tf_turn_cable_space_eyoung :
        The cable space width (m) the transverse smearing is built on. `i_tf_turns_
        integer` picks which field supplies it (`base.py:2745-2749`):
        `.superconducting_tfcoil.dx_tf_turn_cable_space_average` on the averaged arm,
        `.superconducting_tfcoil.dr_tf_turn_cable_space` on the integer one. Named for
        its role rather than for either field, because the two node arms below spell the
        read differently and this function is shared between them.
    dia_tf_turn_coolant_channel, dx_tf_turn_steel :
        Central coolant channel diameter and steel conduit thickness of one turn (m).
    f_a_tf_turn_cable_copper :
        Copper fraction of the cable space.
    dx_tf_side_case_average, dx_tf_wp_toroidal_average :
        Average sidewall case thickness and average winding pack toroidal width (m).
    a_tf_coil_inboard_insulation, a_tf_wp_steel, a_tf_wp_conductor,
    a_tf_wp_with_insulation :
        Inboard insulation, winding pack steel, winding pack conductor and total winding
        pack areas (m^2).
    c_tf_total :
        Total current in the TF coil set (A) -- the winding pack layers' current density.
    vforce :
        Inboard vertical tension per coil (N), from `tf_field_and_force_clamped_joints`.
    a_tf_coil_inboard_case, a_tf_turn_steel :
        Inboard case area and one turn's steel conduit area (m^2) -- together the steel
        that carries `vforce`.
    n_tf_graded_layers :
        How many equal-thickness sub-layers the winding pack is split into. PROCESS's
        `.tfcoil.n_tf_graded_layers`, an input (`input.py:1076`, range 1-20) whose
        default and only tracked value is `1`. A count, not a model choice, so a static
        kwarg rather than a port (`_audit/switch_elimination_design.md` §3 kind (b)).
    n_radial_array :
        Test points per layer. **The node never passes this**: on the ported path it is
        fixed at `N_RADIAL_ARRAY` by `superconducting.py:2100` and is not a run input at
        all. It is a parameter only because PROCESS's own signature has one, and because
        `test_stress.py` replays PROCESS's unit-test sample, which was taken at `100`.

    Returns
    -------
    :
        `(sig_tf_wp, sig_tf_case, str_wp, casestr, insstrain)` -- peak Tresca stress in
        the winding pack and in the nose case (Pa), vertical strain in the winding pack,
        in the case, and radial strain in the turn insulation.
    """
    n_tf_layer = N_TF_BUCKING + n_tf_graded_layers + 1

    (
        r_wp_inner_eff,
        r_wp_outer_eff,
        eyoung_wp_trans_eff,
        poisson_wp_trans_eff,
        eyoung_wp_axial_eff,
        poisson_wp_axial_eff,
        eyoung_wp_stiffest_leg,
    ) = _winding_pack_smeared_properties(
        r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
        r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
        tan_theta_coil=tan_theta_coil,
        rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
        a_tf_coil_inboard_steel=a_tf_coil_inboard_steel,
        a_tf_plasma_case=a_tf_plasma_case,
        a_tf_coil_nose_case=a_tf_coil_nose_case,
        eyoung_steel=eyoung_steel,
        poisson_steel=poisson_steel,
        eyoung_cond_axial=eyoung_cond_axial,
        poisson_cond_axial=poisson_cond_axial,
        eyoung_cond_trans=eyoung_cond_trans,
        poisson_cond_trans=poisson_cond_trans,
        eyoung_ins=eyoung_ins,
        poisson_ins=poisson_ins,
        eyoung_copper=eyoung_copper,
        poisson_copper=poisson_copper,
        dx_tf_turn_insulation=dx_tf_turn_insulation,
        dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        dx_tf_wp_insulation=dx_tf_wp_insulation,
        n_tf_coil_turns=n_tf_coil_turns,
        dx_tf_turn_cable_space_eyoung=dx_tf_turn_cable_space_eyoung,
        dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
        dx_tf_turn_steel=dx_tf_turn_steel,
        dx_tf_side_case_average=dx_tf_side_case_average,
        dx_tf_wp_toroidal_average=dx_tf_wp_toroidal_average,
        a_tf_coil_inboard_insulation=a_tf_coil_inboard_insulation,
        a_tf_wp_steel=a_tf_wp_steel,
        a_tf_wp_conductor=a_tf_wp_conductor,
        a_tf_wp_with_insulation=a_tf_wp_with_insulation,
    )

    # ---- the layer stack (`base.py:2664-2698`, `:2906-2946`) -----------------------
    # Layer 0 is the nose casing, which carries no current and is steel throughout.
    # Layers 1..n_tf_graded_layers are the winding pack. The last layer is the
    # plasma-side case, again currentless steel.
    dr_wp_layer = (r_wp_outer_eff - r_wp_inner_eff) / n_tf_graded_layers
    j_wp = c_tf_total / (jnp.pi * (r_wp_outer_eff**2 - r_wp_inner_eff**2))

    radtf = [r_tf_inboard_in]
    jeff = [0.0]
    eyoung_trans = [eyoung_steel]
    poisson_trans = [poisson_steel]
    eyoung_axial = [eyoung_steel]
    poisson_axial = [poisson_steel]
    for ii in range(n_tf_graded_layers):
        radtf.append(r_wp_inner_eff + ii * dr_wp_layer)
        jeff.append(j_wp)
        eyoung_trans.append(eyoung_wp_trans_eff)
        poisson_trans.append(poisson_wp_trans_eff)
        eyoung_axial.append(eyoung_wp_axial_eff)
        poisson_axial.append(poisson_wp_axial_eff)
    radtf.append(r_wp_outer_eff)
    jeff.append(0.0)
    eyoung_trans.append(eyoung_steel)
    poisson_trans.append(poisson_steel)
    eyoung_axial.append(eyoung_steel)
    poisson_axial.append(poisson_steel)
    radtf.append(r_wp_outer_eff + dr_tf_plasma_case)

    # The plane-stress solver treats the front case as a full annulus; the real case is
    # only part of one, so its axial stiffness is scaled by the area ratio and the
    # scaling is undone again on the way out (PROCESS Issue #1509, `base.py:2949-2962`).
    f_tf_stress_front_case = (
        a_tf_plasma_case
        / rad_tf_coil_inboard_toroidal_half
        / (radtf[n_tf_layer] ** 2 - radtf[n_tf_layer - 1] ** 2)
    )

    sig_tf_r, sig_tf_t, _, _ = plane_stress(
        nu=jnp.stack([jnp.asarray(v, dtype=float) for v in poisson_trans]),
        rad=jnp.stack([jnp.asarray(v, dtype=float) for v in radtf]),
        ey=jnp.stack([jnp.asarray(v, dtype=float) for v in eyoung_trans]),
        j=jnp.stack([jnp.asarray(v, dtype=float) for v in jeff]),
        n_radial_array=n_radial_array,
    )

    # ---- vertical stress and the three strains (`base.py:2980-2999`) ---------------
    # One value for the whole leg: the tension divided by all the steel carrying it.
    # PROCESS's own comment at `:2983` doubts that this is an array equation, and it is
    # right -- every element is the same number.
    sig_tf_z_uniform = vforce / (
        a_tf_coil_inboard_case + a_tf_turn_steel * n_tf_coil_turns
    )
    str_wp = sig_tf_z_uniform / eyoung_wp_axial_eff
    casestr = sig_tf_z_uniform / eyoung_steel
    insstrain = (
        sig_tf_r[n_radial_array - 1]
        * eyoung_wp_stiffest_leg
        / eyoung_wp_trans_eff
        / eyoung_ins
    )

    # ---- unsmearing (`base.py:3068-3131`) -----------------------------------------
    # `i_tf_sup == 1` with the plane-stress model: the vertical factor is 1 (`:3084`),
    # so only the transverse stresses are lifted back onto the steel conduit, and only
    # over the winding pack layers.
    fac_sig_t = fac_sig_r = eyoung_wp_stiffest_leg / eyoung_wp_trans_eff
    wp_slice = slice(
        N_TF_BUCKING * n_radial_array,
        (N_TF_BUCKING + n_tf_graded_layers) * n_radial_array,
    )
    sig_tf_r = sig_tf_r.at[wp_slice].multiply(fac_sig_r)
    sig_tf_t = sig_tf_t.at[wp_slice].multiply(fac_sig_t)

    # The front case's axial stiffness correction, undone (`base.py:3123-3131`).
    sig_tf_z = jnp.full(n_tf_layer * n_radial_array, sig_tf_z_uniform)
    sig_tf_z = sig_tf_z.at[
        (N_TF_BUCKING + n_tf_graded_layers) * n_radial_array :
    ].divide(f_tf_stress_front_case)

    # ---- the reduction (`base.py:3196-3231`) --------------------------------------
    s_shear_tf = tresca_stress(sig_tf_r, sig_tf_t, sig_tf_z)
    peak = s_shear_tf[_layer_peak_indices(s_shear_tf, n_tf_layer, n_radial_array)]
    return peak[N_TF_BUCKING], peak[N_TF_BUCKING - 1], str_wp, casestr, insstrain


# ---------------------------------------------------------------------------
# The nodes
# ---------------------------------------------------------------------------


class TfFieldAndForce(ExplicitFunction):
    """The family that owns the TF coil's in-plane force and vertical tension.

    `(itart, i_cp_joints)` decides it, and only the clamped-joint arm is written --
    see this module's docstring. A family base rather than a bare class because the
    two arms own *different* fields: the sliding-joint arm additionally owns
    `.tfcoil.f_vforce_inboard`, which this one reads.
    """


class TfFieldAndForceClampedJoints(TfFieldAndForce):
    """No sliding centrepost joints -- every superconducting coil unless an input file
    sets `i_cp_joints = 1` alongside `itart = 1`.
    """

    cforce = OutputInto(tfcoil)
    vforce = OutputInto(tfcoil)
    vforce_outboard = OutputInto(tfcoil)
    vforce_inboard_tot = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_outboard_in=From(superconducting_tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        b_tf_inboard_peak_symmetric=From(tfcoil),
        c_tf_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        rmajor=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        f_vforce_inboard=From(tfcoil),
    ):
        return tf_field_and_force_clamped_joints(
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_outboard_in=r_tf_outboard_in,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric,
            c_tf_total=c_tf_total,
            n_tf_coils=n_tf_coils,
            dr_tf_plasma_case=dr_tf_plasma_case,
            rmajor=rmajor,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            f_vforce_inboard=f_vforce_inboard,
        )


class TfStress(ExplicitFunction):
    """The family that owns the TF coil's peak stresses and strains.

    `(i_tf_stress_model, i_tf_bucking)` decides it and only `(1, 1)` is written. A
    family base for the same reason as `TfFieldAndForce`: the bucked-and-wedged arms
    additionally own `.tfcoil.sig_tf_cs_bucked`, which `stresscl` leaves as `None` here
    (`base.py:3230-3231`).
    """


class TfStressPlaneStressBuckedCase(TfStress):
    """`i_tf_stress_model == 1` (generalised plane stress) with `i_tf_bucking == 1`.

    The reference arm of `large_tokamak_eval`, `large_tokamak_nof` and
    `low_aspect_ratio_DEMO`: none of the three sets either switch, so both take their
    defaults (`tfcoil_variables.py:211`, and `-1` resolved to `1` at
    `init.py:891-895`).

    Abstract: `i_tf_turns_integer` splits it once more into the two subclasses below,
    which differ in exactly one read -- the cable-space width the transverse smearing is
    built on. Everything else, including all five outputs, is shared.

    `n_tf_graded_layers` is a static field rather than a `__call__` parameter -- a
    count that fixes the layer stack's *shape*, so it cannot be a traced input.
    """

    n_tf_graded_layers: int = 1

    sig_tf_wp = OutputInto(tfcoil)
    sig_tf_case = OutputInto(tfcoil)
    str_wp = OutputInto(tfcoil)
    casestr = OutputInto(tfcoil)
    insstrain = OutputInto(tfcoil)


class TfStressPlaneStressBuckedCaseAveragedTurn(TfStressPlaneStressBuckedCase):
    """`i_tf_turns_integer == 0` -- the turn is described by one averaged cable-space
    width, `.superconducting_tfcoil.dx_tf_turn_cable_space_average`
    (`base.py:2745-2749`). `large_tokamak_eval`'s and `large_tokamak_nof`'s arm.
    """

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        a_tf_coil_inboard_steel=From(superconducting_tfcoil),
        a_tf_plasma_case=From(superconducting_tfcoil),
        a_tf_coil_nose_case=From(superconducting_tfcoil),
        eyoung_steel=From(tfcoil),
        poisson_steel=From(tfcoil),
        eyoung_cond_axial=From(tfcoil),
        poisson_cond_axial=From(tfcoil),
        eyoung_cond_trans=From(tfcoil),
        poisson_cond_trans=From(tfcoil),
        eyoung_ins=From(tfcoil),
        poisson_ins=From(tfcoil),
        eyoung_copper=From(tfcoil),
        poisson_copper=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        dx_tf_turn_cable_space_average=From(superconducting_tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_side_case_average=From(superconducting_tfcoil),
        dx_tf_wp_toroidal_average=From(superconducting_tfcoil),
        a_tf_coil_inboard_insulation=From(superconducting_tfcoil),
        a_tf_wp_steel=From(tfcoil),
        a_tf_wp_conductor=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        c_tf_total=From(tfcoil),
        vforce=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
    ):
        return tf_stress_plane_stress_bucked_case(
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            tan_theta_coil=tan_theta_coil,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            dr_tf_plasma_case=dr_tf_plasma_case,
            a_tf_coil_inboard_steel=a_tf_coil_inboard_steel,
            a_tf_plasma_case=a_tf_plasma_case,
            a_tf_coil_nose_case=a_tf_coil_nose_case,
            eyoung_steel=eyoung_steel,
            poisson_steel=poisson_steel,
            eyoung_cond_axial=eyoung_cond_axial,
            poisson_cond_axial=poisson_cond_axial,
            eyoung_cond_trans=eyoung_cond_trans,
            poisson_cond_trans=poisson_cond_trans,
            eyoung_ins=eyoung_ins,
            poisson_ins=poisson_ins,
            eyoung_copper=eyoung_copper,
            poisson_copper=poisson_copper,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            n_tf_coil_turns=n_tf_coil_turns,
            dx_tf_turn_cable_space_eyoung=dx_tf_turn_cable_space_average,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_side_case_average=dx_tf_side_case_average,
            dx_tf_wp_toroidal_average=dx_tf_wp_toroidal_average,
            a_tf_coil_inboard_insulation=a_tf_coil_inboard_insulation,
            a_tf_wp_steel=a_tf_wp_steel,
            a_tf_wp_conductor=a_tf_wp_conductor,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            c_tf_total=c_tf_total,
            vforce=vforce,
            a_tf_coil_inboard_case=a_tf_coil_inboard_case,
            a_tf_turn_steel=a_tf_turn_steel,
            n_tf_graded_layers=self.n_tf_graded_layers,
        )


class TfStressPlaneStressBuckedCaseIntegerTurn(TfStressPlaneStressBuckedCase):
    """`i_tf_turns_integer == 1` -- rectangular turns on a fixed layers x pancakes grid,
    so the smearing uses the *radial* cable-space dimension
    `.superconducting_tfcoil.dr_tf_turn_cable_space` instead (`base.py:2745-2749`).
    `low_aspect_ratio_DEMO`'s arm.

    Identical to its sibling in every other read and in all five outputs. Written out
    rather than parameterised because a read is a declaration: the two arms read
    different fields, and on this one `dx_tf_turn_cable_space_average` has no producer
    at all (`CiccIntegerTurnGeometry` does not own it), so declaring it would put a
    stale `DataStructure` value into the answer -- the exact defect
    `functional_process/boundary.py` exists to catch.
    """

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        a_tf_coil_inboard_steel=From(superconducting_tfcoil),
        a_tf_plasma_case=From(superconducting_tfcoil),
        a_tf_coil_nose_case=From(superconducting_tfcoil),
        eyoung_steel=From(tfcoil),
        poisson_steel=From(tfcoil),
        eyoung_cond_axial=From(tfcoil),
        poisson_cond_axial=From(tfcoil),
        eyoung_cond_trans=From(tfcoil),
        poisson_cond_trans=From(tfcoil),
        eyoung_ins=From(tfcoil),
        poisson_ins=From(tfcoil),
        eyoung_copper=From(tfcoil),
        poisson_copper=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        dr_tf_turn_cable_space=From(superconducting_tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_side_case_average=From(superconducting_tfcoil),
        dx_tf_wp_toroidal_average=From(superconducting_tfcoil),
        a_tf_coil_inboard_insulation=From(superconducting_tfcoil),
        a_tf_wp_steel=From(tfcoil),
        a_tf_wp_conductor=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        c_tf_total=From(tfcoil),
        vforce=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
    ):
        return tf_stress_plane_stress_bucked_case(
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            tan_theta_coil=tan_theta_coil,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            dr_tf_plasma_case=dr_tf_plasma_case,
            a_tf_coil_inboard_steel=a_tf_coil_inboard_steel,
            a_tf_plasma_case=a_tf_plasma_case,
            a_tf_coil_nose_case=a_tf_coil_nose_case,
            eyoung_steel=eyoung_steel,
            poisson_steel=poisson_steel,
            eyoung_cond_axial=eyoung_cond_axial,
            poisson_cond_axial=poisson_cond_axial,
            eyoung_cond_trans=eyoung_cond_trans,
            poisson_cond_trans=poisson_cond_trans,
            eyoung_ins=eyoung_ins,
            poisson_ins=poisson_ins,
            eyoung_copper=eyoung_copper,
            poisson_copper=poisson_copper,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            n_tf_coil_turns=n_tf_coil_turns,
            dx_tf_turn_cable_space_eyoung=dr_tf_turn_cable_space,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_side_case_average=dx_tf_side_case_average,
            dx_tf_wp_toroidal_average=dx_tf_wp_toroidal_average,
            a_tf_coil_inboard_insulation=a_tf_coil_inboard_insulation,
            a_tf_wp_steel=a_tf_wp_steel,
            a_tf_wp_conductor=a_tf_wp_conductor,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            c_tf_total=c_tf_total,
            vforce=vforce,
            a_tf_coil_inboard_case=a_tf_coil_inboard_case,
            a_tf_turn_steel=a_tf_turn_steel,
            n_tf_graded_layers=self.n_tf_graded_layers,
        )
