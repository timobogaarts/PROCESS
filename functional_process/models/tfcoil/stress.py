"""Pure functions for the TF coil in-plane force, vertical tension, and the bucked-case
stress models, extracted from `functional_process/cottax/tfcoil/stress.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants) that wire
these functions to `VarPath`s; read its module docstring for scope and the switch
table. The audit record is
`functional_process/_audit/units/models/tfcoil/stress.md` and mirrors these functions,
not the declarations that call them.
"""

import jax.numpy as jnp

from functional_process.vocabulary import constants

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


# ---------------------------------------------------------------------------
# The second layer solver -- `process/models/tfcoil/base.py:3719-4234`
# ---------------------------------------------------------------------------


def _lame_row(r_sq):
    """`[r^2, 1, 0, 0, 0]` as a length-5 row -- PROCESS's `rad_row_helper`.

    `base.py` reuses one `(1, 5)` scratch array for this and overwrites its first two
    entries before every inner product; here each use builds its own row, which is the
    same value and one fewer aliasing hazard.
    """
    zero = jnp.zeros_like(r_sq)
    return jnp.stack([r_sq, jnp.ones_like(r_sq), zero, zero, zero])


def extended_plane_strain(
    *,
    nu_t,
    nu_zt,
    ey_t,
    ey_z,
    rad,
    d_curr,
    v_force,
    i_tf_bucking,
    n_radial_array=N_RADIAL_ARRAY,
):
    """Axisymmetric extended plane strain in a stack of current-carrying layers.

    Ports `extended_plane_strain`, `process/models/tfcoil/base.py:3719-4234` -- the
    solver `i_tf_stress_model in {0, 2}` selects, and the live one on both tracked
    spherical tokamaks. Derivation: PROCESS issue #1414.

    **Why it is a second function and not a second branch of `plane_stress`.** It solves
    a different problem: transverse-isotropic materials (two moduli and three Poisson's
    ratios per layer rather than one of each), a genuine axial degree of freedom driven
    by a prescribed total tension rather than a stress read off afterwards, and an
    optional *slip* boundary at layer `i_tf_bucking` below which the layers carry no net
    axial force. It therefore returns three strain arrays and an axial stress array that
    `plane_stress` never computes, and takes `v_force` where `plane_stress` takes
    nothing.

    **The linear system is 4x4 whatever the layer count.** Each layer's radial
    displacement is Lame's `u = A r + B / r` plus a particular integral for the Lorentz
    body force; continuity across an interface is a fixed `5 x 5` transfer matrix, so the
    whole stack collapses onto the outermost layer's solution vector
    `(A, B, eps_z, 1, eps_z_slip)`. Four scalar conditions -- zero radial stress outside,
    zero radial stress (or zero displacement, at a zero inner radius) inside, prescribed
    total axial force, zero axial force on the slip layers -- are inner products with
    that vector, so stacking them gives one `4 x 4` solve. The `1` in slot 3 is what
    carries the inhomogeneous terms, which is why the solve drops that column into the
    right-hand side.

    PROCESS's `nlayers` argument is the length of `d_curr` here, static at trace time,
    for the same reason `plane_stress` drops it.

    Parameters
    ----------
    nu_t :
        Per-layer transverse (radial-toroidal) Poisson's ratio, length `nlayers`.
    nu_zt :
        Per-layer axial-transverse Poisson's ratio, length `nlayers`.
    ey_t, ey_z :
        Per-layer transverse and axial Young's moduli (Pa), length `nlayers`.
    rad :
        Layer boundary radii (m), length `nlayers + 1`, inboard to outboard.
    d_curr :
        Per-layer uniform current density (A/m^2), length `nlayers`.
    v_force :
        Total axial tension carried by the force-carrying layers (N). On the TF coil
        this is `.superconducting_tfcoil.vforce_inboard_tot`, the *whole set's* inboard
        tension, not one coil's -- `base.py:3021` passes `vforce_inboard_tot` where the
        plane-stress arm divides `vforce` by the steel area itself.
    i_tf_bucking :
        The index of the innermost layer that carries axial force; layers inboard of it
        are decoupled (the "slip" layers) and their axial force is separately
        constrained to zero. A **static** argument: it decides how many rows of the
        transfer-matrix construction exist, not a value inside them. PROCESS clamps it
        up to `1` (`base.py:3919`) and so does this.
    n_radial_array :
        Test points per layer. Unlike `plane_stress`'s, this grid is **closed**: the step
        is `(rad[ii+1] - rad[ii]) / (n_radial_array - 1)` (`base.py:4160`), so the last
        point of a layer sits exactly on its outer radius.

    Returns
    -------
    :
        `(rradius, sigr, sigt, sigz, str_r, str_t, str_z, r_deflect)` in PROCESS's own
        return order -- the radius each quantity was evaluated at (m), the three
        principal stresses (Pa), the three principal strains, and the radial
        displacement (m); each of length `nlayers * n_radial_array`, laid out layer by
        layer.

    Notes
    -----
    **A zero inner radius produces `nan`, and that is reproduced rather than repaired.**
    At `rad[0] == 0` the innermost test point is `r = 0`; the inner boundary condition
    forces `B = 0` there, so `b_plot / r` is `0 / 0` and PROCESS's own unit test records
    the first element of `sigr`, `sigt`, `sigz`, `str_r`, `str_t` and `r_deflect` as
    `nan` (`tests/unit/models/tfcoil/test_tfcoil.py`'s `nan_init`). `f_int_a_plot`'s
    `f_rec_fac * log(rad[1] / 0)` is `0 * inf` and `nan` for the same reason. Both are
    left unguarded here so the port returns what PROCESS returns; the guard PROCESS
    *does* have -- on `f_int_a[0]`, `base.py:3979-3980` -- is reproduced below.
    **This is not reachable from the ported node**: `stresscl` raises
    `ProcessValueError` at `base.py:2524-2527` whenever `r_tf_inboard_in` is zero and
    `i_tf_stress_model != 2`, and the arm registered in `indat.py` is `0`.
    """
    nlayers = len(d_curr)
    # `base.py:3917-3919`. Static, so a Python `int` and Python `max`.
    nonslip_layer = max(int(i_tf_bucking), 1)

    nu_t = jnp.asarray(nu_t, dtype=float)
    nu_zt = jnp.asarray(nu_zt, dtype=float)
    ey_t = jnp.asarray(ey_t, dtype=float)
    ey_z = jnp.asarray(ey_z, dtype=float)
    rad = jnp.asarray(rad, dtype=float)
    d_curr = jnp.asarray(d_curr, dtype=float)

    # ---- stiffness tensor factors (`base.py:3925-3944`, writeup §3 and §12) --------
    nu_tz = nu_zt * ey_t / ey_z
    denominator = 1.0 - nu_t - 2.0 * nu_tz * nu_zt
    ey_bar_z = ey_z * (1.0 - nu_t) / denominator
    ey_bar_t = ey_t * (1.0 - nu_tz * nu_zt) / denominator / (1.0 + nu_t)
    nu_bar_t = (nu_t + nu_tz * nu_zt) / (1.0 - nu_tz * nu_zt)
    nu_bar_tz = nu_tz / (1.0 - nu_t)
    nu_bar_zt = nu_zt * (1.0 + nu_t) / (1.0 - nu_tz * nu_zt)

    # ---- Lorentz force parameters (`base.py:3946-3990`, writeup §13) ---------------
    r_inner = rad[:nlayers]
    r_outer = rad[1 : nlayers + 1]
    currents = jnp.pi * d_curr * (r_outer**2 - r_inner**2)
    # `currents_enclosed[ii]` is the sum of layers 0..ii-1, so a shifted cumulative sum.
    currents_enclosed = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum(currents)[: nlayers - 1],
    ])
    f_lin_fac = constants.RMU0 / 2.0 * d_curr**2
    f_rec_fac = (
        constants.RMU0
        / 2.0
        * (d_curr * currents_enclosed / jnp.pi - d_curr**2 * r_inner**2)
    )

    # A zero inner radius makes `log(r_outer / r_inner)` infinite; PROCESS lets that
    # happen and then overwrites element 0 when `f_rec_fac[0] == 0`. It always is when
    # `rad[0] == 0` -- `currents_enclosed[0]` is identically zero, so
    # `f_rec_fac[0] = -RMU0/2 * d_curr[0]^2 * rad[0]^2` -- so the overwrite covers
    # exactly the case that would have been `nan`. The `where` on the divisor is the
    # `safe_math` idiom: it keeps the *untaken* branch's tangent finite.
    safe_r_inner = jnp.where(r_inner == 0.0, 1.0, r_inner)
    f_int_a = 0.5 * f_lin_fac * (r_outer**2 - r_inner**2) + f_rec_fac * jnp.log(
        r_outer / safe_r_inner
    )
    f_int_a = f_int_a.at[0].set(
        jnp.where(
            f_rec_fac[0] == 0.0,
            0.5 * f_lin_fac[0] * (rad[1] ** 2 - rad[0] ** 2),
            f_int_a[0],
        )
    )
    f_int_b = 0.25 * f_lin_fac * (r_outer**4 - r_inner**4) + 0.5 * f_rec_fac * (
        r_outer**2 - r_inner**2
    )

    # ---- within-layer transfer matrices (`base.py:3992-4010`, writeup §5) ----------
    # `m_int[kk]` carries the solution vector from layer `kk`'s outer radius to its
    # inner one. PROCESS stores these as `(5, 5, nlayers)`; the layer index leads here,
    # which is the same matrices and the ordinary `A @ B` for the products below.
    m_int = jnp.repeat(jnp.eye(5)[None, :, :], nlayers, axis=0)
    m_int = m_int.at[:, 0, 3].set(-0.5 / ey_bar_t * f_int_a)
    m_int = m_int.at[:, 1, 3].set(0.5 / ey_bar_t * f_int_b)

    # ---- between-layer transfer matrices (`base.py:4012-4058`, writeup §6 and §15) --
    # `m_ext[kk]` carries it from layer `kk`'s inner radius to layer `kk-1`'s outer one.
    # `m_ext[0]` is left as zeros, exactly as PROCESS leaves it: the only use of it is
    # the final update of the innermost iteration, whose result is never read.
    m_ext = jnp.zeros((nlayers, 5, 5))
    for kk in range(1, nonslip_layer - 1):
        ey_fac = ey_bar_t[kk] / ey_bar_t[kk - 1]
        m_ext = m_ext.at[kk, 0, 2].set(0.0)
        m_ext = m_ext.at[kk, 0, 4].set(
            0.5 * (ey_fac * nu_bar_zt[kk] - nu_bar_zt[kk - 1])
        )
    if nonslip_layer > 1:
        # The slip interface itself: axial strain switches sides here.
        kk = nonslip_layer - 1
        ey_fac = ey_bar_t[kk] / ey_bar_t[kk - 1]
        m_ext = m_ext.at[kk, 0, 2].set(0.5 * ey_fac * nu_bar_zt[kk])
        m_ext = m_ext.at[kk, 0, 4].set(0.5 * (-nu_bar_zt[kk - 1]))
    for kk in range(nonslip_layer, nlayers):
        ey_fac = ey_bar_t[kk] / ey_bar_t[kk - 1]
        m_ext = m_ext.at[kk, 0, 2].set(
            0.5 * (ey_fac * nu_bar_zt[kk] - nu_bar_zt[kk - 1])
        )
        m_ext = m_ext.at[kk, 0, 4].set(0.0)
    for kk in range(1, nlayers):
        # Written after the three loops above, because rows 1 and 2 below read the
        # `m_ext[kk, 0, 2]` and `m_ext[kk, 0, 4]` they set (`base.py:4048-4051`).
        ey_fac = ey_bar_t[kk] / ey_bar_t[kk - 1]
        m_ext = m_ext.at[kk, 0, 0].set(
            0.5 * (ey_fac * (1.0 + nu_bar_t[kk]) + 1.0 - nu_bar_t[kk - 1])
        )
        # `base.py:4041`: the `B` column is dropped at a zero interface radius, where
        # `1 / r^2` is meaningless. Double `where`, so the untaken branch never divides.
        safe_rad = jnp.where(rad[kk] > 0.0, rad[kk], 1.0)
        m_ext = m_ext.at[kk, 0, 1].set(
            jnp.where(
                rad[kk] > 0.0,
                0.5
                / safe_rad**2
                * (1.0 - nu_bar_t[kk - 1] - ey_fac * (1.0 - nu_bar_t[kk])),
                0.0,
            )
        )
        m_ext = m_ext.at[kk, 1, 0].set(rad[kk] ** 2 * (1.0 - m_ext[kk, 0, 0]))
        m_ext = m_ext.at[kk, 1, 1].set(1.0 - rad[kk] ** 2 * m_ext[kk, 0, 1])
        m_ext = m_ext.at[kk, 1, 2].set(-(rad[kk] ** 2) * m_ext[kk, 0, 2])
        m_ext = m_ext.at[kk, 1, 4].set(-(rad[kk] ** 2) * m_ext[kk, 0, 4])
        m_ext = m_ext.at[kk, 2, 2].set(1.0)
        m_ext = m_ext.at[kk, 3, 3].set(1.0)
        m_ext = m_ext.at[kk, 4, 4].set(1.0)

    # ---- outermost-to-each transfer matrices (`base.py:4060-4070`, writeup §7) -----
    m_tot_layers = [None] * nlayers
    m_tot_layers[nlayers - 1] = m_int[nlayers - 1]
    for kk in range(nlayers - 2, -1, -1):
        m_tot_layers[kk] = m_int[kk] @ (m_ext[kk + 1] @ m_tot_layers[kk + 1])
    m_tot = jnp.stack(m_tot_layers)

    # ---- the axial force inner products (`base.py:4072-4147`, writeup §8) ----------
    # PROCESS sums with Python's `sum` over a numpy array (left to right from `0`);
    # `jnp.sum` may associate differently, which is a rounding difference on a sum of
    # `nlayers` terms and nothing else.
    ey_bar_z_area = jnp.pi * jnp.sum(
        ey_bar_z[nonslip_layer - 1 : nlayers]
        * (rad[nonslip_layer : nlayers + 1] ** 2 - rad[nonslip_layer - 1 : nlayers] ** 2)
    )
    ey_bar_z_area_slip = jnp.pi * jnp.sum(
        ey_bar_z[: nonslip_layer - 1]
        * (rad[1:nonslip_layer] ** 2 - rad[: nonslip_layer - 1] ** 2)
    )

    v_force_row = (
        2.0
        * jnp.pi
        * ey_bar_z[nlayers - 1]
        * nu_bar_tz[nlayers - 1]
        * _lame_row(rad[nlayers] ** 2)
    )
    v_force_row -= (
        2.0
        * jnp.pi
        * ey_bar_z[nonslip_layer - 1]
        * nu_bar_tz[nonslip_layer - 1]
        * (_lame_row(rad[nonslip_layer - 1] ** 2) @ m_tot[nonslip_layer - 1])
    )
    for kk in range(nonslip_layer, nlayers):
        v_force_row += (
            2.0
            * jnp.pi
            * (ey_bar_z[kk - 1] * nu_bar_tz[kk - 1] - ey_bar_z[kk] * nu_bar_tz[kk])
            * (_lame_row(rad[kk] ** 2) @ m_tot[kk])
        )
    v_force_row = v_force_row.at[2].add(ey_bar_z_area)

    if nonslip_layer > 1:
        v_force_row_slip = (
            2.0
            * jnp.pi
            * ey_bar_z[nonslip_layer - 2]
            * nu_bar_tz[nonslip_layer - 2]
            * (_lame_row(rad[nonslip_layer - 1] ** 2) @ m_tot[nonslip_layer - 1])
        )
        v_force_row_slip -= (
            2.0
            * jnp.pi
            * ey_bar_z[0]
            * nu_bar_tz[0]
            * (_lame_row(rad[0] ** 2) @ m_tot[0])
        )
        for kk in range(1, nonslip_layer - 1):
            v_force_row_slip += (
                2.0
                * jnp.pi
                * (ey_bar_z[kk - 1] * nu_bar_tz[kk - 1] - ey_bar_z[kk] * nu_bar_tz[kk])
                * (_lame_row(rad[kk] ** 2) @ m_tot[kk])
            )
        v_force_row_slip = v_force_row_slip.at[4].add(ey_bar_z_area_slip)
    else:
        # `base.py:4145-4147`: with no slip layers the fourth condition is vacuous, and
        # a unit entry in the `eps_z_slip` column is what keeps the 4x4 non-singular.
        v_force_row_slip = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])

    # ---- boundary conditions and the 4x4 solve (`base.py:4149-4198`, writeup §9-10) -
    zero = jnp.zeros(())
    bc_outer = jnp.stack([
        (1.0 + nu_bar_t[nlayers - 1]) * rad[nlayers] ** 2,
        -1.0 + nu_bar_t[nlayers - 1],
        nu_bar_zt[nlayers - 1] * rad[nlayers] ** 2,
        zero,
        zero,
    ])
    if nonslip_layer > 1:
        # The innermost layer is a slip layer, so its axial strain is `eps_z_slip`.
        bc_inner = jnp.stack([
            (1.0 + nu_bar_t[0]) * rad[0] ** 2,
            -1.0 + nu_bar_t[0],
            zero,
            zero,
            nu_bar_zt[0] * rad[0] ** 2,
        ])
    else:
        bc_inner = jnp.stack([
            (1.0 + nu_bar_t[0]) * rad[0] ** 2,
            -1.0 + nu_bar_t[0],
            nu_bar_zt[0] * rad[0] ** 2,
            zero,
            zero,
        ])
    bc_inner @= m_tot[0]

    m_bc = jnp.stack([
        bc_outer,
        bc_inner,
        v_force_row.at[3].add(-v_force),
        v_force_row_slip,
    ])

    # Column 3 multiplies the constant `1.0`, so it is the right-hand side; the other
    # four columns are the unknowns `(A, B, eps_z, eps_z_slip)`.
    m_toinv = jnp.concatenate([m_bc[:, :3], m_bc[:, 4:5]], axis=1)
    solution = jnp.linalg.solve(m_toinv, -m_bc[:, 3])
    a_vec_solution = jnp.stack([
        solution[0],
        solution[1],
        solution[2],
        jnp.ones(()),
        solution[3],
    ])

    # ---- the radial distributions (`base.py:4200-4232`) ---------------------------
    # PROCESS recomputes `a_vec_layer` from `a_vec_solution` at the end of every
    # iteration rather than accumulating, so layer `ii`'s vector is a closed form:
    # `m_ext[ii+1] @ m_tot[ii+1] @ a_vec_solution`, and the outermost is the solution
    # itself.
    layer_vectors = [None] * nlayers
    layer_vectors[nlayers - 1] = a_vec_solution
    for ii in range(nlayers - 2, -1, -1):
        layer_vectors[ii] = m_ext[ii + 1] @ (m_tot[ii + 1] @ a_vec_solution)

    offsets = jnp.arange(n_radial_array)
    rradius_parts, sigr_parts, sigt_parts, sigz_parts = [], [], [], []
    str_r_parts, str_t_parts, str_z_parts, deflect_parts = [], [], [], []
    for ii in range(nlayers):
        a_layer = layer_vectors[ii][0]
        b_layer = layer_vectors[ii][1]
        dradius = (rad[ii + 1] - rad[ii]) / (n_radial_array - 1)
        radius = rad[ii] + dradius * offsets

        f_int_a_plot = 0.5 * f_lin_fac[ii] * (rad[ii + 1] ** 2 - radius**2) + f_rec_fac[
            ii
        ] * jnp.log(rad[ii + 1] / radius)
        f_int_b_plot = 0.25 * f_lin_fac[ii] * (
            rad[ii + 1] ** 4 - radius**4
        ) + 0.5 * f_rec_fac[ii] * (rad[ii + 1] ** 2 - radius**2)
        a_plot = a_layer - 0.5 / ey_bar_t[ii] * f_int_a_plot
        b_plot = b_layer + 0.5 / ey_bar_t[ii] * f_int_b_plot

        str_r = a_plot - b_plot / radius**2
        str_t = a_plot + b_plot / radius**2
        # A slip layer takes `eps_z_slip`, everything else the force-carrying `eps_z`.
        # `ii` and `nonslip_layer` are both static, so this is a Python `if`.
        str_z = jnp.full(
            (n_radial_array,),
            a_vec_solution[4] if ii < nonslip_layer - 1 else a_vec_solution[2],
        )

        rradius_parts.append(radius)
        deflect_parts.append(a_plot * radius + b_plot / radius)
        str_r_parts.append(str_r)
        str_t_parts.append(str_t)
        str_z_parts.append(str_z)
        sigr_parts.append(
            ey_bar_t[ii] * (str_r + nu_bar_t[ii] * str_t + nu_bar_zt[ii] * str_z)
        )
        sigt_parts.append(
            ey_bar_t[ii] * (str_t + nu_bar_t[ii] * str_r + nu_bar_zt[ii] * str_z)
        )
        sigz_parts.append(ey_bar_z[ii] * (str_z + nu_bar_tz[ii] * (str_r + str_t)))

    return (
        jnp.concatenate(rradius_parts),
        jnp.concatenate(sigr_parts),
        jnp.concatenate(sigt_parts),
        jnp.concatenate(sigz_parts),
        jnp.concatenate(str_r_parts),
        jnp.concatenate(str_t_parts),
        jnp.concatenate(str_z_parts),
        jnp.concatenate(deflect_parts),
    )


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


def tf_stress_extended_plane_strain_bucked_case(
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
    vforce_inboard_tot,
    n_tf_graded_layers=1,
    n_radial_array=N_RADIAL_ARRAY,
):
    """Peak Tresca stresses and winding-pack strain, extended-plane-strain solver.

    Ports `TFCoil.stresscl`, `process/models/tfcoil/base.py:2222-3274`, on the
    `(i_tf_sup, i_tf_stress_model, i_tf_bucking) == (1, 0, 1)` cell -- the one both
    tracked spherical tokamaks take (`spherical_tokamak_eval.IN.DAT:350`,
    `st_regression.IN.DAT:1223`). `i_tf_stress_model == 2` reaches the same solver
    through the same `elif` (`base.py:3005`) and differs only in two zero-bore guards
    this cell cannot reach; it is nonetheless refused in `indat.py` rather than aliased
    here, because "same code path on the tracked values" is a measurement nobody has
    made for a file that sets `2`.

    **The sibling of `tf_stress_plane_stress_bucked_case`, not a switch on it.** Five
    things change with `i_tf_stress_model`, and only one of them is a formula:

    1. The solver is `extended_plane_strain`, which additionally **reads** the axial
       moduli and Poisson's ratios (`eyoung_axial`, `poisson_axial`) that the
       plane-stress arm assembles and never uses, and the total inboard tension
       `.superconducting_tfcoil.vforce_inboard_tot` in place of one coil's
       `.tfcoil.vforce`.
    2. The vertical stress is *solved for*, radially resolved, rather than being
       `vforce / (steel area)` broadcast over the whole leg -- so
       `.tfcoil.a_tf_coil_inboard_case` and `.tfcoil.a_tf_turn_steel`, the two areas
       that divide it there, are **not read here at all**.
    3. `.tfcoil.str_wp` comes off `str_tf_z` at the winding pack's inner face
       (`base.py:3034`) instead of off that uniform stress (`:2988`).
    4. The vertical unsmearing factor is `eyoung_steel / eyoung_wp_axial_eff` rather
       than `1.0` (`base.py:3083-3087`), so the axial stress is lifted onto the steel
       conduit as well as the two transverse ones.
    5. `.tfcoil.casestr` and `.tfcoil.insstrain` are **not owned**. `stresscl` sets
       both only inside the `i_tf_stress_model == 1` branch (`base.py:2991-2998`) and
       leaves them at the `None` they were initialised to (`:2520-2521`); the store at
       `superconducting.py:2224-2231` then writes that `None` over the
       `DataStructure`'s `0.0`. Their only reader anywhere is the printer
       (`base.py:3646`, `:3653`) -- measured, not assumed -- so nothing downstream can
       tell, but a port that returned a number for them would be inventing one.

    That is four differences in the reads-set and one in the owns-set, which is why this
    is a second occupant and not a `i_tf_stress_model` keyword
    (`_audit/next_steps.md` §14.2).

    Parameters
    ----------
    vforce_inboard_tot :
        Inboard vertical tension summed over the whole TF coil set (N), from
        `tf_field_and_force_clamped_joints`. **Not** `.tfcoil.vforce`: the extended
        plane strain solver's axial condition is on the layer stack's total tension,
        and `base.py:3021` passes `vforce_inboard_tot` there.

    Every other parameter is the corresponding one of
    `tf_stress_plane_stress_bucked_case`, read the same way and used for the same thing;
    see that function's docstring.

    Returns
    -------
    :
        `(sig_tf_wp, sig_tf_case, str_wp)` -- peak Tresca stress in the winding pack and
        in the nose case (Pa), and the vertical strain at the winding pack's inner face.
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
    # The same stack the plane-stress arm builds -- nose casing, winding pack, plasma-
    # side case -- with the *axial* pair carried through this time, because this solver
    # reads it.
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

    # PROCESS Issue #1509, `base.py:2949-2962`. On *this* arm the round trip is not
    # neutral: the scaled modulus enters the solve through `ey_z`, and only the
    # resulting vertical stress is scaled back out -- which is `stress.md`'s OQ3, now
    # live.
    f_tf_stress_front_case = (
        a_tf_plasma_case
        / rad_tf_coil_inboard_toroidal_half
        / (radtf[n_tf_layer] ** 2 - radtf[n_tf_layer - 1] ** 2)
    )
    eyoung_axial[n_tf_layer - 1] *= f_tf_stress_front_case

    def _stack(values):
        return jnp.stack([jnp.asarray(v, dtype=float) for v in values])

    _, sig_tf_r, sig_tf_t, sig_tf_z, _, _, str_tf_z, _ = extended_plane_strain(
        nu_t=_stack(poisson_trans),
        nu_zt=_stack(poisson_axial),
        ey_t=_stack(eyoung_trans),
        ey_z=_stack(eyoung_axial),
        rad=_stack(radtf),
        d_curr=_stack(jeff),
        v_force=vforce_inboard_tot,
        i_tf_bucking=N_TF_BUCKING,
        n_radial_array=n_radial_array,
    )

    # ---- the strain in the conductor (`base.py:3034`) ------------------------------
    # The *first* station of the winding pack layer, i.e. its inner face -- not a peak
    # and not an average.
    str_wp = str_tf_z[N_TF_BUCKING * n_radial_array]

    # ---- unsmearing (`base.py:3068-3131`) -----------------------------------------
    # `i_tf_sup == 1` with a plane-*strain* model: all three factors bite.
    fac_sig_z = eyoung_steel / eyoung_wp_axial_eff
    fac_sig_t = fac_sig_r = eyoung_wp_stiffest_leg / eyoung_wp_trans_eff
    wp_slice = slice(
        N_TF_BUCKING * n_radial_array,
        (N_TF_BUCKING + n_tf_graded_layers) * n_radial_array,
    )
    sig_tf_r = sig_tf_r.at[wp_slice].multiply(fac_sig_r)
    sig_tf_t = sig_tf_t.at[wp_slice].multiply(fac_sig_t)
    sig_tf_z = sig_tf_z.at[wp_slice].multiply(fac_sig_z)

    # The front case's axial stiffness correction, undone (`base.py:3123-3131`).
    sig_tf_z = sig_tf_z.at[
        (N_TF_BUCKING + n_tf_graded_layers) * n_radial_array :
    ].divide(f_tf_stress_front_case)

    # ---- the reduction (`base.py:3196-3231`) --------------------------------------
    s_shear_tf = tresca_stress(sig_tf_r, sig_tf_t, sig_tf_z)
    peak = s_shear_tf[_layer_peak_indices(s_shear_tf, n_tf_layer, n_radial_array)]
    return peak[N_TF_BUCKING], peak[N_TF_BUCKING - 1], str_wp
