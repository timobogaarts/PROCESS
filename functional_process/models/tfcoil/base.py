"""Pure-functional port of `process/models/tfcoil/base.py` -- the device-agnostic TF
coil layer that `SuperconductingTFCoil.run_base_superconducting_tf` reaches by
inheritance.

Audit record: `functional_process/_audit/units/models/tfcoil/base.md`.

**Scope is the minimal closure of `.tokamak.cicc_superconducting_tf_coil`'s ten boundary
reads** (`_audit/tokamak_boundary.md`), not the whole file. In scope and ported here:
`circumference`, `tf_global_geometry` (split three ways, see below), `tf_current`,
`tf_coil_shape_inner`, `tf_coil_self_inductance`, `tf_stored_magnetic_energy`,
`generic_tf_coil_area_and_masses`, plus the one inline formula
`run_base_tf` writes for `.tfcoil.r_b_tf_inboard_peak`
(`process/models/tfcoil/base.py:166-171`).

Deliberately **out** of scope, with reasons:

- `tf_field_and_force` (`base.py:1623`) -- feeds only `stresscl`; none of the ten
  boundary reads depends on it, and it carries three more switches (`i_tf_sup`,
  `itart`, `i_cp_joints`).
- `stresscl` and the elasticity helpers (`base.py:2222-4670`) -- `numba.njit`,
  ~2400 lines, and every output is a stress, none of which is on the boundary.
- `cntrpst` (`base.py:1211`) -- the TART centrepost, `itart == 1` only.
- `he_density`/`he_cp`/`he_visco`/`he_th_cond`/`al_th_cond` (`base.py:1827-2065`) --
  CoolProp-backed property lookups reached only from `cntrpst`. **They are not on this
  slot's CoolProp path**; see `functional_process/models/tfcoil/quench.py` for the one
  that is.

## `tf_global_geometry` is three nodes here, not one

`process/models/tfcoil/base.py:214-372` computes eleven quantities under three
independent switches, and the three groups of outputs are disjoint -- nothing computed
after a branch is read by another branch. Splitting is therefore semantics-preserving
and removes two invented edges the composite node would have declared:

| group | switch | node(s) here |
|---|---|---|
| the nine unconditional geometry outputs (only `a_tf_inboard_total` branches) | `i_tf_case_geom` | `TfGlobalGeometryCircularCase` / `TfGlobalGeometryStraightCase` |
| `.tfcoil.dr_tf_plasma_case` | `i_f_dr_tf_plasma_case` | `DrTfPlasmaCaseFromInput` / `DrTfPlasmaCaseFromFraction` |
| `.tfcoil.dx_tf_side_case_min` | `tfc_sidewall_is_fraction` | `DxTfSideCaseMinFromFraction` only -- see below |

**`tfc_sidewall_is_fraction == False` gets no node at all.** The branch is
`dx_tf_side_case_min = data.tfcoil.dx_tf_side_case_min` (`base.py:358`) -- a verbatim
read-back of the field it is about to be written to, i.e. an identity. Conditional
ownership, exactly the shape `models/power/thermal_cryo.py`'s
`calculate_p_fw_blkt_coolant_pump_mw` records: on that arm the field is a run input and
nothing produces it. `large_tokamak_eval.IN.DAT` does not set `tfc_sidewall_is_fraction`,
so it takes the `False` default (`tfcoil_variables.py:95`) and this is the live arm.

**`.tfcoil.dr_tf_plasma_case` is a genuine self-loop on the `False` arm**, and unlike
the sidewall one it is *not* an identity: `base.py:328` reads the entering value and
`base.py:333-340` raises it to `(r_tf_inboard_in + dr_tf_inboard) * (1 - cos(pi/n))`
when it is below that. `DrTfPlasmaCaseFromInput` is therefore a `FixedPointFunction`
(`step`, minting `^cond.tfcoil.dr_tf_plasma_case`), which is what cottax's "a node may
not read what it owns" requires. Its fixed point is reached in one step because
`jnp.maximum(x, m)` is idempotent in `x` and `m` does not depend on `x`; at the
reference point the clamp binds (`dr_tf_plasma_case` defaults to `0.0`,
`tfcoil_variables.py:77`, and is not in the input file) so the derivative with respect
to its own entering value is exactly zero there.

## A PROCESS defect, ported faithfully

`base.py:344-346` calls `logger.error("dr_tf_plasma_case too small to accommodate the
WP, forced to minimum value")` **unconditionally** -- it sits outside the `if` at 333
whose body it describes. Logging only, no value effect, so the port drops it (a pure
function does not log); recorded in `base.md` as defect **D1** rather than silently
fixed.
"""

import jax
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    From,
    OutputInto,
)

from functional_process.paths import build, physics, superconducting_tfcoil, tfcoil
from functional_process.vocabulary import constants

N_TF_INDUCTANCE_INTERVALS = 100
"""`NINTERVALS` in `tf_coil_self_inductance` (`process/models/tfcoil/base.py:2118`).

A quadrature resolution, not a model choice -- kind (b) in
`_audit/switch_elimination_design.md` §3 -- so it stays a module constant rather than
becoming a port or an occupant.
"""

F_STRAIGHT_TF_INBOARD = 0.6
"""`FSTRAIGHT` in `tf_coil_shape_inner` (`process/models/tfcoil/base.py:491`): the
fraction of the inboard half-height that is straight."""


# ---------------------------------------------------------------------------
# `circumference` -- `process/models/tfcoil/base.py:1185-1209`
# ---------------------------------------------------------------------------


def circumference(aaa, bbb):
    """Ramanujan's ellipse-circumference approximation (m). Unchanged.

    Ports `TFCoil.circumference`, `process/models/tfcoil/base.py:1185-1209`.

    Parameters
    ----------
    aaa, bbb :
        The two semi-axes of the ellipse `(x/a)^2 + (y/b)^2 = 1`.

    Returns
    -------
    :
        Approximate circumference (m).
    """
    hh = (aaa - bbb) ** 2 / (aaa + bbb) ** 2
    return jnp.pi * (aaa + bbb) * (1.0 + (3.0 * hh) / (10.0 + jnp.sqrt(4.0 - 3.0 * hh)))


# ---------------------------------------------------------------------------
# `tf_global_geometry`, split three ways -- see the module docstring
# ---------------------------------------------------------------------------


def calculate_tf_global_geometry(
    *,
    n_tf_coils,
    r_tf_inboard_out,
    r_tf_inboard_in,
    r_tf_outboard_mid,
    dr_tf_outboard,
    a_tf_inboard_total,
):
    """The nine outputs of `tf_global_geometry` that no switch touches.

    Ports `process/models/tfcoil/base.py:278-317`, with `a_tf_inboard_total` lifted out
    as an argument because it is the one quantity `i_tf_case_geom` decides
    (`base.py:282-293`) -- the two arms below supply it.

    Parameters
    ----------
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    r_tf_inboard_out, r_tf_inboard_in :
        Outer/inner radius of the inboard leg (m). `.build.*`.
    r_tf_outboard_mid :
        Mid-plane radius of the outboard leg (m). `.build.r_tf_outboard_mid`.
    dr_tf_outboard :
        Radial thickness of the outboard leg (m). `.build.dr_tf_outboard`.
    a_tf_inboard_total :
        Total inboard-leg mid-plane cross-section (m2), from the `i_tf_case_geom` arm.

    Returns
    -------
    :
        `(rad_tf_coil_inboard_toroidal_half, tan_theta_coil, a_tf_inboard_total,
        r_tf_outboard_in, r_tf_outboard_out, dx_tf_inboard_out_toroidal,
        a_tf_leg_outboard, dr_tf_full_midplane, dr_tf_internal_midplane)`.
    """
    rad_tf_coil_inboard_toroidal_half = jnp.pi / n_tf_coils
    tan_theta_coil = jnp.tan(rad_tf_coil_inboard_toroidal_half)

    dx_tf_inboard_out_toroidal = (
        2.0 * r_tf_inboard_out * jnp.sin(rad_tf_coil_inboard_toroidal_half)
    )

    r_tf_outboard_in = r_tf_outboard_mid - (dr_tf_outboard * 0.5)
    r_tf_outboard_out = r_tf_outboard_mid + (dr_tf_outboard * 0.5)

    a_tf_leg_outboard = dx_tf_inboard_out_toroidal * dr_tf_outboard

    dr_tf_full_midplane = r_tf_outboard_out - r_tf_inboard_in
    dr_tf_internal_midplane = r_tf_outboard_in - r_tf_inboard_out

    return (
        rad_tf_coil_inboard_toroidal_half,
        tan_theta_coil,
        a_tf_inboard_total,
        r_tf_outboard_in,
        r_tf_outboard_out,
        dx_tf_inboard_out_toroidal,
        a_tf_leg_outboard,
        dr_tf_full_midplane,
        dr_tf_internal_midplane,
    )


def a_tf_inboard_total_circular_case(*, r_tf_inboard_out, r_tf_inboard_in):
    """`i_tf_case_geom == TFPlasmaCaseType.CIRCULAR` (0). `base.py:284`."""
    return jnp.pi * (r_tf_inboard_out**2 - r_tf_inboard_in**2)


def a_tf_inboard_total_straight_case(*, n_tf_coils, r_tf_inboard_out, r_tf_inboard_in):
    """`i_tf_case_geom == TFPlasmaCaseType.STRAIGHT` (1). `base.py:287-293`."""
    rad_tf_coil_inboard_toroidal_half = jnp.pi / n_tf_coils
    return (
        n_tf_coils
        * jnp.sin(rad_tf_coil_inboard_toroidal_half)
        * jnp.cos(rad_tf_coil_inboard_toroidal_half)
        * r_tf_inboard_out**2
        - jnp.pi * r_tf_inboard_in**2
    )


def calculate_tf_global_geometry_circular_case(
    *,
    n_tf_coils,
    r_tf_inboard_out,
    r_tf_inboard_in,
    r_tf_outboard_mid,
    dr_tf_outboard,
):
    """`tf_global_geometry`'s nine unswitched outputs, circular front case."""
    return calculate_tf_global_geometry(
        n_tf_coils=n_tf_coils,
        r_tf_inboard_out=r_tf_inboard_out,
        r_tf_inboard_in=r_tf_inboard_in,
        r_tf_outboard_mid=r_tf_outboard_mid,
        dr_tf_outboard=dr_tf_outboard,
        a_tf_inboard_total=a_tf_inboard_total_circular_case(
            r_tf_inboard_out=r_tf_inboard_out, r_tf_inboard_in=r_tf_inboard_in
        ),
    )


def calculate_tf_global_geometry_straight_case(
    *,
    n_tf_coils,
    r_tf_inboard_out,
    r_tf_inboard_in,
    r_tf_outboard_mid,
    dr_tf_outboard,
):
    """`tf_global_geometry`'s nine unswitched outputs, straight front case."""
    return calculate_tf_global_geometry(
        n_tf_coils=n_tf_coils,
        r_tf_inboard_out=r_tf_inboard_out,
        r_tf_inboard_in=r_tf_inboard_in,
        r_tf_outboard_mid=r_tf_outboard_mid,
        dr_tf_outboard=dr_tf_outboard,
        a_tf_inboard_total=a_tf_inboard_total_straight_case(
            n_tf_coils=n_tf_coils,
            r_tf_inboard_out=r_tf_inboard_out,
            r_tf_inboard_in=r_tf_inboard_in,
        ),
    )


def dr_tf_plasma_case_minimum(*, r_tf_inboard_in, dr_tf_inboard, n_tf_coils):
    """The floor `tf_global_geometry` imposes on the plasma-facing case thickness (m).

    `process/models/tfcoil/base.py:333-340`: enough radial room that the winding pack
    does not clip the edges of the front case.
    """
    return (r_tf_inboard_in + dr_tf_inboard) * (1.0 - jnp.cos(jnp.pi / n_tf_coils))


def dr_tf_plasma_case_from_input(
    *, dr_tf_plasma_case, r_tf_inboard_in, dr_tf_inboard, n_tf_coils
):
    """`i_f_dr_tf_plasma_case == False`: the input value, clamped from below.

    Ports `process/models/tfcoil/base.py:326-340`'s `else` arm plus the clamp. The
    entering `dr_tf_plasma_case` is the same `.tfcoil.dr_tf_plasma_case` the result is
    written back to -- see the module docstring for why the node is a
    `FixedPointFunction`.
    """
    return jnp.maximum(
        dr_tf_plasma_case,
        dr_tf_plasma_case_minimum(
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_inboard=dr_tf_inboard,
            n_tf_coils=n_tf_coils,
        ),
    )


def dr_tf_plasma_case_from_fraction(
    *, f_dr_tf_plasma_case, dr_tf_inboard, r_tf_inboard_in, n_tf_coils
):
    """`i_f_dr_tf_plasma_case == True`: a fraction of the inboard thickness, clamped.

    Ports `process/models/tfcoil/base.py:323-340`'s `if` arm plus the clamp. No
    self-read: the entering `.tfcoil.dr_tf_plasma_case` is never consulted on this arm.
    """
    return jnp.maximum(
        f_dr_tf_plasma_case * dr_tf_inboard,
        dr_tf_plasma_case_minimum(
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_inboard=dr_tf_inboard,
            n_tf_coils=n_tf_coils,
        ),
    )


def dx_tf_side_case_min_from_fraction(
    *, casths_fraction, r_tf_inboard_in, dr_tf_nose_case, n_tf_coils
):
    """`tfc_sidewall_is_fraction == True`. `process/models/tfcoil/base.py:352-356`."""
    return (
        casths_fraction
        * (r_tf_inboard_in + dr_tf_nose_case)
        * jnp.tan(jnp.pi / n_tf_coils)
    )


# ---------------------------------------------------------------------------
# `run_base_tf`'s inline `.tfcoil.r_b_tf_inboard_peak` -- `base.py:166-171`
# ---------------------------------------------------------------------------


def calculate_r_b_tf_inboard_peak(
    *,
    r_tf_inboard_out,
    dr_tf_plasma_case,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    """Radius at which the peak inboard toroidal field occurs (m).

    Written inline in `run_base_tf` (`process/models/tfcoil/base.py:166-171`) rather
    than in a `calculate_*`; it is a real `DataStructure` field
    (`.tfcoil.r_b_tf_inboard_peak`) with a real consumer (`tf_current`), so it gets its
    own node here instead of becoming a local of a bigger one.
    """
    return (
        r_tf_inboard_out
        - dr_tf_plasma_case
        - dx_tf_wp_insulation
        - dx_tf_wp_insertion_gap
    )


# ---------------------------------------------------------------------------
# `tf_current` -- `process/models/tfcoil/base.py:374-426`
# ---------------------------------------------------------------------------


def tf_current(
    *,
    n_tf_coils,
    b_plasma_toroidal_on_axis,
    rmajor,
    r_b_tf_inboard_peak,
    a_tf_inboard_total,
):
    """Peak inboard field and the TF current that produces it. Unchanged.

    Ports `TFCoil.tf_current`, `process/models/tfcoil/base.py:374-426`. No switch.

    Returns
    -------
    :
        `(b_tf_inboard_peak_symmetric, c_tf_total, c_tf_coil, j_tf_coil_full_area)` --
        peak axisymmetric field (T), total TF current (A), current per coil (A),
        inboard-leg average current density (A/m2).
    """
    b_tf_inboard_peak_symmetric = (
        b_plasma_toroidal_on_axis * rmajor / r_b_tf_inboard_peak
    )
    c_tf_total = (
        b_tf_inboard_peak_symmetric * r_b_tf_inboard_peak * (2 * jnp.pi / constants.RMU0)
    )
    c_tf_coil = c_tf_total / n_tf_coils
    j_tf_coil_full_area = c_tf_total / a_tf_inboard_total

    return b_tf_inboard_peak_symmetric, c_tf_total, c_tf_coil, j_tf_coil_full_area


# ---------------------------------------------------------------------------
# `tf_coil_shape_inner` -- `process/models/tfcoil/base.py:428-580`
# ---------------------------------------------------------------------------


def _d_shape_length_and_arcs(*, r_tf_arc, z_tf_arc, dr_tf_inboard):
    """The four-arc sum shared by both `i_tf_shape == D_SHAPE`, `itart == 0` arms.

    Ports `process/models/tfcoil/base.py:519-526` verbatim, including the accumulation
    order (`len_tf_coil` starts at `z[0] - z[4]` and each quarter-circumference is added
    in arc order) so the float64 result is bit-comparable with PROCESS's.
    """
    tfa = jnp.abs(r_tf_arc[1:] - r_tf_arc[:-1])
    tfb = jnp.abs(z_tf_arc[1:] - z_tf_arc[:-1])

    len_tf_coil = z_tf_arc[0] - z_tf_arc[4]
    for ii in range(4):
        aa = tfa[ii] + 0.5 * dr_tf_inboard
        bb = tfb[ii] + 0.5 * dr_tf_inboard
        len_tf_coil = len_tf_coil + 0.25 * circumference(aa, bb)

    return len_tf_coil, tfa, tfb


def tf_coil_shape_inner_d_shape_single_null(
    *,
    r_tf_inboard_out,
    rmajor,
    rminor,
    r_tf_outboard_in,
    z_tf_inside_half,
    z_tf_top,
    dr_tf_inboard,
):
    """`i_tf_shape == D_SHAPE (1)`, `itart == 0`, `i_single_null == SINGLE_NULL (1)`.

    Ports `process/models/tfcoil/base.py:498-526` taking the `else` at line 512. This
    is `large_tokamak_eval.IN.DAT`'s arm: it sets `i_single_null = 1` (line 307) and
    `itart` takes its `0` default, which makes `process/core/init.py:775-776` promote
    `i_tf_shape` from `DEFAULT (0)` to `D_SHAPE (1)`.

    Returns
    -------
    :
        `(len_tf_coil, tfa, tfb, r_tf_arc, z_tf_arc)` -- total inner-edge length (m),
        the four arc semi-axes in R and Z (m), and the five arc points (m).
    """
    r_arc_1 = rmajor - 0.2 * rminor
    r_tf_arc = jnp.stack([
        r_tf_inboard_out,
        r_arc_1,
        r_tf_outboard_in,
        r_arc_1,
        r_tf_inboard_out,
    ])
    z_tf_arc = jnp.stack([
        F_STRAIGHT_TF_INBOARD * (z_tf_top - dr_tf_inboard),
        z_tf_top - dr_tf_inboard,
        jnp.zeros_like(z_tf_top),
        -z_tf_inside_half,
        -F_STRAIGHT_TF_INBOARD * z_tf_inside_half,
    ])

    len_tf_coil, tfa, tfb = _d_shape_length_and_arcs(
        r_tf_arc=r_tf_arc, z_tf_arc=z_tf_arc, dr_tf_inboard=dr_tf_inboard
    )
    return len_tf_coil, tfa, tfb, r_tf_arc, z_tf_arc


def tf_coil_shape_inner_d_shape_double_null(
    *,
    r_tf_inboard_out,
    rmajor,
    rminor,
    r_tf_outboard_in,
    z_tf_inside_half,
    dr_tf_inboard,
):
    """`i_tf_shape == D_SHAPE (1)`, `itart == 0`, `i_single_null == DOUBLE_NULL (0)`.

    Ports `process/models/tfcoil/base.py:498-526` taking the `if` at line 506. Note
    what changes with the null count and what does not: this arm reads
    `.build.z_tf_inside_half` only and **not** `.build.z_tf_top`, which is the invented
    edge a single node carrying `i_single_null` as a static kwarg would have declared.
    """
    r_arc_1 = rmajor - 0.2 * rminor
    r_tf_arc = jnp.stack([
        r_tf_inboard_out,
        r_arc_1,
        r_tf_outboard_in,
        r_arc_1,
        r_tf_inboard_out,
    ])
    z_tf_arc = jnp.stack([
        F_STRAIGHT_TF_INBOARD * z_tf_inside_half,
        z_tf_inside_half,
        jnp.zeros_like(z_tf_inside_half),
        -z_tf_inside_half,
        -F_STRAIGHT_TF_INBOARD * z_tf_inside_half,
    ])

    len_tf_coil, tfa, tfb = _d_shape_length_and_arcs(
        r_tf_arc=r_tf_arc, z_tf_arc=z_tf_arc, dr_tf_inboard=dr_tf_inboard
    )
    return len_tf_coil, tfa, tfb, r_tf_arc, z_tf_arc


def tf_coil_shape_inner_picture_frame_tart(
    *,
    r_cp_top,
    r_tf_outboard_in,
    z_tf_inside_half,
    z_tf_top,
    dr_tf_inboard,
    r_tf_outboard_mid,
):
    """`i_tf_shape == PICTURE_FRAME (2)`, `itart == 1` -- both ST regression files' arm.

    Ports `process/models/tfcoil/base.py:551-578` taking the `itart == 1` sub-branches
    at `:555-556` and `:575-578`. `spherical_tokamak_eval.IN.DAT` (`itart = 1` line 283,
    `i_tf_shape = 2` line 357) and `st_regression.IN.DAT` (`itart = 1` line 66,
    `i_tf_shape = 2` line 803) both reach it: the D-shape/`itart == 1` branch at
    `:528-549` is guarded by `i_tf_shape == D_SHAPE` and is *not* what a spherical
    tokamak with a picture-frame coil takes, whatever the name "the TART arm" suggests.

    **`tfa` and `tfb` are returned as exact zeros, and that is PROCESS's answer, not a
    stub.** `:495-496` allocates them `np.zeros(4)` and the picture-frame branch never
    assigns an element -- there are no elliptical arcs to take semi-axes of. The two
    fields are written to `data` anyway (`base.py:186-190`), so the occupant owns them
    and must produce the zeros. Defect D5 in `base.md`: `.tfcoil.tfa`/`.tfcoil.tfb`
    silently mean "unset" on this arm rather than being absent.

    Note also that `z_tf_arc[2] = 0` while `r_tf_arc[2] = r_tf_outboard_in`, so points
    1, 2 and 3 share a radius: the "arcs" of the picture frame are a straight outboard
    leg, and only `r_tf_arc[0]`/`r_tf_arc[4]` (`r_cp_top`) differ.

    Returns
    -------
    :
        `(len_tf_coil, tfa, tfb, r_tf_arc, z_tf_arc)` -- total inner-edge length (m),
        the four arc semi-axes in R and Z (m, identically zero here), and the five arc
        points (m).
    """
    r_tf_arc = jnp.stack([
        r_cp_top,
        r_tf_outboard_in,
        r_tf_outboard_in,
        r_tf_outboard_in,
        r_cp_top,
    ])
    z_tf_arc = jnp.stack([
        z_tf_top - dr_tf_inboard,
        z_tf_top - dr_tf_inboard,
        jnp.zeros_like(z_tf_top),
        -z_tf_inside_half,
        -z_tf_inside_half,
    ])

    len_tf_coil = z_tf_inside_half + z_tf_top + 2.0 * (r_tf_outboard_mid - r_cp_top)

    tfa = jnp.zeros_like(r_tf_arc[:4])
    tfb = jnp.zeros_like(z_tf_arc[:4])

    return len_tf_coil, tfa, tfb, r_tf_arc, z_tf_arc


# ---------------------------------------------------------------------------
# `tf_coil_self_inductance` -- `process/models/tfcoil/base.py:2066-2191`
# ---------------------------------------------------------------------------


def _inductance_arc_sum(*, x0, y0, ai, bi, ao, bo, z_straight, outward):
    """One of the two 100-interval sweeps in `tf_coil_self_inductance`.

    Ports `process/models/tfcoil/base.py:2133-2181`. `lax.scan` rather than a
    vectorised `r = x0 -/+ (k + 1/2) * dr`, deliberately: PROCESS advances the major
    radius by repeated `r -= dr` / `r += dr`, and reproducing that recurrence exactly
    keeps the port's float64 answer bit-comparable *and*, more importantly, keeps the
    `x0 - r < ai` branch flipping at exactly the same interval PROCESS flips it at. A
    reassociated `r` differs in the last bits, which is harmless for the sum and not
    harmless at a branch boundary.

    Parameters
    ----------
    x0, y0 :
        The arc-join point (`r_tf_arc[1]`, `z_tf_arc[1]`), m.
    ai, bi, ao, bo :
        Inner/outer semi-axes of the arc, m.
    z_straight :
        The straight-section contribution added outside the bore -- `z_tf_arc[0]` on the
        inboard sweep (`base.py:2151`) and `0.0` on the outboard one (`base.py:2176`).
    outward :
        `True` to step `r` outward (`r += dr`, the outboard arc), `False` inward.
    """
    dr = ao / N_TF_INDUCTANCE_INTERVALS
    step = dr if outward else -dr
    r_start = x0 + dr / 2.0 if outward else x0 - dr / 2.0

    def body(carry, _):
        r, acc = carry
        b = constants.RMU0 / (2.0 * jnp.pi * r)

        offset = (r - x0) if outward else (x0 - r)
        in_bore = offset < ai

        # `1 - ((r - x0) / ai)**2` is negative exactly where `in_bore` is false, so the
        # inner square root is guarded by substitution (the double-`where` idiom) rather
        # than clipped -- a clip would leave a zero-derivative artefact on the taken
        # branch, and a bare `sqrt` would leak NaN into the gradient of the untaken one.
        inner_arg = 1.0 - ((r - x0) / ai) ** 2
        s_inner = bi * jnp.sqrt(jnp.where(in_bore, inner_arg, 1.0))
        s_outer = bo * jnp.sqrt(1.0 - ((r - x0) / ao) ** 2)

        h_bore = jnp.where(in_bore, y0 + s_inner, 0.0)
        h_thick = jnp.where(in_bore, s_outer - (y0 + s_inner), s_outer + z_straight)

        acc = acc + b * dr * (2.0 * h_bore + h_thick)
        return (r + step, acc), None

    (_, total), _ = jax.lax.scan(
        body,
        (r_start, jnp.zeros_like(x0 * ao)),
        None,
        length=N_TF_INDUCTANCE_INTERVALS,
    )
    return total


def tf_coil_self_inductance_d_shape(*, dr_tf_inboard, r_tf_arc, z_tf_arc):
    """`itart == 0` and `i_tf_shape == 1`: the two-sweep numerical integration.

    Ports `process/models/tfcoil/base.py:2120-2181`. This arm reads **only** the coil
    arc geometry and the inboard thickness; `z_tf_inside_half`, `dr_tf_outboard`,
    `r_tf_outboard_mid` and `r_tf_inboard_mid` belong to the other arm and are not read
    here (PROCESS's own unit test says as much: *"the following 4 params are not used
    ... however they are provided because the function is Numba compiled"*,
    `tests/unit/models/tfcoil/test_tfcoil.py:597-599`).

    Returns
    -------
    :
        Self-inductance of one TF coil (H).
    """
    x0 = r_tf_arc[1]
    y0 = z_tf_arc[1]

    ai_in = r_tf_arc[1] - r_tf_arc[0]
    bi_in = (z_tf_arc[1] - z_tf_arc[3]) / 2.0 - z_tf_arc[0]

    ai_out = r_tf_arc[2] - r_tf_arc[1]
    bi_out = (z_tf_arc[1] - z_tf_arc[3]) / 2.0

    inboard = _inductance_arc_sum(
        x0=x0,
        y0=y0,
        ai=ai_in,
        bi=bi_in,
        ao=ai_in + dr_tf_inboard,
        bo=bi_in + dr_tf_inboard,
        z_straight=z_tf_arc[0],
        outward=False,
    )
    outboard = _inductance_arc_sum(
        x0=x0,
        y0=y0,
        ai=ai_out,
        bi=bi_out,
        ao=ai_out + dr_tf_inboard,
        bo=bi_out + dr_tf_inboard,
        z_straight=0.0,
        outward=True,
    )
    return inboard + outboard


def tf_coil_self_inductance_picture_frame(
    *, z_tf_inside_half, dr_tf_outboard, r_tf_outboard_mid, r_tf_inboard_mid
):
    """Everything that is not (`itart == 0` and `i_tf_shape == 1`): the closed form.

    Ports `process/models/tfcoil/base.py:2182-2189`. Named for what PROCESS's own
    comment calls it ("Picture frame TF coil"), but note it is the source's `else`, so
    it also catches `itart == 1`.
    """
    return (
        (z_tf_inside_half + dr_tf_outboard)
        * constants.RMU0
        / jnp.pi
        * jnp.log(r_tf_outboard_mid / r_tf_inboard_mid)
    )


# ---------------------------------------------------------------------------
# `tf_stored_magnetic_energy` -- `process/models/tfcoil/base.py:582-634`
# ---------------------------------------------------------------------------


def tf_stored_magnetic_energy(*, ind_tf_coil, c_tf_total, n_tf_coils):
    """Stored magnetic energy of the TF coil set. Unchanged.

    Ports `TFCoil.tf_stored_magnetic_energy`, `process/models/tfcoil/base.py:582-634`.
    No switch.

    Returns
    -------
    :
        `(e_tf_magnetic_stored_total, e_tf_magnetic_stored_total_gj,
        e_tf_coil_magnetic_stored)` -- total (J), total (GJ), per coil (J).
    """
    e_tf_magnetic_stored_total = 0.5 * ind_tf_coil * c_tf_total**2
    e_tf_magnetic_stored_total_gj = e_tf_magnetic_stored_total * 1.0e-9
    e_tf_coil_magnetic_stored = e_tf_magnetic_stored_total / n_tf_coils

    return (
        e_tf_magnetic_stored_total,
        e_tf_magnetic_stored_total_gj,
        e_tf_coil_magnetic_stored,
    )


# ---------------------------------------------------------------------------
# `generic_tf_coil_area_and_masses` -- `process/models/tfcoil/base.py:2193-2218`
# ---------------------------------------------------------------------------


def generic_tf_coil_area_and_masses(
    *,
    r_tf_inboard_out,
    r_tf_inboard_in,
    rad_tf_coil_inboard_toroidal_half,
    tan_theta_coil,
    len_tf_coil,
    r_tf_inboard_mid,
    r_tf_outboard_mid,
):
    """Cryostat-facing TF surface areas. Unchanged apart from `np.` -> `jnp.`.

    Ports `TFCoil.generic_tf_coil_area_and_masses`,
    `process/models/tfcoil/base.py:2193-2218`. Despite the name it computes no mass at
    all on this path -- every mass in the superconducting chain is
    `superconducting_tf_coil_areas_and_masses`'s
    (`functional_process/models/tfcoil/superconducting.py`). `wbtf` is a local of the
    source too, so it stays a local here.

    This is the sole producer of `.tfcoil.tfcryoarea`, one of the slot's ten boundary
    reads; `models/stellarator/coils/calculate.py::TfCryoArea` produces the *same*
    `VarPath` on the stellarator from a different formula -- see `base.md` §"Shared with
    the stellarator".

    Returns
    -------
    :
        `(tfocrn, tficrn, tfcryoarea)` -- inner/outer corner half-widths (m) and the
        total two-shell cryostat surface area (m2).
    """
    wbtf = (
        r_tf_inboard_out * jnp.sin(rad_tf_coil_inboard_toroidal_half)
        - r_tf_inboard_in * tan_theta_coil
    )
    tfocrn = r_tf_inboard_in * tan_theta_coil
    tficrn = tfocrn + wbtf

    tfcryoarea = (
        2.0 * len_tf_coil * 2.0 * jnp.pi * 0.5 * (r_tf_inboard_mid + r_tf_outboard_mid)
    )
    return tfocrn, tficrn, tfcryoarea


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class TfGlobalGeometry(ExplicitFunction):
    """The family that owns `tf_global_geometry`'s nine unswitched outputs.

    One occupant per `i_tf_case_geom` value. The two arms' reads-sets are identical
    (both read the same five fields); they are still separate classes, because a switch
    selects an occupant and never a static kwarg -- `_audit/next_steps.md` §14.2, and
    the `istore` precedent it names for two arms differing only in a literal.
    """


class TfGlobalGeometryCircularCase(TfGlobalGeometry):
    """`i_tf_case_geom == TFPlasmaCaseType.CIRCULAR` (0) -- `large_tokamak_eval`'s arm.

    The input file does not set `i_tf_case_geom`, so it takes the `0` default
    (`process/data_structure/tfcoil_variables.py:234`).
    """

    rad_tf_coil_inboard_toroidal_half = OutputInto(superconducting_tfcoil)
    tan_theta_coil = OutputInto(superconducting_tfcoil)
    a_tf_inboard_total = OutputInto(tfcoil)
    r_tf_outboard_in = OutputInto(superconducting_tfcoil)
    r_tf_outboard_out = OutputInto(superconducting_tfcoil)
    dx_tf_inboard_out_toroidal = OutputInto(tfcoil)
    a_tf_leg_outboard = OutputInto(tfcoil)
    dr_tf_full_midplane = OutputInto(tfcoil)
    dr_tf_internal_midplane = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        r_tf_inboard_out=From(build),
        r_tf_inboard_in=From(build),
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
    ):
        return calculate_tf_global_geometry_circular_case(
            n_tf_coils=n_tf_coils,
            r_tf_inboard_out=r_tf_inboard_out,
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_outboard_mid=r_tf_outboard_mid,
            dr_tf_outboard=dr_tf_outboard,
        )


class TfGlobalGeometryStraightCase(TfGlobalGeometry):
    """`i_tf_case_geom == TFPlasmaCaseType.STRAIGHT` (1)."""

    rad_tf_coil_inboard_toroidal_half = OutputInto(superconducting_tfcoil)
    tan_theta_coil = OutputInto(superconducting_tfcoil)
    a_tf_inboard_total = OutputInto(tfcoil)
    r_tf_outboard_in = OutputInto(superconducting_tfcoil)
    r_tf_outboard_out = OutputInto(superconducting_tfcoil)
    dx_tf_inboard_out_toroidal = OutputInto(tfcoil)
    a_tf_leg_outboard = OutputInto(tfcoil)
    dr_tf_full_midplane = OutputInto(tfcoil)
    dr_tf_internal_midplane = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        r_tf_inboard_out=From(build),
        r_tf_inboard_in=From(build),
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
    ):
        return calculate_tf_global_geometry_straight_case(
            n_tf_coils=n_tf_coils,
            r_tf_inboard_out=r_tf_inboard_out,
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_outboard_mid=r_tf_outboard_mid,
            dr_tf_outboard=dr_tf_outboard,
        )


class DrTfPlasmaCaseFromInput(FixedPointFunction):
    """`i_f_dr_tf_plasma_case == False` -- `large_tokamak_eval`'s arm, and a self-loop.

    `step` reads the entering `.tfcoil.dr_tf_plasma_case` and writes the minted
    `^cond.tfcoil.dr_tf_plasma_case`; the `FixedPoint` this class also declares reads
    that copy and owns the real path, so neither piece reads what it owns. See the
    module docstring for why the loop is real (a clamp, not an identity) and why it
    closes in one step.
    """

    dr_tf_plasma_case = OutputInto(tfcoil)

    def step(
        self,
        dr_tf_plasma_case=From(tfcoil),
        r_tf_inboard_in=From(build),
        dr_tf_inboard=From(build),
        n_tf_coils=From(tfcoil),
    ):
        return dr_tf_plasma_case_from_input(
            dr_tf_plasma_case=dr_tf_plasma_case,
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_inboard=dr_tf_inboard,
            n_tf_coils=n_tf_coils,
        )


class DrTfPlasmaCaseFromFraction(ExplicitFunction):
    """`i_f_dr_tf_plasma_case == True`: a plain node, no loop.

    The fraction arm never reads the entering `.tfcoil.dr_tf_plasma_case`, so this one
    is an `ExplicitFunction` where its sibling has to be a `FixedPointFunction` -- the
    clearest possible demonstration that the loop is a property of one *arm*, not of
    the quantity.
    """

    dr_tf_plasma_case = OutputInto(tfcoil)

    def __call__(
        self,
        f_dr_tf_plasma_case=From(tfcoil),
        dr_tf_inboard=From(build),
        r_tf_inboard_in=From(build),
        n_tf_coils=From(tfcoil),
    ):
        return dr_tf_plasma_case_from_fraction(
            f_dr_tf_plasma_case=f_dr_tf_plasma_case,
            dr_tf_inboard=dr_tf_inboard,
            r_tf_inboard_in=r_tf_inboard_in,
            n_tf_coils=n_tf_coils,
        )


class DxTfSideCaseMinFromFraction(ExplicitFunction):
    """`tfc_sidewall_is_fraction == True`.

    The `False` arm has no node -- see the module docstring.
    """

    dx_tf_side_case_min = OutputInto(tfcoil)

    def __call__(
        self,
        casths_fraction=From(tfcoil),
        r_tf_inboard_in=From(build),
        dr_tf_nose_case=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return dx_tf_side_case_min_from_fraction(
            casths_fraction=casths_fraction,
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            n_tf_coils=n_tf_coils,
        )


class RBTfInboardPeak(ExplicitFunction):
    """cottax node: `run_base_tf`'s inline `.tfcoil.r_b_tf_inboard_peak`."""

    r_b_tf_inboard_peak = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        dr_tf_plasma_case=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return calculate_r_b_tf_inboard_peak(
            r_tf_inboard_out=r_tf_inboard_out,
            dr_tf_plasma_case=dr_tf_plasma_case,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )


class TfCurrent(ExplicitFunction):
    """cottax node: `tf_current`, ports declared. No switch, so no family."""

    b_tf_inboard_peak_symmetric = OutputInto(tfcoil)
    c_tf_total = OutputInto(tfcoil)
    c_tf_coil = OutputInto(superconducting_tfcoil)
    j_tf_coil_full_area = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        b_plasma_toroidal_on_axis=From(physics),
        rmajor=From(physics),
        r_b_tf_inboard_peak=From(tfcoil),
        a_tf_inboard_total=From(tfcoil),
    ):
        return tf_current(
            n_tf_coils=n_tf_coils,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            rmajor=rmajor,
            r_b_tf_inboard_peak=r_b_tf_inboard_peak,
            a_tf_inboard_total=a_tf_inboard_total,
        )


class TfCoilShape(ExplicitFunction):
    """The family that owns `.tfcoil.len_tf_coil` and the arc arrays.

    Three switches decide it -- `i_tf_shape`, `itart`, `i_single_null` -- and the arms
    read genuinely different variables (`r_cp_top` on the two `itart == 1` arms,
    `z_tf_top` on all but the D-shape double-null one, `r_tf_outboard_mid` on the
    picture frame). Three occupants are written: the two `i_tf_shape == D_SHAPE`,
    `itart == 0` arms and the `i_tf_shape == PICTURE_FRAME`, `itart == 1` one the two
    ST regression files take. See `base.md` for the UNPORTED list.
    """


class TfCoilShapeDShapeSingleNull(TfCoilShape):
    """`i_tf_shape == 1`, `itart == 0`, `i_single_null == 1` -- the reference arm."""

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        r_tf_outboard_in=From(superconducting_tfcoil),
        z_tf_inside_half=From(build),
        z_tf_top=From(build),
        dr_tf_inboard=From(build),
    ):
        return tf_coil_shape_inner_d_shape_single_null(
            r_tf_inboard_out=r_tf_inboard_out,
            rmajor=rmajor,
            rminor=rminor,
            r_tf_outboard_in=r_tf_outboard_in,
            z_tf_inside_half=z_tf_inside_half,
            z_tf_top=z_tf_top,
            dr_tf_inboard=dr_tf_inboard,
        )


class TfCoilShapeDShapeDoubleNull(TfCoilShape):
    """`i_tf_shape == 1`, `itart == 0`, `i_single_null == 0`.

    Does not read `z_tf_top`.
    """

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        r_tf_outboard_in=From(superconducting_tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
    ):
        return tf_coil_shape_inner_d_shape_double_null(
            r_tf_inboard_out=r_tf_inboard_out,
            rmajor=rmajor,
            rminor=rminor,
            r_tf_outboard_in=r_tf_outboard_in,
            z_tf_inside_half=z_tf_inside_half,
            dr_tf_inboard=dr_tf_inboard,
        )


class TfCoilShapePictureFrameTart(TfCoilShape):
    """`i_tf_shape == 2`, `itart == 1` -- both ST regression files' arm.

    The reads-set is the measurement: it reads `.build.r_cp_top` and
    `.build.r_tf_outboard_mid`, which neither D-shape sibling touches, and reads
    **neither** `.physics.rmajor` nor `.physics.rminor`, which both siblings do. A
    single node carrying `i_tf_shape`/`itart` as static kwargs would have declared all
    four edges on every arm; three of the four would have been invented.

    `.build.r_cp_top` has **no producer in this port** -- `process/models/build.py`'s
    `calculate_radial_build` writes it at `:1750-1813` and that slice is not ported --
    so it enters the ST graphs as a declared boundary input. See `base.md`'s dated
    section; it is a lost producer, recorded as one, not stubbed.
    """

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)

    def __call__(
        self,
        r_cp_top=From(build),
        r_tf_outboard_in=From(superconducting_tfcoil),
        z_tf_inside_half=From(build),
        z_tf_top=From(build),
        dr_tf_inboard=From(build),
        r_tf_outboard_mid=From(build),
    ):
        return tf_coil_shape_inner_picture_frame_tart(
            r_cp_top=r_cp_top,
            r_tf_outboard_in=r_tf_outboard_in,
            z_tf_inside_half=z_tf_inside_half,
            z_tf_top=z_tf_top,
            dr_tf_inboard=dr_tf_inboard,
            r_tf_outboard_mid=r_tf_outboard_mid,
        )


class TfCoilSelfInductance(ExplicitFunction):
    """The family that owns `.tfcoil.ind_tf_coil`. `(itart, i_tf_shape)` decides it."""


class TfCoilSelfInductanceDShape(TfCoilSelfInductance):
    """`itart == 0` and `i_tf_shape == 1` -- the reference arm.

    Reads three things; the composite PROCESS function takes nine, and the other six
    belong to the sibling arm. That gap is the measurement this split exists to make.
    """

    ind_tf_coil = OutputInto(tfcoil)

    def __call__(
        self,
        dr_tf_inboard=From(build),
        r_tf_arc=From(tfcoil),
        z_tf_arc=From(tfcoil),
    ):
        return tf_coil_self_inductance_d_shape(
            dr_tf_inboard=dr_tf_inboard, r_tf_arc=r_tf_arc, z_tf_arc=z_tf_arc
        )


class TfCoilSelfInductancePictureFrame(TfCoilSelfInductance):
    """Everything else (`i_tf_shape == 2`, or `itart == 1`): the closed form."""

    ind_tf_coil = OutputInto(tfcoil)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dr_tf_outboard=From(build),
        r_tf_outboard_mid=From(build),
        r_tf_inboard_mid=From(build),
    ):
        return tf_coil_self_inductance_picture_frame(
            z_tf_inside_half=z_tf_inside_half,
            dr_tf_outboard=dr_tf_outboard,
            r_tf_outboard_mid=r_tf_outboard_mid,
            r_tf_inboard_mid=r_tf_inboard_mid,
        )


class TfStoredMagneticEnergy(ExplicitFunction):
    """cottax node: `tf_stored_magnetic_energy`. Owns one of the slot's ten reads."""

    e_tf_magnetic_stored_total = OutputInto(tfcoil)
    e_tf_magnetic_stored_total_gj = OutputInto(tfcoil)
    e_tf_coil_magnetic_stored = OutputInto(tfcoil)

    def __call__(
        self,
        ind_tf_coil=From(tfcoil),
        c_tf_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return tf_stored_magnetic_energy(
            ind_tf_coil=ind_tf_coil, c_tf_total=c_tf_total, n_tf_coils=n_tf_coils
        )


class GenericTfCoilAreaAndMasses(ExplicitFunction):
    """cottax node: `generic_tf_coil_area_and_masses`. Owns `.tfcoil.tfcryoarea`."""

    tfocrn = OutputInto(tfcoil)
    tficrn = OutputInto(tfcoil)
    tfcryoarea = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        r_tf_inboard_in=From(build),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        len_tf_coil=From(tfcoil),
        r_tf_inboard_mid=From(build),
        r_tf_outboard_mid=From(build),
    ):
        return generic_tf_coil_area_and_masses(
            r_tf_inboard_out=r_tf_inboard_out,
            r_tf_inboard_in=r_tf_inboard_in,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            tan_theta_coil=tan_theta_coil,
            len_tf_coil=len_tf_coil,
            r_tf_inboard_mid=r_tf_inboard_mid,
            r_tf_outboard_mid=r_tf_outboard_mid,
        )
