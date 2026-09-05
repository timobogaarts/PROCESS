"""Harness cases for the ported TF coil stress chain
(`functional_process/cottax/tfcoil/stress.py`).

Audit record: `functional_process/_audit/units/models/tfcoil/stress.md`.

Six contracts: the four elasticity-smearing helpers, the `plane_stress` layer solver,
`tf_field_and_force` and `stresscl` itself. Where PROCESS's own
`tests/unit/models/tfcoil/test_tfcoil.py` has a parametrisation on the ported arm, the
**input** half is lifted verbatim and the expected half is not transcribed -- the
contract calls the PROCESS reference itself and diffs the two.

**Two of PROCESS's parametrisations are on arms this port does not have, and are not
used.** `test_tf_field_and_force`'s two samples are both `i_tf_sup = 0`, `itart = 1`,
`i_cp_joints = 1` -- the *resistive sliding-joint* arm, which is exactly the one
`indat.py` refuses -- so the clamped superconducting arm is sampled at the converged
point of `large_tokamak_nof` instead, plus fuzz. `test_plane_stress`'s third sample has
`rad[2] < rad[1]` (non-monotonic layer boundaries) and PROCESS's own test guards it with
`skip_if_incompatible_system` because the resulting matrix is ill-conditioned enough for
the answer to depend on the LAPACK build; comparing two solvers there measures the
condition number, not the port, so it is left out and said so here.

**The adapters drop PROCESS's `n` arguments rather than binding them.**
`eyoung_parallel_array` and `eyoung_t_nested_squares` take a member count *and* arrays
that may be longer than it (PROCESS's own sample passes `n = 4` with five-element
arrays); the port takes exactly the members, so the adapter slices. That is a signature
change, not a behaviour change -- `range(n)` never reads past `n` -- and it is recorded
in `stress.md` rather than hidden here.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.tfcoil.stress import (
    extended_plane_strain,
    eyoung_parallel,
    eyoung_parallel_array,
    eyoung_series,
    eyoung_t_nested_squares,
    plane_stress,
    tf_field_and_force_clamped_joints,
    tf_stress_extended_plane_strain_bucked_case,
    tf_stress_plane_stress_bucked_case,
)
from process.models.tfcoil import base as tfcoil_base
from process.models.tfcoil.base import TFCoil

# `large_tokamak_nof.IN.DAT` at PROCESS's converged point, measured with a `SingleRun`.
# The reference configuration for this unit: `i_tf_sup = 1`, `i_tf_stress_model = 1`,
# `i_tf_bucking = 1` (resolved from `-1`), `i_tf_turns_integer = 0`, `i_tf_tresca = 0`.
_CONVERGED_NOF = {
    "r_tf_inboard_in": 2.776781736333239,
    "r_tf_wp_inboard_inner": 2.978393629565446,
    "r_tf_wp_inboard_outer": 3.5493428322288025,
    "tan_theta_coil": 0.198912367379658,
    "rad_tf_coil_inboard_toroidal_half": 0.19634954084936207,
    "dr_tf_plasma_case": 0.07599050121231316,
    "a_tf_coil_inboard_steel": 0.6772330797567163,
    "a_tf_plasma_case": 0.07476512623743048,
    "a_tf_coil_nose_case": 0.25056108436158886,
    "eyoung_steel": 205000000000.0,
    "poisson_steel": 0.3,
    "eyoung_cond_axial": 0.0,
    "poisson_cond_axial": 0.3,
    "eyoung_cond_trans": 0.0,
    "poisson_cond_trans": 0.3,
    "eyoung_ins": 20000000000.0,
    "poisson_ins": 0.34,
    "eyoung_copper": 117000000000.0,
    "poisson_copper": 0.35,
    "dx_tf_turn_insulation": 0.0008,
    "dx_tf_wp_insertion_gap": 0.01,
    "dx_tf_wp_insulation": 0.008,
    "n_tf_coil_turns": 150.0258353015685,
    "dx_tf_turn_cable_space_eyoung": 0.04501894028884075,
    "dia_tf_turn_coolant_channel": 0.01,
    "f_a_tf_turn_cable_copper": 0.8997880037570971,
    "dx_tf_turn_steel": 0.008085108218129558,
    "dx_tf_side_case_average": 0.0783922143888241,
    "dx_tf_wp_toroidal_average": 1.141663084468358,
    "a_tf_coil_inboard_insulation": 0.056265716649540734,
    "a_tf_wp_steel": 0.2623909245570688,
    "a_tf_wp_conductor": 0.19774291703590527,
    "a_tf_wp_with_insulation": 0.651831627787397,
    "c_tf_total": 198249624.59807792,
    "vforce": 182709032.97324997,
    "a_tf_coil_inboard_case": 0.4148421551996475,
    "a_tf_turn_steel": 0.001748971595656402,
}
"""Every argument of `tf_stress_plane_stress_bucked_case` at that point."""

_CONVERGED_NOF_FORCE = {
    "r_tf_wp_inboard_outer": 3.5493428322288025,
    "r_tf_wp_inboard_inner": 2.978393629565446,
    "r_tf_outboard_in": 14.559557920183472,
    "dx_tf_wp_insulation": 0.008,
    "dx_tf_wp_insertion_gap": 0.01,
    "b_tf_inboard_peak_symmetric": 11.228002157011968,
    "c_tf_total": 198249624.59807792,
    "n_tf_coils": 16.0,
    "dr_tf_plasma_case": 0.07599050121231316,
    "rmajor": 8.000000000138815,
    "b_plasma_toroidal_on_axis": 4.956240617090718,
    "f_vforce_inboard": 0.5,
}
"""The same point, for `tf_field_and_force_clamped_joints`."""


# ---------------------------------------------------------------------------
# The elasticity-smearing helpers
# ---------------------------------------------------------------------------


class TestEyoungParallel(Tier1Contract):
    """`eyoung_parallel` -> `eyoung_parallel`, unchanged.

    PROCESS's own `test_eyoung_parallel` parametrisation
    (`tests/unit/models/tfcoil/test_tfcoil.py:1665-1691`), both samples. Both have
    `eyoung_j_1 == eyoung_j_2 == 0`, which is the branch this composite does *not*
    guard -- it is an area-weighted average and zero is not special in it.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = staticmethod(tfcoil_base.eyoung_parallel)
    ported = eyoung_parallel

    samples = [
        legacy_sample(
            "eyoung-parallel-baseline2018-1",
            eyoung_j_1=0.0,
            a_1=0.010000000000000002,
            poisson_j_perp_1=0.30000001192092896,
            eyoung_j_2=0.0,
            a_2=0.0,
            poisson_j_perp_2=0.0,
        ),
        legacy_sample(
            "eyoung-parallel-baseline2018-2",
            eyoung_j_1=0.0,
            a_1=0.020661087836601012,
            poisson_j_perp_1=0.30000001192092896,
            eyoung_j_2=0.0,
            a_2=0.010000000000000002,
            poisson_j_perp_2=0.30000001192092896,
        ),
    ]

    fuzz_bounds = {
        "eyoung_j_1": (0.0, 2.1e11),
        "a_1": (1.0e-3, 1.0),
        "poisson_j_perp_1": (0.2, 0.4),
        "eyoung_j_2": (0.0, 2.1e11),
        "a_2": (1.0e-3, 1.0),
        "poisson_j_perp_2": (0.2, 0.4),
    }


class TestEyoungSeries(Tier1Contract):
    """`eyoung_series` -> `eyoung_series`, with the zero guard as a double `jnp.where`.

    PROCESS's own `test_eyoung_series` parametrisation (`test_tfcoil.py:1846-1872`),
    both samples -- and both are on the **zero branch**, which is the one that matters:
    `eyoung_cond_axial` and `eyoung_cond_trans` are `0.0` on the reference run too, so
    this is not an edge case of the function but its normal operating point. The fuzz
    bounds start at `0.0` for the same reason, so roughly half the draws take it.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = staticmethod(tfcoil_base.eyoung_series)
    ported = eyoung_series

    samples = [
        legacy_sample(
            "eyoung-series-baseline2018-1",
            eyoung_j_1=0.0,
            l_1=0.003949573550844649,
            poisson_j_perp_1=0.30000001192092896,
            eyoung_j_2=117000000000.0,
            l_2=0.016711514285756363,
            poisson_j_perp_2=0.34999999999999998,
        ),
        legacy_sample(
            "eyoung-series-baseline2018-2",
            eyoung_j_1=0.0,
            l_1=0.020661087836601012,
            poisson_j_perp_1=0.30000001192092896,
            eyoung_j_2=0.0,
            l_2=0.010000000000000002,
            poisson_j_perp_2=0.29999999999999999,
        ),
    ]

    fuzz_bounds = {
        "eyoung_j_1": (0.0, 2.1e11),
        "l_1": (1.0e-4, 0.1),
        "poisson_j_perp_1": (0.2, 0.4),
        "eyoung_j_2": (0.0, 2.1e11),
        "l_2": (1.0e-4, 0.1),
        "poisson_j_perp_2": (0.2, 0.4),
    }


def _reference_eyoung_parallel_array(eyoung_j_in, a_in, poisson_j_perp_in):
    """`eyoung_parallel_array` with PROCESS's leading `n` supplied from the shapes."""
    return tfcoil_base.eyoung_parallel_array(
        len(eyoung_j_in),
        np.asarray(eyoung_j_in, dtype=float),
        np.asarray(a_in, dtype=float),
        np.asarray(poisson_j_perp_in, dtype=float),
    )


class TestEyoungParallelArray(Tier1Contract):
    """`eyoung_parallel_array` -> `eyoung_parallel_array`, `n` dropped.

    PROCESS's own `test_eyoung_parallel_array` sample (`test_tfcoil.py:1918-1940`),
    whose `n = 5` matches its five-element arrays exactly -- so nothing is sliced here
    and the adapter only reorders the count away.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = _reference_eyoung_parallel_array
    ported = eyoung_parallel_array

    samples = [
        legacy_sample(
            "eyoung-parallel-array-baseline2018",
            eyoung_j_in=np.array([
                205000000000.0,
                20000000000.0,
                117000000000.0,
                0.0,
                0.0,
            ]),
            a_in=np.array([
                0.29370123076207649,
                0.11646247019991701,
                0.13374756938078641,
                0.031609694578447076,
                0.1297552160314831,
            ]),
            poisson_j_perp_in=np.array([
                0.29999999999999999,
                0.34000000000000002,
                0.34999999999999998,
                0.30000001192092896,
                0.29999999999999999,
            ]),
        ),
    ]


def _reference_eyoung_t_nested_squares(eyoung_j_in, l_in, poisson_j_perp_in):
    """`eyoung_t_nested_squares` with `n` taken from the (already sliced) shapes."""
    return tfcoil_base.eyoung_t_nested_squares(
        len(eyoung_j_in),
        np.asarray(eyoung_j_in, dtype=float),
        np.asarray(l_in, dtype=float),
        np.asarray(poisson_j_perp_in, dtype=float),
    )


class TestEyoungTNestedSquares(Tier1Contract):
    """`eyoung_t_nested_squares` -> `eyoung_t_nested_squares`, `n` dropped.

    PROCESS's own `test_eyoung_t_nested_squares` sample (`test_tfcoil.py:1780-1804`)
    with its trailing member removed: that sample passes `n = 4` with five-element
    arrays, and PROCESS reads only the first four. Slicing at the call site rather than
    inside the port is what makes the dropped argument a signature change and not a
    behaviour change.

    The fourth return, `eyoung_stiffest`, is the one PROCESS's own test still asserts on
    (the other three assertions are commented out in `test_tfcoil.py:1832-1840`); this
    contract checks all four, because the reference is called rather than transcribed.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = _reference_eyoung_t_nested_squares
    ported = eyoung_t_nested_squares

    samples = [
        legacy_sample(
            "eyoung-t-nested-squares-baseline2018",
            eyoung_j_in=np.array([0.0, 0.0, 205000000000.0, 20000000000.0]),
            l_in=np.array([
                0.010000000000000002,
                0.020661087836601012,
                0.016,
                0.0041799999999999997,
            ]),
            poisson_j_perp_in=np.array([
                0.29999999999999999,
                0.30000001192092896,
                0.29999999999999999,
                0.34000000000000002,
            ]),
        ),
    ]


# ---------------------------------------------------------------------------
# The layer solver
# ---------------------------------------------------------------------------


def _reference_plane_stress(nu, rad, ey, j, n_radial_array):
    """`plane_stress` with `nlayers` supplied from the shapes, keyword for keyword."""
    return tfcoil_base.plane_stress(
        nu=np.asarray(nu, dtype=float),
        rad=np.asarray(rad, dtype=float),
        ey=np.asarray(ey, dtype=float),
        j=np.asarray(j, dtype=float),
        nlayers=len(j),
        n_radial_array=int(n_radial_array),
    )


class TestPlaneStress(Tier1Contract):
    """`plane_stress` -> `plane_stress`, `np.linalg.solve` becoming `jnp.linalg.solve`.

    PROCESS's own `test_plane_stress` first sample (`test_tfcoil.py:1373-1389`) and the
    layer stack the ported `stresscl` builds at `large_tokamak_nof`'s converged point.
    The third PROCESS sample is deliberately absent -- see this module's docstring.

    Both returns are 1500- and 300-element radial distributions, and the contract
    compares every element and (under `--fp-gradients`) differentiates every one of them
    with respect to every component of all four arrays, so this is the densest case in
    the module by a wide margin.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = _reference_plane_stress
    ported = plane_stress
    static_argnames = ("n_radial_array",)

    gradient_floor = 1.0e-8
    """The only contract in the port that sets it. Measured, at `--fp-gradients` on both
    samples: two rows need it, and neither is a disagreement about the model.

    `d(sigr[0])/d(nu[1])` is the one `gradient_safety` cannot reach. `sigr` at the
    innermost radius *is* the first boundary condition, so both implementations return
    exactly `0.0` and PROCESS's finite difference is exactly `0.0` with an error bar of
    exactly `0.0`; the port's `jacfwd`, propagating a tangent through the same cancelling
    expression, returns `-3.8e-10` against derivatives of order `1e6` in that column.
    The second, `d(sigr[91])/d(nu[1])`, misses by `1.5e-3` against an allowance of
    `1.4e-3` on a derivative of `-1.9e6` -- 8e-10 relative, and the same size as the
    value disagreement the two solves already show (`6.6e-7` on stresses of `5e7`,
    1.3e-14 relative).

    Both are the ill-conditioning PROCESS's own source warns about
    (`base.py:4404-4412`, "you can get above-floating point differences in the result of
    this function depending on system") and its own unit test guards with
    `skip_if_incompatible_system`. `1e-8` is ~4 orders above the worse of the two
    (`1.5e-3` against a column scale of `1.9e6` is `8e-10`) and 8 orders below a wrong
    derivative."""

    samples = [
        legacy_sample(
            "plane-stress-baseline2018",
            nu=np.array([0.29999999999999999, 0.30904421667064924, 0.29999999999999999]),
            rad=np.array([
                2.9939411851091102,
                3.5414797139565706,
                4.0876202904571599,
                4.1476202904571595,
            ]),
            ey=np.array([205000000000.0, 43126670035.025253, 205000000000.0]),
            j=np.array([0.0, 18097185.781970859, 0.0]),
            n_radial_array=100,
        ),
        legacy_sample(
            "plane-stress-large-tokamak-nof-converged",
            # The three layers `tf_stress_plane_stress_bucked_case` assembles at the
            # reference point: nose casing, winding pack, plasma-side case.
            nu=np.array([0.29999999999999999, 0.30348733486899632, 0.29999999999999999]),
            rad=np.array([
                2.776781736333239,
                2.9977681590072023,
                3.5724314013552201,
                3.6484219025675331,
            ]),
            ey=np.array([205000000000.0, 46455045800.47422, 205000000000.0]),
            j=np.array([0.0, 16713619.898244653, 0.0]),
            n_radial_array=100,
        ),
    ]


# ---------------------------------------------------------------------------
# `tf_field_and_force`
# ---------------------------------------------------------------------------


def _reference_tf_field_and_force(
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
    """`tf_field_and_force` pinned to the superconducting clamped-joint arm.

    `i_tf_sup = 1` and `i_cp_joints = 0` are the arm the port implements, and `itart` and
    `r_cp_top` are held at values that cannot reach the sliding-joint branch -- passing
    them would declare reads the ported node does not have. The fifth return,
    `f_vforce_inboard`, is dropped: on this arm PROCESS returns the argument unchanged,
    so comparing it would test the adapter rather than the port.
    """
    cforce, vforce, vforce_outboard, vforce_inboard_tot, _f_vforce_inboard = (
        TFCoil.tf_field_and_force(
            i_tf_sup=1,
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
            r_cp_top=0.0,
            itart=0,
            i_cp_joints=0,
            f_vforce_inboard=f_vforce_inboard,
        )
    )
    return cforce, vforce, vforce_outboard, vforce_inboard_tot


class TestTfFieldAndForceClampedJoints(Tier1Contract):
    """`tf_field_and_force` -> `tf_field_and_force_clamped_joints`.

    **No legacy sample**, and that is a measured statement rather than an omission:
    PROCESS's own `test_tf_field_and_force` has two, and both are `i_tf_sup = 0` with
    `itart = 1` and `i_cp_joints = 1` -- the resistive sliding-joint arm, which this port
    refuses. The converged point of `large_tokamak_nof` stands in for it, and the fuzz
    bounds are narrow around that point on purpose: the vertical-tension integral is a
    sum of six logarithms of radius ratios, so a draw that puts the outer winding-pack
    radius inside the inner one is `nan` on both sides and tests nothing.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = _reference_tf_field_and_force
    ported = tf_field_and_force_clamped_joints

    samples = [
        legacy_sample(
            "tf-field-and-force-large-tokamak-nof-converged", **_CONVERGED_NOF_FORCE
        )
    ]

    fuzz_bounds = {
        "r_tf_wp_inboard_inner": (2.5, 3.1),
        "r_tf_wp_inboard_outer": (3.3, 3.9),
        "r_tf_outboard_in": (12.0, 16.0),
        "b_tf_inboard_peak_symmetric": (8.0, 14.0),
        "c_tf_total": (1.0e8, 3.0e8),
        "dr_tf_plasma_case": (0.03, 0.12),
        "rmajor": (6.0, 10.0),
        "b_plasma_toroidal_on_axis": (3.0, 7.0),
        "f_vforce_inboard": (0.3, 0.7),
    }
    fuzz_fixed = {
        "dx_tf_wp_insulation": 0.008,
        "dx_tf_wp_insertion_gap": 0.01,
        "n_tf_coils": 16.0,
    }


# ---------------------------------------------------------------------------
# `stresscl`
# ---------------------------------------------------------------------------


def _reference_stresscl(*, n_radial_array=500, n_tf_graded_layers=1, **kwargs):
    """`stresscl` pinned to `(i_tf_sup, i_tf_stress_model, i_tf_bucking) == (1, 1, 1)`.

    Every argument the ported cell does not reach is supplied here rather than passed
    through, and each is inert on this arm: the nine `.pf_coil`/`.build` CS-layer
    arguments are read only at `i_tf_bucking >= 2` (`base.py:2531-2650`),
    `eyoung_res_tf_buck`/`eyoung_al`/`poisson_al`/`fcoolcp` only at `i_tf_sup != 1`,
    `vforce_inboard_tot` only at `i_tf_stress_model != 1`, and `i_tf_tresca` only for
    layers at or beyond `i_tf_bucking + 1`, which is neither of the two this node
    reports. Supplying them is what makes "the port declares no edge PROCESS does not
    make here" a tested claim rather than an asserted one -- a value that mattered would
    show up as a diff.

    `dx_tf_turn_cable_space_average` and `dr_tf_turn_cable_space` are the two arms of
    `i_tf_turns_integer`; the ported function takes whichever one applies as
    `dx_tf_turn_cable_space_eyoung`, so the adapter routes it to the averaged slot and
    pins the switch to `0`.

    Returns the five values the node owns, in the node's order.
    """
    out = TFCoil.stresscl(
        n_tf_layer=1 + n_tf_graded_layers + 1,
        n_radial_array=int(n_radial_array),
        n_tf_wp_stress_layers=5,
        i_tf_bucking=1,
        i_tf_sup=1,
        i_tf_stress_model=1,
        i_tf_tresca=0,
        i_tf_turns_integer=0,
        n_tf_graded_layers=n_tf_graded_layers,
        dx_tf_turn_cable_space_average=kwargs.pop("dx_tf_turn_cable_space_eyoung"),
        dr_tf_turn_cable_space=0.0,
        # Inert on this arm -- see the docstring.
        dr_bore=0.0,
        dr_cs=0.0,
        i_tf_inside_cs=0,
        dr_tf_inboard=0.0,
        dr_cs_tf_gap=0.0,
        i_pf_conductor=0,
        j_cs_flat_top_end=0.0,
        j_cs_pulse_start=0.0,
        c_pf_coil_turn_peak_input=np.ones(64),
        n_pf_coils_in_group=np.ones(12, dtype=np.int64),
        f_dr_dz_cs_turn=70.0 / 22.0,
        radius_cs_turn_corners=0.003,
        f_a_cs_turn_steel=0.8,
        a_cs_poloidal=1.0,
        eyoung_res_tf_buck=150000000000.0,
        eyoung_al=69000000000.0,
        poisson_al=0.35,
        fcoolcp=0.3,
        vforce_inboard_tot=0.0,
        **kwargs,
    )
    sig_tf_wp, sig_tf_case, _sig_tf_cs_bucked, str_wp, casestr, insstrain = out[27:33]
    return sig_tf_wp, sig_tf_case, str_wp, casestr, insstrain


class TestTfStressPlaneStressBuckedCase(Tier1Contract):
    """`stresscl` -> `tf_stress_plane_stress_bucked_case`.

    PROCESS's own `test_stresscl` sample (`test_tfcoil.py:1093-1204`), whose four
    switches are exactly the ported cell -- `i_tf_sup = 1`, `i_tf_stress_model = 1`,
    `i_tf_bucking = 1`, `i_tf_tresca = 0` -- and the converged point of
    `large_tokamak_nof`. The first is taken at `n_radial_array = 100` because that is
    what PROCESS's sample uses; the second at the `500` the pipeline actually runs.

    **The fuzz holds the geometry fixed and varies the material and load.** Ten of the
    thirty-seven arguments place the four layer boundaries, and the plane-stress system
    is singular unless they stay ordered and positive; a draw that crosses two of them
    is `nan` on both sides. The moduli, areas, currents and the vertical tension carry
    no such constraint, so those are what moves.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = _reference_stresscl
    ported = tf_stress_plane_stress_bucked_case
    static_argnames = ("n_radial_array", "n_tf_graded_layers")

    samples = [
        legacy_sample(
            "stresscl-baseline2018",
            r_tf_inboard_in=2.9939411851091102,
            r_tf_wp_inboard_inner=3.5185911851091101,
            r_tf_wp_inboard_outer=4.06120206347512,
            tan_theta_coil=0.19891236737965801,
            rad_tf_coil_inboard_toroidal_half=0.19634954084936207,
            dr_tf_plasma_case=0.060000000000000012,
            a_tf_coil_inboard_steel=1.2952181546825934,
            a_tf_plasma_case=0.18607458590131154,
            a_tf_coil_nose_case=0.70261616505511615,
            eyoung_steel=205000000000.0,
            poisson_steel=0.29999999999999999,
            eyoung_cond_axial=0.0,
            poisson_cond_axial=0.30000001192092896,
            eyoung_cond_trans=0.0,
            poisson_cond_trans=0.30000001192092896,
            eyoung_ins=20000000000.0,
            poisson_ins=0.34000000000000002,
            eyoung_copper=117000000000.0,
            poisson_copper=0.34999999999999998,
            dx_tf_turn_insulation=0.002,
            dx_tf_wp_insertion_gap=0.01,
            dx_tf_wp_insulation=0.0080000000000000019,
            n_tf_coil_turns=200.0,
            # PROCESS's sample is `i_tf_turns_integer = INTEGER`, so its cable-space
            # width for the smearing is the *radial* one.
            dx_tf_turn_cable_space_eyoung=0.030661087836601014,
            dia_tf_turn_coolant_channel=0.010000000000000002,
            f_a_tf_turn_cable_copper=0.80884,
            dx_tf_turn_steel=0.0080000000000000002,
            dx_tf_side_case_average=0.10396600719086938,
            dx_tf_wp_toroidal_average=1.299782604942499,
            a_tf_coil_inboard_insulation=0.11646247019991701,
            a_tf_wp_steel=0.29370123076207649,
            a_tf_wp_conductor=0.1653572639592335,
            a_tf_wp_with_insulation=0.70527618095271016,
            c_tf_total=236885604.60000002,
            vforce=250545611.13801825,
            a_tf_coil_inboard_case=1.0015169239205168,
            a_tf_turn_steel=0.0014685061538103825,
            n_radial_array=100,
        ),
        legacy_sample(
            "stresscl-large-tokamak-nof-converged", **_CONVERGED_NOF, n_radial_array=500
        ),
    ]

    fuzz_bounds = {
        "eyoung_steel": (1.5e11, 2.5e11),
        "poisson_steel": (0.25, 0.35),
        "eyoung_cond_axial": (0.0, 1.0e11),
        "poisson_cond_axial": (0.25, 0.35),
        "eyoung_cond_trans": (0.0, 1.0e11),
        "poisson_cond_trans": (0.25, 0.35),
        "eyoung_ins": (1.0e10, 3.0e10),
        "poisson_ins": (0.3, 0.4),
        "eyoung_copper": (1.0e11, 1.3e11),
        "poisson_copper": (0.3, 0.4),
        "f_a_tf_turn_cable_copper": (0.6, 0.95),
        "a_tf_wp_steel": (0.15, 0.35),
        "a_tf_wp_conductor": (0.12, 0.28),
        "c_tf_total": (1.5e8, 2.5e8),
        "vforce": (1.2e8, 2.4e8),
        "a_tf_coil_inboard_case": (0.3, 0.55),
        "a_tf_turn_steel": (1.2e-3, 2.4e-3),
        "n_tf_coil_turns": (100.0, 200.0),
    }
    fuzz_fixed = {
        key: value
        for key, value in _CONVERGED_NOF.items()
        if key
        not in {
            "eyoung_steel",
            "poisson_steel",
            "eyoung_cond_axial",
            "poisson_cond_axial",
            "eyoung_cond_trans",
            "poisson_cond_trans",
            "eyoung_ins",
            "poisson_ins",
            "eyoung_copper",
            "poisson_copper",
            "f_a_tf_turn_cable_copper",
            "a_tf_wp_steel",
            "a_tf_wp_conductor",
            "c_tf_total",
            "vforce",
            "a_tf_coil_inboard_case",
            "a_tf_turn_steel",
            "n_tf_coil_turns",
        }
    } | {"n_radial_array": 100}


# ---------------------------------------------------------------------------
# `extended_plane_strain`
# ---------------------------------------------------------------------------

# `spherical_tokamak_eval.IN.DAT` at PROCESS's converged point, measured with a
# `SingleRun`. Live switch values there: `i_tf_sup = 1`, `i_tf_stress_model = 0`,
# `i_tf_bucking = 1`, `i_tf_turns_integer = 0`, `i_tf_tresca = 0`, `itart = 1`,
# `i_cp_joints = 0`, `n_tf_graded_layers = 1`. That run's own three answers are
# `sig_tf_wp = 302391298.07593733`, `sig_tf_case = 390959416.2857399`,
# `str_wp = 0.0010498113096371342`, and `.tfcoil.casestr` / `.tfcoil.insstrain` are
# **`None`** in the converged `DataStructure` -- which is the measurement behind this
# arm owning three fields rather than five.
_CONVERGED_ST = {
    "r_tf_inboard_in": 0.30885540192106586,
    "r_tf_wp_inboard_inner": 0.4381694923304086,
    "r_tf_wp_inboard_outer": 1.1676646529646093,
    "tan_theta_coil": 0.2679491924311227,
    "rad_tf_coil_inboard_toroidal_half": 0.2617993877991494,
    "dr_tf_plasma_case": 0.041190748956456545,
    "a_tf_coil_inboard_steel": 0.2524170599339434,
    "a_tf_plasma_case": 0.017242815688178703,
    "a_tf_coil_nose_case": 0.026470758397272677,
    "eyoung_steel": 205000000000.0,
    "poisson_steel": 0.3,
    "eyoung_cond_axial": 0.0,
    "poisson_cond_axial": 0.3,
    "eyoung_cond_trans": 0.0,
    "poisson_cond_trans": 0.3,
    "eyoung_ins": 20000000000.0,
    "poisson_ins": 0.34,
    "eyoung_copper": 117000000000.0,
    "poisson_copper": 0.35,
    "dx_tf_turn_insulation": 0.0008,
    "dx_tf_wp_insertion_gap": 0.01,
    "dx_tf_wp_insulation": 0.018,
    "n_tf_coil_turns": 80.35714282107192,
    "dx_tf_turn_cable_space_eyoung": 0.025501408508296027,
    "dia_tf_turn_coolant_channel": 0.005,
    "f_a_tf_turn_cable_copper": 0.69,
    "dx_tf_turn_steel": 0.008,
    "dx_tf_side_case_average": 0.0763146550516272,
    "dx_tf_wp_toroidal_average": 0.27765265230686775,
    "a_tf_coil_inboard_insulation": 0.0443988263059861,
    "a_tf_wp_steel": 0.09736114275723116,
    "a_tf_wp_conductor": 0.027362227248714482,
    "a_tf_wp_with_insulation": 0.20254626619511037,
    "c_tf_total": 67499999.96970041,
    "vforce_inboard_tot": 551215676.8768669,
}
"""Every argument of `tf_stress_extended_plane_strain_bucked_case` at that point."""


def _reference_extended_plane_strain(
    nu_t, nu_zt, ey_t, ey_z, rad, d_curr, v_force, i_tf_bucking, n_radial_array
):
    """`extended_plane_strain` with `nlayers` supplied from the shapes."""
    return tfcoil_base.extended_plane_strain(
        nu_t=np.asarray(nu_t, dtype=float),
        nu_zt=np.asarray(nu_zt, dtype=float),
        ey_t=np.asarray(ey_t, dtype=float),
        ey_z=np.asarray(ey_z, dtype=float),
        rad=np.asarray(rad, dtype=float),
        d_curr=np.asarray(d_curr, dtype=float),
        v_force=float(v_force),
        nlayers=len(d_curr),
        n_radial_array=int(n_radial_array),
        i_tf_bucking=int(i_tf_bucking),
    )


_FNSF_SLIP_STACK = {
    "nu_t": np.array([
        0.30000000502169133,
        0.34000000000000002,
        0.29999999999999999,
        0.30901178507421895,
        0.29999999999999999,
    ]),
    "nu_zt": np.array([
        0.31163570564277626,
        0.34000000000000002,
        0.29999999999999999,
        0.31377709779186291,
        0.29999999999999999,
    ]),
    "ey_t": np.array([
        118643750000.0,
        2500000000.0,
        205000000000.0,
        43163597776.087654,
        205000000000.0,
    ]),
    "ey_z": np.array([
        48005309351.198608,
        2500000000.0,
        205000000000.0,
        124208626934.75433,
        390554854819.81116,
    ]),
    "rad": np.array([
        2.3322000000000003,
        2.8846200000000004,
        2.9346200000000002,
        3.4817726429672304,
        4.0290604740948242,
        4.0890604740948238,
    ]),
    "d_curr": np.array([0.0, 0.0, 0.0, 18343613.563061949, 0.0]),
    "v_force": 4051971733.3410816,
    "i_tf_bucking": 3,
    "n_radial_array": 100,
}
"""PROCESS's own `test_extended_plane_strain` second sample (`test_tfcoil.py:1521-1573`)
-- the FNSF five-layer stack at `i_tf_bucking = 3`, the only oracle anywhere for the
`nonslip_layer > 1` branches. Used by
`test_extended_plane_strain_matches_process_on_a_slip_stack` rather than by the contract;
that function says why."""


class TestExtendedPlaneStrain(Tier1Contract):
    """`extended_plane_strain` -> `extended_plane_strain`, `np.linalg.solve` on the
    4x4 becoming `jnp.linalg.solve`.

    One sample: the three-layer stack the ported `stresscl` builds at
    `spherical_tokamak_eval`'s converged point, which is the `nonslip_layer == 1` cell
    the node actually reaches and the one where the slip force row degenerates to
    `[0, 0, 0, 0, 1]`. Eight returns of `nlayers * n_radial_array` elements each -- 12000
    numbers, every one compared at `rtol = 1e-12`.

    **Neither of PROCESS's own two parametrisations is here, and each is absent for its
    own measured reason.**

    Its *first* sample has `rad[0] == 0`, so the innermost test point is `r = 0`; the
    inner boundary condition forces `B = 0` there, `b_plot / r` is `0 / 0`, and PROCESS's
    own test declares the first element of `sigr`, `sigt`, `sigz`, `str_r`, `str_t` and
    `r_deflect` to be `nan` (its `nan_init` field). The port returns `nan` in exactly
    those six places and nowhere else -- measured -- but `test_outputs_finite` exists
    precisely to fail on a `nan`, so carrying the sample would mean disabling the check
    that makes the others worth anything. The point is unreachable from the node in any
    case: `stresscl` raises `ProcessValueError` 245 at a zero inner radius unless
    `i_tf_stress_model == 2` (`base.py:2524-2527`), and `2` is refused in `indat.py`.

    Its *second* sample is `_FNSF_SLIP_STACK`, and it agrees -- but at `1e-13` of each
    array's *scale*, not of each element, which an elementwise `rtol` cannot express.
    `test_extended_plane_strain_matches_process_on_a_slip_stack` below carries it with
    the criterion that fits, and the numbers are there.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = _reference_extended_plane_strain
    ported = extended_plane_strain
    static_argnames = ("n_radial_array", "i_tf_bucking")

    gradient_floor = 1.0e-8
    """The second contract in the port to set one, and the same size as
    `TestPlaneStress`'s for the same reason. Measured, at `--fp-gradients`: five of the
    12000 x 22 derivative components need it, and every one is a *boundary condition*
    rather than a disagreement about the model.

    Three are on `sigr[1499]`, the outermost radial station -- which **is** the outer
    boundary condition, zero radial stress. The port's `jacfwd` propagates the same
    cancelling expression to an exact `0.0` (or `2.1e-19`) there while PROCESS's finite
    difference, straddling the same cancellation with `epsfcn = 1e-3`, reports `-5.7e-5`
    with a Richardson error bar of `8.7e-18` -- an error bar that is small because the
    three perturbed evaluations agree with each other, not because the derivative is
    known. The other two, on `sigr[8]` and `sigr[1497]`, miss by `1.7e-8` and `1.2e-7`
    *relative* on derivatives of `2.3e5` and `-4.9e2`.

    `1e-8` is the same order above the worst of these as it is on `TestPlaneStress`, and
    eight orders below a derivative that is wrong rather than cancelled."""

    samples = [
        legacy_sample(
            "extended-plane-strain-spherical-tokamak-eval-converged",
            # The three layers `tf_stress_extended_plane_strain_bucked_case` assembles
            # at that point: nose casing, winding pack, plasma-side case. `ey_z[2]` is
            # already scaled by `f_tf_stress_front_case` (`base.py:2960`), which is
            # what makes this solver read a modulus its plane-stress sibling ignores.
            nu_t=np.array([0.3, 0.3064907778275952, 0.3]),
            nu_zt=np.array([0.3, 0.3086653348093896, 0.3]),
            ey_t=np.array([205000000000.0, 70097031864.01279, 205000000000.0]),
            ey_z=np.array([
                205000000000.0,
                146170145629.56116,
                136363213453.44171,
            ]),
            rad=np.array([
                0.30885540192106586,
                0.4432860345278621,
                1.1812995718120933,
                1.2224903207685498,
            ]),
            d_curr=np.array([0.0, 17920369.937168237, 0.0]),
            v_force=551215676.8768669,
            i_tf_bucking=1,
            n_radial_array=500,
        ),
    ]


def test_extended_plane_strain_matches_process_on_a_slip_stack():
    """The `nonslip_layer > 1` branches, against PROCESS, at `1e-13` of array scale.

    `_FNSF_SLIP_STACK` is the only oracle in existence for the half of
    `extended_plane_strain` that `i_tf_bucking == 1` cannot reach: two axially decoupled
    inner layers, so the slip interface row of `m_ext`, the slip axial-force row, the
    `eps_z_slip` inner boundary condition and the `str_z = a_vec_solution[4]` branch all
    run. Without it those are written and untested.

    **It is here and not in `TestExtendedPlaneStrain.samples` because of what the
    disagreement is, and that was measured rather than guessed.** The 4x4 this sample
    produces has condition number `3.4e13` -- `extended_plane_strain` does no row
    equilibration, where `plane_stress` does (`base.py:4404-4419`) -- so `cond * eps` is
    `7.5e-3` and the *achievable* accuracy of the raw solution vector is far worse than
    anything either implementation delivers. What both actually deliver agrees to about
    `1e-13` of each output array's scale:

    | array | max abs difference | array scale | ratio |
    |---|---|---|---|
    | `sigr` | `2.2e-06` | `5.7e+07` | `3.8e-14` |
    | `sigt` | `6.8e-06` | `2.2e+08` | `3.1e-14` |
    | `sigz` | `1.7e-06` | `4.2e+08` | `4.1e-15` |
    | `str_r` | `1.8e-16` | `7.8e-03` | `2.3e-14` |
    | `str_t` | `5.1e-17` | `1.5e-03` | `3.5e-14` |
    | `str_z` | `5.4e-18` | `1.2e-03` | `4.6e-15` |
    | `r_deflect`| `1.2e-16` | `4.3e-03` | `2.8e-14` |

    Elementwise, two of those are unbounded in *relative* terms and neither is a
    disagreement about the model: `sigr[0]` **is** the inner boundary condition, so
    PROCESS returns a hard `0.0` and the port returns `-2.1e-6` from the same cancelling
    expression; and the largest relative miss on `sigz`, `1.25e-12`, is on an element of
    `1.3e6` inside an array whose scale is `4.2e8`. A per-element `rtol` that passed both
    would have to be `1e-11` or looser on 12000 numbers that mostly agree to `1e-15`,
    which is a worse test than this one. So the criterion here is the one that matches
    the claim: **`1e-13` of each array's own scale, checked array by array.**
    """
    reference = _reference_extended_plane_strain(**_FNSF_SLIP_STACK)
    ported = extended_plane_strain(**_FNSF_SLIP_STACK)
    names = (
        "rradius",
        "sigr",
        "sigt",
        "sigz",
        "str_r",
        "str_t",
        "str_z",
        "r_deflect",
    )
    for name, expected_raw, actual_raw in zip(names, reference, ported, strict=True):
        expected = np.asarray(expected_raw, dtype=float)
        actual = np.asarray(actual_raw, dtype=float)
        assert np.all(np.isfinite(actual)), name
        scale = np.max(np.abs(expected))
        error = np.max(np.abs(actual - expected))
        assert error <= 1.0e-13 * scale, (
            f"{name}: max |diff| {error:g} exceeds 1e-13 of the array scale {scale:g}"
        )


# ---------------------------------------------------------------------------
# `stresscl`, the `i_tf_stress_model == 0` arm
# ---------------------------------------------------------------------------


def _reference_stresscl_extended(*, n_radial_array=500, n_tf_graded_layers=1, **kwargs):
    """`stresscl` pinned to `(i_tf_sup, i_tf_stress_model, i_tf_bucking) == (1, 0, 1)`.

    The sibling of `_reference_stresscl`, and the inert-argument list moves by exactly
    the three arguments the arms disagree about: `vforce_inboard_tot` is **live** here
    and passed through, while `vforce`, `a_tf_coil_inboard_case` and `a_tf_turn_steel`
    join the inert set, because the vertical stress is solved for rather than divided
    out of a steel area (`base.py:2980-2989` runs only at `i_tf_stress_model == 1`).
    They are supplied as deliberately absurd values -- `vforce = 0`, both areas `1.0`,
    against a real machine's `4.6e7` and `1.6e-1` -- so that "the port declares no edge
    PROCESS does not make here" is tested rather than asserted: if any of the three
    reached the answer, the diff would be enormous rather than marginal.

    Returns the **three** values the node owns. `casestr` and `insstrain` come back as
    `None` from this arm and are not returned; that they do is checked by
    `test_casestr_and_insstrain_are_none_on_this_arm` below rather than left as a claim
    in prose.
    """
    sig_tf_wp, sig_tf_case, str_wp = _stresscl_extended_raw(
        n_radial_array=n_radial_array,
        n_tf_graded_layers=n_tf_graded_layers,
        **kwargs,
    )[:3]
    return sig_tf_wp, sig_tf_case, str_wp


def _stresscl_extended_raw(*, n_radial_array=500, n_tf_graded_layers=1, **kwargs):
    """`(sig_tf_wp, sig_tf_case, str_wp, casestr, insstrain)` from PROCESS itself."""
    out = TFCoil.stresscl(
        n_tf_layer=1 + n_tf_graded_layers + 1,
        n_radial_array=int(n_radial_array),
        n_tf_wp_stress_layers=5,
        i_tf_bucking=1,
        i_tf_sup=1,
        i_tf_stress_model=0,
        i_tf_tresca=0,
        i_tf_turns_integer=0,
        n_tf_graded_layers=n_tf_graded_layers,
        dx_tf_turn_cable_space_average=kwargs.pop("dx_tf_turn_cable_space_eyoung"),
        dr_tf_turn_cable_space=0.0,
        # Inert on this arm -- see `_reference_stresscl` for the CS-layer and
        # conductor-model group, and this function's docstring for the three that are
        # inert *here* and live on the plane-stress arm.
        dr_bore=0.0,
        dr_cs=0.0,
        i_tf_inside_cs=0,
        dr_tf_inboard=0.0,
        dr_cs_tf_gap=0.0,
        i_pf_conductor=0,
        j_cs_flat_top_end=0.0,
        j_cs_pulse_start=0.0,
        c_pf_coil_turn_peak_input=np.ones(64),
        n_pf_coils_in_group=np.ones(12, dtype=np.int64),
        f_dr_dz_cs_turn=70.0 / 22.0,
        radius_cs_turn_corners=0.003,
        f_a_cs_turn_steel=0.8,
        a_cs_poloidal=1.0,
        eyoung_res_tf_buck=150000000000.0,
        eyoung_al=69000000000.0,
        poisson_al=0.35,
        fcoolcp=0.3,
        vforce=0.0,
        a_tf_coil_inboard_case=1.0,
        a_tf_turn_steel=1.0,
        **kwargs,
    )
    sig_tf_wp, sig_tf_case, _cs_bucked, str_wp, casestr, insstrain = out[27:33]
    return sig_tf_wp, sig_tf_case, str_wp, casestr, insstrain


def test_casestr_and_insstrain_are_none_on_this_arm():
    """`stresscl` returns `None` for both at `i_tf_stress_model == 0`.

    The measurement behind `TfStressExtendedPlaneStrainBuckedCaseAveragedTurn` owning
    three fields where its plane-stress sibling owns five. Both are initialised to
    `None` (`base.py:2520-2521`) and assigned only inside the
    `i_tf_stress_model == 1` branch (`:2991-2998`); `superconducting.py:2224-2231`
    then writes that `None` over the `DataStructure`'s `0.0`, so a port that returned a
    number for either would be inventing one. Their only reader anywhere is the printer
    (`base.py:3646`, `:3653`), which is why nothing downstream reports it.

    The same `None` was observed end to end: a `SingleRun` on
    `spherical_tokamak_eval.IN.DAT` leaves `.tfcoil.casestr` and `.tfcoil.insstrain` at
    `None` in the converged `DataStructure`.
    """
    _, _, _, casestr, insstrain = _stresscl_extended_raw(
        n_radial_array=100, **_CONVERGED_ST
    )
    assert casestr is None
    assert insstrain is None


class TestTfStressExtendedPlaneStrainBuckedCase(Tier1Contract):
    """`stresscl` -> `tf_stress_extended_plane_strain_bucked_case`.

    `spherical_tokamak_eval`'s converged point at the `500` stations the pipeline runs,
    and the same geometry at `100`. The two agree to every printed digit here, which is
    worth recording rather than assuming: on this stack the peak Tresca stress in each
    of the three layers sits on a layer boundary, and `extended_plane_strain`'s grid is
    **closed** (`base.py:4160` divides by `n_radial_array - 1`, not `n_radial_array`),
    so both resolutions sample it exactly. `stress.md`'s OQ1 -- how much of the
    reported stress is a quadrature artefact -- is therefore *smaller* on this arm than
    on the plane-stress one, whose grid is open at the outer end and cannot land on the
    outer boundary at all.

    There is no PROCESS unit-test parametrisation for this cell: `test_stresscl`'s
    single sample is `i_tf_stress_model = 1`. The oracle is `stresscl` itself, called on
    the same converged point, and the port additionally reproduces that run's own
    `DataStructure` fields exactly -- `sig_tf_wp = 302391298.07593733`,
    `sig_tf_case = 390959416.2857399`, `str_wp = 0.0010498113096371342`, relative
    difference `0.0` on all three.

    **The fuzz holds the geometry fixed and varies the material and load**, for the same
    reason as on the sibling contract: the ten arguments that place the four layer
    boundaries have to stay ordered and positive or the transfer matrices are singular
    on both sides.
    """

    audit_record = "models/tfcoil/stress.md"
    reference = _reference_stresscl_extended
    ported = tf_stress_extended_plane_strain_bucked_case
    static_argnames = ("n_radial_array", "n_tf_graded_layers")

    samples = [
        legacy_sample(
            "stresscl-extended-spherical-tokamak-eval-converged",
            **_CONVERGED_ST,
            n_radial_array=500,
        ),
        legacy_sample(
            "stresscl-extended-spherical-tokamak-eval-converged-100",
            **_CONVERGED_ST,
            n_radial_array=100,
        ),
    ]

    fuzz_bounds = {
        "eyoung_steel": (1.5e11, 2.5e11),
        "poisson_steel": (0.25, 0.35),
        "eyoung_cond_axial": (0.0, 1.0e11),
        "poisson_cond_axial": (0.25, 0.35),
        "eyoung_cond_trans": (0.0, 1.0e11),
        "poisson_cond_trans": (0.25, 0.35),
        "eyoung_ins": (1.0e10, 3.0e10),
        "poisson_ins": (0.3, 0.4),
        "eyoung_copper": (1.0e11, 1.3e11),
        "poisson_copper": (0.3, 0.4),
        "f_a_tf_turn_cable_copper": (0.6, 0.95),
        "a_tf_wp_steel": (0.06, 0.14),
        "a_tf_wp_conductor": (0.018, 0.04),
        "c_tf_total": (5.0e7, 9.0e7),
        "vforce_inboard_tot": (3.5e8, 7.5e8),
        "n_tf_coil_turns": (60.0, 120.0),
    }
    fuzz_fixed = {
        key: value
        for key, value in _CONVERGED_ST.items()
        if key
        not in {
            "eyoung_steel",
            "poisson_steel",
            "eyoung_cond_axial",
            "poisson_cond_axial",
            "eyoung_cond_trans",
            "poisson_cond_trans",
            "eyoung_ins",
            "poisson_ins",
            "eyoung_copper",
            "poisson_copper",
            "f_a_tf_turn_cable_copper",
            "a_tf_wp_steel",
            "a_tf_wp_conductor",
            "c_tf_total",
            "vforce_inboard_tot",
            "n_tf_coil_turns",
        }
    } | {"n_radial_array": 100}
