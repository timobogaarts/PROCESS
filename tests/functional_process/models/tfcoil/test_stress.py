"""Harness cases for the ported TF coil stress chain
(`functional_process/models/tfcoil/stress.py`).

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
from functional_process.models.tfcoil.stress import (
    eyoung_parallel,
    eyoung_parallel_array,
    eyoung_series,
    eyoung_t_nested_squares,
    plane_stress,
    tf_field_and_force_clamped_joints,
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
