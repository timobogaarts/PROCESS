"""Harness cases for the ported tokamak build (`functional_process/models/build.py`).

Audit record: `functional_process/_audit/units/models/build.md`.

**Every reference here is a real PROCESS call, not a re-derivation.** `build.py` has no
`calculate_*` staticmethod for the radial or vertical build -- both are `self.data`-
mutating methods hundreds of lines long -- so the adapters below build a real
`DataStructure`, set the fields the sample names plus a fixed baseline for everything
else the method reads, call `Build.calculate_radial_build` /
`Build.calculate_vertical_build` /
`Build.divgeom` through it, and read the answer back off `data`. That is this harness's
standard "close the `data` back-door" technique (`_audit/test_harness.md` § As built) and
it is what makes these tier-1 cases genuine diffs rather than two copies of the same
transcription.

The baseline is `tests/regression/input_files/large_tokamak_eval.IN.DAT` at convergence,
read off a live `SingleRun` once and written down here as literals -- no cached solve, so
the suite stays fast. `BASELINE` therefore also *is* the evidence for every "the live
value is X" claim in the audit record.

**Two contracts are composites of several ported functions.**
`TestOutboardBuildChain` and `TestRippleSuperconducting` exist because PROCESS draws its
function boundaries in different places than the graph does: the outboard leg radius is
produced by a stretch of `calculate_radial_build` that calls the ripple fit twice, and
`dx_tf_wp_conductor_max` is a local inside that fit. Testing the chain end to end against
PROCESS's own boundary is the only 1:1 comparison available, and it is what justifies the
finer split the nodes use -- exactly the trade `models/physics/confinement_time.py`'s
`plasma_power_loss_mw` docstring already records.
"""

import copy

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.build import (
    calculate_divertor_geometry_conventional,
    calculate_dr_shld_vv_gap_outboard,
    calculate_dr_tf_inboard,
    calculate_dr_tf_outboard_superconducting,
    calculate_dr_tf_wp_with_insulation,
    calculate_dx_tf_wp_conductor_max_superconducting,
    calculate_r_shld_inboard_inner,
    calculate_r_shld_outboard_outer,
    calculate_r_tf_outboard_mid,
    calculate_r_tf_outboard_mid_unrippled,
    calculate_z_plasma_xpoint,
    calculate_z_tf_inside_half,
    plasma_outboard_edge_toroidal_ripple_fitted,
)
from process.core.model import DataStructure
from process.models.build import Build

# ---------------------------------------------------------------------------
# The baseline: `large_tokamak_eval.IN.DAT` at convergence.
#
# Every value below was read off a live `process.main.SingleRun` of
# `tests/regression/input_files/large_tokamak_eval.IN.DAT`. Only fields
# `calculate_radial_build` / `calculate_vertical_build` / `divgeom` actually read are
# listed; everything else stays at its `DataStructure` default.
# ---------------------------------------------------------------------------

BASELINE = {
    "physics": {
        "rmajor": 8.0,
        "rminor": 2.6666666666666665,
        "kappa": 1.85,
        "triang": 0.5,
        "i_single_null": 1,
        "itart": 0,
    },
    "build": {
        # radial
        "dr_bore": 2.003843190236783,
        "dr_cs": 0.546816593988753,
        "dr_cs_tf_gap": 0.08,
        "dr_tf_inboard": 1.2,
        "i_tf_inside_cs": 0,
        "i_cs_precomp": 1,
        "fseppc": 350000000.0,
        "fcspc": 0.6,
        "sigallpc": 300000000.0,
        "dr_shld_vv_gap_inboard": 0.02,
        "dr_vv_inboard": 0.3,
        "dr_shld_thermal_inboard": 0.05,
        "dr_shld_inboard": 0.3,
        "dr_shld_outboard": 0.8,
        "dr_shld_blkt_gap": 0.02,
        "dr_blkt_inboard": 0.7,
        "dr_blkt_outboard": 1.0,
        "dr_fw_inboard": 0.018000000000000002,
        "dr_fw_outboard": 0.018000000000000002,
        "dr_fw_plasma_gap_inboard": 0.25,
        "dr_fw_plasma_gap_outboard": 0.25,
        "dr_vv_outboard": 0.3,
        "gapomin": 0.234,
        "dr_shld_thermal_outboard": 0.05,
        "dr_tf_shld_gap": 0.05,
        "f_dr_tf_outboard_inboard": 1.19,
        "i_r_cp_top": 0,
        # vertical
        "dz_fw_plasma_gap": 0.6,
        "dz_shld_upper": 0.6,
        "dz_shld_lower": 0.7,
        "dz_vv_upper": 0.3,
        "dz_vv_lower": 0.3,
        "dz_shld_vv_gap": 0.163,
        "dz_shld_thermal": 0.05,
        "dz_xpoint_divertor": 0.0,
        # divertor geometry
        "plsepi": 1.0,
        "plsepo": 1.5,
        "plleni": 1.0,
        "plleno": 1.0,
    },
    "tfcoil": {
        "i_tf_sup": 1,
        "i_tf_shape": 1,
        "i_tf_wp_geom": 1,
        "n_tf_coils": 16.0,
        "dr_tf_plasma_case": 0.07491064938739048,
        "dr_tf_nose_case": 0.2816873221155309,
        "dx_tf_wp_primary_toroidal": 1.2533980800120443,
        "dx_tf_wp_insulation": 0.008,
        "dx_tf_wp_insertion_gap": 0.01,
        "ripple_b_tf_plasma_edge_max": 0.6,
        "drtop": 0.0,
    },
    "superconducting_tfcoil": {
        # Read by `plasma_outboard_edge_toroidal_ripple`'s `i_tf_sup == 1` arm and then
        # never used by it -- see the port's module docstring on `i_tf_wp_geom`. Set to
        # the converged values anyway so the reference runs the real code path.
        "r_tf_wp_inboard_inner": 2.9802947938573854,
        "r_tf_wp_inboard_centre": 3.4019958081059247,
        "r_tf_wp_inboard_outer": 3.8236968223544636,
    },
    "fwbs": {
        "blktmodel": 0,
    },
    "divertor": {
        "dz_divertor": 0.62,
        "betai": 1.0,
        "betao": 1.0,
        "n_divertors": 1,
    },
    "numerics": {
        # `ixc = [4, 6]` on the reference run, so iteration variable 140 is absent and
        # `process/models/build.py:1685` does not run. Zero active variables is the same
        # empty `ixc[0:n]` slice as far as the `140 in ...` test is concerned.
        "n_iteration_variables": 0,
    },
}


def _data(**overrides):
    """A `DataStructure` at `BASELINE`, with `area__field=value` overrides applied."""
    data = DataStructure()
    for area, fields in BASELINE.items():
        target = getattr(data, area)
        for name, value in fields.items():
            setattr(target, name, value)
    for key, value in overrides.items():
        area, name = key.split("__", 1)
        setattr(getattr(data, area), name, value)
    return data


def _build(**overrides):
    """A `Build` bound to `_data(**overrides)`."""
    model = Build()
    model.data = _data(**overrides)
    return model


def _radial(**overrides):
    """Run PROCESS's real `calculate_radial_build` and hand back the mutated data."""
    model = _build(**overrides)
    model.calculate_radial_build(output=False)
    return model.data


def _vertical(**overrides):
    """Run PROCESS's real `calculate_vertical_build` and hand back the mutated data."""
    model = _build(**overrides)
    model.calculate_vertical_build(output=False)
    return model.data


# ---------------------------------------------------------------------------
# Vertical build
# ---------------------------------------------------------------------------


def _reference_z_plasma_xpoint(rminor, kappa):
    data = _vertical(physics__rminor=rminor, physics__kappa=kappa)
    return data.build.z_plasma_xpoint_upper, data.build.z_plasma_xpoint_lower


class TestZPlasmaXpoint(Tier1Contract):
    """`calculate_z_plasma_xpoint` vs `Build.calculate_vertical_build:167-172`."""

    audit_record = "models/build.md"
    reference = _reference_z_plasma_xpoint
    ported = calculate_z_plasma_xpoint

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged", rminor=2.6666666666666665, kappa=1.85
        ),
    ]

    fuzz_bounds = {"rminor": (1.5, 4.0), "kappa": (1.2, 2.2)}


def _reference_divertor_geometry_conventional(
    rmajor, rminor, kappa, triang, plsepi, plsepo, plleni, plleno, betai, betao
):
    model = _build(
        physics__rmajor=rmajor,
        physics__rminor=rminor,
        physics__kappa=kappa,
        physics__triang=triang,
        build__plsepi=plsepi,
        build__plsepo=plsepo,
        build__plleni=plleni,
        build__plleno=plleno,
        divertor__betai=betai,
        divertor__betao=betao,
    )
    divht = model.divgeom(output=False)
    return divht, model.data.build.rspo


class TestDivertorGeometryConventional(Tier1Contract):
    """`calculate_divertor_geometry_conventional` vs `Build.divgeom`, `itart == 0`."""

    audit_record = "models/build.md"
    reference = _reference_divertor_geometry_conventional
    ported = calculate_divertor_geometry_conventional

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            rmajor=8.0,
            rminor=2.6666666666666665,
            kappa=1.85,
            triang=0.5,
            plsepi=1.0,
            plsepo=1.5,
            plleni=1.0,
            plleno=1.0,
            betai=1.0,
            betao=1.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (6.0, 10.0),
        "rminor": (2.0, 3.5),
        "kappa": (1.5, 2.1),
        # `triang` is kept away from 1, where `(triang - 1)**2` in `rci` is a division
        # by zero, and away from 0 only for variety -- 0 is well inside the domain.
        "triang": (0.2, 0.7),
        "plsepi": (0.7, 1.5),
        "plsepo": (1.0, 2.0),
        "plleni": (0.7, 1.5),
        "plleno": (0.7, 1.5),
        "betai": (0.8, 1.3),
        "betao": (0.8, 1.3),
    }


def _reference_z_tf_inside_half(
    z_plasma_xpoint_upper,
    dz_xpoint_divertor,
    dz_divertor,
    dz_shld_lower,
    dz_vv_lower,
    dz_shld_vv_gap,
    dz_shld_thermal,
    dr_tf_shld_gap,
):
    """`Build.calculate_vertical_build:807-816`, reached through the real method.

    Two of the eight arguments cannot be set on `data` directly, because
    `calculate_vertical_build` computes them before it uses them:

    - `z_plasma_xpoint_upper` is overwritten at `:167` with `rminor * kappa`, so it is
      fed as `rminor = z_plasma_xpoint_upper` with `kappa = 1.0`;
    - `dz_xpoint_divertor` is overwritten at `:800-801` with `divgeom`'s answer *only
      when the input is below `1e-5`*, so any sample value above that threshold is kept
      as given. Every sample here is, which is also the branch a run that sets
      `dz_xpoint_divertor` in its input file takes.
    """
    data = _vertical(
        physics__rminor=z_plasma_xpoint_upper,
        physics__kappa=1.0,
        build__dz_xpoint_divertor=dz_xpoint_divertor,
        divertor__dz_divertor=dz_divertor,
        build__dz_shld_lower=dz_shld_lower,
        build__dz_vv_lower=dz_vv_lower,
        build__dz_shld_vv_gap=dz_shld_vv_gap,
        build__dz_shld_thermal=dz_shld_thermal,
        build__dr_tf_shld_gap=dr_tf_shld_gap,
    )
    return data.build.z_tf_inside_half


class TestZTfInsideHalf(Tier1Contract):
    """`calculate_z_tf_inside_half` vs `Build.calculate_vertical_build:807-816`.

    The tokamak occupant of `.build.z_tf_inside_half`; the stellarator's is
    `models/stellarator/coils/calculate.py::ZTfInsideHalf`, a different formula for the
    same field on a different device.
    """

    audit_record = "models/build.md"
    reference = _reference_z_tf_inside_half
    ported = calculate_z_tf_inside_half

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            z_plasma_xpoint_upper=4.933333333333334,
            dz_xpoint_divertor=2.001883830794158,
            dz_divertor=0.62,
            dz_shld_lower=0.7,
            dz_vv_lower=0.3,
            dz_shld_vv_gap=0.163,
            dz_shld_thermal=0.05,
            dr_tf_shld_gap=0.05,
        ),
    ]

    fuzz_bounds = {
        "z_plasma_xpoint_upper": (3.0, 7.0),
        "dz_xpoint_divertor": (1.0, 3.0),
        "dz_divertor": (0.3, 1.0),
        "dz_shld_lower": (0.4, 1.0),
        "dz_vv_lower": (0.2, 0.6),
        "dz_shld_vv_gap": (0.1, 0.3),
        "dz_shld_thermal": (0.02, 0.1),
        "dr_tf_shld_gap": (0.02, 0.1),
    }


# ---------------------------------------------------------------------------
# Radial build
# ---------------------------------------------------------------------------


def _reference_dr_tf_wp_with_insulation(
    dr_tf_inboard, dr_tf_plasma_case, dr_tf_nose_case
):
    data = _radial(
        build__dr_tf_inboard=dr_tf_inboard,
        tfcoil__dr_tf_plasma_case=dr_tf_plasma_case,
        tfcoil__dr_tf_nose_case=dr_tf_nose_case,
    )
    return data.tfcoil.dr_tf_wp_with_insulation


class TestDrTfWpWithInsulation(Tier1Contract):
    """`calculate_dr_tf_wp_with_insulation` vs `calculate_radial_build:1743-1747`.

    The live arm on `large_tokamak_eval.IN.DAT`: `ixc = [4, 6]`, so iteration variable
    140 is absent and PROCESS runs this assignment rather than its inverse.
    """

    audit_record = "models/build.md"
    reference = _reference_dr_tf_wp_with_insulation
    ported = calculate_dr_tf_wp_with_insulation

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            dr_tf_inboard=1.2,
            dr_tf_plasma_case=0.07491064938739048,
            dr_tf_nose_case=0.2816873221155309,
        ),
    ]

    fuzz_bounds = {
        "dr_tf_inboard": (0.8, 1.8),
        "dr_tf_plasma_case": (0.03, 0.2),
        "dr_tf_nose_case": (0.1, 0.5),
    }


def _reference_dr_tf_inboard(
    dr_tf_wp_with_insulation, dr_tf_plasma_case, dr_tf_nose_case
):
    """`calculate_radial_build:1685-1689`, with iteration variable 140 made active.

    `ixc[0] = 140` with one active variable is what puts PROCESS on this arm; the
    reference run is not on it (see `TestDrTfWpWithInsulation`), so this contract is the
    only thing exercising the other half of the pair.
    """
    data = _data(
        tfcoil__dr_tf_wp_with_insulation=dr_tf_wp_with_insulation,
        tfcoil__dr_tf_plasma_case=dr_tf_plasma_case,
        tfcoil__dr_tf_nose_case=dr_tf_nose_case,
    )
    ixc = copy.deepcopy(data.numerics.ixc)
    ixc[0] = 140
    data.numerics.ixc = ixc
    data.numerics.n_iteration_variables = 1
    model = Build()
    model.data = data
    model.calculate_radial_build(output=False)
    return data.build.dr_tf_inboard


class TestDrTfInboardFromWindingPack(Tier1Contract):
    """`calculate_dr_tf_inboard` vs `calculate_radial_build:1685-1689`, `140 in ixc`."""

    audit_record = "models/build.md"
    reference = _reference_dr_tf_inboard
    ported = calculate_dr_tf_inboard

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged-wp",
            dr_tf_wp_with_insulation=0.8434020284970785,
            dr_tf_plasma_case=0.07491064938739048,
            dr_tf_nose_case=0.2816873221155309,
        ),
    ]

    fuzz_bounds = {
        "dr_tf_wp_with_insulation": (0.4, 1.4),
        "dr_tf_plasma_case": (0.03, 0.2),
        "dr_tf_nose_case": (0.1, 0.5),
    }


def _reference_r_shld_inboard_inner(
    rmajor,
    rminor,
    dr_fw_plasma_gap_inboard,
    dr_fw_inboard,
    dr_blkt_inboard,
    dr_shld_inboard,
):
    data = _radial(
        physics__rmajor=rmajor,
        physics__rminor=rminor,
        build__dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
        build__dr_fw_inboard=dr_fw_inboard,
        build__dr_blkt_inboard=dr_blkt_inboard,
        build__dr_shld_inboard=dr_shld_inboard,
    )
    return data.build.r_shld_inboard_inner


class TestRShldInboardInner(Tier1Contract):
    """`calculate_r_shld_inboard_inner` vs `calculate_radial_build:1873-1880`."""

    audit_record = "models/build.md"
    reference = _reference_r_shld_inboard_inner
    ported = calculate_r_shld_inboard_inner

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            rmajor=8.0,
            rminor=2.6666666666666665,
            dr_fw_plasma_gap_inboard=0.25,
            dr_fw_inboard=0.018000000000000002,
            dr_blkt_inboard=0.7,
            dr_shld_inboard=0.3,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (6.0, 10.0),
        "rminor": (2.0, 3.5),
        "dr_fw_plasma_gap_inboard": (0.1, 0.5),
        "dr_fw_inboard": (0.01, 0.05),
        "dr_blkt_inboard": (0.3, 1.0),
        "dr_shld_inboard": (0.2, 0.6),
    }


# ---------------------------------------------------------------------------
# The outboard chain, end to end
# ---------------------------------------------------------------------------


def _ported_outboard_build(
    rmajor,
    rminor,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_shld_outboard,
    dr_shld_blkt_gap,
    dr_vv_outboard,
    gapomin,
    dr_shld_thermal_outboard,
    dr_tf_shld_gap,
    dr_tf_inboard,
    ripple_b_tf_plasma_edge_max,
    n_tf_coils,
    dx_tf_wp_primary_toroidal,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    """The six ported functions `calculate_radial_build:1883-1977` covers, composed.

    Mirrors PROCESS's own order exactly, including its *two* calls to the ripple fit:
    the first supplies `r_tf_outboard_midmin`, the second the surviving
    `.tfcoil.ripple_b_tf_plasma_edge` at the final leg radius.
    """
    r_shld_outboard_outer = calculate_r_shld_outboard_outer(
        rmajor,
        rminor,
        dr_fw_plasma_gap_outboard,
        dr_fw_outboard,
        dr_blkt_outboard,
        dr_shld_outboard,
    )
    dr_tf_outboard = calculate_dr_tf_outboard_superconducting(dr_tf_inboard)
    r_tf_outboard_mid_unrippled = calculate_r_tf_outboard_mid_unrippled(
        r_shld_outboard_outer,
        dr_shld_blkt_gap,
        dr_vv_outboard,
        gapomin,
        dr_shld_thermal_outboard,
        dr_tf_shld_gap,
        dr_tf_outboard,
    )
    dx_tf_wp_conductor_max = calculate_dx_tf_wp_conductor_max_superconducting(
        dx_tf_wp_primary_toroidal, dx_tf_wp_insulation, dx_tf_wp_insertion_gap
    )
    _, r_tf_outboard_midmin = plasma_outboard_edge_toroidal_ripple_fitted(
        ripple_b_tf_plasma_edge_max,
        r_tf_outboard_mid_unrippled,
        n_tf_coils,
        rmajor,
        rminor,
        dx_tf_wp_conductor_max,
    )
    r_tf_outboard_mid = calculate_r_tf_outboard_mid(
        r_tf_outboard_mid_unrippled, r_tf_outboard_midmin
    )
    dr_shld_vv_gap_outboard = calculate_dr_shld_vv_gap_outboard(
        r_tf_outboard_mid,
        dr_tf_outboard,
        dr_vv_outboard,
        r_shld_outboard_outer,
        dr_shld_thermal_outboard,
        dr_tf_shld_gap,
        dr_shld_blkt_gap,
    )
    ripple_b_tf_plasma_edge, _ = plasma_outboard_edge_toroidal_ripple_fitted(
        ripple_b_tf_plasma_edge_max,
        r_tf_outboard_mid,
        n_tf_coils,
        rmajor,
        rminor,
        dx_tf_wp_conductor_max,
    )
    return (
        r_shld_outboard_outer,
        dr_tf_outboard,
        r_tf_outboard_mid,
        dr_shld_vv_gap_outboard,
        ripple_b_tf_plasma_edge,
    )


def _reference_outboard_build(
    rmajor,
    rminor,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_shld_outboard,
    dr_shld_blkt_gap,
    dr_vv_outboard,
    gapomin,
    dr_shld_thermal_outboard,
    dr_tf_shld_gap,
    dr_tf_inboard,
    ripple_b_tf_plasma_edge_max,
    n_tf_coils,
    dx_tf_wp_primary_toroidal,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    data = _radial(
        physics__rmajor=rmajor,
        physics__rminor=rminor,
        build__dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
        build__dr_fw_outboard=dr_fw_outboard,
        build__dr_blkt_outboard=dr_blkt_outboard,
        build__dr_shld_outboard=dr_shld_outboard,
        build__dr_shld_blkt_gap=dr_shld_blkt_gap,
        build__dr_vv_outboard=dr_vv_outboard,
        build__gapomin=gapomin,
        build__dr_shld_thermal_outboard=dr_shld_thermal_outboard,
        build__dr_tf_shld_gap=dr_tf_shld_gap,
        build__dr_tf_inboard=dr_tf_inboard,
        tfcoil__ripple_b_tf_plasma_edge_max=ripple_b_tf_plasma_edge_max,
        tfcoil__n_tf_coils=n_tf_coils,
        tfcoil__dx_tf_wp_primary_toroidal=dx_tf_wp_primary_toroidal,
        tfcoil__dx_tf_wp_insulation=dx_tf_wp_insulation,
        tfcoil__dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
    )
    return (
        data.build.r_shld_outboard_outer,
        data.build.dr_tf_outboard,
        data.build.r_tf_outboard_mid,
        data.build.dr_shld_vv_gap_outboard,
        data.tfcoil.ripple_b_tf_plasma_edge,
    )


class TestOutboardBuildChain(Tier1Contract):
    """The whole outboard closure vs `calculate_radial_build:1883-1977`.

    Covers `calculate_r_shld_outboard_outer`,
    `calculate_dr_tf_outboard_superconducting`,
    `calculate_dx_tf_wp_conductor_max_superconducting`,
    `calculate_r_tf_outboard_mid_unrippled`,
    `plasma_outboard_edge_toroidal_ripple_fitted`, `calculate_r_tf_outboard_mid` and
    `calculate_dr_shld_vv_gap_outboard` in one diff, because PROCESS has no boundary
    inside that stretch to compare against.

    Switches pinned by the baseline: `i_tf_sup = 1`, `i_tf_shape = 1`,
    `blktmodel = 0`, `140 not in ixc`.

    Fuzz bounds are kept inside the region where PROCESS moves the leg out
    (`r_tf_outboard_midmin > r_tf_outboard_mid_unrippled`, true by ~1 m at the converged
    point), so no sample straddles `calculate_r_tf_outboard_mid`'s `maximum` kink -- the
    one place in this unit where a finite-difference gradient and an autodiff one would
    legitimately disagree.
    """

    audit_record = "models/build.md"
    reference = _reference_outboard_build
    ported = _ported_outboard_build

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            rmajor=8.0,
            rminor=2.6666666666666665,
            dr_fw_plasma_gap_outboard=0.25,
            dr_fw_outboard=0.018000000000000002,
            dr_blkt_outboard=1.0,
            dr_shld_outboard=0.8,
            dr_shld_blkt_gap=0.02,
            dr_vv_outboard=0.3,
            gapomin=0.234,
            dr_shld_thermal_outboard=0.05,
            dr_tf_shld_gap=0.05,
            dr_tf_inboard=1.2,
            ripple_b_tf_plasma_edge_max=0.6,
            n_tf_coils=16.0,
            dx_tf_wp_primary_toroidal=1.2533980800120443,
            dx_tf_wp_insulation=0.008,
            dx_tf_wp_insertion_gap=0.01,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (7.5, 8.5),
        "rminor": (2.4, 2.9),
        "dr_fw_plasma_gap_outboard": (0.2, 0.3),
        "dr_fw_outboard": (0.01, 0.03),
        "dr_blkt_outboard": (0.8, 1.2),
        "dr_shld_outboard": (0.6, 1.0),
        "dr_shld_blkt_gap": (0.01, 0.04),
        "dr_vv_outboard": (0.2, 0.4),
        "gapomin": (0.15, 0.35),
        "dr_shld_thermal_outboard": (0.03, 0.08),
        "dr_tf_shld_gap": (0.03, 0.08),
        "dr_tf_inboard": (1.0, 1.4),
        "ripple_b_tf_plasma_edge_max": (0.4, 0.8),
        "n_tf_coils": (16.0, 18.0),
        "dx_tf_wp_primary_toroidal": (1.0, 1.5),
        "dx_tf_wp_insulation": (0.005, 0.012),
        "dx_tf_wp_insertion_gap": (0.005, 0.015),
    }


def _ported_ripple_superconducting(
    ripple_b_tf_plasma_edge_max,
    r_tf_outboard_mid,
    n_tf_coils,
    rmajor,
    rminor,
    dx_tf_wp_primary_toroidal,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    """`calculate_dx_tf_wp_conductor_max_superconducting` feeding the fitted ripple."""
    return plasma_outboard_edge_toroidal_ripple_fitted(
        ripple_b_tf_plasma_edge_max,
        r_tf_outboard_mid,
        n_tf_coils,
        rmajor,
        rminor,
        calculate_dx_tf_wp_conductor_max_superconducting(
            dx_tf_wp_primary_toroidal, dx_tf_wp_insulation, dx_tf_wp_insertion_gap
        ),
    )


def _reference_ripple_superconducting(
    ripple_b_tf_plasma_edge_max,
    r_tf_outboard_mid,
    n_tf_coils,
    rmajor,
    rminor,
    dx_tf_wp_primary_toroidal,
    dx_tf_wp_insulation,
    dx_tf_wp_insertion_gap,
):
    """`Build.plasma_outboard_edge_toroidal_ripple`, minus its third return value.

    The dropped value is `flag`, a fitted-range diagnostic. It is a step function of
    `n_tf_coils` with a threshold at exactly `16`, the value the reference run uses, so a
    central difference of it against a port that does not compute it would compare a jump
    against a zero. Nothing in the graph reads `.build.ripflag`; see the port's
    `plasma_outboard_edge_toroidal_ripple_fitted` docstring.
    """
    ripple, r_tf_outboard_midmin, _flag = Build.plasma_outboard_edge_toroidal_ripple(
        ripple_b_tf_plasma_edge_max=ripple_b_tf_plasma_edge_max,
        r_tf_outboard_mid=r_tf_outboard_mid,
        n_tf_coils=n_tf_coils,
        rmajor=rmajor,
        rminor=rminor,
        r_tf_wp_inboard_inner=BASELINE["superconducting_tfcoil"][
            "r_tf_wp_inboard_inner"
        ],
        r_tf_wp_inboard_centre=BASELINE["superconducting_tfcoil"][
            "r_tf_wp_inboard_centre"
        ],
        r_tf_wp_inboard_outer=BASELINE["superconducting_tfcoil"][
            "r_tf_wp_inboard_outer"
        ],
        dx_tf_wp_primary_toroidal=dx_tf_wp_primary_toroidal,
        i_tf_shape=BASELINE["tfcoil"]["i_tf_shape"],
        i_tf_sup=BASELINE["tfcoil"]["i_tf_sup"],
        dx_tf_wp_insulation=dx_tf_wp_insulation,
        dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        i_tf_wp_geom=BASELINE["tfcoil"]["i_tf_wp_geom"],
    )
    return ripple, r_tf_outboard_midmin


class TestRippleSuperconducting(Tier1Contract):
    """The ripple fit vs `Build.plasma_outboard_edge_toroidal_ripple`, `i_tf_sup == 1`,
    `i_tf_shape == 1`.
    """

    audit_record = "models/build.md"
    reference = _reference_ripple_superconducting
    ported = _ported_ripple_superconducting

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            ripple_b_tf_plasma_edge_max=0.6,
            r_tf_outboard_mid=14.978406000060053,
            n_tf_coils=16.0,
            rmajor=8.0,
            rminor=2.6666666666666665,
            dx_tf_wp_primary_toroidal=1.2533980800120443,
            dx_tf_wp_insulation=0.008,
            dx_tf_wp_insertion_gap=0.01,
        ),
        legacy_sample(
            "large_tokamak_eval-unrippled-leg",
            # The radius PROCESS evaluates the fit at on its *first* call, before the
            # leg is moved out: `calculate_radial_build:1901-1909`'s stack.
            ripple_b_tf_plasma_edge_max=0.6,
            r_tf_outboard_mid=13.988666666666669,
            n_tf_coils=16.0,
            rmajor=8.0,
            rminor=2.6666666666666665,
            dx_tf_wp_primary_toroidal=1.2533980800120443,
            dx_tf_wp_insulation=0.008,
            dx_tf_wp_insertion_gap=0.01,
        ),
    ]

    fuzz_bounds = {
        "ripple_b_tf_plasma_edge_max": (0.4, 0.8),
        "r_tf_outboard_mid": (13.0, 16.0),
        "n_tf_coils": (16.0, 18.0),
        "rmajor": (7.5, 8.5),
        "rminor": (2.4, 2.9),
        "dx_tf_wp_primary_toroidal": (1.0, 1.5),
        "dx_tf_wp_insulation": (0.005, 0.012),
        "dx_tf_wp_insertion_gap": (0.005, 0.015),
    }
