"""Harness cases for the ported tokamak build (`functional_process/cottax/build.py`).

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
from functional_process.cottax.build import (
    calculate_divertor_geometry_conventional,
    calculate_divertor_geometry_spherical_tokamak,
    calculate_dr_shld_vv_gap_outboard,
    calculate_dr_tf_inboard,
    calculate_dr_tf_inner_bore,
    calculate_dr_tf_outboard_superconducting,
    calculate_dr_tf_wp_with_insulation,
    calculate_dx_tf_wp_conductor_max_superconducting,
    calculate_dz_blkt_upper,
    calculate_r_cp_top_from_tf_inboard_out,
    calculate_r_shld_inboard_inner,
    calculate_r_shld_outboard_outer,
    calculate_r_tf_inboard_radii_no_cs_precomp,
    calculate_r_tf_inboard_radii_tf_outside_cs,
    calculate_r_tf_outboard_mid,
    calculate_r_tf_outboard_mid_unrippled,
    calculate_rbld,
    calculate_tf_top_height_double_null,
    calculate_tf_top_height_single_null,
    calculate_vacuum_vessel_and_shield_radii,
    calculate_z_plasma_xpoint,
    calculate_z_tf_inside_half,
    plasma_outboard_edge_toroidal_ripple_fitted,
    plasma_outboard_edge_toroidal_ripple_picture_frame,
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


def _reference_dz_blkt_upper(dr_blkt_inboard, dr_blkt_outboard):
    """`Build.calculate_radial_build:1665-1667`, read back off `data`.

    A radial-build line reached through `_radial`, not `_vertical`, despite writing a
    vertical thickness -- PROCESS puts it there and the adapter follows the source
    rather than the field's name.
    """
    return _radial(
        build__dr_blkt_inboard=dr_blkt_inboard,
        build__dr_blkt_outboard=dr_blkt_outboard,
    ).build.dz_blkt_upper


class TestDzBlktUpper(Tier1Contract):
    """`calculate_dz_blkt_upper` vs `Build.calculate_radial_build:1665-1667`."""

    audit_record = "models/build.md"
    reference = _reference_dz_blkt_upper
    ported = calculate_dz_blkt_upper

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            dr_blkt_inboard=0.7,
            dr_blkt_outboard=1.0,
        ),
    ]
    """`BASELINE`'s own two blanket thicknesses -- both are run inputs at
    `blktmodel == 0`, which is the arm every tracked tokamak takes."""

    fuzz_bounds = {"dr_blkt_inboard": (0.2, 1.5), "dr_blkt_outboard": (0.3, 2.0)}


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


def _reference_divertor_geometry_spherical_tokamak(rminor):
    """`Build.divgeom` with `itart == 1` -- the early return at `build.py:862-863`.

    The `assert` is the write-set claim the occupant split rests on: the early return
    never reaches the `.build.rspo` write at `:912`, so the reference checks on every
    sample that the real `divgeom` left `rspo` exactly as it found it.
    """
    model = _build(physics__itart=1, physics__rminor=rminor)
    rspo_before = model.data.build.rspo
    divht = model.divgeom(output=False)
    assert model.data.build.rspo == rspo_before
    return divht


class TestDivertorGeometrySphericalTokamak(Tier1Contract):
    """`calculate_divertor_geometry_spherical_tokamak` vs `Build.divgeom`, `itart == 1`.

    One read, one output, and the whole arm is `1.75 * rminor` -- the case exists less
    for the arithmetic than for the gradient check and for pinning the write-set (see
    the reference's `assert`). The legacy sample's `rminor = 2.5` is
    `spherical_tokamak_eval.IN.DAT`'s *input* geometry (`rmajor = 4.5`, `aspect = 1.8`);
    `rmajor` is an iteration variable on that file, so no converged value is written
    down here.
    """

    audit_record = "models/build.md"
    reference = _reference_divertor_geometry_spherical_tokamak
    ported = calculate_divertor_geometry_spherical_tokamak

    samples = [
        legacy_sample("spherical_tokamak_eval-input", rminor=2.5),
    ]

    fuzz_bounds = {"rminor": (0.8, 3.0)}


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


# `z_tf_inside_half` is not settable on `data`: `calculate_vertical_build` computes it
# at `:807` from eight fields and then uses it at `:820`/`:840`. Four of those eight --
# `dz_xpoint_divertor`, `dz_divertor`, `dz_shld_lower`, `dz_vv_lower` -- appear nowhere
# in the `z_tf_top` block, so the *lower* build can be solved backwards for a requested
# half-height while every field the two expressions share keeps its sample value. These
# three constants are the fixed part of that solve; `dz_shld_lower` is the free one.
# All three are above the `1e-5` latch or plainly positive, so the reference runs the
# same branch a real single-null tokamak does.
_LOWER_BUILD = {"dz_xpoint_divertor": 2.0, "dz_divertor": 0.62, "dz_vv_lower": 0.3}

_Z_XPOINT = 4.933333333333334
"""`large_tokamak_eval`'s converged `.build.z_plasma_xpoint_upper`.

Used by the double-null adapter only, where `z_plasma_xpoint_upper` is in `:807`'s stack
but *not* in that arm's `z_tf_top` expression, so it is free to be pinned rather than
sampled."""


def _dz_shld_lower_for(
    z_tf_inside_half,
    z_plasma_xpoint_upper,
    dz_shld_vv_gap,
    dz_shld_thermal,
    dr_tf_shld_gap,
):
    """The lower-shield thickness that makes `:807`'s stack equal `z_tf_inside_half`."""
    return z_tf_inside_half - (
        z_plasma_xpoint_upper
        + _LOWER_BUILD["dz_xpoint_divertor"]
        + _LOWER_BUILD["dz_divertor"]
        + _LOWER_BUILD["dz_vv_lower"]
        + dz_shld_vv_gap
        + dz_shld_thermal
        + dr_tf_shld_gap
    )


def _reference_tf_top_height_single_null(
    z_tf_inside_half,
    dr_tf_inboard,
    dr_tf_shld_gap,
    dz_shld_thermal,
    dz_shld_vv_gap,
    dz_vv_upper,
    dz_shld_upper,
    dr_shld_blkt_gap,
    dz_blkt_upper,
    dr_fw_inboard,
    dr_fw_outboard,
    dz_fw_plasma_gap,
    z_plasma_xpoint_upper,
):
    """`Build.calculate_vertical_build:826-841`, reached through the real method.

    Two of the thirteen arguments are not `data` fields the method leaves alone:

    - `z_plasma_xpoint_upper` is overwritten at `:167` with `rminor * kappa`, so it is
      fed as `rminor = z_plasma_xpoint_upper` with `kappa = 1.0`, exactly as
      `_reference_z_tf_inside_half` does;
    - `z_tf_inside_half` is computed at `:807`, so it is set indirectly through
      `dz_shld_lower` -- see `_dz_shld_lower_for`.

    Everything else is passed straight through. `dz_fw_plasma_gap` is only rewritten in
    `calculate_radial_build` (`:1669-1680`), never here, so the sample value survives.
    """
    data = _vertical(
        physics__i_single_null=1,
        physics__rminor=z_plasma_xpoint_upper,
        physics__kappa=1.0,
        build__dz_xpoint_divertor=_LOWER_BUILD["dz_xpoint_divertor"],
        divertor__dz_divertor=_LOWER_BUILD["dz_divertor"],
        build__dz_vv_lower=_LOWER_BUILD["dz_vv_lower"],
        build__dz_shld_lower=_dz_shld_lower_for(
            z_tf_inside_half,
            z_plasma_xpoint_upper,
            dz_shld_vv_gap,
            dz_shld_thermal,
            dr_tf_shld_gap,
        ),
        build__dr_tf_inboard=dr_tf_inboard,
        build__dr_tf_shld_gap=dr_tf_shld_gap,
        build__dz_shld_thermal=dz_shld_thermal,
        build__dz_shld_vv_gap=dz_shld_vv_gap,
        build__dz_vv_upper=dz_vv_upper,
        build__dz_shld_upper=dz_shld_upper,
        build__dr_shld_blkt_gap=dr_shld_blkt_gap,
        build__dz_blkt_upper=dz_blkt_upper,
        build__dr_fw_inboard=dr_fw_inboard,
        build__dr_fw_outboard=dr_fw_outboard,
        build__dz_fw_plasma_gap=dz_fw_plasma_gap,
    )
    return data.build.z_tf_top, data.build.dz_tf_upper_lower_midplane


class TestTfTopHeightSingleNull(Tier1Contract):
    """`calculate_tf_top_height_single_null` vs
    `Build.calculate_vertical_build:826-841`.

    `.build.z_tf_top` and `.build.dz_tf_upper_lower_midplane` had **no producer at all**
    in this port until 2026-08-30 -- both sat frozen at the cold `0.0` while PROCESS
    computed `8.656` m and `-1.234` m on `large_tokamak_nof`
    (`missing_producers_tokamak.txt`). `z_tf_top` is read by
    `models/tfcoil/base.py::TfCoilShapeDShapeSingleNull`, which places the coil's arcs
    from it, so the cold graph drew a TF coil whose top was on the midplane.

    The offset is checked alongside the height rather than separately because it is a
    *difference* of the two vertical stacks and the interesting failure is a sign or a
    dropped term, which a test of the height alone cannot see.
    """

    audit_record = "models/build.md"
    reference = _reference_tf_top_height_single_null
    ported = calculate_tf_top_height_single_null

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            z_tf_inside_half=8.818217164127492,
            dr_tf_inboard=1.2,
            dr_tf_shld_gap=0.05,
            dz_shld_thermal=0.05,
            dz_shld_vv_gap=0.163,
            dz_vv_upper=0.3,
            dz_shld_upper=0.6,
            dr_shld_blkt_gap=0.02,
            dz_blkt_upper=0.85,
            dr_fw_inboard=0.018000000000000002,
            dr_fw_outboard=0.018000000000000002,
            dz_fw_plasma_gap=0.6,
            z_plasma_xpoint_upper=4.933333333333334,
        ),
    ]

    fuzz_bounds = {
        "z_tf_inside_half": (7.0, 10.0),
        "dr_tf_inboard": (1.0, 1.4),
        "dr_tf_shld_gap": (0.03, 0.08),
        "dz_shld_thermal": (0.02, 0.1),
        "dz_shld_vv_gap": (0.1, 0.3),
        "dz_vv_upper": (0.2, 0.4),
        "dz_shld_upper": (0.4, 0.8),
        "dr_shld_blkt_gap": (0.01, 0.04),
        "dz_blkt_upper": (0.6, 1.1),
        "dr_fw_inboard": (0.01, 0.03),
        "dr_fw_outboard": (0.01, 0.03),
        "dz_fw_plasma_gap": (0.4, 0.8),
        "z_plasma_xpoint_upper": (4.0, 6.0),
    }


def _reference_tf_top_height_double_null(z_tf_inside_half, dr_tf_inboard):
    """`Build.calculate_vertical_build:820-824`, the `i_single_null == 0` arm.

    `z_tf_inside_half` is reached the same indirect way as in the single-null adapter;
    the three fields it shares with the `z_tf_top` expression on that arm are none at
    all, so `_LOWER_BUILD`'s constants and the `BASELINE` upper build are all that the
    solve needs.
    """
    data = _vertical(
        physics__i_single_null=0,
        physics__rminor=_Z_XPOINT,
        physics__kappa=1.0,
        build__dz_xpoint_divertor=_LOWER_BUILD["dz_xpoint_divertor"],
        divertor__dz_divertor=_LOWER_BUILD["dz_divertor"],
        build__dz_vv_lower=_LOWER_BUILD["dz_vv_lower"],
        build__dz_shld_lower=_dz_shld_lower_for(
            z_tf_inside_half,
            _Z_XPOINT,
            BASELINE["build"]["dz_shld_vv_gap"],
            BASELINE["build"]["dz_shld_thermal"],
            BASELINE["build"]["dr_tf_shld_gap"],
        ),
        build__dr_tf_inboard=dr_tf_inboard,
    )
    return data.build.z_tf_top, data.build.dz_tf_upper_lower_midplane


class TestTfTopHeightDoubleNull(Tier1Contract):
    """`calculate_tf_top_height_double_null` vs
    `Build.calculate_vertical_build:820-824`.

    Not the arm any *assembling* machine takes today, and the honest form of that is
    worth stating: the two tracked inputs with `i_single_null = 0`
    (`spherical_tokamak_eval.IN.DAT:292`, `st_regression.IN.DAT:638`) are both refused by
    `machine_from_indat` for an unrelated reason -- `i_tf_turn_type == 2`, the CroCo
    turn, `indat._refuse_unported_switch`. So this arm is written and tested against
    PROCESS, and it becomes reachable the moment that refusal lifts; it is not a
    speculative arm, because `build.py:820-824` is the code a double-null run runs.

    The second returned value is PROCESS's literal `0.0e0`; it is compared anyway,
    because an arm that silently stopped owning it would orphan every consumer of
    `.build.dz_tf_upper_lower_midplane` on those machines.
    """

    audit_record = "models/build.md"
    reference = _reference_tf_top_height_double_null
    ported = calculate_tf_top_height_double_null

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged-geometry",
            z_tf_inside_half=8.818217164127492,
            dr_tf_inboard=1.2,
        ),
    ]

    fuzz_bounds = {
        "z_tf_inside_half": (7.0, 10.0),
        "dr_tf_inboard": (1.0, 1.4),
    }


# ---------------------------------------------------------------------------
# Radial build
# ---------------------------------------------------------------------------


def _reference_dz_blkt_upper(dr_blkt_inboard, dr_blkt_outboard):
    """`calculate_radial_build:1664-1667`. Unconditional -- `blktmodel = 0` in
    `BASELINE`, so neither operand is rewritten above it.
    """
    data = _radial(
        build__dr_blkt_inboard=dr_blkt_inboard,
        build__dr_blkt_outboard=dr_blkt_outboard,
    )
    return data.build.dz_blkt_upper


class TestDzBlktUpper(Tier1Contract):
    """`calculate_dz_blkt_upper` vs `calculate_radial_build:1664-1667`.

    Landed 2026-08-30 as `calculate_tf_top_height_single_null`'s missing dependency, and
    a `missing_producers_tokamak.txt` row in its own right (`models/fw.py` and
    `models/vacuum/vacuum.py` read it too).
    """

    audit_record = "models/build.md"
    reference = _reference_dz_blkt_upper
    ported = calculate_dz_blkt_upper

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            dr_blkt_inboard=0.7,
            dr_blkt_outboard=1.0,
        ),
    ]

    fuzz_bounds = {
        "dr_blkt_inboard": (0.3, 1.0),
        "dr_blkt_outboard": (0.8, 1.2),
    }


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


def _reference_tf_inboard_radii(
    dr_bore,
    dr_cs,
    fseppc,
    fcspc,
    sigallpc,
    dr_cs_tf_gap,
    dr_tf_inboard,
):
    data = _radial(
        build__dr_bore=dr_bore,
        build__dr_cs=dr_cs,
        build__fseppc=fseppc,
        build__fcspc=fcspc,
        build__sigallpc=sigallpc,
        build__dr_cs_tf_gap=dr_cs_tf_gap,
        build__dr_tf_inboard=dr_tf_inboard,
    )
    return (
        data.build.dr_cs_bore,
        data.build.dr_cs_precomp,
        data.build.r_tf_inboard_in,
        data.build.r_tf_inboard_mid,
        data.build.r_tf_inboard_out,
    )


class TestTfInboardRadii(Tier1Contract):
    """`calculate_r_tf_inboard_radii_tf_outside_cs` vs
    `calculate_radial_build:1691-1735` at `(i_tf_inside_cs, i_cs_precomp) = (0, 1)`
    (both baked into `BASELINE`). Added 2026-08-27, `cold_boundary.md` producer 2.
    """

    audit_record = "models/build.md"
    reference = _reference_tf_inboard_radii
    ported = calculate_r_tf_inboard_radii_tf_outside_cs

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            dr_bore=2.003843190236783,
            dr_cs=0.546816593988753,
            fseppc=350000000.0,
            fcspc=0.6,
            sigallpc=300000000.0,
            dr_cs_tf_gap=0.08,
            dr_tf_inboard=1.2,
        ),
    ]

    fuzz_bounds = {
        "dr_bore": (0.5, 4.0),
        "dr_cs": (0.2, 1.5),
        "fseppc": (1.0e8, 6.0e8),
        "fcspc": (0.3, 0.9),
        "sigallpc": (1.0e8, 6.0e8),
        "dr_cs_tf_gap": (0.02, 0.3),
        "dr_tf_inboard": (0.5, 2.0),
    }


def _reference_tf_inboard_radii_no_precomp(
    dr_bore,
    dr_cs,
    dr_cs_tf_gap,
    dr_tf_inboard,
):
    """`calculate_radial_build` at `i_cs_precomp = 0` (the `BASELINE` value flipped).

    `fseppc`/`fcspc`/`sigallpc` stay at their `BASELINE` values deliberately: PROCESS
    must not read them on this arm, and leaving them nonzero means a wrong arm would
    show up as a value disagreement, not a division error.
    """
    data = _radial(
        build__i_cs_precomp=0,
        build__dr_bore=dr_bore,
        build__dr_cs=dr_cs,
        build__dr_cs_tf_gap=dr_cs_tf_gap,
        build__dr_tf_inboard=dr_tf_inboard,
    )
    return (
        data.build.dr_cs_bore,
        data.build.dr_cs_precomp,
        data.build.r_tf_inboard_in,
        data.build.r_tf_inboard_mid,
        data.build.r_tf_inboard_out,
    )


class TestTfInboardRadiiNoCsPrecomp(Tier1Contract):
    """`calculate_r_tf_inboard_radii_no_cs_precomp` vs
    `calculate_radial_build:1691-1735` at `(i_tf_inside_cs, i_cs_precomp) = (0, 0)` --
    the live cell on both tracked spherical-tokamak files
    (`spherical_tokamak_eval.IN.DAT:70-71`, `st_regression.IN.DAT:1811`/`:1845`).
    Added 2026-08-27, ST frontier wave.

    The legacy point is `spherical_tokamak_eval.IN.DAT`'s input radial build
    (`dr_bore`, `:61`; `dr_cs_tf_gap = 0.0`, `:67`; `dr_cs`, `:77`;
    `dr_tf_inboard = 0.9`, `:345` -- the file's literals) -- input values, not
    converged ones, since no converged reference for this cell has been solved yet.
    `dr_cs_tf_gap = 0.0` is the file's actual value and exercises the zero-gap edge.
    """

    audit_record = "models/build.md"
    reference = _reference_tf_inboard_radii_no_precomp
    ported = calculate_r_tf_inboard_radii_no_cs_precomp

    samples = [
        legacy_sample(
            "spherical_tokamak_eval-input",
            dr_bore=0.23375250334739459,
            dr_cs=0.20016400484967947,
            dr_cs_tf_gap=0.0,
            dr_tf_inboard=0.9,
        ),
    ]

    fuzz_bounds = {
        "dr_bore": (0.1, 4.0),
        "dr_cs": (0.1, 1.5),
        "dr_cs_tf_gap": (0.0, 0.3),
        "dr_tf_inboard": (0.2, 2.0),
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
    # The inboard half of the bore, from the same `BASELINE` fields PROCESS reads at
    # `:1691-1735`. Only `dr_tf_inboard` varies with the sample; the CS build above it
    # is fixed, exactly as `_reference_ripple_superconducting` fixes the winding pack.
    *_, r_tf_inboard_mid, _ = calculate_r_tf_inboard_radii_tf_outside_cs(
        BASELINE["build"]["dr_bore"],
        BASELINE["build"]["dr_cs"],
        BASELINE["build"]["fseppc"],
        BASELINE["build"]["fcspc"],
        BASELINE["build"]["sigallpc"],
        BASELINE["build"]["dr_cs_tf_gap"],
        dr_tf_inboard,
    )
    return (
        r_shld_outboard_outer,
        dr_tf_outboard,
        r_tf_outboard_mid,
        dr_shld_vv_gap_outboard,
        ripple_b_tf_plasma_edge,
        calculate_dr_tf_inner_bore(
            r_tf_outboard_mid, dr_tf_outboard, r_tf_inboard_mid, dr_tf_inboard
        ),
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
        data.build.dr_tf_inner_bore,
    )


class TestOutboardBuildChain(Tier1Contract):
    """The whole outboard closure vs `calculate_radial_build:1883-1977`.

    Covers `calculate_r_shld_outboard_outer`,
    `calculate_dr_tf_outboard_superconducting`,
    `calculate_dx_tf_wp_conductor_max_superconducting`,
    `calculate_r_tf_outboard_mid_unrippled`,
    `plasma_outboard_edge_toroidal_ripple_fitted`, `calculate_r_tf_outboard_mid`,
    `calculate_dr_shld_vv_gap_outboard` and `calculate_dr_tf_inner_bore` in one diff,
    because PROCESS has no boundary inside that stretch to compare against.

    **`calculate_dr_tf_inner_bore` joined this contract on 2026-08-30 rather than
    getting one of its own**, and the reason is the same one that created the contract:
    two of its four arguments (`.build.r_tf_outboard_mid`, `.build.dr_tf_outboard`) are
    produced *inside* this stretch and cannot be set on `data` independently -- at
    `i_tf_sup == 1` the source makes `dr_tf_outboard` equal to `dr_tf_inboard`, so a
    sample naming them separately is unreachable through PROCESS. Adding it here tests
    it at the post-ripple radius, which is where PROCESS's own surviving write
    (`:1949-1955`) is taken.

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


def _reference_ripple_picture_frame(
    ripple_b_tf_plasma_edge_max,
    r_tf_outboard_mid,
    n_tf_coils,
    rmajor,
    rminor,
):
    """`Build.plasma_outboard_edge_toroidal_ripple` with `i_tf_shape = 2`, minus `flag`.

    The winding-pack kwargs are the D-shape `BASELINE` values on purpose: the
    picture-frame arm computes `dx_tf_wp_conductor_max` and never uses it
    (`build.py:1551-1580` then `:1582-1590`), so if the port were secretly reading the
    winding pack the fixed values here would make that a value failure, not a
    coincidence. `flag` is dropped as everywhere else -- and on this arm it is the
    literal `0` from `:1581`, never reassigned, so nothing diagnostic is lost either.
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
        dx_tf_wp_primary_toroidal=BASELINE["tfcoil"]["dx_tf_wp_primary_toroidal"],
        i_tf_shape=2,
        i_tf_sup=BASELINE["tfcoil"]["i_tf_sup"],
        dx_tf_wp_insulation=BASELINE["tfcoil"]["dx_tf_wp_insulation"],
        dx_tf_wp_insertion_gap=BASELINE["tfcoil"]["dx_tf_wp_insertion_gap"],
        i_tf_wp_geom=BASELINE["tfcoil"]["i_tf_wp_geom"],
    )
    return ripple, r_tf_outboard_midmin


class TestRipplePictureFrame(Tier1Contract):
    """`plasma_outboard_edge_toroidal_ripple_picture_frame` vs
    `Build.plasma_outboard_edge_toroidal_ripple`, `i_tf_shape == 2`.

    The legacy point is `spherical_tokamak_eval.IN.DAT` / `st_regression.IN.DAT` input
    values, which agree on every field this arm reads: `rmajor = 4.5`,
    `rminor = 4.5 / 1.8 = 2.5`, `n_tf_coils = 12`,
    `ripple_b_tf_plasma_edge_max = 1.0`. No converged reference for either file has
    been solved, so `r_tf_outboard_mid` is set at the radius the 1% limit itself
    forces, `midmin = 7.0 / 0.01 ** (1/12) = 10.274594873354488 m` -- the value
    `calculate_r_tf_outboard_mid` would adopt whenever the stacked build is tighter.
    Fuzz covers the neighbourhood on every input.
    """

    audit_record = "models/build.md"
    reference = _reference_ripple_picture_frame
    ported = plasma_outboard_edge_toroidal_ripple_picture_frame

    samples = [
        legacy_sample(
            "spherical_tokamak_eval-inputs",
            ripple_b_tf_plasma_edge_max=1.0,
            r_tf_outboard_mid=10.274594873354488,
            n_tf_coils=12.0,
            rmajor=4.5,
            rminor=2.5,
        ),
    ]

    fuzz_bounds = {
        "ripple_b_tf_plasma_edge_max": (0.5, 1.5),
        "r_tf_outboard_mid": (8.0, 12.0),
        "n_tf_coils": (10.0, 14.0),
        "rmajor": (4.0, 5.0),
        "rminor": (2.2, 2.8),
    }


# ---------------------------------------------------------------------------
# Inboard vacuum vessel and neutronic shield (2026-08-29)
# ---------------------------------------------------------------------------


def _place_tf_leg(r_tf_inboard_out, **overrides):
    """The `dr_tf_inboard` that makes PROCESS's own radial build land on
    `r_tf_inboard_out`.

    `r_tf_inboard_out` is the one argument of the two functions below that PROCESS
    **computes** rather than reads -- `calculate_radial_build` writes it 140 lines
    above the block under test -- so it cannot simply be overridden onto the
    `DataStructure` the way every other argument here can: the method would overwrite
    it before reaching the lines being tested. Moving an *upstream* thickness instead
    drives PROCESS's real chain to the requested radius, so the reference stays
    PROCESS's own code with every branch intact.

    **`dr_tf_inboard` is the lever, not `dr_bore`, and the difference is exactness.**
    `r_tf_inboard_out = r_tf_inboard_in + dr_tf_inboard` and nothing upstream of it
    reads `dr_tf_inboard`, so one probe gives the answer with slope exactly 1. The bore
    looks like the more natural lever and is not: `dr_cs_precomp` is
    `fseppc / (2*pi*fcspc*sigallpc*(2*dr_bore + dr_cs))`, a hyperbola in `dr_bore`, so a
    two-point secant on the bore lands ~1e-4 out -- measured, by trying it. The
    reference asserts the landing either way, which is how that showed up as a failing
    test rather than as a quietly wrong oracle.
    """
    probe = _radial(**overrides).build
    return probe.dr_tf_inboard + (r_tf_inboard_out - probe.r_tf_inboard_out)


def _reference_vacuum_vessel_and_shield_radii(
    r_tf_inboard_out,
    dr_tf_shld_gap,
    dr_shld_thermal_inboard,
    dr_shld_vv_gap_inboard,
    dr_vv_inboard,
    dr_shld_inboard,
):
    """`calculate_radial_build:1833-1860`, the `TF_OUTSIDE_CS` arm."""
    overrides = {
        "build__dr_tf_shld_gap": dr_tf_shld_gap,
        "build__dr_shld_thermal_inboard": dr_shld_thermal_inboard,
        "build__dr_shld_vv_gap_inboard": dr_shld_vv_gap_inboard,
        "build__dr_vv_inboard": dr_vv_inboard,
        "build__dr_shld_inboard": dr_shld_inboard,
    }
    data = _radial(
        build__dr_tf_inboard=_place_tf_leg(r_tf_inboard_out, **overrides), **overrides
    )
    assert abs(data.build.r_tf_inboard_out - r_tf_inboard_out) < 1e-9, (
        "the placement did not land where it was asked -- `_place_tf_leg`'s unit-slope "
        "argument no longer holds"
    )
    return (
        data.build.r_vv_inboard_out,
        data.build.r_sh_inboard_in,
        data.build.r_sh_inboard_out,
    )


class TestVacuumVesselAndShieldRadii(Tier1Contract):
    """`calculate_vacuum_vessel_and_shield_radii` vs
    `calculate_radial_build:1833-1860`, `i_tf_inside_cs == TF_OUTSIDE_CS`.

    The producer that closed the cold tokamak SAND probe's last non-finite condition
    (`.build.r_vv_inboard_out`, which divides in `vv_stress_on_quench`) and the
    centrepost cluster's one declared-but-unproduced read (`.build.r_sh_inboard_out`).
    """

    audit_record = "models/build.md"
    reference = _reference_vacuum_vessel_and_shield_radii
    ported = calculate_vacuum_vessel_and_shield_radii

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            r_tf_inboard_out=3.8986074717418546,
            dr_tf_shld_gap=0.05,
            dr_shld_thermal_inboard=0.05,
            dr_shld_vv_gap_inboard=0.02,
            dr_vv_inboard=0.3,
            dr_shld_inboard=0.3,
        ),
    ]

    fuzz_bounds = {
        # Placed by moving `dr_tf_inboard` off its 1.2 m baseline, so the range is
        # bounded below by the radius at which that thickness would go negative.
        "r_tf_inboard_out": (3.0, 6.0),
        "dr_tf_shld_gap": (0.01, 0.2),
        "dr_shld_thermal_inboard": (0.01, 0.2),
        "dr_shld_vv_gap_inboard": (0.005, 0.2),
        "dr_vv_inboard": (0.1, 0.8),
        "dr_shld_inboard": (0.1, 1.0),
    }


def _reference_rbld(
    r_sh_inboard_out,
    dr_shld_blkt_gap,
    dr_blkt_inboard,
    dr_fw_inboard,
    dr_fw_plasma_gap_inboard,
    rminor,
):
    """`calculate_radial_build:1862-1870`.

    `r_sh_inboard_out` is placed through `dr_tf_inboard` for the reason
    `_place_tf_leg` gives, offset by the five thicknesses that separate it from the TF
    leg on this arm -- all at their baseline here, so the offset is a constant of the
    run measured once from a probe, not a second fit.
    """
    overrides = {
        "build__dr_shld_blkt_gap": dr_shld_blkt_gap,
        "build__dr_blkt_inboard": dr_blkt_inboard,
        "build__dr_fw_inboard": dr_fw_inboard,
        "build__dr_fw_plasma_gap_inboard": dr_fw_plasma_gap_inboard,
        "physics__rminor": rminor,
    }
    probe = _radial(**overrides)
    offset = probe.build.r_sh_inboard_out - probe.build.r_tf_inboard_out
    data = _radial(
        build__dr_tf_inboard=_place_tf_leg(r_sh_inboard_out - offset, **overrides),
        **overrides,
    )
    assert abs(data.build.r_sh_inboard_out - r_sh_inboard_out) < 1e-9, (
        "the placement did not land where it was asked"
    )
    return data.build.rbld


class TestRbld(Tier1Contract):
    """`calculate_rbld` vs `calculate_radial_build:1862-1870`.

    PROCESS's own comment on this accumulation is "should be equal to `rmajor`" and
    constraint 11 is the equation that asserts it -- active on `low_aspect_ratio_DEMO`,
    `spherical_tokamak_eval` and `st_regression`, which is why the value is produced
    rather than left at the boundary.
    """

    audit_record = "models/build.md"
    reference = _reference_rbld
    ported = calculate_rbld

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            r_sh_inboard_out=4.6186074717418535,
            dr_shld_blkt_gap=0.02,
            dr_blkt_inboard=0.7,
            dr_fw_inboard=0.018000000000000002,
            dr_fw_plasma_gap_inboard=0.25,
            rminor=2.6666666666666665,
        ),
    ]

    fuzz_bounds = {
        # Same bound as `TestVacuumVesselAndShieldRadii`, shifted by the constant
        # offset between the two radii on this arm.
        "r_sh_inboard_out": (4.0, 7.0),
        "dr_shld_blkt_gap": (0.005, 0.1),
        "dr_blkt_inboard": (0.3, 1.2),
        "dr_fw_inboard": (0.005, 0.08),
        "dr_fw_plasma_gap_inboard": (0.05, 0.6),
        "rminor": (1.5, 4.0),
    }


def _reference_r_cp_top(r_tf_inboard_out):
    """`Build.calculate_radial_build`'s `else` at `:1812-1813`, `itart == 0`.

    Driven through `_place_tf_leg` for the reason that helper gives: PROCESS computes
    `r_tf_inboard_out` 60 lines above the block under test, so it cannot be overridden
    onto the `DataStructure` directly. `BASELINE`'s `itart` is `0`, which is the arm.
    """
    data = _radial(build__dr_tf_inboard=_place_tf_leg(r_tf_inboard_out))
    assert abs(data.build.r_tf_inboard_out - r_tf_inboard_out) < 1e-12, (
        "the placement did not land where it was asked"
    )
    return data.build.r_cp_top


class TestRCpTop(Tier1Contract):
    """`calculate_r_cp_top_from_tf_inboard_out` vs `calculate_radial_build:1812-1813`.

    An identity, tested for the same two reasons `TestDivertorGeometrySphericalTokamak`
    is: it pins the *write set* (that this arm writes `r_cp_top` and leaves
    `.build.f_r_cp` alone, which the three refused arms do not -- see the `assert` in
    the sibling contract below) and it gives the gradient check a row to look at.

    The legacy value is `st_regression.IN.DAT` at PROCESS's own solution, where
    `r_cp_top == r_tf_inboard_out == 1.3405301988363134` -- one of the four frozen
    boundary paths `optimise_design.md` §26.2 found, and the only one this port's own
    pins had already reported (`reference_provider_st_regression.txt`'s single
    `computed` row).
    """

    audit_record = "models/build.md"
    reference = _reference_r_cp_top
    ported = calculate_r_cp_top_from_tf_inboard_out

    samples = [
        legacy_sample("st_regression-converged", r_tf_inboard_out=1.3405301988363134),
        legacy_sample(
            "spherical_tokamak_eval-converged", r_tf_inboard_out=1.208855401921066
        ),
    ]

    fuzz_bounds = {"r_tf_inboard_out": (0.8, 5.0)}


def _reference_r_cp_top_superconducting_spherical_tokamak(r_tf_inboard_out):
    """The same PROCESS lines reached with `itart = 1`, `i_tf_sup = 1` **and**
    `i_r_cp_top = 2`, `f_r_cp = 1.4` -- both tracked spherical tokamaks' own switches.

    **This adapter exists to execute a claim rather than assert it.** PROCESS's guard
    is `if itart == 1 and i_tf_sup != 1:` (`build.py:1750`), so a superconducting
    spherical tokamak takes the `else` and `.build.i_r_cp_top` is never read -- even
    though both ST input files set it to `2`, whose formula would give
    `f_r_cp * r_tf_inboard_out = 1.4 * r_tf_inboard_out`. `f_r_cp` is set to `1.4` here
    for exactly that reason: were the dispatch keyed on `i_r_cp_top` first, this
    reference would return 40% more than the port and the value test would fail rather
    than quietly agree. The `assert` pins the other half of the claim -- the three
    refused arms all *write* `.build.f_r_cp` and this one must not.

    Same lever, same helper, same `else` branch as `_reference_r_cp_top`: two
    contracts, because `Tier1Contract` takes one reference and the two switch settings
    are two different journeys to it.
    """
    f_r_cp_before = 1.4
    data = _radial(
        build__dr_tf_inboard=_place_tf_leg(r_tf_inboard_out),
        physics__itart=1,
        tfcoil__i_tf_sup=1,
        build__i_r_cp_top=2,
        build__f_r_cp=f_r_cp_before,
    )
    assert abs(data.build.r_tf_inboard_out - r_tf_inboard_out) < 1e-12, (
        "the placement did not land where it was asked"
    )
    assert data.build.f_r_cp == f_r_cp_before, (
        "this arm must leave .build.f_r_cp alone; the three refused arms write it"
    )
    return data.build.r_cp_top


class TestRCpTopSuperconductingSphericalTokamak(Tier1Contract):
    """`calculate_r_cp_top_from_tf_inboard_out` vs the same two PROCESS lines reached
    through `itart = 1, i_tf_sup = 1, i_r_cp_top = 2` -- the two tracked spherical
    tokamaks' literal configuration, and the executable form of
    `indat._r_cp_top_arm`'s claim that `i_r_cp_top` is inert on both of them.
    """

    audit_record = "models/build.md"
    reference = _reference_r_cp_top_superconducting_spherical_tokamak
    ported = calculate_r_cp_top_from_tf_inboard_out

    samples = [
        legacy_sample("st_regression-converged", r_tf_inboard_out=1.3405301988363134),
        legacy_sample(
            "spherical_tokamak_eval-converged", r_tf_inboard_out=1.208855401921066
        ),
    ]

    fuzz_bounds = {"r_tf_inboard_out": (0.8, 5.0)}
