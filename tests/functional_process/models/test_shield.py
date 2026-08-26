"""Harness cases for the ported shield model (`functional_process/models/shield.py`).

`calculate_elliptical_shield_volumes` and `calculate_dshaped_shield_volumes` are diffed
directly against `Shield`'s own `@staticmethod`s -- they take no `self.data` access at
all, so no adapter is needed and `reference = staticmethod(Shield.<name>)` is exact.

`calculate_shield_half_height_{single,double}_null` have no standalone PROCESS function
of their own shape -- both are one branch of `Shield.calculate_shield_half_height`, which
*is* a real staticmethod taking `n_divertors` as a plain argument -- so each is diffed
against a thin adapter that calls the real `Shield.calculate_shield_half_height` with
`n_divertors` fixed to the value that selects the branch under test (dummy zeros for the
other branch's unused parameters, which that branch's arithmetic never touches).

`calculate_shield_volumes_elliptical` (the coverage-factor-adjusted composite) has no
PROCESS function of this shape either -- it is `Shield.run()`'s coverage-factor block,
inline code -- so it is diffed against `_run_shield`, which builds a real `DataStructure`
with `large_tokamak_eval.IN.DAT`'s own switch values (`itart=0`,
`i_fw_blkt_vv_shape=ELLIPTICAL_SHAPED` (the dataclass default), `n_divertors=1`), calls
the real, bound `Shield.run()`, and reads `.blanket.vol_shld_inboard`/
`.blanket.vol_shld_outboard`/`.fwbs.vol_shld_total` back -- the same "close the `data`
backdoor" technique `plasma_geometry.md`'s `_run_plasma_geom` uses.

Sample values are a physically-plausible ITER/DEMO-scale synthetic operating point (no
recorded legacy expectation for this exact function shape in `tests/unit`), chosen to
keep both elliptical radii comfortably positive (`r_2 = r_1 - r_shld_inboard_inner -
dr_shld_inboard`, `r_3 = r_shld_outboard_outer - r_1 - dr_shld_outboard`) -- same
provenance class `plasma_geometry.md`'s `sauter_geometry`/`plasma_poloidal_perimeter`
samples used.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.shield import (
    calculate_dshaped_shield_volumes,
    calculate_elliptical_shield_volumes,
    calculate_shield_half_height_double_null,
    calculate_shield_half_height_single_null,
    calculate_shield_volumes_elliptical,
)
from process.core.model import DataStructure
from process.models.build import FwBlktVVShape
from process.models.shield import Shield


def _reference_half_height_double_null(
    z_plasma_xpoint_lower, dz_xpoint_divertor, dz_divertor
):
    return Shield.calculate_shield_half_height(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        n_divertors=2,
        z_plasma_xpoint_upper=0.0,
        dr_fw_plasma_gap_inboard=0.0,
        dr_fw_plasma_gap_outboard=0.0,
        dr_fw_inboard=0.0,
        dr_fw_outboard=0.0,
        dz_blkt_upper=0.0,
    )


def _reference_half_height_single_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    z_plasma_xpoint_upper,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_inboard,
    dr_fw_outboard,
    dz_blkt_upper,
):
    return Shield.calculate_shield_half_height(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        n_divertors=1,
        z_plasma_xpoint_upper=z_plasma_xpoint_upper,
        dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
        dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
        dr_fw_inboard=dr_fw_inboard,
        dr_fw_outboard=dr_fw_outboard,
        dz_blkt_upper=dz_blkt_upper,
    )


def _run_shield(
    r_shld_inboard_inner,
    r_shld_outboard_outer,
    rmajor,
    triang,
    dr_shld_inboard,
    rminor,
    dr_shld_outboard,
    dz_shld_upper,
    fvolsi,
    fvolso,
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    z_plasma_xpoint_upper,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_inboard,
    dr_fw_outboard,
    dz_blkt_upper,
):
    """Build a `DataStructure`, run the real `Shield.run()`, return `data`.

    Switch values match `large_tokamak_eval.IN.DAT`: `itart=0`,
    `i_fw_blkt_vv_shape=ELLIPTICAL_SHAPED` (the dataclass default, unset in the IN.DAT),
    `n_divertors=1` (`i_single_null=1`, resolved before any model runs).
    """
    data = DataStructure()
    data.build.r_shld_inboard_inner = r_shld_inboard_inner
    data.build.r_shld_outboard_outer = r_shld_outboard_outer
    data.build.dr_shld_inboard = dr_shld_inboard
    data.build.dr_shld_outboard = dr_shld_outboard
    data.build.dz_shld_upper = dz_shld_upper
    data.build.z_plasma_xpoint_lower = z_plasma_xpoint_lower
    data.build.z_plasma_xpoint_upper = z_plasma_xpoint_upper
    data.build.dr_fw_plasma_gap_inboard = dr_fw_plasma_gap_inboard
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.build.dz_blkt_upper = dz_blkt_upper
    data.build.dz_xpoint_divertor = dz_xpoint_divertor

    data.divertor.dz_divertor = dz_divertor
    data.divertor.n_divertors = 1

    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.physics.triang = triang
    data.physics.itart = 0

    data.fwbs.fvolsi = fvolsi
    data.fwbs.fvolso = fvolso
    data.fwbs.i_fw_blkt_vv_shape = FwBlktVVShape.ELLIPTICAL_SHAPED

    shield = Shield()
    shield.data = data
    shield.run()
    return data


def _reference_calculate_shield_volumes_elliptical(
    r_shld_inboard_inner,
    r_shld_outboard_outer,
    rmajor,
    triang,
    dr_shld_inboard,
    rminor,
    dz_shld_half,
    dr_shld_outboard,
    dz_shld_upper,
    fvolsi,
    fvolso,
):
    """`dz_shld_half` is dropped: `_run_shield` computes its own via the real
    `Shield.run()`'s half-height call, using a fixed single-null geometry -- the
    composite under test is only the volumes-plus-coverage-factors half, so this adapter
    matches the *sample*'s `dz_shld_half` by holding the half-height inputs fixed at
    values known to reproduce it (see the sample below).
    """
    data = _run_shield(
        r_shld_inboard_inner=r_shld_inboard_inner,
        r_shld_outboard_outer=r_shld_outboard_outer,
        rmajor=rmajor,
        triang=triang,
        dr_shld_inboard=dr_shld_inboard,
        rminor=rminor,
        dr_shld_outboard=dr_shld_outboard,
        dz_shld_upper=dz_shld_upper,
        fvolsi=fvolsi,
        fvolso=fvolso,
        z_plasma_xpoint_lower=4.0,
        dz_xpoint_divertor=0.5,
        dz_divertor=0.3,
        z_plasma_xpoint_upper=4.0,
        dr_fw_plasma_gap_inboard=0.25,
        dr_fw_plasma_gap_outboard=0.25,
        dr_fw_inboard=0.05,
        dr_fw_outboard=0.05,
        dz_blkt_upper=0.3,
    )
    assert abs(data.blanket.dz_shld_half - dz_shld_half) < 1e-9, (
        "the fixed half-height inputs in this adapter must reproduce the sample's own "
        "dz_shld_half -- see the module docstring"
    )
    return (
        data.blanket.vol_shld_inboard,
        data.blanket.vol_shld_outboard,
        data.fwbs.vol_shld_total,
    )


class TestCalculateShieldHalfHeightDoubleNull(Tier1Contract):
    """`calculate_shield_half_height_double_null` -> `n_divertors == 2` branch."""

    audit_record = "models/shield.md"
    reference = staticmethod(_reference_half_height_double_null)
    ported = calculate_shield_half_height_double_null

    samples = [
        legacy_sample(
            "shield_half_height_double_null-synthetic",
            z_plasma_xpoint_lower=4.0,
            dz_xpoint_divertor=0.5,
            dz_divertor=0.3,
        ),
    ]

    fuzz_bounds = {
        "z_plasma_xpoint_lower": (1.0, 8.0),
        "dz_xpoint_divertor": (0.1, 1.0),
        "dz_divertor": (0.1, 1.0),
    }


class TestCalculateShieldHalfHeightSingleNull(Tier1Contract):
    """`calculate_shield_half_height_single_null` -> `n_divertors != 2` branch, live on
    `large_tokamak_eval.IN.DAT`.
    """

    audit_record = "models/shield.md"
    reference = staticmethod(_reference_half_height_single_null)
    ported = calculate_shield_half_height_single_null

    samples = [
        legacy_sample(
            "shield_half_height_single_null-synthetic",
            z_plasma_xpoint_lower=4.0,
            dz_xpoint_divertor=0.5,
            dz_divertor=0.3,
            z_plasma_xpoint_upper=4.0,
            dr_fw_plasma_gap_inboard=0.25,
            dr_fw_plasma_gap_outboard=0.25,
            dr_fw_inboard=0.05,
            dr_fw_outboard=0.05,
            dz_blkt_upper=0.3,
        ),
    ]

    fuzz_bounds = {
        "z_plasma_xpoint_lower": (1.0, 8.0),
        "dz_xpoint_divertor": (0.1, 1.0),
        "dz_divertor": (0.1, 1.0),
        "z_plasma_xpoint_upper": (1.0, 8.0),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "dr_fw_inboard": (0.01, 0.2),
        "dr_fw_outboard": (0.01, 0.2),
        "dz_blkt_upper": (0.0, 0.6),
    }


class TestCalculateEllipticalShieldVolumes(Tier1Contract):
    """`calculate_elliptical_shield_volumes` -> the same, unchanged."""

    audit_record = "models/shield.md"
    reference = staticmethod(Shield.calculate_elliptical_shield_volumes)
    ported = calculate_elliptical_shield_volumes

    samples = [
        legacy_sample(
            "elliptical_shield_volumes-synthetic",
            r_shld_inboard_inner=5.0,
            r_shld_outboard_outer=13.0,
            rmajor=8.0,
            triang=0.5,
            dr_shld_inboard=0.4,
            rminor=2.5,
            dz_shld_half=4.7,
            dr_shld_outboard=0.8,
            dz_shld_upper=0.3,
        ),
    ]

    fuzz_bounds = {
        "r_shld_inboard_inner": (2.5, 3.5),
        "r_shld_outboard_outer": (12.0, 15.0),
        "rmajor": (6.0, 10.0),
        "triang": (0.0, 0.3),
        "dr_shld_inboard": (0.2, 0.4),
        "rminor": (2.0, 3.0),
        "dz_shld_half": (2.0, 6.0),
        "dr_shld_outboard": (0.5, 1.0),
        "dz_shld_upper": (0.1, 0.8),
    }


class TestCalculateDshapedShieldVolumes(Tier1Contract):
    """`calculate_dshaped_shield_volumes` -> the same, unchanged.

    Ported for completeness; not wired to an occupant class (not live on
    `large_tokamak_eval.IN.DAT`).
    """

    audit_record = "models/shield.md"
    reference = staticmethod(Shield.calculate_dshaped_shield_volumes)
    ported = calculate_dshaped_shield_volumes

    samples = [
        legacy_sample(
            "dshaped_shield_volumes-synthetic",
            r_shld_inboard_inner=5.0,
            dr_shld_inboard=0.4,
            dr_fw_inboard=0.03,
            dr_fw_plasma_gap_inboard=0.25,
            rminor=2.5,
            dr_fw_plasma_gap_outboard=0.25,
            dr_fw_outboard=0.03,
            dr_blkt_inboard=0.4,
            dr_blkt_outboard=0.6,
            dz_shld_half=4.7,
            dr_shld_outboard=0.8,
            dz_shld_upper=0.3,
        ),
    ]

    fuzz_bounds = {
        "r_shld_inboard_inner": (2.5, 3.5),
        "dr_shld_inboard": (0.2, 0.4),
        "dr_fw_inboard": (0.01, 0.1),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "rminor": (2.0, 3.0),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "dr_fw_outboard": (0.01, 0.1),
        "dr_blkt_inboard": (0.1, 0.6),
        "dr_blkt_outboard": (0.1, 0.8),
        "dz_shld_half": (2.0, 6.0),
        "dr_shld_outboard": (0.5, 1.0),
        "dz_shld_upper": (0.1, 0.8),
    }


class TestCalculateShieldVolumesElliptical(Tier1Contract):
    """`calculate_shield_volumes_elliptical` -> `Shield.run()`'s own
    `.blanket.vol_shld_inboard`/`.blanket.vol_shld_outboard`/`.fwbs.vol_shld_total`, the
    node that owns this unit's one target output.
    """

    audit_record = "models/shield.md"
    reference = staticmethod(_reference_calculate_shield_volumes_elliptical)
    ported = calculate_shield_volumes_elliptical

    static_argnames = ("dz_shld_half",)
    """Not a switch -- held out of differentiation here because `_run_shield` cannot
    hold it fixed while perturbing it independently (the real `Shield.run()` always
    recomputes it from the half-height inputs, never reads it as a boundary input; see
    the module docstring). `d(output)/d(dz_shld_half)` is already fully covered by
    `TestCalculateEllipticalShieldVolumes` above, which diffs `calculate_elliptical_
    shield_volumes` directly against the real `Shield.calculate_elliptical_shield_
    volumes` staticmethod, `dz_shld_half` included as an ordinary differentiable
    argument."""

    samples = [
        legacy_sample(
            "shield_volumes_elliptical-large_tokamak_eval",
            r_shld_inboard_inner=5.0,
            r_shld_outboard_outer=13.0,
            rmajor=8.0,
            triang=0.5,
            dr_shld_inboard=0.4,
            rminor=2.5,
            dz_shld_half=4.7,
            dr_shld_outboard=0.8,
            dz_shld_upper=0.3,
            fvolsi=1.0,
            fvolso=0.64,
        ),
    ]

    fuzz_bounds = {
        "r_shld_inboard_inner": (2.5, 3.5),
        "r_shld_outboard_outer": (12.0, 15.0),
        "rmajor": (6.0, 10.0),
        "triang": (0.0, 0.3),
        "dr_shld_inboard": (0.2, 0.4),
        "rminor": (2.0, 3.0),
        "dz_shld_half": (4.7, 4.7),
        "dr_shld_outboard": (0.5, 1.0),
        "dz_shld_upper": (0.1, 0.8),
        "fvolsi": (0.5, 1.0),
        "fvolso": (0.3, 1.0),
    }
