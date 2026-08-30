"""Harness cases for the ported subset of `blankets/blanket_library.py`.

The four functions of the `component_volumes` chain that lie on the minimal closure
producing `.tokamak.ccfe_hcpb`'s boundary variables -- see
`functional_process/_audit/units/models/blankets/blanket_library.md`.

Every reference is PROCESS's own callable, driven in-process (the point of the
`process_port` env). Three of the four are `@staticmethod`s on `BlanketLibrary` and need
no adapter beyond a keyword rename; `apply_coverage_factors` is `self`-bound and gets the
usual bind-a-`DataStructure` treatment.

Legacy sample points are lifted verbatim from
`tests/unit/models/blankets/test_blanket_library.py`, whose own docstrings record that
they were generated from `large_tokamak.IN.DAT`/`large_tokamak_eval.IN.DAT` -- the same
reference run `_audit/tokamak_call_surface.md` traced, so they are on the operating point
this port targets rather than near it.

2026-08-27: the two `n_divertors == 2` occupants joined
(`TestBlktHalfHeightDoubleNull`, `TestApplyCoverageFactorsDoubleNull`). The half-height
adapter **poisons** the five parameters the double-null arm does not read with `nan`
rather than zeroing them, so "PROCESS does not look at these" is executed rather than
asserted: were the branch not taken, the reference would return `nan` and the value
comparison would fail instead of quietly agreeing on a zero.

2026-08-27 (the D-shaped wave): `TestDshapedBlktAreas` and `TestDshapedBlktVolumes`
joined, making both shape slots total. Their references are `@staticmethod`s like the
elliptical pair's, so they need no adapter -- and they need no `nan` poisoning either,
because the D-shaped arm takes a *different signature* rather than a subset of the same
one: `triang`, `rmajor` and the two outboard shield radii are absent from the PROCESS
staticmethod's own parameter list, so there is nothing to poison. That is a stronger
guarantee than a poisoned argument, not a weaker one.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.blankets.blanket_library import (
    apply_coverage_factors_double_null,
    apply_coverage_factors_single_null,
    calculate_blkt_half_height_double_null,
    calculate_blkt_half_height_single_null,
    calculate_blkt_inboard_poloidal_plasma_angle,
    calculate_dshaped_blkt_areas,
    calculate_dshaped_blkt_volumes,
    calculate_elliptical_blkt_areas,
    calculate_elliptical_blkt_volumes,
)
from process.core.model import DataStructure
from process.models.blankets.hcpb import CCFE_HCPB
from process.models.fw import FirstWall

_AUDIT_RECORD = "models/blankets/blanket_library.md"

# Reference-run geometry, read out of the assembled `DataStructure` after four
# `_call_models_once` passes on `tests/regression/input_files/large_tokamak_eval.IN.DAT`.
# Used as the second legacy point for each contract, so every unit is checked at the
# operating point the tokamak port is actually being assembled for and not only at the
# point PROCESS's own unit tests happen to carry.
_RUN_GEOMETRY = {
    "rmajor": 8.0,
    "rminor": 2.6666666666666665,
    "triang": 0.5,
    "r_shld_inboard_inner": 4.065333333333334,
    "dr_shld_inboard": 0.3,
    "dr_blkt_inboard": 0.7,
    "r_shld_outboard_outer": 12.734666666666667,
    "dr_shld_outboard": 0.8,
    "dr_blkt_outboard": 1.0,
    "dz_blkt_half": 5.953275248730413,
}

_GEOMETRY_FUZZ = {
    # Deliberately tight around the reference point. `_eshellarea`/`_eshellvol` divide by
    # `r1 - r_shld_inboard_inner - dr_shld_inboard - dr_blkt_inboard`, which goes through
    # zero a short way outside these bounds; a wider box would spend most draws outside
    # the model's domain rather than exercising it.
    "rmajor": (7.5, 8.5),
    "rminor": (2.4, 2.9),
    "triang": (0.4, 0.6),
    "r_shld_inboard_inner": (3.8, 4.3),
    "dr_shld_inboard": (0.2, 0.4),
    "dr_blkt_inboard": (0.6, 0.8),
    "r_shld_outboard_outer": (12.2, 13.2),
    "dr_shld_outboard": (0.6, 1.0),
    "dr_blkt_outboard": (0.8, 1.2),
    "dz_blkt_half": (5.5, 6.5),
}

_DSHAPED_GEOMETRY_FUZZ = {
    # The D-shaped arm reads a different set: no `rmajor`, no `triang`, no outboard
    # shield radius, and four first-wall thicknesses/gaps the elliptical arm never sees.
    # `dshellarea`/`dshellvol` divide by `r2`, the width across the plasma, which stays
    # comfortably positive over this box -- so unlike `_GEOMETRY_FUZZ` the bounds here
    # are not defending a near-singularity, only keeping the draws physical.
    "r_shld_inboard_inner": (3.8, 4.3),
    "dr_shld_inboard": (0.2, 0.4),
    "dr_blkt_inboard": (0.6, 0.8),
    "dr_fw_inboard": (0.01, 0.1),
    "dr_fw_plasma_gap_inboard": (0.05, 0.5),
    "rminor": (2.4, 2.9),
    "dr_fw_plasma_gap_outboard": (0.05, 0.5),
    "dr_fw_outboard": (0.01, 0.1),
    "dz_blkt_half": (5.5, 6.5),
}

_COVERAGE_FUZZ = {
    # `f_ster_div_single` stops well short of `0.5`, which is where the double-null
    # arm's `1 - 2 * f_ster_div_single` changes sign; both arms are exercised over the
    # same box so a difference between them is the branch and not the domain.
    "a_blkt_total_surface_full_coverage": (800.0, 3000.0),
    "a_blkt_inboard_surface_full_coverage": (300.0, 900.0),
    "f_ster_div_single": (0.02, 0.2),
    "f_a_fw_outboard_hcd": (0.0, 0.1),
    "vol_blkt_total_full_coverage": (600.0, 2500.0),
    "vol_blkt_inboard_full_coverage": (150.0, 600.0),
}


def _blanket_library():
    """A `CCFE_HCPB` with a fresh `DataStructure`, used as a `BlanketLibrary`.

    `CCFE_HCPB(OutboardBlanket, InboardBlanket)` is how `blanket_library.py` is reached
    at all on the tokamak path (`hcpb.py:25`; `models.blanket_library` itself is
    constructed at `main.py:678` and never called), so binding the subclass rather than
    `BlanketLibrary` directly is the call site the port is modelling.
    """
    model = CCFE_HCPB(fw=FirstWall())
    model.data = DataStructure()
    return model


def _reference_blkt_half_height_single_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
    z_plasma_xpoint_upper,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_inboard,
    dr_fw_outboard,
):
    """`calculate_blkt_half_height` at `n_divertors == 1`.

    `n_divertors` is supplied by the adapter, not by the port: it selected the occupant.
    """
    return CCFE_HCPB.calculate_blkt_half_height(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        z_plasma_xpoint_upper=z_plasma_xpoint_upper,
        dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
        dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
        dr_fw_inboard=dr_fw_inboard,
        dr_fw_outboard=dr_fw_outboard,
        dz_blkt_upper=dz_blkt_upper,
        n_divertors=1,
    )


def _reference_blkt_half_height_double_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
):
    """`calculate_blkt_half_height` at `n_divertors == 2`.

    The five parameters this arm does not read are passed as `nan`, not `0.0` -- see the
    module docstring. PROCESS's signature still demands values for them.
    """
    return CCFE_HCPB.calculate_blkt_half_height(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        z_plasma_xpoint_upper=np.nan,
        dr_fw_plasma_gap_inboard=np.nan,
        dr_fw_plasma_gap_outboard=np.nan,
        dr_fw_inboard=np.nan,
        dr_fw_outboard=np.nan,
        dz_blkt_upper=dz_blkt_upper,
        n_divertors=2,
    )


def _reference_elliptical_blkt_areas(**kwargs):
    """`calculate_elliptical_blkt_areas`, already a bare `@staticmethod`."""
    return CCFE_HCPB.calculate_elliptical_blkt_areas(**kwargs)


def _reference_elliptical_blkt_volumes(**kwargs):
    """`calculate_elliptical_blkt_volumes`, already a bare `@staticmethod`."""
    return CCFE_HCPB.calculate_elliptical_blkt_volumes(**kwargs)


def _reference_apply_coverage_factors_single_null(
    a_blkt_total_surface_full_coverage,
    a_blkt_inboard_surface_full_coverage,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    vol_blkt_total_full_coverage,
    vol_blkt_inboard_full_coverage,
):
    """Bind a `DataStructure` and call `apply_coverage_factors()` at `n_divertors == 1`.

    Writing this adapter is where the audit's "close the `data` back-door" claim gets
    tested rather than asserted: if the port read a field this seeds nothing into, the
    two would disagree.
    """
    model = _blanket_library()
    data = model.data

    data.divertor.n_divertors = 1
    data.build.a_blkt_total_surface_full_coverage = a_blkt_total_surface_full_coverage
    data.build.a_blkt_inboard_surface_full_coverage = (
        a_blkt_inboard_surface_full_coverage
    )
    data.fwbs.f_ster_div_single = f_ster_div_single
    data.fwbs.f_a_fw_outboard_hcd = f_a_fw_outboard_hcd
    data.fwbs.vol_blkt_total_full_coverage = vol_blkt_total_full_coverage
    data.fwbs.vol_blkt_inboard_full_coverage = vol_blkt_inboard_full_coverage

    model.apply_coverage_factors()

    return (
        data.build.a_blkt_outboard_surface,
        data.build.a_blkt_total_surface,
        data.fwbs.vol_blkt_outboard,
        data.fwbs.vol_blkt_inboard,
        data.build.a_blkt_inboard_surface,
        data.fwbs.vol_blkt_total,
    )


def _reference_apply_coverage_factors_double_null(
    a_blkt_total_surface_full_coverage,
    a_blkt_inboard_surface_full_coverage,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    vol_blkt_total_full_coverage,
    vol_blkt_inboard_full_coverage,
):
    """Bind a `DataStructure` and call `apply_coverage_factors()` at `n_divertors == 2`.

    Same six fields as the single-null adapter -- the arms differ by a literal, not by a
    read -- so this is where PROCESS's areas-doubled/volumes-not asymmetry gets executed
    rather than argued about: if the port had "fixed" the volume line, the two would
    disagree here.
    """
    model = _blanket_library()
    data = model.data

    data.divertor.n_divertors = 2
    data.build.a_blkt_total_surface_full_coverage = a_blkt_total_surface_full_coverage
    data.build.a_blkt_inboard_surface_full_coverage = (
        a_blkt_inboard_surface_full_coverage
    )
    data.fwbs.f_ster_div_single = f_ster_div_single
    data.fwbs.f_a_fw_outboard_hcd = f_a_fw_outboard_hcd
    data.fwbs.vol_blkt_total_full_coverage = vol_blkt_total_full_coverage
    data.fwbs.vol_blkt_inboard_full_coverage = vol_blkt_inboard_full_coverage

    model.apply_coverage_factors()

    return (
        data.build.a_blkt_outboard_surface,
        data.build.a_blkt_total_surface,
        data.fwbs.vol_blkt_outboard,
        data.fwbs.vol_blkt_inboard,
        data.build.a_blkt_inboard_surface,
        data.fwbs.vol_blkt_total,
    )


class TestBlktHalfHeightSingleNull(Tier1Contract):
    """`calculate_blkt_half_height_single_null` -> the `n_divertors != 2` arm.

    Legacy point is `test_blanket_library.py::test_calculate_blkt_half_height`'s single
    parametrised case (`n_divertors=1`, from `large_tokamak.IN.DAT`).
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_blkt_half_height_single_null
    ported = calculate_blkt_half_height_single_null

    samples = [
        legacy_sample(
            "half-height-large-tokamak",
            z_plasma_xpoint_lower=4.93333333333333333,
            dz_xpoint_divertor=2.0018838307941582,
            dz_divertor=0.62000000000000011,
            dz_blkt_upper=0.85000000000000009,
            z_plasma_xpoint_upper=4.93333333333333333,
            dr_fw_plasma_gap_inboard=0.25,
            dr_fw_plasma_gap_outboard=0.25,
            dr_fw_inboard=0.018000000000000002,
            dr_fw_outboard=0.018000000000000002,
        ),
    ]

    fuzz_bounds = {
        "z_plasma_xpoint_lower": (3.0, 7.0),
        "dz_xpoint_divertor": (1.0, 3.0),
        "dz_divertor": (0.3, 1.0),
        "dz_blkt_upper": (0.5, 1.2),
        "z_plasma_xpoint_upper": (3.0, 7.0),
        "dr_fw_plasma_gap_inboard": (0.1, 0.5),
        "dr_fw_plasma_gap_outboard": (0.1, 0.5),
        "dr_fw_inboard": (0.005, 0.05),
        "dr_fw_outboard": (0.005, 0.05),
    }


class TestBlktHalfHeightDoubleNull(Tier1Contract):
    """`calculate_blkt_half_height_double_null` -> the `n_divertors == 2` arm.

    Same geometry point as the single-null case above, minus the five parameters this
    arm does not read -- so the two contracts share an operating point and differ only
    in the branch, which is the comparison worth being able to make.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_blkt_half_height_double_null
    ported = calculate_blkt_half_height_double_null

    samples = [
        legacy_sample(
            "half-height-double-null-large-tokamak",
            z_plasma_xpoint_lower=4.93333333333333333,
            dz_xpoint_divertor=2.0018838307941582,
            dz_divertor=0.62000000000000011,
            dz_blkt_upper=0.85000000000000009,
        ),
    ]

    fuzz_bounds = {
        "z_plasma_xpoint_lower": (3.0, 7.0),
        "dz_xpoint_divertor": (1.0, 3.0),
        "dz_divertor": (0.3, 1.0),
        "dz_blkt_upper": (0.5, 1.2),
    }


class TestEllipticalBlktAreas(Tier1Contract):
    """`calculate_elliptical_blkt_areas` -> the same, unchanged.

    Two legacy points: `test_blanket_library.py::test_calculate_elliptical_blkt_areas`'s
    parametrised case, and the reference run's own converged geometry (the two differ in
    `r_shld_inboard_inner`/`r_shld_outboard_outer`, which move with the radial build).
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_elliptical_blkt_areas
    ported = calculate_elliptical_blkt_areas

    samples = [
        legacy_sample(
            "areas-large-tokamak-eval",
            rmajor=8,
            rminor=2.6666666666666665,
            triang=0.5,
            r_shld_inboard_inner=4.0833333333333339,
            dr_shld_inboard=0.30000000000000004,
            dr_blkt_inboard=0.70000000000000007,
            r_shld_outboard_outer=12.716666666666667,
            dr_shld_outboard=0.80000000000000004,
            dr_blkt_outboard=1,
            dz_blkt_half=5.9532752487304119,
        ),
        legacy_sample("areas-reference-run", **_RUN_GEOMETRY),
    ]

    fuzz_bounds = _GEOMETRY_FUZZ


class TestEllipticalBlktVolumes(Tier1Contract):
    """`calculate_elliptical_blkt_volumes` -> the same, unchanged."""

    audit_record = _AUDIT_RECORD
    reference = _reference_elliptical_blkt_volumes
    ported = calculate_elliptical_blkt_volumes

    samples = [
        legacy_sample(
            "volumes-large-tokamak-eval",
            rmajor=8,
            rminor=2.6666666666666665,
            triang=0.5,
            r_shld_inboard_inner=4.0833333333333339,
            dr_shld_inboard=0.30000000000000004,
            dr_blkt_inboard=0.70000000000000007,
            r_shld_outboard_outer=12.716666666666667,
            dr_shld_outboard=0.80000000000000004,
            dr_blkt_outboard=1,
            dz_blkt_half=5.9532752487304119,
            dz_blkt_upper=0.85000000000000009,
        ),
        legacy_sample("volumes-reference-run", **_RUN_GEOMETRY, dz_blkt_upper=0.85),
    ]

    fuzz_bounds = {**_GEOMETRY_FUZZ, "dz_blkt_upper": (0.5, 1.2)}


class TestApplyCoverageFactorsSingleNull(Tier1Contract):
    """`apply_coverage_factors_single_null` -> `apply_coverage_factors()` at
    `n_divertors == 1`.

    Legacy point is `test_blanket_library.py::test_apply_coverage_factors`'s parametrised
    case; the second is the reference run's own converged full-coverage areas and
    volumes, where `f_ster_div_single` is `divertor.py:42`'s computed value rather than
    the `0.115` the PROCESS test pins.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_apply_coverage_factors_single_null
    ported = apply_coverage_factors_single_null

    samples = [
        legacy_sample(
            "coverage-large-tokamak-eval",
            a_blkt_total_surface_full_coverage=1766.3354109399943,
            a_blkt_inboard_surface_full_coverage=664.9687712975541,
            f_ster_div_single=0.115,
            f_a_fw_outboard_hcd=0,
            vol_blkt_total_full_coverage=1336.207205897842,
            vol_blkt_inboard_full_coverage=315.83946385183026,
        ),
        legacy_sample(
            "coverage-reference-run",
            a_blkt_total_surface_full_coverage=1766.3354109399945,
            a_blkt_inboard_surface_full_coverage=663.622172160947,
            f_ster_div_single=0.0725040362777958,
            f_a_fw_outboard_hcd=0.0,
            vol_blkt_total_full_coverage=1338.8701833977761,
            vol_blkt_inboard_full_coverage=315.9239262058935,
        ),
    ]

    fuzz_bounds = _COVERAGE_FUZZ


class TestApplyCoverageFactorsDoubleNull(Tier1Contract):
    """`apply_coverage_factors_double_null` -> `apply_coverage_factors()` at
    `n_divertors == 2`.

    The same two legacy points as the single-null case, since the arms take the same six
    inputs. `f_ster_div_single` stays below `0.5` throughout so `1 - 2 * f_ster_div_
    single` remains positive -- above it the doubled coverage would exceed the whole
    sphere, which is not a regime PROCESS's formula means anything in.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_apply_coverage_factors_double_null
    ported = apply_coverage_factors_double_null

    samples = [
        legacy_sample(
            "coverage-double-null-large-tokamak-eval",
            a_blkt_total_surface_full_coverage=1766.3354109399943,
            a_blkt_inboard_surface_full_coverage=664.9687712975541,
            f_ster_div_single=0.115,
            f_a_fw_outboard_hcd=0,
            vol_blkt_total_full_coverage=1336.207205897842,
            vol_blkt_inboard_full_coverage=315.83946385183026,
        ),
        legacy_sample(
            "coverage-double-null-reference-run",
            a_blkt_total_surface_full_coverage=1766.3354109399945,
            a_blkt_inboard_surface_full_coverage=663.622172160947,
            f_ster_div_single=0.0725040362777958,
            f_a_fw_outboard_hcd=0.0,
            vol_blkt_total_full_coverage=1338.8701833977761,
            vol_blkt_inboard_full_coverage=315.9239262058935,
        ),
    ]

    fuzz_bounds = _COVERAGE_FUZZ


def _reference_dshaped_blkt_areas(**kwargs):
    """`calculate_dshaped_blkt_areas`, already a bare `@staticmethod`."""
    return CCFE_HCPB.calculate_dshaped_blkt_areas(**kwargs)


def _reference_dshaped_blkt_volumes(**kwargs):
    """`calculate_dshaped_blkt_volumes`, already a bare `@staticmethod`."""
    return CCFE_HCPB.calculate_dshaped_blkt_volumes(**kwargs)


class TestDshapedBlktAreas(Tier1Contract):
    """`calculate_dshaped_blkt_areas` -> the same, unchanged.

    No legacy sample: `tests/unit/models/blankets/test_blanket_library.py` parametrises
    only the elliptical pair, and the reference run this file's `_RUN_GEOMETRY` came from
    (`large_tokamak_eval.IN.DAT`) is elliptical, so its converged geometry does not carry
    the `dr_fw_*`/`dr_fw_plasma_gap_*` values this arm needs at a self-consistent point.
    The fuzz box below is anchored on the same radial build all the same -- `r1` walks
    outwards from `r_shld_inboard_inner` exactly as in the elliptical case -- so the two
    contracts exercise comparable geometry even though only one of them has a legacy
    point.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_dshaped_blkt_areas
    ported = calculate_dshaped_blkt_areas

    fuzz_bounds = _DSHAPED_GEOMETRY_FUZZ


class TestDshapedBlktVolumes(Tier1Contract):
    """`calculate_dshaped_blkt_volumes` -> the same, unchanged.

    `dr_blkt_inboard` doubles as `dshellvol`'s `drin`, whose inboard term
    `rmajor**2 - (rmajor - drin)**2` needs `drin < rmajor`; the box keeps it two orders
    of magnitude below `r1`, so the constraint is never near.
    """

    audit_record = _AUDIT_RECORD
    reference = _reference_dshaped_blkt_volumes
    ported = calculate_dshaped_blkt_volumes

    fuzz_bounds = {
        **_DSHAPED_GEOMETRY_FUZZ,
        "dr_blkt_outboard": (0.8, 1.2),
        "dz_blkt_upper": (0.5, 1.2),
    }


class TestBlktInboardPoloidalPlasmaAngle(Tier1Contract):
    """`calculate_blkt_inboard_poloidal_plasma_angle` ->
    `BlanketLibrary.calculate_blkt_inboard_poloidal_plasma_angle`, unchanged.

    Added 2026-08-30 with the producer. A `@staticmethod` with the same three
    parameters, so the reference is PROCESS's own callable with no adapter at all --
    which makes the point about why this one was missing: the *function* was always
    trivially portable, and what was absent was anyone asking whether the field it
    writes had an owner (see the port function's docstring).

    The legacy point is `large_tokamak_eval` at convergence and reproduces PROCESS's own
    `deg_blkt_inboard_poloidal_plasma = 127.79709387998703`.

    `rminor + dr_fw_plasma_gap_inboard` is the divisor and is bounded away from zero by
    both fuzz ranges; PROCESS has no guard there either, and a machine with no minor
    radius is not a case worth agreeing on.
    """

    audit_record = _AUDIT_RECORD
    reference = staticmethod(
        CCFE_HCPB.calculate_blkt_inboard_poloidal_plasma_angle,
    )
    ported = calculate_blkt_inboard_poloidal_plasma_angle

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            rminor=2.6666666666666665,
            dz_blkt_half=5.953275248730413,
            dr_fw_plasma_gap_inboard=0.25,
        ),
    ]

    fuzz_bounds = {
        "rminor": (1.0, 4.0),
        "dz_blkt_half": (1.0, 10.0),
        "dr_fw_plasma_gap_inboard": (0.05, 1.0),
    }
