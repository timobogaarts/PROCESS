"""Harness cases for the ported subset of `process/models/fw.py`
(`.tokamak.first_wall`).

Four units:

- `TestCalculateFirstWallHalfHeight`, `TestCalculateEllipticalFirstWallAreas`,
  `TestApplyFirstWallCoverageFactors` -- each a real PROCESS `@staticmethod`, diffed
  directly (with `n_divertors` fixed by the reference adapter where PROCESS's own
  signature still takes it).
- `TestCalculateFirstWallOutputs` -- the whole `FirstWall.run()` pipeline.
  `p_fw_alpha_mw` and `pflux_fw_neutron_mw` are inline `run()` arithmetic with no
  isolated PROCESS function of their own (same shape as `confinement_time.md`'s
  `plasma_power_loss_mw`), so this composite is their only diff against real PROCESS,
  not a standalone contract.
"""

import functools

import numpy as np

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.process_reference import process_reference
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.fw import (
    apply_first_wall_coverage_factors,
    apply_first_wall_coverage_factors_double_null,
    calculate_dshaped_first_wall_areas,
    calculate_elliptical_first_wall_areas,
    calculate_first_wall_half_height,
    calculate_first_wall_half_height_double_null,
    calculate_first_wall_outputs,
    calculate_first_wall_outputs_double_null,
    calculate_first_wall_outputs_dshaped_double_null,
    calculate_radiated_wall_load_scaled_plasma_surface,
    set_fw_geometry,
)
from process.core.model import DataStructure
from process.models.fw import FirstWall

_reference_first_wall_half_height = functools.partial(
    FirstWall.calculate_first_wall_half_height,
    n_divertors=1,
)


class TestCalculateFirstWallHalfHeight(Tier1Contract):
    """`calculate_first_wall_half_height` -> `FirstWall.calculate_first_wall_half_height`
    at `n_divertors == 1`.
    """

    audit_record = "models/fw.md"
    reference = _reference_first_wall_half_height
    ported = calculate_first_wall_half_height

    fuzz_bounds = {
        "z_plasma_xpoint_lower": (-10.0, -1.0),
        "dz_xpoint_divertor": (0.1, 2.0),
        "dz_divertor": (0.2, 2.0),
        "dz_blkt_upper": (0.1, 1.5),
        "z_plasma_xpoint_upper": (1.0, 10.0),
        "dz_fw_plasma_gap": (0.05, 1.0),
        "dr_fw_inboard": (0.01, 0.1),
        "dr_fw_outboard": (0.01, 0.1),
    }


_reference_first_wall_half_height_double_null = functools.partial(
    FirstWall.calculate_first_wall_half_height,
    z_plasma_xpoint_upper=np.nan,
    dz_fw_plasma_gap=np.nan,
    n_divertors=2,
)


class TestCalculateFirstWallHalfHeightDoubleNull(Tier1Contract):
    """`calculate_first_wall_half_height_double_null` ->
    `FirstWall.calculate_first_wall_half_height` at `n_divertors == 2`.
    """

    audit_record = "models/fw.md"
    reference = _reference_first_wall_half_height_double_null
    ported = calculate_first_wall_half_height_double_null

    # `z_plasma_xpoint_lower` is positive here where the single-null contract's box has
    # it negative. `process/models/build.py:170-172` assigns it `rminor * kappa` -- a
    # magnitude, always positive -- and on this arm the half-height *is* `z_bottom`, so
    # a negative draw would put the whole downstream geometry through a negative
    # half-height. The single-null arm averages it against a positive `z_top` and
    # survives its own unphysical box; this one does not, so it uses the real sign.
    fuzz_bounds = {
        "z_plasma_xpoint_lower": (3.0, 7.0),
        "dz_xpoint_divertor": (0.1, 2.0),
        "dz_divertor": (0.2, 2.0),
        "dz_blkt_upper": (0.1, 1.5),
        "dr_fw_inboard": (0.01, 0.1),
        "dr_fw_outboard": (0.01, 0.1),
    }


class TestCalculateEllipticalFirstWallAreas(Tier1Contract):
    """`calculate_elliptical_first_wall_areas` -> `FirstWall.
    calculate_elliptical_first_wall_areas`, unchanged signature.
    """

    audit_record = "models/fw.md"
    reference = staticmethod(FirstWall.calculate_elliptical_first_wall_areas)
    ported = calculate_elliptical_first_wall_areas

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "triang": (0.0, 0.8),
        "dz_fw_half": (1.0, 15.0),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
    }


_reference_first_wall_coverage_factors = functools.partial(
    FirstWall.apply_first_wall_coverage_factors,
    n_divertors=1,
)


class TestApplyFirstWallCoverageFactors(Tier1Contract):
    """`apply_first_wall_coverage_factors` -> `FirstWall.
    apply_first_wall_coverage_factors` at `n_divertors == 1`.

    Fuzz bounds keep `f_ster_div_single + f_a_fw_outboard_hcd` comfortably below 1 so
    `a_fw_outboard` stays positive -- PROCESS's own `ProcessValueError` guard on a
    non-credible outboard area (dropped in the port, `fw.md` § deviations) is never
    exercised here.
    """

    audit_record = "models/fw.md"
    reference = _reference_first_wall_coverage_factors
    ported = apply_first_wall_coverage_factors

    fuzz_bounds = {
        "f_ster_div_single": (0.05, 0.3),
        "f_a_fw_outboard_hcd": (0.05, 0.3),
        "a_fw_inboard_full_coverage": (50.0, 2000.0),
        "a_fw_outboard_full_coverage": (50.0, 2000.0),
    }


_reference_first_wall_coverage_factors_double_null = functools.partial(
    FirstWall.apply_first_wall_coverage_factors,
    n_divertors=2,
)


class TestApplyFirstWallCoverageFactorsDoubleNull(Tier1Contract):
    """`apply_first_wall_coverage_factors_double_null` -> `FirstWall.
    apply_first_wall_coverage_factors` at `n_divertors == 2`.

    `f_ster_div_single` is capped tighter than the single-null contract's box:
    `1 - 2 * f_ster_div_single - f_a_fw_outboard_hcd` has to stay positive, or PROCESS
    raises its `ProcessValueError` (which the port does not reproduce -- `fw.md`
    § deviations) and the two would disagree for a reason that is not a porting
    question.
    """

    audit_record = "models/fw.md"
    reference = _reference_first_wall_coverage_factors_double_null
    ported = apply_first_wall_coverage_factors_double_null

    fuzz_bounds = {
        "f_ster_div_single": (0.05, 0.2),
        "f_a_fw_outboard_hcd": (0.05, 0.3),
        "a_fw_inboard_full_coverage": (50.0, 2000.0),
        "a_fw_outboard_full_coverage": (50.0, 2000.0),
    }


def _fw():
    """A `FirstWall` instance with a fresh `DataStructure` attached."""
    model = FirstWall()
    model.data = DataStructure()
    return model


def _fw_single_null():
    """`FirstWall`, at the one switch combination the port bakes in (`itart=0`,
    `i_fw_blkt_vv_shape=2` -- both already PROCESS defaults; `n_divertors=1`,
    `i_pflux_fw_neutron=1` -- the latter also already the default).

    `.physics.a_plasma_surface` is a plain nonzero value: `run()` unconditionally
    computes `pflux_fw_rad_mw` (out of this unit's scope, `fw.md` § scope discipline)
    immediately after the lines this test cares about, dividing by it, so it must not
    be left at its zero default.
    """
    model = _fw()
    model.data.divertor.n_divertors = 1
    model.data.physics.itart = 0
    model.data.fwbs.i_fw_blkt_vv_shape = 2
    model.data.physics.i_pflux_fw_neutron = 1
    model.data.physics.a_plasma_surface = 1000.0
    return model


_reference_first_wall_outputs = process_reference(
    _fw_single_null,
    "run",
    (
        "first_wall.a_fw_inboard",
        "first_wall.a_fw_outboard",
        "first_wall.a_fw_total",
        "physics.p_fw_alpha_mw",
        "physics.pflux_fw_neutron_mw",
    ),
)


class TestCalculateFirstWallOutputs(Tier1Contract):
    """`calculate_first_wall_outputs` -> real `FirstWall.run()`, the whole live
    pipeline (own contract for `p_fw_alpha_mw`/`pflux_fw_neutron_mw` too -- neither
    has an isolated PROCESS function, see module docstring).
    """

    audit_record = "models/fw.md"
    reference = _reference_first_wall_outputs
    ported = calculate_first_wall_outputs

    samples = FROM_FILE

    # Narrower than the shared DOMAIN: widening gives
    # reference ProcessValueError: fhole+f_ster_div_single+f_a_fw_outboard_hcd is too high for
    fuzz_bounds = {
        "z_plasma_xpoint_lower": (-10.0, -1.0),
        "dz_xpoint_divertor": (0.1, 2.0),
        "dz_divertor": (0.2, 2.0),
        "dz_blkt_upper": (0.1, 1.5),
        "z_plasma_xpoint_upper": (1.0, 10.0),
        "dz_fw_plasma_gap": (0.05, 1.0),
        "dr_fw_inboard": (0.01, 0.1),
        "dr_fw_outboard": (0.01, 0.1),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "triang": (0.0, 0.8),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "f_ster_div_single": (0.05, 0.3),
        "f_a_fw_outboard_hcd": (0.05, 0.3),
        "p_alpha_total_mw": (10.0, 1000.0),
        "f_p_alpha_plasma_deposited": (0.7, 1.0),
        "ffwal": (0.8, 1.2),
        "pflux_plasma_surface_neutron_avg_mw": (0.1, 5.0),
    }


def _fw_double_null():
    """`FirstWall` at `n_divertors = 2`, otherwise the same configuration as
    `_fw_single_null`.

    `.build.z_plasma_xpoint_upper` and `.build.dz_fw_plasma_gap` are seeded with `nan`,
    not left at their defaults: `process/models/fw.py` reads them at `:51-52` only, as
    arguments of the half-height call, so on this arm nothing may touch them and a `nan`
    proves it.
    """
    model = _fw()
    model.data.build.z_plasma_xpoint_upper = np.nan
    model.data.build.dz_fw_plasma_gap = np.nan
    model.data.divertor.n_divertors = 2
    model.data.physics.itart = 0
    model.data.fwbs.i_fw_blkt_vv_shape = 2
    model.data.physics.i_pflux_fw_neutron = 1
    # Same out-of-scope `pflux_fw_rad_mw` division as the single-null factory.
    model.data.physics.a_plasma_surface = 1000.0
    return model


_reference_first_wall_outputs_double_null = process_reference(
    _fw_double_null,
    "run",
    (
        "first_wall.a_fw_inboard",
        "first_wall.a_fw_outboard",
        "first_wall.a_fw_total",
        "physics.p_fw_alpha_mw",
        "physics.pflux_fw_neutron_mw",
    ),
)


class TestCalculateFirstWallOutputsDoubleNull(Tier1Contract):
    """`calculate_first_wall_outputs_double_null` -> real `FirstWall.run()` at
    `n_divertors == 2`.

    `f_ster_div_single` is pulled down from the single-null box's `0.3` ceiling because
    this arm subtracts `2 * f_ster_div_single`, and `z_plasma_xpoint_lower` takes its
    real (positive) sign for the reason
    `TestCalculateFirstWallHalfHeightDoubleNull` records: on this arm the half-height is
    `z_bottom` itself, and a negative one drives `eshellarea` to negative areas and
    PROCESS's `ProcessValueError` -- a domain question, not a porting one. The rest of
    the point is the single-null contract's, so the two composites meet at one geometry.
    """

    audit_record = "models/fw.md"
    reference = _reference_first_wall_outputs_double_null
    ported = calculate_first_wall_outputs_double_null

    samples = FROM_FILE

    # Narrower than the shared DOMAIN: widening gives
    # reference ProcessValueError: fhole+f_ster_div_single+f_a_fw_outboard_hcd is too high for
    fuzz_bounds = {
        "z_plasma_xpoint_lower": (3.0, 7.0),
        "dz_xpoint_divertor": (0.1, 2.0),
        "dz_divertor": (0.2, 2.0),
        "dz_blkt_upper": (0.1, 1.5),
        "dr_fw_inboard": (0.01, 0.1),
        "dr_fw_outboard": (0.01, 0.1),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "triang": (0.0, 0.8),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "f_ster_div_single": (0.05, 0.2),
        "f_a_fw_outboard_hcd": (0.05, 0.3),
        "p_alpha_total_mw": (10.0, 1000.0),
        "f_p_alpha_plasma_deposited": (0.7, 1.0),
        "ffwal": (0.8, 1.2),
        "pflux_plasma_surface_neutron_avg_mw": (0.1, 5.0),
    }


_reference_set_fw_geometry = process_reference(
    _fw, "set_fw_geometry", ("build.dr_fw_inboard", "build.dr_fw_outboard")
)


class TestSetFwGeometry(Tier1Contract):
    """`set_fw_geometry` -> `FirstWall.set_fw_geometry` (`fw.py:347-352`)."""

    audit_record = "models/fw.md"
    reference = _reference_set_fw_geometry
    ported = set_fw_geometry

    samples = FROM_FILE

    fuzz = True


class TestCalculateDshapedFirstWallAreas(Tier1Contract):
    """`calculate_dshaped_first_wall_areas` -> `FirstWall.
    calculate_dshaped_first_wall_areas`, unchanged signature.

    No `nan` poisoning and none possible: PROCESS's own D-shaped staticmethod simply has
    no `triang` parameter, so the "this arm does not read triangularity" claim is carried
    by the signature rather than by a poisoned value. That is the stronger form.
    """

    audit_record = "models/fw.md"
    reference = staticmethod(FirstWall.calculate_dshaped_first_wall_areas)
    ported = calculate_dshaped_first_wall_areas

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "dz_fw_half": (1.0, 15.0),
        "dr_fw_plasma_gap_inboard": (0.05, 0.5),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
    }


def _fw_dshaped_double_null():
    """`FirstWall` at `n_divertors = 2` **and** the D-shaped shape arm --
    `spherical_tokamak_eval.IN.DAT`/`st_regression.IN.DAT`'s own configuration.

    **Three fields are poisoned with `nan`**, one more than `_fw_double_null`:
    `.build.z_plasma_xpoint_upper` and `.build.dz_fw_plasma_gap` (read at
    `process/models/fw.py:51-52` only, as arguments of the half-height call, which this
    arm's branch does not use) and now `.physics.triang` as well, which `fw.py` reads at
    `:82` only, as an argument of the *elliptical* area call. On this arm nothing may
    touch any of the three, and a `nan` proves it.

    `itart = 1` **and** `i_fw_blkt_vv_shape = D_SHAPED` are both set, as both ST files
    set both; either alone selects the same arm.
    """
    model = _fw()
    model.data.build.z_plasma_xpoint_upper = np.nan
    model.data.build.dz_fw_plasma_gap = np.nan
    model.data.divertor.n_divertors = 2
    model.data.physics.itart = 1
    model.data.fwbs.i_fw_blkt_vv_shape = 1
    model.data.physics.triang = np.nan
    model.data.physics.i_pflux_fw_neutron = 1
    # Same out-of-scope `pflux_fw_rad_mw` division as the other two factories.
    model.data.physics.a_plasma_surface = 1000.0
    return model


_reference_first_wall_outputs_dshaped_double_null = process_reference(
    _fw_dshaped_double_null,
    "run",
    (
        "first_wall.a_fw_inboard",
        "first_wall.a_fw_outboard",
        "first_wall.a_fw_total",
        "physics.p_fw_alpha_mw",
        "physics.pflux_fw_neutron_mw",
    ),
)


class TestCalculateFirstWallOutputsDshapedDoubleNull(Tier1Contract):
    """`calculate_first_wall_outputs_dshaped_double_null` -> real `FirstWall.run()` at
    the D-shaped double-null cell -- what `FirstWallDShapedDoubleNull` wraps, and the
    configuration both spherical-tokamak input files select.

    Bounds are the spherical-tokamak operating point rather than the large-tokamak one
    the other two composites use: `rmajor` around `1.8 * rminor` (both files set
    `aspect = 1.8`) instead of around `3 * rminor`. The D-shaped inboard radius is
    `rmajor - rminor - dr_fw_plasma_gap_inboard`, which a conventional-aspect box would
    keep far from the ST regime where it goes small.
    """

    audit_record = "models/fw.md"
    reference = _reference_first_wall_outputs_dshaped_double_null
    ported = calculate_first_wall_outputs_dshaped_double_null

    samples = FROM_FILE

    # Narrower than the shared DOMAIN: widening gives
    # reference ProcessValueError: fhole+f_ster_div_single+f_a_fw_outboard_hcd is too high for
    fuzz_bounds = {
        "z_plasma_xpoint_lower": (3.0, 7.0),
        "dz_xpoint_divertor": (0.1, 2.0),
        "dz_divertor": (0.2, 2.0),
        "dz_blkt_upper": (0.1, 1.5),
        "dr_fw_inboard": (0.01, 0.1),
        "dr_fw_outboard": (0.01, 0.1),
        "rmajor": (3.0, 5.0),
        "rminor": (1.5, 2.4),
        "dr_fw_plasma_gap_inboard": (0.05, 0.4),
        "dr_fw_plasma_gap_outboard": (0.05, 0.5),
        "f_ster_div_single": (0.05, 0.2),
        "f_a_fw_outboard_hcd": (0.05, 0.3),
        "p_alpha_total_mw": (10.0, 1000.0),
        "f_p_alpha_plasma_deposited": (0.7, 1.0),
        "ffwal": (0.8, 1.2),
        "pflux_plasma_surface_neutron_avg_mw": (0.1, 5.0),
    }


def _fw_radiated_wall_load():
    """`FirstWall`, fixed at the D-shaped double-null spherical-tokamak point (both
    tracked ST files' arm, and the one whose `.constraints.pflux_fw_rad_max_mw` was the
    frozen boundary path) so `run()` reaches `fw.py:130-144` -- four lines inside
    `run()` with no `calculate_*` staticmethod of their own.

    **Nothing in this contract depends on the geometry being right**, and that is the
    useful property: `run()` computes `.first_wall.a_fw_total` from all of it and the
    `i_pflux_fw_neutron == 1` arm of these four lines then reads *none* of it -- only
    `ffwal`, `p_plasma_rad_mw` and `a_plasma_surface`. The geometry is here only to let
    `run()` reach line 130 at all.
    """
    model = _fw()
    data = model.data
    data.build.z_plasma_xpoint_lower = 4.0
    data.build.dz_xpoint_divertor = 0.5
    data.divertor.dz_divertor = 0.4
    data.build.dz_blkt_upper = 0.5
    data.build.z_plasma_xpoint_upper = 4.0
    data.build.dz_fw_plasma_gap = 0.2
    data.divertor.n_divertors = 2
    data.build.dr_fw_inboard = 0.03
    data.build.dr_fw_outboard = 0.03
    data.physics.itart = 1
    data.fwbs.i_fw_blkt_vv_shape = 1
    data.physics.rmajor = 3.6
    data.physics.rminor = 2.0
    data.physics.triang = 0.5
    data.build.dr_fw_plasma_gap_inboard = 0.1
    data.build.dr_fw_plasma_gap_outboard = 0.2
    data.fwbs.f_ster_div_single = 0.1
    data.fwbs.f_a_fw_outboard_hcd = 0.1
    data.physics.p_alpha_total_mw = 100.0
    data.physics.f_p_alpha_plasma_deposited = 0.95
    data.physics.pflux_plasma_surface_neutron_avg_mw = 1.0
    data.physics.i_pflux_fw_neutron = 1
    return model


_reference_radiated_wall_load_scaled_plasma_surface = process_reference(
    _fw_radiated_wall_load,
    "run",
    ("physics.pflux_fw_rad_mw", "constraints.pflux_fw_rad_max_mw"),
)


class TestRadiatedWallLoadScaledPlasmaSurface(Tier1Contract):
    """`calculate_radiated_wall_load_scaled_plasma_surface` -> real `FirstWall.run()`
    at `i_pflux_fw_neutron == 1` (`fw.py:130-144`).

    `f_fw_rad_max` is `1.0` on both files, so the two outputs coincide there; the fuzz
    box moves it away from 1 so the peaking factor is actually exercised.
    `a_plasma_surface` is bounded away from zero -- PROCESS divides by it with no guard.
    """

    audit_record = "models/fw.md"
    reference = _reference_radiated_wall_load_scaled_plasma_surface
    ported = calculate_radiated_wall_load_scaled_plasma_surface

    samples = FROM_FILE

    fuzz = True
