"""Harness cases for the ported subset of `process/models/fw.py`
(`.tokamak.first_wall`).

Audit record: `functional_process/_audit/units/models/fw.md`. Four units:

- `TestCalculateFirstWallHalfHeight`, `TestCalculateEllipticalFirstWallAreas`,
  `TestApplyFirstWallCoverageFactors` -- each a real PROCESS `@staticmethod`, diffed
  directly (with `n_divertors` fixed by the reference adapter where PROCESS's own
  signature still takes it).
- `TestCalculateFirstWallOutputs` -- the whole `FirstWall.run()` pipeline.
  `p_fw_alpha_mw` and `pflux_fw_neutron_mw` are inline `run()` arithmetic with no
  isolated PROCESS function of their own (same shape as `confinement_time.md`'s
  `plasma_power_loss_mw`), so this composite is their only diff against real PROCESS,
  not a standalone contract.

2026-08-27: three `n_divertors == 2` contracts joined -- the half-height, the coverage
factors and the composite. Both double-null adapters **poison** the two inputs that arm
does not read (`z_plasma_xpoint_upper`, `dz_fw_plasma_gap`) with `nan` rather than
zeroing them, so "PROCESS does not look at these" is executed rather than asserted: were
the branch not taken, the reference would return `nan` and the value comparison would
fail instead of quietly agreeing on a zero. The composite can do it too because
`process/models/fw.py` reads both fields in exactly one place -- `:51-52`, the arguments
of the half-height call -- so a `nan` in `.build` reaches nothing else in `run()`.

2026-08-27 (the D-shaped wave): two more contracts --
`TestCalculateDshapedFirstWallAreas` (a bare PROCESS staticmethod, no adapter) and
`TestCalculateFirstWallOutputsDshapedDoubleNull` (the D-shaped double-null composite,
what `FirstWallDShapedDoubleNull` wraps and what both spherical-tokamak input files
select). The composite's adapter poisons a *third* field, `.physics.triang`, which
`fw.py` reads at `:82` only, inside the elliptical area call.
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract, legacy_sample
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


def _reference_first_wall_half_height(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
    z_plasma_xpoint_upper,
    dz_fw_plasma_gap,
    dr_fw_inboard,
    dr_fw_outboard,
):
    """`FirstWall.calculate_first_wall_half_height` at `n_divertors == 1`."""
    return FirstWall.calculate_first_wall_half_height(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        dz_blkt_upper=dz_blkt_upper,
        z_plasma_xpoint_upper=z_plasma_xpoint_upper,
        dz_fw_plasma_gap=dz_fw_plasma_gap,
        n_divertors=1,
        dr_fw_inboard=dr_fw_inboard,
        dr_fw_outboard=dr_fw_outboard,
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


def _reference_first_wall_half_height_double_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
    dr_fw_inboard,
    dr_fw_outboard,
):
    """`FirstWall.calculate_first_wall_half_height` at `n_divertors == 2`.

    The two parameters this arm does not read go in as `nan` -- see the module
    docstring.
    """
    return FirstWall.calculate_first_wall_half_height(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        dz_blkt_upper=dz_blkt_upper,
        z_plasma_xpoint_upper=np.nan,
        dz_fw_plasma_gap=np.nan,
        n_divertors=2,
        dr_fw_inboard=dr_fw_inboard,
        dr_fw_outboard=dr_fw_outboard,
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


def _reference_first_wall_coverage_factors(
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    a_fw_inboard_full_coverage,
    a_fw_outboard_full_coverage,
):
    """`FirstWall.apply_first_wall_coverage_factors` at `n_divertors == 1`."""
    return FirstWall.apply_first_wall_coverage_factors(
        n_divertors=1,
        f_ster_div_single=f_ster_div_single,
        f_a_fw_outboard_hcd=f_a_fw_outboard_hcd,
        a_fw_inboard_full_coverage=a_fw_inboard_full_coverage,
        a_fw_outboard_full_coverage=a_fw_outboard_full_coverage,
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


def _reference_first_wall_coverage_factors_double_null(
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    a_fw_inboard_full_coverage,
    a_fw_outboard_full_coverage,
):
    """`FirstWall.apply_first_wall_coverage_factors` at `n_divertors == 2`."""
    return FirstWall.apply_first_wall_coverage_factors(
        n_divertors=2,
        f_ster_div_single=f_ster_div_single,
        f_a_fw_outboard_hcd=f_a_fw_outboard_hcd,
        a_fw_inboard_full_coverage=a_fw_inboard_full_coverage,
        a_fw_outboard_full_coverage=a_fw_outboard_full_coverage,
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


def _reference_first_wall_outputs(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
    z_plasma_xpoint_upper,
    dz_fw_plasma_gap,
    dr_fw_inboard,
    dr_fw_outboard,
    rmajor,
    rminor,
    triang,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    p_alpha_total_mw,
    f_p_alpha_plasma_deposited,
    ffwal,
    pflux_plasma_surface_neutron_avg_mw,
):
    """Call PROCESS's real `FirstWall.run()` through the port's signature, at the one
    switch combination the port bakes in (`itart=0`, `i_fw_blkt_vv_shape=2` --
    both already PROCESS defaults; `n_divertors=1`, `i_pflux_fw_neutron=1` -- the
    latter also already the default).
    """
    data = DataStructure()
    data.build.z_plasma_xpoint_lower = z_plasma_xpoint_lower
    data.build.dz_xpoint_divertor = dz_xpoint_divertor
    data.divertor.dz_divertor = dz_divertor
    data.build.dz_blkt_upper = dz_blkt_upper
    data.build.z_plasma_xpoint_upper = z_plasma_xpoint_upper
    data.build.dz_fw_plasma_gap = dz_fw_plasma_gap
    data.divertor.n_divertors = 1
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.physics.itart = 0
    data.fwbs.i_fw_blkt_vv_shape = 2
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.physics.triang = triang
    data.build.dr_fw_plasma_gap_inboard = dr_fw_plasma_gap_inboard
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.fwbs.f_ster_div_single = f_ster_div_single
    data.fwbs.f_a_fw_outboard_hcd = f_a_fw_outboard_hcd
    data.physics.p_alpha_total_mw = p_alpha_total_mw
    data.physics.f_p_alpha_plasma_deposited = f_p_alpha_plasma_deposited
    data.physics.i_pflux_fw_neutron = 1
    data.physics.ffwal = ffwal
    data.physics.pflux_plasma_surface_neutron_avg_mw = (
        pflux_plasma_surface_neutron_avg_mw
    )
    # `run()` unconditionally computes `pflux_fw_rad_mw` (out of this unit's scope,
    # `fw.md` § scope discipline) immediately after the lines this test cares about,
    # dividing by `a_plasma_surface` -- give it a nonzero value so that unrelated line
    # does not crash before the target outputs are read back.
    data.physics.a_plasma_surface = 1000.0

    fw = FirstWall()
    fw.data = data
    fw.run()

    return (
        fw.data.first_wall.a_fw_inboard,
        fw.data.first_wall.a_fw_outboard,
        fw.data.first_wall.a_fw_total,
        fw.data.physics.p_fw_alpha_mw,
        fw.data.physics.pflux_fw_neutron_mw,
    )


class TestCalculateFirstWallOutputs(Tier1Contract):
    """`calculate_first_wall_outputs` -> real `FirstWall.run()`, the whole live
    pipeline (own contract for `p_fw_alpha_mw`/`pflux_fw_neutron_mw` too -- neither
    has an isolated PROCESS function, see module docstring).
    """

    audit_record = "models/fw.md"
    reference = _reference_first_wall_outputs
    ported = calculate_first_wall_outputs

    samples = [
        legacy_sample(
            "large-tokamak-plausible",
            z_plasma_xpoint_lower=-5.5,
            dz_xpoint_divertor=0.5,
            dz_divertor=0.65,
            dz_blkt_upper=0.6,
            z_plasma_xpoint_upper=5.5,
            dz_fw_plasma_gap=0.25,
            dr_fw_inboard=0.03,
            dr_fw_outboard=0.03,
            rmajor=8.8901,
            rminor=2.8677741935483869,
            triang=0.5,
            dr_fw_plasma_gap_inboard=0.25,
            dr_fw_plasma_gap_outboard=0.25,
            f_ster_div_single=0.1,
            f_a_fw_outboard_hcd=0.1,
            p_alpha_total_mw=400.0,
            f_p_alpha_plasma_deposited=0.95,
            ffwal=1.0,
            pflux_plasma_surface_neutron_avg_mw=1.0,
        ),
    ]

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


def _reference_first_wall_outputs_double_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
    dr_fw_inboard,
    dr_fw_outboard,
    rmajor,
    rminor,
    triang,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    p_alpha_total_mw,
    f_p_alpha_plasma_deposited,
    ffwal,
    pflux_plasma_surface_neutron_avg_mw,
):
    """Real `FirstWall.run()` at `n_divertors = 2`, otherwise the same configuration as
    `_reference_first_wall_outputs`.

    `.build.z_plasma_xpoint_upper` and `.build.dz_fw_plasma_gap` are seeded with `nan`,
    not left at their defaults: `process/models/fw.py` reads them at `:51-52` only, as
    arguments of the half-height call, so on this arm nothing may touch them and a `nan`
    proves it.
    """
    data = DataStructure()
    data.build.z_plasma_xpoint_lower = z_plasma_xpoint_lower
    data.build.dz_xpoint_divertor = dz_xpoint_divertor
    data.divertor.dz_divertor = dz_divertor
    data.build.dz_blkt_upper = dz_blkt_upper
    data.build.z_plasma_xpoint_upper = np.nan
    data.build.dz_fw_plasma_gap = np.nan
    data.divertor.n_divertors = 2
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.physics.itart = 0
    data.fwbs.i_fw_blkt_vv_shape = 2
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.physics.triang = triang
    data.build.dr_fw_plasma_gap_inboard = dr_fw_plasma_gap_inboard
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.fwbs.f_ster_div_single = f_ster_div_single
    data.fwbs.f_a_fw_outboard_hcd = f_a_fw_outboard_hcd
    data.physics.p_alpha_total_mw = p_alpha_total_mw
    data.physics.f_p_alpha_plasma_deposited = f_p_alpha_plasma_deposited
    data.physics.i_pflux_fw_neutron = 1
    data.physics.ffwal = ffwal
    data.physics.pflux_plasma_surface_neutron_avg_mw = (
        pflux_plasma_surface_neutron_avg_mw
    )
    # Same out-of-scope `pflux_fw_rad_mw` division as the single-null adapter.
    data.physics.a_plasma_surface = 1000.0

    fw = FirstWall()
    fw.data = data
    fw.run()

    return (
        fw.data.first_wall.a_fw_inboard,
        fw.data.first_wall.a_fw_outboard,
        fw.data.first_wall.a_fw_total,
        fw.data.physics.p_fw_alpha_mw,
        fw.data.physics.pflux_fw_neutron_mw,
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

    samples = [
        legacy_sample(
            "large-tokamak-plausible-double-null",
            z_plasma_xpoint_lower=5.5,
            dz_xpoint_divertor=0.5,
            dz_divertor=0.65,
            dz_blkt_upper=0.6,
            dr_fw_inboard=0.03,
            dr_fw_outboard=0.03,
            rmajor=8.8901,
            rminor=2.8677741935483869,
            triang=0.5,
            dr_fw_plasma_gap_inboard=0.25,
            dr_fw_plasma_gap_outboard=0.25,
            f_ster_div_single=0.1,
            f_a_fw_outboard_hcd=0.1,
            p_alpha_total_mw=400.0,
            f_p_alpha_plasma_deposited=0.95,
            ffwal=1.0,
            pflux_plasma_surface_neutron_avg_mw=1.0,
        ),
    ]

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


def _reference_set_fw_geometry(radius_fw_channel, dr_fw_wall):
    """PROCESS's real `FirstWall.set_fw_geometry`, through the `data` back-door -- it
    is an instance method with no arguments, so the inputs go in as fields and the
    answers come back off `.build`.
    """
    data = DataStructure()
    data.fwbs.radius_fw_channel = radius_fw_channel
    data.fwbs.dr_fw_wall = dr_fw_wall
    fw = FirstWall()
    fw.data = data
    fw.set_fw_geometry()
    return fw.data.build.dr_fw_inboard, fw.data.build.dr_fw_outboard


class TestSetFwGeometry(Tier1Contract):
    """`set_fw_geometry` -> `FirstWall.set_fw_geometry` (`fw.py:347-352`), added
    2026-08-27 (`cold_boundary.md` producer 1). The legacy point is the reference
    run's two dataclass defaults, whose sum `0.018` is the value `sr.run()`
    reproduces exactly (`cold_boundary.md` Task A).
    """

    audit_record = "models/fw.md"
    reference = _reference_set_fw_geometry
    ported = set_fw_geometry

    samples = [
        legacy_sample(
            "fwbs-defaults",
            radius_fw_channel=0.006,
            dr_fw_wall=0.003,
        ),
    ]

    fuzz_bounds = {
        "radius_fw_channel": (0.002, 0.02),
        "dr_fw_wall": (0.001, 0.01),
    }


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


def _reference_first_wall_outputs_dshaped_double_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
    dr_fw_inboard,
    dr_fw_outboard,
    rmajor,
    rminor,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    p_alpha_total_mw,
    f_p_alpha_plasma_deposited,
    ffwal,
    pflux_plasma_surface_neutron_avg_mw,
):
    """Real `FirstWall.run()` at `n_divertors = 2` **and** the D-shaped shape arm --
    `spherical_tokamak_eval.IN.DAT`/`st_regression.IN.DAT`'s own configuration.

    **Three fields are poisoned with `nan`**, one more than the elliptical double-null
    adapter: `.build.z_plasma_xpoint_upper` and `.build.dz_fw_plasma_gap` (read at
    `process/models/fw.py:51-52` only, as arguments of the half-height call, which this
    arm's branch does not use) and now `.physics.triang` as well, which `fw.py` reads at
    `:82` only, as an argument of the *elliptical* area call. On this arm nothing may
    touch any of the three, and a `nan` proves it.

    `itart = 1` **and** `i_fw_blkt_vv_shape = D_SHAPED` are both set, as both ST files
    set both; either alone selects the same arm.
    """
    data = DataStructure()
    data.build.z_plasma_xpoint_lower = z_plasma_xpoint_lower
    data.build.dz_xpoint_divertor = dz_xpoint_divertor
    data.divertor.dz_divertor = dz_divertor
    data.build.dz_blkt_upper = dz_blkt_upper
    data.build.z_plasma_xpoint_upper = np.nan
    data.build.dz_fw_plasma_gap = np.nan
    data.divertor.n_divertors = 2
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.physics.itart = 1
    data.fwbs.i_fw_blkt_vv_shape = 1
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.physics.triang = np.nan
    data.build.dr_fw_plasma_gap_inboard = dr_fw_plasma_gap_inboard
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.fwbs.f_ster_div_single = f_ster_div_single
    data.fwbs.f_a_fw_outboard_hcd = f_a_fw_outboard_hcd
    data.physics.p_alpha_total_mw = p_alpha_total_mw
    data.physics.f_p_alpha_plasma_deposited = f_p_alpha_plasma_deposited
    data.physics.i_pflux_fw_neutron = 1
    data.physics.ffwal = ffwal
    data.physics.pflux_plasma_surface_neutron_avg_mw = (
        pflux_plasma_surface_neutron_avg_mw
    )
    # Same out-of-scope `pflux_fw_rad_mw` division as the other two adapters.
    data.physics.a_plasma_surface = 1000.0

    fw = FirstWall()
    fw.data = data
    fw.run()

    return (
        fw.data.first_wall.a_fw_inboard,
        fw.data.first_wall.a_fw_outboard,
        fw.data.first_wall.a_fw_total,
        fw.data.physics.p_fw_alpha_mw,
        fw.data.physics.pflux_fw_neutron_mw,
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

    samples = [
        legacy_sample(
            "spherical-tokamak-plausible",
            z_plasma_xpoint_lower=4.0,
            dz_xpoint_divertor=0.5,
            dz_divertor=0.4,
            dz_blkt_upper=0.5,
            dr_fw_inboard=0.03,
            dr_fw_outboard=0.03,
            rmajor=3.6,
            rminor=2.0,
            dr_fw_plasma_gap_inboard=0.1,
            dr_fw_plasma_gap_outboard=0.2,
            f_ster_div_single=0.1,
            f_a_fw_outboard_hcd=0.1,
            p_alpha_total_mw=100.0,
            f_p_alpha_plasma_deposited=0.95,
            ffwal=1.0,
            pflux_plasma_surface_neutron_avg_mw=1.0,
        ),
    ]

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


def _reference_radiated_wall_load_scaled_plasma_surface(
    ffwal, p_plasma_rad_mw, a_plasma_surface, f_fw_rad_max
):
    """Real `FirstWall.run()` at `i_pflux_fw_neutron == 1`, read back off `data`.

    `fw.py:130-144` has no `calculate_*` staticmethod of its own -- it is four lines
    inside `run()` -- so this is the same "close the `data` back-door" adapter the three
    composites above use, and for the same reason. The geometry is fixed at the
    D-shaped double-null spherical-tokamak point (both tracked ST files' arm, and the
    one whose `.constraints.pflux_fw_rad_max_mw` was the frozen boundary path); the four
    fields the ported function actually reads are the only ones that vary.

    **Nothing in this contract depends on the geometry being right**, and that is the
    useful property: `run()` computes `.first_wall.a_fw_total` from all of it and the
    `i_pflux_fw_neutron == 1` arm of these four lines then reads *none* of it -- only
    `ffwal`, `p_plasma_rad_mw` and `a_plasma_surface`. The geometry is here to let
    `run()` reach line 130 at all.
    """
    data = DataStructure()
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

    data.physics.ffwal = ffwal
    data.physics.p_plasma_rad_mw = p_plasma_rad_mw
    data.physics.a_plasma_surface = a_plasma_surface
    data.constraints.f_fw_rad_max = f_fw_rad_max

    fw = FirstWall()
    fw.data = data
    fw.run()

    return fw.data.physics.pflux_fw_rad_mw, fw.data.constraints.pflux_fw_rad_max_mw


class TestRadiatedWallLoadScaledPlasmaSurface(Tier1Contract):
    """`calculate_radiated_wall_load_scaled_plasma_surface` -> real `FirstWall.run()`
    at `i_pflux_fw_neutron == 1` (`fw.py:130-144`).

    The legacy points are PROCESS's own converged answers on the two files where
    constraint 67 is active and this port had no producer:
    `.constraints.pflux_fw_rad_max_mw` was frozen at `0.0` against a bound of `1.2`
    where PROCESS reads `0.36324` (`st_regression`) and `0.49896`
    (`spherical_tokamak_eval`) -- `optimise_design.md` §26.2, rank 5. Neither is
    binding at PROCESS's answer, so what the freeze cost was a zero Jacobian row and a
    report that said `0.0`.

    `f_fw_rad_max` is `1.0` on both files, so the two outputs coincide there; the fuzz
    box moves it away from 1 so the peaking factor is actually exercised.
    `a_plasma_surface` is bounded away from zero -- PROCESS divides by it with no guard.
    """

    audit_record = "models/fw.md"
    reference = _reference_radiated_wall_load_scaled_plasma_surface
    ported = calculate_radiated_wall_load_scaled_plasma_surface

    samples = [
        legacy_sample(
            "st_regression-converged",
            ffwal=0.92,
            p_plasma_rad_mw=319.9999999940988,
            a_plasma_surface=810.4940458049573,
            f_fw_rad_max=1.0,
        ),
        legacy_sample(
            "spherical_tokamak_eval-converged",
            ffwal=0.92,
            p_plasma_rad_mw=439.57059868288485,
            a_plasma_surface=810.4940458049573,
            f_fw_rad_max=1.0,
        ),
    ]

    fuzz_bounds = {
        "ffwal": (0.8, 1.2),
        "p_plasma_rad_mw": (10.0, 1000.0),
        "a_plasma_surface": (100.0, 2000.0),
        "f_fw_rad_max": (0.5, 3.0),
    }
