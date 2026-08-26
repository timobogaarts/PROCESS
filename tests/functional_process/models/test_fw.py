"""Harness cases for the ported subset of `process/models/fw.py`
(`.tokamak.first_wall`).

Audit record: `functional_process/_audit/units/models/fw.md`. Four units:

- `TestCalculateFirstWallHalfHeight`, `TestCalculateEllipticalFirstWallAreas`,
  `TestApplyFirstWallCoverageFactors` -- each a real PROCESS `@staticmethod`, diffed
  directly (with `n_divertors=1` fixed by the reference adapter where PROCESS's own
  signature still takes it).
- `TestCalculateFirstWallOutputs` -- the whole `FirstWall.run()` pipeline.
  `p_fw_alpha_mw` and `pflux_fw_neutron_mw` are inline `run()` arithmetic with no
  isolated PROCESS function of their own (same shape as `confinement_time.md`'s
  `plasma_power_loss_mw`), so this composite is their only diff against real PROCESS,
  not a standalone contract.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.fw import (
    apply_first_wall_coverage_factors,
    calculate_elliptical_first_wall_areas,
    calculate_first_wall_half_height,
    calculate_first_wall_outputs,
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
