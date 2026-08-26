"""Pure-functional port of `process/models/fw.py` (`FirstWall`, `.tokamak.first_wall`)
-- the minimal closure for `.first_wall.a_fw_total`, `.physics.p_fw_alpha_mw` and
`.physics.pflux_fw_neutron_mw`, `tokamak_boundary.md`'s three reads of this slot.

Audit record: `functional_process/_audit/units/models/fw.md`.

`fw.py` imports `FluidProperties` (CoolProp) but reaches it **zero** times on the live
run: every CoolProp site is inside `FirstWall.fw_temp` (thermal-hydraulics), reached
only via `.fwbs.i_p_coolant_pumping == 2` (`MECHANICAL`), and
`tests/regression/input_files/large_tokamak_eval.IN.DAT:172` sets `i_p_coolant_pumping =
3`. `fw_temp` also produces nothing on this slot's boundary (`n_fw_*_channels`,
`radius_fw_channel_*_bend`, thermal quantities -- none of them read by anything in
`tokamak_boundary.md`), so it is UNPORTED, dormant rather than absent -- a second
tokamak input file with mechanical pumping would wake it, and a CoolProp wrapping
policy would be needed first (`_audit/next_steps.md` §5).

Three switches are baked into the one occupant below, each at the value live on the
reference input (none of the three appears in `large_tokamak_eval.IN.DAT`, so all three
take their PROCESS default):

- `.physics.itart` (default `0`) and `.fwbs.i_fw_blkt_vv_shape` (default `2`,
  `ELLIPTICAL_SHAPED` -- `process/models/build.py:26-30`) jointly select the elliptical
  area formula over the D-shaped one (`fw.py:58-86`).
- `.divertor.n_divertors` -- **not** the `DataStructure` field's own default of `2`
  (`divertor_variables.py:94`), but `1`, derived by `process/core/init.py:606-616` from
  `.physics.i_single_null = 1` (`large_tokamak_eval.IN.DAT:307`); see
  `divertor.md`/`structure.md` for the same correction. Selects the single-null branch
  of both `calculate_first_wall_half_height` and `apply_first_wall_coverage_factors`.
- `.physics.i_pflux_fw_neutron` (default `1`, `physics_variables.py:1006`) selects the
  `ffwal`-scaled formula for `pflux_fw_neutron_mw` over the `a_fw_total`-normalised one.

All four are read-to-branch switches under the wave-1 policy ("no switch is a static
kwarg"), so none of the port functions below takes them as a parameter at all -- each
function *is* the occupant for its live value, and the alternative arms are UNPORTED
(see `fw.md` § switches touched for the per-switch reads-set evidence).
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.engineering.ivc_functions import eshellarea
from functional_process.paths import build, divertor, first_wall, fwbs, physics


def calculate_first_wall_half_height(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
    z_plasma_xpoint_upper,
    dz_fw_plasma_gap,
    dr_fw_inboard,
    dr_fw_outboard,
):
    """First wall internal half-height (m), `n_divertors == 1` (single null) -- the
    value live on `large_tokamak_eval.IN.DAT` (see module docstring). Ports
    `FirstWall.calculate_first_wall_half_height`,
    `process/models/fw.py:151-199`, `n_divertors == 2` branch dropped (UNPORTED).

    Parameters
    ----------
    z_plasma_xpoint_lower :
        Height of the lower plasma X-point (m). `.build.z_plasma_xpoint_lower`.
    dz_xpoint_divertor :
        Vertical distance from the X-point to the divertor (m).
        `.build.dz_xpoint_divertor`.
    dz_divertor :
        Vertical height of the divertor (m). `.divertor.dz_divertor`.
    dz_blkt_upper :
        Upper blanket vertical thickness (m). `.build.dz_blkt_upper`.
    z_plasma_xpoint_upper :
        Height of the upper plasma X-point (m). `.build.z_plasma_xpoint_upper`.
    dz_fw_plasma_gap :
        Upper first-wall/plasma vertical gap (m). `.build.dz_fw_plasma_gap`.
    dr_fw_inboard :
        Inboard first wall radial thickness (m). `.build.dr_fw_inboard`.
    dr_fw_outboard :
        Outboard first wall radial thickness (m). `.build.dr_fw_outboard`.

    Returns
    -------
    :
        First wall internal half-height (m). `.fwbs.dz_fw_half`.
    """
    z_bottom = (
        z_plasma_xpoint_lower
        + dz_xpoint_divertor
        + dz_divertor
        - dz_blkt_upper
        - 0.5e0 * (dr_fw_inboard + dr_fw_outboard)
    )
    z_top = z_plasma_xpoint_upper + dz_fw_plasma_gap
    return 0.5e0 * (z_top + z_bottom)


def calculate_elliptical_first_wall_areas(
    rmajor,
    rminor,
    triang,
    dz_fw_half,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
):
    """First wall areas at full (100%) coverage, elliptical cross-section --
    `itart == 0` and `.fwbs.i_fw_blkt_vv_shape == ELLIPTICAL_SHAPED`, the combination
    live on `large_tokamak_eval.IN.DAT` (see module docstring). Ports
    `FirstWall.calculate_elliptical_first_wall_areas`, `process/models/fw.py:232-284`,
    unchanged (`eshellarea` -> `functional_process.models.engineering.ivc_functions.
    eshellarea`, the shared elliptical shell-area helper).

    Parameters
    ----------
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    rminor :
        Plasma minor radius (m). `.physics.rminor`.
    triang :
        Plasma triangularity. `.physics.triang`.
    dz_fw_half :
        First wall internal half-height (m). `.fwbs.dz_fw_half`.
    dr_fw_plasma_gap_inboard :
        Inboard scrape-off gap (m). `.build.dr_fw_plasma_gap_inboard`.
    dr_fw_plasma_gap_outboard :
        Outboard scrape-off gap (m). `.build.dr_fw_plasma_gap_outboard`.

    Returns
    -------
    tuple
        `(a_fw_inboard_full_coverage, a_fw_outboard_full_coverage,
        a_fw_total_full_coverage)`, m^2 -- local intermediates, not written to `data`
        (see `fw.md` § scope discipline).
    """
    r1 = rmajor - rminor * triang
    r2 = r1 - (rmajor - rminor - dr_fw_plasma_gap_inboard)
    r3 = (rmajor + rminor + dr_fw_plasma_gap_outboard) - r1

    return eshellarea(rshell=r1, rmini=r2, rmino=r3, zminor=dz_fw_half)


def apply_first_wall_coverage_factors(
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    a_fw_inboard_full_coverage,
    a_fw_outboard_full_coverage,
):
    """First wall areas after divertor/HCD coverage factors, `n_divertors == 1` (single
    null) -- the value live on `large_tokamak_eval.IN.DAT` (see module docstring).
    Ports `FirstWall.apply_first_wall_coverage_factors`,
    `process/models/fw.py:287-345`, `n_divertors == 2` branch dropped (UNPORTED). The
    source's `ProcessValueError` guard against a non-credible outboard area
    (`:337-343`) is also dropped -- a traced function cannot raise on a data-dependent
    condition (`fw.md` § deviations); the port simply returns the (possibly
    non-physical) value.

    Parameters
    ----------
    f_ster_div_single :
        Fractional area of first wall sterically blocked by a single divertor.
        `.fwbs.f_ster_div_single` -- produced by `.tokamak.divertor`'s
        `DivertorHeatFluxSplit` (this pass's other unit).
    f_a_fw_outboard_hcd :
        Fractional area of outboard first wall covered by HCD components.
        `.fwbs.f_a_fw_outboard_hcd`.
    a_fw_inboard_full_coverage :
        Inboard first wall area at full coverage (m^2).
    a_fw_outboard_full_coverage :
        Outboard first wall area at full coverage (m^2).

    Returns
    -------
    tuple
        `(a_fw_inboard, a_fw_outboard, a_fw_total)`, m^2.
    """
    a_fw_outboard = a_fw_outboard_full_coverage * (
        1.0e0 - f_ster_div_single - f_a_fw_outboard_hcd
    )
    a_fw_inboard = a_fw_inboard_full_coverage * (1.0e0 - f_ster_div_single)
    a_fw_total = a_fw_inboard + a_fw_outboard
    return a_fw_inboard, a_fw_outboard, a_fw_total


def calculate_p_fw_alpha_mw(p_alpha_total_mw, f_p_alpha_plasma_deposited):
    """Power transported to the first wall by escaped alpha particles (MW). Ports
    `FirstWall.run`, `process/models/fw.py:146-149`, unchanged.

    Parameters
    ----------
    p_alpha_total_mw :
        Total alpha particle power (MW). `.physics.p_alpha_total_mw`.
    f_p_alpha_plasma_deposited :
        Fraction of alpha power deposited in the plasma.
        `.physics.f_p_alpha_plasma_deposited`.

    Returns
    -------
    :
        `.physics.p_fw_alpha_mw`.
    """
    return p_alpha_total_mw * (1.0e0 - f_p_alpha_plasma_deposited)


def calculate_pflux_fw_neutron_mw_ffwal(ffwal, pflux_plasma_surface_neutron_avg_mw):
    """Nominal mean neutron load on the first wall (MW/m^2), `i_pflux_fw_neutron == 1`
    -- the value live on `large_tokamak_eval.IN.DAT` (default,
    `physics_variables.py:1006`; see module docstring). Ports `FirstWall.run`,
    `process/models/fw.py:121-125`, unchanged. The `i_pflux_fw_neutron == 0` arm
    (`p_neutron_total_mw / a_fw_total`) is UNPORTED.

    Parameters
    ----------
    ffwal :
        First wall load fraction factor. `.physics.ffwal`.
    pflux_plasma_surface_neutron_avg_mw :
        Average neutron flux on the plasma surface (MW/m^2).
        `.physics.pflux_plasma_surface_neutron_avg_mw`.

    Returns
    -------
    :
        `.physics.pflux_fw_neutron_mw`.
    """
    return ffwal * pflux_plasma_surface_neutron_avg_mw


def calculate_first_wall_outputs(
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
    """`.tokamak.first_wall`'s whole live-configuration pipeline: `FirstWall.run()`
    end to end, at the one switch combination live on `large_tokamak_eval.IN.DAT` (see
    module docstring). Composes the five functions above in `run()`'s own order
    (`process/models/fw.py:44-149`); no PROCESS function has this exact shape (`run()`
    itself is the stateful shell this mirrors), so this composite -- not any one of its
    five parts alone -- is what `test_fw.py` diffs against a real
    `FirstWall.run()` call.

    Returns
    -------
    tuple
        `(a_fw_inboard, a_fw_outboard, a_fw_total, p_fw_alpha_mw, pflux_fw_neutron_mw)`.
    """
    dz_fw_half = calculate_first_wall_half_height(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        dz_blkt_upper=dz_blkt_upper,
        z_plasma_xpoint_upper=z_plasma_xpoint_upper,
        dz_fw_plasma_gap=dz_fw_plasma_gap,
        dr_fw_inboard=dr_fw_inboard,
        dr_fw_outboard=dr_fw_outboard,
    )

    (
        a_fw_inboard_full_coverage,
        a_fw_outboard_full_coverage,
        _a_fw_total_full_coverage,
    ) = calculate_elliptical_first_wall_areas(
        rmajor=rmajor,
        rminor=rminor,
        triang=triang,
        dz_fw_half=dz_fw_half,
        dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
        dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
    )

    a_fw_inboard, a_fw_outboard, a_fw_total = apply_first_wall_coverage_factors(
        f_ster_div_single=f_ster_div_single,
        f_a_fw_outboard_hcd=f_a_fw_outboard_hcd,
        a_fw_inboard_full_coverage=a_fw_inboard_full_coverage,
        a_fw_outboard_full_coverage=a_fw_outboard_full_coverage,
    )

    p_fw_alpha_mw = calculate_p_fw_alpha_mw(
        p_alpha_total_mw=p_alpha_total_mw,
        f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
    )

    pflux_fw_neutron_mw = calculate_pflux_fw_neutron_mw_ffwal(
        ffwal=ffwal,
        pflux_plasma_surface_neutron_avg_mw=pflux_plasma_surface_neutron_avg_mw,
    )

    return (
        a_fw_inboard,
        a_fw_outboard,
        a_fw_total,
        p_fw_alpha_mw,
        pflux_fw_neutron_mw,
    )


class FirstWall(ExplicitFunction):
    """cottax node: `.tokamak.first_wall`.

    Bakes in `itart == 0`, `.fwbs.i_fw_blkt_vv_shape == ELLIPTICAL_SHAPED`,
    `.divertor.n_divertors == 1` and `.physics.i_pflux_fw_neutron == 1` -- the
    combination live on `large_tokamak_eval.IN.DAT` (see module docstring). Owns all
    three of `.tokamak.first_wall`'s declared boundary outputs
    (`.first_wall.a_fw_total`, `.physics.p_fw_alpha_mw`, `.physics.pflux_fw_neutron_mw`)
    plus `a_fw_inboard`/`a_fw_outboard`, siblings of `a_fw_total` from the same source
    function and read elsewhere in `process/` (`blankets/dcll.py`, `blankets/hcpb.py`,
    `stellarator.py`). Thin wrap of `calculate_first_wall_outputs` -- no arithmetic of
    its own.
    """

    a_fw_inboard = OutputInto(first_wall)
    a_fw_outboard = OutputInto(first_wall)
    a_fw_total = OutputInto(first_wall)
    p_fw_alpha_mw = OutputInto(physics)
    pflux_fw_neutron_mw = OutputInto(physics)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
        z_plasma_xpoint_upper=From(build),
        dz_fw_plasma_gap=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        p_alpha_total_mw=From(physics),
        f_p_alpha_plasma_deposited=From(physics),
        ffwal=From(physics),
        pflux_plasma_surface_neutron_avg_mw=From(physics),
    ):
        return calculate_first_wall_outputs(
            z_plasma_xpoint_lower=z_plasma_xpoint_lower,
            dz_xpoint_divertor=dz_xpoint_divertor,
            dz_divertor=dz_divertor,
            dz_blkt_upper=dz_blkt_upper,
            z_plasma_xpoint_upper=z_plasma_xpoint_upper,
            dz_fw_plasma_gap=dz_fw_plasma_gap,
            dr_fw_inboard=dr_fw_inboard,
            dr_fw_outboard=dr_fw_outboard,
            rmajor=rmajor,
            rminor=rminor,
            triang=triang,
            dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
            f_ster_div_single=f_ster_div_single,
            f_a_fw_outboard_hcd=f_a_fw_outboard_hcd,
            p_alpha_total_mw=p_alpha_total_mw,
            f_p_alpha_plasma_deposited=f_p_alpha_plasma_deposited,
            ffwal=ffwal,
            pflux_plasma_surface_neutron_avg_mw=pflux_plasma_surface_neutron_avg_mw,
        )
