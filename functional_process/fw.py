"""Pure-functional port of `process/models/fw.py` (`FirstWall`, `.tokamak.first_wall`)
-- the minimal closure for `.first_wall.a_fw_total`, `.physics.p_fw_alpha_mw` and
`.physics.pflux_fw_neutron_mw`, `tokamak_boundary.md`'s three reads of this slot.

Audit record: `functional_process/_audit/units/models/fw.md`.

2026-08-27: `set_fw_geometry` (`fw.py:347-352`) joined the closure as its own node,
`FirstWallGeometry` -- `cold_boundary.md`'s producer 1, the two boundary zeros behind
7 of the cold MDA's 11 non-finite roots. See that class's docstring for why it is a
second node beside `FirstWall` rather than two more outputs of it.

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
function *is* the occupant for its live value (see `fw.md` § switches touched for the
per-switch reads-set evidence).

2026-08-27 (the double-null wave): `n_divertors` is no longer one of the unported arms.
`FirstWall` became a family of two -- `FirstWallSingleNull` (unchanged, still the
`large_tokamak_eval.IN.DAT` occupant) and `FirstWallDoubleNull` -- because the two
`n_divertors` branches this file takes differ in *reads*: a double-null first wall never
looks at `.build.z_plasma_xpoint_upper` or `.build.dz_fw_plasma_gap`, so folding the arms
into one node would declare two edges a double-null machine does not have. The other two
switches are untouched and still refused; a double-null machine that is also spherical or
D-shaped stops at `('fw_blkt_vv_shape_arm', 0)` instead.

2026-08-27 (the D-shaped wave): that last sentence no longer holds for the *double-null*
D-shaped machine. `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` are both
D-shaped (`i_fw_blkt_vv_shape = 1`) **and** spherical (`itart = 1`) **and** double-null
(`i_single_null = 0`), so `FirstWallDShapedDoubleNull` joins the family and the shape
refusal moves off that cell.

**The shape and the divertor count are a genuine product here, and this is one of only
two slots where they are.** `FirstWall.run()` (`process/models/fw.py:44-149`) branches on
`n_divertors` twice -- once for the half-height (`:46-56`) and once for the coverage
factors (`:103-109`) -- with the shape branch (`:58-86`) sitting *between* them. The two
predicates are independent, but because this port keeps `run()` as **one** composite node
rather than three, the occupant grid is 2 (shape) x 2 (divertor count) = four cells, of
which three are written:

| | single null | double null |
|---|---|---|
| **elliptical** | `FirstWallSingleNull` | `FirstWallDoubleNull` |
| **D-shaped** | UNPORTED, `('first_wall_arm', -2)` | `FirstWallDShapedDoubleNull` |

`models/vacuum/vacuum.py` pays the same product for the same reason;
`blankets/blanket_library.py` and `models/shield.py` do **not**, because wave 1 had
already split their `run()`s into one slot per branch, so each of their slots is keyed on
exactly one predicate. The difference is in the decomposition, not in PROCESS.

The unwritten cell is refused rather than written because no input file in this
repository selects it -- the wave's reachability-first discipline. Its every
ingredient exists (`calculate_dshaped_first_wall_areas` below, and the single-null
half-height and coverage
arms), so it is one composite function and one class away, with no new arithmetic;
that is what `('first_wall_arm', -2)` now says.
"""

from functional_process.models.engineering.ivc_functions import dshellarea, eshellarea


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
    `process/models/fw.py:151-199`, `n_divertors == 2` branch split off into
    `calculate_first_wall_half_height_double_null`.

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


def calculate_first_wall_half_height_double_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
    dr_fw_inboard,
    dr_fw_outboard,
):
    """First wall internal half-height (m), `n_divertors == 2` (double null). Ports the
    `if n_divertors == 2` arm of `FirstWall.calculate_first_wall_half_height`,
    `process/models/fw.py:186-197`.

    A double-null machine is vertically symmetric, so PROCESS sets `z_top = z_bottom`
    and `0.5 * (z_top + z_bottom)` collapses to `z_bottom` -- exactly, in floating point
    as well as algebraically. Written reduced, as `shield.py` and
    `blankets/blanket_library.py` do for the same branch.

    **Two fewer reads than the single-null sibling**: `z_plasma_xpoint_upper` and
    `dz_fw_plasma_gap` are the whole of the `else` arm's `z_top`, and a double-null
    machine reads neither.

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
    dr_fw_inboard :
        Inboard first wall radial thickness (m). `.build.dr_fw_inboard`.
    dr_fw_outboard :
        Outboard first wall radial thickness (m). `.build.dr_fw_outboard`.

    Returns
    -------
    :
        First wall internal half-height (m). `.fwbs.dz_fw_half`.
    """
    return (
        z_plasma_xpoint_lower
        + dz_xpoint_divertor
        + dz_divertor
        - dz_blkt_upper
        - 0.5e0 * (dr_fw_inboard + dr_fw_outboard)
    )


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


def calculate_dshaped_first_wall_areas(
    rmajor,
    rminor,
    dz_fw_half,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
):
    """First wall areas at full (100%) coverage, D-shaped cross-section --
    `itart == 1 or .fwbs.i_fw_blkt_vv_shape == D_SHAPED`, the combination live on
    `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` (which satisfy both
    disjuncts). Ports `FirstWall.calculate_dshaped_first_wall_areas`,
    `process/models/fw.py:201-230`, unchanged (`dshellarea` ->
    `functional_process.models.engineering.ivc_functions.dshellarea`, the shared
    D-shaped shell-area helper added in the same wave).

    **One read fewer than the elliptical sibling: no `triang`.** A D-shaped first wall's
    inboard section is a cylinder at `rmajor - rminor - dr_fw_plasma_gap_inboard`, a
    radius the plasma's triangularity does not enter; the elliptical arm instead centres
    both its ellipses at `rmajor - rminor * triang`. Declaring `.physics.triang` on this
    arm would invent an edge, which is the whole reason the two arms are occupants.

    Parameters
    ----------
    rmajor :
        Plasma major radius (m). `.physics.rmajor`.
    rminor :
        Plasma minor radius (m). `.physics.rminor`.
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
        a_fw_total_full_coverage)`, m^2 -- local intermediates, not written to `data`.
    """
    # Major radius to the outer edge of the inboard (cylindrical) section.
    r1 = rmajor - rminor - dr_fw_plasma_gap_inboard

    # Horizontal distance between inside edges.
    r2 = (rmajor + rminor + dr_fw_plasma_gap_outboard) - r1

    return dshellarea(rmajor=r1, rminor=r2, zminor=dz_fw_half)


def apply_first_wall_coverage_factors(
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    a_fw_inboard_full_coverage,
    a_fw_outboard_full_coverage,
):
    """First wall areas after divertor/HCD coverage factors, `n_divertors == 1` (single
    null) -- the value live on `large_tokamak_eval.IN.DAT` (see module docstring).
    Ports `FirstWall.apply_first_wall_coverage_factors`,
    `process/models/fw.py:287-345`, `n_divertors == 2` branch split off into
    `apply_first_wall_coverage_factors_double_null`. The
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


def apply_first_wall_coverage_factors_double_null(
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    a_fw_inboard_full_coverage,
    a_fw_outboard_full_coverage,
):
    """First wall areas after divertor/HCD coverage factors, `n_divertors == 2` (double
    null). Ports the `if n_divertors == 2` arm of
    `FirstWall.apply_first_wall_coverage_factors`, `process/models/fw.py:320-327`.

    Two divertors block twice the solid angle, so both the inboard and the outboard area
    lose `2 * f_ster_div_single` where the single-null arm loses `f_ster_div_single`.
    **Unlike `blanket_library.py`'s coverage factors, this arm is symmetric** -- both
    assignments sit inside the `if`, so there is no doubled-here-not-there defect to
    transcribe. Worth stating, because the two functions look like the same edit and
    only one of them was made completely.

    The source's `ProcessValueError` guard against a non-credible outboard area
    (`fw.py:337-343`) is dropped here for the same reason as in the single-null
    sibling: a traced function cannot raise on a data-dependent condition (`fw.md`
    § deviations).

    Parameters
    ----------
    f_ster_div_single :
        Fractional area of first wall sterically blocked by a **single** divertor.
        `.fwbs.f_ster_div_single` -- produced by `.tokamak.divertor`'s
        `DivertorHeatFluxSplit`.
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
        1.0e0 - 2.0e0 * f_ster_div_single - f_a_fw_outboard_hcd
    )
    a_fw_inboard = a_fw_inboard_full_coverage * (1.0e0 - 2.0e0 * f_ster_div_single)
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
    """`.tokamak.first_wall`'s single-null pipeline: `FirstWall.run()` end to end, at
    the switch combination live on `large_tokamak_eval.IN.DAT` (see module docstring).
    Composes the five functions above in `run()`'s own order
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


def calculate_first_wall_outputs_double_null(
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
    """`.tokamak.first_wall`'s double-null pipeline: `FirstWall.run()` end to end at
    `n_divertors == 2`, otherwise the same configuration as
    `calculate_first_wall_outputs` (elliptical, `i_pflux_fw_neutron == 1`).

    Two parameters fewer than the single-null composite -- `z_plasma_xpoint_upper` and
    `dz_fw_plasma_gap` -- because the half-height arm this one calls does not read them.
    That difference is the whole reason this is a second composite rather than one with
    a traced `n_divertors`.

    Returns
    -------
    tuple
        `(a_fw_inboard, a_fw_outboard, a_fw_total, p_fw_alpha_mw, pflux_fw_neutron_mw)`.
    """
    dz_fw_half = calculate_first_wall_half_height_double_null(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        dz_blkt_upper=dz_blkt_upper,
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

    (
        a_fw_inboard,
        a_fw_outboard,
        a_fw_total,
    ) = apply_first_wall_coverage_factors_double_null(
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


def calculate_first_wall_outputs_dshaped_double_null(
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
    """`.tokamak.first_wall`'s D-shaped double-null pipeline: `FirstWall.run()` end to
    end at `n_divertors == 2` **and** the D-shaped arm of the shape branch, still at
    `i_pflux_fw_neutron == 1`. The configuration live on
    `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`.

    Three parameters fewer than the elliptical double-null composite it otherwise
    mirrors: `triang` (the D-shaped area formula does not read it) is dropped on top of
    the two `z_plasma_xpoint_upper`/`dz_fw_plasma_gap` the double-null half-height
    already dropped. Sixteen reads against the elliptical single-null composite's
    nineteen.

    Returns
    -------
    tuple
        `(a_fw_inboard, a_fw_outboard, a_fw_total, p_fw_alpha_mw, pflux_fw_neutron_mw)`.
    """
    dz_fw_half = calculate_first_wall_half_height_double_null(
        z_plasma_xpoint_lower=z_plasma_xpoint_lower,
        dz_xpoint_divertor=dz_xpoint_divertor,
        dz_divertor=dz_divertor,
        dz_blkt_upper=dz_blkt_upper,
        dr_fw_inboard=dr_fw_inboard,
        dr_fw_outboard=dr_fw_outboard,
    )

    (
        a_fw_inboard_full_coverage,
        a_fw_outboard_full_coverage,
        _a_fw_total_full_coverage,
    ) = calculate_dshaped_first_wall_areas(
        rmajor=rmajor,
        rminor=rminor,
        dz_fw_half=dz_fw_half,
        dr_fw_plasma_gap_inboard=dr_fw_plasma_gap_inboard,
        dr_fw_plasma_gap_outboard=dr_fw_plasma_gap_outboard,
    )

    (
        a_fw_inboard,
        a_fw_outboard,
        a_fw_total,
    ) = apply_first_wall_coverage_factors_double_null(
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


def set_fw_geometry(radius_fw_channel, dr_fw_wall):
    """First wall radial thickness, inboard and outboard (m). Ports
    `FirstWall.set_fw_geometry`, `process/models/fw.py:347-352`, unchanged: both sides
    are `2 * radius_fw_channel + 2 * dr_fw_wall`, and the outboard is assigned the
    inboard's value rather than recomputed, reproduced here by returning the same
    intermediate twice.

    Added 2026-08-27 (`cold_boundary.md` producer 1): `.build.dr_fw_inboard`/
    `.build.dr_fw_outboard` were the boundary zeros behind 7 of the cold tokamak MDA's
    11 non-finite roots (both hcpb coolant void fractions and the five
    `nuclear_heating_magnets` outputs).

    Parameters
    ----------
    radius_fw_channel :
        First wall coolant channel radius (m). `.fwbs.radius_fw_channel`.
    dr_fw_wall :
        Wall thickness of the first wall coolant channels (m). `.fwbs.dr_fw_wall`.

    Returns
    -------
    tuple
        `(dr_fw_inboard, dr_fw_outboard)`, m -- `.build.dr_fw_inboard`,
        `.build.dr_fw_outboard`.
    """
    dr_fw_inboard = 2 * radius_fw_channel + 2 * dr_fw_wall
    return dr_fw_inboard, dr_fw_inboard


def calculate_radiated_wall_load_scaled_plasma_surface(
    ffwal,
    p_plasma_rad_mw,
    a_plasma_surface,
    f_fw_rad_max,
):
    """Photon (radiation) wall load and the peak the constraint bounds, MW/m^2.

    Ports `FirstWall.run`'s `process/models/fw.py:130-144`, the
    `i_pflux_fw_neutron == 1` arm, unchanged. Two statements of PROCESS in one
    function because the second is the first times a peaking factor and PROCESS writes
    them adjacently with nothing between:

    ```
    pflux_fw_rad_mw     = ffwal * p_plasma_rad_mw / a_plasma_surface   # :131-135
    pflux_fw_rad_max_mw = pflux_fw_rad_mw * f_fw_rad_max               # :142-144
    ```

    **The stellarator has its own port of the same two lines**, and the two are
    genuinely different code: `models/stellarator/plasma_physics.py::
    _radiated_wall_load` ports `process/models/stellarator/stellarator.py:2241-2251`,
    which additionally computes `.physics.rad_fraction_total` from the heating powers
    in the same breath. `FirstWall.run` does not -- the tokamak's radiation fraction is
    `exhaust.py::calculate_radiation_fraction`, a different quantity against a
    different denominator, produced by a different slot. Same two `VarPath`s, two
    devices, two nodes, never both in one graph; the correspondence
    `build.py::calculate_dz_blkt_upper` and `ZTfInsideHalf` already record.

    **No `i_pflux_fw_neutron` refusal is minted for this node**, because the one that
    already exists covers it. `indat._first_wall_arm` refuses
    `i_pflux_fw_neutron != 1` at `('first_wall_arm', -3)`, and a graph that has this
    node has `.tokamak.first_wall` too, so no assembled machine can reach the
    `a_fw_total`-normalised arm here without having been refused there first. Writing a
    second, identical refusal would say the two arms can be chosen apart, and they
    cannot.

    Parameters
    ----------
    ffwal :
        Factor to scale plasma surface area to first-wall area. `.physics.ffwal`.
    p_plasma_rad_mw :
        Total radiated power from the plasma (MW). `.physics.p_plasma_rad_mw`.
    a_plasma_surface :
        Plasma surface area (m^2). `.physics.a_plasma_surface`.
    f_fw_rad_max :
        Peaking factor for the radiation wall load. `.constraints.f_fw_rad_max`.

    Returns
    -------
    :
        `(pflux_fw_rad_mw, pflux_fw_rad_max_mw)` (MW/m^2).
    """
    pflux_fw_rad_mw = ffwal * p_plasma_rad_mw / a_plasma_surface
    pflux_fw_rad_max_mw = pflux_fw_rad_mw * f_fw_rad_max
    return pflux_fw_rad_mw, pflux_fw_rad_max_mw
