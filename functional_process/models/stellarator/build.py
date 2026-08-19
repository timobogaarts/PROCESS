"""Pure-functional port of `process/models/stellarator/build.py`'s `st_build` (unit #2).

Audit record: `functional_process/models/stellarator/build.md`. The source is one
straight-line function gated by two switches, `.fwbs.blktmodel` and
`.heat_transport.ipowerflow`. Both are split per `_audit/traceability_policy.md`'s
default and `_audit/naming_convention.md`'s "switches are not ports" -- see the record
for the reasoning. Three tier-1 functions result:

- `calculate_blktmodel_blanket_thickness` -- the `blktmodel > 0` preamble. Only
  instantiated as a node when `blktmodel > 0`; when it isn't, `dr_blkt_inboard`/
  `dr_blkt_outboard` are plain external inputs to `calculate_build` instead (this is
  `conditional-ownership-by-run-config`, the same pattern as `.physics.aspect` in
  `stellarator_C_geometry.md` -- a graph-assembly-time decision, not resolved here).
- `calculate_build` -- everything the source runs unconditionally. Reads
  `dr_blkt_inboard`/`dr_blkt_outboard` as ordinary explicit args regardless of where
  they came from, which is exactly what the source does too (it never branches on
  `blktmodel` again after the preamble). Returns `a_fw_total_unadjusted`, an invented
  intermediate (not a real PROCESS field) rather than the final `.first_wall.a_fw_total`,
  since which of the two functions below owns that name is an `ipowerflow` graph-assembly
  choice, not something `calculate_build` should decide for itself.
- `calculate_a_fw_total_no_powerflow` / `calculate_a_fw_total_with_powerflow` -- the
  `ipowerflow` split. Whichever is wired to `calculate_build`'s output owns
  `.first_wall.a_fw_total`.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FromExactly,
    Output,
)


def calculate_blktmodel_blanket_thickness(
    blbuith,
    blbmith,
    blbpith,
    blbuoth,
    blbmoth,
    blbpoth,
    dr_shld_inboard,
    dr_shld_outboard,
):
    """Blanket thickness and top shield thickness, when `blktmodel > 0`.

    Ports the `if data.fwbs.blktmodel > 0:` block of `st_build`. Not instantiated as a
    node when `blktmodel <= 0` -- see module docstring.

    Parameters
    ----------
    blbuith, blbmith, blbpith :
        Inboard blanket first wall/breeder/back-plate sub-thicknesses (m).
        `.build.blbuith`, `.build.blbmith`, `.build.blbpith`.
    blbuoth, blbmoth, blbpoth :
        Outboard counterparts (m). `.build.blbuoth`, `.build.blbmoth`, `.build.blbpoth`.
    dr_shld_inboard, dr_shld_outboard :
        Inboard/outboard shield thicknesses (m). `.build.dr_shld_inboard`,
        `.build.dr_shld_outboard`.

    Returns
    -------
    :
        `(dr_blkt_inboard, dr_blkt_outboard, dz_shld_upper)` -- inboard/outboard blanket
        thickness (m) and top shield thickness (m).
    """
    dr_blkt_inboard = blbuith + blbmith + blbpith
    dr_blkt_outboard = blbuoth + blbmoth + blbpoth
    dz_shld_upper = 0.5 * (dr_shld_inboard + dr_shld_outboard)
    return dr_blkt_inboard, dr_blkt_outboard, dz_shld_upper


def calculate_build(
    dr_blkt_inboard,
    dr_blkt_outboard,
    radius_fw_channel,
    dr_fw_wall,
    rmajor,
    rminor,
    dr_cs,
    dr_cs_tf_gap,
    dr_tf_inboard,
    dr_shld_vv_gap_inboard,
    dr_vv_inboard,
    dr_shld_inboard,
    dr_fw_plasma_gap_inboard,
    r_coil_minor,
    f_coil_shape,
    stella_config_derivative_min_lcfs_coils_dist,
    f_st_rmajor,
    stella_config_rminor_ref,
    dr_fw_plasma_gap_outboard,
    dr_shld_outboard,
    gapomin,
    dr_vv_outboard,
    a_plasma_surface,
):
    """The stellarator radial build, minus the `blktmodel`/`ipowerflow`-gated pieces.

    Ports the unconditional body of `st_build` -- everything after the `blktmodel`
    preamble and before the `ipowerflow` branch. See module docstring for why those two
    are separate functions.

    Parameters
    ----------
    dr_blkt_inboard, dr_blkt_outboard :
        Inboard/outboard blanket thickness (m). `.build.dr_blkt_inboard`,
        `.build.dr_blkt_outboard` -- from `calculate_blktmodel_blanket_thickness` or an
        external input, depending on `blktmodel` (see module docstring).
    radius_fw_channel, dr_fw_wall :
        First wall coolant channel radius / wall thickness (m). `.fwbs.radius_fw_channel`,
        `.fwbs.dr_fw_wall`.
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    dr_cs, dr_cs_tf_gap, dr_tf_inboard :
        Central solenoid thickness, CS-TF gap, TF coil inboard leg thickness (m).
        `.build.dr_cs`, `.build.dr_cs_tf_gap`, `.build.dr_tf_inboard`.
    dr_shld_vv_gap_inboard, dr_vv_inboard, dr_shld_inboard :
        Inboard shield-VV gap, VV thickness, shield thickness (m). `.build.*`.
    dr_fw_plasma_gap_inboard :
        Inboard scrape-off thickness (m). `.build.dr_fw_plasma_gap_inboard`.
    r_coil_minor, f_coil_shape :
        Coil minor radius (m), coil shape factor. `.stellarator.r_coil_minor`,
        `.stellarator.f_coil_shape`.
    stella_config_derivative_min_lcfs_coils_dist, stella_config_rminor_ref :
        Reference-configuration scaling terms. `.stellarator_config.*`.
    f_st_rmajor :
        Major-radius scaling factor. `.stellarator.f_st_rmajor`.
    dr_fw_plasma_gap_outboard, dr_shld_outboard :
        Outboard scrape-off / shield thickness (m). `.build.*`.
    gapomin :
        Minimum outboard shield-VV gap (m). `.build.gapomin`.
    dr_vv_outboard :
        Outboard VV thickness (m). `.build.dr_vv_outboard`.
    a_plasma_surface :
        Plasma surface area (m2). `.physics.a_plasma_surface`.

    Returns
    -------
    :
        `(dz_blkt_upper, dr_fw_inboard, dr_fw_outboard, dr_bore, rbld,
        required_radial_space, available_radial_space, r_shld_inboard_inner,
        r_shld_outboard_outer, dr_tf_outboard, dr_shld_vv_gap_outboard,
        r_tf_outboard_mid, rspo, a_fw_total_unadjusted)`.

        **`.build.z_tf_inside_half` is deliberately not returned here, even though
        `st_build` (this function's source) computes it.** Real PROCESS has two
        independent, differently-formulated writers of this one field --
        `st_build`'s (this one) and `st_coil`'s (`coils/calculate.py`,
        `calculate_z_tf_inside_half`) -- and `stellarator.py`'s `run()` calls them in
        *opposite* order depending on the `output` flag: mid-solve (`output=False`)
        `st_coil` runs first so `st_build` wins transiently, but the final report pass
        every real run ends with (`output=True`, needed to write `OUT.DAT`/
        `MFILE.DAT`) runs `st_build` first so `st_coil` wins for good -- confirmed
        directly against a converged run via the block-by-block MDA-vs-PROCESS
        comparison harness (`functional_process/mda_harness.py`), which caught this
        port's `Build` node claiming ownership under the wrong (`st_build`'s) formula.
        `st_coil`'s formula is the one PROCESS's real answer keeps, so
        `coils/calculate.py`'s new `ZTfInsideHalf` node owns `.build.z_tf_inside_half`
        instead -- this is an "ordering artifact" in the same family
        `_audit/next_steps.md` §5 already tracks several instances of (two producers,
        one wins by call order, not represented structurally), not a bug in either
        formula.
    """
    dz_blkt_upper = 0.5 * (dr_blkt_inboard + dr_blkt_outboard)

    dr_fw_inboard = 2.0 * radius_fw_channel + 2.0 * dr_fw_wall
    dr_fw_outboard = dr_fw_inboard

    dr_bore = rmajor - (
        dr_cs
        + dr_cs_tf_gap
        + dr_tf_inboard
        + dr_shld_vv_gap_inboard
        + dr_vv_inboard
        + dr_shld_inboard
        + dr_blkt_inboard
        + dr_fw_inboard
        + dr_fw_plasma_gap_inboard
        + rminor
    )

    rbld = (
        dr_bore
        + dr_cs
        + dr_cs_tf_gap
        + dr_tf_inboard
        + dr_shld_vv_gap_inboard
        + dr_vv_inboard
        + dr_shld_inboard
        + dr_blkt_inboard
        + dr_fw_inboard
        + dr_fw_plasma_gap_inboard
        + rminor
    )

    required_radial_space = (
        dr_tf_inboard / 2.0
        + dr_shld_vv_gap_inboard
        + dr_vv_inboard
        + dr_shld_inboard
        + dr_blkt_inboard
        + dr_fw_inboard
        + dr_fw_plasma_gap_inboard
    )

    available_radial_space = (
        r_coil_minor * f_coil_shape - rminor
    ) + stella_config_derivative_min_lcfs_coils_dist * (
        rminor - f_st_rmajor * stella_config_rminor_ref
    )

    r_shld_inboard_inner = (
        rmajor
        - rminor
        - dr_fw_plasma_gap_inboard
        - dr_fw_inboard
        - dr_blkt_inboard
        - dr_shld_inboard
    )

    r_shld_outboard_outer = (
        rmajor
        + rminor
        + dr_fw_plasma_gap_outboard
        + dr_fw_outboard
        + dr_blkt_outboard
        + dr_shld_outboard
    )

    dr_tf_outboard = dr_tf_inboard

    dr_shld_vv_gap_outboard = gapomin
    r_tf_outboard_mid = (
        r_shld_outboard_outer
        + dr_vv_outboard
        + dr_shld_vv_gap_outboard
        + 0.5 * dr_tf_outboard
    )

    # `st_build`'s own `z_tf_inside_half` -- computed (matching the source line for
    # line) but not returned: PROCESS's real, final answer for this field is
    # `st_coil`'s formula instead. See this function's own Returns docstring for why.
    _z_tf_inside_half_st_build = 0.5 * (
        (
            dr_shld_vv_gap_inboard
            + dr_vv_inboard
            + dr_shld_inboard
            + dr_blkt_inboard
            + dr_fw_inboard
            + dr_fw_plasma_gap_inboard
            + rminor
        )
        + (
            rminor
            + dr_fw_plasma_gap_outboard
            + dr_fw_outboard
            + dr_blkt_outboard
            + dr_shld_outboard
            + dr_vv_outboard
            + dr_shld_vv_gap_outboard
        )
    )

    rspo = rmajor

    awall = rminor + 0.5 * (dr_fw_plasma_gap_inboard + dr_fw_plasma_gap_outboard)
    a_fw_total_unadjusted = a_plasma_surface * awall / rminor

    return (
        dz_blkt_upper,
        dr_fw_inboard,
        dr_fw_outboard,
        dr_bore,
        rbld,
        required_radial_space,
        available_radial_space,
        r_shld_inboard_inner,
        r_shld_outboard_outer,
        dr_tf_outboard,
        dr_shld_vv_gap_outboard,
        r_tf_outboard_mid,
        rspo,
        a_fw_total_unadjusted,
    )


def calculate_a_fw_total_no_powerflow(a_fw_total_unadjusted, fhole):
    """First wall area, `ipowerflow == 0` branch.

    Parameters
    ----------
    a_fw_total_unadjusted :
        `calculate_build`'s raw first wall area (m2), before this adjustment.
    fhole :
        Hole fraction. `.fwbs.fhole`.

    Returns
    -------
    :
        `.first_wall.a_fw_total` (m2).
    """
    return (1.0 - fhole) * a_fw_total_unadjusted


def calculate_a_fw_total_with_powerflow(
    a_fw_total_unadjusted, fhole, f_ster_div_single, f_a_fw_outboard_hcd
):
    """First wall area, `ipowerflow != 0` branch.

    Parameters
    ----------
    a_fw_total_unadjusted :
        `calculate_build`'s raw first wall area (m2), before this adjustment.
    fhole :
        Hole fraction. `.fwbs.fhole`.
    f_ster_div_single :
        Divertor solid-angle fraction. `.fwbs.f_ster_div_single`.
    f_a_fw_outboard_hcd :
        Outboard HCD first-wall-area fraction. `.fwbs.f_a_fw_outboard_hcd`.

    Returns
    -------
    :
        `.first_wall.a_fw_total` (m2).
    """
    return (
        1.0 - fhole - f_ster_div_single - f_a_fw_outboard_hcd
    ) * a_fw_total_unadjusted


class BlktmodelBlanketThickness(ExplicitFunction):
    """cottax node: `calculate_blktmodel_blanket_thickness`, ports declared.

    Only instantiate this node when `blktmodel > 0` -- see module docstring.
    """

    dr_blkt_inboard = Output(lambda s: s.build.dr_blkt_inboard)
    dr_blkt_outboard = Output(lambda s: s.build.dr_blkt_outboard)
    dz_shld_upper = Output(lambda s: s.build.dz_shld_upper)

    def __call__(
        self,
        blbuith=FromExactly(lambda s: s.build.blbuith),
        blbmith=FromExactly(lambda s: s.build.blbmith),
        blbpith=FromExactly(lambda s: s.build.blbpith),
        blbuoth=FromExactly(lambda s: s.build.blbuoth),
        blbmoth=FromExactly(lambda s: s.build.blbmoth),
        blbpoth=FromExactly(lambda s: s.build.blbpoth),
        dr_shld_inboard=FromExactly(lambda s: s.build.dr_shld_inboard),
        dr_shld_outboard=FromExactly(lambda s: s.build.dr_shld_outboard),
    ):
        return calculate_blktmodel_blanket_thickness(
            blbuith,
            blbmith,
            blbpith,
            blbuoth,
            blbmoth,
            blbpoth,
            dr_shld_inboard,
            dr_shld_outboard,
        )


class Build(ExplicitFunction):
    """cottax node: `calculate_build`, ports declared.

    `dr_blkt_inboard`/`dr_blkt_outboard` read from wherever the `blktmodel`
    graph-assembly choice puts them -- `BlktmodelBlanketThickness`'s outputs, or an
    external input, per module docstring.

    **Does not own `.build.z_tf_inside_half`** -- `calculate_build`'s own Returns
    docstring explains why: real PROCESS has two independent writers of that field,
    and this port's own comparison against a converged PROCESS run showed the other
    one (`coils/calculate.py`'s `ZTfInsideHalf`) is the one whose value survives.
    """
    dz_blkt_upper = Output(lambda s: s.build.dz_blkt_upper)
    dr_fw_inboard = Output(lambda s: s.build.dr_fw_inboard)
    dr_fw_outboard = Output(lambda s: s.build.dr_fw_outboard)
    dr_bore = Output(lambda s: s.build.dr_bore)
    rbld = Output(lambda s: s.build.rbld)
    required_radial_space = Output(lambda s: s.build.required_radial_space)
    available_radial_space = Output(lambda s: s.build.available_radial_space)
    r_shld_inboard_inner = Output(lambda s: s.build.r_shld_inboard_inner)
    r_shld_outboard_outer = Output(lambda s: s.build.r_shld_outboard_outer)
    dr_tf_outboard = Output(lambda s: s.build.dr_tf_outboard)
    dr_shld_vv_gap_outboard = Output(lambda s: s.build.dr_shld_vv_gap_outboard)
    r_tf_outboard_mid = Output(lambda s: s.build.r_tf_outboard_mid)
    rspo = Output(lambda s: s.build.rspo)
    # Invented intermediate, not a real PROCESS field -- see module docstring.
    a_fw_total_unadjusted = Output(lambda s: s.first_wall.a_fw_total_unadjusted)

    def __call__(
        self,
        dr_blkt_inboard=FromExactly(lambda s: s.build.dr_blkt_inboard),
        dr_blkt_outboard=FromExactly(lambda s: s.build.dr_blkt_outboard),
        radius_fw_channel=FromExactly(lambda s: s.fwbs.radius_fw_channel),
        dr_fw_wall=FromExactly(lambda s: s.fwbs.dr_fw_wall),
        rmajor=FromExactly(lambda s: s.physics.rmajor),
        rminor=FromExactly(lambda s: s.physics.rminor),
        dr_cs=FromExactly(lambda s: s.build.dr_cs),
        dr_cs_tf_gap=FromExactly(lambda s: s.build.dr_cs_tf_gap),
        dr_tf_inboard=FromExactly(lambda s: s.build.dr_tf_inboard),
        dr_shld_vv_gap_inboard=FromExactly(lambda s: s.build.dr_shld_vv_gap_inboard),
        dr_vv_inboard=FromExactly(lambda s: s.build.dr_vv_inboard),
        dr_shld_inboard=FromExactly(lambda s: s.build.dr_shld_inboard),
        dr_fw_plasma_gap_inboard=FromExactly(lambda s: s.build.dr_fw_plasma_gap_inboard),
        r_coil_minor=FromExactly(lambda s: s.stellarator.r_coil_minor),
        f_coil_shape=FromExactly(lambda s: s.stellarator.f_coil_shape),
        stella_config_derivative_min_lcfs_coils_dist=FromExactly(
            lambda s: s.stellarator_config.stella_config_derivative_min_lcfs_coils_dist
        ),
        f_st_rmajor=FromExactly(lambda s: s.stellarator.f_st_rmajor),
        stella_config_rminor_ref=FromExactly(
            lambda s: s.stellarator_config.stella_config_rminor_ref
        ),
        dr_fw_plasma_gap_outboard=FromExactly(lambda s: s.build.dr_fw_plasma_gap_outboard),
        dr_shld_outboard=FromExactly(lambda s: s.build.dr_shld_outboard),
        gapomin=FromExactly(lambda s: s.build.gapomin),
        dr_vv_outboard=FromExactly(lambda s: s.build.dr_vv_outboard),
        a_plasma_surface=FromExactly(lambda s: s.physics.a_plasma_surface),
    ):
        return calculate_build(
            dr_blkt_inboard,
            dr_blkt_outboard,
            radius_fw_channel,
            dr_fw_wall,
            rmajor,
            rminor,
            dr_cs,
            dr_cs_tf_gap,
            dr_tf_inboard,
            dr_shld_vv_gap_inboard,
            dr_vv_inboard,
            dr_shld_inboard,
            dr_fw_plasma_gap_inboard,
            r_coil_minor,
            f_coil_shape,
            stella_config_derivative_min_lcfs_coils_dist,
            f_st_rmajor,
            stella_config_rminor_ref,
            dr_fw_plasma_gap_outboard,
            dr_shld_outboard,
            gapomin,
            dr_vv_outboard,
            a_plasma_surface,
        )


class AFwTotalNoPowerflow(ExplicitFunction):
    """cottax node: `calculate_a_fw_total_no_powerflow`. Instantiate iff `ipowerflow == 0`."""

    a_fw_total = Output(lambda s: s.first_wall.a_fw_total)

    def __call__(
        self,
        a_fw_total_unadjusted=FromExactly(lambda s: s.first_wall.a_fw_total_unadjusted),
        fhole=FromExactly(lambda s: s.fwbs.fhole),
    ):
        return calculate_a_fw_total_no_powerflow(a_fw_total_unadjusted, fhole)


class AFwTotalWithPowerflow(ExplicitFunction):
    """cottax node: `calculate_a_fw_total_with_powerflow`. Instantiate iff `ipowerflow != 0`."""

    a_fw_total = Output(lambda s: s.first_wall.a_fw_total)    

    def __call__(
        self,
        a_fw_total_unadjusted=FromExactly(lambda s: s.first_wall.a_fw_total_unadjusted),
        fhole=FromExactly(lambda s: s.fwbs.fhole),
        f_ster_div_single=FromExactly(lambda s: s.fwbs.f_ster_div_single),
        f_a_fw_outboard_hcd=FromExactly(lambda s: s.fwbs.f_a_fw_outboard_hcd),
    ):
        return calculate_a_fw_total_with_powerflow(
            a_fw_total_unadjusted, fhole, f_ster_div_single, f_a_fw_outboard_hcd
        )
