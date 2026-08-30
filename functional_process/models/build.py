"""Pure-functional port of `process/models/build.py`'s `Build` -- the tokamak radial and
vertical build.

Audit record: `functional_process/_audit/units/models/build.md`. Read it first: this
module is deliberately **not** a port of the whole file. `build.py` is 2360 lines of
which `tokamak_call_surface.md` measures 2306 as entered on `large_tokamak_eval`, and the
overwhelming majority of that is `po.obuild`/`po.ovarre` reporting inside
`if output:` blocks. What is ported here is the **minimal closure that produces the six
`.tokamak.build` boundary variables** of `_audit/tokamak_boundary.md`:

    .build.dr_tf_inboard  .build.dr_tf_outboard  .build.r_shld_inboard_inner
    .build.r_shld_outboard_outer  .build.r_tf_outboard_mid  .build.z_tf_inside_half

and nothing else. Everything those six read that this module does not produce is a
declared input -- a run input, or another slot's product.

**One of the six is not produced by this model on the run being modelled.**
`.build.dr_tf_inboard` is written at `process/models/build.py:1685` *only* when
iteration variable 140 (`dr_tf_wp_with_insulation`) is active;
`large_tokamak_eval.IN.DAT` runs with `ixc = [4, 6]`, so on that run `dr_tf_inboard` is
the plain input `large_tokamak_eval.IN.DAT:74` (`1.2`) and build.py runs the *inverse*
assignment at `:1743` instead, producing `.tfcoil.dr_tf_wp_with_insulation`. Both arms
are written here (`DrTfInboardFromWindingPack` / `DrTfWpWithInsulationFromInboardBuild`)
and the record says which is live; `tokamak_boundary.md`'s attribution of
`.build.dr_tf_inboard` to `.tokamak.build` is an artefact of its mechanical `ast` walk,
which cannot see the `ixc` guard. See the record's "contradiction with
tokamak_boundary.md".

**`.build.z_tf_inside_half` has a stellarator twin.** `models/stellarator/coils/
calculate.py::ZTfInsideHalf` owns the same `VarPath` from `st_coil`'s formula; this
module's `ZTfInsideHalf` owns it from `build.py:807`'s. Same field, two devices, two
formulas, two occupants -- exactly the shape the stellarator's own record already
describes for its two writers, and the reason that node was carved out of `st_coil` in
the first place. They are never in one graph.

Switches, one occupant class per value, no static kwargs
(`_audit/naming_convention.md` § "switches are not ports"):

| switch | live value | occupant |
|---|---|---|
| `140 in numerics.ixc` | absent | `DrTfWpWithInsulationFromInboardBuild` |
| `.tfcoil.i_tf_sup` | `1` (SC) | `DrTfOutboardSuperconducting`, `WpConductorMaxWidth-` |
| `.tfcoil.i_tf_shape` | `1` (D-shape) | `TfOutboardMidDShape` |
| `.physics.itart` | `0` | `DivertorGeometryConventional` |
| input `dz_xpoint_divertor < 1e-5` | true (`0.0`) | `DivertorGeometryConventional` |
| `.fwbs.blktmodel` | `0` | no occupant -- `dr_blkt_*`/`dz_shld_upper` are run inputs |

`itart == 1` with `dz_xpoint_divertor` left at `0.0` is `DivertorGeometrySpherical-`
`Tokamak` (not live on the reference run); `itart == 1` with it set is the slot's `None`
arm, because `divgeom`'s early return is computed and discarded at `build.py:800` and
nothing is owned. `i_tf_shape == 2` (picture frame) is `TfOutboardMidPictureFrame` /
`TfOutboardEdgeRipplePictureFrame` (not live on the reference run; live on both tracked
spherical-tokamak inputs), and `0` is a meta-value `indat._tf_shape` resolves.
`i_tf_sup` values `0`/`2` and the `itart == 0` + `dz_xpoint_divertor` set (`rspo`-only)
arm are UNPORTED -- see the record. `.tfcoil.i_tf_wp_geom` is **not** a switch of this unit at all: the
`i_tf_sup == 1` arm of `plasma_outboard_edge_toroidal_ripple` computes `r_wp_min`/
`r_wp_max` from it and then never uses either (`process/models/build.py:1551-1572`), so
declaring it, or the three `.superconducting_tfcoil.r_tf_wp_inboard_*` radii it selects
between, would be three invented edges.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.safe_math import safe_sqrt
from functional_process.paths import build, divertor, physics, tfcoil

# ---------------------------------------------------------------------------
# Vertical build -- `Build.calculate_vertical_build`, the part outside `if output:`
# (`process/models/build.py:152-172` and `:797-842`). Everything between those two
# ranges is reporting.
# ---------------------------------------------------------------------------


def calculate_z_plasma_xpoint(rminor, kappa):
    """X-point heights above and below the midplane (m).

    Ports `process/models/build.py:167-172`, unchanged. Top-down plasma symmetry is
    assumed by the source, so the two are the same number written twice; both are kept
    because both are separate `.build.*` fields with separate readers.

    Parameters
    ----------
    rminor :
        Plasma minor radius (m). `.physics.rminor`.
    kappa :
        Plasma separatrix elongation. `.physics.kappa`.

    Returns
    -------
    :
        `(z_plasma_xpoint_upper, z_plasma_xpoint_lower)` (m).
    """
    z_plasma_xpoint_upper = rminor * kappa
    z_plasma_xpoint_lower = rminor * kappa
    return z_plasma_xpoint_upper, z_plasma_xpoint_lower


def calculate_dz_blkt_upper(dr_blkt_inboard, dr_blkt_outboard):
    """Top/bottom blanket thickness (m). Ports `process/models/build.py:1665-1667`,
    unchanged -- the mean of the two radial blanket thicknesses.

    **`models/stellarator/build.py` computes the same expression**
    (`st_build`, `:168` there, `process/models/stellarator/build.py:38`), which is why
    `.build.dz_blkt_upper` was not on the boundary of the stellarator graph and *was*
    on the tokamak's: the stellarator's producer is one of `st_build`'s fourteen
    outputs, and there was no tokamak node for the same line. Same field,
    two devices, two nodes, never both in one graph -- the correspondence
    `ZTfInsideHalf` already records for `.build.z_tf_inside_half`, and the reason each
    of them is carved out as a node of its own rather than folded into a neighbour.

    Its own node rather than part of any existing one because nothing in this file's
    ported closure reads it: `fw.py` and `shield.py` do (the upper blanket and shield
    stack), and both were reading `0.0`.

    Parameters
    ----------
    dr_blkt_inboard :
        Inboard blanket thickness (m). `.build.dr_blkt_inboard`.
    dr_blkt_outboard :
        Outboard blanket thickness (m). `.build.dr_blkt_outboard`.

    Returns
    -------
    :
        `.build.dz_blkt_upper` (m).
    """
    return 0.5 * (dr_blkt_inboard + dr_blkt_outboard)


def calculate_divertor_geometry_conventional(
    rmajor,
    rminor,
    kappa,
    triang,
    plsepi,
    plsepo,
    plleni,
    plleno,
    betai,
    betao,
):
    """Divertor height and outer strike point radius, conventional-tokamak arm.

    Ports the computing half of `Build.divgeom` (`process/models/build.py:862-943`), the
    `itart != 1` path. The inboard and outboard plasma surfaces are approximated by arcs
    and followed past the X-point; the height is the span between the highest plate top
    and the lowest plate bottom.

    Two departures from the source, both mechanical:

    - `triu` and `tril` (`:868-869`) are both `.physics.triang`; only `tril` enters the
      arithmetic (`triu` is used solely by the reporting block), so this signature takes
      `triang` once.
    - `rplti`, `rplbi`, `rplto`, `rplbo` (`:916-940`) are the *radial* plate-end
      coordinates. They are computed by the source and never read by it outside the
      reporting block, so they are not computed here. The four `z*` coordinates, which
      are what `divht` is built from, are.

    Parameters
    ----------
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    kappa, triang :
        Plasma separatrix elongation and triangularity. `.physics.kappa`,
        `.physics.triang`.
    plsepi, plsepo :
        Poloidal length, X-point to inboard/outboard strike point (m). `.build.plsepi`,
        `.build.plsepo`.
    plleni, plleno :
        Inboard/outboard divertor plate length (m). `.build.plleni`, `.build.plleno`.
    betai, betao :
        Poloidal plane angle between inboard/outboard divertor leg and plate (rad).
        `.divertor.betai`, `.divertor.betao`.

    Returns
    -------
    :
        `(dz_xpoint_divertor, rspo)` -- divertor height (m) and outer strike point
        radius (m).
    """
    # Radius of the outer and inner plasma arcs.
    rco = 0.5 * safe_sqrt(
        (rminor**2 * ((triang + 1.0) ** 2 + kappa**2) ** 2) / ((triang + 1.0) ** 2)
    )
    rci = 0.5 * safe_sqrt(
        (rminor**2 * ((triang - 1.0) ** 2 + kappa**2) ** 2) / ((triang - 1.0) ** 2)
    )

    # Angles between vertical and the divertor legs. The inboard arc angle is the
    # outboard leg angle and vice versa -- the source's own naming, kept.
    thetao = jnp.arcsin(1.0 - (rminor * (1.0 - triang)) / rci)
    thetai = jnp.arcsin(1.0 - (rminor * (1.0 + triang)) / rco)

    # Lower X-point.
    rxpt = rmajor - triang * rminor
    zxpt = -1.0 * kappa * rminor

    # Strike points.
    zspi = zxpt - plsepi * jnp.sin(thetai)
    rspo = rxpt + plsepo * jnp.cos(thetao)
    zspo = zxpt - plsepo * jnp.sin(thetao)

    # Plate ends, vertical coordinates only.
    zplti = zspi + (plleni / 2.0) * jnp.sin(thetai + betai)
    zplbi = zspi - (plleni / 2.0) * jnp.sin(thetai + betai)
    zplto = zspo + (plleno / 2.0) * jnp.sin(thetao + betao)
    zplbo = zspo - (plleno / 2.0) * jnp.sin(thetao + betao)

    dz_xpoint_divertor = jnp.maximum(zplti, zplto) - jnp.minimum(zplbo, zplbi)
    return dz_xpoint_divertor, rspo


def calculate_divertor_geometry_spherical_tokamak(rminor):
    """Divertor height, spherical-tokamak arm.

    Ports `process/models/build.py:862-863`, the whole of it: `divgeom` opens with
    `if itart == 1: return 1.75e0 * rminor` -- "TART option: Peng SOFT paper", per the
    method's own docstring -- and that early return is the entire arm. None of the
    conventional arc geometry runs, and in particular control never reaches the
    `.build.rspo` write at `:912`, which is why this arm's occupant owns one field
    where the conventional arm's owns two.

    Parameters
    ----------
    rminor :
        Plasma minor radius (m). `.physics.rminor`.

    Returns
    -------
    :
        `dz_xpoint_divertor` -- divertor height (m).
    """
    return 1.75 * rminor


def calculate_z_tf_inside_half(
    z_plasma_xpoint_upper,
    dz_xpoint_divertor,
    dz_divertor,
    dz_shld_lower,
    dz_vv_lower,
    dz_shld_vv_gap,
    dz_shld_thermal,
    dr_tf_shld_gap,
):
    """Half-height to the inside edge of the TF coil (m).

    Ports `process/models/build.py:807-816`, unchanged. TF coils are assumed vertically
    symmetric, so the source uses the *lower* build for both halves regardless of
    `i_single_null` -- its own comment says so, and that is why this function carries no
    `i_single_null` arm even though the surrounding method has several.

    Parameters
    ----------
    z_plasma_xpoint_upper :
        Upper X-point height (m). `.build.z_plasma_xpoint_upper`.
    dz_xpoint_divertor :
        Divertor height, X-point to divertor structure (m). `.build.dz_xpoint_divertor`.
    dz_divertor :
        Divertor structure vertical thickness (m). `.divertor.dz_divertor`.
    dz_shld_lower, dz_vv_lower :
        Lower shield / vacuum vessel vertical thickness (m). `.build.dz_shld_lower`,
        `.build.dz_vv_lower`.
    dz_shld_vv_gap, dz_shld_thermal, dr_tf_shld_gap :
        Vessel-thermal shield gap, thermal shield thickness, TF-thermal shield gap (m).
        `.build.dz_shld_vv_gap`, `.build.dz_shld_thermal`, `.build.dr_tf_shld_gap`.

    Returns
    -------
    :
        `.build.z_tf_inside_half` (m).
    """
    return (
        z_plasma_xpoint_upper
        + dz_xpoint_divertor
        + dz_divertor
        + dz_shld_lower
        + dz_vv_lower
        + dz_shld_vv_gap
        + dz_shld_thermal
        + dr_tf_shld_gap
    )


def calculate_tf_top_height_double_null(z_tf_inside_half, dr_tf_inboard):
    """Height to the top of the TF coil and the up-down offset (m), double null.

    Ports `process/models/build.py:820-824`, the
    `i_single_null == DivertorNumberModels.DOUBLE_NULL` arm. A double-null machine is
    up-down symmetric, so the top of the coil is the inside half-height plus one
    inboard-leg thickness and the offset between the upper and lower halves is exactly
    zero -- the source assigns the literal `0.0e0`.

    **`.build.z_tf_top` had no producer in this port until 2026-08-30**, and it is not
    an unread accumulation: `models/tfcoil/base.py::TfCoilShapeDShapeSingleNull` and
    `TfCoilShapePictureFrameTart` both read it to place the coil's arcs, and
    `models/pfcoil/geometry.py` places the divertor PF coils from it. Frozen at the cold
    `0.0` it puts the top of the TF coil at the midplane. See
    `_audit/units/models/build.md` § "the vertical build's two missing producers".

    Parameters
    ----------
    z_tf_inside_half :
        Half-height to the inside edge of the TF coil (m). `.build.z_tf_inside_half`.
    dr_tf_inboard :
        Inboard TF coil radial thickness (m), used here as the coil's vertical
        thickness. `.build.dr_tf_inboard`.

    Returns
    -------
    :
        `(.build.z_tf_top, .build.dz_tf_upper_lower_midplane)` (m).
    """
    z_tf_top = z_tf_inside_half + dr_tf_inboard
    return z_tf_top, jnp.zeros_like(z_tf_top)


def calculate_tf_top_height_single_null(
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
    """Height to the top of the TF coil and the up-down offset (m), single null.

    Ports `process/models/build.py:826-841`, the
    `i_single_null == DivertorNumberModels.SINGLE_NULL` arm and the reference tokamak's.
    The upper build is stacked independently of the lower one -- a single-null machine
    has a divertor below and a blanket above -- so `z_tf_top` is **not**
    `z_tf_inside_half + dr_tf_inboard` here, and `dz_tf_upper_lower_midplane` is
    precisely that difference: how much further the coil reaches above the midplane than
    below it. It is routinely negative (`-1.234` m on `large_tokamak_nof`), because the
    divertor stack below is taller than the blanket stack above.

    `z_tf_inside_half` and `dr_tf_inboard` appear **only** in the offset, never in the
    top height, so the two returned values are not two views of one number: the source
    computes them from disjoint halves of the vertical build and the second subtracts
    the first arm's expression from the first.

    `.build.dz_blkt_upper` is produced by `calculate_dz_blkt_upper` below (also landed
    2026-08-30, and for the same reason). The other twelve reads are run inputs or other
    slots' products.

    Parameters
    ----------
    z_tf_inside_half, dr_tf_inboard :
        Inside half-height and inboard leg thickness (m) -- the *lower* half's reach,
        subtracted to form the offset. `.build.z_tf_inside_half`,
        `.build.dr_tf_inboard`.
    dr_tf_shld_gap, dz_shld_thermal, dz_shld_vv_gap, dz_vv_upper, dz_shld_upper :
        TF-thermal shield gap, thermal shield thickness, vessel-thermal shield gap,
        upper vessel and upper shield thicknesses (m).
    dr_shld_blkt_gap, dz_blkt_upper :
        Shield-blanket gap and upper blanket thickness (m). `.build.dr_shld_blkt_gap`,
        `.build.dz_blkt_upper`.
    dr_fw_inboard, dr_fw_outboard :
        Inboard and outboard first wall thicknesses (m); the source takes their mean.
    dz_fw_plasma_gap, z_plasma_xpoint_upper :
        Upper first wall-plasma gap and upper X-point height (m).

    Returns
    -------
    :
        `(.build.z_tf_top, .build.dz_tf_upper_lower_midplane)` (m).
    """
    z_tf_top = (
        dr_tf_inboard
        + dr_tf_shld_gap
        + dz_shld_thermal
        + dz_shld_vv_gap
        + dz_vv_upper
        + dz_shld_upper
        + dr_shld_blkt_gap
        + dz_blkt_upper
        + 0.5 * (dr_fw_inboard + dr_fw_outboard)
        + dz_fw_plasma_gap
        + z_plasma_xpoint_upper
    )
    return z_tf_top, z_tf_top - (z_tf_inside_half + dr_tf_inboard)


# ---------------------------------------------------------------------------
# Radial build -- `Build.calculate_radial_build`, the part outside `if output:`
# (`process/models/build.py:1649-1977`).
# ---------------------------------------------------------------------------


def calculate_dz_blkt_upper(dr_blkt_inboard, dr_blkt_outboard):
    """Top/bottom blanket thickness (m) -- the mean of the two radial thicknesses.

    Ports `process/models/build.py:1664-1667`, unconditional: it sits *below* the
    `blktmodel > 0` block that may have just rewritten `dr_blkt_inboard`/`_outboard`,
    and runs on every configuration. On the reference tokamak `.fwbs.blktmodel` is `0`,
    so both operands are run inputs (`large_tokamak_nof.IN.DAT`'s `dr_blkt_inboard`/
    `dr_blkt_outboard`) and this is a producer with no unproduced dependency.

    **Landed 2026-08-30 as `calculate_tf_top_height_single_null`'s missing dependency.**
    It is on `missing_producers_tokamak.txt` in its own right -- `models/fw.py` and
    `models/vacuum/vacuum.py` read it too -- but the reason it is ported *now* is that
    the single-null `z_tf_top` stack reads it, and landing that producer on top of a
    frozen `0.0` would have produced a number that only looked produced.

    Parameters
    ----------
    dr_blkt_inboard, dr_blkt_outboard :
        Inboard and outboard blanket radial thicknesses (m). `.build.dr_blkt_inboard`,
        `.build.dr_blkt_outboard`.

    Returns
    -------
    :
        `.build.dz_blkt_upper` (m).
    """
    return 0.5 * (dr_blkt_inboard + dr_blkt_outboard)


def calculate_dr_tf_inboard(
    dr_tf_wp_with_insulation, dr_tf_plasma_case, dr_tf_nose_case
):
    """Inboard TF coil radial thickness (m), from the winding pack.

    Ports `process/models/build.py:1685-1689`. **Only runs when iteration variable 140
    (`dr_tf_wp_with_insulation`) is active**; otherwise `dr_tf_inboard` is an input and
    `calculate_dr_tf_wp_with_insulation` below runs instead. The two are exact inverses
    of one another and exactly one of them runs per run -- `conditional-ownership-by-run
    -config`, the same shape `models/stellarator/build.md` records for
    `.build.dr_blkt_inboard`.

    Parameters
    ----------
    dr_tf_wp_with_insulation :
        Winding pack radial thickness including ground insulation (m).
        `.tfcoil.dr_tf_wp_with_insulation`.
    dr_tf_plasma_case, dr_tf_nose_case :
        Plasma-side and nose case radial thickness (m). `.tfcoil.dr_tf_plasma_case`,
        `.tfcoil.dr_tf_nose_case`.

    Returns
    -------
    :
        `.build.dr_tf_inboard` (m).
    """
    return dr_tf_wp_with_insulation + dr_tf_plasma_case + dr_tf_nose_case


def calculate_dr_tf_wp_with_insulation(
    dr_tf_inboard, dr_tf_plasma_case, dr_tf_nose_case
):
    """Winding pack radial thickness including ground insulation (m).

    Ports `process/models/build.py:1743-1747`, the arm that runs when iteration variable
    140 is *not* active -- the live arm on `large_tokamak_eval.IN.DAT` (`ixc = [4, 6]`).
    Exact inverse of `calculate_dr_tf_inboard`; see there.

    `process/models/build.py:1743` is the sole writer of this field on the tokamak path
    (the only other writer anywhere is `models/stellarator/coils/calculate.py:489`), so
    owning it here creates no dual ownership with `models/tfcoil/**`, which only reads
    it.

    Parameters
    ----------
    dr_tf_inboard :
        Inboard TF coil radial thickness (m). `.build.dr_tf_inboard`.
    dr_tf_plasma_case, dr_tf_nose_case :
        Plasma-side and nose case radial thickness (m). `.tfcoil.dr_tf_plasma_case`,
        `.tfcoil.dr_tf_nose_case`.

    Returns
    -------
    :
        `.tfcoil.dr_tf_wp_with_insulation` (m).
    """
    return dr_tf_inboard - dr_tf_plasma_case - dr_tf_nose_case


def calculate_r_tf_inboard_radii_tf_outside_cs(
    dr_bore, dr_cs, fseppc, fcspc, sigallpc, dr_cs_tf_gap, dr_tf_inboard
):
    """The CS-to-TF slice of the inboard radial build (m): CS bore radius, CS
    pre-compression structure thickness, and the inboard TF leg's inner, middle and
    plasma-facing radii. The `(i_tf_inside_cs == TF_OUTSIDE_CS, i_cs_precomp ==
    CS_PRECOMPRESSION_STRUCTURE_PRESENT)` arm -- both the live values on
    `large_tokamak_eval.IN.DAT` (`i_tf_inside_cs` defaults to `0`,
    `build_variables.py:189`, never set in the file; `i_cs_precomp` defaults to `1`,
    `:183`, same).

    Ports `process/models/build.py:1691-1735` as one contiguous slice: `dr_cs_bore =
    dr_bore` (else-arm, `:1698-1699`), the pre-compression thickness (`:1702-1713`),
    the else-arm `r_tf_inboard_in` (`:1717-1725`) and the unconditional
    `r_tf_inboard_mid`/`r_tf_inboard_out` (`:1727-1735`). Added 2026-08-27
    (`cold_boundary.md` producer 2): `r_tf_inboard_in`/`r_tf_inboard_out` were the two
    boundary zeros behind 3 of the cold tokamak MDA's 11 non-finite roots
    (`a_tf_inboard_total = pi*(out^2 - in^2) = 0` fed the TF current density and both
    CICC inboard fractions). The slice is taken whole rather than at the record's
    `:1720` line so `dr_cs_precomp` and `dr_cs_bore` are produced instead of read
    stale -- `dr_cs_bore` was a standing boundary input with a wrong cold value
    (`1.42` for a converged `2.00384`, the record's 29-name overwrite list) read by
    `pfcoil/currents.py::CSFluxSwing`, and nothing else produces either field.

    Parameters
    ----------
    dr_bore :
        Machine bore radius (m). `.build.dr_bore`.
    dr_cs :
        Central solenoid radial thickness (m). `.build.dr_cs`.
    fseppc :
        CS separation force (N). `.build.fseppc`.
    fcspc :
        Fraction of space occupied by pre-compression structure. `.build.fcspc`.
    sigallpc :
        Allowable stress in the pre-compression structure (Pa). `.build.sigallpc`.
    dr_cs_tf_gap :
        Gap between the CS and the inboard TF leg (m). `.build.dr_cs_tf_gap`.
    dr_tf_inboard :
        Inboard TF coil radial thickness (m). `.build.dr_tf_inboard`.

    Returns
    -------
    tuple
        `(dr_cs_bore, dr_cs_precomp, r_tf_inboard_in, r_tf_inboard_mid,
        r_tf_inboard_out)`, all m -- `.build.` each.
    """
    dr_cs_bore = dr_bore

    dr_cs_precomp = fseppc / (
        2.0e0 * jnp.pi * fcspc * sigallpc * (2.0 * dr_cs_bore + dr_cs)
    )

    r_tf_inboard_in = dr_bore + dr_cs + dr_cs_precomp + dr_cs_tf_gap
    r_tf_inboard_mid = r_tf_inboard_in + 0.5e0 * dr_tf_inboard
    r_tf_inboard_out = r_tf_inboard_in + dr_tf_inboard
    return dr_cs_bore, dr_cs_precomp, r_tf_inboard_in, r_tf_inboard_mid, r_tf_inboard_out


def calculate_r_tf_inboard_radii_no_cs_precomp(
    dr_bore, dr_cs, dr_cs_tf_gap, dr_tf_inboard
):
    """The CS-to-TF slice of the inboard radial build (m) with **no CS pre-compression
    structure**: the `(i_tf_inside_cs == TF_OUTSIDE_CS, i_cs_precomp == 0)` arm -- the
    live cell on both tracked spherical-tokamak files
    (`spherical_tokamak_eval.IN.DAT:70-71` sets `i_cs_precomp = 0`,
    `i_tf_inside_cs = 0`; `st_regression.IN.DAT:1811`/`:1845` the same).

    Ports `process/models/build.py:1691-1735` with the `i_cs_precomp` else-arm taken:
    `dr_cs_bore = dr_bore` (else-arm, `:1698-1699`), `dr_cs_precomp = 0.0e0` (the
    literal at `:1714` -- `fseppc`/`fcspc`/`sigallpc` are never read, which is why this
    is a different occupant and not a kwarg on the sibling), the else-arm
    `r_tf_inboard_in` (`:1717-1725`, its `+ dr_cs_precomp` term absorbed as the exact
    zero it is on this arm) and the unconditional `r_tf_inboard_mid`/`r_tf_inboard_out`
    (`:1727-1735`). Same write-set as `calculate_r_tf_inboard_radii_tf_outside_cs`, so
    the two occupants are interchangeable in the slot; `dr_cs_precomp` stays an output
    because PROCESS writes the field on this arm too and downstream readers
    (`pfcoil/currents.py::CSFluxSwing`'s chain via `.build.dr_cs_bore`, the vertical
    CS stack) must see the produced zero, not a stale boundary value.

    Parameters
    ----------
    dr_bore :
        Machine bore radius (m). `.build.dr_bore`.
    dr_cs :
        Central solenoid radial thickness (m). `.build.dr_cs`.
    dr_cs_tf_gap :
        Gap between the CS and the inboard TF leg (m). `.build.dr_cs_tf_gap`.
    dr_tf_inboard :
        Inboard TF coil radial thickness (m). `.build.dr_tf_inboard`.

    Returns
    -------
    tuple
        `(dr_cs_bore, dr_cs_precomp, r_tf_inboard_in, r_tf_inboard_mid,
        r_tf_inboard_out)`, all m -- `.build.` each. `dr_cs_precomp` is identically
        zero on this arm.
    """
    dr_cs_bore = dr_bore

    dr_cs_precomp = jnp.zeros_like(jnp.asarray(dr_bore))

    r_tf_inboard_in = dr_bore + dr_cs + dr_cs_tf_gap
    r_tf_inboard_mid = r_tf_inboard_in + 0.5e0 * dr_tf_inboard
    r_tf_inboard_out = r_tf_inboard_in + dr_tf_inboard
    return dr_cs_bore, dr_cs_precomp, r_tf_inboard_in, r_tf_inboard_mid, r_tf_inboard_out


def calculate_r_shld_inboard_inner(
    rmajor,
    rminor,
    dr_fw_plasma_gap_inboard,
    dr_fw_inboard,
    dr_blkt_inboard,
    dr_shld_inboard,
):
    """Radius to the inner edge of the inboard shield (m).

    Ports `process/models/build.py:1873-1880`, unchanged. Note this is built *inwards
    from the plasma*, not accumulated outwards from `r_tf_inboard_in`, so it is
    independent of the whole `i_tf_inside_cs`/`i_cs_precomp` central-solenoid chain
    above it in the source method.

    `dr_blkt_inboard` is a run input on `large_tokamak_eval` (`blktmodel = 0`); under
    `blktmodel > 0` it is instead produced by `process/models/build.py:1650-1654`, the
    block `models/stellarator/build.py::BlktmodelBlanketThickness` already ports
    verbatim. Either way it arrives here as an ordinary read.

    Parameters
    ----------
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    dr_fw_plasma_gap_inboard :
        Inboard plasma-first wall gap (m). `.build.dr_fw_plasma_gap_inboard`.
    dr_fw_inboard :
        Inboard first wall thickness (m). `.build.dr_fw_inboard`.
    dr_blkt_inboard :
        Inboard blanket thickness (m). `.build.dr_blkt_inboard`.
    dr_shld_inboard :
        Inboard shield thickness (m). `.build.dr_shld_inboard`.

    Returns
    -------
    :
        `.build.r_shld_inboard_inner` (m).
    """
    return (
        rmajor
        - rminor
        - dr_fw_plasma_gap_inboard
        - dr_fw_inboard
        - dr_blkt_inboard
        - dr_shld_inboard
    )


def calculate_r_shld_outboard_outer(
    rmajor,
    rminor,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_shld_outboard,
):
    """Radius to the outer edge of the outboard shield (m).

    Ports `process/models/build.py:1883-1890`, unchanged. Outboard mirror of
    `calculate_r_shld_inboard_inner`; the same note about `dr_blkt_outboard` applies.

    Parameters
    ----------
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    dr_fw_plasma_gap_outboard :
        Outboard plasma-first wall gap (m). `.build.dr_fw_plasma_gap_outboard`.
    dr_fw_outboard :
        Outboard first wall thickness (m). `.build.dr_fw_outboard`.
    dr_blkt_outboard :
        Outboard blanket thickness (m). `.build.dr_blkt_outboard`.
    dr_shld_outboard :
        Outboard shield thickness (m). `.build.dr_shld_outboard`.

    Returns
    -------
    :
        `.build.r_shld_outboard_outer` (m).
    """
    return (
        rmajor
        + rminor
        + dr_fw_plasma_gap_outboard
        + dr_fw_outboard
        + dr_blkt_outboard
        + dr_shld_outboard
    )


def calculate_dr_tf_outboard_superconducting(dr_tf_inboard):
    """Outboard TF coil leg thickness (m), superconducting coil (`i_tf_sup == 1`).

    Ports `process/models/build.py:1898`: the outboard leg is the same thickness as the
    inboard one. The `i_tf_sup != 1` arm (`:1894-1896`) scales it by
    `.build.f_dr_tf_outboard_inboard` and is UNPORTED -- see the module docstring.

    Parameters
    ----------
    dr_tf_inboard :
        Inboard TF coil radial thickness (m). `.build.dr_tf_inboard`.

    Returns
    -------
    :
        `.build.dr_tf_outboard` (m).
    """
    return dr_tf_inboard


def calculate_dx_tf_wp_conductor_max_superconducting(
    dx_tf_wp_primary_toroidal, dx_tf_wp_insulation, dx_tf_wp_insertion_gap
):
    """Maximum toroidal conductor width of the winding pack (m), superconducting coil.

    Ports `process/models/build.py:1570-1572`, the `i_tf_sup == 1` arm of
    `plasma_outboard_edge_toroidal_ripple`. Split out of that function because it is the
    only part of it `i_tf_sup` branches, and because its two arms read entirely
    different variables.

    **The output is a mint**: PROCESS keeps `dx_tf_wp_conductor_max` as a local, so
    `.tfcoil.dx_tf_wp_conductor_max` names a place `DataStructure` does not have. It is
    put in `.tfcoil` because that is the area every one of its inputs lives in.

    `r_wp_min`/`r_wp_max` (`:1551-1567`) are computed alongside this in the source and
    then never used on this arm -- see the module docstring on `i_tf_wp_geom`.

    Parameters
    ----------
    dx_tf_wp_primary_toroidal :
        Primary toroidal winding pack thickness (m).
        `.tfcoil.dx_tf_wp_primary_toroidal`.
    dx_tf_wp_insulation, dx_tf_wp_insertion_gap :
        Ground insulation thickness and insertion gap (m).
        `.tfcoil.dx_tf_wp_insulation`, `.tfcoil.dx_tf_wp_insertion_gap`.

    Returns
    -------
    :
        `.tfcoil.dx_tf_wp_conductor_max` (m), a mint.
    """
    return dx_tf_wp_primary_toroidal - 2.0 * (
        dx_tf_wp_insulation + dx_tf_wp_insertion_gap
    )


def plasma_outboard_edge_toroidal_ripple_fitted(
    ripple_b_tf_plasma_edge_max,
    r_tf_outboard_mid,
    n_tf_coils,
    rmajor,
    rminor,
    dx_tf_wp_conductor_max,
):
    """TF ripple at the outboard plasma edge, and the leg radius that would produce the
    allowed maximum.

    Ports the `i_tf_shape != PICTURE_FRAME` arm of
    `Build.plasma_outboard_edge_toroidal_ripple` (`process/models/build.py:1591-1623`) --
    the MAGINT-fitted correlation. Both `numpy` kludges the source applies are kept:
    `base` is clamped at `1e-6` (`:1611-1613`) and a non-finite `r_tf_outboard_midmin` is
    replaced by `3 (R + a)` (`:1619-1623`), both as `jnp.where`-shaped selections rather
    than Python branches, since both test a traced quantity.

    **`flag` is not returned.** The source's third return value (`:1626-1634`) is a
    fitted-range diagnostic that PROCESS only turns into a log warning; nothing in the
    graph reads `.build.ripflag`. It is also a step function of `n_tf_coils` with a
    threshold at exactly `16`, which is exactly the value `large_tokamak_eval` runs at,
    so a finite-difference gradient of it against this port compares a jump to a zero.
    Excluding it is deliberate; see the audit record's "what is not ported".

    Parameters
    ----------
    ripple_b_tf_plasma_edge_max :
        Maximum allowed ripple at the plasma edge (per cent).
        `.tfcoil.ripple_b_tf_plasma_edge_max`.
    r_tf_outboard_mid :
        Radius to the centre of the outboard TF leg (m), the point the ripple is
        evaluated at. `.build.r_tf_outboard_mid`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.
    dx_tf_wp_conductor_max :
        Maximum toroidal conductor width of the winding pack (m), the mint above.

    Returns
    -------
    :
        `(ripple_b_tf_plasma_edge, r_tf_outboard_midmin)` -- ripple at
        `r_tf_outboard_mid` (per cent) and the minimum leg radius meeting
        `ripple_b_tf_plasma_edge_max` (m).
    """
    # Winding pack to coil toroidal length ratio at the plasma centre.
    x = dx_tf_wp_conductor_max * n_tf_coils / rmajor

    c1 = 0.875 - 0.0557 * x
    c2 = 1.617 + 0.0832 * x

    ripple_b_tf_plasma_edge = (
        100.0 * c1 * ((rmajor + rminor) / r_tf_outboard_mid) ** (n_tf_coils - c2)
    )

    # `if base <= 1e-6: base = 1e-6` -- the source's own kludge against a negative or
    # complex root.
    base = jnp.maximum(0.01 * ripple_b_tf_plasma_edge_max / c1, 1e-6)

    r_tf_outboard_midmin = (rmajor + rminor) / (base ** (1.0 / (n_tf_coils - c2)))
    r_tf_outboard_midmin = jnp.where(
        jnp.isinf(r_tf_outboard_midmin),
        (rmajor + rminor) * 3.0,
        r_tf_outboard_midmin,
    )

    return ripple_b_tf_plasma_edge, r_tf_outboard_midmin


def plasma_outboard_edge_toroidal_ripple_picture_frame(
    ripple_b_tf_plasma_edge_max,
    r_tf_outboard_mid,
    n_tf_coils,
    rmajor,
    rminor,
):
    """TF ripple at the outboard plasma edge for the picture-frame coil, and the leg
    radius that would produce the allowed maximum.

    Ports the `i_tf_shape == PICTURE_FRAME` arm of
    `Build.plasma_outboard_edge_toroidal_ripple` (`process/models/build.py:1582-1590`) --
    "Ken McClements ST picture frame coil analytical ripple calc" (2022, approximate to
    ~10% of numerical per the source docstring). The exponent is the bare `n_tf_coils`,
    not the fitted arm's `n_tf_coils - c2`, and neither `c1`/`c2` nor the winding pack
    appears: `dx_tf_wp_conductor_max` is computed upstream (`:1551-1580`) and then dead
    on this arm, which is why this function does not take it and the picture-frame
    occupants below do not read it.

    **The source's kludges do not exist on this arm, and none are added.** The fitted
    arm clamps `base` at `1e-6` and replaces a non-finite `r_tf_outboard_midmin`; here
    `(0.01 * ripple_b_tf_plasma_edge_max) ** (1 / n_tf_coils)` is used unguarded
    (`:1588-1590`), so a zero `ripple_b_tf_plasma_edge_max` divides by zero exactly as
    PROCESS would. `flag` is not returned, as for the sibling -- and on this arm it is
    not even a dropped diagnostic: the source sets `flag = 0` at `:1581` and never
    reassigns it inside the picture-frame branch, so the value is identically zero.

    Parameters
    ----------
    ripple_b_tf_plasma_edge_max :
        Maximum allowed ripple at the plasma edge (per cent).
        `.tfcoil.ripple_b_tf_plasma_edge_max`.
    r_tf_outboard_mid :
        Radius to the centre of the outboard TF leg (m), the point the ripple is
        evaluated at. `.build.r_tf_outboard_mid`.
    n_tf_coils :
        Number of TF coils (outer legs, for an ST). `.tfcoil.n_tf_coils`.
    rmajor, rminor :
        Plasma major/minor radius (m). `.physics.rmajor`, `.physics.rminor`.

    Returns
    -------
    :
        `(ripple_b_tf_plasma_edge, r_tf_outboard_midmin)` -- ripple at
        `r_tf_outboard_mid` (per cent) and the minimum leg radius meeting
        `ripple_b_tf_plasma_edge_max` (m).
    """
    ripple_b_tf_plasma_edge = (
        100.0 * ((rmajor + rminor) / r_tf_outboard_mid) ** n_tf_coils
    )

    r_tf_outboard_midmin = (rmajor + rminor) / (
        (0.01 * ripple_b_tf_plasma_edge_max) ** (1.0 / n_tf_coils)
    )

    return ripple_b_tf_plasma_edge, r_tf_outboard_midmin


def calculate_r_tf_outboard_mid_unrippled(
    r_shld_outboard_outer,
    dr_shld_blkt_gap,
    dr_vv_outboard,
    gapomin,
    dr_shld_thermal_outboard,
    dr_tf_shld_gap,
    dr_tf_outboard,
):
    """Radius to the centre of the outboard TF leg (m), before the ripple constraint.

    Ports `process/models/build.py:1901-1909` -- the outboard build stacked up from the
    shield with the *minimum* vessel-TF gap `gapomin`. This is the value the source
    assigns first and then may raise; `calculate_r_tf_outboard_mid` below is that raise.

    Returns a mint: PROCESS overwrites `.build.r_tf_outboard_mid` in place at `:1939`,
    so the pre-ripple value has no name of its own in `DataStructure`. Named
    `r_tf_outboard_mid_unrippled` under `.build`.

    Parameters
    ----------
    r_shld_outboard_outer :
        Radius to the outer edge of the outboard shield (m).
        `.build.r_shld_outboard_outer`.
    dr_shld_blkt_gap, dr_vv_outboard, gapomin :
        Vessel-blanket gap, outboard vessel thickness, minimum vessel-TF gap (m).
        `.build.dr_shld_blkt_gap`, `.build.dr_vv_outboard`, `.build.gapomin`.
    dr_shld_thermal_outboard, dr_tf_shld_gap :
        Outboard thermal shield thickness and TF-thermal shield gap (m).
        `.build.dr_shld_thermal_outboard`, `.build.dr_tf_shld_gap`.
    dr_tf_outboard :
        Outboard TF leg thickness (m). `.build.dr_tf_outboard`.

    Returns
    -------
    :
        `.build.r_tf_outboard_mid_unrippled` (m), a mint.
    """
    return (
        r_shld_outboard_outer
        + dr_shld_blkt_gap
        + dr_vv_outboard
        + gapomin
        + dr_shld_thermal_outboard
        + dr_tf_shld_gap
        + 0.5 * dr_tf_outboard
    )


def calculate_r_tf_outboard_mid(r_tf_outboard_mid_unrippled, r_tf_outboard_midmin):
    """Radius to the centre of the outboard TF leg (m), after the ripple constraint.

    Ports `process/models/build.py:1937-1956`: *"if the ripple is too large then move the
    outboard TF coil leg"*. The source writes this as
    `if r_tf_outboard_midl > r_tf_outboard_mid:` and assigns in the taken branch only,
    which is `jnp.maximum` of the two -- **and the two arguments are independent**:
    `r_tf_outboard_midmin` is a function of `rmajor`, `rminor`, `n_tf_coils` and the
    winding pack only (see `plasma_outboard_edge_toroidal_ripple_fitted`), never of
    `r_tf_outboard_mid`, so this is not a fixed point and there is no cycle here.

    Parameters
    ----------
    r_tf_outboard_mid_unrippled :
        The stacked-up outboard build (m), from
        `calculate_r_tf_outboard_mid_unrippled`.
    r_tf_outboard_midmin :
        The minimum leg radius meeting the ripple limit (m), from
        `plasma_outboard_edge_toroidal_ripple_fitted`.

    Returns
    -------
    :
        `.build.r_tf_outboard_mid` (m).
    """
    return jnp.maximum(r_tf_outboard_mid_unrippled, r_tf_outboard_midmin)


def calculate_dr_tf_inner_bore(
    r_tf_outboard_mid, dr_tf_outboard, r_tf_inboard_mid, dr_tf_inboard
):
    """TF coil horizontal bore at the midplane (m) -- inner face to inner face.

    Ports `process/models/build.py:1911-1913` **and** `:1949-1955`, which are the same
    expression written twice: PROCESS computes it once from the stacked-up
    `r_tf_outboard_mid` and then recomputes it verbatim inside the "if the ripple is too
    large, move the outboard leg" branch. That first write is a
    `redundant-duplicate-write` -- only the second survives when the branch is taken,
    and when it is not, the two are equal. This port evaluates it once, at the *final*
    `.build.r_tf_outboard_mid` (`calculate_r_tf_outboard_mid`'s `jnp.maximum`), which
    reproduces whichever of the source's two writes lands. Same resolution as
    `TfOutboardEdgeRipple`'s.

    Read by `models/structure.py`'s support-structure masses and
    `models/costs/costs_2015.py`'s Account 22 magnet costs; unproduced in this port
    until 2026-08-30, when it was frozen at the cold `0.0` against PROCESS's `11.794` m
    on `large_tokamak_nof`.

    Parameters
    ----------
    r_tf_outboard_mid, dr_tf_outboard :
        Radius to the centre of the outboard TF leg and its radial thickness (m).
        `.build.r_tf_outboard_mid`, `.build.dr_tf_outboard`.
    r_tf_inboard_mid, dr_tf_inboard :
        Radius to the centre of the inboard TF leg and its radial thickness (m).
        `.build.r_tf_inboard_mid`, `.build.dr_tf_inboard`.

    Returns
    -------
    :
        `.build.dr_tf_inner_bore` (m).
    """
    return (r_tf_outboard_mid - 0.5 * dr_tf_outboard) - (
        r_tf_inboard_mid - 0.5 * dr_tf_inboard
    )


def calculate_dr_shld_vv_gap_outboard(
    r_tf_outboard_mid,
    dr_tf_outboard,
    dr_vv_outboard,
    r_shld_outboard_outer,
    dr_shld_thermal_outboard,
    dr_tf_shld_gap,
    dr_shld_blkt_gap,
):
    """Gap between the outboard vacuum vessel and thermal shield (m).

    Ports `process/models/build.py:1940-1956`. The source writes this in two arms --
    the subtraction below when the TF leg was moved out, and the literal `gapomin` when
    it was not (`:1956`) -- and **the two are the same expression**: substituting
    `calculate_r_tf_outboard_mid_unrippled`'s definition into the subtraction leaves
    exactly `gapomin`. Written unconditionally here, so the port has no branch where the
    source's own algebra has none. Verified numerically as well as algebraically; see the
    audit record's "deviations".

    Parameters
    ----------
    r_tf_outboard_mid :
        Radius to the centre of the outboard TF leg (m), post-ripple.
        `.build.r_tf_outboard_mid`.
    dr_tf_outboard, dr_vv_outboard :
        Outboard TF leg and vacuum vessel thickness (m). `.build.dr_tf_outboard`,
        `.build.dr_vv_outboard`.
    r_shld_outboard_outer :
        Radius to the outer edge of the outboard shield (m).
        `.build.r_shld_outboard_outer`.
    dr_shld_thermal_outboard, dr_tf_shld_gap, dr_shld_blkt_gap :
        Outboard thermal shield thickness, TF-thermal shield gap, vessel-blanket gap
        (m). `.build.dr_shld_thermal_outboard`, `.build.dr_tf_shld_gap`,
        `.build.dr_shld_blkt_gap`.

    Returns
    -------
    :
        `.build.dr_shld_vv_gap_outboard` (m).
    """
    return (
        r_tf_outboard_mid
        - 0.5 * dr_tf_outboard
        - dr_vv_outboard
        - r_shld_outboard_outer
        - dr_shld_thermal_outboard
        - dr_tf_shld_gap
        - dr_shld_blkt_gap
    )


# ---------------------------------------------------------------------------
# cottax nodes. One occupant class per switch value; see the module docstring's table
# for which value each answers and `_audit/units/models/build.md` for what is UNPORTED.
# ---------------------------------------------------------------------------


def calculate_vacuum_vessel_and_shield_radii(
    r_tf_inboard_out,
    dr_tf_shld_gap,
    dr_shld_thermal_inboard,
    dr_shld_vv_gap_inboard,
    dr_vv_inboard,
    dr_shld_inboard,
):
    """The inboard vacuum-vessel and neutronic-shield radii (m), accumulated outwards
    from the TF coil's outer edge.

    Ports `process/models/build.py:1833-1860`, the
    `i_tf_inside_cs == TF_OUTSIDE_CS` arm and the two assignments that follow it
    unconditionally. `r_sh_inboard_in` is `r_vv_inboard_out` -- PROCESS assigns one to
    the other at `:1855` and this port keeps both, because both names are read
    downstream and a node that owned only one would leave the other a boundary input
    with a silent duplicate for a producer.

    **`.build.r_vv_inboard_out` was the last non-finite condition in the cold tokamak
    SAND probe** (`_audit/next_steps.md` §16.3): a boundary input, zero cold, dividing
    in `vv_stress_on_quench`'s `tf_vv_frac = r_tf_inboard_out / r_vv_inboard_out`, which
    made c65 `nan` at the cold seed and nowhere else. **`.build.r_sh_inboard_out` is the
    read `blankets/hcpb.py`'s centrepost cluster declared and nothing produced**
    (that module's open note 3, and `consolidation_round_3.md` §4's last item): the same
    three lines close both.

    `rbld` is *not* computed here even though PROCESS writes it four lines later --
    see `calculate_rbld`, which reads the plasma minor radius and would otherwise give
    every output above a dependency on `.physics.rminor` that PROCESS does not give
    them. The source method is one straight line; its read sets are not, and this file
    already splits it that way (`calculate_r_shld_inboard_inner` is the same block,
    built inwards from the plasma).

    Parameters
    ----------
    r_tf_inboard_out :
        Outer radius of the inboard TF leg (m). `.build.r_tf_inboard_out`.
    dr_tf_shld_gap :
        TF-thermal shield gap (m). `.build.dr_tf_shld_gap`.
    dr_shld_thermal_inboard :
        Inboard thermal shield thickness (m). `.build.dr_shld_thermal_inboard`.
    dr_shld_vv_gap_inboard :
        Inboard thermal shield-vacuum vessel gap (m). `.build.dr_shld_vv_gap_inboard`.
    dr_vv_inboard :
        Inboard vacuum vessel thickness (m). `.build.dr_vv_inboard`.
    dr_shld_inboard :
        Inboard neutronic shield thickness (m). `.build.dr_shld_inboard`.

    Returns
    -------
    :
        `(.build.r_vv_inboard_out, .build.r_sh_inboard_in, .build.r_sh_inboard_out)`
        (m).
    """
    r_vv_inboard_out = (
        r_tf_inboard_out
        + dr_tf_shld_gap
        + dr_shld_thermal_inboard
        + dr_shld_vv_gap_inboard
        + dr_vv_inboard
    )
    r_sh_inboard_in = r_vv_inboard_out
    return r_vv_inboard_out, r_sh_inboard_in, r_sh_inboard_in + dr_shld_inboard


def calculate_rbld(
    r_sh_inboard_out,
    dr_shld_blkt_gap,
    dr_blkt_inboard,
    dr_fw_inboard,
    dr_fw_plasma_gap_inboard,
    rminor,
):
    """Radial build to the centre of the plasma (m). Ports
    `process/models/build.py:1862-1870`, unchanged.

    PROCESS's own comment is "should be equal to `rmajor`", and constraint 11 is the
    equation that says so. **That constraint is active on three of the four tracked
    tokamak files** (`low_aspect_ratio_DEMO`, `spherical_tokamak_eval`,
    `st_regression`; `large_tokamak_eval` is the one that omits it), which is why this
    is produced rather than left at the boundary as an unread accumulation.

    Its own node, not folded into `calculate_vacuum_vessel_and_shield_radii`, because
    it is the one line of that block that reads the plasma: a single node would hand
    `.build.r_vv_inboard_out` a dependency on `.physics.rminor` that PROCESS does not
    give it, and an invented edge is how a false cycle gets created.

    Parameters
    ----------
    r_sh_inboard_out :
        Plasma-facing radius of the inboard shield (m). `.build.r_sh_inboard_out`.
    dr_shld_blkt_gap :
        Shield-blanket gap (m). `.build.dr_shld_blkt_gap`.
    dr_blkt_inboard :
        Inboard blanket thickness (m). `.build.dr_blkt_inboard`.
    dr_fw_inboard :
        Inboard first wall thickness (m). `.build.dr_fw_inboard`.
    dr_fw_plasma_gap_inboard :
        Inboard plasma-first wall gap (m). `.build.dr_fw_plasma_gap_inboard`.
    rminor :
        Plasma minor radius (m). `.physics.rminor`.

    Returns
    -------
    :
        `.build.rbld` (m).
    """
    return (
        r_sh_inboard_out
        + dr_shld_blkt_gap
        + dr_blkt_inboard
        + dr_fw_inboard
        + dr_fw_plasma_gap_inboard
        + rminor
    )


class PlasmaXpointHeights(ExplicitFunction):
    """cottax node: `calculate_z_plasma_xpoint`. No switch."""

    z_plasma_xpoint_upper = OutputInto(build)
    z_plasma_xpoint_lower = OutputInto(build)

    def __call__(self, rminor=From(physics), kappa=From(physics)):
        return calculate_z_plasma_xpoint(rminor, kappa)


class DzBlktUpper(ExplicitFunction):
    """cottax node: `calculate_dz_blkt_upper`, owning `.build.dz_blkt_upper`. No switch.

    **Unswitched even though PROCESS's write sits under one.** `build.py:1650-1663`
    recomputes `dr_blkt_inboard`/`dr_blkt_outboard` from the sub-layer thicknesses when
    `.fwbs.blktmodel > 0`, and this line then runs regardless -- so `blktmodel` decides
    where the two *reads* come from, not whether this node owns its output or what
    formula it uses. `blktmodel > 0` has no producer for either read anywhere in this
    port (this module's own switch table: "no occupant -- `dr_blkt_*`/`dz_shld_upper`
    are run inputs"), which is a gap one level up, in that arm's absence, not a second
    occupant for this slot.
    """

    dz_blkt_upper = OutputInto(build)

    def __call__(self, dr_blkt_inboard=From(build), dr_blkt_outboard=From(build)):
        return calculate_dz_blkt_upper(dr_blkt_inboard, dr_blkt_outboard)


class DivertorGeometryConventional(ExplicitFunction):
    """cottax node: `calculate_divertor_geometry_conventional`. Answers
    `.physics.itart == 0`.

    **Also answers the input `.build.dz_xpoint_divertor < 1e-5`.** `process/models/
    build.py:800-801` assigns `dz_xpoint_divertor = divht` only when the input is
    effectively zero, and keeps the user's value otherwise -- so whether this node owns
    `.build.dz_xpoint_divertor` or that field is a plain input is a run-configuration
    fact, `conditional-ownership-by-run-config` again. `large_tokamak_eval.IN.DAT` does
    not set it, so it takes `build_variables.py:326`'s default `0.0` and this node owns
    it. `.build.rspo` is owned either way -- `divgeom` is called unconditionally at
    `:798` and writes it at `:912`.

    A note on faithfulness that matters and is easy to miss: in PROCESS the assignment is
    a **latch**, since after the first pipeline pass `dz_xpoint_divertor` is no longer
    below `1e-5`, so every later pass keeps the *first* pass's value. This node
    recomputes it every evaluation. On `large_tokamak_eval` the two are identical --
    every input to `calculate_divertor_geometry_conventional` is a run constant there
    (`ixc = [4, 6]` is temperature and density), and calling `Build.divgeom` on the
    converged `DataStructure` reproduces the stored `2.001883830794158` exactly. On a run
    where the plasma shape is an iteration variable they would differ, and PROCESS's own
    answer would be the stale one. Recorded in the audit record's "deviations".
    """

    dz_xpoint_divertor = OutputInto(build)
    rspo = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
        triang=From(physics),
        plsepi=From(build),
        plsepo=From(build),
        plleni=From(build),
        plleno=From(build),
        betai=From(divertor),
        betao=From(divertor),
    ):
        return calculate_divertor_geometry_conventional(
            rmajor,
            rminor,
            kappa,
            triang,
            plsepi,
            plsepo,
            plleni,
            plleno,
            betai,
            betao,
        )


class DivertorGeometrySphericalTokamak(ExplicitFunction):
    """cottax node: `calculate_divertor_geometry_spherical_tokamak`. Answers
    `.physics.itart == 1` **and** the input `.build.dz_xpoint_divertor < 1e-5`.

    The write-set is why this is a separate occupant and not a kwarg on the
    conventional node: `divgeom`'s early return at `process/models/build.py:863` never
    reaches the `.build.rspo` write at `:912`, so on a spherical tokamak nothing
    produces `rspo` -- the field keeps its `DataStructure` default (or the input file's
    value) and no ported tokamak node reads it. One owned field against the
    conventional arm's two: a different write-set, not a different formula for the
    same one.

    The `:800-801` latch gates this arm exactly as it gates the conventional one --
    `dz_xpoint_divertor = divht` only when the entering value is effectively zero. Both
    tracked spherical-tokamak inputs (`spherical_tokamak_eval.IN.DAT:91`,
    `st_regression.IN.DAT:1989`) set `dz_xpoint_divertor = 0.75`, so on both of them
    the `1.75 * rminor` is computed and *discarded* at `:800` and `divgeom` owns
    nothing at all. That configuration is the slot's `None` arm
    (`indat.py::_divertor_geometry_arm`, arm `-3`) -- absence rather than refusal, by
    `UNPORTED`'s own rule, because PROCESS itself computes nothing that survives. This
    node is therefore live only on a spherical-tokamak run that *leaves*
    `dz_xpoint_divertor` at its `0.0` default.
    """

    dz_xpoint_divertor = OutputInto(build)

    def __call__(self, rminor=From(physics)):
        return calculate_divertor_geometry_spherical_tokamak(rminor)


class ZTfInsideHalf(ExplicitFunction):
    """cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`. No
    switch.

    **The tokamak occupant of a field the stellarator also produces.**
    `models/stellarator/coils/calculate.py::ZTfInsideHalf` owns the same `VarPath` from
    `st_coil`'s formula (a coil-geometry expression in
    `.stellarator_config.stella_config_maximal_coil_height` and the coil minor radius);
    this one owns it from `build.py:807`'s vertical stack. Same name, same field,
    different device, different reads -- and never both in one graph, so there is no
    dual-ownership question to settle, only a correspondence to record. That stellarator
    node exists as a separate node at all for the same reason this one does: the field
    has two writers and which one wins was an ordering artefact until it was made
    structural.
    """

    z_tf_inside_half = OutputInto(build)

    def __call__(
        self,
        z_plasma_xpoint_upper=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_shld_lower=From(build),
        dz_vv_lower=From(build),
        dz_shld_vv_gap=From(build),
        dz_shld_thermal=From(build),
        dr_tf_shld_gap=From(build),
    ):
        return calculate_z_tf_inside_half(
            z_plasma_xpoint_upper,
            dz_xpoint_divertor,
            dz_divertor,
            dz_shld_lower,
            dz_vv_lower,
            dz_shld_vv_gap,
            dz_shld_thermal,
            dr_tf_shld_gap,
        )


class TfTopHeight(ExplicitFunction):
    """The family that owns `.build.z_tf_top` and
    `.build.dz_tf_upper_lower_midplane`. Switched on `.physics.i_single_null`.

    Two arms, both written, and they own the **same two fields** -- which is what makes
    this a genuine two-armed slot rather than two nodes (`configuration.py`'s
    exclusivity rule). They read very different things: the double-null arm reads two
    fields and the single-null arm thirteen, because a symmetric machine can reflect the
    lower build while a single-null one has to stack the upper one from scratch.

    `.physics.i_single_null` is already answered once in `indat.machine_from_indat` --
    `_n_divertors` derives `.divertor.n_divertors` from it there -- so this slot costs no
    new switch read, only a new registry.
    """


class TfTopHeightSingleNull(TfTopHeight):
    """cottax node: `calculate_tf_top_height_single_null`. Answers
    `.physics.i_single_null == 1`, the arm `large_tokamak_eval`/`_nof` and
    `low_aspect_ratio_DEMO` take.
    """

    z_tf_top = OutputInto(build)
    dz_tf_upper_lower_midplane = OutputInto(build)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
        dr_tf_shld_gap=From(build),
        dz_shld_thermal=From(build),
        dz_shld_vv_gap=From(build),
        dz_vv_upper=From(build),
        dz_shld_upper=From(build),
        dr_shld_blkt_gap=From(build),
        dz_blkt_upper=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
        dz_fw_plasma_gap=From(build),
        z_plasma_xpoint_upper=From(build),
    ):
        return calculate_tf_top_height_single_null(
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
        )


class TfTopHeightDoubleNull(TfTopHeight):
    """cottax node: `calculate_tf_top_height_double_null`. Answers
    `.physics.i_single_null == 0`.

    **Written and tested but not yet reachable through `machine_from_indat`**: the two
    tracked inputs that set `i_single_null = 0` (`spherical_tokamak_eval.IN.DAT:292`,
    `st_regression.IN.DAT:638`) are refused earlier for an unrelated reason
    (`i_tf_turn_type == 2`, the CroCo turn). Written anyway rather than left `UNPORTED`,
    because it is four lines of the same source block and refusing it would make this
    slot the only one in `Build` whose refusal is about a switch it does not read.

    Owns `.build.dz_tf_upper_lower_midplane` even though the source assigns it a
    literal `0.0e0`: the field is read (`models/pfcoil/geometry.py` offsets the lower
    divertor coils by it) and a constant is still a producer. Leaving it unowned here
    would make the two arms' write-sets differ and turn an ordinary switch into a
    partial-overlap orphan -- exactly the hazard `boundary.orphaned_by` exists to catch.
    """

    z_tf_top = OutputInto(build)
    dz_tf_upper_lower_midplane = OutputInto(build)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
    ):
        return calculate_tf_top_height_double_null(z_tf_inside_half, dr_tf_inboard)


class BlktUpperThickness(ExplicitFunction):
    """cottax node: `calculate_dz_blkt_upper`. No switch -- `process/models/build.py:
    1664-1667` sits below the `.fwbs.blktmodel` branch and runs on every configuration.
    """

    dz_blkt_upper = OutputInto(build)

    def __call__(
        self,
        dr_blkt_inboard=From(build),
        dr_blkt_outboard=From(build),
    ):
        return calculate_dz_blkt_upper(dr_blkt_inboard, dr_blkt_outboard)


class DrTfInboardFromWindingPack(ExplicitFunction):
    """cottax node: `calculate_dr_tf_inboard`. Answers `140 in numerics.ixc`.

    **Not the occupant on `large_tokamak_eval.IN.DAT`**, whose `ixc` is `[4, 6]`. Written
    because this is the only way `.build.dr_tf_inboard` -- one of the six variables
    `tokamak_boundary.md` attributes to this slot -- is ever produced. On the reference
    run its sibling `DrTfWpWithInsulationFromInboardBuild` is the occupant instead and
    `.build.dr_tf_inboard` is a boundary input.
    """

    dr_tf_inboard = OutputInto(build)

    def __call__(
        self,
        dr_tf_wp_with_insulation=From(tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        dr_tf_nose_case=From(tfcoil),
    ):
        return calculate_dr_tf_inboard(
            dr_tf_wp_with_insulation, dr_tf_plasma_case, dr_tf_nose_case
        )


class DrTfWpWithInsulationFromInboardBuild(ExplicitFunction):
    """cottax node: `calculate_dr_tf_wp_with_insulation`. Answers
    `140 not in numerics.ixc` -- the live arm on `large_tokamak_eval.IN.DAT`.
    """

    dr_tf_wp_with_insulation = OutputInto(tfcoil)

    def __call__(
        self,
        dr_tf_inboard=From(build),
        dr_tf_plasma_case=From(tfcoil),
        dr_tf_nose_case=From(tfcoil),
    ):
        return calculate_dr_tf_wp_with_insulation(
            dr_tf_inboard, dr_tf_plasma_case, dr_tf_nose_case
        )


class TfInboardRadiiTfOutsideCs(ExplicitFunction):
    """cottax node: `calculate_r_tf_inboard_radii_tf_outside_cs`. Answers the joint
    key `(.build.i_tf_inside_cs, .build.i_cs_precomp) == (TF_OUTSIDE_CS,
    CS_PRECOMPRESSION_STRUCTURE_PRESENT)` -- both defaults, both live. `TF_INSIDE_CS`
    is UNPORTED (its arm computes `r_tf_inboard_in` from `dr_bore` alone and
    `dr_cs_bore` from the TF thickness -- a genuinely different reads-set); the
    no-precompression arm (`dr_cs_precomp = 0.0`, reading none of
    `fseppc`/`fcspc`/`sigallpc`) is `TfInboardRadiiNoCsPrecomp` (2026-08-27).
    """

    dr_cs_bore = OutputInto(build)
    dr_cs_precomp = OutputInto(build)
    r_tf_inboard_in = OutputInto(build)
    r_tf_inboard_mid = OutputInto(build)
    r_tf_inboard_out = OutputInto(build)

    def __call__(
        self,
        dr_bore=From(build),
        dr_cs=From(build),
        fseppc=From(build),
        fcspc=From(build),
        sigallpc=From(build),
        dr_cs_tf_gap=From(build),
        dr_tf_inboard=From(build),
    ):
        return calculate_r_tf_inboard_radii_tf_outside_cs(
            dr_bore,
            dr_cs,
            fseppc,
            fcspc,
            sigallpc,
            dr_cs_tf_gap,
            dr_tf_inboard,
        )


class TfInboardRadiiNoCsPrecomp(ExplicitFunction):
    """cottax node: `calculate_r_tf_inboard_radii_no_cs_precomp`. Answers the joint
    key `(.build.i_tf_inside_cs, .build.i_cs_precomp) == (TF_OUTSIDE_CS, 0)` -- the
    live cell on both tracked spherical-tokamak files (2026-08-27, ST frontier wave).
    Same write-set as `TfInboardRadiiTfOutsideCs`; a strict-subset reads-set
    (`fseppc`/`fcspc`/`sigallpc` never read, `dr_cs_precomp` produced as the exact
    zero PROCESS writes at `build.py:1714`).
    """

    dr_cs_bore = OutputInto(build)
    dr_cs_precomp = OutputInto(build)
    r_tf_inboard_in = OutputInto(build)
    r_tf_inboard_mid = OutputInto(build)
    r_tf_inboard_out = OutputInto(build)

    def __call__(
        self,
        dr_bore=From(build),
        dr_cs=From(build),
        dr_cs_tf_gap=From(build),
        dr_tf_inboard=From(build),
    ):
        return calculate_r_tf_inboard_radii_no_cs_precomp(
            dr_bore,
            dr_cs,
            dr_cs_tf_gap,
            dr_tf_inboard,
        )


class ShldInboardInnerRadius(ExplicitFunction):
    """cottax node: `calculate_r_shld_inboard_inner`. No switch."""

    r_shld_inboard_inner = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_shld_inboard=From(build),
    ):
        return calculate_r_shld_inboard_inner(
            rmajor,
            rminor,
            dr_fw_plasma_gap_inboard,
            dr_fw_inboard,
            dr_blkt_inboard,
            dr_shld_inboard,
        )


class ShldOutboardOuterRadius(ExplicitFunction):
    """cottax node: `calculate_r_shld_outboard_outer`. No switch."""

    r_shld_outboard_outer = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dr_blkt_outboard=From(build),
        dr_shld_outboard=From(build),
    ):
        return calculate_r_shld_outboard_outer(
            rmajor,
            rminor,
            dr_fw_plasma_gap_outboard,
            dr_fw_outboard,
            dr_blkt_outboard,
            dr_shld_outboard,
        )


class DrTfOutboardSuperconducting(ExplicitFunction):
    """cottax node: `calculate_dr_tf_outboard_superconducting`. Answers
    `.tfcoil.i_tf_sup == 1`.

    The `i_tf_sup != 1` arm reads `.build.f_dr_tf_outboard_inboard` as well and is
    UNPORTED; declaring that read here instead of splitting would be an invented edge on
    the superconducting run, which never reads it.
    """

    dr_tf_outboard = OutputInto(build)

    def __call__(self, dr_tf_inboard=From(build)):
        return calculate_dr_tf_outboard_superconducting(dr_tf_inboard)


class WpConductorMaxWidthSuperconducting(ExplicitFunction):
    """cottax node: `calculate_dx_tf_wp_conductor_max_superconducting`. Answers
    `.tfcoil.i_tf_sup == 1`.

    Owns the mint `.tfcoil.dx_tf_wp_conductor_max`. The resistive arm
    (`process/models/build.py:1577-1579`) computes the same quantity from
    `.superconducting_tfcoil.r_tf_wp_inboard_outer` and `.tfcoil.n_tf_coils` -- a
    disjoint reads-set -- and is UNPORTED.
    """

    dx_tf_wp_conductor_max = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_wp_primary_toroidal=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return calculate_dx_tf_wp_conductor_max_superconducting(
            dx_tf_wp_primary_toroidal, dx_tf_wp_insulation, dx_tf_wp_insertion_gap
        )


class TfOutboardMidUnrippled(ExplicitFunction):
    """cottax node: `calculate_r_tf_outboard_mid_unrippled`. No switch.

    Owns the mint `.build.r_tf_outboard_mid_unrippled` -- the value PROCESS assigns to
    `.build.r_tf_outboard_mid` at `:1901` and then overwrites in place at `:1939`.
    Separating the two is what keeps `.build.r_tf_outboard_mid` from being a field a node
    both reads and owns.
    """

    r_tf_outboard_mid_unrippled = OutputInto(build)

    def __call__(
        self,
        r_shld_outboard_outer=From(build),
        dr_shld_blkt_gap=From(build),
        dr_vv_outboard=From(build),
        gapomin=From(build),
        dr_shld_thermal_outboard=From(build),
        dr_tf_shld_gap=From(build),
        dr_tf_outboard=From(build),
    ):
        return calculate_r_tf_outboard_mid_unrippled(
            r_shld_outboard_outer,
            dr_shld_blkt_gap,
            dr_vv_outboard,
            gapomin,
            dr_shld_thermal_outboard,
            dr_tf_shld_gap,
            dr_tf_outboard,
        )


class TfOutboardMidDShape(ExplicitFunction):
    """cottax node: the ripple constraint on the outboard TF leg. Answers
    `.tfcoil.i_tf_shape == 1` (D-shape).

    Composes `plasma_outboard_edge_toroidal_ripple_fitted` and
    `calculate_r_tf_outboard_mid`, which is exactly what
    `process/models/build.py:1916-1956` does: call the ripple fit at the stacked-up
    radius, take the larger of that radius and the fit's minimum. It does **not** own
    `.tfcoil.ripple_b_tf_plasma_edge`, because PROCESS's own answer for that field comes
    from its *second* call to the fit, at the final radius -- see
    `TfOutboardEdgeRipple` below, which is that second call.

    `i_tf_shape == 2` (picture frame) is `TfOutboardMidPictureFrame` below -- a
    different formula reading neither the winding pack nor `c1`/`c2`. `0` (auto-select)
    is a meta-value `init.py` resolves before any model runs; `indat._tf_shape` answers
    it and it names no occupant.
    """

    r_tf_outboard_mid = OutputInto(build)

    def __call__(
        self,
        r_tf_outboard_mid_unrippled=From(build),
        ripple_b_tf_plasma_edge_max=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        rminor=From(physics),
        dx_tf_wp_conductor_max=From(tfcoil),
    ):
        _, r_tf_outboard_midmin = plasma_outboard_edge_toroidal_ripple_fitted(
            ripple_b_tf_plasma_edge_max,
            r_tf_outboard_mid_unrippled,
            n_tf_coils,
            rmajor,
            rminor,
            dx_tf_wp_conductor_max,
        )
        return calculate_r_tf_outboard_mid(
            r_tf_outboard_mid_unrippled, r_tf_outboard_midmin
        )


class TfOutboardEdgeRipple(ExplicitFunction):
    """cottax node: `plasma_outboard_edge_toroidal_ripple_fitted`, evaluated at the final
    leg radius. Answers `.tfcoil.i_tf_shape == 1` (D-shape).

    This is `process/models/build.py:1958-1977`, the source's **second** call to the
    ripple fit -- the one whose `ripple` lands in `.tfcoil.ripple_b_tf_plasma_edge` and
    survives into the converged answer. The first call's `ripple` (`:1916-1935`) is
    overwritten unconditionally and is a `redundant-duplicate-write`; only its
    `r_tf_outboard_midmin` is load-bearing, and that is what `TfOutboardMidDShape` above
    keeps.

    `r_tf_outboard_midmin` is returned by the pure function but is not owned here: it is
    identical to the one `TfOutboardMidDShape` already used (it does not depend on
    `r_tf_outboard_mid`), so binding it a second time would mint two names for one value.
    """

    ripple_b_tf_plasma_edge = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        ripple_b_tf_plasma_edge_max=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        rminor=From(physics),
        dx_tf_wp_conductor_max=From(tfcoil),
    ):
        ripple_b_tf_plasma_edge, _ = plasma_outboard_edge_toroidal_ripple_fitted(
            ripple_b_tf_plasma_edge_max,
            r_tf_outboard_mid,
            n_tf_coils,
            rmajor,
            rminor,
            dx_tf_wp_conductor_max,
        )
        return ripple_b_tf_plasma_edge


class TfOutboardMidPictureFrame(ExplicitFunction):
    """cottax node: the ripple constraint on the outboard TF leg. Answers
    `.tfcoil.i_tf_shape == 2` (picture frame), the shape both tracked spherical-tokamak
    inputs select explicitly (`spherical_tokamak_eval.IN.DAT:357`,
    `st_regression.IN.DAT:803`) and the one `init.py:728-729` would resolve `0` to on
    `itart == 1` anyway.

    Same composition as `TfOutboardMidDShape` -- the shell around the ripple call
    (`process/models/build.py:1916-1956`) is shape-agnostic, only the fit inside
    branches -- with `plasma_outboard_edge_toroidal_ripple_picture_frame` in place of
    the fitted correlation. The reads-set drops `dx_tf_wp_conductor_max`: the
    picture-frame formula never touches the winding pack, which is exactly why this is
    a different occupant and not a kwarg of the D-shape one.
    """

    r_tf_outboard_mid = OutputInto(build)

    def __call__(
        self,
        r_tf_outboard_mid_unrippled=From(build),
        ripple_b_tf_plasma_edge_max=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        _, r_tf_outboard_midmin = plasma_outboard_edge_toroidal_ripple_picture_frame(
            ripple_b_tf_plasma_edge_max,
            r_tf_outboard_mid_unrippled,
            n_tf_coils,
            rmajor,
            rminor,
        )
        return calculate_r_tf_outboard_mid(
            r_tf_outboard_mid_unrippled, r_tf_outboard_midmin
        )


class TfOutboardEdgeRipplePictureFrame(ExplicitFunction):
    """cottax node: `plasma_outboard_edge_toroidal_ripple_picture_frame`, evaluated at
    the final leg radius. Answers `.tfcoil.i_tf_shape == 2` (picture frame).

    The picture-frame occupant of the second-ripple-call slot -- `process/models/
    build.py:1958-1977` is unconditional, so the picture frame fills it too, and the
    same `redundant-duplicate-write` reasoning as `TfOutboardEdgeRipple` applies: only
    the second call's `ripple` survives into `.tfcoil.ripple_b_tf_plasma_edge`, and
    `r_tf_outboard_midmin` is not re-owned because it is the same value
    `TfOutboardMidPictureFrame` already used.

    `ripple_b_tf_plasma_edge_max` is read for the same reason the D-shape sibling reads
    it: the shared pure function computes both outputs at once. The ripple itself
    depends only on the radius ratio and `n_tf_coils` (`:1585`).
    """

    ripple_b_tf_plasma_edge = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        ripple_b_tf_plasma_edge_max=From(tfcoil),
        n_tf_coils=From(tfcoil),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        ripple_b_tf_plasma_edge, _ = plasma_outboard_edge_toroidal_ripple_picture_frame(
            ripple_b_tf_plasma_edge_max,
            r_tf_outboard_mid,
            n_tf_coils,
            rmajor,
            rminor,
        )
        return ripple_b_tf_plasma_edge


class ShldVvGapOutboard(ExplicitFunction):
    """cottax node: `calculate_dr_shld_vv_gap_outboard`. No switch -- see that
    function's docstring for why the source's two arms are one expression.
    """

    dr_shld_vv_gap_outboard = OutputInto(build)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
        dr_vv_outboard=From(build),
        r_shld_outboard_outer=From(build),
        dr_shld_thermal_outboard=From(build),
        dr_tf_shld_gap=From(build),
        dr_shld_blkt_gap=From(build),
    ):
        return calculate_dr_shld_vv_gap_outboard(
            r_tf_outboard_mid,
            dr_tf_outboard,
            dr_vv_outboard,
            r_shld_outboard_outer,
            dr_shld_thermal_outboard,
            dr_tf_shld_gap,
            dr_shld_blkt_gap,
        )


class TfInnerBore(ExplicitFunction):
    """cottax node: `calculate_dr_tf_inner_bore`. No switch -- both of PROCESS's writes
    (`:1911`, `:1949`) are the same expression and neither is guarded by a switch, only
    by the ripple test that `calculate_r_tf_outboard_mid` already resolves.

    Reads `.build.r_tf_outboard_mid` (post-ripple), so it sits downstream of
    `tf_outboard_mid` in the same way `ShldVvGapOutboard` does.
    """

    dr_tf_inner_bore = OutputInto(build)

    def __call__(
        self,
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
        r_tf_inboard_mid=From(build),
        dr_tf_inboard=From(build),
    ):
        return calculate_dr_tf_inner_bore(
            r_tf_outboard_mid, dr_tf_outboard, r_tf_inboard_mid, dr_tf_inboard
        )


class VacuumVesselAndShieldRadiiTfOutsideCs(ExplicitFunction):
    """cottax node: `calculate_vacuum_vessel_and_shield_radii`. Answers
    `.build.i_tf_inside_cs == TF_OUTSIDE_CS`, the live arm on every tracked file.

    `TF_INSIDE_CS` is UNPORTED and the refusal is the same one
    `TfInboardRadiiTfOutsideCs` already carries for the neighbouring slot: that arm
    accumulates three further central-solenoid thicknesses (`dr_cs`, `dr_cs_tf_gap`,
    `dr_cs_precomp`) into the same radius, a genuinely different reads-set.

    **Keyed on `i_tf_inside_cs` alone**, not on the joint
    `(i_tf_inside_cs, i_cs_precomp)` key `tf_inboard_radii_arm` uses. The switch is
    asked twice on purpose: this block's arm does not depend on whether the CS carries
    precompression structure, and reusing the joint answer would make it look as though
    it did. Same reasoning as `pfcoil/superconductor.py`'s two-slot split.
    """

    r_vv_inboard_out = OutputInto(build)
    r_sh_inboard_in = OutputInto(build)
    r_sh_inboard_out = OutputInto(build)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        dr_tf_shld_gap=From(build),
        dr_shld_thermal_inboard=From(build),
        dr_shld_vv_gap_inboard=From(build),
        dr_vv_inboard=From(build),
        dr_shld_inboard=From(build),
    ):
        return calculate_vacuum_vessel_and_shield_radii(
            r_tf_inboard_out,
            dr_tf_shld_gap,
            dr_shld_thermal_inboard,
            dr_shld_vv_gap_inboard,
            dr_vv_inboard,
            dr_shld_inboard,
        )


class RadialBuildToPlasmaCentre(ExplicitFunction):
    """cottax node: `calculate_rbld`. No switch -- PROCESS writes it below the
    `i_tf_inside_cs` branch, on both arms.
    """

    rbld = OutputInto(build)

    def __call__(
        self,
        r_sh_inboard_out=From(build),
        dr_shld_blkt_gap=From(build),
        dr_blkt_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        rminor=From(physics),
    ):
        return calculate_rbld(
            r_sh_inboard_out,
            dr_shld_blkt_gap,
            dr_blkt_inboard,
            dr_fw_inboard,
            dr_fw_plasma_gap_inboard,
            rminor,
        )
