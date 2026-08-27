"""Pure-functional port of `process/models/blankets/blanket_library.py`'s tokamak
`component_volumes` chain.

**Why this file exists at all.** `blankets/blanket_library.py` is one of the three files
`tokamak_call_surface.md` §A found *reached with no `.run()` in `caller.py`*:
`models.blanket_library` is constructed at `main.py:678` and never called, and the file
runs only because `CCFE_HCPB(OutboardBlanket, InboardBlanket)` (`hcpb.py:25`) inherits
from `BlanketLibrary` (`blanket_library.py:56`). Fourteen of its functions are entered on
the reference tokamak run; this port covers the **four** of them that lie on the minimal
closure producing `.tokamak.ccfe_hcpb`'s boundary variables -- everything downstream of
`.fwbs.vol_blkt_total`, which `CCFE_HCPB.component_masses` needs and nothing else in the
tokamak surface produces.

The other ten entered functions (`set_blanket_module_geometry`,
`pipe_hydraulic_diameter`, the four poloidal-segment/module-geometry helpers, the two
poloidal-plasma-angle helpers) are **deliberately out of scope**: measured, every one of
their writes lands in `.blanket.*` or in `.fwbs.b_bz_liq`/`a_bz_liq`/
`radius_blkt_channel*`, and none of those reaches any of the sixteen variables
`_audit/tokamak_boundary.md` §`.tokamak.ccfe_hcpb` lists. See
`_audit/units/models/blankets/blanket_library.md` (read it first) for the evidence table.

**Switches.** `component_volumes` (`blanket_library.py:91-94`) chooses D-shaped vs
elliptical blanket geometry on `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`; the
reference run has `itart = 0` and `i_fw_blkt_vv_shape = 2` (`ELLIPTICAL_SHAPED`), so only
the elliptical arm is ported. `calculate_blkt_half_height` and `apply_coverage_factors`
branch on `n_divertors == 2`; the reference run has `n_divertors = 1`. Every unported arm
is named as UNPORTED in the audit record rather than folded into a `jnp.where` -- the
union-of-arms reads is the invented-edge defect this port exists to remove
(`next_steps.md` §14.2).

2026-08-27 (the double-null wave, for the two spherical-tokamak input files that set
`i_single_null = 0`): **both** `n_divertors` slots are now total -- the double-null arms
of the half-height and of the coverage factors are written beside their single-null
siblings, each a separate occupant with its own reads-set. The half-height's two arms
differ by five reads and the coverage factors' by a literal, and per `next_steps.md`
§14.2 (the `istore` precedent) a literal is enough: a switch value selects an occupant.
`n_divertors` is still a parameter of nothing.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import blanket, build, divertor, fwbs, physics


def _eshellarea(rshell, rmini, rmino, zminor):
    """Inboard/outboard/total surface area of a two-ellipse toroidal shell.

    Ports `process/models/engineering/ivc_functions.py:99-130` verbatim.

    **Filed here rather than in a `models/engineering/ivc_functions.py` port**, and
    module-private to say so. `ivc_functions.py` is not a `Model` at all -- three plain
    module functions imported by `fw.py:14-16`, `shield.py:12-13` and `vacuum.py:13`
    (`tokamak_call_surface.md` §A) -- so it has no slot in the model tree and no natural
    owner among the units being ported in this wave. Duplicating the twelve lines here is
    the smaller debt than three agents each half-owning a shared module; the
    consolidation pass should lift `_eshellarea`/`_eshellvol` into one
    `functional_process/models/engineering/ivc_functions.py` and have this file import
    them.

    Parameters
    ----------
    rshell :
        Major radius of the centre of both ellipses (m).
    rmini :
        Horizontal distance from `rshell` to the inboard elliptical shell (m).
    rmino :
        Horizontal distance from `rshell` to the outboard elliptical shell (m).
    zminor :
        Vertical internal half-height of the shell (m).

    Returns
    -------
    tuple
        `(ain, aout, ain + aout)` -- inboard, outboard and total surface area (m2).
    """
    elong_inboard = zminor / rmini
    ain = 2.0 * jnp.pi * elong_inboard * (jnp.pi * rshell * rmini - 2.0 * rmini * rmini)

    elong_outboard = zminor / rmino
    aout = (
        2.0 * jnp.pi * elong_outboard * (jnp.pi * rshell * rmino + 2.0 * rmino * rmino)
    )

    return ain, aout, ain + aout


def _eshellvol(rshell, rmini, rmino, zminor, drin, drout, dz):
    """Inboard/outboard/total volume of a two-ellipse toroidal shell.

    Ports `process/models/engineering/ivc_functions.py:170-246` verbatim. Same filing
    note as `_eshellarea` above.

    Parameters
    ----------
    rshell :
        Major radius of the centre of both ellipses (m).
    rmini :
        Horizontal distance from `rshell` to the outer edge of the inboard shell (m).
    rmino :
        Horizontal distance from `rshell` to the inner edge of the outboard shell (m).
    zminor :
        Vertical internal half-height of the shell (m).
    drin, drout :
        Horizontal thickness of the inboard/outboard shell at the midplane (m).
    dz :
        Vertical thickness of the shell at top/bottom (m).

    Returns
    -------
    tuple
        `(vin, vout, vin + vout)` -- inboard, outboard and total volume (m3).
    """
    # Inboard section: outer (higher R) surface minus inner (lower R) surface.
    a = rmini
    b = zminor
    v1 = 2.0 * jnp.pi * (b / a) * (0.5 * jnp.pi * rshell * a**2 - (2.0 / 3.0) * a**3)

    a = rmini + drin
    b = zminor + dz
    v2 = 2.0 * jnp.pi * (b / a) * (0.5 * jnp.pi * rshell * a**2 - (2.0 / 3.0) * a**3)

    vin = v2 - v1

    # Outboard section: the same difference with the sign of the cubic term flipped.
    a = rmino
    b = zminor
    v1 = 2.0 * jnp.pi * (b / a) * (0.5 * jnp.pi * rshell * a**2 + (2.0 / 3.0) * a**3)

    a = rmino + drout
    b = zminor + dz
    v2 = 2.0 * jnp.pi * (b / a) * (0.5 * jnp.pi * rshell * a**2 + (2.0 / 3.0) * a**3)

    vout = v2 - v1

    return vin, vout, vin + vout


def calculate_blkt_half_height_single_null(
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
    """Blanket half-height, single-null arm (`n_divertors == 1`).

    Ports the `else` arm of `blanket_library.py:220-229`'s `if n_divertors == 2`. The
    double-null arm (`z_top = z_bottom`) is `calculate_blkt_half_height_double_null`
    below. Note the arms genuinely differ in *reads*: the double-null arm needs neither
    `z_plasma_xpoint_upper` nor any of the four gap/first-wall thicknesses, so folding
    them into one `jnp.where` would declare five edges a double-null machine does not
    have.

    `n_divertors` is therefore **not** a parameter of this function. That is the point:
    the switch chose the occupant, so it is gone from the body.

    Parameters
    ----------
    z_plasma_xpoint_lower :
        Lower vertical position of the plasma X-point (m).
        `.build.z_plasma_xpoint_lower`.
    dz_xpoint_divertor :
        Vertical distance from X-point to divertor (m). `.build.dz_xpoint_divertor`.
    dz_divertor :
        Vertical thickness of the divertor (m). `.divertor.dz_divertor`.
    dz_blkt_upper :
        Vertical thickness of the upper blanket (m). `.build.dz_blkt_upper`.
    z_plasma_xpoint_upper :
        Upper vertical position of the plasma X-point (m).
        `.build.z_plasma_xpoint_upper`.
    dr_fw_plasma_gap_inboard, dr_fw_plasma_gap_outboard :
        First-wall/plasma radial gaps (m). `.build.dr_fw_plasma_gap_inboard`/`_outboard`.
    dr_fw_inboard, dr_fw_outboard :
        First-wall radial thicknesses (m). `.build.dr_fw_inboard`/`_outboard`.

    Returns
    -------
    :
        Blanket half-height `dz_blkt_half` (m). `.blanket.dz_blkt_half`.
    """
    z_bottom = z_plasma_xpoint_lower + dz_xpoint_divertor + dz_divertor - dz_blkt_upper

    z_top = z_plasma_xpoint_upper + 0.5 * (
        dr_fw_plasma_gap_inboard
        + dr_fw_plasma_gap_outboard
        + dr_fw_inboard
        + dr_fw_outboard
    )

    return 0.5 * (z_top + z_bottom)


def calculate_blkt_half_height_double_null(
    z_plasma_xpoint_lower,
    dz_xpoint_divertor,
    dz_divertor,
    dz_blkt_upper,
):
    """Blanket half-height, double-null arm (`n_divertors == 2`).

    Ports the `if n_divertors == 2` arm of `blanket_library.py:220-229`. A double-null
    machine is vertically symmetric, so PROCESS sets `z_top = z_bottom` and the
    `0.5 * (z_top + z_bottom)` that follows collapses to `z_bottom` itself -- written
    out reduced, as `shield.py`'s `calculate_shield_half_height_double_null` does for
    the identical branch. (The reduction is exact in floating point, not merely
    algebraic: `x + x` is representable and `0.5 * (x + x)` is `x` to the bit.)

    **Five fewer reads than the single-null sibling**, which is why this is a second
    occupant rather than a `jnp.where` over one: `z_plasma_xpoint_upper` and the four
    gap/first-wall thicknesses are not read by a double-null machine at all, and
    declaring them would invent five edges.

    Parameters
    ----------
    z_plasma_xpoint_lower :
        Lower vertical position of the plasma X-point (m).
        `.build.z_plasma_xpoint_lower`.
    dz_xpoint_divertor :
        Vertical distance from X-point to divertor (m). `.build.dz_xpoint_divertor`.
    dz_divertor :
        Vertical thickness of the divertor (m). `.divertor.dz_divertor`.
    dz_blkt_upper :
        Vertical thickness of the upper blanket (m). `.build.dz_blkt_upper`.

    Returns
    -------
    :
        Blanket half-height `dz_blkt_half` (m). `.blanket.dz_blkt_half`.
    """
    return z_plasma_xpoint_lower + dz_xpoint_divertor + dz_divertor - dz_blkt_upper


def calculate_elliptical_blkt_areas(
    rmajor,
    rminor,
    triang,
    r_shld_inboard_inner,
    dr_shld_inboard,
    dr_blkt_inboard,
    r_shld_outboard_outer,
    dr_shld_outboard,
    dr_blkt_outboard,
    dz_blkt_half,
):
    """Full-coverage blanket surface areas, elliptical arm.

    Ports the `@staticmethod` `calculate_elliptical_blkt_areas`
    (`blanket_library.py:381-449`) verbatim -- already pure in the source, no `self` to
    close.

    Parameters
    ----------
    rmajor, rminor, triang :
        Plasma major radius (m), minor radius (m), triangularity. `.physics.rmajor`,
        `.physics.rminor`, `.physics.triang`.
    r_shld_inboard_inner :
        Inner radius of the inboard shield (m). `.build.r_shld_inboard_inner`.
    dr_shld_inboard, dr_blkt_inboard :
        Inboard shield/blanket radial thicknesses (m). `.build.dr_shld_inboard`,
        `.build.dr_blkt_inboard`.
    r_shld_outboard_outer :
        Outer radius of the outboard shield (m). `.build.r_shld_outboard_outer`.
    dr_shld_outboard, dr_blkt_outboard :
        Outboard shield/blanket radial thicknesses (m). `.build.dr_shld_outboard`,
        `.build.dr_blkt_outboard`.
    dz_blkt_half :
        Blanket half-height (m). `.blanket.dz_blkt_half`.

    Returns
    -------
    tuple
        `(a_blkt_inboard_surface_full_coverage, a_blkt_outboard_surface_full_coverage,
        a_blkt_total_surface_full_coverage)` (m2).
    """
    # Major radius to the centre of both ellipses -- coincident in radius with the top
    # of the plasma.
    r1 = rmajor - rminor * triang

    r2 = r1 - r_shld_inboard_inner - dr_shld_inboard - dr_blkt_inboard
    r3 = r_shld_outboard_outer - r1 - dr_shld_outboard - dr_blkt_outboard

    return _eshellarea(r1, r2, r3, dz_blkt_half)


def calculate_elliptical_blkt_volumes(
    rmajor,
    rminor,
    triang,
    r_shld_inboard_inner,
    dr_shld_inboard,
    dr_blkt_inboard,
    r_shld_outboard_outer,
    dr_shld_outboard,
    dr_blkt_outboard,
    dz_blkt_half,
    dz_blkt_upper,
):
    """Full-coverage blanket volumes, elliptical arm.

    Ports the `@staticmethod` `calculate_elliptical_blkt_volumes`
    (`blanket_library.py:451-530`) verbatim -- already pure in the source.

    Parameters
    ----------
    rmajor, rminor, triang, r_shld_inboard_inner, dr_shld_inboard, dr_blkt_inboard,
    r_shld_outboard_outer, dr_shld_outboard, dr_blkt_outboard, dz_blkt_half :
        As `calculate_elliptical_blkt_areas` above.
    dz_blkt_upper :
        Vertical thickness of the upper blanket (m). `.build.dz_blkt_upper`.

    Returns
    -------
    tuple
        `(vol_blkt_inboard_full_coverage, vol_blkt_outboard_full_coverage,
        vol_blkt_total_full_coverage)` (m3).
    """
    r1 = rmajor - rminor * triang

    r2 = r1 - r_shld_inboard_inner - dr_shld_inboard - dr_blkt_inboard
    r3 = r_shld_outboard_outer - r1 - dr_shld_outboard - dr_blkt_outboard

    return _eshellvol(
        rshell=r1,
        rmini=r2,
        rmino=r3,
        zminor=dz_blkt_half,
        drin=dr_blkt_inboard,
        drout=dr_blkt_outboard,
        dz=dz_blkt_upper,
    )


def apply_coverage_factors_single_null(
    a_blkt_total_surface_full_coverage,
    a_blkt_inboard_surface_full_coverage,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    vol_blkt_total_full_coverage,
    vol_blkt_inboard_full_coverage,
):
    """Blanket areas and volumes after divertor/HCD coverage, single-null arm.

    Ports the `n_divertors != 2` arm of `apply_coverage_factors`
    (`blanket_library.py:532-584`), closing its `self.data` back-door. The double-null
    arm differs only in a literal (`2.0 * f_ster_div_single` in place of
    `f_ster_div_single`, `blanket_library.py:544`) and is
    `apply_coverage_factors_double_null` below: under `next_steps.md` §14.2 a switch
    value selects an occupant even when the arms differ only in a literal -- the
    `istore` precedent -- so it is a second class, not a `jnp.where`.

    Parameters
    ----------
    a_blkt_total_surface_full_coverage, a_blkt_inboard_surface_full_coverage :
        Blanket surface areas at 100 % coverage (m2).
        `.build.a_blkt_total_surface_full_coverage`,
        `.build.a_blkt_inboard_surface_full_coverage` -- `EllipticalBlanketAreas`'s
        outputs.
    f_ster_div_single :
        Divertor solid-angle fraction per divertor. `.fwbs.f_ster_div_single`
        (`divertor.py:42`).
    f_a_fw_outboard_hcd :
        Fraction of the outboard first-wall area taken by HCD apparatus.
        `.fwbs.f_a_fw_outboard_hcd`.
    vol_blkt_total_full_coverage, vol_blkt_inboard_full_coverage :
        Blanket volumes at 100 % coverage (m3). `EllipticalBlanketVolumes`'s outputs.

    Returns
    -------
    tuple
        `(a_blkt_outboard_surface, a_blkt_total_surface, vol_blkt_outboard,
        vol_blkt_inboard, a_blkt_inboard_surface, vol_blkt_total)` -- PROCESS's own
        write order.
    """
    covered = 1.0 - f_ster_div_single - f_a_fw_outboard_hcd

    a_blkt_outboard_surface = (
        a_blkt_total_surface_full_coverage * covered
        - a_blkt_inboard_surface_full_coverage
    )
    a_blkt_total_surface = a_blkt_inboard_surface_full_coverage + a_blkt_outboard_surface

    vol_blkt_outboard = (
        vol_blkt_total_full_coverage * covered - vol_blkt_inboard_full_coverage
    )
    vol_blkt_inboard = vol_blkt_inboard_full_coverage

    a_blkt_inboard_surface = a_blkt_inboard_surface_full_coverage

    vol_blkt_total = vol_blkt_inboard_full_coverage + vol_blkt_outboard

    return (
        a_blkt_outboard_surface,
        a_blkt_total_surface,
        vol_blkt_outboard,
        vol_blkt_inboard,
        a_blkt_inboard_surface,
        vol_blkt_total,
    )


def apply_coverage_factors_double_null(
    a_blkt_total_surface_full_coverage,
    a_blkt_inboard_surface_full_coverage,
    f_ster_div_single,
    f_a_fw_outboard_hcd,
    vol_blkt_total_full_coverage,
    vol_blkt_inboard_full_coverage,
):
    """Blanket areas and volumes after divertor/HCD coverage, double-null arm
    (`n_divertors == 2`).

    Ports the `if n_divertors == 2` arm of `apply_coverage_factors`
    (`blanket_library.py:538-548`) plus the unbranched tail (`:561-584`) it shares with
    its single-null sibling.

    **The two divertors are counted in the areas and not in the volumes, and that is
    PROCESS's own asymmetry, transcribed rather than repaired.** Only the
    `a_blkt_outboard_surface` assignment sits inside the `if`; `vol_blkt_outboard`
    (`blanket_library.py:565-573`) is written *below* the branch and uses the
    single-divertor `1 - f_ster_div_single - f_a_fw_outboard_hcd` on both arms. So a
    double-null machine subtracts two divertors' solid angle from its blanket *surface*
    and one divertor's from its blanket *volume*. Nothing in `blanket_library.py`
    justifies the difference and no comment mentions it; it reads as an arm that was
    edited where the branch was and not where it was not. Recorded as a defect in
    `_audit/units/models/blankets/blanket_library.md` (§ 2026-08-27) and reproduced
    exactly here, per `traceability_policy.md`: the port's job is to agree with PROCESS,
    including where PROCESS is wrong.

    Parameters
    ----------
    a_blkt_total_surface_full_coverage, a_blkt_inboard_surface_full_coverage :
        Blanket surface areas at 100 % coverage (m2).
        `.build.a_blkt_total_surface_full_coverage`,
        `.build.a_blkt_inboard_surface_full_coverage` -- `EllipticalBlanketAreas`'s
        outputs.
    f_ster_div_single :
        Divertor solid-angle fraction **per divertor**. `.fwbs.f_ster_div_single`
        (`divertor.py:42`) -- the `2.0 *` below is where the second divertor enters.
    f_a_fw_outboard_hcd :
        Fraction of the outboard first-wall area taken by HCD apparatus.
        `.fwbs.f_a_fw_outboard_hcd`.
    vol_blkt_total_full_coverage, vol_blkt_inboard_full_coverage :
        Blanket volumes at 100 % coverage (m3). `EllipticalBlanketVolumes`'s outputs.

    Returns
    -------
    tuple
        `(a_blkt_outboard_surface, a_blkt_total_surface, vol_blkt_outboard,
        vol_blkt_inboard, a_blkt_inboard_surface, vol_blkt_total)` -- PROCESS's own
        write order.
    """
    # `blanket_library.py:541-547` -- inside the `n_divertors == 2` branch.
    covered_area = 1.0 - 2.0 * f_ster_div_single - f_a_fw_outboard_hcd
    # `blanket_library.py:565-572` -- below the branch, one divertor on both arms.
    covered_volume = 1.0 - f_ster_div_single - f_a_fw_outboard_hcd

    a_blkt_outboard_surface = (
        a_blkt_total_surface_full_coverage * covered_area
        - a_blkt_inboard_surface_full_coverage
    )
    a_blkt_total_surface = a_blkt_inboard_surface_full_coverage + a_blkt_outboard_surface

    vol_blkt_outboard = (
        vol_blkt_total_full_coverage * covered_volume - vol_blkt_inboard_full_coverage
    )
    vol_blkt_inboard = vol_blkt_inboard_full_coverage

    a_blkt_inboard_surface = a_blkt_inboard_surface_full_coverage

    vol_blkt_total = vol_blkt_inboard_full_coverage + vol_blkt_outboard

    return (
        a_blkt_outboard_surface,
        a_blkt_total_surface,
        vol_blkt_outboard,
        vol_blkt_inboard,
        a_blkt_inboard_surface,
        vol_blkt_total,
    )


class BlanketHalfHeight(ExplicitFunction):
    """The family that owns `.blanket.dz_blkt_half`: one occupant per `n_divertors` arm
    of `BlanketLibrary.calculate_blkt_half_height`.

    Both arms are written (2026-08-27), so this slot is total. They are separate
    occupants and not one node with a `jnp.where` because the double-null arm reads five
    fields fewer -- see `calculate_blkt_half_height_double_null`.
    """


class BlanketHalfHeightSingleNull(BlanketHalfHeight):
    """cottax node: `calculate_blkt_half_height_single_null`. `n_divertors == 1`."""

    dz_blkt_half = OutputInto(blanket)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
        z_plasma_xpoint_upper=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
    ):
        return calculate_blkt_half_height_single_null(
            z_plasma_xpoint_lower,
            dz_xpoint_divertor,
            dz_divertor,
            dz_blkt_upper,
            z_plasma_xpoint_upper,
            dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard,
            dr_fw_inboard,
            dr_fw_outboard,
        )


class BlanketHalfHeightDoubleNull(BlanketHalfHeight):
    """cottax node: `calculate_blkt_half_height_double_null`. `n_divertors == 2`.

    Live on `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`, both of which
    set `i_single_null = 0` (`:292`, `:638`), from which `init.py:606-617` derives
    `n_divertors = 2`.
    """

    dz_blkt_half = OutputInto(blanket)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
    ):
        return calculate_blkt_half_height_double_null(
            z_plasma_xpoint_lower,
            dz_xpoint_divertor,
            dz_divertor,
            dz_blkt_upper,
        )


class EllipticalBlanketAreas(ExplicitFunction):
    """cottax node: `calculate_elliptical_blkt_areas`.

    Occupies the elliptical arm of `component_volumes`' shape decision
    (`itart == 0` and `.fwbs.i_fw_blkt_vv_shape == ELLIPTICAL_SHAPED`).
    """

    a_blkt_inboard_surface_full_coverage = OutputInto(build)
    a_blkt_outboard_surface_full_coverage = OutputInto(build)
    a_blkt_total_surface_full_coverage = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        r_shld_outboard_outer=From(build),
        dr_shld_outboard=From(build),
        dr_blkt_outboard=From(build),
        dz_blkt_half=From(blanket),
    ):
        return calculate_elliptical_blkt_areas(
            rmajor,
            rminor,
            triang,
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            r_shld_outboard_outer,
            dr_shld_outboard,
            dr_blkt_outboard,
            dz_blkt_half,
        )


class EllipticalBlanketVolumes(ExplicitFunction):
    """cottax node: `calculate_elliptical_blkt_volumes`. Same arm as the areas above."""

    vol_blkt_inboard_full_coverage = OutputInto(fwbs)
    vol_blkt_outboard_full_coverage = OutputInto(fwbs)
    vol_blkt_total_full_coverage = OutputInto(fwbs)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        r_shld_outboard_outer=From(build),
        dr_shld_outboard=From(build),
        dr_blkt_outboard=From(build),
        dz_blkt_half=From(blanket),
        dz_blkt_upper=From(build),
    ):
        return calculate_elliptical_blkt_volumes(
            rmajor,
            rminor,
            triang,
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            r_shld_outboard_outer,
            dr_shld_outboard,
            dr_blkt_outboard,
            dz_blkt_half,
            dz_blkt_upper,
        )


class BlanketCoverageFactors(ExplicitFunction):
    """The family that owns `.fwbs.vol_blkt_total` and the five fields written beside
    it: one occupant per `n_divertors` arm of `BlanketLibrary.apply_coverage_factors`.

    `.fwbs.vol_blkt_total` is what the whole of this file exists to reach:
    `CCFE_HCPB.component_masses` (`hcpb.py:306`, `:419`, `:425`, `:444`) reads it and
    nothing else in the tokamak call surface writes it.

    Both arms are written (2026-08-27); the slot is total. The arms read the same six
    fields and differ by one literal, which is enough to make them occupants rather than
    a parameter (`next_steps.md` §14.2, the `istore` precedent).
    """


class BlanketCoverageFactorsSingleNull(BlanketCoverageFactors):
    """cottax node: `apply_coverage_factors_single_null`. `n_divertors == 1`."""

    a_blkt_outboard_surface = OutputInto(build)
    a_blkt_total_surface = OutputInto(build)
    vol_blkt_outboard = OutputInto(fwbs)
    vol_blkt_inboard = OutputInto(fwbs)
    a_blkt_inboard_surface = OutputInto(build)
    vol_blkt_total = OutputInto(fwbs)

    def __call__(
        self,
        a_blkt_total_surface_full_coverage=From(build),
        a_blkt_inboard_surface_full_coverage=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        vol_blkt_total_full_coverage=From(fwbs),
        vol_blkt_inboard_full_coverage=From(fwbs),
    ):
        return apply_coverage_factors_single_null(
            a_blkt_total_surface_full_coverage,
            a_blkt_inboard_surface_full_coverage,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            vol_blkt_total_full_coverage,
            vol_blkt_inboard_full_coverage,
        )


class BlanketCoverageFactorsDoubleNull(BlanketCoverageFactors):
    """cottax node: `apply_coverage_factors_double_null`. `n_divertors == 2`.

    Live on `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`. Carries
    PROCESS's areas-doubled/volumes-not asymmetry unrepaired -- see the function's
    docstring.
    """

    a_blkt_outboard_surface = OutputInto(build)
    a_blkt_total_surface = OutputInto(build)
    vol_blkt_outboard = OutputInto(fwbs)
    vol_blkt_inboard = OutputInto(fwbs)
    a_blkt_inboard_surface = OutputInto(build)
    vol_blkt_total = OutputInto(fwbs)

    def __call__(
        self,
        a_blkt_total_surface_full_coverage=From(build),
        a_blkt_inboard_surface_full_coverage=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        vol_blkt_total_full_coverage=From(fwbs),
        vol_blkt_inboard_full_coverage=From(fwbs),
    ):
        return apply_coverage_factors_double_null(
            a_blkt_total_surface_full_coverage,
            a_blkt_inboard_surface_full_coverage,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            vol_blkt_total_full_coverage,
            vol_blkt_inboard_full_coverage,
        )
