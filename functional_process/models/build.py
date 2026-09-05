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
| `(.physics.itart, .tfcoil.i_tf_sup)` | not (`1`, not-`1`) | `RCpTopFromTfInboardOut` |

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

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.build import (
    calculate_divertor_geometry_conventional,
    calculate_divertor_geometry_spherical_tokamak,
    calculate_dr_shld_vv_gap_outboard,
    calculate_dr_tf_inboard,
    calculate_dr_tf_inner_bore,
    calculate_dr_tf_outboard_superconducting,
    calculate_dr_tf_wp_with_insulation,
    calculate_dx_tf_wp_conductor_max_superconducting,
    calculate_dz_blkt_upper,
    calculate_r_cp_top_from_tf_inboard_out,
    calculate_r_shld_inboard_inner,
    calculate_r_shld_outboard_outer,
    calculate_r_tf_inboard_radii_no_cs_precomp,
    calculate_r_tf_inboard_radii_tf_outside_cs,
    calculate_r_tf_outboard_mid,
    calculate_r_tf_outboard_mid_unrippled,
    calculate_rbld,
    calculate_tf_top_height_double_null,
    calculate_tf_top_height_single_null,
    calculate_vacuum_vessel_and_shield_radii,
    calculate_z_plasma_xpoint,
    calculate_z_tf_inside_half,
    plasma_outboard_edge_toroidal_ripple_fitted,
    plasma_outboard_edge_toroidal_ripple_picture_frame,
)
from functional_process.models.safe_math import safe_sqrt  # noqa: F401
from functional_process.paths import build, divertor, physics, tfcoil


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


class RCpTopFromTfInboardOut(ExplicitFunction):
    """cottax node: `calculate_r_cp_top_from_tf_inboard_out`, ports declared.

    Answers `(.physics.itart, .tfcoil.i_tf_sup)` -- every cell except the resistive
    spherical tokamak (`itart == 1` and `i_tf_sup != 1`), which is
    `indat._r_cp_top_arm`'s refused arm `-1`. That is *all five* tracked **tokamak**
    configurations' cell -- the two spherical ones included, because both are
    superconducting -- and the stellarators have no `.tokamak.build` at all.
    """

    r_cp_top = OutputInto(build)

    def __call__(self, r_tf_inboard_out=From(build)):
        return calculate_r_cp_top_from_tf_inboard_out(r_tf_inboard_out)
