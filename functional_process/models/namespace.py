"""The namespaces of the model modules that are files rather than packages.

`models/build.py` and `models/divertor.py` each hold more than one node and therefore
need a `ModelNamespace` to gather them, but neither is a package, so there is no
`models/<subsystem>/namespace.py` for the class to live in. It lives here, in the same
directory as the modules it names, which is as close to `model_tree_design.md` §11's
"beside the nodes it names" as a flat module allows.

**Only multi-node modules appear here.** `models/fw.py`, `models/structure.py`,
`models/cryostat.py` and `models/vacuum/vacuum.py`'s `VacuumVesselElliptical` each
contribute exactly one node to their `Tokamak` slot, and a slot may hold a node as
readily as a namespace (`Physics.fusion_rates` has always been one), so those four are
bound directly and have nothing here.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.build import (
    BlktUpperThickness,
    DivertorGeometryConventional,
    DivertorGeometrySphericalTokamak,
    DrTfInboardFromWindingPack,
    DrTfOutboardSuperconducting,
    DrTfWpWithInsulationFromInboardBuild,
    PlasmaXpointHeights,
    RadialBuildToPlasmaCentre,
    ShldInboardInnerRadius,
    ShldOutboardOuterRadius,
    ShldVvGapOutboard,
    TfInboardRadiiTfOutsideCs,
    TfInnerBore,
    VacuumVesselAndShieldRadiiTfOutsideCs,
    TfOutboardEdgeRipple,
    TfOutboardMidDShape,
    TfOutboardMidUnrippled,
    TfTopHeight,
    WpConductorMaxWidthSuperconducting,
    ZTfInsideHalf,
)
from functional_process.models.divertor import (
    DivertorHeatFluxSplit,
    DivertorHeatLoadWade,
)


class Build(ModelNamespace):
    """The tokamak's radial and vertical build -- eighteen slots, twenty-four classes.

    `process/models/build.py::Build`, `caller.py:288`. The structural spine of the
    device, and with no stellarator counterpart at all: `models/stellarator/build.py`'s
    `Build` is a different model of a different machine.

    Eighteen slots and twenty-four occupant classes, because six slots hold more than
    one arm: `dr_tf_inboard_winding_pack`, `tf_inboard_radii`, `tf_outboard_mid`,
    `tf_outboard_edge_ripple`, `tf_top_height` (added 2026-08-30) and
    `divertor_geometry` (whose third disposition, `None`, is absence rather than a
    class). `models/build.py` declares twenty-five classes; the twenty-fifth,
    `TfTopHeight`, is that slot's abstract family base and occupies nothing.

    **Seven of the eighteen slots are switched, and only two of the switches are an
    `i_*` integer alone** (`tf_inboard_radii`'s `.build.i_tf_inside_cs`, added
    2026-08-27, and `tf_top_height`'s `.physics.i_single_null`, added 2026-08-30).
    `.tfcoil.i_tf_sup` and `.tfcoil.i_tf_shape` are ordinary switches;
    the other two are conditions on things that are not switches at all -- whether
    iteration variable 140 is active, and whether the *input* value of
    `.build.dz_xpoint_divertor` is effectively zero. Both are still resolved exactly
    where every switch is, in `machine_from_indat`, and for the same reason: neither can
    change between two evaluations of one assembled graph, because `ixc` is fixed for a
    solve and an input is an input.

    **This namespace produces five of the six variables `tokamak_boundary.md` attributes
    to it, plus one it does not.** `.build.dr_tf_inboard` stays a boundary input on
    `large_tokamak_eval.IN.DAT`, because `process/models/build.py:1685` is guarded by
    `140 in ixc` and that file's `ixc` is `[4, 6]`; what runs instead is the inverse
    assignment producing `.tfcoil.dr_tf_wp_with_insulation`, which is not in the boundary
    file at all. `build.md` § "contradiction with `tokamak_boundary.md`" is that
    measurement -- an `ast` walk over `Assign` targets cannot see an `ixc` guard, which
    is a limit of that method rather than a mistake in it.
    """

    plasma_xpoint_heights: PlasmaXpointHeights = PlasmaXpointHeights()
    """`.build.z_plasma_xpoint_upper`/`_lower`. Unswitched."""

    divertor_geometry: (
        DivertorGeometryConventional | DivertorGeometrySphericalTokamak | None
    ) = dataclasses.field(kw_only=True)
    """`.physics.itart`, **and** the input `.build.dz_xpoint_divertor < 1e-5`.

    Two conditions, one slot, three dispositions. `process/models/build.py:800-801`
    assigns `dz_xpoint_divertor = divht` only when the input is effectively zero and
    keeps the user's value otherwise, so whether a node *owns* that field is a
    run-configuration fact -- `conditional-ownership-by-run-config`, the same shape
    `build.md` uses to close `next_steps.md` §2's `dz_shld_upper` flag.
    `large_tokamak_eval.IN.DAT` never sets it, so it takes `build_variables.py:326`'s
    default `0.0` and `DivertorGeometryConventional` owns it (plus `.build.rspo`).
    `itart == 1` selects `DivertorGeometrySphericalTokamak` (`build.py:863`'s
    `1.75 * rminor`, which never reaches the `rspo` write) when the input is effectively
    zero -- and `None` when the run sets it, as both tracked spherical-tokamak inputs
    do: the early return is computed and discarded at `:800` and PROCESS computes
    nothing that survives, so absence, not refusal.
    """

    z_tf_inside_half: ZTfInsideHalf = ZTfInsideHalf()
    """`.build.z_tf_inside_half`, from the vertical stack at `build.py:807`.

    **The third source-level writer of one field, and the second in this port.**
    `models/stellarator/coils/calculate.py::ZTfInsideHalf` owns the same `VarPath` from
    `st_coil`'s coil-geometry formula, and `st_build` has a third formula that the
    stellarator's own `Build` node deliberately does not own. Never two in one graph, so
    there is nothing to settle -- but a field with that many producers is exactly the
    "which writer wins" check `_audit/test_harness.md` says is owed."""

    tf_top_height: TfTopHeight = dataclasses.field(kw_only=True)
    """`.physics.i_single_null` -- `.build.z_tf_top` and
    `.build.dz_tf_upper_lower_midplane`, both arms written (`build.py:820-841`).

    Added 2026-08-30. The single-null arm is what every assembling tokamak takes; the
    double-null one is written but not yet reachable, its two inputs being refused
    earlier for `i_tf_turn_type == 2`. Both fields were missing producers on
    `large_tokamak_nof`
    (`missing_producers_tokamak.txt`), frozen at `0.0` against PROCESS's `8.656` m and
    `-1.234` m -- and `.build.z_tf_top` is *read*, by
    `models/tfcoil/base.py::TfCoilShapeDShapeSingleNull` (which places the coil's arcs
    from it) and `models/pfcoil/geometry.py` (which places the divertor PF coils from
    it), so the cold graph was drawing a TF coil whose top sat on the midplane."""

    blkt_upper_thickness: BlktUpperThickness = BlktUpperThickness()
    """`.build.dz_blkt_upper`, the mean of the two radial blanket thicknesses
    (`build.py:1664-1667`). Unswitched.

    Added 2026-08-30 as the slot above's missing dependency -- the single-null
    `z_tf_top` stack reads it, and it was itself on `missing_producers_tokamak.txt`.
    Its two operands are run inputs whenever `.fwbs.blktmodel == 0`, which is every
    tracked tokamak."""

    dr_tf_inboard_winding_pack: (
        DrTfInboardFromWindingPack | DrTfWpWithInsulationFromInboardBuild
    ) = dataclasses.field(kw_only=True)
    """Whether iteration variable 140 is active -- and the two arms own **different
    fields**, being exact inverses of one relation.

    `140 in ixc` produces `.build.dr_tf_inboard` from the winding pack; `140 not in ixc`
    produces `.tfcoil.dr_tf_wp_with_insulation` from the inboard build. Both are written
    and tested; `large_tokamak_eval.IN.DAT`'s `ixc = [4, 6]` selects the second.

    Named for the pair rather than for either occupant, the same way
    `PhysicsProfiles.parameterisation` is: the rule that a slot name is the snake_case of
    its occupant's class needs a shared stem to apply to, and here there is no family
    base class because the two arms share no output to declare."""

    tf_inboard_radii: TfInboardRadiiTfOutsideCs = dataclasses.field(kw_only=True)
    """`(.build.i_tf_inside_cs, .build.i_cs_precomp)` -- `(0, 1)` (both defaults, both
    live) is written; `TF_INSIDE_CS` and the no-precompression arm are UNPORTED
    (`indat._tf_inboard_radii_arm`). Added 2026-08-27, `cold_boundary.md` producer 2:
    `.build.r_tf_inboard_in` and `.build.r_tf_inboard_out` were two of the six cold
    boundary zeros (3 of the 11 roots), and the slice is taken whole
    (`build.py:1691-1735`) so `dr_cs_bore` -- a standing boundary input with a wrong
    cold value, read by `CSFluxSwing` -- and `dr_cs_precomp` are produced rather than
    read stale; the node's own docstring carries the argument."""

    vacuum_vessel_and_shield_radii: VacuumVesselAndShieldRadiiTfOutsideCs = (
        dataclasses.field(kw_only=True)
    )
    """`.build.i_tf_inside_cs` -- `TF_OUTSIDE_CS` is written, `TF_INSIDE_CS` UNPORTED
    (`indat.VACUUM_VESSEL_AND_SHIELD_RADII`). Added 2026-08-29: this is
    `_audit/next_steps.md` §16.3's last cold blocker and
    `consolidation_round_3.md` §4's last producer, in three lines of the same block --
    `.build.r_vv_inboard_out` was the one non-finite condition left in the cold tokamak
    SAND probe (it divides in `vv_stress_on_quench`), and `.build.r_sh_inboard_out` was
    the read `blankets/hcpb.py`'s centrepost cluster declared with nothing producing
    it."""

    radial_build_to_plasma_centre: RadialBuildToPlasmaCentre = (
        RadialBuildToPlasmaCentre()
    )
    """`.build.rbld`, PROCESS's own "should be equal to `rmajor`" accumulation, which
    is what constraint 11 says -- active on three of the four tracked tokamak files.
    Split from the slot above rather than folded into it: it is the one line of that
    block that reads `.physics.rminor`, and one node would hand the vacuum-vessel radius
    a dependency PROCESS does not give it."""

    shld_inboard_inner_radius: ShldInboardInnerRadius = ShldInboardInnerRadius()
    shld_outboard_outer_radius: ShldOutboardOuterRadius = ShldOutboardOuterRadius()
    """The two shield radii, built inwards and outwards from the plasma. Unswitched, and
    the reason the whole central-solenoid radial chain is outside this closure:
    `r_shld_inboard_inner` is not accumulated outwards from the bore."""

    dr_tf_outboard: DrTfOutboardSuperconducting = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_sup`. The non-superconducting arm scales by
    `.build.f_dr_tf_outboard_inboard`, which this arm never reads -- a disjoint
    reads-set, so an occupant and not a kwarg."""

    wp_conductor_max_width: WpConductorMaxWidthSuperconducting = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_tf_sup`, and the owner of the mint `.tfcoil.dx_tf_wp_conductor_max`.

    The resistive arm computes the same quantity from
    `.superconducting_tfcoil.r_tf_wp_inboard_outer` and `.tfcoil.n_tf_coils` instead of
    the three `dx_tf_wp_*` fields -- again disjoint, again UNPORTED."""

    tf_outboard_mid_unrippled: TfOutboardMidUnrippled = TfOutboardMidUnrippled()
    """The mint `.build.r_tf_outboard_mid_unrippled`: the value PROCESS assigns to
    `.build.r_tf_outboard_mid` at `:1901` and then overwrites in place at `:1939`.
    Minting it is what keeps `.build.r_tf_outboard_mid` from being read and owned by one
    node -- the same resolution `physics.py`'s `p_plasma_separatrix_mw_raw` uses."""

    tf_outboard_mid: TfOutboardMidDShape = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_shape`. The auto-select value `0` never reaches a model: PROCESS's
    own `init.py:728`/`:775` resolves it to picture-frame or D-shape from
    `.physics.itart` before any model runs, and `machine_from_indat` reproduces that
    resolution rather than the raw default."""

    tf_outboard_edge_ripple: TfOutboardEdgeRipple = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_shape`, and PROCESS's **second** call to the same ripple fit -- the
    one whose answer survives into `.tfcoil.ripple_b_tf_plasma_edge`. Two nodes rather
    than one, so that neither reads a field it owns; the cost is that this one recomputes
    `r_tf_outboard_midmin` and discards it, a duplicated computation the graph cannot see
    is duplicated (`build.md` OQ2)."""

    shld_vv_gap_outboard: ShldVvGapOutboard = ShldVvGapOutboard()
    """`.build.dr_shld_vv_gap_outboard`. Unswitched -- the source's two arms are one
    expression."""

    tf_inner_bore: TfInnerBore = TfInnerBore()
    """`.build.dr_tf_inner_bore`, the midplane bore between the two TF legs
    (`build.py:1911-1913`, rewritten verbatim at `:1949-1955`). Unswitched.

    Added 2026-08-30, another `missing_producers_tokamak.txt` row: `11.794` m in
    PROCESS, `0.0` here, read by `models/structure.py`'s support masses and
    `costs/costs_2015.py`'s Account 22. Evaluated at the post-ripple
    `.build.r_tf_outboard_mid`, so it reproduces whichever of the source's two identical
    writes lands -- the same resolution `tf_outboard_edge_ripple` uses one slot up."""


class Divertor(ModelNamespace):
    """The tokamak divertor -- two nodes, one of them switched.

    `process/models/divertor.py::Divertor`, `caller.py:324`. **Not**
    `.stellarator.divertor`, which is `models/stellarator/divertor.py` and is one half of
    the one cycle the stellarator graph has.

    A namespace of two rather than two slots of `Tokamak`, because they are one PROCESS
    `Model` and one of them is switched while the other is not -- putting the unswitched
    one beside `.tokamak.build` in the device namespace would say they are separate
    subsystems, which they are not. `divertor.md`'s open question asked for this choice
    to be made here; it is.
    """

    heat_flux_split: DivertorHeatFluxSplit = DivertorHeatFluxSplit()
    """`.fwbs.f_ster_div_single`, `.fwbs.p_div_nuclear_heat_total_mw` and
    `.divertor.deg_div_poloidal_plasma`. **Not** switch-gated: it runs whatever
    `i_div_heat_load` is.

    It reads `.divertor.n_divertors` as an ordinary multiplier, which is the cleanest
    illustration in this wave of the policy `_audit/switch_kwarg_survey.md` needs: a
    switch read *arithmetically* is an ordinary input port, and the same field read to
    *branch* selects an occupant. One integer, both roles, in one file."""

    heat_load: DivertorHeatLoadWade = dataclasses.field(kw_only=True)
    """`.divertor.i_div_heat_load` -- `2` (Wade) on `large_tokamak_eval.IN.DAT:139`.

    `0` (user input, which reads nothing and computes nothing) and `1` (Peng chamber,
    `divtart`, a tight-aspect-ratio model reading six fields Wade never touches) are
    both UNPORTED. Wade's own double-null arm **is** written (2026-08-27), as a second
    occupant: it reads `.physics.f_p_div_lower` and takes a `max` the single-null arm
    does not, so neither function has an `n_divertors` argument -- each *is* the
    occupant for its value."""
