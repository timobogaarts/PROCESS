"""Pure-functional port of the tier-1 functions in `coils/calculate.py` (registry unit #9).

Audit record: `functional_process/_audit/units/models/stellarator/coils/calculate.md`.
That file's `st_coil` (593 lines) is the orchestrator called directly from
`Stellarator.run()`; most of its body is 12 short, independent helper functions, 10 of
which are tier-1 (pure arithmetic, no internal solve, no calls into any other file) and
were ported first. The other two -- `winding_pack_total_size` (a 200-point sampled curve
fed into `intersect`, a Newton-Raphson root-find in `coils/coils.py`) and `st_coil`
itself (the orchestrator, which also calls `coils/mass.py`, `coils/quench.py`,
`coils/forces.py`, `coils/output.py`) -- were blocked on registry units #10-14. Units
#10 (partially, see below), #11, #12 and #14 are now ported, and #13 is confirmed pure
reporting with nothing to port, which unblocked both: `winding_pack_total_size` is
ported below (tier-2, a `Tier2Contract`, same pattern as `coils.py`'s own `intersect`),
and `st_coil` is ported as a plain composed function (tier-3; see the record for why it
gets no `cottax` node of its own).

`winding_pack_total_size` calls `intersect`/`bmax_from_awp` (`coils/coils.py`, already
ported) directly, and needs `jcrit_from_material`'s dispatch on `i_tf_sc_mat` -- which
itself is **not** ported (`coils.py` remains out of this unit's boundary; see
`coils.md`). The eight `jcrit_*` functions below are therefore a local restatement of
that dispatch, scoped to this unit's own solve, calling the real ported material models
in `functional_process/models/physics/superconductors.py` directly. They are not the
audited port of `jcrit_from_material` -- that stays unit #10's to do; see the record's
"switches touched" section.

**One function per `i_tf_sc_mat` value, and one node class per value on top of them**
(`_audit/next_steps.md` §14.2's binding policy, §14.5). `i_tf_sc_mat` used to be an
`eqx.field(static=True)` on a single `WindingPackIntersectInputs`, which therefore
declared the union of all eight branches' reads -- six of them dead at the value every
run here holds, and one of the six (`.tfcoil.j_tf_wp`, live on Bi-2212 alone) the sole
back-edge closing the four-node coils SCC (`_audit/switch_kwarg_survey.md` §4.6). The
composite `_critical_current_density_by_material` / `winding_pack_pre_intersect` /
`winding_pack_curves` / `winding_pack_total_size` chain is kept, unchanged in signature
and in numbers, because it is what PROCESS's own `winding_pack_total_size` is diffed
against at every material; the *graph* uses the per-material occupants instead.

Every function below keeps its original name (already `calculate_*`-shaped in the
source, so nothing to rename per `naming_convention.md`) and takes exactly the fields it
reads as explicit arguments -- no `data: DataStructure` parameter anywhere, unlike the
source (`_audit/traceability_policy.md`: closing the `data` back-door is the whole
point).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.physics.superconductors import (
    bi2212,  # noqa: F401
    gl_nbti,  # noqa: F401
    gl_rebco,  # noqa: F401
    itersc,  # noqa: F401
    jcrit_nbti,  # noqa: F401
    jcrit_rebco,  # noqa: F401
    western_superconducting_nb3sn,  # noqa: F401
)
from functional_process.models.stellarator.coils.coils import (
    bmax_from_awp,  # noqa: F401
    intersect,  # noqa: F401
)
from functional_process.models.stellarator.coils.forces import (
    calculate_centering_force_avg_mn,  # noqa: F401
    calculate_centering_force_max_mn,  # noqa: F401
    calculate_centering_force_min_mn,  # noqa: F401
    calculate_max_force_density,  # noqa: F401
    calculate_max_force_density_mnm,  # noqa: F401
    calculate_max_lateral_force_density,  # noqa: F401
    calculate_max_radial_force_density,  # noqa: F401
    calculate_maximum_stress,  # noqa: F401
)
from functional_process.models.stellarator.coils.mass import (
    calculate_coils_mass,  # noqa: F401
)
from functional_process.models.stellarator.coils.quench import (
    calculate_quench_protection,  # noqa: F401
)
from functional_process.paths import (
    build,
    constraints,
    stellarator,
    stellarator_config,
    tfcoil,
)
from functional_process.stellarator.coils.calculate import (
    calculate_bi2212_winding_pack_intersect_inputs,
    calculate_casing,
    calculate_coil_coil_toroidal_gap,  # noqa: F401
    calculate_coil_cross_sectional_area,
    calculate_coil_half_widths,
    calculate_coil_radial_thickness,
    calculate_coil_toroidal_thickness,
    calculate_coils_summary_variables,
    calculate_current,
    calculate_durham_nbti_winding_pack_intersect_inputs,
    calculate_horizontal_ports,
    calculate_inductance,  # noqa: F401
    calculate_len_tf_coil,
    calculate_plasma_facing_coil_area,
    calculate_stored_magnetic_energy,
    calculate_tfcryoarea,
    calculate_user_defined_nb3sn_winding_pack_intersect_inputs,
    calculate_vertical_ports,
    calculate_winding_pack_geometry,
    calculate_z_tf_inside_half,
    jcrit_bi2212,  # noqa: F401
    jcrit_croco_rebco,
    jcrit_durham_nbti,  # noqa: F401
    jcrit_durham_rebco,
    jcrit_iter_nb3sn,
    jcrit_old_lubell_nbti,
    jcrit_user_defined_nb3sn,  # noqa: F401
    jcrit_wst_nb3sn,
    select_coil_coil_toroidal_gap,
    st_coil,  # noqa: F401
    winding_pack_curves,  # noqa: F401
    winding_pack_post_intersect,
    winding_pack_pre_intersect,  # noqa: F401
    winding_pack_pre_intersect_for,
    winding_pack_total_size,  # noqa: F401
)
from functional_process.vocabulary import (
    SuperconductorModel,  # noqa: F401
)


class CoilToroidalThickness(ExplicitFunction):
    dx_tf_inboard_out_toroidal = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_wp_primary_toroidal=From(tfcoil),
        dx_tf_side_case_min=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
    ):
        return calculate_coil_toroidal_thickness(
            dx_tf_wp_primary_toroidal, dx_tf_side_case_min, dx_tf_wp_insulation
        )


class CoilRadialThickness(ExplicitFunction):
    dr_tf_inboard = OutputInto(build)

    def __call__(
        self,
        dr_tf_nose_case=From(tfcoil),
        dr_tf_wp_with_insulation=From(tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
    ):
        return calculate_coil_radial_thickness(
            dr_tf_nose_case,
            dr_tf_wp_with_insulation,
            dr_tf_plasma_case,
            dx_tf_wp_insulation,
        )


class CoilCrossSectionalArea(ExplicitFunction):
    a_tf_leg_outboard = OutputInto(tfcoil)
    a_tf_coil_inboard_case = OutputInto(tfcoil)

    def __call__(
        self,
        a_tf_wp_with_insulation=From(tfcoil),
        dr_tf_inboard=From(build),
        dx_tf_inboard_out_toroidal=From(tfcoil),
    ):
        return calculate_coil_cross_sectional_area(
            a_tf_wp_with_insulation, dr_tf_inboard, dx_tf_inboard_out_toroidal
        )


class CoilHalfWidths(ExplicitFunction):
    tfocrn = OutputInto(tfcoil)
    tficrn = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_inboard_out_toroidal=From(tfcoil),
    ):
        return calculate_coil_half_widths(dx_tf_inboard_out_toroidal)


class PlasmaFacingCoilArea(ExplicitFunction):
    tfsai = OutputInto(tfcoil)
    tfsao = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        dx_tf_inboard_out_toroidal=From(tfcoil),
        len_tf_coil=From(tfcoil),
    ):
        return calculate_plasma_facing_coil_area(
            n_tf_coils, dx_tf_inboard_out_toroidal, len_tf_coil
        )


class CoilCoilToroidalGap(ExplicitFunction):
    toroidalgap = OutputInto(tfcoil)

    def __call__(
        self,
        stella_config_dmin=From(stellarator_config),
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        stella_config_coil_rmajor=From(stellarator_config),
        stella_config_coil_rminor=From(stellarator_config),
        dx_tf_inboard_out_toroidal=From(tfcoil),
    ):
        return select_coil_coil_toroidal_gap(
            stella_config_dmin,
            r_coil_major,
            r_coil_minor,
            stella_config_coil_rmajor,
            stella_config_coil_rminor,
            dx_tf_inboard_out_toroidal,
        )


class CoilsSummaryVariables(ExplicitFunction):
    a_tf_inboard_total = OutputInto(tfcoil)
    c_tf_total = OutputInto(tfcoil)
    j_tf_coil_full_area = OutputInto(tfcoil)
    r_b_tf_inboard_peak_symmetric = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        a_tf_leg_outboard=From(tfcoil),
        coilcurrent=From(stellarator),
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        dr_tf_wp_with_insulation=From(tfcoil),
    ):
        return calculate_coils_summary_variables(
            n_tf_coils,
            a_tf_leg_outboard,
            coilcurrent,
            r_coil_major,
            r_coil_minor,
            dr_tf_wp_with_insulation,
        )


class StoredMagneticEnergy(ExplicitFunction):
    e_tf_magnetic_stored_total_gj = OutputInto(tfcoil)

    def __call__(
        self,
        stella_config_inductance=From(stellarator_config),
        f_st_rmajor=From(stellarator),
        r_coil_minor=From(stellarator),
        stella_config_coil_rminor=From(stellarator_config),
        f_st_n_coils=From(stellarator),
        c_tf_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return calculate_stored_magnetic_energy(
            stella_config_inductance,
            f_st_rmajor,
            r_coil_minor,
            stella_config_coil_rminor,
            f_st_n_coils,
            c_tf_total,
            n_tf_coils,
        )


class WindingPackGeometry(ExplicitFunction):
    a_tf_turn_cable_space_no_void = OutputInto(tfcoil)
    a_tf_turn_steel = OutputInto(tfcoil)

    def __call__(
        self,
        dx_tf_turn_general=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
    ):
        return calculate_winding_pack_geometry(
            dx_tf_turn_general, dx_tf_turn_steel, dx_tf_turn_insulation
        )


class CoilCurrent(ExplicitFunction):
    """`coilcurrent` has no PROCESS storage location -- it is a local in `st_coil`,
    threaded manually into `winding_pack_total_size` and
    `calculate_coils_summary_variables`. `.stellarator.coilcurrent` is an invented
    `VarPath` (per `naming_convention.md`: port the existing name where one exists;
    mint one where it doesn't), needed because `CoilsSummaryVariables` below reads it --
    without minting it, that node would have no way to source this input at all.
    """

    coilcurrent = OutputInto(stellarator)
    f_st_i_total = OutputInto(stellarator)

    def __call__(
        self,
        f_st_b=From(stellarator),
        stella_config_i0=From(stellarator_config),
        f_st_rmajor=From(stellarator),
        f_st_n_coils=From(stellarator),
    ):
        return calculate_current(f_st_b, stella_config_i0, f_st_rmajor, f_st_n_coils)


class WindingPackIntersectInputs(ExplicitFunction):
    """The family that owns the *pre*-`intersect` half of `winding_pack_total_size`:
    the sampled `(wp_width_r, lhs, rhs)` curves `coils.py`'s `Intersect`
    (`ImplicitFunction`/`RootFind`) needs as its own `From`s, and the starting guess it
    is driven from. One occupant per `i_tf_sc_mat` value.

    This, together with `coils.py`'s `Intersect` and `WindingPackTotalSizePost` below,
    replaces the single `WindingPackTotalSize` node an earlier pass wrote (which called
    `intersect` eagerly, in the middle of its own `__call__`) -- see
    `_audit/next_steps.md` §7 and `coils.py`'s `Intersect` docstring for why splitting
    the *structural* declaration around `intersect` is worth doing even though nothing
    else in the graph needs `intersect`'s internal unknowns visible (§7's own test for
    that, unchanged): it makes the root-find's solver algorithm a first-class, swappable
    `Drive` choice, not something hardcoded inside one node's body.

    Mints `.stellarator.wp_width_r`/`.lhs`/`.rhs` at exactly the `VarPath`s `Intersect`
    reads -- `coils.md`'s own sketch of this split already proposed these names for this
    exact call site, not a fresh invention here.

    **`i_tf_sc_mat` was an `eqx.field(static=True)` on this class and is gone**
    (`_audit/next_steps.md` §14.2's binding policy, §14.5). The eight branches read
    genuinely different `.tfcoil.*` fields, so one node carrying all eight declared six
    reads that are dead at `ITER_NB3SN` -- and one of the six, `.tfcoil.j_tf_wp`, was
    measured to be **the sole back-edge closing the four-node coils SCC**
    (`_audit/switch_kwarg_survey.md` §4.6). Only `Bi2212...` reads it, so on every other
    material the block collapses to `Intersect` and its own `^problem`, which is the
    cycle the model genuinely has.

    **`wp_width_r_min_guess` is an `Output` here**, which it was not before. It is
    `intersect`'s `xin` (`calculate.py:452-458`), and the old arrangement discarded it
    on the grounds that "a starting guess is a property of the algorithm, not an edge of
    the model" -- so `mda.ROOT_FIND_SEEDS` re-derived it from `.stellarator.r_coil_minor`
    read out of the *block's context*, which only held `r_coil_minor` because the
    invented `j_tf_wp` edge dragged this node into the block. Remove the invented edge
    and that seed loses its source. `cottax.rewrites.Supply` is the mechanism that was
    missing: `Assign` opens `^guess.stellarator.wp_width_r_min` as a boundary input and
    `Supply` points that port at this output instead (`mda.supply_starts`), so PROCESS's
    own starting guess reaches the driver as an ordinary graph edge and the boundary
    loses a `guess` entry rather than gaining a fallback.

    `fraction_area_superconductor_of_wp` (return-only, reporting) is still discarded, as
    the pre-split `WindingPackTotalSize` discarded it, for the same reporting-only
    reason.
    """

    wp_width_r = OutputInto(stellarator)
    lhs = OutputInto(stellarator)
    rhs = OutputInto(stellarator)
    wp_width_r_min_guess = OutputInto(stellarator)

    sample_lower_divisor = 40.0
    guess_divisor = 10.0
    """`_MATERIAL_SAMPLING`'s row for this occupant's material, as plain class
    attributes -- the ordinary pair by default, overridden by the one occupant PROCESS
    treats differently. Not `eqx.field`s: they are a property of the class, and there is
    no constructor argument that could set them (which is exactly what "the switch
    selects a class" means)."""

    def _curves(
        self,
        jcrit,
        r_coil_major,
        r_coil_minor,
        coilcurrent,
        n_tf_coils,
        stella_config_a1,
        stella_config_a2,
        stella_config_wp_ratio,
        tftmp,
        tmargmin,
        f_a_tf_turn_cable_copper,
        f_a_tf_turn_cable_space_extra_void,
        f_j_tf_wp_critical_max,
        a_tf_turn_cable_space_no_void,
        dx_tf_turn_general,
    ):
        """The occupant's four outputs, from its own `jcrit` law and its own divisors.

        The fourteen reads every material shares, in one place, so an occupant's body is
        its material's law and nothing else. Not a port surface: `_params` reads
        `__call__`'s signature only (`ExplicitFunction._signature_of`), so what is
        declared is still each occupant's own parameter list.
        """
        wp_width_r, lhs, rhs, _fraction, wp_width_r_min_guess = (
            winding_pack_pre_intersect_for(
                jcrit,
                self.sample_lower_divisor,
                self.guess_divisor,
                r_coil_major,
                r_coil_minor,
                coilcurrent,
                n_tf_coils,
                stella_config_a1,
                stella_config_a2,
                stella_config_wp_ratio,
                tftmp,
                tmargmin,
                f_a_tf_turn_cable_copper,
                f_a_tf_turn_cable_space_extra_void,
                f_j_tf_wp_critical_max,
                a_tf_turn_cable_space_no_void,
                dx_tf_turn_general,
            )
        )
        return wp_width_r, lhs, rhs, wp_width_r_min_guess


class IterNb3snWindingPackIntersectInputs(WindingPackIntersectInputs):
    """`i_tf_sc_mat == ITER_NB3SN` (1) -- PROCESS's own default and this run's value.

    Reads no material field at all: `jcrit_iter_nb3sn`'s `bc20m`/`tc0m` are literals.
    **Six reads leave with this occupant** -- `.tfcoil.b_crit_upper_nbti`, `.bcritsc`,
    `.fhts`, `.t_crit_nbti`, `.tcritsc` and `.j_tf_wp`.
    """

    def __call__(
        self,
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        tftmp=From(tfcoil),
        tmargmin=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_j_tf_wp_critical_max=From(constraints),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
    ):
        return self._curves(
            jcrit_iter_nb3sn,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
        )


class Bi2212WindingPackIntersectInputs(WindingPackIntersectInputs):
    """`i_tf_sc_mat == BI2212` (2).

    **The one occupant that reads `.tfcoil.j_tf_wp`**, which `WindingPackTotalSizePost`
    owns -- so this is the one material for which the coils block is genuinely a
    four-node cycle rather than `Intersect` and its `^problem`. Also the one that reads
    `.tfcoil.fhts`.

    A consequence worth stating rather than working around: with this occupant the node
    that produces `wp_width_r_min_guess` is *inside* the driven block, and cottax refuses
    a `Start` produced inside its own block (*"the driver reads its data before the block
    runs"*). `mda.supply_starts` therefore leaves this machine's start at the boundary --
    see its own docstring.
    """

    def __call__(
        self,
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        tftmp=From(tfcoil),
        tmargmin=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_j_tf_wp_critical_max=From(constraints),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
        fhts=From(tfcoil),
        j_tf_wp=From(tfcoil),
    ):
        return calculate_bi2212_winding_pack_intersect_inputs(
            self,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
            fhts,
            j_tf_wp,
        )


class OldLubellNbtiWindingPackIntersectInputs(WindingPackIntersectInputs):
    """`i_tf_sc_mat == OLD_LUBELL_NBTI` (3). Literals only, no material read."""

    def __call__(
        self,
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        tftmp=From(tfcoil),
        tmargmin=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_j_tf_wp_critical_max=From(constraints),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
    ):
        return self._curves(
            jcrit_old_lubell_nbti,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
        )


class UserDefinedNb3snWindingPackIntersectInputs(WindingPackIntersectInputs):
    """`i_tf_sc_mat == USER_DEFINED_NB3SN` (4) -- the only occupant reading
    `.tfcoil.bcritsc`/`.tfcoil.tcritsc`, which are exactly what "user-defined" means
    here.
    """

    def __call__(
        self,
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        tftmp=From(tfcoil),
        tmargmin=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_j_tf_wp_critical_max=From(constraints),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
        bcritsc=From(tfcoil),
        tcritsc=From(tfcoil),
    ):
        return calculate_user_defined_nb3sn_winding_pack_intersect_inputs(
            self,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
            bcritsc,
            tcritsc,
        )


class WstNb3snWindingPackIntersectInputs(WindingPackIntersectInputs):
    """`i_tf_sc_mat == WST_NB3SN` (5). Literals only, no material read."""

    def __call__(
        self,
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        tftmp=From(tfcoil),
        tmargmin=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_j_tf_wp_critical_max=From(constraints),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
    ):
        return self._curves(
            jcrit_wst_nb3sn,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
        )


class CrocoRebcoWindingPackIntersectInputs(WindingPackIntersectInputs):
    """`i_tf_sc_mat == CROCO_REBCO` (6) -- the one occupant with different sampling.

    `_MATERIAL_SAMPLING`'s only non-default row: the sweep starts at
    `r_coil_minor / 150` and the guess at `(r_coil_minor / 20) ** 2`, PROCESS's own
    "if REBCO, start at smaller winding pack ratios" (`calculate.py:455-458`). No
    material read -- `jcrit_rebco` takes only field and temperature.
    """

    sample_lower_divisor = 150.0
    guess_divisor = 20.0

    def __call__(
        self,
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        tftmp=From(tfcoil),
        tmargmin=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_j_tf_wp_critical_max=From(constraints),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
    ):
        return self._curves(
            jcrit_croco_rebco,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
        )


class DurhamNbtiWindingPackIntersectInputs(WindingPackIntersectInputs):
    """`i_tf_sc_mat == DURHAM_NBTI` (7) -- the only occupant reading
    `.tfcoil.b_crit_upper_nbti`/`.tfcoil.t_crit_nbti`.
    """

    def __call__(
        self,
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        tftmp=From(tfcoil),
        tmargmin=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_j_tf_wp_critical_max=From(constraints),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
        b_crit_upper_nbti=From(tfcoil),
        t_crit_nbti=From(tfcoil),
    ):
        return calculate_durham_nbti_winding_pack_intersect_inputs(
            self,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
            b_crit_upper_nbti,
            t_crit_nbti,
        )


class DurhamRebcoWindingPackIntersectInputs(WindingPackIntersectInputs):
    """`i_tf_sc_mat == DURHAM_REBCO` (8). Literals only, and the **ordinary** sampling
    divisors: PROCESS's two REBCO special cases test `i_tf_sc_mat == 6` exactly.
    """

    def __call__(
        self,
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        tftmp=From(tfcoil),
        tmargmin=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_j_tf_wp_critical_max=From(constraints),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
    ):
        return self._curves(
            jcrit_durham_rebco,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            tftmp,
            tmargmin,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_extra_void,
            f_j_tf_wp_critical_max,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
        )


class WindingPackTotalSizePost(ExplicitFunction):
    """cottax node: the *post*-`intersect` half of `winding_pack_total_size` --
    everything downstream of the resolved crossing point.

    Reads `.stellarator.wp_width_r_min` as a plain, ordinary `From` -- `coils.py`'s
    `Intersect` (its `RootFind` problem, specifically) owns that `VarPath`, not this
    node, so this is a genuine cross-node edge, not a self-loop (see `Intersect`'s own
    docstring for why the pair below it is *not* a self-loop either). Together with
    `WindingPackIntersectInputs` above and `coils.py`'s `Intersect`, this is
    `WindingPackTotalSize`'s (an earlier pass's node) replacement -- see that class'
    removal note and `_audit/next_steps.md` §7 for why the split is worth doing now.

    `.tfcoil.a_tf_wp_with_insulation`/`.tfcoil.a_tf_wp_no_insulation` are minted here,
    at the same `VarPath`s the pre-split `WindingPackTotalSize` already minted them at
    (unchanged by this split) -- `coils/mass.py`'s `CoilsMass` and `coils/forces.py`'s
    `MaxForceDensity` (etc.) already declared `From`s at exactly these two paths; this
    node is still their producer. See that removed class' own docstring (preserved
    below in this module's history/`calculate.md`) for the full reasoning, including the
    real port bug (`CoilCrossSectionalArea`'s `a_tf_wp_with_insulation` `From`) that
    discovering this producer's correct path fixed.

    **Owns `.tfcoil.j_tf_wp`.** Unlike the pre-intersect half, nothing in
    `winding_pack_post_intersect` *reads* `j_tf_wp` (the material dispatch that does is
    entirely upstream, in `WindingPackIntersectInputs`), but it does *produce* the fresh
    `j_tf_wp_new` value -- previously discarded here because an earlier pass gave sole
    ownership of `.tfcoil.j_tf_wp` to a separate `WindingPackJTfWp` `FixedPointFunction`
    that duplicated this entire computation just to isolate that one value. That class is
    gone; this node now declares `j_tf_wp` as an ordinary `Output` instead, and
    `WindingPackIntersectInputs` reads the real `.tfcoil.j_tf_wp` as an ordinary `From`
    -- together with `coils.py`'s `Intersect` sitting between them, this closes a genuine
    multi-node cycle (see `winding_pack_total_size`'s own docstring), not a self-loop on
    one node, so no `FixedPointFunction`/`Cut` is needed here either.
    """

    b_tf_inboard_peak_symmetric = OutputInto(tfcoil)
    dx_tf_wp_primary_toroidal = OutputInto(tfcoil)
    dx_tf_wp_secondary_toroidal = OutputInto(tfcoil)
    dr_tf_wp_with_insulation = OutputInto(tfcoil)
    j_tf_wp = OutputInto(tfcoil)
    n_tf_coil_turns = OutputInto(tfcoil)
    c_tf_turn = OutputInto(tfcoil)
    a_tf_wp_conductor = OutputInto(tfcoil)
    a_tf_wp_extra_void = OutputInto(tfcoil)
    a_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    a_tf_wp_steel = OutputInto(tfcoil)
    a_tf_wp_no_insulation = OutputInto(tfcoil)
    a_tf_wp_with_insulation = OutputInto(tfcoil)

    def __call__(
        self,
        wp_width_r_min=From(stellarator),
        r_coil_major=From(stellarator),
        r_coil_minor=From(stellarator),
        coilcurrent=From(stellarator),
        n_tf_coils=From(tfcoil),
        stella_config_a1=From(stellarator_config),
        stella_config_a2=From(stellarator_config),
        stella_config_wp_ratio=From(stellarator_config),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        dx_tf_turn_general=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
    ):
        # `winding_pack_post_intersect`'s return tuple is already in this exact order
        # (see its own docstring) -- the unpack-then-repack this used to do was an
        # identity transform, so the declaration now just delegates directly.
        return winding_pack_post_intersect(
            wp_width_r_min,
            r_coil_major,
            r_coil_minor,
            coilcurrent,
            n_tf_coils,
            stella_config_a1,
            stella_config_a2,
            stella_config_wp_ratio,
            f_a_tf_turn_cable_space_extra_void,
            a_tf_turn_cable_space_no_void,
            dx_tf_turn_general,
            dx_tf_wp_insulation,
            a_tf_turn_steel,
        )


class CoilCasing(ExplicitFunction):
    dr_tf_plasma_case = OutputInto(tfcoil)
    dx_tf_side_case_min = OutputInto(tfcoil)

    def __call__(self, dr_tf_nose_case=From(tfcoil)):
        return calculate_casing(dr_tf_nose_case)


class VerticalPorts(ExplicitFunction):
    vporttmax = OutputInto(stellarator)
    vportpmax = OutputInto(stellarator)
    vportamax = OutputInto(stellarator)

    def __call__(
        self,
        stella_config_max_portsize_width=From(stellarator_config),
        f_st_rmajor=From(stellarator),
        f_st_n_coils=From(stellarator),
    ):
        return calculate_vertical_ports(
            stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
        )


class HorizontalPorts(ExplicitFunction):
    hporttmax = OutputInto(stellarator)
    hportpmax = OutputInto(stellarator)
    hportamax = OutputInto(stellarator)

    def __call__(
        self,
        stella_config_max_portsize_width=From(stellarator_config),
        f_st_rmajor=From(stellarator),
        f_st_n_coils=From(stellarator),
    ):
        return calculate_horizontal_ports(
            stella_config_max_portsize_width, f_st_rmajor, f_st_n_coils
        )


class ZTfInsideHalf(ExplicitFunction):
    """cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`.

    **Why this node, not `build.py`'s `Build`, owns this field**: real PROCESS has two
    independent writers of `.build.z_tf_inside_half` -- `st_build`'s formula (what
    `Build` computes) and `st_coil`'s formula (what this node computes, ported here).
    `stellarator.py`'s `run()` calls them in opposite order depending on the `output`
    flag; every real run ends with an `output=True` report pass that runs `st_build`
    then `st_coil`, so `st_coil`'s value is what survives into the converged answer --
    confirmed directly against a real run via the block-by-block MDA-vs-PROCESS
    comparison harness (`functional_process/mda_harness.py`), which caught `Build`
    claiming this field under the wrong (transient, `st_build`) formula. See
    `build.py`'s `calculate_build`/`Build` docstrings for the fuller account, and
    `_audit/next_steps.md` §5 for this session's other "ordering artifact" findings --
    same shape: two producers, one wins by call order, not represented structurally
    until now.
    """

    z_tf_inside_half = OutputInto(build)

    def __call__(
        self,
        stella_config_maximal_coil_height=From(stellarator_config),
        r_coil_minor=From(stellarator),
        stella_config_coil_rminor=From(stellarator_config),
    ):
        return calculate_z_tf_inside_half(
            stella_config_maximal_coil_height, r_coil_minor, stella_config_coil_rminor
        )


class LenTfCoil(ExplicitFunction):
    """cottax node: `calculate_len_tf_coil`, owning `.tfcoil.len_tf_coil`.

    Carved out of `st_coil`'s inline geometry block like `ZTfInsideHalf` and
    `TfCryoArea`. Four registered nodes read `.tfcoil.len_tf_coil` -- `StructureMasses`,
    `PlasmaFacingCoilArea`, `CoilsMass`, `TfMagnetCostSuperconducting` -- and until this
    landed it was a **boundary input** with no producer, so all four consumed a frozen
    seed. Cold, that seed is `0.0`, which is what made
    `TfMagnetCostSuperconducting`'s `.costs.c22211`/`.c2221` come out `nan`
    (`costs.md`'s cold-start finding): every coil mass in `coils/mass.py` is
    proportional to `len_tf_coil`, so `costtfcu = uccu * m_tf_coil_copper /
    (len_tf_coil * n_tf_coil_turns)` is `0.0 / 0.0` there.

    **The stale-vs-fresh question this node was held back for, resolved.**
    `st_coil` calls `calculate_plasma_facing_coil_area` at
    `process/models/stellarator/coils/calculate.py:68`, **19 lines before** `:87` writes
    `len_tf_coil` -- so within one `Caller` round `PlasmaFacingCoilArea` reads the
    *previous* round's value, and the eager port preserves that faithfully with a
    separate `len_tf_coil_stale` parameter (`calculate.md:124`). Giving the field a
    producer switches the declared `PlasmaFacingCoilArea` node from stale to fresh, and
    the question was whether that needs modelling as a `FixedPointFunction` self-loop.

    **It does not, and the reason is structural rather than numerical.** There is no
    feedback path: `len_tf_coil`'s own inputs are two `stellarator_config` boundary
    values plus `.stellarator.r_coil_minor`/`.tfcoil.n_tf_coils`, owned by
    `StellaratorScalingFactors`, which is **not reachable from any of the four readers**
    (measured -- `_audit/boundary_inputs_audit.md` §4c (c1)). So the loop equation would
    be `x = g()` with `g` not depending on `x`: a degenerate fixed point, which
    `sand.degenerate_fixed_points` drops on sight, exactly as it already drops
    `EtaTurbineStep` and `CplifeAvail`. Modelling PROCESS's read-before-write as a cycle
    would not be more faithful -- it would add a block that is deleted for being an
    identity. The staleness is a property of PROCESS's Gauss-Seidel *schedule*, not of
    the dependency structure, and this port does not model PROCESS's round structure at
    all.

    The honest caveat: `Caller.call_models` checks idempotence on the objective and
    constraints at `rtol=1e-6`, not per field, so stale and fresh can differ
    *transiently* while upstream is still moving. They cannot differ at a converged
    point, which is what every harness here compares.
    """

    len_tf_coil = OutputInto(tfcoil)

    def __call__(
        self,
        stella_config_coillength=From(stellarator_config),
        r_coil_minor=From(stellarator),
        stella_config_coil_rminor=From(stellarator_config),
        n_tf_coils=From(tfcoil),
    ):
        return calculate_len_tf_coil(
            stella_config_coillength,
            r_coil_minor,
            stella_config_coil_rminor,
            n_tf_coils,
        )


class TfCryoArea(ExplicitFunction):
    """cottax node: `calculate_tfcryoarea`, owning `.tfcoil.tfcryoarea`.

    Carved out of `st_coil`'s inline geometry block exactly as `ZTfInsideHalf` (above)
    was, and for the same structural reason: the eager `st_coil` orchestrator is
    deliberately not registered, so a formula that lives only inside it has no owner
    in the graph and its output stays a boundary input.

    **Why it was worth carving out now.** `.tfcoil.tfcryoarea` is an input of
    `thermal_cryo.py`'s cryogenic-load nodes (`CryoQLoadsStep`, via
    `Power.cryo`'s `qss` term). Registering those without this node would have traded
    two boundary inputs (`.heat_transport.helpow`,
    `.heat_transport.p_cryo_plant_electric_mw`) for one new one -- see
    `_audit/boundary_inputs_audit.md` §4c (c1)'s "sibling gap in the same three lines"
    and §7 items 4 and 7.

    **Its two siblings in the same block are deliberately left alone**:
    `.tfcoil.len_tf_coil` carries an unresolved stale-vs-fresh design decision
    (`PlasmaFacingCoilArea` reads it 19 lines before `st_coil` writes it, and the
    eager port preserves that with a separate `len_tf_coil_stale` parameter --
    `calculate.md:124`), and `min_bending_radius` has no reader at all. `tfcryoarea`
    has neither complication: nothing reads it before `st_coil` writes it.
    """

    tfcryoarea = OutputInto(tfcoil)

    def __call__(
        self,
        stella_config_coilsurface=From(stellarator_config),
        f_st_rmajor=From(stellarator),
        r_coil_minor=From(stellarator),
        stella_config_coil_rminor=From(stellarator_config),
    ):
        return calculate_tfcryoarea(
            stella_config_coilsurface,
            f_st_rmajor,
            r_coil_minor,
            stella_config_coil_rminor,
        )
