"""Pure-functional port of the tier-1 functions in `coils/calculate.py` (registry unit
#9).
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
from functional_process.cottax.stellarator.coils.coils import (
    bmax_from_awp,  # noqa: F401
    intersect,  # noqa: F401
)
from functional_process.cottax.stellarator.coils.forces import (
    calculate_centering_force_avg_mn,  # noqa: F401
    calculate_centering_force_max_mn,  # noqa: F401
    calculate_centering_force_min_mn,  # noqa: F401
    calculate_max_force_density,  # noqa: F401
    calculate_max_force_density_mnm,  # noqa: F401
    calculate_max_lateral_force_density,  # noqa: F401
    calculate_max_radial_force_density,  # noqa: F401
    calculate_maximum_stress,  # noqa: F401
)
from functional_process.cottax.stellarator.coils.mass import (
    calculate_coils_mass,  # noqa: F401
)
from functional_process.cottax.stellarator.coils.quench import (
    calculate_quench_protection,  # noqa: F401
)
from functional_process.cottax.paths import (
    build,
    constraints,
    stellarator,
    stellarator_config,
    tfcoil,
)
from functional_process.models.stellarator.coils.calculate import (
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
    `calculate_coils_summary_variables`.
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
    """The family that owns the *pre*-`intersect` half of `winding_pack_total_size`: the
    sampled `(wp_width_r, lhs, rhs)` curves `coils.py`'s `Intersect`
    (`ImplicitFunction`/`RootFind`) needs as its own `From`s, and the starting guess it
    is driven from.
    """

    wp_width_r = OutputInto(stellarator)
    lhs = OutputInto(stellarator)
    rhs = OutputInto(stellarator)
    wp_width_r_min_guess = OutputInto(stellarator)

    sample_lower_divisor = 40.0
    guess_divisor = 10.0
    """`_MATERIAL_SAMPLING`'s row for this occupant's material, as plain class
    attributes -- the ordinary pair by default, overridden by the one occupant PROCESS
    treats differently.
    """

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
    """`i_tf_sc_mat == ITER_NB3SN` (1) -- PROCESS's own default and this run's value."""

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
    """`i_tf_sc_mat == BI2212` (2)."""

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
    """`i_tf_sc_mat == CROCO_REBCO` (6) -- the one occupant with different sampling."""

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
    """`i_tf_sc_mat == DURHAM_REBCO` (8)."""

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
    """cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`."""

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
    """cottax node: `calculate_len_tf_coil`, owning `.tfcoil.len_tf_coil`."""

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
    """cottax node: `calculate_tfcryoarea`, owning `.tfcoil.tfcryoarea`."""

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
