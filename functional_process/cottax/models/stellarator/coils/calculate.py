"""Pure-functional port of the tier-1 functions in `coils/calculate.py` (registry unit
#9).
"""

import jax.numpy as jnp  # noqa: F401
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.models.stellarator.coils.coils import (
    bmax_from_awp,  # noqa: F401
    intersect,  # noqa: F401
)
from functional_process.cottax.models.stellarator.coils.forces import (
    calculate_centering_force_avg_mn,  # noqa: F401
    calculate_centering_force_max_mn,  # noqa: F401
    calculate_centering_force_min_mn,  # noqa: F401
    calculate_max_force_density,  # noqa: F401
    calculate_max_force_density_mnm,  # noqa: F401
    calculate_max_lateral_force_density,  # noqa: F401
    calculate_max_radial_force_density,  # noqa: F401
    calculate_maximum_stress,  # noqa: F401
)
from functional_process.cottax.models.stellarator.coils.mass import (
    calculate_coils_mass,  # noqa: F401
)
from functional_process.cottax.models.stellarator.coils.quench import (
    calculate_quench_protection,  # noqa: F401
)
from functional_process.cottax.paths import (
    build,
    constraints,
    stellarator,
    stellarator_config,
    tfcoil,
)
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.physics.superconductors import (
    bi2212,  # noqa: F401
    gl_nbti,  # noqa: F401
    gl_rebco,  # noqa: F401
    itersc,  # noqa: F401
    jcrit_nbti,  # noqa: F401
    jcrit_rebco,  # noqa: F401
    western_superconducting_nb3sn,  # noqa: F401
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


class CoilToroidalThickness(WrapsFunction):
    fn = calculate_coil_toroidal_thickness

    dx_tf_wp_primary_toroidal = From(tfcoil)
    dx_tf_side_case_min = From(tfcoil)
    dx_tf_wp_insulation = From(tfcoil)

    dx_tf_inboard_out_toroidal = OutputInto(tfcoil)


class CoilRadialThickness(WrapsFunction):
    fn = calculate_coil_radial_thickness

    dr_tf_nose_case = From(tfcoil)
    dr_tf_wp_with_insulation = From(tfcoil)
    dr_tf_plasma_case = From(tfcoil)
    dx_tf_wp_insulation = From(tfcoil)

    dr_tf_inboard = OutputInto(build)


class CoilCrossSectionalArea(WrapsFunction):
    fn = calculate_coil_cross_sectional_area

    a_tf_wp_with_insulation = From(tfcoil)
    dr_tf_inboard = From(build)
    dx_tf_inboard_out_toroidal = From(tfcoil)

    a_tf_leg_outboard = OutputInto(tfcoil)
    a_tf_coil_inboard_case = OutputInto(tfcoil)


class CoilHalfWidths(WrapsFunction):
    fn = calculate_coil_half_widths

    dx_tf_inboard_out_toroidal = From(tfcoil)

    tfocrn = OutputInto(tfcoil)
    tficrn = OutputInto(tfcoil)


class PlasmaFacingCoilArea(WrapsFunction):
    fn = calculate_plasma_facing_coil_area

    n_tf_coils = From(tfcoil)
    dx_tf_inboard_out_toroidal = From(tfcoil)
    len_tf_coil = From(tfcoil)

    tfsai = OutputInto(tfcoil)
    tfsao = OutputInto(tfcoil)


class CoilCoilToroidalGap(WrapsFunction):
    fn = select_coil_coil_toroidal_gap

    stella_config_dmin = From(stellarator_config)
    r_coil_major = From(stellarator)
    r_coil_minor = From(stellarator)
    stella_config_coil_rmajor = From(stellarator_config)
    stella_config_coil_rminor = From(stellarator_config)
    dx_tf_inboard_out_toroidal = From(tfcoil)

    toroidalgap = OutputInto(tfcoil)


class CoilsSummaryVariables(WrapsFunction):
    fn = calculate_coils_summary_variables

    n_tf_coils = From(tfcoil)
    a_tf_leg_outboard = From(tfcoil)
    coilcurrent = From(stellarator)
    r_coil_major = From(stellarator)
    r_coil_minor = From(stellarator)
    awp_rad = FromExactly(tfcoil.dr_tf_wp_with_insulation)

    a_tf_inboard_total = OutputInto(tfcoil)
    c_tf_total = OutputInto(tfcoil)
    j_tf_coil_full_area = OutputInto(tfcoil)
    r_b_tf_inboard_peak_symmetric = OutputInto(tfcoil)


class StoredMagneticEnergy(WrapsFunction):
    fn = calculate_stored_magnetic_energy

    stella_config_inductance = From(stellarator_config)
    f_st_rmajor = From(stellarator)
    r_coil_minor = From(stellarator)
    stella_config_coil_rminor = From(stellarator_config)
    f_st_n_coils = From(stellarator)
    c_tf_total = From(tfcoil)
    n_tf_coils = From(tfcoil)

    e_tf_magnetic_stored_total_gj = OutputInto(tfcoil)


class WindingPackGeometry(WrapsFunction):
    fn = calculate_winding_pack_geometry

    dx_tf_turn_general = From(tfcoil)
    dx_tf_turn_steel = From(tfcoil)
    dx_tf_turn_insulation = From(tfcoil)

    a_tf_turn_cable_space_no_void = OutputInto(tfcoil)
    a_tf_turn_steel = OutputInto(tfcoil)


class CoilCurrent(WrapsFunction):
    """`coilcurrent` has no PROCESS storage location -- it is a local in `st_coil`,
    threaded manually into `winding_pack_total_size` and
    `calculate_coils_summary_variables`.
    """

    fn = calculate_current

    f_st_b = From(stellarator)
    stella_config_i0 = From(stellarator_config)
    f_st_rmajor = From(stellarator)
    f_st_n_coils = From(stellarator)

    coilcurrent = OutputInto(stellarator)
    f_st_i_total = OutputInto(stellarator)


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
        """The occupant's four outputs, from its own `jcrit` law and its own divisors."""
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


class WindingPackTotalSizePost(WrapsFunction):
    """cottax node: the *post*-`intersect` half of `winding_pack_total_size` --
    everything downstream of the resolved crossing point.

    `winding_pack_post_intersect`'s return tuple is already in this exact order (see its
    own docstring), so the declaration below delegates directly.
    """

    fn = winding_pack_post_intersect

    wp_width_r_min = From(stellarator)
    r_coil_major = From(stellarator)
    r_coil_minor = From(stellarator)
    coilcurrent = From(stellarator)
    n_tf_coils = From(tfcoil)
    stella_config_a1 = From(stellarator_config)
    stella_config_a2 = From(stellarator_config)
    stella_config_wp_ratio = From(stellarator_config)
    f_a_tf_turn_cable_space_extra_void = From(tfcoil)
    a_tf_turn_cable_space_no_void = From(tfcoil)
    dx_tf_turn_general = From(tfcoil)
    dx_tf_wp_insulation = From(tfcoil)
    a_tf_turn_steel = From(tfcoil)

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


class CoilCasing(WrapsFunction):
    fn = calculate_casing

    dr_tf_nose_case = From(tfcoil)

    dr_tf_plasma_case = OutputInto(tfcoil)
    dx_tf_side_case_min = OutputInto(tfcoil)


class VerticalPorts(WrapsFunction):
    fn = calculate_vertical_ports

    stella_config_max_portsize_width = From(stellarator_config)
    f_st_rmajor = From(stellarator)
    f_st_n_coils = From(stellarator)

    vporttmax = OutputInto(stellarator)
    vportpmax = OutputInto(stellarator)
    vportamax = OutputInto(stellarator)


class HorizontalPorts(WrapsFunction):
    fn = calculate_horizontal_ports

    stella_config_max_portsize_width = From(stellarator_config)
    f_st_rmajor = From(stellarator)
    f_st_n_coils = From(stellarator)

    hporttmax = OutputInto(stellarator)
    hportpmax = OutputInto(stellarator)
    hportamax = OutputInto(stellarator)


class ZTfInsideHalf(WrapsFunction):
    """cottax node: `calculate_z_tf_inside_half`, owning `.build.z_tf_inside_half`."""

    fn = calculate_z_tf_inside_half

    stella_config_maximal_coil_height = From(stellarator_config)
    r_coil_minor = From(stellarator)
    stella_config_coil_rminor = From(stellarator_config)

    z_tf_inside_half = OutputInto(build)


class LenTfCoil(WrapsFunction):
    """cottax node: `calculate_len_tf_coil`, owning `.tfcoil.len_tf_coil`."""

    fn = calculate_len_tf_coil

    stella_config_coillength = From(stellarator_config)
    r_coil_minor = From(stellarator)
    stella_config_coil_rminor = From(stellarator_config)
    n_tf_coils = From(tfcoil)

    len_tf_coil = OutputInto(tfcoil)


class TfCryoArea(WrapsFunction):
    """cottax node: `calculate_tfcryoarea`, owning `.tfcoil.tfcryoarea`."""

    fn = calculate_tfcryoarea

    stella_config_coilsurface = From(stellarator_config)
    f_st_rmajor = From(stellarator)
    r_coil_minor = From(stellarator)
    stella_config_coil_rminor = From(stellarator_config)

    tfcryoarea = OutputInto(tfcoil)
