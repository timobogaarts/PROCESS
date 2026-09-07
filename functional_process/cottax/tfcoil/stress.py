"""Pure-functional port of the tokamak TF coil's **stress chain** --
`process/models/tfcoil/base.py`'s `tf_field_and_force` (`:1623-1821`) and `stresscl`
(`:2222-3274`), plus the four elasticity-smearing helpers and the `plane_stress` layer
solver `stresscl` calls (`:3659-3717`, `:4236-4670`).
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import (
    build,
    physics,
    superconducting_tfcoil,
    tfcoil,
)
from functional_process.models.tfcoil.stress import (
    extended_plane_strain,  # noqa: F401 -- re-exported for tests/.../test_stress.py
    eyoung_parallel,  # noqa: F401 -- re-exported for tests/.../test_stress.py
    eyoung_parallel_array,  # noqa: F401 -- re-exported for tests/.../test_stress.py
    eyoung_series,  # noqa: F401 -- re-exported for tests/.../test_stress.py
    eyoung_t_nested_squares,  # noqa: F401 -- re-exported for tests/.../test_stress.py
    plane_stress,  # noqa: F401 -- re-exported for tests/.../test_stress.py
    tf_field_and_force_clamped_joints,
    tf_stress_extended_plane_strain_bucked_case,
    tf_stress_plane_stress_bucked_case,
)

# ---------------------------------------------------------------------------
# The nodes
# ---------------------------------------------------------------------------


class TfFieldAndForce(ExplicitFunction):
    """The family that owns the TF coil's in-plane force and vertical tension."""


class TfFieldAndForceClampedJoints(TfFieldAndForce):
    """No sliding centrepost joints -- every superconducting coil unless an input file
    sets `i_cp_joints = 1` alongside `itart = 1`.
    """

    cforce = OutputInto(tfcoil)
    vforce = OutputInto(tfcoil)
    vforce_outboard = OutputInto(tfcoil)
    vforce_inboard_tot = OutputInto(superconducting_tfcoil)

    def __call__(
        self,
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_outboard_in=From(superconducting_tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        b_tf_inboard_peak_symmetric=From(tfcoil),
        c_tf_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        rmajor=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        f_vforce_inboard=From(tfcoil),
    ):
        return tf_field_and_force_clamped_joints(
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_outboard_in=r_tf_outboard_in,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            b_tf_inboard_peak_symmetric=b_tf_inboard_peak_symmetric,
            c_tf_total=c_tf_total,
            n_tf_coils=n_tf_coils,
            dr_tf_plasma_case=dr_tf_plasma_case,
            rmajor=rmajor,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            f_vforce_inboard=f_vforce_inboard,
        )


class TfStress(ExplicitFunction):
    """The family that owns the TF coil's peak stresses and strains."""


class TfStressPlaneStressBuckedCase(TfStress):
    """`i_tf_stress_model == 1` (generalised plane stress) with `i_tf_bucking == 1`."""

    n_tf_graded_layers: int = 1

    sig_tf_wp = OutputInto(tfcoil)
    sig_tf_case = OutputInto(tfcoil)
    str_wp = OutputInto(tfcoil)
    casestr = OutputInto(tfcoil)
    insstrain = OutputInto(tfcoil)


class TfStressPlaneStressBuckedCaseAveragedTurn(TfStressPlaneStressBuckedCase):
    """`i_tf_turns_integer == 0` -- the turn is described by one averaged cable-space
    width, `.superconducting_tfcoil.dx_tf_turn_cable_space_average`
    (`base.py:2745-2749`).
    """

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        a_tf_coil_inboard_steel=From(superconducting_tfcoil),
        a_tf_plasma_case=From(superconducting_tfcoil),
        a_tf_coil_nose_case=From(superconducting_tfcoil),
        eyoung_steel=From(tfcoil),
        poisson_steel=From(tfcoil),
        eyoung_cond_axial=From(tfcoil),
        poisson_cond_axial=From(tfcoil),
        eyoung_cond_trans=From(tfcoil),
        poisson_cond_trans=From(tfcoil),
        eyoung_ins=From(tfcoil),
        poisson_ins=From(tfcoil),
        eyoung_copper=From(tfcoil),
        poisson_copper=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        dx_tf_turn_cable_space_average=From(superconducting_tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_side_case_average=From(superconducting_tfcoil),
        dx_tf_wp_toroidal_average=From(superconducting_tfcoil),
        a_tf_coil_inboard_insulation=From(superconducting_tfcoil),
        a_tf_wp_steel=From(tfcoil),
        a_tf_wp_conductor=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        c_tf_total=From(tfcoil),
        vforce=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
    ):
        return tf_stress_plane_stress_bucked_case(
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            tan_theta_coil=tan_theta_coil,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            dr_tf_plasma_case=dr_tf_plasma_case,
            a_tf_coil_inboard_steel=a_tf_coil_inboard_steel,
            a_tf_plasma_case=a_tf_plasma_case,
            a_tf_coil_nose_case=a_tf_coil_nose_case,
            eyoung_steel=eyoung_steel,
            poisson_steel=poisson_steel,
            eyoung_cond_axial=eyoung_cond_axial,
            poisson_cond_axial=poisson_cond_axial,
            eyoung_cond_trans=eyoung_cond_trans,
            poisson_cond_trans=poisson_cond_trans,
            eyoung_ins=eyoung_ins,
            poisson_ins=poisson_ins,
            eyoung_copper=eyoung_copper,
            poisson_copper=poisson_copper,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            n_tf_coil_turns=n_tf_coil_turns,
            dx_tf_turn_cable_space_eyoung=dx_tf_turn_cable_space_average,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_side_case_average=dx_tf_side_case_average,
            dx_tf_wp_toroidal_average=dx_tf_wp_toroidal_average,
            a_tf_coil_inboard_insulation=a_tf_coil_inboard_insulation,
            a_tf_wp_steel=a_tf_wp_steel,
            a_tf_wp_conductor=a_tf_wp_conductor,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            c_tf_total=c_tf_total,
            vforce=vforce,
            a_tf_coil_inboard_case=a_tf_coil_inboard_case,
            a_tf_turn_steel=a_tf_turn_steel,
            n_tf_graded_layers=self.n_tf_graded_layers,
        )


class TfStressPlaneStressBuckedCaseIntegerTurn(TfStressPlaneStressBuckedCase):
    """`i_tf_turns_integer == 1` -- rectangular turns on a fixed layers x pancakes grid,
    so the smearing uses the *radial* cable-space dimension
    `.superconducting_tfcoil.dr_tf_turn_cable_space` instead (`base.py:2745-2749`).
    """

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        a_tf_coil_inboard_steel=From(superconducting_tfcoil),
        a_tf_plasma_case=From(superconducting_tfcoil),
        a_tf_coil_nose_case=From(superconducting_tfcoil),
        eyoung_steel=From(tfcoil),
        poisson_steel=From(tfcoil),
        eyoung_cond_axial=From(tfcoil),
        poisson_cond_axial=From(tfcoil),
        eyoung_cond_trans=From(tfcoil),
        poisson_cond_trans=From(tfcoil),
        eyoung_ins=From(tfcoil),
        poisson_ins=From(tfcoil),
        eyoung_copper=From(tfcoil),
        poisson_copper=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        dr_tf_turn_cable_space=From(superconducting_tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_side_case_average=From(superconducting_tfcoil),
        dx_tf_wp_toroidal_average=From(superconducting_tfcoil),
        a_tf_coil_inboard_insulation=From(superconducting_tfcoil),
        a_tf_wp_steel=From(tfcoil),
        a_tf_wp_conductor=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        c_tf_total=From(tfcoil),
        vforce=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
    ):
        return tf_stress_plane_stress_bucked_case(
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            tan_theta_coil=tan_theta_coil,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            dr_tf_plasma_case=dr_tf_plasma_case,
            a_tf_coil_inboard_steel=a_tf_coil_inboard_steel,
            a_tf_plasma_case=a_tf_plasma_case,
            a_tf_coil_nose_case=a_tf_coil_nose_case,
            eyoung_steel=eyoung_steel,
            poisson_steel=poisson_steel,
            eyoung_cond_axial=eyoung_cond_axial,
            poisson_cond_axial=poisson_cond_axial,
            eyoung_cond_trans=eyoung_cond_trans,
            poisson_cond_trans=poisson_cond_trans,
            eyoung_ins=eyoung_ins,
            poisson_ins=poisson_ins,
            eyoung_copper=eyoung_copper,
            poisson_copper=poisson_copper,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            n_tf_coil_turns=n_tf_coil_turns,
            dx_tf_turn_cable_space_eyoung=dr_tf_turn_cable_space,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_side_case_average=dx_tf_side_case_average,
            dx_tf_wp_toroidal_average=dx_tf_wp_toroidal_average,
            a_tf_coil_inboard_insulation=a_tf_coil_inboard_insulation,
            a_tf_wp_steel=a_tf_wp_steel,
            a_tf_wp_conductor=a_tf_wp_conductor,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            c_tf_total=c_tf_total,
            vforce=vforce,
            a_tf_coil_inboard_case=a_tf_coil_inboard_case,
            a_tf_turn_steel=a_tf_turn_steel,
            n_tf_graded_layers=self.n_tf_graded_layers,
        )


class TfStressExtendedPlaneStrainBuckedCaseAveragedTurn(TfStress):
    """`i_tf_stress_model == 0` (extended plane strain) with `i_tf_bucking == 1` and
    `i_tf_turns_integer == 0`.
    """

    n_tf_graded_layers: int = 1

    sig_tf_wp = OutputInto(tfcoil)
    sig_tf_case = OutputInto(tfcoil)
    str_wp = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_in=From(build),
        r_tf_wp_inboard_inner=From(superconducting_tfcoil),
        r_tf_wp_inboard_outer=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        dr_tf_plasma_case=From(tfcoil),
        a_tf_coil_inboard_steel=From(superconducting_tfcoil),
        a_tf_plasma_case=From(superconducting_tfcoil),
        a_tf_coil_nose_case=From(superconducting_tfcoil),
        eyoung_steel=From(tfcoil),
        poisson_steel=From(tfcoil),
        eyoung_cond_axial=From(tfcoil),
        poisson_cond_axial=From(tfcoil),
        eyoung_cond_trans=From(tfcoil),
        poisson_cond_trans=From(tfcoil),
        eyoung_ins=From(tfcoil),
        poisson_ins=From(tfcoil),
        eyoung_copper=From(tfcoil),
        poisson_copper=From(tfcoil),
        dx_tf_turn_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        dx_tf_turn_cable_space_average=From(superconducting_tfcoil),
        dia_tf_turn_coolant_channel=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        dx_tf_turn_steel=From(tfcoil),
        dx_tf_side_case_average=From(superconducting_tfcoil),
        dx_tf_wp_toroidal_average=From(superconducting_tfcoil),
        a_tf_coil_inboard_insulation=From(superconducting_tfcoil),
        a_tf_wp_steel=From(tfcoil),
        a_tf_wp_conductor=From(tfcoil),
        a_tf_wp_with_insulation=From(superconducting_tfcoil),
        c_tf_total=From(tfcoil),
        vforce_inboard_tot=From(superconducting_tfcoil),
    ):
        return tf_stress_extended_plane_strain_bucked_case(
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_wp_inboard_inner=r_tf_wp_inboard_inner,
            r_tf_wp_inboard_outer=r_tf_wp_inboard_outer,
            tan_theta_coil=tan_theta_coil,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            dr_tf_plasma_case=dr_tf_plasma_case,
            a_tf_coil_inboard_steel=a_tf_coil_inboard_steel,
            a_tf_plasma_case=a_tf_plasma_case,
            a_tf_coil_nose_case=a_tf_coil_nose_case,
            eyoung_steel=eyoung_steel,
            poisson_steel=poisson_steel,
            eyoung_cond_axial=eyoung_cond_axial,
            poisson_cond_axial=poisson_cond_axial,
            eyoung_cond_trans=eyoung_cond_trans,
            poisson_cond_trans=poisson_cond_trans,
            eyoung_ins=eyoung_ins,
            poisson_ins=poisson_ins,
            eyoung_copper=eyoung_copper,
            poisson_copper=poisson_copper,
            dx_tf_turn_insulation=dx_tf_turn_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            n_tf_coil_turns=n_tf_coil_turns,
            dx_tf_turn_cable_space_eyoung=dx_tf_turn_cable_space_average,
            dia_tf_turn_coolant_channel=dia_tf_turn_coolant_channel,
            f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
            dx_tf_turn_steel=dx_tf_turn_steel,
            dx_tf_side_case_average=dx_tf_side_case_average,
            dx_tf_wp_toroidal_average=dx_tf_wp_toroidal_average,
            a_tf_coil_inboard_insulation=a_tf_coil_inboard_insulation,
            a_tf_wp_steel=a_tf_wp_steel,
            a_tf_wp_conductor=a_tf_wp_conductor,
            a_tf_wp_with_insulation=a_tf_wp_with_insulation,
            c_tf_total=c_tf_total,
            vforce_inboard_tot=vforce_inboard_tot,
            n_tf_graded_layers=self.n_tf_graded_layers,
        )
