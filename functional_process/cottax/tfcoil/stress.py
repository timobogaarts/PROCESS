"""Pure-functional port of the tokamak TF coil's **stress chain** --
`process/models/tfcoil/base.py`'s `tf_field_and_force` (`:1623-1821`) and `stresscl`
(`:2222-3274`), plus the four elasticity-smearing helpers and the `plane_stress` layer
solver `stresscl` calls (`:3659-3717`, `:4236-4670`).

Audit record: `functional_process/_audit/units/models/tfcoil/stress.md`.

**Why this is a module of its own and not part of `models/tfcoil/base.py`.** That file's
scope is the minimal closure of `.tokamak.cicc_superconducting_tf_coil`'s ten *boundary*
reads, and its docstring excludes both of these functions for one stated reason -- "feeds
only stresses, which no boundary read depends on". That was true of the boundary and
false of the **constraint surface**, which was added later: `large_tokamak_nof.IN.DAT:
146-147` activates constraints **31** (`sig_tf_case <= sig_tf_case_max`) and **32**
(`sig_tf_wp <= sig_tf_wp_max`), and with no producer for either operand the port
evaluated both as `0 <= max` -- two *dropped* constraints, not two wrong numbers, which
is worse because no residual reports it. `.tfcoil.str_wp` is the third: at
`i_str_wp == 1` (PROCESS's default, `tfcoil_variables.py:508`) it is the strain the
Nb3Sn critical-current surface (constraint 33) and the temperature margin (constraint
36) read, and **zero strain is the peak of that fit**, so its absence was optimistic
rather than neutral.

## What is ported: one cell of a four-switch space

`stresscl` has 65 parameters and four internal switches. This module writes the single
cell the tracked tokamaks take, and nothing else:

| switch | value ported | why the others are not here |
|---|---|---|
| `i_tf_sup` | `1` (superconducting) | **resolved above this file.** `caller.py:295-316` picks `CICCSuperconductingTFCoil` at `i_tf_sup == 1`; a resistive machine has a different occupant of `.tokamak`'s TF slot entirely, not a different arm of this node |
| `i_tf_stress_model` | `1` (generalised plane stress) **and** `0` (extended plane strain) | Two solvers, two wrappers, two occupants -- `plane_stress` and `extended_plane_strain` (`base.py:3719-4234`). `2` reaches the same code as `0` on every tracked value but is refused rather than aliased, because nobody has measured a file that sets it |
| `i_tf_bucking` | `1` (nose casing bucks, no CS layer) | `>= 2` (bucked-and-wedged) prepends a **central-solenoid** layer whose properties are rebuilt from scratch out of nine `.pf_coil` fields (`base.py:2531-2650`); `3` adds a Kapton interlayer on top. Neither reads-set is written |
| `i_tf_tresca` | **not read at all** | measured, not assumed: its two branches (`base.py:3196`, `:3218`) are both gated on `ii >= i_tf_bucking + 1`, and the two layers this node reports are `n_tf_bucking` and `n_tf_bucking - 1`. Neither can reach the gate, so no CEA out-of-plane correction and no von Mises array is computed here |
| `i_tf_turns_integer` | **both arms written** | it picks *which field* the cable-space width for the transverse smearing comes from -- `.superconducting_tfcoil.dx_tf_turn_cable_space_average` at `0`, `.superconducting_tfcoil.dr_tf_turn_cable_space` at `1` (`base.py:2745-2749`). A different read, so a different occupant; `low_aspect_ratio_DEMO` is the integer one and it assembles today |

The two `i_tf_stress_model` wrappers are **siblings, not a switch**: the plane-strain
one reads the axial moduli and `.superconducting_tfcoil.vforce_inboard_tot` where the
plane-stress one reads neither, does not read `.tfcoil.a_tf_coil_inboard_case` /
`.tfcoil.a_tf_turn_steel` at all, and owns three fields rather than five. Only its
averaged-turn arm is written; `(0, 1, 1)` is refused, because both tracked spherical
tokamaks set `i_tf_turns_integer = 0`.

`tf_field_and_force` carries a fifth: `itart == 1 and i_cp_joints == 1` (a spherical
tokamak with *sliding* centrepost joints) splits the vertical tension between centrepost
and legs by a different closed form and **owns `.tfcoil.f_vforce_inboard`**, which the
clamped-joint arm reads and returns unchanged. That arm is
unreachable through `machine_from_indat` for any occupant of this namespace:
`init.py:752-756` resolves `i_cp_joints == -1` (the default,
`tfcoil_variables.py:589`) to `0` for every superconducting coil, so only an input file
that sets `i_cp_joints = 1` *and* `itart = 1` on a superconducting machine reaches it,
and none of the tracked files does. Refused in `indat.py` rather than left inferred.

## What is deliberately not returned

`stresscl` returns 34 values. This node owns **five** -- the three the constraint
surface reads (`sig_tf_wp`, `sig_tf_case`, `str_wp`) and the two strains PROCESS stores
beside them (`casestr`, `insstrain`, `base.py:2991-2998`, one line each off quantities
already computed). The other 29 are either

* **reporting arrays** consumed only by `out_stress` (`base.py:3275`) -- the nine
  `sig_tf_*`/`str_tf_*` radial distributions, `radial_array`, `deflect`,
  `s_shear_cea_tf_cond`, the smeared moduli. PROCESS stores none of them in
  `DataStructure` on this arm; they are local variables of
  `SuperconductingTFCoil.tf_stress` passed straight to the printer
  (`superconducting.py:2105-2138`), so there is no `VarPath` to own; or
* **`sig_tf_cs_bucked`**, which `stresscl` leaves as `None` unless `i_tf_bucking >= 2`
  (`base.py:3230-3231`) -- conditional ownership, and this node is the arm that does not
  own it.

`.tfcoil.n_rad_per_layer` is likewise **not** a port: it is an `InputVariable`
(`input.py:1074`, range 1-500) that `SuperconductingTFCoil.tf_stress` overwrites with
`500` unconditionally at `superconducting.py:2100` before every call, so on this path it
is a constant of the model and not of the run. `N_RADIAL_ARRAY` below.

## JAX notes

Nothing here is untraceable -- no CoolProp, no external call. Three transcriptions are
not literal and each is marked at its site:

1. `plane_stress`'s `np.linalg.solve` becomes `jnp.linalg.solve` on the same
   row-equilibrated matrix. Same LU-with-partial-pivoting; agreement is to rounding, not
   to the bit, and PROCESS's own comment at `base.py:4404-4412` says as much about its
   own portability.
2. The `argmax` reduction over the radial array (`base.py:3199-3226`) becomes
   `jnp.argmax` over a `(n_tf_layer, n_radial_array)` reshape, with the degenerate case
   PROCESS's `ii_max = 0` initialiser produces spelled out -- see `_layer_peak_indices`.
3. `eyoung_series`'s division-by-zero guard is the double-`jnp.where` idiom of
   `models/safe_math.py`, and it is **load-bearing on the reference run**:
   `.tfcoil.eyoung_cond_axial` and `.tfcoil.eyoung_cond_trans` are both `0.0` there, so
   the zero branch is the one taken and an unguarded `l / eyoung` would put `inf` into
   the primal and `nan` into every tangent.
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
    """The family that owns the TF coil's in-plane force and vertical tension.

    `(itart, i_cp_joints)` decides it, and only the clamped-joint arm is written --
    see this module's docstring. A family base rather than a bare class because the
    two arms own *different* fields: the sliding-joint arm additionally owns
    `.tfcoil.f_vforce_inboard`, which this one reads.
    """


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
    """The family that owns the TF coil's peak stresses and strains.

    `(i_tf_stress_model, i_tf_bucking)` decides it and only `(1, 1)` is written. A
    family base for the same reason as `TfFieldAndForce`: the bucked-and-wedged arms
    additionally own `.tfcoil.sig_tf_cs_bucked`, which `stresscl` leaves as `None` here
    (`base.py:3230-3231`).
    """


class TfStressPlaneStressBuckedCase(TfStress):
    """`i_tf_stress_model == 1` (generalised plane stress) with `i_tf_bucking == 1`.

    The reference arm of `large_tokamak_eval`, `large_tokamak_nof` and
    `low_aspect_ratio_DEMO`: none of the three sets either switch, so both take their
    defaults (`tfcoil_variables.py:211`, and `-1` resolved to `1` at
    `init.py:891-895`).

    Abstract: `i_tf_turns_integer` splits it once more into the two subclasses below,
    which differ in exactly one read -- the cable-space width the transverse smearing is
    built on. Everything else, including all five outputs, is shared.

    `n_tf_graded_layers` is a static field rather than a `__call__` parameter -- a
    count that fixes the layer stack's *shape*, so it cannot be a traced input.
    """

    n_tf_graded_layers: int = 1

    sig_tf_wp = OutputInto(tfcoil)
    sig_tf_case = OutputInto(tfcoil)
    str_wp = OutputInto(tfcoil)
    casestr = OutputInto(tfcoil)
    insstrain = OutputInto(tfcoil)


class TfStressPlaneStressBuckedCaseAveragedTurn(TfStressPlaneStressBuckedCase):
    """`i_tf_turns_integer == 0` -- the turn is described by one averaged cable-space
    width, `.superconducting_tfcoil.dx_tf_turn_cable_space_average`
    (`base.py:2745-2749`). `large_tokamak_eval`'s and `large_tokamak_nof`'s arm.
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
    `low_aspect_ratio_DEMO`'s arm.

    Identical to its sibling in every other read and in all five outputs. Written out
    rather than parameterised because a read is a declaration: the two arms read
    different fields, and on this one `dx_tf_turn_cable_space_average` has no producer
    at all (`CiccIntegerTurnGeometry` does not own it), so declaring it would put a
    stale `DataStructure` value into the answer -- the exact defect
    `functional_process/cottax/boundary.py` exists to catch.
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

    The arm of both tracked spherical tokamaks -- `spherical_tokamak_eval` and
    `st_regression` set all three switches explicitly (`:350/:354/:358` and
    `:1223/:1042/:1180`) -- and the last blocker that stood between the port and either
    of them.

    **It owns three fields where its plane-stress sibling owns five**, which is why
    `TfStress` is a family base and neither class is the whole of it: `.tfcoil.casestr`
    and `.tfcoil.insstrain` are computed only inside `stresscl`'s
    `i_tf_stress_model == 1` branch and left `None` here (`base.py:2520-2521`,
    `:2991-2998`). Not owning them is the `next_steps.md` §14.2 answer to a switch that
    would otherwise create a dead read.

    **Only the averaged-turn arm exists, and that was measured.** The integer-turn arm
    of this solver is a real PROCESS branch, but no tracked file reaches it: both ST
    files set `i_tf_turns_integer = 0`. It is refused in `indat.py` as `(0, 1, 1)`
    rather than written blind, for the same reason its plane-stress counterpart names --
    the two arms read *different* fields, and on the integer arm
    `.superconducting_tfcoil.dx_tf_turn_cable_space_average` has no producer.

    `n_tf_graded_layers` is a static field for the same reason as on the sibling: it
    fixes the layer stack's shape, so it cannot be a traced input. Both ST files leave
    it at PROCESS's default of `1`.
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
