"""Pure-functional port of `process/models/tfcoil/base.py` -- the device-agnostic TF
coil layer that `SuperconductingTFCoil.run_base_superconducting_tf` reaches by
inheritance.

Audit record: `functional_process/_audit/units/models/tfcoil/base.md`.

**Scope is the minimal closure of `.tokamak.cicc_superconducting_tf_coil`'s ten boundary
reads** (`_audit/tokamak_boundary.md`), not the whole file. In scope and ported here:
`circumference`, `tf_global_geometry` (split three ways, see below), `tf_current`,
`tf_coil_shape_inner`, `tf_coil_self_inductance`, `tf_stored_magnetic_energy`,
`generic_tf_coil_area_and_masses`, plus the one inline formula
`run_base_tf` writes for `.tfcoil.r_b_tf_inboard_peak`
(`process/models/tfcoil/base.py:166-171`).

Deliberately **out** of scope, with reasons:

- `tf_field_and_force` (`base.py:1623`) -- feeds only `stresscl`; none of the ten
  boundary reads depends on it, and it carries three more switches (`i_tf_sup`,
  `itart`, `i_cp_joints`).
- `stresscl` and the elasticity helpers (`base.py:2222-4670`) -- `numba.njit`,
  ~2400 lines, and every output is a stress, none of which is on the boundary.
- `cntrpst` (`base.py:1211`) -- the TART centrepost, `itart == 1` only.
- `he_density`/`he_cp`/`he_visco`/`he_th_cond`/`al_th_cond` (`base.py:1827-2065`) --
  CoolProp-backed property lookups reached only from `cntrpst`. **They are not on this
  slot's CoolProp path**; see `functional_process/cottax/tfcoil/quench.py` for the one
  that is.

## `tf_global_geometry` is three nodes here, not one

`process/models/tfcoil/base.py:214-372` computes eleven quantities under three
independent switches, and the three groups of outputs are disjoint -- nothing computed
after a branch is read by another branch. Splitting is therefore semantics-preserving
and removes two invented edges the composite node would have declared:

| group | switch | node(s) here |
|---|---|---|
| the nine unconditional geometry outputs (only `a_tf_inboard_total` branches) | `i_tf_case_geom` | `TfGlobalGeometryCircularCase` / `TfGlobalGeometryStraightCase` |
| `.tfcoil.dr_tf_plasma_case` | `i_f_dr_tf_plasma_case` | `DrTfPlasmaCaseFromInput` / `DrTfPlasmaCaseFromFraction` |
| `.tfcoil.dx_tf_side_case_min` | `tfc_sidewall_is_fraction` | `DxTfSideCaseMinFromFraction` only -- see below |

**`tfc_sidewall_is_fraction == False` gets no node at all.** The branch is
`dx_tf_side_case_min = data.tfcoil.dx_tf_side_case_min` (`base.py:358`) -- a verbatim
read-back of the field it is about to be written to, i.e. an identity. Conditional
ownership, exactly the shape `models/power/thermal_cryo.py`'s
`calculate_p_fw_blkt_coolant_pump_mw` records: on that arm the field is a run input and
nothing produces it. `large_tokamak_eval.IN.DAT` does not set `tfc_sidewall_is_fraction`,
so it takes the `False` default (`tfcoil_variables.py:95`) and this is the live arm.

**`.tfcoil.dr_tf_plasma_case` is a genuine self-loop on the `False` arm**, and unlike
the sidewall one it is *not* an identity: `base.py:328` reads the entering value and
`base.py:333-340` raises it to `(r_tf_inboard_in + dr_tf_inboard) * (1 - cos(pi/n))`
when it is below that. `DrTfPlasmaCaseFromInput` is therefore a `FixedPointFunction`
(`step`, minting `^cond.tfcoil.dr_tf_plasma_case`), which is what cottax's "a node may
not read what it owns" requires. Its fixed point is reached in one step because
`jnp.maximum(x, m)` is idempotent in `x` and `m` does not depend on `x`; at the
reference point the clamp binds (`dr_tf_plasma_case` defaults to `0.0`,
`tfcoil_variables.py:77`, and is not in the input file) so the derivative with respect
to its own entering value is exactly zero there.

## A PROCESS defect, ported faithfully

`base.py:344-346` calls `logger.error("dr_tf_plasma_case too small to accommodate the
WP, forced to minimum value")` **unconditionally** -- it sits outside the `if` at 333
whose body it describes. Logging only, no value effect, so the port drops it (a pure
function does not log); recorded in `base.md` as defect **D1** rather than silently
fixed.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    FixedPointFunction,
    From,
    OutputInto,
)

from functional_process.paths import build, physics, superconducting_tfcoil, tfcoil
from functional_process.models.tfcoil.base import (
    calculate_r_b_tf_inboard_peak,
    calculate_tf_global_geometry_circular_case,
    calculate_tf_global_geometry_straight_case,
    circumference,  # noqa: F401 -- re-exported for tests/.../test_base.py
    dr_tf_plasma_case_from_fraction,
    dr_tf_plasma_case_from_input,
    dx_tf_side_case_min_from_fraction,
    generic_tf_coil_area_and_masses,
    tf_coil_self_inductance_d_shape,
    tf_coil_self_inductance_picture_frame,
    tf_coil_shape_inner_d_shape_double_null,
    tf_coil_shape_inner_d_shape_single_null,
    tf_coil_shape_inner_picture_frame_tart,
    tf_current,
    tf_stored_magnetic_energy,
)

# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class TfGlobalGeometry(ExplicitFunction):
    """The family that owns `tf_global_geometry`'s nine unswitched outputs.

    One occupant per `i_tf_case_geom` value. The two arms' reads-sets are identical
    (both read the same five fields); they are still separate classes, because a switch
    selects an occupant and never a static kwarg -- `_audit/next_steps.md` §14.2, and
    the `istore` precedent it names for two arms differing only in a literal.
    """


class TfGlobalGeometryCircularCase(TfGlobalGeometry):
    """`i_tf_case_geom == TFPlasmaCaseType.CIRCULAR` (0) -- `large_tokamak_eval`'s arm.

    The input file does not set `i_tf_case_geom`, so it takes the `0` default
    (`process/data_structure/tfcoil_variables.py:234`).
    """

    rad_tf_coil_inboard_toroidal_half = OutputInto(superconducting_tfcoil)
    tan_theta_coil = OutputInto(superconducting_tfcoil)
    a_tf_inboard_total = OutputInto(tfcoil)
    r_tf_outboard_in = OutputInto(superconducting_tfcoil)
    r_tf_outboard_out = OutputInto(superconducting_tfcoil)
    dx_tf_inboard_out_toroidal = OutputInto(tfcoil)
    a_tf_leg_outboard = OutputInto(tfcoil)
    dr_tf_full_midplane = OutputInto(tfcoil)
    dr_tf_internal_midplane = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        r_tf_inboard_out=From(build),
        r_tf_inboard_in=From(build),
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
    ):
        return calculate_tf_global_geometry_circular_case(
            n_tf_coils=n_tf_coils,
            r_tf_inboard_out=r_tf_inboard_out,
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_outboard_mid=r_tf_outboard_mid,
            dr_tf_outboard=dr_tf_outboard,
        )


class TfGlobalGeometryStraightCase(TfGlobalGeometry):
    """`i_tf_case_geom == TFPlasmaCaseType.STRAIGHT` (1)."""

    rad_tf_coil_inboard_toroidal_half = OutputInto(superconducting_tfcoil)
    tan_theta_coil = OutputInto(superconducting_tfcoil)
    a_tf_inboard_total = OutputInto(tfcoil)
    r_tf_outboard_in = OutputInto(superconducting_tfcoil)
    r_tf_outboard_out = OutputInto(superconducting_tfcoil)
    dx_tf_inboard_out_toroidal = OutputInto(tfcoil)
    a_tf_leg_outboard = OutputInto(tfcoil)
    dr_tf_full_midplane = OutputInto(tfcoil)
    dr_tf_internal_midplane = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        r_tf_inboard_out=From(build),
        r_tf_inboard_in=From(build),
        r_tf_outboard_mid=From(build),
        dr_tf_outboard=From(build),
    ):
        return calculate_tf_global_geometry_straight_case(
            n_tf_coils=n_tf_coils,
            r_tf_inboard_out=r_tf_inboard_out,
            r_tf_inboard_in=r_tf_inboard_in,
            r_tf_outboard_mid=r_tf_outboard_mid,
            dr_tf_outboard=dr_tf_outboard,
        )


class DrTfPlasmaCaseFromInput(FixedPointFunction):
    """`i_f_dr_tf_plasma_case == False` -- `large_tokamak_eval`'s arm, and a self-loop.

    `step` reads the entering `.tfcoil.dr_tf_plasma_case` and writes the minted
    `^cond.tfcoil.dr_tf_plasma_case`; the `FixedPoint` this class also declares reads
    that copy and owns the real path, so neither piece reads what it owns. See the
    module docstring for why the loop is real (a clamp, not an identity) and why it
    closes in one step.
    """

    dr_tf_plasma_case = OutputInto(tfcoil)

    def step(
        self,
        dr_tf_plasma_case=From(tfcoil),
        r_tf_inboard_in=From(build),
        dr_tf_inboard=From(build),
        n_tf_coils=From(tfcoil),
    ):
        return dr_tf_plasma_case_from_input(
            dr_tf_plasma_case=dr_tf_plasma_case,
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_inboard=dr_tf_inboard,
            n_tf_coils=n_tf_coils,
        )


class DrTfPlasmaCaseFromFraction(ExplicitFunction):
    """`i_f_dr_tf_plasma_case == True`: a plain node, no loop.

    The fraction arm never reads the entering `.tfcoil.dr_tf_plasma_case`, so this one
    is an `ExplicitFunction` where its sibling has to be a `FixedPointFunction` -- the
    clearest possible demonstration that the loop is a property of one *arm*, not of
    the quantity.
    """

    dr_tf_plasma_case = OutputInto(tfcoil)

    def __call__(
        self,
        f_dr_tf_plasma_case=From(tfcoil),
        dr_tf_inboard=From(build),
        r_tf_inboard_in=From(build),
        n_tf_coils=From(tfcoil),
    ):
        return dr_tf_plasma_case_from_fraction(
            f_dr_tf_plasma_case=f_dr_tf_plasma_case,
            dr_tf_inboard=dr_tf_inboard,
            r_tf_inboard_in=r_tf_inboard_in,
            n_tf_coils=n_tf_coils,
        )


class DxTfSideCaseMinFromFraction(ExplicitFunction):
    """`tfc_sidewall_is_fraction == True`.

    The `False` arm has no node -- see the module docstring.
    """

    dx_tf_side_case_min = OutputInto(tfcoil)

    def __call__(
        self,
        casths_fraction=From(tfcoil),
        r_tf_inboard_in=From(build),
        dr_tf_nose_case=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return dx_tf_side_case_min_from_fraction(
            casths_fraction=casths_fraction,
            r_tf_inboard_in=r_tf_inboard_in,
            dr_tf_nose_case=dr_tf_nose_case,
            n_tf_coils=n_tf_coils,
        )


class RBTfInboardPeak(ExplicitFunction):
    """cottax node: `run_base_tf`'s inline `.tfcoil.r_b_tf_inboard_peak`."""

    r_b_tf_inboard_peak = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        dr_tf_plasma_case=From(tfcoil),
        dx_tf_wp_insulation=From(tfcoil),
        dx_tf_wp_insertion_gap=From(tfcoil),
    ):
        return calculate_r_b_tf_inboard_peak(
            r_tf_inboard_out=r_tf_inboard_out,
            dr_tf_plasma_case=dr_tf_plasma_case,
            dx_tf_wp_insulation=dx_tf_wp_insulation,
            dx_tf_wp_insertion_gap=dx_tf_wp_insertion_gap,
        )


class TfCurrent(ExplicitFunction):
    """cottax node: `tf_current`, ports declared. No switch, so no family."""

    b_tf_inboard_peak_symmetric = OutputInto(tfcoil)
    c_tf_total = OutputInto(tfcoil)
    c_tf_coil = OutputInto(superconducting_tfcoil)
    j_tf_coil_full_area = OutputInto(tfcoil)

    def __call__(
        self,
        n_tf_coils=From(tfcoil),
        b_plasma_toroidal_on_axis=From(physics),
        rmajor=From(physics),
        r_b_tf_inboard_peak=From(tfcoil),
        a_tf_inboard_total=From(tfcoil),
    ):
        return tf_current(
            n_tf_coils=n_tf_coils,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            rmajor=rmajor,
            r_b_tf_inboard_peak=r_b_tf_inboard_peak,
            a_tf_inboard_total=a_tf_inboard_total,
        )


class TfCoilShape(ExplicitFunction):
    """The family that owns `.tfcoil.len_tf_coil` and the arc arrays.

    Three switches decide it -- `i_tf_shape`, `itart`, `i_single_null` -- and the arms
    read genuinely different variables (`r_cp_top` on the two `itart == 1` arms,
    `z_tf_top` on all but the D-shape double-null one, `r_tf_outboard_mid` on the
    picture frame). Three occupants are written: the two `i_tf_shape == D_SHAPE`,
    `itart == 0` arms and the `i_tf_shape == PICTURE_FRAME`, `itart == 1` one the two
    ST regression files take. See `base.md` for the UNPORTED list.
    """


class TfCoilShapeDShapeSingleNull(TfCoilShape):
    """`i_tf_shape == 1`, `itart == 0`, `i_single_null == 1` -- the reference arm."""

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        r_tf_outboard_in=From(superconducting_tfcoil),
        z_tf_inside_half=From(build),
        z_tf_top=From(build),
        dr_tf_inboard=From(build),
    ):
        return tf_coil_shape_inner_d_shape_single_null(
            r_tf_inboard_out=r_tf_inboard_out,
            rmajor=rmajor,
            rminor=rminor,
            r_tf_outboard_in=r_tf_outboard_in,
            z_tf_inside_half=z_tf_inside_half,
            z_tf_top=z_tf_top,
            dr_tf_inboard=dr_tf_inboard,
        )


class TfCoilShapeDShapeDoubleNull(TfCoilShape):
    """`i_tf_shape == 1`, `itart == 0`, `i_single_null == 0`.

    Does not read `z_tf_top`.
    """

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        rmajor=From(physics),
        rminor=From(physics),
        r_tf_outboard_in=From(superconducting_tfcoil),
        z_tf_inside_half=From(build),
        dr_tf_inboard=From(build),
    ):
        return tf_coil_shape_inner_d_shape_double_null(
            r_tf_inboard_out=r_tf_inboard_out,
            rmajor=rmajor,
            rminor=rminor,
            r_tf_outboard_in=r_tf_outboard_in,
            z_tf_inside_half=z_tf_inside_half,
            dr_tf_inboard=dr_tf_inboard,
        )


class TfCoilShapePictureFrameTart(TfCoilShape):
    """`i_tf_shape == 2`, `itart == 1` -- both ST regression files' arm.

    The reads-set is the measurement: it reads `.build.r_cp_top` and
    `.build.r_tf_outboard_mid`, which neither D-shape sibling touches, and reads
    **neither** `.physics.rmajor` nor `.physics.rminor`, which both siblings do. A
    single node carrying `i_tf_shape`/`itart` as static kwargs would have declared all
    four edges on every arm; three of the four would have been invented.

    `.build.r_cp_top` has **no producer in this port** -- `process/models/build.py`'s
    `calculate_radial_build` writes it at `:1750-1813` and that slice is not ported --
    so it enters the ST graphs as a declared boundary input. See `base.md`'s dated
    section; it is a lost producer, recorded as one, not stubbed.
    """

    len_tf_coil = OutputInto(tfcoil)
    tfa = OutputInto(tfcoil)
    tfb = OutputInto(tfcoil)
    r_tf_arc = OutputInto(tfcoil)
    z_tf_arc = OutputInto(tfcoil)

    def __call__(
        self,
        r_cp_top=From(build),
        r_tf_outboard_in=From(superconducting_tfcoil),
        z_tf_inside_half=From(build),
        z_tf_top=From(build),
        dr_tf_inboard=From(build),
        r_tf_outboard_mid=From(build),
    ):
        return tf_coil_shape_inner_picture_frame_tart(
            r_cp_top=r_cp_top,
            r_tf_outboard_in=r_tf_outboard_in,
            z_tf_inside_half=z_tf_inside_half,
            z_tf_top=z_tf_top,
            dr_tf_inboard=dr_tf_inboard,
            r_tf_outboard_mid=r_tf_outboard_mid,
        )


class TfCoilSelfInductance(ExplicitFunction):
    """The family that owns `.tfcoil.ind_tf_coil`. `(itart, i_tf_shape)` decides it."""


class TfCoilSelfInductanceDShape(TfCoilSelfInductance):
    """`itart == 0` and `i_tf_shape == 1` -- the reference arm.

    Reads three things; the composite PROCESS function takes nine, and the other six
    belong to the sibling arm. That gap is the measurement this split exists to make.
    """

    ind_tf_coil = OutputInto(tfcoil)

    def __call__(
        self,
        dr_tf_inboard=From(build),
        r_tf_arc=From(tfcoil),
        z_tf_arc=From(tfcoil),
    ):
        return tf_coil_self_inductance_d_shape(
            dr_tf_inboard=dr_tf_inboard, r_tf_arc=r_tf_arc, z_tf_arc=z_tf_arc
        )


class TfCoilSelfInductancePictureFrame(TfCoilSelfInductance):
    """Everything else (`i_tf_shape == 2`, or `itart == 1`): the closed form."""

    ind_tf_coil = OutputInto(tfcoil)

    def __call__(
        self,
        z_tf_inside_half=From(build),
        dr_tf_outboard=From(build),
        r_tf_outboard_mid=From(build),
        r_tf_inboard_mid=From(build),
    ):
        return tf_coil_self_inductance_picture_frame(
            z_tf_inside_half=z_tf_inside_half,
            dr_tf_outboard=dr_tf_outboard,
            r_tf_outboard_mid=r_tf_outboard_mid,
            r_tf_inboard_mid=r_tf_inboard_mid,
        )


class TfStoredMagneticEnergy(ExplicitFunction):
    """cottax node: `tf_stored_magnetic_energy`. Owns one of the slot's ten reads."""

    e_tf_magnetic_stored_total = OutputInto(tfcoil)
    e_tf_magnetic_stored_total_gj = OutputInto(tfcoil)
    e_tf_coil_magnetic_stored = OutputInto(tfcoil)

    def __call__(
        self,
        ind_tf_coil=From(tfcoil),
        c_tf_total=From(tfcoil),
        n_tf_coils=From(tfcoil),
    ):
        return tf_stored_magnetic_energy(
            ind_tf_coil=ind_tf_coil, c_tf_total=c_tf_total, n_tf_coils=n_tf_coils
        )


class GenericTfCoilAreaAndMasses(ExplicitFunction):
    """cottax node: `generic_tf_coil_area_and_masses`. Owns `.tfcoil.tfcryoarea`."""

    tfocrn = OutputInto(tfcoil)
    tficrn = OutputInto(tfcoil)
    tfcryoarea = OutputInto(tfcoil)

    def __call__(
        self,
        r_tf_inboard_out=From(build),
        r_tf_inboard_in=From(build),
        rad_tf_coil_inboard_toroidal_half=From(superconducting_tfcoil),
        tan_theta_coil=From(superconducting_tfcoil),
        len_tf_coil=From(tfcoil),
        r_tf_inboard_mid=From(build),
        r_tf_outboard_mid=From(build),
    ):
        return generic_tf_coil_area_and_masses(
            r_tf_inboard_out=r_tf_inboard_out,
            r_tf_inboard_in=r_tf_inboard_in,
            rad_tf_coil_inboard_toroidal_half=rad_tf_coil_inboard_toroidal_half,
            tan_theta_coil=tan_theta_coil,
            len_tf_coil=len_tf_coil,
            r_tf_inboard_mid=r_tf_inboard_mid,
            r_tf_outboard_mid=r_tf_outboard_mid,
        )
