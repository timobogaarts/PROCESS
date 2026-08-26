"""The tokamak TF coil's namespace -- the nineteen slots of
`.tokamak.cicc_superconducting_tf_coil`.

Beside the nodes it names (`model_tree_design.md` §11), and spanning three modules
because PROCESS's own `CICCSuperconductingTFCoil` does: `base.py` is reached through it
by *inheritance* rather than by any call in `caller.py`
(`tokamak_call_surface.md` §A row 3), `quench.py` through one call inside it. That is
why this is a `namespace.py` in the package rather than a class at the foot of one
module -- there is no single module the slots are all beside.

**Nine of the nineteen slots are switched, and every switch is answered by
`indat.py`**, never here. The families and the switch each answers are in the class
docstrings below; the reference-run arm of every one is
`tests/regression/input_files/large_tokamak_eval.IN.DAT`'s, and `tfcoil/base.md`,
`tfcoil/superconducting.md` and `tfcoil/quench.md` carry the per-arm evidence.

**One slot is legitimately empty on the reference run.** `dx_tf_side_case_min` has a
node only when `.tfcoil.tfc_sidewall_is_fraction` is `True`; PROCESS's default is
`False` (`tfcoil_variables.py:95`), on which `.tfcoil.dx_tf_side_case_min` is simply an
input and no code computes it. `None` is the honest occupant, the same spelling
`costs.cost_of_electricity` and `power.cryo_q_nuc` already use -- an unowned read is a
correct answer for that field.

**`.tfcoil.c_tf_turn` is a boundary input on this configuration, by measurement.**
`superconducting.md` finding 1: at `i_dx_tf_turn_general_input == False` and
`i_dx_tf_turn_cable_space_general_input == False` nothing under `process/models/`
produces it -- it is iteration variable 60 and enters from the input file. So this slot
produces **nine** of the ten variables `_audit/tokamak_boundary.md` lists against it,
and the tenth is an unknown rather than a missing node.
"""

import dataclasses

from cottax.interfaces.pytree_namespace_module import ModelNamespace

from functional_process.models.tfcoil.base import (
    DrTfPlasmaCaseFromFraction,
    DrTfPlasmaCaseFromInput,
    DxTfSideCaseMinFromFraction,
    GenericTfCoilAreaAndMasses,
    RBTfInboardPeak,
    TfCoilSelfInductance,
    TfCoilShape,
    TfCurrent,
    TfGlobalGeometry,
    TfStoredMagneticEnergy,
)
from functional_process.models.tfcoil.quench import TfCoilDumpQuenchVoltage
from functional_process.models.tfcoil.superconducting import (
    CiccAveragedTurnGeometry,
    CiccInboardAreasAndFractions,
    DxTfSideCase,
    PeakBTfInboardWithRipple,
    SuperconductingTfCoilAreasAndMasses,
    SuperconductingTfWpGeometry,
    TfCaseAreas,
    TfTurnArea,
    TfWpCurrents,
)


class CiccSuperconductingTfCoil(ModelNamespace):
    """A cable-in-conduit superconducting TF coil: the tokamak's magnet system.

    Named for `process/models/tfcoil/superconducting.py::CICCSuperconductingTFCoil`,
    which is what `caller.py:306` runs at `i_tf_sup == 1` and
    `.superconducting_tfcoil.i_tf_turn_type == 1`. Both of those switches are resolved
    **above every model**, in `caller.py:295-316`, which is why neither appears as a slot
    here: a machine whose `i_tf_turn_type` is `2` has a different occupant of
    `.tokamak.cicc_superconducting_tf_coil` (`CROCOSuperconductingTFCoil`, unported),
    not a different slot inside this one.

    Nineteen slots, eighteen of them occupied on the reference run. Written in
    dependency order rather than in file order, because the
    chain is long and legible: global geometry and case thickness, then the current, then
    the winding pack, then the turns, then the areas and masses, then the two quantities
    the quench and energy limits read.
    """

    # ---- base.py: the geometry every TF coil model shares --------------------------

    tf_global_geometry: TfGlobalGeometry = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_case_geom` -- circular or straight front case, two occupants whose
    reads-sets are identical and which are separate classes anyway (`next_steps.md`
    §14.2). Owns nine unswitched outputs including `.tfcoil.a_tf_inboard_total`, which
    everything below depends on."""

    dr_tf_plasma_case: DrTfPlasmaCaseFromInput | DrTfPlasmaCaseFromFraction = (
        dataclasses.field(kw_only=True)
    )
    """`.tfcoil.i_f_dr_tf_plasma_case` -- and the one slot in this port whose two arms
    are *different kinds of node*, which is why the annotation is a union and not a
    family base class: there is no class both arms could inherit from, because
    `FixedPointFunction` and `ExplicitFunction` are the two kinds. The `False` arm is a
    `FixedPointFunction` because PROCESS clamps `dr_tf_plasma_case` in place; the `True`
    arm computes it from a fraction and never reads the entering value, so it is an
    `ExplicitFunction`. The
    loop is a property of the arm, not of the quantity (`base.md` OQ1)."""

    dx_tf_side_case_min: DxTfSideCaseMinFromFraction | None = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.tfc_sidewall_is_fraction` -- **`None` on the reference run.**

    PROCESS computes `.tfcoil.dx_tf_side_case_min` only when the sidewall thickness is
    given as a fraction; at the default `False` (`tfcoil_variables.py:95`) it is an
    input and there is no arm at all. Absence, spelled as absence -- not a refusal,
    because PROCESS itself computes nothing here."""

    r_b_tf_inboard_peak: RBTfInboardPeak = RBTfInboardPeak()
    """`.tfcoil.r_b_tf_inboard_peak`, the radius the peak inboard field is quoted at.
    Unswitched."""

    tf_current: TfCurrent = TfCurrent()
    """`.tfcoil.c_tf_total` and the peak symmetric field. Unswitched."""

    tf_coil_shape: TfCoilShape = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_shape` x `.physics.itart` x `.physics.i_single_null` -- and the
    producer of `.tfcoil.len_tf_coil`, one of the two `VarPath`s this device shares with
    the stellarator from a completely different formula (`base.md` §"Shared with the
    stellarator"). Two of the arms are written, both D-shaped and conventional-aspect."""

    tf_coil_self_inductance: TfCoilSelfInductance = dataclasses.field(kw_only=True)
    """`(.physics.itart, .tfcoil.i_tf_shape)` -- the D-shape arm integrates the arcs,
    the picture-frame arm is a closed form reading four entirely different fields."""

    tf_stored_magnetic_energy: TfStoredMagneticEnergy = TfStoredMagneticEnergy()
    """`.tfcoil.e_tf_coil_magnetic_stored` and its two totals. Unswitched, and read by
    `tf_coil_dump_quench_voltage` below."""

    generic_tf_coil_area_and_masses: GenericTfCoilAreaAndMasses = (
        GenericTfCoilAreaAndMasses()
    )
    """`.tfcoil.tfcryoarea`, `.tfcoil.tfocrn` and `.tfcoil.tficrn`. Unswitched.

    `.tfcoil.tfcryoarea` is the second of the two `VarPath`s this device shares with the
    stellarator from an entirely different formula (`.tfcoil.len_tf_coil` is the other,
    from `tf_coil_shape` above): a stellarator's is a modular-coil scaling, a tokamak's
    is this coil's surface area. One variable, two device-specific producers, two
    slots -- which is what a slot is for, and never both in one graph."""

    # ---- superconducting.py: the winding pack, the turns and the masses -------------

    superconducting_tf_wp_geometry: SuperconductingTfWpGeometry = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_tf_wp_geom` -- rectangular, double-rectangular or trapezoidal. All
    three arms read the same seven fields; one class per value regardless, per the
    binding policy. Note the value is **not** read from the input file as written:
    `init.py:977-989` resolves the `-1` `UNSET` default from `i_tf_turns_integer`, and
    `machine_from_indat` reproduces that resolution rather than the raw default."""

    tf_case_areas: TfCaseAreas = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_case_geom`, the same switch `tf_global_geometry` answers -- one
    input value filling two slots, the shape `PhysicsConfinementTime.tail` already
    records for `i_rad_loss`."""

    dx_tf_side_case: DxTfSideCase = dataclasses.field(kw_only=True)
    """`.tfcoil.i_tf_wp_geom` again; the trapezoidal arm reads one field where the other
    two read three."""

    tf_wp_currents: TfWpCurrents = TfWpCurrents()
    """`.tfcoil.j_tf_wp`, and **not** a `FixedPointFunction` here.

    `models/stellarator/namespace.py`'s counterpart carries a long argument about
    whether this field needs one. On the tokamak path it does not: PROCESS's body
    (`superconducting.py:1963-1970`) never consults the entering value, so there is no
    self-reference to cut."""

    peak_b_tf_inboard_with_ripple: PeakBTfInboardWithRipple = dataclasses.field(
        kw_only=True
    )
    """`round(.tfcoil.n_tf_coils)` -- treated as a switch, because the arms select
    different fit coefficients **and** own different numbers of outputs: the
    flat-allowance fallback returns before three of the four are assigned. A build-time
    branch is legitimate here only because `n_tf_coils` is not an iteration variable
    (`superconducting.md` OQ2)."""

    cicc_averaged_turn_geometry: CiccAveragedTurnGeometry = dataclasses.field(
        kw_only=True
    )
    """`.tfcoil.i_dx_tf_turn_general_input` together with
    `.tfcoil.i_dx_tf_turn_cable_space_general_input` -- two booleans, three arms,
    differing in which of `c_tf_turn` /
    `dx_tf_turn_general` / `dx_tf_turn_cable_space_general` each reads and which it
    owns. The written arm reads `.tfcoil.c_tf_turn` and does not own it, which is why
    that field is this slot's one boundary input."""

    cicc_inboard_areas_and_fractions: CiccInboardAreasAndFractions = (
        CiccInboardAreasAndFractions()
    )
    """The nine inboard areas and steel/insulation fractions. Unswitched."""

    tf_turn_area: TfTurnArea = TfTurnArea()
    """`.tfcoil.a_tf_turn` -- one division written inline in `run` rather than in any
    function (`superconducting.py:2700-2704`), which is why it is a node of its own."""

    superconducting_tf_coil_areas_and_masses: SuperconductingTfCoilAreasAndMasses = (
        dataclasses.field(kw_only=True)
    )
    """`.physics.itart` -- the conventional arm owns ten fields, the spherical arm two
    more (`whtcp`, `whttflgs`). Conditional ownership, so occupants and not a kwarg."""

    # ---- quench.py -----------------------------------------------------------------

    tf_coil_dump_quench_voltage: TfCoilDumpQuenchVoltage = TfCoilDumpQuenchVoltage()
    """`.tfcoil.v_tf_coil_dump_quench_kv`, three reads and no switch.

    Its companion output `.tfcoil.j_tf_wp_quench_heat_max` has **no node**: computing it
    needs helium density and specific heat at 75 quadrature temperatures, neither of
    which has a `VarPath`, and minting two is an open policy call
    (`quench.md` OQ1, `next_steps.md` §5). So the CoolProp dependency in the tokamak
    scope stays outside the assembled graph entirely -- it is not wrapped, it is not
    approximated, it is not there."""
